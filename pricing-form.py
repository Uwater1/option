import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import math
from numba import jit, vectorize, float64

# --- CONFIGURATION ---
MODEL_FILE = "iv_surface_prod.json"
DEFAULT_RATE = 0.0364

COMMODITY_TICKERS = {"gold", "silver", "longterm"}
STOCK_TICKERS     = {"aapl", "amzn", "goog"}
INDEX_TICKERS     = {"sp500", "nq100", "dowjones"}

def resolve_asset_class(token: str):
    """Resolve a -t value to (is_stock, is_index, is_commodity) flags."""
    t = token.strip().lower()
    if t in ("", "none"):
        return 0, 0, 0
    if t == "stock":     return 1, 0, 0
    if t == "index":     return 0, 1, 0
    if t == "commodity":  return 0, 0, 1
    if t in STOCK_TICKERS:     return 1, 0, 0
    if t in INDEX_TICKERS:     return 0, 1, 0
    if t in COMMODITY_TICKERS: return 0, 0, 1
    print(f"Warning: unknown ticker/class '{token}', defaulting all asset flags to 0")
    return 0, 0, 0

# --- NUMBA FUNCTIONS ---
@jit(nopython=True)
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / 1.4142135623730951))

@jit(nopython=True)
def norm_pdf(x):
    return 0.3989422804014327 * math.exp(-0.5 * x * x)

@vectorize([float64(float64, float64, float64, float64, float64, float64)], target='parallel')
def black_scholes_numba(S, K, T, r, sigma, is_put):
    if T <= 1e-6:
        if is_put == 1.0:
            return max(0.0, K - S)
        else:
            return max(0.0, S - K)
            
    if sigma <= 0 or S <= 0 or K <= 0:
         return 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if is_put == 0.0:
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        
    return price

# --- FEATURE PREP ---
def prepare_features(df):
    df = df.copy()
    
    df['net_moneyness'] = df['strike'] / df['underlying']
    df['log_moneyness'] = np.log(df['net_moneyness'])
    df['moneyness_sq'] = df['log_moneyness'] ** 2
    
    # Avoid negative sqrt
    df['sqrt_dte'] = np.sqrt(np.maximum(df['days'], 0.001))
    df['inv_dte'] = 1.0 / np.maximum(df['days'], 0.001)
    
    df['vix_sq'] = df['vix'] ** 2
    df['vix_x_dte'] = df['vix'] * df['sqrt_dte']
    df['vix_x_log_moneyness'] = df['vix'] * df['log_moneyness']
    
    # Bucketed Moneyness
    df['otm_amount'] = df['log_moneyness'] * (1 - 2 * df['is_put'])
    
    threshold = df['vix'] * 0.02 + df['sqrt_dte'] * 0.5
    df['is_atm'] = (np.abs(df['otm_amount']) <= 0.02).astype(int)
    df['is_otm'] = ((df['otm_amount'] > 0.02) & (df['otm_amount'] <= threshold)).astype(int)
    df['is_deep_otm'] = (df['otm_amount'] > threshold).astype(int)
    df['is_itm'] = ((df['otm_amount'] < -0.02) & (df['otm_amount'] >= -threshold)).astype(int)
    df['is_deep_itm'] = (df['otm_amount'] < -threshold).astype(int)
    
    df['dte_under_15'] = (df['days'] < 15).astype(int)
    df['dte_15_to_40'] = ((df['days'] >= 15) & (df['days'] <= 40)).astype(int)
    df['dte_over_40'] = (df['days'] > 40).astype(int)
    
    df['atm_iv_proxy'] = df['vix'] / 100.0
    
    features = [
        'net_moneyness', 'log_moneyness', 'moneyness_sq', 'days', 'sqrt_dte', 'inv_dte',
        'is_put', 'vix', 'vix_sq', 'vix_x_dte', 'vix_x_log_moneyness',
        'is_atm', 'is_otm', 'is_deep_otm', 'is_itm', 'is_deep_itm',
        'dte_under_15', 'dte_15_to_40', 'dte_over_40',
        'atm_iv_proxy',
        'is_stock', 'is_index', 'is_commodity'
    ]
    return df[features]

def get_strike_step(underlying):
    if underlying <= 20:
        return 0.5
    elif underlying <= 100:
        return 1.0
    elif underlying <= 500:
        return 5.0
    else:
        return 10.0

def main():
    parser = argparse.ArgumentParser(description="Price option chain form.")
    parser.add_argument("underlying", type=float, help="Underlying price")
    parser.add_argument("days", type=int, help="Days to expiration")
    parser.add_argument("vix", type=float, help="Volatility Index")
    parser.add_argument("rate", type=float, nargs='?', default=DEFAULT_RATE, help="Risk-free rate (default: 0.0364)")
    parser.add_argument("-t", type=str, default="", help="Asset type or ticker (stock, index, commodity, aapl, gold ...)")
    
    args = parser.parse_args()
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found.")
        print("Please ask the developer to train the production model first.")
        return

    # Load Model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    step = get_strike_step(args.underlying)
    center_strike = round(args.underlying / step) * step
    
    # 10 strikes below and 10 above
    strikes = [center_strike + i * step for i in range(-10, 11)]
    
    # Find the 2 strikes closest to underlying
    sorted_by_dist = sorted(strikes, key=lambda x: abs(x - args.underlying))
    closest_two = sorted_by_dist[:2]
    
    print(f"\nUnderlying: {args.underlying} | Days: {args.days} | VIX: {args.vix} | Rate: {args.rate}")
    print("-" * 55)
    print(f"{'CALLS (Price) | IV%':>22} | {'STRIKE':^7} | {'PUTS (Price) | IV%':<22}")
    print("-" * 55)
    
    # Pre-build dataframes for batch prediction
    # This is much faster than running xgboost predict in a loop 40 times
    rows = []
    for K in strikes:
        rows.append({'strike': K, 'is_put': 0.0})
        rows.append({'strike': K, 'is_put': 1.0})
        
    df_batch = pd.DataFrame(rows)
    df_batch['underlying'] = args.underlying
    df_batch['days'] = args.days
    df_batch['vix'] = args.vix

    is_stock, is_index, is_commodity = resolve_asset_class(args.t)
    df_batch['is_stock'] = is_stock
    df_batch['is_index'] = is_index
    df_batch['is_commodity'] = is_commodity
    
    X_batch = prepare_features(df_batch)
    iv_log_preds = model.predict(X_batch)
    iv_preds = np.exp(iv_log_preds)
    
    T = args.days / 365.0
    
    for i, K in enumerate(strikes):
        is_closest = K in closest_two
        marker = "*" if is_closest else " "
        
        iv_call = iv_preds[i * 2]
        iv_put = iv_preds[i * 2 + 1]
        
        sigma_c = iv_call / 100.0
        sigma_p = iv_put / 100.0
        
        price_c = black_scholes_numba(args.underlying, K, T, args.rate, sigma_c, 0.0)
        price_p = black_scholes_numba(args.underlying, K, T, args.rate, sigma_p, 1.0)
        
        # Format string
        c_str = f"${price_c:.2f} | {iv_call:.1f}%"
        p_str = f"${price_p:.2f} | {iv_put:.1f}%"
        
        strike_fmt = f"{K:g}" # remove trailing zeros 
        strike_str = f"{strike_fmt}{marker}"
        
        print(f"{c_str:>22} | {strike_str:^7} | {p_str:<22}")

if __name__ == "__main__":
    main()
