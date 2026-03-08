import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import math
from numba import jit, vectorize, float64

# --- CONFIGURATION ---
MODEL_FILE = "iv_prod_xgb.json"
DEFAULT_RATE = 0.0364

COMMODITY_TICKERS = {"gold", "silver", "longterm"}
STOCK_TICKERS     = {"aapl", "amzn", "goog"}
INDEX_TICKERS     = {"sp500", "nq100", "dowjones"}
KNOWN_TICKERS = sorted(COMMODITY_TICKERS | STOCK_TICKERS | INDEX_TICKERS)

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
    # Do not print warning here, handle in main
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

@jit(nopython=True)
def calculate_greeks_numba(S, K, T, r, sigma, is_put):
    if T <= 1e-6 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    nd1 = norm_pdf(d1)
    
    delta = 0.0
    rho = 0.0
    theta = 0.0
    
    if is_put == 0.0:
        delta = norm_cdf(d1)
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
        theta = (-S * nd1 * sigma / (2 * sqrt_T) 
                 - r * K * math.exp(-r * T) * norm_cdf(d2))
    else:
        delta = -norm_cdf(-d1)
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)
        theta = (-S * nd1 * sigma / (2 * sqrt_T) 
                 + r * K * math.exp(-r * T) * norm_cdf(-d2))

    gamma = nd1 / (S * sigma * sqrt_T)
    vega = S * sqrt_T * nd1 / 100.0
    theta = theta / 365.0 

    return delta, gamma, vega, theta, rho

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
    ] + [f'ticker_{t}' for t in KNOWN_TICKERS]
    return df[features]

def get_strike_step(underlying):
    if underlying <= 20:
        return 0.5
    elif underlying <= 100:
        return 1.0
    elif underlying <= 500:
        return 2.5
    else:
        return 5.0

def main():
    parser = argparse.ArgumentParser(description="Price option chain form.")
    parser.add_argument("underlying", type=float, help="Underlying price")
    parser.add_argument("days", type=int, help="Days to expiration")
    parser.add_argument("vix", type=float, help="Volatility Index")
    parser.add_argument("rate", type=float, nargs='?', default=DEFAULT_RATE, help="Risk-free rate (default: 0.0364)")
    parser.add_argument("-t", type=str, default="", help="Specific ticker (aapl, gold, sp500...)")
    
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
    
    # 16 strikes below and 16 above
    strikes = [center_strike + i * step for i in range(-16, 17)]
    
    # Find the 2 strikes closest to underlying
    sorted_by_dist = sorted(strikes, key=lambda x: abs(x - args.underlying))
    closest_two = sorted_by_dist[:2]
    
    print(f"\nUnderlying: {args.underlying} | Days: {args.days} | VIX: {args.vix} | Rate: {args.rate}")
    print("-" * 125)
    print(f"{'CALLS (Price | IV% | Delta | Gamma | Vega | Theta)':>56} | {'STRIKE':^7} | {'PUTS (Price | IV% | Delta | Gamma | Vega | Theta)':<56}")
    print("-" * 125)
    
    ticker = args.t.strip().lower()
    if ticker and ticker not in KNOWN_TICKERS:
        print(f"Warning: unknown ticker '{ticker}', defaulting all ticker flags to 0")

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
    for kt in KNOWN_TICKERS:
        df_batch[f'ticker_{kt}'] = 1 if ticker == kt else 0
    
    X_batch = prepare_features(df_batch)
    # Reorder columns to exactly match what the model was trained on
    X_batch = X_batch[model.get_booster().feature_names]
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
        
        dc, gc, vc, tc, rc = calculate_greeks_numba(args.underlying, K, T, args.rate, sigma_c, 0.0)
        dp, gp, vp, tp, rp = calculate_greeks_numba(args.underlying, K, T, args.rate, sigma_p, 1.0)
        
        # Format string
        c_str = f"${price_c:5.2f} | {iv_call:5.1f}% | {dc:6.3f} | {gc:5.3f} | {vc:5.3f} | {tc:6.3f}"
        p_str = f"${price_p:5.2f} | {iv_put:5.1f}% | {dp:6.3f} | {gp:5.3f} | {vp:5.3f} | {tp:6.3f}"
        
        strike_fmt = f"{K:g}" # remove trailing zeros 
        strike_str = f"{strike_fmt}{marker}"
        
        print(f"{c_str:>56} | {strike_str:^7} | {p_str:<56}")

if __name__ == "__main__":
    main()
