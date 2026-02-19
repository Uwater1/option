import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import glob
import math
from numba import jit, vectorize, float64

# --- CONFIGURATION ---
MODEL_FILE = "iv_surface_prod.json"
DEFAULT_RATE = 0.0364

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
    """
    Prepare features for the XGBoost model.
    Expects numerical columns: strike, underlying, days, vix, is_put
    """
    df = df.copy()
    
    # Calculations
    df['log_moneyness'] = np.log(df['strike'] / df['underlying'])
    df['moneyness_sq'] = df['log_moneyness'] ** 2
    
    # Avoid negative sqrt
    df['sqrt_dte'] = np.sqrt(df['days'])
    
    df['vix_sq'] = df['vix'] ** 2
    df['vix_x_dte'] = df['vix'] * df['sqrt_dte']
    
    features = [
        'log_moneyness', 
        'moneyness_sq', 
        'days', 
        'sqrt_dte', 
        'is_put', 
        'vix', 
        'vix_sq', 
        'vix_x_dte'
    ]
    return df[features]

def main():
    parser = argparse.ArgumentParser(description="Price option.")
    parser.add_argument("underlying", type=float, help="Underlying price")
    parser.add_argument("strike", type=float, help="Strike price")
    parser.add_argument("days", type=int, help="Days to expiration")
    parser.add_argument("vix", type=float, help="Volatility Index")
    parser.add_argument("rate", type=float, nargs='?', default=DEFAULT_RATE, help="Risk-free rate (default: 0.0364)")
    
    args = parser.parse_args()
    
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found.")
        print("Please ask the developer to train the production model first.")
        return

    # Load Model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    # Pricing both Call and Put
    types = ["call", "put"]
    
    print(f"\nUnderlying: {args.underlying} | Strike: {args.strike} | Days: {args.days} | VIX: {args.vix} | Rate: {args.rate}")
    print("-" * 65)
    print(f"{'Type':<6} {'Price':<10} {'IV':<8} {'Delta':<8} {'Gamma':<8} {'Vega':<8} {'Theta':<8}")
    print("-" * 65)
    
    for t in types:
        is_put = 1.0 if t == "put" else 0.0
        
        # Prepare single row data
        data = {
            'underlying': [args.underlying],
            'strike': [args.strike],
            'days': [args.days],
            'vix': [args.vix],
            'is_put': [is_put]
        }
        df = pd.DataFrame(data)
        
        X = prepare_features(df)
        predicted_iv = model.predict(X)[0]
        
        sigma = predicted_iv / 100.0
        T = args.days / 365.0
        
        # Numba calculation
        price = black_scholes_numba(args.underlying, args.strike, T, args.rate, sigma, is_put)
        delta, gamma, vega, theta, rho = calculate_greeks_numba(
            args.underlying, args.strike, T, args.rate, sigma, is_put
        )
        
        print(f"{t.upper():<6} ${price:<9.2f} {predicted_iv:<7.2f}% {delta:<8.4f} {gamma:<8.4f} {vega:<8.4f} {theta:<8.4f}")

if __name__ == "__main__":
    main()
