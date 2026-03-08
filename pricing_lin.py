import argparse
import os
import numpy as np
import json
import math

# --- CONFIGURATION ---
PARAMS_FILE = "iv_lin_params.json"
DEFAULT_RATE = 0.0364

# --- BLACK-SCHOLES MATH (Pure Python, Zero Overhead) ---

def cdf(x):
    """Standard Normal CDF using math.erf (zero dependency)"""
    return 0.5 * (1 + math.erf(x / 1.4142135623730951))

def get_features(underlying, strike, days, vix):
    log_moneyness = math.log(strike / underlying)
    moneyness_sq = log_moneyness ** 2
    sqrt_dte = math.sqrt(max(days, 0.001))
    inv_dte = 1.0 / max(days, 0.001)
    vix_sq = vix ** 2
    vix_x_dte = vix * sqrt_dte
    vix_x_log_moneyness = vix * log_moneyness
    
    # Order must match iv_lin_params.json['features']
    return np.array([
        log_moneyness, moneyness_sq, float(days), sqrt_dte, inv_dte,
        vix, vix_sq, vix_x_dte, vix_x_log_moneyness
    ])

def predict_iv(features, params):
    weights = np.array(params['weights'])
    mean = np.array(params['mean'])
    scale = np.array(params['scale'])
    intercept = params['intercept']
    
    scaled_features = (features - mean) / scale
    y_log = np.dot(scaled_features, weights) + intercept
    return math.exp(y_log)

def main():
    parser = argparse.ArgumentParser(description="Price option (ElasticNet Standalone, Pure Python).")
    parser.add_argument("underlying", type=float, help="Underlying price")
    parser.add_argument("strike", type=float, help="Strike price")
    parser.add_argument("days", type=int, help="Days to expiration")
    parser.add_argument("vix", type=float, help="Volatility Index")
    parser.add_argument("rate", type=float, nargs='?', default=DEFAULT_RATE, help="Risk-free rate")
    args = parser.parse_args()

    if not os.path.exists(PARAMS_FILE):
        print(f"Error: Params file '{PARAMS_FILE}' not found. Run train_lin_model.py first.")
        return

    with open(PARAMS_FILE, 'r') as f:
        config = json.load(f)

    print(f"\nUnderlying: {args.underlying} | Strike: {args.strike} | Days: {args.days} | VIX: {args.vix} | Rate: {args.rate}")
    print("-" * 35)
    print(f"{'Type':<6} {'Price':<10} {'IV':<8}")
    print("-" * 35)

    feats = get_features(args.underlying, args.strike, args.days, args.vix)
    T = args.days / 365.0

    for t in ["call", "put"]:
        params = config[t]
        predicted_iv = predict_iv(feats, params)
        sigma = predicted_iv / 100.0
        
        # Black-Scholes Formula
        if T <= 1e-6:
            price = max(0.0, args.strike - args.underlying) if t == "put" else max(0.0, args.underlying - args.strike)
        elif sigma <= 0:
            price = max(0.0, args.strike - args.underlying) if t == "put" else max(0.0, args.underlying - args.strike)
        else:
            d1 = (math.log(args.underlying / args.strike) + (args.rate + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if t == "call":
                price = args.underlying * cdf(d1) - args.strike * math.exp(-args.rate * T) * cdf(d2)
            else:
                price = args.strike * math.exp(-args.rate * T) * cdf(-d2) - args.underlying * cdf(-d1)

        print(f"{t.upper():<6} ${max(0, price):<9.2f} {predicted_iv:<7.2f}%")

if __name__ == "__main__":
    main()
