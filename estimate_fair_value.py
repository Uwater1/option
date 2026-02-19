import argparse
import numpy as np
import os
import ctypes
import sys
import pandas as pd

# Try importing XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not installed. Spread prediction might be less accurate (falling back to polynomial model).")

# Load C++ Library (with fallback to pure Python)
from scipy.stats import norm
from scipy.optimize import brentq

def _python_implied_volatility(price, S, K, T, r, is_call):
    """Pure Python fallback for implied volatility calculation."""
    option_type = 'call' if is_call else 'put'
    
    def objective(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return bs_price - price
    
    intrinsic = max(0, S - K) if is_call else max(0, K - S)
    if price <= intrinsic + 1e-5:
        return 0.0
    try:
        return brentq(objective, 0.001, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan

def _python_vega(S, K, T, r, sigma):
    """Pure Python fallback for vega calculation."""
    T = max(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def _python_gamma(S, K, T, r, sigma):
    """Pure Python fallback for gamma calculation."""
    T = max(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def _python_delta(S, K, T, r, sigma, is_call):
    """Pure Python fallback for delta calculation."""
    T = max(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)

def _python_theta(S, K, T, r, sigma, is_call):
    """Pure Python fallback for theta calculation."""
    T = max(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return (term1 - term2) / 365.0 if is_call else (term1 + term3) / 365.0

def _python_greeks_all(S, K, T, r, sigma, is_call):
    """Calculate all Greeks in one pass, sharing d1/d2 computation."""
    T = max(T, 1e-10)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = norm.pdf(d1)
    exp_rT = np.exp(-r * T)

    delta = norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    term1 = -(S * pdf_d1 * sigma) / (2 * sqrt_T)
    if is_call:
        theta = (term1 - r * K * exp_rT * norm.cdf(d2)) / 365.0
    else:
        theta = (term1 + r * K * exp_rT * norm.cdf(-d2)) / 365.0

    return delta, gamma, vega, theta

lib_path = os.path.join(os.path.dirname(__file__), 'libbs.so')
try:
    lib = ctypes.CDLL(lib_path)
    lib.implied_volatility.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.implied_volatility.restype = ctypes.c_double
    lib.black_scholes_vega.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.black_scholes_vega.restype = ctypes.c_double
    lib.black_scholes_gamma.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.black_scholes_gamma.restype = ctypes.c_double
    lib.black_scholes_delta.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.black_scholes_delta.restype = ctypes.c_double
    lib.black_scholes_theta.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.black_scholes_theta.restype = ctypes.c_double
    CPP_LIB_AVAILABLE = True
except OSError:
    print("Warning: Could not load C++ library libbs.so. Falling back to pure Python implementation.")
    CPP_LIB_AVAILABLE = False

def fast_iv(price, S, K, T, r, is_call):
    if CPP_LIB_AVAILABLE:
        return lib.implied_volatility(price, S, K, T, r, is_call)
    return _python_implied_volatility(price, S, K, T, r, is_call)

def fast_vega(S, K, T, r, sigma):
    if CPP_LIB_AVAILABLE:
        return lib.black_scholes_vega(S, K, T, r, sigma)
    return _python_vega(S, K, T, r, sigma)

def fast_gamma(S, K, T, r, sigma):
    if CPP_LIB_AVAILABLE:
        return lib.black_scholes_gamma(S, K, T, r, sigma)
    return _python_gamma(S, K, T, r, sigma)

def fast_delta(S, K, T, r, sigma, is_call):
    if CPP_LIB_AVAILABLE:
        return lib.black_scholes_delta(S, K, T, r, sigma, is_call)
    return _python_delta(S, K, T, r, sigma, is_call)

def fast_theta(S, K, T, r, sigma, is_call):
    if CPP_LIB_AVAILABLE:
        return lib.black_scholes_theta(S, K, T, r, sigma, is_call)
    return _python_theta(S, K, T, r, sigma, is_call)


# Module-level cache for XGBoost models
_xgb_model_cache = {}

def predict_spread_poly(moneyness, time_to_expire_years, iv, last_price):
    M, T, IV = moneyness, time_to_expire_years, iv
    rel_spread = (POLY_COEFFS['b0'] + 
                  POLY_COEFFS['b1']*M + 
                  POLY_COEFFS['b2']*T + 
                  POLY_COEFFS['b3']*IV + 
                  POLY_COEFFS['b4']*M**2 + 
                  POLY_COEFFS['b5']*T**2 + 
                  POLY_COEFFS['b6']*T*IV)
    return max(0.001, min(0.5, rel_spread)) * last_price

# Feature names for XGBoost DMatrix — production-only features (v2)
_XGB_FEATURE_NAMES = [
    'moneyness', 'time_to_expire_years', 'iv', 'gamma', 'vega',
    'delta', 'theta', 'is_call', 'dist_from_atm', 'inv_price',
    'log_price', 'sqrt_time', 'iv_moneyness', 'iv_squared', 'time_iv',
    'intrinsic_ratio', 'time_value_ratio',
    # NEW FEATURES
    'strike_mod_5', 'strike_mod_1', 'is_low_price', 
    'vanna', 'volga', 'cash_gamma'
]

def predict_spread_xgb(moneyness, time_to_expire_years, iv, gamma, vega, delta, theta,
                        is_call, last_price, strike, underlying_price, model_path="spread_xgb.json"):
    if not XGB_AVAILABLE:
        return None
    
    # Load and cache the model (only once per model_path)
    if model_path not in _xgb_model_cache:
        if not os.path.exists(model_path):
            return None
        model = xgb.Booster()
        model.load_model(model_path)
        _xgb_model_cache[model_path] = model
    model = _xgb_model_cache[model_path]
    
    # Compute production-available features (Basic)
    dist_from_atm = abs(moneyness - 1.0)
    inv_price = 1.0 / max(last_price, 0.01)
    log_price = np.log(max(last_price, 0.01))
    sqrt_time = np.sqrt(max(time_to_expire_years, 0.001))
    iv_moneyness = iv * moneyness
    iv_squared = iv ** 2
    time_iv = time_to_expire_years * iv
    
    # Intrinsic value & time value ratios
    if is_call:
        intrinsic = max(0, underlying_price - strike)
    else:
        intrinsic = max(0, strike - underlying_price)
    intrinsic_ratio = min(intrinsic / max(last_price, 0.01), 10.0)
    time_value = max(last_price - intrinsic, 0)
    time_value_ratio = time_value / max(last_price, 0.01)

    # --- NEW MICROSTRUCTURE FEATURES ---
    # 1. Strike Roundness
    strike_mod_5 = 1 if (strike % 5 == 0) else 0
    strike_mod_1 = 1 if (strike % 1 == 0) else 0

    # 2. Tick Size Regime
    is_low_price = 1 if (last_price < 3.0) else 0

    # 3. Advanced Greeks (Vanna / Volga / Cash Gamma)
    # Re-calculate parameters for d1/d2
    T = max(time_to_expire_years, 0.001)
    sigma = max(iv, 0.001)
    r = 0.05
    sqrt_T = np.sqrt(T)
    
    d1 = (np.log(underlying_price / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = 0.39894228 * np.exp(-0.5 * d1 * d1) # 1/sqrt(2pi) approx 0.3989
    
    vanna = -pdf_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    cash_gamma = gamma * (underlying_price ** 2) / 100.0
    
    # Build feature array (must match _XGB_FEATURE_NAMES order)
    features = np.array([[moneyness, time_to_expire_years, iv, gamma, vega, delta, theta,
                          int(is_call), dist_from_atm, inv_price,
                          log_price, sqrt_time, iv_moneyness, iv_squared, time_iv,
                          intrinsic_ratio, time_value_ratio,
                          
                          strike_mod_5, strike_mod_1, is_low_price,
                          vanna, volga, cash_gamma]])
                          
    dtest = xgb.DMatrix(features, feature_names=_XGB_FEATURE_NAMES)
    
    # Predict the RESIDUAL (correction)
    residual_correction = model.predict(dtest)[0]
    return residual_correction

# Latest trained coefficients for the Baseline Layer
# IMPORTANT: These must match the coefficients used in train_xgboost_model.py
# because the XGBoost model predicts residuals relative to this baseline.
POLY_COEFFS = {
    'b0': 0.093480, 'b1': -0.153181, 'b2': 0.012322, 
    'b3': 0.012915, 'b4': 0.071723, 'b5': 0.002246, 'b6': -0.042443
}

def estimate_fair_value(last_price, strike, underlying_price, days_to_expire, option_type='call'):
    """
    Estimate fair bid/ask spread for an option.
    
    Production-ready: only requires inputs available from a pricing model.
    
    Args:
        last_price: Option fair price (estimated by another model)
        strike: Option strike price
        underlying_price: Current price of the underlying asset
        days_to_expire: Calendar days until expiration
        option_type: 'call' or 'put'
    """
    # 1. Calculate Factors
    if days_to_expire <= 0:
        return None
        
    time_years = days_to_expire / 365.0
    moneyness = underlying_price / strike
    
    # 2. Calculate IV and Greeks
    r = 0.05
    is_call = 1 if option_type.lower() == 'call' else 0
    iv = fast_iv(last_price, underlying_price, strike, time_years, r, is_call)
    
    if iv <= 0.001: iv = 0.3  # Fallback if IV fails
    if iv > 5.0: iv = 5.0     # Clamp to match training range [0.001, 5.0]
    
    # Calculate all Greeks in one pass (avoids redundant d1/d2 computation)
    if CPP_LIB_AVAILABLE:
        theta = fast_theta(underlying_price, strike, time_years, r, iv, is_call)
        delta = fast_delta(underlying_price, strike, time_years, r, iv, is_call)
        gamma = fast_gamma(underlying_price, strike, time_years, r, iv)
        vega = fast_vega(underlying_price, strike, time_years, r, iv)
    else:
        delta, gamma, vega, theta = _python_greeks_all(
            underlying_price, strike, time_years, r, iv, is_call)
    
    # 3. Layered Prediction
    # Layer 1: Polynomial Baseline
    baseline_spread = predict_spread_poly(moneyness, time_years, iv, last_price)
    
    # Layer 2: XGBoost Correction (production-only features)
    correction = predict_spread_xgb(moneyness, time_years, iv, gamma, vega, delta, theta,
                                     is_call, last_price, strike, underlying_price)
    
    if correction is not None:
        predicted_spread = max(0.01, baseline_spread + correction)
        method = "Layered (Poly + XGB Correction)"
    else:
        predicted_spread = baseline_spread
        method = "Polynomial (Baseline Only)"
    
    # 4. Derive Bid/Ask (centered around fair price)
    fair_bid = last_price - (predicted_spread / 2)
    fair_ask = last_price + (predicted_spread / 2)
    
    return {
        "strike": strike,
        "underlying": underlying_price,
        "moneyness": moneyness,
        "days": days_to_expire,
        "iv": iv,
        "gamma": gamma,
        "vega": vega,
        "spread": predicted_spread,
        "bid": fair_bid,
        "ask": fair_ask,
        "method": method
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Fair Bid/Ask using Advanced Machine Learning")
    parser.add_argument("last_price", type=float, help="Option Fair Price (from pricing model)")
    parser.add_argument("strike", type=float, help="Option Strike Price")
    parser.add_argument("underlying", type=float, help="Underlying Asset Price")
    parser.add_argument("days_to_expire", type=float, help="Days to Expiration")
    parser.add_argument("--type", type=str, default="call", choices=['call', 'put'], help="Option Type")
    
    args = parser.parse_args()
    
    res = estimate_fair_value(args.last_price, args.strike, args.underlying, args.days_to_expire, args.type)
    
    if res:
        print("\n--- Advanced Fair Value Estimation ---")
        print(f"Option:            {args.type.upper()} Strike ${args.strike:.2f}")
        print(f"Underlying:        ${res['underlying']:.2f} (Moneyness: {res['moneyness']:.2f})")
        print(f"Time:              {res['days']} days")
        print("-" * 30)
        print(f"Implied Vol (IV):  {res['iv']:.2%}")
        print(f"Gamma:             {res['gamma']:.4f}")
        print(f"Vega:              {res['vega']:.4f}")
        print("-" * 30)
        print(f"Method Used:       {res['method']}")
        print(f"Predicted Spread:  ${res['spread']:.2f}")
        print("-" * 30)
        print(f"Fair Bid:          ${res['bid']:.2f}")
        print(f"Fair Ask:          ${res['ask']:.2f}")

