import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
import glob
import re
from datetime import datetime

MODEL_FILE = "iv_surface_model.json"

import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import norm
import glob
import re
from datetime import datetime
import math
from numba import jit, vectorize, float64

MODEL_FILE = "iv_surface_model.json"

@jit(nopython=True)
def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1 + math.erf(x / 1.4142135623730951))

@jit(nopython=True)
def norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return 0.3989422804014327 * math.exp(-0.5 * x * x)

@vectorize([float64(float64, float64, float64, float64, float64, float64)], target='parallel')
def black_scholes_numba(S, K, T, r, sigma, is_put):
    """
    Vectorized Black-Scholes pricing using Numba.
    is_put: 1.0 for put, 0.0 for call
    """
    if T <= 1e-6:
        # Intrinsic value at expiry
        if is_put == 1.0:
            return max(0.0, K - S)
        else:
            return max(0.0, S - K)
            
    # Avoid zero/negative volatility/prices
    if sigma <= 0 or S <= 0 or K <= 0:
         return 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if is_put == 0.0:
        # Call
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        # Put
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        
    return price

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Wrapper for Black-Scholes option price.
    """
    is_put = 1.0 if option_type == "put" else 0.0
    return black_scholes_numba(float(S), float(K), float(T), float(r), float(sigma), is_put)

@jit(nopython=True)
def calculate_greeks_numba(S, K, T, r, sigma, is_put):
    """
    Calculate Option Greeks using Numba.
    Returns a tuple: (delta, gamma, vega, theta, rho)
    """
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
        # Call
        delta = norm_cdf(d1)
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
        theta = (-S * nd1 * sigma / (2 * sqrt_T) 
                 - r * K * math.exp(-r * T) * norm_cdf(d2))
    else:
        # Put
        delta = -norm_cdf(-d1)
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)
        theta = (-S * nd1 * sigma / (2 * sqrt_T) 
                 + r * K * math.exp(-r * T) * norm_cdf(-d2))

    gamma = nd1 / (S * sigma * sqrt_T)
    vega = S * sqrt_T * nd1 / 100.0  # Divide by 100
    
    theta = theta / 365.0 

    return delta, gamma, vega, theta, rho

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Wrapper for Greeks calculation.
    """
    is_put = 1.0 if option_type == "put" else 0.0
    d, g, v, t, rho = calculate_greeks_numba(float(S), float(K), float(T), float(r), float(sigma), is_put)
    return {
        "Delta": d,
        "Gamma": g,
        "Vega": v,
        "Theta": t,
        "Rho": rho
    }

def parse_contract_symbol(symbol):
    """
    Parse contract symbol to extract expiration date and type.
    Format: Ticker + YYMMDD + Type + Strike
    Example: AAPL260220P00100000
    """
    match = re.search(r'(\d{6})([CP])', symbol)
    if match:
        date_str = match.group(1)
        type_str = match.group(2)
        
        # Parse date
        expiry = datetime.strptime(date_str, "%y%m%d")
        
        # Parse type
        is_put = 1 if type_str == 'P' else 0
        
        return expiry, is_put
    return None, None

def enrich_data(df):
    """
    Calculate daysToExpiration and is_put if not present.
    Also handles column renaming.
    """
    df = df.copy()
    
    # Rename 'strike' to 'strikePrice' if needed
    if 'strike' in df.columns and 'strikePrice' not in df.columns:
        df.rename(columns={'strike': 'strikePrice'}, inplace=True)
        
    # Check if we have contractSymbol to parse
    if 'contractSymbol' in df.columns:
        # Vectorized string operations are much faster than applying a function row-by-row
        # Extract date part and type part
        # Format: Ticker + YYMMDD + Type + Strike
        # Regex: ...(\d{6})([CP])...
        
        # We need to target the 6 digits before C/P
        # Assuming contractSymbol is standard OCC format or similar
        # AAPL260220P00100000 -> 260220 is date, P is type
        
        extracted = df['contractSymbol'].astype(str).str.extract(r'(\d{6})([CP])')
        
        if not extracted.empty and extracted.shape[1] == 2:
            df['expiry_str'] = extracted[0]
            df['type_str'] = extracted[1]
            
            # Parse dates
            df['expiry_dt'] = pd.to_datetime(df['expiry_str'], format='%y%m%d', errors='coerce')
            
            # Parse type
            df['is_put'] = (df['type_str'] == 'P').astype(int)
            
            # Parse trade date for DTE calculation
            if 'lastTradeDate' in df.columns:
                df['trade_date_dt'] = pd.to_datetime(df['lastTradeDate'], utc=True).dt.tz_convert(None)
                
                # Calculate days to expiry
                df['daysToExpiration'] = (df['expiry_dt'] - df['trade_date_dt']).dt.days
    
    return df

def prepare_features(df):
    """
    Prepare features for the XGBoost model.
    """
    df = df.copy()
    
    # Ensure numerical types and drop rows with issues
    cols = ['strikePrice', 'underlyingPriceAtTrade', 'daysToExpiration', 'volatilityIndex', 'is_put']
    
    # Check existence to avoid KeyError, though caller should check
    missing = [c for c in cols if c not in df.columns]
    if missing:
        # If running from batch/price command, we might be missing some columns if not provided
        # But for training, we filtered already.
        pass

    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df = df.dropna(subset=[c for c in cols if c in df.columns])
    
    # Calculations
    if all(c in df.columns for c in ['strikePrice', 'underlyingPriceAtTrade']):
        df['log_moneyness'] = np.log(df['strikePrice'] / df['underlyingPriceAtTrade'])
        df['moneyness_sq'] = df['log_moneyness'] ** 2
    
    if 'daysToExpiration' in df.columns:
        # Avoid negative sqrt
        df = df[df['daysToExpiration'] >= 0]
        df['sqrt_dte'] = np.sqrt(df['daysToExpiration'])
    
    if 'volatilityIndex' in df.columns:
        df['vix_sq'] = df['volatilityIndex'] ** 2
        if 'sqrt_dte' in df.columns:
            df['vix_x_dte'] = df['volatilityIndex'] * df['sqrt_dte']
    
    features = [
        'log_moneyness', 
        'moneyness_sq', 
        'daysToExpiration', 
        'sqrt_dte', 
        'is_put', 
        'volatilityIndex', 
        'vix_sq', 
        'vix_x_dte'
    ]
    
    # Return only available features? XGBoost needs exact features.
    # For training, we guarantee all.
    return df[features]

def train(args):
    print(f"Searching for data in {args.data_dir}...")
    files = glob.glob(os.path.join(args.data_dir, "**/*.csv"), recursive=True)
    print(f"Found {len(files)} CSV files.")
    
    all_data = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            
            # Enrich data (calculate daysToExpiration, etc)
            df = enrich_data(df)

            required_cols = ['lastPrice', 'volume', 'underlyingPriceAtTrade', 'volatilityIndex', 'impliedVolatility', 'strikePrice', 'daysToExpiration', 'is_put']
            
            # Debug missing columns
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                # print(f"Skipping {f}: Missing columns {missing}")
                continue

            # Filter low quality data
            if 'lastPrice' in df.columns:
                df = df[df['lastPrice'] > 0.01]
            if 'volume' in df.columns:
                df = df[df['volume'] >= 5]
            
            df = df.dropna(subset=['underlyingPriceAtTrade', 'volatilityIndex', 'impliedVolatility'])
            
            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            # print(f"Error processing {f}: {e}") # Reduce noise
            continue

    if not all_data:
        print("No valid data found to train on.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Training on {len(full_df)} records...")
    
    X = prepare_features(full_df)
    # y is impliedVolatility. 
    # Make sure to align indices because prepare_features might drop rows (e.g. days<0)
    y = full_df.loc[X.index, 'impliedVolatility']
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X, y)
    
    model.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    print("\nFeature Importances:")
    importances = model.feature_importances_
    for name, imp in zip(X.columns, importances):
        print(f"{name}: {imp:.4f}")

def price_option(args):
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found. Run 'train' first.")
        return

    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    is_put = 1 if args.type == "put" else 0
    
    # Create DataFrame
    data = {
        'underlyingPriceAtTrade': [args.underlying],
        'strikePrice': [args.strike],
        'daysToExpiration': [args.days],
        'volatilityIndex': [args.vix],
        'is_put': [is_put]
    }
    df = pd.DataFrame(data)
    
    X = prepare_features(df)
    predicted_iv = model.predict(X)[0]
    
    # Convert IV % to decimal for BS
    sigma = predicted_iv / 100.0
    T = args.days / 365.0
    
    bs_price = black_scholes(
        args.underlying, 
        args.strike, 
        T, 
        args.rate, 
        sigma, 
        args.type
    )
    
    greeks = calculate_greeks(
        args.underlying, 
        args.strike, 
        T, 
        args.rate, 
        sigma, 
        args.type
    )
    
    print("\n---------------------------------")
    print(f"Option: {args.type.upper()} {args.strike} (Expires in {args.days} days)")
    print(f"Underlying: {args.underlying}")
    print(f"VIX: {args.vix}")
    print(f"Predicted IV: {predicted_iv:.2f}%")
    print(f"Estimated Price: ${bs_price:.2f}")
    print("---------------------------------")
    print("Greeks:")
    for k, v in greeks.items():
        print(f"  {k}: {v:.4f}")
    print("---------------------------------")

def batch_price(args):
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found. Run 'train' first.")
        return

    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"File {args.file} not found.")
        return
        
    df = enrich_data(df)
    
    # User overrides (broadcast to all rows)
    if args.underlying is not None:
         df['underlyingPriceAtTrade'] = args.underlying
    if args.vix is not None:
         df['volatilityIndex'] = args.vix
    
    # Ensure is_put is set if not from contract symbol
    if 'is_put' not in df.columns:
         # Try to guess or fail? 
         # Assuming batch files processed via enrich_data() work because of contractSymbol
         pass

    X = prepare_features(df)
    
    # Predict
    # Only predict for rows that survived prepare_features
    
    # Let's subset original df to match X index
    df_subset = df.loc[X.index].copy()
    
    predicted_ivs = model.predict(X)
    df_subset['predicted_iv'] = predicted_ivs
    
    # Vectorized Black-Scholes using Numba
    T_arr = df_subset['daysToExpiration'].values / 365.0
    sigma_arr = df_subset['predicted_iv'].values / 100.0
    S_arr = df_subset['underlyingPriceAtTrade'].values
    K_arr = df_subset['strikePrice'].values
    
    # Ensure all arrays are float64 for Numba
    T_arr = T_arr.astype(np.float64)
    sigma_arr = sigma_arr.astype(np.float64)
    S_arr = S_arr.astype(np.float64)
    K_arr = K_arr.astype(np.float64)
    
    # Handle option type
    if 'is_put' in df_subset.columns:
        is_put_arr = df_subset['is_put'].values.astype(np.float64)
    else:
        # Default to call if not found (though enrich_data should handle it)
        is_put_arr = np.zeros(len(df_subset), dtype=np.float64)
        
    r_arr = np.full(len(df_subset), args.rate, dtype=np.float64)
    
    # Call vectorized function
    prices = black_scholes_numba(S_arr, K_arr, T_arr, r_arr, sigma_arr, is_put_arr)
    
    df_subset['model_price'] = prices
    
    # print results
    pd.set_option('display.max_rows', 20)
    print(df_subset[['strikePrice', 'lastPrice', 'predicted_iv', 'model_price']].head(20))

def main():
    parser = argparse.ArgumentParser(description="Option Pricing with XGBoost IV Surface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train the IV surface model")
    train_parser.add_argument("--data-dir", default="options_data", help="Directory containing option data")
    
    # Price
    price_parser = subparsers.add_parser("price", help="Price a single option")
    price_parser.add_argument("--underlying", type=float, required=True, help="Underlying price")
    price_parser.add_argument("--strike", type=float, required=True, help="Strike price")
    price_parser.add_argument("--days", type=int, required=True, help="Days to expiration")
    price_parser.add_argument("--rate", type=float, required=True, help="Risk-free rate")
    price_parser.add_argument("--vix", type=float, required=True, help="Volatility Index")
    price_parser.add_argument("--type", choices=["call", "put"], required=True, help="Option type")

    # Batch
    batch_parser = subparsers.add_parser("batch", help="Batch price options from a file")
    batch_parser.add_argument("--file", required=True, help="CSV file with options")
    batch_parser.add_argument("--underlying", type=float, required=True, help="Underlying price")
    batch_parser.add_argument("--rate", type=float, required=True, help="Risk-free rate")
    batch_parser.add_argument("--vix", type=float, required=True, help="Volatility Index")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "price":
        price_option(args)
    elif args.command == "batch":
        batch_price(args)

if __name__ == "__main__":
    main()
