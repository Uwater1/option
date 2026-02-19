import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import glob
import re
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Reuse functions from pricing.py / price_options.py logic
# For training script, we need robust regex parsers

MODEL_FILE = "iv_surface_prod.json"

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
    Mapping to 'pricing.py' feature names:
    pricing.py expects: strike, underlying, days, vix, is_put
    
    BUT the model trained here must match what pricing.py uses for prediction.
    pricing.py:
        df['log_moneyness'] = np.log(df['strike'] / df['underlying'])
        ...
        ['log_moneyness', 'moneyness_sq', 'days', 'sqrt_dte', 'is_put', 'vix', 'vix_sq', 'vix_x_dte']
    
    So we must map our training data columns to these names.
    Training Data cols: strikePrice, underlyingPriceAtTrade, daysToExpiration, volatilityIndex, is_put
    """
    df = df.copy()
    
    # Rename training cols to match pricing.py expected input cols
    # strikePrice -> strike
    # underlyingPriceAtTrade -> underlying
    # daysToExpiration -> days
    # volatilityIndex -> vix
    
    rename_map = {
        'strikePrice': 'strike',
        'underlyingPriceAtTrade': 'underlying',
        'daysToExpiration': 'days',
        'volatilityIndex': 'vix' 
    }
    df = df.rename(columns=rename_map)
    
    # Ensure numerical types
    cols = ['strike', 'underlying', 'days', 'vix', 'is_put']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df = df.dropna(subset=cols)
    
    # Now use exact logic from pricing.py
    df['log_moneyness'] = np.log(df['strike'] / df['underlying'])
    df['moneyness_sq'] = df['log_moneyness'] ** 2
    
    df = df[df['days'] >= 0]
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
    
    return df, features

def load_data_from_range(data_dir, date_range):
    """
    Load data from directories matching the date range.
    date_range: list of strings 'YYYY-MM-DD'
    """
    all_data = []
    
    for date_str in date_range:
        path = os.path.join(data_dir, date_str)
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist.")
            continue
            
        print(f"Loading data from {date_str}...")
        files = glob.glob(os.path.join(path, "*.csv"))
        
        for f in files:
            try:
                df = pd.read_csv(f)
                df = enrich_data(df)
                
                # Basic filtering
                if 'lastPrice' in df.columns:
                     df = df[df['lastPrice'] > 0.01]
                if 'volume' in df.columns:
                     df = df[df['volume'] >= 5]

                required = ['strikePrice', 'underlyingPriceAtTrade', 'daysToExpiration', 'volatilityIndex', 'is_put', 'impliedVolatility']
                if not all(col in df.columns for col in required):
                    continue
                    
                df = df.dropna(subset=required)
                
                if len(df) > 0:
                    all_data.append(df)
            except:
                continue
                
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="options_data")
    args = parser.parse_args()
    
    # 1. Define Date Ranges
    train_dates = pd.date_range(start="2026-02-10", end="2026-02-17").strftime("%Y-%m-%d").tolist()
    test_dates = ["2026-02-18"]
    
    print("--- Training Phase ---")
    train_df = load_data_from_range(args.data_dir, train_dates)
    if train_df.empty:
        print("No training data found!")
        return
        
    print(f"Training samples: {len(train_df)}")
    
    X_train_full, feature_names = prepare_features(train_df)
    X_train = X_train_full[feature_names]
    y_train = train_df.loc[X_train.index, 'impliedVolatility']
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=8, learning_rate=0.05)
    model.fit(X_train, y_train)
    model.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    # 2. Testing Phase
    print("\n--- Testing Phase ---")
    test_df = load_data_from_range(args.data_dir, test_dates)
    if test_df.empty:
        print("No testing data found!")
        return

    print(f"Testing samples: {len(test_df)}")
    
    X_test_full, _ = prepare_features(test_df)
    X_test = X_test_full[feature_names]
    y_test = test_df.loc[X_test.index, 'impliedVolatility']
    
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Quality Metrics (Test Set 2026-02-18):")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Error analysis
    test_df_subset = test_df.loc[X_test.index].copy()
    test_df_subset['predicted_IV'] = y_pred
    test_df_subset['error'] = test_df_subset['predicted_IV'] - test_df_subset['impliedVolatility']
    
    print("\nSample Predictions:")
    print(test_df_subset[['strikePrice', 'daysToExpiration', 'impliedVolatility', 'predicted_IV', 'error']].head(10))

if __name__ == "__main__":
    main()
