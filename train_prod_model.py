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
    df = df.copy()
    
    rename_map = {
        'strikePrice': 'strike',
        'underlyingPriceAtTrade': 'underlying',
        'daysToExpiration': 'days',
        'volatilityIndex': 'vix' 
    }
    df = df.rename(columns=rename_map)
    
    cols = ['strike', 'underlying', 'days', 'vix', 'is_put']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=cols)
    df = df[df['days'] >= 0.9]
    
    df['log_moneyness'] = np.log(df['strike'] / df['underlying'])
    df['moneyness_sq'] = df['log_moneyness'] ** 2
    
    df['sqrt_dte'] = np.sqrt(np.maximum(df['days'], 0.001))
    df['inv_dte'] = 1.0 / np.maximum(df['days'], 0.001)
    
    df['vix_sq'] = df['vix'] ** 2
    df['vix_x_dte'] = df['vix'] * df['sqrt_dte']
    df['vix_x_log_moneyness'] = df['vix'] * df['log_moneyness']
    
    # Bucketed Moneyness
    # Normalized OTM amount: Positive = OTM, Negative = ITM
    # Call: log_mon > 0 is OTM -> log_mon
    # Put: log_mon < 0 is OTM -> -log_mon
    df['otm_amount'] = df['log_moneyness'] * (1 - 2 * df['is_put'])
    
    # 5-Bucket Moneyness
    df['is_atm'] = (np.abs(df['otm_amount']) <= 0.02).astype(int)
    df['is_otm'] = ((df['otm_amount'] > 0.02) & (df['otm_amount'] <= 0.1)).astype(int)
    df['is_deep_otm'] = (df['otm_amount'] > 0.1).astype(int)
    df['is_itm'] = ((df['otm_amount'] < -0.02) & (df['otm_amount'] >= -0.1)).astype(int)
    df['is_deep_itm'] = (df['otm_amount'] < -0.1).astype(int)
    
    # Bucketed DTE
    df['dte_under_15'] = (df['days'] < 15).astype(int)
    df['dte_15_to_40'] = ((df['days'] >= 15) & (df['days'] <= 40)).astype(int)
    df['dte_over_40'] = (df['days'] > 40).astype(int)
    
    # ATM IV proxy from VIX (matches pricing.py)
    df['atm_iv_proxy'] = df['vix'] / 100.0
    
    features = [
        'log_moneyness', 'moneyness_sq', 'days', 'sqrt_dte', 'inv_dte', 
        'is_put', 'vix', 'vix_sq', 'vix_x_dte', 'vix_x_log_moneyness',
        'is_atm', 'is_otm', 'is_deep_otm', 'is_itm', 'is_deep_itm',
        'dte_under_15', 'dte_15_to_40', 'dte_over_40',
        'atm_iv_proxy'
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
                     df = df[df['volume'] >= 10]

                # Filter extreme IVs and near-expiration options
                if 'impliedVolatility' in df.columns:
                    df = df[(df['impliedVolatility'] > 1.0) & (df['impliedVolatility'] < 150.0)]
                if 'daysToExpiration' in df.columns:
                    df = df[df['daysToExpiration'] >= 0.9]

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

def huber_loss(y_true, y_pred, delta=1.0):
    """Calculate Huber loss."""
    residual = np.abs(y_true - y_pred)
    quadratic = np.minimum(residual, delta)
    linear = residual - quadratic
    return np.mean(0.5 * quadratic ** 2 + delta * linear)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="options_data")
    args = parser.parse_args()
    
    # 1. Load full dataset (Dynamic listing from data directory)
    all_dates = sorted([
        d for d in os.listdir(args.data_dir) 
        if os.path.isdir(os.path.join(args.data_dir, d)) and re.match(r'^\d{4}-\d{2}-\d{2}$', d)
    ])
    
    if not all_dates:
        print(f"Error: No date directories found in {args.data_dir}")
        return
    
    print(f"Found {len(all_dates)} dates: {all_dates[0]} to {all_dates[-1]}")    
    print("--- Loading Data ---")
    full_df = load_data_from_range(args.data_dir, all_dates)
    if full_df.empty:
        print("No data found!")
        return
    print(f"Total samples after filtering: {len(full_df)}")
    
    # 2. Prepare features
    X_full, feature_names = prepare_features(full_df)
    X = X_full[feature_names]
    y_raw = full_df.loc[X.index, 'impliedVolatility']
    
    # Log-transform the target
    y = np.log(y_raw)
    
    # 3. Train/Validation split (80/20, shuffled)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # 4. Train with early stopping
    print("\n--- Training Phase ---")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    model.save_model(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Best iteration: {model.best_iteration}")
    
    # 5. Evaluate on the full validation set
    y_val_pred_log = model.predict(X_val)
    y_val_pred = np.exp(y_val_pred_log)
    y_val_actual = np.exp(y_val)
    
    mse = mean_squared_error(y_val_actual, y_val_pred)
    mae = mean_absolute_error(y_val_actual, y_val_pred)
    r2 = r2_score(y_val_actual, y_val_pred)
    huber = huber_loss(y_val_actual, y_val_pred, delta=1.0)
    
    print("\n" + "=" * 55)
    print("Model Quality Metrics (Validation Set):")
    print("=" * 55)
    print(f"RMSE:       {np.sqrt(mse):.4f}")
    print(f"MAE:        {mae:.4f}")
    print(f"R²:         {r2:.4f}")
    print(f"Huber Loss: {huber:.4f}  (delta=1.0)")
    
    # 6. Error analysis
    val_df = X_val.copy()
    val_df['actual_IV'] = y_val_actual.values
    val_df['predicted_IV'] = y_val_pred
    val_df['abs_error'] = np.abs(val_df['predicted_IV'] - val_df['actual_IV'])
    
    print("\n--- Top 10 Worst Predictions ---")
    worst = val_df.nlargest(10, 'abs_error')[['log_moneyness', 'days', 'vix', 'is_put', 'actual_IV', 'predicted_IV', 'abs_error']]
    print(worst.to_string(index=False))
    
    # Per-bucket error analysis
    print("\n--- Error by Moneyness Bucket ---")
    for name, mask in [
        ('Deep ITM', val_df['is_deep_itm'] == 1),
        ('ITM', val_df['is_itm'] == 1),
        ('ATM', val_df['is_atm'] == 1),
        ('OTM', val_df['is_otm'] == 1),
        ('Deep OTM', val_df['is_deep_otm'] == 1)
    ]:
        subset = val_df[mask]
        if len(subset) > 0:
            bucket_mae = np.mean(subset['abs_error'])
            bucket_rmse = np.sqrt(np.mean(subset['abs_error'] ** 2))
            print(f"  {name:10s}  n={len(subset):5d}  MAE={bucket_mae:.4f}  RMSE={bucket_rmse:.4f}")
    
    print("\n--- Error by DTE Bucket ---")
    for name, mask in [('<15d', val_df['dte_under_15'] == 1), ('15-40d', val_df['dte_15_to_40'] == 1), ('>40d', val_df['dte_over_40'] == 1)]:
        subset = val_df[mask]
        if len(subset) > 0:
            bucket_mae = np.mean(subset['abs_error'])
            bucket_rmse = np.sqrt(np.mean(subset['abs_error'] ** 2))
            print(f"  {name:10s}  n={len(subset):5d}  MAE={bucket_mae:.4f}  RMSE={bucket_rmse:.4f}")

if __name__ == "__main__":
    main()
