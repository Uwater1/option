import argparse
import os
import numpy as np
import pandas as pd
import re
import glob
import multiprocessing as mp
import json
import pytz
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constants from train_prod_model.py
COMMODITY_TICKERS = {"gold", "silver", "longterm"}
STOCK_TICKERS     = {"aapl", "amzn", "goog"}
INDEX_TICKERS     = {"sp500", "nq100", "dowjones"}
KNOWN_TICKERS = sorted(COMMODITY_TICKERS | STOCK_TICKERS | INDEX_TICKERS)

IRLS_C                 = 1.345
IRLS_BASE_THRESHOLD    = 3.5
IRLS_MIN_ABS_DEV       = 3.0
IRLS_MIN_POINTS        = 10
IRLS_MAX_ITER          = 100
IRLS_L2_PENALTY        = 1e-4
IV_MIN                 = 2.0
IV_MAX                 = 150.0

# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA PIPELINE (Copied from train_prod_model.py to avoid heavy imports)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_data(df):
    df = df.copy()
    if 'strike' in df.columns and 'strikePrice' not in df.columns:
        df.rename(columns={'strike': 'strikePrice'}, inplace=True)
    if 'contractSymbol' in df.columns:
        extracted = df['contractSymbol'].astype(str).str.extract(r'(\d{6})([CP])')
        if not extracted.empty and extracted.shape[1] == 2:
            df['expiry_str'] = extracted[0]
            df['type_str'] = extracted[1]
            df['expiry_dt'] = pd.to_datetime(df['expiry_str'], format='%y%m%d', errors='coerce')
            df['is_put'] = (df['type_str'] == 'P').astype(int)
            if 'lastTradeDate' in df.columns:
                trade_date_utc = pd.to_datetime(df['lastTradeDate'], utc=True)
                _ny_tz = pytz.timezone('America/New_York')
                def _localize_expiry(naive_dt):
                    if pd.isna(naive_dt): return pd.NaT
                    dt = naive_dt.replace(hour=16, minute=0, second=0, microsecond=0)
                    return pd.Timestamp(_ny_tz.localize(dt.to_pydatetime()).astimezone(pytz.utc))
                expiry_utc = df['expiry_dt'].apply(_localize_expiry)
                df['daysToExpiration'] = (expiry_utc - trade_date_utc).dt.total_seconds() / 86400.0
    return df

def filter_arbitrage_irls(df: pd.DataFrame) -> pd.DataFrame:
    needed = {'strikePrice', 'underlyingPriceAtTrade', 'daysToExpiration', 'impliedVolatility'}
    if not needed.issubset(df.columns): return df
    work = df.copy()
    for c in needed: work[c] = pd.to_numeric(work[c], errors='coerce')
    work = work.dropna(subset=list(needed))
    work = work[(work['daysToExpiration'] > 0) & (work['underlyingPriceAtTrade'] > 0)]
    if len(work) < IRLS_MIN_POINTS: return df
    m = np.log(work['strikePrice'] / work['underlyingPriceAtTrade']).values
    t = np.sqrt(work['daysToExpiration']).values
    y = work['impliedVolatility'].values
    X = np.column_stack([np.ones(len(m)), m, m ** 2, t, t ** 2, m * t, m ** 2 * t])
    w = np.exp(-2.0 * np.abs(m))
    beta = np.zeros(X.shape[1])
    I_reg = np.eye(X.shape[1]); I_reg[0, 0] = 0
    for _ in range(IRLS_MAX_ITER):
        W = np.diag(w); XtW = X.T @ W
        try: beta_new = np.linalg.solve(XtW @ X + IRLS_L2_PENALTY * I_reg, XtW @ y)
        except np.linalg.LinAlgError: return df
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new; break
        beta = beta_new
        residuals = y - X @ beta; mad = np.median(np.abs(residuals))
        mad_scale = 1.4826 * mad if mad > 1e-10 else 1.0; r = residuals / mad_scale
        abs_r = np.abs(r); w = np.where(abs_r < IRLS_C, 1.0, IRLS_C / np.maximum(abs_r, 1e-10))
    final_residuals = y - X @ beta; mad = np.median(np.abs(final_residuals))
    mad_scale = 1.4826 * mad if mad > 1e-10 else 1.0; std_residuals = np.abs(final_residuals) / mad_scale
    dyn_threshold = IRLS_BASE_THRESHOLD + 1.5 * np.abs(m) + 0.1 / np.maximum(t, 0.05)
    keep_mask = (std_residuals <= dyn_threshold) | (np.abs(final_residuals) <= IRLS_MIN_ABS_DEV) | (np.abs(m) <= 0.02)
    return df.loc[df.index.isin(work.index[keep_mask])]

def process_file(f):
    try:
        ticker = os.path.basename(f).split('_')[0].lower()
        if ticker not in KNOWN_TICKERS: return None
        
        if f.endswith('.parquet'):
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
            
        df = enrich_data(df)
        if 'lastPrice' in df.columns: df = df[df['lastPrice'] > 0.01]
        if 'volume' in df.columns: df = df[df['volume'] >= 10]
        if 'impliedVolatility' in df.columns:
            df = df[(df['impliedVolatility'] > IV_MIN) & (df['impliedVolatility'] < IV_MAX)]
        if 'daysToExpiration' in df.columns: df = df[df['daysToExpiration'] >= 1.1]
        df = filter_arbitrage_irls(df)
        required = ['strikePrice', 'underlyingPriceAtTrade', 'daysToExpiration', 'volatilityIndex', 'is_put', 'impliedVolatility']
        if not all(col in df.columns for col in required): return None
        return df.dropna(subset=required)
    except: return None

def load_data_from_range(data_dir, date_range):
    all_files = []
    for date_str in date_range:
        path = os.path.join(data_dir, date_str)
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "*.parquet"))
            if not files:
                files = glob.glob(os.path.join(path, "*.csv"))
            all_files.extend(files)
    print(f"Loading {len(all_files)} files...")
    all_data = []
    with mp.Pool(mp.cpu_count()) as pool:
        for df in pool.imap_unordered(process_file, all_files, chunksize=100):
            if df is not None: all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLIFIED FEATURE PREP
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features_lin(df):
    df = df.copy()
    df.rename(columns={'strikePrice':'strike', 'underlyingPriceAtTrade':'underlying', 
                       'daysToExpiration':'days', 'volatilityIndex':'vix'}, inplace=True)
    
    # Calculate base features
    df['log_moneyness'] = np.log(df['strike'] / df['underlying'])
    df['moneyness_sq'] = df['log_moneyness'] ** 2
    df['abs_log_moneyness'] = np.abs(df['log_moneyness'])
    
    df['sqrt_dte'] = np.sqrt(np.maximum(df['days'], 0.001))
    df['inv_dte'] = 1.0 / np.maximum(df['days'], 0.001)
    df['inv_sqrt_dte'] = 1.0 / np.sqrt(np.maximum(df['days'], 1e-4))
    
    df['log_vix'] = np.log(np.maximum(df['vix'], 1.0))
    # df['vix_sq'] = df['vix'] ** 2  # Dropped per request
    
    df['vix_x_dte'] = df['vix'] * df['sqrt_dte']
    df['vix_x_log_moneyness'] = df['vix'] * df['log_moneyness']
    df['log_moneyness_x_sqrt_dte'] = df['log_moneyness'] * df['sqrt_dte']
    
    features = [
        'log_moneyness', 'moneyness_sq', 'abs_log_moneyness',
        'days', 'sqrt_dte', 'inv_dte', 'inv_sqrt_dte',
        'vix', 'log_vix', 'vix_x_dte', 'vix_x_log_moneyness',
        'log_moneyness_x_sqrt_dte'
    ]
    return df[features], features

def train_and_export(X, y, name):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    
    model = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train_sc, y_train)
    
    pred_val = model.predict(X_val_sc)
    rmse = np.sqrt(mean_squared_error(np.exp(y_val), np.exp(pred_val)))
    mae = mean_absolute_error(np.exp(y_val), np.exp(pred_val))
    r2 = r2_score(y_val, pred_val)
    
    print(f"\n{name} Model Performance: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    return {
        'weights': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="options_data")
    args = parser.parse_args()
    
    all_dates = sorted([d for d in os.listdir(args.data_dir) 
                        if os.path.isdir(os.path.join(args.data_dir, d)) 
                        and re.match(r'^\d{4}-\d{2}-\d{2}$', d)])
    
    print(f"Training on {len(all_dates)} dates...")
    full_df = load_data_from_range(args.data_dir, all_dates)
    if 'contractSymbol' in full_df.columns and 'lastTradeDate' in full_df.columns:
        full_df = full_df.drop_duplicates(subset=['contractSymbol', 'lastTradeDate'])
    
    # 1. Feature Engineering (Simplified)
    X_all, feature_names = prepare_features_lin(full_df)
    y_all = np.log(full_df['impliedVolatility'])
    
    # 2. Split Call/Put
    is_put = full_df['is_put'] == 1
    
    print("\nTraining Call Model...")
    call_params = train_and_export(X_all[~is_put], y_all[~is_put], "CALL")
    
    print("\nTraining Put Model...")
    put_params = train_and_export(X_all[is_put], y_all[is_put], "PUT")
    
    # 3. Export to JSON
    export_data = {
        'features': feature_names,
        'call': call_params,
        'put': put_params
    }
    
    with open("iv_lin_params.json", "w") as f:
        json.dump(export_data, f, indent=2)
    print("\nSaved parameters to iv_lin_params.json")

if __name__ == "__main__":
    main()
