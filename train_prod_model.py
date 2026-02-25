import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import glob
import re
import multiprocessing as mp
from datetime import datetime
import pytz
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Reuse functions from pricing.py / price_options.py logic
# For training script, we need robust regex parsers

MODEL_FILE = "iv_surface_prod.json"

# ── Asset-class lookup tables ────────────────────────────────────────────────
# Keys are the lowercase ticker prefix used in CSV filenames; values are the
# canonical Yahoo Finance / brokerage symbol (kept for reference).
COMMODITY_TICKERS = {"gold", "silver", "longterm"}   # gold, silver, LongTerm-bond ETF
STOCK_TICKERS     = {"aapl", "amzn", "goog"}
INDEX_TICKERS     = {"sp500", "nq100", "dowjones"}
# ─────────────────────────────────────────────────────────────────────────────

# ── IRLS Surface Filter ──────────────────────────────────────────────────────
IRLS_C                 = 1.345   # Huber tuning constant
IRLS_BASE_THRESHOLD    = 3.5     # base max |std residual| (relaxed from 2.5)
IRLS_MIN_ABS_DEV       = 3.0     # min IV points deviation to be an outlier
IRLS_MIN_POINTS        = 10      # skip fit if snapshot has fewer points
IRLS_MAX_ITER          = 50      # max IRLS iterations
IRLS_L2_PENALTY        = 1e-4    # Ridge regularization for stability
IV_MIN                 = 2.0     # tightened lower bound (was 1.0)
IV_MAX                 = 150.0   # unchanged upper bound
# ─────────────────────────────────────────────────────────────────────────────

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
                trade_date_utc = pd.to_datetime(df['lastTradeDate'], utc=True)

                # Options expire at 16:00 New York Time — localize correctly to handle DST
                _ny_tz = pytz.timezone('America/New_York')

                def _localize_expiry(naive_dt):
                    if pd.isna(naive_dt):
                        return pd.NaT
                    dt = naive_dt.replace(hour=16, minute=0, second=0, microsecond=0)
                    return pd.Timestamp(
                        _ny_tz.localize(dt.to_pydatetime()).astimezone(pytz.utc)
                    )

                expiry_utc = df['expiry_dt'].apply(_localize_expiry)

                # Use fractional days (via total_seconds) to preserve intra-day precision
                df['daysToExpiration'] = (
                    (expiry_utc - trade_date_utc).dt.total_seconds() / 86400.0
                )
    
    return df


def filter_arbitrage_irls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a robust parametric IV surface via IRLS per (ticker, trade_date) snapshot.

    Surface model (7 terms):
        IV ≈ β₀ + β₁·m + β₂·m² + β₃·t + β₄·t² + β₅·m·t + β₆·m²·t
    where m = log(strike/underlying)  and  t = sqrt(DTE).

    Points with |standardized Huber-MAD residual| > dyn_threshold AND
    absolute residual > IRLS_MIN_ABS_DEV are removed as arbitrage-consistent outliers.
    ATM options (|m| <= 0.02) are strictly preserved.
    """
    # ── Identify required columns ────────────────────────────────────────────
    needed = {'strikePrice', 'underlyingPriceAtTrade', 'daysToExpiration',
               'impliedVolatility'}
    if not needed.issubset(df.columns):
        return df

    work = df.copy()
    for c in needed:
        work[c] = pd.to_numeric(work[c], errors='coerce')
    work = work.dropna(subset=list(needed))
    work = work[work['daysToExpiration'] > 0]
    work = work[work['underlyingPriceAtTrade'] > 0]

    if len(work) < IRLS_MIN_POINTS:
        return df  # too few points to fit 7 parameters reliably

    # ── Build design matrix ──────────────────────────────────────────────────
    m = np.log(work['strikePrice'] / work['underlyingPriceAtTrade']).values
    t = np.sqrt(work['daysToExpiration']).values
    y = work['impliedVolatility'].values

    X = np.column_stack([
        np.ones(len(m)),   # β₀ intercept
        m,                  # β₁ skew
        m ** 2,             # β₂ smile
        t,                  # β₃ term level
        t ** 2,             # β₄ term curvature
        m * t,              # β₅ skew × term
        m ** 2 * t,         # β₆ smile × term
    ])

    # ── IRLS with Huber weights ──────────────────────────────────────────────
    w = np.exp(-2.0 * np.abs(m))  # initial weights inversely prop to moneyness dist
    beta = np.zeros(X.shape[1])
    
    I_reg = np.eye(X.shape[1])
    I_reg[0, 0] = 0  # Do not regularize intercept

    for _ in range(IRLS_MAX_ITER):
        W = np.diag(w)
        XtW = X.T @ W
        try:
            beta_new = np.linalg.solve(XtW @ X + IRLS_L2_PENALTY * I_reg, XtW @ y)
        except np.linalg.LinAlgError:
            return df  # singular matrix — skip filter

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

        residuals = y - X @ beta
        mad = np.median(np.abs(residuals))
        mad_scale = 1.4826 * mad if mad > 1e-10 else 1.0  # robust sigma
        r = residuals / mad_scale

        # Huber weights: w = min(1, IRLS_C / |r|)
        abs_r = np.abs(r)
        w = np.where(abs_r < IRLS_C, 1.0, IRLS_C / np.maximum(abs_r, 1e-10))

    # ── Final standardized residuals for outlier detection ───────────────────
    final_residuals = y - X @ beta
    mad = np.median(np.abs(final_residuals))
    mad_scale = 1.4826 * mad if mad > 1e-10 else 1.0
    std_residuals = np.abs(final_residuals) / mad_scale

    # ── Map results back to original df index ────────────────────────────────
    # Dynamic threshold: relax for deep OTM/ITM and ultra-short DTE
    dyn_threshold = IRLS_BASE_THRESHOLD + 1.5 * np.abs(m) + 0.1 / np.maximum(t, 0.05)
    
    keep_mask = (std_residuals <= dyn_threshold) | (np.abs(final_residuals) <= IRLS_MIN_ABS_DEV)
    keep_mask = keep_mask | (np.abs(m) <= 0.02)  # Strictly save ATM options (+-2%)
    
    keep_idx = work.index[keep_mask]
    return df.loc[df.index.isin(keep_idx)]


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
    df = df[df['days'] >= 1.1]
    
    df['net_moneyness'] = df['strike'] / df['underlying']
    df['log_moneyness'] = np.log(df['net_moneyness'])
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
    threshold = df['vix'] * 0.004 + df['sqrt_dte'] * 0.001
    df['is_atm'] = (np.abs(df['otm_amount']) <= 0.02).astype(int)
    df['is_otm'] = ((df['otm_amount'] > 0.02) & (df['otm_amount'] < threshold)).astype(int)
    df['is_deep_otm'] = (df['otm_amount'] >= threshold).astype(int)
    df['is_itm'] = ((df['otm_amount'] < -0.02) & (df['otm_amount'] > -threshold)).astype(int)
    df['is_deep_itm'] = (df['otm_amount'] <= -threshold).astype(int)
    
    # Bucketed DTE
    df['dte_under_15'] = (df['days'] < 15).astype(int)
    df['dte_15_to_40'] = ((df['days'] >= 15) & (df['days'] <= 40)).astype(int)
    df['dte_over_40'] = (df['days'] > 40).astype(int)
    
    # ATM IV proxy from VIX (matches pricing.py)
    df['atm_iv_proxy'] = df['vix'] / 100.0
    
    features = [
        'net_moneyness', 'log_moneyness', 'moneyness_sq', 'days', 'sqrt_dte', 'inv_dte',
        'is_put', 'vix', 'vix_sq', 'vix_x_dte', 'vix_x_log_moneyness',
        'is_atm', 'is_otm', 'is_deep_otm', 'is_itm', 'is_deep_itm',
        'dte_under_15', 'dte_15_to_40', 'dte_over_40',
        'atm_iv_proxy',
        # Asset-class flags
        'is_stock', 'is_index', 'is_commodity',
    ]
    
    return df, features

def _ticker_from_filename(filepath: str) -> str:
    """Extract the lowercase ticker prefix from a CSV filename.

    Expected naming convention: <ticker>_<YYYYMMDD>_<calls|puts>_*.csv
    e.g.  aapl_20260320_calls_263_88.csv  →  'aapl'
    """
    basename = os.path.basename(filepath)
    return basename.split('_')[0].lower()


def process_file(f):
    try:
        # Quick check for required columns in header before full parse
        with open(f, 'r') as file:
            header = file.readline()
        if 'volatilityIndex' not in header:
            return None

        df = pd.read_csv(f)
        df = enrich_data(df)

        # ── Asset-class features derived from filename ticker ────────────────
        ticker = _ticker_from_filename(f)
        df['is_commodity'] = int(ticker in COMMODITY_TICKERS)
        df['is_stock']     = int(ticker in STOCK_TICKERS)
        df['is_index']     = int(ticker in INDEX_TICKERS)
        # ────────────────────────────────────────────────────────────────────

        # Basic filtering
        if 'lastPrice' in df.columns:
            df = df[df['lastPrice'] > 0.01]
        if 'volume' in df.columns:
            df = df[df['volume'] >= 10]

        # Filter extreme IVs and near-expiration options
        if 'impliedVolatility' in df.columns:
            df = df[(df['impliedVolatility'] > IV_MIN) & (df['impliedVolatility'] < IV_MAX)]
        if 'daysToExpiration' in df.columns:
            df = df[df['daysToExpiration'] >= 1.1]

        # Robust surface filter: remove arbitrage-consistent outliers via IRLS
        df = filter_arbitrage_irls(df)

        required = ['strikePrice', 'underlyingPriceAtTrade', 'daysToExpiration',
                    'volatilityIndex', 'is_put', 'impliedVolatility']
        if not all(col in df.columns for col in required):
            return None

        df = df.dropna(subset=required)

        if len(df) > 0:
            return df
    except:
        pass
    return None

def load_data_from_range(data_dir, date_range):
    """
    Load data from directories matching the date range.
    date_range: list of strings 'YYYY-MM-DD'
    """
    all_files = []
    for date_str in date_range:
        path = os.path.join(data_dir, date_str)
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist.")
            continue
        files = glob.glob(os.path.join(path, "*.csv"))
        all_files.extend(files)
        
    print(f"Found {len(all_files)} files to process. Starting parallel load...")
    
    all_data = []
    with mp.Pool(mp.cpu_count()) as pool:
        for df in pool.imap_unordered(process_file, all_files, chunksize=100):
            if df is not None:
                all_data.append(df)
                
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
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        early_stopping_rounds=50,
        eval_metric='rmse',
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
