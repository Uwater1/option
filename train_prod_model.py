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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import optuna
import ydf
import lightgbm as lgb
import catboost as cb
import subprocess

def detect_cuda():
    """Detect if CUDA is available via nvidia-smi."""
    try:
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return True
    except:
        return False

CUDA_AVAILABLE = detect_cuda()

# Reuse functions from pricing.py / price_options.py logic
# For training script, we need robust regex parsers

MODEL_FILE = "iv_surface_prod.json"

# ── Asset-class lookup tables ────────────────────────────────────────────────
# Keys are the lowercase ticker prefix used in CSV filenames; values are the
# canonical Yahoo Finance / brokerage symbol (kept for reference).
COMMODITY_TICKERS = {"gold", "silver", "longterm"}   # gold, silver, LongTerm-bond ETF
STOCK_TICKERS     = {"aapl", "amzn", "goog"}
INDEX_TICKERS     = {"sp500", "nq100", "dowjones"}
KNOWN_TICKERS = sorted(COMMODITY_TICKERS | STOCK_TICKERS | INDEX_TICKERS)
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
                trade_date_utc = pd.to_datetime(df['lastTradeDate'], unit='s', utc=True)

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
    df['is_otm'] = ((df['otm_amount'] > 0.02) & (df['otm_amount'] <= threshold)).astype(int)
    df['is_deep_otm'] = (df['otm_amount'] > threshold).astype(int)
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
        'is_stock', 'is_index', 'is_commodity',
    ] + [f'ticker_{t}' for t in KNOWN_TICKERS]
    
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
        ticker = _ticker_from_filename(f)
        if ticker not in KNOWN_TICKERS:
            return None

        if f.endswith('.parquet'):
            # Parquet files are already structured, no need for header check
            df = pd.read_parquet(f)
        else:
            # Quick check for required columns in header before full parse for CSV
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
        for t in KNOWN_TICKERS:
            df[f'ticker_{t}'] = int(ticker == t)
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
            
        # Try parquet first
        files = glob.glob(os.path.join(path, "*.parquet"))
        if not files:
            # Fall back to CSV if no parquet files
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

# ── Optuna Tuning Functions ──────────────────────────────────────────────────

def tune_xgb(X_train, y_train, X_val, y_val, n_trials=50):
    """Tune XGBoost hyperparameters via Optuna."""
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 3000,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'early_stopping_rounds': 100,
            'eval_metric': 'rmse',
        }
        if CUDA_AVAILABLE:
            params.update({'tree_method': 'hist', 'device': 'cuda'})
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return root_mean_squared_error(y_val, model.predict(X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', study_name='xgb_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def tune_lgb(X_train, y_train, X_val, y_val, n_trials=50):
    """Tune LightGBM hyperparameters via Optuna."""
    def objective(trial):
        params = {
            'n_estimators': 3000,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'verbosity': -1,
        }
        if CUDA_AVAILABLE:
            params.update({'device': 'gpu'})
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        return root_mean_squared_error(y_val, model.predict(X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', study_name='lgb_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def tune_cb(X_train, y_train, X_val, y_val, n_trials=50):
    """Tune CatBoost hyperparameters via Optuna."""
    def objective(trial):
        params = {
            'iterations': 3000,
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'early_stopping_rounds': 50,
            'verbose': False,
        }
        if CUDA_AVAILABLE:
            params.update({'task_type': 'GPU'})
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        return root_mean_squared_error(y_val, model.predict(X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', study_name='cb_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="options_data")
    parser.add_argument("--models", nargs="+", default=["xgb"], 
                        choices=["xgb", "lgb", "cb", "ydf", "all"],
                        help="Specify which models to train. 'all' trains all 4.")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before training.")
    parser.add_argument("--tune-trials", type=int, default=30,
                        help="Number of Optuna trials per model (default: 30).")
    args = parser.parse_args()
    
    # Resolve 'all' target
    if "all" in args.models:
        train_targets = {"xgb", "lgb", "cb", "ydf"}
        if CUDA_AVAILABLE:
            print("CUDA detected and '--models all' requested. Skipping CPU-primarily models (ydf) to optimize for GPU runtime.")
            train_targets.discard("ydf")
    else:
        train_targets = set(args.models)
    
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
    # Drop exact duplicates caused by options with no new trades appearing in
    # multiple date directories (same contractSymbol + lastTradeDate → identical row).
    if 'contractSymbol' in full_df.columns and 'lastTradeDate' in full_df.columns:
        full_df = full_df.drop_duplicates(subset=['contractSymbol', 'lastTradeDate'])
    else:
        full_df = full_df.drop_duplicates()
    
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
    
    # 4. Train models
    print("\n--- Training Phase ---")
    models = {}
    predictions = {}

    # --- XGBoost ---
    if "xgb" in train_targets:
        xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 2000,
            'max_depth': 10,
            'learning_rate': 0.0335,
            'min_child_weight': 9,
            'subsample': 0.866,
            'colsample_bytree': 0.9315,
            'reg_lambda': 8.2831,
            'reg_alpha': 0.2753,
            'gamma': 0.0071,
            'early_stopping_rounds': 100,
            'eval_metric': 'rmse',
        }
        if CUDA_AVAILABLE:
            xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
        if args.tune:
            print(f"\n⏳ Tuning XGBoost ({args.tune_trials} trials)...")
            best = tune_xgb(X_train, y_train, X_val, y_val, n_trials=args.tune_trials)
            xgb_params.update(best)
            print(f"  ✅ XGBoost best params:")
            for k, v in best.items():
                print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")

        print("\nTraining XGBoost...")
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        model_xgb.save_model("iv_prod_xgb.json")
        models['xgb'] = model_xgb
        predictions['xgb'] = model_xgb.predict(X_val)
        print("XGBoost saved to iv_prod_xgb.json")

    # --- LightGBM ---
    if "lgb" in train_targets:
        lgb_params = {
            'max_depth': 10,
            'learning_rate': 0.0495,
            'num_leaves': 163,
            'min_child_samples': 11,
            'subsample': 0.7260,
            'colsample_bytree': 0.5424,
            'reg_lambda': 9.0980,
            'reg_alpha': 0.0227,
        }
        if CUDA_AVAILABLE:
            lgb_params.update({'device': 'gpu'})
        if args.tune:
            print(f"\n⏳ Tuning LightGBM ({args.tune_trials} trials)...")
            best = tune_lgb(X_train, y_train, X_val, y_val, n_trials=args.tune_trials)
            lgb_params.update(best)
            print(f"  ✅ LightGBM best params:")
            for k, v in best.items():
                print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")

        print("\nTraining LightGBM...")
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        model_lgb.booster_.save_model("iv_prod_lgb.txt")
        models['lgb'] = model_lgb
        predictions['lgb'] = model_lgb.predict(X_val)
        print("LightGBM saved to iv_prod_lgb.txt")

    # --- CatBoost ---
    if "cb" in train_targets:
        cb_params = {
            'iterations': 2000,
            'depth': 8,
            'learning_rate': 0.2468,
            'l2_leaf_reg': 2.3247,
            'subsample': 0.7155,
            'colsample_bylevel': 0.9375,
            'min_data_in_leaf': 50,
            'early_stopping_rounds': 100,
            'verbose': False,
        }
        if CUDA_AVAILABLE:
            cb_params.update({'task_type': 'GPU'})
        if args.tune:
            print(f"\n⏳ Tuning CatBoost ({args.tune_trials} trials)...")
            best = tune_cb(X_train, y_train, X_val, y_val, n_trials=args.tune_trials)
            cb_params.update(best)
            print(f"  ✅ CatBoost best params:")
            for k, v in best.items():
                print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")

        print("\nTraining CatBoost...")
        model_cb = cb.CatBoostRegressor(**cb_params)
        model_cb.fit(X_train, y_train, eval_set=(X_val, y_val))
        model_cb.save_model("iv_prod_cb.cbm")
        models['cb'] = model_cb
        predictions['cb'] = model_cb.predict(X_val)
        print("CatBoost saved to iv_prod_cb.cbm")

    # --- YDF ---
    if "ydf" in train_targets:
        train_ds = X_train.copy()
        train_ds['target'] = y_train
        val_ds = X_val.copy()
        val_ds['target'] = y_val

        learner_kwargs = {
            'label': "target",
            'task': ydf.Task.REGRESSION,
            'num_trees': 2000,
            'early_stopping_num_trees_look_ahead': 50,
        }
        print("\nTraining YDF (no tuner)...")

        model_ydf = ydf.GradientBoostedTreesLearner(**learner_kwargs).train(train_ds, valid=val_ds)

        model_ydf.save("iv_prod_ydf")
        models['ydf'] = model_ydf
        predictions['ydf'] = model_ydf.predict(X_val)
        print("YDF saved to directory iv_prod_ydf")

    # 5. Evaluate on the full validation set
    print("\n" + "=" * 65)
    print(f"{'Model':<15} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10} | {'Huber Loss'}")
    print("-" * 65)
    
    results = []
    y_val_actual = np.exp(y_val)
    
    for name, pred_log in predictions.items():
        y_val_pred = np.exp(pred_log)
        
        mse = mean_squared_error(y_val_actual, y_val_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_actual, y_val_pred)
        r2 = r2_score(y_val_actual, y_val_pred)
        huber = huber_loss(y_val_actual, y_val_pred, delta=1.0)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Huber': huber,
            'y_val_pred': y_val_pred
        })
        print(f"{name:<15} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | {huber:.4f}")
        
    print("=" * 65)

    best_model = min(results, key=lambda x: x['RMSE'])
    best_name = best_model['Model']
    print(f"\n=> Best Model based on RMSE is: {best_name}")

    # 6. Error analysis for the best model
    val_df = X_val.copy()
    val_df['actual_IV'] = y_val_actual.values
    val_df['predicted_IV'] = best_model['y_val_pred']
    val_df['abs_error'] = np.abs(val_df['predicted_IV'] - val_df['actual_IV'])
    
    print(f"\n--- Top 10 Worst Predictions ({best_name}) ---")
    worst = val_df.nlargest(10, 'abs_error')[['log_moneyness', 'days', 'vix', 'is_put', 'actual_IV', 'predicted_IV', 'abs_error']]
    print(worst.to_string(index=False))
    
    # Per-bucket error analysis
    print(f"\n--- Error by Moneyness Bucket ({best_name}) ---")
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
    
    print(f"\n--- Error by DTE Bucket ({best_name}) ---")
    for name, mask in [('<15d', val_df['dte_under_15'] == 1), ('15-40d', val_df['dte_15_to_40'] == 1), ('>40d', val_df['dte_over_40'] == 1)]:
        subset = val_df[mask]
        if len(subset) > 0:
            bucket_mae = np.mean(subset['abs_error'])
            bucket_rmse = np.sqrt(np.mean(subset['abs_error'] ** 2))
            print(f"  {name:10s}  n={len(subset):5d}  MAE={bucket_mae:.4f}  RMSE={bucket_rmse:.4f}")

if __name__ == "__main__":
    main()
