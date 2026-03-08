import argparse
import os
import numpy as np
import re
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Reuse heavy data pipeline from the production training script
from train_prod_model import (
    load_data_from_range, prepare_features, huber_loss,
)

MODEL_FILE = "iv_lin_elasticnet.joblib"


def main():
    parser = argparse.ArgumentParser(description="Train Elastic Net IV model (lightweight)")
    parser.add_argument("--data-dir", default="options_data")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning.")
    parser.add_argument("--tune-trials", type=int, default=80,
                        help="Number of Optuna trials (default: 80).")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="ElasticNet alpha (default: 0.001)")
    parser.add_argument("--l1-ratio", type=float, default=0.5,
                        help="ElasticNet l1_ratio (default: 0.5)")
    args = parser.parse_args()

    # 1. Load data
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

    # Deduplicate
    if 'contractSymbol' in full_df.columns and 'lastTradeDate' in full_df.columns:
        full_df = full_df.drop_duplicates(subset=['contractSymbol', 'lastTradeDate'])
    else:
        full_df = full_df.drop_duplicates()

    # 2. Prepare features
    X_full, feature_names = prepare_features(full_df)
    X = X_full[feature_names]
    y_raw = full_df.loc[X.index, 'impliedVolatility']
    y = np.log(y_raw)  # log-transform target (same as tree models)

    # 3. Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # 4. Scale features (critical for linear models)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # 5. Optional Optuna tuning
    alpha = args.alpha
    l1_ratio = args.l1_ratio

    if args.tune:
        import optuna
        print(f"\n⏳ Tuning ElasticNet ({args.tune_trials} trials)...")

        def objective(trial):
            a = trial.suggest_float('alpha', 1e-5, 1.0, log=True)
            lr = trial.suggest_float('l1_ratio', 0.01, 1.0)
            model = ElasticNet(alpha=a, l1_ratio=lr, max_iter=10000)
            model.fit(X_train_sc, y_train)
            pred = model.predict(X_val_sc)
            return np.sqrt(mean_squared_error(y_val, pred))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', study_name='elasticnet_tune')
        study.optimize(objective, n_trials=args.tune_trials, show_progress_bar=True)
        alpha = study.best_params['alpha']
        l1_ratio = study.best_params['l1_ratio']
        print(f"  ✅ Best alpha: {alpha:.6f}, l1_ratio: {l1_ratio:.4f}")

    # 6. Train final model
    print(f"\n--- Training ElasticNet (alpha={alpha}, l1_ratio={l1_ratio}) ---")
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train_sc, y_train)

    # 7. Save model + scaler + feature names
    artifact = {'model': model, 'scaler': scaler, 'features': feature_names}
    joblib.dump(artifact, MODEL_FILE)
    print(f"Saved to {MODEL_FILE} ({os.path.getsize(MODEL_FILE):,} bytes)")

    # 8. Evaluate
    pred_log = model.predict(X_val_sc)
    y_val_actual = np.exp(y_val)
    y_val_pred = np.exp(pred_log)

    rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
    mae = mean_absolute_error(y_val_actual, y_val_pred)
    r2 = r2_score(y_val_actual, y_val_pred)
    huber = huber_loss(y_val_actual, y_val_pred, delta=1.0)

    print(f"\n{'='*55}")
    print(f"{'Model':<15} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10} | {'Huber'}")
    print(f"{'-'*55}")
    print(f"{'ElasticNet':<15} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | {huber:.4f}")
    print(f"{'='*55}")

    # Non-zero coefficients
    n_nonzero = np.sum(model.coef_ != 0)
    print(f"\nNon-zero coefficients: {n_nonzero}/{len(feature_names)}")
    for fname, coef in sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True):
        if coef != 0:
            print(f"  {fname:30s}  {coef:+.6f}")

    # 9. Per-bucket error analysis
    val_df = X_val.copy()
    val_df['actual_IV'] = y_val_actual.values
    val_df['predicted_IV'] = y_val_pred
    val_df['abs_error'] = np.abs(val_df['predicted_IV'] - val_df['actual_IV'])

    print(f"\n--- Error by Moneyness Bucket ---")
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

    print(f"\n--- Error by DTE Bucket ---")
    for name, mask in [('<15d', val_df['dte_under_15'] == 1), ('15-40d', val_df['dte_15_to_40'] == 1), ('>40d', val_df['dte_over_40'] == 1)]:
        subset = val_df[mask]
        if len(subset) > 0:
            bucket_mae = np.mean(subset['abs_error'])
            bucket_rmse = np.sqrt(np.mean(subset['abs_error'] ** 2))
            print(f"  {name:10s}  n={len(subset):5d}  MAE={bucket_mae:.4f}  RMSE={bucket_rmse:.4f}")


if __name__ == "__main__":
    main()
