# option
Download option data using yfinance

## IV Prediction Model (`iv_surface_prod.json`)
Recent improvements were made to the Implied Volatility (IV) prediction model (`train_prod_model.py` and `pricing.py`):
- **Data Filtering**: Excluded extreme IVs (outside 1% to 150%) and near-expiration options (DTE < 0.9).
- **Target Transformation**: Predicting `log(IV)` to stabilize variance.
- **Feature Engineering**: Added features like `inv_dte`, `vix_x_log_moneyness`, VIX-based `atm_iv_proxy`, and explicit buckets for Moneyness (Deep ITM, ITM, ATM, OTM, Deep OTM) and DTE.
- **Training Strategy**: Used full historical dataset (80/20 train/val split) with early stopping.
- **Performance**: Significant reduction in error (RMSE ~2.77, MAE ~1.01). Error analysis reveals that Deep ITM or very short-dated (<15d) options carry the highest variance, while ATM and longer-dated predictions are highly accurate.
- **Model Tuning**: Hyperparameters can now be optimized with Optuna by using `--tune` and `--tune-trials <N>`.

## Usage
### Training the Model
Train the core models and optionally optimize hyperparameters using Optuna:
```bash
python train_prod_model.py --models xgb lgb cb ydf --tune --tune-trials 30
```

### Pricing a Specific Option (`pricing.py`)
Predict Implied Volatility and Option Greeks (Delta, Gamma, Vega, Theta) for a specific underlying price, strike, DTE, and VIX. You can specify a known ticker via the `-t` argument for model precision:
```bash
python pricing.py <underlying> <strike> <days> <vix> -t aapl
```

### Pricing an Option Chain (`pricing-form.py`)
Predict IV, Price, and Greeks for a chain of strikes around the given underlying price:
```bash
python pricing-form.py <underlying> <days> <vix> -t sp500
```
