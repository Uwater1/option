# option
Download option data using yfinance

## IV Prediction Model (`iv_surface_prod.json`)
Recent improvements were made to the Implied Volatility (IV) prediction model (`train_prod_model.py` and `pricing.py`):
- **Data Filtering**: Excluded extreme IVs (outside 1% to 150%) and near-expiration options (DTE < 0.9).
- **Target Transformation**: Predicting `log(IV)` to stabilize variance.
- **Feature Engineering**: Added features like `inv_dte`, `vix_x_log_moneyness`, VIX-based `atm_iv_proxy`, and explicit buckets for Moneyness (Deep ITM, ITM, ATM, OTM, Deep OTM) and DTE.
- **Training Strategy**: Used full historical dataset (80/20 train/val split) with early stopping.
- **Performance**: Significant reduction in error (RMSE ~2.77, MAE ~1.01). Error analysis reveals that Deep ITM or very short-dated (<15d) options carry the highest variance, while ATM and longer-dated predictions are highly accurate.
