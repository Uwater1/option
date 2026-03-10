# option
Download option data using yfinance

## IV Prediction Model (`iv_prod_xgb.json`)
Recent improvements were made to the Implied Volatility (IV) prediction model (`train_prod_model.py` and `pricing.py`):
- **Asset Classes**: Added support for specific tickers (`aapl`, `gold`, `sp500`, etc.) and general asset classes (Stock, Index, Commodity).
- **Data Filtering**: Excluded extreme IVs (outside 1% to 150%) and near-expiration options (DTE < 1.1).
- **Target Transformation**: Predicting `log(IV)` to stabilize variance for better performance across different volatility regimes.
- **Feature Engineering**: Added features like `inv_dte`, `vix_x_log_moneyness`, VIX-based `atm_iv_proxy`, and explicit buckets for Moneyness and DTE.
- **Training Strategy**: Parallel loading of historical CSV data with 80/20 train/val split and Early Stopping.
- **Model Tuning**: Integrated **Optuna** for hyperparameter optimization across XGBoost, LightGBM, and CatBoost.

## Usage

Data are stored in Parquet format, in the `options_data` and `spread` directories.
You can see some examples in DATA.md

Ensure you are in the virtual environment:
```bash
source venv/bin/activate
```

### 1. Training the Model
Train one or more models (XGBoost, LightGBM, CatBoost, YDF). The best model for general prediction is saved as `iv_prod_xgb.json`.

```bash
# Train XGBoost with Optuna tuning
python train_prod_model.py --models xgb --tune --tune-trials 50

# Train all supported models
python train_prod_model.py --models all
```

### 2. Pricing a Specific Option (`pricing.py`)
Predicts IV, Price, and Greeks (Delta, Gamma, Vega, Theta, Rho) for a single strike.

**Syntax:**
```bash
python pricing.py <underlying> <strike> <days> <vix> [rate] [-t ticker]
```

**Example:**
```bash
python pricing.py 500 505 30 18.5 -t sp500
```

### 3. Pricing an Option Chain (`pricing-form.py`)
Generates a beautiful console-rendered option chain (10 strikes above/below ATM) with IV skew and Greeks for both Calls and Puts.

**Syntax:**
```bash
python pricing-form.py <underlying> <days> <vix> [rate] [-t ticker]
```

**Example:**
```bash
python pricing-form.py 150.5 45 22.1 -t aapl
```

---
**Note:** Supported tickers for the `-t` flag include: `gold`, `silver`, `longterm`, `aapl`, `amzn`, `goog`, `sp500`, `nq100`, `dowjones`. Use `stock`, `index`, or `commodity` for general asset class defaults.
