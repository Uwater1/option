# Option Pricing Script — ML Volatility Surface

## Problem

Using `volatilityIndex` (VIX) directly as σ in Black-Scholes is inaccurate because:
1. **Term structure**: Near-term options (<20 days) have different IV than longer-dated ones
2. **Volatility smile/skew**: Deep OTM/ITM options have higher IV than ATM
3. **Put premium**: Puts are systematically more expensive than calls at the same strike

## Solution: XGBoost IV Surface Model

Train an **XGBoost** regressor to predict the actual Black-Scholes IV (`impliedVolatility` column) from structural features, then use that predicted IV in Black-Scholes to price options.

---

## Proposed Changes

### New Script

#### [NEW] [price_options.py](file:///home/hallo/Documents/option/price_options.py)

**Three sub-commands:**

```
# 1. Train the IV surface model (run once, saves model to disk)
python price_options.py train [--data-dir options_data]

# 2. Price a single option using the trained model
python price_options.py price \
    --underlying 263.88 --strike 260 --days 275 \
    --rate 0.0364 --vix 30.5 --type put

# 3. Price all options in a CSV file (batch mode)
python price_options.py batch \
    --file options_data/2026-02-17/aapl_20261120_puts_263_88.csv \
    --underlying 263.88 --rate 0.0364 --vix 30.5
```

---

### Training Data & Features

Load all CSV files from [options_data/](file:///home/hallo/Documents/option/volatility_viewer.py#22-76) that have valid `impliedVolatility` and `volatilityIndex`.

Filter out low quality data:
* lastPrice > 0.01
* volume >= 5
* underlyingPriceAtTrade, volatilityIndex must exist

| Feature | Description | Captures |
|---|---|---|
| `log_moneyness` | `log(K / S)` | Smile / skew shape |
| `moneyness_sq` | `log(K/S)²` | Smile curvature |
| `days_to_expiry` | Calendar days until expiration | Term structure |
| `sqrt_dte` | `√(days_to_expiry)` | Non-linear time decay |
| `is_put` | 1 for put, 0 for call | Put premium |
| `vix` | `volatilityIndex` at trade time | Market-wide vol level |
| `vix_sq` | `vix²` | Non-linear VIX sensitivity |
| `vix_x_dte` | `vix × sqrt_dte` | VIX × term interaction |

**Target:** `impliedVolatility` (Black-Scholes IV in %, already in the CSVs)

**Model saved to:** `iv_surface_model.json` (XGBoost native format)

---

### Pricing Logic

After predicting `σ_pred` from the model:

```
Call = S·N(d1) − K·e^{−rT}·N(d2)
Put  = K·e^{−rT}·N(−d2) − S·N(−d1)

d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T
```

Also outputs Greeks: **Δ, Γ, ν (Vega), Θ, ρ**.

---

## Verification Plan

```bash
cd /home/hallo/Documents/option && source venv/bin/activate

# 1. Train the model
python price_options.py train

# 2. Price a single ATM put (should be close to market lastPrice ~19.87)
python price_options.py price \
    --underlying 266.025 --strike 260 --days 275 \
    --rate 0.0364 --vix 30.29 --type put

# 3. Batch price the AAPL Nov 2026 puts and compare to lastPrice
python price_options.py batch \
    --file options_data/2026-02-17/aapl_20261120_puts_263_88.csv \
    --underlying 263.88 --rate 0.0364 --vix 30.5
```

**Expected:**
- Train: prints feature importances (log_moneyness and is_put should rank high)
- Price: predicted IV should be ~28–30% for near-ATM, higher for deep OTM
- Batch: predicted prices should be close to `lastPrice` in the CSV
