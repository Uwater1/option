# Options Data Structure

This repository manages financial options data for various tickers, stored efficiently in **Parquet** format.

## Directory Structure

- `options_data/`: Daily options snapshots downloaded at market close.
  - Organized by date: `options_data/YYYY-MM-DD/`
  - Filename pattern: `{ticker}_{expiration}_[calls|puts]_{underlying}_{vix}.parquet`
- `spread/`: Intra-day option spread data.
  - Organized by date and hour: `spread/YYYYMMDD_HH/`
  - Filename pattern: `{TICKER}_{YYYYMMDD_HH}-{underlying}-{vix}.parquet`

---

## 1. Options Data (`options_data/`)

This data represents the end-of-day state of the option chains.

### Schema Definitions

| Column | Description |
|--------|-------------|
| `contractSymbol` | Unique identifier for the option contract. |
| `lastTradeDate` | Timestamp of the last recorded trade for this contract. |
| `strike` | The strike price of the option. |
| `lastPrice` | The price at which the last trade occurred. |
| `bid` | Highest price a buyer is willing to pay. |
| `ask` | Lowest price a seller is willing to accept. |
| `volume` | Number of contracts traded during the session. |
| `openInterest` | Total number of open contracts. |
| `inTheMoney` | Boolean (0/1) indicating if the option is ITM. |
| `IV_yf` | Implied Volatility provided by Yahoo Finance. |
| `underlyingPriceAtTrade` | Price of the underlying asset at the time of the data capture. |
| `impliedVolatility` | Calculated Implied Volatility (Black-Scholes). |
| `riskFreeRate` | The risk-free interest rate used in calculations (e.g., ^IRX). |
| `volatilityIndex` | The market volatility index at the time (e.g., ^VIX). |

### Example Data (CSV Format)

```csv
contractSymbol,lastTradeDate,strike,lastPrice,bid,ask,volume,openInterest,inTheMoney,IV_yf,underlyingPriceAtTrade,impliedVolatility,riskFreeRate,volatilityIndex
AAPL260309C00200000,2026-03-09 16:16:57+00:00,200.0,57.73,58.3,61.5,2.0,1,1,182.81,257.72,,0.0364,32.39
AAPL260309C00215000,2026-03-06 20:59:07+00:00,215.0,42.36,43.3,46.5,1.0,1,1,135.94,257.44,,0.0364,33.71
AAPL260309C00220000,2026-03-09 17:36:12+00:00,220.0,37.13,38.3,40.85,4.0,3,1,223.44,256.235,204.103,0.0364,32.88
AAPL260309C00225000,2026-03-09 16:17:33+00:00,225.0,32.72,33.3,36.5,4.0,2,1,106.25,257.73,,0.0364,32.4
AAPL260309C00230000,2026-03-09 16:18:17+00:00,230.0,27.8,28.35,31.5,4.0,1,1,100.78,257.73,95.4261,0.0364,32.4
```

---

## 2. Spread Data (`spread/`)

This data focuses on liquidity and bid-ask spreads captured during trading hours.

### Schema Definitions

| Column | Description |
|--------|-------------|
| `contractSymbol` | Unique identifier for the option contract. |
| `lastTradeDate` | Timestamp of last trade. |
| `strike` | Option strike price. |
| `lastPrice` | Last traded price. |
| `underlyingPriceAtTrade` | Spot price of the underlying asset. |
| `volatilityIndex` | Market volatility (VIX). |
| `bid` | Current bid price. |
| `ask` | Current ask price. |
| `bid_ask_spread` | Calculated difference between ask and bid. |
| `days_to_expire` | Time remaining until expiration (in days). |
| `optionType` | 'c' for Call, 'p' for Put. |

### Example Data (CSV Format)

```csv
contractSymbol,lastTradeDate,strike,lastPrice,volume,openInterest,underlyingPriceAtTrade,volatilityIndex,bid,ask,bid_ask_spread,days_to_expire,optionType
AAPL260309C00305000,2026-02-23 19:16:51,305.0,0.04,,1,263.705,33.32,0.0,0.01,0.01,14.03,c
AAPL260309P00287500,2026-02-24 19:57:10,287.5,15.75,,0,263.705,33.32,29.7,31.85,2.15,13.002,p
AAPL260309P00300000,2026-02-25 19:26:59,300.0,25.4,,0,263.705,33.32,41.85,44.25,2.4,12.023,p
AAPL260309P00282500,2026-02-26 20:55:14,282.5,10.46,,1,263.705,33.32,24.7,26.85,2.15,10.962,p
AAPL260309P00277500,2026-02-27 16:07:53,277.5,9.35,70.0,10,263.705,33.32,19.7,21.85,2.15,10.161,p
```
