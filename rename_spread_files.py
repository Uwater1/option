"""
Rename spread CSV files to the full convention used by download_option_spread.py:

    TICKER_YYYYMMDD_HH-PPP_pp-VVV_vv.csv

where PPP_pp is the underlying price at HH:00 NYT that day,
  and VVV_vv is the corresponding volatility index (e.g. VIX) value.

Handles two input forms:
  1. TICKER_YYYYMMDD_HH.csv          (no suffixes yet)
  2. TICKER_YYYYMMDD_HH-PPP_pp.csv   (price already present, VIX missing)

Skips files that already have both suffixes.
"""

import os
import re
import pytz
import pandas as pd
import yfinance as yf
from datetime import datetime

SPREAD_ROOT = os.path.join(os.path.dirname(__file__), "spread")
NY_TZ = pytz.timezone("America/New_York")

VOLATILITY_MAP = {
    "SPY": "^VIX",
    "QQQ": "^VXN",
    "AMZN": "^VXAZN",
    "AAPL": "^VXAPL",
    "GOOG": "^VXGOG",
    "SLV": "^VXSLV",
    "GLD": "^GVZ",
}

_intraday_cache = {}


def get_intraday_data(ticker_symbol: str) -> pd.DataFrame:
    if ticker_symbol in _intraday_cache:
        return _intraday_cache[ticker_symbol]
    tk = yf.Ticker(ticker_symbol)
    hist = tk.history(period="5d", interval="1m")
    if not hist.empty:
        hist.index = pd.to_datetime(hist.index, utc=True)
    _intraday_cache[ticker_symbol] = hist
    return hist


def get_price_at_hour(ticker: str, date_str: str, hour: int) -> str | None:
    """
    Fetch the 1-minute close price of `ticker` at `hour`:00 NYT on `date_str`
    (YYYYMMDD).  Returns a string like '245_67', or None on failure.
    """
    try:
        target_naive = datetime.strptime(f"{date_str} {hour:02d}:00:00", "%Y%m%d %H:%M:%S")
        target_dt = NY_TZ.localize(target_naive).astimezone(pytz.utc)
        target_ts = pd.Timestamp(target_dt)

        hist = get_intraday_data(ticker)
        if hist.empty:
            return None

        idx = hist.index.get_indexer([target_ts], method="nearest")[0]
        if idx < 0:
            return None

        price = float(hist["Close"].iloc[idx])
        return f"{price:.2f}".replace(".", "_")
    except Exception as e:
        print(f"  [warn] Could not fetch price for {ticker} at {date_str} {hour:02d}h: {e}")
        return None


def main():
    # Form 1: TICKER_YYYYMMDD_HH.csv  (no suffixes)
    pattern_no_suffix = re.compile(r"^([A-Z]+)_(\d{8})_(\d{2})\.csv$")
    # Form 2: TICKER_YYYYMMDD_HH-PPP_pp.csv  (price present, VIX missing)
    pattern_price_only = re.compile(r"^([A-Z]+)_(\d{8})_(\d{2})-([\d_]+)\.csv$")
    # Form 3 (fully renamed): TICKER_YYYYMMDD_HH-PPP_pp-VVV_vv.csv  → skip
    pattern_full = re.compile(r"^([A-Z]+)_(\d{8})_(\d{2})-[\d_]+-[\d_]+\.csv$")

    for dirpath, _, filenames in os.walk(SPREAD_ROOT):
        for fname in filenames:
            if not fname.endswith(".csv"):
                continue

            # Already fully renamed — skip
            if pattern_full.match(fname):
                continue

            price_str = None  # will be fetched or extracted below

            m1 = pattern_no_suffix.match(fname)
            m2 = pattern_price_only.match(fname)

            if m1:
                ticker, date_str, hour_str = m1.group(1), m1.group(2), m1.group(3)
                # price not yet in filename — fetch it
                price_str = None
            elif m2:
                ticker, date_str, hour_str = m2.group(1), m2.group(2), m2.group(3)
                price_str = m2.group(4)  # already embedded in filename
            else:
                continue  # unrecognised format

            hour = int(hour_str)
            print(f"Processing {fname} ...")

            # --- underlying price (fetch only if not already known) ---
            if price_str is None:
                price_str = get_price_at_hour(ticker, date_str, hour)
                if price_str is None:
                    print(f"  Skipping {fname} — could not determine underlying price.")
                    continue

            # --- volatility index ---
            vol_symbol = VOLATILITY_MAP.get(ticker)
            vix_str = None
            if vol_symbol:
                vix_str = get_price_at_hour(vol_symbol, date_str, hour)
                if vix_str is None:
                    print(f"  [warn] Could not fetch VIX ({vol_symbol}) for {fname}; omitting VIX suffix.")

            # Build new filename: TICKER_YYYYMMDD_HH-PPP_pp[-VVV_vv].csv
            if vix_str:
                new_fname = f"{ticker}_{date_str}_{hour_str}-{price_str}-{vix_str}.csv"
            else:
                new_fname = f"{ticker}_{date_str}_{hour_str}-{price_str}.csv"

            old_path = os.path.join(dirpath, fname)
            new_path = os.path.join(dirpath, new_fname)
            os.rename(old_path, new_path)
            print(f"  {fname}  →  {new_fname}")


if __name__ == "__main__":
    main()
