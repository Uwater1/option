"""
Rename spread CSV files from  TICKER_YYYYMMDD_HH.csv
                           to  TICKER_YYYYMMDD_HH_PPP_pp.csv
where PPP_pp is the underlying price at HH:00 NYT that day.

Only renames files that don't already have a price suffix (i.e. exactly 3
underscore-separated parts: TICKER, YYYYMMDD, HH).
"""

import os
import re
import pytz
import pandas as pd
import yfinance as yf
from datetime import datetime

SPREAD_ROOT = os.path.join(os.path.dirname(__file__), "spread")
NY_TZ = pytz.timezone("America/New_York")


def get_price_at_hour(ticker: str, date_str: str, hour: int) -> str | None:
    """
    Fetch the 1-minute close price of `ticker` at `hour`:00 NYT on `date_str`
    (YYYYMMDD).  Returns a string like '245_67', or None on failure.
    """
    try:
        target_naive = datetime.strptime(f"{date_str} {hour:02d}:00:00", "%Y%m%d %H:%M:%S")
        target_dt = NY_TZ.localize(target_naive).astimezone(pytz.utc)
        target_ts = pd.Timestamp(target_dt)

        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d", interval="1m")
        if hist.empty:
            return None

        hist.index = pd.to_datetime(hist.index, utc=True)
        idx = hist.index.get_indexer([target_ts], method="nearest")[0]
        if idx < 0:
            return None

        price = float(hist["Close"].iloc[idx])
        return f"{price:.2f}".replace(".", "_")
    except Exception as e:
        print(f"  [warn] Could not fetch price for {ticker} at {date_str} {hour:02d}h: {e}")
        return None


def main():
    # Pattern: exactly TICKER_YYYYMMDD_HH.csv  (no price yet)
    pattern = re.compile(r"^([A-Z]+)_(\d{8})_(\d{2})\.csv$")

    for dirpath, _, filenames in os.walk(SPREAD_ROOT):
        for fname in filenames:
            m = pattern.match(fname)
            if not m:
                continue  # already renamed or different format

            ticker, date_str, hour_str = m.group(1), m.group(2), m.group(3)
            hour = int(hour_str)

            print(f"Processing {fname} ...")
            price_str = get_price_at_hour(ticker, date_str, hour)
            if price_str is None:
                print(f"  Skipping {fname} — could not determine price.")
                continue

            new_fname = f"{ticker}_{date_str}_{hour_str}_{price_str}.csv"
            old_path = os.path.join(dirpath, fname)
            new_path = os.path.join(dirpath, new_fname)
            os.rename(old_path, new_path)
            print(f"  {fname}  →  {new_fname}")


if __name__ == "__main__":
    main()
