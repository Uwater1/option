import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import argparse
import time

# Import shared resources from download_options.py
from download_options import VOLATILITY_MAP, TICKERS, get_stock_price_at_time, _intraday_cache

def update_file(file_path, ticker_symbol, vol_symbol):
    """Adds volatilityIndex column to a single CSV file."""
    try:
        df = pd.read_csv(file_path)
        
        # Check if already updated
        if 'volatilityIndex' in df.columns:
            # Check for NaNs, if all are numeric, we can skip
            if not df['volatilityIndex'].isna().all():
                print(f"      Skipping {os.path.basename(file_path)} (already contains volatility index)")
                return False
        
        if 'lastTradeDate' not in df.columns:
            print(f"      Warning: lastTradeDate column missing in {os.path.basename(file_path)}")
            return False

        # Convert lastTradeDate to datetime
        df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'], utc=True)
        
        # Calculate volatility index values
        print(f"      Fetching volatility index ({vol_symbol}) for {len(df)} rows...")
        df['volatilityIndex'] = df['lastTradeDate'].apply(
            lambda x: get_stock_price_at_time(vol_symbol, x, np.nan) if pd.notna(x) else np.float32(np.nan)
        ).astype(np.float32)
        
        # Save back to CSV
        df.to_csv(file_path, index=False)
        print(f"      Successfully updated {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        print(f"      Error updating {os.path.basename(file_path)}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update historical options data with volatility index readings.")
    parser.add_argument("--date", type=str, help="Specific date folder to update (YYYY-MM-DD). If omitted, scans all.")
    parser.add_argument("--ticker", type=str, help="Specific ticker symbol to update (e.g., SPY). If omitted, scans all in map.")
    args = parser.parse_args()

    base_folder = "options_data"
    
    if args.date:
        date_folders = [args.date] if os.path.exists(os.path.join(base_folder, args.date)) else []
    else:
        # Scan all date folders
        date_folders = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])

    if not date_folders:
        print(f"No date folders found in {base_folder}")
        return

    print(f"Starting historical update for {len(date_folders)} date folders...")

    for date_str in date_folders:
        print(f"\nProcessing folder: {date_str}")
        date_path = os.path.join(base_folder, date_str)
        
        # Clear cache for each date to keep memory low
        _intraday_cache.clear()

        # Identify tickers to update
        if args.ticker:
            active_tickers = {k: v for k, v in TICKERS.items() if v == args.ticker}
        else:
            active_tickers = TICKERS

        for name, symbol in active_tickers.items():
            vol_symbol = VOLATILITY_MAP.get(symbol)
            if not vol_symbol:
                continue
                
            print(f"    Updating files for {name} ({symbol}) using {vol_symbol}...")
            
            # Find files for this ticker
            matching_files = [f for f in os.listdir(date_path) if f.startswith(f"{name}_") and f.endswith(".csv")]
            
            if not matching_files:
                print(f"      No files found for {name}")
                continue
                
            for csv_file in matching_files:
                file_path = os.path.join(date_path, csv_file)
                update_file(file_path, symbol, vol_symbol)
                # Small sleep to prevent rate limiting
                time.sleep(0.1)

    print("\nHistorical update complete.")

if __name__ == "__main__":
    main()
