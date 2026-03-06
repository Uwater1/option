import yfinance as yf
import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz

TICKERS = {
    "spy": "SPY",
    "nq100": "QQQ",
    "aapl": "AAPL",
    "amzn": "AMZN",
    "goog": "GOOG",
    "silver": "SLV",
    "gold": "GLD",
    "dowjones" : "DIA",
    "longterm" : "TLT",
}

VOLATILITY_MAP = {
    "SPY": "^VIX",
    "QQQ": "^VXN",
    "AMZN": "^VXAZN",
    "AAPL": "^VXAPL",
    "GOOG": "^VXGOG",
    "SLV":"^VXSLV",
    "GLD": "^GVZ",
    "TLT":"^VXTLT",
    "DIA": "^VXD",
}

_intraday_cache = {}

def get_intraday_data(ticker_symbol):
    if ticker_symbol in _intraday_cache:
        return _intraday_cache[ticker_symbol]
    
    tk = yf.Ticker(ticker_symbol)
    hist = tk.history(period="5d", interval="1m")
    if not hist.empty:
        hist.index = pd.to_datetime(hist.index, utc=True)
    _intraday_cache[ticker_symbol] = hist
    return hist

def get_price_at_time(ticker_symbol, target_datetime, fallback_price):
    if pd.isna(target_datetime):
        return fallback_price
    
    try:
        if target_datetime.tzinfo is None:
            target_datetime = target_datetime.replace(tzinfo=pd.Timestamp.now(tz='UTC').tzinfo)

        # 1-min bar index = bar open time. If second >= 40, 'nearest' would
        # snap to the next bar's open (only a few seconds away). Instead,
        # floor to the current minute and use 'ffill' to get the bar that
        # contains the trade, then read its Close.
        if target_datetime.second >= 40:
            lookup_time = target_datetime.replace(second=0, microsecond=0)
            lookup_method = 'ffill'
        else:
            lookup_time = target_datetime
            lookup_method = 'nearest'

        hist = get_intraday_data(ticker_symbol)
        if hist.empty:
            return fallback_price
            
        closest_idx = hist.index.get_indexer([lookup_time], method=lookup_method)[0]
        if closest_idx >= 0:
            return float(hist['Close'].iloc[closest_idx])
    except Exception:
        pass
    
    return fallback_price

def main():
    spread_folder = "spread"
    if not os.path.exists(spread_folder):
        os.makedirs(spread_folder)
        
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    # 50-min threshold: 10:50 -> _11, 11:10 -> _11, 11:50 -> _12
    current_date = (now_ny + timedelta(minutes=10)).strftime("%Y%m%d_%H")
    
    # We will track when we started to enforce a time limit
    start_time = time.time()
    MAX_DURATION_SECONDS = 10 * 60 # 10 minutes max allowed
    
    for name, ticker_symbol in TICKERS.items():
        if time.time() - start_time > MAX_DURATION_SECONDS:
            print("Nearing 10 minute timeout. Stopping further downloads.")
            break
            
        try:
            print(f"Downloading spread data for {name} ({ticker_symbol})...")
            tk = yf.Ticker(ticker_symbol)
            
            # Get current price
            hist = tk.history(period="1d")
            current_price = hist['Close'].iloc[-1] if not hist.empty else tk.info.get('regularMarketPrice', np.nan)
                
            dates = tk.options
            if not dates:
                print(f"No options found for {name}.")
                continue
                
            all_spreads_for_ticker = []
            
            # Loop through all expirations
            for date in dates:
                if time.time() - start_time > MAX_DURATION_SECONDS:
                    print(f"Timeout reached while processing {ticker_symbol} expirations.")
                    break
                    
                try:
                    chain = tk.option_chain(date)
                    calls = chain.calls
                    puts = chain.puts
                    
                    dfs_to_concat = []
                    if not calls.empty:
                        calls['optionType'] = 'c'
                        dfs_to_concat.append(calls)
                    if not puts.empty:
                        puts['optionType'] = 'p'
                        dfs_to_concat.append(puts)
                        
                    if not dfs_to_concat:
                        continue
                        
                    df = pd.concat(dfs_to_concat, ignore_index=True)
                    
                    if 'ask' in df.columns and 'bid' in df.columns:
                        df['bid_ask_spread'] = df['ask'] - df['bid']
                    else:
                        df['bid_ask_spread'] = np.nan
                        
                    if 'lastTradeDate' in df.columns:
                        df['underlyingPriceAtTrade'] = df['lastTradeDate'].apply(
                            lambda x: get_price_at_time(ticker_symbol, x, current_price)
                        )
                        # Calculate days_to_expire (options expire at 16:00 New York Time)
                        # Use localize first, then replace hour to correctly handle DST
                        exp_date_naive = pd.to_datetime(date).replace(hour=16, minute=0, second=0, microsecond=0)
                        ny_tz_exp = pytz.timezone('America/New_York')
                        exp_dt = ny_tz_exp.localize(exp_date_naive.to_pydatetime()).astimezone(pytz.utc)
                        exp_dt = pd.Timestamp(exp_dt)
                        df['days_to_expire'] = (exp_dt - pd.to_datetime(df['lastTradeDate'], utc=True)).dt.total_seconds() / (24 * 3600)
                    else:
                        df['underlyingPriceAtTrade'] = current_price
                        df['days_to_expire'] = np.nan
                        
                    vol_symbol = VOLATILITY_MAP.get(ticker_symbol)
                    if vol_symbol and 'lastTradeDate' in df.columns:
                        v_tk = yf.Ticker(vol_symbol)
                        v_hist = v_tk.history(period="1d")
                        vol_current_price = v_hist['Close'].iloc[-1] if not v_hist.empty else np.nan
                        df['volatilityIndex'] = df['lastTradeDate'].apply(
                            lambda x: get_price_at_time(vol_symbol, x, vol_current_price)
                        )
                    else:
                        df['volatilityIndex'] = np.nan
                    
                    required_cols = [
                        'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'volume', 'openInterest',
                        'underlyingPriceAtTrade', 'volatilityIndex','bid','ask', 'bid_ask_spread', 'days_to_expire', 'optionType'
                    ]
                    
                    # Round numerical columns to 3 decimal places
                    float_cols = ['strike', 'lastPrice', 'underlyingPriceAtTrade', 'volatilityIndex','bid','ask','bid_ask_spread', 'days_to_expire']
                    for col in float_cols:
                        if col in df.columns:
                            df[col] = df[col].round(3)
                            
                    available_cols = [c for c in required_cols if c in df.columns]
                    df = df[available_cols]
                    
                    all_spreads_for_ticker.append(df)
                    time.sleep(0.1) # Small delay to avoid rate limiting
                    
                except Exception as e:
                    print(f"  Error fetching {date} for {name}: {e}")

            if all_spreads_for_ticker:
                final_ticker_df = pd.concat(all_spreads_for_ticker, ignore_index=True)
                underlying_price = f"{current_price:.2f}".replace(".", "_")
                underlying_vix = f"{vol_current_price:.2f}".replace(".", "_")
                file_path = os.path.join(spread_folder, f"{ticker_symbol}_{current_date}-{underlying_price}-{underlying_vix}.csv")
                final_ticker_df.to_csv(file_path, index=False)
                print(f"Successfully saved {ticker_symbol} spread data to {file_path}")
            
        except Exception as e:
            print(f"Error downloading spread data for {name}: {e}")

if __name__ == "__main__":
    main()
