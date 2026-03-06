import yfinance as yf
import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        # Sort index for merge_asof
        hist = hist.sort_index()
    _intraday_cache[ticker_symbol] = hist
    return hist

def download_expiration(tk, ticker_symbol, date, current_price, vol_current_price, vol_hist, ticker_hist):
    """Downloads and processes a single expiration date."""
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
            return None
            
        df = pd.concat(dfs_to_concat, ignore_index=True)
        
        if 'ask' in df.columns and 'bid' in df.columns:
            df['bid_ask_spread'] = df['ask'] - df['bid']
        else:
            df['bid_ask_spread'] = np.nan
            
        if 'lastTradeDate' in df.columns:
            df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'], utc=True)
            
            # Vectorized price lookup using merge_asof for underlyingPriceAtTrade
            if not ticker_hist.empty:
                df = df.sort_values('lastTradeDate')
                df = pd.merge_asof(
                    df, 
                    ticker_hist[['Close']].rename(columns={'Close': 'underlyingPriceAtTrade'}),
                    left_on='lastTradeDate', 
                    right_index=True, 
                    direction='nearest'
                )
            else:
                df['underlyingPriceAtTrade'] = current_price
            
            # Vectorized price lookup for volatilityIndex
            if not vol_hist.empty:
                # If df was already sorted above, merge_asof will workfine. 
                # If not, we ensure it's sorted by lastTradeDate.
                df = pd.merge_asof(
                    df,
                    vol_hist[['Close']].rename(columns={'Close': 'volatilityIndex'}),
                    left_on='lastTradeDate',
                    right_index=True,
                    direction='nearest'
                )
            else:
                df['volatilityIndex'] = vol_current_price

            # Calculate days_to_expire
            exp_date_naive = pd.to_datetime(date).replace(hour=16, minute=0, second=0, microsecond=0)
            ny_tz_exp = pytz.timezone('America/New_York')
            exp_dt = ny_tz_exp.localize(exp_date_naive.to_pydatetime()).astimezone(pytz.utc)
            exp_dt = pd.Timestamp(exp_dt)
            df['days_to_expire'] = (exp_dt - df['lastTradeDate']).dt.total_seconds() / (24 * 3600)
            # Remove timezone offset (+00:00) by converting to naive timezone
            df['lastTradeDate'] = df['lastTradeDate'].dt.tz_localize(None)
        else:
            df['underlyingPriceAtTrade'] = current_price
            df['volatilityIndex'] = vol_current_price
            df['days_to_expire'] = np.nan
            
        required_cols = [
            'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'volume', 'openInterest',
            'underlyingPriceAtTrade', 'volatilityIndex','bid','ask', 'bid_ask_spread', 'days_to_expire', 'optionType'
        ]
        
        float_cols = ['strike', 'lastPrice', 'underlyingPriceAtTrade', 'volatilityIndex','bid','ask','bid_ask_spread', 'days_to_expire']
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].round(3)
                
        available_cols = [c for c in required_cols if c in df.columns]
        return df[available_cols]
        
    except Exception as e:
        print(f"  Error fetching {date}: {e}")
        return None

def main():
    spread_folder = "spread"
    if not os.path.exists(spread_folder):
        os.makedirs(spread_folder)
        
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    current_date = (now_ny + timedelta(minutes=10)).strftime("%Y%m%d_%H")
    
    start_time = time.time()
    MAX_DURATION_SECONDS = 10 * 60 
    
    for name, ticker_symbol in TICKERS.items():
        if time.time() - start_time > MAX_DURATION_SECONDS:
            print("Nearing 10 minute timeout. Stopping further downloads.")
            break
            
        try:
            tk = yf.Ticker(ticker_symbol)
            
            # Get current price and historical data for vectorization
            ticker_hist = get_intraday_data(ticker_symbol)
            current_price = ticker_hist['Close'].iloc[-1] if not ticker_hist.empty else tk.info.get('regularMarketPrice', np.nan)
                
            dates = tk.options
            if not dates:
                print(f"No options found for {name}.")
                continue
            
            # Prepare volatility data once per ticker
            vol_symbol = VOLATILITY_MAP.get(ticker_symbol)
            vol_hist = pd.DataFrame()
            vol_current_price = np.nan
            if vol_symbol:
                vol_hist = get_intraday_data(vol_symbol)
                vol_current_price = vol_hist['Close'].iloc[-1] if not vol_hist.empty else np.nan

            all_spreads_for_ticker = []
            
            # Use ThreadPoolExecutor to download expirations in parallel
            # Even on one core, this speeds up I/O bound network requests.
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_date = {
                    executor.submit(download_expiration, tk, ticker_symbol, date, current_price, vol_current_price, vol_hist, ticker_hist): date 
                    for date in dates
                }
                
                for future in as_completed(future_to_date):
                    if time.time() - start_time > MAX_DURATION_SECONDS:
                        # We don't cancel pending futures as ThreadPoolExecutor doesn't support it easily,
                        # but we stop processing results.
                        continue
                    
                    df = future.result()
                    if df is not None:
                        all_spreads_for_ticker.append(df)

            if all_spreads_for_ticker:
                final_ticker_df = pd.concat(all_spreads_for_ticker, ignore_index=True)
                underlying_price_str = f"{current_price:.2f}".replace(".", "_")
                underlying_vix_str = f"{vol_current_price:.2f}".replace(".", "_")
                file_path = os.path.join(spread_folder, f"{ticker_symbol}_{current_date}-{underlying_price_str}-{underlying_vix_str}.csv")
                final_ticker_df.to_csv(file_path, index=False)
                print(f"Successfully saved {ticker_symbol} spread data to {file_path}")
            
        except Exception as e:
            print(f"Error downloading spread data for {name}: {e}")

if __name__ == "__main__":
    main()
