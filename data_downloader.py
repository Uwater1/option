#!/usr/bin/env python3
"""
Stock Data Downloader

This script downloads stock data at various intervals (including 1d with max range) 
using yfinance and saves them as CSV files for later analysis.

Author: Kilo Code
Date: 2026-01-11
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import argparse
from typing import List, Optional


class StockDataDownloader:
    """Downloads and saves stock data for various intervals."""
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data downloader.
        
        Args:
            output_dir (str): Directory to save downloaded CSV files
        """
        self.output_dir = output_dir
        self.ensure_output_directory()
    
    def get_default_days_for_interval(self, interval: str) -> int:
        """
        Get default number of days based on interval.
        
        Args:
            interval (str): Time interval (1m, 2m, 5m, 15m, 30m, 1h, 1d, etc.)
        
        Returns:
            int: Default number of days for the interval (returns 0 for 1d to use 'max' period)
        """
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            if minutes == 1:
                return 8
            elif 2 <= minutes <= 59:
                return 60
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            if 1 <= hours <= 23:
                return 730
        elif interval.endswith('d'):
            # For daily data, return 0 to signal use of 'max' period
            return 0
        
        # Default fallback
        return 60
    
    def ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✅ Created output directory: {self.output_dir}")
    
    def download_stock_data(self, ticker: str, interval: str = '5m', days: Optional[int] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """
        Download interval data for a specific stock.
        
        Args:
            ticker (str): Stock ticker symbol
            interval (str): Time interval (1m, 2m, 5m, 15m, 30m, 1h, etc.)
            days (int, optional): Number of days to download (if no specific dates provided)
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
        
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Use default days if not provided
            if days is None:
                days = self.get_default_days_for_interval(interval)
            
            print(f"📥 Downloading {interval} data for {ticker}...")
            
            # Suppress yfinance warnings for cleaner output
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            # Download data with specified interval
            if start_date and end_date:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )
            else:
                # For 1d data with days=0, use max period; otherwise use N days
                if days == 0:
                    print(f"   Using max available range for {interval} data...")
                    data = yf.download(
                        ticker,
                        period='max',
                        interval=interval,
                        progress=False
                    )
                else:
                    # Download last N days of data
                    data = yf.download(
                        ticker,
                        period=f'{days}d',
                        interval=interval,
                        progress=False
                    )
            
            if data.empty:
                print(f"❌ No data found for {ticker}")
                return False
            
                        
            # Generate filename
            filename = f"{ticker}_{interval}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to CSV
            data.to_csv(filepath)
            
            print(f"✅ Successfully saved {len(data)} data points to {filepath}")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading data for {ticker}: {str(e)}")
            return False
    
    def download_multiple_stocks(self, tickers: List[str], interval: str = '5m', days: Optional[int] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[str]:
        """
        Download data for multiple stocks.
        
        Args:
            tickers (List[str]): List of stock ticker symbols
            interval (str): Time interval (1m, 2m, 5m, 15m, 30m, 1h, etc.)
            days (int, optional): Number of days to download (if no specific dates provided)
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
        
        Returns:
            List[str]: List of successfully downloaded tickers
        """
        successful_downloads = []
        
        print(f"🚀 Starting download for {len(tickers)} stocks...")
        print("-" * 50)
        
        for ticker in tickers:
            if self.download_stock_data(ticker, interval, days, start_date, end_date):
                successful_downloads.append(ticker)
            print()  # Empty line for readability
        
        print(f"📊 Download completed!")
        print(f"   Successfully downloaded: {len(successful_downloads)} stocks")
        print(f"   Failed downloads: {len(tickers) - len(successful_downloads)} stocks")
        
        return successful_downloads


def main():
    """Main function to run the data downloader."""
    parser = argparse.ArgumentParser(
        description='Stock Data Downloader - Downloads interval data for stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python data_downloader.py IVV 5m
  python data_downloader.py SPY AAPL 1m
  python data_downloader.py QQQ 15m --days 30
  python data_downloader.py BTC-USD 1h --start-date 2024-01-01 --end-date 2024-01-31
  python data_downloader.py SPY 1d              # Downloads max available 1d data
  python data_downloader.py AAPL 1d --days 365  # Downloads 1 year of 1d data

Default time periods:
  1min ~ 4min: 8 days
  5min ~ 59min: 60 days  
  1h ~ 23h: 730 days
  1d: max available range
        '''
    )
    
    parser.add_argument('tickers', nargs='+', help='Stock ticker symbols (e.g., IVV SPY AAPL)')
    parser.add_argument('interval', help='Time interval (1m, 2m, 5m, 15m, 30m, 1h, etc.)')
    parser.add_argument('--days', type=int, help='Number of days to download (overrides default)')
    parser.add_argument('--start-date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', help='End date in YYYY-MM-DD format')
    parser.add_argument('--output-dir', default='data', help='Output directory for CSV files (default: data)')
    
    args = parser.parse_args()
    
    print("Stock Data Downloader")
    print(f"Downloads {args.interval} interval data for stocks")
    print("=" * 60)
    
    # Create downloader and download data
    downloader = StockDataDownloader(args.output_dir)
    successful = downloader.download_multiple_stocks(
        args.tickers, 
        args.interval, 
        args.days, 
        args.start_date, 
        args.end_date
    )
    
    if successful:
        print(f"\n✅ Download completed successfully!")
        print(f"📁 Data saved to: {downloader.output_dir}/")
        print(f"📄 Files created:")
        for ticker in successful:
            print(f"   - {ticker}_{args.interval}.csv")
    else:
        print(f"\n❌ No data was downloaded successfully.")


if __name__ == "__main__":
    main()