#!/usr/bin/env python3
"""
Volatility Curve Viewer

This script plots the implied volatility curve (skew) for a given security on a specific day.
It filters out missing and bad data points automatically.

Usage:
    python volatility_viewer.py --ticker AAPL --date 2026-02-11 --expiry 20260218
    python volatility_viewer.py --ticker AAPL --date 2026-02-11  # plots all expiries
"""

import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_options_data(ticker, date, expiry=None):
    """
    Load options data for a given ticker and date.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'aapl')
        date: Date in YYYY-MM-DD format
        expiry: Optional expiry date in YYYYMMDD format
        
    Returns:
        DataFrame with combined calls and puts data
    """
    data_dir = f"options_data/{date}"
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Build file pattern
    ticker_lower = ticker.lower()
    if expiry:
        pattern = f"{data_dir}/{ticker_lower}_{expiry}_*.csv"
    else:
        pattern = f"{data_dir}/{ticker_lower}_*_*.csv"
    
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No data files found for ticker {ticker} on {date}")
    
    # Load all matching files
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Add option type based on filename
            if '_calls_' in file:
                df['optionType'] = 'call'
            elif '_puts_' in file:
                df['optionType'] = 'put'
            
            # Extract expiry from filename
            basename = os.path.basename(file)
            parts = basename.split('_')
            if len(parts) >= 2:
                df['expiryDate'] = parts[1]
            
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid data files could be loaded")
    
    return pd.concat(dfs, ignore_index=True)


def filter_data(df):
    """
    Filter out bad data points.
    
    Args:
        df: DataFrame with options data
        
    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    
    # Remove rows with missing impliedVolatility
    df = df[df['impliedVolatility'].notna()].copy()
    
    # Remove rows where IV is zero or negative
    df = df[df['impliedVolatility'] > 0].copy()
    
    # Remove rows where IV is excessively high (> 500%)
    df = df[df['impliedVolatility'] <= 500].copy()
    
    # Remove rows with missing strike
    df = df[df['strike'].notna()].copy()
    
    filtered_count = len(df)
    removed_count = original_count - filtered_count
    
    if removed_count > 0:
        print(f"Filtered out {removed_count} bad/missing data points ({removed_count/original_count*100:.1f}%)")
    
    return df


def plot_volatility_curve(df, ticker, date, expiry=None, output_file=None):
    """
    Plot the volatility curve.
    
    Args:
        df: DataFrame with filtered options data
        ticker: Stock ticker symbol
        date: Date string
        expiry: Optional expiry date string
        output_file: Optional output filename
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare expiries
    if 'expiryDate' in df.columns:
        expiries = sorted(df['expiryDate'].dropna().unique())
    else:
        expiries = []
    
    num_expiries = len(expiries)
    
    # If we have many expiries, use a colormap
    if num_expiries > 1:
        # Use viridis colormap
        cmap = plt.get_cmap('viridis')
        # Create a matching normalization
        norm = plt.Normalize(0, num_expiries - 1)
        
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Add colorbar
        cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(0, num_expiries-1, min(10, num_expiries)))
        
        # Select tick labels
        if num_expiries <= 10:
            tick_labels = expiries
        else:
            # Pick ~10 evenly spaced dates
            idx = np.linspace(0, num_expiries-1, 10).astype(int)
            tick_labels = [expiries[i] for i in idx]
            
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('Expiration Date', rotation=270, labelpad=15)
    
    # Loop and plot
    for i, exp in enumerate(expiries):
        # Determine color
        if num_expiries > 1:
            color = cmap(norm(i))
        else:
            color = None # Auto color
            
        if exp:
            exp_df = df[df['expiryDate'] == exp]
            label_suffix = f" ({exp})"
        else:
            exp_df = df
            label_suffix = ""
        
        # Plot calls - Solid line, Circle marker
        calls = exp_df[exp_df['optionType'] == 'call'].sort_values('strike')
        if len(calls) > 0:
            label = f'Calls{label_suffix}' if num_expiries <= 5 else '_nolegend_'
            ax.plot(calls['strike'], calls['impliedVolatility'], 
                   marker='o', linestyle='-', color=color, label=label, 
                   alpha=0.7, markersize=3, linewidth=1.5)
        
        # Plot puts - Dashed line, X marker
        puts = exp_df[exp_df['optionType'] == 'put'].sort_values('strike')
        if len(puts) > 0:
            label = f'Puts{label_suffix}' if num_expiries <= 5 else '_nolegend_'
            ax.plot(puts['strike'], puts['impliedVolatility'], 
                   marker='x', linestyle='--', color=color, label=label, 
                   alpha=0.7, markersize=3, linewidth=1.5)

    ax.set_xlabel('Strike Price', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    
    title = f'Volatility Curve - {ticker.upper()} ({date})'
    if expiry:
        title += f' - Expiry: {expiry}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    # Custom legend for many expiries case
    if num_expiries > 5:
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='black', lw=2, linestyle='-', marker='o'),
                        Line2D([0], [0], color='black', lw=2, linestyle='--', marker='x')]
        ax.legend(custom_lines, ['Calls', 'Puts'], loc='upper right')
    else:
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot volatility curve for options data')
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--expiry', help='Optional expiry date in YYYYMMDD format')
    parser.add_argument('--output', help='Output filename for the plot')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data for {args.ticker} on {args.date}...")
        df = load_options_data(args.ticker, args.date, args.expiry)
        print(f"Loaded {len(df)} data points")
        
        # Filter data
        df = filter_data(df)
        print(f"Remaining {len(df)} valid data points")
        
        if len(df) == 0:
            print("Error: No valid data points remaining after filtering")
            return
        
        # Generate output filename if not provided
        output_file = args.output
        if not output_file:
            expiry_str = f"_{args.expiry}" if args.expiry else ""
            output_file = f"volatility_{args.ticker.lower()}_{args.date}{expiry_str}.png"
        
        # Plot
        plot_volatility_curve(df, args.ticker, args.date, args.expiry, output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
