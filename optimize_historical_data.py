import os
import pandas as pd
import numpy as np
import pathlib

def optimize_dataframe(df):
    """
    Applies dtype and compression optimizations to the dataframe.
    """
    # 1. Categories
    cat_cols = ['contractSymbol', 'optionType']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # 2. Int8 for ITM
    if 'inTheMoney' in df.columns:
        df['inTheMoney'] = df['inTheMoney'].astype(np.int8)

    # 3. Float32 for all numerical columns
    # We include float32 in the selection to re-cast and ensure consistency,
    # and also cover common column names just in case they were object/int
    num_cols = [
        'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest',
        'IV_yf', 'underlyingPriceAtTrade', 'impliedVolatility',
        'riskFreeRate', 'volatilityIndex', 'bid_ask_spread', 'days_to_expire'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # 4. Datetime optimization
    if 'lastTradeDate' in df.columns:
        df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'], utc=True).dt.tz_localize(None)

    return df

def migrate_folder(input_dir):
    """
    Recursively finds all Parquet files and optimizes them.
    """
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return

    parquet_files = list(pathlib.Path(input_dir).rglob('*.parquet'))
    if not parquet_files:
        print(f"No Parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} Parquet files in {input_dir}. Optimizing...")

    success_count = 0
    error_count = 0

    for i, parquet_path in enumerate(parquet_files, 1):
        try:
            # Read Parquet
            df = pd.read_parquet(parquet_path)

            # Optimize
            df = optimize_dataframe(df)

            # Save back with zstd compression safely using a temporary file
            temp_path = parquet_path.with_suffix('.tmp')
            try:
                df.to_parquet(temp_path, index=False, compression='zstd')
                temp_path.replace(parquet_path)
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise e

            success_count += 1
            if i % 100 == 0:
                print(f"  Processed {i}/{len(parquet_files)} files...")

        except Exception as e:
            print(f"  Error processing {parquet_path}: {e}")
            error_count += 1

    print(f"\nOptimization complete for {input_dir}:")
    print(f"  Successfully optimized: {success_count}")
    print(f"  Errors: {error_count}")

if __name__ == "__main__":
    directories = ["options_data", "spread"]
    for directory in directories:
        print(f"\n--- Optimizing {directory} ---")
        migrate_folder(directory)
