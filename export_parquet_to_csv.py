import os
import pandas as pd
import pathlib
import sys
import argparse

def revert_file_or_folder(path):
    """
    Reverts a specific Parquet file or all Parquet files in a directory back to CSV.
    Original Parquet files are kept by default unless specified otherwise.
    """
    path_obj = pathlib.Path(path)
    
    if not path_obj.exists():
        print(f"Error: Path '{path}' does not exist.")
        return

    # Determine files to process
    if path_obj.is_file():
        if path_obj.suffix.lower() == '.parquet':
            parquet_files = [path_obj]
        else:
            print(f"Error: '{path}' is not a .parquet file.")
            return
    else:
        parquet_files = list(path_obj.rglob('*.parquet'))
        if not parquet_files:
            print(f"No Parquet files found in directory '{path}'.")
            return
        
    print(f"Found {len(parquet_files)} Parquet files to convert back to CSV.")
    
    success_count = 0
    error_count = 0
    
    for i, pq_path in enumerate(parquet_files, 1):
        try:
            # Read Parquet
            df = pd.read_parquet(pq_path)
            
            # Form CSV path
            csv_path = pq_path.with_suffix('.csv')
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            success_count += 1
            if i % 100 == 0:
                print(f"  Processed {i}/{len(parquet_files)} files...")
                
        except Exception as e:
            print(f"  Error processing {pq_path}: {e}")
            error_count += 1

    print(f"\nReversion complete:")
    print(f"  Successfully converted to CSV: {success_count}")
    print(f"  Errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet files back to CSV format without losing data.")
    parser.add_argument("path", help="Path to a specific .parquet file or a directory containing .parquet files.")
    
    args = parser.parse_args()
    
    revert_file_or_folder(args.path)
