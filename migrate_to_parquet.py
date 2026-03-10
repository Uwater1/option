import os
import pandas as pd
import pathlib
import sys

def migrate_folder(input_dir):
    """
    Migrates all CSV files in a directory (and subdirectories) to Parquet,
    overwriting the original files so the same directory structure is maintained
    but with .parquet extensions instead of .csv
    """
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return

    csv_files = list(pathlib.Path(input_dir).rglob('*.csv'))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {input_dir}. Migrating...")

    success_count = 0
    error_count = 0

    for i, csv_path in enumerate(csv_files, 1):
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Form Parquet path
            parquet_path = csv_path.with_suffix('.parquet')
            
            # Save to Parquet
            df.to_parquet(parquet_path, index=False)
            
            # Delete original CSV to save space and avoid duplicate processing by training scripts
            os.remove(csv_path)
            
            success_count += 1
            if i % 100 == 0:
                print(f"  Processed {i}/{len(csv_files)} files...")
                
        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")
            error_count += 1

    print(f"\nMigration complete for {input_dir}:")
    print(f"  Successfully migrated: {success_count}")
    print(f"  Errors: {error_count}")


if __name__ == "__main__":
    directories_to_migrate = [
        "options_data",
        "spread",
        "options_data_test"
    ]
    
    print("WARNING: This script will convert your existing CSVs to Parquet and delete the original CSVs.")
    print("If you need to convert back, you can use the 'export_parquet_to_csv.py' script.")
    
    confirm = input("Are you sure you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Migration cancelled.")
        sys.exit(0)

    for directory in directories_to_migrate:
        print(f"\n--- Migrating {directory} ---")
        migrate_folder(directory)
