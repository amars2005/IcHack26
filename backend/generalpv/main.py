# main.py
import argparse
import pandas as pd
from builder import DatasetBuilder
from preprocessor import DatasetPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Build and preprocess StatsBomb dataset')
    parser.add_argument('--no-build', action='store_true', 
                        help='Skip building dataset, use existing statsbomb_chained_dataset.csv')
    parser.add_argument('--no-pre', action='store_true',
                        help='Skip preprocessing step')
    parser.add_argument('--build-workers', type=int, default=12,
                        help='Number of parallel build workers (default: 12)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of matches to process (for testing)')
    parser.add_argument('--pre-workers', type=int, default=1,
                        help='Number of workers for preprocessing (default: 1, recommended)')
    args = parser.parse_args()
    
    output_filename = "statsbomb_chained_dataset.csv"
    
    if not args.no_build:
        # Build dataset from scratch
        builder = DatasetBuilder(max_workers=args.build_workers)
        
        print("Starting data extraction...")
        raw_data = builder.build_dataset(limit=args.limit)
        
        if raw_data.empty:
            print("No data found. Exiting.")
            return

        final_data = builder.process_chains(raw_data)
        final_data.to_csv(output_filename, index=False)

        print("\n--- BUILD DONE ---")
        print(f"Data saved to {output_filename}")
        print(f"Total Rows: {len(final_data)}")
        print(f"Total Matches: {final_data['match_id'].nunique()}")
        print(f"Columns: {list(final_data.columns)}")
    else:
        print(f"Skipping build, loading existing {output_filename}...")
        try:
            final_data = pd.read_csv(output_filename)
            print(f"Loaded {len(final_data)} rows from {output_filename}")
        except FileNotFoundError:
            print(f"Error: {output_filename} not found. Run without --no-build first.")
            return

    if not args.no_pre:
        print("\nApplying preprocessing...")
        preprocessor = DatasetPreprocessor()
        processed_data = preprocessor.preprocess_dataframe(final_data, max_workers=args.pre_workers)
        
        if processed_data.empty:
            print("No data after preprocessing. Exiting.")
            return
        
        processed_output_filename = "statsbomb_normalised_dataset.csv"
        processed_data.to_csv(processed_output_filename, index=False)
        print(f"\n--- PREPROCESSING DONE ---")
        print(f"Preprocessed data saved to {processed_output_filename}")
        print(f"Preprocessed rows: {len(processed_data)}")
    else:
        print("\nSkipping preprocessing.")

if __name__ == "__main__":
    main()