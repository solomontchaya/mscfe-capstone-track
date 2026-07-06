import os
import time
from utils import process_local_chunks

if __name__ == "__main__":
    
    # Establish a fixed structural anchor based on this script's location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))     
    BASE_DIR = os.path.dirname(SCRIPT_DIR)                       
    
    # Build absolute paths to your data folders
    BASE_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Target growth universe
    TARGET_UNIVERSE = ["AMZN", "AAPL", "NVDA", "TSLA"]
    
    # Uniform structural window boundaries
    START_HORIZON = "2020-01-01"
    END_HORIZON = "2022-03-01"
    
    total_start_time = time.time()
    
    for ticker in TARGET_UNIVERSE:
        ticker_start = time.time()
        print("\n" + "="*60)
        print(f"LAUNCHING PIPELINE FOR TARGET ASSET FLUX: {ticker}")
        print("="*60)
        
        # Point to the specific raw data folder for this asset
        ticker_raw_folder = os.path.join(BASE_RAW_DIR, ticker)
        
        # Output clean panels as [TICKER].csv inside data/processed/
        output_file_name = f"{ticker}.csv"
        destination_path = os.path.join(PROCESSED_DATA_DIR, output_file_name)
        
        # Verify the source directory actually exists before processing
        if not os.path.exists(ticker_raw_folder):
            print(f"Execution Error: Raw data directory missing for {ticker}")
            print(f"Expected Path: {ticker_raw_folder}")
            continue
            
        process_local_chunks(
            raw_data_dir=ticker_raw_folder,
            output_csv_path=destination_path,
            ticker=ticker,
            start_date=START_HORIZON,
            end_date=END_HORIZON
        )
        
        elapsed = time.time() - ticker_start
        print(f"Completed execution cycle for {ticker} in {elapsed:.2f} seconds.")
        
    print("\n" + "="*60)
    print("ALL UNIVERSAL PANELS PROCESSED AND LOG-SHIFTED FOR REGIME PROCESSING")
    print(f"Total pipeline execution time: {(time.time() - total_start_time)/60:.2f} minutes.")
    print("="*60)