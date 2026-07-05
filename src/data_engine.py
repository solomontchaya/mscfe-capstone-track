import os
import time
from utils.process_data import process_local_chunks

if __name__ == "__main__":
    
    # Adjust these paths to point to your local directories inside VS Code
    # Assuming your raw data sits inside directories named after each ticker:
    # e.g., data/raw/TSLA/, data/raw/AAPL/, etc.
    BASE_RAW_DIR = "../data/raw/StockTwits_2020_2022_Raw"
    PROCESSED_DATA_DIR = "../data/processed/StockTwits_2020_2022_Raw"
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # The five major growth tickers present in your Kaggle data download
    TARGET_UNIVERSE = ["AMZN", "AAPL", "NVDA", "TSLA"]
    
    START_HORIZON = "2020-01-01"
    END_HORIZON = "2022-03-01"
    
    total_start_time = time.time()
    
    for ticker in TARGET_UNIVERSE:
        ticker_start = time.time()
        print("\n" + "="*60)
        print(f"LAUNCHING PIPELINE FOR TARGET ASSET FLUX: {ticker}")
        print("="*60)
        
        # Point specifically to the directory holding that asset's files
        ticker_raw_folder = os.path.join(BASE_RAW_DIR, ticker)
        
        output_file_name = f"{ticker}.csv"
        destination_path = os.path.join(PROCESSED_DATA_DIR, output_file_name)
        
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
    print("ALL UNIVERSES PREPROCESSED SEAMLESSLY FOR MODULE 3 ARCHITECTURE")
    print(f"Total processing time: {(time.time() - total_start_time)/60:.2f} minutes.")
    print("="*60)