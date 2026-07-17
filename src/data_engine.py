import os
import time
from utils import process_s3_sentiment_pipeline

if __name__ == "__main__":
    
    # Establish a fixed structural anchor based on this script's location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))     
    BASE_DIR = os.path.dirname(SCRIPT_DIR)                       
    
    # Target processed directory path
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Target growth universe
    TARGET_UNIVERSE = ["AMZN", "AAPL", "NVDA", "TSLA"]
    
    # Expanded timeline window (2012 to end of 2022)
    START_HORIZON = "2012-01-01"
    END_HORIZON = "2022-12-31"
    
    total_start_time = time.time()
    
    for ticker in TARGET_UNIVERSE:
        ticker_start = time.time()
        print("\n" + "="*60)
        print(f"LAUNCHING S3 SENTIMENT PIPELINE FOR TARGET ASSET: {ticker}")
        print("="*60)
        
        # Output clean panels as [TICKER].csv inside data/processed/
        output_file_name = f"{ticker}.csv"
        destination_path = os.path.join(PROCESSED_DATA_DIR, output_file_name)
        
        # Stream, process, and align using the sentiment-only S3 pipeline
        process_s3_sentiment_pipeline(
            output_csv_path=destination_path,
            target_ticker=ticker,
            start_date=START_HORIZON,
            end_date=END_HORIZON
        )
        
        elapsed = time.time() - ticker_start
        print(f"Completed execution cycle for {ticker} in {elapsed:.2f} seconds.")
        
    print("\n" + "="*60)
    print("ALL S3 UNIVERSAL PANELS PROCESSED AND LOG-SHIFTED")
    print(f"Total pipeline execution time: {(time.time() - total_start_time)/60:.2f} minutes.")
    print("="*60)