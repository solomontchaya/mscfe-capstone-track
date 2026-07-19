import os
import time
from utils import process_s3_sentiment_pipeline

if __name__ == "__main__":
    
    # 1. Flexible path anchoring for processed output targets
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))     
    if os.path.basename(SCRIPT_DIR) == "src":
        BASE_DIR = os.path.dirname(SCRIPT_DIR)
    else:
        BASE_DIR = SCRIPT_DIR
    
    # Target processed directory path
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 2. Project Universe Configuration
    TARGET_UNIVERSE = ["AAPL", "AMD", "SPY", "TSLA"]
    
    # Expanded timeline window (2012 to end of 2022)
    START_HORIZON = "2012-01-01"
    END_HORIZON = "2022-12-31"
    
    # 4. Public S3 Ingestion Path Configuration
    BUCKET_DIR = "stocktwits-nyu/dataset/v1/data/csv/symbol_sentiments"
    STORAGE_OPTIONS = {
        "anon": True,
        "client_kwargs": {"region_name": "us-west-2"}
    }
    
    print("="*70)
    print("INITIALIZING ASYNCHRONOUS AWS S3 GLOBAL INGESTION PIPELINE")
    print("="*70)
    
    # Establish connection with the public S3 space
    fs = s3fs.S3FileSystem(**STORAGE_OPTIONS)
    all_files = sorted(fs.glob(f"{BUCKET_DIR}/symbol_sentiments_*.csv"))
    print(f"[S3 CONNECT] Located {len(all_files)} global chunks to slice mine.")
    
    # Dictionary to collect dataframes dynamically per asset to prevent continuous IO hits
    universe_collections = {ticker: [] for ticker in TARGET_UNIVERSE}
    
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
    print("="*70)