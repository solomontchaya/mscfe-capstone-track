import os
import time
import s3fs
import ast
import pandas as pd

if __name__ == "__main__":
    
    # 1. Flexible path anchoring for processed output targets
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))     
    if os.path.basename(SCRIPT_DIR) == "src":
        BASE_DIR = os.path.dirname(SCRIPT_DIR)
    else:
        BASE_DIR = SCRIPT_DIR
    
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 2. Project Universe Configuration
    TARGET_UNIVERSE = ["AAPL", "AMD", "SPY", "TSLA"]
    
    # 3. Synchronized Structural Boundaries
    START_HORIZON = "2014-01-01"
    END_HORIZON = "2022-03-01"
    
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
    
    # Loop over global files, process and extract rows matching the target assets
    for idx, file_path in enumerate(all_files):
        file_start = time.time()
        print(f"\nStreaming Chunks from Global File [{idx + 1}/{len(all_files)}]: {os.path.basename(file_path)}")
        
        try:
            # Read relevant features point-in-time
            df_chunk = pd.read_csv(
                f"s3://{file_path}", 
                storage_options=STORAGE_OPTIONS
            ).dropna(subset=["created_at", "symbol_list"])
            
            # Filter structural time-horizon boundaries
            df_chunk["created_at"] = pd.to_datetime(df_chunk["created_at"], errors='coerce')
            df_chunk = df_chunk[
                (df_chunk["created_at"] >= START_HORIZON) & 
                (df_chunk["created_at"] <= END_HORIZON)
            ]
            
            if df_chunk.empty:
                print(f" -> Skipping: No logs found inside {START_HORIZON} to {END_HORIZON} window.")
                continue
                
            # Safely evaluate string representations of symbol arrays
            df_chunk["symbol_list"] = df_chunk["symbol_list"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            # Explode lists to isolate unique rows per asset symbol
            df_exploded = df_chunk.explode("symbol_list")
            
            # Route matching elements straight to our collection lists
            for ticker in TARGET_UNIVERSE:
                df_ticker_slice = df_exploded[df_exploded["symbol_list"] == ticker]
                if not df_ticker_slice.empty:
                    universe_collections[ticker].append(df_ticker_slice)
                    
            print(f" -> Mining cycle completed in {time.time() - file_start:.2f} seconds.")
            
        except Exception as e:
            print(f" [WARNING] Error compiling file slice {idx}: {e}")
            continue

    print("\n" + "="*70)
    print("Writing Extracted Assets out to Panel Layers inside data/processed/")
    print("="*70)
    
    # Combine and export individual data structures
    for ticker in TARGET_UNIVERSE:
        if len(universe_collections[ticker]) > 0:
            df_final_panel = pd.concat(universe_collections[ticker]).sort_values(by="created_at")
            
            # Deduplicate or process final metrics here if your utils expects specific dimensions
            destination_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}.csv")
            df_final_panel.to_csv(destination_path, index=False)
            
            print(f"=== Success! Processed panel saved: {destination_path} (Shape: {df_final_panel.shape}) ===")
        else:
            print(f"No relevant logs recovered for target asset {ticker} within timeline parameters.")
            
    print("\n" + "="*70)
    print("ALL UNIVERSAL PANELS PROCESSED AND LOG-SHIFTED FOR REGIME PROCESSING")
    print(f"Total pipeline execution time: {(time.time() - total_start_time)/60:.2f} minutes.")
    print("="*70)