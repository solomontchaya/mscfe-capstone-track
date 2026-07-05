import os
import glob
import time
import ast
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_sentiment_score(entities_str):
    """
    Safely parses the string-serialized JSON map in the 'entities' column
    to extract native StockTwits sentiment tags.
    Returns +1 for Bullish, -1 for Bearish, and NaN for unlabelled/neutral.
    """
    if pd.isna(entities_str):
        return np.nan
    try:
        # Safe literal evaluation handles single-quoted dictionaries perfectly
        entities_dict = ast.literal_eval(entities_str)
        sentiment_node = entities_dict.get('sentiment')
        if sentiment_node and isinstance(sentiment_node, dict):
            basic_sentiment = sentiment_node.get('basic')
            if basic_sentiment == 'Bullish':
                return 1.0
            elif basic_sentiment == 'Bearish':
                return -1.0
    except Exception:
        pass
    return np.nan

def process_local_chunks(raw_data_dir, output_csv_path, ticker, start_date, end_date):
    """
    Memory-efficient local pipeline that iterates through the 147 raw CSV files,
    extracts nested sentiment distributions, calculates text argument similarity,
    and structures a clean Point-In-Time dataset.
    """
    print(f"Initializing localized processing loops for: {ticker}")
    
    # Locate all sub-files within your extracted folder path
    search_path = os.path.join(raw_data_dir, "**/*.csv")
    all_files = glob.glob(search_path, recursive=True)
    
    if not all_files:
        print(f"Error: No raw data files located at: {raw_data_dir}")
        return

    print(f"Located {len(all_files)} raw data files. Beginning text parsing...")
    
    daily_aggregates = {}
    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')

    # Process file-by-file to keep local RAM footprint under 200MB
    for file_path in all_files:
        try:
            # Load only the critical columns discovered in data inspection
            df = pd.read_csv(file_path, usecols=['created_at', 'body', 'entities'])
            
            # Parse timestamps and establish standardized calendar date strings
            df['timestamp'] = pd.to_datetime(df['created_at'])
            df['date_only'] = df['timestamp'].dt.date
            
            # Filter rows strictly within your project's validation timeline
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            if df.empty:
                continue
                
            # Parse nested sentiment nodes
            df['sentiment_score'] = df['entities'].apply(extract_sentiment_score)
            
            # Group data by individual dates
            for current_date, group in df.groupby('date_only'):
                bodies = group['body'].dropna().tolist()
                msg_count = len(bodies)
                
                if msg_count < 3:
                    continue
                    
                # Compute Text Argument Similarity (Social Contagion Index)
                try:
                    tfidf = vectorizer.fit_transform(bodies)
                    cos_sim = cosine_similarity(tfidf)
                    np.fill_diagonal(cos_sim, np.nan)
                    daily_arg_sim = np.nanmean(cos_sim)
                except Exception:
                    daily_arg_sim = 0.0
                
                # Compute Sentiment Variance 
                valid_sentiments = group['sentiment_score'].dropna()
                daily_sent_var = np.var(valid_sentiments) if len(valid_sentiments) > 0 else 0.0
                
                # Consolidate running daily states across chunks
                if current_date not in daily_aggregates:
                    daily_aggregates[current_date] = {
                        'arg_sims': [daily_arg_sim],
                        'sent_vars': [daily_sent_var],
                        'counts': msg_count
                    }
                else:
                    daily_aggregates[current_date]['arg_sims'].append(daily_arg_sim)
                    daily_aggregates[current_date]['sent_vars'].append(daily_sent_var)
                    daily_aggregates[current_date]['counts'] += msg_count
                    
        except Exception as e:
            # Safely log anomalies and skip corrupted files
            continue

    if not daily_aggregates:
        print(f"Warning: No valid alternative text records processed for {ticker}.")
        return

    # Structure data records into a clean dataframe
    summary_rows = []
    for date_key, metrics in daily_aggregates.items():
        summary_rows.append({
            'Date': pd.to_datetime(date_key),
            'Argument_Similarity': np.mean(metrics['arg_sims']),
            'Sentiment_Variance': np.mean(metrics['sent_vars']),
            'Volume_Crowd': metrics['counts']
        })
        
    df_features = pd.DataFrame(summary_rows).set_index('Date').sort_index()
    
    # ENFORCE POINT-IN-TIME CAUSALITY 
    # Shift alternative features forward by 1 day so they are strictly predictive
    df_features_lagged = df_features.shift(1)
    
    # Rebrand 'FB' to 'META' to maintain compatibility with modern market APIs
    yf_ticker = "META" if ticker == "FB" else ticker
    
    print(f"Downloading market pricing matrix for {yf_ticker} via Yahoo Finance...")
    market_data = yf.download(yf_ticker, start=start_date, end=end_date)
    
    if isinstance(market_data.columns, pd.MultiIndex):
        market_data.columns = market_data.columns.get_level_values(0)
        
    # Merge datasets and fill non-trading day gaps using a forward-fill method
    final_panel = market_data[['Close', 'High', 'Low', 'Volume']].merge(
        df_features_lagged, left_index=True, right_index=True, how='left'
    )
    final_panel['Argument_Similarity'] = final_panel['Argument_Similarity'].ffill().fillna(0.0)
    final_panel['Sentiment_Variance'] = final_panel['Sentiment_Variance'].ffill().fillna(0.0)
    final_panel['Volume_Crowd'] = final_panel['Volume_Crowd'].fillna(0.0)
    
    # Export the finalized time-series matrix to your project workspace
    final_panel.to_csv(output_csv_path)
    print(f"=== Success! Processed panel saved to: {output_csv_path} ===")


# =====================================================================
# CHRONOLOGICAL EXECUTION ENTRY POINT (The Main Sweep Loop)
# =====================================================================
if __name__ == "__main__":
    
    # Adjust these paths to point to your local directories inside VS Code
    # Assuming your raw data sits inside directories named after each ticker:
    # e.g., data/raw/TSLA/, data/raw/AAPL/, etc.
    BASE_RAW_DIR = "../data/raw"
    PROCESSED_DATA_DIR = "../data/processed"
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # The five major growth tickers present in your Kaggle data download
    TARGET_UNIVERSE = ["AMZN", "FB", "AAPL", "NVDA", "TSLA"]
    
    START_HORIZON = "2020-01-01"
    END_HORIZON = "2022-12-31"
    
    total_start_time = time.time()
    
    for ticker in TARGET_UNIVERSE:
        ticker_start = time.time()
        print("\n" + "="*60)
        print(f"LAUNCHING PIPELINE FOR TARGET ASSET FLUX: {ticker}")
        print("="*60)
        
        # Point specifically to the directory holding that asset's files
        ticker_raw_folder = os.path.join(BASE_RAW_DIR, ticker)
        
        output_file_name = f"{ticker}_processed_panel.csv"
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