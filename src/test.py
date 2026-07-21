import s3fs
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Configuration ---
BUCKET_DIR = "stocktwits-nyu/dataset/v1/data/csv/symbol_sentiments"
STORAGE_OPTIONS = {
    "anon": True,
    "client_kwargs": {"region_name": "us-west-2"}
}

print("📡 Connecting to S3 and scanning all files...")
fs = s3fs.S3FileSystem(**STORAGE_OPTIONS)
all_files = sorted(fs.glob(f"{BUCKET_DIR}/symbol_sentiments_*.csv"))
print(f"Found {len(all_files)} total files to process.")

# --- Step 1: Global Ticker Tally ---
print("\n📊 Phase 1: Identifying overall Top Tickers...")
global_counts = Counter()

# We can skim chunks to quickly find our heavy hitters across the entire timeline
for i, file_path in enumerate(all_files):
    try:
        # Read just the symbol column to get a global count rapidly
        df_skim = pd.read_csv(f"s3://{file_path}", storage_options=STORAGE_OPTIONS, usecols=["symbol_list"]).dropna()
        for row in df_skim["symbol_list"]:
            try:
                # Handle string-encoded lists safely
                symbols = ast.literal_eval(row) if isinstance(row, str) else row
                if isinstance(symbols, list):
                    global_counts.update(symbols)
            except:
                continue
    except Exception as e:
        print(f"Skipping file {i} due to error: {e}")

# Extract Top 10 Tickers
top_10_tuples = global_counts.most_common(10)
top_10_tickers = [ticker for ticker, count in top_10_tuples]
print(f"🎯 Global Top 10 Tickers identified: {top_10_tickers}")


# --- Step 2: Time-Series Distribution Tracking (2012 - 2022) ---
print("\n⏳ Phase 2: Extracting 2012-2022 yearly distribution for the Top 10...")
# Initialize a matrix structure: list of dicts to convert to DataFrame
distribution_data = []

for file_path in all_files:
    # Read both timestamp and symbols
    df_chunk = pd.read_csv(
        f"s3://{file_path}", 
        storage_options=STORAGE_OPTIONS, 
        usecols=["created_at", "symbol_list"]
    ).dropna()
    
    # Parse out the year efficiently
    df_chunk["Year"] = pd.to_datetime(df_chunk["created_at"], errors='coerce').dt.year
    df_chunk = df_chunk[(df_chunk["Year"] >= 2012) & (df_chunk["Year"] <= 2022)]
    
    if df_chunk.empty:
        continue
        
    # Explode and count
    df_chunk["symbol_list"] = df_chunk["symbol_list"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_exploded = df_chunk.explode("symbol_list")
    
    # Filter strictly for rows containing our top 10 tickers
    df_filtered = df_exploded[df_exploded["symbol_list"].isin(top_10_tickers)]
    
    # Group by Ticker and Year to get historical counts
    counts = df_filtered.groupby(["symbol_list", "Year"]).size().reset_index(name="Volume")
    distribution_data.append(counts)

# Combine results from all files
df_distribution = pd.concat(distribution_data).groupby(["symbol_list", "Year"]).sum().reset_index()
df_pivot = df_distribution.pivot(index="symbol_list", columns="Year", values="Volume").fillna(0)

print("\n--- Raw Data Volume Matrix (2012-2022) ---")
print(df_pivot)


# --- Step 3: Visualization ---
print("\n🎨 Phase 3: Generating Visualizations...")
sns.set_theme(style="whitegrid")

# Chart 1: Volume Growth Trends
plt.figure(figsize=(12, 6))
for ticker in top_10_tickers:
    if ticker in df_pivot.index:
        plt.plot(df_pivot.columns, df_pivot.loc[ticker], marker='o', label=ticker, linewidth=2)

plt.title("Sentiment Volume Trajectory (2012 - 2022)", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Sentiment Records", fontsize=12)
plt.xticks(df_pivot.columns)
plt.legend(title="Tickers", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("sentiment_trends.png", dpi=300)
plt.show()

# Chart 2: Relative Density Heatmap (Percentage distribution per stock across years)
# Normalize rows to sum to 100% so we see WHERE the data is concentrated relative to its own lifetime
df_heatmap_norm = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12, 6))
sns.heatmap(df_heatmap_norm, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '% of Total Lifetime Sentiment'})
plt.title("Where is the Sentiment Concentrated? (Normalized Yearly Distribution %)", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Ticker Symbol", fontsize=12)
plt.tight_layout()
plt.savefig("sentiment_heatmap.png", dpi=300)
plt.show()