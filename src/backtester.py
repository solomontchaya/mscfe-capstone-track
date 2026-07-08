import os
import numpy as np
import pandas as pd
from portfolio_engine import load_saved_posterior, generate_bayesian_inputs, optimize_portfolio

def load_historical_backtest_data(processed_dir):
    """
    Dynamically loads and combines individual ticker CSVs containing regime alignments.
    """
    print("[DATA] Ingesting and compiling per-ticker regime datasets...")
    
    tickers = ["TSLA", "AAPL", "AMZN", "NVDA"]
    combined_records = []
    
    for ticker in tickers:
        # Build the dynamic file name matching your directory structure
        file_name = f"{ticker}_with_regimes.csv"
        file_path = os.path.join(processed_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing expected data asset: {file_path}")
            
        # Read individual asset frame
        df_ticker = pd.read_csv(file_path)
        
        # Inject the Ticker identifier so the cross-sectional indexer can find it
        df_ticker['Ticker'] = ticker
        
        combined_records.append(df_ticker)
        
    # Stack all tickers vertically into one uniform dataset
    master_df = pd.concat(combined_records, ignore_index=True)
    
    # Ensure uniform datetime indexing
    master_df['Date'] = pd.to_datetime(master_df['Date'])
    
    return master_df

def run_rolling_backtest():
    # 1. Structural Path mapping
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    tickers = ["TSLA", "AAPL", "AMZN", "NVDA"]
    
    # Load the real dataset into memory before looping
    try:
        master_df = load_historical_backtest_data(PROCESSED_DIR)
    except Exception as e:
        print(f"Data Load Error: {e}")
        return
    
    # 2. Define historical execution timeline (Post 6-month warmup phase)
    rebalance_dates = pd.date_range(start="2020-07-01", end="2022-12-31", freq='W-FRI')
    
    portfolio_records = []
    current_weights = np.array([0.25, 0.25, 0.25, 0.25]) # Start equal-weighted
    
    print(f"Initializing Backtest Engine across {len(rebalance_dates)} periods...")
    
    for date in rebalance_dates:
        date_str = date.strftime('%Y-%m-%d')
        
        # Filter down our master dataset to just the rows matching this specific rebalance date
        period_df = master_df[master_df['Date'] == date]
        
        if period_df.empty or len(period_df) < len(tickers):
            # Safe catch for data gaps or the initial warmup edge cases
            continue

        # Ensure consistent indexing alignment across your tickers
        period_df = period_df.set_index('Ticker').reindex(tickers)
        
        # =====================================================================
        # MATCHING YOUR EXACT CSV HEADERS
        # =====================================================================
        REGIME_COL = 'Hidden_State'        
        # =====================================================================

        # Safe verification: Check if our exact regime column target exists and isn't null
        if REGIME_COL not in period_df.columns or period_df[REGIME_COL].isna().any():
            continue

        # 1. Look up the true active regime state for this week
        predicted_regime = int(period_df[REGIME_COL].iloc[0]) 
        
        # 2. Extract real point-in-time Stocktwits crowd signals
        real_features = period_df[['Argument_Similarity', 'Sentiment_Variance']]
        
        # 3. Pull actual returns realized by these assets over the UPCOMING week
        # We look up the date row inside master_df for the next sequential rebalance step
        next_date = rebalance_dates[rebalance_dates.get_loc(date) + 1] if date != rebalance_dates[-1] else None
        
        if next_date is not None:
            next_period_df = master_df[master_df['Date'] == next_date].set_index('Ticker').reindex(tickers)
            # Ensure no missing return tokens are passed into portfolio linear combination
            if next_period_df['log_ret'].isna().any():
                continue
            forward_returns = next_period_df['log_ret'].values
        else:
            forward_returns = period_df['log_ret'].values # Fallback for terminal calculation
        
        try:
            # A. Load the pre-compiled posterior trace asset for the active regime
            idata = load_saved_posterior(BASE_DIR, predicted_regime)
            
            # B. Conditional Bayesian aggregation of crowd forecasts
            mu_b, sigma_b = generate_bayesian_inputs(idata, real_features, tickers)
            
            # C. Optimize target portfolio allocation weights
            new_weights = optimize_portfolio(mu_b, sigma_b)
            
            # D. Apply Transaction Frictions & Turnover Penalties
            turnover = np.sum(np.abs(new_weights - current_weights))
            tx_cost = turnover * 0.0010  # 10 bps execution penalty
            
            # E. Calculate Net Portfolio Yield for this step
            raw_p_return = np.dot(new_weights, forward_returns)
            net_p_return = raw_p_return - tx_cost
            
            portfolio_records.append({
                'Date': date,
                'Regime': predicted_regime,
                'Raw_Return': raw_p_return,
                'Net_Return': net_p_return,
                'Turnover': turnover,
                'Weights': new_weights
            })
            
            # Update allocation state for next turnover check
            current_weights = new_weights
            print(f"Processed {date_str} | Regime: {predicted_regime} | Net Return: {net_p_return:.4f}")
            
        except FileNotFoundError:
            print(f"Missing trace file for Regime {predicted_regime}. Run training loop first.")
            return

    # 3. Compile and summarize Performance Metrics
    if not portfolio_records:
        print("Backtest finished with no records generated. Verify date-matching configurations.")
        return
        
    df_results = pd.DataFrame(portfolio_records)
    df_results.set_index('Date', inplace=True)
    
    # Calculate performance analytics
    cum_returns = (1 + df_results['Net_Return']).cumprod() - 1
    total_return = cum_returns.iloc[-1]
    ann_sharpe = (df_results['Net_Return'].mean() / (df_results['Net_Return'].std() + 1e-8)) * np.sqrt(52)
    
    print("\n" + "="*50)
    print("SWING-TRADE STRATEGY PERFORMANCE REPORT (2020-2022)")
    print("="*50)
    print(f"Total Cumulative Return: {total_return * 100:.2f}%")
    print(f"Annualized Sharpe Ratio: {ann_sharpe:.4f}")
    print(f"Average Weekly Turnover: {df_results['Turnover'].mean() * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_rolling_backtest()