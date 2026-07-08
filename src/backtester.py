import os
import numpy as np
import pandas as pd
from portfolio_engine import load_saved_posterior, generate_bayesian_inputs, optimize_portfolio

def load_historical_backtest_data(processed_dir):
    """
    UPDATE 1: Load your real compiled 2020-2022 CSV master dataset.
    """
    print("[DATA] Loading 2020-2022 Stocktwits & Asset Matrix...")
    
    # ASSUMPTION: You have a master CSV file containing rows for each date/ticker combo.
    # Adjust the filename ('master_data.csv') to match your actual file.
    csv_path = os.path.join(processed_dir, "master_data.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find historical data file at {csv_path}. Please place your 2020-2022 data here.")
        
    df = pd.read_csv(csv_path)
    
    # Ensure standard datetime parsing
    df['Date'] = pd.to_datetime(df['Date'])
    return df

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
        print(f"❌ Data Load Error: {e}")
        return
    
    # 2. Define historical execution timeline (Post 6-month warmup phase)
    rebalance_dates = pd.date_range(start="2020-07-01", end="2022-12-31", freq='W-FRI')
    
    portfolio_records = []
    current_weights = np.array([0.25, 0.25, 0.25, 0.25]) # Start equal-weighted
    
    print(f"🚀 Initializing Backtest Engine across {len(rebalance_dates)} periods...")
    
    for date in rebalance_dates:
        date_str = date.strftime('%Y-%m-%d')
        
        # Filter down our master dataset to just the rows matching this specific rebalance date
        period_df = master_df[master_df['Date'] == date]
        
        if period_df.empty or len(period_df) < len(tickers):
            print(f"⚠️ Warning: Missing or incomplete data data for {date_str}. Skipping week.")
            continue

        # Ensure consistent indexing alignment across your tickers
        period_df = period_df.set_index('Ticker').reindex(tickers)
        
        # =====================================================================
        # UPDATE 2: REPLACING DATA LOOKUP PLACEHOLDERS WITH REAL TIME SERIES
        # =====================================================================
        
        # 1. Look up the true active regime probability or predicted discrete state output from your HMM
        # Assumes a column named 'Predicted_Regime' exists in your CSV
        predicted_regime = int(period_df['Predicted_Regime'].iloc[0]) 
        
        # 2. Extract real point-in-time Stocktwits crowd signals
        real_features = period_df[['Argument_Similarity', 'Sentiment_Variance']]
        
        # 3. Pull actual returns realized by these stocks over the upcoming week
        forward_returns = period_df['Forward_Return'].values 
        
        # =====================================================================
        
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
            print(f"❌ Missing trace file for Regime {predicted_regime}. Run training loop first.")
            return

    # 3. Compile and summarize Performance Metrics
    if not portfolio_records:
        print("❌ Backtest finished with no records generated.")
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