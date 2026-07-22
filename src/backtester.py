import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from portfolio_engine import load_saved_posterior, generate_bayesian_inputs, optimize_portfolio

def load_historical_backtest_data(processed_dir, tickers):
    """
    Dynamically loads and combines individual ticker CSVs containing regime alignments.
    Now accepts a dynamic 'tickers' list to prevent asset mismatch issues.
    """
    print("[DATA] Ingesting and compiling per-ticker regime datasets...")
    
    combined_records = []
    
    for ticker in tickers:
        file_name = f"{ticker}_with_regimes.csv"
        file_path = os.path.join(processed_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing expected data asset: {file_path}\n"
                f"Please ensure that HMM training has successfully run for target asset: '{ticker}'."
            )
            
        # Read individual asset frame
        df_ticker = pd.read_csv(file_path)
        
        # Inject the Ticker identifier so the cross-sectional indexer can find it
        df_ticker['Ticker'] = ticker
        
        combined_records.append(df_ticker)
        
    # Stack all tickers vertically into one uniform dataset
    master_df = pd.concat(combined_records, ignore_index=True)
    
    # Ensure uniform naive datetime indexing (removing timezone offsets and time components)
    master_df['Date'] = pd.to_datetime(master_df['Date']).dt.tz_localize(None).dt.normalize()
    
    return master_df

def plot_backtest_results(df_results, tickers, output_path):
    """
    Renders a four-panel diagnostic dashboard summarizing backtest performance:
    cumulative net returns with regime shading, drawdown, allocation weights
    over time, and per-period turnover.
    """
    sns.set_theme(style="whitegrid")

    # Expand the per-period weight vectors into individual ticker columns
    weights_df = pd.DataFrame(
        df_results['Weights'].tolist(),
        index=df_results.index,
        columns=tickers
    )

    cum_returns = (1 + df_results['Net_Return']).cumprod() - 1
    running_max = (1 + cum_returns).cummax()
    drawdown = (1 + cum_returns) / running_max - 1
    cum_benchmark = (1 + df_results['Benchmark_Return']).cumprod() - 1

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # --- Panel 1: Cumulative Net Return vs. Equal-Weight Benchmark, with Regime Shading ---
    ax1.plot(cum_returns.index, cum_returns.values * 100, color='#1f77b4',
              linewidth=1.8, label='Strategy (Net)', zorder=3)
    ax1.plot(cum_benchmark.index, cum_benchmark.values * 100, color='#555555',
              linewidth=1.4, linestyle='--', label='Equal-Weight Benchmark', zorder=2)
    ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--')

    # Shade background by active regime (contiguous blocks)
    regime_colors = {0: '#a6d8a8', 1: '#f4a6a6'}
    regime_labels_used = set()
    dates = df_results.index.to_list()
    regimes = df_results['Regime'].to_list()
    seg_start = dates[0]
    seg_regime = regimes[0]
    for i in range(1, len(dates) + 1):
        if i == len(dates) or regimes[i] != seg_regime:
            seg_end = dates[i] if i < len(dates) else dates[-1]
            label = f"Regime {seg_regime}" if seg_regime not in regime_labels_used else None
            ax1.axvspan(seg_start, seg_end, color=regime_colors.get(seg_regime, '#dddddd'),
                        alpha=0.25, label=label)
            regime_labels_used.add(seg_regime)
            if i < len(dates):
                seg_start = dates[i]
                seg_regime = regimes[i]

    ax1.set_title("Cumulative Net Return vs. Equal-Weight Benchmark", fontsize=12, weight='bold')
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Panel 2: Drawdown ---
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color='#d62728', alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values * 100, color='#d62728', linewidth=1.2)
    ax2.set_title("Portfolio Drawdown", fontsize=12, weight='bold')
    ax2.set_ylabel("Drawdown (%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Panel 3: Allocation Weights Over Time (Stacked Area) ---
    colors = sns.color_palette("muted", len(tickers))
    ax3.stackplot(weights_df.index, [weights_df[t] * 100 for t in tickers],
                  labels=tickers, colors=colors, alpha=0.85)
    ax3.set_title("Portfolio Allocation Weights Over Time", fontsize=12, weight='bold')
    ax3.set_ylabel("Weight (%)")
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', frameon=True, fontsize=9, ncol=len(tickers))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Panel 4: Turnover per Rebalance ---
    ax4.bar(df_results.index, df_results['Turnover'] * 100, width=4,
            color='#9467bd', edgecolor='black', alpha=0.8)
    ax4.set_title("Turnover per Rebalance", fontsize=12, weight='bold')
    ax4.set_ylabel("Turnover (%)")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[VISUAL] Backtest results dashboard exported to: {output_path}")
    plt.close()

def run_rolling_backtest():
    # 1. Structural Path mapping
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    # Defining target backtest assets (Aligned with the Bayesian Backtest Simulator)
    tickers = ["AAPL", "AMD", "SPY", "TSLA"]

    # Turnover controls: with weak/noisy Bayesian mu estimates, an unconstrained
    # Sharpe-maximizer can flip between concentrated corner solutions week to
    # week. These two knobs bias the optimizer toward smoother reallocation.
    # Set both to 0/None to fall back to the original unconstrained behavior.
    TURNOVER_PENALTY = 0.05      # Soft L1 penalty (lambda) on |w_new - w_prev|
    MAX_WEIGHT_CHANGE = 0.15     # Hard cap: no single asset's weight can move
                                  # more than this per rebalance (None disables)
    
    # Load the real dataset into memory before looping, passing the target tickers dynamically
    try:
        master_df = load_historical_backtest_data(PROCESSED_DIR, tickers)
    except Exception as e:
        print(f"Data Load Error: {e}")
        return
    
    print(f"[DATA] Successfully loaded master dataset with {len(master_df)} rows.")
    print(f"[DATA] Date range in dataset: {master_df['Date'].min().strftime('%Y-%m-%d')} to {master_df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Define Column Mapping constants outside the loop to avoid UnboundLocalErrors
    REGIME_COL = 'Hidden_State'

    # 2. Define historical execution timeline (Dynamically matches dataset date range)
    min_data_date = master_df['Date'].min()
    max_data_date = master_df['Date'].max()
    
    # Attempt a 6-month warmup phase. If overall timeline is too narrow, fall back to min date.
    warmup_start = min_data_date + pd.Timedelta(weeks=26)
    if warmup_start >= max_data_date:
        warmup_start = min_data_date
        
    rebalance_dates = pd.date_range(start=warmup_start, end=max_data_date, freq='W-FRI')
    
    # Create normalized DataFrames to merge weekly target dates with the closest actual data date
    rebalance_df = pd.DataFrame({'Rebalance_Date': rebalance_dates})
    rebalance_df['Rebalance_Date'] = rebalance_df['Rebalance_Date'].dt.normalize()
    
    available_dates = pd.DataFrame({'Actual_Date': master_df['Date'].unique()}).sort_values('Actual_Date')
    
    # Align each target rebalance date with the nearest actual date within 4 days (covers holiday offsets)
    aligned = pd.merge_asof(
        rebalance_df,
        available_dates,
        left_on='Rebalance_Date',
        right_on='Actual_Date',
        direction='nearest',
        tolerance=pd.Timedelta(days=4)
    ).dropna().reset_index(drop=True)
    
    portfolio_records = []
    current_weights = np.array([0.25, 0.25, 0.25, 0.25]) # Start equal-weighted
    
    skip_reasons = {
        'empty_or_missing_tickers': 0,
        'missing_regime_col': 0,
        'missing_forward_returns': 0,
        'other_error': 0
    }
    
    print(f"Initializing Backtest Engine across {len(aligned)} aligned periods...")
    
    for idx, row in aligned.iterrows():
        rebalance_date = row['Rebalance_Date']
        actual_date = row['Actual_Date']
        date_str = actual_date.strftime('%Y-%m-%d')
        
        # Filter down our master dataset to just the rows matching this specific rebalance date
        period_df = master_df[master_df['Date'] == actual_date]
        
        if period_df.empty or len(period_df) < len(tickers):
            skip_reasons['empty_or_missing_tickers'] += 1
            continue

        # Ensure consistent indexing alignment across your tickers
        period_df = period_df.set_index('Ticker').reindex(tickers)

        # Safe verification: Check if our exact regime column target exists and isn't null
        if REGIME_COL not in period_df.columns or period_df[REGIME_COL].isna().any():
            skip_reasons['missing_regime_col'] += 1
            continue

        # 1. Look up the true active regime state for this week
        predicted_regime = int(period_df[REGIME_COL].iloc[0]) 
        
        # 2. Extract real point-in-time Stocktwits crowd signals
        real_features = period_df[['Sentiment_Mean', 'Sentiment_Variance']]
        
        # 3. Pull actual returns realized by these assets over the UPCOMING week
        if idx + 1 < len(aligned):
            next_actual_date = aligned.loc[idx + 1, 'Actual_Date']
            next_period_df = master_df[master_df['Date'] == next_actual_date].set_index('Ticker').reindex(tickers)
            if next_period_df['log_ret'].isna().any():
                skip_reasons['missing_forward_returns'] += 1
                continue
            forward_returns = next_period_df['log_ret'].values
        else:
            forward_returns = period_df['log_ret'].values # Fallback for terminal calculation
        
        try:
            # A. Load the pre-compiled posterior trace asset for the active regime
            idata = load_saved_posterior(BASE_DIR, predicted_regime)
            
            # B. Conditional Bayesian aggregation of crowd forecasts
            mu_b, sigma_b = generate_bayesian_inputs(idata, real_features, tickers)
            
            # SAFEGUARD: Flatten expected return vector to 1D array to match optimizer matrix-dot layout
            mu_b_flat = mu_b.flatten()
            
            # C. Optimize target portfolio allocation weights, biased toward
            #    the current allocation to curb noise-driven corner-flipping
            new_weights = optimize_portfolio(
                mu_b_flat, sigma_b,
                prev_weights=current_weights,
                turnover_penalty=TURNOVER_PENALTY,
                max_weight_change=MAX_WEIGHT_CHANGE
            )
            
            # D. Apply Transaction Frictions & Turnover Penalties
            turnover = np.sum(np.abs(new_weights - current_weights))
            tx_cost = turnover * 0.0010  # 10 bps execution penalty
            
            # E. Calculate Net Portfolio Yield for this step
            raw_p_return = np.dot(new_weights, forward_returns)
            net_p_return = raw_p_return - tx_cost

            # F. Passive equal-weight buy-and-hold benchmark over the SAME
            #    forward-return window, for a like-for-like comparison. No
            #    transaction cost applied since the benchmark weights never
            #    change (a static equal split has zero turnover by definition).
            benchmark_weights = np.ones(len(tickers)) / len(tickers)
            benchmark_return = np.dot(benchmark_weights, forward_returns)

            portfolio_records.append({
                'Date': actual_date,
                'Regime': predicted_regime,
                'Raw_Return': raw_p_return,
                'Net_Return': net_p_return,
                'Benchmark_Return': benchmark_return,
                'Turnover': turnover,
                'Weights': new_weights
            })
            
            # Update allocation state for next turnover check
            current_weights = new_weights
            print(f"Processed {date_str} | Regime: {predicted_regime} | Net Return: {net_p_return:.4f}")
            
        except FileNotFoundError:
            print(f"Missing trace file for Regime {predicted_regime}. Run training loop first.")
            return
        except Exception as e:
            skip_reasons['other_error'] += 1
            if skip_reasons['other_error'] <= 5:
                print(f"[DEBUG] Operational loop error on {date_str}: {e}")

    # 3. Compile and summarize Performance Metrics
    if not portfolio_records:
        print("\nBacktest finished with no records generated.")
        print("Detailed skip diagnostics:")
        print(f" - Empty rows or less than {len(tickers)} tickers: {skip_reasons['empty_or_missing_tickers']}")
        print(f" - Missing column '{REGIME_COL}' or null values: {skip_reasons['missing_regime_col']}")
        print(f" - Missing forward log return data: {skip_reasons['missing_forward_returns']}")
        print(f" - Unhandled calculation exceptions: {skip_reasons['other_error']}")
        print("\nVerify that HMM outputs and feature columns match the structural schema requirements.")
        return
        
    df_results = pd.DataFrame(portfolio_records)
    df_results.set_index('Date', inplace=True)
    
    # Calculate performance analytics
    cum_returns = (1 + df_results['Net_Return']).cumprod() - 1
    total_return = cum_returns.iloc[-1]
    ann_sharpe = (df_results['Net_Return'].mean() / (df_results['Net_Return'].std() + 1e-8)) * np.sqrt(52)

    cum_benchmark = (1 + df_results['Benchmark_Return']).cumprod() - 1
    benchmark_total_return = cum_benchmark.iloc[-1]
    benchmark_sharpe = (df_results['Benchmark_Return'].mean() / (df_results['Benchmark_Return'].std() + 1e-8)) * np.sqrt(52)

    print("\n" + "="*50)
    print(f"SWING-TRADE STRATEGY PERFORMANCE REPORT ({min_data_date.year}-{max_data_date.year})")
    print("="*50)
    print(f"{'Metric':<28}{'Strategy':>12}{'Equal-Wt Benchmark':>22}")
    print(f"{'Total Cumulative Return':<28}{total_return*100:>11.2f}%{benchmark_total_return*100:>21.2f}%")
    print(f"{'Annualized Sharpe Ratio':<28}{ann_sharpe:>12.4f}{benchmark_sharpe:>22.4f}")
    print(f"{'Average Weekly Turnover':<28}{df_results['Turnover'].mean()*100:>11.2f}%{'—':>22}")
    print("="*50)

    # 4. Render diagnostic visuals summarizing the backtest run
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    chart_path = os.path.join(REPORTS_DIR, "backtest_results_dashboard.png")
    plot_backtest_results(df_results, tickers, chart_path)

if __name__ == "__main__":
    run_rolling_backtest()