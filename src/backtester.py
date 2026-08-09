import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
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

def plot_backtest_results(df_results, tickers, output_path, title_suffix=""):
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

    ax1.set_title(f"Cumulative Net Return vs. Equal-Weight Benchmark{title_suffix}", fontsize=12, weight='bold')
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Panel 2: Drawdown ---
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color='#d62728', alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values * 100, color='#d62728', linewidth=1.2)
    ax2.set_title(f"Portfolio Drawdown{title_suffix}", fontsize=12, weight='bold')
    ax2.set_ylabel("Drawdown (%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Panel 3: Allocation Weights Over Time (Stacked Area) ---
    colors = sns.color_palette("muted", len(tickers))
    ax3.stackplot(weights_df.index, [weights_df[t] * 100 for t in tickers],
                  labels=tickers, colors=colors, alpha=0.85)
    ax3.set_title(f"Portfolio Allocation Weights Over Time{title_suffix}", fontsize=12, weight='bold')
    ax3.set_ylabel("Weight (%)")
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', frameon=True, fontsize=9, ncol=len(tickers))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Panel 4: Turnover per Rebalance ---
    ax4.bar(df_results.index, df_results['Turnover'] * 100, width=4,
            color='#9467bd', edgecolor='black', alpha=0.8)
    ax4.set_title(f"Turnover per Rebalance{title_suffix}", fontsize=12, weight='bold')
    ax4.set_ylabel("Turnover (%)")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[VISUAL] Backtest results dashboard exported to: {output_path}")
    plt.close()


def run_segment_backtest(master_df, tickers, base_dir, reports_dir,
                          segment_label, segment_start, segment_end,
                          turnover_penalty, max_weight_change):
    """
    Runs the weekly walk-forward backtest restricted to a single date segment
    (e.g. 'Validation' or 'Test'). This is the same per-period mechanics as
    before, just scoped to segment_start..segment_end and reported under a
    clearly labeled heading, so validation-stage results (used to eyeball
    turnover-control settings) are never confused with the held-out test
    result that should be reported once and left alone.

    Portfolio weights reset to equal-weight at the start of each segment --
    each segment is evaluated independently rather than as one continuous
    portfolio path spanning the train/validation/test boundary.

    Returns df_results (or None if the segment produced no records).
    """
    REGIME_COL = 'Hidden_State'

    segment_start_ts = pd.Timestamp(segment_start)
    segment_end_ts = pd.Timestamp(segment_end)

    segment_df = master_df[(master_df['Date'] >= segment_start_ts) & (master_df['Date'] <= segment_end_ts)]
    if segment_df.empty:
        print(f"\n[{segment_label}] No data found in range {segment_start} to {segment_end}. Skipping segment.")
        return None

    rebalance_dates = pd.date_range(start=segment_start_ts, end=segment_end_ts, freq='W-FRI')
    rebalance_df = pd.DataFrame({'Rebalance_Date': rebalance_dates})
    rebalance_df['Rebalance_Date'] = rebalance_df['Rebalance_Date'].dt.normalize()

    available_dates = pd.DataFrame({'Actual_Date': segment_df['Date'].unique()}).sort_values('Actual_Date')

    aligned = pd.merge_asof(
        rebalance_df,
        available_dates,
        left_on='Rebalance_Date',
        right_on='Actual_Date',
        direction='nearest',
        tolerance=pd.Timedelta(days=4)
    ).dropna().reset_index(drop=True)

    if aligned.empty:
        print(f"\n[{segment_label}] No aligned rebalance dates found in range {segment_start} to {segment_end}. Skipping segment.")
        return None

    portfolio_records = []
    current_weights = np.array([1.0 / len(tickers)] * len(tickers))  # Start equal-weighted at segment start

    direction_predicted = []
    direction_actual = []

    skip_reasons = {
        'empty_or_missing_tickers': 0,
        'missing_regime_col': 0,
        'missing_forward_returns': 0,
        'other_error': 0
    }

    print(f"\n[{segment_label}] Initializing backtest across {len(aligned)} aligned periods "
          f"({segment_start} to {segment_end})...")

    for idx, row in aligned.iterrows():
        actual_date = row['Actual_Date']
        date_str = actual_date.strftime('%Y-%m-%d')

        # NOTE: forward returns are looked up against the FULL master_df, not
        # segment_df, so a rebalance on the last Friday of a segment can still
        # correctly evaluate against the following week's realized return
        # even if that date falls just past segment_end. This is standard
        # walk-forward practice -- forward returns are realized market
        # outcomes, not model parameters, so using the next available price
        # print introduces no train/validation/test leakage.
        period_df = master_df[master_df['Date'] == actual_date]

        if period_df.empty or len(period_df) < len(tickers):
            skip_reasons['empty_or_missing_tickers'] += 1
            continue

        period_df = period_df.set_index('Ticker').reindex(tickers)

        if REGIME_COL not in period_df.columns or period_df[REGIME_COL].isna().any():
            skip_reasons['missing_regime_col'] += 1
            continue

        predicted_regime = int(period_df[REGIME_COL].iloc[0])
        real_features = period_df[['Sentiment_Mean', 'Sentiment_Variance']]

        # Forward return lookup: use the NEXT SCHEDULED REBALANCE DATE within
        # this segment (matching the weekly holding period turnover costs are
        # computed against) for every period except the last. Only the
        # segment's final period needs to look past segment_end into the
        # full master_df, so its return isn't computed against same-day data.
        if idx + 1 < len(aligned):
            next_actual_date = aligned.loc[idx + 1, 'Actual_Date']
        else:
            all_dates_sorted = np.sort(master_df['Date'].unique())
            later_dates = all_dates_sorted[all_dates_sorted > np.datetime64(actual_date)]
            next_actual_date = pd.Timestamp(later_dates[0]) if len(later_dates) > 0 else None

        if next_actual_date is not None:
            next_period_df = master_df[master_df['Date'] == next_actual_date].set_index('Ticker').reindex(tickers)
            if next_period_df['log_ret'].isna().any():
                skip_reasons['missing_forward_returns'] += 1
                continue
            forward_returns = next_period_df['log_ret'].values
        else:
            forward_returns = period_df['log_ret'].values  # Fallback for terminal calculation

        try:
            idata = load_saved_posterior(base_dir, predicted_regime)
            mu_b, sigma_b = generate_bayesian_inputs(idata, real_features, tickers)
            mu_b_flat = mu_b.flatten()

            predicted_direction = (mu_b_flat > 0).astype(int)
            actual_direction = (forward_returns > 0).astype(int)
            direction_predicted.extend(predicted_direction.tolist())
            direction_actual.extend(actual_direction.tolist())

            new_weights = optimize_portfolio(
                mu_b_flat, sigma_b,
                prev_weights=current_weights,
                turnover_penalty=turnover_penalty,
                max_weight_change=max_weight_change
            )

            turnover = np.sum(np.abs(new_weights - current_weights))
            tx_cost = turnover * 0.0010  # 10 bps execution penalty

            raw_p_return = np.dot(new_weights, forward_returns)
            net_p_return = raw_p_return - tx_cost

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

            current_weights = new_weights
            print(f"[{segment_label}] {date_str} | Regime: {predicted_regime} | Net Return: {net_p_return:.4f}")

        except FileNotFoundError:
            print(f"Missing trace file for Regime {predicted_regime}. Run training loop first.")
            return None
        except Exception as e:
            skip_reasons['other_error'] += 1
            if skip_reasons['other_error'] <= 5:
                print(f"[DEBUG] Operational loop error on {date_str}: {e}")

    if not portfolio_records:
        print(f"\n[{segment_label}] Backtest finished with no records generated.")
        print("Detailed skip diagnostics:")
        print(f" - Empty rows or less than {len(tickers)} tickers: {skip_reasons['empty_or_missing_tickers']}")
        print(f" - Missing column 'Hidden_State' or null values: {skip_reasons['missing_regime_col']}")
        print(f" - Missing forward log return data: {skip_reasons['missing_forward_returns']}")
        print(f" - Unhandled calculation exceptions: {skip_reasons['other_error']}")
        return None

    df_results = pd.DataFrame(portfolio_records)
    df_results.set_index('Date', inplace=True)

    cum_returns = (1 + df_results['Net_Return']).cumprod() - 1
    total_return = cum_returns.iloc[-1]
    ann_sharpe = (df_results['Net_Return'].mean() / (df_results['Net_Return'].std() + 1e-8)) * np.sqrt(52)

    cum_benchmark = (1 + df_results['Benchmark_Return']).cumprod() - 1
    benchmark_total_return = cum_benchmark.iloc[-1]
    benchmark_sharpe = (df_results['Benchmark_Return'].mean() / (df_results['Benchmark_Return'].std() + 1e-8)) * np.sqrt(52)

    print("\n" + "="*50)
    print(f"SWING-TRADE STRATEGY PERFORMANCE REPORT — {segment_label.upper()} ({segment_start} to {segment_end})")
    print("="*50)
    print(f"{'Metric':<28}{'Strategy':>12}{'Equal-Wt Benchmark':>22}")
    print(f"{'Total Cumulative Return':<28}{total_return*100:>11.2f}%{benchmark_total_return*100:>21.2f}%")
    print(f"{'Annualized Sharpe Ratio':<28}{ann_sharpe:>12.4f}{benchmark_sharpe:>22.4f}")
    print(f"{'Average Weekly Turnover':<28}{df_results['Turnover'].mean()*100:>11.2f}%{'—':>22}")
    print("="*50)

    y_pred = np.array(direction_predicted)
    y_true = np.array(direction_actual)

    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        accuracy = (y_pred == y_true).mean()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        majority_class = int(y_true.mean() > 0.5)
        baseline_pred = np.full_like(y_true, majority_class)
        baseline_f1 = f1_score(y_true, baseline_pred, pos_label=1, zero_division=0)

        print(f"\nDIRECTIONAL SKILL DIAGNOSTIC — {segment_label.upper()}")
        print("-"*50)
        print(f"Observations (assets x periods): {len(y_true)}")
        print(f"Accuracy:                      {accuracy*100:.2f}%")
        print(f"Precision (predicted 'up'):    {precision:.4f}")
        print(f"Recall (predicted 'up'):       {recall:.4f}")
        print(f"F1 Score:                      {f1:.4f}")
        print(f"  vs. majority-class baseline: {baseline_f1:.4f}")
        print(f"Confusion Matrix  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
        print("-"*50)
    else:
        print(f"\n[{segment_label}] Skipped F1 scoring: insufficient class variation in realized returns.")

    chart_path = os.path.join(reports_dir, f"backtest_results_dashboard_{segment_label.lower()}.png")
    plot_backtest_results(df_results, tickers, chart_path, title_suffix=f" — {segment_label}")

    return df_results


def run_rolling_backtest():
    # 1. Structural Path mapping
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    tickers = ["AAPL", "AMD", "SPY", "TSLA"]

    # --- Train / Validation / Test split boundaries -----------------------
    # Must match regime_pipeline.py's TRAIN_END and regime_models.py's
    # TRAIN_END exactly -- those two scripts are what actually enforce the
    # split (HMM fit + Bayesian posteriors both stop at TRAIN_END). This
    # backtester never touches train-period data at all: it only evaluates
    # the already-fitted models across the validation and test windows,
    # each reported separately.
    #
    # Intended usage: inspect the Validation report/dashboard first, and if
    # you want to adjust TURNOVER_PENALTY / MAX_WEIGHT_CHANGE below, do so
    # based on validation results only. Once you're satisfied, run once
    # more and treat the Test report as final -- re-tuning after looking at
    # Test results defeats the purpose of holding it out.
    VALIDATION_START = "2020-01-15"
    VALIDATION_END = "2020-12-31"
    TEST_START = "2021-01-15"
    TEST_END = "2022-03-01"
    # ------------------------------------------------------------------------

    # Turnover controls: with weak/noisy Bayesian mu estimates, an unconstrained
    # Sharpe-maximizer can flip between concentrated corner solutions week to
    # week. These two knobs bias the optimizer toward smoother reallocation.
    # Tune these against the Validation segment's results only.
    TURNOVER_PENALTY = 0.05      # Soft L1 penalty (lambda) on |w_new - w_prev|
    MAX_WEIGHT_CHANGE = 0.15     # Hard cap: no single asset's weight can move
                                  # more than this per rebalance (None disables)

    try:
        master_df = load_historical_backtest_data(PROCESSED_DIR, tickers)
    except Exception as e:
        print(f"Data Load Error: {e}")
        return

    print(f"[DATA] Successfully loaded master dataset with {len(master_df)} rows.")
    print(f"[DATA] Date range in dataset: {master_df['Date'].min().strftime('%Y-%m-%d')} to {master_df['Date'].max().strftime('%Y-%m-%d')}")

    # --- Validation segment --------------------------------------------------
    validation_results = run_segment_backtest(
        master_df, tickers, BASE_DIR, REPORTS_DIR,
        segment_label="Validation",
        segment_start=VALIDATION_START,
        segment_end=VALIDATION_END,
        turnover_penalty=TURNOVER_PENALTY,
        max_weight_change=MAX_WEIGHT_CHANGE
    )

    # --- Test segment (held out; report once, do not re-tune afterward) -----
    test_results = run_segment_backtest(
        master_df, tickers, BASE_DIR, REPORTS_DIR,
        segment_label="Test",
        segment_start=TEST_START,
        segment_end=TEST_END,
        turnover_penalty=TURNOVER_PENALTY,
        max_weight_change=MAX_WEIGHT_CHANGE
    )

    print("\n" + "="*50)
    print("RUN COMPLETE")
    print("="*50)
    print("Validation segment:", "OK" if validation_results is not None else "NO RECORDS")
    print("Test segment:      ", "OK" if test_results is not None else "NO RECORDS")
    print("Dashboards saved to:", REPORTS_DIR)


if __name__ == "__main__":
    run_rolling_backtest()