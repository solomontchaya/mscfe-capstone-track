# regime_models.py
import os
import sys
import arviz as az
from utils import load_regime_data, fit_hierarchical_bayes

if __name__ == "__main__":
    # --sentiment-column=Sentiment_Extremized runs the ablation using the
    # skill-weighted + ANOVA-extremized signal instead of the original
    # naive Sentiment_Mean. Output files get a matching suffix so neither
    # run overwrites the other -- run both and diff the two summaries.
    SENTIMENT_COLUMN = "Sentiment_Mean"
    for arg in sys.argv:
        if arg.startswith("--sentiment-column="):
            SENTIMENT_COLUMN = arg.split("=", 1)[1]
    OUTPUT_SUFFIX = "" if SENTIMENT_COLUMN == "Sentiment_Mean" else f"_{SENTIMENT_COLUMN.lower()}"

    # Structural project root directory mapping
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    # Target directory for trace persistence
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    
    TARGET_UNIVERSE = ["AAPL", "AMD", "SPY", "TSLA"]

    # Must match regime_pipeline.py's TRAIN_END exactly -- the regime labels
    # in the loaded CSVs were decoded across the full series, but the
    # Bayesian posteriors must only ever see rows up to this date, or the
    # coefficients would carry validation/test hindsight.
    TRAIN_END = "2019-12-31"

    # 1. Compile dataset
    print(f"[PRE-FLIGHT] Checking processed files in {PROCESSED_DIR}...")
    print(f"[PRE-FLIGHT] Sentiment predictor for this run: {SENTIMENT_COLUMN}")
    df_universe_full = load_regime_data(PROCESSED_DIR, TARGET_UNIVERSE)

    df_universe = df_universe_full[df_universe_full.index <= TRAIN_END].copy()
    print(f"[PRE-FLIGHT] Loaded {len(df_universe_full)} total rows across "
          f"{len(TARGET_UNIVERSE)} assets; restricting to {len(df_universe)} "
          f"train-only rows through {TRAIN_END} for posterior fitting.")
    
    print(f"[PRE-FLIGHT] Successfully compiled {len(df_universe)} combined rows across {len(TARGET_UNIVERSE)} assets.")
    print("Row breakdown by asset:")
    if 'Asset' in df_universe.columns:
        print(df_universe['Asset'].value_counts())
    
    # Dictionary container to hold posteriors for downstream portfolio feeding
    regime_posteriors = {}
    
    # 2. Iterate sequentially through both regimes
    for regime in [0, 1]:
        print("\n" + "="*75)
        print(f"COMPUTING HIERARCHICAL BAYES POSTERIORS FOR REGIME STATE: {regime} "
              f"(sentiment_column={SENTIMENT_COLUMN})")
        print("="*75)
        
        # Track sampling duration
        import time
        start_sampling = time.time()
        
        # Fit model
        idata, assets = fit_hierarchical_bayes(df_universe, regime_id=regime, sentiment_column=SENTIMENT_COLUMN)
        regime_posteriors[regime] = idata
        
        sampling_duration = time.time() - start_sampling
        print(f"Sampling for Regime {regime} finished in {sampling_duration/60:.2f} minutes.")
        
        # 3. Run Quality and Convergence Diagnostics ---
        summary = az.summary(idata, var_names=['beta_sim', 'beta_var'])
        
        # Version-agnostic column slicing filter for ArviZ HDI naming variations
        hdi_cols = [c for c in summary.columns if 'hdi' in c.lower()]
        target_cols = ['mean', 'sd'] + hdi_cols + ['r_hat', 'ess_bulk']
        valid_cols = [c for c in target_cols if c in summary.columns]
        
        print(f"\n[DIAGNOSTICS] Convergence Summary for Regime {regime}:")
        print(summary[valid_cols])

        # Save the summary table itself -- this is your Table 4.1/4.2 data,
        # previously only visible in console output. Saving it lets you
        # diff the Sentiment_Mean run against the Sentiment_Extremized run
        # directly instead of re-transcribing console output by hand.
        summary_csv_path = os.path.join(
            DATA_OUTPUT_DIR, f"regime_{regime}_coefficients{OUTPUT_SUFFIX}.csv"
        )
        summary[valid_cols].to_csv(summary_csv_path)
        print(f"[SERIALIZE] Coefficient summary saved to {summary_csv_path}")
        
        # CRITICAL FIX: Explicitly cast to float to protect against ValueError string formatting exceptions
        max_rhat = float(summary['r_hat'].max())
        print(f"\nMax Gelman-Rubin (R-hat) Score: {max_rhat:.4f}")
        
        if max_rhat > 1.05:
            print("WARNING: MCMC chains haven't fully mixed. Consider expanding tuning bounds or increasing samples.")
        else:
            print("SUCCESS: MCMC chains successfully converged without structural leakage.")
            
        # Write Posterior Trace Asset directly to disk
        output_file_path = os.path.join(DATA_OUTPUT_DIR, f"regime_{regime}_posterior{OUTPUT_SUFFIX}.nc")
        print(f"[SERIALIZE] Preserving trace context to {output_file_path}...")
        idata.to_netcdf(output_file_path)
        print(f"NetCDF Asset Saved Successfully for Regime {regime}.")
            
    print("\n" + "="*75)
    print("BAYESIAN REGIME PARAMETER CORES COMPILED FOR PORTFOLIO ENGINE GENERATION")
    print("="*75)