import os
import arviz as az
from utils import load_regime_data, fit_hierarchical_bayes

if __name__ == "__main__":
    # Structural project root directory mapping
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    # Target directory for trace persistence
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    
    TARGET_UNIVERSE = ["TSLA", "AAPL", "AMZN", "NVDA"]
    
    # 1. Compile dataset
    print(f"[PRE-FLIGHT] Checking processed files in {PROCESSED_DIR}...")
    df_universe = load_regime_data(PROCESSED_DIR, TARGET_UNIVERSE)
    
    print(f"[PRE-FLIGHT] Successfully compiled {len(df_universe)} combined rows across {len(TARGET_UNIVERSE)} assets.")
    print("Row breakdown by asset:")
    if 'Asset' in df_universe.columns:
        print(df_universe['Asset'].value_counts())
    
    # Dictionary container to hold posteriors for downstream portfolio feeding
    regime_posteriors = {}
    
    # 2. Iterate sequentially through both regimes
    for regime in [0, 1]:
        print("\n" + "="*75)
        print(f"COMPUTING HIERARCHICAL BAYES POSTERIORS FOR REGIME STATE: {regime}")
        print("="*75)
        
        # Track sampling duration
        import time
        start_sampling = time.time()
        
        # Fit model
        idata, assets = fit_hierarchical_bayes(df_universe, regime_id=regime)
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
        
        # CRITICAL FIX: Explicitly cast to float to protect against ValueError string formatting exceptions
        max_rhat = float(summary['r_hat'].max())
        print(f"\nMax Gelman-Rubin (R-hat) Score: {max_rhat:.4f}")
        
        if max_rhat > 1.05:
            print("WARNING: MCMC chains haven't fully mixed. Consider expanding tuning bounds or increasing samples.")
        else:
            print("SUCCESS: MCMC chains successfully converged without structural leakage.")
            
        # Write Posterior Trace Asset directly to disk
        output_file_path = os.path.join(DATA_OUTPUT_DIR, f"regime_{regime}_posterior.nc")
        print(f"[SERIALIZE] Preserving trace context to {output_file_path}...")
        idata.to_netcdf(output_file_path)
        print(f"NetCDF Asset Saved Successfully for Regime {regime}.")
            
    print("\n" + "="*75)
    print("BAYESIAN REGIME PARAMETER CORES COMPILED FOR PORTFOLIO ENGINE GENERATION")
    print("="*75)