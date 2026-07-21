import os
import pandas as pd
import numpy as np
from utils import fit_market_hmm, generate_regime_features, process_local_chunks

if __name__ == "__main__":
    # Suppress PyMC C++ compilation warnings globally
    os.environ["PYTENSOR_FLAGS"] = "cxx="
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      
    if os.path.basename(SCRIPT_DIR) == "src":
        BASE_DIR = os.path.dirname(SCRIPT_DIR)
    else:
        BASE_DIR = SCRIPT_DIR
                        
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    
<<<<<<< HEAD
    # The 4 fully validated universe tickers
    VALIDATED_UNIVERSE = ["TSLA", "AAPL", "AMZN", "NVDA"]
=======
    # Analysis configuration bounds
    START_DATE = "2014-01-01"
    END_DATE = "2015-12-31" 
    VALIDATED_UNIVERSE = ["AAPL", "AMD", "SPY", "TSLA"]
>>>>>>> 8ab4591 (Backtest Update)
    
    print("="*70)
    print("STARTING MULTI-ASSET HIDDEN MARKOV MODEL REGIME PIPELINE")
    print("="*70)
    
    fitted_models = {}
    
    for ticker in VALIDATED_UNIVERSE:
        print(f"\nProcessing Regime Space for Matrix Flux: {ticker}")
        print("-" * 50)
        
<<<<<<< HEAD
        # Prioritize matching the exact [TICKER].csv structure outputted by S3 pipeline
        potential_files = [
            f"{ticker}.csv",
            f"{ticker}_processed_panel.csv"
        ]
=======
        panel_filename = f"{ticker}_processed_panel.csv"
        panel_path = os.path.join(PROCESSED_DATA_DIR, panel_filename)
>>>>>>> 8ab4591 (Backtest Update)
        
        # Check and merge upstream files if missing
        if not os.path.exists(panel_path):
            print(f"Target panel not found at {panel_path}. Orchestrating upstream merge engine...")
            process_local_chunks(
                raw_data_dir=PROCESSED_DATA_DIR, 
                output_csv_path=panel_path, 
                ticker=ticker, 
                start_date=START_DATE, 
                end_date=END_DATE
            )
        
        if not os.path.exists(panel_path):
            print(f"Skipping {ticker}: Failed to construct integrated market-sentiment panel.")
            continue
            
        try:
            # 1. Feature generation layer
            df_feat = generate_regime_features(panel_path)
            
<<<<<<< HEAD
            # 2. Gaussian HMM State Decoding Engine (2-state system: Low vs High Volatility)
            # Fits HMM over returns/spreads with multistart seed search for global convergence
=======
            # 2. Gaussian HMM State Decoding Engine
>>>>>>> 8ab4591 (Backtest Update)
            hmm_model, df_regimes = fit_market_hmm(df_feat, n_regimes=2)
            fitted_models[ticker] = hmm_model
            
<<<<<<< HEAD
            # 3. Save the enriched panel directly as a downstream input for the Bayesian pipeline
=======
            # 3. Save the enriched panel
>>>>>>> 8ab4591 (Backtest Update)
            output_destination = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_with_regimes.csv")
            df_regimes.to_csv(output_destination)
            
            # 4. Clean diagnostic logging
            print(f"Success! Enriched regime tensor saved to: {output_destination}")
            print(f"Matrix Dimension Profile: {df_regimes.shape}")
            print(f"Decoded State Counts (0 = Low-Vol, 1 = High-Vol):\n{df_regimes['Hidden_State'].value_counts()}")
            print("\nStationary State Transition Probability Matrix:")
            print(np.round(hmm_model.transmat_, 4))
            
        except Exception as e:
            print(f"Critical execution fault processing ticker {ticker}: {str(e)}")
            
    print("\n" + "="*70)
    print("ALL TARGET REGIME MATRICES COMPILED FOR HIERARCHICAL BAYES INPUT")
    print("="*70)