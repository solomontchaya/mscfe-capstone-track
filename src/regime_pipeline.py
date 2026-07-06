import os
import pandas as pd
import numpy as np
from utils import fit_market_hmm, generate_regime_features

if __name__ == "__main__":
    
    # Establish a fixed structural anchor based on this script's location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      
    BASE_DIR = os.path.dirname(SCRIPT_DIR)                       
    
    # Anchor directly to the verified processed absolute directory layout
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    # The 4 fully validated, truncated tickers matching your 545-row panels
    VALIDATED_UNIVERSE = ["TSLA", "AAPL", "AMZN", "NVDA"]
    
    print("="*70)
    print("STARTING MULTI-ASSET HIDDEN MARKOV MODEL REGIME PIPELINE")
    print("="*70)
    
    # Dictionary to store fitted models if needed for downstream serialization/pickling
    fitted_models = {}
    
    for ticker in VALIDATED_UNIVERSE:
        print(f"\nProcessing Regime Space for Matrix Flux: {ticker}")
        print("-" * 50)
        
        # Build paths dynamically based on your file naming structure
        potential_files = [
            f"{ticker}_processed_panel.csv",
            f"{ticker}.csv"
        ]
        
        data_path = None
        for filename in potential_files:
            test_path = os.path.join(PROCESSED_DATA_DIR, filename)
            if os.path.exists(test_path):
                data_path = test_path
                break
                
        if data_path is None:
            print(f"Skipping {ticker}: No verified source file found in {PROCESSED_DATA_DIR}")
            continue
            
        try:
            # 1. Feature generation layer (Extracts log returns and high-low spread matrices)
            df_feat = generate_regime_features(data_path)
            
            # 2. Gaussian HMM State Decoding Engine (2-state system: Low vs High Volatility)
            hmm_model, df_regimes = fit_market_hmm(df_feat, n_regimes=2)
            
            # Save the trained model instance in memory
            fitted_models[ticker] = hmm_model
            
            # 3. Save the enriched panel directly back over the target file or an explicit copy
            output_destination = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_with_regimes.csv")
            df_regimes.to_csv(output_destination)
            
            # 4. Diagnostics output
            print(f"Success! Enriched regime tensor saved to: {output_destination}")
            print(f"Matrix Dimension Profile: {df_regimes.shape}")
            print(f"Decoded State Counts:\n{df_regimes['Hidden_State'].value_counts()}")
            print("\nStationary State Transition Probability Matrix:")
            print(np.round(hmm_model.transmat_, 4))
            
        except Exception as e:
            print(f"Critical execution fault processing ticker {ticker}: {str(e)}")
            
    print("\n" + "="*70)
    print("ALL TARGET REGIME MATRICES COMPILED FOR HIERARCHICAL BAYES INPUT")
    print("="*70)