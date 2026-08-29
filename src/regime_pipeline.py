# regime_pipeline.py
import os
import pandas as pd
import numpy as np
from utils import fit_market_hmm, generate_regime_features, process_local_chunks, merge_skill_extremized_signal

if __name__ == "__main__":
    # Suppress PyMC C++ compilation warnings globally
    os.environ["PYTENSOR_FLAGS"] = "cxx="
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      
    if os.path.basename(SCRIPT_DIR) == "src":
        BASE_DIR = os.path.dirname(SCRIPT_DIR)
    else:
        BASE_DIR = SCRIPT_DIR
                        
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    # Analysis configuration bounds
    START_DATE = "2014-01-01"
    END_DATE = "2022-03-01"
    VALIDATED_UNIVERSE = ["AAPL", "AMD", "SPY", "TSLA"]

    # --- Train / Validation / Test split boundaries -----------------------
    # The HMM is fit ONLY on data up to TRAIN_END. Regime labels for the
    # validation and test windows are then decoded (not re-fit) using that
    # train-only model, so no regime boundary anywhere in validation/test
    # was chosen with hindsight of data past TRAIN_END. These three
    # boundaries must stay in sync with the matching constants in
    # regime_models.py and backtester.py -- see methodology_and_results.md
    # for the full split table and rationale.
    TRAIN_END = "2019-12-31"          # HMM + Bayesian fitting window ends here
    VALIDATION_START = "2020-01-15"   # ~2-week embargo after TRAIN_END
    VALIDATION_END = "2020-12-31"
    TEST_START = "2021-01-15"         # ~2-week embargo after VALIDATION_END
    TEST_END = END_DATE
    # ------------------------------------------------------------------------
    
    print("="*70)
    print("STARTING MULTI-ASSET HIDDEN MARKOV MODEL REGIME PIPELINE")
    print("="*70)
    
    fitted_models = {}
    
    for ticker in VALIDATED_UNIVERSE:
        print(f"\nProcessing Regime Space for Matrix Flux: {ticker}")
        print("-" * 50)
        
        panel_filename = f"{ticker}_processed_panel.csv"
        panel_path = os.path.join(PROCESSED_DATA_DIR, panel_filename)
        
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

            # 1.05 Merge in the Stage 1.5 skill-weighted + ANOVA-extremized
            #      signal (skill_weighting.py output) as a new
            #      'Sentiment_Extremized' column, alongside the existing
            #      naive 'Sentiment_Mean' -- lets fit_hierarchical_bayes()
            #      run the ablation between the two without touching the
            #      original column. No-ops (prints a warning, leaves the
            #      panel unchanged) if skill_weighting.py hasn't been run
            #      for this ticker yet.
            df_feat = merge_skill_extremized_signal(df_feat, ticker, PROCESSED_DATA_DIR)

            # 1.1 Train-only slice for HMM fitting. Everything after
            #     TRAIN_END (validation + test) is decoded using this
            #     model's .predict()/.predict_proba(), never used to fit it.
            df_feat_train = df_feat[df_feat.index <= TRAIN_END]
            if len(df_feat_train) < 100:
                print(f"Skipping {ticker}: train-window slice has only "
                      f"{len(df_feat_train)} rows (<100), too little data "
                      f"to fit an HMM before TRAIN_END={TRAIN_END}.")
                continue

            # 2. Gaussian HMM State Decoding Engine -- fit on train, apply to full
            hmm_model, df_regimes = fit_market_hmm(
                df_feat_train, n_regimes=2, df_apply=df_feat
            )
            fitted_models[ticker] = hmm_model
            
            # 3. Save the enriched panel
            output_destination = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_with_regimes.csv")
            df_regimes.to_csv(output_destination)
            
            # 4. Clean diagnostic logging
            print(f"Success! Enriched regime tensor saved to: {output_destination}")
            print(f"Matrix Dimension Profile: {df_regimes.shape} "
                  f"(fit on {len(df_feat_train)} train-only rows through {TRAIN_END}, "
                  f"decoded across the full {len(df_feat)}-row series)")
            print(f"Decoded State Counts (0 = Low-Vol, 1 = High-Vol):\n{df_regimes['Hidden_State'].value_counts()}")
            print("\nStationary State Transition Probability Matrix (train-only fit):")
            print(np.round(hmm_model.transmat_, 4))
            
        except Exception as e:
            print(f"Critical execution fault processing ticker {ticker}: {str(e)}")
            
    print("\n" + "="*70)
    print("ALL TARGET REGIME MATRICES COMPILED FOR HIERARCHICAL BAYES INPUT")
    print("="*70)