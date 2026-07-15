import os
import numpy as np
import pandas as pd
from utils import load_saved_posterior, generate_bayesian_inputs, optimize_portfolio

if __name__ == "__main__":
    # Structural project root directory mapping
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    
    # Setup out-of-sample data points for asset allocation
    tickers = ["TSLA", "AAPL", "AMZN", "NVDA"]
    mock_features = pd.DataFrame({
        'Argument_Similarity': [0.65, 0.50, 0.55, 0.70],
        'Sentiment_Variance': [2.1, 1.2, 1.5, 3.4]
    }, index=tickers)
    
    # Active Hidden Markov Model state detected for allocation (1 = High Volatility / Stressed)
    current_regime = 1 
    
    print(f"[ENG] Ingesting Posteriors for Active Regime State: {current_regime}")
    try:
        # Load serialized posterior NetCDF trace
        idata = load_saved_posterior(BASE_DIR, current_regime)
        
        # 1. Coordinate Alignment Hook
        # Extract the coordinate labels directly from ArviZ to prevent column-index misalignments
        if hasattr(idata, "posterior") and "Asset" in idata.posterior.coords:
            trace_assets = list(idata.posterior.coords["Asset"].values)
            # Reorder our incoming ticker features to match the exact index mapping of the sampler
            mock_features = mock_features.reindex(trace_assets)
            tickers = trace_assets
            print(f"[ALIGN] Aligning feature matrix order with posterior coordinates: {tickers}")
        
        # 2. Extract conditioned inputs
        mu_b, sigma_b = generate_bayesian_inputs(idata, mock_features, tickers)
        
        # SAFE-GUARD: Flatten ONLY the expected return vector mu_b to 1-D (shape: 4,)
        # Keep sigma_b as a 2-D covariance matrix (shape: 4x4)
        mu_b_flat = mu_b.flatten()
        
        # Extract the diagonal (variance of each asset) for representation printing
        if sigma_b.ndim == 2:
            diag_uncertainty = np.diag(sigma_b)
        else:
            diag_uncertainty = sigma_b.flatten()
        
        print("\n--- Conditional Bayesian Vectors Derived ---")
        for i, ticker in enumerate(tickers):
            print(f"{ticker} -> Expected Return: {mu_b_flat[i]:.6f} | Marginal Variance (Uncertainty): {diag_uncertainty[i]:.6f}")
            
        # 3. Run allocation (Dynamic Mean-Variance / Kelly Optimization)
        # Passing mu_b_flat (1D vector) and sigma_b (2D covariance matrix)
        optimal_weights = optimize_portfolio(mu_b_flat, sigma_b)
        
        print("\n=======================================================")
        print("OPTIMAL ALLOCATION WEIGHTS (BAYESIAN REGIME CONDITIONED)")
        print("=======================================================")
        for ticker, weight in zip(tickers, optimal_weights):
            print(f"{ticker}: {weight * 100:.2f}%")
        print("=======================================================")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Hint: Please save your netCDF files in regime_models.py first.")
    