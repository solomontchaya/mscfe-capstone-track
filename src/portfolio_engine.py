import os
import pandas as pd
from utils import load_saved_posterior, generate_bayesian_inputs, optimize_portfolio

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    
    # Setup dummy 'current' out-of-sample data points for illustration
    tickers = ["TSLA", "AAPL", "AMZN", "NVDA"]
    mock_features = pd.DataFrame({
        'Argument_Similarity': [0.65, 0.50, 0.55, 0.70],
        'Sentiment_Variance': [2.1, 1.2, 1.5, 3.4]
    }, index=tickers)
    
    # Assume the current Markov hidden state detected by your HMM layer is Regime 0
    current_regime = 1 
    
    print(f"[ENG] Ingesting Posteriors for Active Regime State: {current_regime}")
    try:
        idata = load_saved_posterior(BASE_DIR, current_regime)
        
        # 1. Extract conditioned inputs
        mu_b, sigma_b = generate_bayesian_inputs(idata, mock_features, tickers)
        
        print("\n--- Conditional Bayesian Vectors Derived ---")
        for i, ticker in enumerate(tickers):
            print(f"{ticker} -> Expected Return: {mu_b[i]:.4f}")
            
        # 2. Run allocation
        optimal_weights = optimize_portfolio(mu_b, sigma_b)
        
        print("=======================================================")
        print("OPTIMAL ALLOCATION WEIGHTS (BAYESIAN REGIME CONDITIONED)")
        print("=======================================================")
        for ticker, weight in zip(tickers, optimal_weights):
            print(f"{ticker}: {weight * 100:.2f}%")
        print("=======================================================")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Hint: Please save your netCDF files in regime_models.py first.")