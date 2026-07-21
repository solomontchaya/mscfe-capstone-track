import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from utils import load_saved_posterior, generate_bayesian_inputs, optimize_portfolio

def plot_efficient_frontier(mu_b, sigma_b, tickers, optimal_weights, output_path):
    """
    Generates a rigorous Efficient Frontier scatter plot mapping asset space, 
    random portfolios, and highlights the current optimal allocation.
    """
    sns.set_theme(style="whitegrid")
    n_assets = len(tickers)
    
    # 1. Simulate Random Portfolios for Contextual Background Density
    np.random.seed(42)
    n_simulations = 5000
    sim_returns = np.zeros(n_simulations)
    sim_vols = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        weights = np.random.dirichlet(np.ones(n_assets))
        sim_returns[i] = np.dot(weights, mu_b)
        sim_vols[i] = np.sqrt(np.dot(weights.T, np.dot(sigma_b, weights)))
        
    # 2. Compute Target Optimal Portfolio Space Metrics
    opt_return = np.dot(optimal_weights, mu_b)
    opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(sigma_b, optimal_weights)))
    
    # 3. Render the Primary Visualization Engine
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Frontier Scatter
    scatter = ax1.scatter(sim_vols, sim_returns, c=sim_returns/sim_vols, 
                          cmap='viridis', marker='o', s=4, alpha=0.4, label='Random Portfolios')
    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label('Sharpe Ratio (Ex-Ante)', fontsize=11)
    
    # Plot Individual Assets
    for i, ticker in enumerate(tickers):
        asset_vol = np.sqrt(sigma_b[i, i])
        ax1.scatter(asset_vol, mu_b[i], color='red', marker='X', s=100, zorder=5)
        ax1.annotate(ticker, (asset_vol, mu_b[i]), textcoords="offset points", 
                     xytext=(5,5), ha='left', weight='bold', fontsize=10)
        
    # Plot Maximum Sharpe Ratio (MSR) Target Node
    ax1.scatter(opt_vol, opt_return, color='magenta', marker='*', s=250, 
                zorder=10, label='Optimal Portfolio (MSR)')
    
    ax1.set_title("Bayesian Efficient Frontier Space (Active Regime)", fontsize=13, weight='bold')
    ax1.set_xlabel("Expected Portfolio Volatility", fontsize=11)
    ax1.set_ylabel("Expected Portfolio Return", fontsize=11)
    ax1.legend(loc='best', frameon=True)
    
    # 4. Render Asset Allocation Breakdown Bar Graph
    colors = sns.color_palette("muted", n_assets)
    bars = ax2.bar(tickers, optimal_weights * 100, color=colors, edgecolor='black', alpha=0.85)
    
    # Annotate heights cleanly
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', weight='bold')
                     
    ax2.set_title("Optimal Regime-Conditioned Allocations", fontsize=13, weight='bold')
    ax2.set_xlabel("Asset Class", fontsize=11)
    ax2.set_ylabel("Portfolio Weight (%)", fontsize=11)
    ax2.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[VISUAL] Efficient frontier analysis successfully exported to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Structural project root directory mapping
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    tickers = ["AAPL", "AMD", "SPY", "TSLA"]
    
    mock_features = pd.DataFrame({
        'Sentiment_Mean': [0.65, 0.50, 0.55, 0.70],     
        'Sentiment_Variance': [2.1, 1.2, 1.5, 3.4]
    }, index=tickers)
    mock_features.index.name = 'ticker'
    
    current_regime = 1  # 0: Low-Vol Expansion, 1: High-Vol Stressed Bear
    
    print(f"[ENG] Ingesting Posteriors for Active Regime State: {current_regime}")
    try:
        # Load serialized posterior NetCDF trace
        idata = load_saved_posterior(BASE_DIR, current_regime)
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
            print(f"{ticker} -> Expected Return: {mu_b[i]:.6f}")
            
        optimal_weights = optimize_portfolio(mu_b, sigma_b)
        
        print("\n=========================================================")
        print("OPTIMAL ALLOCATION WEIGHTS (BAYESIAN REGIME CONDITIONED)")
        print("=========================================================")
        for ticker, weight in zip(tickers, optimal_weights):
            print(f"{ticker}: {weight * 100:.2f}%")
        print("=========================================================")
        
        # Trigger explicit chart building pipelines
        output_chart_file = os.path.join(REPORTS_DIR, f"regime_{current_regime}_optimization.png")
        plot_efficient_frontier(mu_b, sigma_b, tickers, optimal_weights, output_chart_file)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR]: Trace target file missing: {e}")
    except ValueError as e:
        print(f"\n[ERROR]: Dimensionality or index mismatch occurred inside optimization block: {e}")
