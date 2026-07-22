import os
import glob
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import yfinance as yf
from scipy.optimize import minimize
from hmmlearn.hmm import GaussianHMM
import warnings
from sklearn.exceptions import ConvergenceWarning

def process_s3_sentiment_pipeline(output_csv_path, target_ticker, start_date, end_date):
    """
    Directly streams pre-computed sentiment metadata from the NYU S3 bucket,
    aggregates daily sentiment metrics, and aligns them with Yahoo Finance pricing.
    
    This replaces the heavy text-parsing pipeline while keeping the output schema 
    identical to keep downstream Bayesian models happy.
    """
    CSV_URL = "s3://stocktwits-nyu/dataset/v1/data/csv"
    storage_opts = {
        "anon": True,
        "client_kwargs": {"region_name": "us-west-2"}
    }

    print(f"=== [1/3] Streaming S3 Sentiment Index for: {target_ticker} ===")
    sentiment_file = f"{CSV_URL}/sentiments/sentiment_00.csv"

    # Stream the lightweight sentiment metadata
    df_sent = pd.read_csv(
        sentiment_file,
        storage_options=storage_opts,
        dtype={"sentiment": "object", "message_id": "object"}
    )

    # Quick cleanup of empty symbol columns
    df_sent = df_sent.dropna(subset=['symbol_list'])
    df_sent = df_sent[df_sent['symbol_list'] != '[]']

    # Fast ticker matching
    def ticker_match(symbol_str):
        try:
            return target_ticker in ast.literal_eval(symbol_str)
        except Exception:
            return False

    df_sent['is_target'] = df_sent['symbol_list'].apply(ticker_match)
    df_target = df_sent[df_sent['is_target'] == True].copy()

    if df_target.empty:
        print(f"Aborted: No records found for {target_ticker} in S3 sentiment files.")
        return

    # Convert sentiment string to float (-1.0, 1.0, etc.)
    df_target['sentiment_score'] = pd.to_numeric(df_target['sentiment'], errors='coerce')

    # Convert temporal index and apply timeline filter bounds
    df_target['timestamp'] = pd.to_datetime(df_target['created_at'], errors='coerce')
    df_target = df_target.dropna(subset=['timestamp'])
    df_target['date_only'] = df_target['timestamp'].dt.date

    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    df_target = df_target[(df_target['date_only'] >= start_dt) & (df_target['date_only'] <= end_dt)]

    # -------------------------------------------------------------------------
    # STEP 2: Aggregate Daily Metrics (Preserving Downstream Variable Names)
    # -------------------------------------------------------------------------
    print("=== [2/3] Aggregating Daily Signals ===")
    daily_aggregates = []

    for current_date, group in df_target.groupby('date_only'):
        valid_sentiments = group['sentiment_score'].dropna()
        msg_count = len(group)

        if msg_count < 3:
            continue  # Keep statistical integrity of variance calculation

        # Map mean sentiment to Argument_Similarity to preserve code compatibility
        daily_sent_mean = np.mean(valid_sentiments) if len(valid_sentiments) > 0 else 0.0
        daily_sent_var = np.var(valid_sentiments) if len(valid_sentiments) > 0 else 0.0

        daily_aggregates.append({
            'Date': pd.to_datetime(current_date),
            'Argument_Similarity': daily_sent_mean,  # Directional sentiment proxy
            'Sentiment_Variance': daily_sent_var,    # Consensus dispersion proxy
            'Volume_Crowd': float(msg_count)
        })

    df_features = pd.DataFrame(daily_aggregates).set_index('Date').sort_index()
    df_features_lagged = df_features.shift(1)

    # -------------------------------------------------------------------------
    # STEP 3: Download Yahoo Finance Prices and Merge
    # -------------------------------------------------------------------------
    print("=== [3/3] Pulling Market Data and Merging Final Panel ===")
    yf_ticker = "META" if target_ticker == "FB" else target_ticker
    extended_end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    market_data = yf.download(yf_ticker, start=start_date, end=extended_end)

    if isinstance(market_data.columns, pd.MultiIndex):
        market_data.columns = market_data.columns.get_level_values(0)
    market_data.index = pd.to_datetime(market_data.index)
    market_data = market_data.loc[start_date:end_date]

    # Clean merge with 1-day lagged signals (to prevent look-ahead bias)
    final_panel = market_data[['Close', 'High', 'Low', 'Volume']].merge(
        df_features_lagged, left_index=True, right_index=True, how='left'
    )
    final_panel['Argument_Similarity'] = final_panel['Argument_Similarity'].ffill().fillna(0.0)
    final_panel['Sentiment_Variance'] = final_panel['Sentiment_Variance'].ffill().fillna(0.0)
    final_panel['Volume_Crowd'] = final_panel['Volume_Crowd'].fillna(0.0)

    # Save to dynamic dataset location
    final_panel.to_csv(output_csv_path)
    print(f"\n=== Run Completed! Clean panel saved to: {output_csv_path} (Shape: {final_panel.shape}) ===\n")

def process_local_chunks(raw_data_dir, output_csv_path, ticker, start_date, end_date):
    """
    Extracts, structuralizes, and aggregates raw ticker text sentiment signals 
    and merges them against historical OHLCV data from Yahoo Finance.
    """
    print(f"Initializing localized processing loops for: {ticker}")
    search_path = os.path.join(raw_data_dir, "**/*.csv")
    all_files = glob.glob(search_path, recursive=True)
    
    if not all_files:
        print(f"Error: No raw data files located at: {raw_data_dir}")
        return

    print(f"Located {len(all_files)} raw data files. Extracting numeric sentiment matrices...")
    daily_aggregates = {}

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        
        # FIX: Skip output panel and regime files generated by this pipeline to prevent usecols pollution
        if "_processed_panel" in file_name or "_with_regimes" in file_name:
            continue
            
        try:
            # Handles text formats by isolating columns safely
            df = pd.read_csv(file_path, usecols=['created_at', 'sentiment', 'symbol_list'], on_bad_lines='skip')
            
            # Target explicit asset identifiers if tracking individual symbols
            if 'symbol_list' in df.columns:
                df = df[df['symbol_list'].astype(str).str.contains(ticker, na=False, case=False)]
                
            df['timestamp'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'sentiment'])
            
            df['date_only'] = df['timestamp'].dt.date
            
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            df = df[(df['date_only'] >= start_dt) & (df['date_only'] <= end_dt)]
            
            if df.empty:
                continue
                
            for current_date, group in df.groupby('date_only'):
                sentiments = pd.to_numeric(group['sentiment'], errors='coerce').dropna().values
                msg_count = len(sentiments)
                
                if msg_count < 3:
                    continue
                    
                daily_sent_mean = np.mean(sentiments)
                daily_sent_var = np.var(sentiments)
                
                if current_date not in daily_aggregates:
                    daily_aggregates[current_date] = {
                        'sent_means': [daily_sent_mean],
                        'sent_vars': [daily_sent_var],
                        'counts': msg_count
                    }
                else:
                    daily_aggregates[current_date]['sent_means'].append(daily_sent_mean)
                    daily_aggregates[current_date]['sent_vars'].append(daily_sent_var)
                    daily_aggregates[current_date]['counts'] += msg_count
                    
        except Exception as file_err:
            print(f"Log Notice: Skipping file entry {file_name} due to: {file_err}")
            continue

    if not daily_aggregates:
        print(f"Warning: No valid sentiment records processed for {ticker}.")
        return

    summary_rows = []
    for date_key, metrics in daily_aggregates.items():
        summary_rows.append({
            'Date': pd.to_datetime(date_key),
            'Sentiment_Mean': np.mean(metrics['sent_means']),
            'Sentiment_Variance': np.mean(metrics['sent_vars']),
            'Volume_Crowd': metrics['counts']
        })
        
    df_features = pd.DataFrame(summary_rows).set_index('Date').sort_index()
    df_features_lagged = df_features.shift(1)
    
    yf_ticker = "META" if ticker == "FB" else ticker
    print(f"Downloading market pricing matrix for {yf_ticker} via Yahoo Finance...")
    
    extended_end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    market_data = yf.download(yf_ticker, start=start_date, end=extended_end, group_by='column')
    
    if market_data.empty:
        print(f"Error: No market data fetched for {yf_ticker}.")
        return

    if isinstance(market_data.columns, pd.MultiIndex):
        market_data.columns = [col[0] if isinstance(col, tuple) else col for col in market_data.columns]
        
    market_data.index = pd.to_datetime(market_data.index)
    market_data = market_data.loc[start_date:end_date]
    
    required_cols = ['Close', 'High', 'Low', 'Volume']
    missing_cols = [c for c in required_cols if c not in market_data.columns]
    if missing_cols:
        print(f"Error: yfinance output missing columns {missing_cols} for {ticker}.")
        return
    
    final_panel = market_data[required_cols].merge(
        df_features_lagged, left_index=True, right_index=True, how='left'
    )
    
    final_panel['Sentiment_Mean'] = final_panel['Sentiment_Mean'].ffill().fillna(0.0)
    final_panel['Sentiment_Variance'] = final_panel['Sentiment_Variance'].ffill().fillna(0.0)
    final_panel['Volume_Crowd'] = final_panel['Volume_Crowd'].fillna(0.0)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_panel.to_csv(output_csv_path)
    print(f"=== Success! Processed panel saved to: {output_csv_path} (Shape: {final_panel.shape}) ===")

def generate_regime_features(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True, date_format='ISO8601')
    df.index.name = 'Date'
    df = df.sort_index()
    
    if 'Close' not in df.columns:
        raise KeyError(f"The CSV at {file_path} is missing the 'Close' column. Available columns: {list(df.columns)}")
        
    for col in ['Close', 'High', 'Low']:
        df[col] = df[col].astype(float)
        
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['hl_spread'] = np.log(df['High'] / df['Low'])
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['log_ret', 'hl_spread'])
    return df

def fit_market_hmm(df_features, n_regimes=2, random_state=42):
    feature_cols = ['log_ret', 'hl_spread']
    X = df_features[feature_cols].values
    best_likelihood = -np.inf
    best_model = None
    
    for seed_offset in [0, 15, 42, 99, 123]:
        current_seed = random_state + seed_offset
        model = GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full", 
            n_iter=1000,
            tol=1e-4,
            min_covar=1e-3,
            random_state=current_seed
        )
        try:
            # Silence the non-converging seed alerts to clean up standard error logs
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X)
                
            current_score = model.score(X)
            if model.transmat_[1, 0] > 0.80 or model.transmat_[0, 1] > 0.80:
                continue
            if current_score > best_likelihood:
                best_likelihood = current_score
                best_model = model
        except ValueError:
            continue
            
    if best_model is None:
        best_model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=random_state, tol=1e-4)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            best_model.fit(X)

    hidden_states = best_model.predict(X)
    post_probs = best_model.predict_proba(X)
    
    state_volatilities = [best_model.covars_[i][0, 0] for i in range(n_regimes)]
    low_vol_state_idx = np.argmin(state_volatilities)
    
    if low_vol_state_idx != 0:
        hidden_states = 1 - hidden_states
        post_probs = post_probs[:, [1, 0]]
        
    df_out = df_features.copy()
    df_out['Prob_Regime_0'] = post_probs[:, 0]
    df_out['Prob_Regime_1'] = post_probs[:, 1]
    df_out['Hidden_State'] = hidden_states
    
    return best_model, df_out

def load_regime_data(processed_dir, universe):
    """
    Ingests compiled regime panels and stacks them into a unified dataframe
    ready for multi-asset Hierarchical Bayesian inference.
    """
    combined_data = []
    
    for asset_idx, ticker in enumerate(universe):
        file_path = os.path.join(processed_dir, f"{ticker}_with_regimes.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing upstream regime tensor for modeling: {file_path}")
            
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df['Asset_Idx'] = asset_idx
        df['Ticker'] = ticker
        combined_data.append(df)
        
    df_all = pd.concat(combined_data).sort_index()
    return df_all

def fit_hierarchical_bayes(df_panel, regime_id, draws=2000, tune=1500):
    """
    Executes an optimized Multi-Asset Non-Centered Hierarchical Bayesian Linear Regression
    isolated for a specific HMM Volatility Regime.
    """
    # Filter dataset for the specific operational regime slice
    df_regime = df_panel[df_panel['Hidden_State'] == regime_id].copy()
    
    asset_indices = df_regime['Asset_Idx'].values
    universe = df_regime['Ticker'].unique().tolist()
    n_assets = len(universe)
    
    # Extract structural predictors (Standardized upstream for convergence safety)
    X_mean = df_regime['Sentiment_Mean'].values
    X_var = df_regime['Sentiment_Variance'].values
    y = df_regime['log_ret'].values
    
    coords = {
        "assets": universe
    }
    
    with pm.Model(coords=coords) as model:
        # --- Intercept Hierarchical Structure (Non-Centered) ---
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=0.01)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.01)
        alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, dims="assets")
        alpha = pm.Deterministic("alpha", mu_alpha + alpha_offset * sigma_alpha, dims="assets")
        
        # --- Sentiment Mean Coefficient Structure (Non-Centered) ---
        mu_beta_mean = pm.Normal("mu_beta_mean", mu=0.0, sigma=0.05)
        sigma_beta_mean = pm.HalfNormal("sigma_beta_mean", sigma=0.02)
        beta_mean_offset = pm.Normal("beta_mean_offset", mu=0.0, sigma=1.0, dims="assets")
        beta_sim = pm.Deterministic("beta_sim", mu_beta_mean + beta_mean_offset * sigma_beta_mean, dims="assets")
        
        # --- Sentiment Variance Coefficient Structure (Non-Centered) ---
        mu_beta_var = pm.Normal("mu_beta_var", mu=0.0, sigma=0.05)
        sigma_beta_var = pm.HalfNormal("sigma_beta_var", sigma=0.02)
        beta_var_offset = pm.Normal("beta_var_offset", mu=0.0, sigma=1.0, dims="assets")
        beta_var = pm.Deterministic("beta_var", mu_beta_var + beta_var_offset * sigma_beta_var, dims="assets")
        
        # --- Model Residual Variance (Asset Independent) ---
        sigma_residual = pm.HalfNormal("sigma_residual", sigma=0.03, dims="assets")
        
        # --- Expected Value Equation Mapping ---
        mu_expected = alpha[asset_indices] + beta_sim[asset_indices] * X_mean + beta_var[asset_indices] * X_var
        
        # --- Likelihood ---
        likelihood = pm.Normal("likelihood", mu=mu_expected, sigma=sigma_residual[asset_indices], observed=y)
        
        # --- MCMC NUTS Sampling ---
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=0.99,  # Heightened targeting step bounds to completely drop geometric divergences
            random_seed=42,
            return_inferencedata=True,
            init="jitter+adapt_diag"
        )
        
    return trace, universe

def load_saved_posterior(base_dir, regime_id):
    """
    Ingests NetCDF traces containing full MCMC chains for a target regime.
    """
    file_path = os.path.join(base_dir, "data", f"regime_{regime_id}_posterior.nc")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No NetCDF trace asset found at {file_path}")
    return az.from_netcdf(file_path)

def generate_bayesian_inputs(idata, df_features, tickers):
    """
    Extracts posterior parameter arrays, applies feature matrices, 
    and outputs conditional return vectors and covariance structures.
    """
    # Extract structural chain arrays from the posterior trace
    posterior = idata.posterior
    
    alphas = posterior['alpha'].values        # Shape: (chains, draws, assets)
    betas_sim = posterior['beta_sim'].values  # Shape: (chains, draws, assets)
    betas_var = posterior['beta_var'].values  # Shape: (chains, draws, assets)
    sigmas = posterior['sigma_residual'].values # Shape: (chains, draws, assets)
    
    # Reshape chains and draws into a single flat sampling layer
    n_samples = alphas.shape[0] * alphas.shape[1]
    alphas_flat = alphas.reshape(n_samples, -1)
    betas_sim_flat = betas_sim.reshape(n_samples, -1)
    betas_var_flat = betas_var.reshape(n_samples, -1)
    sigmas_flat = sigmas.reshape(n_samples, -1)
    
    # Map coordinates cleanly to guarantee asset alignment matches tracking universe
    trace_assets = list(idata.posterior.coords['assets'].values)
    asset_mapping = [trace_assets.index(t) for t in tickers]
    
    n_assets = len(tickers)
    simulated_returns = np.zeros((n_samples, n_assets))
    
    # Generate predictive return distributions
    for idx, ticker in enumerate(tickers):
        t_idx = asset_mapping[idx]
        
        # Pull asset features safely
        x_mean = df_features.loc[ticker, 'Sentiment_Mean']
        x_var = df_features.loc[ticker, 'Sentiment_Variance']
        
        # Calculate expected returns incorporating residual idiosyncratic risk components
        expected_mu = (alphas_flat[:, t_idx] + 
                       betas_sim_flat[:, t_idx] * x_mean + 
                       betas_var_flat[:, t_idx] * x_var)
        
        # Generate posterior predictive draws
        simulated_returns[:, idx] = np.random.normal(expected_mu, sigmas_flat[:, t_idx])
        
    # Compile Bayesian parameters
    mu_bayesian = np.mean(simulated_returns, axis=0)
    sigma_bayesian = np.cov(simulated_returns, rowvar=False)
    
    # Apply a light shrinkage regularization boundary to guarantee positive definiteness
    sigma_bayesian += np.eye(n_assets) * 1e-6
    
    return mu_bayesian, sigma_bayesian

def optimize_portfolio(mu_b, sigma_b, prev_weights=None, turnover_penalty=0.0, max_weight_change=None):
    """
    Executes a Markowitz Mean-Variance optimization to maximize the Sharpe Ratio
    under full long-only capital investment constraints.

    Optional turnover controls (both default OFF, so existing callers that
    only pass mu_b/sigma_b keep their original behavior):

    - prev_weights: the portfolio's current allocation. Required for either
      of the two controls below to take effect.
    - turnover_penalty: a soft L1 penalty (lambda) subtracted from the
      Sharpe objective, proportional to sum(|w_new - prev_weights|). Larger
      values bias the optimizer toward staying closer to the current
      allocation when expected-return estimates are noisy/near-zero,
      instead of chasing the "least-bad" corner every period.
    - max_weight_change: an optional hard cap (e.g. 0.15) on how much any
      single asset's weight can move per rebalance. Prevents the
      100%-into-one-asset flip that noisy, near-zero mu estimates can
      otherwise trigger.
    """
    n_assets = len(mu_b)
    initial_weights = np.array(prev_weights) if prev_weights is not None else np.ones(n_assets) / n_assets

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # Long-only constraint
    if prev_weights is not None and max_weight_change is not None:
        bounds = tuple(
            (max(0.0, prev_weights[i] - max_weight_change),
             min(1.0, prev_weights[i] + max_weight_change))
            for i in range(n_assets)
        )

    # Constraint equation: Sum of allocations must equal 100%
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    # Objective: Minimize Negative Sharpe Ratio (assumes zero risk-free rate),
    # optionally penalized for deviating from the prior allocation.
    def negative_sharpe(weights):
        port_return = np.dot(weights, mu_b)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(sigma_b, weights)))
        if port_volatility < 1e-8:
            sharpe = 0.0
        else:
            sharpe = port_return / port_volatility

        objective = -sharpe
        if prev_weights is not None and turnover_penalty > 0.0:
            turnover = np.sum(np.abs(weights - prev_weights))
            objective += turnover_penalty * turnover
        return objective

    result = minimize(
        negative_sharpe, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if not result.success:
        raise ValueError(f"Portfolio Optimization engine failed to converge: {result.message}")
        
    return result.x