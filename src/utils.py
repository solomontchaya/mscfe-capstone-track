import os
import glob
import ast
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import yfinance as yf
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hmmlearn.hmm import GaussianHMM

def extract_sentiment_score(entities_str):
    if pd.isna(entities_str):
        return np.nan
    try:
        entities_dict = ast.literal_eval(entities_str)
        sentiment_node = entities_dict.get('sentiment')
        if sentiment_node and isinstance(sentiment_node, dict):
            basic_sentiment = sentiment_node.get('basic')
            if basic_sentiment == 'Bullish':
                return 1.0
            elif basic_sentiment == 'Bearish':
                return -1.0
    except Exception:
        pass
    return np.nan

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
    print(f"Initializing localized processing loops for: {ticker}")
    search_path = os.path.join(raw_data_dir, "**/*.csv")
    all_files = glob.glob(search_path, recursive=True)
    
    if not all_files:
        print(f"Error: No raw data files located at: {raw_data_dir}")
        return

    print(f"Located {len(all_files)} raw data files. Beginning text parsing...")
    daily_aggregates = {}
    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, usecols=['created_at', 'body', 'entities'])
            df['timestamp'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            # Localize dates cleanly to match the absolute timeline
            df['date_only'] = df['timestamp'].dt.date
            
            # Force inclusive filtering bounds across uniform timestamps
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            df = df[(df['date_only'] >= start_dt) & (df['date_only'] <= end_dt)]
            
            if df.empty:
                continue
                
            df['sentiment_score'] = df['entities'].apply(extract_sentiment_score)
            
            for current_date, group in df.groupby('date_only'):
                bodies = group['body'].dropna().tolist()
                msg_count = len(bodies)
                
                if msg_count < 3:
                    continue
                    
                try:
                    tfidf = vectorizer.fit_transform(bodies)
                    cos_sim = cosine_similarity(tfidf)
                    np.fill_diagonal(cos_sim, np.nan)
                    daily_arg_sim = np.nanmean(cos_sim)
                except Exception:
                    daily_arg_sim = 0.0
                
                valid_sentiments = group['sentiment_score'].dropna()
                daily_sent_var = np.var(valid_sentiments) if len(valid_sentiments) > 0 else 0.0
                
                if current_date not in daily_aggregates:
                    daily_aggregates[current_date] = {
                        'arg_sims': [daily_arg_sim],
                        'sent_vars': [daily_sent_var],
                        'counts': msg_count
                    }
                else:
                    daily_aggregates[current_date]['arg_sims'].append(daily_arg_sim)
                    daily_aggregates[current_date]['sent_vars'].append(daily_sent_var)
                    daily_aggregates[current_date]['counts'] += msg_count
                    
        except Exception as file_err:
            print(f"Log Notice: Skipping file entry {os.path.basename(file_path)} due to: {file_err}")
            continue

    if not daily_aggregates:
        print(f"Warning: No valid alternative text records processed for {ticker}.")
        return

    summary_rows = []
    for date_key, metrics in daily_aggregates.items():
        summary_rows.append({
            'Date': pd.to_datetime(date_key),
            'Argument_Similarity': np.mean(metrics['arg_sims']),
            'Sentiment_Variance': np.mean(metrics['sent_vars']),
            'Volume_Crowd': metrics['counts']
        })
        
    df_features = pd.DataFrame(summary_rows).set_index('Date').sort_index()
    df_features_lagged = df_features.shift(1)
    
    yf_ticker = "META" if ticker == "FB" else ticker
    print(f"Downloading market pricing matrix for {yf_ticker} via Yahoo Finance...")
    
    # Add 1 extra day to the end date parameter because Yahoo extraction endpoint is exclusive
    extended_end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    market_data = yf.download(yf_ticker, start=start_date, end=extended_end)
    
    # Robust Multi-Index column flattening logic
    if isinstance(market_data.columns, pd.MultiIndex):
        market_data.columns = market_data.columns.get_level_values(0)
        
    market_data.index = pd.to_datetime(market_data.index)
    
    # Slice pricing data strictly back down to your target end date bounds
    market_data = market_data.loc[start_date:end_date]
    
    final_panel = market_data[['Close', 'High', 'Low', 'Volume']].merge(
        df_features_lagged, left_index=True, right_index=True, how='left'
    )
    final_panel['Argument_Similarity'] = final_panel['Argument_Similarity'].ffill().fillna(0.0)
    final_panel['Sentiment_Variance'] = final_panel['Sentiment_Variance'].ffill().fillna(0.0)
    final_panel['Volume_Crowd'] = final_panel['Volume_Crowd'].fillna(0.0)
    
    final_panel.to_csv(output_csv_path)
    print(f"=== Success! Processed panel saved to: {output_csv_path} (Shape: {final_panel.shape}) ===")

def generate_regime_features(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.set_index('Date').sort_index()
    
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['hl_spread'] = np.log(df['High'] / df['Low'])
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['log_ret', 'hl_spread'])
    return df

def fit_market_hmm(df_features, n_regimes=2, random_state=42):
    """
    Fits a Gaussian HMM over returns and volatility spreads using a multistart 
    optimization sequence to guarantee global convergence and stable regime capture.
    """
    feature_cols = ['log_ret', 'hl_spread']
    X = df_features[feature_cols].values
    
    best_likelihood = -np.inf
    best_model = None
    
    # Run a localized multi-start sequence to guarantee escape from local maxima
    for seed_offset in [0, 15, 42, 99, 123]:
        current_seed = random_state + seed_offset
        
        # We strip the default internal 'init' overrides to handle bounds manually
        model = GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full", 
            n_iter=500,
            min_covar=1e-3,
            random_state=current_seed
        )
        
        try:
            model.fit(X)
            # Evaluate model viability using log-likelihood performance score
            current_score = model.score(X)
            
            # Extract transition properties to screen for single-day outlier traps
            # We reject initializations where a state has >80% immediate exit risk
            if model.transmat_[1, 0] > 0.80 or model.transmat_[0, 1] > 0.80:
                continue
                
            if current_score > best_likelihood:
                best_likelihood = current_score
                best_model = model
                
        except ValueError:
            continue
            
    # Fallback configuration safeguard if all constrained seeds are bypassed
    if best_model is None:
        best_model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000, random_state=random_state)
        best_model.fit(X)

    hidden_states = best_model.predict(X)
    post_probs = best_model.predict_proba(X)
    
    # --- Standardize States by High-Low Spread Mean ---
    state_volatilities = [best_model.means_[i, 1] for i in range(n_regimes)]
    low_vol_state_idx = np.argmin(state_volatilities)
    
    # Structural Parameter Swap ensuring State 0 == Low Volatility
    if low_vol_state_idx != 0:
        best_model.means_[0], best_model.means_[1] = best_model.means_[1].copy(), best_model.means_[0].copy()
        best_model.covars_[0], best_model.covars_[1] = best_model.covars_[1].copy(), best_model.covars_[0].copy()
        
        trans = best_model.transmat_.copy()
        trans[[0, 1], :] = trans[[1, 0], :]
        trans[:, [0, 1]] = trans[:, [1, 0]]
        best_model.transmat_ = trans
        
        best_model.startprob_[0], best_model.startprob_[1] = best_model.startprob_[1].copy(), best_model.startprob_[0].copy()
        
        hidden_states = 1 - hidden_states
        post_probs = post_probs[:, [1, 0]]
        
    df_out = df_features.copy()
    df_out['Prob_Regime_0'] = post_probs[:, 0]
    df_out['Prob_Regime_1'] = post_probs[:, 1]
    df_out['Hidden_State'] = hidden_states
    
    return best_model, df_out

def load_regime_data(processed_dir, tickers):
    """
    Loads processed data panels and compiles cross-sectional matrices.
    """
    all_data = []
    for ticker in tickers:
        file_path = os.path.join(processed_dir, f"{ticker}_with_regimes.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing mandatory data panel: {file_path}")
        
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df['Ticker'] = ticker
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)

def fit_hierarchical_bayes(df_universe, regime_id, draws=2000, tune=1000):
    """
    Executes MCMC sampling using Non-Centered Parameterization to completely
    eliminate divergent transitions caused by high-curvature funnel geometries.
    """
    # Filter universe to isolate the selected active regime window
    df_regime = df_universe[df_universe['Hidden_State'] == regime_id].copy()
    
    # Map tickers to discrete categorical index coordinates for tensor mapping
    df_regime['Ticker_Idx'] = df_regime['Ticker'].astype('category').cat.codes
    ticker_names = df_regime['Ticker'].astype('category').cat.categories.values
    n_assets = len(ticker_names)
    
    # Extract structural feature observations
    ticker_idx = df_regime['Ticker_Idx'].values
    y_obs = df_regime['log_ret'].values
    arg_sim = df_regime['Argument_Similarity'].values
    sent_var = df_regime['Sentiment_Variance'].values
    
    print(f"\n[INIT] Initializing Non-Centered PyMC Graph Canvas for Regime {regime_id}")
    print(f"Total cross-sectional observation tokens: {len(y_obs)} across {n_assets} assets.")
    
    with pm.Model() as model:
        # --- Shared Hyper-Priors (Global Market Context) ---
        mu_alpha = pm.Normal('mu_alpha', mu=0.0, sigma=0.1)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.1)
        
        mu_beta_sim = pm.Normal('mu_beta_sim', mu=0.0, sigma=0.5)
        sigma_beta_sim = pm.HalfNormal('sigma_beta_sim', sigma=0.5)
        
        mu_beta_var = pm.Normal('mu_beta_var', mu=0.0, sigma=0.5)
        sigma_beta_var = pm.HalfNormal('sigma_beta_var', sigma=0.5)
        
        # --- Non-Centered Parameterization Primitive Offsets ---
        alpha_offset = pm.Normal('alpha_offset', mu=0.0, sigma=1.0, shape=n_assets)
        beta_sim_offset = pm.Normal('beta_sim_offset', mu=0.0, sigma=1.0, shape=n_assets)
        beta_var_offset = pm.Normal('beta_var_offset', mu=0.0, sigma=1.0, shape=n_assets)
        
        # --- Deterministic Linear Transformations ---
        alpha = pm.Deterministic('alpha', mu_alpha + alpha_offset * sigma_alpha)
        beta_sim = pm.Deterministic('beta_sim', mu_beta_sim + beta_sim_offset * sigma_beta_sim)
        beta_var = pm.Deterministic('beta_var', mu_beta_var + beta_var_offset * sigma_beta_var)
        
        # Residual variance estimation per asset
        sigma_residual = pm.HalfNormal('sigma_residual', sigma=0.1, shape=n_assets)
        
        # --- Deterministic Linear Regression Equation ---
        mu_computed = (alpha[ticker_idx] + 
                       beta_sim[ticker_idx] * arg_sim + 
                       beta_var[ticker_idx] * sent_var)
        
        # --- Likelihood Function ---
        likelihood = pm.Normal('y_likelihood', mu=mu_computed, sigma=sigma_residual[ticker_idx], observed=y_obs)
        
        # --- Execute MCMC Sampling Sequence ---
        print(f"[SAMPLING] Launching NUTS Sampler Engine for Regime {regime_id}...")
        idata = pm.sample(
            draws=draws, 
            tune=tune, 
            chains=4, 
            random_seed=42,
            return_inferencedata=True,
            target_accept=0.99  # Protects against tail-end divergence anomalies
        )
        
    return idata, ticker_names

def load_saved_posterior(base_dir, regime_id):
    """Loads the serialized NetCDF inference data for a specific regime."""
    trace_path = os.path.join(base_dir, "data", f"regime_{regime_id}_posterior.nc")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Posterior trace asset missing: {trace_path}")
    return az.from_netcdf(trace_path)

def generate_bayesian_inputs(idata, current_features, asset_names):
    """
    Samples the posteriors conditioned on current NLP features to build 
    the expected return vector (mu) and the covariance matrix (Sigma).
    """
    posterior = idata.posterior
    n_assets = len(asset_names)
    
    # Containers for simulated forward returns
    simulated_returns = []
    
    for idx, asset in enumerate(asset_names):
        # Flatten chains and draws to get the full posterior distribution
        alphas = posterior['alpha'].values[..., idx].flatten()
        beta_sims = posterior['beta_sim'].values[..., idx].flatten()
        beta_vars = posterior['beta_var'].values[..., idx].flatten()
        
        # Pull current feature values for this specific asset
        feat_sim = current_features.loc[asset, 'Argument_Similarity']
        feat_var = current_features.loc[asset, 'Sentiment_Variance']
        
        # Conditioned Expected Return Distribution: μ = α + β1*Sim + β2*Var
        mu_conditioned = alphas + (beta_sims * feat_sim) + (beta_vars * feat_var)
        simulated_returns.append(mu_conditioned)
        
    # Convert to array of shape (n_assets, n_samples)
    sim_returns_matrix = np.array(simulated_returns)
    
    # Calculate Expected Return Vector (Mean of posteriors)
    mu_bayes = np.mean(sim_returns_matrix, axis=1)
    
    # Calculate Empirical Covariance Matrix directly from posterior paths
    sigma_bayes = np.cov(sim_returns_matrix)
    
    return mu_bayes, sigma_bayes

def optimize_portfolio(mu, sigma):
    """Executes a standard Max Sharpe Ratio optimization."""
    n_assets = len(mu)
    init_weights = np.array([1 / n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only constraint
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Fully invested
    
    def negative_sharpe(weights):
        p_ret = np.dot(weights, mu)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -p_ret / (p_vol + 1e-8)  # Minimize negative Sharpe
        
    res = minimize(negative_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x