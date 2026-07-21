## Assessing the Viability of Swing-Trade Strategies Using Bayesian Aggregation of Behavioural Crowd Forecasts With Continuous Fundamental Regimes


Swing-trading strategy that combines **Hidden Markov Model regime detection**,
**hierarchical Bayesian return modeling on crowd sentiment**, and
**mean-variance portfolio optimization**, evaluated with a walk-forward
backtest.

The strategy's core idea: social-media sentiment (Stocktwits) may carry
predictive signal about near-term returns, but the strength of that
relationship likely depends on the prevailing market regime. Rather than fit
one pooled model across all conditions, this project fits **separate
Bayesian return models per volatility regime**, then uses whichever model is
active at each rebalance date to drive portfolio construction.

> **Status:** research / proof-of-concept pipeline. See [Limitations](#limitations) before drawing any performance conclusions.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Methodology Notes](#methodology-notes)
- [Limitations](#limitations)
- [License](#license)

---

## How It Works

```
data_engine.py  ──▶  regime_pipeline.py  ──▶  regime_models.py  ──▶  portfolio_engine.py  ──▶  backtester.py
   (ingest raw         (feature engineering       (hierarchical Bayes       (single-period          (walk-forward
    sentiment data)      + HMM regime fitting)      posterior sampling)       allocation demo)         backtest + report)
```

| Stage | Script | What it does | Reads | Writes |
|---|---|---|---|---|
| 1 | `data_engine.py` | Streams raw Stocktwits sentiment records from a public S3 bucket for the target universe/date range | S3 (`stocktwits-nyu` public bucket) | `data/processed/{TICKER}.csv` |
| 2 | `regime_pipeline.py` | Builds lagged sentiment features + price features, fits a per-ticker Gaussian HMM to classify Low-Vol / High-Vol regimes | `data/processed/{TICKER}.csv`, Yahoo Finance | `data/processed/{TICKER}_with_regimes.csv` |
| 3 | `regime_models.py` | Fits a hierarchical Bayesian regression (PyMC/NUTS) per regime, pooling statistical strength across tickers | `data/processed/{TICKER}_with_regimes.csv` (all tickers) | `data/regime_0_posterior.nc`, `data/regime_1_posterior.nc` |
| 4 | `portfolio_engine.py` | Demonstrates a single-period optimal allocation for a given regime + feature snapshot; plots the efficient frontier | `data/regime_{n}_posterior.nc` | `reports/regime_{n}_optimization.png` |
| 5 | `backtester.py` | Runs a weekly walk-forward simulation: detect active regime → generate Bayesian (μ, Σ) → optimize weights → apply turnover cost → realize forward return | `data/processed/{TICKER}_with_regimes.csv`, posteriors | Console performance report + `reports/backtest_results_dashboard.png` |

Shared logic (feature engineering, HMM fitting, the PyMC model definition,
and the Markowitz optimizer) lives in `utils.py` and is imported by the
later stages.

---

## Project Structure

```
.
├── src/
│   ├── data_engine.py        # Stage 1 — S3 sentiment ingestion
│   ├── regime_pipeline.py    # Stage 2 — feature engineering + HMM regime detection
│   ├── regime_models.py      # Stage 3 — hierarchical Bayesian model fitting
│   ├── portfolio_engine.py   # Stage 4 — single-period optimization + efficient frontier plot
│   ├── backtester.py         # Stage 5 — walk-forward backtest + results dashboard
│   └── utils.py              # Shared feature engineering, HMM, Bayesian model, optimizer functions
├── data/
│   ├── processed/            # Per-ticker CSVs at each pipeline stage
│   ├── regime_0_posterior.nc # Serialized MCMC trace — Low-Volatility regime
│   └── regime_1_posterior.nc # Serialized MCMC trace — High-Volatility regime
├── reports/
│   ├── regime_{n}_optimization.png
│   └── backtest_results_dashboard.png
└── README.md
```

`data/` and `reports/` are created automatically on first run — you don't
need to create them by hand.

---

## Requirements

- Python 3.10+
- A C++ compiler toolchain (for PyMC's `pytensor` backend) — on Windows,
  installing `g++` via `conda install m2w64-toolchain` (or `conda install
  gxx` in a conda environment) avoids the `g++ not available` warning at
  startup. The pipeline still runs without it, just with slower/uncompiled
  sampling.

### Python packages

```
pandas
numpy
scipy
s3fs
pymc
arviz
hmmlearn
scikit-learn
yfinance
matplotlib
seaborn
```

A minimal `requirements.txt`:

```text
pandas
numpy
scipy
s3fs
pymc
arviz
hmmlearn
scikit-learn
yfinance
matplotlib
seaborn
```

---

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd <your-repo-name>

# Create and activate a virtual environment
python -m venv env
source env/bin/activate        # macOS/Linux
env\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run the pipeline stages **in order** — each stage expects the previous
stage's output files to already exist. Every script raises a clear
`FileNotFoundError` (rather than failing silently) if a prerequisite hasn't
been run yet.

```bash
# 1. Ingest sentiment data for the target universe
python src/data_engine.py

# 2. Engineer features and fit per-ticker HMM regimes
python src/regime_pipeline.py

# 3. Fit hierarchical Bayesian return models (one per regime)
python src/regime_models.py

# 4. (Optional) Inspect a single-period optimal allocation + efficient frontier
python src/portfolio_engine.py

# 5. Run the full walk-forward backtest
python src/backtester.py
```

Step 5 prints a performance summary to the console:

```
==================================================
SWING-TRADE STRATEGY PERFORMANCE REPORT (2014-2015)
==================================================
Total Cumulative Return: X.XX%
Annualized Sharpe Ratio: X.XXXX
Average Weekly Turnover: X.XX%
==================================================
```

and saves a diagnostic dashboard to `reports/backtest_results_dashboard.png`.

---

## Configuration

Key parameters are defined as constants near the top of each script — edit
these directly to customize a run:

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `TARGET_UNIVERSE` / `VALIDATED_UNIVERSE` / `tickers` | `data_engine.py`, `regime_pipeline.py`, `backtester.py` | `["AAPL", "AMD", "SPY", "TSLA"]` | Asset universe (must match across all three) |
| `START_HORIZON` / `START_DATE` | `data_engine.py`, `regime_pipeline.py` | `2014-01-01` | Start of data pull / feature window |
| `END_HORIZON` / `END_DATE` | `data_engine.py`, `regime_pipeline.py` | `2022-03-01` / `2015-12-31` | End of data pull / feature window — **note:** these two currently differ; align them for a full-horizon run |
| `n_regimes` | `regime_pipeline.py` (`fit_market_hmm` call) | `2` | Number of HMM hidden states |
| Rebalance frequency | `backtester.py` (`pd.date_range(..., freq='W-FRI')`) | Weekly, Fridays | Backtest rebalance cadence |
| Warm-up window | `backtester.py` | 26 weeks | Burn-in period before the first rebalance |
| Transaction cost | `backtester.py` | 10 bps × turnover | Per-rebalance cost assumption |
| MCMC sampling config | `utils.fit_hierarchical_bayes` | 4 chains / 2,000 draws / 1,500 tuning | PyMC/NUTS sampler settings |

---

## Outputs

- **Console report** — cumulative return, annualized Sharpe ratio, average
  weekly turnover, plus skip-reason diagnostics if any rebalance periods
  were dropped (missing regime labels, missing forward returns, etc.).
- **`reports/regime_{n}_optimization.png`** — efficient frontier scatter
  (simulated random portfolios + individual assets + optimal portfolio) and
  a bar chart of optimal weights, for a single regime/feature snapshot.
- **`reports/backtest_results_dashboard.png`** — four-panel summary of the
  full walk-forward run:
  1. Cumulative net return, shaded by active regime
  2. Portfolio drawdown
  3. Allocation weights over time (stacked area, by ticker)
  4. Turnover per rebalance

---

## Methodology Notes

- **Anti-leakage design:** sentiment features are lagged before merging
  with price data, and each rebalance evaluates portfolio weights against
  the **next** period's realized return, not the current one.
- **Regime canonicalization:** HMM states are relabeled so Regime 0 is
  always the lower-variance ("Low-Volatility") state and Regime 1 the
  higher-variance ("High-Volatility/Stressed") state, making labels
  comparable across tickers.
- **Hierarchical pooling:** the Bayesian model shares population-level
  priors across all four tickers within a regime, so assets with sparser
  sentiment data borrow statistical strength from the others rather than
  being fit in total isolation.

---

## Limitations

- **In-sample regime labeling.** The backtest currently uses the *fitted*
  (smoothed) HMM state for each date rather than a causal, real-time
  filtered regime estimate. A live/forward-testing deployment would need
  an online filtering approach to avoid look-ahead in the regime label
  itself.
- **Static posteriors.** Bayesian models are fit once per regime
  (`regime_models.py`) rather than re-estimated on a rolling basis, so
  the backtest is out-of-sample on realized returns but not on model
  parameters.
- **Simplified transaction costs.** A flat 10 bps per unit of turnover;
  no bid-ask spread or market-impact modeling.
- **Small universe.** Four assets, chosen to validate the pipeline
  end-to-end — not intended as a diversified, production-ready portfolio.
- **Date range mismatch.** `data_engine.py` defaults to a 2014–2022 pull
  while `regime_pipeline.py` defaults to 2014–2015 — align `START_DATE`/
  `END_DATE` across scripts before running a full-horizon backtest.

---
