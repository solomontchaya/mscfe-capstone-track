## Assessing the Viability of Swing-Trade Strategies Using Bayesian Aggregation of Behavioural Crowd Forecasts With Continuous Fundamental Regimes

Swing-trading strategy that combines **skilled-minority sentiment identification with ANOVA-style extremizing**, **Hidden Markov Model regime detection**, **hierarchical Bayesian return modeling on crowd sentiment**, and **mean-variance portfolio optimization**, evaluated with a walk-forward backtest split into **validation** and **test** segments, plus a directional-skill and statistical-significance diagnostic on top of the standard return metrics.

The strategy's core idea: social-media sentiment (Stocktwits) may carry
predictive signal about near-term returns, but the strength of that
relationship likely depends on (a) *who* is posting — most crowd sentiment is
noise, and only a "skilled minority" of forecasters have any real call-timing
edge — and (b) the prevailing market regime. Rather than fit one pooled model
across all conditions on the raw, unweighted sentiment, this project first
isolates and extremizes the skilled minority's signal, then fits **separate
Bayesian return models per volatility regime** on top of it, using whichever
model is active at each rebalance date to drive portfolio construction.

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
data_engine.py  ──▶  skill_weighting.py  ──▶  regime_pipeline.py  ──▶  regime_models.py  ──▶  portfolio_engine.py  ──▶  backtester.py
   (ingest raw         (Stage 1.5 --            (feature engineering       (hierarchical Bayes       (single-period          (walk-forward
    sentiment data)     skilled-minority          + HMM regime fitting)      posterior sampling)       allocation demo)         backtest + report,
                        ID + ANOVA-style                                                                                          split into
                        extremizing)                                                                                              validation/test)
```

| Stage | Script                | What it does                                                                                                                                             | Reads                                                    | Writes                                                                |
| ----- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------- |
| 1     | `data_engine.py`      | Streams raw Stocktwits sentiment records from a public S3 bucket for the target universe/date range                                                      | S3 (`stocktwits-nyu` public bucket)                      | `data/processed/{TICKER}.csv`                                         |
| 1.5   | `skill_weighting.py`  | Runs a causal (no-look-ahead) one-sided sign test per Stocktwits user against that ticker's own running base rate, keeps only the FDR-corrected "skilled minority," pools their calls into a daily bullish probability, and extremizes it via a per-day ANOVA-derived logit transform | `data/processed/{TICKER}.csv`, Yahoo Finance (forward returns) | `data/processed/{TICKER}_skill_extremized.csv`                        |
| 2     | `regime_pipeline.py`  | Builds lagged sentiment features + price features, fits a per-ticker Gaussian HMM to classify Low-Vol / High-Vol regimes                                 | `data/processed/{TICKER}.csv`, Yahoo Finance             | `data/processed/{TICKER}_with_regimes.csv`                            |
| 3     | `regime_models.py`    | Fits a hierarchical Bayesian regression (PyMC/NUTS) per regime, pooling statistical strength across tickers                                              | `data/processed/{TICKER}_with_regimes.csv` (all tickers) | `data/regime_0_posterior.nc`, `data/regime_1_posterior.nc`            |
| 4     | `portfolio_engine.py` | Demonstrates a single-period optimal allocation for a given regime + feature snapshot; plots the efficient frontier                                      | `data/regime_{n}_posterior.nc`                           | `reports/regime_{n}_optimization.png`                                 |
| 5     | `backtester.py`       | Runs a weekly walk-forward simulation over a **validation** period and a subsequent **test** period: detect active regime → generate Bayesian (μ, Σ) → optimize weights → apply turnover cost → realize forward return. Reports return/Sharpe metrics, a directional-skill diagnostic, and a statistical-significance test for each segment. | `data/processed/{TICKER}_with_regimes.csv`, posteriors   | Console performance report + `reports/backtest_results_dashboard_validation.png`, `reports/backtest_results_dashboard_test.png` |

Shared logic (feature engineering, HMM fitting, the PyMC model definition,
and the Markowitz optimizer) lives in `utils.py` and is imported by the
later stages.

> **Not yet wired together:** `skill_weighting.py` runs standalone today — it
> reads `data_engine.py`'s output and writes `{TICKER}_skill_extremized.csv`,
> but `regime_pipeline.py` does not yet consume that file. See
> [Limitations](#limitations).

---

## Project Structure

```
.
├── src/
│   ├── data_engine.py         # Stage 1   — S3 sentiment ingestion
│   ├── skill_weighting.py     # Stage 1.5 — skilled-minority ID + ANOVA-style extremizing
│   ├── regime_pipeline.py     # Stage 2   — feature engineering + HMM regime detection
│   ├── regime_models.py       # Stage 3   — hierarchical Bayesian model fitting
│   ├── portfolio_engine.py    # Stage 4   — single-period optimization + efficient frontier plot
│   ├── backtester.py          # Stage 5   — validation/test walk-forward backtest + results dashboards
│   └── utils.py               # Shared feature engineering, HMM, Bayesian model, optimizer functions
├── data/
│   ├── processed/             # Per-ticker CSVs at each pipeline stage
│   │   └── {TICKER}_skill_extremized.csv  # Stage 1.5 daily skill-weighted/extremized signal
│   ├── regime_0_posterior.nc  # Serialized MCMC trace — Low-Volatility regime
│   └── regime_1_posterior.nc  # Serialized MCMC trace — High-Volatility regime
├── reports/
│   ├── regime_{n}_optimization.png
│   ├── backtest_results_dashboard_validation.png
│   └── backtest_results_dashboard_test.png
├── README.md
└── requirements.txt
```

`data/` and `reports/` are created automatically on first run — you don't
need to create them by hand.

---

## Requirements

- Python 3.10+
- A C++ compiler toolchain (for PyMC's `pytensor` backend) — on Windows,
installing `g++` via `conda install m2w64-toolchain` (or `conda install gxx` in a conda environment) avoids the `g++ not available` warning at
startup. The pipeline still runs without it, just with slower/uncompiled
sampling.
- **`user_id` must be retained by `data_engine.py`'s S3 ingestion.**
  `skill_weighting.py` needs a `user_id` column in
  `data/processed/{TICKER}.csv` to build a per-forecaster track record, and
  fails loudly (`KeyError`) if it's missing. See
  [Limitations](#limitations).

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

---

## Installation

```
# Clone the repo
git clone https://github.com/solomontchaya/mscfe-capstone-track.git
cd mscfe-capstone-track

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

```
# 1. Ingest sentiment data for the target universe
python src/data_engine.py

# 1.5. Identify the skilled-minority of forecasters and extremize their
#      pooled daily signal (requires user_id to already be retained in
#      Stage 1's output — see Requirements)
python src/skill_weighting.py

# 2. Engineer features and fit per-ticker HMM regimes
python src/regime_pipeline.py

# 3. Fit hierarchical Bayesian return models (one per regime)
python src/regime_models.py

# 4. (Optional) Inspect a single-period optimal allocation + efficient frontier
python src/portfolio_engine.py

# 5. Run the full walk-forward backtest (validation, then test)
python src/backtester.py
```

`skill_weighting.py` also supports two standalone modes, useful before
wiring it into the rest of the pipeline:

```
# Run a synthetic self-test with no network/S3 access required — simulates
# a handful of genuinely skilled users among a crowd of noise traders and
# confirms they're the ones flagged
python src/skill_weighting.py --demo

# Use a rolling base rate (e.g. the trailing 60 trading days) instead of the
# default full-history expanding base rate as the sign test's null, to check
# how much of the "skilled" classification is really just local momentum
python src/skill_weighting.py --rolling-window=60
```

Step 1.5 prints, per ticker, how many users and messages were loaded, a
skill-classification diagnostic (raw vs. FDR-corrected skilled counts, the
overall unconditional hit rate as a leakage sanity check, and directional-
consistency stats), and how many daily extremized signal rows were produced,
before writing `data/processed/{TICKER}_skill_extremized.csv`.

Step 5 first logs the sentiment column being used, ingests the compiled
master dataset, then runs each aligned rebalance period for the validation
segment followed by the test segment, printing a performance report,
directional-skill diagnostic, and statistical-significance test for each:

```
[PRE-FLIGHT] Backtest sentiment_column for this run: Sentiment_Mean
[DATA] Ingesting and compiling per-ticker regime datasets...
[DATA] Successfully loaded master dataset with N rows.

[Validation] Initializing backtest across K aligned periods (START to END, sentiment_column=Sentiment_Mean)...
[Validation] YYYY-MM-DD | Regime: 0/1 | Net Return: X.XXXX
...

==================================================
SWING-TRADE STRATEGY PERFORMANCE REPORT — VALIDATION (START to END) [sentiment_column=Sentiment_Mean]
==================================================
Metric                          Strategy    Equal-Wt Benchmark
Total Cumulative Return            4.45%                 0.19%
Annualized Sharpe Ratio           0.3130                0.0866
Average Weekly Turnover           11.54%                    —
==================================================

DIRECTIONAL SKILL DIAGNOSTIC — VALIDATION
--------------------------------------------------
Observations (assets x periods): 200
Accuracy:                      54.00%
Precision (predicted 'up'):    0.5057
Recall (predicted 'up'):       0.4731
F1 Score:                      0.4889
  vs. majority-class baseline: 0.0000
Confusion Matrix  [[TN=64, FP=43], [FN=49, TP=44]]
--------------------------------------------------

STATISTICAL SIGNIFICANCE — VALIDATION (Strategy vs. Benchmark)
--------------------------------------------------
Observations (weekly periods):  50
Mean weekly excess return:      0.1107%
t-statistic:                    0.3507
p-value (two-sided):            0.7273
95% bootstrap CI on mean:       [-0.5080%, 0.6958%]
Verdict: <statement on whether excess return is distinguishable from zero>
--------------------------------------------------
[VISUAL] Backtest results dashboard exported to: reports/backtest_results_dashboard_validation.png
```

The same block is then repeated for the **[Test]** segment, writing
`reports/backtest_results_dashboard_test.png`, and the run finishes with a
short summary confirming which sentiment column was used and that both
segments completed:

```
==================================================
RUN COMPLETE
==================================================
Sentiment column used: Sentiment_Mean
Validation segment: OK
Test segment:       OK
Dashboards saved to: <reports directory>
```

---

## Configuration

Key parameters are defined as constants near the top of each script — edit
these directly to customize a run:

| Parameter                                            | Location                                                | Default                               | Purpose                                                                                                     |
| ----------------------------------------------------- | ------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `TARGET_UNIVERSE` / `VALIDATED_UNIVERSE` / `tickers` | `data_engine.py`, `skill_weighting.py`, `regime_pipeline.py`, `backtester.py` | `["AAPL", "AMD", "SPY", "TSLA"]`      | Asset universe (must match across all scripts)                                                                |
| `START_HORIZON` / `START_DATE`                       | `data_engine.py`, `skill_weighting.py`, `regime_pipeline.py`                  | `2014-01-01`                          | Start of data pull / feature window                                                                         |
| `END_HORIZON` / `END_DATE`                           | `data_engine.py`, `skill_weighting.py`, `regime_pipeline.py`                  | `2022-03-01` / `2015-12-31`           | End of data pull / feature window — **note:** `regime_pipeline.py`'s default currently differs from the other two; align them for a full-horizon run |
| `FORWARD_HORIZON_DAYS`                               | `skill_weighting.py`                                    | `1` (next trading day)                | Horizon used to score whether a forecaster's directional call was a "hit" — should match whatever horizon the strategy ultimately trades on |
| `min_track_record`                                   | `skill_weighting.build_forecaster_track_record`         | `20`                                   | Minimum number of prior scored calls a user needs before their sign-test p-value is treated as meaningful   |
| `embargo_days`                                       | `skill_weighting.build_forecaster_track_record`         | `1`                                    | Calendar-day buffer between a message and the cutoff for "prior" calls, to avoid forward-return leakage      |
| `base_rate_window`                                   | `skill_weighting.build_forecaster_track_record` (`--rolling-window=N` CLI flag) | `None` (full-history expanding)       | Null hypothesis for the sign test: expanding causal base rate vs. a rolling N-day window                     |
| `k_min` / `k_max`                                    | `skill_weighting.extremize_daily_signal`                | `1.0` / `3.0`                          | Bounds on the per-day ANOVA-derived logit-extremizing exponent                                                |
| FDR threshold `q`                                    | `skill_weighting.benjamini_hochberg_correction`         | `0.05`                                 | Target false-discovery rate when flagging users as "skilled" across the whole universe of forecasters        |
| `n_regimes`                                          | `regime_pipeline.py` (`fit_market_hmm` call)            | `2`                                   | Number of HMM hidden states                                                                                 |
| `sentiment_column`                                   | `backtester.py`                                         | `Sentiment_Mean`                      | Which engineered sentiment feature drives the Bayesian return model at backtest time; logged at `[PRE-FLIGHT]` |
| Validation window                                    | `backtester.py`                                         | `2020-01-15` to `2020-12-31`          | Aligned rebalance periods used for the validation segment                                                    |
| Test window                                          | `backtester.py`                                         | `2021-01-15` to `2022-03-01`          | Aligned rebalance periods used for the (later, out-of-sample) test segment                                   |
| Rebalance frequency                                  | `backtester.py` (`pd.date_range(..., freq='W-FRI')`)    | Weekly, Fridays                       | Backtest rebalance cadence                                                                                  |
| Warm-up window                                       | `backtester.py`                                         | 26 weeks                              | Burn-in period before the first rebalance                                                                   |
| Transaction cost                                     | `backtester.py`                                         | 10 bps × turnover                     | Per-rebalance cost assumption                                                                               |
| MCMC sampling config                                 | `utils.fit_hierarchical_bayes`                          | 4 chains / 2,000 draws / 1,500 tuning | PyMC/NUTS sampler settings                                                                                  |

---

## Outputs

- **`data/processed/{TICKER}_skill_extremized.csv`** (Stage 1.5) — one row
  per (ticker, date) with:
  - `n_skilled` — number of FDR-corrected skilled forecasters that day
  - `Sentiment_Mean_SkillWeighted` — pooled directional score weighted by
    each skilled forecaster's hit-rate strength (`1 - p_value`)
  - `extremizing_k` — the per-day ANOVA-derived extremizing exponent
    actually used (worth logging as a robustness/ablation figure)
  - `Sentiment_Extremized` — the ANOVA-extremized bullish probability,
    mapped back to a signed `[-1, 1]` score for drop-in compatibility with
    the existing `Sentiment_Mean` feature slot
- **Skill-classification diagnostic** (console, Stage 1.5) — raw vs.
  FDR-corrected counts and share of skilled users, the overall unconditional
  hit rate (should sit near 0.50 — a leakage sanity check), average hit rate
  vs. base rate, and, once FDR-skilled users exist, the share of their calls
  that are bullish and a directional-consistency check (flags forecasters
  who are simply permabulls/permabears rather than genuine two-sided
  callers).
- **Console report** (per segment: validation, then test) —
  - Cumulative return, annualized Sharpe ratio, and average weekly turnover
    for the strategy vs. an equal-weight benchmark.
  - **Directional skill diagnostic** — accuracy, precision/recall/F1 on the
    predicted-up-vs-down calls (asset × period observations), a confusion
    matrix, and the F1 score against a majority-class baseline.
  - **Statistical significance test** — mean weekly excess return over the
    benchmark, a t-statistic and two-sided p-value, and a 95% bootstrap
    confidence interval on the mean, with a plain-language verdict on
    whether the excess return is distinguishable from zero at that sample
    size.
  - Skip-reason diagnostics if any rebalance periods were dropped (missing
    regime labels, missing forward returns, etc.).
- **`reports/regime_{n}_optimization.png`** — efficient frontier scatter
(simulated random portfolios + individual assets + optimal portfolio) and
a bar chart of optimal weights, for a single regime/feature snapshot.
- **`reports/backtest_results_dashboard_validation.png`** and
  **`reports/backtest_results_dashboard_test.png`** — four-panel summaries
  of each walk-forward segment:
  1. Cumulative net return, shaded by active regime
  2. Portfolio drawdown
  3. Allocation weights over time (stacked area, by ticker)
  4. Turnover per rebalance

---

## Methodology Notes

- **Skilled-minority identification (Stage 1.5):** rather than trust the raw,
  unweighted crowd, each Stocktwits user's bullish/bearish calls are scored
  against the realized forward-return direction using a **causal, no-look-
  ahead** running record — a user's classification on date *t* never uses
  information realized on or after *t*. Users are tested with a one-sided
  sign test against that ticker's own **causal base rate** (not a flat 50%),
  so a permabull during a genuine uptrend isn't mistaken for a skilled
  forecaster. A **Benjamini-Hochberg FDR correction across users** (not per
  message) is then applied, since a flat p<0.05 cutoff would flag far more
  than 5% of users by chance alone across tens of thousands of testers.
- **ANOVA-style extremizing (Stage 1.5):** the FDR-corrected skilled
  minority's calls are pooled into a daily bullish probability, then
  extremized via a logit transform whose exponent *k* is derived from a
  one-way ANOVA decomposition (between- vs. within-forecaster variance) of
  that day's individual sentiment scores. This is a documented design
  choice inspired by, but not a literal reproduction of, extremizing
  estimators in the forecast-aggregation literature. With only one message
  per forecaster per day, the within-group term collapses and this
  degenerates to `k_min`.
- **Validation/test split:** the walk-forward backtest is run over two
  non-overlapping periods — an earlier **validation** window and a later,
  strictly out-of-sample **test** window — each reported independently so
  that any apparent edge on the validation period can be checked for
  persistence rather than taken at face value.
- **Anti-leakage design:** sentiment features are lagged before merging
with price data, and each rebalance evaluates portfolio weights against
the **next** period's realized return, not the current one. The same
embargo discipline is used in Stage 1.5's forecaster track record.
- **Regime canonicalization:** HMM states are relabeled so Regime 0 is
always the lower-variance ("Low-Volatility") state and Regime 1 the
higher-variance ("High-Volatility/Stressed") state, making labels
comparable across tickers.
- **Hierarchical pooling:** the Bayesian model shares population-level
priors across all four tickers within a regime, so assets with sparser
sentiment data borrow statistical strength from the others rather than
being fit in total isolation.
- **Significance testing:** excess return over the benchmark is tested with
  a paired t-test and a bootstrap confidence interval on the weekly mean,
  since a positive average excess return can otherwise be indistinguishable
  from noise at a few dozen weekly observations.

---

## Limitations

- **`skill_weighting.py` is not yet wired into the pipeline.** It reads
  `data_engine.py`'s output and writes
  `data/processed/{TICKER}_skill_extremized.csv` on its own, but
  `regime_pipeline.py` does not yet consume `Sentiment_Extremized` in place
  of (or alongside, as an ablation) the existing `Sentiment_Mean` feature.
- **Blocking data prerequisite:** `data_engine.py`'s current S3 read
  discards `user_id`, which `skill_weighting.py` requires to build a
  per-forecaster track record. Stage 1.5 will fail with a `KeyError` until
  `data_engine.py` is patched to retain that column.
- **Forward-horizon alignment risk:** the horizon used to score a
  forecaster's "hit" in Stage 1.5 (`FORWARD_HORIZON_DAYS`, default 1 day)
  must match whatever horizon the strategy ultimately trades on (currently
  weekly rebalancing in `backtester.py`) — using mismatched horizons across
  the two stages introduces a subtle label mismatch between how skill was
  measured and what the strategy predicts.
- **Extremizing factor can degenerate.** The ANOVA-derived exponent
  collapses to `k_min` whenever a forecaster posts only one message per
  ticker-day; it only becomes informative with multiple posts per
  forecaster per day, or when computed over a short rolling window instead
  of a single day.
- **In-sample regime labeling.** The backtest currently uses the *fitted* (smoothed) HMM state for each date rather than a causal, real-time
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
- **Statistically inconclusive edge.** In both the validation and test
  segments, the mean weekly excess return over the equal-weight benchmark
  has not been statistically distinguishable from zero (two-sided p-values
  well above conventional thresholds, with bootstrap confidence intervals
  spanning zero). Higher cumulative return and Sharpe ratio than the
  benchmark on a given run should not be read as evidence of a robust edge
  without a larger sample or additional out-of-sample periods.

---