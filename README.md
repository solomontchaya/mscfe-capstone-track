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

> **Status:** research / proof-of-concept pipeline. The skilled-minority +
> extremizing branch (Stage 1.5) is now fully wired into the rest of the
> pipeline and has been run end-to-end as a head-to-head ablation against
> the original naive-sentiment approach. **The ablation result is a
> negative one** — see [Ablation Findings](#ablation-findings) before
> assuming the more sophisticated signal is an improvement. See
> [Limitations](#limitations) before drawing any performance conclusions
> more broadly.

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
- [Ablation Findings](#ablation-findings)
- [Limitations](#limitations)
- [License](#license)

---

## How It Works

```
data_engine.py  ──▶  skill_weighting.py  ──▶  regime_pipeline.py  ──▶  regime_models.py  ──▶  portfolio_engine.py  ──▶  backtester.py
   (ingest raw         (Stage 1.5 --            (feature engineering       (hierarchical Bayes       (single-period          (walk-forward
    sentiment data,     skilled-minority          + HMM regime fitting;      posterior sampling,       allocation demo,        backtest + report,
    retains user_id)    ID + ANOVA-style           merges in the Stage       one model per             one per                 split into
                        extremizing; writes         1.5 signal, lagged        sentiment_column          sentiment_column)       validation/test,
                        Sentiment_Extremized)       one trading day)          x regime)                                        one per
                                                                                                                                 sentiment_column)
```

Stages 3–6 (`regime_models.py`, `portfolio_engine.py`, `backtester.py`) all
accept a `--sentiment-column=` flag, so the same pipeline can be run once
with the original `Sentiment_Mean` and once with the new
`Sentiment_Extremized`, producing distinctly-suffixed output files for a
direct, apples-to-apples comparison. See
[Ablation Findings](#ablation-findings) for what that comparison found.

| Stage | Script                | What it does                                                                                                                                             | Reads                                                    | Writes                                                                |
| ----- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------- |
| 1     | `data_engine.py`      | Streams raw Stocktwits sentiment records (including `user_id`) from a public S3 bucket for the target universe/date range                                | S3 (`stocktwits-nyu` public bucket)                      | `data/processed/{TICKER}.csv`                                         |
| 1.5   | `skill_weighting.py`  | Runs a causal (no-look-ahead) one-sided sign test per Stocktwits user against that ticker's own running base rate, keeps only the FDR-corrected "skilled minority," pools their calls into a daily bullish probability, and extremizes it via a per-day ANOVA-derived logit transform | `data/processed/{TICKER}.csv`, Yahoo Finance (forward returns) | `data/processed/{TICKER}_skill_extremized.csv`                        |
| 2     | `regime_pipeline.py`  | Builds lagged sentiment features (`Sentiment_Mean`) + price features, **merges in the Stage 1.5 output as `Sentiment_Extremized`** (also lagged one trading day, matching `Sentiment_Mean`'s existing look-ahead-bias control), fits a per-ticker Gaussian HMM to classify Low-Vol / High-Vol regimes | `data/processed/{TICKER}.csv`, `data/processed/{TICKER}_skill_extremized.csv`, Yahoo Finance | `data/processed/{TICKER}_with_regimes.csv` (carries both sentiment columns) |
| 3     | `regime_models.py`    | Fits a hierarchical Bayesian regression (PyMC/NUTS) per regime, pooling statistical strength across tickers, using whichever `--sentiment-column` is passed | `data/processed/{TICKER}_with_regimes.csv` (all tickers) | `data/regime_{n}_posterior{suffix}.nc`, `data/regime_{n}_coefficients{suffix}.csv` |
| 4     | `portfolio_engine.py` | Demonstrates a single-period optimal allocation for a given regime + feature snapshot; plots the efficient frontier                                      | `data/regime_{n}_posterior{suffix}.nc`                   | `reports/regime_{n}_optimization{suffix}.png`                         |
| 5     | `backtester.py`       | Runs a weekly walk-forward simulation over a **validation** period and a subsequent **test** period: detect active regime → generate Bayesian (μ, Σ) → optimize weights → apply turnover cost → realize forward return. Reports return/Sharpe metrics, a directional-skill diagnostic, and a statistical-significance test for each segment. | `data/processed/{TICKER}_with_regimes.csv`, posteriors   | Console performance report + `reports/backtest_results_dashboard_{segment}{suffix}.png` |
| —     | `check_momentum_confound.py` | Diagnostic (not part of the main sequence): checks whether `Sentiment_Extremized` correlates with the ticker's own trailing price momentum more than `Sentiment_Mean` does — used to help distinguish "the extremized signal is genuine crowd information" from "it's just an indirect encoding of recent trend" | `data/processed/{TICKER}_with_regimes.csv`               | `data/momentum_confound_diagnostic.csv`                               |

`{suffix}` is empty for the default `Sentiment_Mean` run and
`_sentiment_extremized` when `--sentiment-column=Sentiment_Extremized` is
passed, so neither run's outputs overwrite the other's.

Shared logic (feature engineering, HMM fitting, the PyMC model definition,
the skill-extremized merge, and the Markowitz optimizer) lives in
`utils.py` and is imported by the later stages.

> **Not yet wired in:** `deflated_sharpe.py` (Bailey & López de Prado's
> Deflated Sharpe Ratio) exists as a standalone, tested module but is not
> currently called from `backtester.py`. See
> [Limitations](#limitations) if you want to add it to the reported
> metrics. The herding/crash-shield component described in the original
> project design (cosine-similarity / Gini-index detection of social
> contagion) was investigated and intentionally not built — see
> [Limitations](#limitations) for why.

---

## Project Structure

```
.
├── src/
│   ├── data_engine.py              # Stage 1   — S3 sentiment ingestion (retains user_id)
│   ├── skill_weighting.py          # Stage 1.5 — skilled-minority ID + ANOVA-style extremizing
│   ├── regime_pipeline.py          # Stage 2   — feature engineering, Stage 1.5 merge, HMM regime detection
│   ├── regime_models.py            # Stage 3   — hierarchical Bayesian model fitting (--sentiment-column)
│   ├── portfolio_engine.py         # Stage 4   — single-period optimization + efficient frontier plot
│   ├── backtester.py               # Stage 5   — validation/test walk-forward backtest + results dashboards
│   ├── check_momentum_confound.py  # Diagnostic — Sentiment_Extremized vs. trailing momentum correlation
│   ├── deflated_sharpe.py          # Standalone, NOT yet wired into backtester.py — see Limitations
│   └── utils.py                    # Shared feature engineering, HMM, Bayesian model, optimizer, merge functions
├── data/
│   ├── processed/                        # Per-ticker CSVs at each pipeline stage
│   │   ├── {TICKER}_skill_extremized.csv # Stage 1.5 daily skill-weighted/extremized signal
│   │   └── {TICKER}_with_regimes.csv     # Stage 2 output; carries Sentiment_Mean AND Sentiment_Extremized
│   ├── regime_{n}_posterior.nc                       # Serialized MCMC trace, Sentiment_Mean
│   ├── regime_{n}_posterior_sentiment_extremized.nc  # Serialized MCMC trace, Sentiment_Extremized
│   ├── regime_{n}_coefficients.csv                   # Table 4.1/4.2-style summary, Sentiment_Mean
│   ├── regime_{n}_coefficients_sentiment_extremized.csv
│   └── momentum_confound_diagnostic.csv
├── reports/
│   ├── regime_{n}_optimization{_sentiment_extremized}.png
│   ├── backtest_results_dashboard_validation{_sentiment_extremized}.png
│   └── backtest_results_dashboard_test{_sentiment_extremized}.png
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
- `data_engine.py` retains `user_id` from the raw S3 rows by default —
  `skill_weighting.py` needs it to build a per-forecaster track record.

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

No additional dependencies were introduced by the skill-weighting /
extremizing work — the Benjamini-Hochberg FDR correction is implemented
directly in `skill_weighting.py` without requiring `statsmodels`.

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
# 1. Ingest sentiment data for the target universe (retains user_id)
python src/data_engine.py

# 1.5. Identify the skilled-minority of forecasters and extremize their
#      pooled daily signal
python src/skill_weighting.py

# 2. Engineer features, merge in the Stage 1.5 signal, and fit per-ticker
#    HMM regimes
python src/regime_pipeline.py

# 3. Fit hierarchical Bayesian return models (one per regime)
python src/regime_models.py

# 4. (Optional) Inspect a single-period optimal allocation + efficient frontier
python src/portfolio_engine.py

# 5. Run the full walk-forward backtest (validation, then test)
python src/backtester.py
```

**To run the head-to-head ablation** against the original naive
`Sentiment_Mean` feature, repeat steps 3–5 with the flag:

```
python src/regime_models.py --sentiment-column=Sentiment_Extremized
python src/portfolio_engine.py --sentiment-column=Sentiment_Extremized
python src/backtester.py --sentiment-column=Sentiment_Extremized
```

Output files from the two runs are suffixed differently
(`_sentiment_extremized`), so nothing gets overwritten and both can be
diffed directly — this is exactly what produced the comparison in
[Ablation Findings](#ablation-findings).

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

And `check_momentum_confound.py` runs independently once
`regime_pipeline.py` has produced `{TICKER}_with_regimes.csv`:

```
python src/check_momentum_confound.py
```

Step 1.5 prints, per ticker, how many users and messages were loaded, a
skill-classification diagnostic (raw vs. FDR-corrected skilled counts, the
overall unconditional hit rate as a leakage sanity check, average hit rate
vs. the ticker's causal base rate, share of flagged calls that are bullish,
and a directional-consistency check), and how many daily extremized signal
rows were produced, before writing
`data/processed/{TICKER}_skill_extremized.csv`.

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
Total Cumulative Return            2.37%                 0.19%
Annualized Sharpe Ratio           0.2203                0.0866
Average Weekly Turnover           11.13%                    —
==================================================

DIRECTIONAL SKILL DIAGNOSTIC — VALIDATION
--------------------------------------------------
Observations (assets x periods): 200
Accuracy:                      54.00%
Precision (predicted 'up'):    0.5059
Recall (predicted 'up'):       0.4624
F1 Score:                      0.4831
  vs. majority-class baseline: 0.0000
Confusion Matrix  [[TN=65, FP=42], [FN=50, TP=43]]
--------------------------------------------------

STATISTICAL SIGNIFICANCE — VALIDATION (Strategy vs. Benchmark)
--------------------------------------------------
Observations (weekly periods):  50
Mean weekly excess return:      0.0685%
t-statistic:                    0.2207
p-value (two-sided):            0.8263
95% bootstrap CI on mean:       [-0.5437%, 0.6406%]
Verdict: Excess return is NOT statistically distinguishable from zero -- the observed gap vs. the benchmark could plausibly be noise at this sample size.
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

**Reproducibility check:** running `backtester.py` twice in a row with the
same `--sentiment-column` should now produce byte-identical output
(including every per-period `Net Return` line) — see
[Methodology Notes](#methodology-notes) for why this wasn't always true and
what was fixed.

---

## Configuration

Key parameters are defined as constants near the top of each script — edit
these directly to customize a run:

| Parameter                                            | Location                                                | Default                               | Purpose                                                                                                     |
| ----------------------------------------------------- | ------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `TARGET_UNIVERSE` / `tickers`                        | `data_engine.py`, `skill_weighting.py`, `regime_pipeline.py`, `backtester.py` | `["AAPL", "AMD", "SPY", "TSLA"]`      | Asset universe (must match across all scripts)                                                                |
| `START_DATE` / `END_DATE`                            | `data_engine.py`, `skill_weighting.py`, `regime_pipeline.py`                  | `2014-01-01` / `2022-03-01`           | Start/end of data pull and feature window — kept consistent across all three scripts                        |
| `FORWARD_HORIZON_DAYS`                               | `skill_weighting.py`                                    | `1` (next trading day)                | Horizon used to score whether a forecaster's directional call was a "hit" — see Limitations re: alignment with the weekly backtest horizon |
| `min_track_record`                                   | `skill_weighting.build_forecaster_track_record`         | `20`                                   | Minimum number of prior scored calls a user needs before their sign-test p-value is treated as meaningful   |
| `embargo_days`                                       | `skill_weighting.build_forecaster_track_record`         | `1`                                    | Buffer between a message and the cutoff for "prior" calls, to avoid forward-return leakage      |
| `base_rate_window`                                   | `skill_weighting.build_forecaster_track_record` (`--rolling-window=N` CLI flag) | `None` (full-history expanding)       | Null hypothesis for the sign test: expanding causal base rate vs. a rolling N-day window                     |
| `k_min` / `k_max`                                    | `skill_weighting.extremize_daily_signal`                | `1.0` / `3.0`                          | Bounds on the per-day ANOVA-derived logit-extremizing exponent                                                |
| FDR threshold `q`                                    | `skill_weighting.benjamini_hochberg_correction`         | `0.05`                                 | Target false-discovery rate when flagging users as "skilled" across the whole universe of forecasters        |
| `n_regimes`                                          | `regime_pipeline.py` (`fit_market_hmm` call)            | `2`                                   | Number of HMM hidden states                                                                                 |
| `--sentiment-column=`                                | `regime_models.py`, `portfolio_engine.py`, `backtester.py` (CLI flag) | `Sentiment_Mean`                      | Which engineered sentiment feature drives the Bayesian return model — `Sentiment_Mean` (naive) or `Sentiment_Extremized` (Stage 1.5 output); logged at `[PRE-FLIGHT]`. Output files are suffixed accordingly so both runs coexist. |
| `random_seed` (posterior-predictive draw)            | `utils.generate_bayesian_inputs`                        | `42`, varied per rebalance period in `backtester.py` (`42 + idx`) | Seeds the Monte Carlo draw from the posterior — see Methodology Notes; previously unseeded, which was a real reproducibility bug |
| Validation window                                    | `backtester.py`                                         | `2020-01-15` to `2020-12-31`          | Aligned rebalance periods used for the validation segment                                                    |
| Test window                                          | `backtester.py`                                         | `2021-01-15` to `2022-03-01`          | Aligned rebalance periods used for the (later, out-of-sample) test segment                                   |
| Rebalance frequency                                  | `backtester.py` (`pd.date_range(..., freq='W-FRI')`)    | Weekly, Fridays                       | Backtest rebalance cadence                                                                                  |
| Warm-up window                                       | `backtester.py`                                         | 26 weeks                              | Burn-in period before the first rebalance                                                                   |
| Transaction cost                                     | `backtester.py`                                         | 10 bps × turnover                     | Per-rebalance cost assumption                                                                               |
| MCMC sampling config                                 | `utils.fit_hierarchical_bayes`                          | 4 chains / 2,000 draws / 1,500 tuning, `random_seed=42` | PyMC/NUTS sampler settings                                                                    |

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
  vs. base rate, share of FDR-skilled calls that are bullish, and a
  directional-consistency check (flags forecasters who are simply
  permabulls/permabears rather than genuine two-sided callers — see
  [Ablation Findings](#ablation-findings) for what this found).
- **`data/processed/{TICKER}_with_regimes.csv`** (Stage 2) — carries both
  `Sentiment_Mean` and `Sentiment_Extremized` (each independently lagged
  one trading day) alongside HMM regime labels and price features.
- **`data/regime_{n}_coefficients{suffix}.csv`** (Stage 3) — the
  posterior mean/sd/r_hat/ess table per asset and coefficient (a
  machine-readable version of what previously only printed to console),
  suffixed so the `Sentiment_Mean` and `Sentiment_Extremized` runs can be
  diffed directly.
- **`data/momentum_confound_diagnostic.csv`** — per-ticker Pearson
  correlation of `Sentiment_Extremized` and `Sentiment_Mean` against 5/10/
  20/60-day trailing momentum, restricted to the training window.
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
- **`reports/regime_{n}_optimization{suffix}.png`** — efficient frontier
scatter (simulated random portfolios + individual assets + optimal
portfolio) and a bar chart of optimal weights, for a single regime/feature
snapshot.
- **`reports/backtest_results_dashboard_{validation,test}{suffix}.png`** —
  four-panel summaries of each walk-forward segment:
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
  On the real dataset, this brought the flagged-skilled share down from an
  implausible ~35% (raw, flat null) to single digits per ticker — still
  above the ~3% figure reported for a comparable setting in Gomez-Cram et
  al., and diagnostic work traced the remainder to a specific, reportable
  finding — see [Ablation Findings](#ablation-findings).
- **ANOVA-style extremizing (Stage 1.5):** the FDR-corrected skilled
  minority's calls are pooled into a daily bullish probability, then
  extremized via a logit transform whose exponent *k* is derived from a
  one-way ANOVA decomposition (between- vs. within-forecaster variance) of
  that day's individual sentiment scores. This is a documented design
  choice inspired by, but not a literal reproduction of, extremizing
  estimators in the forecast-aggregation literature. With only one message
  per forecaster per day, the within-group term collapses and this
  degenerates to `k_min`.
- **Look-ahead-bias parity between the two sentiment features.** The
  original `Sentiment_Mean` feature is lagged one trading day before
  merging with price data (a deliberate, documented control in
  `process_local_chunks`). Early in development, the `Sentiment_Extremized`
  merge (`utils.merge_skill_extremized_signal`) did **not** apply the same
  lag — meaning an early ablation run was regressing same-day sentiment
  against the same-day return it could react to, not predict. This produced
  a dramatic, spurious-looking result (Bayesian coefficients jumping from
  indistinguishable-from-zero to strongly significant) that did not survive
  once the lag was corrected to match `Sentiment_Mean` exactly (a
  **positional**, trading-day shift — not a calendar-day `Timedelta`, which
  would misalign across weekends). The corrected, lagged version is what's
  in the current codebase and what produced the results in
  [Ablation Findings](#ablation-findings).
- **Reproducibility of the posterior-predictive draw.**
  `utils.generate_bayesian_inputs` draws Monte Carlo samples from the fitted
  posterior to build the (μ, Σ) inputs to the optimizer at each rebalance.
  This previously used the global, unseeded numpy RNG, so **the exact same
  backtest command, run twice in a row with no code or data changes, could
  produce meaningfully different results** (observed swings of ~50% in
  reported Sharpe ratio across identical re-runs during development). This
  is now a locally-seeded generator (`random_seed` parameter, varied
  per-rebalance-period so different periods still get independent noise),
  and repeated runs now match exactly — verified directly before trusting
  any of the numbers in [Ablation Findings](#ablation-findings).
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

## Ablation Findings

This is the headline result of the skilled-minority / extremizing work, and
it's a negative one: **on this dataset, the more sophisticated,
skill-weighted and extremized sentiment signal does not outperform the
original naive daily-mean sentiment — it underperforms it on most metrics.**

| Metric | Validation (2020) — Mean → Extremized | Test (2021–2022) — Mean → Extremized |
| --- | --- | --- |
| Total cumulative return | 2.37% → **-4.26%** | 13.13% → 10.43% |
| Annualized Sharpe ratio | 0.220 → **-0.192** | 0.800 → 0.670 |
| Average weekly turnover | 11.13% → 11.64% | 12.78% → 18.14% |
| Directional accuracy | 54.00% → **47.00%** | 53.39% → 52.54% |
| Excess return vs. benchmark | Not statistically significant | Not statistically significant |

(F1 score is higher under `Sentiment_Extremized` in both segments, but this
is a confusion-matrix artifact, not better skill — the extremized signal
predicts "up" far more aggressively, which inflates recall and F1 under
class imbalance while precision and raw accuracy both fall.)

**Why:** a diagnostic chain traced this to the composition of the
FDR-corrected "skilled minority" itself. A directional-consistency check —
what fraction of a flagged user's calls sit in their single most common
direction — found the flagged population is **~90% fixed-stance** on
average (AAPL 90.2%, AMD 93.6%, SPY 89.3%, TSLA 93.1%), with a majority of
individual flagged users calling the same direction ≥95% of the time. That's
not the calibration-and-updating behavior the forecasting-skill literature
(e.g. Mellers et al., cited in the accompanying paper) defines as genuine
skill — it's closer to persistently bullish or bearish posters whose
historical "accuracy" derives from riding each ticker's own trend rather
than from two-sided, information-driven calls. Extremizing that population's
signal amplifies a trend-following bias, not a forecasting edge, which is
consistent with why it underperforms simple averaging once measured fairly
(after the look-ahead-bias and reproducibility fixes described in
[Methodology Notes](#methodology-notes)).

This should be read as a genuine, reportable finding rather than a dead
end: naive sentiment averaging showing no signal is one result; a more
sophisticated, causally-safeguarded skill-weighting scheme *also* showing no
exploitable signal — and for an identifiable, literature-consistent reason
— is a stronger and more useful one.

---

## Limitations

- **Forward-horizon alignment risk.** The horizon used to score a
  forecaster's "hit" in Stage 1.5 (`FORWARD_HORIZON_DAYS`, default 1 day)
  does not match the weekly rebalance horizon `backtester.py` ultimately
  trades on. Using mismatched horizons across the two stages introduces a
  subtle label mismatch between how skill was measured and what the
  strategy predicts — worth aligning (e.g. `FORWARD_HORIZON_DAYS=5`) as a
  follow-up robustness check.
- **Base-rate window choice affects results, and not cleanly in one
  direction.** Comparing the default expanding (full-history) causal base
  rate against a 60-day rolling alternative (`--rolling-window=60`) gave
  mixed results across tickers — the rolling window reduced the bullish
  skew of flagged calls for three of four tickers but *increased* the
  flagged-skilled rate for three of four (including a notable jump for
  SPY). The directional-consistency finding above holds under both windows,
  but the flagged population's exact size is sensitive to this choice —
  worth treating the expanding-window result as primary and the rolling
  result as a documented robustness check, not as a second, equally-valid
  headline number.
- **Extremizing factor can degenerate.** The ANOVA-derived exponent
  collapses to `k_min` whenever a forecaster posts only one message per
  ticker-day; it only becomes informative with multiple posts per
  forecaster per day, or when computed over a short rolling window instead
  of a single day.
- **In-sample regime labeling.** The backtest currently uses the *fitted*
  (smoothed) HMM state for each date — `hmmlearn`'s default Viterbi
  decoding finds the globally optimal state path across the *entire* input
  sequence, so a regime label for an early date can, in principle, be
  influenced by data from much later in the sequence, even though the HMM's
  emission/transition *parameters* are fit strictly on the training window
  only. A live/forward-testing deployment would need an online, causal
  filtering approach (not full-sequence Viterbi decoding) to fully avoid
  this.
- **Static posteriors.** Bayesian models are fit once per regime
(`regime_models.py`) rather than re-estimated on a rolling basis, so
the backtest is out-of-sample on realized returns but not on model
parameters.
- **Simplified transaction costs.** A flat 10 bps per unit of turnover;
no bid-ask spread or market-impact modeling.
- **Small universe.** Four assets, chosen to validate the pipeline
end-to-end — not intended as a diversified, production-ready portfolio.
- **Raw data completeness.** The most recent full `data_engine.py` S3
  ingestion run logged 8 of 34 source chunks failing with
  `Could not connect to the endpoint URL` (a contiguous-looking pattern
  more consistent with transient rate-limiting than missing files). These
  chunks were not confirmed retried before downstream stages were run —
  worth re-running `data_engine.py` (or specifically the failed chunk
  indices) and confirming row counts before treating any reported figures
  as final; the current numbers may understate true message volume by
  roughly a quarter.
- **Deflated Sharpe Ratio not yet wired in.** `deflated_sharpe.py`
  implements Bailey & López de Prado's DSR (corrects the reported Sharpe
  for the number of configurations tried, track length, and
  skew/kurtosis) and is tested standalone, but `backtester.py` does not yet
  call it. Given how many configurations have now genuinely been tried in
  this project's history (turnover-control ablation, base-rate window
  choice, sentiment-column choice, at minimum), wiring this in and
  reporting it honestly alongside the t-test/bootstrap result would
  meaningfully strengthen the significance-testing section.
- **Herding/crash-shield component not built.** The original project
  design referenced a social-contagion "crash shield" using cosine
  similarity of message arguments and a Gini index of sentiment dispersion
  (Li et al.-style). This needs raw message **text**, which the
  `symbol_sentiments` S3 path currently ingested does not carry (only a
  pre-scored `sentiment` value). A lighter substitute achievable from
  existing data — a Gini index of the per-message sentiment *score*
  distribution each day, as a herding/consensus proxy in place of the
  current `Sentiment_Variance` — was identified but not implemented.
- **Statistically inconclusive edge.** In both the validation and test
  segments, for both `Sentiment_Mean` and `Sentiment_Extremized`, the mean
  weekly excess return over the equal-weight benchmark is not statistically
  distinguishable from zero (two-sided p-values well above conventional
  thresholds, with bootstrap confidence intervals spanning zero). Higher
  cumulative return and Sharpe ratio than the benchmark on a given run
  should not be read as evidence of a robust edge without a larger sample
  or additional out-of-sample periods.

---