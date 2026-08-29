# skill_weighting
"""
Stage 1.5 -- Skilled-Minority Identification and ANOVA-Style Extremizing
=========================================================================

Sits between data_engine.py (raw per-message ingestion) and
regime_pipeline.py (feature engineering + HMM regime detection).

Implements two pieces of the literature review that the current pipeline
promises but doesn't build:

1. Skilled-minority weighting (Gomez-Cram et al. 2026; Mellers et al. 2015):
   For each Stocktwits user, track a *causal* (no-look-ahead) running record
   of whether their bullish/bearish call matched the sign of the forward
   return, then run a one-sided sign test (H0: p=0.5, H1: p>0.5) against
   that record as of each date. Only users whose p-value clears a threshold
   are treated as "skilled" and contribute to the daily aggregate signal.

2. ANOVA-style extremizing (Satopaa et al. 2014):
   Pools the skilled minority's calls into a daily probability of "bullish",
   then extremizes that pooled probability via a logit transform whose
   exponent k is derived from a one-way ANOVA decomposition of that day's
   individual sentiment scores (between-forecaster variance vs.
   within-forecaster variance) -- the "ANOVA-style extremizing factor"
   named in the abstract.

IMPORTANT -- data prerequisite:
    This requires `user_id` to be retained in the raw ingestion step.
    Your current data_engine.py / utils.process_s3_sentiment_pipeline
    read only ['created_at', 'sentiment', 'symbol_list'] (or similar) and
    discard `user_id`, even though the NYU Stocktwits S3 schema includes
    it (message_id, user_id, created_at, sentiment, symbol_list, ...).
    See `patch_data_engine.md` in this same delivery for the minimal diff
    needed to retain it.

Expected input schema (one row per message, per ticker, already exploded
so each row is a single (user_id, ticker) pair for one message):
    columns: ['user_id', 'ticker', 'created_at', 'sentiment_score']
    - sentiment_score: signed continuous or discrete Stocktwits value.
      Only its SIGN is used for the directional call (bullish=+1, bearish=-1).
      Neutral (0) messages are dropped from skill tracking (no directional
      claim to score) but can still be folded into Sentiment_Variance
      elsewhere in your existing pipeline.

Expected forward-return input:
    A per-(ticker, date) realized forward return, e.g. the next trading
    day's or next N-day's log return -- whatever horizon you use to score
    a "hit". This must be the SAME forward horizon used to build your
    forecaster track record and to build the ANOVA-style extremized signal
    that regime_pipeline.py will later consume, or you introduce a subtle
    label mismatch between how skill was measured and what the strategy
    predicts.

Output:
    A daily panel per ticker with:
        - n_skilled: number of forecasters classified skilled that day
        - Sentiment_Mean_SkillWeighted: skill-weighted pooled directional score
        - Sentiment_Extremized: ANOVA-extremized probability of "bullish",
          mapped back to a [-1, 1] signed score for drop-in compatibility
          with the existing Sentiment_Mean feature slot.
        - extremizing_k: the per-day k factor actually used (log this --
          it's a natural robustness/ablation figure for the paper)

This module has NO network calls. It operates on an in-memory per-message
dataframe you build once you've retained user_id upstream.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


# ---------------------------------------------------------------------------
# Config -- mirrors data_engine.py / regime_pipeline.py so this stage stays
# consistent with the rest of the pipeline without re-typing constants.
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

TARGET_UNIVERSE = ["AAPL", "AMD", "SPY", "TSLA"]
START_DATE = "2014-01-01"
END_DATE = "2022-03-01"

# Forward-return horizon (in trading days) used to score whether a message's
# directional call was a "hit". 1 = next trading day. Align this with
# whatever horizon backtester.py ultimately rebalances on (currently weekly)
# once you wire the two stages together -- scoring skill against a different
# horizon than the one the strategy trades on is a real mismatch to avoid.
FORWARD_HORIZON_DAYS = 1


# ---------------------------------------------------------------------------
# Step 1: Causal, no-look-ahead forecaster track record + one-sided sign test
# ---------------------------------------------------------------------------

def build_forecaster_track_record(
    df_messages: pd.DataFrame,
    df_forward_hits: pd.DataFrame,
    min_track_record: int = 20,
    embargo_days: int = 1,
    base_rate_window: int = None,
) -> pd.DataFrame:
    """
    For every (user_id, ticker, date) message, compute that user's one-sided
    sign-test p-value using ONLY calls strictly before (date - embargo_days),
    so a user's classification on date t never uses information realized on
    or after t. This mirrors the train/validation/test embargo discipline
    already used elsewhere in your pipeline (regime_pipeline.py, backtester.py).

    Parameters
    ----------
    df_messages : columns ['user_id', 'ticker', 'created_at', 'sentiment_score']
        One row per message. `created_at` should already be normalized to a
        trading-day date (tz-naive).
    df_forward_hits : columns ['ticker', 'date', 'forward_return']
        The realized forward return used to score whether a call was a hit.
        Must use the same horizon as the rest of your pipeline.
    min_track_record : minimum number of PRIOR scored calls a user must have
        before they are eligible to be classified (otherwise the sign test
        has essentially no power and the p-value is not meaningful).
    embargo_days : buffer (in calendar days) between the message date and the
        cutoff for "prior" calls used to compute the running record, to keep
        forward-return realization lag from leaking into the record.
    base_rate_window : if None (default), the sign test's null is the
        ticker's full-history EXPANDING causal base rate of "up" days. If
        set to an integer (e.g. 60), the null instead uses a ROLLING
        window of that many trading days immediately preceding the cutoff.
        A rolling window tracks LOCAL trend/momentum strength rather than
        the lifetime average -- important because a stock's up-day rate is
        not constant over 8 years, and a permabull active during a locally
        strong uptrend can still clear a stale lifetime-average null even
        with zero real skill. Run both and compare
        pct_of_fdr_skilled_calls_that_are_bullish across the two: if it
        drops substantially under the rolling window, the residual "skill"
        under the expanding window was mostly short-horizon momentum, not
        individual forecasting ability.

    Returns
    -------
    DataFrame with columns:
        ['user_id', 'ticker', 'date', 'sentiment_score', 'direction',
         'n_prior_calls', 'n_prior_hits', 'base_rate_null', 'p_value_skill', 'is_skilled']
    """
    df = df_messages.copy()
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.normalize()
    df = df[df["sentiment_score"] != 0].copy()  # drop neutral, no directional claim
    df["direction"] = np.sign(df["sentiment_score"]).astype(int)

    hits = df_forward_hits.copy()
    hits["date"] = pd.to_datetime(hits["date"]).dt.normalize()
    hits["forward_direction"] = np.sign(hits["forward_return"]).astype(int)

    df = df.merge(
        hits[["ticker", "date", "forward_direction"]],
        left_on=["ticker", "created_at"],
        right_on=["ticker", "date"],
        how="left",
    )
    df = df.dropna(subset=["forward_direction"])
    df["hit"] = (df["direction"] == df["forward_direction"]).astype(int)

    df = df.sort_values(["user_id", "created_at"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Causal, no-look-ahead base rate of "up" days for this ticker.
    # A flat p=0.5 null is wrong for a single trending asset: if the
    # ticker spent most of the window trending up, a permanently-bullish
    # user gets an inflated hit rate purely from matching the trend, with
    # zero actual directional skill. Testing against the asset's own
    # running base rate (estimated from PRIOR days only, same embargo
    # discipline as everything else) isolates genuine call-timing skill
    # from simple trend-following.
    # ------------------------------------------------------------------
    hits_sorted = hits.sort_values("date").reset_index(drop=True)
    hit_dates = hits_sorted["date"].values
    up_flags = (hits_sorted["forward_direction"].values == 1).astype(int)
    cum_up = np.concatenate(([0], np.cumsum(up_flags)))  # cum_up[i] = up-days among first i

    # ------------------------------------------------------------------
    # Vectorized per-user pass. The original implementation re-scanned
    # each user's ENTIRE message history for every single message
    # (O(n^2) per user), which is why this crawled for power users with
    # thousands of posts. This version uses one searchsorted lookup into
    # a precomputed cumulative-hit array per user (O(n log n) per user).
    # ------------------------------------------------------------------
    n_prior_arr = np.empty(len(df), dtype=np.int64)
    n_prior_hits_arr = np.empty(len(df), dtype=np.int64)
    base_rate_arr = np.full(len(df), 0.5)  # fallback for rows before any price history exists

    for user_id, group_idx in df.groupby("user_id").indices.items():
        group_idx = np.sort(group_idx)
        dates = df["created_at"].values[group_idx]
        hits_arr = df["hit"].values[group_idx]

        cum_hits = np.concatenate(([0], np.cumsum(hits_arr)))
        cutoffs = dates - np.timedelta64(embargo_days, "D")
        n_prior = np.searchsorted(dates, cutoffs, side="left")

        n_prior_arr[group_idx] = n_prior
        n_prior_hits_arr[group_idx] = cum_hits[n_prior]

        # Ticker-level causal base rate as of each row's own cutoff --
        # independent of this user's own history, just "how often has
        # this ticker gone up" either over its full prior history
        # (base_rate_window=None) or over the trailing window only.
        base_idx = np.searchsorted(hit_dates, cutoffs, side="left")
        if base_rate_window is None:
            window_start = np.zeros_like(base_idx)
        else:
            window_start = np.clip(base_idx - base_rate_window, 0, None)
        window_len = base_idx - window_start
        valid = window_len >= min_track_record
        up_in_window = cum_up[np.clip(base_idx, 1, None)] - cum_up[window_start]
        rates = np.where(valid, up_in_window / np.clip(window_len, 1, None), 0.5)
        base_rate_arr[group_idx] = rates

    df["n_prior_calls"] = n_prior_arr
    df["n_prior_hits"] = n_prior_hits_arr
    df["base_rate_null"] = base_rate_arr

    eligible = df["n_prior_calls"] >= min_track_record
    p_value = np.full(len(df), np.nan)
    # Exact one-sided binomial (sign test) p-value against the ticker's OWN
    # causal base rate, vectorized: P(X >= k | n, base_rate) = sf(k-1, n, base_rate).
    p_value[eligible.values] = stats.binom.sf(
        df.loc[eligible, "n_prior_hits"].values - 1,
        df.loc[eligible, "n_prior_calls"].values,
        df.loc[eligible, "base_rate_null"].values,
    )
    df["p_value_skill"] = p_value
    df["is_skilled"] = df["p_value_skill"] < 0.05  # raw, uncorrected -- see benjamini_hochberg_correction()
    df.loc[~eligible, ["n_prior_calls", "n_prior_hits"]] = np.nan

    # The earlier merge (left_on='created_at', right_on='date') left both
    # 'created_at' and 'date' as separate columns with identical values --
    # drop the redundant one from df_forward_hits and keep 'created_at' as
    # the canonical 'date' going forward.
    df = df.drop(columns=["date"]).rename(columns={"created_at": "date"})

    return df[["user_id", "ticker", "date", "sentiment_score", "direction",
               "n_prior_calls", "n_prior_hits", "base_rate_null", "p_value_skill", "is_skilled"]].reset_index(drop=True)


def benjamini_hochberg_correction(df_track: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """
    Applies Benjamini-Hochberg FDR correction ACROSS DISTINCT USERS (using
    each user's most recent/most-informed p-value, not one test per message)
    rather than the raw p<0.05 cutoff on every message.

    With tens of thousands of users each tested repeatedly, a flat p<0.05
    cutoff will flag far more than 5% of users by chance alone. This
    controls the expected proportion of false positives among users flagged
    'skilled' to no more than q.

    Adds an 'is_skilled_fdr' column to the input, applied per user (i.e. a
    user is either FDR-skilled or not; that status applies to all of their
    rows, using their final/best-informed p-value as of the end of the
    observed window).
    """
    df = df_track.copy()
    per_user_latest = (
        df.dropna(subset=["p_value_skill"])
        .sort_values("date")
        .groupby("user_id")
        .tail(1)[["user_id", "p_value_skill"]]
        .reset_index(drop=True)
    )

    m = len(per_user_latest)
    if m == 0:
        df["is_skilled_fdr"] = False
        return df

    sorted_p = per_user_latest.sort_values("p_value_skill").reset_index(drop=True)
    ranks = np.arange(1, m + 1)
    bh_threshold = ranks / m * q
    passing = sorted_p["p_value_skill"].values <= bh_threshold
    # Largest rank where p_(i) <= (i/m)*q -- everything at or below that rank passes.
    if passing.any():
        cutoff_rank = np.max(np.where(passing)[0])
        skilled_users = set(sorted_p.loc[:cutoff_rank, "user_id"])
    else:
        skilled_users = set()

    df["is_skilled_fdr"] = df["user_id"].isin(skilled_users)
    return df


def summarize_skill_classification(df_track: pd.DataFrame, df_messages_raw: pd.DataFrame = None) -> dict:
    """
    Diagnostic report to run BEFORE trusting the skilled-minority output.
    Distinguishes two very different explanations for a high raw-flagged
    rate: (a) many messages from a few prolific flagged users (expected,
    fixable via FDR), vs (b) a broadly elevated hit rate across almost
    everyone (points to a leakage bug, not a power-user artifact).
    """
    classified = df_track.dropna(subset=["p_value_skill"])
    n_classified_msgs = len(classified)
    n_skilled_msgs_raw = int(classified["is_skilled"].sum())

    n_distinct_users_classified = classified["user_id"].nunique()
    n_distinct_users_skilled_raw = classified.loc[classified["is_skilled"], "user_id"].nunique()

    # Unconditional hit rate across ALL classified messages, ignoring skill
    # status entirely -- this is the key check for broad leakage. It should
    # sit close to 0.50 if there's no systematic bias; if it's e.g. 0.55+,
    # something upstream (timestamp alignment, horizon, timezone) is likely
    # leaking information into every user's record, not just a skilled few.
    overall_hit_rate = float(
        (classified["n_prior_hits"] / classified["n_prior_calls"]).mean()
    )
    overall_hit_rate_vs_base = float(
        (classified["n_prior_hits"] / classified["n_prior_calls"] - classified["base_rate_null"]).mean()
    ) if "base_rate_null" in classified.columns else np.nan

    avg_base_rate = float(classified["base_rate_null"].mean()) if "base_rate_null" in classified.columns else np.nan

    report = {
        "n_classified_messages": n_classified_msgs,
        "pct_messages_flagged_skilled_raw": n_skilled_msgs_raw / n_classified_msgs if n_classified_msgs else np.nan,
        "n_distinct_users_classified": n_distinct_users_classified,
        "n_distinct_users_skilled_raw": n_distinct_users_skilled_raw,
        "pct_distinct_users_skilled_raw": (
            n_distinct_users_skilled_raw / n_distinct_users_classified if n_distinct_users_classified else np.nan
        ),
        "overall_unconditional_hit_rate": overall_hit_rate,
        "avg_causal_base_rate": avg_base_rate,
        "avg_hit_rate_minus_base_rate": overall_hit_rate_vs_base,
    }

    if "is_skilled_fdr" in df_track.columns:
        classified_fdr = df_track.dropna(subset=["p_value_skill"])
        skilled_rows = classified_fdr[classified_fdr["is_skilled_fdr"]]
        n_users_fdr = skilled_rows["user_id"].nunique()
        report["n_distinct_users_skilled_fdr"] = n_users_fdr
        report["pct_distinct_users_skilled_fdr"] = (
            n_users_fdr / n_distinct_users_classified if n_distinct_users_classified else np.nan
        )
        # If FDR-skilled users are overwhelmingly bullish, that confirms
        # they're being flagged for riding the trend, not for genuine
        # two-sided call-timing skill.
        if len(skilled_rows) > 0:
            report["pct_of_fdr_skilled_calls_that_are_bullish"] = float((skilled_rows["direction"] == 1).mean())

        # Directional-consistency check: a genuinely skilled forecaster
        # (per Mellers et al. 2014/2015's own definition, cited in the
        # paper -- calibration and updating with new information) should
        # sometimes call bearish and sometimes bullish, tracking the
        # asset's actual direction over time. A user who is simply a fixed
        # permabull/permabear (always the same direction) isn't exhibiting
        # "skill" in that sense, regardless of which base rate the sign
        # test used -- they're a static bet, not a two-sided forecaster.
        # This is computed independent of window choice, so it's a useful
        # cross-check when the expanding-vs-rolling comparison is mixed.
        if len(skilled_rows) > 0:
            per_user_modal_share = (
                skilled_rows.groupby("user_id")["direction"]
                .apply(lambda s: s.value_counts(normalize=True).iloc[0])
            )
            report["avg_fdr_skilled_directional_consistency"] = float(per_user_modal_share.mean())
            report["pct_fdr_skilled_users_always_same_direction"] = float((per_user_modal_share >= 0.95).mean())

    return report


# ---------------------------------------------------------------------------
# Step 2: ANOVA-style extremizing factor, applied to the pooled skilled signal
# ---------------------------------------------------------------------------

def _anova_extremizing_factor(scores: np.ndarray, groups: np.ndarray,
                               k_min: float = 1.0, k_max: float = 3.0) -> float:
    """
    Derive a per-day extremizing exponent k from a one-way ANOVA decomposition
    of that day's individual (skilled-forecaster) sentiment scores, grouped
    by forecaster.

    Rationale (Satopaa et al. 2014): naive pooling under-corrects for
    forecaster overconfidence/correlated error. The more the pooled forecast
    hugs the group mean relative to within-forecaster noise (i.e. the higher
    the between/within variance ratio, analogous to an ANOVA F-ratio), the
    more the group's shared signal should be pushed away from the neutral
    midpoint before being used -- hence "extremizing". This is a documented
    design choice, not a literal reproduction of Satopaa et al.'s estimator;
    flag it as such in the methodology writeup.

    With only one score per forecaster per day (the typical case), the
    "within-group" term collapses and this degenerates gracefully to
    k = k_min. The factor becomes informative once forecasters post more
    than one message per ticker-day, or when this is computed over a
    short rolling window (e.g. 3-5 trading days) instead of a single day --
    recommended if your typical daily message-per-user count is low.
    """
    if len(scores) < 2 or len(np.unique(groups)) < 2:
        return k_min

    df_anova = pd.DataFrame({"score": scores, "group": groups})
    group_means = df_anova.groupby("group")["score"].mean()
    grand_mean = df_anova["score"].mean()
    group_sizes = df_anova.groupby("group")["score"].size()

    ss_between = float((group_sizes * (group_means - grand_mean) ** 2).sum())
    ss_within = float(
        df_anova.groupby("group")["score"]
        .apply(lambda s: ((s - s.mean()) ** 2).sum())
        .sum()
    )

    df_between = len(group_means) - 1
    df_within = len(df_anova) - len(group_means)

    if df_between <= 0 or df_within <= 0 or ss_within <= 1e-12:
        return k_max  # all remaining disagreement is between-forecaster -> fully extremize

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_ratio = ms_between / (ms_within + 1e-12)

    # Map the (unbounded) F-ratio onto [k_min, k_max] with a saturating
    # transform so a handful of noisy days can't blow up the extremizing
    # exponent.
    k = k_min + (k_max - k_min) * (1 - np.exp(-f_ratio / 4.0))
    return float(np.clip(k, k_min, k_max))


def extremize_daily_signal(
    df_track_record: pd.DataFrame,
    k_min: float = 1.0,
    k_max: float = 3.0,
    eps: float = 1e-4,
) -> pd.DataFrame:
    """
    Pools the SKILLED subset of each day's calls into a probability of
    "bullish", derives that day's ANOVA-style extremizing factor k, and
    applies the Satopaa-style logit extremizing transform:

        logit(p_pooled) = log(p / (1-p))
        logit(p_extreme) = k * logit(p_pooled)
        p_extreme = sigmoid(logit(p_extreme))

    Returns one row per (ticker, date) -- drop this straight into
    regime_pipeline.py in place of (or alongside, for an ablation) the
    current naive Sentiment_Mean.
    """
    df = df_track_record[df_track_record["is_skilled"] == True].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "date", "n_skilled", "Sentiment_Mean_SkillWeighted",
            "extremizing_k", "Sentiment_Extremized",
        ])

    records = []
    for (ticker, date), group in df.groupby(["ticker", "date"]):
        n_skilled = len(group)
        # Pooled probability of "bullish" among the skilled minority.
        p_pooled = np.clip((group["direction"] == 1).mean(), eps, 1 - eps)

        # Weight each skilled forecaster's contribution by their own
        # historical hit-rate strength (1 - p_value), so a barely-significant
        # forecaster counts less than a strongly significant one.
        weights = 1.0 - group["p_value_skill"].values
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_skilled) / n_skilled
        weighted_mean = float(np.dot(weights, group["sentiment_score"].values))

        k = _anova_extremizing_factor(
            group["sentiment_score"].values, group["user_id"].values, k_min, k_max
        )

        logit_p = np.log(p_pooled / (1 - p_pooled))
        logit_p_extreme = k * logit_p
        p_extreme = 1.0 / (1.0 + np.exp(-logit_p_extreme))

        # Map [0,1] probability back to a signed [-1, 1] score so this can
        # be dropped straight into the existing Sentiment_Mean feature slot
        # that regime_pipeline.py / utils.fit_hierarchical_bayes expects.
        signed_extreme_score = 2 * p_extreme - 1

        records.append({
            "ticker": ticker,
            "date": date,
            "n_skilled": n_skilled,
            "Sentiment_Mean_SkillWeighted": weighted_mean,
            "extremizing_k": k,
            "Sentiment_Extremized": signed_extreme_score,
        })

    return pd.DataFrame(records).sort_values(["ticker", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Real data loading -- reads data_engine.py's output directly, no re-fetch
# ---------------------------------------------------------------------------

def load_ticker_messages(ticker: str, processed_dir: str = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """
    Reads data/processed/{ticker}.csv (data_engine.py's output) and returns
    it reshaped to the ['user_id', 'ticker', 'created_at', 'sentiment_score']
    schema this module expects.

    data_engine.py applies NO usecols filter when reading from S3, so every
    raw column -- including user_id -- should already be on disk. This
    function auto-detects column names defensively and fails loudly with a
    clear message if user_id is missing, rather than silently proceeding
    without skill information.
    """
    path = os.path.join(processed_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run data_engine.py first (Stage 1) -- "
            f"this stage reads its output directly, it does not hit S3."
        )

    df = pd.read_csv(path)
    print(f"[{ticker}] Loaded {len(df):,} rows. Columns found: {df.columns.tolist()}")

    if "user_id" not in df.columns:
        raise KeyError(
            f"'user_id' not found in {path}. Columns present: {df.columns.tolist()}.\n"
            f"data_engine.py should have written it through untouched (no usecols "
            f"filter is applied on the S3 read) -- if it's genuinely absent from "
            f"the raw Stocktwits-NYU rows for this file, skill weighting cannot "
            f"be computed for this ticker without a different data source."
        )

    # Best-effort mapping for the remaining expected columns, since exact
    # names can vary slightly across data_engine.py revisions.
    created_col = "created_at" if "created_at" in df.columns else next(
        (c for c in df.columns if "date" in c.lower() or "created" in c.lower()), None
    )
    sentiment_col = "sentiment" if "sentiment" in df.columns else next(
        (c for c in df.columns if "sentiment" in c.lower()), None
    )
    if created_col is None or sentiment_col is None:
        raise KeyError(
            f"Could not identify created_at/sentiment columns in {path}. "
            f"Columns present: {df.columns.tolist()}. Update load_ticker_messages() "
            f"with the exact column names once confirmed."
        )

    out = pd.DataFrame({
        "user_id": df["user_id"],
        "ticker": ticker,
        "created_at": df[created_col],
        "sentiment_score": pd.to_numeric(df[sentiment_col], errors="coerce"),
    }).dropna(subset=["sentiment_score"])

    print(f"[{ticker}] {out['user_id'].nunique():,} unique users, "
          f"{len(out):,} scoreable messages after cleaning.")
    return out


def build_forward_returns(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    max_retries: int = 3,
    retry_wait_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    Pulls Close prices via yfinance (same source as utils.py) and computes
    the forward log return over `horizon_days`, used to score whether a
    message's directional call was a hit.

    Retries on transient failures (rate limiting, momentary "possibly
    delisted" responses that are really just a dropped API call) before
    giving up -- run_ticker_pipeline() still refuses to write output if
    all retries return a suspiciously small result.

    Returns columns: ['ticker', 'date', 'forward_return']
    """
    yf_ticker = "META" if ticker == "FB" else ticker
    extended_end = (pd.to_datetime(end_date) + pd.Timedelta(days=horizon_days + 5)).strftime("%Y-%m-%d")

    market_data = pd.DataFrame()
    for attempt in range(1, max_retries + 1):
        market_data = yf.download(yf_ticker, start=start_date, end=extended_end, progress=False)
        if len(market_data) > 500:  # sane result, stop retrying
            break
        print(f"[{ticker}] yfinance returned {len(market_data)} rows on attempt "
              f"{attempt}/{max_retries} -- retrying in {retry_wait_seconds}s..." if attempt < max_retries
              else f"[{ticker}] yfinance still returning only {len(market_data)} rows after "
                   f"{max_retries} attempts.")
        if attempt < max_retries:
            time.sleep(retry_wait_seconds)

    if isinstance(market_data.columns, pd.MultiIndex):
        market_data.columns = market_data.columns.get_level_values(0)
    market_data.index = pd.to_datetime(market_data.index)

    close = market_data["Close"] if "Close" in market_data.columns else pd.Series(dtype=float)
    forward_log_ret = np.log(close.shift(-horizon_days) / close)

    df_fwd = pd.DataFrame({
        "ticker": ticker,
        "date": market_data.index,
        "forward_return": forward_log_ret.values,
    }).dropna(subset=["forward_return"])

    df_fwd = df_fwd[(df_fwd["date"] >= start_date) & (df_fwd["date"] <= end_date)]
    return df_fwd.reset_index(drop=True)


def run_ticker_pipeline(
    ticker: str,
    processed_dir: str = PROCESSED_DATA_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    min_track_record: int = 20,
    embargo_days: int = 1,
    base_rate_window: int = None,
) -> pd.DataFrame:
    """
    Full Stage 1.5 run for one ticker: load messages -> build forward
    returns -> causal sign test -> ANOVA extremizing. Writes
    data/processed/{ticker}_skill_extremized.csv and also returns it.

    base_rate_window : None uses the full-history expanding base rate as
        the sign test's null (default). Pass e.g. 60 to use a rolling
        60-trading-day window instead -- run both and compare
        pct_of_fdr_skilled_calls_that_are_bullish in the printed
        diagnostic; a large drop under the rolling window indicates the
        expanding-window result was substantially driven by local/short-
        -horizon momentum rather than individual forecasting skill.
    """
    print(f"\n{'='*70}\nSTAGE 1.5 -- SKILL WEIGHTING: {ticker}\n{'='*70}")

    df_messages = load_ticker_messages(ticker, processed_dir)
    df_messages = df_messages[
        (pd.to_datetime(df_messages["created_at"]) >= start_date)
        & (pd.to_datetime(df_messages["created_at"]) <= end_date)
    ]

    print(f"[{ticker}] Pulling forward returns from Yahoo Finance...")
    df_fwd = build_forward_returns(ticker, start_date, end_date)
    print(f"[{ticker}] {len(df_fwd):,} trading days with forward returns.")

    # Guard against a transient Yahoo Finance failure (rate limiting, a
    # blank/delisted-style response, momentary API hiccup) silently
    # proceeding with near-empty data and overwriting a previous GOOD
    # output file with an empty one. Refuse to write in that case instead.
    MIN_EXPECTED_TRADING_DAYS = 500  # well below any real full-window count; just a sanity floor
    if len(df_fwd) < MIN_EXPECTED_TRADING_DAYS:
        raise RuntimeError(
            f"[{ticker}] Only {len(df_fwd)} trading days returned from Yahoo Finance "
            f"(expected roughly {pd.bdate_range(start_date, end_date).size:,} for "
            f"{start_date} to {end_date}). This looks like a transient API failure, "
            f"not a real data gap -- refusing to overwrite "
            f"{ticker}_skill_extremized.csv with near-empty output. Re-run this ticker "
            f"on its own once the Yahoo Finance call succeeds."
        )

    print(f"[{ticker}] Building causal forecaster track record "
          f"(min_track_record={min_track_record}, embargo_days={embargo_days}, "
          f"base_rate_window={base_rate_window or 'expanding/full-history'})...")
    df_track = build_forecaster_track_record(
        df_messages, df_fwd, min_track_record=min_track_record, embargo_days=embargo_days,
        base_rate_window=base_rate_window,
    )
    df_track = benjamini_hochberg_correction(df_track, q=0.05)

    report = summarize_skill_classification(df_track)
    print(f"\n[{ticker}] --- SKILL CLASSIFICATION DIAGNOSTIC ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  (Sanity check: overall_unconditional_hit_rate should sit near 0.50.")
    print(f"   pct_distinct_users_skilled_fdr should be closer to the ~3% figure")
    print(f"   reported in Gomez-Cram et al. than to pct_..._raw.)")
    print(f"[{ticker}] --- END DIAGNOSTIC ---\n")

    print(f"[{ticker}] Applying ANOVA-style extremizing to pooled FDR-corrected skilled signal...")
    df_track_for_extremizing = df_track.drop(columns=["is_skilled"]).rename(
        columns={"is_skilled_fdr": "is_skilled"}
    )
    df_daily = extremize_daily_signal(df_track_for_extremizing)
    print(f"[{ticker}] {len(df_daily):,} daily extremized signal rows produced.")

    out_path = os.path.join(processed_dir, f"{ticker}_skill_extremized.csv")
    df_daily.to_csv(out_path, index=False)
    print(f"[{ticker}] Saved: {out_path}")

    return df_daily


# ---------------------------------------------------------------------------
# Quick self-test on synthetic data (no network, no real S3 pull required)
# ---------------------------------------------------------------------------

def _run_self_test():
    rng = np.random.default_rng(42)

    n_users = 40
    n_days = 120
    tickers = ["AAPL"]
    dates = pd.bdate_range("2019-01-01", periods=n_days)

    # Simulate: 3 users are genuinely skilled (65% hit rate), the rest are noise (50%).
    true_skill = {f"user_{i}": (0.65 if i < 3 else 0.50) for i in range(n_users)}

    # Simulate a forward-return sign series first (ground truth to score against).
    fwd_dir = rng.choice([-1, 1], size=n_days)
    df_hits = pd.DataFrame({
        "ticker": "AAPL",
        "date": dates,
        "forward_return": fwd_dir * rng.uniform(0.001, 0.02, size=n_days),
    })

    messages = []
    for d_idx, date in enumerate(dates):
        for u_idx in range(n_users):
            user = f"user_{u_idx}"
            if rng.random() < 0.7:  # not every user posts every day
                p_correct = true_skill[user]
                correct = rng.random() < p_correct
                direction = fwd_dir[d_idx] if correct else -fwd_dir[d_idx]
                messages.append({
                    "user_id": user,
                    "ticker": "AAPL",
                    "created_at": date,
                    "sentiment_score": float(direction) * rng.uniform(0.5, 1.0),
                })
    df_messages = pd.DataFrame(messages)

    print(f"Simulated {len(df_messages)} messages across {n_users} users, {n_days} days.")

    track = build_forecaster_track_record(df_messages, df_hits, min_track_record=20, embargo_days=1)
    print(f"\nTrack-record rows: {len(track)} "
          f"(rows with n_prior_calls >= 20: {(track['n_prior_calls'] >= 20).sum()})")

    classified = track.dropna(subset=["p_value_skill"])
    skilled_rate_by_user = classified.groupby("user_id")["is_skilled"].mean().sort_values(ascending=False)
    print("\nFraction of dates each user was classified 'skilled' (top 8):")
    print(skilled_rate_by_user.head(8))
    print("\n(Compare to ground truth: user_0/1/2 should dominate this list -- they're the")
    print(" only genuinely 65%-skilled simulated users; everyone else is 50/50 noise.)")

    daily_signal = extremize_daily_signal(track)
    print(f"\nExtremized daily signal rows: {len(daily_signal)}")
    print(daily_signal.head(10).to_string(index=False))

    print("\nSelf-test complete -- no errors. Ready to wire into regime_pipeline.py.")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _run_self_test()
    else:
        # --rolling-window=60 uses a rolling base-rate null instead of the
        # default full-history expanding one. Run once without this flag,
        # then once with it, and compare pct_of_fdr_skilled_calls_that_are_bullish
        # in the printed diagnostic between the two runs.
        rolling_window = None
        for arg in sys.argv:
            if arg.startswith("--rolling-window="):
                rolling_window = int(arg.split("=")[1])

        for ticker in TARGET_UNIVERSE:
            try:
                run_ticker_pipeline(ticker, base_rate_window=rolling_window)
            except Exception as e:
                print(f"[{ticker}] FAILED: {e}")
                continue

        print(f"\n{'='*70}\nSTAGE 1.5 COMPLETE FOR ALL TICKERS\n{'='*70}")
        print("Run with --demo instead to see the synthetic self-test:")
        print("  python src/skill_weighting.py --demo")
        print("Run with --rolling-window=60 to compare against a rolling base rate:")
        print("  python src/skill_weighting.py --rolling-window=60")