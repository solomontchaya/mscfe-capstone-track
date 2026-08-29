"""
Momentum-Confound Diagnostic
============================

Checks whether Sentiment_Extremized's dramatic jump in the Table 4.1/4.2
ablation is real forecast signal, or whether it's substantially just an
encoding of the ticker's OWN trailing momentum -- a real possibility given
that the skill-classification sign test selected users based on whether
their historical calls correlated with THIS SAME ticker's own past returns,
and the flagged 'skilled' cohort turned out to be ~90%+ fixed-stance
(permabull/permabear), per the directional-consistency diagnostic.

Run this against your real data/processed/{ticker}_with_regimes.csv files
(produced by regime_pipeline.py, now containing both Sentiment_Mean and
Sentiment_Extremized after the merge_skill_extremized_signal() wiring).

Usage:
    python src/check_momentum_confound.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

TARGET_UNIVERSE = ["AAPL", "AMD", "SPY", "TSLA"]
TRAIN_END = "2019-12-31"  # match regime_models.py -- this is the window the ablation was fit on
MOMENTUM_WINDOWS = [5, 10, 20, 60]  # trading days


def check_ticker(ticker: str) -> dict:
    path = os.path.join(PROCESSED_DIR, f"{ticker}_with_regimes.csv")
    if not os.path.exists(path):
        print(f"[{ticker}] {path} not found -- run regime_pipeline.py first.")
        return {}

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Sentiment_Extremized" not in df.columns:
        print(f"[{ticker}] No Sentiment_Extremized column found -- "
              f"regime_pipeline.py may need re-running after skill_weighting.py.")
        return {}

    df = df.sort_index()
    df_train = df[df.index <= TRAIN_END].copy()  # match the exact window the ablation was fit on

    results = {"ticker": ticker}
    for window in MOMENTUM_WINDOWS:
        # Trailing (causal, uses only past log_ret) cumulative momentum as of
        # each date -- excludes today's own return via shift(1) so this isn't
        # trivially correlated with the CONTEMPORANEOUS log_ret target either.
        df_train[f"momentum_{window}d"] = (
            df_train["log_ret"].shift(1).rolling(window).sum()
        )

        valid = df_train.dropna(subset=[f"momentum_{window}d", "Sentiment_Extremized", "Sentiment_Mean"])
        if len(valid) < 30:
            continue

        r_extremized, p_extremized = stats.pearsonr(valid[f"momentum_{window}d"], valid["Sentiment_Extremized"])
        r_naive, p_naive = stats.pearsonr(valid[f"momentum_{window}d"], valid["Sentiment_Mean"])

        results[f"corr_extremized_vs_{window}d_momentum"] = r_extremized
        results[f"corr_naive_vs_{window}d_momentum"] = r_naive

    return results


if __name__ == "__main__":
    print("=" * 90)
    print("MOMENTUM-CONFOUND DIAGNOSTIC")
    print("Comparing correlation of Sentiment_Extremized vs. Sentiment_Mean")
    print("against the ticker's OWN trailing momentum (train window only, matching the ablation)")
    print("=" * 90)

    all_results = []
    for ticker in TARGET_UNIVERSE:
        r = check_ticker(ticker)
        if r:
            all_results.append(r)
            print(f"\n[{ticker}]")
            for window in MOMENTUM_WINDOWS:
                key_ext = f"corr_extremized_vs_{window}d_momentum"
                key_naive = f"corr_naive_vs_{window}d_momentum"
                if key_ext in r:
                    print(f"  {window:>3}d momentum -- corr(Sentiment_Extremized): {r[key_ext]:+.4f}  "
                          f"|  corr(Sentiment_Mean): {r[key_naive]:+.4f}")

    if all_results:
        df_summary = pd.DataFrame(all_results)
        out_path = os.path.join(BASE_DIR, "data", "momentum_confound_diagnostic.csv")
        df_summary.to_csv(out_path, index=False)
        print(f"\nSaved summary to {out_path}")

        print("\n" + "=" * 90)
        print("HOW TO READ THIS:")
        print("If corr(Sentiment_Extremized) is substantially higher than corr(Sentiment_Mean)")
        print("against the SAME momentum window, that's direct evidence the extremizing/skill-")
        print("selection step introduced a momentum-like structure that wasn't in the naive")
        print("signal -- i.e. the Table 4.1/4.2 ablation jump is likely momentum in disguise,")
        print("not new crowd information. If both are similarly low, the jump needs a different")
        print("explanation (worth revisiting the variance-compression hypothesis instead).")
        print("=" * 90)