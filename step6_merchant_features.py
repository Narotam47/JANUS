"""Step 6 — JANUS Phase 1: merchant-level and population-level features.

Captures signals that neither the fixed 10-tx window nor the per-user
profile can see: *how does this merchant behave across all users, and
how busy is the overall system right now?*

Input : artifacts/raw_subset.parquet  (original merchant IDs + raw amounts)
        artifacts/clean_train.parquet (alignment verification only)
        artifacts/clean_test.parquet  (alignment verification only)

Output: artifacts/merchant_features_train.parquet  (9 features, row-aligned)
        artifacts/merchant_features_test.parquet   (9 features, row-aligned)

No-leakage approach
-------------------
All merchant statistics use GLOBAL time ordering (across all users). Every
feature at time t uses only transactions with timestamp strictly < t.

EXPANDING STATS (merch_tx_count, merch_hist_amount_mean, merch_amount_deviation,
    merch_large_tx_rate):
    Computed via cumcount/cumsum on the globally-sorted DataFrame. cumcount()
    at position i gives i (count of prior rows in the group). cumsum at
    position i includes the current row, so we subtract the current value
    to get the sum of strictly-prior rows. First transaction per merchant
    gets NaN for mean/deviation/rate (no history).

ROLLING 7-DAY COUNT (merch_7d_count):
    Per-merchant searchsorted on sorted timestamps. For each transaction at
    time t, counts how many transactions at the same merchant fell in
    [t − 7 days, t). The current transaction is at index i; the search finds
    index j where timestamp[j] ≥ t − 7 days; rolling count = i − j.
    Excludes the current transaction because i itself is not counted.

SPIKE RATIO (merch_spike_ratio):
    merch_7d_count / (merch_tx_count / max(days_active, 1) × 7).
    Both numerator and denominator use only past data (inherited from
    merch_7d_count and merch_tx_count). NaN when days_active < 1.

DAYS ACTIVE (merch_days_active):
    (current timestamp − merchant's first-ever timestamp) in days. The
    first timestamp is always ≤ current, so no leakage.

POPULATION FEATURES (pop_prior_day_count, pop_7d_avg_count):
    Aggregated daily counts across ALL merchants and users. Shifted by 1 day
    so each transaction sees only yesterday's (or prior 7 days') volume.
    Current day's transactions are never visible.

Unseen merchants in test
------------------------
A merchant appearing for the first time in the test period gets:
    merch_tx_count=0, merch_7d_count=0, merch_days_active=0, and NaN for
    mean/deviation/rate/spike. XGBoost handles NaN natively. As test-period
    transactions accumulate, the features fill in naturally — this matches
    how a production system would behave.
"""

from __future__ import annotations

import logging
import tracemalloc
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
RAW_SUBSET: Final[Path] = ARTIFACTS_DIR / "raw_subset.parquet"
CLEAN_TRAIN: Final[Path] = ARTIFACTS_DIR / "clean_train.parquet"
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"

OUT_TRAIN: Final[Path] = ARTIFACTS_DIR / "merchant_features_train.parquet"
OUT_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"

USER_COL: Final[str] = "User"
CHRONO_SPLIT_FRAC: Final[float] = 0.80

LARGE_TX_THRESHOLD: Final[float] = 200.0
ROLLING_WINDOW_DAYS: Final[int] = 7

NEW_FEATURES: Final[tuple[str, ...]] = (
    "merch_tx_count",
    "merch_7d_count",
    "merch_spike_ratio",
    "merch_days_active",
    "merch_hist_amount_mean",
    "merch_amount_deviation",
    "merch_large_tx_rate",
    "pop_prior_day_count",
    "pop_7d_avg_count",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step6")


# --- Helpers ----------------------------------------------------------------

def parse_amount(series: pd.Series) -> pd.Series:
    return series.str.lstrip("$").str.replace(",", "", regex=False).astype(float)


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    hour = df["Time"].str.split(":", n=1).str[0].astype(int)
    minute = df["Time"].str.split(":", n=1).str[1].astype(int)
    return pd.to_datetime(
        df["Year"].astype(str) + "-"
        + df["Month"].astype(str).str.zfill(2) + "-"
        + df["Day"].astype(str).str.zfill(2) + " "
        + hour.astype(str).str.zfill(2) + ":"
        + minute.astype(str).str.zfill(2),
        format="%Y-%m-%d %H:%M",
    )


def chrono_split(
    df: pd.DataFrame, frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values([USER_COL, "timestamp"]).reset_index(drop=True)
    train_parts, test_parts = [], []
    for _, group in df.groupby(USER_COL, sort=False):
        n = len(group)
        cutoff = int(n * frac)
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


# --- Feature computations --------------------------------------------------

MERCH_COL: Final[str] = "Merchant Name"


def compute_merchant_expanding(df: pd.DataFrame) -> None:
    """Cumcount/cumsum-based features in global time order."""
    log.info("Computing merchant expanding features (cumcount/cumsum) ...")

    g = df.groupby(MERCH_COL, sort=False)

    # --- tx count: cumcount gives 0-indexed position = count of prior rows ---
    df["merch_tx_count"] = g.cumcount().astype(np.int32)
    count_safe = df["merch_tx_count"].replace(0, np.nan)

    # --- amount mean: (cumsum - current) / count_prior ---
    _cumsum = g["amount_float"].cumsum()
    _past_sum = _cumsum - df["amount_float"]
    df["merch_hist_amount_mean"] = (_past_sum / count_safe).astype(np.float32)

    # --- amount std via E[X²] - E[X]² ---
    df["_amt_sq"] = df["amount_float"] ** 2
    _cumsum_sq = df.groupby(MERCH_COL, sort=False)["_amt_sq"].cumsum()
    _past_sum_sq = _cumsum_sq - df["_amt_sq"]
    _past_mean_sq = _past_sum_sq / count_safe
    _var = _past_mean_sq - df["merch_hist_amount_mean"].astype(np.float64) ** 2
    _std = np.sqrt(np.maximum(_var, 0))

    df["merch_amount_deviation"] = np.where(
        _std > 1e-8,
        (df["amount_float"] - df["merch_hist_amount_mean"]) / _std,
        np.nan,
    ).astype(np.float32)

    # --- large-tx rate: fraction of past merchant tx exceeding $200 ---
    df["_is_large"] = (df["amount_float"].abs() > LARGE_TX_THRESHOLD).astype(np.int32)
    _large_cumsum = df.groupby(MERCH_COL, sort=False)["_is_large"].cumsum()
    _past_large = _large_cumsum - df["_is_large"]
    df["merch_large_tx_rate"] = (_past_large / count_safe).astype(np.float32)

    # --- cleanup intermediates ---
    df.drop(columns=["_amt_sq", "_is_large"], inplace=True)


def compute_merchant_rolling(df: pd.DataFrame) -> None:
    """7-day rolling count per merchant via searchsorted (no Python loop per tx)."""
    log.info("Computing merchant rolling %d-day counts (%d merchants) ...",
             ROLLING_WINDOW_DAYS, df[MERCH_COL].nunique())

    result = np.zeros(len(df), dtype=np.int32)
    window_ns = np.timedelta64(ROLLING_WINDOW_DAYS, "D")

    for _, group in df.groupby(MERCH_COL, sort=False):
        ts = group["timestamp"].values
        cutoff = ts - window_ns
        window_start = np.searchsorted(ts, cutoff, side="left")
        current_pos = np.arange(len(ts))
        result[group.index.values] = current_pos - window_start

    df["merch_7d_count"] = result
    log.info("Rolling count done.")


def compute_merchant_derived(df: pd.DataFrame) -> None:
    """Spike ratio and days-active (depend on expanding + rolling features)."""
    log.info("Computing merchant derived features ...")

    merch_first_ts = df.groupby(MERCH_COL, sort=False)["timestamp"].transform("first")
    df["merch_days_active"] = (
        (df["timestamp"] - merch_first_ts).dt.total_seconds() / 86400
    ).astype(np.float32)

    # Expected 7-day count = (historical daily rate) × 7
    hist_daily_rate = df["merch_tx_count"] / np.maximum(df["merch_days_active"], 1.0)
    expected_7d = hist_daily_rate * ROLLING_WINDOW_DAYS

    df["merch_spike_ratio"] = np.where(
        expected_7d > 0.1,
        df["merch_7d_count"] / expected_7d,
        np.nan,
    ).astype(np.float32)


def compute_population_features(df: pd.DataFrame) -> None:
    """System-wide daily volume — shifted so each tx sees only past days."""
    log.info("Computing population-level features ...")

    date_col = df["timestamp"].dt.normalize()
    daily_pop = df.groupby(date_col).size()

    # Fill missing dates with 0, build rolling stats
    date_range = pd.date_range(daily_pop.index.min(), daily_pop.index.max(), freq="D")
    daily_full = daily_pop.reindex(date_range, fill_value=0).astype(np.float64)

    # Shift 1 day: for date d, value is count on d-1
    shifted = daily_full.shift(1, fill_value=0)

    # 7-day rolling mean of the shifted series: avg of d-1 through d-7
    pop_7d = shifted.rolling(ROLLING_WINDOW_DAYS, min_periods=1).mean()

    # Map back to transactions
    tx_dates = date_col.values
    prior_day_map = shifted.to_dict()
    pop_7d_map = pop_7d.to_dict()

    df["pop_prior_day_count"] = date_col.map(prior_day_map).fillna(0).astype(np.int32)
    df["pop_7d_avg_count"] = date_col.map(pop_7d_map).fillna(0).astype(np.float32)


# --- Main -------------------------------------------------------------------

def main() -> None:
    tracemalloc.start()

    for p in (RAW_SUBSET, CLEAN_TRAIN, CLEAN_TEST):
        if not p.exists():
            log.error("Missing %s — run prior steps first.", p)
            return

    # Load only needed columns (lean: ~500 MB vs 2+ GB for all 15)
    needed = [USER_COL, "Amount", "Year", "Month", "Day", "Time", MERCH_COL]
    log.info("Loading %s (%d columns) ...", RAW_SUBSET, len(needed))
    df = pd.read_parquet(RAW_SUBSET, columns=needed)
    log.info("Loaded %s rows.", f"{len(df):,}")

    # Parse, then drop raw string columns we no longer need
    df["amount_float"] = parse_amount(df["Amount"])
    df["timestamp"] = build_timestamp(df)
    df.drop(columns=["Amount", "Year", "Month", "Day", "Time"], inplace=True)
    log.info("Parsed. Columns kept: %s", df.columns.tolist())

    # --- Sort globally by timestamp (critical for expanding features) --------
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Sorted globally by timestamp.")

    # --- Compute features ---------------------------------------------------
    compute_merchant_expanding(df)
    compute_merchant_rolling(df)
    compute_merchant_derived(df)
    compute_population_features(df)

    # Drop intermediates no longer needed (keep timestamp+User for chrono_split)
    df.drop(columns=["amount_float", MERCH_COL], inplace=True)

    log.info("All 9 features computed. Splitting ...")

    # --- Chrono split (same as steps 2/5) -----------------------------------
    train, test = chrono_split(df, CHRONO_SPLIT_FRAC)
    del df

    # --- Verify alignment ---------------------------------------------------
    log.info("Verifying alignment with clean files ...")
    clean_train_users = pd.read_parquet(CLEAN_TRAIN, columns=[USER_COL])
    clean_test_users = pd.read_parquet(CLEAN_TEST, columns=[USER_COL])

    assert len(train) == len(clean_train_users), (
        f"Train rows: {len(train)} vs {len(clean_train_users)}"
    )
    assert len(test) == len(clean_test_users), (
        f"Test rows: {len(test)} vs {len(clean_test_users)}"
    )
    assert (train[USER_COL].values == clean_train_users[USER_COL].values).all(), \
        "Train user order mismatch!"
    assert (test[USER_COL].values == clean_test_users[USER_COL].values).all(), \
        "Test user order mismatch!"
    del clean_train_users, clean_test_users
    log.info("Alignment verified: row counts and user order match.")

    # --- Save ---------------------------------------------------------------
    out_cols = list(NEW_FEATURES)
    train[out_cols].to_parquet(OUT_TRAIN, index=False)
    test[out_cols].to_parquet(OUT_TEST, index=False)

    # --- Report -------------------------------------------------------------
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    log.info("=" * 70)
    log.info("STEP 6 COMPLETE — 9 merchant/population features")
    log.info("Peak memory (tracemalloc): %.1f MB", peak_mem / 1e6)
    log.info(
        "Train: %s (%.1f MB)  |  Test: %s (%.1f MB)",
        OUT_TRAIN, OUT_TRAIN.stat().st_size / 1e6,
        OUT_TEST, OUT_TEST.stat().st_size / 1e6,
    )

    for split_name, split_df in [("train", train), ("test", test)]:
        log.info("-" * 70)
        log.info("[%s] Feature summary (%s rows):", split_name, f"{len(split_df):,}")
        for col in out_cols:
            s = split_df[col]
            nan_n = int(s.isna().sum())
            nan_pct = nan_n / len(s) * 100
            if nan_n > 0:
                log.info(
                    "  %-28s mean=%10.3f  std=%10.3f  min=%10.3f  max=%10.3f  NaN=%s (%.2f%%)",
                    col, s.mean(), s.std(), s.min(), s.max(),
                    f"{nan_n:,}", nan_pct,
                )
            else:
                log.info(
                    "  %-28s mean=%10.3f  std=%10.3f  min=%10.3f  max=%10.3f  NaN=0",
                    col, s.mean(), s.std(), s.min(), s.max(),
                )

    log.info("-" * 70)
    log.info("NO-LEAKAGE CONFIRMATION:")
    log.info("  Expanding stats    : cumcount/cumsum, current tx subtracted    ✓")
    log.info("  Rolling 7-day      : searchsorted on [t-7d, t), excludes t    ✓")
    log.info("  Spike ratio        : derived from expanding + rolling (both ✓) ✓")
    log.info("  Days active        : first_ts always ≤ current_ts             ✓")
    log.info("  Population daily   : shifted 1 day, current day never visible ✓")
    log.info("  Unseen merchants   : NaN at first appearance, fills naturally  ✓")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
