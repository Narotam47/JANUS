"""Step 5 — JANUS Phase 1: per-user historical profile features.

Computes features from the full per-user transaction history — signals that
a fixed 10-transaction window cannot capture: lifetime spending profile,
merchant familiarity, and year-over-year patterns.

Input : artifacts/raw_subset.parquet   (original string values — needed for
        unscaled amounts and raw merchant IDs)
        artifacts/clean_train.parquet  (row-count verification only)
        artifacts/clean_test.parquet   (row-count verification only)

Output: artifacts/user_features_train.parquet  (14 new features, row-aligned with clean_train)
        artifacts/user_features_test.parquet   (14 new features, row-aligned with clean_test)

No-leakage approach per feature group
--------------------------------------
AMOUNT HISTORY (user_hist_mean, _std, _max, user_tx_count, user_amount_zscore,
    user_amount_to_max_ratio):
    Computed via shift(1).expanding() — at position i, the expanding window
    covers the SHIFTED series [NaN, amount[0], ..., amount[i-1]], so only
    transactions strictly before position i contribute. The current
    transaction's amount is never in its own history.

MERCHANT FAMILIARITY (merchant_user_count, mcc_user_count, is_new_merchant):
    Computed via groupby([User, Merchant]).cumcount(). cumcount() at position j
    within its group returns j (0-indexed), which equals the number of prior
    rows in that group. At the first occurrence, cumcount=0 (no prior visits).

SAME-PERIOD-LAST-YEAR (same_month_last_year_mean, amount_vs_last_year_ratio):
    For a transaction in (Year=Y, Month=M), looks up the user's average amount
    in (Year=Y-1, Month=M). The ENTIRE prior year is strictly in the past, so
    no leakage even when using whole-month aggregates. Transactions in the
    first year of data (no Y-1) get NaN — XGBoost handles this natively.

SEASONAL MARKERS (is_weekend, is_holiday_season, day_of_year):
    Derived from the transaction's own timestamp — no dependency on other
    rows, no leakage possible.
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

OUT_TRAIN: Final[Path] = ARTIFACTS_DIR / "user_features_train.parquet"
OUT_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"

USER_COL: Final[str] = "User"
CHRONO_SPLIT_FRAC: Final[float] = 0.80

NEW_FEATURES: Final[tuple[str, ...]] = (
    "user_hist_mean",
    "user_hist_std",
    "user_hist_max",
    "user_tx_count",
    "user_amount_zscore",
    "user_amount_to_max_ratio",
    "merchant_user_count",
    "mcc_user_count",
    "is_new_merchant",
    "same_month_last_year_mean",
    "amount_vs_last_year_ratio",
    "is_weekend",
    "is_holiday_season",
    "day_of_year",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step5")


# --- Helpers ----------------------------------------------------------------

def parse_amount(series: pd.Series) -> pd.Series:
    return series.str.lstrip("$").str.replace(",", "", regex=False).astype(float)


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Identical to step 2 — same parse, same sort order."""
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
    """Identical to step 2 — sort by (User, timestamp), 80/20 per user."""
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

def compute_amount_features(df: pd.DataFrame) -> None:
    """Expanding amount stats. shift(1) excludes the current transaction."""
    log.info("Computing per-user expanding amount features ...")

    def _per_user(group: pd.DataFrame) -> pd.DataFrame:
        amt = group["amount_float"]
        shifted = amt.shift(1)
        return pd.DataFrame(
            {
                "user_hist_mean": shifted.expanding(min_periods=1).mean(),
                "user_hist_std": shifted.expanding(min_periods=2).std(),
                "user_hist_max": shifted.expanding(min_periods=1).max(),
                "user_tx_count": shifted.expanding().count(),
            },
            index=group.index,
        )

    feats = df.groupby(USER_COL, sort=False, group_keys=False).apply(
        _per_user, include_groups=False,
    )
    for col in feats.columns:
        df[col] = feats[col]

    df["user_amount_zscore"] = (
        (df["amount_float"] - df["user_hist_mean"]) / df["user_hist_std"]
    )
    df["user_amount_to_max_ratio"] = np.where(
        df["user_hist_max"].abs() > 1e-8,
        df["amount_float"] / df["user_hist_max"],
        np.nan,
    )

    for col in ("user_hist_mean", "user_hist_std", "user_hist_max",
                "user_amount_zscore", "user_amount_to_max_ratio"):
        df[col] = df[col].astype(np.float32)
    df["user_tx_count"] = df["user_tx_count"].astype(np.int32)


def compute_merchant_features(df: pd.DataFrame) -> None:
    """Per-user merchant/MCC familiarity via cumcount (no-leakage by construction)."""
    log.info("Computing merchant familiarity features ...")
    df["merchant_user_count"] = (
        df.groupby([USER_COL, "Merchant Name"], sort=False)
        .cumcount()
        .astype(np.int32)
    )
    df["mcc_user_count"] = (
        df.groupby([USER_COL, "MCC"], sort=False)
        .cumcount()
        .astype(np.int32)
    )
    df["is_new_merchant"] = (df["merchant_user_count"] == 0).astype(np.int8)


def compute_same_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same-month-last-year comparison.

    No leakage: looks up (Year-1, Month), which is a complete year in the past.
    Returns the (possibly new) DataFrame — merge may create a copy.
    """
    log.info("Computing same-period-last-year features ...")
    year = df["Year"].astype(int)
    month = df["Month"].astype(int)

    df["_yr"] = year
    df["_mo"] = month

    monthly_avg = (
        df.groupby([USER_COL, "_yr", "_mo"], sort=False)["amount_float"]
        .mean()
        .reset_index()
        .rename(columns={"amount_float": "_ly_mean", "_yr": "_ly_yr", "_mo": "_ly_mo"})
    )

    df["_ly_yr"] = year - 1
    df["_ly_mo"] = month
    df = df.merge(monthly_avg, on=[USER_COL, "_ly_yr", "_ly_mo"], how="left")

    df["same_month_last_year_mean"] = df.pop("_ly_mean").astype(np.float32)
    df["amount_vs_last_year_ratio"] = np.where(
        df["same_month_last_year_mean"].abs() > 1e-8,
        df["amount_float"] / df["same_month_last_year_mean"],
        np.nan,
    ).astype(np.float32)

    df.drop(columns=["_yr", "_mo", "_ly_yr", "_ly_mo"], inplace=True)
    return df


def compute_seasonal_features(df: pd.DataFrame) -> None:
    """Timestamp-derived markers — no temporal dependency, no leakage."""
    log.info("Computing seasonal markers ...")
    dow = df["timestamp"].dt.dayofweek
    month = df["timestamp"].dt.month

    df["is_weekend"] = (dow >= 5).astype(np.int8)
    df["is_holiday_season"] = month.isin([11, 12]).astype(np.int8)
    df["day_of_year"] = df["timestamp"].dt.dayofyear.astype(np.int16)


# --- Main -------------------------------------------------------------------

def main() -> None:
    tracemalloc.start()

    for p in (RAW_SUBSET, CLEAN_TRAIN, CLEAN_TEST):
        if not p.exists():
            log.error("Missing %s — run prior steps first.", p)
            return

    # Load only the columns we need (saves ~40% RAM vs loading all 15)
    needed = [USER_COL, "Amount", "Year", "Month", "Day", "Time",
              "Merchant Name", "MCC"]
    log.info("Loading %s (columns: %s) ...", RAW_SUBSET, needed)
    df = pd.read_parquet(RAW_SUBSET, columns=needed)
    log.info("Loaded %s rows.", f"{len(df):,}")

    # Parse
    df["amount_float"] = parse_amount(df["Amount"])
    df["timestamp"] = build_timestamp(df)
    df = df.sort_values([USER_COL, "timestamp"]).reset_index(drop=True)
    log.info("Parsed and sorted.")

    # --- Compute features ---------------------------------------------------
    compute_amount_features(df)
    compute_merchant_features(df)
    df = compute_same_period_features(df)
    compute_seasonal_features(df)

    log.info("All 14 features computed. Splitting ...")

    # --- Split (same logic as step 2) ---------------------------------------
    train, test = chrono_split(df, CHRONO_SPLIT_FRAC)
    del df

    # --- Alignment verification ---------------------------------------------
    log.info("Verifying row alignment with clean files ...")
    clean_train_users = pd.read_parquet(CLEAN_TRAIN, columns=[USER_COL])
    clean_test_users = pd.read_parquet(CLEAN_TEST, columns=[USER_COL])

    assert len(train) == len(clean_train_users), (
        f"Train row count mismatch: {len(train)} vs {len(clean_train_users)}"
    )
    assert len(test) == len(clean_test_users), (
        f"Test row count mismatch: {len(test)} vs {len(clean_test_users)}"
    )

    train_user_counts = train[USER_COL].value_counts().sort_index()
    clean_train_counts = clean_train_users[USER_COL].value_counts().sort_index()
    assert (train_user_counts == clean_train_counts).all(), "Per-user train counts differ!"

    test_user_counts = test[USER_COL].value_counts().sort_index()
    clean_test_counts = clean_test_users[USER_COL].value_counts().sort_index()
    assert (test_user_counts == clean_test_counts).all(), "Per-user test counts differ!"

    assert (train[USER_COL].values == clean_train_users[USER_COL].values).all(), \
        "Train user order differs — rows are not aligned!"
    assert (test[USER_COL].values == clean_test_users[USER_COL].values).all(), \
        "Test user order differs — rows are not aligned!"

    del clean_train_users, clean_test_users
    log.info("Alignment verified: row counts, per-user counts, and row order all match.")

    # --- Select and save ----------------------------------------------------
    out_cols = list(NEW_FEATURES)
    train[out_cols].to_parquet(OUT_TRAIN, index=False)
    test[out_cols].to_parquet(OUT_TEST, index=False)

    # --- Report -------------------------------------------------------------
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    log.info("=" * 70)
    log.info("STEP 5 COMPLETE — 14 user-profile features")
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
            if nan_n == len(s):
                log.info("  %-30s  all NaN", col)
            elif nan_n > 0:
                log.info(
                    "  %-30s  mean=%10.3f  std=%10.3f  min=%10.3f  max=%10.3f  NaN=%s (%.1f%%)",
                    col, s.mean(), s.std(), s.min(), s.max(),
                    f"{nan_n:,}", nan_pct,
                )
            else:
                log.info(
                    "  %-30s  mean=%10.3f  std=%10.3f  min=%10.3f  max=%10.3f  NaN=0",
                    col, s.mean(), s.std(), s.min(), s.max(),
                )

    log.info("-" * 70)
    log.info("NO-LEAKAGE CONFIRMATION:")
    log.info("  Amount history     : shift(1).expanding() — current tx excluded ✓")
    log.info("  Merchant familiarity: cumcount() — counts prior rows only    ✓")
    log.info("  Same-period-last-yr: (Year-1, Month) lookup — full year past ✓")
    log.info("  Seasonal markers   : timestamp-derived, no other-row deps    ✓")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
