"""Step 7 — JANUS Phase 1: first-occurrence and novelty flags.

Uncertainty/novelty signals — these flag when a transaction is unlike
anything the user has done before.  They feed both the fraud model and
the confidence score in step 10.

Input : artifacts/raw_subset.parquet  (original IDs + raw amounts)
        artifacts/clean_train.parquet (alignment verification only)
        artifacts/clean_test.parquet  (alignment verification only)

Output: artifacts/flags_train.parquet  (10 features, row-aligned)
        artifacts/flags_test.parquet   (10 features, row-aligned)

No-leakage approach
-------------------
All features use ONLY transactions strictly before the current one.

AMOUNT NOVELTY (is_largest_ever, is_2x_hist_max, is_5x_hist_max):
    Uses shift(1).expanding().max() per user — the expanding window at
    position i covers [amount[0], ..., amount[i-1]].  The current
    transaction is never in its own comparison set.  First tx per user
    has no history → NaN.

FIRST-OCCURRENCE FLAGS (first_mcc, first_city_state, first_channel,
    first_tx_this_month, first_tx_this_year):
    Computed via groupby cumcount().  cumcount() at position j within its
    group returns j (0-indexed) = count of prior rows in that group.
    At the first occurrence, cumcount=0 (no prior visits) → flag=1.
    Always defined (no NaN).

DORMANCY (days_since_last_tx, is_dormant):
    Gap computed via diff() per user.  Dormancy threshold uses
    shift(1).expanding() on the gap series, so the current gap is never
    in its own mean/std computation.  Flag fires when the current gap
    exceeds mean + 2×std of all prior gaps.  NaN until 3+ transactions
    (need ≥2 prior gaps for a meaningful std).
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

OUT_TRAIN: Final[Path] = ARTIFACTS_DIR / "flags_train.parquet"
OUT_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

USER_COL: Final[str] = "User"
CHRONO_SPLIT_FRAC: Final[float] = 0.80

NEW_FEATURES: Final[tuple[str, ...]] = (
    "is_largest_ever",
    "is_2x_hist_max",
    "is_5x_hist_max",
    "first_mcc",
    "first_city_state",
    "first_channel",
    "days_since_last_tx",
    "is_dormant",
    "first_tx_this_month",
    "first_tx_this_year",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step7")


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

def compute_first_occurrence_flags(df: pd.DataFrame) -> None:
    """Cumcount-based first-occurrence flags — always defined (no NaN)."""
    log.info("Computing first-occurrence flags (cumcount) ...")

    df["first_mcc"] = (
        df.groupby([USER_COL, "MCC"], sort=False).cumcount() == 0
    ).astype(np.int8)

    df["_cs"] = df["Merchant City"].fillna("") + "|" + df["Merchant State"].fillna("")
    df["first_city_state"] = (
        df.groupby([USER_COL, "_cs"], sort=False).cumcount() == 0
    ).astype(np.int8)
    df.drop(columns=["_cs"], inplace=True)

    df["first_channel"] = (
        df.groupby([USER_COL, "Use Chip"], sort=False).cumcount() == 0
    ).astype(np.int8)

    _year = df["timestamp"].dt.year
    _month = df["timestamp"].dt.month

    df["first_tx_this_month"] = (
        df.groupby([USER_COL, _year, _month], sort=False).cumcount() == 0
    ).astype(np.int8)

    df["first_tx_this_year"] = (
        df.groupby([USER_COL, _year], sort=False).cumcount() == 0
    ).astype(np.int8)

    log.info("  first_mcc / first_city_state / first_channel / "
             "first_tx_this_month / first_tx_this_year — done.")


def compute_amount_novelty(df: pd.DataFrame) -> None:
    """is_largest_ever, is_2x_hist_max, is_5x_hist_max via shift(1).expanding()."""
    log.info("Computing amount novelty flags (per-user expanding max) ...")

    def _per_user(group: pd.DataFrame) -> pd.DataFrame:
        amt = group["amount_float"]
        hist_max = amt.shift(1).expanding(min_periods=1).max()
        has_history = hist_max.notna()
        return pd.DataFrame(
            {
                "is_largest_ever": np.where(
                    has_history, (amt > hist_max).astype(float), np.nan
                ),
                "is_2x_hist_max": np.where(
                    has_history, (amt > 2 * hist_max).astype(float), np.nan
                ),
                "is_5x_hist_max": np.where(
                    has_history, (amt > 5 * hist_max).astype(float), np.nan
                ),
            },
            index=group.index,
        )

    feats = df.groupby(USER_COL, sort=False, group_keys=False).apply(
        _per_user, include_groups=False,
    )
    for col in ("is_largest_ever", "is_2x_hist_max", "is_5x_hist_max"):
        df[col] = feats[col].astype(np.float32)


def compute_dormancy(df: pd.DataFrame) -> None:
    """days_since_last_tx and is_dormant via expanding gap stats."""
    log.info("Computing dormancy features (per-user expanding gap stats) ...")

    def _per_user(group: pd.DataFrame) -> pd.DataFrame:
        ts = group["timestamp"]
        gap_days = ts.diff().dt.total_seconds() / 86400

        shifted_gap = gap_days.shift(1)
        gap_mean = shifted_gap.expanding(min_periods=1).mean()
        gap_std = shifted_gap.expanding(min_periods=2).std()

        threshold = gap_mean + 2 * gap_std
        is_dormant = np.where(
            gap_days.notna() & gap_std.notna(),
            (gap_days > threshold).astype(float),
            np.nan,
        )

        return pd.DataFrame(
            {
                "days_since_last_tx": gap_days,
                "is_dormant": is_dormant,
            },
            index=group.index,
        )

    feats = df.groupby(USER_COL, sort=False, group_keys=False).apply(
        _per_user, include_groups=False,
    )
    df["days_since_last_tx"] = feats["days_since_last_tx"].astype(np.float32)
    df["is_dormant"] = feats["is_dormant"].astype(np.float32)


# --- Main -------------------------------------------------------------------

def main() -> None:
    tracemalloc.start()

    for p in (RAW_SUBSET, CLEAN_TRAIN, CLEAN_TEST):
        if not p.exists():
            log.error("Missing %s — run prior steps first.", p)
            return

    needed = [USER_COL, "Amount", "Year", "Month", "Day", "Time",
              "MCC", "Merchant City", "Merchant State", "Use Chip", "Is Fraud?"]
    log.info("Loading %s (%d columns) ...", RAW_SUBSET, len(needed))
    df = pd.read_parquet(RAW_SUBSET, columns=needed)
    log.info("Loaded %s rows.", f"{len(df):,}")

    df["amount_float"] = parse_amount(df["Amount"])
    df["timestamp"] = build_timestamp(df)
    df["label"] = (df["Is Fraud?"] == "Yes").astype(np.int8)
    df.drop(columns=["Amount", "Year", "Month", "Day", "Time", "Is Fraud?"],
            inplace=True)

    df = df.sort_values([USER_COL, "timestamp"]).reset_index(drop=True)
    log.info("Sorted by [User, timestamp].")

    # --- Cumcount-based flags (while string columns are in memory) ----------
    compute_first_occurrence_flags(df)

    df.drop(columns=["MCC", "Merchant City", "Merchant State", "Use Chip"],
            inplace=True)
    log.info("Dropped categorical columns to free memory.")

    # --- Expanding-based features -------------------------------------------
    compute_amount_novelty(df)
    compute_dormancy(df)

    log.info("All 10 features computed. Splitting ...")

    # --- Chrono split -------------------------------------------------------
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
    log.info("STEP 7 COMPLETE — 10 first-occurrence and novelty flags")
    log.info("Peak memory (tracemalloc): %.1f MB", peak_mem / 1e6)
    log.info(
        "Train: %s (%.1f MB)  |  Test: %s (%.1f MB)",
        OUT_TRAIN, OUT_TRAIN.stat().st_size / 1e6,
        OUT_TEST, OUT_TEST.stat().st_size / 1e6,
    )

    for split_name, split_df in [("train", train), ("test", test)]:
        total_fraud = int(split_df["label"].sum())
        n_rows = len(split_df)
        base_pct = total_fraud / n_rows * 100
        log.info("-" * 70)
        log.info("[%s] %s rows, %s fraud (%.3f%%)",
                 split_name, f"{n_rows:,}", f"{total_fraud:,}", base_pct)
        log.info("  %-25s %10s %7s  %6s/%-8s %8s %8s %8s",
                 "flag", "fires", "fire%",
                 "fraud", "fired", "fraud%", "lift", "cover%")

        for col in out_cols:
            s = split_df[col]

            if col == "days_since_last_tx":
                nan_n = int(s.isna().sum())
                log.info(
                    "  %-25s [continuous]  mean=%.2f  std=%.2f  "
                    "p50=%.2f  p95=%.1f  max=%.0f  NaN=%s",
                    col, s.mean(), s.std(), s.median(),
                    s.quantile(0.95), s.max(), f"{nan_n:,}",
                )
                continue

            n_flagged = int((s == 1).sum())
            pct_flagged = n_flagged / n_rows * 100
            nan_n = int(s.isna().sum())

            if n_flagged > 0:
                fraud_in = int(split_df.loc[s == 1, "label"].sum())
                fraud_pct = fraud_in / n_flagged * 100
                lift = fraud_pct / base_pct if base_pct > 0 else 0
                cover = fraud_in / max(total_fraud, 1) * 100
            else:
                fraud_in, fraud_pct, lift, cover = 0, 0.0, 0.0, 0.0

            log.info(
                "  %-25s %10s %6.2f%%  %6d/%-8s %7.3f%% %7.1fx %7.1f%%  NaN=%s",
                col, f"{n_flagged:,}", pct_flagged,
                fraud_in, f"{n_flagged:,}", fraud_pct, lift, cover,
                f"{nan_n:,}",
            )

    log.info("-" * 70)
    log.info("NO-LEAKAGE CONFIRMATION:")
    log.info("  Amount novelty     : shift(1).expanding().max() — current excluded ✓")
    log.info("  First-occurrence   : cumcount() — counts prior rows only           ✓")
    log.info("  Dormancy gap stats : shift(1).expanding() on gap series            ✓")
    log.info("  All flags are per-user, computed in [User, timestamp] order        ✓")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
