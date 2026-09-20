"""Step 2 — JANUS Phase 0: preprocess the raw TabFormer subset into
model-ready features.

Input : artifacts/raw_subset.parquet  (step 1 output, all columns as strings)
Output: artifacts/clean_train.parquet
        artifacts/clean_test.parquet
        artifacts/scaler.pkl          (StandardScaler, fitted on train only)
        artifacts/encoders.pkl        (label encoders + category mappings)

Design decisions driven by the real data
-----------------------------------------
1. FRAUD IS 0.12% — extreme imbalance. This step does NOT upsample or
   downsample; that belongs in the modelling step where it can be tuned
   independently. But we preserve the natural rate in the chronological
   split and print the per-split balance so downstream steps can verify.

2. AMOUNT has negatives (refunds, min -$500, ~5% of rows). log1p fails
   on negatives. We use signed-log: sign(x) * log1p(|x|). This preserves
   the refund signal while compressing the heavy right tail.

3. ERRORS? is a multi-label comma-separated field with trailing commas
   (24 unique combos, 98.4% empty). We one-hot the 7 individual error
   types extracted by splitting on commas.

4. EMPTY STRING semantics (per column):
   - Merchant State / Zip: empty IFF Use Chip == "Online Transaction"
     (confirmed 100% correlation). Semantics: *not-applicable* for online
     purchases. We fill State with "ONLINE", Zip with "00000".
     The ~63k records with empty Zip but non-empty State (non-online) are
     genuinely missing; we fill those Zips with "MISSING".
   - Errors?: empty means no error occurred. We leave the 7 OHE columns
     as 0 — the semantics are already correct.
   - All other columns: no empty values observed in the subset.

5. HIGH-CARDINALITY categoricals:
   - Merchant Name (56k unique): hashed integer IDs in the raw data.
     Too high-cardinality for one-hot. We frequency-encode: each name →
     its log-frequency in the training set. Unseen names at test time → 0.
   - Merchant City (11k unique): same treatment as Merchant Name.
   - MCC (109 unique): low enough for label-encoding (ordinal int).
   - Merchant State (197 + "ONLINE"): label-encoded.
   - Zip (23k unique): too many for one-hot. Frequency-encoded like
     Merchant Name.
   - Use Chip (3 unique): one-hot encoded.

6. CHRONOLOGICAL SPLIT: sorted by (User, Year, Month, Day, Time).
   Last 20% of each user's history → test, first 80% → train.
   This prevents future-leakage that a random split would cause.

7. SCALER: StandardScaler fitted on train numeric columns only, then
   applied to both train and test. Saved as scaler.pkl.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
INPUT_PARQUET: Final[Path] = ARTIFACTS_DIR / "raw_subset.parquet"

OUTPUT_TRAIN: Final[Path] = ARTIFACTS_DIR / "clean_train.parquet"
OUTPUT_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"
SCALER_PATH: Final[Path] = ARTIFACTS_DIR / "scaler.pkl"
ENCODERS_PATH: Final[Path] = ARTIFACTS_DIR / "encoders.pkl"

LABEL_COL: Final[str] = "Is Fraud?"
USER_COL: Final[str] = "User"
CHRONO_SPLIT_FRAC: Final[float] = 0.80

ERROR_TYPES: Final[tuple[str, ...]] = (
    "Bad CVV", "Bad Card Number", "Bad Expiration",
    "Bad PIN", "Bad Zipcode", "Insufficient Balance", "Technical Glitch",
)

FREQ_ENCODE_COLS: Final[tuple[str, ...]] = ("Merchant Name", "Merchant City", "Zip")
LABEL_ENCODE_COLS: Final[tuple[str, ...]] = ("MCC", "Merchant State")
OHE_COLS: Final[tuple[str, ...]] = ("Use Chip",)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step2")


# --- Helpers ----------------------------------------------------------------

def signed_log1p(series: pd.Series) -> pd.Series:
    """sign(x) * log1p(|x|) — handles negatives (refunds) safely."""
    return np.sign(series) * np.log1p(np.abs(series))


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


def hours_since_last(df: pd.DataFrame) -> pd.Series:
    """Per-user hours since previous transaction. First transaction → 0."""
    delta = df.groupby(USER_COL)["timestamp"].diff()
    return delta.dt.total_seconds().div(3600).fillna(0.0)


def explode_errors(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot the 7 individual error types from the comma-separated Errors? field."""
    raw = df["Errors?"].fillna("")
    for err in ERROR_TYPES:
        col_name = "err_" + err.lower().replace(" ", "_")
        df[col_name] = raw.str.contains(err, regex=False).astype(np.int8)
    return df


def clean_zip(series: pd.Series, use_chip: pd.Series) -> pd.Series:
    """Strip .0 suffix, fill empties by semantics."""
    z = series.str.replace(r"\.0$", "", regex=True)
    online = use_chip == "Online Transaction"
    z = z.where(z != "", other="MISSING")         # genuinely missing
    z = z.where(~(online & (series == "")), other="00000")  # online → not-applicable
    return z


def clean_state(series: pd.Series, use_chip: pd.Series) -> pd.Series:
    online = use_chip == "Online Transaction"
    s = series.copy()
    s = s.where(~(online & (s == "")), other="ONLINE")
    return s


def frequency_encode(
    train: pd.DataFrame, test: pd.DataFrame, col: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Replace a high-cardinality string column with log-frequency from train."""
    freq = train[col].value_counts()
    freq_map = np.log1p(freq).to_dict()
    train[col] = train[col].map(freq_map).astype(np.float32)
    test[col] = test[col].map(freq_map).fillna(0.0).astype(np.float32)
    return train, test, freq_map


def label_encode(
    train: pd.DataFrame, test: pd.DataFrame, col: str
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    known = set(le.classes_)
    test[col] = test[col].apply(lambda x: x if x in known else le.classes_[0])
    test[col] = le.transform(test[col])
    return train, test, le


def chrono_split(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split: first `frac` of each user's history → train.

    Adds a _chrono_rank column (global row order before split) so the leakage
    check can verify temporal ordering without needing the timestamp column.
    """
    df = df.sort_values([USER_COL, "timestamp"]).reset_index(drop=True)
    df["_chrono_rank"] = np.arange(len(df), dtype=np.int64)
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


def _fraud_line(series: pd.Series) -> str:
    total = len(series)
    fraud = int(series.sum())
    rate = fraud / max(total, 1)
    return f"{fraud:,} / {total:,} ({rate:.4%})"


# --- Main -------------------------------------------------------------------

def main() -> None:
    if not INPUT_PARQUET.exists():
        log.error("Input not found at '%s'. Run step1 first.", INPUT_PARQUET)
        return

    log.info("Reading %s ...", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET)
    log.info("Loaded %s rows, %d columns.", f"{len(df):,}", len(df.columns))

    # --- Type conversions ---------------------------------------------------
    log.info("Parsing types ...")

    df["amount_raw"] = parse_amount(df["Amount"])
    df["amount_log"] = signed_log1p(df["amount_raw"]).astype(np.float32)
    log.info(
        "Amount: min=%.2f, max=%.2f, negatives=%s, zeros=%s",
        df["amount_raw"].min(), df["amount_raw"].max(),
        f'{(df["amount_raw"] < 0).sum():,}',
        f'{(df["amount_raw"] == 0).sum():,}',
    )

    df["timestamp"] = build_timestamp(df)
    df["label"] = (df[LABEL_COL] == "Yes").astype(np.int8)

    # --- Chronological sort + hours_since_last (before split) ---------------
    df = df.sort_values([USER_COL, "timestamp"]).reset_index(drop=True)
    df["hours_since_last"] = hours_since_last(df).astype(np.float32)

    # --- Clean categoricals -------------------------------------------------
    df["Zip"] = clean_zip(df["Zip"], df["Use Chip"])
    df["Merchant State"] = clean_state(df["Merchant State"], df["Use Chip"])

    # --- Explode Errors? into OHE columns -----------------------------------
    df = explode_errors(df)

    # --- Extract time features before dropping raw columns ------------------
    df["hour"] = df["timestamp"].dt.hour.astype(np.int8)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.int8)
    df["month"] = df["timestamp"].dt.month.astype(np.int8)

    # --- One-hot Use Chip (3 values) ----------------------------------------
    chip_dummies = pd.get_dummies(df["Use Chip"], prefix="chip", dtype=np.int8)
    df = pd.concat([df, chip_dummies], axis=1)

    # --- Drop raw columns that are now encoded (keep timestamp for split) ---
    drop_before_split = [
        "Amount", "Year", "Month", "Day", "Time",
        LABEL_COL, "Use Chip", "Errors?", "Card",
    ]
    df.drop(columns=drop_before_split, inplace=True)

    # --- Chronological split ------------------------------------------------
    log.info("Chronological train/test split (%.0f%%/%.0f%%) per user ...",
             CHRONO_SPLIT_FRAC * 100, (1 - CHRONO_SPLIT_FRAC) * 100)
    train, test = chrono_split(df, CHRONO_SPLIT_FRAC)
    del df

    train.drop(columns=["timestamp"], inplace=True)
    test.drop(columns=["timestamp"], inplace=True)

    log.info("Train: %s rows, fraud: %s", f"{len(train):,}", _fraud_line(train["label"]))
    log.info("Test : %s rows, fraud: %s", f"{len(test):,}", _fraud_line(test["label"]))

    # --- Encode high-cardinality cols (fit on train) ------------------------
    encoders: dict = {"freq_maps": {}, "label_encoders": {}}

    for col in FREQ_ENCODE_COLS:
        train, test, freq_map = frequency_encode(train, test, col)
        encoders["freq_maps"][col] = freq_map
        log.info("Freq-encoded %s (%d classes)", col, len(freq_map))

    for col in LABEL_ENCODE_COLS:
        train, test, le = label_encode(train, test, col)
        encoders["label_encoders"][col] = le
        log.info("Label-encoded %s (%d classes)", col, len(le.classes_))

    # --- StandardScaler on numeric columns (fit on train) -------------------
    numeric_cols = [
        "amount_raw", "amount_log", "hours_since_last",
        "hour", "day_of_week", "month",
    ] + [c for c in FREQ_ENCODE_COLS]

    scaler = StandardScaler()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols]).astype(np.float32)
    test[numeric_cols] = scaler.transform(test[numeric_cols]).astype(np.float32)

    # --- Leakage check ------------------------------------------------------
    user_overlap = set(train[USER_COL].unique()) & set(test[USER_COL].unique())
    log.info("Users in both train and test: %d (expected = %d, all users)",
             len(user_overlap), train[USER_COL].nunique())

    train_max_rank = train.groupby(USER_COL)["_chrono_rank"].max()
    test_min_rank = test.groupby(USER_COL)["_chrono_rank"].min()
    common = train_max_rank.index.intersection(test_min_rank.index)
    leaks = int((train_max_rank.loc[common] >= test_min_rank.loc[common]).sum())
    log.info("Temporal leakage (train rank >= test rank for same user): %d", leaks)
    assert leaks == 0, "Temporal leakage detected!"

    train.drop(columns=["_chrono_rank"], inplace=True)
    test.drop(columns=["_chrono_rank"], inplace=True)

    # --- Save ---------------------------------------------------------------
    train.to_parquet(OUTPUT_TRAIN, index=False)
    test.to_parquet(OUTPUT_TEST, index=False)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    log.info("=" * 70)
    log.info("STEP 2 COMPLETE")
    log.info("Train : %s (%s rows, fraud %s)",
             OUTPUT_TRAIN, f"{len(train):,}", _fraud_line(train["label"]))
    log.info("Test  : %s (%s rows, fraud %s)",
             OUTPUT_TEST, f"{len(test):,}", _fraud_line(test["label"]))
    log.info("Scaler: %s  (%d features)", SCALER_PATH, len(numeric_cols))
    log.info("Encoders: %s", ENCODERS_PATH)
    log.info("Final columns (%d): %s", len(train.columns), train.columns.tolist())
    log.info("=" * 70)


if __name__ == "__main__":
    main()
