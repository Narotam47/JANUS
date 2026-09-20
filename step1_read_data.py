"""Step 1 — JANUS Phase 0: read the raw TabFormer transactions and carve out a
memory-safe dev subset.

What this script does
---------------------
1. Streams the full raw CSV (24M+ rows) in chunks so the whole file is never in RAM.
2. Pass 1 (cheap): scans only the [User, label] columns to learn the full set of
   users, the total row count, and the overall fraud balance.
3. Samples a fixed dev subset of ~500-1000 users (seeded => reproducible) and keeps
   each sampled user's *entire* transaction history.
4. Pass 2: streams the full file again, filters to the sampled users, and writes
   the kept rows straight to `artifacts/raw_subset.parquet`, one chunk per row
   group (pyarrow.ParquetWriter) — again never materialising the whole subset.
5. Prints sanity checks: row counts, required-column presence, and fraud balance.

Why the subset is written as raw strings (dtype=str)
----------------------------------------------------
Step 1 is "get the raw data faithfully". Type-casting and feature parsing happen
in step 2. Reading everything as strings (a) preserves the raw values verbatim
(e.g. Amount stays "$134.09", no int/float guessing) and (b) guarantees an
identical Parquet schema for every chunk, so the streamed row groups line up.

Canonical TabFormer (IBM card_transaction.v1.csv) columns, for reference:
    User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
    Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?
If your file uses different names, edit USER_COL / LABEL_COL / EXPECTED_COLUMNS.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --- Config -----------------------------------------------------------------
# Path to the raw TabFormer CSV. Override without editing this file via the
# JANUS_RAW_CSV env var, e.g.:
#     JANUS_RAW_CSV=/data/card_transaction.v1.csv python step1_read_data.py
RAW_CSV_PATH: Final[Path] = Path(
    os.environ.get("JANUS_RAW_CSV", "Data/card_transaction.v1.csv")
)

# Where the dev subset lands.
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
OUTPUT_PARQUET: Final[Path] = ARTIFACTS_DIR / "raw_subset.parquet"

# The two columns we cannot proceed without.
USER_COL: Final[str] = "User"         # groups transactions into per-user histories
LABEL_COL: Final[str] = "Is Fraud?"   # fraud flag; "Yes"/"No" in TabFormer

# Full set of columns we expect. Missing ones are warned about, not fatal.
EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    "User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
    "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
    "Errors?", "Is Fraud?",
)

# Dev-subset sampling.
# NOTE: TabFormer averages ~12k transactions per user (24M rows / 2000 users),
# so 750 users is roughly ~9M rows. Lower this for a smaller/faster dev subset.
N_DEV_USERS: Final[int] = 750       # anywhere in the 500-1000 range you asked for
RANDOM_SEED: Final[int] = 42        # fixed so the subset is reproducible
CHUNK_SIZE: Final[int] = 1_000_000  # rows per read_csv chunk

# Read everything as raw strings: faithful copy + stable per-chunk Parquet schema.
# keep_default_na=False keeps empty fields as "" instead of NaN (raw fidelity).
READ_KWARGS: Final[dict] = dict(dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step1")


def check_columns(path: Path) -> list[str]:
    """Read only the header, verify required columns exist, warn on missing expected ones.

    Returns the actual column list. Raises KeyError if a required column is absent.
    """
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    log.info("Columns found (%d): %s", len(columns), columns)

    missing_required = [c for c in (USER_COL, LABEL_COL) if c not in columns]
    if missing_required:
        raise KeyError(
            f"Required column(s) {missing_required} not in file. Found {columns}. "
            "Edit USER_COL / LABEL_COL at the top of this file to match your schema."
        )

    missing_expected = [c for c in EXPECTED_COLUMNS if c not in columns]
    if missing_expected:
        log.warning("Expected TabFormer columns missing (non-fatal): %s", missing_expected)
    return columns


def scan_universe(path: Path) -> tuple[int, dict[str, int], list[str]]:
    """Pass 1 — stream only [User, label] to collect totals cheaply.

    Returns (total_rows, overall_label_counts, sorted_unique_users).
    """
    total_rows = 0
    label_counts: dict[str, int] = {}
    users: set[str] = set()

    for chunk in pd.read_csv(path, usecols=[USER_COL, LABEL_COL], **READ_KWARGS):
        total_rows += len(chunk)
        users.update(chunk[USER_COL].unique().tolist())
        for value, count in chunk[LABEL_COL].value_counts().items():
            label_counts[value] = label_counts.get(value, 0) + int(count)

    return total_rows, label_counts, sorted(users)


def sample_dev_users(all_users: list[str], n: int, seed: int) -> set[str]:
    """Pick n users deterministically (or all of them if fewer than n exist)."""
    if len(all_users) <= n:
        log.warning(
            "Only %d users available; taking all (fewer than requested %d).",
            len(all_users), n,
        )
        return set(all_users)
    return set(random.Random(seed).sample(all_users, n))


def write_subset(
    path: Path, dev_users: set[str], out: Path
) -> tuple[int, dict[str, int], set[str]]:
    """Pass 2 — stream full rows, keep sampled users, write straight to Parquet.

    Each surviving chunk is appended as its own Parquet row group, so peak memory
    is one chunk — not the whole subset. Returns
    (subset_rows, subset_label_counts, users_actually_seen).
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    writer: "pq.ParquetWriter | None" = None
    subset_rows = 0
    label_counts: dict[str, int] = {}
    users_seen: set[str] = set()

    try:
        for chunk in pd.read_csv(path, **READ_KWARGS):
            keep = chunk[chunk[USER_COL].isin(dev_users)]
            if keep.empty:
                continue

            subset_rows += len(keep)
            users_seen.update(keep[USER_COL].unique().tolist())
            for value, count in keep[LABEL_COL].value_counts().items():
                label_counts[value] = label_counts.get(value, 0) + int(count)

            table = pa.Table.from_pandas(keep, preserve_index=False)
            if writer is None:                       # first surviving chunk fixes the schema
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError(
            "No rows matched the sampled users — subset is empty, nothing written."
        )

    return subset_rows, label_counts, users_seen


def _fraud_line(counts: dict[str, int]) -> str:
    """Format a label distribution as 'Yes=1,234 (0.1234%), No=...'."""
    total = sum(counts.values()) or 1
    return ", ".join(
        f"{label}={n:,} ({n / total:.4%})" for label, n in sorted(counts.items())
    )


def main() -> None:
    if not RAW_CSV_PATH.exists():
        log.error(
            "Raw CSV not found at '%s'. Set JANUS_RAW_CSV or fix RAW_CSV_PATH.",
            RAW_CSV_PATH,
        )
        return

    log.info("Reading raw TabFormer CSV: %s", RAW_CSV_PATH.resolve())
    check_columns(RAW_CSV_PATH)

    log.info("Pass 1/2 — scanning user universe + overall fraud balance ...")
    total_rows, overall_counts, all_users = scan_universe(RAW_CSV_PATH)

    dev_users = sample_dev_users(all_users, N_DEV_USERS, RANDOM_SEED)
    log.info(
        "Sampled %d dev users (seed=%d) out of %d total.",
        len(dev_users), RANDOM_SEED, len(all_users),
    )

    log.info("Pass 2/2 — extracting full histories for sampled users -> %s ...", OUTPUT_PARQUET)
    subset_rows, subset_counts, users_seen = write_subset(
        RAW_CSV_PATH, dev_users, OUTPUT_PARQUET
    )

    # --- Sanity checks (printed so you can eyeball the output) ---------------
    log.info("=" * 70)
    log.info("SANITY CHECKS")
    log.info("Full dataset : %s rows, %d users", f"{total_rows:,}", len(all_users))
    log.info("Full fraud   : %s", _fraud_line(overall_counts))
    log.info("-" * 70)
    log.info(
        "Dev subset   : %s rows, %d users (of %d sampled)",
        f"{subset_rows:,}", len(users_seen), len(dev_users),
    )
    log.info("Subset fraud : %s", _fraud_line(subset_counts))
    log.info("Avg tx/user  : %.0f", subset_rows / max(len(users_seen), 1))
    log.info(
        "Written to   : %s (%.1f MB)",
        OUTPUT_PARQUET.resolve(), OUTPUT_PARQUET.stat().st_size / 1e6,
    )
    log.info("=" * 70)

    # Loud, not silent, on the two things most likely to go wrong.
    assert subset_rows > 0, "Subset is empty — check USER_COL / sampling."
    if users_seen != dev_users:
        log.warning(
            "Sampled %d users but only %d appeared in the data.",
            len(dev_users), len(users_seen),
        )


if __name__ == "__main__":
    main()
