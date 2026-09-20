"""Step 3 — JANUS Phase 0: build sliding sequence windows for the detector.

Input : artifacts/clean_train.parquet, artifacts/clean_test.parquet
Output: artifacts/windows_train.npy   (N_train, 210)  float32
        artifacts/labels_train.npy    (N_train,)       int8
        artifacts/windows_test.npy    (N_test, 210)    float32
        artifacts/labels_test.npy     (N_test,)        int8

Design decisions
----------------
WINDOW LABELING — last transaction's label.
  At 0.12% fraud, "any fraud in window" inflates positives by up to 10x,
  teaching the model to flag entire *neighborhoods* of fraud rather than
  detecting the fraudulent transaction itself. "Last" preserves detection
  semantics: "given this 10-tx history, is the current transaction fraud?"
  This keeps the natural 0.12% rate through windowing, and downstream
  imbalance handling (SMOTE, class weights, focal loss) can be calibrated
  against the true base rate.

STRIDE — 1 (no skip).
  At 0.12% fraud, stride>1 discards positive windows proportionally. Each
  fraud transaction produces exactly one positive window (where it is the
  last element). Stride=5 would lose ~80% of the already-scarce 9,220
  train fraud labels — unacceptable. Stride=1 preserves every positive
  example. The large window count (~7.3M) is handled via numpy memmap
  (peak RAM ≈ DataFrame + one user's feature array, not the full tensor).

MEMORY — per-user processing + numpy memmap output.
  Pass 1 counts windows. Pass 2 pre-allocates .npy files on disk via
  numpy.lib.format.open_memmap, then fills them one user at a time using
  numpy.lib.stride_tricks.sliding_window_view (vectorised, no Python loop
  over individual windows). Peak RSS is the DataFrame plus one user's
  arrays (~12k × 21 features ≈ 1 MB), not the ~6 GB output tensor.

USER BOUNDARIES — guaranteed by the per-user groupby loop. A window never
  mixes transactions from different users.
"""

from __future__ import annotations

import gc
import logging
import tracemalloc
from pathlib import Path
from typing import Final

import numpy as np
from numpy.lib.format import open_memmap
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
TRAIN_PATH: Final[Path] = ARTIFACTS_DIR / "clean_train.parquet"
TEST_PATH: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"

WINDOW_SIZE: Final[int] = 10
STRIDE: Final[int] = 1

USER_COL: Final[str] = "User"
LABEL_COL: Final[str] = "label"

WINDOWS_TRAIN: Final[Path] = ARTIFACTS_DIR / "windows_train.npy"
LABELS_TRAIN: Final[Path] = ARTIFACTS_DIR / "labels_train.npy"
WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
LABELS_TEST: Final[Path] = ARTIFACTS_DIR / "labels_test.npy"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step3")


# ---------------------------------------------------------------------------

def count_windows(df: pd.DataFrame) -> tuple[int, int, int]:
    """Pass 1 — count total windows, users skipped, rows dropped (cheap)."""
    total_win = 0
    users_skipped = 0
    rows_dropped = 0

    for _, group in df.groupby(USER_COL, sort=False):
        n = len(group)
        if n < WINDOW_SIZE:
            users_skipped += 1
            rows_dropped += n
            continue
        n_win = (n - WINDOW_SIZE) // STRIDE + 1
        total_win += n_win
        rows_used = (n_win - 1) * STRIDE + WINDOW_SIZE
        rows_dropped += n - rows_used

    return total_win, users_skipped, rows_dropped


def build_windows(
    df: pd.DataFrame,
    out_windows: Path,
    out_labels: Path,
    split_name: str,
) -> tuple[int, int]:
    """Build sliding windows per user, write to .npy via memmap.

    Returns (total_windows, positive_windows).
    """
    feature_cols = [c for c in df.columns if c not in (USER_COL, LABEL_COL)]
    n_features = len(feature_cols)
    flat_dim = WINDOW_SIZE * n_features

    log.info("[%s] %d features × %d window = %d flat dim",
             split_name, n_features, WINDOW_SIZE, flat_dim)

    # Pass 1: count
    total_win, skipped, dropped = count_windows(df)
    log.info(
        "[%s] Total windows: %s | users skipped (<W=%d): %d | rows dropped: %s",
        split_name, f"{total_win:,}", WINDOW_SIZE, skipped, f"{dropped:,}",
    )
    if total_win == 0:
        raise RuntimeError(f"No windows for {split_name}")

    # Allocate .npy memmap files (proper npy format, loadable with np.load)
    X = open_memmap(
        str(out_windows), dtype=np.float32, mode="w+", shape=(total_win, flat_dim),
    )
    y = open_memmap(
        str(out_labels), dtype=np.int8, mode="w+", shape=(total_win,),
    )

    # Pass 2: fill per user — vectorised within each user
    idx = 0
    pos_count = 0
    users_processed = 0

    for _, group in df.groupby(USER_COL, sort=False):
        n = len(group)
        if n < WINDOW_SIZE:
            continue

        feats = group[feature_cols].values.astype(np.float32)
        labels = group[LABEL_COL].values.astype(np.int8)

        # sliding_window_view: (n, F) → (n-W+1, W, F), zero-copy view
        windowed = sliding_window_view(feats, window_shape=WINDOW_SIZE, axis=0)
        windowed = windowed[::STRIDE]
        n_win = len(windowed)
        flat = windowed.reshape(n_win, flat_dim).copy()

        win_labels = labels[WINDOW_SIZE - 1 :: STRIDE][:n_win]

        X[idx : idx + n_win] = flat
        y[idx : idx + n_win] = win_labels
        pos_count += int(win_labels.sum())
        idx += n_win
        users_processed += 1

    X.flush(); del X
    y.flush(); del y

    log.info("[%s] Filled %s windows from %d users.", split_name, f"{idx:,}", users_processed)
    return total_win, pos_count


def main() -> None:
    tracemalloc.start()

    for path in (TRAIN_PATH, TEST_PATH):
        if not path.exists():
            log.error("Input not found at '%s'. Run step2 first.", path)
            return

    for split_name, in_path, w_path, l_path in [
        ("train", TRAIN_PATH, WINDOWS_TRAIN, LABELS_TRAIN),
        ("test", TEST_PATH, WINDOWS_TEST, LABELS_TEST),
    ]:
        log.info("=" * 70)
        log.info("[%s] Loading %s ...", split_name, in_path)
        df = pd.read_parquet(in_path)
        log.info("[%s] %s rows loaded.", split_name, f"{len(df):,}")

        total, positives = build_windows(df, w_path, l_path, split_name)
        del df
        gc.collect()

        rate = positives / max(total, 1) * 100
        w_mb = w_path.stat().st_size / 1e6
        l_mb = l_path.stat().st_size / 1e6

        log.info(
            "[%s] Positive windows: %s / %s (%.4f%%)",
            split_name, f"{positives:,}", f"{total:,}", rate,
        )
        log.info(
            "[%s] Files: %s (%.1f MB), %s (%.1f MB)",
            split_name, w_path.name, w_mb, l_path.name, l_mb,
        )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    log.info("=" * 70)
    log.info("STEP 3 COMPLETE")
    log.info("Peak memory (tracemalloc): %.1f MB", peak / 1e6)

    # Final verification — reload shapes via mmap (no extra RAM)
    for name, w_path, l_path in [
        ("train", WINDOWS_TRAIN, LABELS_TRAIN),
        ("test", WINDOWS_TEST, LABELS_TEST),
    ]:
        X = np.load(str(w_path), mmap_mode="r")
        y = np.load(str(l_path), mmap_mode="r")
        pos = int(y.sum())
        log.info(
            "[%s] X: %s  y: %s  positive: %s (%.4f%%)",
            name, X.shape, y.shape, f"{pos:,}", pos / len(y) * 100,
        )
        del X, y

    log.info("=" * 70)


if __name__ == "__main__":
    main()
