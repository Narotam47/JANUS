"""Step 8 — JANUS Phase 1: retrain XGBoost with enriched features.

Merges Phase 0 sliding-window vectors (210 dims) with Phase 1 features
(14 user-profile + 9 merchant/population + 10 first-occurrence = 33 dims)
and retrains XGBoost to measure the lift from enriched features alone.

Feature alignment
-----------------
Window arrays are per-WINDOW (last-transaction labelled).
Feature parquets are per-TRANSACTION (row-aligned with clean_train/test).

Each window's enriched features come from its LAST transaction — the one
being classified.  For a user with N ≥ 10 transactions, windows map to
transactions at indices [9, 10, ..., N-1].  Users with < 10 transactions
(11 test users, 0 train) are skipped — matching step 3's skip logic.

Memory strategy
---------------
Window arrays are 6.2 GB (train) + 1.5 GB (test).  To avoid loading them
fully into RAM, a merged memmap is built by copying windows chunk by chunk
(500k rows at a time) and appending the enriched features.  XGBoost trains
directly from the memmap (tree_method="hist" streams data page-by-page).
Temporary merged files are deleted after training.

Training config is IDENTICAL to step 4 (same XGB_PARAMS + scale_pos_weight)
so the comparison is attributable to the features alone.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")

WINDOWS_TRAIN: Final[Path] = ARTIFACTS_DIR / "windows_train.npy"
LABELS_TRAIN: Final[Path] = ARTIFACTS_DIR / "labels_train.npy"
WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
LABELS_TEST: Final[Path] = ARTIFACTS_DIR / "labels_test.npy"

CLEAN_TRAIN: Final[Path] = ARTIFACTS_DIR / "clean_train.parquet"
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"

USER_FEATURES_TRAIN: Final[Path] = ARTIFACTS_DIR / "user_features_train.parquet"
USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TRAIN: Final[Path] = ARTIFACTS_DIR / "merchant_features_train.parquet"
MERCH_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TRAIN: Final[Path] = ARTIFACTS_DIR / "flags_train.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

BASELINE_METRICS: Final[Path] = ARTIFACTS_DIR / "metrics_baseline.json"
MODEL_PATH: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
METRICS_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"

MERGED_TRAIN: Final[Path] = ARTIFACTS_DIR / "_tmp_merged_train.npy"
MERGED_TEST: Final[Path] = ARTIFACTS_DIR / "_tmp_merged_test.npy"

USER_COL: Final[str] = "User"
LABEL_COL: Final[str] = "label"
WINDOW_SIZE: Final[int] = 10
RANDOM_SEED: Final[int] = 42
CHUNK_SIZE: Final[int] = 500_000

XGB_PARAMS: Final[dict] = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="aucpr",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

WATCH_LIST: Final[dict[str, str]] = {
    "user_tx_count": "distribution shift (train mu=8.7k, test mu=19.6k)",
    "merch_amount_deviation": "extreme z-scores (±70M) from near-zero-std merchants",
    "first_channel": "near-zero fraud coverage (0 fraud / 2,191 fires in train)",
    "is_5x_hist_max": "near-zero fraud coverage (1 fraud / 217 fires in train)",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step8")


# --- Helpers ----------------------------------------------------------------

def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0:
        return 1.0
    weight = neg / pos
    log.info("Class balance: neg=%s, pos=%s, scale_pos_weight=%.1f",
             f"{neg:,}", f"{pos:,}", weight)
    return weight


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    best_idx = int(np.argmax(f1))
    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1[best_idx]),
        "precision_at_best": float(precision[best_idx]),
        "recall_at_best": float(recall[best_idx]),
        "n_thresholds_swept": len(thresholds),
    }


def threshold_sweep_summary(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    rows = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        idx = int(np.searchsorted(thresholds, t))
        if idx >= len(thresholds):
            idx = len(thresholds) - 1
        p, r = float(precision[idx]), float(recall[idx])
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({"threshold": float(thresholds[idx]),
                      "precision": round(p, 4), "recall": round(r, 4),
                      "f1": round(f, 4)})
    return rows


def measure_latency(model: XGBClassifier, X_single: np.ndarray,
                    n_runs: int = 1000) -> dict:
    X_single = X_single.reshape(1, -1)
    for _ in range(10):
        model.predict_proba(X_single)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        model.predict_proba(X_single)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    arr = np.array(times)
    return {
        "n_runs": n_runs,
        "mean_ms": round(float(arr.mean()), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
    }


# --- Feature alignment -----------------------------------------------------

def compute_window_feature_indices(clean_path: Path) -> np.ndarray:
    """Row indices in the per-transaction feature files that map to each
    window's last (classified) transaction."""
    users = pd.read_parquet(clean_path, columns=[USER_COL])[USER_COL]
    indices: list[np.ndarray] = []
    offset = 0
    for _, group in users.groupby(users, sort=False):
        n = len(group)
        if n >= WINDOW_SIZE:
            indices.append(np.arange(offset + WINDOW_SIZE - 1, offset + n))
        offset += n
    return np.concatenate(indices)


def load_enriched(split: str) -> pd.DataFrame:
    """Load and horizontally concatenate the three feature parquets."""
    suffix = f"_{split}.parquet"
    u = pd.read_parquet(ARTIFACTS_DIR / f"user_features{suffix}")
    m = pd.read_parquet(ARTIFACTS_DIR / f"merchant_features{suffix}")
    f = pd.read_parquet(ARTIFACTS_DIR / f"flags{suffix}")
    return pd.concat([u, m, f], axis=1)


def create_merged_memmap(
    windows_path: Path,
    enriched_arr: np.ndarray,
    out_path: Path,
    split_name: str,
) -> tuple[int, int]:
    X_base = np.load(str(windows_path), mmap_mode="r")
    n_win, n_base = X_base.shape
    n_new = enriched_arr.shape[1]
    n_total = n_base + n_new

    assert enriched_arr.shape[0] == n_win, (
        f"[{split_name}] enriched rows {enriched_arr.shape[0]} != windows {n_win}"
    )

    log.info("[%s] Building merged memmap: %s × %d (%d base + %d enriched) → %s",
             split_name, f"{n_win:,}", n_total, n_base, n_new, out_path.name)

    merged = open_memmap(
        str(out_path), mode="w+", dtype=np.float32, shape=(n_win, n_total),
    )

    for start in range(0, n_win, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_win)
        merged[start:end, :n_base] = X_base[start:end]

    merged[:, n_base:] = enriched_arr
    merged.flush()

    sz_gb = out_path.stat().st_size / 1e9
    log.info("[%s] Merged memmap written: %.2f GB", split_name, sz_gb)
    del merged, X_base
    return n_win, n_total


def build_feature_names(
    clean_path: Path, enriched_cols: list[str]
) -> list[str]:
    schema = pq.read_schema(str(clean_path))
    base_cols = [f.name for f in schema if f.name not in (USER_COL, LABEL_COL)]
    names: list[str] = []
    for t in range(WINDOW_SIZE):
        for col in base_cols:
            names.append(f"{col}_t{t}")
    names.extend(enriched_cols)
    return names


# --- Main -------------------------------------------------------------------

def get_enriched_col_names() -> list[str]:
    """Read enriched column names from parquet schemas (zero data loaded)."""
    cols: list[str] = []
    for name in ("user_features_train", "merchant_features_train", "flags_train"):
        schema = pq.read_schema(str(ARTIFACTS_DIR / f"{name}.parquet"))
        cols.extend(f.name for f in schema)
    return cols


def main() -> None:
    tracemalloc.start()
    import gc

    required = [
        WINDOWS_TRAIN, LABELS_TRAIN, WINDOWS_TEST, LABELS_TEST,
        CLEAN_TRAIN, CLEAN_TEST,
        USER_FEATURES_TRAIN, USER_FEATURES_TEST,
        MERCH_FEATURES_TRAIN, MERCH_FEATURES_TEST,
        FLAGS_TRAIN, FLAGS_TEST,
        BASELINE_METRICS,
    ]
    for p in required:
        if not p.exists():
            log.error("Missing %s — run prior steps first.", p)
            return

    y_train = np.load(str(LABELS_TRAIN), mmap_mode="r")
    y_test = np.load(str(LABELS_TEST), mmap_mode="r")
    enriched_cols = get_enriched_col_names()
    log.info("Enriched features: %d columns", len(enriched_cols))

    # ---- 1. Build merged memmaps if needed ---------------------------------
    if MERGED_TRAIN.exists() and MERGED_TEST.exists():
        log.info("Merged temp files already exist — skipping merge step.")
        probe = np.load(str(MERGED_TRAIN), mmap_mode="r")
        n_win_train, n_total = probe.shape
        del probe
    else:
        log.info("Computing window → transaction index mapping ...")
        train_idx = compute_window_feature_indices(CLEAN_TRAIN)
        test_idx = compute_window_feature_indices(CLEAN_TEST)
        assert len(train_idx) == len(y_train), "Train index/window count mismatch"
        assert len(test_idx) == len(y_test), "Test index/window count mismatch"
        log.info("Feature index counts match window counts. ✓")

        log.info("Loading enriched features ...")
        enriched_train_df = load_enriched("train")
        enriched_test_df = load_enriched("test")

        enriched_train = enriched_train_df.iloc[train_idx].values.astype(np.float32)
        del enriched_train_df
        enriched_test = enriched_test_df.iloc[test_idx].values.astype(np.float32)
        del enriched_test_df
        gc.collect()

        n_win_train, n_total = create_merged_memmap(
            WINDOWS_TRAIN, enriched_train, MERGED_TRAIN, "train",
        )
        del enriched_train; gc.collect()
        create_merged_memmap(
            WINDOWS_TEST, enriched_test, MERGED_TEST, "test",
        )
        del enriched_test; gc.collect()

    # ---- 2. Train ----------------------------------------------------------
    # Load X_train into RAM (6.6 GB) — XGBoost reads it hundreds of times.
    # Skip eval_set to avoid loading X_test (1.7 GB) during training —
    # predict on test separately afterwards.
    log.info("Loading merged train array into RAM ...")
    X_train = np.load(str(MERGED_TRAIN))
    log.info("X_train: %s (%.1f GB)", X_train.shape,
             X_train.nbytes / 1e9)

    spw = compute_scale_pos_weight(y_train)
    model = XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)

    log.info("Training enriched XGBoost (n_estimators=%d, max_depth=%d, %d features) ...",
             XGB_PARAMS["n_estimators"], XGB_PARAMS["max_depth"], n_total)
    log.info("(eval_set omitted to save 1.7 GB RAM — test eval after training)")
    t0 = time.time()
    model.fit(X_train, y_train, verbose=50)
    train_secs = time.time() - t0
    log.info("Training time: %.1f s", train_secs)

    # Free train array before loading test
    del X_train; gc.collect()

    # ---- 3. Evaluate -------------------------------------------------------
    log.info("Loading merged test array for evaluation ...")
    X_test = np.load(str(MERGED_TEST))
    log.info("X_test: %s", X_test.shape)

    log.info("Predicting on test set ...")
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    best_f1_info = find_best_f1_threshold(y_test, y_prob)
    best_t = best_f1_info["best_threshold"]

    log.info("=" * 70)
    log.info("ENRICHED MODEL — HEADLINE METRICS")
    log.info("AUC-PR  (primary) : %.6f", auc_pr)
    log.info("ROC-AUC (secondary): %.6f", roc_auc)

    log.info("-" * 70)
    log.info("BEST-F1 THRESHOLD ANALYSIS")
    log.info("Threshold: %.4f", best_t)
    log.info("F1: %.4f | Precision: %.4f | Recall: %.4f",
             best_f1_info["best_f1"],
             best_f1_info["precision_at_best"],
             best_f1_info["recall_at_best"])

    y_pred_best = (y_prob >= best_t).astype(int)
    y_pred_50 = (y_prob >= 0.5).astype(int)

    log.info("-" * 70)
    log.info("Classification report @ threshold=%.4f (best F1):", best_t)
    log.info("\n%s", classification_report(y_test, y_pred_best, digits=4))

    sweep = threshold_sweep_summary(y_test, y_prob)
    log.info("-" * 70)
    log.info("THRESHOLD SWEEP:")
    log.info("%-12s %-12s %-12s %-12s", "Threshold", "Precision", "Recall", "F1")
    for row in sweep:
        log.info("%-12.4f %-12.4f %-12.4f %-12.4f",
                 row["threshold"], row["precision"], row["recall"], row["f1"])

    # ---- 4. Latency -------------------------------------------------------
    log.info("-" * 70)
    log.info("Measuring single-window CPU inference latency ...")
    sample_row = np.array(X_test[0], dtype=np.float32)
    latency = measure_latency(model, sample_row)
    log.info("Latency (1000 runs): mean=%.3f ms, p95=%.3f ms",
             latency["mean_ms"], latency["p95_ms"])

    # ---- 5. Feature importance (top 30) ------------------------------------
    log.info("-" * 70)
    log.info("TOP 30 FEATURES BY IMPORTANCE (gain)")
    feature_names = build_feature_names(CLEAN_TRAIN, enriched_cols)
    assert len(feature_names) == n_total, (
        f"Feature names {len(feature_names)} != total features {n_total}"
    )

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:30]

    log.info("%-4s  %-40s  %10s  %s", "Rank", "Feature", "Importance", "Notes")
    for rank, idx in enumerate(top_idx, 1):
        name = feature_names[idx]
        imp = importances[idx]
        warning = ""
        for key, msg in WATCH_LIST.items():
            if key in name:
                warning = f"  ⚠ {msg}"
                break
        log.info("%-4d  %-40s  %10.4f%s", rank, name, imp, warning)

    # ---- 6. Side-by-side comparison ----------------------------------------
    log.info("=" * 70)
    log.info("SIDE-BY-SIDE COMPARISON: baseline → enriched")

    with open(BASELINE_METRICS) as f:
        baseline = json.load(f)

    b_auc_pr = baseline["headline_metrics"]["AUC_PR"]
    b_roc_auc = baseline["headline_metrics"]["ROC_AUC"]
    b_f1 = baseline["best_f1_threshold"]["best_f1"]
    b_prec = baseline["best_f1_threshold"]["precision_at_best"]
    b_rec = baseline["best_f1_threshold"]["recall_at_best"]
    b_thresh = baseline["best_f1_threshold"]["best_threshold"]
    b_lat = baseline["latency_cpu_single_window"]["mean_ms"]

    e_f1 = best_f1_info["best_f1"]
    e_prec = best_f1_info["precision_at_best"]
    e_rec = best_f1_info["recall_at_best"]

    log.info("%-20s %12s %12s %12s", "Metric", "Baseline", "Enriched", "Delta")
    log.info("%-20s %12.6f %12.6f %+12.4f", "AUC-PR", b_auc_pr, auc_pr, auc_pr - b_auc_pr)
    log.info("%-20s %12.6f %12.6f %+12.4f", "ROC-AUC", b_roc_auc, roc_auc, roc_auc - b_roc_auc)
    log.info("%-20s %12.4f %12.4f %+12.4f", "Best F1", b_f1, e_f1, e_f1 - b_f1)
    log.info("%-20s %12.4f %12.4f %+12.4f", "Precision@F1", b_prec, e_prec, e_prec - b_prec)
    log.info("%-20s %12.4f %12.4f %+12.4f", "Recall@F1", b_rec, e_rec, e_rec - b_rec)
    log.info("%-20s %12.4f %12.4f %+12.4f", "Threshold@F1", b_thresh, best_t, best_t - b_thresh)
    log.info("%-20s %12.3f %12.3f %+12.3f ms",
             "Latency (mean)", b_lat, latency["mean_ms"], latency["mean_ms"] - b_lat)
    log.info("%-20s %12d %12d %+12d", "Features",
             baseline["data"]["features_per_window"], n_total,
             n_total - baseline["data"]["features_per_window"])

    # ---- 7. Peak memory ---------------------------------------------------
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_mem / 1e6
    log.info("-" * 70)
    log.info("Peak memory (tracemalloc, excludes mmap): %.1f MB", peak_mb)

    # ---- 8. Save model + metrics ------------------------------------------
    n_test_win = int(X_test.shape[0])
    del X_test

    model.save_model(str(MODEL_PATH))
    log.info("Model saved: %s (%.1f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)

    metrics = {
        "model": "XGBoost enriched (Phase 1 features)",
        "phase": "Phase 1 — enriched defender",
        "data": {
            "train_windows": n_win_train,
            "train_positives": int(y_train.sum()),
            "train_positive_rate": round(float(y_train.sum() / len(y_train)), 6),
            "test_windows": n_test_win,
            "test_positives": int(y_test.sum()),
            "test_positive_rate": round(float(y_test.sum() / len(y_test)), 6),
            "features_total": n_total,
            "features_window": n_total - len(enriched_cols),
            "features_enriched": len(enriched_cols),
            "scale_pos_weight": round(spw, 1),
        },
        "headline_metrics": {
            "AUC_PR": round(auc_pr, 6),
            "ROC_AUC": round(roc_auc, 6),
        },
        "best_f1_threshold": best_f1_info,
        "at_default_threshold_0.5": {
            "f1": round(float(f1_score(y_test, y_pred_50)), 6),
            "precision": round(float(
                y_pred_50[y_test == 1].sum() / max(y_pred_50.sum(), 1)
            ), 6),
            "recall": round(float(
                y_pred_50[y_test == 1].sum() / max(y_test.sum(), 1)
            ), 6),
        },
        "threshold_sweep": sweep,
        "latency_cpu_single_window": latency,
        "training": {
            "time_seconds": round(train_secs, 1),
            "xgb_params": {**XGB_PARAMS, "scale_pos_weight": round(spw, 1)},
        },
        "vs_baseline": {
            "AUC_PR_delta": round(auc_pr - b_auc_pr, 6),
            "ROC_AUC_delta": round(roc_auc - b_roc_auc, 6),
            "best_f1_delta": round(e_f1 - b_f1, 6),
        },
        "feature_importance_top30": [
            {"rank": rank, "feature": feature_names[idx],
             "importance": round(float(importances[idx]), 6)}
            for rank, idx in enumerate(np.argsort(importances)[::-1][:30], 1)
        ],
        "peak_memory_mb": round(peak_mb, 1),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Metrics saved: %s", METRICS_PATH)

    # ---- 9. Cleanup temp merged arrays ------------------------------------
    for tmp in (MERGED_TRAIN, MERGED_TEST):
        if tmp.exists():
            sz = tmp.stat().st_size / 1e9
            tmp.unlink()
            log.info("Cleaned up %s (%.2f GB)", tmp.name, sz)

    log.info("=" * 70)
    log.info("STEP 8 COMPLETE — enriched model trained and evaluated.")
    log.info("Compare: %s vs %s", METRICS_PATH, BASELINE_METRICS)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
