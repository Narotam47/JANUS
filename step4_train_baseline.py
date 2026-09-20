"""Step 4 — JANUS Phase 0: train the baseline XGBoost defender and record
reference metrics that every later result in this project gets compared against.

Input : artifacts/windows_train.npy, labels_train.npy
        artifacts/windows_test.npy,  labels_test.npy
Output: artifacts/model_baseline.json  (XGBoost model in JSON format)
        artifacts/metrics_baseline.json (all metrics + threshold analysis)

Why AUC-PR leads at this base rate
-----------------------------------
With 0.12% fraud (795:1 imbalance), ROC-AUC is dominated by the true-negative
rate and can look deceptively high even for a mediocre model — a classifier
that flags 1% of transactions as fraud catches most fraud but also generates
massive false-positive volume, yet still shows ROC-AUC > 0.95 because TNR
barely moves. AUC-PR (average precision) focuses on the precision-recall
tradeoff in the rare positive class and is far more informative: a random
classifier scores ~0.0012 AUC-PR, so any meaningful signal is immediately
visible.

Imbalance handling
------------------
scale_pos_weight = count(neg) / count(pos) ≈ 795. This tells XGBoost to
weight each positive sample 795× more than each negative in the gradient,
equivalent to replicating each fraud row 795 times but without the memory
or overfitting cost. No synthetic data or resampling at this stage — that
belongs in Phase 4 (GenAI augmentation).
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Final

import numpy as np
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

MODEL_PATH: Final[Path] = ARTIFACTS_DIR / "model_baseline.json"
METRICS_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_baseline.json"

RANDOM_SEED: Final[int] = 42

# XGBoost training params — conservative defaults for a reproducible baseline.
# tree_method="hist" is memory-efficient for large datasets.
XGB_PARAMS: Final[dict] = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="aucpr",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step4")


# --- Helpers ----------------------------------------------------------------

def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0:
        log.warning("No positives — defaulting scale_pos_weight=1.0")
        return 1.0
    weight = neg / pos
    log.info("Class balance: neg=%s, pos=%s, scale_pos_weight=%.1f",
             f"{neg:,}", f"{pos:,}", weight)
    return weight


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Sweep precision-recall curve to find the threshold maximising F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    precision = precision[:-1]
    recall = recall[:-1]

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


def threshold_sweep_summary(
    y_true: np.ndarray, y_prob: np.ndarray
) -> list[dict]:
    """Report metrics at a handful of representative thresholds."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]

    targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for t in targets:
        idx = np.searchsorted(thresholds, t)
        if idx >= len(thresholds):
            idx = len(thresholds) - 1
        p = float(precision[idx])
        r = float(recall[idx])
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({
            "threshold": float(thresholds[idx]),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
        })
    return rows


def measure_latency(model: XGBClassifier, X_single: np.ndarray, n_runs: int = 1000) -> dict:
    """Measure single-window CPU inference latency."""
    X_single = X_single.reshape(1, -1)

    # Warm up
    for _ in range(10):
        model.predict_proba(X_single)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        model.predict_proba(X_single)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)  # ms

    times_arr = np.array(times)
    return {
        "n_runs": n_runs,
        "mean_ms": round(float(times_arr.mean()), 4),
        "median_ms": round(float(np.median(times_arr)), 4),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 4),
        "p99_ms": round(float(np.percentile(times_arr, 99)), 4),
        "min_ms": round(float(times_arr.min()), 4),
        "max_ms": round(float(times_arr.max()), 4),
    }


# --- Main -------------------------------------------------------------------

def main() -> None:
    tracemalloc.start()

    for p in (WINDOWS_TRAIN, LABELS_TRAIN, WINDOWS_TEST, LABELS_TEST):
        if not p.exists():
            log.error("Missing %s — run step3 first.", p)
            return

    # Load via memmap — no full-array copy into RAM
    log.info("Loading data via memmap ...")
    X_train = np.load(str(WINDOWS_TRAIN), mmap_mode="r")
    y_train = np.load(str(LABELS_TRAIN), mmap_mode="r")
    X_test = np.load(str(WINDOWS_TEST), mmap_mode="r")
    y_test = np.load(str(LABELS_TEST), mmap_mode="r")

    log.info("X_train: %s  y_train: %s", X_train.shape, y_train.shape)
    log.info("X_test:  %s  y_test:  %s", X_test.shape, y_test.shape)

    # --- Train ---------------------------------------------------------------
    spw = compute_scale_pos_weight(y_train)

    model = XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)

    log.info("Training XGBoost baseline (n_estimators=%d, max_depth=%d) ...",
             XGB_PARAMS["n_estimators"], XGB_PARAMS["max_depth"])
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    train_secs = time.time() - t0
    log.info("Training time: %.1f s", train_secs)

    # --- Predict (probabilities) ---------------------------------------------
    log.info("Predicting on test set ...")
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Headline metrics ----------------------------------------------------
    auc_pr = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    log.info("=" * 70)
    log.info("HEADLINE METRICS")
    log.info("AUC-PR  (primary) : %.6f", auc_pr)
    log.info("ROC-AUC (secondary): %.6f", roc_auc)
    log.info(
        "Note: a random classifier scores AUC-PR ≈ %.4f at this base rate.",
        y_test.sum() / len(y_test),
    )

    # --- Best-F1 threshold ---------------------------------------------------
    best_f1_info = find_best_f1_threshold(y_test, y_prob)
    best_t = best_f1_info["best_threshold"]

    log.info("-" * 70)
    log.info("BEST-F1 THRESHOLD ANALYSIS")
    log.info("Threshold: %.4f", best_t)
    log.info("F1: %.4f | Precision: %.4f | Recall: %.4f",
             best_f1_info["best_f1"],
             best_f1_info["precision_at_best"],
             best_f1_info["recall_at_best"])
    log.info("(swept %s thresholds from precision-recall curve)",
             f"{best_f1_info['n_thresholds_swept']:,}")

    # --- Classification report at best threshold and at 0.5 ------------------
    y_pred_best = (y_prob >= best_t).astype(int)
    y_pred_50 = (y_prob >= 0.5).astype(int)

    log.info("-" * 70)
    log.info("Classification report @ threshold=%.4f (best F1):", best_t)
    log.info("\n%s", classification_report(y_test, y_pred_best, digits=4))

    log.info("Classification report @ threshold=0.5 (default):")
    log.info("\n%s", classification_report(y_test, y_pred_50, digits=4))

    # --- Threshold sweep table -----------------------------------------------
    sweep = threshold_sweep_summary(y_test, y_prob)
    log.info("-" * 70)
    log.info("THRESHOLD SWEEP (precision-recall tradeoff):")
    log.info("%-12s %-12s %-12s %-12s", "Threshold", "Precision", "Recall", "F1")
    for row in sweep:
        log.info("%-12.4f %-12.4f %-12.4f %-12.4f",
                 row["threshold"], row["precision"], row["recall"], row["f1"])

    # --- Single-window CPU latency -------------------------------------------
    log.info("-" * 70)
    log.info("Measuring single-window CPU inference latency ...")
    latency = measure_latency(model, X_test[0])
    log.info("Latency (1000 runs): mean=%.3f ms, median=%.3f ms, p95=%.3f ms, p99=%.3f ms",
             latency["mean_ms"], latency["median_ms"],
             latency["p95_ms"], latency["p99_ms"])

    # --- Peak memory ---------------------------------------------------------
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_mem / 1e6
    log.info("-" * 70)
    log.info("Peak memory (tracemalloc): %.1f MB", peak_mb)

    # --- Save model ----------------------------------------------------------
    model.save_model(str(MODEL_PATH))
    log.info("Model saved: %s (%.1f MB)",
             MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)

    # --- Save metrics --------------------------------------------------------
    metrics = {
        "model": "XGBoost baseline",
        "phase": "Phase 0 — baseline defender",
        "data": {
            "train_windows": int(X_train.shape[0]),
            "train_positives": int(y_train.sum()),
            "train_positive_rate": round(float(y_train.sum() / len(y_train)), 6),
            "test_windows": int(X_test.shape[0]),
            "test_positives": int(y_test.sum()),
            "test_positive_rate": round(float(y_test.sum() / len(y_test)), 6),
            "features_per_window": int(X_train.shape[1]),
            "scale_pos_weight": round(spw, 1),
        },
        "headline_metrics": {
            "AUC_PR": round(auc_pr, 6),
            "ROC_AUC": round(roc_auc, 6),
            "note": "AUC-PR is the primary metric. At 0.12% base rate, "
                    "ROC-AUC is inflated by the trivially high TNR.",
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
        "peak_memory_mb": round(peak_mb, 1),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Metrics saved: %s", METRICS_PATH)

    log.info("=" * 70)
    log.info("STEP 4 COMPLETE — Phase 0 baseline established.")
    log.info("Every future result compares against: %s", METRICS_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
