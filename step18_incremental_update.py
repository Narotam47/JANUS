"""Step 18 — PSI-triggered XGBoost warm-start update with a validation gate.

Flow: check step 17's drift trigger -> if it fired, warm-start-continue
training `model_enriched.json` on a "recent data" proxy -> evaluate both
the current and updated model on the SAME held-out test set -> promote the
update only if it doesn't regress AUC-PR or F1 beyond a small tolerance.

HONEST CONTEXT (read before trusting any number here as a real production
decision): the "recent data" used for warm-start fine-tuning is the
chronological TAIL of the TRAINING set (the last 20% of training windows
by row position in windows_train.npy) — not genuinely new, newly-labeled
live data. Step 17's own honest-context note applies here too: this is one
dataset's chronological split, not an independent later population. In
production, "recent data" would be newly labeled transactions collected
from the live stream since the last training run. The warm-start MECHANISM
(XGBoost's `xgb_model` continuation, the validation gate, the promote/
reject artifact contract) is unaffected by that distinction — this run
demonstrates the machinery works end-to-end, not that a real-world update
was warranted.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from step13_shap_explain import build_feature_names, compute_window_feature_indices
from step17_drift_monitor import should_retrain

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")

CLEAN_TRAIN: Final[Path] = ARTIFACTS_DIR / "clean_train.parquet"
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"
WINDOWS_TRAIN: Final[Path] = ARTIFACTS_DIR / "windows_train.npy"
LABELS_TRAIN: Final[Path] = ARTIFACTS_DIR / "labels_train.npy"
WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
LABELS_TEST: Final[Path] = ARTIFACTS_DIR / "labels_test.npy"
USER_FEATURES_TRAIN: Final[Path] = ARTIFACTS_DIR / "user_features_train.parquet"
USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TRAIN: Final[Path] = ARTIFACTS_DIR / "merchant_features_train.parquet"
MERCH_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TRAIN: Final[Path] = ARTIFACTS_DIR / "flags_train.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

MODEL_ENRICHED: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
METRICS_ENRICHED: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"
PSI_REPORT: Final[Path] = ARTIFACTS_DIR / "psi_report.json"

MODEL_UPDATED_PATH: Final[Path] = ARTIFACTS_DIR / "model_updated.json"
METRICS_UPDATED_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_updated.json"
REJECTED_PATH: Final[Path] = ARTIFACTS_DIR / "update_rejected.json"

RECENT_FRACTION: Final[float] = 0.20
N_ADDITIONAL_ROUNDS: Final[int] = 75

AUC_PR_MAX_DROP: Final[float] = 0.005
F1_MAX_DROP: Final[float] = 0.01

HONEST_CONTEXT: Final[str] = (
    "The 'recent data' used for warm-start fine-tuning is the chronological tail of the "
    "TRAINING set (last 20% of training windows by row position) — not genuinely new live "
    "data. In production this would be newly labeled transactions from the live stream since "
    "the last training run. The warm-start mechanism, validation gate, and promote/reject "
    "artifact contract are unaffected by that distinction; this run demonstrates the machinery "
    "works end-to-end, not that a real-world update was warranted."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step18")


# === Data loading =============================================================

def load_recent_training_slice(fraction: float = RECENT_FRACTION) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Last `fraction` of training windows by row position — the 'recent
    data' proxy for warm-start fine-tuning. A contiguous tail slice, so
    this is a plain sequential read off the 6.2GB memmap, same spirit as
    step 8's loading approach, just a slice instead of the whole file.
    """
    idx = compute_window_feature_indices(CLEAN_TRAIN)
    n_windows = len(idx)
    n_recent = int(n_windows * fraction)
    log.info("Training windows: %s total; using most recent %s (%.0f%%) as the 'recent data' proxy.",
              f"{n_windows:,}", f"{n_recent:,}", fraction * 100)

    user_feats = pd.read_parquet(USER_FEATURES_TRAIN).iloc[idx].reset_index(drop=True).iloc[-n_recent:].reset_index(drop=True)
    merch_feats = pd.read_parquet(MERCH_FEATURES_TRAIN).iloc[idx].reset_index(drop=True).iloc[-n_recent:].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TRAIN).iloc[idx].reset_index(drop=True).iloc[-n_recent:].reset_index(drop=True)
    feature_names = build_feature_names(
        CLEAN_TRAIN, list(user_feats.columns), list(merch_feats.columns), list(flags.columns),
    )

    X_win = np.load(str(WINDOWS_TRAIN), mmap_mode="r")[-n_recent:]
    y = np.asarray(np.load(str(LABELS_TRAIN), mmap_mode="r")[-n_recent:])

    enriched = np.hstack([
        user_feats.values.astype(np.float32),
        merch_feats.values.astype(np.float32),
        flags.values.astype(np.float32),
    ])
    X = np.hstack([np.asarray(X_win, dtype=np.float32), enriched])
    assert X.shape[1] == 243 and len(feature_names) == 243
    return X, y, feature_names


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    """The canonical held-out test set — unchanged from steps 8/11/12/13.
    Used ONLY for gate evaluation, never for warm-start fitting."""
    idx = compute_window_feature_indices(CLEAN_TEST)
    y_test = np.asarray(np.load(str(LABELS_TEST), mmap_mode="r"))

    user_feats = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(MERCH_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TEST).iloc[idx].reset_index(drop=True)

    X_win = np.load(str(WINDOWS_TEST), mmap_mode="r")
    enriched = np.hstack([
        user_feats.values.astype(np.float32),
        merch_feats.values.astype(np.float32),
        flags.values.astype(np.float32),
    ])
    X = np.hstack([np.asarray(X_win, dtype=np.float32), enriched])
    return X, y_test


def evaluate(model: XGBClassifier, X: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "auc_pr": float(average_precision_score(y, prob)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


# === Main =====================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("STEP 18 — PSI-TRIGGERED WARM-START UPDATE")
    log.info("=" * 70)

    if not PSI_REPORT.exists():
        log.error("Missing %s — run step 17 first.", PSI_REPORT)
        return

    with open(PSI_REPORT) as f:
        psi_report = json.load(f)
    psi_by_feature = {r["feature"]: r["psi"] for r in psi_report["per_feature"]}

    decision, reason = should_retrain(psi_by_feature)
    log.info("should_retrain() -> %s", decision)
    log.info("  Reason: %s", reason)

    if not decision:
        log.info("-" * 70)
        log.info("No drift detected, skipping update.")
        log.info("(Not the branch this run takes: step 17 found should_retrain()==True, driven "
                  "by mechanical population-volume artifacts of the chronological train/test "
                  "split — see step 17's report. This early-exit path is here so the script "
                  "behaves correctly whenever PSI comes back clean, e.g. after a future retrain "
                  "resets the reference distribution.)")
        log.info("=" * 70)
        return

    log.info("-" * 70)
    log.info("Drift trigger fired — proceeding with warm-start update.")
    log.info("HONEST CONTEXT: %s", HONEST_CONTEXT)

    with open(METRICS_ENRICHED) as f:
        base_metrics = json.load(f)
    current_threshold = base_metrics["best_f1_threshold"]["best_threshold"]
    xgb_params = dict(base_metrics["training"]["xgb_params"])
    log.info("-" * 70)
    log.info("Base hyperparameters (from metrics_enriched.json, reused unchanged): %s", xgb_params)
    log.info("Current best-F1 threshold (held FIXED for the gate comparison): %.4f", current_threshold)

    log.info("-" * 70)
    log.info("Loading recent training slice (warm-start fine-tuning data) ...")
    X_recent, y_recent, _ = load_recent_training_slice()
    log.info("Recent slice: %s windows, %s fraud (%.4f%%)",
              f"{len(y_recent):,}", f"{int(y_recent.sum()):,}", y_recent.mean() * 100)

    log.info("Loading held-out test set (gate evaluation only, unchanged from steps 8-13) ...")
    X_test, y_test = load_test_set()
    log.info("Test set: %s windows, %s fraud (%.4f%%)",
              f"{len(y_test):,}", f"{int(y_test.sum()):,}", y_test.mean() * 100)

    log.info("-" * 70)
    log.info("Evaluating CURRENT model (model_enriched.json) on the test set ...")
    current_model = XGBClassifier()
    current_model.load_model(str(MODEL_ENRICHED))
    current_eval = evaluate(current_model, X_test, y_test, current_threshold)
    log.info("  AUC-PR=%.6f  F1@%.4f=%.6f  precision=%.4f  recall=%.4f",
              current_eval["auc_pr"], current_threshold, current_eval["f1"],
              current_eval["precision"], current_eval["recall"])
    n_trees_before = current_model.get_booster().num_boosted_rounds()

    log.info("-" * 70)
    log.info("Warm-starting from model_enriched.json (%d existing trees), training %d "
              "ADDITIONAL rounds on the recent slice via xgb_model continuation — NOT "
              "retraining from scratch ...", n_trees_before, N_ADDITIONAL_ROUNDS)
    updated_params = dict(xgb_params)
    updated_params["n_estimators"] = N_ADDITIONAL_ROUNDS
    updated_model = XGBClassifier(**updated_params)
    t0 = time.perf_counter()
    updated_model.fit(X_recent, y_recent, xgb_model=str(MODEL_ENRICHED))
    train_time = time.perf_counter() - t0
    n_trees_after = updated_model.get_booster().num_boosted_rounds()
    log.info("Warm-start training complete in %.1fs. Trees: %d -> %d (+%d).",
              train_time, n_trees_before, n_trees_after, n_trees_after - n_trees_before)
    if n_trees_after != n_trees_before + N_ADDITIONAL_ROUNDS:
        log.warning("Expected +%d trees, got +%d — warm-start continuation may not have "
                     "behaved as expected; proceeding with evaluation regardless.",
                     N_ADDITIONAL_ROUNDS, n_trees_after - n_trees_before)

    log.info("-" * 70)
    log.info("Evaluating UPDATED model on the SAME held-out test set ...")
    updated_eval = evaluate(updated_model, X_test, y_test, current_threshold)
    log.info("  AUC-PR=%.6f  F1@%.4f=%.6f  precision=%.4f  recall=%.4f",
              updated_eval["auc_pr"], current_threshold, updated_eval["f1"],
              updated_eval["precision"], updated_eval["recall"])

    log.info("-" * 70)
    log.info("VALIDATION GATE")
    auc_pr_drop = current_eval["auc_pr"] - updated_eval["auc_pr"]
    f1_drop = current_eval["f1"] - updated_eval["f1"]
    auc_pr_ok = auc_pr_drop <= AUC_PR_MAX_DROP
    f1_ok = f1_drop <= F1_MAX_DROP
    log.info("  AUC-PR : %.6f -> %.6f   (change=%+.6f, max allowed drop=%.3f)  -> %s",
              current_eval["auc_pr"], updated_eval["auc_pr"], -auc_pr_drop, AUC_PR_MAX_DROP,
              "PASS" if auc_pr_ok else "FAIL")
    log.info("  F1@%.4f: %.6f -> %.6f   (change=%+.6f, max allowed drop=%.3f)  -> %s",
              current_threshold, current_eval["f1"], updated_eval["f1"], -f1_drop, F1_MAX_DROP,
              "PASS" if f1_ok else "FAIL")

    promote = auc_pr_ok and f1_ok
    warm_start_record = {
        "base_model": str(MODEL_ENRICHED),
        "additional_rounds": N_ADDITIONAL_ROUNDS,
        "trees_before": n_trees_before,
        "trees_after": n_trees_after,
        "recent_slice_fraction": RECENT_FRACTION,
        "recent_slice_size": int(len(y_recent)),
        "recent_slice_fraud_count": int(y_recent.sum()),
        "recent_slice_fraud_rate": float(y_recent.mean()),
        "train_time_seconds": round(train_time, 2),
        "xgb_params": updated_params,
    }
    gate_record = {
        "current_model_metrics": current_eval,
        "updated_model_metrics": updated_eval,
        "threshold_used": current_threshold,
        "auc_pr_drop": round(auc_pr_drop, 6),
        "f1_drop": round(f1_drop, 6),
        "auc_pr_max_drop": AUC_PR_MAX_DROP,
        "f1_max_drop": F1_MAX_DROP,
    }

    log.info("-" * 70)
    if promote:
        log.info("GATE PASSED — promoting updated model.")
        updated_model.save_model(str(MODEL_UPDATED_PATH))
        REJECTED_PATH.unlink(missing_ok=True)  # keep artifacts/ from showing contradictory outcomes

        metrics_updated = {
            "description": "Warm-start incremental update of model_enriched.json, triggered by "
                            "step 17's PSI drift monitor. PROMOTED — validation gate passed.",
            "honest_context": HONEST_CONTEXT,
            "trigger": {"should_retrain_decision": decision, "reason": reason},
            "warm_start": warm_start_record,
            "gate": {**gate_record, "decision": "promoted"},
        }
        with open(METRICS_UPDATED_PATH, "w") as f:
            json.dump(metrics_updated, f, indent=2, default=str)
        log.info("Saved: %s", MODEL_UPDATED_PATH)
        log.info("Saved: %s", METRICS_UPDATED_PATH)
        log.info("model_enriched.json is UNCHANGED on disk; model_updated.json is the newly "
                  "promoted candidate for whatever deployment step swaps it in.")
    else:
        failed = []
        if not auc_pr_ok:
            failed.append(f"AUC-PR dropped {auc_pr_drop:.6f} (allowed <= {AUC_PR_MAX_DROP})")
        if not f1_ok:
            failed.append(f"F1@{current_threshold:.4f} dropped {f1_drop:.6f} (allowed <= {F1_MAX_DROP})")
        log.warning("GATE FAILED — keeping current model. Reasons: %s", "; ".join(failed))

        MODEL_UPDATED_PATH.unlink(missing_ok=True)
        METRICS_UPDATED_PATH.unlink(missing_ok=True)  # keep artifacts/ from showing contradictory outcomes

        rejection = {
            "description": "Warm-start incremental update REJECTED — validation gate failed. "
                            "model_enriched.json remains the active model; no promotion occurred.",
            "honest_context": HONEST_CONTEXT,
            "trigger": {"should_retrain_decision": decision, "reason": reason},
            "warm_start": warm_start_record,
            "gate": {**gate_record, "decision": "rejected", "failed_conditions": failed},
        }
        with open(REJECTED_PATH, "w") as f:
            json.dump(rejection, f, indent=2, default=str)
        log.info("Saved: %s", REJECTED_PATH)
        log.info("model_enriched.json remains the active model — no promotion occurred.")

    log.info("=" * 70)


if __name__ == "__main__":
    main()
