"""Step 10 — Confidence score for JANUS fraud predictions.

Quantifies how much the model should be trusted on a given transaction.
Returns a scalar in [0, 1] and a short reason string naming the dominant
factor, intended to feed the SHAP rationale panel.

Five components, each in [0, 1], weighted-averaged into the final score:

  certainty    (0.35)  Distance from the decision threshold.  A prediction
                        right at the boundary is maximally uncertain.
  history      (0.15)  Depth of the user's transaction history.  More prior
                        transactions → stabler user-profile features.
  familiarity  (0.20)  Merchant and MCC visit counts for this user.  A first
                        visit to an unknown merchant is inherently harder to
                        classify.
  novelty      (0.20)  Penalty when first-occurrence flags fire.  first_mcc
                        gets the heaviest penalty (step 8b showed FP regression
                        there).
  completeness (0.10)  Fraction of enriched features that are non-NaN.  Mostly
                        full in the main-model regime, but conceptually important
                        at system boundaries.

The dominant-factor reason is the component farthest below 1.0.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")

WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
LABELS_TEST: Final[Path] = ARTIFACTS_DIR / "labels_test.npy"
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"

USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

MODEL_ENRICHED: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
METRICS_ENRICHED: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_confidence.json"

USER_COL: Final[str] = "User"
WINDOW_SIZE: Final[int] = 10

# --- Component parameters ---------------------------------------------------
W_CERTAINTY: Final[float] = 0.35
W_HISTORY: Final[float] = 0.15
W_FAMILIARITY: Final[float] = 0.20
W_NOVELTY: Final[float] = 0.20
W_COMPLETENESS: Final[float] = 0.10

CERT_BELOW_SCALE: Final[float] = 0.30
CERT_ABOVE_SCALE: Final[float] = 0.04
HISTORY_FULL: Final[float] = 200.0
FAM_MERCH_LOG: Final[float] = math.log1p(50)
FAM_MCC_LOG: Final[float] = math.log1p(30)
NOV_FIRST_MCC: Final[float] = 0.30
NOV_FIRST_CITY_STATE: Final[float] = 0.15
NOV_FIRST_CHANNEL: Final[float] = 0.10
COMP_SCALE: Final[float] = 5.0

_COMPONENT_NAMES: Final[list[str]] = [
    "certainty", "history", "familiarity", "novelty", "completeness",
]
_WEIGHTS: Final[list[float]] = [
    W_CERTAINTY, W_HISTORY, W_FAMILIARITY, W_NOVELTY, W_COMPLETENESS,
]
_REASONS: Final[dict[str, str]] = {
    "certainty": "prediction near decision threshold",
    "history": "thin user history",
    "familiarity": "unfamiliar merchant or category",
    "novelty": "first-time merchant/location/channel",
    "completeness": "missing feature data",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step10")


# === Reusable confidence function ===========================================

def confidence(
    prob: float,
    threshold: float,
    user_tx_count: float,
    merchant_user_count: float,
    mcc_user_count: float,
    first_mcc: int,
    first_city_state: int,
    first_channel: int,
    n_nan_features: int,
    total_features: int = 243,
) -> tuple[float, str]:
    """Score how much to trust the model on this transaction.

    Parameters
    ----------
    prob : float
        Model's predicted fraud probability.
    threshold : float
        Decision threshold (e.g. 0.9497 from best-F1 sweep).
    user_tx_count : float
        Number of prior transactions for this user.
    merchant_user_count : float
        Times this user has visited this merchant before.
    mcc_user_count : float
        Times this user has transacted at this MCC before.
    first_mcc, first_city_state, first_channel : int
        First-occurrence flags (1 = first time, 0 = repeat).
    n_nan_features : int
        Count of NaN values across the model's enriched features.
    total_features : int
        Total model features (default 243).

    Returns
    -------
    score : float
        Confidence in [0, 1].  Higher = model is more trustworthy here.
    reason : str
        Short string naming the dominant factor driving confidence down.
    """
    if prob < threshold:
        certainty = min((threshold - prob) / CERT_BELOW_SCALE, 1.0)
    else:
        certainty = min((prob - threshold) / CERT_ABOVE_SCALE, 1.0)

    history = min(user_tx_count / HISTORY_FULL, 1.0)

    fam_merch = min(math.log1p(merchant_user_count) / FAM_MERCH_LOG, 1.0)
    fam_mcc = min(math.log1p(mcc_user_count) / FAM_MCC_LOG, 1.0)
    familiarity = min(fam_merch, fam_mcc)

    novelty = max(
        1.0
        - NOV_FIRST_MCC * float(first_mcc)
        - NOV_FIRST_CITY_STATE * float(first_city_state)
        - NOV_FIRST_CHANNEL * float(first_channel),
        0.0,
    )

    nan_frac = n_nan_features / max(total_features, 1)
    completeness = max(1.0 - nan_frac * COMP_SCALE, 0.0)

    components = {
        "certainty": certainty,
        "history": history,
        "familiarity": familiarity,
        "novelty": novelty,
        "completeness": completeness,
    }

    score = sum(_WEIGHTS[i] * components[n] for i, n in enumerate(_COMPONENT_NAMES))

    weakest = min(components, key=components.get)
    if components[weakest] >= 0.85:
        reason = "all signals strong"
    else:
        reason = _REASONS[weakest]

    return round(score, 4), reason


# === Vectorized batch computation (for validation) ==========================

def confidence_batch(
    prob: np.ndarray,
    threshold: float,
    user_tx_count: np.ndarray,
    merchant_user_count: np.ndarray,
    mcc_user_count: np.ndarray,
    first_mcc: np.ndarray,
    first_city_state: np.ndarray,
    first_channel: np.ndarray,
    n_nan: np.ndarray,
    total_features: int = 243,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized confidence computation.

    Returns (scores array, component_matrix [N×5] for dominant-factor analysis).
    """
    certainty = np.where(
        prob < threshold,
        np.minimum((threshold - prob) / CERT_BELOW_SCALE, 1.0),
        np.minimum((prob - threshold) / CERT_ABOVE_SCALE, 1.0),
    )

    history = np.minimum(user_tx_count / HISTORY_FULL, 1.0)

    fam_merch = np.minimum(np.log1p(merchant_user_count) / FAM_MERCH_LOG, 1.0)
    fam_mcc = np.minimum(np.log1p(mcc_user_count) / FAM_MCC_LOG, 1.0)
    familiarity = np.minimum(fam_merch, fam_mcc)

    novelty = np.clip(
        1.0
        - NOV_FIRST_MCC * first_mcc
        - NOV_FIRST_CITY_STATE * first_city_state
        - NOV_FIRST_CHANNEL * first_channel,
        0.0, 1.0,
    )

    nan_frac = n_nan / max(total_features, 1)
    completeness = np.clip(1.0 - nan_frac * COMP_SCALE, 0.0, 1.0)

    components = np.column_stack([
        certainty, history, familiarity, novelty, completeness,
    ])

    scores = (
        W_CERTAINTY * certainty
        + W_HISTORY * history
        + W_FAMILIARITY * familiarity
        + W_NOVELTY * novelty
        + W_COMPLETENESS * completeness
    )

    return scores, components


# === Validation =============================================================

def compute_window_feature_indices(clean_path: Path) -> np.ndarray:
    users = pd.read_parquet(clean_path, columns=[USER_COL])[USER_COL]
    indices: list[np.ndarray] = []
    offset = 0
    for _, group in users.groupby(users, sort=False):
        n = len(group)
        if n >= WINDOW_SIZE:
            indices.append(np.arange(offset + WINDOW_SIZE - 1, offset + n))
        offset += n
    return np.concatenate(indices)


def main() -> None:
    for p in (WINDOWS_TEST, LABELS_TEST, CLEAN_TEST,
              USER_FEATURES_TEST, MERCH_FEATURES_TEST, FLAGS_TEST,
              MODEL_ENRICHED, METRICS_ENRICHED):
        if not p.exists():
            log.error("Missing %s — run prior steps first.", p)
            return

    # ---- Load and align ----------------------------------------------------
    log.info("Loading data and aligning to windows ...")
    idx = compute_window_feature_indices(CLEAN_TEST)
    y_test = np.load(str(LABELS_TEST), mmap_mode="r")
    n_test = len(y_test)
    assert len(idx) == n_test

    user_feats = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(MERCH_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TEST).iloc[idx].reset_index(drop=True)

    with open(METRICS_ENRICHED) as f:
        threshold = json.load(f)["best_f1_threshold"]["best_threshold"]
    log.info("Decision threshold: %.4f", threshold)

    # ---- Predict with enriched model ---------------------------------------
    log.info("Building merged test array and predicting ...")
    model = XGBClassifier()
    model.load_model(str(MODEL_ENRICHED))

    X_win = np.load(str(WINDOWS_TEST), mmap_mode="r")
    u = user_feats.values.astype(np.float32)
    m = merch_feats.values.astype(np.float32)
    fl_arr = flags.values.astype(np.float32)
    enriched = np.hstack([u, m, fl_arr])
    X = np.hstack([X_win, enriched])
    del X_win, enriched, u, m, fl_arr

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    del X, model
    log.info("Predictions complete: %d flagged as fraud.", pred.sum())

    # ---- Compute NaN counts per window ------------------------------------
    nan_user = user_feats.isna().sum(axis=1).values
    nan_merch = merch_feats.isna().sum(axis=1).values
    nan_flags = flags.isna().sum(axis=1).values
    n_nan = nan_user + nan_merch + nan_flags
    del nan_user, nan_merch, nan_flags

    # ---- Compute confidence scores -----------------------------------------
    log.info("Computing confidence scores ...")
    scores, components = confidence_batch(
        prob=prob,
        threshold=threshold,
        user_tx_count=user_feats["user_tx_count"].values,
        merchant_user_count=user_feats["merchant_user_count"].values,
        mcc_user_count=user_feats["mcc_user_count"].values,
        first_mcc=flags["first_mcc"].values,
        first_city_state=flags["first_city_state"].values,
        first_channel=flags["first_channel"].values,
        n_nan=n_nan,
    )

    log.info("Confidence distribution: mean=%.3f, p5=%.3f, p25=%.3f, "
             "p50=%.3f, p75=%.3f, p95=%.3f",
             scores.mean(), np.percentile(scores, 5),
             np.percentile(scores, 25), np.percentile(scores, 50),
             np.percentile(scores, 75), np.percentile(scores, 95))

    # ---- Component distribution -------------------------------------------
    log.info("-" * 70)
    log.info("COMPONENT DISTRIBUTIONS")
    for i, name in enumerate(_COMPONENT_NAMES):
        c = components[:, i]
        log.info("  %-15s  mean=%.3f  <0.5: %6.2f%%  <0.3: %6.2f%%  =1.0: %6.1f%%",
                 name, c.mean(),
                 (c < 0.5).mean() * 100,
                 (c < 0.3).mean() * 100,
                 (c == 1.0).mean() * 100)

    # ---- Dominant factor distribution -------------------------------------
    weakest_idx = components.argmin(axis=1)
    log.info("-" * 70)
    log.info("DOMINANT FACTOR (lowest component per transaction)")
    for i, name in enumerate(_COMPONENT_NAMES):
        pct = (weakest_idx == i).mean() * 100
        log.info("  %-15s  %.1f%%", name, pct)

    # ==== VALIDATION TABLE ==================================================
    log.info("=" * 70)
    log.info("VALIDATION: MODEL PERFORMANCE BY CONFIDENCE BUCKET")
    log.info("(enriched model at threshold %.4f)", threshold)
    log.info("")

    bucket_edges = [0.0, 0.40, 0.60, 0.75, 0.90, 1.01]
    bucket_labels = ["very low [0,.4)", "low [.4,.6)", "moderate [.6,.75)",
                     "high [.75,.9)", "very high [.9,1]"]

    log.info("%-20s %10s %6s  %8s %8s %8s  %7s",
             "Bucket", "Windows", "Fraud",
             "Precis", "Recall", "F1", "FP rate")

    validation: list[dict] = []
    for i in range(len(bucket_labels)):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        fraud_n = int(y_test[mask].sum())

        if n == 0:
            log.info("%-20s %10s %6s  %8s %8s %8s  %7s",
                     bucket_labels[i], "0", "—", "—", "—", "—", "—")
            validation.append({
                "bucket": bucket_labels[i], "lo": lo, "hi": hi,
                "n_windows": 0, "n_fraud": 0,
            })
            continue

        y_b = y_test[mask]
        p_b = pred[mask]

        flagged = int(p_b.sum())
        tp = int(((p_b == 1) & (y_b == 1)).sum())
        fp = int(((p_b == 1) & (y_b == 0)).sum())
        fn = int(((p_b == 0) & (y_b == 1)).sum())

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1) if fraud_n > 0 else None
        f1 = 2 * prec * rec / max(prec + rec, 1e-9) if rec is not None else None
        n_legit = n - fraud_n
        fp_rate = fp / max(n_legit, 1)

        log.info("%-20s %10s %6d  %7.1f%% %7.1f%% %7.1f%%  %6.3f%%",
                 bucket_labels[i], f"{n:,}", fraud_n,
                 prec * 100,
                 rec * 100 if rec is not None else 0,
                 f1 * 100 if f1 is not None else 0,
                 fp_rate * 100)

        validation.append({
            "bucket": bucket_labels[i], "lo": lo, "hi": hi,
            "n_windows": n, "n_fraud": fraud_n,
            "n_flagged": flagged, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4) if prec else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "fp_rate": round(fp_rate, 6),
        })

    # Monotonicity check
    precs = [v.get("precision") for v in validation if v.get("precision") is not None]
    if len(precs) >= 3:
        increasing = all(precs[i] <= precs[i + 1] + 0.01
                         for i in range(len(precs) - 1))
        if increasing:
            log.info("")
            log.info("→ MONOTONIC: precision increases with confidence bucket.")
            log.info("  The confidence score discriminates real model quality.")
            mono_verdict = "monotonic"
        else:
            log.info("")
            log.info("→ NON-MONOTONIC: precision does not strictly increase.")
            log.info("  Confidence captures some signal but is not fully calibrated.")
            mono_verdict = "non_monotonic"
    else:
        mono_verdict = "insufficient_data"

    # ==== FIRST-MCC CHECK ===================================================
    log.info("=" * 70)
    log.info("FIRST-MCC CHECK (step 8b regression cases)")
    log.info("")

    first_mcc_mask = (flags["first_mcc"] == 1).values
    legit = y_test == 0
    first_mcc_legit = first_mcc_mask & legit

    n_fml = int(first_mcc_legit.sum())
    scores_fml = scores[first_mcc_legit]

    log.info("First-MCC legitimate transactions: %s", f"{n_fml:,}")
    log.info("Confidence distribution:")
    log.info("  mean=%.3f, p10=%.3f, p25=%.3f, p50=%.3f",
             scores_fml.mean(),
             np.percentile(scores_fml, 10),
             np.percentile(scores_fml, 25),
             np.percentile(scores_fml, 50))

    # What buckets do they land in?
    log.info("")
    log.info("Bucket distribution of first-MCC legit transactions:")
    fml_bucket_dist: dict = {}
    for i in range(len(bucket_labels)):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        n_in = int(((scores_fml >= lo) & (scores_fml < hi)).sum())
        pct = n_in / max(n_fml, 1) * 100
        log.info("  %-24s  %5d  (%5.1f%%)", bucket_labels[i], n_in, pct)
        fml_bucket_dist[bucket_labels[i]] = {
            "count": n_in, "pct": round(pct, 1),
        }

    low_conf_fml = int((scores_fml < 0.60).sum())
    low_conf_pct = low_conf_fml / max(n_fml, 1) * 100
    log.info("")
    if low_conf_pct > 50:
        log.info("→ %d / %d (%.0f%%) of first-MCC legit transactions land in",
                 low_conf_fml, n_fml, low_conf_pct)
        log.info("  low-confidence buckets (<0.60). The confidence score correctly")
        log.info("  identifies these as candidates for step-up authentication.")
        fml_verdict = "correctly_low_confidence"
    elif low_conf_pct > 20:
        log.info("→ %.0f%% of first-MCC legit land below 0.60 confidence.", low_conf_pct)
        log.info("  Partial coverage — some first-MCC cases still get high confidence.")
        fml_verdict = "partial_coverage"
    else:
        log.info("→ Only %.0f%% of first-MCC legit land below 0.60.", low_conf_pct)
        log.info("  The confidence score does NOT reliably flag first-MCC cases.")
        fml_verdict = "insufficient_coverage"

    # Compare first-MCC FP rate by confidence
    log.info("")
    log.info("First-MCC legit FP rates by confidence:")
    fml_fp_by_conf: dict = {}
    for i in range(len(bucket_labels)):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask_b = first_mcc_legit & (scores >= lo) & (scores < hi)
        n_b = int(mask_b.sum())
        if n_b == 0:
            continue
        fp_b = int((pred[mask_b] == 1).sum())
        log.info("  %-24s  n=%5d  FP=%3d  (%.2f%%)",
                 bucket_labels[i], n_b, fp_b, fp_b / n_b * 100)
        fml_fp_by_conf[bucket_labels[i]] = {
            "n": n_b, "fp": fp_b, "fp_rate": round(fp_b / n_b, 4),
        }

    del user_feats, merch_feats, flags

    # ==== SINGLE-FUNCTION DEMO ==============================================
    log.info("=" * 70)
    log.info("FUNCTION DEMO — example calls")
    cases = [
        ("Normal legit (prob=0.01)", 0.01, threshold, 5000, 200, 300, 0, 0, 0, 0),
        ("Near threshold (prob=0.93)", 0.93, threshold, 5000, 200, 300, 0, 0, 0, 0),
        ("Fraud detected (prob=0.98)", 0.98, threshold, 5000, 200, 300, 0, 0, 0, 0),
        ("First MCC, legit (prob=0.4)", 0.40, threshold, 5000, 200, 300, 1, 0, 0, 0),
        ("New user, first merchant", 0.50, threshold, 15, 0, 0, 1, 1, 0, 0),
        ("Many NaN features", 0.20, threshold, 5000, 200, 300, 0, 0, 0, 10),
    ]

    for label, *args in cases:
        s, r = confidence(*args)
        log.info("  %-35s  score=%.3f  reason=%s", label, s, r)

    # ==== SAVE ==============================================================
    log.info("=" * 70)
    output = {
        "description": "Confidence score validation on the enriched model's test set.",
        "threshold": threshold,
        "component_weights": dict(zip(_COMPONENT_NAMES, _WEIGHTS)),
        "score_distribution": {
            "mean": round(float(scores.mean()), 4),
            "p5": round(float(np.percentile(scores, 5)), 4),
            "p25": round(float(np.percentile(scores, 25)), 4),
            "p50": round(float(np.percentile(scores, 50)), 4),
            "p75": round(float(np.percentile(scores, 75)), 4),
            "p95": round(float(np.percentile(scores, 95)), 4),
        },
        "validation_by_bucket": validation,
        "monotonicity": mono_verdict,
        "first_mcc_check": {
            "n_first_mcc_legit": n_fml,
            "confidence_mean": round(float(scores_fml.mean()), 4),
            "confidence_p25": round(float(np.percentile(scores_fml, 25)), 4),
            "bucket_distribution": fml_bucket_dist,
            "low_conf_pct": round(low_conf_pct, 1),
            "verdict": fml_verdict,
            "fp_by_confidence": fml_fp_by_conf,
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
