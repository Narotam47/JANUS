"""Step 29 — ablation studies: does each component earn its place?

Two families of ablation, evaluated against the same held-out test set as
every other step:

FEATURE ABLATIONS — zero out a feature group in the ALREADY-TRAINED
enriched model's input, no retraining. This is explicitly an
approximation, not a clean causal test: a model retrained without a
feature group could learn to compensate via correlated features (e.g.
`merchant_user_count` and `mcc_user_count` are correlated — zeroing one
while the model still sees the other is not the same as never having had
either). It measures "how much does the trained model currently RELY on
this group's presence at inference time," not "how much predictive value
does this group fundamentally carry."

A second, sharper caveat, specific to zeroing (not raised if you don't
think about what zero actually means per feature): zero is NOT a neutral
"this information is absent" signal for most of these columns.
  * For flags (first_mcc, is_largest_ever, ...): 0 is the ordinary,
    majority category ("not a first occurrence") — zeroing is a
    reasonably clean ablation.
  * For counts (user_tx_count, merchant_user_count, merch_tx_count, ...):
    0 is an extreme, specific value ("brand new, zero prior visits") —
    zeroing does not simulate "no information," it simulates "every
    transaction looks brand new," which can itself be a strong (mis)signal.
  * For unscaled means/ratios (user_hist_mean, user_amount_to_max_ratio,
    ...): 0 is often outside the realistic range entirely (a user with a
    literal $0 historical average) — an extreme extrapolation, not a
    neutral removal.
This is reported plainly per ablation below, not glossed over.

ARCHITECTURE ABLATIONS — re-apply step 10/11's routing logic to the
SAME probability outputs with one component disabled, using the already-
validated `confidence_batch`/`route_batch` functions directly (no
reimplementation, no new risk of subtly inconsistent logic).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from xgboost import XGBClassifier

from step10_confidence import confidence_batch
from step11_routing import DEFAULT_CONF_STEPUP, route_batch
from step13_shap_explain import build_feature_names, compute_window_feature_indices

# --- Config -------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"
WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
LABELS_TEST: Final[Path] = ARTIFACTS_DIR / "labels_test.npy"
USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"
MODEL_ENRICHED: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
METRICS_ENRICHED: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "ablation_table.json"

WINDOW_FEATURE_COUNT: Final[int] = 210
RARE_ZSCORE_THRESHOLD: Final[float] = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step29")


# === Data loading =============================================================

def load_test_set() -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    idx = compute_window_feature_indices(CLEAN_TEST)
    y_test = np.asarray(np.load(str(LABELS_TEST), mmap_mode="r"))

    user_feats = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(MERCH_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TEST).iloc[idx].reset_index(drop=True)
    feature_names = build_feature_names(
        CLEAN_TEST, list(user_feats.columns), list(merch_feats.columns), list(flags.columns),
    )

    X_win = np.load(str(WINDOWS_TEST), mmap_mode="r")
    enriched = np.hstack([
        user_feats.values.astype(np.float32),
        merch_feats.values.astype(np.float32),
        flags.values.astype(np.float32),
    ])
    X_test = np.hstack([np.asarray(X_win, dtype=np.float32), enriched])
    return X_test, y_test, feature_names, user_feats, flags, merch_feats, list(user_feats.columns), list(merch_feats.columns)


def best_f1_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float, float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y, prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx]), float(f1s[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])


def rare_purchase_slice_masks(user_feats: pd.DataFrame, flags: pd.DataFrame) -> dict[str, np.ndarray]:
    """Slice membership is ALWAYS computed from the ORIGINAL (unablated)
    data — "is this transaction objectively a rare purchase" is a ground-
    truth property, independent of what a feature-ablated model can see."""
    amount_unusual = (user_feats["user_amount_zscore"] > RARE_ZSCORE_THRESHOLD).values
    new_mcc = (flags["first_mcc"] == 1).values
    new_location = (flags["first_city_state"] == 1).values
    return {
        "rare_any": amount_unusual | new_mcc | new_location,
        "amount_unusual": amount_unusual,
        "new_mcc": new_mcc,
        "new_location": new_location,
    }


def evaluate_variant(prob: np.ndarray, y: np.ndarray, rare_masks: dict[str, np.ndarray]) -> dict:
    auc_pr = float(average_precision_score(y, prob))
    thr, f1, prec, rec = best_f1_threshold(y, prob)
    pred = prob >= thr
    legit = y == 0

    rare_fp: dict[str, float] = {}
    for name, mask in rare_masks.items():
        slice_legit = mask & legit
        n_legit = int(slice_legit.sum())
        n_fp = int((pred & slice_legit).sum())
        rare_fp[name] = round(n_fp / max(n_legit, 1), 6)

    return {
        "auc_pr": round(auc_pr, 6),
        "best_threshold": round(thr, 6),
        "f1": round(f1, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "rare_purchase_fp_rate": rare_fp,
    }


# === Main =====================================================================

def main() -> None:
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 29 — ABLATION STUDIES")
    log.info("=" * 70)

    for p in (MODEL_ENRICHED, METRICS_ENRICHED):
        if not p.exists():
            log.error("Missing required artifact: %s", p)
            return

    log.info("Loading held-out test set ...")
    X_test, y_test, feature_names, user_feats, flags, merch_feats, user_cols, merch_cols = load_test_set()
    rare_masks = rare_purchase_slice_masks(user_feats, flags)
    position = user_feats["user_tx_count"].values.astype(int)
    log.info("Test set: %s windows, %s fraud (%.4f%%). Rare-purchase slice: %s legit transactions.",
              f"{len(y_test):,}", f"{int(y_test.sum()):,}", y_test.mean() * 100,
              f"{int((rare_masks['rare_any'] & (y_test == 0)).sum()):,}")

    model = XGBClassifier()
    model.load_model(str(MODEL_ENRICHED))
    with open(METRICS_ENRICHED) as f:
        enriched_recorded_threshold = json.load(f)["best_f1_threshold"]["best_threshold"]

    flags_cols = feature_names[WINDOW_FEATURE_COUNT + len(user_cols) + len(merch_cols):]
    user_idx = np.array([feature_names.index(c) for c in user_cols])
    merch_idx = np.array([feature_names.index(c) for c in merch_cols])
    flags_idx = np.array([feature_names.index(c) for c in flags_cols])
    enriched_idx = np.arange(WINDOW_FEATURE_COUNT, len(feature_names))
    assert len(user_idx) == 14 and len(merch_idx) == 9 and len(flags_idx) == 10

    # =========================================================================
    # FULL SYSTEM (reference)
    # =========================================================================
    log.info("-" * 70)
    log.info("Scoring FULL SYSTEM (enriched model, all 243 features) ...")
    prob_full = model.predict_proba(X_test)[:, 1]
    full_result = evaluate_variant(prob_full, y_test, rare_masks)
    log.info("  AUC-PR=%.6f  F1=%.6f  rare_any FP=%.4f%%",
              full_result["auc_pr"], full_result["f1"], full_result["rare_purchase_fp_rate"]["rare_any"] * 100)

    # =========================================================================
    # FEATURE ABLATIONS
    # =========================================================================
    log.info("=" * 70)
    log.info("FEATURE ABLATIONS (zero-out at evaluation time, NOT retrained)")

    feature_ablations = {
        "minus_user_profile (14 cols zeroed)": user_idx,
        "minus_merchant_features (9 cols zeroed)": merch_idx,
        "minus_first_occurrence_flags (10 cols zeroed)": flags_idx,
        "window_only (all 33 enriched cols zeroed)": enriched_idx,
    }

    feature_results: dict[str, dict] = {}
    for name, idx in feature_ablations.items():
        X_variant = X_test.copy()
        X_variant[:, idx] = 0.0
        prob_variant = model.predict_proba(X_variant)[:, 1]
        result = evaluate_variant(prob_variant, y_test, rare_masks)
        result["auc_pr_delta_vs_full"] = round(result["auc_pr"] - full_result["auc_pr"], 6)
        result["f1_delta_vs_full"] = round(result["f1"] - full_result["f1"], 6)
        feature_results[name] = result
        log.info("  %-46s AUC-PR=%.6f (%+.6f)  F1=%.6f (%+.6f)  rare_any FP=%.4f%%",
                  name, result["auc_pr"], result["auc_pr_delta_vs_full"],
                  result["f1"], result["f1_delta_vs_full"], result["rare_purchase_fp_rate"]["rare_any"] * 100)
        del X_variant

    with open(ARTIFACTS_DIR / "metrics_baseline.json") as f:
        step4_baseline_aucpr = json.load(f)["headline_metrics"]["AUC_PR"]
    window_only_aucpr = feature_results["window_only (all 33 enriched cols zeroed)"]["auc_pr"]
    log.info("-" * 70)
    log.info("Sanity check — 'window_only' zeroed-ablation (AUC-PR=%.4f) vs. step 4's actual "
              "separately-trained 210-feature baseline (AUC-PR=%.4f): delta=%+.4f. These are NOT "
              "expected to match exactly — zeroing enriched columns on a model TRAINED with them "
              "present is a different object than a model that never had them; the trees still "
              "encode splits calibrated assuming those features carry real information.",
              window_only_aucpr, step4_baseline_aucpr, window_only_aucpr - step4_baseline_aucpr)

    # =========================================================================
    # ARCHITECTURE ABLATIONS — ROUTING
    # =========================================================================
    log.info("=" * 70)
    log.info("ARCHITECTURE ABLATIONS (same probabilities, routing logic re-applied)")

    n_nan = (
        user_feats.isna().sum(axis=1).values
        + merch_feats.isna().sum(axis=1).values
        + flags.isna().sum(axis=1).values
    )
    confidence_score, _ = confidence_batch(
        prob=prob_full, threshold=enriched_recorded_threshold,
        user_tx_count=user_feats["user_tx_count"].values,
        merchant_user_count=user_feats["merchant_user_count"].values,
        mcc_user_count=user_feats["mcc_user_count"].values,
        first_mcc=flags["first_mcc"].values,
        first_city_state=flags["first_city_state"].values,
        first_channel=flags["first_channel"].values,
        n_nan=n_nan,
    )

    def _routing_stats(decisions: np.ndarray, label: str) -> dict:
        is_block = decisions == "block"
        is_stepup = decisions == "step_up" if "step_up" in decisions else np.zeros_like(is_block)
        legit = y_test == 0
        fraud = y_test == 1
        block_tp = int((is_block & fraud).sum())
        block_fp = int((is_block & legit).sum())
        block_precision = block_tp / max(block_tp + block_fp, 1)
        rare_legit = rare_masks["rare_any"] & legit
        rare_hard_blocked = int((is_block & rare_legit).sum())
        n_rare_legit = int(rare_legit.sum())
        new_mcc_legit = rare_masks["new_mcc"] & legit
        new_mcc_hard_blocked = int((is_block & new_mcc_legit).sum())
        n_new_mcc_legit = int(new_mcc_legit.sum())
        stats = {
            "n_block": int(is_block.sum()),
            "n_stepup": int(is_stepup.sum()),
            "n_approve": int((decisions == "approve").sum()),
            "block_tp": block_tp,
            "block_fp": block_fp,
            "block_precision": round(block_precision, 6),
            "rare_purchase_hard_block_rate": round(rare_hard_blocked / max(n_rare_legit, 1), 6),
            "new_mcc_hard_block_rate": round(new_mcc_hard_blocked / max(n_new_mcc_legit, 1), 6),
        }
        log.info("  %-38s blocks=%6s (prec=%.1f%%)  rare-purchase hard-block rate=%.3f%%  "
                  "new-mcc hard-block rate=%.3f%%",
                  label, f"{stats['n_block']:,}", block_precision * 100,
                  stats["rare_purchase_hard_block_rate"] * 100, stats["new_mcc_hard_block_rate"] * 100)
        return stats

    log.info("Full 3-way router (block/step-up/approve, confidence-aware):")
    decisions_full, _ = route_batch(prob_full, confidence_score, position)
    routing_full = _routing_stats(decisions_full, "full_router")

    log.info("Binary routing only (no step-up tier at all — block if prob>=threshold, else approve):")
    decisions_binary = np.where(prob_full >= enriched_recorded_threshold, "block", "approve")
    routing_binary = _routing_stats(decisions_binary, "binary_only")

    log.info("No confidence scoring (keeps cold-start policy, drops the confidence-based step-up "
              "carve-out for established users — conf_stepup=0.0):")
    decisions_noconf, _ = route_batch(prob_full, confidence_score, position, conf_stepup=0.0)
    routing_noconf = _routing_stats(decisions_noconf, "no_confidence")

    log.info("-" * 70)
    log.info("Step-up tier's contribution (full router vs binary-only):")
    log.info("  Rare-purchase hard-block rate: %.3f%% (binary) -> %.3f%% (full router), "
              "change=%+.3fpp", routing_binary["rare_purchase_hard_block_rate"] * 100,
              routing_full["rare_purchase_hard_block_rate"] * 100,
              (routing_full["rare_purchase_hard_block_rate"] - routing_binary["rare_purchase_hard_block_rate"]) * 100)
    log.info("  Block precision: %.1f%% (binary) -> %.1f%% (full router), change=%+.1fpp",
              routing_binary["block_precision"] * 100, routing_full["block_precision"] * 100,
              (routing_full["block_precision"] - routing_binary["block_precision"]) * 100)

    log.info("Confidence scoring's specific contribution (full router vs no-confidence):")
    log.info("  New-MCC hard-block rate (the step 8b/10 regression case): %.3f%% (no-confidence) "
              "-> %.3f%% (full router with confidence), change=%+.3fpp",
              routing_noconf["new_mcc_hard_block_rate"] * 100, routing_full["new_mcc_hard_block_rate"] * 100,
              (routing_full["new_mcc_hard_block_rate"] - routing_noconf["new_mcc_hard_block_rate"]) * 100)
    log.info("  Rare-purchase hard-block rate: %.3f%% (no-confidence) -> %.3f%% (full router), "
              "change=%+.3fpp", routing_noconf["rare_purchase_hard_block_rate"] * 100,
              routing_full["rare_purchase_hard_block_rate"] * 100,
              (routing_full["rare_purchase_hard_block_rate"] - routing_noconf["rare_purchase_hard_block_rate"]) * 100)
    log.info("  (AUC-PR/F1 are UNCHANGED across all three routing variants — routing doesn't "
              "touch the underlying model's probability output, only the decision layer built "
              "on top of it. The metrics that matter for these ablations are block precision and "
              "rare-purchase hard-block rate, not AUC-PR/F1.)")

    total_runtime = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("Total runtime: %.1fs", total_runtime)

    output = {
        "description": "Ablation studies: feature-group zero-out (no retraining) and routing-"
                        "architecture ablations, evaluated against the enriched model's real "
                        "probability outputs on the held-out test set.",
        "methodology_caveats": {
            "feature_ablation_is_approximation": (
                "Zeroing a feature group on the ALREADY-TRAINED enriched model measures how much "
                "the trained model currently relies on that group at inference time — not the "
                "group's fundamental predictive value in a model retrained without it. A retrained "
                "model could compensate via correlated features that remain present."
            ),
            "zero_is_not_neutral": (
                "Zero is not a 'this information is absent' signal for most of these columns: for "
                "flags, 0 is the ordinary majority category (a fairly clean ablation); for counts "
                "(user_tx_count, merchant_user_count, ...), 0 means 'brand new' — an extreme, "
                "specific value, not an absence of information; for unscaled means/ratios "
                "(user_hist_mean, user_amount_to_max_ratio, ...), 0 is often outside the realistic "
                "range altogether. Interpret magnitudes accordingly."
            ),
            "window_only_vs_step4_baseline": (
                f"The window_only ablation (AUC-PR={window_only_aucpr:.4f}) is NOT expected to "
                f"exactly reproduce step 4's separately-trained 210-feature baseline "
                f"(AUC-PR={step4_baseline_aucpr:.4f}, delta={window_only_aucpr - step4_baseline_aucpr:+.4f}) "
                "— it's the enriched model's trees, calibrated assuming enriched features carry "
                "real information, with those inputs forced to zero, not a model that never saw them."
            ),
        },
        "full_system": full_result,
        "feature_ablations": feature_results,
        "routing_ablations": {
            "full_router": routing_full,
            "binary_only": routing_binary,
            "no_confidence": routing_noconf,
        },
        "runtime_seconds": round(total_runtime, 2),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
