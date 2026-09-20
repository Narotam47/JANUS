"""Step 11 — Operational routing: block / approve / step-up.

Turns model probability + confidence score + user position into an
operational decision.  Three tiers:

  BLOCK     — automated rejection.  High fraud probability AND high
               confidence in the prediction context.
  STEP-UP   — verification request (SMS OTP, in-app confirm, etc.).
               Used when confidence in the prediction context is too low
               to act on the probability either way, or for the
               first-ever cold-start transaction.
  APPROVE   — automated clearance.  Model is confident the transaction
               is legitimate, OR the user is in the cold-start safe zone
               (positions 1–8, zero fraud in this dataset).

Routing rules, evaluated in order:
  1. position == 0             → step_up   (cold_start_first_tx)
  2. 1 ≤ position ≤ 8         → approve   (cold_start_safe)
  3. confidence < conf_stepup  → step_up   (low_confidence)
  4. prob ≥ prob_block         → block     (high_prob_high_conf)
  5. otherwise                 → approve   (confident_approve)

Configurable thresholds (keyword arguments with documented defaults):

  prob_block  = 0.9497  Best-F1 threshold from step 8.  Combined with the
                         confidence gate (rule 3), blocks only fire when
                         both model output and prediction context agree.

  conf_stepup = 0.75    From step 10 validation: precision is 67% for
                         confidence < 0.75 vs 85%+ above.  All 29 first-MCC
                         false positives from step 8b sit at confidence
                         ≤ 0.75, so this threshold converts them from wrong
                         blocks into verification requests.

NOTE — a fourth "suspicious_review" tier (prob in [0.90, prob_block) with
high confidence → step_up) was in an earlier version of this router and was
removed after evaluation showed it could never fire. Reason, precisely:

  confidence = 0.35*certainty + 0.15*history + 0.20*familiarity
             + 0.20*novelty + 0.10*completeness

  certainty = min((threshold - prob) / CERT_BELOW_SCALE, 1.0)   for prob < threshold

  With threshold = prob_block = 0.9497 and CERT_BELOW_SCALE = 0.30 (step 10),
  any prob in [0.90, 0.9497) has certainty <= (0.9497 - 0.90) / 0.30 = 0.1657.
  Even if history, familiarity, novelty, and completeness are all a perfect
  1.0 — the best case possible — the ceiling on confidence in that probability
  band is:

      0.35 * 0.1657 + 0.15 + 0.20 + 0.20 + 0.10 = 0.708

  which is BELOW conf_stepup = 0.75. So no transaction with prob in
  [0.90, prob_block) can ever reach "high confidence" under the step 10
  formula — rule 3 (low_confidence) always claims it first. This is a
  structural property of the certainty formula and the gap between
  prob_block and conf_stepup, not a coincidence of this dataset; it holds
  for any input. See `main()` for an empirical confirmation on the test set.
  Removing the dead tier collapses the router back to the exact 3-way
  design (block / approve / step-up) that was originally specified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from step10_confidence import (
    CERT_BELOW_SCALE,
    W_CERTAINTY,
    W_COMPLETENESS,
    W_FAMILIARITY,
    W_HISTORY,
    W_NOVELTY,
    confidence,
    confidence_batch,
)

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

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_routing.json"

USER_COL: Final[str] = "User"
WINDOW_SIZE: Final[int] = 10

# --- Default thresholds (documented in module docstring) --------------------
DEFAULT_PROB_BLOCK: Final[float] = 0.9497
DEFAULT_CONF_STEPUP: Final[float] = 0.75
DEFAULT_COLDSTART_MAX: Final[int] = 8

# Diagnostic-only band used to empirically confirm the dead-zone analysis in
# the module docstring. Not a routing threshold.
DIAGNOSTIC_REVIEW_BAND_LO: Final[float] = 0.90

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step11")


# === Reusable route function ================================================

def route(
    prob: float,
    confidence_score: float,
    confidence_reason: str,
    position: int,
    *,
    prob_block: float = DEFAULT_PROB_BLOCK,
    conf_stepup: float = DEFAULT_CONF_STEPUP,
    coldstart_max: int = DEFAULT_COLDSTART_MAX,
) -> dict:
    """Route a transaction to block, approve, or step-up.

    Parameters
    ----------
    prob : float
        Enriched model's predicted fraud probability.
    confidence_score : float
        Confidence score from step10_confidence.confidence().
    confidence_reason : str
        Dominant factor from step10_confidence.confidence().
    position : int
        User's transaction index (user_tx_count): 0 = first-ever tx.
    prob_block : float
        Probability at or above which → block (if confidence is high).
    conf_stepup : float
        Confidence below which → step-up regardless of probability.
    coldstart_max : int
        Maximum position for cold-start safe zone (default 8).

    Returns
    -------
    dict with keys: decision, probability, confidence_score,
    confidence_reason, rule.
    """
    base = {
        "probability": prob,
        "confidence_score": confidence_score,
        "confidence_reason": confidence_reason,
    }

    if position == 0:
        return {**base, "decision": "step_up", "rule": "cold_start_first_tx"}

    if 1 <= position <= coldstart_max:
        return {**base, "decision": "approve", "rule": "cold_start_safe"}

    if confidence_score < conf_stepup:
        return {**base, "decision": "step_up", "rule": "low_confidence"}

    if prob >= prob_block:
        return {**base, "decision": "block", "rule": "high_prob_high_conf"}

    return {**base, "decision": "approve", "rule": "confident_approve"}


# === Unit tests =============================================================

def test_route() -> None:
    """Assertions for each routing path, including edge cases."""
    T = DEFAULT_PROB_BLOCK

    cases = [
        # (label, prob, conf, reason, position, expected_decision, expected_rule)
        ("cold_start_first_tx",
         0.50, 0.90, "all signals strong", 0, "step_up", "cold_start_first_tx"),
        ("cold_start_pos5",
         0.99, 0.90, "all signals strong", 5, "approve", "cold_start_safe"),
        ("cold_start_boundary_pos8",
         0.99, 0.99, "all signals strong", 8, "approve", "cold_start_safe"),
        ("main_model_starts_pos9",
         0.99, 0.99, "all signals strong", 9, "block", "high_prob_high_conf"),
        ("high_prob_high_conf",
         0.98, 0.90, "all signals strong", 1000, "block", "high_prob_high_conf"),
        ("low_prob_high_conf",
         0.01, 0.95, "all signals strong", 1000, "approve", "confident_approve"),
        ("first_mcc_case_high_prob_low_conf",
         0.96, 0.60, "first-time merchant/location/channel", 5000,
         "step_up", "low_confidence"),
        ("exactly_at_block_thresh",
         T, 0.75, "all signals strong", 100, "block", "high_prob_high_conf"),
        ("just_below_conf_thresh",
         0.96, 0.7499, "unfamiliar merchant or category", 100,
         "step_up", "low_confidence"),
        ("low_prob_low_conf",
         0.05, 0.40, "thin user history", 50,
         "step_up", "low_confidence"),
        # Hypothetical only — the step 10 certainty formula can never actually
        # produce confidence >= conf_stepup for prob this far below prob_block
        # (see module docstring), but route() must still have a defined
        # fallback if it were ever passed such a combination.
        ("hypothetical_midprob_highconf",
         0.93, 0.80, "prediction near decision threshold", 5000,
         "approve", "confident_approve"),
        ("just_below_block_thresh",
         T - 0.0001, 0.80, "all signals strong", 100,
         "approve", "confident_approve"),
    ]

    passed = 0
    for label, prob, conf, reason, pos, exp_dec, exp_rule in cases:
        v = route(prob, conf, reason, pos)
        ok = v["decision"] == exp_dec and v["rule"] == exp_rule
        if not ok:
            log.error("FAIL %s: got decision=%s rule=%s, expected %s / %s",
                      label, v["decision"], v["rule"], exp_dec, exp_rule)
        else:
            passed += 1

    assert passed == len(cases), f"{passed}/{len(cases)} tests passed"
    log.info("All %d unit tests PASSED.", len(cases))


# === Vectorized routing (for evaluation) ====================================

def route_batch(
    prob: np.ndarray,
    conf_scores: np.ndarray,
    position: np.ndarray,
    prob_block: float = DEFAULT_PROB_BLOCK,
    conf_stepup: float = DEFAULT_CONF_STEPUP,
    coldstart_max: int = DEFAULT_COLDSTART_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized routing. Returns (decisions, rules) as string arrays."""
    n = len(prob)
    decision = np.empty(n, dtype="U10")
    rule = np.empty(n, dtype="U25")

    cs0 = position == 0
    cs_safe = (position >= 1) & (position <= coldstart_max)
    main = position > coldstart_max

    low_conf = main & (conf_scores < conf_stepup)
    high_prob_conf = main & (conf_scores >= conf_stepup) & (prob >= prob_block)
    low_prob = main & (conf_scores >= conf_stepup) & (prob < prob_block)

    decision[cs0] = "step_up"
    rule[cs0] = "cold_start_first_tx"

    decision[cs_safe] = "approve"
    rule[cs_safe] = "cold_start_safe"

    decision[low_conf] = "step_up"
    rule[low_conf] = "low_confidence"

    decision[high_prob_conf] = "block"
    rule[high_prob_conf] = "high_prob_high_conf"

    decision[low_prob] = "approve"
    rule[low_prob] = "confident_approve"

    return decision, rule


# === Helpers ================================================================

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


# === Main ===================================================================

def main() -> None:
    # ---- Run unit tests first ----------------------------------------------
    log.info("Running unit tests ...")
    test_route()

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

    position = user_feats["user_tx_count"].values.astype(int)

    with open(METRICS_ENRICHED) as f:
        threshold = json.load(f)["best_f1_threshold"]["best_threshold"]

    # ---- Predict -----------------------------------------------------------
    log.info("Predicting with enriched model ...")
    model = XGBClassifier()
    model.load_model(str(MODEL_ENRICHED))

    X_win = np.load(str(WINDOWS_TEST), mmap_mode="r")
    u = user_feats.values.astype(np.float32)
    m = merch_feats.values.astype(np.float32)
    fl = flags.values.astype(np.float32)
    enriched = np.hstack([u, m, fl])
    X = np.hstack([X_win, enriched])
    del X_win, enriched, u, m, fl

    prob = model.predict_proba(X)[:, 1]
    del X, model

    # Binary model reference (what the old model would do)
    binary_pred = (prob >= threshold).astype(int)
    log.info("Binary model: %d flagged as fraud.", binary_pred.sum())

    # ---- Compute confidence ------------------------------------------------
    log.info("Computing confidence scores ...")
    nan_user = user_feats.isna().sum(axis=1).values
    nan_merch = merch_feats.isna().sum(axis=1).values
    nan_flags = flags.isna().sum(axis=1).values
    n_nan = nan_user + nan_merch + nan_flags
    del nan_user, nan_merch, nan_flags

    conf_scores, _ = confidence_batch(
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

    # ---- Structural check: is the review band (prob in [0.90, prob_block))
    # actually reachable with high confidence? -------------------------------
    log.info("=" * 70)
    log.info("STRUCTURAL CHECK — why 'suspicious_review' was removed")
    band_lo = DIAGNOSTIC_REVIEW_BAND_LO
    band_mask = (prob >= band_lo) & (prob < threshold)
    n_band = int(band_mask.sum())

    max_certainty_in_band = min((threshold - band_lo) / CERT_BELOW_SCALE, 1.0)
    theoretical_conf_ceiling = (
        W_CERTAINTY * max_certainty_in_band
        + W_HISTORY + W_FAMILIARITY + W_NOVELTY + W_COMPLETENESS
    )
    log.info("  Band: prob in [%.4f, %.4f) — n=%s transactions", band_lo, threshold, f"{n_band:,}")
    log.info("  Theoretical confidence ceiling in this band (best case on all"
              " other components): %.4f", theoretical_conf_ceiling)
    log.info("  conf_stepup threshold: %.4f", DEFAULT_CONF_STEPUP)
    if n_band > 0:
        band_conf = conf_scores[band_mask]
        log.info("  Observed confidence in band: mean=%.4f  max=%.4f  p99=%.4f",
                  band_conf.mean(), band_conf.max(), np.percentile(band_conf, 99))
        n_would_be_high_conf = int((band_conf >= DEFAULT_CONF_STEPUP).sum())
        log.info("  Observed transactions in band with confidence >= %.2f: %d / %d",
                  DEFAULT_CONF_STEPUP, n_would_be_high_conf, n_band)
    if theoretical_conf_ceiling < DEFAULT_CONF_STEPUP:
        log.info("  → Ceiling (%.4f) < conf_stepup (%.4f): a 'suspicious_review'",
                  theoretical_conf_ceiling, DEFAULT_CONF_STEPUP)
        log.info("    tier is mathematically UNREACHABLE for any input, not just this")
        log.info("    dataset. This is why route() collapses that case into low_confidence.")
    else:
        log.info("  → Ceiling exceeds conf_stepup: the tier would be reachable in")
        log.info("    principle. (Not the case with current defaults.)")

    # ---- Route all test transactions ---------------------------------------
    log.info("Routing %s test transactions ...", f"{n_test:,}")
    decisions, rules = route_batch(prob, conf_scores, position)

    # ---- Three-way split ---------------------------------------------------
    is_block = decisions == "block"
    is_stepup = decisions == "step_up"
    is_approve = decisions == "approve"

    n_block = int(is_block.sum())
    n_stepup = int(is_stepup.sum())
    n_approve = int(is_approve.sum())

    log.info("=" * 70)
    log.info("ROUTING DISTRIBUTION")
    log.info("%-12s  %10s  %6s", "Decision", "Count", "%")
    log.info("%-12s  %10s  %5.2f%%", "block", f"{n_block:,}", n_block / n_test * 100)
    log.info("%-12s  %10s  %5.2f%%", "step_up", f"{n_stepup:,}", n_stepup / n_test * 100)
    log.info("%-12s  %10s  %5.2f%%", "approve", f"{n_approve:,}", n_approve / n_test * 100)

    # ---- Rule breakdown ----------------------------------------------------
    log.info("-" * 70)
    log.info("RULE BREAKDOWN")
    unique_rules, counts = np.unique(rules, return_counts=True)
    for r, c in sorted(zip(unique_rules, counts), key=lambda x: -x[1]):
        log.info("  %-25s  %10s  (%5.2f%%)", r, f"{c:,}", c / n_test * 100)

    # ---- Metrics per decision ----------------------------------------------
    fraud = y_test == 1
    legit = y_test == 0
    n_fraud = int(fraud.sum())
    n_legit = int(legit.sum())

    # BLOCKS
    log.info("=" * 70)
    log.info("BLOCK QUALITY (precision of blocks)")
    block_tp = int((is_block & fraud).sum())
    block_fp = int((is_block & legit).sum())
    block_prec = block_tp / max(block_tp + block_fp, 1)
    log.info("  Blocks: %d total = %d TP + %d FP", n_block, block_tp, block_fp)
    log.info("  Block precision: %.1f%%", block_prec * 100)

    # STEP-UPS
    log.info("-" * 70)
    log.info("STEP-UP ANALYSIS")
    stepup_fraud = int((is_stepup & fraud).sum())
    stepup_legit = int((is_stepup & legit).sum())
    log.info("  Step-ups: %d total = %d fraud + %d legit",
             n_stepup, stepup_fraud, stepup_legit)

    # Of step-ups, how many would have been FPs under binary model?
    stepup_was_binary_fp = int((is_stepup & legit & (binary_pred == 1)).sum())
    binary_fp_total = int((legit & (binary_pred == 1)).sum())
    log.info("")
    log.info("  FPs rescued by routing (binary FPs → step-up instead of block):")
    log.info("    %d / %d binary FPs now route to step-up (%.1f%%)",
             stepup_was_binary_fp, binary_fp_total,
             stepup_was_binary_fp / max(binary_fp_total, 1) * 100)

    # Of step-ups, how many are fraud caught by the review zone?
    stepup_fraud_was_binary_fn = int(
        (is_stepup & fraud & (binary_pred == 0)).sum()
    )
    log.info("")
    log.info("  Fraud caught by step-up that binary model MISSED:")
    log.info("    %d additional fraud routed to step-up (would have been approved)",
             stepup_fraud_was_binary_fn)

    # APPROVALS
    log.info("-" * 70)
    log.info("APPROVAL ANALYSIS (cost of approving)")
    approve_fraud = int((is_approve & fraud).sum())
    approve_legit = int((is_approve & legit).sum())
    log.info("  Approvals: %d total = %d fraud (missed) + %d legit",
             n_approve, approve_fraud, approve_legit)
    log.info("  Missed fraud rate in approvals: %.4f%% (%d / %s)",
             approve_fraud / max(n_approve, 1) * 100,
             approve_fraud, f"{n_approve:,}")

    # ---- Comparison: binary model vs router --------------------------------
    log.info("=" * 70)
    log.info("BINARY MODEL vs ROUTER — SIDE BY SIDE")

    binary_tp = int((fraud & (binary_pred == 1)).sum())
    binary_fp = int((legit & (binary_pred == 1)).sum())
    binary_fn = int((fraud & (binary_pred == 0)).sum())
    binary_prec = binary_tp / max(binary_tp + binary_fp, 1)
    binary_recall = binary_tp / n_fraud

    # The binary model has only one "flagged" action, so its recall is a
    # single number. The router has two distinct flagged actions (block and
    # step-up) that carry very different operational cost, so its recall must
    # be reported both ways — collapsing them into one number hides that a
    # step-up is a request for verification, not a rejection.
    recall_block_only = block_tp / n_fraud
    recall_block_plus_stepup = (block_tp + stepup_fraud) / n_fraud

    log.info("%-30s  %10s  %10s", "", "Binary", "Router")
    log.info("%-30s  %10d  %10d", "Fraud in hard action",
             binary_tp, block_tp)
    log.info("%-30s  %10s  %10d", "Fraud additionally in step-up",
             "—", stepup_fraud)
    log.info("%-30s  %10d  %10d", "Fraud missed (approved)",
             binary_fn, approve_fraud)
    log.info("")
    log.info("  Recall, BLOCK-ONLY (directly comparable to binary's flag action):")
    log.info("    binary: %.1f%%   router (block only): %.1f%%",
              binary_recall * 100, recall_block_only * 100)
    log.info("  Recall, BLOCK + STEP-UP (fraud flagged in ANY form, assumes")
    log.info("  step-up verification successfully catches the fraud):")
    log.info("    router (block + step-up): %.1f%%", recall_block_plus_stepup * 100)
    log.info("")
    log.info("%-30s  %10d  %10d", "Legit blocked (hard FPs)",
             binary_fp, block_fp)
    log.info("%-30s  %10s  %10d", "Legit step-up (soft FPs)",
             "—", stepup_legit)
    log.info("%-30s  %9.1f%%  %9.1f%%", "Block precision",
             binary_prec * 100, block_prec * 100)
    log.info("")
    log.info("  Step-up rate, of LEGITIMATE transactions only (the honest")
    log.info("  customer-friction number — all-transaction denominators dilute")
    log.info("  it with the 99.87%% of traffic that is fraud-free by construction):")
    stepup_rate_of_legit = stepup_legit / n_legit
    log.info("    %.4f%% (%d / %s legit transactions)",
              stepup_rate_of_legit * 100, stepup_legit, f"{n_legit:,}")

    # ---- First-MCC FP check ------------------------------------------------
    log.info("=" * 70)
    log.info("FIRST-MCC FALSE POSITIVE CHECK (the 8b regression)")

    first_mcc_legit = (flags["first_mcc"] == 1).values & legit
    first_mcc_binary_fp = first_mcc_legit & (binary_pred == 1)
    n_fml_fp = int(first_mcc_binary_fp.sum())
    log.info("  First-MCC legit FPs under binary model: %d", n_fml_fp)

    if n_fml_fp > 0:
        fml_decisions = decisions[first_mcc_binary_fp]
        fml_block = int((fml_decisions == "block").sum())
        fml_stepup = int((fml_decisions == "step_up").sum())
        fml_approve = int((fml_decisions == "approve").sum())

        log.info("  Under router:")
        log.info("    block:   %d  (still wrong blocks)", fml_block)
        log.info("    step_up: %d  (rescued → verification)", fml_stepup)
        log.info("    approve: %d  (rescued → approved)", fml_approve)
        log.info("")

        rescued = fml_stepup + fml_approve
        log.info("  → %d / %d first-MCC FPs rescued (%.0f%%)",
                 rescued, n_fml_fp, rescued / n_fml_fp * 100)

        fml_rules = rules[first_mcc_binary_fp]
        unique_r, counts_r = np.unique(fml_rules, return_counts=True)
        log.info("  Rules that fired:")
        for r, c in sorted(zip(unique_r, counts_r), key=lambda x: -x[1]):
            log.info("    %-25s  %d", r, c)
    else:
        log.info("  (no first-MCC FPs found)")

    del user_feats, merch_feats, flags

    # ---- Summary -----------------------------------------------------------
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("  Routing converts %d hard FP blocks into step-up verifications.",
             stepup_was_binary_fp)
    log.info("  Block precision improves: %.1f%% → %.1f%%.",
             binary_prec * 100, block_prec * 100)
    log.info("  Recall, block-only: %.1f%% (binary: %.1f%%).",
             recall_block_only * 100, binary_recall * 100)
    log.info("  Recall, block+step-up (assumes step-up catches the fraud): %.1f%%.",
             recall_block_plus_stepup * 100)
    log.info("  Missed fraud in approvals: %d (%.4f%% of approved).",
             approve_fraud, approve_fraud / max(n_approve, 1) * 100)
    log.info("  Customer friction: %.4f%% of LEGITIMATE transactions go to step-up.",
             stepup_rate_of_legit * 100)
    log.info("  'suspicious_review' tier removed — mathematically unreachable")
    log.info("  (confidence ceiling %.4f < conf_stepup %.4f); see structural check above.",
             theoretical_conf_ceiling, DEFAULT_CONF_STEPUP)
    log.info("=" * 70)

    # ---- Save --------------------------------------------------------------
    output = {
        "description": "Three-tier routing evaluation on the enriched model's test set.",
        "thresholds": {
            "prob_block": DEFAULT_PROB_BLOCK,
            "conf_stepup": DEFAULT_CONF_STEPUP,
            "coldstart_max": DEFAULT_COLDSTART_MAX,
        },
        "structural_check_suspicious_review": {
            "description": "Diagnostic band [0.90, prob_block) evaluated for "
                            "whether a 'high prob + high confidence but below "
                            "block threshold' review tier could ever fire.",
            "band_lo": DIAGNOSTIC_REVIEW_BAND_LO,
            "band_hi": threshold,
            "n_in_band": n_band,
            "theoretical_confidence_ceiling": round(theoretical_conf_ceiling, 4),
            "conf_stepup": DEFAULT_CONF_STEPUP,
            "structurally_unreachable": bool(theoretical_conf_ceiling < DEFAULT_CONF_STEPUP),
            "observed_n_high_conf_in_band": n_would_be_high_conf if n_band > 0 else 0,
        },
        "routing_distribution": {
            "block": {"count": n_block, "pct": round(n_block / n_test * 100, 4)},
            "step_up": {"count": n_stepup, "pct": round(n_stepup / n_test * 100, 4)},
            "approve": {"count": n_approve, "pct": round(n_approve / n_test * 100, 4)},
        },
        "block_quality": {
            "tp": block_tp, "fp": block_fp,
            "precision": round(block_prec, 4),
        },
        "stepup_analysis": {
            "total": n_stepup,
            "fraud_caught": stepup_fraud,
            "legit_friction": stepup_legit,
            "stepup_rate_of_legit": round(stepup_rate_of_legit, 6),
            "binary_fps_rescued": stepup_was_binary_fp,
            "binary_fns_caught": stepup_fraud_was_binary_fn,
        },
        "approval_cost": {
            "total": n_approve,
            "missed_fraud": approve_fraud,
            "missed_fraud_rate": round(approve_fraud / max(n_approve, 1), 6),
        },
        "vs_binary": {
            "binary_tp": binary_tp, "binary_fp": binary_fp,
            "binary_recall": round(binary_recall, 4),
            "binary_block_precision": round(binary_prec, 4),
            "router_block_precision": round(block_prec, 4),
            "recall_block_only": round(recall_block_only, 4),
            "recall_block_plus_stepup": round(recall_block_plus_stepup, 4),
        },
        "first_mcc_fps": {
            "binary_fps": n_fml_fp,
            "rescued_to_stepup": int(fml_stepup) if n_fml_fp > 0 else 0,
            "rescued_to_approve": int(fml_approve) if n_fml_fp > 0 else 0,
            "still_blocked": int(fml_block) if n_fml_fp > 0 else 0,
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
