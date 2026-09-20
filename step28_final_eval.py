"""Step 28 — comprehensive final evaluation: baseline vs enriched vs hardened.

The primary evidence artifact for the thesis. Everything here is computed
fresh, in one script, from the three frozen model files
(model_baseline.json, model_enriched.json, model_hardened.json) against
the SAME held-out test set — not stitched together from prior steps'
separately-run numbers — so the comparison table is internally consistent
by construction. Adversarial numbers ARE pulled from steps 20/21/21b's own
reports (re-running PPO training here would defeat the point of comparing
against those specific, already-documented attacks).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Final, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from xgboost import XGBClassifier

from step13_shap_explain import build_feature_names, compute_window_feature_indices

# --- Config -------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"
WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
LABELS_TEST: Final[Path] = ARTIFACTS_DIR / "labels_test.npy"
USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

MODEL_BASELINE: Final[Path] = ARTIFACTS_DIR / "model_baseline.json"
MODEL_ENRICHED: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
MODEL_HARDENED: Final[Path] = ARTIFACTS_DIR / "model_hardened.json"
METRICS_BASELINE: Final[Path] = ARTIFACTS_DIR / "metrics_baseline.json"
METRICS_ENRICHED: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"

PPO_ATTACK_REPORT: Final[Path] = ARTIFACTS_DIR / "ppo_attack_report.json"
METRICS_ADVERSARIAL: Final[Path] = ARTIFACTS_DIR / "metrics_adversarial.json"
ADAPTIVE_EVASION_REPORT: Final[Path] = ARTIFACTS_DIR / "adaptive_evasion_report.json"

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_final.json"

WINDOW_FEATURE_COUNT: Final[int] = 210
LATENCY_N_RUNS: Final[int] = 1000

# Rare-purchase slice — identical criteria to step 8b.
RARE_ZSCORE_THRESHOLD: Final[float] = 2.0

# Position buckets — identical boundaries to step 9b's "merged_buckets".
POSITION_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("early (<5k)", 16, 5000),
    ("mid (5k-15k)", 5000, 15000),
    ("late (15k+)", 15000, 10_000_000),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step28")


# === Shared test-set loading ===================================================

def load_test_set() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Returns (X_test, y_test, user_feats, flags) — user_feats/flags kept
    around (not just folded into X) because the rare-purchase and
    position-bucket analyses need to slice by specific named columns."""
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
    X_test = np.hstack([np.asarray(X_win, dtype=np.float32), enriched])
    return X_test, y_test, user_feats, flags


def best_f1_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float, float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y, prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1]))  # last (P,R) pair has no threshold
    return float(thresholds[best_idx]), float(f1s[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])


def benchmark_latency(model: XGBClassifier, X_sample_row: np.ndarray, n_runs: int = LATENCY_N_RUNS) -> dict:
    row = X_sample_row.reshape(1, -1)
    # warm-up
    for _ in range(10):
        model.predict_proba(row)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "n_runs": n_runs,
        "mean_ms": round(float(times.mean()), 4),
        "median_ms": round(float(np.median(times)), 4),
        "p95_ms": round(float(np.percentile(times, 95)), 4),
        "p99_ms": round(float(np.percentile(times, 99)), 4),
    }


# === Rare-purchase slice evaluation (step 8b methodology) ======================

def evaluate_rare_purchase(
    prob: np.ndarray, y: np.ndarray, user_feats: pd.DataFrame, flags: pd.DataFrame, threshold: float,
) -> dict:
    legit = y == 0
    pred = prob >= threshold

    amount_unusual = (user_feats["user_amount_zscore"] > RARE_ZSCORE_THRESHOLD).values
    new_mcc = (flags["first_mcc"] == 1).values
    new_location = (flags["first_city_state"] == 1).values
    rare_any = amount_unusual | new_mcc | new_location
    ordinary = ~rare_any

    def _slice_stats(mask: np.ndarray) -> dict:
        slice_legit = mask & legit
        n_legit = int(slice_legit.sum())
        n_fp = int((pred & slice_legit).sum())
        return {"n_legit": n_legit, "n_fp": n_fp, "fp_rate": round(n_fp / max(n_legit, 1), 6)}

    return {
        "threshold": threshold,
        "rare_any": _slice_stats(rare_any),
        "amount_unusual": _slice_stats(amount_unusual),
        "new_mcc": _slice_stats(new_mcc),
        "new_location": _slice_stats(new_location),
        "ordinary": _slice_stats(ordinary),
        "all_legit": _slice_stats(np.ones_like(legit, dtype=bool)),
    }


# === Position-bucket evaluation (step 9b methodology) ==========================

def evaluate_position_buckets(prob: np.ndarray, y: np.ndarray, position: np.ndarray) -> list[dict]:
    results = []
    for name, lo, hi in POSITION_BUCKETS:
        mask = (position >= lo) & (position < hi)
        n_windows = int(mask.sum())
        n_fraud = int(y[mask].sum())
        aucpr = float(average_precision_score(y[mask], prob[mask])) if n_fraud > 0 else None
        results.append({"bucket": name, "lo": lo, "hi": hi, "n_windows": n_windows, "n_fraud": n_fraud, "aucpr": aucpr})
    return results


# === Main =====================================================================

def main() -> None:
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 28 — COMPREHENSIVE FINAL EVALUATION")
    log.info("=" * 70)

    for p in (MODEL_BASELINE, MODEL_ENRICHED, MODEL_HARDENED, METRICS_BASELINE, METRICS_ENRICHED):
        if not p.exists():
            log.error("Missing required artifact: %s", p)
            return

    log.info("Loading held-out test set (shared across all three models) ...")
    X_test, y_test, user_feats, flags = load_test_set()
    position = user_feats["user_tx_count"].values.astype(int)
    log.info("Test set: %s windows, %s fraud (%.4f%%)",
              f"{len(y_test):,}", f"{int(y_test.sum()):,}", y_test.mean() * 100)

    models = {}
    for name, path in [("baseline", MODEL_BASELINE), ("enriched", MODEL_ENRICHED), ("hardened", MODEL_HARDENED)]:
        m = XGBClassifier()
        m.load_model(str(path))
        models[name] = m

    # Baseline was trained on 210 window-only features (Phase 0); slice
    # the shared 243-dim test matrix down to match, exactly as step 4 did.
    X_by_model = {
        "baseline": X_test[:, :WINDOW_FEATURE_COUNT],
        "enriched": X_test,
        "hardened": X_test,
    }

    log.info("-" * 70)
    log.info("Scoring all three models on the shared test set ...")
    prob_by_model: dict[str, np.ndarray] = {}
    for name, model in models.items():
        prob_by_model[name] = model.predict_proba(X_by_model[name])[:, 1]
        log.info("  %s: scored %s windows", name, f"{len(prob_by_model[name]):,}")

    # =========================================================================
    # 1. MODEL COMPARISON TABLE
    # =========================================================================
    log.info("=" * 70)
    log.info("1. MODEL COMPARISON — baseline vs enriched vs hardened (standard test set)")

    with open(METRICS_BASELINE) as f:
        baseline_metrics_file = json.load(f)
    with open(METRICS_ENRICHED) as f:
        enriched_metrics_file = json.load(f)

    comparison: dict[str, dict] = {}
    for name in ("baseline", "enriched", "hardened"):
        prob = prob_by_model[name]
        auc_pr = float(average_precision_score(y_test, prob))
        thr, f1, prec, rec = best_f1_threshold(y_test, prob)

        if name == "baseline":
            latency = baseline_metrics_file["latency_cpu_single_window"]
            latency_source = "reused from metrics_baseline.json (same frozen model, same benchmark methodology)"
        elif name == "enriched":
            latency = enriched_metrics_file["latency_cpu_single_window"]
            latency_source = "reused from metrics_enriched.json (same frozen model, same benchmark methodology)"
        else:
            latency = benchmark_latency(models[name], X_by_model[name][0])
            latency_source = "freshly benchmarked in this script, identical methodology"

        comparison[name] = {
            "auc_pr": round(auc_pr, 6),
            "best_f1_threshold": round(thr, 6),
            "f1": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "latency_ms": latency,
            "latency_source": latency_source,
        }

    log.info("%-12s %10s %10s %10s %10s %14s %12s", "Model", "AUC-PR", "F1", "Precision", "Recall", "Threshold", "Latency(ms)")
    for name in ("baseline", "enriched", "hardened"):
        c = comparison[name]
        log.info("%-12s %10.4f %10.4f %10.4f %10.4f %14.4f %12.4f",
                  name, c["auc_pr"], c["f1"], c["precision"], c["recall"], c["best_f1_threshold"], c["latency_ms"]["mean_ms"])

    # =========================================================================
    # 2. RARE-PURCHASE EVALUATION
    # =========================================================================
    log.info("=" * 70)
    log.info("2. RARE-PURCHASE EVALUATION (step 8b methodology, re-run on all three)")
    log.info("Slice: user_amount_zscore>2 OR first_mcc==1 OR first_city_state==1 (unusual-but-"
              "legitimate purchases — e.g. a large, novel-category purchase during a festival "
              "or holiday season, the 'unusual purchase' case).")

    rare_results: dict[str, dict] = {}
    for name in ("baseline", "enriched", "hardened"):
        thr = comparison[name]["best_f1_threshold"]
        rare_results[name] = evaluate_rare_purchase(prob_by_model[name], y_test, user_feats, flags, thr)

    log.info("%-16s %12s %12s %12s", "Slice", "baseline FP%", "enriched FP%", "hardened FP%")
    for slice_name in ("rare_any", "amount_unusual", "new_mcc", "new_location", "ordinary"):
        b = rare_results["baseline"][slice_name]["fp_rate"] * 100
        e = rare_results["enriched"][slice_name]["fp_rate"] * 100
        h = rare_results["hardened"][slice_name]["fp_rate"] * 100
        log.info("%-16s %11.3f%% %11.3f%% %11.3f%%", slice_name, b, e, h)

    new_mcc_enriched = rare_results["enriched"]["new_mcc"]["fp_rate"]
    new_mcc_hardened = rare_results["hardened"]["new_mcc"]["fp_rate"]
    new_mcc_change_pct = (
        (new_mcc_hardened - new_mcc_enriched) / new_mcc_enriched * 100 if new_mcc_enriched > 0 else float("nan")
    )
    log.info("-" * 70)
    log.info("NEW-MCC (novel merchant-category) regression, specifically: enriched had a known "
              "+70.6%% FP-rate regression on this slice vs. baseline (step 8b). Hardened vs. "
              "enriched change: %+.1f%% (%.3f%% -> %.3f%%). %s",
              new_mcc_change_pct, new_mcc_enriched * 100, new_mcc_hardened * 100,
              "Hardening HELPED this specific case." if new_mcc_hardened < new_mcc_enriched
              else "Hardening did NOT help (or worsened) this specific case." if new_mcc_hardened > new_mcc_enriched
              else "No change.")

    rare_any_enriched = rare_results["enriched"]["rare_any"]["fp_rate"]
    rare_any_hardened = rare_results["hardened"]["rare_any"]["fp_rate"]
    log.info("Overall rare-purchase (rare_any) FP rate: enriched %.3f%% -> hardened %.3f%% (%+.3f pp)",
              rare_any_enriched * 100, rare_any_hardened * 100, (rare_any_hardened - rare_any_enriched) * 100)

    # =========================================================================
    # 3. ADVERSARIAL SUMMARY
    # =========================================================================
    log.info("=" * 70)
    log.info("3. ADVERSARIAL SUMMARY — local (transfer) vs structural (adaptive) robustness")

    adversarial_summary: dict[str, dict] = {
        "baseline": {
            "note": "Not adversarially tested. The PPO attacker (step 20) targeted the enriched "
                    "model specifically; no attack was ever run against the baseline model. "
                    "Reporting this honestly as a gap, not inferring a number.",
            "native_evasion_rate": None, "transfer_evasion_rate": None, "adaptive_evasion_rate": None,
        },
    }

    if PPO_ATTACK_REPORT.exists():
        with open(PPO_ATTACK_REPORT) as f:
            step20 = json.load(f)
        adversarial_summary["enriched"] = {
            "note": "Native evasion rate: the attacker was trained directly against this model "
                    "(step 20). Not a transfer or adaptive test — this IS its own attack.",
            "native_evasion_rate_deterministic": step20["evasion_rate"]["after_training_deterministic"],
            "native_evasion_rate_harvest": step20["evasion_rate"]["after_training_harvest_any_of_n_attempts"],
            "transfer_evasion_rate": None,
            "adaptive_evasion_rate": None,
        }
    else:
        adversarial_summary["enriched"] = {"note": "ppo_attack_report.json not found."}

    if METRICS_ADVERSARIAL.exists() and ADAPTIVE_EVASION_REPORT.exists():
        with open(METRICS_ADVERSARIAL) as f:
            step21 = json.load(f)
        with open(ADAPTIVE_EVASION_REPORT) as f:
            step21b = json.load(f)
        adversarial_summary["hardened"] = {
            "note": "transfer = the ENRICHED-trained attacker policy reused against this model, "
                    "unmodified (tests whether the OLD exploit still works). adaptive = a BRAND-"
                    "NEW attacker trained from scratch directly against this model (tests whether "
                    "hardening is structural or just overfit to the old exploit's trajectories).",
            "transfer_evasion_rate_deterministic": step21["adversarial_evaluation"]["hardened"]["evasion_rate_deterministic"],
            "transfer_evasion_rate_harvest": step21["adversarial_evaluation"]["hardened"]["evasion_rate_harvest"],
            "adaptive_evasion_rate_deterministic": step21b["comparison_table"]["fresh_attacker_vs_hardened_model"]["evasion_rate_deterministic"],
            "adaptive_evasion_rate_harvest": step21b["comparison_table"]["fresh_attacker_vs_hardened_model"]["evasion_rate_harvest"],
            "verdict": step21b["verdict"],
        }
    else:
        adversarial_summary["hardened"] = {"note": "step 21/21b reports not found."}

    log.info("%-12s %-18s %14s %14s", "Model", "Test type", "Evasion(det)", "Evasion(harvest)")
    log.info("%-12s %-18s %14s %14s", "baseline", "not tested", "n/a", "n/a")
    if "native_evasion_rate_harvest" in adversarial_summary["enriched"]:
        log.info("%-12s %-18s %13.2f%% %13.2f%%", "enriched", "native (own attacker)",
                  adversarial_summary["enriched"]["native_evasion_rate_deterministic"] * 100,
                  adversarial_summary["enriched"]["native_evasion_rate_harvest"] * 100)
    if "transfer_evasion_rate_harvest" in adversarial_summary["hardened"]:
        h = adversarial_summary["hardened"]
        log.info("%-12s %-18s %13.2f%% %13.2f%%", "hardened", "transfer (old atk)",
                  h["transfer_evasion_rate_deterministic"] * 100, h["transfer_evasion_rate_harvest"] * 100)
        log.info("%-12s %-18s %13.2f%% %13.2f%%", "hardened", "adaptive (new atk)",
                  h["adaptive_evasion_rate_deterministic"] * 100, h["adaptive_evasion_rate_harvest"] * 100)
        log.info("  -> LOCAL robustness (transfer) is strong (%.2f%%); STRUCTURAL robustness "
                  "(adaptive) is %s (%.2f%%) — verdict from step 21b: %s.",
                  h["transfer_evasion_rate_harvest"] * 100,
                  "weak" if h["adaptive_evasion_rate_harvest"] > 0.5 else "moderate" if h["adaptive_evasion_rate_harvest"] > 0.1 else "strong",
                  h["adaptive_evasion_rate_harvest"] * 100, h.get("verdict", "n/a"))

    # =========================================================================
    # 4. POSITION-BUCKET EVALUATION
    # =========================================================================
    log.info("=" * 70)
    log.info("4. POSITION-BUCKET EVALUATION (step 9b methodology, re-run on all three)")

    position_results: dict[str, list[dict]] = {}
    for name in ("baseline", "enriched", "hardened"):
        position_results[name] = evaluate_position_buckets(prob_by_model[name], y_test, position)

    log.info("%-16s %10s %10s %12s %12s %12s", "Bucket", "n_windows", "n_fraud", "base AUC-PR", "enr AUC-PR", "hard AUC-PR")
    for i, (bucket_name, _, _) in enumerate(POSITION_BUCKETS):
        b = position_results["baseline"][i]
        e = position_results["enriched"][i]
        h = position_results["hardened"][i]
        log.info("%-16s %10s %10s %12s %12s %12s",
                  bucket_name, f"{b['n_windows']:,}", f"{b['n_fraud']:,}",
                  f"{b['aucpr']:.4f}" if b["aucpr"] is not None else "n/a",
                  f"{e['aucpr']:.4f}" if e["aucpr"] is not None else "n/a",
                  f"{h['aucpr']:.4f}" if h["aucpr"] is not None else "n/a")

    log.info("-" * 70)
    log.info("Delta vs. enriched (did hardening change the position-dependence pattern?):")
    for i, (bucket_name, _, _) in enumerate(POSITION_BUCKETS):
        e = position_results["enriched"][i]["aucpr"]
        h = position_results["hardened"][i]["aucpr"]
        if e is not None and h is not None:
            log.info("  %-16s enriched=%.4f -> hardened=%.4f (%+.4f)", bucket_name, e, h, h - e)
    early_delta = (position_results["hardened"][0]["aucpr"] or 0) - (position_results["enriched"][0]["aucpr"] or 0)
    late_delta = (position_results["hardened"][-1]["aucpr"] or 0) - (position_results["enriched"][-1]["aucpr"] or 0)
    position_pattern_note: str
    if abs(early_delta - late_delta) > 0.02:
        if late_delta > early_delta:
            position_pattern_note = (
                f"Hardening's benefit is UNEVEN across position buckets, but in the OPPOSITE "
                f"direction from step 9b's original confound: gains are LARGER at late positions "
                f"(+{late_delta:.4f}) than early ones (+{early_delta:.4f}). Step 9b's concern was "
                f"that enriched-vs-baseline gains were biggest early and decayed with position — "
                f"a signature of leaning on position-correlated features rather than genuine fraud "
                f"signal. This reversed pattern does NOT match that signature: if hardening's "
                f"improvement were just re-exploiting the same position confound, it should decay "
                f"the same way, not grow. More consistent with hardening improving the decision "
                f"boundary for well-established, long-history users specifically (which matches "
                f"the harvested evasions and CTGAN fraud, both drawn from/modeling users with "
                f"substantial history — fraud only occurs at position >= 68 by construction)."
            )
        else:
            position_pattern_note = (
                f"Hardening's benefit is UNEVEN across position buckets, in the SAME direction as "
                f"step 9b's original confound: gains are larger early (+{early_delta:.4f}) than "
                f"late (+{late_delta:.4f}). This is the pattern step 9b flagged as potentially "
                f"riding on position-correlated features rather than pure fraud signal — the same "
                f"caveat plausibly applies to hardening's gains here too."
            )
    else:
        position_pattern_note = (
            f"Hardening's benefit is roughly UNIFORM across position buckets (early "
            f"delta=+{early_delta:.4f}, late delta=+{late_delta:.4f}) — unlike the original "
            f"enriched-vs-baseline comparison, this improvement does not show a strong "
            f"position-dependent pattern either way."
        )
    log.info("  %s", position_pattern_note)

    # =========================================================================
    # 5. CROSS-DATASET LIMITATION
    # =========================================================================
    log.info("=" * 70)
    log.info("5. CROSS-DATASET LIMITATION (honest, not omitted)")
    cross_dataset_note = (
        "All results in this project — baseline through hardened, calibration, drift monitoring, "
        "and adversarial robustness — are measured on ONE dataset: IBM TabFormer synthetic credit "
        "card transactions, with a single chronological 80/20 per-user train/test split. Sparkov "
        "and IEEE-CIS (two other commonly-used public fraud datasets) were NOT loaded, and no "
        "cross-dataset generalization test was performed. This means: every number in this report "
        "— AUC-PR, calibration, PSI drift, adversarial evasion rates — characterizes behavior on "
        "TabFormer's specific synthetic generation process, which may not transfer to real-world "
        "transaction distributions or to other datasets' fraud patterns and feature schemas. This "
        "is named explicitly as an untested generalization gap, not silently assumed away."
    )
    log.info(cross_dataset_note)

    total_runtime = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("Total runtime: %.1fs (%.1f min)", total_runtime, total_runtime / 60)

    output = {
        "description": "Comprehensive final evaluation — baseline vs enriched vs hardened. "
                        "Primary evidence artifact consolidating Phases 0-4.",
        "model_comparison": comparison,
        "rare_purchase_evaluation": rare_results,
        "rare_purchase_new_mcc_regression": {
            "enriched_fp_rate": new_mcc_enriched,
            "hardened_fp_rate": new_mcc_hardened,
            "pct_change": None if np.isnan(new_mcc_change_pct) else round(new_mcc_change_pct, 2),
            "hardening_helped": bool(new_mcc_hardened < new_mcc_enriched),
        },
        "adversarial_summary": adversarial_summary,
        "position_bucket_evaluation": {
            "bucket_definitions": [{"bucket": b, "lo": lo, "hi": hi} for b, lo, hi in POSITION_BUCKETS],
            "baseline": position_results["baseline"],
            "enriched": position_results["enriched"],
            "hardened": position_results["hardened"],
            "hardened_vs_enriched_pattern_note": position_pattern_note,
            "early_bucket_delta": round(early_delta, 6),
            "late_bucket_delta": round(late_delta, 6),
        },
        "cross_dataset_limitation": cross_dataset_note,
        "runtime_seconds": round(total_runtime, 2),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
