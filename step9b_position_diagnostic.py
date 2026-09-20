"""Step 9b — Position diagnostic.

Tests whether the enriched model's improvement over baseline is partly
an artifact of learning transaction position (user_tx_count) rather than
genuine user-familiarity signal.

Motivation: the TabFormer synthetic dataset has zero fraud before position
68, and fraud rate increases with position up to ~0.21% around position
5000–10000 before settling at ~0.11%.  If enriched features like
merchant_user_count or first_mcc correlate strongly with position, the
model may be exploiting the synthetic data's position→fraud relationship
rather than learning transferable behavioral patterns.

Three tests:
  1. Per-bucket AUC-PR:  is the enriched model's advantage uniform across
     position ranges, or does it concentrate where position is most
     informative?
  2. Spearman correlations:  do the top enriched features proxy for position?
  3. Rare-purchase position skew:  does the 8b FP-reduction slice sit
     at unusual positions?

Output: artifacts/metrics_position_diagnostic.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")

WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
LABELS_TEST: Final[Path] = ARTIFACTS_DIR / "labels_test.npy"
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"

USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

MODEL_BASELINE: Final[Path] = ARTIFACTS_DIR / "model_baseline.json"
MODEL_ENRICHED: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
METRICS_BASELINE: Final[Path] = ARTIFACTS_DIR / "metrics_baseline.json"
METRICS_ENRICHED: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_position_diagnostic.json"

USER_COL: Final[str] = "User"
WINDOW_SIZE: Final[int] = 10
ZSCORE_THRESHOLD: Final[float] = 2.0
CORR_SAMPLE: Final[int] = 200_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step9b")


# --- Helpers ----------------------------------------------------------------

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


def bucket_aucpr(
    y: np.ndarray,
    prob_base: np.ndarray,
    prob_enr: np.ndarray,
    position: np.ndarray,
    buckets: list[tuple[str, int, int]],
) -> list[dict]:
    rows = []
    for label, lo, hi in buckets:
        mask = (position >= lo) & (position < hi)
        n = int(mask.sum())
        f = int(y[mask].sum())

        if f < 2:
            rows.append({
                "bucket": label, "lo": lo, "hi": hi,
                "n_windows": n, "n_fraud": f,
                "aucpr_base": None, "aucpr_enr": None, "delta": None,
                "note": "too few fraud for AUC-PR",
            })
            continue

        ap_base = float(average_precision_score(y[mask], prob_base[mask]))
        ap_enr = float(average_precision_score(y[mask], prob_enr[mask]))
        delta = ap_enr - ap_base

        rows.append({
            "bucket": label, "lo": lo, "hi": hi,
            "n_windows": n, "n_fraud": f,
            "aucpr_base": round(ap_base, 4),
            "aucpr_enr": round(ap_enr, 4),
            "delta": round(delta, 4),
        })
    return rows


# --- Main -------------------------------------------------------------------

def main() -> None:
    for p in (WINDOWS_TEST, LABELS_TEST, CLEAN_TEST,
              USER_FEATURES_TEST, MERCH_FEATURES_TEST, FLAGS_TEST,
              MODEL_BASELINE, MODEL_ENRICHED,
              METRICS_BASELINE, METRICS_ENRICHED):
        if not p.exists():
            log.error("Missing %s — run prior steps first.", p)
            return

    # ---- Alignment ---------------------------------------------------------
    log.info("Computing window → transaction indices ...")
    idx = compute_window_feature_indices(CLEAN_TEST)
    y_test = np.load(str(LABELS_TEST), mmap_mode="r")
    n_test = len(y_test)
    assert len(idx) == n_test

    # ---- Load features aligned to windows ----------------------------------
    user_feats = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TEST).iloc[idx].reset_index(drop=True)

    position = user_feats["user_tx_count"].values.astype(np.float64)
    log.info("Position range: [%d, %d], mean=%.0f, median=%.0f",
             position.min(), position.max(), position.mean(), np.median(position))

    # ---- Predict with both models ------------------------------------------
    log.info("Loading baseline model ...")
    model_base = XGBClassifier()
    model_base.load_model(str(MODEL_BASELINE))
    X_base = np.load(str(WINDOWS_TEST), mmap_mode="r")
    prob_base = model_base.predict_proba(X_base)[:, 1]
    del X_base, model_base

    log.info("Loading enriched model and building merged array ...")
    model_enr = XGBClassifier()
    model_enr.load_model(str(MODEL_ENRICHED))
    X_win = np.load(str(WINDOWS_TEST), mmap_mode="r")
    u = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].values.astype(np.float32)
    m = pd.read_parquet(MERCH_FEATURES_TEST).iloc[idx].values.astype(np.float32)
    f = pd.read_parquet(FLAGS_TEST).iloc[idx].values.astype(np.float32)
    enriched = np.hstack([u, m, f])
    del u, m, f
    X_enr = np.hstack([X_win, enriched])
    del X_win, enriched
    log.info("X_enriched: %s (%.1f GB)", X_enr.shape, X_enr.nbytes / 1e9)
    prob_enr = model_enr.predict_proba(X_enr)[:, 1]
    del X_enr, model_enr

    # ---- Thresholds from step 8/8b -----------------------------------------
    with open(METRICS_BASELINE) as fh:
        thresh_base = json.load(fh)["best_f1_threshold"]["best_threshold"]
    with open(METRICS_ENRICHED) as fh:
        thresh_enr = json.load(fh)["best_f1_threshold"]["best_threshold"]

    # ====================================================================
    # PART 1 — Per-bucket AUC-PR
    # ====================================================================
    log.info("=" * 70)
    log.info("PART 1: AUC-PR BY TRANSACTION POSITION BUCKET")

    # A. User's suggested buckets (some will have too few fraud)
    user_buckets = [
        ("16–500", 16, 500),
        ("500–2k", 500, 2000),
        ("2k–5k", 2000, 5000),
        ("5k–10k", 5000, 10000),
        ("10k+", 10000, 100000),
    ]
    user_results = bucket_aucpr(y_test, prob_base, prob_enr, position, user_buckets)

    log.info("")
    log.info("A. Requested buckets (some too small for reliable AUC-PR):")
    log.info("%-12s %10s %6s  %8s %8s %8s  %s",
             "Bucket", "Windows", "Fraud", "Base", "Enrich", "Δ", "Note")

    for r in user_results:
        note = r.get("note", "")
        if r["aucpr_base"] is None:
            log.info("%-12s %10s %6d  %8s %8s %8s  %s",
                     r["bucket"], f"{r['n_windows']:,}", r["n_fraud"],
                     "—", "—", "—", note)
        else:
            log.info("%-12s %10s %6d  %8.4f %8.4f %+7.4f",
                     r["bucket"], f"{r['n_windows']:,}", r["n_fraud"],
                     r["aucpr_base"], r["aucpr_enr"], r["delta"])

    # B. Merged buckets for reliable comparison
    merged_buckets = [
        ("early (<5k)", 16, 5000),
        ("mid (5k–15k)", 5000, 15000),
        ("late (15k+)", 15000, 100000),
    ]
    merged_results = bucket_aucpr(y_test, prob_base, prob_enr, position, merged_buckets)

    log.info("")
    log.info("B. Merged buckets (≥50 fraud each for reliable AUC-PR):")
    log.info("%-16s %10s %6s  %8s %8s %8s",
             "Bucket", "Windows", "Fraud", "Base", "Enrich", "Δ")

    for r in merged_results:
        if r["aucpr_base"] is not None:
            log.info("%-16s %10s %6d  %8.4f %8.4f %+7.4f",
                     r["bucket"], f"{r['n_windows']:,}", r["n_fraud"],
                     r["aucpr_base"], r["aucpr_enr"], r["delta"])

    # C. Quartile-based buckets
    q25, q50, q75 = np.percentile(position, [25, 50, 75])
    quartile_buckets = [
        (f"Q1 (<{q25:.0f})", 0, int(q25)),
        (f"Q2 ({q25:.0f}–{q50:.0f})", int(q25), int(q50)),
        (f"Q3 ({q50:.0f}–{q75:.0f})", int(q50), int(q75)),
        (f"Q4 (>{q75:.0f})", int(q75), 100000),
    ]
    quartile_results = bucket_aucpr(y_test, prob_base, prob_enr, position, quartile_buckets)

    log.info("")
    log.info("C. Quartile buckets (equal-count, most balanced):")
    log.info("%-20s %10s %6s  %8s %8s %8s",
             "Bucket", "Windows", "Fraud", "Base", "Enrich", "Δ")

    for r in quartile_results:
        if r["aucpr_base"] is not None:
            log.info("%-20s %10s %6d  %8.4f %8.4f %+7.4f",
                     r["bucket"], f"{r['n_windows']:,}", r["n_fraud"],
                     r["aucpr_base"], r["aucpr_enr"], r["delta"])

    # Interpret
    deltas = [r["delta"] for r in quartile_results if r["delta"] is not None]
    if deltas:
        d_min, d_max, d_mean = min(deltas), max(deltas), np.mean(deltas)
        spread = d_max - d_min
        log.info("")
        log.info("Quartile Δ range: [%+.4f, %+.4f], mean=%+.4f, spread=%.4f",
                 d_min, d_max, d_mean, spread)
        if spread < 0.03:
            log.info("→ UNIFORM: AUC-PR improvement is consistent across positions.")
            log.info("  Features generalize; position is not the primary driver.")
            uniformity = "uniform"
        elif d_min < 0.01 and d_max > 0.05:
            log.info("→ CONCENTRATED: improvement is position-dependent.")
            q_labels = [r["bucket"] for r in quartile_results
                        if r["delta"] is not None and r["delta"] == d_max]
            log.info("  Largest gain in: %s", q_labels)
            uniformity = "concentrated"
        else:
            log.info("→ MIXED: some variation across positions, not extreme.")
            uniformity = "mixed"
    else:
        uniformity = "insufficient_data"

    # ====================================================================
    # PART 2 — Spearman correlations
    # ====================================================================
    log.info("=" * 70)
    log.info("PART 2: SPEARMAN CORRELATIONS (feature vs transaction position)")
    log.info("(subsample of %s for speed)", f"{CORR_SAMPLE:,}")

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_test, size=min(CORR_SAMPLE, n_test), replace=False)
    pos_sample = position[sample_idx]

    features_to_check = [
        ("user_tx_count", user_feats["user_tx_count"].values),
        ("merchant_user_count", user_feats["merchant_user_count"].values),
        ("mcc_user_count", user_feats["mcc_user_count"].values),
        ("is_new_merchant", user_feats["is_new_merchant"].values),
        ("first_mcc", flags["first_mcc"].values),
        ("first_city_state", flags["first_city_state"].values),
        ("is_largest_ever", flags["is_largest_ever"].values),
        ("is_2x_hist_max", flags["is_2x_hist_max"].values),
        ("user_amount_zscore", user_feats["user_amount_zscore"].values),
        ("user_hist_std", user_feats["user_hist_std"].values),
    ]

    # Also check enriched model's score advantage
    score_advantage = prob_enr - prob_base

    log.info("")
    log.info("%-24s  %8s  %12s  %s",
             "Feature", "ρ", "p-value", "Interpretation")

    correlation_results: dict = {}
    for name, vals in features_to_check:
        v = vals[sample_idx]
        valid = ~(np.isnan(pos_sample) | np.isnan(v))
        if valid.sum() < 100:
            log.info("%-24s  %8s  %12s  too few valid pairs", name, "—", "—")
            continue
        rho, pval = spearmanr(pos_sample[valid], v[valid])
        rho = float(rho)

        if abs(rho) > 0.7:
            interp = "STRONG proxy for position"
        elif abs(rho) > 0.3:
            interp = "moderate correlation"
        elif abs(rho) > 0.1:
            interp = "weak correlation"
        else:
            interp = "negligible"

        log.info("%-24s  %+7.4f  %12.2e  %s", name, rho, pval, interp)
        correlation_results[name] = {
            "spearman_rho": round(rho, 4),
            "p_value": float(f"{pval:.4e}"),
            "interpretation": interp,
        }

    # Score advantage vs position
    sa = score_advantage[sample_idx]
    valid = ~np.isnan(sa)
    if valid.sum() > 100:
        rho_sa, pval_sa = spearmanr(pos_sample[valid], sa[valid])
        log.info("")
        log.info("%-24s  %+7.4f  %12.2e  %s",
                 "score_advantage",
                 rho_sa, pval_sa,
                 "STRONG" if abs(rho_sa) > 0.3 else
                 "moderate" if abs(rho_sa) > 0.1 else "weak/none")
        log.info("  (score_advantage = prob_enriched − prob_baseline)")
        log.info("  If positive: enriched model's advantage grows with position.")
        log.info("  If near zero: advantage is position-independent.")
        correlation_results["score_advantage"] = {
            "spearman_rho": round(float(rho_sa), 4),
            "p_value": float(f"{pval_sa:.4e}"),
            "note": "prob_enriched − prob_baseline vs transaction position",
        }

    # Flag the top enriched features that are strong position proxies
    log.info("")
    strong_proxies = [name for name, r in correlation_results.items()
                      if abs(r.get("spearman_rho", 0)) > 0.5
                      and name != "user_tx_count"
                      and name != "score_advantage"]
    if strong_proxies:
        log.info("⚠ Features that are STRONG position proxies (|ρ|>0.5):")
        for sp in strong_proxies:
            rho = correlation_results[sp]["spearman_rho"]
            log.info("    %s (ρ=%+.4f)", sp, rho)
        log.info("  These features may partly encode position rather than")
        log.info("  genuine behavioral signal.")
    else:
        log.info("No enriched features are strong position proxies (|ρ|>0.5).")

    # ====================================================================
    # PART 3 — Rare-purchase position distribution
    # ====================================================================
    log.info("=" * 70)
    log.info("PART 3: RARE-PURCHASE SLICE — POSITION DISTRIBUTION")

    legit = y_test == 0
    amount_unusual = (user_feats["user_amount_zscore"] > ZSCORE_THRESHOLD).values
    new_mcc = (flags["first_mcc"] == 1).values
    new_location = (flags["first_city_state"] == 1).values
    rare_any = amount_unusual | new_mcc | new_location
    ordinary = ~rare_any

    rare_legit_mask = legit & rare_any
    ordinary_legit_mask = legit & ordinary
    all_legit_mask = legit

    slices_pos = {
        "rare_legit": position[rare_legit_mask],
        "ordinary_legit": position[ordinary_legit_mask],
        "all_legit": position[all_legit_mask],
    }

    log.info("")
    log.info("%-18s %8s  %8s %8s %8s %8s %8s",
             "Slice", "Count", "Mean", "p10", "p25", "p50", "p75")

    pos_distribution: dict = {}
    for label, arr in slices_pos.items():
        pcts = np.percentile(arr, [10, 25, 50, 75, 90])
        log.info("%-18s %8s  %8.0f %8.0f %8.0f %8.0f %8.0f",
                 label, f"{len(arr):,}",
                 arr.mean(), pcts[0], pcts[1], pcts[2], pcts[3])
        pos_distribution[label] = {
            "count": int(len(arr)),
            "mean": round(float(arr.mean()), 1),
            "p10": round(float(pcts[0]), 0),
            "p25": round(float(pcts[1]), 0),
            "p50": round(float(pcts[2]), 0),
            "p75": round(float(pcts[3]), 0),
            "p90": round(float(pcts[4]), 0),
        }

    # Position skew analysis
    rare_mean = slices_pos["rare_legit"].mean()
    all_mean = slices_pos["all_legit"].mean()
    skew_ratio = rare_mean / all_mean

    log.info("")
    log.info("Position skew: rare_legit mean=%.0f vs all_legit mean=%.0f "
             "(ratio=%.3f)", rare_mean, all_mean, skew_ratio)

    if skew_ratio < 0.85:
        log.info("→ Rare-legit transactions SKEW EARLY — they sit at lower")
        log.info("  positions than the overall population. Some of the 8b FP")
        log.info("  reduction may be position-driven.")
        skew_verdict = "skews_early"
    elif skew_ratio > 1.15:
        log.info("→ Rare-legit transactions SKEW LATE.")
        skew_verdict = "skews_late"
    else:
        log.info("→ Rare-legit transactions have SIMILAR position distribution")
        log.info("  to the overall population. Position skew is not a concern")
        log.info("  for the 8b FP-reduction claim.")
        skew_verdict = "no_significant_skew"

    # Per-sub-slice position breakdown
    log.info("")
    log.info("Sub-slice position distributions:")
    sub_slices = [
        ("amount_unusual", legit & amount_unusual),
        ("new_mcc", legit & new_mcc),
        ("new_location", legit & new_location),
    ]
    sub_pos_results: dict = {}
    for label, mask in sub_slices:
        arr = position[mask]
        if len(arr) == 0:
            continue
        pcts = np.percentile(arr, [10, 25, 50, 75])
        log.info("  %-18s  n=%8s  mean=%8.0f  p25=%8.0f  p50=%8.0f",
                 label, f"{len(arr):,}", arr.mean(), pcts[1], pcts[2])
        sub_pos_results[label] = {
            "count": int(len(arr)),
            "mean": round(float(arr.mean()), 1),
            "p50": round(float(np.median(arr)), 0),
        }

    # Rare-legit FP rate by position bucket
    log.info("")
    log.info("Rare-legit FP rate by position (connecting parts 1 and 3):")
    pred_base = (prob_base >= thresh_base).astype(int)
    pred_enr = (prob_enr >= thresh_enr).astype(int)

    rare_fp_by_pos: dict = {}
    for label, lo, hi in merged_buckets:
        pos_mask = (position >= lo) & (position < hi)
        rl = rare_legit_mask & pos_mask
        n_rl = int(rl.sum())
        if n_rl == 0:
            continue
        fp_b = int((pred_base[rl] == 1).sum())
        fp_e = int((pred_enr[rl] == 1).sum())
        rate_b = fp_b / n_rl * 100
        rate_e = fp_e / n_rl * 100
        rel = (rate_e - rate_b) / max(rate_b, 1e-9) * 100

        log.info("  %-16s  n_rare_legit=%6s  FP base=%3d (%.2f%%)  "
                 "FP enr=%3d (%.2f%%)  Δrel=%+.1f%%",
                 label, f"{n_rl:,}", fp_b, rate_b, fp_e, rate_e, rel)

        rare_fp_by_pos[label] = {
            "n_rare_legit": n_rl,
            "fp_baseline": fp_b, "fp_rate_baseline": round(rate_b / 100, 6),
            "fp_enriched": fp_e, "fp_rate_enriched": round(rate_e / 100, 6),
            "relative_change_pct": round(rel, 1),
        }

    del user_feats, flags

    # ====================================================================
    # SYNTHESIS
    # ====================================================================
    log.info("=" * 70)
    log.info("SYNTHESIS — DOES POSITION CONFOUND THE STEP 8 / 8B CLAIMS?")
    log.info("")

    # Gather evidence
    evidence_for: list[str] = []
    evidence_against: list[str] = []

    if uniformity == "uniform":
        evidence_against.append(
            "AUC-PR improvement is uniform across position quartiles"
        )
    elif uniformity == "concentrated":
        evidence_for.append(
            "AUC-PR improvement concentrates in certain position ranges"
        )
    else:
        evidence_against.append(
            "AUC-PR improvement shows some variation but not extreme concentration"
        )

    if strong_proxies:
        evidence_for.append(
            f"Features {strong_proxies} are strong position proxies (|ρ|>0.5)"
        )
    else:
        evidence_against.append(
            "No enriched feature (besides user_tx_count itself) is a strong "
            "position proxy (|ρ|>0.5)"
        )

    sa_rho = correlation_results.get("score_advantage", {}).get("spearman_rho", 0)
    if abs(sa_rho) > 0.1:
        evidence_for.append(
            f"Score advantage correlates with position (ρ={sa_rho:+.4f})"
        )
    else:
        evidence_against.append(
            f"Score advantage is position-independent (ρ={sa_rho:+.4f})"
        )

    if skew_verdict == "skews_early":
        evidence_for.append(
            f"Rare-legit slice skews early (mean pos ratio={skew_ratio:.3f})"
        )
    elif skew_verdict == "no_significant_skew":
        evidence_against.append(
            "Rare-legit slice has similar position distribution to overall"
        )

    log.info("Evidence FOR position confound:")
    for e in evidence_for:
        log.info("  + %s", e)
    if not evidence_for:
        log.info("  (none)")

    log.info("")
    log.info("Evidence AGAINST position confound:")
    for e in evidence_against:
        log.info("  − %s", e)
    if not evidence_against:
        log.info("  (none)")

    log.info("")
    n_for = len(evidence_for)
    n_against = len(evidence_against)
    if n_for == 0:
        verdict = "NO position confound detected."
        log.info("VERDICT: %s", verdict)
        log.info("The enriched model's improvement appears genuine.")
    elif n_against == 0:
        verdict = "STRONG position confound detected."
        log.info("VERDICT: %s", verdict)
        log.info("The step 8 / 8b claims are substantially undermined.")
    elif n_for > n_against:
        verdict = "LIKELY partial position confound."
        log.info("VERDICT: %s", verdict)
        log.info("Some of the enriched model's improvement may be position-driven.")
    else:
        verdict = "UNLIKELY to be primarily position-driven."
        log.info("VERDICT: %s", verdict)
        log.info("Position is a factor but the features also carry genuine signal.")

    log.info("=" * 70)

    # ---- Save --------------------------------------------------------------
    output = {
        "description": (
            "Diagnostic checking whether enriched model improvement is "
            "confounded by transaction position in the synthetic data."
        ),
        "part1_bucket_aucpr": {
            "user_requested_buckets": user_results,
            "merged_buckets": merged_results,
            "quartile_buckets": quartile_results,
            "uniformity": uniformity,
        },
        "part2_spearman_correlations": correlation_results,
        "part3_rare_purchase_position": {
            "position_distribution": pos_distribution,
            "sub_slices": sub_pos_results,
            "skew_ratio": round(skew_ratio, 3),
            "skew_verdict": skew_verdict,
            "rare_fp_by_position": rare_fp_by_pos,
        },
        "synthesis": {
            "evidence_for_confound": evidence_for,
            "evidence_against_confound": evidence_against,
            "verdict": verdict,
        },
    }

    with open(OUT_PATH, "w") as fout:
        json.dump(output, fout, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
