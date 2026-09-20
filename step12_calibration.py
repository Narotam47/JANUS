"""Step 12 — Probability calibration and monotonicity checks.

Two independent diagnostics on the enriched model's raw probability output,
neither of which changes the served model:

Task 1 — Calibration analysis.  At a 0.12% base rate, most transactions get
a predicted probability near zero, so equal-population ("decile") binning
degenerates: the bottom several deciles collapse into a single near-zero
bin.  This is reported honestly rather than papered over.  A fixed,
log-spaced binning scheme is used alongside the quantile scheme to give a
meaningful reliability diagram across the full dynamic range.  Brier score,
Brier Skill Score (vs. the always-predict-base-rate baseline), and Expected
Calibration Error (ECE) are reported for the raw model.  A held-out split of
the test set is then used to check, empirically, whether Platt scaling or
isotonic regression would improve calibration — as a reporting exercise
only.  The served enriched model is not retrained or modified.

Task 2 — Monotonicity check on `user_amount_zscore`.  For a random sample of
test transactions, the feature is swept across a realistic grid while every
other feature is held fixed at its observed value (an individual conditional
expectation / ICE sweep), and the resulting predicted-probability path is
checked for non-decreasing behavior.  This isolates the feature's effect
from confounds, unlike a raw feature-vs-fraud-rate correlation on real data
(which is also reported, separately, for contrast).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
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

PLOT_PATH: Final[Path] = ARTIFACTS_DIR / "calibration_plot.png"
OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_calibration.json"

USER_COL: Final[str] = "User"
WINDOW_SIZE: Final[int] = 10

# --- Calibration binning ------------------------------------------------
N_QUANTILE_BINS: Final[int] = 10
FIXED_BIN_EDGES: Final[list[float]] = [
    0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0 + 1e-9,
]

# --- Monotonicity check ---------------------------------------------------
MONO_FEATURE: Final[str] = "user_amount_zscore"
MONO_SAMPLE_SIZE: Final[int] = 3000
MONO_GRID_POINTS: Final[int] = 41
MONO_SEED: Final[int] = 42
MONO_PCTL_LO: Final[float] = 0.5
MONO_PCTL_HI: Final[float] = 99.5
VIOLATION_EPS_ABS: Final[float] = 1e-4
VIOLATION_EPS_REL: Final[float] = 0.01

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step12")


# === Helpers =================================================================

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


def _bin_stats(prob: np.ndarray, y: np.ndarray, bin_id: np.ndarray) -> list[dict]:
    """Per-bin count, mean predicted prob, observed fraction positive."""
    out: list[dict] = []
    for b in sorted(pd.unique(bin_id)):
        mask = bin_id == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "bin": str(b),
            "n": n,
            "n_fraud": int(y[mask].sum()),
            "mean_predicted": round(float(prob[mask].mean()), 8),
            "observed_fraction": round(float(y[mask].mean()), 8),
        })
    return out


def _ece(bin_stats: list[dict], n_total: int) -> float:
    return sum(
        (b["n"] / n_total) * abs(b["observed_fraction"] - b["mean_predicted"])
        for b in bin_stats
    )


def _brier_skill_score(y: np.ndarray, prob: np.ndarray) -> tuple[float, float, float]:
    bs_model = brier_score_loss(y, prob)
    base_rate = float(y.mean())
    bs_baseline = base_rate * (1.0 - base_rate)
    bss = 1.0 - bs_model / bs_baseline if bs_baseline > 0 else float("nan")
    return bs_model, bs_baseline, bss


# === Main =====================================================================

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
    y_test = np.asarray(y_test)

    user_feats = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(MERCH_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TEST).iloc[idx].reset_index(drop=True)

    with open(METRICS_ENRICHED) as f:
        threshold = json.load(f)["best_f1_threshold"]["best_threshold"]

    # ---- Predict -------------------------------------------------------------
    log.info("Predicting with enriched model ...")
    model = XGBClassifier()
    model.load_model(str(MODEL_ENRICHED))

    X_win = np.load(str(WINDOWS_TEST), mmap_mode="r")
    win_width = X_win.shape[1]
    u = user_feats.values.astype(np.float32)
    m = merch_feats.values.astype(np.float32)
    fl = flags.values.astype(np.float32)
    enriched = np.hstack([u, m, fl])
    X = np.hstack([X_win, enriched])
    del enriched, u, m, fl

    prob = model.predict_proba(X)[:, 1]

    # Pull out a random sample of full feature rows for the Task 2 ICE sweep
    # before X is freed.
    rng = np.random.default_rng(MONO_SEED)
    sample_idx = rng.choice(n_test, size=min(MONO_SAMPLE_SIZE, n_test), replace=False)
    sample_rows = X[sample_idx].astype(np.float32).copy()
    del X, X_win

    log.info("Predictions complete. Base rate: %.4f%% (%d / %s fraud)",
              y_test.mean() * 100, int(y_test.sum()), f"{n_test:,}")

    # =========================================================================
    # TASK 1 — CALIBRATION ANALYSIS
    # =========================================================================
    log.info("=" * 70)
    log.info("TASK 1 — CALIBRATION ANALYSIS")

    bs_model, bs_baseline, bss = _brier_skill_score(y_test, prob)
    log.info("  Brier score (model):            %.8f", bs_model)
    log.info("  Brier score (baseline=const p): %.8f  (p=%.4f%%)",
              bs_baseline, y_test.mean() * 100)
    log.info("  Brier Skill Score:               %.4f  (1.0=perfect, 0.0=no better than baseline)", bss)
    log.info("  Note: raw Brier score is trivially tiny at 0.12%% base rate —")
    log.info("  the skill score against the constant-base-rate baseline is the")
    log.info("  number that actually says whether the model beats naive guessing.")

    # ---- Quantile ("decile") binning: expected to degenerate ---------------
    log.info("-" * 70)
    log.info("Quantile-decile binning (equal population per bin):")
    try:
        q_bin_id, q_bin_edges = pd.qcut(prob, q=N_QUANTILE_BINS, retbins=True, duplicates="drop")
        n_effective_bins = q_bin_id.categories.size
    except ValueError:
        # All-identical values (extreme degenerate case) — single bin.
        q_bin_id = pd.Series(["all"] * n_test)
        n_effective_bins = 1
    quantile_bins = _bin_stats(prob, y_test, np.asarray(q_bin_id))
    ece_quantile = _ece(quantile_bins, n_test)

    log.info("  Requested %d bins -> %d effective (unique) bins after dedup.",
              N_QUANTILE_BINS, n_effective_bins)
    if n_effective_bins < N_QUANTILE_BINS:
        log.info("  → COLLAPSED: probability mass is concentrated near 0 at this")
        log.info("    base rate, so quantile bin edges coincide and several nominal")
        log.info("    deciles merge into one. This is expected, not a bug.")
    for b in quantile_bins:
        log.info("    n=%9s  fraud=%5d  mean_pred=%.6f  observed=%.6f",
                  f"{b['n']:,}", b["n_fraud"], b["mean_predicted"], b["observed_fraction"])
    log.info("  ECE (quantile scheme): %.6f", ece_quantile)

    # ---- Fixed log-scale binning: the informative reliability diagram ------
    log.info("-" * 70)
    log.info("Fixed log-scale binning (informative across full dynamic range):")
    fixed_bin_id = np.digitize(prob, FIXED_BIN_EDGES[1:-1])
    fixed_bins = _bin_stats(prob, y_test, fixed_bin_id)
    ece_fixed = _ece(fixed_bins, n_test)
    for i, b in enumerate(fixed_bins):
        lo = FIXED_BIN_EDGES[int(b["bin"])]
        hi = FIXED_BIN_EDGES[int(b["bin"]) + 1]
        log.info("    [%9.1e, %9.1e)  n=%9s  fraud=%5d  mean_pred=%.6f  observed=%.6f",
                  lo, hi, f"{b['n']:,}", b["n_fraud"], b["mean_predicted"], b["observed_fraction"])
    log.info("  ECE (fixed log-scale scheme): %.6f  <- headline ECE", ece_fixed)

    # Direction of miscalibration in the highest bin (where the router acts)
    top_bin = fixed_bins[-1]
    if top_bin["mean_predicted"] > top_bin["observed_fraction"] + 0.01:
        direction = "overconfident"
    elif top_bin["mean_predicted"] < top_bin["observed_fraction"] - 0.01:
        direction = "underconfident"
    else:
        direction = "approximately calibrated"
    log.info("  Top bin (prob >= 0.97): mean_predicted=%.4f vs observed=%.4f -> %s",
              top_bin["mean_predicted"], top_bin["observed_fraction"], direction)

    calibration_poor = ece_fixed > 0.01 or direction != "approximately calibrated"
    log.info("  Verdict: calibration is %s (headline ECE=%.4f).",
              "POOR" if calibration_poor else "reasonable", ece_fixed)

    # ---- Would Platt scaling / isotonic regression help? Empirical check ---
    log.info("-" * 70)
    log.info("Would post-hoc recalibration help? (diagnostic only — main model unchanged)")
    cal_idx, hold_idx = train_test_split(
        np.arange(n_test), test_size=0.5, random_state=MONO_SEED, stratify=y_test,
    )
    prob_cal, y_cal = prob[cal_idx], y_test[cal_idx]
    prob_hold, y_hold = prob[hold_idx], y_test[hold_idx]

    platt = LogisticRegression()
    platt.fit(prob_cal.reshape(-1, 1), y_cal)
    prob_hold_platt = platt.predict_proba(prob_hold.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_cal, y_cal)
    prob_hold_iso = iso.transform(prob_hold)

    def _holdout_report(name: str, p_hold: np.ndarray) -> dict:
        bs, _, skill = _brier_skill_score(y_hold, p_hold)
        bin_id = np.digitize(p_hold, FIXED_BIN_EDGES[1:-1])
        bins = _bin_stats(p_hold, y_hold, bin_id)
        ece = _ece(bins, len(y_hold))
        log.info("    %-10s  Brier=%.8f  BSS=%.4f  ECE=%.6f", name, bs, skill, ece)
        return {"brier_score": round(bs, 8), "brier_skill_score": round(skill, 4), "ece": round(ece, 6)}

    log.info("  Holdout (50%% of test, unseen by calibrators): n=%s, fraud=%d",
              f"{len(y_hold):,}", int(y_hold.sum()))
    raw_report = _holdout_report("raw", prob_hold)
    platt_report = _holdout_report("platt", prob_hold_platt)
    iso_report = _holdout_report("isotonic", prob_hold_iso)

    ece_improvement_platt = raw_report["ece"] - platt_report["ece"]
    ece_improvement_iso = raw_report["ece"] - iso_report["ece"]
    if ece_improvement_iso > 0.001 or ece_improvement_platt > 0.001:
        recal_verdict = (
            "isotonic" if iso_report["ece"] <= platt_report["ece"] else "platt"
        )
        log.info("  → Recalibration HELPS: %s reduces ECE from %.6f to %.6f on holdout.",
                  recal_verdict, raw_report["ece"],
                  min(platt_report["ece"], iso_report["ece"]))
    else:
        recal_verdict = "none"
        log.info("  → Recalibration does NOT meaningfully help on this holdout split.")

    needs_calibration_warning = calibration_poor
    log.info("-" * 70)
    log.info("  Raw probability outputs %s a calibration warning in the API/dashboard.",
              "DO need" if needs_calibration_warning else "do NOT need")
    if needs_calibration_warning:
        log.info("  Reason: headline ECE=%.4f and/or top-bin probabilities are %s.",
                  ece_fixed, direction)
        log.info("  Raw scores should be described as a RANKING signal (higher = riskier),")
        log.info("  not a literal fraud probability, until a calibration layer is deployed.")

    # ---- Reliability diagram plot -------------------------------------------
    log.info("Rendering reliability diagram -> %s", PLOT_PATH)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # Panel 1: quantile-decile scheme (degenerate)
    ax = axes[0]
    qx = [b["mean_predicted"] for b in quantile_bins]
    qy = [b["observed_fraction"] for b in quantile_bins]
    qn = [b["n"] for b in quantile_bins]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    sizes = 30 + 300 * np.array(qn) / max(qn)
    ax.scatter(qx, qy, s=sizes, alpha=0.7, color="tab:red")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"Quantile deciles\n({n_effective_bins} effective bins — collapsed by skew)")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: fixed log-scale scheme (informative)
    ax = axes[1]
    fx = np.array([max(b["mean_predicted"], 1e-8) for b in fixed_bins])
    fy = np.array([b["observed_fraction"] for b in fixed_bins])
    fn = np.array([b["n"] for b in fixed_bins])
    order = np.argsort(fx)
    diag_x = np.geomspace(fx.min(), 1.0, 100)
    ax.plot(diag_x, diag_x, "k--", lw=1, label="perfect calibration")
    sizes = 30 + 300 * fn[order] / fn.max()
    ax.scatter(fx[order], fy[order], s=sizes, alpha=0.8, color="tab:blue")
    ax.plot(fx[order], fy[order], color="tab:blue", alpha=0.4, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Mean predicted probability (log scale)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"Fixed log-scale bins\nECE={ece_fixed:.4f}, direction={direction}")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3: raw vs platt vs isotonic on holdout
    ax = axes[2]

    def _panel3_points(p_hold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bin_id = np.digitize(p_hold, FIXED_BIN_EDGES[1:-1])
        bins = _bin_stats(p_hold, y_hold, bin_id)
        x = np.array([max(b["mean_predicted"], 1e-8) for b in bins])
        y = np.array([b["observed_fraction"] for b in bins])
        o = np.argsort(x)
        return x[o], y[o]

    ax.plot(diag_x, diag_x, "k--", lw=1, label="perfect calibration")
    for name, p_hold, color in [
        ("raw", prob_hold, "tab:blue"),
        ("platt", prob_hold_platt, "tab:orange"),
        ("isotonic", prob_hold_iso, "tab:green"),
    ]:
        x, y = _panel3_points(p_hold)
        ax.plot(x, y, marker="o", ms=4, alpha=0.8, color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Mean predicted probability (log scale, holdout)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"Recalibration check (holdout)\nbest option: {recal_verdict}")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    # =========================================================================
    # TASK 2 — MONOTONICITY CHECK on user_amount_zscore
    # =========================================================================
    log.info("=" * 70)
    log.info("TASK 2 — MONOTONICITY CHECK on '%s'", MONO_FEATURE)

    mono_col_idx = win_width + user_feats.columns.get_loc(MONO_FEATURE)
    zscore_full = user_feats[MONO_FEATURE].values
    p_lo, p_hi = np.percentile(zscore_full, [MONO_PCTL_LO, MONO_PCTL_HI])
    grid = np.linspace(p_lo, p_hi, MONO_GRID_POINTS)
    log.info("  Feature column index in full vector: %d", mono_col_idx)
    log.info("  Sweep grid: [%.4f, %.4f] over %d points (p%.1f-p%.1f of observed values)",
              p_lo, p_hi, MONO_GRID_POINTS, MONO_PCTL_LO, MONO_PCTL_HI)
    log.info("  Sample size: %d transactions (all other features held at observed values)",
              len(sample_idx))

    n_sample = sample_rows.shape[0]
    n_grid = len(grid)
    tiled = np.repeat(sample_rows, n_grid, axis=0)
    tiled[:, mono_col_idx] = np.tile(grid, n_sample)

    log.info("  Predicting %s grid points (%d transactions x %d grid values) ...",
              f"{n_sample * n_grid:,}", n_sample, n_grid)
    probs_grid = model.predict_proba(tiled)[:, 1].reshape(n_sample, n_grid)
    del tiled

    diffs = np.diff(probs_grid, axis=1)
    row_range = probs_grid.max(axis=1) - probs_grid.min(axis=1)
    eps_row = np.maximum(VIOLATION_EPS_ABS, VIOLATION_EPS_REL * row_range)[:, None]

    violation_mask = diffs < -eps_row
    n_violations_per_row = violation_mask.sum(axis=1)
    has_violation = n_violations_per_row > 0

    strict_violation_mask = diffs < -1e-9
    strict_has_violation = strict_violation_mask.any(axis=1)

    max_drop_per_row = np.where(diffs < 0, -diffs, 0.0).max(axis=1)

    aggregate_pdp = probs_grid.mean(axis=0)
    agg_spearman, agg_spearman_p = spearmanr(grid, aggregate_pdp)

    pct_practically_monotonic = float((~has_violation).mean() * 100)
    pct_strictly_monotonic = float((~strict_has_violation).mean() * 100)
    mean_violations_per_row = float(n_violations_per_row.mean())
    mean_max_drop_all = float(max_drop_per_row.mean())
    mean_max_drop_violators = (
        float(max_drop_per_row[has_violation].mean()) if has_violation.any() else 0.0
    )
    pct_steps_violating = float(violation_mask.mean() * 100)

    log.info("  Per-transaction ICE sweep results:")
    log.info("    Practically monotonic (tolerance-adjusted): %.2f%% of %d rows",
              pct_practically_monotonic, n_sample)
    log.info("    Strictly monotonic (any decrease at all):    %.2f%% of %d rows",
              pct_strictly_monotonic, n_sample)
    log.info("    Mean violations per row:                     %.3f", mean_violations_per_row)
    log.info("    Mean worst single-step drop (all rows):      %.6f", mean_max_drop_all)
    log.info("    Mean worst single-step drop (violators only):%.6f", mean_max_drop_violators)
    log.info("    Fraction of all grid-steps that are violations: %.3f%%", pct_steps_violating)
    log.info("  Aggregate PDP (mean prob per grid point) vs grid value:")
    log.info("    Spearman rho = %.4f  (p=%.2e)", agg_spearman, agg_spearman_p)
    if agg_spearman > 0.9:
        log.info("    → Aggregate relationship is STRONGLY monotonic increasing.")
    elif agg_spearman > 0.5:
        log.info("    → Aggregate relationship is monotonic increasing overall, with noise.")
    else:
        log.info("    → Aggregate relationship is NOT clearly monotonic — investigate further.")

    severity = "none"
    if pct_practically_monotonic < 90:
        severity = "material"
    elif pct_practically_monotonic < 99:
        severity = "minor"
    log.info("  Violation severity verdict: %s", severity)

    # ---- Empirical (confounded) real-data comparison ------------------------
    log.info("-" * 70)
    log.info("For contrast: empirical (confounded) real-data trend, no holding-fixed:")
    emp_bin_id, emp_edges = pd.qcut(zscore_full, q=10, retbins=True, duplicates="drop")
    emp_n_bins = emp_bin_id.categories.size
    emp_rows = []
    for i, cat in enumerate(sorted(pd.unique(emp_bin_id))):
        mask = np.asarray(emp_bin_id) == cat
        emp_rows.append({
            "bin": i,
            "n": int(mask.sum()),
            "mean_zscore": round(float(zscore_full[mask].mean()), 4),
            "mean_predicted": round(float(prob[mask].mean()), 6),
            "observed_fraud_rate": round(float(y_test[mask].mean()), 6),
        })
        log.info("    bin %2d  mean_zscore=%8.3f  n=%9s  mean_pred=%.6f  observed_fraud_rate=%.6f",
                  i, emp_rows[-1]["mean_zscore"], f"{emp_rows[-1]['n']:,}",
                  emp_rows[-1]["mean_predicted"], emp_rows[-1]["observed_fraud_rate"])
    emp_spearman_pred, _ = spearmanr(
        [r["mean_zscore"] for r in emp_rows], [r["mean_predicted"] for r in emp_rows]
    )
    emp_spearman_obs, _ = spearmanr(
        [r["mean_zscore"] for r in emp_rows], [r["observed_fraud_rate"] for r in emp_rows]
    )
    log.info("  Spearman(zscore bin, mean predicted prob) = %.4f", emp_spearman_pred)
    log.info("  Spearman(zscore bin, observed fraud rate) = %.4f", emp_spearman_obs)
    log.info("  (This mixes in confounds — e.g. high-zscore transactions may also be")
    log.info("   new-merchant or late-position — unlike the ICE sweep above.)")

    del user_feats, merch_feats, flags

    # =========================================================================
    # SAVE
    # =========================================================================
    log.info("=" * 70)
    output = {
        "description": "Calibration and monotonicity diagnostics for the enriched "
                        "model's raw probability output. Reporting only — the "
                        "served model is unchanged.",
        "task1_calibration": {
            "base_rate": round(float(y_test.mean()), 6),
            "brier_score_model": round(bs_model, 8),
            "brier_score_baseline": round(bs_baseline, 8),
            "brier_skill_score": round(bss, 4),
            "quantile_scheme": {
                "requested_bins": N_QUANTILE_BINS,
                "effective_bins": n_effective_bins,
                "collapsed": n_effective_bins < N_QUANTILE_BINS,
                "bins": quantile_bins,
                "ece": round(ece_quantile, 6),
            },
            "fixed_log_scheme": {
                "bin_edges": FIXED_BIN_EDGES,
                "bins": fixed_bins,
                "ece": round(ece_fixed, 6),
                "headline_ece": round(ece_fixed, 6),
            },
            "top_bin_direction": direction,
            "calibration_verdict": "poor" if calibration_poor else "reasonable",
            "recalibration_check_holdout": {
                "n_holdout": len(y_hold),
                "raw": raw_report,
                "platt": platt_report,
                "isotonic": iso_report,
                "recommended": recal_verdict,
                "ece_improvement_platt": round(ece_improvement_platt, 6),
                "ece_improvement_isotonic": round(ece_improvement_iso, 6),
            },
            "needs_api_dashboard_calibration_warning": needs_calibration_warning,
            "note": "Diagnostic recalibration was fit and evaluated on a 50/50 "
                    "split of the TEST set purely to answer whether Platt/isotonic "
                    "would help; no calibrator is applied to the served model.",
        },
        "task2_monotonicity": {
            "feature": MONO_FEATURE,
            "method": "ICE sweep: grid over feature value, all other features held "
                      "at each sampled transaction's observed values.",
            "sample_size": int(n_sample),
            "grid_points": int(n_grid),
            "grid_range": [round(float(p_lo), 4), round(float(p_hi), 4)],
            "pct_rows_practically_monotonic": round(pct_practically_monotonic, 2),
            "pct_rows_strictly_monotonic": round(pct_strictly_monotonic, 2),
            "mean_violations_per_row": round(mean_violations_per_row, 4),
            "mean_worst_drop_all_rows": round(mean_max_drop_all, 6),
            "mean_worst_drop_violators_only": round(mean_max_drop_violators, 6),
            "pct_grid_steps_violating": round(pct_steps_violating, 4),
            "aggregate_pdp_spearman_rho": round(float(agg_spearman), 4),
            "aggregate_pdp_spearman_p": float(agg_spearman_p),
            "violation_severity": severity,
            "empirical_confounded_comparison": {
                "n_bins": emp_n_bins,
                "bins": emp_rows,
                "spearman_zscore_vs_predicted": round(float(emp_spearman_pred), 4),
                "spearman_zscore_vs_observed_fraud_rate": round(float(emp_spearman_obs), 4),
            },
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)
    log.info("Saved: %s", PLOT_PATH)


if __name__ == "__main__":
    main()
