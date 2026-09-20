"""Step 17 — PSI (Population Stability Index) drift monitor.

Compares the training-time feature distribution (reference) against the
held-out test set's distribution (monitoring proxy) across all 243 model
features — the same 210 window + 33 enriched features steps 8/11/12/13
already use, reconstructed the identical way (windows_*.npy flattened
base-feature-major/timestep-minor, enriched parquets aligned via
`compute_window_feature_indices`). `clean_train.parquet` / `clean_test.parquet`
alone only carry the 21 RAW per-transaction columns before windowing — the
full 243-feature model space requires combining them with the windowed and
enriched artifacts exactly as every prior modeling step does; this script
does that, not just a comparison of the 21 raw columns.

HONEST CONTEXT — read before treating any number here as "real-world drift":
the "monitoring" distribution is the held-out TEST set: a chronological
slice of the SAME TabFormer dataset, split 80/20 per user (step 2). It is
NOT independent live traffic. PSI values here reflect whatever temporal
distribution shift exists WITHIN this one synthetic dataset between each
user's earlier 80% and later 20% of transactions — things like a feature's
support widening over a longer observation window (e.g. user_tx_count,
which is mechanically larger by construction in the test period), not
necessarily anything resembling real-world concept drift. In production,
the reference would be a rolling recent-training window and the monitoring
window would be genuinely live, unseen transactions from a different time
period and population. Treat this as validating the PSI *machinery*, not
as evidence about real deployment drift.

Reference sampling: the training set has 7,330,071 windows (6.2GB on
disk). PSI is a proportion-based statistic over 10 bins — a large random
sample gives statistically indistinguishable bin proportions to using the
full population, at a fraction of the I/O. The reference here is a
300,000-row random sample of the training windows (sorted before mmap
fancy-indexing for better disk locality; order doesn't matter for an
aggregate statistic). The monitoring side uses the FULL test set
(1,827,841 rows), matching every prior step's practice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from step13_shap_explain import build_feature_names, compute_window_feature_indices

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")

CLEAN_TRAIN: Final[Path] = ARTIFACTS_DIR / "clean_train.parquet"
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"
WINDOWS_TRAIN: Final[Path] = ARTIFACTS_DIR / "windows_train.npy"
WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
USER_FEATURES_TRAIN: Final[Path] = ARTIFACTS_DIR / "user_features_train.parquet"
USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TRAIN: Final[Path] = ARTIFACTS_DIR / "merchant_features_train.parquet"
MERCH_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TRAIN: Final[Path] = ARTIFACTS_DIR / "flags_train.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "psi_report.json"

N_BINS: Final[int] = 10
EPS: Final[float] = 1e-4
N_REF_SAMPLE: Final[int] = 300_000
SEED: Final[int] = 42

WATCH_THRESHOLD: Final[float] = 0.10
RETRAIN_THRESHOLD: Final[float] = 0.25
PCT_FEATURES_WATCH_THRESHOLD: Final[float] = 0.20  # "more than 20% of features exceed 0.1"

TOP_N_DRIFTED: Final[int] = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step17")


# === Data loading =============================================================

def _load_split_matrix(
    clean_path: Path, windows_path: Path, user_path: Path, merch_path: Path, flags_path: Path,
    row_positions: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """Builds the 243-col window+enriched matrix, optionally restricted to
    `row_positions` (window-array row indices) for a subsample. Mirrors
    steps 11/12/13's exact construction: window block base-feature-major /
    timestep-minor, enriched = hstack([user, merch, flags]) aligned via
    `compute_window_feature_indices`.
    """
    idx = compute_window_feature_indices(clean_path)  # window-position -> clean-row-position

    user_feats = pd.read_parquet(user_path).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(merch_path).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(flags_path).iloc[idx].reset_index(drop=True)
    feature_names = build_feature_names(
        clean_path, list(user_feats.columns), list(merch_feats.columns), list(flags.columns),
    )

    X_win = np.load(str(windows_path), mmap_mode="r")
    if row_positions is not None:
        X_win = X_win[row_positions]
        user_feats = user_feats.iloc[row_positions].reset_index(drop=True)
        merch_feats = merch_feats.iloc[row_positions].reset_index(drop=True)
        flags = flags.iloc[row_positions].reset_index(drop=True)
    else:
        X_win = np.asarray(X_win)

    enriched = np.hstack([
        user_feats.values.astype(np.float32),
        merch_feats.values.astype(np.float32),
        flags.values.astype(np.float32),
    ])
    X = np.hstack([X_win.astype(np.float32), enriched])
    return X, feature_names


def load_reference_and_monitoring() -> tuple[np.ndarray, np.ndarray, list[str]]:
    for p in (CLEAN_TRAIN, CLEAN_TEST, WINDOWS_TRAIN, WINDOWS_TEST,
              USER_FEATURES_TRAIN, USER_FEATURES_TEST, MERCH_FEATURES_TRAIN, MERCH_FEATURES_TEST,
              FLAGS_TRAIN, FLAGS_TEST):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    log.info("Sampling %s reference windows from training set (of 7,330,071 total) ...",
              f"{N_REF_SAMPLE:,}")
    n_windows_train = np.load(str(WINDOWS_TRAIN), mmap_mode="r").shape[0]
    rng = np.random.default_rng(SEED)
    sample_positions = np.sort(rng.choice(n_windows_train, size=N_REF_SAMPLE, replace=False))

    X_ref, feature_names = _load_split_matrix(
        CLEAN_TRAIN, WINDOWS_TRAIN, USER_FEATURES_TRAIN, MERCH_FEATURES_TRAIN, FLAGS_TRAIN,
        row_positions=sample_positions,
    )
    log.info("Reference matrix: %s", X_ref.shape)

    log.info("Loading full monitoring (test) set — the 'live traffic' proxy ...")
    X_mon, feature_names_mon = _load_split_matrix(
        CLEAN_TEST, WINDOWS_TEST, USER_FEATURES_TEST, MERCH_FEATURES_TEST, FLAGS_TEST,
        row_positions=None,
    )
    log.info("Monitoring matrix: %s", X_mon.shape)

    assert feature_names == feature_names_mon, "Train/test feature name ordering mismatch!"
    assert X_ref.shape[1] == X_mon.shape[1] == 243, (
        f"Expected 243 features, got ref={X_ref.shape[1]}, mon={X_mon.shape[1]}"
    )
    return X_ref, X_mon, feature_names


# === PSI computation ===========================================================

def psi_category(
    psi: float, watch_threshold: float = WATCH_THRESHOLD, retrain_threshold: float = RETRAIN_THRESHOLD,
) -> str:
    if psi >= retrain_threshold:
        return "retrain"
    if psi >= watch_threshold:
        return "watch"
    return "stable"


def compute_psi_for_feature(ref: np.ndarray, mon: np.ndarray, n_bins: int = N_BINS, eps: float = EPS) -> dict:
    """PSI for one feature, handling: NaN as its own explicit bin (missingness
    rate IS a driftable quantity, not something to silently drop), constant
    features (2-bin equals/not-equals split), binary/flag features (2 bins
    at the natural midpoint instead of 10 quantile bins), and zero-count
    bins (epsilon floor before the log-ratio, to avoid log(0)/div-by-0).
    """
    ref = ref.astype(np.float64)
    mon = mon.astype(np.float64)
    ref_nan_mask = np.isnan(ref)
    mon_nan_mask = np.isnan(mon)
    ref_nan_rate = float(ref_nan_mask.mean()) if len(ref) else 0.0
    mon_nan_rate = float(mon_nan_mask.mean()) if len(mon) else 0.0

    ref_valid = ref[~ref_nan_mask]
    mon_valid = mon[~mon_nan_mask]
    has_missing_bin = (ref_nan_rate > 0) or (mon_nan_rate > 0)

    if len(ref_valid) == 0:
        # Reference is entirely missing in the sample — no numeric bins are
        # definable at all. Fall back to comparing missing-rate directly.
        feature_type = "all_missing_in_reference"
        n_bins_used = 2
        ref_props = np.array([ref_nan_rate, 1.0 - ref_nan_rate])
        mon_props = np.array([mon_nan_rate, 1.0 - mon_nan_rate])
    else:
        unique_vals = np.unique(ref_valid)
        n_unique = len(unique_vals)

        if n_unique <= 1:
            feature_type = "constant"
            const_val = unique_vals[0]
            ref_eq = float(np.mean(np.isclose(ref_valid, const_val)))
            mon_eq = float(np.mean(np.isclose(mon_valid, const_val))) if len(mon_valid) else 0.0
            numeric_ref = np.array([ref_eq, 1.0 - ref_eq])
            numeric_mon = np.array([mon_eq, 1.0 - mon_eq])
            n_bins_used = 2
        elif n_unique == 2:
            feature_type = "binary"
            lo, hi = float(unique_vals[0]), float(unique_vals[1])
            mid = (lo + hi) / 2.0
            ref_lo = float(np.mean(ref_valid <= mid))
            mon_lo = float(np.mean(mon_valid <= mid)) if len(mon_valid) else 0.0
            numeric_ref = np.array([ref_lo, 1.0 - ref_lo])
            numeric_mon = np.array([mon_lo, 1.0 - mon_lo])
            n_bins_used = 2
        else:
            feature_type = "continuous"
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.unique(np.quantile(ref_valid, quantiles))
            if len(edges) < 2:
                edges = np.array([ref_valid.min(), ref_valid.max() + 1e-9])
            edges[0] = -np.inf
            edges[-1] = np.inf
            n_bins_used = len(edges) - 1
            ref_counts, _ = np.histogram(ref_valid, bins=edges)
            mon_counts, _ = np.histogram(mon_valid, bins=edges)
            numeric_ref = ref_counts / max(len(ref_valid), 1)
            numeric_mon = mon_counts / max(len(mon_valid), 1)

        if has_missing_bin:
            ref_props = np.concatenate([numeric_ref * (1.0 - ref_nan_rate), [ref_nan_rate]])
            mon_props = np.concatenate([numeric_mon * (1.0 - mon_nan_rate), [mon_nan_rate]])
            n_bins_used += 1
        else:
            ref_props, mon_props = numeric_ref, numeric_mon

    ref_props = np.maximum(ref_props, eps)
    mon_props = np.maximum(mon_props, eps)
    psi = float(np.sum((mon_props - ref_props) * np.log(mon_props / ref_props)))

    return {
        "psi": round(psi, 6),
        "category": psi_category(psi),
        "feature_type": feature_type,
        "n_bins_used": int(n_bins_used),
        "ref_nan_rate": round(ref_nan_rate, 6),
        "mon_nan_rate": round(mon_nan_rate, 6),
    }


def compute_all_psi(X_ref: np.ndarray, X_mon: np.ndarray, feature_names: list[str]) -> list[dict]:
    results = []
    for j, name in enumerate(feature_names):
        r = compute_psi_for_feature(X_ref[:, j], X_mon[:, j])
        r["feature"] = name
        results.append(r)
    return results


# === Retrain trigger (importable by step 18) ==================================

def should_retrain(
    psi_by_feature: dict[str, float] | list[dict],
    single_feature_threshold: float = RETRAIN_THRESHOLD,
    pct_features_threshold: float = PCT_FEATURES_WATCH_THRESHOLD,
    watch_threshold: float = WATCH_THRESHOLD,
) -> tuple[bool, str]:
    """Retrain trigger: fires if ANY feature's PSI >= single_feature_threshold
    (default 0.25), OR if more than pct_features_threshold (default 20%) of
    features have PSI >= watch_threshold (default 0.1).

    Accepts either {feature_name: psi} or the list-of-dicts format
    `compute_all_psi` returns. Returns (decision, human-readable reason).
    """
    if isinstance(psi_by_feature, dict):
        items = list(psi_by_feature.items())
    else:
        items = [(r["feature"], r["psi"]) for r in psi_by_feature]

    if not items:
        return False, "No PSI data provided."

    max_feature, max_psi = max(items, key=lambda kv: kv[1])
    n_total = len(items)
    n_at_or_above_watch = sum(1 for _, v in items if v >= watch_threshold)
    pct_at_or_above_watch = n_at_or_above_watch / n_total

    if max_psi >= single_feature_threshold:
        return True, (
            f"Feature '{max_feature}' has PSI={max_psi:.4f}, at or above the single-feature "
            f"retrain threshold ({single_feature_threshold})."
        )
    if pct_at_or_above_watch > pct_features_threshold:
        return True, (
            f"{n_at_or_above_watch}/{n_total} features ({pct_at_or_above_watch * 100:.1f}%) have "
            f"PSI >= {watch_threshold}, above the {pct_features_threshold * 100:.0f}% population "
            f"threshold (worst: '{max_feature}' at PSI={max_psi:.4f})."
        )
    return False, (
        f"No trigger: worst feature '{max_feature}' PSI={max_psi:.4f} (< {single_feature_threshold}); "
        f"{n_at_or_above_watch}/{n_total} features ({pct_at_or_above_watch * 100:.1f}%) at/above "
        f"watch threshold {watch_threshold} (<= {pct_features_threshold * 100:.0f}% population threshold)."
    )


# === Main =====================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("STEP 17 — PSI DRIFT MONITOR")
    log.info("Reference: %s random-sampled training windows. Monitoring: full test set.",
              f"{N_REF_SAMPLE:,}")
    log.info("HONEST CONTEXT: monitoring = held-out chronological test slice of the SAME "
              "TabFormer dataset, not independent live traffic. See module docstring.")
    log.info("=" * 70)

    X_ref, X_mon, feature_names = load_reference_and_monitoring()

    log.info("Computing PSI for %d features (%d bins, epsilon=%.1e) ...",
              len(feature_names), N_BINS, EPS)
    results = compute_all_psi(X_ref, X_mon, feature_names)
    results_sorted = sorted(results, key=lambda r: -r["psi"])

    n_stable = sum(1 for r in results if r["category"] == "stable")
    n_watch = sum(1 for r in results if r["category"] == "watch")
    n_retrain = sum(1 for r in results if r["category"] == "retrain")

    log.info("=" * 70)
    log.info("CATEGORY BREAKDOWN (%d features total)", len(results))
    log.info("  stable  (PSI < %.2f):              %3d  (%.1f%%)",
              WATCH_THRESHOLD, n_stable, n_stable / len(results) * 100)
    log.info("  watch   (%.2f <= PSI < %.2f):       %3d  (%.1f%%)",
              WATCH_THRESHOLD, RETRAIN_THRESHOLD, n_watch, n_watch / len(results) * 100)
    log.info("  retrain (PSI >= %.2f):              %3d  (%.1f%%)",
              RETRAIN_THRESHOLD, n_retrain, n_retrain / len(results) * 100)

    log.info("-" * 70)
    log.info("TOP %d MOST DRIFTED FEATURES", TOP_N_DRIFTED)
    log.info("  %-30s %10s  %-10s  %-22s  %8s  %8s",
              "Feature", "PSI", "Category", "Type", "RefNaN%", "MonNaN%")
    top10 = results_sorted[:TOP_N_DRIFTED]
    for r in top10:
        log.info("  %-30s %10.4f  %-10s  %-22s  %7.2f%%  %7.2f%%",
                  r["feature"], r["psi"], r["category"], r["feature_type"],
                  r["ref_nan_rate"] * 100, r["mon_nan_rate"] * 100)

    log.info("-" * 70)
    psi_by_feature = {r["feature"]: r["psi"] for r in results}
    decision, reason = should_retrain(psi_by_feature)
    log.info("SHOULD_RETRAIN() DECISION: %s", decision)
    log.info("  Reason: %s", reason)
    if decision:
        log.info("  NOTE: given the honest context above (test set is a chronological slice of "
                  "the SAME dataset, not independent live data), a positive trigger here most "
                  "likely reflects mechanical within-dataset shift (e.g. user_tx_count being "
                  "larger by construction later in each user's history) rather than genuine "
                  "real-world concept drift. Treat this run as validating the MACHINERY works, "
                  "not as a real retrain decision.")
    else:
        log.info("  Distributions are close enough that should_retrain() does NOT fire on this "
                  "data — expected, since train/test here are two halves of one dataset rather "
                  "than genuinely different populations.")

    log.info("=" * 70)

    output = {
        "description": "PSI drift report: training distribution (reference, sampled) vs. "
                        "held-out test distribution (monitoring proxy).",
        "honest_context": (
            "The monitoring set is a chronological held-out slice of the SAME TabFormer "
            "dataset (last 20% of each user's transactions), not independent live traffic. "
            "PSI here reflects within-dataset temporal shift, not real-world drift. In "
            "production the reference would be a rolling training window and the monitoring "
            "window would be genuinely live, unseen transactions."
        ),
        "config": {
            "n_bins": N_BINS,
            "epsilon": EPS,
            "reference_sample_size": N_REF_SAMPLE,
            "monitoring_sample_size": int(X_mon.shape[0]),
            "watch_threshold": WATCH_THRESHOLD,
            "retrain_threshold": RETRAIN_THRESHOLD,
            "pct_features_watch_threshold": PCT_FEATURES_WATCH_THRESHOLD,
            "seed": SEED,
        },
        "summary": {
            "n_features": len(results),
            "n_stable": n_stable,
            "n_watch": n_watch,
            "n_retrain": n_retrain,
        },
        "top_10_drifted": [
            {"feature": r["feature"], "psi": r["psi"], "category": r["category"], "feature_type": r["feature_type"]}
            for r in top10
        ],
        "should_retrain": {
            "decision": decision,
            "reason": reason,
            "thresholds_used": {
                "single_feature_threshold": RETRAIN_THRESHOLD,
                "pct_features_threshold": PCT_FEATURES_WATCH_THRESHOLD,
                "watch_threshold": WATCH_THRESHOLD,
            },
        },
        "per_feature": results_sorted,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
