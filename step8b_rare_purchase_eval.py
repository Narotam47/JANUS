"""Step 8b — Rare legitimate purchase evaluation.

Measures false-positive rates on the specific problem this project
targets: legitimate but unusual purchases that a fraud model might
wrongly flag — the "annual sale at a new store in a different city" case.

Slice definitions
-----------------
A test-set legitimate transaction (Is Fraud? == 0) qualifies as a
"rare legitimate purchase" if ANY of these hold:

  1. AMOUNT-UNUSUAL: user_amount_zscore > 2 — the transaction amount is
     more than 2 standard deviations above the user's historical mean.
     This is the core "annual sale" signal: a legitimate user spending
     far more than their typical pattern.  (~40k test transactions)

  2. NEW-MCC: first_mcc == 1 — the user has never transacted at this
     merchant-category code before.  (~1.2k test transactions)

  3. NEW-LOCATION: first_city_state == 1 — the user has never transacted
     in this city/state combination before.  (~18.5k test transactions)

Justification: these three criteria capture the dimensions along which
a legitimate purchase looks "fraud-like" — unusual spend level, unfamiliar
merchant type, or unfamiliar geography.  A transaction that triggers any
of these would make a human reviewer look twice, yet is genuinely
legitimate.  The union gives a slice large enough for reliable rates.

Each model is evaluated at its OWN best-F1 threshold (baseline 0.9593,
enriched 0.9497), since that's the operating point each model would
actually use in production.

Output: artifacts/metrics_rare_purchase.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
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

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_rare_purchase.json"

USER_COL: Final[str] = "User"
WINDOW_SIZE: Final[int] = 10

ZSCORE_THRESHOLD: Final[float] = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step8b")


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


def fp_rate_on_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slice_mask: np.ndarray,
) -> dict:
    legit_in_slice = (~y_true.astype(bool)) & slice_mask
    n_legit = int(legit_in_slice.sum())
    if n_legit == 0:
        return {"n_legit": 0, "n_fp": 0, "fp_rate": None}
    fp = int((y_pred[legit_in_slice] == 1).sum())
    return {"n_legit": n_legit, "n_fp": fp, "fp_rate": round(fp / n_legit, 6)}


# --- Main -------------------------------------------------------------------

def main() -> None:
    for p in (WINDOWS_TEST, LABELS_TEST, CLEAN_TEST,
              USER_FEATURES_TEST, MERCH_FEATURES_TEST, FLAGS_TEST,
              MODEL_BASELINE, MODEL_ENRICHED,
              METRICS_BASELINE, METRICS_ENRICHED):
        if not p.exists():
            log.error("Missing %s — run prior steps first.", p)
            return

    # ---- Load labels and thresholds ----------------------------------------
    y_test = np.load(str(LABELS_TEST), mmap_mode="r")
    n_test = len(y_test)

    with open(METRICS_BASELINE) as f:
        base_metrics = json.load(f)
    with open(METRICS_ENRICHED) as f:
        enr_metrics = json.load(f)

    thresh_base = base_metrics["best_f1_threshold"]["best_threshold"]
    thresh_enr = enr_metrics["best_f1_threshold"]["best_threshold"]
    log.info("Thresholds: baseline=%.4f  enriched=%.4f", thresh_base, thresh_enr)

    # ---- Load enriched features aligned to windows -------------------------
    log.info("Computing window → transaction indices ...")
    idx = compute_window_feature_indices(CLEAN_TEST)
    assert len(idx) == n_test

    user_feats = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TEST).iloc[idx].reset_index(drop=True)
    log.info("Features loaded and aligned to %s test windows.", f"{n_test:,}")

    # ---- Build slice masks -------------------------------------------------
    legit = y_test == 0

    amount_unusual = (user_feats["user_amount_zscore"] > ZSCORE_THRESHOLD).values
    new_mcc = (flags["first_mcc"] == 1).values
    new_location = (flags["first_city_state"] == 1).values

    rare_any = amount_unusual | new_mcc | new_location
    ordinary = ~rare_any

    n_legit_total = int(legit.sum())
    n_rare_legit = int((legit & rare_any).sum())
    n_ordinary_legit = int((legit & ordinary).sum())

    log.info("=" * 70)
    log.info("SLICE DEFINITIONS")
    log.info("  Total test windows:           %s", f"{n_test:,}")
    log.info("  Total legitimate:             %s", f"{n_legit_total:,}")
    log.info("")
    log.info("  Amount-unusual (zscore>%.1f):  %s legit  (%.2f%%)",
             ZSCORE_THRESHOLD,
             f"{int((legit & amount_unusual).sum()):,}",
             int((legit & amount_unusual).sum()) / n_legit_total * 100)
    log.info("  New-MCC (first_mcc=1):        %s legit  (%.2f%%)",
             f"{int((legit & new_mcc).sum()):,}",
             int((legit & new_mcc).sum()) / n_legit_total * 100)
    log.info("  New-location (first_city_st): %s legit  (%.2f%%)",
             f"{int((legit & new_location).sum()):,}",
             int((legit & new_location).sum()) / n_legit_total * 100)
    log.info("")
    log.info("  RARE (union of above):        %s legit  (%.2f%%)",
             f"{n_rare_legit:,}", n_rare_legit / n_legit_total * 100)
    log.info("  ORDINARY (complement):        %s legit  (%.2f%%)",
             f"{n_ordinary_legit:,}", n_ordinary_legit / n_legit_total * 100)

    del user_feats, flags

    # ---- Predict with baseline model ---------------------------------------
    log.info("-" * 70)
    log.info("Loading baseline model and predicting ...")
    model_base = XGBClassifier()
    model_base.load_model(str(MODEL_BASELINE))

    X_base = np.load(str(WINDOWS_TEST), mmap_mode="r")
    prob_base = model_base.predict_proba(X_base)[:, 1]
    pred_base = (prob_base >= thresh_base).astype(int)
    del X_base, model_base

    # ---- Predict with enriched model ---------------------------------------
    log.info("Loading enriched model and building merged test array ...")
    model_enr = XGBClassifier()
    model_enr.load_model(str(MODEL_ENRICHED))

    X_windows = np.load(str(WINDOWS_TEST), mmap_mode="r")
    u = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].values.astype(np.float32)
    m = pd.read_parquet(MERCH_FEATURES_TEST).iloc[idx].values.astype(np.float32)
    f = pd.read_parquet(FLAGS_TEST).iloc[idx].values.astype(np.float32)
    enriched = np.hstack([u, m, f])
    del u, m, f

    X_enr = np.hstack([X_windows, enriched])
    del X_windows, enriched

    log.info("X_enriched: %s (%.1f GB)", X_enr.shape, X_enr.nbytes / 1e9)
    prob_enr = model_enr.predict_proba(X_enr)[:, 1]
    pred_enr = (prob_enr >= thresh_enr).astype(int)
    del X_enr, model_enr

    # ---- Compute FP rates per slice ----------------------------------------
    slices = {
        "rare_any": rare_any,
        "amount_unusual": amount_unusual,
        "new_mcc": new_mcc,
        "new_location": new_location,
        "ordinary": ordinary,
        "all_legit": np.ones(n_test, dtype=bool),
    }

    results: dict = {}

    log.info("=" * 70)
    log.info("FALSE-POSITIVE RATES ON LEGITIMATE TRANSACTIONS")
    log.info("(each model at its own best-F1 threshold)")
    log.info("")
    log.info("%-22s %8s  %10s %10s  %10s %10s  %8s",
             "Slice", "Legit",
             "Base FP", "Base FP%",
             "Enr FP", "Enr FP%",
             "Δ rel%")

    for name, mask in slices.items():
        r_base = fp_rate_on_slice(y_test, pred_base, mask)
        r_enr = fp_rate_on_slice(y_test, pred_enr, mask)

        n = r_base["n_legit"]
        fp_b = r_base["n_fp"]
        fp_e = r_enr["n_fp"]
        rate_b = r_base["fp_rate"] if r_base["fp_rate"] is not None else 0
        rate_e = r_enr["fp_rate"] if r_enr["fp_rate"] is not None else 0

        if rate_b > 0:
            rel_change = (rate_e - rate_b) / rate_b * 100
        else:
            rel_change = 0.0

        log.info("%-22s %8s  %10s %9.4f%%  %10s %9.4f%%  %+7.1f%%",
                 name, f"{n:,}",
                 f"{fp_b:,}", rate_b * 100,
                 f"{fp_e:,}", rate_e * 100,
                 rel_change)

        results[name] = {
            "n_legit": n,
            "baseline": {"n_fp": fp_b, "fp_rate": rate_b,
                          "threshold": thresh_base},
            "enriched": {"n_fp": fp_e, "fp_rate": rate_e,
                          "threshold": thresh_enr},
            "relative_change_pct": round(rel_change, 2),
        }

    # ---- Also report: fraud caught (TP) to check we didn't lose recall -----
    log.info("-" * 70)
    fraud_mask = y_test == 1
    n_fraud = int(fraud_mask.sum())
    tp_base = int((pred_base[fraud_mask] == 1).sum())
    tp_enr = int((pred_enr[fraud_mask] == 1).sum())
    log.info("FRAUD RECALL CHECK (test set: %s fraud)", f"{n_fraud:,}")
    log.info("  Baseline: %d / %d caught (%.1f%%)", tp_base, n_fraud, tp_base / n_fraud * 100)
    log.info("  Enriched: %d / %d caught (%.1f%%)", tp_enr, n_fraud, tp_enr / n_fraud * 100)

    results["fraud_recall"] = {
        "n_fraud": n_fraud,
        "baseline": {"tp": tp_base, "recall": round(tp_base / n_fraud, 4)},
        "enriched": {"tp": tp_enr, "recall": round(tp_enr / n_fraud, 4)},
    }

    # ---- Summary -----------------------------------------------------------
    rare = results["rare_any"]
    ordi = results["ordinary"]
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("  Rare-purchase FP rate:     baseline %.4f%% → enriched %.4f%%  (%+.1f%% relative)",
             rare["baseline"]["fp_rate"] * 100,
             rare["enriched"]["fp_rate"] * 100,
             rare["relative_change_pct"])
    log.info("  Ordinary FP rate:          baseline %.4f%% → enriched %.4f%%  (%+.1f%% relative)",
             ordi["baseline"]["fp_rate"] * 100,
             ordi["enriched"]["fp_rate"] * 100,
             ordi["relative_change_pct"])

    rare_reduction = -rare["relative_change_pct"] if rare["relative_change_pct"] < 0 else 0
    ordi_reduction = -ordi["relative_change_pct"] if ordi["relative_change_pct"] < 0 else 0

    if rare_reduction > ordi_reduction + 5:
        log.info("  → Enriched model reduces FP disproportionately on rare purchases.")
    elif abs(rare_reduction - ordi_reduction) <= 5:
        log.info("  → FP reduction is roughly uniform across rare and ordinary transactions.")
    else:
        log.info("  → FP reduction is concentrated on ordinary transactions, not rare ones.")

    log.info("=" * 70)

    # ---- Save --------------------------------------------------------------
    output = {
        "description": "False-positive rates on legitimate but unusual purchases",
        "slice_criteria": {
            "amount_unusual": f"user_amount_zscore > {ZSCORE_THRESHOLD}",
            "new_mcc": "first_mcc == 1 (first transaction at this MCC for user)",
            "new_location": "first_city_state == 1 (first transaction in this city/state for user)",
            "rare_any": "union of the three above",
            "ordinary": "complement of rare_any (none of the above fire)",
        },
        "thresholds": {
            "baseline": thresh_base,
            "enriched": thresh_enr,
        },
        "results": results,
    }

    with open(OUT_PATH, "w") as fout:
        json.dump(output, fout, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
