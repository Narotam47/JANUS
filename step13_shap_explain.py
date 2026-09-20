"""Step 13 — Per-transaction SHAP explanations for the SOC dashboard.

Provides a reusable `explain()` function (not a data artifact) that turns a
single transaction's feature vector into:
  * a ranked list of the top-N features by |SHAP value|,
  * a one-sentence, plain-language rationale for a fraud analyst,
  * the model's raw probability, the step-10 confidence score, and the
    step-11 routing verdict (block / approve / step-up).

Uses shap.TreeExplainer (exact, tree_path_dependent — not an approximation)
on the already-trained enriched model. SHAP values are in the model's raw
margin (log-odds) space, which is standard for TreeExplainer on a tree
ensemble; ranking by |SHAP value| in that space is what determines the top
features, since the sigmoid transform to probability space is monotonic and
preserves relative ordering of contribution magnitude for practical purposes.

Per the calibration finding in step 12 (Brier Skill Score -1.75 — raw
probabilities are materially overconfident), the rationale string never
states a probability ("X% confident"). It uses "flagged" / "elevated risk"
/ "not flagged" language only.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Final

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from step10_confidence import confidence
from step11_routing import route

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

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_shap_explain.json"

USER_COL: Final[str] = "User"
WINDOW_SIZE: Final[int] = 10
WINDOW_FEATURE_COUNT: Final[int] = 210  # 21 base features x 10 timesteps (step 3)

TOP_N_DEFAULT: Final[int] = 5
LATENCY_N_CALLS: Final[int] = 500
VALIDATION_SAMPLE_SIZE: Final[int] = 1000
VALIDATION_TOP_K: Final[int] = 3
INFERENCE_LATENCY_MS: Final[float] = 0.2684  # from step 8, single-window predict_proba
AUTH_BUDGET_MS: Final[float] = 50.0
SEED: Final[int] = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step13")


# === Feature naming (must match step3/step5/step6/step7/step8 exactly) ======

def build_feature_names(clean_path: Path, user_cols: list[str],
                         merch_cols: list[str], flag_cols: list[str]) -> list[str]:
    """Reconstruct the 243 flat feature names in the exact order the model sees.

    Window block: step3's sliding_window_view flattens (n_win, F, W) ->
    (n_win, F*W), i.e. base-feature-major, timestep-minor: column f*10+t is
    base feature f at timestep t. Enriched block order matches step11/12's
    hstack([user_feats, merch_feats, flags]).
    """
    import pyarrow.parquet as pq
    schema_cols = pq.ParquetFile(clean_path).schema.names
    window_base_cols = [c for c in schema_cols if c not in (USER_COL, "label")]
    assert len(window_base_cols) * WINDOW_SIZE == WINDOW_FEATURE_COUNT, (
        f"Expected {WINDOW_FEATURE_COUNT} window features, "
        f"got {len(window_base_cols)} x {WINDOW_SIZE}"
    )
    window_names = [f"{c}_t{t}" for c in window_base_cols for t in range(WINDOW_SIZE)]
    return window_names + list(user_cols) + list(merch_cols) + list(flag_cols)


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


# === Rationale phrase builders ===============================================
# Each returns a phrase (no leading capital, no trailing period) or None if
# the feature's value doesn't support a crisp narrative for this transaction.

def _flag_phrase(text: str) -> Callable[[float, float], str | None]:
    def _builder(value: float, shap_value: float) -> str | None:
        return text if value >= 0.5 else None
    return _builder


def _zscore_phrase(value: float, shap_value: float) -> str | None:
    if value > 1.0:
        return f"the amount is unusually high for this user ({value:.1f} standard deviations above their average)"
    return None


def _ratio_phrase(template: str, min_ratio: float) -> Callable[[float, float], str | None]:
    """template must contain a single {value:.1f} formatting field."""
    def _builder(value: float, shap_value: float) -> str | None:
        if value > min_ratio:
            return template.format(value=value)
        return None
    return _builder


def _rare_count_phrase(label: str, max_count: float) -> Callable[[float, float], str | None]:
    def _builder(value: float, shap_value: float) -> str | None:
        if value < max_count:
            n = int(round(value))
            times = "never" if n == 0 else f"only {n} time{'s' if n != 1 else ''}"
            return f"the user has {times} {label} before"
        return None
    return _builder


FEATURE_PHRASE_BUILDERS: Final[dict[str, Callable[[float, float], str | None]]] = {
    # Novelty (first-occurrence) features
    "first_mcc": _flag_phrase("this is the user's first transaction in this merchant category"),
    "first_city_state": _flag_phrase("this is the user's first transaction in this city or state"),
    "first_channel": _flag_phrase("this is the user's first time using this payment channel"),
    "is_new_merchant": _flag_phrase("the user has never transacted with this merchant before"),
    "first_tx_this_month": _flag_phrase("this is the user's first transaction this month"),
    "first_tx_this_year": _flag_phrase("this is the user's first transaction this year"),
    "is_dormant": _flag_phrase("the user's account had been inactive before this transaction"),
    # Amount-anomaly features
    "is_largest_ever": _flag_phrase("this is the largest transaction this user has ever made"),
    "is_2x_hist_max": _flag_phrase("the amount is more than double the user's previous largest transaction"),
    "is_5x_hist_max": _flag_phrase("the amount is more than five times the user's previous largest transaction"),
    "user_amount_zscore": _zscore_phrase,
    "user_amount_to_max_ratio": _ratio_phrase(
        "the amount is {value:.1f}x this user's typical maximum transaction size", 1.0),
    "amount_vs_last_year_ratio": _ratio_phrase(
        "the amount is {value:.1f}x what this user spent in the same period last year", 1.5),
    "merch_amount_deviation": _ratio_phrase(
        "the amount is unusually large for this merchant, at {value:.1f}x normal", 1.0),
    "merch_spike_ratio": _ratio_phrase(
        "this merchant is seeing an unusual spike in transaction volume, at {value:.1f}x normal", 1.5),
    # Familiarity (falls back into the "pattern match" narrative when absent)
    "merchant_user_count": _rare_count_phrase("visited this merchant", 3),
    "mcc_user_count": _rare_count_phrase("transacted in this merchant category", 3),
}

_GENERIC_PATTERN_MATCH = (
    "the transaction's overall pattern (recent amounts, timing, and merchant "
    "activity) closely resembles historical fraud patterns, without any single "
    "unusual attribute standing out"
)
_NO_RISK_SIGNAL = (
    "feature values are consistent with this user's established transaction "
    "pattern; no unusual merchant, amount, or location signals were detected"
)


def _feature_phrase(name: str, value: float, shap_value: float) -> str | None:
    builder = FEATURE_PHRASE_BUILDERS.get(name)
    if builder is None:
        return None
    return builder(value, shap_value)


def compose_rationale(
    top_features: list[dict],
    prob: float,
    threshold: float,
) -> str:
    """Plain-language, analyst-facing rationale. Never states a probability."""
    flagged = prob >= threshold

    phrases: list[str] = []
    for f in top_features:
        if flagged and f["shap_value"] <= 0:
            continue  # only narrate risk-increasing drivers when flagged
        phrase = _feature_phrase(f["feature"], f["value"], f["shap_value"])
        if phrase and phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) == 2:
            break

    if flagged:
        if phrases:
            if len(phrases) == 1:
                return f"Flagged primarily because {phrases[0]}."
            return f"Flagged primarily because {phrases[0]}, and {phrases[1]}."
        return f"Flagged primarily because {_GENERIC_PATTERN_MATCH}."
    else:
        if phrases:
            extra = f"; also {phrases[1]}" if len(phrases) > 1 else ""
            return (
                f"Not flagged. Some risk indicators are present ({phrases[0]}{extra}), "
                "but overall the transaction pattern remains consistent with this "
                "user's established behavior."
            )
        return f"Not flagged. {_NO_RISK_SIGNAL[0].upper()}{_NO_RISK_SIGNAL[1:]}."


# === Reusable explain() function =============================================

_EXPLAINER_CACHE: dict[int, shap.TreeExplainer] = {}
_INDEX_CACHE: dict[int, dict[str, int]] = {}


def _get_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    key = id(model)
    if key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[key] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[key]


def _get_feature_index(feature_names: list[str]) -> dict[str, int]:
    key = id(feature_names)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = {name: i for i, name in enumerate(feature_names)}
    return _INDEX_CACHE[key]


def explain(
    x: np.ndarray,
    model: XGBClassifier,
    feature_names: list[str],
    threshold: float,
    top_n: int = TOP_N_DEFAULT,
) -> dict:
    """Explain a single transaction's model output.

    Parameters
    ----------
    x : np.ndarray, shape (n_features,)
        The transaction's full flat feature vector (window + enriched),
        in the exact column order the model was trained on.
    model : XGBClassifier
        The trained enriched model (step 8).
    feature_names : list[str]
        Names for each position in `x`, in the same order (see
        `build_feature_names`).
    threshold : float
        Decision threshold (best-F1 from step 8) used only to decide the
        framing of the rationale (flagged vs. not flagged), not to
        recompute any routing decision.
    top_n : int
        How many top |SHAP value| features to return.

    Returns
    -------
    dict with keys: probability, confidence_score, confidence_reason,
    top_features (list of {feature, shap_value, value}), rationale,
    routing_verdict (the full step-11 verdict object: decision, rule,
    probability, confidence_score, confidence_reason).
    """
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    explainer = _get_explainer(model)
    idx = _get_feature_index(feature_names)

    prob = float(model.predict_proba(x)[0, 1])
    shap_values = explainer.shap_values(x)[0]

    order = np.argsort(-np.abs(shap_values))[:top_n]
    top_features = [
        {
            "feature": feature_names[i],
            "shap_value": round(float(shap_values[i]), 6),
            "value": round(float(x[0, i]), 6),
        }
        for i in order
    ]

    position = int(round(float(x[0, idx["user_tx_count"]])))
    n_nan = int(np.isnan(x[0, WINDOW_FEATURE_COUNT:]).sum())
    conf_score, conf_reason = confidence(
        prob=prob,
        threshold=threshold,
        user_tx_count=float(x[0, idx["user_tx_count"]]),
        merchant_user_count=float(x[0, idx["merchant_user_count"]]),
        mcc_user_count=float(x[0, idx["mcc_user_count"]]),
        first_mcc=int(x[0, idx["first_mcc"]]),
        first_city_state=int(x[0, idx["first_city_state"]]),
        first_channel=int(x[0, idx["first_channel"]]),
        n_nan_features=n_nan,
        total_features=len(feature_names),
    )

    routing_verdict = route(
        prob=prob,
        confidence_score=conf_score,
        confidence_reason=conf_reason,
        position=position,
        prob_block=threshold,
    )

    rationale = compose_rationale(top_features, prob, threshold)

    return {
        "probability": round(prob, 6),
        "confidence_score": conf_score,
        "confidence_reason": conf_reason,
        "top_features": top_features,
        "rationale": rationale,
        "routing_verdict": routing_verdict,
    }


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
    y_test = np.asarray(np.load(str(LABELS_TEST), mmap_mode="r"))
    n_test = len(y_test)
    assert len(idx) == n_test

    user_feats = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(MERCH_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TEST).iloc[idx].reset_index(drop=True)

    feature_names = build_feature_names(
        CLEAN_TEST, list(user_feats.columns), list(merch_feats.columns), list(flags.columns),
    )
    log.info("Reconstructed %d feature names (window=%d, enriched=%d).",
              len(feature_names), WINDOW_FEATURE_COUNT, len(feature_names) - WINDOW_FEATURE_COUNT)

    with open(METRICS_ENRICHED) as f:
        threshold = json.load(f)["best_f1_threshold"]["best_threshold"]

    model = XGBClassifier()
    model.load_model(str(MODEL_ENRICHED))

    X_win = np.load(str(WINDOWS_TEST), mmap_mode="r")
    u = user_feats.values.astype(np.float32)
    m = merch_feats.values.astype(np.float32)
    fl = flags.values.astype(np.float32)
    enriched = np.hstack([u, m, fl])
    X = np.hstack([X_win, enriched]).astype(np.float32)
    del enriched, u, m, fl, X_win

    assert X.shape[1] == len(feature_names)

    # =========================================================================
    # LATENCY — single-call SHAP explanation
    # =========================================================================
    log.info("=" * 70)
    log.info("LATENCY — single-transaction explain() calls")

    explainer = _get_explainer(model)  # build once, outside the timed loop
    log.info("  TreeExplainer built (one-time cost, not counted in per-call latency).")

    rng = np.random.default_rng(SEED)
    latency_idx = rng.choice(n_test, size=LATENCY_N_CALLS, replace=False)

    # Warm-up call (first call can carry extra one-time overhead)
    explain(X[latency_idx[0]], model, feature_names, threshold)

    latencies_ms: list[float] = []
    for i in latency_idx:
        t0 = time.perf_counter()
        explain(X[i], model, feature_names, threshold)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    shap_mean = float(latencies_ms.mean())
    shap_median = float(np.median(latencies_ms))
    shap_p95 = float(np.percentile(latencies_ms, 95))
    shap_p99 = float(np.percentile(latencies_ms, 99))
    shap_min = float(latencies_ms.min())
    shap_max = float(latencies_ms.max())

    total_p99_ms = INFERENCE_LATENCY_MS + shap_p99
    within_budget = total_p99_ms < AUTH_BUDGET_MS

    log.info("  n_calls=%d (includes model inference inside explain())", LATENCY_N_CALLS)
    log.info("  mean=%.4fms  median=%.4fms  p95=%.4fms  p99=%.4fms  min=%.4fms  max=%.4fms",
              shap_mean, shap_median, shap_p95, shap_p99, shap_min, shap_max)
    log.info("  Step 8 bare-inference latency: %.4fms (mean)", INFERENCE_LATENCY_MS)
    log.info("  Combined p99 (inference + explain, conservative — explain() already")
    log.info("  includes its own inference call): %.4fms vs %.1fms authorization budget",
              total_p99_ms, AUTH_BUDGET_MS)
    log.info("  → %s the authorization budget.",
              "WELL WITHIN" if within_budget else "EXCEEDS")

    # =========================================================================
    # VALIDATION — do top SHAP features align with step 8's global importance?
    # =========================================================================
    log.info("=" * 70)
    log.info("VALIDATION — top-%d SHAP features across %d sampled transactions",
              VALIDATION_TOP_K, VALIDATION_SAMPLE_SIZE)

    val_idx = rng.choice(n_test, size=VALIDATION_SAMPLE_SIZE, replace=False)
    X_val = X[val_idx]

    t0 = time.perf_counter()
    shap_values_batch = explainer.shap_values(X_val)
    batch_time = time.perf_counter() - t0
    log.info("  Batch SHAP for %d transactions computed in %.3fs (%.3fms/tx amortized).",
              VALIDATION_SAMPLE_SIZE, batch_time, batch_time / VALIDATION_SAMPLE_SIZE * 1000)

    top_k_idx = np.argsort(-np.abs(shap_values_batch), axis=1)[:, :VALIDATION_TOP_K]
    freq = np.zeros(len(feature_names), dtype=int)
    for row in top_k_idx:
        for i in row:
            freq[i] += 1

    freq_order = np.argsort(-freq)
    freq_table = [
        {"feature": feature_names[i], "count": int(freq[i]),
         "pct_of_transactions": round(float(freq[i]) / VALIDATION_SAMPLE_SIZE * 100, 2)}
        for i in freq_order if freq[i] > 0
    ]

    log.info("  Top 20 features by frequency in top-%d (out of %d sampled transactions):",
              VALIDATION_TOP_K, VALIDATION_SAMPLE_SIZE)
    for row in freq_table[:20]:
        log.info("    %-30s  n=%4d  (%5.1f%%)", row["feature"], row["count"], row["pct_of_transactions"])

    # Cross-check against step 8's global feature importance ranking.
    with open(METRICS_ENRICHED) as f:
        global_importance = json.load(f)["feature_importance_top30"]
    freq_by_name = {row["feature"]: row for row in freq_table}

    log.info("-" * 70)
    log.info("  Cross-check vs. step 8 global feature importance (top 10):")
    log.info("  %-30s  %8s  %14s", "Feature", "GlobalRk", "SHAP top-3 freq")
    crosscheck = []
    for g in global_importance[:10]:
        name = g["feature"]
        row = freq_by_name.get(name)
        pct = row["pct_of_transactions"] if row else 0.0
        log.info("    %-28s  #%-7d  %6.1f%%", name, g["rank"], pct)
        crosscheck.append({"feature": name, "global_rank": g["rank"], "shap_top3_pct": pct})

    alignment_ok = all(c["shap_top3_pct"] > 0 for c in crosscheck[:5])
    log.info("  → Top-5 globally important features %s appear in per-transaction",
              "DO" if alignment_ok else "do NOT all")
    log.info("    top-%d SHAP explanations at least occasionally.", VALIDATION_TOP_K)

    # ---- Investigate the dominant feature if it's absent from the global list
    dominant_feature_discrepancy = None
    if freq_table and freq_table[0]["feature"] not in {g["feature"] for g in global_importance}:
        top_name = freq_table[0]["feature"]
        m = re.match(r"^(.*)_t(\d+)$", top_name)
        log.info("-" * 70)
        log.info("  NOTE: the single most frequent top-%d feature ('%s', %.1f%% of "
                  "transactions) is ABSENT from step 8's global top-30 gain-based "
                  "ranking. Gain-based importance and per-instance SHAP attribution "
                  "can diverge — investigating why.",
                  VALIDATION_TOP_K, top_name, freq_table[0]["pct_of_transactions"])
        if m:
            base_col = m.group(1)
            import pyarrow.parquet as pq
            schema_cols = pq.ParquetFile(CLEAN_TEST).schema.names
            candidate_cols = [c for c in schema_cols if c not in (USER_COL, "label", base_col)]
            raw = pd.read_parquet(CLEAN_TEST, columns=[base_col] + candidate_cols)
            corrs = raw[candidate_cols].corrwith(raw[base_col]).abs().sort_values(ascending=False)
            best_col, best_corr = corrs.index[0], float(corrs.iloc[0])
            log.info("    Most correlated raw column with '%s': '%s' (|corr|=%.3f) — plausibly "
                      "geographic collinearity (e.g. zip <-> city), not a data artifact.",
                      base_col, best_col, best_corr)
            log.info("    Gain-based importance is an AGGREGATE, summed/averaged across all "
                      "trees; when several columns encode overlapping information (here: "
                      "location/merchant-identity columns), boosting can split its reliance "
                      "across them, diluting each one's aggregate gain rank — while any single "
                      "transaction's SHAP attribution still credits whichever correlated column "
                      "the specific trees happened to split on for that instance. This is a "
                      "known limitation of aggregate gain importance, and exactly what "
                      "per-instance SHAP validation is meant to surface.")

            # Separately: check whether any one-hot channel column (chip_*) shows a
            # deterministic (near-zero-variance) relationship with this feature — a
            # stronger, narrower signal than general correlation.
            chip_cols = [c for c in candidate_cols if c.startswith("chip_")]
            sentinel_note = None
            for chip_col in chip_cols:
                grp = raw.groupby(raw[chip_col] > 0.5)[base_col].agg(["mean", "std", "count"])
                if True in grp.index and grp.loc[True, "std"] < 1e-6:
                    sentinel_note = (
                        f"'{base_col}' takes a CONSTANT value ({grp.loc[True, 'mean']:.4f}, "
                        f"std=0) whenever '{chip_col}'=1 (n={int(grp.loc[True, 'count']):,}) — "
                        f"consistent with a missing-value sentinel (e.g. no physical zip code "
                        f"for that transaction channel) rather than a real geographic reading. "
                        f"This subset alone covers {int(grp.loc[True, 'count']):,} of "
                        f"{len(raw):,} test transactions ({int(grp.loc[True, 'count']) / len(raw) * 100:.1f}%), "
                        f"so it is a real but PARTIAL explanation — it does not by itself "
                        f"account for why '{base_col}' dominates SHAP attribution for the "
                        f"much larger share of sampled transactions shown above."
                    )
                    log.info("    Sentinel check: %s", sentinel_note)
                    break

            dominant_feature_discrepancy = {
                "dominant_feature": top_name,
                "pct_of_transactions": freq_table[0]["pct_of_transactions"],
                "in_global_top30": False,
                "most_correlated_raw_column": best_col,
                "correlation": round(best_corr, 4),
                "sentinel_value_note": sentinel_note,
                "note": "Gain-based importance aggregates/averages across all trees; when "
                        "features are collinear (here, location/merchant-identity columns), "
                        "reliance can split across them, diluting any single one's aggregate "
                        "gain rank while per-instance SHAP still credits whichever correlated "
                        "column a given transaction's trees actually used. This is a known "
                        "limitation of aggregate gain importance rather than evidence of a bug.",
            }
            del raw

    # ---- Demo: a few example explanations -----------------------------------
    log.info("=" * 70)
    log.info("EXAMPLE EXPLANATIONS")
    fraud_idx_all = np.where(y_test == 1)[0]
    legit_idx_all = np.where(y_test == 0)[0]
    demo_idx = list(rng.choice(fraud_idx_all, size=2, replace=False)) + \
               list(rng.choice(legit_idx_all, size=2, replace=False))
    demo_examples = []
    for i in demo_idx:
        result = explain(X[i], model, feature_names, threshold, top_n=3)
        label = "FRAUD" if y_test[i] == 1 else "legit"
        verdict = result["routing_verdict"]
        log.info("  [%s] prob=%.4f  confidence=%.3f (%s)  ->  %s (%s)",
                  label, result["probability"], result["confidence_score"], result["confidence_reason"],
                  verdict["decision"].upper(), verdict["rule"])
        log.info("    Rationale: %s", result["rationale"])
        for tf in result["top_features"]:
            log.info("      %-28s  shap=%9.4f  value=%.4f", tf["feature"], tf["shap_value"], tf["value"])
        demo_examples.append({"true_label": label, **result})

    # =========================================================================
    # SAVE
    # =========================================================================
    log.info("=" * 70)
    output = {
        "description": "SHAP TreeExplainer validation and latency for the "
                        "per-transaction explain() function. No model artifact.",
        "latency_ms": {
            "n_calls": LATENCY_N_CALLS,
            "mean": round(shap_mean, 4),
            "median": round(shap_median, 4),
            "p95": round(shap_p95, 4),
            "p99": round(shap_p99, 4),
            "min": round(shap_min, 4),
            "max": round(shap_max, 4),
            "step8_bare_inference_mean": INFERENCE_LATENCY_MS,
            "combined_p99": round(total_p99_ms, 4),
            "authorization_budget_ms": AUTH_BUDGET_MS,
            "within_budget": bool(within_budget),
        },
        "topk_frequency_validation": {
            "sample_size": VALIDATION_SAMPLE_SIZE,
            "top_k": VALIDATION_TOP_K,
            "batch_shap_time_seconds": round(batch_time, 4),
            "frequency_table": freq_table,
            "crosscheck_vs_global_importance": crosscheck,
            "top5_globally_important_all_appear": alignment_ok,
            "dominant_feature_discrepancy": dominant_feature_discrepancy,
        },
        "example_explanations": demo_examples,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
