"""Step 21 — adversarial retraining: real data + CTGAN augmentation +
harvested PPO evasions, then measure whether it actually hardens the model.

CTGAN FEATURE MISMATCH — handled explicitly, not glossed over
---------------------------------------------------------------
`synthetic_fraud.parquet` has 33 columns: 21 are the raw window-base
features (Merchant Name, amount_raw, hour, chip_*, ...) and 12 are enriched
user-profile features (user_hist_mean, user_amount_zscore, merchant_user_count,
...) — see step 19. Mapped onto the model's 243-dim space:

  * The 21 raw columns populate the window block's LAST timestep (t9 — "this
    is the current/most-recent transaction"), using their real column names
    to find the right flat index. That's 21 of 210 window slots.
  * The 12 enriched columns populate their exact corresponding slots among
    the 33 enriched features.
  * That's 21 + 12 = 33 features CTGAN actually supplies. The remaining
    210 (189 window slots for the 9 PRIOR timesteps CTGAN never modeled,
    plus 21 enriched features CTGAN excluded: same_month_last_year_mean,
    amount_vs_last_year_ratio, and all 9 merch_*/pop_* + 10 flags features)
    are filled with the MEAN of the corresponding feature among the 9,220
    REAL fraud TRAINING windows — not zero, and not excluded.

Zero-padding was rejected: for the 189 missing window timesteps, zero would
manufacture a fake "9 all-zero prior transactions" pattern with no basis in
reality, and the model could learn to associate that specific artifact with
fraud — a spurious shortcut, not a hardening signal. The empirical real-fraud
mean is a much more defensible fill: "the surrounding context of a typical
real fraud transaction," even though the SPECIFIC synthetic row's actual
prior history is unknown. Excluding CTGAN rows entirely was also rejected —
the task calls for using them as training augmentation, not just as the
validation-only signal step 19 already produced.

HARVESTED EVASIONS: `evaders.parquet`'s 243 feature columns are already a
complete, real window+enriched vector (the actual perturbed transaction that
fooled the original model) — no reconstruction needed, just select the 243
named columns in order and label them fraud (1). They ARE fraud (drawn from
real fraud test transactions), just successfully disguised — training on
them directly targets the disguise the step-20 attacker found.

EXPERIMENTAL CONTROL: hyperparameters (scale_pos_weight=794.0, max_depth=6,
learning_rate=0.1, n_estimators=300, ...) are reused UNCHANGED from
metrics_enriched.json, not recomputed for the new class balance. This is
deliberate, per the task: the only variable that changes is the training
data, so any change in evaluation metrics is attributable to the
augmentation, not to a simultaneously-retuned hyperparameter.
"""

from __future__ import annotations

# Must precede any import touching OpenMP (xgboost, torch) — see step 20.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from xgboost import XGBClassifier  # noqa: E402

import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Final  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from step13_shap_explain import build_feature_names, compute_window_feature_indices  # noqa: E402
from step20_ppo_attacker import (  # noqa: E402
    FraudEvasionEnv,
    HARVEST_ATTEMPTS_PER_TRANSACTION,
    PERTURBABLE_BINARY,
    PERTURBABLE_CONTINUOUS,
    load_fraud_test_vectors,
    rollout_evasion_rate,
)

# --- Config -------------------------------------------------------------
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
SYNTHETIC_FRAUD: Final[Path] = ARTIFACTS_DIR / "synthetic_fraud.parquet"
EVADERS_PATH: Final[Path] = ARTIFACTS_DIR / "evaders.parquet"
PPO_POLICY_PATH: Final[Path] = ARTIFACTS_DIR / "ppo_policy.zip"

MODEL_HARDENED_PATH: Final[Path] = ARTIFACTS_DIR / "model_hardened.json"
METRICS_ADVERSARIAL_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_adversarial.json"

WINDOW_SIZE: Final[int] = 10
LAST_STEP: Final[int] = WINDOW_SIZE - 1
WINDOW_FEATURE_COUNT: Final[int] = 210

SYNTHETIC_RAW_COLS: Final[tuple[str, ...]] = (
    "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
    "amount_raw", "amount_log", "hours_since_last",
    "err_bad_cvv", "err_bad_card_number", "err_bad_expiration", "err_bad_pin",
    "err_bad_zipcode", "err_insufficient_balance", "err_technical_glitch",
    "hour", "day_of_week", "month",
    "chip_Chip Transaction", "chip_Online Transaction", "chip_Swipe Transaction",
)
SYNTHETIC_ENRICHED_COLS: Final[tuple[str, ...]] = (
    "user_hist_mean", "user_hist_std", "user_hist_max", "user_tx_count",
    "user_amount_zscore", "user_amount_to_max_ratio",
    "merchant_user_count", "mcc_user_count", "is_new_merchant",
    "is_weekend", "is_holiday_season", "day_of_year",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step21")


# === Real training data (identical construction to step 8) ===================

def load_real_training_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], list[str]]:
    idx = compute_window_feature_indices(CLEAN_TRAIN)
    y_train = np.asarray(np.load(str(LABELS_TRAIN), mmap_mode="r"))

    user_feats = pd.read_parquet(USER_FEATURES_TRAIN).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(MERCH_FEATURES_TRAIN).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TRAIN).iloc[idx].reset_index(drop=True)
    feature_names = build_feature_names(
        CLEAN_TRAIN, list(user_feats.columns), list(merch_feats.columns), list(flags.columns),
    )
    window_base_cols = [c for c in pq.ParquetFile(CLEAN_TRAIN).schema.names if c not in ("User", "label")]

    log.info("Loading windows_train.npy (%s rows) ...", f"{len(y_train):,}")
    X_win = np.load(str(WINDOWS_TRAIN), mmap_mode="r")
    enriched = np.hstack([
        user_feats.values.astype(np.float32),
        merch_feats.values.astype(np.float32),
        flags.values.astype(np.float32),
    ])
    X_train = np.hstack([np.asarray(X_win, dtype=np.float32), enriched])
    del X_win, enriched
    log.info("Real training matrix: %s, %s fraud (%.4f%%)",
              X_train.shape, f"{int(y_train.sum()):,}", y_train.mean() * 100)

    return X_train, y_train, feature_names, window_base_cols, list(user_feats.columns), list(merch_feats.columns)


def compute_real_fraud_means(
    X_train: np.ndarray, y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fraud_rows = X_train[y_train == 1]
    mean_window = fraud_rows[:, :WINDOW_FEATURE_COUNT].mean(axis=0)
    mean_enriched = fraud_rows[:, WINDOW_FEATURE_COUNT:].mean(axis=0)
    return mean_window, mean_enriched


def build_synthetic_augmentation(
    synthetic_df: pd.DataFrame,
    window_base_cols: list[str],
    enriched_names: list[str],
    mean_window: np.ndarray,
    mean_enriched: np.ndarray,
) -> np.ndarray:
    """Converts CTGAN's 33-column synthetic fraud rows into full 243-dim
    vectors. See module docstring for the exact fill strategy and why."""
    n = len(synthetic_df)
    window_block = np.tile(mean_window, (n, 1)).astype(np.float32)
    enriched_block = np.tile(mean_enriched, (n, 1)).astype(np.float32)

    n_window_filled = 0
    for f_idx, base_col in enumerate(window_base_cols):
        if base_col in synthetic_df.columns:
            flat_idx = f_idx * WINDOW_SIZE + LAST_STEP
            window_block[:, flat_idx] = synthetic_df[base_col].values.astype(np.float32)
            n_window_filled += 1

    n_enriched_filled = 0
    for e_idx, name in enumerate(enriched_names):
        if name in synthetic_df.columns:
            enriched_block[:, e_idx] = synthetic_df[name].values.astype(np.float32)
            n_enriched_filled += 1

    log.info("Synthetic augmentation: %d/%d window slots (t9 only) and %d/%d enriched slots "
              "populated directly from CTGAN; remaining %d slots filled with the real-fraud-"
              "training mean.",
              n_window_filled, WINDOW_FEATURE_COUNT, n_enriched_filled, len(enriched_names),
              (WINDOW_FEATURE_COUNT - n_window_filled) + (len(enriched_names) - n_enriched_filled))

    return np.hstack([window_block, enriched_block]).astype(np.float32)


def load_evaders_as_training_rows(feature_names: list[str]) -> np.ndarray:
    df = pd.read_parquet(EVADERS_PATH)
    X = df[feature_names].values.astype(np.float32)
    log.info("Loaded %s harvested evasions as additional fraud training rows (already full 243-dim).",
              f"{len(df):,}")
    return X


# === Standard test-set evaluation (unchanged from steps 8/11/12/13) ===========

def load_test_set() -> tuple[np.ndarray, np.ndarray]:
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
    return X_test, y_test


def evaluate_standard(model: XGBClassifier, X: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "auc_pr": float(average_precision_score(y, prob)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


# === Adversarial evaluation: reuse step 20's exact attacker/methodology =======

def evaluate_adversarial(model: XGBClassifier, threshold: float) -> dict:
    """Reuses step 20's ALREADY-TRAINED attacker policy (not retrained) and
    the SAME fixed catchable-fraud population it was evaluated on, so the
    only thing that changes between the step-20 number and this one is the
    defender. This answers "does hardening defeat the SPECIFIC exploit
    step 20 found" — a transfer-robustness check, not proof the hardened
    model resists every possible future attack (a fresh adaptive attacker
    could still find something new; that's a natural next iteration, not
    what this step measures).
    """
    X_fraud_all, _, feature_names = load_fraud_test_vectors()
    prob_orig = model.predict_proba(X_fraud_all)[:, 1]
    caught_mask = prob_orig >= threshold
    X_catchable = X_fraud_all[caught_mask]
    n_catchable = len(X_catchable)
    log.info("  Hardened-model catchable pool: %d/%d fraud transactions score >= threshold "
              "(vs. 1,645 for the original model).", n_catchable, len(X_fraud_all))

    continuous_idx = np.array([feature_names.index(n) for n in PERTURBABLE_CONTINUOUS])
    binary_idx = np.array([feature_names.index(n) for n in PERTURBABLE_BINARY])
    continuous_scale = np.array([max(X_catchable[:, i].std(), 1e-6) for i in continuous_idx])

    env = FraudEvasionEnv(model, X_catchable, continuous_idx, continuous_scale, binary_idx, threshold)
    policy = PPO.load(str(PPO_POLICY_PATH))

    det_rate, _ = rollout_evasion_rate(policy, env, n_catchable, deterministic=True, seed_offset=30_000)

    all_successes: dict[int, dict] = {}
    for attempt in range(HARVEST_ATTEMPTS_PER_TRANSACTION):
        _, successes = rollout_evasion_rate(
            policy, env, n_catchable, deterministic=False, seed_offset=40_000 + attempt * n_catchable,
        )
        for s in successes:
            all_successes.setdefault(s["row_idx"], s)
    harvest_rate = len(all_successes) / n_catchable if n_catchable else 0.0

    return {
        "n_catchable": n_catchable,
        "evasion_rate_deterministic": det_rate,
        "evasion_rate_harvest": harvest_rate,
        "n_evaders_harvested": len(all_successes),
    }


# === Main =====================================================================

def main() -> None:
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 21 — ADVERSARIAL RETRAINING")
    log.info("=" * 70)

    for p in (SYNTHETIC_FRAUD, EVADERS_PATH, PPO_POLICY_PATH, MODEL_ENRICHED, METRICS_ENRICHED):
        if not p.exists():
            log.error("Missing required artifact: %s — run steps 8, 19, 20 first.", p)
            return

    with open(METRICS_ENRICHED) as f:
        base_metrics = json.load(f)
    threshold = base_metrics["best_f1_threshold"]["best_threshold"]
    xgb_params = dict(base_metrics["training"]["xgb_params"])
    log.info("Reusing step 8's hyperparameters UNCHANGED: %s", xgb_params)
    log.info("Fixed evaluation threshold (unchanged, for a clean comparison): %.4f", threshold)

    log.info("-" * 70)
    log.info("Loading real training data (identical to step 8) ...")
    X_train_real, y_train_real, feature_names, window_base_cols, user_cols, merch_cols = load_real_training_data()
    enriched_names = feature_names[WINDOW_FEATURE_COUNT:]  # 33 enriched names, in hstack order

    mean_window, mean_enriched = compute_real_fraud_means(X_train_real, y_train_real)

    log.info("-" * 70)
    log.info("Converting CTGAN synthetic fraud (33 cols) to 243-dim window+enriched vectors ...")
    synthetic_df = pd.read_parquet(SYNTHETIC_FRAUD)
    X_synth = build_synthetic_augmentation(synthetic_df, window_base_cols, enriched_names, mean_window, mean_enriched)

    log.info("-" * 70)
    log.info("Loading harvested PPO evasions as additional fraud rows ...")
    X_evaders = load_evaders_as_training_rows(feature_names)

    log.info("-" * 70)
    log.info("Assembling augmented training set ...")
    X_train_aug = np.vstack([X_train_real, X_synth, X_evaders]).astype(np.float32)
    y_train_aug = np.concatenate([
        y_train_real, np.ones(len(X_synth), dtype=y_train_real.dtype), np.ones(len(X_evaders), dtype=y_train_real.dtype),
    ])
    del X_train_real, X_synth, X_evaders
    n_fraud_before = int(y_train_real.sum())
    n_fraud_after = int(y_train_aug.sum())
    log.info("Training set: %s -> %s rows. Fraud: %s -> %s (+%s: %s synthetic + %s evaders).",
              f"{len(y_train_real):,}", f"{len(y_train_aug):,}",
              f"{n_fraud_before:,}", f"{n_fraud_after:,}", f"{n_fraud_after - n_fraud_before:,}",
              f"{len(synthetic_df):,}", f"{n_fraud_after - n_fraud_before - len(synthetic_df):,}")

    log.info("-" * 70)
    log.info("Loading standard held-out test set (unchanged) ...")
    X_test, y_test = load_test_set()

    log.info("-" * 70)
    log.info("Evaluating ORIGINAL model (model_enriched.json) — baseline for the comparison ...")
    original_model = XGBClassifier()
    original_model.load_model(str(MODEL_ENRICHED))
    original_standard = evaluate_standard(original_model, X_test, y_test, threshold)
    log.info("  Standard: AUC-PR=%.6f  F1=%.6f  precision=%.4f  recall=%.4f",
              original_standard["auc_pr"], original_standard["f1"],
              original_standard["precision"], original_standard["recall"])

    log.info("-" * 70)
    log.info("TRAINING hardened model from scratch on the augmented data (same hyperparameters "
              "as step 8, %d estimators) — this is the slow step ...", xgb_params.get("n_estimators", 300))
    t_train0 = time.perf_counter()
    hardened_model = XGBClassifier(**xgb_params)
    hardened_model.fit(X_train_aug, y_train_aug)
    train_time = time.perf_counter() - t_train0
    log.info("Hardened model trained in %.1fs (%.1f min).", train_time, train_time / 60)
    hardened_model.save_model(str(MODEL_HARDENED_PATH))
    log.info("Saved: %s", MODEL_HARDENED_PATH)

    log.info("-" * 70)
    log.info("Evaluating HARDENED model on the standard test set ...")
    hardened_standard = evaluate_standard(hardened_model, X_test, y_test, threshold)
    log.info("  Standard: AUC-PR=%.6f  F1=%.6f  precision=%.4f  recall=%.4f",
              hardened_standard["auc_pr"], hardened_standard["f1"],
              hardened_standard["precision"], hardened_standard["recall"])
    del X_train_aug, X_test

    log.info("-" * 70)
    log.info("ADVERSARIAL EVALUATION — re-running step 20's trained attacker against each model")
    log.info("Against the ORIGINAL model (should reproduce step 20's numbers):")
    original_adversarial = evaluate_adversarial(original_model, threshold)
    log.info("  Evasion rate: deterministic=%.2f%%  harvest(any of %d)=%.2f%%",
              original_adversarial["evasion_rate_deterministic"] * 100, HARVEST_ATTEMPTS_PER_TRANSACTION,
              original_adversarial["evasion_rate_harvest"] * 100)

    log.info("Against the HARDENED model (the headline result):")
    hardened_adversarial = evaluate_adversarial(hardened_model, threshold)
    log.info("  Evasion rate: deterministic=%.2f%%  harvest(any of %d)=%.2f%%",
              hardened_adversarial["evasion_rate_deterministic"] * 100, HARVEST_ATTEMPTS_PER_TRANSACTION,
              hardened_adversarial["evasion_rate_harvest"] * 100)

    # === Comparison table =====================================================
    log.info("=" * 70)
    log.info("FULL COMPARISON — original vs. hardened")
    log.info("%-30s %15s %15s %15s", "Metric", "Original", "Hardened", "Change")
    log.info("%-30s %15.6f %15.6f %+15.6f", "Standard AUC-PR",
              original_standard["auc_pr"], hardened_standard["auc_pr"],
              hardened_standard["auc_pr"] - original_standard["auc_pr"])
    log.info("%-30s %15.6f %15.6f %+15.6f", f"Standard F1@{threshold:.4f}",
              original_standard["f1"], hardened_standard["f1"],
              hardened_standard["f1"] - original_standard["f1"])
    log.info("%-30s %15.6f %15.6f %+15.6f", "Standard precision",
              original_standard["precision"], hardened_standard["precision"],
              hardened_standard["precision"] - original_standard["precision"])
    log.info("%-30s %15.6f %15.6f %+15.6f", "Standard recall",
              original_standard["recall"], hardened_standard["recall"],
              hardened_standard["recall"] - original_standard["recall"])
    log.info("%-30s %14.2f%% %14.2f%% %+14.2fpp", "Adversarial evasion (det.)",
              original_adversarial["evasion_rate_deterministic"] * 100,
              hardened_adversarial["evasion_rate_deterministic"] * 100,
              (hardened_adversarial["evasion_rate_deterministic"] - original_adversarial["evasion_rate_deterministic"]) * 100)
    log.info("%-30s %14.2f%% %14.2f%% %+14.2fpp", "Adversarial evasion (harvest)",
              original_adversarial["evasion_rate_harvest"] * 100,
              hardened_adversarial["evasion_rate_harvest"] * 100,
              (hardened_adversarial["evasion_rate_harvest"] - original_adversarial["evasion_rate_harvest"]) * 100)

    auc_pr_change = hardened_standard["auc_pr"] - original_standard["auc_pr"]
    f1_change = hardened_standard["f1"] - original_standard["f1"]
    evasion_change = hardened_adversarial["evasion_rate_harvest"] - original_adversarial["evasion_rate_harvest"]

    log.info("-" * 70)
    if evasion_change < -0.02:
        log.info("HARDENING WORKED: harvest evasion rate dropped by %.1fpp against the SAME "
                  "attacker policy.", -evasion_change * 100)
    elif evasion_change > 0.02:
        log.warning("HARDENING BACKFIRED: harvest evasion rate INCREASED by %.1fpp. The "
                    "augmented data did not close the gap the step-20 attacker found, and may "
                    "have introduced new weaknesses (e.g. via the mean-imputed synthetic rows).",
                    evasion_change * 100)
    else:
        log.info("Evasion rate is essentially UNCHANGED (%.1fpp) — this augmentation did not "
                  "measurably move the needle against this specific attacker.", evasion_change * 100)

    if auc_pr_change < -0.005 or f1_change < -0.01:
        log.warning("STANDARD PERFORMANCE TRADEOFF: AUC-PR change=%+.4f, F1 change=%+.4f — "
                    "hardening came at a measurable cost to standard-test performance. "
                    "Reporting this plainly rather than only the adversarial win/loss.",
                    auc_pr_change, f1_change)
    else:
        log.info("Standard performance held (AUC-PR change=%+.4f, F1 change=%+.4f) — hardening "
                  "did not come at a meaningful cost on ordinary traffic.", auc_pr_change, f1_change)

    log.info("HONEST SCOPE NOTE: this evaluates whether the SAME attacker policy from step 20 "
              "still works — a transfer-robustness check. It does not prove the hardened model "
              "resists a freshly-adapted attacker; that would require retraining a new PPO "
              "policy against model_hardened.json, a natural next iteration this step does not "
              "attempt.")

    total_runtime = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("Total runtime: %.1fs (%.1f min)", total_runtime, total_runtime / 60)

    output = {
        "description": "Adversarial retraining: real data + CTGAN synthetic fraud + harvested "
                        "PPO evasions, evaluated against the original enriched model on both "
                        "standard and adversarial test sets.",
        "honest_context": {
            "ctgan_feature_mismatch": (
                "synthetic_fraud.parquet supplies 33 of 243 features (21 raw window-base columns "
                "-> the window block's t9 slot; 12 enriched columns -> their exact slots). The "
                "remaining 210 (189 window timesteps t0-t8, plus 21 enriched features CTGAN "
                "doesn't produce: same_month_last_year_mean, amount_vs_last_year_ratio, all 9 "
                "merch_*/pop_* features, all 10 flags) are filled with the MEAN of that feature "
                "among the 9,220 real fraud TRAINING windows — chosen over zero-padding (which "
                "would manufacture a fake all-zero history pattern) and over excluding CTGAN rows "
                "entirely (the task calls for using them as training augmentation)."
            ),
            "evader_labeling": "Harvested evasions are real fraud transactions that were "
                                "successfully disguised by the step-20 attacker — labeled fraud "
                                "(1), and already complete 243-dim vectors, no reconstruction needed.",
            "experimental_control": "Hyperparameters (scale_pos_weight=794.0, max_depth=6, "
                                     "learning_rate=0.1, n_estimators=300) are UNCHANGED from step "
                                     "8, not recomputed for the new class balance — isolates the "
                                     "training-data change as the only variable.",
            "adversarial_eval_scope": "Reuses step 20's ALREADY-TRAINED attacker policy (not "
                                       "retrained) against each model. This is a transfer-"
                                       "robustness check ('does the specific exploit still work'), "
                                       "not proof against a freshly adapted attacker.",
        },
        "training_data": {
            "n_real_rows": int(len(y_train_real)),
            "n_real_fraud": n_fraud_before,
            "n_synthetic_rows": int(len(synthetic_df)),
            "n_evader_rows": int(len(pd.read_parquet(EVADERS_PATH))),
            "n_total_rows": int(len(y_train_aug)),
            "n_total_fraud": n_fraud_after,
            "xgb_params": xgb_params,
            "train_time_seconds": round(train_time, 2),
        },
        "fixed_threshold": threshold,
        "standard_evaluation": {
            "original": original_standard,
            "hardened": hardened_standard,
            "auc_pr_change": round(auc_pr_change, 6),
            "f1_change": round(f1_change, 6),
        },
        "adversarial_evaluation": {
            "original": original_adversarial,
            "hardened": hardened_adversarial,
            "evasion_rate_harvest_change": round(evasion_change, 6),
            "evasion_rate_deterministic_change": round(
                hardened_adversarial["evasion_rate_deterministic"] - original_adversarial["evasion_rate_deterministic"], 6,
            ),
        },
        "outputs": {"model_hardened": str(MODEL_HARDENED_PATH)},
        "runtime_seconds": round(total_runtime, 2),
    }
    with open(METRICS_ADVERSARIAL_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", METRICS_ADVERSARIAL_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
