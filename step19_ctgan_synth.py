"""Step 19 — CTGAN synthetic fraud generation.

Purpose: generate synthetic fraud transactions that respect the JOINT
distribution of real fraud, to augment the adversarial training loop in
steps 20-21. CTGAN is used instead of SMOTE specifically because SMOTE
interpolates each feature independently between nearest neighbors, which
can break cross-feature relationships (e.g. amount vs. amount z-score, or
merchant familiarity vs. category familiarity) and produce transactions
that are jointly implausible even though each individual feature value
looks reasonable in isolation. CTGAN models the joint distribution
directly via a conditional GAN over mixed continuous/discrete columns.

Feature scope: `clean_train.parquet` alone only carries the 21 raw
per-transaction columns (before windowing); the task's named example
features (`user_amount_zscore`, `user_hist_mean`, `merchant_user_count`,
`mcc_user_count`) live in step 5's `user_features_train.parquet`. Both
files are already row-aligned 1:1 (step 5's own alignment check verifies
this — no windowing/`compute_window_feature_indices` needed here, since
this step works at the per-transaction level, not the sequence-window
level steps 8+ use). They are joined column-wise and filtered to the
9,220 fraud rows.

Two enriched columns are excluded from the CTGAN feature set:
`same_month_last_year_mean` and `amount_vs_last_year_ratio` — these need a
full YEAR of prior history (not just "some" history) and are NaN for any
user without one; CTGAN's reference implementation does not accept NaN.
Every other enriched feature used here (hist_mean/std/max, zscore, ratio,
tx_count, merchant/mcc counts) is verified NaN-free among fraud rows below
— expected, since step 9 found zero fraud before transaction position 68,
so every fraud transaction already has substantial prior history.

HONEST CONTEXT: 9,220 training rows is small for a GAN (CTGAN's own
published benchmarks use tens of thousands to millions of rows). Training
may be noisy or fail to converge cleanly — the loss curve is reported
plainly below rather than asserted to look good. The synthetic data
augments an adversarial training loop; it doesn't need to be perfect, but
implausible extrapolations (synthetic values outside the real fraud
sample's observed range) are explicitly flagged, not hidden.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from ctgan import CTGAN
from scipy.stats import pearsonr

# --- Config -----------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
CLEAN_TRAIN: Final[Path] = ARTIFACTS_DIR / "clean_train.parquet"
USER_FEATURES_TRAIN: Final[Path] = ARTIFACTS_DIR / "user_features_train.parquet"

OUT_PARQUET: Final[Path] = ARTIFACTS_DIR / "synthetic_fraud.parquet"
OUT_REPORT: Final[Path] = ARTIFACTS_DIR / "ctgan_synth_report.json"

USER_COL: Final[str] = "User"
LABEL_COL: Final[str] = "label"

EXCLUDE_COLS: Final[tuple[str, ...]] = (
    USER_COL, LABEL_COL, "same_month_last_year_mean", "amount_vs_last_year_ratio",
)
DISCRETE_COLS: Final[tuple[str, ...]] = (
    "Merchant State", "MCC",
    "err_bad_cvv", "err_bad_card_number", "err_bad_expiration", "err_bad_pin",
    "err_bad_zipcode", "err_insufficient_balance", "err_technical_glitch",
    "chip_Chip Transaction", "chip_Online Transaction", "chip_Swipe Transaction",
    "is_new_merchant", "is_weekend", "is_holiday_season",
)

N_SYNTHETIC: Final[int] = 5000
CTGAN_EPOCHS: Final[int] = 1000
CTGAN_BATCH_SIZE: Final[int] = 500
SEED: Final[int] = 42

KEY_CONTINUOUS_FEATURES: Final[tuple[str, ...]] = ("amount_raw", "user_amount_zscore", "user_hist_mean")
CORRELATION_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("amount_raw", "user_amount_zscore"),
    ("merchant_user_count", "mcc_user_count"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step19")


# === Data loading =============================================================

def load_real_fraud() -> pd.DataFrame:
    log.info("Loading %s and %s ...", CLEAN_TRAIN, USER_FEATURES_TRAIN)
    clean = pd.read_parquet(CLEAN_TRAIN)
    user_feats = pd.read_parquet(USER_FEATURES_TRAIN)
    assert len(clean) == len(user_feats), (
        f"Row count mismatch: clean_train={len(clean)} vs user_features_train={len(user_feats)} "
        "— these are supposed to be 1:1 row-aligned per step 5."
    )
    combined = pd.concat([clean.reset_index(drop=True), user_feats.reset_index(drop=True)], axis=1)
    fraud = combined[combined[LABEL_COL] == 1].reset_index(drop=True)
    log.info("Combined per-transaction table: %s rows, %d columns. Fraud rows: %s (%.4f%%).",
              f"{len(combined):,}", combined.shape[1], f"{len(fraud):,}", len(fraud) / len(combined) * 100)
    return fraud


def report_pre_training_stats(fraud: pd.DataFrame, feature_cols: list[str]) -> dict:
    log.info("=" * 70)
    log.info("REAL FRAUD SAMPLE — feature statistics BEFORE training")
    log.info("Shape: %s rows x %d feature columns", f"{len(fraud):,}", len(feature_cols))

    nan_counts = fraud[feature_cols].isna().sum()
    any_nan = nan_counts[nan_counts > 0]
    if len(any_nan):
        log.warning("NaN found in feature columns CTGAN cannot accept: %s", any_nan.to_dict())
    else:
        log.info("No NaN in any selected feature column (expected — step 9 found zero fraud "
                  "before transaction position 68, so every fraud row has full history).")

    stats = {}
    for col in KEY_CONTINUOUS_FEATURES:
        s = fraud[col]
        log.info("  %-22s mean=%10.4f  std=%10.4f  min=%10.4f  max=%10.4f",
                  col, s.mean(), s.std(), s.min(), s.max())
        stats[col] = {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}

    for c1, c2 in CORRELATION_PAIRS:
        r, p = pearsonr(fraud[c1], fraud[c2])
        log.info("  Pearson r(%s, %s) = %.4f (p=%.2e)", c1, c2, r, p)
        stats.setdefault("real_correlations", {})[f"{c1}|{c2}"] = {"r": float(r), "p": float(p)}

    return stats


# === CTGAN training ============================================================

def train_ctgan(fraud: pd.DataFrame, feature_cols: list[str]) -> tuple[CTGAN, float]:
    log.info("=" * 70)
    log.info("TRAINING CTGAN on %s fraud rows, %d features (%d discrete, %d continuous)",
              f"{len(fraud):,}", len(feature_cols), len(DISCRETE_COLS),
              len(feature_cols) - len(DISCRETE_COLS))
    log.info("HONEST CONTEXT: 9,220 rows is small for a GAN. Convergence quality is reported "
              "below, not assumed.")

    model = CTGAN(
        epochs=CTGAN_EPOCHS, batch_size=CTGAN_BATCH_SIZE, verbose=True, enable_gpu=False,
    )
    model.set_random_state(SEED)

    t0 = time.perf_counter()
    model.fit(fraud[feature_cols], discrete_columns=list(DISCRETE_COLS))
    train_time = time.perf_counter() - t0
    log.info("CTGAN training complete in %.1fs (%.1f min).", train_time, train_time / 60)
    return model, train_time


def report_loss_curve(model: CTGAN) -> dict:
    log.info("-" * 70)
    log.info("TRAINING LOSS CURVE — reported honestly, not asserted to look good")
    loss_df = getattr(model, "loss_values", None)
    if loss_df is None or len(loss_df) == 0:
        log.warning("No loss_values recorded by this ctgan version — cannot report a curve.")
        return {"available": False}

    cols = list(loss_df.columns)
    gen_col = next((c for c in cols if "Generator" in c), None)
    disc_col = next((c for c in cols if "iscriminator" in c), None)  # tolerate the lib's own typo variants

    n = len(loss_df)
    checkpoints = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    log.info("  %d epochs logged. Sampled checkpoints:", n)
    for i in checkpoints:
        row = loss_df.iloc[i]
        log.info("    epoch=%4s  generator_loss=%9.4f  discriminator_loss=%9.4f",
                  int(row.get("Epoch", i)), row.get(gen_col, float("nan")), row.get(disc_col, float("nan")))

    gen_last10 = loss_df[gen_col].tail(10)
    disc_last10 = loss_df[disc_col].tail(10)
    gen_std_last10 = float(gen_last10.std())
    disc_std_last10 = float(disc_last10.std())
    # A crude, honest convergence heuristic: is the back end of training still
    # swinging as much as the whole run, or has it settled down?
    gen_std_full = float(loss_df[gen_col].std())
    disc_std_full = float(loss_df[disc_col].std())
    settled = (gen_std_last10 < 0.5 * gen_std_full) and (disc_std_last10 < 0.5 * disc_std_full)

    log.info("  Generator loss:     full-run std=%.4f, last-10-epoch std=%.4f", gen_std_full, gen_std_last10)
    log.info("  Discriminator loss: full-run std=%.4f, last-10-epoch std=%.4f", disc_std_full, disc_std_last10)
    if settled:
        log.info("  -> Loss variance in the last 10 epochs is well below the full-run variance: "
                  "training appears to have SETTLED, not necessarily converged to a Nash "
                  "equilibrium in the formal GAN sense, but stabilized.")
    else:
        log.warning("  -> Loss is STILL OSCILLATING at a magnitude comparable to the full run — "
                    "this does NOT look like clean convergence. Consistent with the honest-"
                    "context note above: 9,220 rows is a small, noisy training set for a GAN. "
                    "The generated data should still be checked below, not assumed good.")

    return {
        "available": True,
        "n_epochs": int(n),
        "generator_loss_final": float(loss_df[gen_col].iloc[-1]),
        "discriminator_loss_final": float(loss_df[disc_col].iloc[-1]),
        "generator_loss_std_full_run": gen_std_full,
        "generator_loss_std_last_10": gen_std_last10,
        "discriminator_loss_std_full_run": disc_std_full,
        "discriminator_loss_std_last_10": disc_std_last10,
        "appears_settled": bool(settled),
    }


# === Generation and evaluation =================================================

def generate_and_compare(model: CTGAN, real_fraud: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    log.info("=" * 70)
    log.info("GENERATING %s synthetic fraud rows ...", f"{N_SYNTHETIC:,}")
    synth = model.sample(N_SYNTHETIC)
    log.info("Generated: %s rows x %d columns.", f"{len(synth):,}", synth.shape[1])

    report: dict = {"key_feature_comparison": {}, "extrapolation_flags": {}, "correlation_comparison": {}}

    log.info("-" * 70)
    log.info("REAL vs SYNTHETIC — key continuous features")
    log.info("  %-22s %8s %10s %10s %10s %10s", "feature", "source", "mean", "std", "min", "max")
    for col in KEY_CONTINUOUS_FEATURES:
        r = real_fraud[col]
        s = synth[col]
        log.info("  %-22s %8s %10.4f %10.4f %10.4f %10.4f", col, "real", r.mean(), r.std(), r.min(), r.max())
        log.info("  %-22s %8s %10.4f %10.4f %10.4f %10.4f", col, "synthetic", s.mean(), s.std(), s.min(), s.max())

        below = int((s < r.min()).sum())
        above = int((s > r.max()).sum())
        n_out = below + above
        pct_out = n_out / len(s) * 100
        if n_out > 0:
            log.warning("    -> %d/%d synthetic rows (%.2f%%) fall OUTSIDE real fraud's observed "
                        "[%.4f, %.4f] range for '%s' — these are EXTRAPOLATIONS, not interpolations "
                        "(%d below min, %d above max).", n_out, len(s), pct_out, r.min(), r.max(), col, below, above)
        else:
            log.info("    -> all synthetic values fall within real fraud's observed range.")

        report["key_feature_comparison"][col] = {
            "real": {"mean": float(r.mean()), "std": float(r.std()), "min": float(r.min()), "max": float(r.max())},
            "synthetic": {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())},
        }
        report["extrapolation_flags"][col] = {
            "n_below_real_min": below, "n_above_real_max": above,
            "pct_outside_real_range": round(pct_out, 3),
        }

    log.info("-" * 70)
    log.info("JOINT PLAUSIBILITY CHECK — correlation preserved? (SMOTE would break this)")
    for c1, c2 in CORRELATION_PAIRS:
        r_real, p_real = pearsonr(real_fraud[c1], real_fraud[c2])
        r_synth, p_synth = pearsonr(synth[c1], synth[c2])
        delta = abs(r_real - r_synth)
        log.info("  (%s, %s): real r=%.4f  |  synthetic r=%.4f  |  |delta|=%.4f",
                  c1, c2, r_real, r_synth, delta)
        report["correlation_comparison"][f"{c1}|{c2}"] = {
            "real_r": float(r_real), "synthetic_r": float(r_synth), "abs_delta": float(delta),
        }

    return synth, report


# === Main =====================================================================

def main() -> None:
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 19 — CTGAN SYNTHETIC FRAUD GENERATION")
    log.info("=" * 70)

    fraud = load_real_fraud()
    feature_cols = [c for c in fraud.columns if c not in EXCLUDE_COLS]

    pre_stats = report_pre_training_stats(fraud, feature_cols)

    model, train_time = train_ctgan(fraud, feature_cols)
    loss_report = report_loss_curve(model)

    synth, gen_report = generate_and_compare(model, fraud, feature_cols)

    synth.to_parquet(OUT_PARQUET, index=False)
    file_size_mb = OUT_PARQUET.stat().st_size / 1e6
    total_runtime = time.perf_counter() - t_start

    log.info("=" * 70)
    log.info("OUTPUT")
    log.info("  %s: %s rows, %d columns, %.2f MB", OUT_PARQUET, f"{len(synth):,}", synth.shape[1], file_size_mb)
    log.info("  Total runtime: %.1fs (%.1f min) — CTGAN training was %.1f%% of that.",
              total_runtime, total_runtime / 60, train_time / total_runtime * 100)

    output = {
        "description": "CTGAN-generated synthetic fraud transactions for steps 20-21's "
                        "adversarial training loop.",
        "honest_context": (
            "Trained on only 9,220 real fraud rows — small for a GAN. See 'loss_curve' for "
            "whether training converged cleanly. 'extrapolation_flags' in "
            "'generation_report.key_feature_comparison' show where synthetic values fall "
            "outside the real fraud sample's observed range."
        ),
        "config": {
            "n_real_fraud_rows": int(len(fraud)),
            "n_feature_cols": len(feature_cols),
            "n_discrete_cols": len(DISCRETE_COLS),
            "excluded_cols": list(EXCLUDE_COLS),
            "ctgan_epochs": CTGAN_EPOCHS,
            "ctgan_batch_size": CTGAN_BATCH_SIZE,
            "n_synthetic_generated": N_SYNTHETIC,
            "seed": SEED,
        },
        "real_fraud_pre_training_stats": pre_stats,
        "loss_curve": loss_report,
        "generation_report": gen_report,
        "output": {
            "path": str(OUT_PARQUET), "n_rows": int(len(synth)), "n_cols": int(synth.shape[1]),
            "file_size_mb": round(file_size_mb, 3),
        },
        "runtime_seconds": {
            "ctgan_training": round(train_time, 2),
            "total": round(total_runtime, 2),
        },
    }
    with open(OUT_REPORT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_REPORT)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
