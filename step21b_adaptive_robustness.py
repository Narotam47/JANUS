"""Step 21b — adaptive robustness check: a FRESH PPO attacker vs. the
hardened model.

Step 21 showed the ORIGINAL attacker's evasion rate against
`model_hardened.json` collapsed from 66.57% to 1.82%. That's necessary but
not sufficient evidence of real hardening — it only proves the SPECIFIC
trajectories the original policy learned no longer work. A hardened model
could still be trivially vulnerable to a freshly-adapted attacker that
searches for a NEW exploit, if the augmentation only patched over the
particular timing-feature manipulation step 20 found rather than closing
the underlying sensitivity.

This script trains a brand-new PPO policy from scratch — same environment,
same action space, same L2 budget, same reward shaping, same episode
length as step 20 — directly against `model_hardened.json`, without ever
loading `ppo_policy.zip`. Whatever evasion rate this fresh attacker
achieves is the honest answer to "is the hardening structural, or did it
just overfit to the original attacker's trajectories?"
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
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from step20_ppo_attacker import (  # noqa: E402
    BUDGET_STD_UNITS,
    HARVEST_ATTEMPTS_PER_TRANSACTION,
    MAX_STEPS_PER_EPISODE,
    PERTURBABLE_BINARY,
    PERTURBABLE_CONTINUOUS,
    RANDOM_SEED,
    TOTAL_TIMESTEPS,
    FraudEvasionEnv,
    load_fraud_test_vectors,
    rollout_evasion_rate,
)

# --- Config -------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
MODEL_ENRICHED: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
MODEL_HARDENED: Final[Path] = ARTIFACTS_DIR / "model_hardened.json"
METRICS_ENRICHED: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"
METRICS_ADVERSARIAL: Final[Path] = ARTIFACTS_DIR / "metrics_adversarial.json"
PPO_ATTACK_REPORT: Final[Path] = ARTIFACTS_DIR / "ppo_attack_report.json"

REPORT_PATH: Final[Path] = ARTIFACTS_DIR / "adaptive_evasion_report.json"
POLICY_OUTPUT_PATH: Final[Path] = ARTIFACTS_DIR / "ppo_policy_adaptive.zip"

# A different seed than step 20's attacker (still the same hyperparameter
# recipe) — this run must not be a coincidental replay of the same policy.
ADAPTIVE_SEED: Final[int] = 4242

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step21b")


def train_fresh_attacker(model: XGBClassifier, threshold: float) -> tuple[PPO, FraudEvasionEnv, np.ndarray, int, dict]:
    """Builds the SAME environment design as step 20, targeting `model`,
    and trains a brand-new PPO policy from scratch (no warm-start from
    ppo_policy.zip). Returns (policy, env, fraud_positions_catchable, n_catchable, train_stats)."""
    X_fraud_all, fraud_positions_all, feature_names = load_fraud_test_vectors()
    prob_orig = model.predict_proba(X_fraud_all)[:, 1]
    caught_mask = prob_orig >= threshold
    X_catchable = X_fraud_all[caught_mask]
    n_catchable = len(X_catchable)
    log.info("Catchable pool against this model: %d/%d fraud transactions score >= %.4f.",
              n_catchable, len(X_fraud_all), threshold)

    continuous_idx = np.array([feature_names.index(n) for n in PERTURBABLE_CONTINUOUS])
    binary_idx = np.array([feature_names.index(n) for n in PERTURBABLE_BINARY])
    continuous_scale = np.array([max(X_catchable[:, i].std(), 1e-6) for i in continuous_idx])

    env = FraudEvasionEnv(
        model, X_catchable, continuous_idx, continuous_scale, binary_idx, threshold,
        max_steps=MAX_STEPS_PER_EPISODE, budget=BUDGET_STD_UNITS,
    )
    monitored_env = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored_env])

    log.info("Initializing a BRAND-NEW PPO policy (seed=%d, NOT loading ppo_policy.zip) ...", ADAPTIVE_SEED)
    policy = PPO(
        "MlpPolicy", vec_env, verbose=0, seed=ADAPTIVE_SEED,
        n_steps=256, batch_size=64, learning_rate=3e-4, gamma=0.95, device="cpu",
    )

    log.info("BEFORE TRAINING — evasion rate with this fresh, untrained (random-init) policy")
    before_rate, _ = rollout_evasion_rate(policy, env, n_catchable, deterministic=False, seed_offset=0)
    log.info("  Evasion rate (untrained): %d/%d = %.2f%%", int(round(before_rate * n_catchable)), n_catchable, before_rate * 100)

    log.info("TRAINING fresh PPO for %s timesteps against the HARDENED model ...", f"{TOTAL_TIMESTEPS:,}")
    t0 = time.perf_counter()
    policy.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    train_time = time.perf_counter() - t0
    log.info("Training complete in %.1fs (%.1f min).", train_time, train_time / 60)

    episode_rewards = monitored_env.get_episode_rewards()
    train_stats = {"train_time_seconds": round(train_time, 2), "n_episodes": len(episode_rewards),
                    "before_training_evasion_rate": before_rate}
    if episode_rewards:
        n = len(episode_rewards)
        q1 = float(np.mean(episode_rewards[: n // 4])) if n >= 4 else float(np.mean(episode_rewards))
        q4 = float(np.mean(episode_rewards[-(n // 4):])) if n >= 4 else float(np.mean(episode_rewards))
        log.info("  Episode reward: first quartile mean=%.4f -> last quartile mean=%.4f (change=%+.4f)", q1, q4, q4 - q1)
        train_stats["reward_first_quartile_mean"] = q1
        train_stats["reward_last_quartile_mean"] = q4

    return policy, env, fraud_positions_all[caught_mask], n_catchable, train_stats


def evaluate_and_analyze(
    policy: PPO, env: FraudEvasionEnv, n_catchable: int,
) -> tuple[float, float, dict, list]:
    log.info("AFTER TRAINING — evasion rate with the trained fresh policy (deterministic)")
    det_rate, _ = rollout_evasion_rate(policy, env, n_catchable, deterministic=True, seed_offset=50_000)
    log.info("  Evasion rate (trained, deterministic): %d/%d = %.2f%%",
              int(round(det_rate * n_catchable)), n_catchable, det_rate * 100)

    log.info("HARVESTING — %d attempt(s) per transaction, stochastic policy", HARVEST_ATTEMPTS_PER_TRANSACTION)
    all_successes: dict[int, dict] = {}
    for attempt in range(HARVEST_ATTEMPTS_PER_TRANSACTION):
        _, successes = rollout_evasion_rate(
            policy, env, n_catchable, deterministic=False, seed_offset=60_000 + attempt * n_catchable,
        )
        for s in successes:
            all_successes.setdefault(s["row_idx"], s)
    harvest_rate = len(all_successes) / n_catchable if n_catchable else 0.0
    log.info("Harvest evasion rate (any of %d attempts): %d/%d = %.2f%%",
              HARVEST_ATTEMPTS_PER_TRANSACTION, len(all_successes), n_catchable, harvest_rate * 100)

    perturb_analysis: dict = {}
    ranked: list = []
    if all_successes:
        prob_drops = [s["prob_before"] - s["prob_after"] for s in all_successes.values()]
        perturb_magnitudes = np.array([
            np.abs(s["perturbed_vector"][env.continuous_idx] - s["orig_vector"][env.continuous_idx]) / env.continuous_scale
            for s in all_successes.values()
        ])
        mean_abs_perturb = perturb_magnitudes.mean(axis=0)
        ranked = sorted(zip(PERTURBABLE_CONTINUOUS, mean_abs_perturb), key=lambda kv: -kv[1])
        log.info("  Mean |perturbation| per continuous feature (std-units), across %d evaders:", len(all_successes))
        for name, v in ranked:
            log.info("    %-28s %.4f", name, v)
        n_flipped_total = sum(s.get("n_binary_flipped", 0) for s in all_successes.values())
        perturb_analysis = {
            "n_evaders": len(all_successes),
            "mean_prob_drop": float(np.mean(prob_drops)),
            "mean_abs_perturbation_per_feature_std_units": {name: float(v) for name, v in ranked},
            "n_binary_flips_total": int(n_flipped_total),
        }
    else:
        log.info("  No evaders harvested — perturbation-targeting analysis not applicable.")

    return det_rate, harvest_rate, perturb_analysis, ranked


def main() -> None:
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 21b — ADAPTIVE ROBUSTNESS CHECK (fresh attacker vs. hardened model)")
    log.info("=" * 70)

    for p in (MODEL_HARDENED, METRICS_ENRICHED, METRICS_ADVERSARIAL, PPO_ATTACK_REPORT):
        if not p.exists():
            log.error("Missing required artifact: %s — run steps 8, 20, 21 first.", p)
            return

    with open(METRICS_ENRICHED) as f:
        threshold = json.load(f)["best_f1_threshold"]["best_threshold"]
    log.info("Routing threshold (unchanged): %.4f", threshold)

    hardened_model = XGBClassifier()
    hardened_model.load_model(str(MODEL_HARDENED))

    log.info("-" * 70)
    policy, env, _, n_catchable, train_stats = train_fresh_attacker(hardened_model, threshold)

    log.info("-" * 70)
    policy.save(str(POLICY_OUTPUT_PATH))
    log.info("Saved fresh attacker policy: %s (not requested, saved for consistency with prior "
              "steps' convention of persisting trained policies).", POLICY_OUTPUT_PATH)

    log.info("-" * 70)
    det_rate, harvest_rate, perturb_analysis, ranked = evaluate_and_analyze(policy, env, n_catchable)

    # --- Pull the reference numbers from steps 20/21 for the comparison table ---
    with open(METRICS_ADVERSARIAL) as f:
        step21_report = json.load(f)
    with open(PPO_ATTACK_REPORT) as f:
        step20_report = json.load(f)

    orig_vs_orig = step21_report["adversarial_evaluation"]["original"]
    orig_vs_hardened = step21_report["adversarial_evaluation"]["hardened"]
    step20_feature_ranking = step20_report.get("evader_analysis", {}).get(
        "mean_abs_perturbation_per_feature_std_units", {},
    )
    step20_top3 = sorted(step20_feature_ranking.items(), key=lambda kv: -kv[1])[:3]
    fresh_top3 = ranked[:3]

    # === Comparison table ======================================================
    log.info("=" * 70)
    log.info("COMPARISON TABLE — original attacker vs. fresh attacker, across both models")
    log.info("%-28s %-18s %10s %14s %14s %-s", "Attacker", "Target model", "Catchable", "Evasion(det)", "Evasion(harv)", "Top perturbed feature")
    log.info("%-28s %-18s %10d %13.2f%% %13.2f%% %-s",
              "Original (ppo_policy.zip)", "original", orig_vs_orig["n_catchable"],
              orig_vs_orig["evasion_rate_deterministic"] * 100, orig_vs_orig["evasion_rate_harvest"] * 100,
              step20_top3[0][0] if step20_top3 else "n/a")
    log.info("%-28s %-18s %10d %13.2f%% %13.2f%% %-s",
              "Original (ppo_policy.zip)", "hardened", orig_vs_hardened["n_catchable"],
              orig_vs_hardened["evasion_rate_deterministic"] * 100, orig_vs_hardened["evasion_rate_harvest"] * 100,
              "n/a (evasion too rare to profile)")
    log.info("%-28s %-18s %10d %13.2f%% %13.2f%% %-s",
              "FRESH (this step)", "hardened", n_catchable, det_rate * 100, harvest_rate * 100,
              fresh_top3[0][0] if fresh_top3 else "n/a (no evaders)")

    log.info("-" * 70)
    log.info("FEATURE-TARGETING COMPARISON: original attacker (vs. original model) vs. fresh "
              "attacker (vs. hardened model)")
    log.info("  %-28s %12s %12s", "Feature", "Original", "Fresh (new)")
    all_feature_names = sorted(set(step20_feature_ranking) | {n for n, _ in ranked})
    fresh_dict = dict(ranked)
    for name in sorted(all_feature_names, key=lambda n: -step20_feature_ranking.get(n, 0)):
        log.info("  %-28s %12.4f %12.4f", name, step20_feature_ranking.get(name, 0.0), fresh_dict.get(name, 0.0))

    # === Verdict ================================================================
    log.info("=" * 70)
    if harvest_rate < 0.10:
        verdict = "structural"
        log.info("VERDICT: STRUCTURAL HARDENING. Fresh-attacker harvest evasion rate is %.2f%% "
                  "(< 10%%) — a brand-new attacker, searching from scratch with no knowledge of "
                  "the original exploit, still can't reliably beat the hardened model. The "
                  "adversarial augmentation closed the underlying sensitivity, not just the "
                  "specific trajectories step 20 found.", harvest_rate * 100)
    elif harvest_rate > 0.50:
        verdict = "overfit"
        log.warning("VERDICT: HARDENING DID NOT STRUCTURALLY WORK. Fresh-attacker harvest evasion "
                    "rate is %.2f%%, comparable to step 20's original 66.57%% against the "
                    "un-hardened model. A new attacker, unaware of the original exploit, finds a "
                    "comparably effective one anyway — the augmentation appears to have overfit "
                    "to the SPECIFIC evasion trajectories in evaders.parquet rather than reducing "
                    "the model's underlying sensitivity to this perturbation space.", harvest_rate * 100)
    else:
        verdict = "partial"
        log.info("VERDICT: PARTIAL HARDENING. Fresh-attacker harvest evasion rate is %.2f%% — "
                  "meaningfully below the original 66.57%%, but above the 10%% bar for calling "
                  "this fully structural. The hardening reduced but did not eliminate the "
                  "underlying vulnerability; a persistent adaptive adversary still finds real, "
                  "if less frequent, evasions.", harvest_rate * 100)

    if fresh_top3 and step20_top3:
        fresh_top_name = fresh_top3[0][0]
        orig_top_name = step20_top3[0][0]
        if fresh_top_name == orig_top_name:
            log.info("The fresh attacker's #1 targeted feature ('%s') is THE SAME as the "
                      "original attacker's — the model may still be more sensitive there than "
                      "elsewhere, even if the specific old trajectory no longer works.", fresh_top_name)
        else:
            log.info("The fresh attacker's #1 targeted feature ('%s') DIFFERS from the original "
                      "('%s') — consistent with the model having closed (or reduced) its "
                      "sensitivity to the original timing-feature attack surface, pushing the "
                      "attacker toward a different exploit.", fresh_top_name, orig_top_name)

    total_runtime = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("Total runtime: %.1fs (%.1f min)", total_runtime, total_runtime / 60)

    output = {
        "description": "Adaptive robustness check: a fresh PPO attacker, trained from scratch "
                        "(no warm-start from ppo_policy.zip), against model_hardened.json.",
        "honest_context": (
            "Step 21's transfer test (reusing the original policy) only shows the SPECIFIC "
            "trajectories the original attacker learned no longer work. This script answers the "
            "stronger question: can a brand-new attacker, searching from scratch with the SAME "
            "environment design, find a DIFFERENT way to evade the hardened model?"
        ),
        "environment": {
            "same_as_step20": True,
            "perturbable_continuous": list(PERTURBABLE_CONTINUOUS),
            "perturbable_binary": list(PERTURBABLE_BINARY),
            "budget_std_units": BUDGET_STD_UNITS,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": ADAPTIVE_SEED,
            "loaded_original_policy": False,
        },
        "training": train_stats,
        "comparison_table": {
            "original_attacker_vs_original_model": orig_vs_orig,
            "original_attacker_vs_hardened_model": orig_vs_hardened,
            "fresh_attacker_vs_hardened_model": {
                "n_catchable": n_catchable,
                "evasion_rate_deterministic": det_rate,
                "evasion_rate_harvest": harvest_rate,
            },
        },
        "feature_targeting_comparison": {
            "original_attacker_ranking": step20_feature_ranking,
            "fresh_attacker_ranking": dict(ranked),
        },
        "fresh_attacker_evader_analysis": perturb_analysis if perturb_analysis else None,
        "verdict": verdict,
        "outputs": {"policy": str(POLICY_OUTPUT_PATH)},
        "runtime_seconds": round(total_runtime, 2),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", REPORT_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
