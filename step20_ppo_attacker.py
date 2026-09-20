"""Step 20 — single PPO attacker in a Gymnasium environment.

Adaptation of Aegis-SDP's `step5_rl_attacker.py` to JANUS's enriched model
and 243-feature space. Same overall shape (Gymnasium env + SB3 PPO +
harvest loop), reworked for: the real frozen `model_enriched.json` as a
fixed target (not retrained here), the full 243-dim window+enriched
feature vector as the observation, a feature-space attack restricted to a
plausible attacker-controlled subset, and a hard perturbation-magnitude
budget so evasion can't degenerate into "make fraud look identical to a
random legitimate transaction."

THREAT MODEL — what the attacker can and cannot touch
-------------------------------------------------------
The attacker crafts ONE fraudulent transaction. It can only control that
transaction's own properties, not the user's or merchant's history (those
already happened). Perturbable features, split by how they're perturbed:

  CONTINUOUS (soft delta, budget-constrained — 8 features):
    amount_raw_t9, amount_log_t9        the transaction's own amount
    hours_since_last_t9, hour_t9,
      day_of_week_t9                    when the transaction happens
    user_amount_zscore,
      user_amount_to_max_ratio          amount framed relative to history
                                         (a fraudster picking a smaller
                                         amount shifts these too)
    days_since_last_tx                  gap since the user's last activity

  BINARY (direct set, soft-penalized — 2 features):
    first_city_state                    pick a familiar-looking location
    first_channel                       pick a familiar-looking channel

  EXPLICITLY EXCLUDED (per the task): user_tx_count, merchant_user_count,
  mcc_user_count, user_hist_mean/std/max, all 9 merch_*/pop_* features,
  first_mcc, is_new_merchant — these are properties of the DATASET (the
  user's/merchant's actual history), not something a single fraudulent
  transaction can rewrite. is_largest_ever/is_2x_hist_max/is_5x_hist_max
  are also left untouched (a known, documented simplification of this
  feature-space attack: a fully consistent problem-space attack would
  recompute these from the perturbed amount, which this environment does
  not do).

This is a FEATURE-SPACE attack (perturbing what the model sees directly),
not a full problem-space attack (reconstructing a consistent raw
transaction end-to-end) — the same simplification Aegis-SDP's reference
script made (it also perturbed amount/hours directly without recomputing
every derived field). Evasion rates here characterize the model's
sensitivity to this specific, bounded perturbation space, not "can an
attacker with unlimited raw-transaction control evade the model."

MAGNITUDE CONSTRAINT: continuous perturbations are tracked as a single
vector normalized by each feature's own std among real fraud test rows,
and hard-capped at a total L2 norm (BUDGET_STD_UNITS) via projection —
exceeding the budget scales the whole cumulative perturbation back onto
the budget's boundary, not just the newest step. Binary flags aren't part
of this L2 budget (flipping a boolean has no meaningful "partial"
projection); instead each flipped flag costs a per-step reward penalty.
"""

from __future__ import annotations

# CRITICAL: must precede any import that touches OpenMP (xgboost, torch, sklearn) —
# without this, PyTorch (via stable_baselines3) and XGBoost can double-load libgomp
# and segfault on macOS/Linux. Verified necessary in Aegis-SDP's step5_rl_attacker.py.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# XGBoost must be imported before torch/stable_baselines3 to win the libgomp race.
from xgboost import XGBClassifier  # noqa: E402

import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Final, Optional  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from step13_shap_explain import build_feature_names, compute_window_feature_indices

# --- Config -------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"
WINDOWS_TEST: Final[Path] = ARTIFACTS_DIR / "windows_test.npy"
LABELS_TEST: Final[Path] = ARTIFACTS_DIR / "labels_test.npy"
USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"
MODEL_ENRICHED: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
METRICS_ENRICHED: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"

POLICY_OUTPUT_PATH: Final[Path] = ARTIFACTS_DIR / "ppo_policy.zip"
EVADERS_OUTPUT_PATH: Final[Path] = ARTIFACTS_DIR / "evaders.parquet"
REPORT_PATH: Final[Path] = ARTIFACTS_DIR / "ppo_attack_report.json"
CURVE_PLOT_PATH: Final[Path] = ARTIFACTS_DIR / "ppo_training_curve.png"

WINDOW_SIZE: Final[int] = 10
LAST_STEP: Final[int] = WINDOW_SIZE - 1  # t9 = the current/most-recent transaction

PERTURBABLE_CONTINUOUS: Final[tuple[str, ...]] = (
    f"amount_raw_t{LAST_STEP}", f"amount_log_t{LAST_STEP}",
    f"hours_since_last_t{LAST_STEP}", f"hour_t{LAST_STEP}", f"day_of_week_t{LAST_STEP}",
    "user_amount_zscore", "user_amount_to_max_ratio", "days_since_last_tx",
)
PERTURBABLE_BINARY: Final[tuple[str, ...]] = ("first_city_state", "first_channel")

BUDGET_STD_UNITS: Final[float] = 6.0        # hard cap: total L2 norm of continuous perturbation, in std-units
MAGNITUDE_PENALTY_COEF: Final[float] = 0.02  # soft per-step penalty on continuous perturbation norm
BINARY_FLIP_PENALTY: Final[float] = 0.10     # soft per-step penalty per binary flag currently flipped
TIME_PENALTY: Final[float] = 0.02
EVASION_BONUS: Final[float] = 5.0
MAX_STEPS_PER_EPISODE: Final[int] = 15
ACTION_STEP_FRACTION: Final[float] = 0.35    # per-step continuous delta, as a fraction of that feature's std

TOTAL_TIMESTEPS: Final[int] = 100_000
HARVEST_ATTEMPTS_PER_TRANSACTION: Final[int] = 3
RANDOM_SEED: Final[int] = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step20")


# === Data loading =============================================================

def load_fraud_test_vectors() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """All real fraud transactions in the held-out test set, as full
    243-dim (window+enriched) vectors — the attacker's starting points."""
    idx = compute_window_feature_indices(CLEAN_TEST)
    y_test = np.asarray(np.load(str(LABELS_TEST), mmap_mode="r"))

    user_feats = pd.read_parquet(USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(MERCH_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(FLAGS_TEST).iloc[idx].reset_index(drop=True)
    feature_names = build_feature_names(
        CLEAN_TEST, list(user_feats.columns), list(merch_feats.columns), list(flags.columns),
    )

    X_win = np.load(str(WINDOWS_TEST), mmap_mode="r")
    enriched = np.hstack([
        user_feats.values.astype(np.float32),
        merch_feats.values.astype(np.float32),
        flags.values.astype(np.float32),
    ])
    X = np.hstack([np.asarray(X_win, dtype=np.float32), enriched])

    fraud_mask = y_test == 1
    X_fraud = X[fraud_mask]
    fraud_positions = np.where(fraud_mask)[0]
    log.info("Loaded %s real fraud transactions from the test set (of %s total).",
              f"{len(X_fraud):,}", f"{len(y_test):,}")
    return X_fraud, fraud_positions, feature_names


# === Gymnasium environment =====================================================

class FraudEvasionEnv(gym.Env):
    """RL environment: perturb a real fraud transaction's feature vector to
    push the JANUS enriched model's fraud probability below the routing
    threshold, within a bounded, attacker-plausible action space."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: XGBClassifier,
        X_fraud: np.ndarray,
        continuous_idx: np.ndarray,
        continuous_scale: np.ndarray,
        binary_idx: np.ndarray,
        threshold: float,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        budget: float = BUDGET_STD_UNITS,
    ) -> None:
        super().__init__()
        self.model = model
        self.X_fraud = X_fraud
        self.continuous_idx = continuous_idx
        self.continuous_scale = continuous_scale  # per-dim std, for normalizing the budget
        self.binary_idx = binary_idx
        self.threshold = threshold
        self.max_steps = max_steps
        self.budget = budget

        n_actions = len(continuous_idx) + len(binary_idx)
        # NaN sentinel: a handful of enriched features (merchant history for a
        # merchant's first appearance, year-over-year comparisons) are
        # legitimately NaN even for fraud rows — XGBoost handles that
        # natively via missing-value routing, but a neural-network policy
        # cannot take NaN as input. The TRUE vector (with real NaN) is what
        # model.predict_proba() always sees, in _predict_prob(); only the
        # policy's observation gets NaN replaced by this out-of-range sentinel.
        self.nan_sentinel = -999.0
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(X_fraud.shape[1],), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._orig_vector: Optional[np.ndarray] = None
        self._current_vector: Optional[np.ndarray] = None
        self._cum_normalized: Optional[np.ndarray] = None  # cumulative continuous perturbation, in std-units
        self._steps = 0
        self._row_idx = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opts = options or {}
        row_idx = opts.get("row_idx")
        if row_idx is None:
            row_idx = int(self._rng.integers(0, len(self.X_fraud)))
        self._row_idx = int(row_idx)
        self._orig_vector = self.X_fraud[self._row_idx].copy()
        self._current_vector = self._orig_vector.copy()
        self._cum_normalized = np.zeros(len(self.continuous_idx), dtype=np.float64)
        self._steps = 0
        return self._sanitized_observation(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        n_cont = len(self.continuous_idx)
        cont_action = action[:n_cont]
        bin_action = action[n_cont:]

        prob_before = self._predict_prob()

        # --- Continuous dims: propose a delta, then project the WHOLE
        # cumulative perturbation back onto the budget ball if it would
        # exceed it (not just clip the newest increment).
        proposed_delta_normalized = cont_action * ACTION_STEP_FRACTION
        proposed_cum = self._cum_normalized + proposed_delta_normalized
        proposed_norm = np.linalg.norm(proposed_cum)
        if proposed_norm > self.budget and proposed_norm > 1e-8:
            proposed_cum = proposed_cum * (self.budget / proposed_norm)
        applied_delta_normalized = proposed_cum - self._cum_normalized
        self._cum_normalized = proposed_cum

        applied_delta_raw = applied_delta_normalized * self.continuous_scale
        self._current_vector[self.continuous_idx] += applied_delta_raw

        # --- Binary dims: direct set via sign(action).
        n_flipped = 0
        for local_i, feat_idx in enumerate(self.binary_idx):
            new_val = 1.0 if bin_action[local_i] > 0 else 0.0
            if new_val != self._orig_vector[feat_idx]:
                n_flipped += 1
            self._current_vector[feat_idx] = new_val

        prob_after = self._predict_prob()

        step_perturb_norm = float(np.linalg.norm(applied_delta_normalized))
        reward = (
            float(prob_before - prob_after)
            - MAGNITUDE_PENALTY_COEF * step_perturb_norm
            - BINARY_FLIP_PENALTY * n_flipped
            - TIME_PENALTY
        )

        self._steps += 1
        terminated = bool(prob_after < self.threshold)
        truncated = self._steps >= self.max_steps
        if terminated:
            reward += EVASION_BONUS

        info = {
            "prob_before": prob_before,
            "prob_after": prob_after,
            "row_idx": self._row_idx,
            "cum_perturbation_l2_std_units": float(np.linalg.norm(self._cum_normalized)),
            "n_binary_flipped": n_flipped,
        }
        return self._sanitized_observation(), float(reward), terminated, truncated, info

    def _sanitized_observation(self) -> np.ndarray:
        """What the POLICY sees: real NaN replaced by an out-of-range
        sentinel. `model.predict_proba()` never uses this — see `_predict_prob`."""
        obs = self._current_vector.astype(np.float32).copy()
        obs[np.isnan(obs)] = self.nan_sentinel
        return obs

    def _predict_prob(self) -> float:
        # Uses self._current_vector directly (with real NaN intact) — this
        # must match production behavior exactly, where XGBoost's native
        # missing-value routing sees genuine NaN, not a sentinel.
        return float(self.model.predict_proba(self._current_vector.reshape(1, -1))[0, 1])


# === Rollout helpers ===========================================================

def rollout_evasion_rate(
    policy: PPO, env: FraudEvasionEnv, n_transactions: int, deterministic: bool, seed_offset: int = 0,
) -> tuple[float, list[dict]]:
    """One rollout per fraud transaction (0..n_transactions-1). Returns
    (evasion_rate, per-episode info dicts for successes)."""
    successes: list[dict] = []
    for row_idx in range(n_transactions):
        obs, _ = env.reset(seed=seed_offset + row_idx, options={"row_idx": row_idx})
        done = False
        last_info: dict = {}
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            last_info = info
            done = bool(terminated or truncated)
            if terminated:
                successes.append({**info, "perturbed_vector": env._current_vector.copy(),
                                   "orig_vector": env._orig_vector.copy(), "n_steps": env._steps})
                break
    return len(successes) / n_transactions, successes


# === Main =====================================================================

def main() -> None:
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 20 — PPO ADVERSARIAL ATTACKER")
    log.info("=" * 70)

    with open(METRICS_ENRICHED) as f:
        threshold = json.load(f)["best_f1_threshold"]["best_threshold"]
    log.info("Target model: %s. Routing threshold (evasion target): %.4f", MODEL_ENRICHED, threshold)

    model = XGBClassifier()
    model.load_model(str(MODEL_ENRICHED))

    X_fraud_all, fraud_positions_all, feature_names = load_fraud_test_vectors()

    # Restrict the attack pool to fraud transactions the model actually
    # catches BEFORE any perturbation (prob >= threshold, i.e. would be
    # BLOCKed). Roughly 638/2283 real fraud transactions already score
    # below threshold with zero tampering (recall is ~72%, not 100%) —
    # "evading" those isn't an adversarial success, it's a pre-existing
    # model miss. Filtering here is what makes "evasion rate" mean
    # "fraction of transactions the model WOULD catch that the attacker
    # can talk it out of," not an inflated number contaminated by misses
    # the attacker had nothing to do with.
    prob_orig = model.predict_proba(X_fraud_all)[:, 1]
    caught_mask = prob_orig >= threshold
    n_already_missed = int((~caught_mask).sum())
    log.info("Of %d real fraud test transactions, %d (%.1f%%) already score below the routing "
              "threshold with ZERO perturbation (pre-existing model misses, recall ~%.1f%%) — "
              "excluded from the attack pool. Evasion is only meaningful against the %d the "
              "model actually catches unperturbed.",
              len(X_fraud_all), n_already_missed, n_already_missed / len(X_fraud_all) * 100,
              caught_mask.mean() * 100, int(caught_mask.sum()))
    X_fraud = X_fraud_all[caught_mask]
    fraud_positions = fraud_positions_all[caught_mask]

    continuous_idx = np.array([feature_names.index(n) for n in PERTURBABLE_CONTINUOUS])
    binary_idx = np.array([feature_names.index(n) for n in PERTURBABLE_BINARY])
    continuous_scale = np.array([max(X_fraud[:, i].std(), 1e-6) for i in continuous_idx])

    log.info("-" * 70)
    log.info("PERTURBABLE FEATURES (attacker-controllable subset of 243)")
    for name, idx, std in zip(PERTURBABLE_CONTINUOUS, continuous_idx, continuous_scale):
        log.info("  [continuous] %-28s idx=%3d  real-fraud std=%.4f", name, idx, std)
    for name, idx in zip(PERTURBABLE_BINARY, binary_idx):
        log.info("  [binary]     %-28s idx=%3d  real-fraud mean=%.4f", name, idx, X_fraud[:, idx].mean())
    log.info("Budget: %.1f std-units total (continuous only); binary flips soft-penalized at %.2f each.",
              BUDGET_STD_UNITS, BINARY_FLIP_PENALTY)

    env = FraudEvasionEnv(model, X_fraud, continuous_idx, continuous_scale, binary_idx, threshold)
    monitored_env = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored_env])

    log.info("-" * 70)
    log.info("Initializing PPO (untrained policy) ...")
    policy = PPO(
        "MlpPolicy", vec_env, verbose=0, seed=RANDOM_SEED,
        n_steps=256, batch_size=64, learning_rate=3e-4, gamma=0.95, device="cpu",
    )

    log.info("-" * 70)
    log.info("BEFORE TRAINING — evasion rate with an untrained (random-init) policy")
    n_fraud = len(X_fraud)
    before_rate, _ = rollout_evasion_rate(policy, env, n_fraud, deterministic=False, seed_offset=0)
    log.info("  Evasion rate (untrained): %d/%d = %.2f%%", int(round(before_rate * n_fraud)), n_fraud, before_rate * 100)

    log.info("-" * 70)
    log.info("TRAINING PPO for %s timesteps ...", f"{TOTAL_TIMESTEPS:,}")
    t_train0 = time.perf_counter()
    policy.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    train_time = time.perf_counter() - t_train0
    log.info("Training complete in %.1fs (%.1f min).", train_time, train_time / 60)

    episode_rewards = monitored_env.get_episode_rewards()
    log.info("Logged %d training episodes.", len(episode_rewards))
    if episode_rewards:
        n = len(episode_rewards)
        q1_mean = float(np.mean(episode_rewards[: n // 4])) if n >= 4 else float(np.mean(episode_rewards))
        q4_mean = float(np.mean(episode_rewards[-(n // 4):])) if n >= 4 else float(np.mean(episode_rewards))
        log.info("  Episode reward: first quartile mean=%.4f -> last quartile mean=%.4f (change=%+.4f)",
                  q1_mean, q4_mean, q4_mean - q1_mean)
        window = max(1, n // 50)
        smoothed = pd.Series(episode_rewards).rolling(window, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(episode_rewards, alpha=0.25, color="tab:blue", label="raw episode reward")
        ax.plot(smoothed, color="tab:blue", lw=2, label=f"rolling mean (window={window})")
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode reward")
        ax.set_title("PPO attacker training curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(CURVE_PLOT_PATH, dpi=150)
        plt.close(fig)
        log.info("Saved training curve: %s", CURVE_PLOT_PATH)

    policy.save(str(POLICY_OUTPUT_PATH))
    log.info("Saved policy: %s", POLICY_OUTPUT_PATH)

    log.info("-" * 70)
    log.info("AFTER TRAINING — evasion rate with the trained policy (deterministic)")
    after_rate, _ = rollout_evasion_rate(policy, env, n_fraud, deterministic=True, seed_offset=10_000)
    log.info("  Evasion rate (trained, deterministic): %d/%d = %.2f%%",
              int(round(after_rate * n_fraud)), n_fraud, after_rate * 100)

    log.info("-" * 70)
    log.info("HARVESTING — %d attempt(s) per transaction, stochastic policy", HARVEST_ATTEMPTS_PER_TRANSACTION)
    all_successes: dict[int, dict] = {}
    for attempt in range(HARVEST_ATTEMPTS_PER_TRANSACTION):
        _, successes = rollout_evasion_rate(
            policy, env, n_fraud, deterministic=False, seed_offset=20_000 + attempt * n_fraud,
        )
        for s in successes:
            all_successes.setdefault(s["row_idx"], s)  # keep first success found per transaction
        log.info("  Attempt %d/%d: %d new cumulative harvested evaders (%d total so far)",
                  attempt + 1, HARVEST_ATTEMPTS_PER_TRANSACTION, len(successes), len(all_successes))

    harvest_rate = len(all_successes) / n_fraud
    log.info("Harvest evasion rate (any of %d stochastic attempts): %d/%d = %.2f%%",
              HARVEST_ATTEMPTS_PER_TRANSACTION, len(all_successes), n_fraud, harvest_rate * 100)

    if not all_successes:
        log.info("=" * 70)
        log.info("NO SUCCESSFUL EVASIONS HARVESTED.")
        log.info("This means the enriched model is ROBUST to this bounded feature-space attack — "
                  "a GOOD result, not a failure. artifacts/evaders.parquet will not be written "
                  "(nothing to write); ppo_policy.zip and the training curve are saved regardless.")
    else:
        log.info("-" * 70)
        log.info("EVADER ANALYSIS")
        prob_drops = [s["prob_before"] - s["prob_after"] for s in all_successes.values()]
        log.info("  Mean probability drop for successful evasions: %.4f (min=%.4f, max=%.4f)",
                  np.mean(prob_drops), np.min(prob_drops), np.max(prob_drops))

        perturb_magnitudes = np.array([
            np.abs(s["perturbed_vector"][continuous_idx] - s["orig_vector"][continuous_idx]) / continuous_scale
            for s in all_successes.values()
        ])
        mean_abs_perturb_per_feature = perturb_magnitudes.mean(axis=0)
        log.info("  Mean |perturbation| per continuous feature (std-units), across %d evaders:", len(all_successes))
        feat_perturb_ranked = sorted(zip(PERTURBABLE_CONTINUOUS, mean_abs_perturb_per_feature), key=lambda kv: -kv[1])
        for name, v in feat_perturb_ranked:
            log.info("    %-28s %.4f", name, v)

        n_flipped_total = sum(s.get("n_binary_flipped", 0) for s in all_successes.values())
        log.info("  Binary flags flipped in successful evasions: %d flips across %d evaders",
                  n_flipped_total, len(all_successes))

        rows = []
        for s in all_successes.values():
            row = {feature_names[i]: s["perturbed_vector"][i] for i in range(len(feature_names))}
            for name, i in zip(PERTURBABLE_CONTINUOUS, continuous_idx):
                row[f"orig_{name}"] = s["orig_vector"][i]
            for name, i in zip(PERTURBABLE_BINARY, binary_idx):
                row[f"orig_{name}"] = s["orig_vector"][i]
            row["test_row_index"] = int(fraud_positions[s["row_idx"]])
            row["prob_before"] = s["prob_before"]
            row["prob_after"] = s["prob_after"]
            row["prob_drop"] = s["prob_before"] - s["prob_after"]
            row["perturbation_l2_std_units"] = s["cum_perturbation_l2_std_units"]
            row["n_steps_to_evade"] = s["n_steps"]
            row["n_binary_flipped"] = s.get("n_binary_flipped", 0)
            rows.append(row)

        evaders_df = pd.DataFrame(rows)
        evaders_df.to_parquet(EVADERS_OUTPUT_PATH, index=False)
        file_size_mb = EVADERS_OUTPUT_PATH.stat().st_size / 1e6
        log.info("Saved: %s (%s rows, %.3f MB)", EVADERS_OUTPUT_PATH, f"{len(evaders_df):,}", file_size_mb)

    total_runtime = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("Total runtime: %.1fs (%.1f min)", total_runtime, total_runtime / 60)

    output = {
        "description": "PPO adversarial attacker vs. the JANUS enriched model — feature-space "
                        "evasion attempt on real held-out fraud transactions.",
        "honest_context": (
            "This is a bounded FEATURE-SPACE attack (perturbs the model's input vector directly, "
            "restricted to attacker-plausible dims with a hard magnitude budget), not a full "
            "problem-space attack. A low evasion rate means the enriched model is robust to this "
            "attack surface — that is a good result for the model, not a failed experiment."
        ),
        "threat_model": {
            "perturbable_continuous": list(PERTURBABLE_CONTINUOUS),
            "perturbable_binary": list(PERTURBABLE_BINARY),
            "excluded_by_design": [
                "user_tx_count", "merchant_user_count", "mcc_user_count",
                "user_hist_mean", "user_hist_std", "user_hist_max",
                "merch_tx_count", "merch_7d_count", "merch_spike_ratio", "merch_days_active",
                "merch_hist_amount_mean", "merch_amount_deviation", "merch_large_tx_rate",
                "pop_prior_day_count", "pop_7d_avg_count", "first_mcc", "is_new_merchant",
            ],
            "budget_std_units": BUDGET_STD_UNITS,
            "binary_flip_penalty": BINARY_FLIP_PENALTY,
            "routing_threshold": threshold,
        },
        "training": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "train_time_seconds": round(train_time, 2),
            "n_episodes": len(episode_rewards),
            "reward_first_quartile_mean": q1_mean if episode_rewards else None,
            "reward_last_quartile_mean": q4_mean if episode_rewards else None,
        },
        "evasion_rate": {
            "n_fraud_test_transactions_total": len(X_fraud_all),
            "n_already_missed_pre_perturbation": n_already_missed,
            "n_fraud_test_transactions_in_attack_pool": n_fraud,
            "note": "Attack pool excludes fraud transactions the model already misses with zero "
                    "perturbation (prob < threshold pre-attack) — evasion rate below is relative "
                    "to the transactions the model actually catches unperturbed, not all fraud.",
            "before_training_random_policy": before_rate,
            "after_training_deterministic": after_rate,
            "after_training_harvest_any_of_n_attempts": harvest_rate,
            "harvest_attempts_per_transaction": HARVEST_ATTEMPTS_PER_TRANSACTION,
        },
        "evader_analysis": (
            {
                "n_evaders": len(all_successes),
                "mean_prob_drop": float(np.mean(prob_drops)),
                "min_prob_drop": float(np.min(prob_drops)),
                "max_prob_drop": float(np.max(prob_drops)),
                "mean_abs_perturbation_per_feature_std_units": {
                    name: float(v) for name, v in zip(PERTURBABLE_CONTINUOUS, mean_abs_perturb_per_feature)
                },
                "n_binary_flips_total": int(n_flipped_total),
            }
            if all_successes else None
        ),
        "outputs": {
            "policy": str(POLICY_OUTPUT_PATH),
            "training_curve_plot": str(CURVE_PLOT_PATH) if episode_rewards else None,
            "evaders_parquet": str(EVADERS_OUTPUT_PATH) if all_successes else None,
        },
        "runtime_seconds": round(total_runtime, 2),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", REPORT_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
