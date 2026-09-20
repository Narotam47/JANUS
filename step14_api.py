"""Step 14 — FastAPI inference service.

Ties together the frozen artifacts from steps 2, 8, 10, 11, and 13 into a
live scoring API.

Architecture
------------
POST /score receives RAW transaction fields (not pre-processed features).
It must reproduce step 2's preprocessing EXACTLY (same encoders, same
scaler, same derived columns) — this is the serving-consistency guarantee:
whatever the model saw at training time is exactly what it sees at serving
time, modulo the two documented gaps below.

Feature assembly needs three kinds of upstream state, of which only ONE is
covered by this step:

  1. PER-USER HISTORY (this step's GET /history/{user_id}) — covers all 14
     step-5 user-profile features and all 10 step-7 first-occurrence flags
     (24 of the 33 enriched features). STUBBED here with an always-empty
     mock; see `get_user_history()` for the full Redis contract step 15
     must implement.

  2. MERCHANT / POPULATION STATE (step 6's 9 features: merch_tx_count,
     merch_7d_count, merch_spike_ratio, merch_days_active,
     merch_hist_amount_mean, merch_amount_deviation, merch_large_tx_rate,
     pop_prior_day_count, pop_7d_avg_count) — these are GLOBAL, cross-user,
     cross-merchant aggregates. A per-user history lookup cannot supply
     them. NOT built yet — out of scope for both this step and step 15.
     `compute_merchant_population_stub()` fills these with exactly the
     fallback the TRAINING pipeline already uses for a merchant it has
     never seen before (see step 6's docstring, "Unseen merchants in
     test"), so this is a documented simplification, not a silent bug.
     A future "merchant/population feature store" service would replace it.

  3. THE 10-TRANSACTION WINDOW (this step, via #1) — the last 9 prior
     transactions (from history) plus the current one, each run through
     the same raw->clean transform as step 2, then flattened in the exact
     base-feature-major / timestep-minor order step 3 used.

Given the stub in #1 always returns empty history, EVERY request in this
step lands in the cold-start regime (position 0) and is routed by the
step-9 policy without ever invoking the main model. That's expected and
correct for this step — the code paths for the populated case are fully
implemented and tested, ready for step 15 to exercise for real.

Calibration caveat (step 12: Brier Skill Score -1.75, top-bin precision
87.3% at raw score 0.99): the response never labels the raw model output
"probability" or "confidence". It is `fraud_score_raw`, with an explicit
note that it's a ranking signal, not a calibrated probability. Use
`confidence_score` (step 10) and `routing_verdict` (step 11) for decisions.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, Optional

import numpy as np
import pyarrow.parquet as pq
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from step11_routing import route
from step13_shap_explain import WINDOW_FEATURE_COUNT, build_feature_names, explain

# --- Config -------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
MODEL_ENRICHED_PATH: Final[Path] = ARTIFACTS_DIR / "model_enriched.json"
MODEL_COLDSTART_PATH: Final[Path] = ARTIFACTS_DIR / "model_coldstart.json"
SCALER_PATH: Final[Path] = ARTIFACTS_DIR / "scaler.pkl"
ENCODERS_PATH: Final[Path] = ARTIFACTS_DIR / "encoders.pkl"
METRICS_ENRICHED_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"
CLEAN_TEST_PATH: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"  # schema source only
USER_FEATURES_TEST_PATH: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
MERCH_FEATURES_TEST_PATH: Final[Path] = ARTIFACTS_DIR / "merchant_features_test.parquet"
FLAGS_TEST_PATH: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

WINDOW_SIZE: Final[int] = 10
MIN_HISTORY_FOR_MAIN_MODEL: Final[int] = WINDOW_SIZE - 1  # 9 prior + current = 10

# Frozen contract from step 2 — the EXACT column order scaler.pkl was fit on.
SCALED_NUMERIC_COLS: Final[tuple[str, ...]] = (
    "amount_raw", "amount_log", "hours_since_last",
    "hour", "day_of_week", "month",
    "Merchant Name", "Merchant City", "Zip",
)
ERROR_TYPES: Final[tuple[str, ...]] = (
    "Bad CVV", "Bad Card Number", "Bad Expiration",
    "Bad PIN", "Bad Zipcode", "Insufficient Balance", "Technical Glitch",
)
CHIP_VALUES: Final[tuple[str, ...]] = (
    "Chip Transaction", "Online Transaction", "Swipe Transaction",
)
# Raw TabFormer quirk: online transactions carry this exact (leading-space)
# sentinel as Merchant City in the training data, not a real city name.
ONLINE_MERCHANT_CITY_SENTINEL: Final[str] = " ONLINE"
ONLINE_ZIP_SENTINEL: Final[str] = "00000"
MISSING_ZIP_SENTINEL: Final[str] = "MISSING"
ONLINE_STATE_SENTINEL: Final[str] = "ONLINE"

AUTH_BUDGET_MS: Final[float] = 50.0

FRAUD_SCORE_NOTE: Final[str] = (
    "Raw model output in [0,1]. NOT a calibrated probability — step 12 found "
    "Brier Skill Score -1.75 (worse than the naive base-rate baseline) and "
    "top-bin precision of only 87.3% when this score reads ~0.99. Treat as a "
    "RANKING signal only; do not display as 'X% confident'. Use "
    "confidence_score and routing_verdict for decisions and dashboard display."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step14")


# === Request / response schemas =============================================

class TransactionRaw(BaseModel):
    """Raw transaction fields, as a real caller would send them — not the
    pre-processed feature vector. Same shape is reused for history items."""

    user_id: str = Field(..., description="Matches the training data's User column.")
    timestamp: datetime = Field(..., description="Transaction time, ISO 8601.")
    amount: float = Field(..., description="Signed raw dollar amount (negative = refund).")
    use_chip: Literal["Chip Transaction", "Online Transaction", "Swipe Transaction"]
    merchant_name: str = Field(..., description="Raw merchant identifier (frequency-encoded).")
    merchant_city: str = Field("", description="Ignored/overridden for online transactions.")
    merchant_state: str = Field("", description="Empty is normalized per step 2's rules.")
    zip_code: str = Field("", description="Empty is normalized per step 2's rules.")
    mcc: str = Field(..., description="Merchant category code, as a string (matches the frozen label encoder).")
    errors: list[str] = Field(
        default_factory=list,
        description=f"Subset of {ERROR_TYPES}; unrecognized strings are ignored.",
    )


class UserHistorySummary(BaseModel):
    """Sufficient statistics for the per-user features — NOT full raw history.

    Maintained incrementally by whatever writes a transaction after it's
    scored (see `get_user_history`'s docstring for the Redis mapping).
    """

    tx_count: int = Field(0, description="Total PRIOR transactions for this user (0 = first-ever).")
    amount_sum: float = 0.0
    amount_sumsq: float = 0.0
    amount_max: float = 0.0
    last_timestamp: Optional[datetime] = None
    merchant_visit_counts: dict[str, int] = Field(default_factory=dict)
    mcc_visit_counts: dict[str, int] = Field(default_factory=dict)
    seen_city_states: list[str] = Field(default_factory=list, description='"{city}|{state}" keys already seen.')
    seen_channels: list[str] = Field(default_factory=list)
    seen_year_months: list[str] = Field(default_factory=list, description='"YYYY-MM" keys already seen.')
    seen_years: list[str] = Field(default_factory=list, description='"YYYY" keys already seen.')
    gap_sum_days: float = 0.0
    gap_sumsq_days: float = 0.0
    gap_count: int = 0
    monthly_amount_avg: dict[str, float] = Field(
        default_factory=dict, description='"YYYY-MM" -> mean amount that month.',
    )


class UserHistory(BaseModel):
    user_id: str
    summary: UserHistorySummary = Field(default_factory=UserHistorySummary)
    recent_transactions: list[TransactionRaw] = Field(
        default_factory=list,
        description=f"Last up to {WINDOW_SIZE - 1} prior transactions, OLDEST first.",
    )


class TopFeature(BaseModel):
    feature: str
    shap_value: float
    value: float


class RoutingVerdict(BaseModel):
    decision: str = Field(..., description='"block", "step_up", or "approve".')
    rule: str
    probability: Optional[float] = None
    confidence_score: Optional[float] = None
    confidence_reason: Optional[str] = None


class ScoreResponse(BaseModel):
    user_id: str
    fraud_score_raw: Optional[float] = Field(None, description=FRAUD_SCORE_NOTE)
    confidence_score: Optional[float] = None
    confidence_reason: Optional[str] = None
    routing_verdict: RoutingVerdict
    top_features: list[TopFeature] = Field(default_factory=list)
    rationale: str
    position: int = Field(..., description="This user's transaction index (0 = first-ever).")
    model_invoked: bool = Field(..., description="False for cold-start transactions — routed by policy, not the model.")
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    model_version: str
    coldstart_model_loaded: bool
    threshold: float
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_scored: int
    block_count: int
    step_up_count: int
    approve_count: int
    uptime_seconds: float


# === Raw -> clean transform (mirrors step 2 exactly, per-row) ===============

def _signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def _clean_zip(zip_code: str, use_chip: str) -> str:
    z = (zip_code or "")
    if z.endswith(".0"):
        z = z[:-2]
    if z == "":
        return ONLINE_ZIP_SENTINEL if use_chip == "Online Transaction" else MISSING_ZIP_SENTINEL
    return z


def _clean_state(state: str, use_chip: str) -> str:
    s = state or ""
    if use_chip == "Online Transaction" and s == "":
        return ONLINE_STATE_SENTINEL
    return s


def _clean_merchant_city(city: str, use_chip: str) -> str:
    # Raw TabFormer data always carries the leading-space sentinel for
    # online transactions' Merchant City — force it rather than trust the
    # caller to know the quirk.
    return ONLINE_MERCHANT_CITY_SENTINEL if use_chip == "Online Transaction" else city


def _encode_freq(col: str, value: str, encoders: dict) -> float:
    return float(encoders["freq_maps"][col].get(value, 0.0))


def _encode_label(col: str, value: str, encoders: dict) -> int:
    le = encoders["label_encoders"][col]
    known = set(le.classes_)
    v = value if value in known else le.classes_[0]
    return int(le.transform([v])[0])


def build_clean_row(
    tx: TransactionRaw, prev_timestamp: Optional[datetime], encoders: dict,
) -> dict[str, float]:
    """Raw transaction -> unscaled clean row, mirroring step 2's per-row logic.

    Numeric columns in SCALED_NUMERIC_COLS are left UNSCALED here — call
    `scale_window_rows()` on the full 10-row window before using it.
    """
    merchant_city = _clean_merchant_city(tx.merchant_city, tx.use_chip)
    merchant_state = _clean_state(tx.merchant_state, tx.use_chip)
    zip_code = _clean_zip(tx.zip_code, tx.use_chip)

    hours_since_last = (
        (tx.timestamp - prev_timestamp).total_seconds() / 3600.0
        if prev_timestamp is not None else 0.0  # matches step 2's fillna(0.0)
    )

    row: dict[str, float] = {
        "Merchant Name": _encode_freq("Merchant Name", tx.merchant_name, encoders),
        "Merchant City": _encode_freq("Merchant City", merchant_city, encoders),
        "Merchant State": float(_encode_label("Merchant State", merchant_state, encoders)),
        "Zip": _encode_freq("Zip", zip_code, encoders),
        "MCC": float(_encode_label("MCC", tx.mcc, encoders)),
        "amount_raw": tx.amount,
        "amount_log": _signed_log1p(tx.amount),
        "hours_since_last": hours_since_last,
        "hour": float(tx.timestamp.hour),
        "day_of_week": float(tx.timestamp.weekday()),
        "month": float(tx.timestamp.month),
    }
    for err in ERROR_TYPES:
        col = "err_" + err.lower().replace(" ", "_")
        row[col] = 1.0 if err in tx.errors else 0.0
    for chip_val in CHIP_VALUES:
        row[f"chip_{chip_val}"] = 1.0 if tx.use_chip == chip_val else 0.0

    return row


def scale_window_rows(rows: list[dict[str, float]], scaler) -> None:
    """Applies the frozen StandardScaler in place, in its fitted column order."""
    arr = np.array([[r[c] for c in SCALED_NUMERIC_COLS] for r in rows], dtype=np.float64)
    scaled = scaler.transform(arr)
    for i, r in enumerate(rows):
        for j, c in enumerate(SCALED_NUMERIC_COLS):
            r[c] = float(scaled[i, j])


def build_window_block(rows: list[dict[str, float]], window_base_cols: list[str]) -> np.ndarray:
    """Flattens 10 clean rows -> 210-dim vector, base-feature-major / timestep-minor
    (column f*10+t), matching step 3's sliding_window_view flattening exactly."""
    assert len(rows) == WINDOW_SIZE
    flat = np.empty(WINDOW_FEATURE_COUNT, dtype=np.float32)
    for f, col in enumerate(window_base_cols):
        for t in range(WINDOW_SIZE):
            flat[f * WINDOW_SIZE + t] = rows[t][col]
    return flat


# === Enriched features computable from PER-USER history (steps 5 & 7) =======

def compute_user_profile_features(tx: TransactionRaw, summary: UserHistorySummary) -> dict[str, float]:
    """Ports step 5's 14 formulas onto live summary stats instead of a DataFrame."""
    n = summary.tx_count
    if n == 0:
        hist_mean = hist_std = hist_max = zscore = ratio_to_max = float("nan")
    else:
        hist_mean = summary.amount_sum / n
        if n >= 2:
            var = max((summary.amount_sumsq - n * hist_mean ** 2) / (n - 1), 0.0)
            hist_std = var ** 0.5
        else:
            hist_std = float("nan")  # matches expanding(min_periods=2)
        hist_max = summary.amount_max
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = (
                float(np.divide(tx.amount - hist_mean, hist_std))
                if not np.isnan(hist_std) else float("nan")
            )
        ratio_to_max = tx.amount / hist_max if abs(hist_max) > 1e-8 else float("nan")

    merchant_user_count = summary.merchant_visit_counts.get(tx.merchant_name, 0)
    mcc_user_count = summary.mcc_visit_counts.get(tx.mcc, 0)

    ym_key = f"{tx.timestamp.year - 1}-{tx.timestamp.month:02d}"
    same_month_last_year_mean = summary.monthly_amount_avg.get(ym_key, float("nan"))
    amount_vs_last_year_ratio = (
        tx.amount / same_month_last_year_mean
        if not np.isnan(same_month_last_year_mean) and abs(same_month_last_year_mean) > 1e-8
        else float("nan")
    )

    return {
        "user_hist_mean": hist_mean,
        "user_hist_std": hist_std,
        "user_hist_max": hist_max,
        "user_tx_count": float(n),
        "user_amount_zscore": zscore,
        "user_amount_to_max_ratio": ratio_to_max,
        "merchant_user_count": float(merchant_user_count),
        "mcc_user_count": float(mcc_user_count),
        "is_new_merchant": 1.0 if merchant_user_count == 0 else 0.0,
        "same_month_last_year_mean": same_month_last_year_mean,
        "amount_vs_last_year_ratio": amount_vs_last_year_ratio,
        "is_weekend": 1.0 if tx.timestamp.weekday() >= 5 else 0.0,
        "is_holiday_season": 1.0 if tx.timestamp.month in (11, 12) else 0.0,
        "day_of_year": float(tx.timestamp.timetuple().tm_yday),
    }


def compute_first_occurrence_flags(tx: TransactionRaw, summary: UserHistorySummary) -> dict[str, float]:
    """Ports step 7's 10 formulas onto live summary stats instead of a DataFrame."""
    n = summary.tx_count
    city_state_key = f"{tx.merchant_city}|{tx.merchant_state}"
    year_month_key = f"{tx.timestamp.year}-{tx.timestamp.month:02d}"
    year_key = str(tx.timestamp.year)

    first_mcc = 1.0 if summary.mcc_visit_counts.get(tx.mcc, 0) == 0 else 0.0
    first_city_state = 1.0 if city_state_key not in summary.seen_city_states else 0.0
    first_channel = 1.0 if tx.use_chip not in summary.seen_channels else 0.0
    first_tx_this_month = 1.0 if year_month_key not in summary.seen_year_months else 0.0
    first_tx_this_year = 1.0 if year_key not in summary.seen_years else 0.0

    if n == 0:
        is_largest_ever = is_2x_hist_max = is_5x_hist_max = float("nan")
    else:
        hist_max = summary.amount_max
        is_largest_ever = 1.0 if tx.amount > hist_max else 0.0
        is_2x_hist_max = 1.0 if tx.amount > 2 * hist_max else 0.0
        is_5x_hist_max = 1.0 if tx.amount > 5 * hist_max else 0.0

    if summary.last_timestamp is None:
        days_since_last_tx = float("nan")
    else:
        days_since_last_tx = (tx.timestamp - summary.last_timestamp).total_seconds() / 86400.0

    if summary.gap_count >= 2 and not np.isnan(days_since_last_tx):
        gap_mean = summary.gap_sum_days / summary.gap_count
        gap_var = max(
            (summary.gap_sumsq_days - summary.gap_count * gap_mean ** 2) / (summary.gap_count - 1), 0.0,
        )
        gap_std = gap_var ** 0.5
        is_dormant = 1.0 if days_since_last_tx > gap_mean + 2 * gap_std else 0.0
    else:
        is_dormant = float("nan")

    return {
        "is_largest_ever": is_largest_ever,
        "is_2x_hist_max": is_2x_hist_max,
        "is_5x_hist_max": is_5x_hist_max,
        "first_mcc": first_mcc,
        "first_city_state": first_city_state,
        "first_channel": first_channel,
        "days_since_last_tx": days_since_last_tx,
        "is_dormant": is_dormant,
        "first_tx_this_month": first_tx_this_month,
        "first_tx_this_year": first_tx_this_year,
    }


def compute_merchant_population_stub() -> dict[str, float]:
    """Step 6's 9 features need GLOBAL merchant/population state that no
    per-user history endpoint can supply — see the module docstring's
    architecture section, gap #2. Defaults exactly match the training
    pipeline's own fallback for a never-before-seen merchant.
    """
    return {
        "merch_tx_count": 0.0,
        "merch_7d_count": 0.0,
        "merch_spike_ratio": float("nan"),
        "merch_days_active": 0.0,
        "merch_hist_amount_mean": float("nan"),
        "merch_amount_deviation": float("nan"),
        "merch_large_tx_rate": float("nan"),
        "pop_prior_day_count": 0.0,
        "pop_7d_avg_count": 0.0,
    }


def assemble_feature_vector(
    tx: TransactionRaw, history: UserHistory, app_state,
) -> tuple[Optional[np.ndarray], int]:
    """Builds the 243-dim model input, or signals cold-start with `None`.

    Returns (vector_or_none, position). position = summary.tx_count
    (0 = user's first-ever transaction), used for cold-start routing
    whether or not a vector was built.
    """
    summary = history.summary
    position = summary.tx_count

    if position < MIN_HISTORY_FOR_MAIN_MODEL:
        return None, position

    recent = history.recent_transactions[-(WINDOW_SIZE - 1):]
    if len(recent) < WINDOW_SIZE - 1:
        # Summary claims enough history but recent_transactions came up
        # short — treat conservatively as insufficient rather than guess.
        return None, position

    rows: list[dict[str, float]] = []
    prev_ts: Optional[datetime] = None
    for wtx in recent + [tx]:
        # NOTE: the oldest window row's hours_since_last ideally uses the
        # transaction BEFORE it, which isn't in `recent` (capped at
        # WINDOW_SIZE-1). It defaults to 0.0 here — the same convention as
        # a genuine first-ever transaction — a minor, documented
        # approximation on one feature of the least-recent timestep.
        rows.append(build_clean_row(wtx, prev_ts, app_state.encoders))
        prev_ts = wtx.timestamp

    scale_window_rows(rows, app_state.scaler)
    window_block = build_window_block(rows, app_state.window_base_cols)

    user_feats = compute_user_profile_features(tx, summary)
    flag_feats = compute_first_occurrence_flags(tx, summary)
    merch_feats = compute_merchant_population_stub()

    enriched = (
        [user_feats[c] for c in app_state.user_cols]
        + [merch_feats[c] for c in app_state.merch_cols]
        + [flag_feats[c] for c in app_state.flag_cols]
    )
    vector = np.concatenate([window_block, np.array(enriched, dtype=np.float32)])
    return vector, position


# === Redis stub — step 15's contract =========================================

def get_user_history(user_id: str) -> UserHistory:
    """Fetch a user's transaction history for feature assembly.

    STEP 15 CONTRACT: replace this function's body with a real Redis client
    call. Both the internal fast-path used by /score AND the public
    GET /history/{user_id} endpoint below call this same function, so
    implementing it here updates both at once.

    Expected real implementation (Redis), all WRITES done by whatever
    service persists a transaction after it's scored (never by this
    read-only endpoint):
      * `summary`'s scalar fields (tx_count, amount_sum, amount_sumsq,
        amount_max, last_timestamp, gap_sum_days, gap_sumsq_days,
        gap_count) -> a Redis HASH "user:{user_id}:summary", updated with
        HINCRBYFLOAT / HSET after each transaction.
      * `merchant_visit_counts` / `mcc_visit_counts` -> Redis HASHes
        "user:{user_id}:merchant_counts" / "...:mcc_counts", updated with
        HINCRBY, read in full with HGETALL.
      * `seen_city_states` / `seen_channels` / `seen_year_months` /
        `seen_years` -> Redis SETs, updated with SADD, read with SMEMBERS.
      * `monthly_amount_avg` -> a Redis HASH keyed by "YYYY-MM", maintained
        as a running average by the writer.
      * `recent_transactions` -> a Redis LIST "user:{user_id}:recent"
        (or a timestamp-scored SORTED SET), holding raw transaction JSON,
        trimmed to the last WINDOW_SIZE-1 (9) entries with LTRIM/ZREMRANGEBYRANK
        after each push, read with LRANGE.

    CURRENT STUB: always returns empty/zeroed history, so every request has
    position=0 and is routed by the cold-start policy (step 9) rather than
    the main model — intentional for this step.
    """
    return UserHistory(user_id=user_id, summary=UserHistorySummary(), recent_transactions=[])


# === App lifecycle ============================================================

def load_artifacts(app: FastAPI) -> None:
    required = (
        MODEL_ENRICHED_PATH, SCALER_PATH, ENCODERS_PATH, METRICS_ENRICHED_PATH,
        CLEAN_TEST_PATH, USER_FEATURES_TEST_PATH, MERCH_FEATURES_TEST_PATH, FLAGS_TEST_PATH,
    )
    for p in required:
        if not p.exists():
            raise RuntimeError(f"Missing required artifact: {p}")

    model = XGBClassifier()
    model.load_model(str(MODEL_ENRICHED_PATH))

    if MODEL_COLDSTART_PATH.exists():
        coldstart_model = XGBClassifier()
        coldstart_model.load_model(str(MODEL_COLDSTART_PATH))
        log.info("Loaded cold-start model from %s", MODEL_COLDSTART_PATH)
    else:
        coldstart_model = None
        log.warning(
            "model_coldstart.json not found — EXPECTED, not a startup error. Step 9 "
            "found ZERO fraud in the cold-start regime (positions 0-8) in the "
            "TabFormer synthetic data (earliest fraud at position 68), so no "
            "supervised cold-start model could be trained. Cold-start transactions "
            "are handled by a fixed routing policy instead (see step 9 / step 11): "
            "position 0 -> step_up, positions 1-8 -> approve. Full finding in "
            "artifacts/metrics_coldstart.json.",
        )

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(METRICS_ENRICHED_PATH) as f:
        metrics_enriched = json.load(f)
    threshold = metrics_enriched["best_f1_threshold"]["best_threshold"]
    model_version = metrics_enriched.get("model", "unknown")

    user_cols = pq.ParquetFile(USER_FEATURES_TEST_PATH).schema.names
    merch_cols = pq.ParquetFile(MERCH_FEATURES_TEST_PATH).schema.names
    flag_cols = pq.ParquetFile(FLAGS_TEST_PATH).schema.names
    window_base_cols = [
        c for c in pq.ParquetFile(CLEAN_TEST_PATH).schema.names if c not in ("User", "label")
    ]
    feature_names = build_feature_names(CLEAN_TEST_PATH, user_cols, merch_cols, flag_cols)

    app.state.model = model
    app.state.coldstart_model = coldstart_model
    app.state.scaler = scaler
    app.state.encoders = encoders
    app.state.threshold = threshold
    app.state.model_version = model_version
    app.state.feature_names = feature_names
    app.state.window_base_cols = window_base_cols
    app.state.user_cols = user_cols
    app.state.merch_cols = merch_cols
    app.state.flag_cols = flag_cols
    app.state.start_time = time.time()
    app.state.counters = {"total_scored": 0, "block": 0, "step_up": 0, "approve": 0}

    log.info(
        "Startup complete. %d features (window=%d, enriched=%d), threshold=%.4f, "
        "cold-start model loaded=%s",
        len(feature_names), WINDOW_FEATURE_COUNT, len(feature_names) - WINDOW_FEATURE_COUNT,
        threshold, coldstart_model is not None,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts(app)
    yield
    log.info("Shutting down.")


app = FastAPI(title="JANUS Fraud Scoring API", lifespan=lifespan)


# === Endpoints ================================================================

@app.post("/score", response_model=ScoreResponse)
async def score_transaction(tx: TransactionRaw) -> ScoreResponse:
    t0 = time.perf_counter()
    state = app.state

    # Internal call — see get_user_history()'s docstring for the Redis
    # contract step 15 implements. This is a direct function call (not an
    # HTTP round-trip to the endpoint below) to avoid paying network
    # latency twice inside the authorization budget.
    history = get_user_history(tx.user_id)

    vector, position = assemble_feature_vector(tx, history, state)

    if vector is None:
        verdict = route(
            prob=0.0, confidence_score=0.0, confidence_reason="cold_start",
            position=position, prob_block=state.threshold,
        )
        if position == 0:
            rationale = (
                "Not evaluated by the model — this is the user's first-ever "
                "transaction, so there is no history to compare against. Routed "
                "to step-up per the cold-start policy (step 9)."
            )
        else:
            rationale = (
                f"Not evaluated by the model — the user has only {position} prior "
                "transaction(s), below the 9 required for the main model. Routed "
                "to approve per the cold-start policy (step 9: zero fraud observed "
                "in this regime in the training data)."
            )
        response = ScoreResponse(
            user_id=tx.user_id,
            fraud_score_raw=None,
            confidence_score=None,
            confidence_reason=None,
            routing_verdict=RoutingVerdict(
                decision=verdict["decision"], rule=verdict["rule"],
                probability=None, confidence_score=None, confidence_reason=None,
            ),
            top_features=[],
            rationale=rationale,
            position=position,
            model_invoked=False,
        )
    else:
        result = explain(vector, state.model, state.feature_names, state.threshold, top_n=5)
        response = ScoreResponse(
            user_id=tx.user_id,
            fraud_score_raw=result["probability"],
            confidence_score=result["confidence_score"],
            confidence_reason=result["confidence_reason"],
            routing_verdict=RoutingVerdict(**result["routing_verdict"]),
            top_features=[TopFeature(**f) for f in result["top_features"]],
            rationale=result["rationale"],
            position=position,
            model_invoked=True,
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    response.latency_ms = round(latency_ms, 4)

    decision = response.routing_verdict.decision
    state.counters["total_scored"] += 1
    state.counters[decision] = state.counters.get(decision, 0) + 1

    log.info(
        "score user=%s position=%d decision=%s latency=%.3fms%s",
        tx.user_id, position, decision, latency_ms,
        "" if vector is not None else " (cold-start, model not invoked)",
    )
    if latency_ms > AUTH_BUDGET_MS:
        log.warning(
            "Request latency %.3fms EXCEEDS the %.1fms authorization budget!",
            latency_ms, AUTH_BUDGET_MS,
        )

    return response


@app.get("/history/{user_id}", response_model=UserHistory)
def get_history_endpoint(user_id: str) -> UserHistory:
    return get_user_history(user_id)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = app.state
    return HealthResponse(
        status="ok",
        model_version=state.model_version,
        coldstart_model_loaded=state.coldstart_model is not None,
        threshold=state.threshold,
        uptime_seconds=round(time.time() - state.start_time, 3),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    state = app.state
    c = state.counters
    return MetricsResponse(
        total_scored=c.get("total_scored", 0),
        block_count=c.get("block", 0),
        step_up_count=c.get("step_up", 0),
        approve_count=c.get("approve", 0),
        uptime_seconds=round(time.time() - state.start_time, 3),
    )


# === Integration test =========================================================

def test_score_endpoint_integration() -> None:
    """Sends a synthetic transaction through the full stack; checks the schema.

    Runnable with pytest (`pytest step14_api.py`) or standalone
    (`python step14_api.py test`).
    """
    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_body = health_resp.json()
        assert health_body["status"] == "ok"
        assert health_body["coldstart_model_loaded"] is False, (
            "Expected no cold-start model per step 9's finding"
        )

        payload = {
            "user_id": "test-user-001",
            "timestamp": "2026-01-15T14:30:00",
            "amount": 42.50,
            "use_chip": "Chip Transaction",
            "merchant_name": "1799189980464955940",
            "merchant_city": "Brooklyn",
            "merchant_state": "NY",
            "zip_code": "11201",
            "mcc": "5411",
            "errors": [],
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()

        for key in (
            "user_id", "fraud_score_raw", "confidence_score", "confidence_reason",
            "routing_verdict", "top_features", "rationale", "position",
            "model_invoked", "latency_ms",
        ):
            assert key in body, f"Missing key in response: {key}"

        assert "decision" in body["routing_verdict"]
        assert "rule" in body["routing_verdict"]

        # Stubbed history -> every request is this user's first-ever transaction.
        assert body["position"] == 0
        assert body["model_invoked"] is False
        assert body["routing_verdict"]["decision"] == "step_up"
        assert body["routing_verdict"]["rule"] == "cold_start_first_tx"
        assert body["fraud_score_raw"] is None
        assert "first-ever" in body["rationale"]
        assert body["latency_ms"] > 0

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics_body = metrics_resp.json()
        assert metrics_body["total_scored"] == 1
        assert metrics_body["step_up_count"] == 1
        assert metrics_body["block_count"] == 0
        assert metrics_body["approve_count"] == 0

        history_resp = client.get("/history/test-user-001")
        assert history_resp.status_code == 200
        assert history_resp.json()["summary"]["tx_count"] == 0

    log.info("Integration test PASSED.")


# === Entry point ===============================================================

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_score_endpoint_integration()
    else:
        main()
