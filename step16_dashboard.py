"""Step 16 — Streamlit SOC (Security Operations Center) dashboard.

Run with:  streamlit run step16_dashboard.py

Two clearly separated halves in this file:

  UPSTREAM SIMULATOR (`_score_and_produce_rows`, `bootstrap_demo_stream`,
  the sidebar "score more" button) — stands in for steps 14/15's live
  scoring service, which in production runs continuously and independently
  of this dashboard. It samples real held-out test-set transactions,
  scores them with the REAL model via step 13's `explain()`, and produces
  the results onto step 15's `verdicts` topic (the mock bus here, since no
  broker is running — see step 15's docstring for how that swap works).

  THE DASHBOARD ITSELF (everything under "RENDER" below) — a pure
  consumer. It only ever reads already-scored verdict messages off the
  `verdicts` topic (`drain_verdicts_topic`) and displays them. It never
  calls `explain()`, `route()`, or the model to decide what to show for a
  given transaction — by the time a verdict reaches the dashboard, scoring
  already happened upstream. The one exception, by design, is the
  threshold panel: it recomputes AGGREGATE counts from a precomputed
  probability/confidence distribution over the full test set using pure
  vector ops (`step11_routing.route_batch`), never the model — exactly
  what the task asks for ("should NOT require re-running the model").

Calibration compliance (step 12: Brier Skill Score -1.75): nothing here is
labeled "fraud probability" or "X% confident this is fraud". The raw
model output is `fraud_score_raw`, shown only inside a collapsed expander
in the detail panel, explicitly labeled a ranking signal.
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timedelta, timezone
from queue import Empty
from typing import Any, Final

import altair as alt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from xgboost import XGBClassifier

import step10_confidence as s10
import step11_routing as s11
import step13_shap_explain as s13
import step14_api as api
import step15_streaming as streaming

# --- Page config --------------------------------------------------------
st.set_page_config(page_title="JANUS — SOC Dashboard", layout="wide", page_icon="🛡️")

# --- Constants ------------------------------------------------------------
DECISION_BADGE: Final[dict[str, str]] = {
    "block": "🟥 BLOCK",
    "step_up": "🟧 STEP-UP",
    "approve": "🟩 APPROVE",
}
DECISION_ORDER: Final[list[str]] = ["block", "step_up", "approve"]
DECISION_CHART_COLORS: Final[list[str]] = ["#d62728", "#ff9f1c", "#2ca02c"]

CONF_TIER_HIGH: Final[float] = 0.85
CONF_TIER_MODERATE: Final[float] = 0.75  # matches step 11's DEFAULT_CONF_STEPUP

DEMO_N_BLOCK: Final[int] = 6
DEMO_N_STEP_UP: Final[int] = 10
DEMO_N_APPROVE: Final[int] = 24
ADDITIONAL_BATCH_SIZE: Final[int] = 8

STEP11_REFERENCE_BLOCK_PRECISION: Final[float] = 0.8588
STEP11_REFERENCE_STEPUP_RATE_OF_LEGIT: Final[float] = 0.001936


# =============================================================================
# UPSTREAM SIMULATOR — stands in for steps 14/15's live scoring service
# =============================================================================

@st.cache_resource(show_spinner="Loading model and test-set arrays (one-time) ...")
def load_scoring_resources() -> dict[str, Any]:
    """Loads everything the simulator and the threshold panel need, once
    per Streamlit process. Mirrors the exact pattern steps 11-13 already
    validated (mmap windows, hstack enriched features, batch confidence +
    routing) — no new computation logic invented here."""
    idx = s13.compute_window_feature_indices(s13.CLEAN_TEST)
    y_test = np.asarray(np.load(str(s13.LABELS_TEST), mmap_mode="r"))

    user_feats = pd.read_parquet(s13.USER_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    merch_feats = pd.read_parquet(s13.MERCH_FEATURES_TEST).iloc[idx].reset_index(drop=True)
    flags = pd.read_parquet(s13.FLAGS_TEST).iloc[idx].reset_index(drop=True)
    user_ids = pd.read_parquet(s13.CLEAN_TEST, columns=["User"])["User"].iloc[idx].reset_index(drop=True)

    feature_names = s13.build_feature_names(
        s13.CLEAN_TEST, list(user_feats.columns), list(merch_feats.columns), list(flags.columns),
    )
    window_base_cols = [c for c in pq.ParquetFile(s13.CLEAN_TEST).schema.names if c not in ("User", "label")]

    with open(s13.METRICS_ENRICHED) as f:
        threshold = json.load(f)["best_f1_threshold"]["best_threshold"]

    model = XGBClassifier()
    model.load_model(str(s13.MODEL_ENRICHED))

    with open(api.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_win = np.load(str(s13.WINDOWS_TEST), mmap_mode="r")
    u = user_feats.values.astype(np.float32)
    m = merch_feats.values.astype(np.float32)
    fl = flags.values.astype(np.float32)
    enriched = np.hstack([u, m, fl])
    X = np.hstack([X_win, enriched]).astype(np.float32)
    del enriched, u, m, fl

    prob = model.predict_proba(X)[:, 1]

    n_nan = (
        user_feats.isna().sum(axis=1).values
        + merch_feats.isna().sum(axis=1).values
        + flags.isna().sum(axis=1).values
    )
    confidence_score, _ = s10.confidence_batch(
        prob=prob, threshold=threshold,
        user_tx_count=user_feats["user_tx_count"].values,
        merchant_user_count=user_feats["merchant_user_count"].values,
        mcc_user_count=user_feats["mcc_user_count"].values,
        first_mcc=flags["first_mcc"].values,
        first_city_state=flags["first_city_state"].values,
        first_channel=flags["first_channel"].values,
        n_nan=n_nan,
    )
    position = user_feats["user_tx_count"].values.astype(int)
    decisions, rules = s11.route_batch(prob, confidence_score, position)

    return {
        "model": model, "scaler": scaler, "threshold": threshold,
        "feature_names": feature_names, "window_base_cols": window_base_cols,
        "X": X, "y_test": y_test, "user_ids": user_ids.values,
        "prob": prob, "confidence_score": confidence_score, "position": position,
        "decisions": decisions, "rules": rules,
    }


def _score_and_produce_rows(row_indices: np.ndarray, res: dict[str, Any]) -> None:
    """Runs the REAL model + explain() on each sampled test-set row and
    produces the result onto the `verdicts` topic — exactly the job
    steps 14/15's live service does. This is the ONLY place in this file
    that calls the model; the dashboard rendering code never does."""
    producer = streaming.make_producer()
    now = datetime.now(timezone.utc)
    amount_col_idx = api.SCALED_NUMERIC_COLS.index("amount_raw")
    amount_flat_idx = res["window_base_cols"].index("amount_raw") * api.WINDOW_SIZE + (api.WINDOW_SIZE - 1)
    scaler = res["scaler"]

    n = len(row_indices)
    for i, row in enumerate(row_indices):
        row = int(row)
        x = res["X"][row]
        result = s13.explain(x, res["model"], res["feature_names"], res["threshold"], top_n=5)

        scaled_amount = float(x[amount_flat_idx])
        real_amount = scaled_amount * scaler.scale_[amount_col_idx] + scaler.mean_[amount_col_idx]

        # Spread scored_at over the recent past so the feed/time-series look live,
        # oldest first. This is a demo-generation timestamp, not the original
        # (years-old) TabFormer transaction date — everything else about the
        # transaction (amount, user, model behavior) is genuinely real.
        scored_at = now - timedelta(seconds=(n - i) * 11)

        verdict = {
            "transaction_id": f"demo-{row}",
            "user_id": str(res["user_ids"][row]),
            "decision": result["routing_verdict"]["decision"],
            "rule": result["routing_verdict"]["rule"],
            "rationale": result["rationale"],
            "top_features": result["top_features"],
            "fraud_score_raw": result["probability"],
            "confidence_score": result["confidence_score"],
            "confidence_reason": result["confidence_reason"],
            "position": int(res["position"][row]),
            "model_invoked": True,
            "scored_at": scored_at.isoformat(),
            "amount": round(float(real_amount), 2),
            "true_label": "fraud" if res["y_test"][row] == 1 else "legit",
            "latency_ms": {"scoring_ms": 0.0, "redis_write_ms": 0.0, "total_e2e_ms": 0.0},
        }
        producer.produce(streaming.VERDICTS_TOPIC, key=verdict["user_id"], value=verdict)
    producer.flush()


@st.cache_resource(show_spinner="Scoring a realistic demo transaction stream (real model, real test-set data) ...")
def bootstrap_demo_stream(seed: int = 7) -> bool:
    """Stratified sample so the feed genuinely contains all three verdict
    types, drawn from real routing outcomes on the held-out test set —
    not fabricated. Runs once per process (st.cache_resource)."""
    res = load_scoring_resources()
    rng = np.random.default_rng(seed)
    decisions = res["decisions"]

    def _sample(label: str, n: int) -> np.ndarray:
        pool = np.where(decisions == label)[0]
        return rng.choice(pool, size=min(n, len(pool)), replace=False)

    sample_idx = np.concatenate([
        _sample("block", DEMO_N_BLOCK),
        _sample("step_up", DEMO_N_STEP_UP),
        _sample("approve", DEMO_N_APPROVE),
    ])
    rng.shuffle(sample_idx)
    _score_and_produce_rows(sample_idx, res)
    return True


def score_additional_batch(res: dict[str, Any]) -> int:
    """Uncached — called fresh each time the sidebar button is clicked."""
    rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
    pool = np.arange(len(res["y_test"]))
    sample_idx = rng.choice(pool, size=ADDITIONAL_BATCH_SIZE, replace=False)
    _score_and_produce_rows(sample_idx, res)
    return len(sample_idx)


def recompute_routing(
    prob: np.ndarray, confidence_score: np.ndarray, position: np.ndarray, y_test: np.ndarray,
    prob_block: float, conf_stepup: float,
) -> dict[str, float]:
    """Pure numpy — no model call. This is what makes the threshold slider
    instant: `route_batch` just re-applies the routing rules to already-
    computed probabilities/confidence scores."""
    decisions, _ = s11.route_batch(prob, confidence_score, position, prob_block=prob_block, conf_stepup=conf_stepup)
    is_block = decisions == "block"
    is_stepup = decisions == "step_up"
    is_approve = decisions == "approve"
    fraud = y_test == 1
    legit = y_test == 0

    block_tp = int((is_block & fraud).sum())
    block_fp = int((is_block & legit).sum())
    approve_fraud = int((is_approve & fraud).sum())

    return {
        "n_block": int(is_block.sum()),
        "n_stepup": int(is_stepup.sum()),
        "n_approve": int(is_approve.sum()),
        "block_precision": block_tp / max(block_tp + block_fp, 1),
        "stepup_rate_of_legit": int((is_stepup & legit).sum()) / max(int(legit.sum()), 1),
        "missed_fraud_rate": approve_fraud / max(int(fraud.sum()), 1),
    }


# =============================================================================
# Small display helpers
# =============================================================================

def mask_user_id(user_id: Any) -> str:
    """Always shows a consistent '••••' mask followed by up to the last 4
    real characters — this dataset's User IDs are short (up to 4 digits
    for 2,000 users), so a length-dependent mask would sometimes render
    with no dots at all for a 4-digit ID, looking unmasked next to rows
    with shorter IDs."""
    s = str(user_id)
    return "••••" + s[-4:]


def confidence_tier(score: Any) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "⚪ N/A"
    if score >= CONF_TIER_HIGH:
        return "🟢 High"
    if score >= CONF_TIER_MODERATE:
        return "🟡 Moderate"
    return "🔴 Low"


def drain_verdicts_topic() -> int:
    """Pulls every currently-queued message off the verdicts topic into
    session state. This is how the dashboard learns about transactions —
    never by calling the model itself."""
    if "verdicts" not in st.session_state:
        st.session_state.verdicts = []
    q = streaming.MockBroker.queue(streaming.VERDICTS_TOPIC)
    drained = 0
    while True:
        try:
            msg = q.get_nowait()
        except Empty:
            break
        st.session_state.verdicts.append(msg.value)
        drained += 1
    return drained


# =============================================================================
# RENDER — the dashboard proper. Reads only st.session_state.verdicts.
# =============================================================================

def render_threshold_panel(res: dict[str, Any]) -> None:
    st.header("⚖️ Routing Threshold — Policy Lever")
    st.caption(
        "Recomputed instantly from precomputed probabilities and confidence scores across the "
        "full 1,827,841-transaction held-out test set — no model re-run needed."
    )

    default_threshold = float(res["threshold"])
    prob_block = st.slider(
        "**Block threshold** — probability required for an automatic BLOCK",
        min_value=0.80, max_value=0.999, value=default_threshold, step=0.001, format="%.3f",
        help=(
            "Raising this: fewer automated blocks. Some transactions that WOULD have been "
            "correctly blocked now fall through to approve instead (since they still pass the "
            "separate confidence bar below) — missed fraud rises. This does NOT change step-up "
            "volume; that's controlled by the confidence threshold below."
        ),
    )
    conf_stepup = st.slider(
        "Step-up confidence threshold — below this, always ask for verification",
        min_value=0.0, max_value=1.0, value=float(s11.DEFAULT_CONF_STEPUP), step=0.01,
        help=(
            "Raising this: MORE transactions get sent to step-up (more customer friction, since "
            "the bar for 'trust the model outright' is higher), but fewer wrong automated calls "
            "slip through. Lowering it: fewer step-ups, more transactions decided automatically."
        ),
    )

    live = recompute_routing(res["prob"], res["confidence_score"], res["position"], res["y_test"], prob_block, conf_stepup)
    baseline = recompute_routing(
        res["prob"], res["confidence_score"], res["position"], res["y_test"],
        default_threshold, s11.DEFAULT_CONF_STEPUP,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Blocks", f"{live['n_block']:,}", delta=f"{live['n_block'] - baseline['n_block']:,}", delta_color="off")
    c2.metric("Step-ups", f"{live['n_stepup']:,}", delta=f"{live['n_stepup'] - baseline['n_stepup']:,}", delta_color="off")
    c3.metric("Approvals", f"{live['n_approve']:,}", delta=f"{live['n_approve'] - baseline['n_approve']:,}", delta_color="off")

    c4, c5, c6 = st.columns(3)
    c4.metric(
        "Block precision", f"{live['block_precision'] * 100:.1f}%",
        delta=f"{(live['block_precision'] - baseline['block_precision']) * 100:+.1f}pp",
    )
    c5.metric(
        "Missed fraud (of all fraud)", f"{live['missed_fraud_rate'] * 100:.2f}%",
        delta=f"{(live['missed_fraud_rate'] - baseline['missed_fraud_rate']) * 100:+.2f}pp",
        delta_color="inverse",
    )
    c6.metric(
        "Step-up rate of legit traffic", f"{live['stepup_rate_of_legit'] * 100:.3f}%",
        delta=f"{(live['stepup_rate_of_legit'] - baseline['stepup_rate_of_legit']) * 100:+.3f}pp",
        delta_color="inverse",
    )
    st.caption(
        f"Baseline (trained threshold {default_threshold:.4f}, default confidence bar "
        f"{s11.DEFAULT_CONF_STEPUP}): block precision {STEP11_REFERENCE_BLOCK_PRECISION * 100:.1f}%, "
        f"step-up rate of legit traffic {STEP11_REFERENCE_STEPUP_RATE_OF_LEGIT * 100:.3f}% "
        f"(step 11's full evaluation)."
    )

    st.markdown("#### Precision – recall tradeoff as the block threshold moves")
    st.caption(
        "Recall = share of ALL fraud caught by block or step-up combined (assumes step-up "
        "verification catches what it flags). Precision = block precision. The confidence "
        "threshold is held fixed at the current step-up slider value while this sweeps the "
        "block threshold — the red dashed line marks where the slider above is right now."
    )
    pr_df = compute_pr_curve(res, conf_stepup)
    pr_long = pr_df.melt(
        id_vars=["threshold", "n_block"], value_vars=["block_precision", "recall"],
        var_name="metric", value_name="rate",
    )
    pr_long["metric"] = pr_long["metric"].map(
        {"block_precision": "Block precision", "recall": "Recall (fraud caught)"}
    )
    line = (
        alt.Chart(pr_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("threshold:Q", title="Block threshold", scale=alt.Scale(domain=[0.80, 1.0])),
            y=alt.Y("rate:Q", title="Rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metric:N", title=None,
                scale=alt.Scale(domain=["Block precision", "Recall (fraud caught)"], range=["#1f77b4", "#9467bd"]),
            ),
            tooltip=["threshold:Q", "metric:N", alt.Tooltip("rate:Q", format=".1%")],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"threshold": [prob_block]}))
        .mark_rule(color="#d62728", strokeDash=[4, 4])
        .encode(x="threshold:Q")
    )
    st.altair_chart((line + marker).properties(height=260), width="stretch")

    with st.expander("Exact values at a few threshold points"):
        sample = pr_df.iloc[:: max(1, len(pr_df) // 6)].copy()
        sample["Block threshold"] = sample["threshold"].map(lambda t: f"{t:.3f}")
        sample["Precision"] = sample["block_precision"].map(lambda v: f"{v * 100:.1f}%")
        sample["Recall"] = sample["recall"].map(lambda v: f"{v * 100:.1f}%")
        sample["# Blocks"] = sample["n_block"]
        st.dataframe(
            sample[["Block threshold", "Precision", "Recall", "# Blocks"]],
            hide_index=True, width="stretch",
        )


def compute_pr_curve(res: dict[str, Any], conf_stepup: float, n_points: int = 25) -> pd.DataFrame:
    """Sweeps the block threshold at a fixed confidence bar, reusing the
    same pure-numpy `recompute_routing` the slider itself calls — no model
    re-run, same reasoning as the live metrics above."""
    thresholds = np.linspace(0.80, 0.999, n_points)
    rows = []
    for t in thresholds:
        r = recompute_routing(res["prob"], res["confidence_score"], res["position"], res["y_test"], t, conf_stepup)
        rows.append({
            "threshold": float(t),
            "block_precision": r["block_precision"],
            "recall": 1.0 - r["missed_fraud_rate"],
            "n_block": r["n_block"],
        })
    return pd.DataFrame(rows)


def render_aggregate_panel(df: pd.DataFrame) -> None:
    st.header("📊 Aggregate Metrics — since dashboard startup")

    total = len(df)
    counts = {d: int((df["decision"] == d).sum()) for d in DECISION_ORDER}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total scored", f"{total:,}")
    c2.metric("Blocked", f"{counts['block']:,} ({counts['block'] / total * 100:.1f}%)")
    c3.metric("Step-up", f"{counts['step_up']:,} ({counts['step_up'] / total * 100:.1f}%)")
    c4.metric("Approved", f"{counts['approve']:,} ({counts['approve'] / total * 100:.1f}%)")

    if "true_label" in df.columns:
        block_df = df[df["decision"] == "block"]
        if len(block_df):
            block_fraud_rate = (block_df["true_label"] == "fraud").mean()
            st.metric(
                "Block precision in this feed (fraud rate among blocks)",
                f"{block_fraud_rate * 100:.1f}%",
                help="Ground truth from the held-out test set these demo transactions are drawn from.",
            )
    st.caption(
        f"Reference — step 11's full 1.8M-transaction evaluation: block precision "
        f"{STEP11_REFERENCE_BLOCK_PRECISION * 100:.1f}%, step-up rate of legitimate traffic "
        f"{STEP11_REFERENCE_STEPUP_RATE_OF_LEGIT * 100:.3f}%."
    )

    ts = df.copy()
    ts["minute"] = ts["scored_at"].dt.floor("min")
    per_minute = ts.groupby(["minute", "decision"]).size().reset_index(name="count")
    chart = (
        alt.Chart(per_minute)
        .mark_bar()
        .encode(
            x=alt.X("minute:T", title="Time"),
            y=alt.Y("count:Q", title="Verdicts / minute"),
            color=alt.Color(
                "decision:N", title="Verdict",
                scale=alt.Scale(domain=DECISION_ORDER, range=DECISION_CHART_COLORS),
            ),
            tooltip=["minute:T", "decision:N", "count:Q"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")


def render_feed_and_detail(df: pd.DataFrame) -> None:
    st.header("📋 Live Transaction Feed")

    view = pd.DataFrame({
        "Time": df["scored_at"].dt.strftime("%H:%M:%S"),
        "User": df["user_id"].apply(mask_user_id),
        "Amount": df["amount"].apply(lambda a: f"${a:,.2f}"),
        "Verdict": df["decision"].map(DECISION_BADGE).fillna(df["decision"]),
        "Confidence": df["confidence_score"].apply(confidence_tier),
    })

    selected_idx = 0
    try:
        event = st.dataframe(
            view, width="stretch", hide_index=True,
            on_select="rerun", selection_mode="single-row", key="feed_table",
        )
        rows = getattr(getattr(event, "selection", None), "rows", None)
        if rows:
            selected_idx = rows[0]
    except TypeError:
        # Older Streamlit without on_select support — fall back to a picker.
        st.dataframe(view, width="stretch", hide_index=True)
        selected_idx = st.selectbox(
            "Select a transaction to inspect", options=list(range(len(df))),
            format_func=lambda i: f"{view.iloc[i]['Time']}  {view.iloc[i]['User']}  {view.iloc[i]['Amount']}  {view.iloc[i]['Verdict']}",
        )

    st.caption("Click a row above to inspect it below (defaults to the most recent transaction).")

    st.header("🔍 Transaction Detail")
    row = df.iloc[selected_idx]

    st.markdown(f"### {DECISION_BADGE.get(row['decision'], row['decision'])}")
    st.caption(f"Rule triggered: `{row['rule']}`")
    st.markdown(f"**Rationale:** {row['rationale']}")

    col_a, col_b = st.columns(2)
    with col_a:
        conf = row["confidence_score"]
        st.metric("Confidence score", f"{conf:.2f}" if pd.notna(conf) else "N/A")
        st.caption(f"Why: {row.get('confidence_reason') or '—'}")
    with col_b:
        st.metric("Amount", f"${row['amount']:,.2f}")
        st.caption(f"User: {mask_user_id(row['user_id'])}  ·  Position: {row.get('position', '—')}")

    st.markdown("#### Top SHAP features")
    top_features = row.get("top_features") or []
    if top_features:
        tf_df = pd.DataFrame(top_features)
        tf_df["Direction"] = tf_df["shap_value"].apply(
            lambda v: "⬆ toward fraud" if v > 0 else "⬇ away from fraud"
        )
        tf_df = tf_df.rename(columns={"feature": "Feature", "shap_value": "SHAP value", "value": "Feature value"})
        st.dataframe(
            tf_df[["Feature", "SHAP value", "Feature value", "Direction"]],
            hide_index=True, width="stretch",
        )
    else:
        st.caption(
            "No SHAP features — this transaction was routed by cold-start policy "
            "(step 9) without invoking the model."
        )

    with st.expander("Raw ranking signal (not a calibrated probability — see step 12)"):
        fsr = row.get("fraud_score_raw")
        if fsr is None or (isinstance(fsr, float) and np.isnan(fsr)):
            st.caption("Not computed for this transaction — cold-start policy, model not invoked.")
        else:
            st.write(f"Raw model output: **{fsr:.4f}**")
            st.caption(
                "This is a RANKING signal, not a fraud probability. Step 12 found a Brier Skill "
                "Score of -1.75 (worse than the naive base-rate baseline) and only 87.3% precision "
                "when this score reads ~0.99. Do not read this as 'X% chance of fraud' — use the "
                "confidence score and routing verdict above for decisions."
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    st.title("🛡️ JANUS — Fraud Detection SOC Dashboard")
    st.caption(
        "Reads scored verdicts from the Kafka `verdicts` topic (in-process mock bus — no broker "
        "reachable in this environment; see step 15). The dashboard never calls the model directly."
    )

    resources = load_scoring_resources()
    bootstrap_demo_stream()

    with st.sidebar:
        st.subheader("Demo controls")
        if st.button("➕ Score 8 more transactions"):
            n = score_additional_batch(resources)
            st.success(f"Produced {n} new verdicts to the `verdicts` topic.")
        st.caption(
            "Simulates steps 14/15's live scoring service producing new messages. The button "
            "itself calls the model (standing in for that service) — the dashboard panels below "
            "never do; they only read what's already on the topic."
        )
        with st.expander("Architecture note"):
            st.write(
                "transactions → [step 14 API + step 15 Redis] → verdicts topic → **this dashboard**\n\n"
                "Kafka: mock in-process bus (no broker reachable). Swap in a real broker by pointing "
                "`step15_streaming.KAFKA_BOOTSTRAP_SERVERS` at one — no other code changes."
            )

    drained = drain_verdicts_topic()
    verdicts = st.session_state.get("verdicts", [])

    if not verdicts:
        st.info("No verdicts yet — waiting on the `verdicts` topic.")
        return

    df = pd.DataFrame(verdicts)
    df["scored_at"] = pd.to_datetime(df["scored_at"])
    df = df.sort_values("scored_at", ascending=False).reset_index(drop=True)

    render_threshold_panel(resources)
    st.divider()
    render_aggregate_panel(df)
    st.divider()
    render_feed_and_detail(df)


if __name__ == "__main__":
    main()
