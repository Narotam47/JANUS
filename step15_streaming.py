"""Step 15 — Streaming: real Redis history store + Kafka consumer/producer.

Two things happen here:

TASK 1 — REDIS. `redis_get_user_history()` and `redis_update_user_history()`
implement, with a real `redis.Redis` client, exactly the data structures
step 14's `get_user_history()` docstring specified:
  * a HASH per user for running sums/counts (amount_sum, amount_sumsq,
    amount_max, tx_count, last_timestamp, gap_sum_days, gap_sumsq_days,
    gap_count),
  * HASHes for merchant_visit_counts / mcc_visit_counts,
  * SETs for first-occurrence membership (seen_city_states, seen_channels,
    seen_year_months, seen_years),
  * HASHes for monthly_amount_avg, tracked as sum+count pairs so the
    average can be computed exactly rather than averaging averages,
  * a capped LIST (RPUSH + LTRIM) for the last WINDOW_SIZE-1 (9) raw
    transactions.
These are monkey-patched onto `step14_api.get_user_history` at consumer
start-up (see `run_consumer_loop`), so step 14's FastAPI app uses the real
implementation with zero changes to step14_api.py itself — it was written
against this exact contract for precisely this reason.

A brand-new user has no keys in Redis at all. HGETALL / SMEMBERS / LRANGE
on a missing key all return empty collections in Redis (not an error), so
`redis_get_user_history` naturally produces the same all-zero UserHistory
step 14's mock did — cold-start routing falls out of that for free, no
special-cased branch needed to avoid crashing on a new user.

TASK 2 — KAFKA. A consumer reads raw transactions from a `transactions`
topic, scores each one by calling step 14's FastAPI app in-process (via
Starlette's TestClient — real ASGI dispatch, no network, but the actual
`/score` code path), writes the user's rolling state back to Redis
(closing the loop for the NEXT transaction), and produces a verdict
message to a `verdicts` topic carrying everything step 16's dashboard
needs (decision, rule, rationale, top-3 SHAP features) without another
model call.

`confluent-kafka` was not installed in this environment; `kafka-python`
is (pure Python, no native deps) — see `KafkaPythonProducer` /
`KafkaPythonConsumer` below, a real adapter built against it. No Kafka
broker is reachable here (no local port 9092, Docker daemon not running),
so `make_producer()` / `make_consumer()` probe for one and fall back to an
in-process mock message bus (`MockProducer` / `MockConsumer`) satisfying
the identical `produce()` / `poll()` / `commit()` interface. To go live:
start a real broker (e.g. `docker run -p 9092:9092 apache/kafka`) and
these factories pick it up automatically — no other code changes.
"""

from __future__ import annotations

import json
import logging
import pickle
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from queue import Empty, Queue
from typing import Any, Optional

import numpy as np
import redis
from fastapi.testclient import TestClient

import step14_api as api

# --- Config -------------------------------------------------------------
REDIS_HOST: str = "localhost"
REDIS_PORT: int = 6379
REDIS_DB: int = 0

KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
KAFKA_CONNECT_TIMEOUT_S: float = 1.5
TRANSACTIONS_TOPIC: str = "transactions"
VERDICTS_TOPIC: str = "verdicts"
CONSUMER_GROUP_ID: str = "janus-scoring-consumer"

WINDOW_SIZE: int = api.WINDOW_SIZE  # 10; recent-transactions cap = WINDOW_SIZE - 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step15")


# =============================================================================
# TASK 1 — REDIS: real implementation of step 14's get_user_history() contract
# =============================================================================

def _k_summary(user_id: str) -> str:
    return f"user:{user_id}:summary"


def _k_merchant_counts(user_id: str) -> str:
    return f"user:{user_id}:merchant_counts"


def _k_mcc_counts(user_id: str) -> str:
    return f"user:{user_id}:mcc_counts"


def _k_seen_city_states(user_id: str) -> str:
    return f"user:{user_id}:seen_city_states"


def _k_seen_channels(user_id: str) -> str:
    return f"user:{user_id}:seen_channels"


def _k_seen_year_months(user_id: str) -> str:
    return f"user:{user_id}:seen_year_months"


def _k_seen_years(user_id: str) -> str:
    return f"user:{user_id}:seen_years"


def _k_monthly_sum(user_id: str) -> str:
    return f"user:{user_id}:monthly_sum"


def _k_monthly_count(user_id: str) -> str:
    return f"user:{user_id}:monthly_count"


def _k_recent(user_id: str) -> str:
    return f"user:{user_id}:recent"


def make_redis_client() -> "redis.Redis":
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def redis_get_user_history(r: "redis.Redis", user_id: str) -> api.UserHistory:
    """Real Redis-backed replacement for step 14's mock `get_user_history()`.

    Gracefully returns an all-zero UserHistory for a user with no keys yet
    (new user, first transaction) — see module docstring for why this
    can't crash: missing-key reads just come back empty.
    """
    raw_summary = r.hgetall(_k_summary(user_id))

    def _f(field: str, default: float = 0.0) -> float:
        v = raw_summary.get(field)
        return float(v) if v is not None else default

    tx_count = int(_f("tx_count", 0.0))
    last_ts_raw = raw_summary.get("last_timestamp")
    last_timestamp = datetime.fromisoformat(last_ts_raw) if last_ts_raw else None

    merchant_counts = {k: int(float(v)) for k, v in r.hgetall(_k_merchant_counts(user_id)).items()}
    mcc_counts = {k: int(float(v)) for k, v in r.hgetall(_k_mcc_counts(user_id)).items()}

    monthly_sum = r.hgetall(_k_monthly_sum(user_id))
    monthly_count = r.hgetall(_k_monthly_count(user_id))
    monthly_avg = {
        ym: float(monthly_sum[ym]) / float(monthly_count[ym])
        for ym in monthly_sum
        if float(monthly_count.get(ym, 0)) > 0
    }

    summary = api.UserHistorySummary(
        tx_count=tx_count,
        amount_sum=_f("amount_sum"),
        amount_sumsq=_f("amount_sumsq"),
        amount_max=_f("amount_max"),
        last_timestamp=last_timestamp,
        merchant_visit_counts=merchant_counts,
        mcc_visit_counts=mcc_counts,
        seen_city_states=list(r.smembers(_k_seen_city_states(user_id))),
        seen_channels=list(r.smembers(_k_seen_channels(user_id))),
        seen_year_months=list(r.smembers(_k_seen_year_months(user_id))),
        seen_years=list(r.smembers(_k_seen_years(user_id))),
        gap_sum_days=_f("gap_sum_days"),
        gap_sumsq_days=_f("gap_sumsq_days"),
        gap_count=int(_f("gap_count", 0.0)),
        monthly_amount_avg=monthly_avg,
    )

    recent_raw = r.lrange(_k_recent(user_id), 0, -1)  # oldest-first (see write side, RPUSH)
    recent = [api.TransactionRaw(**json.loads(x)) for x in recent_raw]

    return api.UserHistory(user_id=user_id, summary=summary, recent_transactions=recent)


def redis_update_user_history(r: "redis.Redis", tx: api.TransactionRaw) -> None:
    """Write-back after scoring: updates this user's rolling Redis state so
    the NEXT transaction sees current history. Must be called with the
    state that existed BEFORE this transaction still readable (i.e. call
    this AFTER scoring, never before) — scoring needs the prior state,
    this function advances it.

    Uses a pipeline for round-trip efficiency, not a WATCH/MULTI
    transaction: this consumer processes one user's transactions strictly
    in arrival order (a real deployment partitions the `transactions`
    topic by user_id, so a single partition — and therefore a single
    consumer — owns all of one user's messages), so there's no concurrent
    writer to race against here.
    """
    user_id = tx.user_id
    summary_key = _k_summary(user_id)

    prior_last_ts_raw = r.hget(summary_key, "last_timestamp")
    prior_last_ts = datetime.fromisoformat(prior_last_ts_raw) if prior_last_ts_raw else None
    prior_tx_count = int(r.hget(summary_key, "tx_count") or 0)
    prior_amount_max_raw = r.hget(summary_key, "amount_max")
    new_amount_max = (
        max(float(prior_amount_max_raw), tx.amount) if prior_tx_count > 0 else tx.amount
    )

    ym = f"{tx.timestamp.year}-{tx.timestamp.month:02d}"

    pipe = r.pipeline()
    pipe.hincrbyfloat(summary_key, "amount_sum", tx.amount)
    pipe.hincrbyfloat(summary_key, "amount_sumsq", tx.amount ** 2)
    pipe.hset(summary_key, "amount_max", new_amount_max)
    pipe.hincrby(summary_key, "tx_count", 1)
    pipe.hset(summary_key, "last_timestamp", tx.timestamp.isoformat())

    if prior_last_ts is not None:
        gap_days = (tx.timestamp - prior_last_ts).total_seconds() / 86400.0
        pipe.hincrbyfloat(summary_key, "gap_sum_days", gap_days)
        pipe.hincrbyfloat(summary_key, "gap_sumsq_days", gap_days ** 2)
        pipe.hincrby(summary_key, "gap_count", 1)

    pipe.hincrby(_k_merchant_counts(user_id), tx.merchant_name, 1)
    pipe.hincrby(_k_mcc_counts(user_id), tx.mcc, 1)

    pipe.sadd(_k_seen_city_states(user_id), f"{tx.merchant_city}|{tx.merchant_state}")
    pipe.sadd(_k_seen_channels(user_id), tx.use_chip)
    pipe.sadd(_k_seen_year_months(user_id), ym)
    pipe.sadd(_k_seen_years(user_id), str(tx.timestamp.year))

    pipe.hincrbyfloat(_k_monthly_sum(user_id), ym, tx.amount)
    pipe.hincrby(_k_monthly_count(user_id), ym, 1)

    pipe.rpush(_k_recent(user_id), tx.model_dump_json())
    pipe.ltrim(_k_recent(user_id), -(WINDOW_SIZE - 1), -1)

    pipe.execute()


def clear_user(r: "redis.Redis", user_id: str) -> None:
    """Test/dev helper — deletes all of a user's keys. Not used in prod."""
    keys = [
        _k_summary(user_id), _k_merchant_counts(user_id), _k_mcc_counts(user_id),
        _k_seen_city_states(user_id), _k_seen_channels(user_id),
        _k_seen_year_months(user_id), _k_seen_years(user_id),
        _k_monthly_sum(user_id), _k_monthly_count(user_id), _k_recent(user_id),
    ]
    r.delete(*keys)


# =============================================================================
# TASK 2 — KAFKA: adapters (real kafka-python + in-process mock) and consumer
# =============================================================================

@dataclass
class BrokerMessage:
    topic: str
    key: Optional[str]
    value: dict


class KafkaPythonProducer:
    """Real producer, built on kafka-python. Used automatically when a
    broker answers at KAFKA_BOOTSTRAP_SERVERS."""

    def __init__(self, bootstrap_servers: str) -> None:
        from kafka import KafkaProducer  # local import: optional dependency

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
        )

    def produce(self, topic: str, key: Optional[str], value: dict) -> None:
        self._producer.send(topic, key=key, value=value)

    def flush(self) -> None:
        self._producer.flush()


class KafkaPythonConsumer:
    """Real consumer, built on kafka-python. Used automatically when a
    broker answers at KAFKA_BOOTSTRAP_SERVERS."""

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str) -> None:
        from kafka import KafkaConsumer  # local import: optional dependency

        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k is not None else None,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )

    def poll(self, timeout: float) -> Optional[BrokerMessage]:
        records = self._consumer.poll(timeout_ms=int(timeout * 1000), max_records=1)
        for tp, batch in records.items():
            for rec in batch:
                return BrokerMessage(topic=tp.topic, key=rec.key, value=rec.value)
        return None

    def commit(self) -> None:
        self._consumer.commit()


class MockBroker:
    """Process-local in-memory message bus. One `Queue` per topic, shared
    across every MockProducer/MockConsumer instance in this process — that
    sharing is what lets a test produce messages, then construct a fresh
    MockConsumer and still see them, exactly as a real broker would let a
    fresh consumer with the same group pick up unread messages."""

    _topics: dict[str, "Queue[BrokerMessage]"] = {}

    @classmethod
    def queue(cls, topic: str) -> "Queue[BrokerMessage]":
        return cls._topics.setdefault(topic, Queue())

    @classmethod
    def reset(cls, topic: str) -> None:
        cls._topics[topic] = Queue()


class MockProducer:
    def produce(self, topic: str, key: Optional[str], value: dict) -> None:
        MockBroker.queue(topic).put(BrokerMessage(topic=topic, key=key, value=value))

    def flush(self) -> None:
        pass


class MockConsumer:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    def poll(self, timeout: float) -> Optional[BrokerMessage]:
        try:
            return MockBroker.queue(self.topic).get(timeout=timeout)
        except Empty:
            return None

    def commit(self) -> None:
        pass  # no offsets to track in-process


def _kafka_broker_reachable(bootstrap_servers: str, timeout: float = KAFKA_CONNECT_TIMEOUT_S) -> bool:
    host, _, port_s = bootstrap_servers.partition(":")
    port = int(port_s) if port_s else 9092
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def make_producer() -> Any:
    if _kafka_broker_reachable(KAFKA_BOOTSTRAP_SERVERS):
        log.info("Kafka broker reachable at %s — using real kafka-python producer.",
                  KAFKA_BOOTSTRAP_SERVERS)
        return KafkaPythonProducer(KAFKA_BOOTSTRAP_SERVERS)
    log.warning(
        "No Kafka broker reachable at %s — falling back to an in-process mock message "
        "bus. To go live: start a real broker (e.g. `docker run -p 9092:9092 "
        "apache/kafka`), leave KAFKA_BOOTSTRAP_SERVERS pointed at it, and this "
        "factory picks it up automatically. No other code changes needed — "
        "MockProducer/MockConsumer and KafkaPythonProducer/KafkaPythonConsumer "
        "expose the identical produce()/poll()/commit() interface.",
        KAFKA_BOOTSTRAP_SERVERS,
    )
    return MockProducer()


def make_consumer(topic: str, group_id: str) -> Any:
    if _kafka_broker_reachable(KAFKA_BOOTSTRAP_SERVERS):
        return KafkaPythonConsumer(KAFKA_BOOTSTRAP_SERVERS, topic, group_id)
    return MockConsumer(topic)


# === Per-message processing ===================================================

def process_transaction_message(
    msg_value: dict,
    redis_client: "redis.Redis",
    client: TestClient,
    producer: Any,
) -> dict:
    """One message: transactions topic -> score -> Redis write-back -> verdicts topic.

    Scores by calling step 14's FastAPI app in-process via TestClient (real
    ASGI dispatch through the actual /score route, no network) — this is
    the "calls the step 14 API" option from the task. Swapping in a real
    network call is a one-line change if the API runs as a separate
    service: replace the `client.post(...)` line with e.g.
    `httpx.post(f"{API_BASE_URL}/score", json=tx_fields)`.

    Returns the verdict dict that was produced.
    """
    t0 = time.perf_counter()

    transaction_id = msg_value.get("transaction_id") or str(uuid.uuid4())
    tx_fields = {k: v for k, v in msg_value.items() if k != "transaction_id"}

    t_score0 = time.perf_counter()
    resp = client.post("/score", json=tx_fields)
    scoring_ms = (time.perf_counter() - t_score0) * 1000.0
    resp.raise_for_status()
    result = resp.json()

    # Close the loop: write THIS transaction into Redis so the next one for
    # this user sees updated history. Must happen after scoring — scoring
    # needs the PRIOR state, this call advances it for next time.
    tx = api.TransactionRaw(**tx_fields)
    t_write0 = time.perf_counter()
    redis_update_user_history(redis_client, tx)
    write_ms = (time.perf_counter() - t_write0) * 1000.0

    e2e_ms = (time.perf_counter() - t0) * 1000.0

    verdict = {
        "transaction_id": transaction_id,
        "user_id": tx.user_id,
        "amount": tx.amount,
        "timestamp": tx.timestamp.isoformat(),
        "decision": result["routing_verdict"]["decision"],
        "rule": result["routing_verdict"]["rule"],
        "rationale": result["rationale"],
        # step 16's dashboard needs 5, matching explain()'s own top_n=5 default
        # in step 14's /score — this was never a 5th model call, just an
        # arbitrary truncation here; widened from :3 to the full 5 already computed.
        "top_features": result["top_features"][:5],
        "fraud_score_raw": result["fraud_score_raw"],
        "confidence_score": result["confidence_score"],
        "confidence_reason": result["confidence_reason"],
        "position": result["position"],
        "model_invoked": result["model_invoked"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": {
            "scoring_ms": round(scoring_ms, 4),
            "redis_write_ms": round(write_ms, 4),
            "total_e2e_ms": round(e2e_ms, 4),
        },
    }
    producer.produce(VERDICTS_TOPIC, key=tx.user_id, value=verdict)
    producer.flush()
    return verdict


def run_consumer_loop(
    max_messages: Optional[int] = None,
    poll_timeout: float = 1.0,
    redis_client: Optional["redis.Redis"] = None,
) -> list[dict]:
    """Drains up to `max_messages` from the transactions topic (or runs
    forever if None), scoring and producing a verdict for each. Returns
    the list of verdicts produced, in order.
    """
    r = redis_client or make_redis_client()
    # Route step 14's history lookups through the real Redis-backed
    # implementation. score_transaction() and get_history_endpoint() in
    # step14_api both call the bare name `get_user_history`, resolved from
    # the module's global namespace at CALL time — so rebinding it here
    # takes effect for both, with zero edits to step14_api.py.
    api.get_user_history = lambda user_id: redis_get_user_history(r, user_id)

    consumer = make_consumer(TRANSACTIONS_TOPIC, CONSUMER_GROUP_ID)
    producer = make_producer()

    verdicts: list[dict] = []
    with TestClient(api.app) as client:
        n = 0
        while max_messages is None or n < max_messages:
            msg = consumer.poll(poll_timeout)
            if msg is None:
                if max_messages is not None:
                    log.warning("Poll timed out with %d/%d messages consumed.", n, max_messages)
                    break
                continue
            verdict = process_transaction_message(msg.value, r, client, producer)
            verdicts.append(verdict)
            consumer.commit()
            n += 1
            log.info(
                "consumed txn=%s user=%s position=%d decision=%s e2e=%.3fms",
                verdict["transaction_id"], verdict["user_id"], verdict["position"],
                verdict["decision"], verdict["latency_ms"]["total_e2e_ms"],
            )
    return verdicts


# =============================================================================
# Test: 10 synthetic transactions for one user, verifying Redis state at each step
# =============================================================================

def _build_synthetic_transactions(user_id: str) -> list[dict]:
    """10 transactions engineered to exercise: merchant_user_count
    incrementing, first_mcc firing only once per MCC, first_channel firing
    once per channel, a dormancy gap (is_dormant), and the position-9
    hand-off from cold-start to the main model."""
    base = datetime(2026, 3, 1, 10, 0, 0)
    # NOTE: gaps between consecutive transactions are deliberately UNEQUAL
    # (1h/2h alternating, then one 120h outlier). Identical repeated gaps
    # give gap_std=0, which puts the is_dormant threshold comparison
    # exactly on a floating-point boundary — and Redis's HINCRBYFLOAT
    # round-trips each running sum through a string, introducing tiny
    # precision drift the offline (pure in-memory float64, via pandas)
    # training computation never sees. Varying the gaps keeps every
    # comparison comfortably clear of that boundary.
    specs = [
        # (hour_offset, merchant, mcc,   amount, chip)
        (0,   "M-A", "5411", 20.00, "Chip Transaction"),    # 0: first-ever
        (1,   "M-A", "5411", 25.00, "Chip Transaction"),    # 1: repeat A/5411
        (3,   "M-B", "5812", 30.00, "Chip Transaction"),    # 2: new merchant+mcc
        (4,   "M-A", "5411", 22.00, "Chip Transaction"),    # 3: A again
        (6,   "M-C", "5812", 28.00, "Online Transaction"),  # 4: new merchant, seen mcc, new channel
        (7,   "M-B", "5812", 31.00, "Chip Transaction"),    # 5: B again
        (9,   "M-A", "5999", 200.00, "Chip Transaction"),   # 6: new mcc for A, big amount
        (129, "M-A", "5411", 24.00, "Chip Transaction"),    # 7: +120h gap -> dormancy
        (130, "M-D", "6011", 15.00, "Chip Transaction"),    # 8: new merchant+mcc
        (132, "M-A", "5411", 26.00, "Chip Transaction"),    # 9: position 9 -> main model
    ]
    out = []
    for i, (h, merchant, mcc, amount, chip) in enumerate(specs):
        ts = base + timedelta(hours=h)
        out.append({
            "transaction_id": f"txn-{i}",
            "user_id": user_id,
            "timestamp": ts.isoformat(),
            "amount": amount,
            "use_chip": chip,
            "merchant_name": merchant,
            "merchant_city": "Testville",
            "merchant_state": "NY",
            "zip_code": "10001",
            "mcc": mcc,
            "errors": [],
        })
    return out


def test_streaming_pipeline() -> None:
    """Precise per-step check: before scoring transaction i, fetch the
    CURRENT Redis-backed history and independently verify
    merchant_user_count, first_mcc, first_channel, and days_since_last_tx
    match hand-computed expectations. Then runs the same 10 messages
    through the actual mock-Kafka consumer loop as a plumbing smoke test.
    """
    r = make_redis_client()
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Redis is not reachable at localhost:6379 — this test needs a real "
            "redis-server running (`redis-server` or `brew services start redis`)."
        ) from e

    user_id = "stream-test-user"
    clear_user(r, user_id)
    api.get_user_history = lambda uid: redis_get_user_history(r, uid)

    with open(api.ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    messages = _build_synthetic_transactions(user_id)

    # Hand-computed expectations, index-aligned with `messages`.
    expected_merchant_user_count = [0, 1, 0, 2, 0, 1, 3, 4, 0, 5]
    expected_first_mcc =           [1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
    expected_first_channel =       [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    expected_is_dormant =          [None, None, None, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    log.info("=" * 70)
    log.info("PER-STEP FEATURE VERIFICATION (before each transaction is scored)")
    log.info("%-4s %-8s %-6s %-10s %-9s %-9s %-9s", "i", "merch", "mcc", "merch_cnt", "first_mcc", "1st_chan", "dormant")

    all_ok = True
    for i, msg in enumerate(messages):
        tx_fields = {k: v for k, v in msg.items() if k != "transaction_id"}
        tx = api.TransactionRaw(**tx_fields)

        history = redis_get_user_history(r, user_id)
        assert history.summary.tx_count == i, (
            f"step {i}: expected tx_count={i} before scoring, got {history.summary.tx_count}"
        )

        user_feats = api.compute_user_profile_features(tx, history.summary)
        flag_feats = api.compute_first_occurrence_flags(tx, history.summary)

        merch_ok = int(user_feats["merchant_user_count"]) == expected_merchant_user_count[i]
        mcc_ok = int(flag_feats["first_mcc"]) == expected_first_mcc[i]
        chan_ok = int(flag_feats["first_channel"]) == expected_first_channel[i]

        exp_dorm = expected_is_dormant[i]
        actual_dorm = flag_feats["is_dormant"]
        if exp_dorm is None:
            dorm_ok = actual_dorm != actual_dorm  # NaN check
        else:
            dorm_ok = actual_dorm == exp_dorm

        step_ok = merch_ok and mcc_ok and chan_ok and dorm_ok
        all_ok = all_ok and step_ok

        log.info(
            "%-4d %-8s %-6s %-10s %-9s %-9s %-9s  %s",
            i, msg["merchant_name"], msg["mcc"],
            f"{int(user_feats['merchant_user_count'])}{'OK' if merch_ok else '!='+str(expected_merchant_user_count[i])}",
            f"{int(flag_feats['first_mcc'])}{'' if mcc_ok else '!='+str(expected_first_mcc[i])}",
            f"{int(flag_feats['first_channel'])}{'' if chan_ok else '!='+str(expected_first_channel[i])}",
            f"{actual_dorm}{'' if dorm_ok else '!='+str(exp_dorm)}",
            "OK" if step_ok else "FAIL",
        )

        # Also check the window-block's hours_since_last for this tx against
        # the true wall-clock gap from the last recorded transaction.
        prev_ts = history.recent_transactions[-1].timestamp if history.recent_transactions else None
        clean_row = api.build_clean_row(tx, prev_ts, encoders)
        if prev_ts is not None:
            expected_hours = (tx.timestamp - prev_ts).total_seconds() / 3600.0
            hrs_ok = np.isclose(clean_row["hours_since_last"], expected_hours)
            assert hrs_ok, f"step {i}: hours_since_last mismatch"

        # Advance state exactly as the consumer would after scoring.
        redis_update_user_history(r, tx)

    assert all_ok, "One or more per-step feature checks FAILED — see log above."
    log.info("-" * 70)
    log.info("ALL PER-STEP FEATURE CHECKS PASSED (merchant_user_count, first_mcc, "
              "first_channel, is_dormant, hours_since_last).")

    final_history = redis_get_user_history(r, user_id)
    assert final_history.summary.tx_count == 10
    assert final_history.summary.merchant_visit_counts["M-A"] == 6
    assert final_history.summary.merchant_visit_counts["M-B"] == 2
    assert final_history.summary.mcc_visit_counts["5411"] == 5
    assert len(final_history.recent_transactions) == WINDOW_SIZE - 1
    log.info("Final Redis state: tx_count=%d, M-A visits=%d, recent_transactions=%d",
              final_history.summary.tx_count, final_history.summary.merchant_visit_counts["M-A"],
              len(final_history.recent_transactions))

    # --- plumbing smoke test: same 10 messages through the mock-Kafka consumer ---
    log.info("=" * 70)
    log.info("CONSUMER/PRODUCER PLUMBING SMOKE TEST (mock Kafka)")
    user_id_2 = "stream-test-user-consumer"
    clear_user(r, user_id_2)
    MockBroker.reset(TRANSACTIONS_TOPIC)
    MockBroker.reset(VERDICTS_TOPIC)

    messages_2 = _build_synthetic_transactions(user_id_2)
    producer = make_producer()
    for msg in messages_2:
        producer.produce(TRANSACTIONS_TOPIC, key=msg["user_id"], value=msg)
    producer.flush()

    verdicts = run_consumer_loop(max_messages=10, poll_timeout=2.0, redis_client=r)
    assert len(verdicts) == 10, f"expected 10 verdicts, got {len(verdicts)}"

    positions = [v["position"] for v in verdicts]
    assert positions == list(range(10)), f"positions out of order: {positions}"
    model_invoked_flags = [v["model_invoked"] for v in verdicts]
    assert model_invoked_flags == [False] * 9 + [True], (
        f"expected cold-start for the first 9 and main-model for the 10th, got {model_invoked_flags}"
    )
    assert verdicts[9]["top_features"], "expected SHAP top features once the main model runs"
    assert len(verdicts[9]["top_features"]) <= 5

    verdicts_queue = MockBroker.queue(VERDICTS_TOPIC)
    assert verdicts_queue.qsize() == 10, "expected 10 messages landed on the verdicts topic"

    e2e_latencies = [v["latency_ms"]["total_e2e_ms"] for v in verdicts]
    log.info("Consumer smoke test: 10/10 verdicts produced, positions 0-9 in order, "
              "cold-start for 0-8 and main-model for 9 (as expected).")
    log.info("End-to-end latency (message received -> verdict written), ms: "
              "mean=%.3f  p50=%.3f  p95=%.3f  max=%.3f",
              np.mean(e2e_latencies), np.percentile(e2e_latencies, 50),
              np.percentile(e2e_latencies, 95), np.max(e2e_latencies))
    log.info(
        "Note: this number reflects in-process TestClient dispatch + local Redis "
        "round-trips, not a real network hop to a separate API service or a real "
        "Kafka broker — those would each add their own latency in a live deployment."
    )

    log.info("=" * 70)
    log.info("STEP 15 TEST SUITE PASSED.")


# === Entry point ===============================================================

def main() -> None:
    log.info("Starting streaming consumer loop (Ctrl+C to stop) ...")
    run_consumer_loop(max_messages=None, poll_timeout=1.0)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_streaming_pipeline()
    else:
        main()
