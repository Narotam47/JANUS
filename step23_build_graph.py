"""Step 23 — user-merchant-card transaction graph for GraphSAGE.

Builds a heterogeneous PyTorch Geometric graph capturing relationships a
per-transaction/per-user model structurally cannot see: which merchants
are shared across users, which cards funnel through which merchants.

Schema (torch_geometric.data.HeteroData)
-----------------------------------------
Nodes:
  user      — one per unique User in the split. Features: [tx_count,
              fraud_count, fraud_rate, mean_amount, std_amount].
  merchant  — one per unique raw Merchant Name (the hashed merchant ID
              from raw_subset.parquet, NOT the frequency-encoded float
              used by the tabular model). Features: [tx_count,
              fraud_count, fraud_rate, mean_amount, unique_user_count].
  card      — one per unique (User, Card) pair. raw_subset's `Card` field
              is a small integer scoped PER USER (card "0" for user A and
              card "0" for user B are different physical cards) — node
              identity is the composite "{user}_{card}" key. Features:
              [tx_count, fraud_count, fraud_rate, mean_amount].

Edges:
  (user, owns, card)         — static, one edge per (user, card) pair seen.
  (user, transacts, merchant) — one edge per transaction. edge_attr =
              [amount_log, hour_of_day, first_mcc, is_new_merchant,
              user_amount_zscore, days_since_last_tx] (NaN -> 0.0, see
              below). edge_label = fraud (0/1), kept SEPARATE from
              edge_attr so a downstream supervised GNN doesn't
              accidentally train on/leak its own target.
  (card, used_at, merchant)  — one edge per transaction, mirroring the
              same event from the card's perspective. Same edge_attr/
              edge_label as the corresponding transacts edge.

Enriched features (user_amount_zscore, first_mcc, is_new_merchant,
days_since_last_tx) are pulled from steps 5/7's ALREADY-COMPUTED,
already-leakage-checked per-transaction outputs — not recomputed here.
They're aligned to this script's own raw_subset split by reproducing
step 2's exact parse -> sort -> chrono_split procedure (imported directly
from step2_preprocess_data, not reimplemented), which guarantees row-for-
row correspondence with clean_train/test.parquet and therefore with
user_features_*.parquet / flags_*.parquet — the same alignment principle
every prior step in this project relies on.

NO LEAKAGE: train and test graphs are built from disjoint transaction sets
(the same chronological per-user 80/20 split as everywhere else in this
project) and have entirely separate node feature computations — a
merchant's "fraud_rate" in the test graph is computed from TEST
transactions only, never contaminated by train statistics.

HONEST CONTEXT ON RING STRUCTURE: see the printed "RING STRUCTURE HONESTY
CHECK" section and the corresponding JSON field. TabFormer is a synthetic
generator; there is no guarantee it encodes genuine coordinated multi-user
fraud rings. This script reports what the graph actually shows (fraud-edge
clustering by merchant) rather than assuming a GNN will find ring
structure just because the graph exists.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from step2_preprocess_data import CHRONO_SPLIT_FRAC, USER_COL, build_timestamp, chrono_split, parse_amount

# --- Config -------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
RAW_SUBSET: Final[Path] = ARTIFACTS_DIR / "raw_subset.parquet"
CLEAN_TRAIN: Final[Path] = ARTIFACTS_DIR / "clean_train.parquet"
CLEAN_TEST: Final[Path] = ARTIFACTS_DIR / "clean_test.parquet"
USER_FEATURES_TRAIN: Final[Path] = ARTIFACTS_DIR / "user_features_train.parquet"
USER_FEATURES_TEST: Final[Path] = ARTIFACTS_DIR / "user_features_test.parquet"
FLAGS_TRAIN: Final[Path] = ARTIFACTS_DIR / "flags_train.parquet"
FLAGS_TEST: Final[Path] = ARTIFACTS_DIR / "flags_test.parquet"

GRAPH_TRAIN_PATH: Final[Path] = ARTIFACTS_DIR / "graph_train.pt"
GRAPH_TEST_PATH: Final[Path] = ARTIFACTS_DIR / "graph_test.pt"
REPORT_PATH: Final[Path] = ARTIFACTS_DIR / "graph_stats.json"

MERCHANT_COL: Final[str] = "Merchant Name"
CARD_COL: Final[str] = "Card"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step23")


# === Split construction (reproduces step 2 exactly, for row alignment) ========

def load_and_split_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading %s ...", RAW_SUBSET)
    df = pd.read_parquet(RAW_SUBSET)
    log.info("Loaded %s rows.", f"{len(df):,}")

    df["amount_float"] = parse_amount(df["Amount"])
    df["timestamp"] = build_timestamp(df)
    df["label"] = (df["Is Fraud?"] == "Yes").astype(np.int8)
    df["card_key"] = df[USER_COL].astype(str) + "_" + df[CARD_COL].astype(str)

    df = df.sort_values([USER_COL, "timestamp"]).reset_index(drop=True)
    train_df, test_df = chrono_split(df, CHRONO_SPLIT_FRAC)
    log.info("Chronological split: train=%s rows, test=%s rows (matches clean_train/test.parquet "
              "row counts and order by construction — same parse/sort/split as step 2).",
              f"{len(train_df):,}", f"{len(test_df):,}")
    return train_df, test_df


def attach_enriched_features(df: pd.DataFrame, user_feat_path: Path, flags_path: Path) -> pd.DataFrame:
    """Attaches a handful of already-computed, already-leakage-checked
    enriched features by ROW POSITION (df here has the exact same row
    order as the corresponding clean_{split}.parquet by construction)."""
    user_feats = pd.read_parquet(user_feat_path, columns=["user_amount_zscore"])
    flags = pd.read_parquet(flags_path, columns=["first_mcc", "days_since_last_tx"])
    assert len(user_feats) == len(df) == len(flags), (
        f"Row count mismatch: raw split={len(df)}, user_feats={len(user_feats)}, flags={len(flags)} "
        "— split reproduction did not match step 2/5/7's row order."
    )
    df = df.reset_index(drop=True)
    df["user_amount_zscore"] = user_feats["user_amount_zscore"].values
    df["first_mcc"] = flags["first_mcc"].values
    df["days_since_last_tx"] = flags["days_since_last_tx"].values
    # is_new_merchant isn't in the trimmed user_feats read above (only pulled
    # user_amount_zscore to keep the read light) — pull it separately.
    is_new = pd.read_parquet(user_feat_path, columns=["is_new_merchant"])
    df["is_new_merchant"] = is_new["is_new_merchant"].values
    return df


# === Graph construction ========================================================

def build_hetero_graph(df: pd.DataFrame, split_name: str) -> tuple[HeteroData, dict]:
    log.info("[%s] Building graph from %s transactions ...", split_name, f"{len(df):,}")

    # --- Node ID mappings (factorize gives dense 0..n-1 indices) ---
    user_codes, user_ids = pd.factorize(df[USER_COL])
    merchant_codes, merchant_ids = pd.factorize(df[MERCHANT_COL])
    card_codes, card_ids = pd.factorize(df["card_key"])
    df = df.assign(_user_idx=user_codes, _merchant_idx=merchant_codes, _card_idx=card_codes)

    n_users, n_merchants, n_cards = len(user_ids), len(merchant_ids), len(card_ids)
    log.info("[%s] Nodes: %s users, %s merchants, %s cards.",
              split_name, f"{n_users:,}", f"{n_merchants:,}", f"{n_cards:,}")

    # --- Node features: per-entity aggregate stats, THIS SPLIT ONLY ---
    def _agg(group_col: str, extra: dict | None = None) -> pd.DataFrame:
        g = df.groupby(group_col)
        out = g["label"].agg(tx_count="count", fraud_count="sum").reset_index()
        out["fraud_rate"] = out["fraud_count"] / out["tx_count"].clip(lower=1)
        amt = g["amount_float"].agg(mean_amount="mean", std_amount="std").reset_index(drop=True)
        out = pd.concat([out, amt], axis=1)
        out["std_amount"] = out["std_amount"].fillna(0.0)
        return out

    user_agg = _agg("_user_idx").sort_values("_user_idx")
    user_x = torch.tensor(
        user_agg[["tx_count", "fraud_count", "fraud_rate", "mean_amount", "std_amount"]].values,
        dtype=torch.float32,
    )

    merch_agg = _agg("_merchant_idx").sort_values("_merchant_idx")
    merch_unique_users = df.groupby("_merchant_idx")["_user_idx"].nunique().sort_index()
    merch_x = torch.tensor(
        np.column_stack([
            merch_agg[["tx_count", "fraud_count", "fraud_rate", "mean_amount"]].values,
            merch_unique_users.values,
        ]),
        dtype=torch.float32,
    )

    card_agg = _agg("_card_idx").sort_values("_card_idx")
    card_x = torch.tensor(
        card_agg[["tx_count", "fraud_count", "fraud_rate", "mean_amount"]].values,
        dtype=torch.float32,
    )

    # --- Edges ---
    # (user, owns, card) — static, one per unique pair.
    owns_pairs = df[["_user_idx", "_card_idx"]].drop_duplicates()
    owns_edge_index = torch.tensor(owns_pairs.values.T, dtype=torch.long)

    # (user, transacts, merchant) and (card, used_at, merchant) — one per transaction.
    amount_log = np.sign(df["amount_float"]) * np.log1p(np.abs(df["amount_float"]))
    hour_of_day = df["timestamp"].dt.hour.astype(np.float32)
    edge_attr_np = np.column_stack([
        amount_log.astype(np.float32),
        hour_of_day,
        df["first_mcc"].fillna(0.0).astype(np.float32),
        df["is_new_merchant"].fillna(0.0).astype(np.float32),
        df["user_amount_zscore"].fillna(0.0).astype(np.float32),
        df["days_since_last_tx"].fillna(0.0).astype(np.float32),
    ])
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
    edge_label = torch.tensor(df["label"].values, dtype=torch.long)

    transacts_edge_index = torch.tensor(df[["_user_idx", "_merchant_idx"]].values.T, dtype=torch.long)
    used_at_edge_index = torch.tensor(df[["_card_idx", "_merchant_idx"]].values.T, dtype=torch.long)

    data = HeteroData()
    data["user"].x = user_x
    data["user"].node_id = list(user_ids)
    data["merchant"].x = merch_x
    data["merchant"].node_id = list(merchant_ids)
    data["card"].x = card_x
    data["card"].node_id = list(card_ids)

    data["user", "owns", "card"].edge_index = owns_edge_index
    data["user", "transacts", "merchant"].edge_index = transacts_edge_index
    data["user", "transacts", "merchant"].edge_attr = edge_attr
    data["user", "transacts", "merchant"].edge_label = edge_label
    data["card", "used_at", "merchant"].edge_index = used_at_edge_index
    data["card", "used_at", "merchant"].edge_attr = edge_attr.clone()
    data["card", "used_at", "merchant"].edge_label = edge_label.clone()

    stats = analyze_graph(df, n_users, n_merchants, n_cards, split_name)
    return data, stats


def analyze_graph(df: pd.DataFrame, n_users: int, n_merchants: int, n_cards: int, split_name: str) -> dict:
    n_edges = len(df)
    n_fraud_edges = int(df["label"].sum())
    fraud_rate = n_fraud_edges / n_edges
    density = n_edges / (n_users * n_merchants)

    log.info("[%s] Edges: %s transacts (user-merchant), %s used_at (card-merchant), fraud rate %.4f%%",
              split_name, f"{n_edges:,}", f"{n_edges:,}", fraud_rate * 100)
    log.info("[%s] Bipartite density (transacts edges / (users x merchants)): %.6e", split_name, density)

    # --- RING STRUCTURE HONESTY CHECK ---
    # Raw "distinct fraud users per merchant" is dominated by a handful of
    # near-universal merchants that most of the 750 users transact with
    # heavily (a merchant with hundreds of transactions from 700+ users will
    # rack up many distinct fraud users by SHEER VOLUME even with zero
    # coordination). The honest question isn't "how many distinct users had
    # fraud here" — it's "more than base-rate chance would predict, given how
    # much traffic this merchant actually has." That needs a null model:
    # for each (user, merchant) pair with n transactions, the probability
    # that user has >=1 fraud transaction there under i.i.d. draws at the
    # OVERALL base rate is 1-(1-base_rate)^n; summing over all users at a
    # merchant gives the EXPECTED distinct-fraud-user count under "no
    # clustering, just volume." Comparing observed to that, not to zero, is
    # what actually distinguishes coordination from popularity.
    fraud_df = df[df["label"] == 1]
    base_rate = float(df["label"].mean())
    if len(fraud_df) > 0 and base_rate > 0:
        tx_per_user_merchant = df.groupby(["_merchant_idx", "_user_idx"]).size()
        prob_ge1_fraud = 1.0 - (1.0 - base_rate) ** tx_per_user_merchant
        expected_per_merchant = prob_ge1_fraud.groupby(level="_merchant_idx").sum()

        observed_per_merchant = fraud_df.groupby("_merchant_idx")["_user_idx"].nunique()
        merchants_with_fraud = len(observed_per_merchant)
        max_distinct_fraud_users = int(observed_per_merchant.max())
        merchants_multi_user_fraud = int((observed_per_merchant >= 2).sum())
        pct_multi = merchants_multi_user_fraud / merchants_with_fraud * 100

        cmp_df = pd.DataFrame({
            "observed": observed_per_merchant,
            "expected": expected_per_merchant.reindex(observed_per_merchant.index),
        }).fillna(0.0)
        cmp_df["excess"] = cmp_df["observed"] - cmp_df["expected"]
        cmp_df["ratio"] = cmp_df["observed"] / cmp_df["expected"].clip(lower=1e-9)
        # "Meaningfully excessive" = at least 2 more distinct fraud users than
        # a pure-volume null model predicts, AND at least 50% over expectation
        # (guards against tiny-expectation merchants where +1 user is a huge
        # ratio but a trivial absolute effect).
        excess_mask = (cmp_df["excess"] >= 2.0) & (cmp_df["ratio"] >= 1.5)
        n_excess_merchants = int(excess_mask.sum())
        pct_excess = n_excess_merchants / merchants_with_fraud * 100
        ratio_clean = cmp_df["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        mean_ratio = float(ratio_clean.mean())
        median_ratio = float(ratio_clean.median())

        top_excess = cmp_df[excess_mask].sort_values("excess", ascending=False).head(5)

        log.info("[%s] RING STRUCTURE HONESTY CHECK (null-model-adjusted, not raw counts):", split_name)
        log.info("  %d distinct merchants have >=1 fraud transaction.", merchants_with_fraud)
        log.info("  Max distinct fraud-committing USERS at any single merchant: %d (RAW, unadjusted "
                  "for that merchant's traffic — see below before treating this as ring evidence).",
                  max_distinct_fraud_users)
        log.info("  Merchants with fraud from >=2 DIFFERENT users: %d/%d (%.1f%%) — raw, unadjusted.",
                  merchants_multi_user_fraud, merchants_with_fraud, pct_multi)
        log.info("  Median observed/expected ratio = %.2fx (typical fraud-touched merchant); mean "
                  "= %.2fx (skewed upward by a small-expectation tail — e.g. a merchant expecting "
                  "0.001 fraud users that gets 1 shows a huge ratio for a trivial absolute effect). "
                  "The 'meaningful excess' % below (requiring >=2 absolute excess users AND >=1.5x) "
                  "is the more honest headline number, not this mean.", median_ratio, mean_ratio)
        log.info("  Under a pure-volume null model (each user's fraud independent at the base "
                  "rate, scaled by how many times they actually visit that merchant): mean "
                  "observed/expected ratio = %.2fx across fraud-touched merchants.", mean_ratio)
        log.info("  Merchants with MEANINGFUL excess (>=2 more distinct fraud users than the null "
                  "model predicts, AND >=1.5x expected): %d/%d (%.1f%%).",
                  n_excess_merchants, merchants_with_fraud, pct_excess)
        if n_excess_merchants > 0:
            log.info("  Top excess merchants (observed vs. null-model-expected distinct fraud users):")
            for m_idx, row in top_excess.iterrows():
                log.info("    merchant_idx=%s  observed=%.0f  expected=%.2f  excess=%+.2f  ratio=%.2fx",
                          m_idx, row["observed"], row["expected"], row["excess"], row["ratio"])

        if pct_excess < 5:
            ring_verdict = "no_evidence"
            log.info("  VERDICT: NO EVIDENCE of coordinated multi-user fraud rings once merchant "
                    "popularity is controlled for. Only %.1f%% of fraud-touched merchants show "
                    "more distinct fraud users than pure volume would predict — the raw '%d "
                    "distinct users at one merchant' number is a POPULARITY artifact (that "
                    "merchant sees traffic from the large majority of all 750 users), not "
                    "coordination. Consistent with TabFormer generating fraud per-user/per-card "
                    "independently. A GNN's structural advantage over shared merchants has little "
                    "genuine ring signal to exploit here.", pct_excess, max_distinct_fraud_users)
        elif pct_excess < 20:
            ring_verdict = "weak_evidence"
            log.info("  VERDICT: WEAK evidence of real (non-popularity) clustering. %.1f%% of "
                    "fraud-touched merchants show more distinct fraud users than a volume-only "
                    "null model predicts — a minority pattern, worth a closer look but not "
                    "grounds to assume a GNN will find strong ring structure.", pct_excess)
        else:
            ring_verdict = "notable_evidence"
            log.info("  VERDICT: NOTABLE excess clustering beyond what merchant popularity alone "
                    "explains. %.1f%% of fraud-touched merchants show meaningfully more distinct "
                    "fraud users than the null model predicts — genuine multi-user co-occurrence, "
                    "worth a GNN's structural attention.", pct_excess)
    else:
        merchants_with_fraud = max_distinct_fraud_users = merchants_multi_user_fraud = 0
        pct_multi = pct_excess = mean_ratio = median_ratio = n_excess_merchants = 0
        ring_verdict = "no_fraud_in_split"
        log.warning("[%s] No fraud transactions in this split — ring-structure check not applicable.", split_name)

    return {
        "n_users": n_users, "n_merchants": n_merchants, "n_cards": n_cards,
        "n_transacts_edges": n_edges, "n_fraud_edges": n_fraud_edges, "fraud_edge_rate": round(fraud_rate, 6),
        "bipartite_density": density,
        "ring_structure_check": {
            "merchants_with_fraud": merchants_with_fraud,
            "max_distinct_fraud_users_at_one_merchant_raw": max_distinct_fraud_users,
            "merchants_with_multi_user_fraud_raw": merchants_multi_user_fraud,
            "pct_fraud_merchants_multi_user_raw": round(pct_multi, 2),
            "null_model": "expected distinct fraud users per merchant under i.i.d. base-rate "
                          "fraud, scaled by each user's actual visit count at that merchant",
            "mean_observed_over_expected_ratio": None if isinstance(mean_ratio, int) or np.isnan(mean_ratio) else round(mean_ratio, 3),
            "median_observed_over_expected_ratio": None if isinstance(median_ratio, int) or np.isnan(median_ratio) else round(median_ratio, 3),
            "merchants_with_meaningful_excess": n_excess_merchants,
            "pct_merchants_with_meaningful_excess": round(pct_excess, 2) if not isinstance(pct_excess, int) else pct_excess,
            "verdict": ring_verdict,
        },
    }


# === Main =====================================================================

def main() -> None:
    t_start = time.perf_counter()
    log.info("=" * 70)
    log.info("STEP 23 — BUILDING USER-MERCHANT-CARD TRANSACTION GRAPH")
    log.info("=" * 70)

    for p in (RAW_SUBSET, CLEAN_TRAIN, CLEAN_TEST, USER_FEATURES_TRAIN, USER_FEATURES_TEST, FLAGS_TRAIN, FLAGS_TEST):
        if not p.exists():
            log.error("Missing required artifact: %s", p)
            return

    train_df, test_df = load_and_split_raw()

    log.info("-" * 70)
    log.info("Attaching enriched features (user_amount_zscore, first_mcc, is_new_merchant, "
              "days_since_last_tx) — reused from steps 5/7, not recomputed ...")
    train_df = attach_enriched_features(train_df, USER_FEATURES_TRAIN, FLAGS_TRAIN)
    test_df = attach_enriched_features(test_df, USER_FEATURES_TEST, FLAGS_TEST)

    all_stats: dict[str, dict] = {}
    for split_name, split_df, out_path in [("train", train_df, GRAPH_TRAIN_PATH), ("test", test_df, GRAPH_TEST_PATH)]:
        log.info("=" * 70)
        graph, stats = build_hetero_graph(split_df, split_name)
        all_stats[split_name] = stats
        torch.save(graph, str(out_path))
        log.info("[%s] Saved: %s", split_name, out_path)
        del graph, split_df

    total_runtime = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("%-8s %8s %10s %8s %14s %14s %10s", "Split", "Users", "Merchants", "Cards", "Edges", "FraudEdges", "Density")
    for name in ("train", "test"):
        s = all_stats[name]
        log.info("%-8s %8s %10s %8s %14s %14s %10.2e",
                  name, f"{s['n_users']:,}", f"{s['n_merchants']:,}", f"{s['n_cards']:,}",
                  f"{s['n_transacts_edges']:,}", f"{s['n_fraud_edges']:,}", s["bipartite_density"])
    log.info("Total runtime: %.1fs", total_runtime)

    output = {
        "description": "User-merchant-card heterogeneous transaction graph for GraphSAGE "
                        "(step 23). Train and test graphs built from disjoint transaction sets "
                        "(same chronological per-user 80/20 split as the rest of the project).",
        "schema": {
            "node_types": ["user", "merchant", "card"],
            "edge_types": [
                ["user", "owns", "card"],
                ["user", "transacts", "merchant"],
                ["card", "used_at", "merchant"],
            ],
            "transacts_edge_attr_columns": [
                "amount_log", "hour_of_day", "first_mcc", "is_new_merchant",
                "user_amount_zscore", "days_since_last_tx",
            ],
        },
        "train": all_stats["train"],
        "test": all_stats["test"],
        "runtime_seconds": round(total_runtime, 2),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", REPORT_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
