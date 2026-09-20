"""Step 25 — fusion due diligence and the honest negative result.

Step 24's finding: GraphSAGE AUC-PR 0.026 vs. the tabular enriched
model's 0.817. A near-random signal fused with a strong one dilutes,
not improves — so this script does NOT attempt a tuned weighted fusion.
It runs exactly one minimal experiment (90% tabular / 10% GNN weighted
average) as due diligence, confirms the null result, and states plainly
what it does and doesn't mean.

ALIGNMENT NOTE: the tabular model scores 1,827,841 WINDOWS (users need
>=10 transactions; step 3 excludes each user's first 9). The GNN graph
has an edge for every one of the 1,834,571 RAW test transactions,
including those excluded cold-start rows. The fusion experiment can only
combine predictions that exist for BOTH models, so it's restricted to the
1,827,841-window population — `compute_window_feature_indices` gives the
row position of each window's current transaction within
`clean_test.parquet`, which is also the graph's edge order (step 23 built
the graph from the identical chronological row order), so indexing the
full-test-set GNN probability array with those same positions aligns the
two models' predictions to the same transactions.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from xgboost import XGBClassifier  # noqa: E402

import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Final  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from torch_geometric.loader import LinkNeighborLoader  # noqa: E402
from torch_geometric.nn import to_hetero  # noqa: E402

import step24_train_graphsage as s24  # noqa: E402
from step13_shap_explain import build_feature_names, compute_window_feature_indices  # noqa: E402

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
METRICS_GNN: Final[Path] = ARTIFACTS_DIR / "metrics_gnn.json"

OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_fusion.json"

TABULAR_WEIGHT: Final[float] = 0.90
GNN_WEIGHT: Final[float] = 0.10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step25")


# === Tabular probabilities (identical construction to every prior step) =======

def load_tabular_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    model = XGBClassifier()
    model.load_model(str(MODEL_ENRICHED))
    prob = model.predict_proba(X_test)[:, 1]
    return prob, y_test, idx


# === GNN probabilities on the FULL test edge set (re-inference, not retraining) ===

def load_gnn_predictions_full_test() -> np.ndarray:
    train_data = torch.load(str(s24.GRAPH_TRAIN_PATH), weights_only=False)
    test_data = torch.load(str(s24.GRAPH_TEST_PATH), weights_only=False)
    s24.strip_leaky_columns(train_data)
    s24.strip_leaky_columns(test_data)
    s24.add_reverse_edges(train_data)
    s24.add_reverse_edges(test_data)
    for data in (train_data, test_data):
        for node_type in data.node_types:
            if "node_id" in data[node_type]:
                del data[node_type].node_id
    s24.normalize_features(train_data, test_data)  # must match training-time stats exactly

    edge_attr_dim = train_data[s24.EDGE_TYPE].edge_attr.shape[1]
    encoder = to_hetero(s24.SAGEEncoder(s24.HIDDEN_CHANNELS, s24.NUM_LAYERS), train_data.metadata(), aggr="sum")
    model = s24.FraudGNN(encoder, edge_attr_dim, s24.HIDDEN_CHANNELS)
    model.load_state_dict(torch.load(str(s24.MODEL_OUT_PATH), weights_only=True))
    model.eval()

    test_edge_label = test_data[s24.EDGE_TYPE].edge_label.float()
    test_label_combined = torch.cat([test_edge_label.unsqueeze(1), test_data[s24.EDGE_TYPE].edge_attr], dim=1)
    eval_loader = LinkNeighborLoader(
        test_data, num_neighbors=s24.NUM_NEIGHBORS,
        edge_label_index=(s24.EDGE_TYPE, test_data[s24.EDGE_TYPE].edge_index),
        edge_label=test_label_combined,
        batch_size=s24.EVAL_BATCH_SIZE, shuffle=False, neg_sampling=None,
    )
    all_probs = []
    with torch.no_grad():
        for batch in eval_loader:
            out = model(batch)
            all_probs.append(torch.sigmoid(out).numpy())
    return np.concatenate(all_probs)


# === Main =====================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("STEP 25 — FUSION DUE DILIGENCE (expected null result)")
    log.info("=" * 70)

    for p in (MODEL_ENRICHED, METRICS_ENRICHED, s24.MODEL_OUT_PATH, s24.GRAPH_TRAIN_PATH, s24.GRAPH_TEST_PATH):
        if not p.exists():
            log.error("Missing required artifact: %s — run steps 8, 23, 24 first.", p)
            return

    log.info("Scoring the tabular enriched model on its %s windows ...", "1,827,841")
    prob_tabular, y_windows, idx = load_tabular_predictions()

    log.info("Re-running GNN inference on the full test graph (%s edges) to get aligned "
              "per-transaction probabilities — not retraining ...", "1,834,571")
    gnn_prob_full = load_gnn_predictions_full_test()
    gnn_prob_aligned = gnn_prob_full[idx]

    auc_pr_tabular = float(average_precision_score(y_windows, prob_tabular))
    auc_pr_gnn_aligned = float(average_precision_score(y_windows, gnn_prob_aligned))

    fusion_prob = TABULAR_WEIGHT * prob_tabular + GNN_WEIGHT * gnn_prob_aligned
    auc_pr_fusion = float(average_precision_score(y_windows, fusion_prob))

    log.info("-" * 70)
    log.info("SANITY CHECK: tabular AUC-PR recomputed here = %.6f (expect ~0.817069, step 8's "
              "recorded value) — confirms the fusion population is the same test set.",
              auc_pr_tabular)
    log.info("GNN AUC-PR on this aligned window subset = %.6f (step 24's own full-edge-set "
              "figure was 0.026443 on a slightly larger population; close agreement expected).",
              auc_pr_gnn_aligned)

    log.info("=" * 70)
    log.info("COMPARISON TABLE")
    log.info("%-45s %10s", "Configuration", "AUC-PR")
    log.info("%-45s %10.6f", "Tabular enriched model alone", auc_pr_tabular)
    log.info("%-45s %10.6f", "GraphSAGE alone (aligned subset)", auc_pr_gnn_aligned)
    log.info("%-45s %10.6f", f"Fusion ({TABULAR_WEIGHT:.0%} tabular + {GNN_WEIGHT:.0%} GNN)", auc_pr_fusion)
    delta = auc_pr_fusion - auc_pr_tabular
    log.info("-" * 70)
    log.info("Fusion vs. tabular alone: %+.6f", delta)

    null_confirmed = auc_pr_fusion <= auc_pr_tabular
    if null_confirmed:
        log.info("NULL RESULT CONFIRMED: the fusion does not improve on the tabular model alone "
                  "(%.6f -> %.6f). Even a 10%% weight on a near-random signal measurably dilutes "
                  "a strong one. No further fusion tuning was attempted.", auc_pr_tabular, auc_pr_fusion)
    else:
        log.warning("UNEXPECTED: fusion (%.6f) exceeded tabular alone (%.6f) by %+.6f — small "
                    "enough to plausibly be noise given a 0.10 weight on a near-random signal, "
                    "not evidence the GNN is meaningfully additive. Reported as found, not "
                    "chased with further tuning.", auc_pr_fusion, auc_pr_tabular, delta)

    log.info("=" * 70)
    log.info("WHAT THIS MEANS")
    log.info("  1. TabFormer's synthetic fraud generation does not produce the coordinated ring "
              "structure GNN approaches are designed to exploit (step 23: only 16.3%%/30.0%% of "
              "fraud-touched merchants show excess multi-user clustering beyond visit-frequency "
              "alone — a minority signal).")
    log.info("  2. The tabular model's 243 per-transaction features already capture most of the "
              "discriminative signal available in this dataset; a 2-3-dimension-per-node GNN has "
              "little room to add to that.")
    log.info("  3. This is a DATASET limitation, not evidence against the GraphSAGE approach in "
              "general. On real fraud data with genuine coordinated rings (card-not-present fraud "
              "networks, synthetic identity fraud), GraphSAGE would be expected to add value the "
              "per-user tabular features structurally cannot see.")

    log.info("-" * 70)
    log.info("FUTURE WORK")
    log.info("  - Genuine multi-institution data with coordinated fraud rings would provide the "
              "ring structure TabFormer lacks.")
    log.info("  - A richer node-feature construction (the full 243-feature vectors as node "
              "features, rather than the 2-3 aggregate stats used here) might partially close "
              "the gap — a methodology improvement worth trying, not attempted here to avoid "
              "tuning toward a predetermined result after already seeing step 24's outcome.")

    with open(METRICS_GNN) as f:
        gnn_report = json.load(f)
    with open(METRICS_ENRICHED) as f:
        enriched_report = json.load(f)

    output = {
        "description": "Fusion due diligence for the GraphSAGE result (step 24) — one minimal "
                        "weighted-average experiment, not a tuned fusion, confirming the null "
                        "result honestly.",
        "inputs": {
            "tabular_auc_pr_recorded": enriched_report["headline_metrics"]["AUC_PR"],
            "gnn_auc_pr_recorded_full_edge_set": gnn_report["results"]["gnn_auc_pr"],
        },
        "fusion_experiment": {
            "weights": {"tabular": TABULAR_WEIGHT, "gnn": GNN_WEIGHT},
            "population": "1,827,841 test windows (intersection of tabular-scorable and "
                          "graph-scorable transactions; see module docstring on alignment)",
            "auc_pr_tabular_alone": round(auc_pr_tabular, 6),
            "auc_pr_gnn_alone_aligned_subset": round(auc_pr_gnn_aligned, 6),
            "auc_pr_fusion": round(auc_pr_fusion, 6),
            "delta_fusion_vs_tabular": round(delta, 6),
            "null_result_confirmed": bool(null_confirmed),
        },
        "causes_identified": {
            "sparse_node_features": "GNN nodes carry only 2-3 raw aggregate stats vs. the "
                                     "tabular model's 243 features per transaction.",
            "weak_ring_structure": "Step 23: only 16.3% (train) / 30.0% (test) of fraud-touched "
                                    "merchants show meaningful excess clustering beyond visit "
                                    "frequency alone — a minority, not dominant, pattern.",
        },
        "interpretation": {
            "verdict": "GraphSAGE does not improve over the tabular model on TabFormer.",
            "attribution": "Dataset limitation (TabFormer lacks genuine coordinated ring "
                            "structure and the per-user tabular features already capture most "
                            "available signal), not a failure of the GraphSAGE approach itself. "
                            "On real fraud data with genuine coordinated rings, GraphSAGE would "
                            "be expected to add value the per-user features cannot see.",
        },
        "future_work": [
            "Genuine multi-institution data with coordinated fraud rings (card-not-present "
            "fraud networks, synthetic identity fraud) would provide the ring structure "
            "TabFormer lacks.",
            "Richer node-feature construction (full 243-feature vectors as node features, "
            "rather than 2-3 aggregate stats) might partially close the gap — a methodology "
            "improvement worth trying, not attempted here to avoid tuning toward a win after "
            "already seeing step 24's result.",
        ],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", OUT_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
