"""Step 24 — GraphSAGE for transaction fraud classification.

TASK FRAMING: the label lives on the (user, transacts, merchant) EDGE, not
on a node — this is edge/link classification via node embeddings (a
GraphSAGE encoder produces user/merchant embeddings through message
passing, then a small classifier head scores each labeled transaction
edge from its two endpoint embeddings + edge features), not node
classification in the literal sense. This is the standard construction
for "is this transaction fraudulent" on a bipartite/heterogeneous graph.

LOADER CHOICE: PyG's `LinkNeighborLoader` is used, not the plain
`NeighborLoader`. Both share the same underlying neighbor-sampling
mechanism the task asks for; `LinkNeighborLoader` is the sibling built
specifically for edge-labeled mini-batches (it samples a batch of labeled
edges, attaches each edge's own k-hop neighborhood, and hands back the
induced subgraph) — `NeighborLoader` alone only samples node
neighborhoods, with no first-class way to attach edge labels for a
specific edge type. Negative sampling is explicitly disabled
(`neg_sampling=None`): this is edge CLASSIFICATION of real, existing
transactions with real 0/1 fraud labels, not link PREDICTION of whether
an edge exists.

LEAKAGE FIX ON STEP 23's SAVED FEATURES (found and fixed here, not
upstream): `graph_train.pt`/`graph_test.pt`'s user/merchant/card node
features include `fraud_count` and `fraud_rate` — aggregates computed
FROM the same within-split labels this model predicts, including each
edge's own label contributing to its own node's aggregate. Feeding a
user's own fraud_rate into the encoder as an input feature would leak the
target through node aggregation, the graph-structured analog of the
exact mistake steps 5-7's `shift(1).expanding()` were built to avoid.
Fixed here by slicing those two columns out of every node type's `.x`
before use — not by rebuilding the graph files (the columns are harmless
to have stored, just must never reach the model as input).

CLASS IMBALANCE: BCEWithLogitsLoss with `pos_weight` = the train split's
negative:positive ratio (~794, the same ratio steps 8's `scale_pos_weight`
used) — the direct neural-net analog of XGBoost's class weighting.
Training batches additionally use ALL positive edges plus a random
subsample of negatives per epoch (documented explicitly below) — training
on the full 7.3M edges per epoch on CPU is not tractable in a reasonable
session; this is a disclosed sampling compromise, not silent scope creep.
Evaluation uses the FULL, un-subsampled test set (1,834,571 edges) — the
AUC-PR headline number is not subject to this compromise.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero

# --- Config -------------------------------------------------------------
ARTIFACTS_DIR: Final[Path] = Path("artifacts")
GRAPH_TRAIN_PATH: Final[Path] = ARTIFACTS_DIR / "graph_train.pt"
GRAPH_TEST_PATH: Final[Path] = ARTIFACTS_DIR / "graph_test.pt"
METRICS_ENRICHED: Final[Path] = ARTIFACTS_DIR / "metrics_enriched.json"

MODEL_OUT_PATH: Final[Path] = ARTIFACTS_DIR / "model_gnn.pt"
METRICS_OUT_PATH: Final[Path] = ARTIFACTS_DIR / "metrics_gnn.json"

EDGE_TYPE: Final[tuple[str, str, str]] = ("user", "transacts", "merchant")

# Columns [1,2] = fraud_count, fraud_rate in every node type's saved .x —
# see module docstring. Excluded from GNN input on all three node types.
LEAKY_COL_INDICES: Final[list[int]] = [1, 2]

HIDDEN_CHANNELS: Final[int] = 64
NUM_LAYERS: Final[int] = 2
NUM_NEIGHBORS: Final[list[int]] = [15, 10]

NEG_SAMPLE_RATIO_PER_EPOCH: Final[int] = 20  # negatives per positive, per training epoch
TRAIN_BATCH_SIZE: Final[int] = 512
EVAL_BATCH_SIZE: Final[int] = 8192
N_EPOCHS: Final[int] = 6
LEARNING_RATE: Final[float] = 1e-3
SEED: Final[int] = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("janus.step24")


# === Model =====================================================================

class SAGEEncoder(nn.Module):
    """Plain, homogeneous-looking SAGEConv stack — made heterogeneous-aware
    via `to_hetero()` at construction time (PyG's standard pattern), so
    each edge type gets its own set of learned weights automatically."""

    def __init__(self, hidden_channels: int, num_layers: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv((-1, -1), hidden_channels) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class EdgeClassifier(nn.Module):
    """Scores a labeled transaction edge from its two endpoint embeddings
    plus the transaction's own edge features."""

    def __init__(self, hidden_channels: int, edge_attr_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_attr_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, z_user: torch.Tensor, z_merchant: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_user, z_merchant, edge_attr], dim=-1)
        return self.mlp(z).squeeze(-1)


class FraudGNN(nn.Module):
    def __init__(self, encoder: nn.Module, edge_attr_dim: int, hidden_channels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = EdgeClassifier(hidden_channels, edge_attr_dim)

    def forward(self, batch) -> torch.Tensor:
        x_dict = self.encoder(batch.x_dict, batch.edge_index_dict)
        store = batch[EDGE_TYPE]
        src, dst = store.edge_label_index
        z_user = x_dict["user"][src]
        z_merchant = x_dict["merchant"][dst]
        # NOTE: store.edge_attr follows the STRUCTURAL sampled edges (every
        # (user,transacts,merchant) edge pulled in during neighborhood
        # expansion — can be far more numerous than the batch's labeled
        # seed edges and has no guaranteed row correspondence with
        # edge_label_index). The seed edges' own features are piggybacked
        # onto `edge_label` instead (columns 1:), which PyG DOES guarantee
        # stays row-aligned with edge_label_index — see combine_label_and_attr.
        combined = store.edge_label
        edge_attr = combined[:, 1:]
        return self.classifier(z_user, z_merchant, edge_attr)


# === Data prep =================================================================

def strip_leaky_columns(data) -> None:
    for node_type in data.node_types:
        x = data[node_type].x
        keep = [i for i in range(x.shape[1]) if i not in LEAKY_COL_INDICES]
        data[node_type].x = x[:, keep].contiguous()
    log.info("Stripped fraud_count/fraud_rate columns from all node types. New dims: %s",
              {nt: data[nt].x.shape[1] for nt in data.node_types})


def normalize_features(train_data, test_data) -> None:
    """Z-score every node type's .x and the (user, transacts, merchant)
    edge_attr, fit on TRAIN only, applied to both splits — standard,
    necessary preprocessing for a neural net, not present for XGBoost
    (tree splits are scale-invariant) but essential here. Raw tx_count/
    mean_amount/etc. span many orders of magnitude across nodes; feeding
    that directly into a Linear layer produces unstable, huge activations
    and cripples training regardless of whether the graph carries useful
    signal — skipping this would make any resulting AUC-PR uninterpretable
    (a broken pipeline, not an honest measurement of the graph approach).
    """
    for node_type in train_data.node_types:
        x_train = train_data[node_type].x
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
        train_data[node_type].x = (x_train - mean) / std
        test_data[node_type].x = (test_data[node_type].x - mean) / std

    attr_train = train_data[EDGE_TYPE].edge_attr
    mean = attr_train.mean(dim=0, keepdim=True)
    std = attr_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_data[EDGE_TYPE].edge_attr = (attr_train - mean) / std
    test_data[EDGE_TYPE].edge_attr = (test_data[EDGE_TYPE].edge_attr - mean) / std
    log.info("Standardized node features (all types) and transacts edge_attr, fit on train only.")


def add_reverse_edges(data) -> None:
    """Step 23's graph stores only the natural transaction direction
    (user->card, user->merchant, card->merchant), so 'user' never appears
    as a destination and never receives a message — `to_hetero`'s traced
    forward pass has nothing to update for it after layer 1, which crashes.
    Standard fix for heterogeneous GNNs: add the reverse of every edge type
    so information propagates both ways (e.g. a merchant's embedding
    should reflect the OTHER users who transacted there, which requires
    user->merchant->user->merchant flow across layers). Reverse edges carry
    only edge_index (structural) — no edge_attr/edge_label duplication,
    since we only ever supervise on the canonical (user, transacts,
    merchant) direction.
    """
    for (src, rel, dst) in list(data.edge_types):
        rev_index = data[src, rel, dst].edge_index.flip(0)
        data[dst, f"rev_{rel}", src].edge_index = rev_index
    log.info("Added reverse edge types for bidirectional message passing: %s",
              [et for et in data.edge_types if et[1].startswith("rev_")])


def make_epoch_edge_selection(edge_label: torch.Tensor, neg_ratio: int, rng: np.random.Generator) -> torch.Tensor:
    """All positive edges + a fresh random subsample of negatives, re-drawn
    each epoch — the documented training-time sampling compromise."""
    pos_idx = torch.where(edge_label == 1)[0]
    neg_idx = torch.where(edge_label == 0)[0]
    n_neg_sample = min(len(neg_idx), len(pos_idx) * neg_ratio)
    chosen_neg = neg_idx[torch.from_numpy(rng.choice(len(neg_idx), size=n_neg_sample, replace=False))]
    combined = torch.cat([pos_idx, chosen_neg])
    return combined[torch.randperm(len(combined))]


# === Main =====================================================================

def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t_start = time.perf_counter()
    tracemalloc.start()

    log.info("=" * 70)
    log.info("STEP 24 — GRAPHSAGE FRAUD CLASSIFICATION")
    log.info("=" * 70)

    for p in (GRAPH_TRAIN_PATH, GRAPH_TEST_PATH, METRICS_ENRICHED):
        if not p.exists():
            log.error("Missing required artifact: %s — run step 23 first.", p)
            return

    device = torch.device("cpu")
    log.info("Device: %s (CPU GraphSAGE on this graph size is expected to be slow — "
              "timing reported honestly below).", device)

    log.info("Loading graphs ...")
    train_data = torch.load(str(GRAPH_TRAIN_PATH), weights_only=False)
    test_data = torch.load(str(GRAPH_TEST_PATH), weights_only=False)
    strip_leaky_columns(train_data)
    strip_leaky_columns(test_data)
    add_reverse_edges(train_data)
    add_reverse_edges(test_data)
    # `node_id` (plain Python lists of raw string IDs, step 23) isn't a
    # tensor — LinkNeighborLoader index-selects every node-store attribute
    # when building sampled subgraphs, which breaks on non-tensor fields.
    # Not needed for training/eval, only human-readable traceability.
    for data in (train_data, test_data):
        for node_type in data.node_types:
            if "node_id" in data[node_type]:
                del data[node_type].node_id
    normalize_features(train_data, test_data)

    train_edge_label = train_data[EDGE_TYPE].edge_label.float()
    test_edge_label = test_data[EDGE_TYPE].edge_label.float()
    n_pos_train = int(train_edge_label.sum())
    n_train = len(train_edge_label)
    pos_weight_value = (n_train - n_pos_train) / max(n_pos_train, 1)
    log.info("Train edges: %s (%s fraud, %.4f%%). pos_weight for BCEWithLogitsLoss: %.2f "
              "(same neg:pos ratio concept as step 8's scale_pos_weight=794).",
              f"{n_train:,}", f"{n_pos_train:,}", train_edge_label.mean().item() * 100, pos_weight_value)
    log.info("Test edges: %s (%s fraud, %.4f%%) — evaluated in FULL, no subsampling.",
              f"{len(test_edge_label):,}", f"{int(test_edge_label.sum()):,}", test_edge_label.mean().item() * 100)

    # Piggyback each edge's own features onto `edge_label` (col 0 = fraud
    # label, cols 1: = edge_attr) — see FraudGNN.forward's note on why
    # `edge_attr` alone can't be trusted to align with the labeled seed
    # edges under LinkNeighborLoader's sampling.
    train_edge_attr = train_data[EDGE_TYPE].edge_attr
    test_edge_attr = test_data[EDGE_TYPE].edge_attr
    train_label_combined = torch.cat([train_edge_label.unsqueeze(1), train_edge_attr], dim=1)
    test_label_combined = torch.cat([test_edge_label.unsqueeze(1), test_edge_attr], dim=1)

    edge_attr_dim = train_edge_attr.shape[1]
    encoder = SAGEEncoder(HIDDEN_CHANNELS, NUM_LAYERS)
    encoder = to_hetero(encoder, train_data.metadata(), aggr="sum")
    model = FraudGNN(encoder, edge_attr_dim, HIDDEN_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value))

    # --- One-batch timing probe before committing to a full schedule ---
    log.info("-" * 70)
    log.info("Timing a single mini-batch before committing to the full training schedule ...")
    probe_idx = make_epoch_edge_selection(train_edge_label, NEG_SAMPLE_RATIO_PER_EPOCH, rng)
    probe_loader = LinkNeighborLoader(
        train_data, num_neighbors=NUM_NEIGHBORS,
        edge_label_index=(EDGE_TYPE, train_data[EDGE_TYPE].edge_index[:, probe_idx[:TRAIN_BATCH_SIZE]]),
        edge_label=train_label_combined[probe_idx[:TRAIN_BATCH_SIZE]],
        batch_size=TRAIN_BATCH_SIZE, shuffle=False, neg_sampling=None,
    )
    t0 = time.perf_counter()
    probe_batch = next(iter(probe_loader))
    model.train()
    optimizer.zero_grad()
    out = model(probe_batch)
    loss = loss_fn(out, probe_batch[EDGE_TYPE].edge_label[:, 0])
    loss.backward()
    optimizer.step()
    probe_time = time.perf_counter() - t0
    n_batches_per_epoch = int(np.ceil((n_pos_train * (1 + NEG_SAMPLE_RATIO_PER_EPOCH)) / TRAIN_BATCH_SIZE))
    est_epoch_time = probe_time * n_batches_per_epoch
    log.info("  One batch (size %d): %.3fs. Epoch has ~%d batches -> estimated %.1fs/epoch "
              "(%.1f min), x%d epochs = %.1f min total training (rough estimate; actual timed "
              "below).", TRAIN_BATCH_SIZE, probe_time, n_batches_per_epoch, est_epoch_time,
              est_epoch_time / 60, N_EPOCHS, est_epoch_time * N_EPOCHS / 60)

    # === Training ===============================================================
    log.info("-" * 70)
    log.info("Training for %d epochs (all %s fraud edges + %dx random negative subsample, "
              "re-drawn each epoch) ...", N_EPOCHS, f"{n_pos_train:,}", NEG_SAMPLE_RATIO_PER_EPOCH)

    epoch_losses: list[float] = []
    t_train0 = time.perf_counter()
    for epoch in range(1, N_EPOCHS + 1):
        epoch_idx = make_epoch_edge_selection(train_edge_label, NEG_SAMPLE_RATIO_PER_EPOCH, rng)
        loader = LinkNeighborLoader(
            train_data, num_neighbors=NUM_NEIGHBORS,
            edge_label_index=(EDGE_TYPE, train_data[EDGE_TYPE].edge_index[:, epoch_idx]),
            edge_label=train_label_combined[epoch_idx],
            batch_size=TRAIN_BATCH_SIZE, shuffle=True, neg_sampling=None,
        )
        model.train()
        total_loss, n_batches = 0.0, 0
        t_epoch0 = time.perf_counter()
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch[EDGE_TYPE].edge_label[:, 0])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        epoch_time = time.perf_counter() - t_epoch0
        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        log.info("  Epoch %d/%d: %d batches, mean loss=%.4f, time=%.1fs",
                  epoch, N_EPOCHS, n_batches, mean_loss, epoch_time)

    train_time = time.perf_counter() - t_train0
    log.info("Training complete: %.1fs (%.1f min) actual, vs. %.1f min rough estimate.",
              train_time, train_time / 60, est_epoch_time * N_EPOCHS / 60)

    # === Full evaluation on the untouched test set ===============================
    log.info("-" * 70)
    log.info("Evaluating on the FULL test set (%s edges, no subsampling) ...", f"{len(test_edge_label):,}")
    eval_loader = LinkNeighborLoader(
        test_data, num_neighbors=NUM_NEIGHBORS,
        edge_label_index=(EDGE_TYPE, test_data[EDGE_TYPE].edge_index),
        edge_label=test_label_combined,
        batch_size=EVAL_BATCH_SIZE, shuffle=False, neg_sampling=None,
    )
    model.eval()
    all_probs, all_labels = [], []
    t_eval0 = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            out = model(batch)
            all_probs.append(torch.sigmoid(out).numpy())
            all_labels.append(batch[EDGE_TYPE].edge_label[:, 0].numpy())
            if (i + 1) % 50 == 0:
                log.info("  ... evaluated %d batches", i + 1)
    eval_time = time.perf_counter() - t_eval0
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    assert len(probs) == len(test_edge_label), f"Expected {len(test_edge_label)} eval predictions, got {len(probs)}"

    auc_pr = float(average_precision_score(labels, probs))
    log.info("Evaluation complete in %.1fs.", eval_time)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_runtime = time.perf_counter() - t_start

    # === Honest comparison =========================================================
    with open(METRICS_ENRICHED) as f:
        enriched_auc_pr = json.load(f)["headline_metrics"]["AUC_PR"]
    delta = auc_pr - enriched_auc_pr

    log.info("=" * 70)
    log.info("HEADLINE RESULT")
    log.info("  GraphSAGE AUC-PR:        %.6f", auc_pr)
    log.info("  Enriched XGBoost AUC-PR: %.6f", enriched_auc_pr)
    log.info("  Delta (GraphSAGE - XGBoost): %+.6f", delta)
    if delta < -0.02:
        verdict = "underperforms"
        log.info("  VERDICT: GraphSAGE UNDERPERFORMS the tabular enriched model, by a large "
                  "margin (%.4f vs %.4f). Two DISTINCT factors plausibly contribute, and it "
                  "would be dishonest to credit only one:", auc_pr, enriched_auc_pr)
        log.info("  (1) SPARSE NODE FEATURES — likely the dominant factor. This run's user/"
                  "merchant/card nodes carry only 2-3 raw aggregate stats each (tx_count, "
                  "mean_amount, +1 more), vs. the tabular model's 243 features per transaction. "
                  "The task's design brief called for 'existing enriched feature vectors where "
                  "available' — step 23's graph only stored simple count/mean aggregates, not "
                  "the full per-user/merchant step 5/6 profiles. A richer-node-feature variant "
                  "was NOT attempted in this run: doing so now, after seeing this result, would "
                  "risk exactly the 'keep tuning until it wins' pattern this step was told to "
                  "avoid. Reporting it as a disclosed follow-up rather than a silent gap.")
        log.info("  (2) WEAK RING STRUCTURE — a real but secondary factor. Step 23 found only a "
                  "minority (16.3%%/30.0%%) of fraud-touched merchants show meaningful excess "
                  "clustering beyond visit-frequency alone. Even with richer node features, this "
                  "caps how much a graph's STRUCTURAL advantage (vs. per-user tabular features) "
                  "could plausibly add here.")
        log.info("  Net: this result does not cleanly isolate 'graphs don't help for this fraud "
                  "problem' from 'this particular graph was under-featured' — both explanations "
                  "are live, and only (2) was something step 23 already measured directly.")
    elif delta > 0.02:
        verdict = "outperforms"
        log.info("  VERDICT: GraphSAGE OUTPERFORMS the tabular enriched model — proceed to step "
                  "25 fusion.")
    else:
        verdict = "comparable"
        log.info("  VERDICT: GraphSAGE is roughly COMPARABLE to the tabular enriched model "
                  "(within +-0.02 AUC-PR).")

    log.info("-" * 70)
    log.info("Resources: peak traced Python memory %.1f MB. Total runtime %.1fs (%.1f min): "
              "%.1fs training + %.1fs evaluation.",
              peak_mem / 1e6, total_runtime, total_runtime / 60, train_time, eval_time)

    torch.save(model.state_dict(), str(MODEL_OUT_PATH))
    log.info("Saved: %s", MODEL_OUT_PATH)

    output = {
        "description": "GraphSAGE edge classifier for transaction fraud, trained on the step 23 "
                        "heterogeneous user-merchant-card graph.",
        "honest_context": {
            "leakage_fix": "Node features fraud_count/fraud_rate (saved in step 23's graph "
                            "files, aggregated from within-split labels including each edge's "
                            "own label) were EXCLUDED from the GNN's input at load time in this "
                            "script — feeding them in would leak the prediction target through "
                            "node aggregation.",
            "training_sampling": f"Each epoch trains on all {n_pos_train:,} positive edges plus "
                                  f"a fresh random {NEG_SAMPLE_RATIO_PER_EPOCH}x negative "
                                  "subsample (re-drawn per epoch) — not the full 7.3M train "
                                  "edges, which is intractable on CPU in this session. "
                                  "Evaluation uses the FULL, un-subsampled test set.",
            "ring_structure_context": "Step 23 found weak-to-notable but minority evidence of "
                                       "merchant-level fraud clustering beyond popularity (16.3% "
                                       "train / 30.0% test of fraud-touched merchants). A "
                                       "GraphSAGE result close to or below the tabular model is "
                                       "consistent with that finding, not a sign of a broken model.",
            "underperformance_attribution": "Two distinct, un-disentangled factors plausibly "
                                             "explain the large gap: (1) sparse node features — "
                                             "this run's nodes carry only 2-3 raw aggregate stats "
                                             "each vs. the tabular model's 243 features; the task's "
                                             "brief called for richer 'existing enriched feature "
                                             "vectors' than step 23's graph actually stored, and a "
                                             "richer variant was not attempted post-hoc to avoid "
                                             "tuning toward a win; (2) weak ring structure per "
                                             "step 23's own measurement, a real but likely smaller "
                                             "factor. This result does not cleanly isolate which "
                                             "explanation dominates.",
        },
        "model_config": {
            "hidden_channels": HIDDEN_CHANNELS, "num_layers": NUM_LAYERS,
            "num_neighbors": NUM_NEIGHBORS, "n_epochs": N_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
            "neg_sample_ratio_per_epoch": NEG_SAMPLE_RATIO_PER_EPOCH,
            "learning_rate": LEARNING_RATE, "pos_weight": round(pos_weight_value, 2),
        },
        "training": {
            "epoch_losses": [round(l, 6) for l in epoch_losses],
            "train_time_seconds": round(train_time, 2),
            "eval_time_seconds": round(eval_time, 2),
            "total_runtime_seconds": round(total_runtime, 2),
            "peak_memory_mb": round(peak_mem / 1e6, 2),
        },
        "results": {
            "gnn_auc_pr": round(auc_pr, 6),
            "enriched_xgboost_auc_pr": enriched_auc_pr,
            "delta": round(delta, 6),
            "verdict": verdict,
        },
        "outputs": {"model": str(MODEL_OUT_PATH)},
    }
    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved: %s", METRICS_OUT_PATH)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
