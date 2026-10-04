"""
Async trace branch — Step 4 of the async-trace plan. A second, independent
model, added purely alongside the existing sync trace branch (trace_model_v3.py,
untouched). See docs/preprocess_deeptralog_{en,vi}.md for the sync/async edge
definitions and the structural-fingerprint evidence that motivated this design:
async faults (F01, F02, F13) leave the sync call graph's edge set unchanged
(0% unseen edges) but shift the async message *count* and/or *temporal order*
— exactly the two things TraceModel's decoder was never built to reconstruct
(its structural decoder is a symmetric presence/absence BCE over `trace_adj`,
and its only supervised attribute columns are error_rate/latency_dev).

Reuses TraceEncoder/GATLayer from trace_model_v3.py UNMODIFIED as the encoder
(same "how do these N service nodes relate" building block); only the decoder
differs, and only because the target is different (a directed, weighted
adjacency instead of a binary undirected one).

Expected input feature layout (async_trace_node_features, col index):
    0  async_out_count   log1p(# async edges where this service is the producer/caller)
    1  async_in_count    log1p(# async edges where this service is the consumer/callee)
    2  avg_lag           log1p(mean (child_start - parent_end) in seconds)  (0 if col1==0).
                      log-scaled, not raw seconds -- a rare slow consumer can
                      be orders of magnitude slower than typical (~2.5ms);
                      see preprocess_deeptralog.py's _build_edges for the
                      2026-09-23 diagnosis of what an unscaled column did.

async_msg_count_adj[i, j] = log1p(count of async messages i -> j in this trace).
Directed and NOT symmetrized (unlike the sync branch's adjacency) -- who sends
to whom is exactly the information a "message count anomaly" (F02) needs, and
symmetrizing would have thrown it away before the encoder ever sees it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trace_model_v3 import TraceEncoder


def async_encoder_inputs(x, adj, order_adj, use_order):
    """Encoder inputs of the async graph -> (node input [B,N,C(+N)], attention mask [B,N,N] bool).

    Without the order relation this is the original (x, adj>0). With it:
      * node IDENTITY: a one-hot of the service is appended for every service present in
        the trace (diagonal of order_adj). Async node features are mostly zeros, so
        without it all services that exchange no message are indistinguishable and no
        head can learn which PAIR of services is unusual (DeepTraLog's TEG nodes carry
        the event/operation content, i.e. they are identifiable).
      * a non-leaking mask: complete graph over the present services plus the message
        edges. The precedence relation itself is NOT in the mask -- it is a target to
        predict; with it in the mask, edge-sharing attention lets the decoder copy the
        relation instead of predicting it, which hid unseen relations.
    """
    if not use_order:
        return x, adj > 0
    if order_adj is None:
        raise ValueError("async_order=True but the batch has no async_temporal_order_adj "
                         "(re-run common/preprocess_deeptralog.py, or pass --async_order False)")
    n = adj.shape[-1]
    present = torch.diagonal(order_adj, dim1=-2, dim2=-1) > 0                          # [B, N]
    ident = torch.eye(n, device=x.device).unsqueeze(0) * present.unsqueeze(-1).float()  # [B, N, N]
    mask = (adj > 0) | (present.unsqueeze(2) & present.unsqueeze(1))
    return torch.cat([x, ident], dim=-1), mask


class AsyncTraceModel(nn.Module):
    """Structure (weighted, directed) + attribute autoencoder for the async subgraph.

    Encoder: TraceEncoder (imported, unmodified) -> node embeddings ZV.
    Structural decoder: adj_hat = softplus((ZV @ W_edge) @ ZV^T)  -- continuous,
        non-negative, asymmetric (W_edge is a free H×H matrix, not required
        symmetric). Computed as Linear + batched matmul (same shape of
        computation as TraceModel's `sigmoid(Z @ Z^T)`, just with the extra
        W_edge for asymmetry) rather than nn.Bilinear over a flattened
        [B*N*N] "batch": an earlier version used nn.Bilinear(H,H,1) on
        zi/zj of shape [B*N*N,H], and its backward pass (the weight
        gradient needs an outer product per "batch" row) allocated a
        [B*N*N,H,H] intermediate -- with the real training batch B*W=640
        and N=35, that is 640*35*35*32*32*4 bytes ≈ 3.2 GB, which is what
        actually overflowed the 3 GB GPU in the first Step 4 run
        (2026-09-23, CUDA OOM "tried to allocate 2.99 GiB"). Linear+bmm
        never materialises anything larger than [B,N,N].
        Loss: MSE against the weighted async_msg_count_adj (log1p(count) target),
        not BCE against a binary target -- Bước 2's fingerprint evidence
        (F02: message count +21% with 0% new edges) is a count anomaly, and a
        presence/absence decoder is structurally blind to a count that merely
        goes up or down while staying nonzero.
        Cell weighting: only a handful of the N*N=1225 (service,service)
        pairs ever carry a real async edge across the whole dataset (~5-10
        pairs, e.g. food-service->delivery-service). Averaging the MSE
        uniformly over all 1225 cells dilutes that real signal by ~1225x,
        AND makes the trace-level loss disproportionately sensitive to any
        single misprediction on one of the ~1220 near-always-zero cells --
        diagnosed 2026-09-23 as the reason some normal F02 traces scored as
        *more* anomalous than every real F02 anomaly (one rare-but-legitimate
        edge type the decoder had barely seen in training dominated that
        trace's average). `edge_mask` (from meta.pkl's `async_edge_mask`,
        built from every normal trace, not just the ones downsampling kept)
        marks which cells were EVER seen with an edge; those get full weight,
        everything else gets `unseen_edge_weight` (small, not zero -- a fault
        that routes a message to a service pair never seen normally is a
        real anomaly the model should still be able to flag, just not let
        1220 near-irrelevant cells drown out the ~10 that matter).
    Attribute decoder: MSE over all async_c columns (no BCE columns here,
        unlike TraceModel -- async_out_count/async_in_count/avg_lag are all
        continuous, count-like or duration-like, not rates in [0,1]).
    """

    def __init__(self, device, **kwargs):
        super(AsyncTraceModel, self).__init__()
        self.use_order = bool(kwargs.get("async_order", False))
        async_kwargs = dict(kwargs)
        async_kwargs["trace_c"] = kwargs["async_c"] + (kwargs["num_services"] if self.use_order else 0)
        self.encoder = TraceEncoder(device, **async_kwargs)
        self.async_c = kwargs["async_c"]
        hidden_size = kwargs["hidden_size"]

        self.edge_decoder = nn.Linear(hidden_size, hidden_size, bias=False)  # W_edge
        self.feat_decoder = nn.Linear(hidden_size, self.async_c)

        # Second relation of the async graph: service-level "Sequence" relation
        # (DeepTraLog's TEG Sequence edges): order_adj[i,j]=1 iff service i finished
        # before service j started; order_adj[i,i]=1 marks a service present in the
        # trace (preprocess_deeptralog._order_adj). Reconstructed by its own bilinear
        # head with BCE, on the same node embeddings -- one branch, two relations.
        if self.use_order:
            self.order_decoder = nn.Linear(hidden_size, hidden_size, bias=False)  # W_order

        self.lambda_attr = kwargs.get("lambda_async_attr", 0.5)

        # Cell weighting for the structural loss -- see class docstring.
        # unseen_edge_weight small-but-nonzero (default 0.05): still lets a
        # genuinely novel edge contribute to the score, just not dominate it.
        self.unseen_edge_weight = float(kwargs.get("async_unseen_edge_weight", 0.05))
        edge_mask = kwargs.get("async_edge_mask")
        num_services = kwargs.get("num_services", 0)
        if edge_mask is not None:
            mask_t = torch.as_tensor(edge_mask, dtype=torch.float32)
        else:
            # No mask available (e.g. meta.pkl predates this feature) --
            # weight everything equally, i.e. today's un-weighted behaviour.
            mask_t = torch.ones((num_services, num_services), dtype=torch.float32)
        cell_weight = mask_t * (1.0 - self.unseen_edge_weight) + self.unseen_edge_weight
        self.register_buffer("cell_weight", cell_weight)  # [N, N], moves with .to(device)

    def forward(self, x, adj, order_adj=None):
        """
        Args:
            x:         [B, N, async_c]
            adj:       [B, N, N]  directed, weighted (log1p count) ground-truth
            order_adj: [B, N, N]  optional precedence relation (see __init__)
        Returns: (z, adj_hat, loss, feats_hat, node_scores, loss_order) -- the
            first five follow TraceModel.forward's convention (`loss` = count
            structure + lambda*attribute, unchanged by the order relation);
            loss_order [B] is the precedence-relation reconstruction error, or
            None when there is no order relation.
        """
        use_order = self.use_order
        x_in, mask = async_encoder_inputs(x, adj, order_adj, use_order)
        z = self.encoder(x_in, mask.float())

        # ── Structural decoder: directed, weighted -- Linear+bmm, see class
        # docstring for why this replaced an earlier nn.Bilinear version.
        zw = self.edge_decoder(z)                                     # [B, N, H]
        adj_hat = F.softplus(torch.bmm(zw, z.transpose(1, 2)))         # [B, N, N]
        loss_struct_cell = F.mse_loss(adj_hat, adj, reduction='none') * self.cell_weight  # [B, N, N]
        loss_struct = loss_struct_cell.sum(dim=[-2, -1]) / self.cell_weight.sum()          # [B]
        loss_struct_per_node = loss_struct_cell.sum(dim=-1) / self.cell_weight.sum(dim=-1).clamp(min=1e-6)  # [B, N]

        # ── Attribute decoder ────────────────────────────────────────────────
        feats_hat = self.feat_decoder(z)                              # [B, N, async_c]
        loss_attr_raw = F.mse_loss(feats_hat, x, reduction='none')    # [B, N, async_c]
        loss_attr_per_node = loss_attr_raw.mean(dim=-1)               # [B, N]
        loss_attr = loss_attr_per_node.mean(dim=-1)                   # [B]

        loss = loss_struct + self.lambda_attr * loss_attr
        node_scores = loss_struct_per_node + self.lambda_attr * loss_attr_per_node  # [B, N]

        loss_order = None
        if use_order:
            n = adj.shape[-1]
            eye = torch.eye(n, device=adj.device)
            present = torch.diagonal(order_adj, dim1=-2, dim2=-1) > 0                       # [B, N]
            pair = (present.unsqueeze(2) & present.unsqueeze(1)).float() * (1.0 - eye)      # [B, N, N] present pairs
            logits = torch.bmm(self.order_decoder(z), z.transpose(1, 2))                    # directed
            bce = F.binary_cross_entropy_with_logits(logits, order_adj * (1.0 - eye), reduction='none')
            loss_order = (bce * pair).sum(dim=[-2, -1]) / pair.sum(dim=[-2, -1]).clamp(min=1.0)   # [B]

        return z, adj_hat, loss, feats_hat, node_scores, loss_order
