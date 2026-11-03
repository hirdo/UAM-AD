"""
ART (Ding et al., ASE'24, bbyldebb/ART) self-supervised representation model, ported to plain torch (the authors'
code needs DGL 0.9 / torch 1.12). Architecture follows models/unified_representation/layers.py:
  Transformer encoder (over the instance axis) -> GRU over the 5 history steps -> 2-layer graph convolution
  (GraphConv with norm='none': sum of in-neighbour features, one weight matrix) -> linear regressor that predicts
  the features of every instance at the next step.

Kept as in the authors' code (including its quirks):
  * the TransformerEncoder is built with the default batch_first=False but is fed [B*5, channel, instance], so its
    attention runs over the B*5 axis, i.e. across the samples of a batch;
  * dropout ("noise_rate") only in the graph encoder.
Changed:
  * DGL GraphConv replaced by a dense adjacency product (same function);
  * the authors keep the last GraphConv in a plain python list, so it is not registered as a module parameter
    (never trained, stays at its random init); here it is registered and trained;
  * instance_dim must be even for the 2 attention heads: the caller pads an all-zero dummy instance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvSum(nn.Module):
    """DGL GraphConv(norm='none'): h_i = sum_{j -> i} x_j W + b (adjacency given densely as adj[j, i])."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, adj_t, x):  # adj_t: [B, S, S] with adj_t[b, i, j] = 1 if edge j -> i
        return self.lin(torch.bmm(adj_t, x))


class Extractor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim,
                 dropout, tf_layers=1, gru_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=tf_in_dim, nhead=num_heads, dim_feedforward=tf_in_dim * 2)
        self.tf = nn.TransformerEncoder(layer, num_layers=tf_layers)
        self.gru = nn.GRU(gnn_in_dim, gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.conv1 = GraphConvSum(gru_hidden_dim, gnn_hidden_dim)
        self.conv2 = GraphConvSum(gnn_hidden_dim, gnn_out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, adj_t, feats):  # feats: [B, series, S, C]
        B, series, S, C = feats.shape
        h = F.leaky_relu(self.tf(feats.permute(0, 1, 3, 2).reshape(-1, C, S)))
        h = h.permute(0, 2, 1).reshape(B, series, S, C).permute(0, 2, 1, 3).reshape(-1, series, C)
        _, h_n = self.gru(h)
        h = F.leaky_relu(h_n[-1]).view(B, S, -1)
        h = self.drop(F.leaky_relu(self.conv1(adj_t, h)))
        h = self.drop(F.leaky_relu(self.conv2(adj_t, h)))
        return h


class AutoRegressor(nn.Module):
    def __init__(self, instance_dim, channel_dim, num_heads=2, tf_layers=1, gnn_hidden_dim=64, gnn_out_dim=32,
                 dropout=0.3, gru_hidden_dim=32, gru_layers=1):
        super().__init__()
        assert gnn_out_dim == gru_hidden_dim, "the regressor reads the graph output with the GRU width"
        self.extractor = Extractor(instance_dim, num_heads, channel_dim, gnn_hidden_dim, gnn_out_dim,
                                   gru_hidden_dim, dropout, tf_layers, gru_layers)
        self.regressor = nn.Linear(gru_hidden_dim, channel_dim)

    def forward(self, adj_t, feats):
        z = self.extractor(adj_t, feats)
        return F.leaky_relu(self.regressor(z))  # [B, S, C]
