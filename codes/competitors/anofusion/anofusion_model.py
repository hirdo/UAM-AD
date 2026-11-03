"""
AnoFusion (Zhao et al., KDD'23, AIOps-Lab-NKU/AnoFusion) model, ported with the minimum changes needed to run on
torch 2.5 / one GPU. Architecture and hyper-parameters are the authors': GTN (5 layers, 6 edge types) over the
6 mutual-information matrices -> GAT (6 heads) on the 20-step window per channel -> GRU -> linear next-step
prediction of every channel.

Differences from the authors' files (none changes the function that is computed):
  * hard-coded `cuda:1` replaced by the single available device;
  * `GTAggregation.weight` was created with torch.Tensor(...) and never initialised (undefined values);
    it is initialised to zeros here, i.e. a uniform softmax over the edge types;
  * the adjacency tensor A is identical for every sample of a batch (it is the global MI matrix) and the authors'
    GAT_GRU only reads `graph[0]`, so the GTN is evaluated once on a batch of 1 instead of B copies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GTAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, out_channels, in_channels, 1, 1))

    def forward(self, A):
        return torch.sum(A * F.softmax(self.weight, dim=2), dim=2)


class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first=True):
        super().__init__()
        self.first = first
        self.layer1 = GTAggregation(in_channels, out_channels)
        if first:
            self.layer2 = GTAggregation(in_channels, out_channels)

    def forward(self, A, H_=None):
        a = self.layer1(A)
        batch, num_node = a.shape[0], a.shape[-1]
        a = a.reshape(-1, num_node, num_node)
        if self.first:
            b = self.layer2(A).reshape(-1, num_node, num_node)
            H = torch.bmm(a, b)
        else:
            H = torch.bmm(H_.reshape(-1, num_node, num_node), a)
        return H.view(batch, -1, num_node, num_node)


def row_norm(H):
    eye = torch.eye(H.shape[1], device=H.device).unsqueeze(0)
    H = H * (1 - eye) + eye
    deg_inv = torch.sum(H, dim=-1).pow(-1)
    deg_inv[deg_inv == float("inf")] = 0
    return torch.bmm(deg_inv.unsqueeze(-1) * eye, H)


class GTN(nn.Module):
    def __init__(self, edge_types, num_channels, num_layers):
        super().__init__()
        self.num_channels = num_channels
        self.layers = nn.ModuleList([GTLayer(edge_types, num_channels, first=(i == 0)) for i in range(num_layers)])

    def forward(self, A):  # A: (B, N, N, C)
        A = A.unsqueeze(1).permute(0, 1, 4, 2, 3)
        H = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                H = layer(A)
            else:
                H = torch.stack([row_norm(H[:, c]) for c in range(self.num_channels)], dim=1)
                H = layer(A, H)
        return H


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.W = nn.Linear(in_c, out_c, bias=False)
        self.b = nn.Parameter(torch.Tensor(out_c))
        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def project(self, inputs):
        """h and the attention energy h h^T do not depend on the graph (the costly N^3 product)."""
        h = self.W(inputs)
        return h, torch.bmm(h, h.transpose(1, 2))

    def attend(self, h, energy, graph):
        out = energy * graph.unsqueeze(0)
        out = out.masked_fill(out == 0, -1e16)
        return torch.bmm(F.softmax(out, dim=2), h) + self.b

    def forward(self, inputs, graph):
        return self.attend(*self.project(inputs), graph)


class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)
        self.act = nn.LeakyReLU()

    def project(self, flow):
        return [h.project(flow) for h in self.heads]

    def forward(self, flow, graph, shared=None):  # shared: self.project(flow), reused across the edge types
        shared = shared or self.project(flow)
        out = self.act(torch.cat([h.attend(hh, e, graph) for h, (hh, e) in zip(self.heads, shared)], dim=-1))
        return self.act(self.out_att(out, graph))


class GATGRU(nn.Module):
    def __init__(self, window, node_num, num_channel):
        super().__init__()
        self.num_channels = num_channel
        self.gru = nn.GRU(input_size=node_num, hidden_size=node_num, num_layers=1, batch_first=True)
        self.linear = nn.Linear(node_num, node_num)
        self.gat = GATNet(in_c=window, hid_c=node_num, out_c=window, n_heads=6)
        self.soft = nn.Parameter(torch.full((num_channel,), 1.0 / num_channel))

    @staticmethod
    def norm(A):
        eye = torch.eye(A.shape[1], device=A.device).unsqueeze(0)
        A = A * (1 - eye) + eye
        d = torch.sum(A, dim=-1).pow(-0.5)
        d = d.unsqueeze(-1) * eye
        return torch.bmm(torch.bmm(d, A), d)

    def forward(self, X, A):  # X: (B, N, window), A: (1, C, N, N)
        return self.forward_graphs(X, [self.norm(A[:, i])[0] for i in range(self.num_channels)])

    def forward_graphs(self, X, graphs):  # graphs: C normalised (N, N) matrices
        out = 0
        shared = self.gat.project(X)  # the heads' h and h h^T are the same for every edge type
        for i, g in enumerate(graphs):
            out = out + self.gat(X, g, shared) * self.soft[i]
        out = out.permute(0, 2, 1)  # (B, window, N)
        _, hn = self.gru(out)
        return self.linear(hn.squeeze(0))


class AnoFusionNet(nn.Module):
    def __init__(self, node_num, window=20, edge_types=6, dropout=0.2):
        super().__init__()
        self.node_num, self.edge_types = node_num, edge_types
        self.gtn = GTN(edge_types, edge_types, num_layers=5)
        self.gat_gru = GATGRU(window, node_num, edge_types)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):  # X: (B, N, window); A: (N, N, C) shared by all samples
        return self.forward_graphs(X, self.graphs(A))

    def graphs(self, A):
        """GTN output and its normalisation depend on A and the weights only, not on the sample: computed once per
        optimiser step / evaluation pass instead of once per micro-batch (same values, large saving on wide datasets)."""
        H = self.gtn(A.unsqueeze(0))
        return [self.gat_gru.norm(H[:, i])[0] for i in range(self.edge_types)]

    def forward_graphs(self, X, graphs):
        return self.gat_gru.forward_graphs(self.dropout(X), graphs)
