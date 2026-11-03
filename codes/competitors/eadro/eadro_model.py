"""
Eadro (Lee et al., ICSE'23, BEbillionaireUSD/Eadro) in plain torch (no DGL). The released snapshot does not run
as is (undefined `trace_dropout`/`metric_dropout`, `self.locator` vs `self.localizer`, DGL 0.9 graph objects), so the
model is rewritten from its structure:
  per service node: trace, metric and log branches (each a dilated causal Conv1d + self-attention over the chunk),
  fused by Linear + GLU -> GATv2 over the service graph (4 heads, max over heads) -> global attention pooling ->
  detector (2 classes) and localizer (one class per service); loss = alpha * detect + (1 - alpha) * locate
  (cross-entropy, localizer ignores normal samples).
Differences from the released code: the log branch takes the 2 per-service log features of service_logs.py (log1p of
the number of lines and of distinct lines; original: event-count vector) and has the structure of the metric branch,
the trace branch takes the 6 trace features of a service as channels (original: 1 latency channel), and the service graph
is the per-sample trace_adj instead of one fixed graph. Hyper-parameters are the authors' defaults.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n].contiguous()


class ConvNet(nn.Module):
    def __init__(self, in_ch, hidden, kernel_sizes, dilation=2):
        super().__init__()
        layers = []
        for i, k in enumerate(kernel_sizes):
            d = dilation ** i
            pad = (k - 1) * d
            layers += [nn.Conv1d(in_ch if i == 0 else hidden[i - 1], hidden[i], k, dilation=d, padding=pad),
                       nn.BatchNorm1d(hidden[i]), nn.ReLU(), Chomp1d(pad)]
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden[-1]

    def forward(self, x):  # [N, T, in_ch]
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)  # [N, T, out]


class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.w = nn.Parameter(torch.empty(seq_len, input_size, 1))
        self.b = nn.Parameter(torch.zeros(seq_len, 1, 1))
        std = math.sqrt(6.0 / (input_size + 1))
        self.w.data.uniform_(-std, std)

    def forward(self, x):  # [N, T, D]
        a = (torch.bmm(x.transpose(1, 0), self.w) + self.b).transpose(1, 0).tanh()  # [N, T, 1]
        return torch.bmm(a.transpose(1, 2), x).squeeze(1)  # [N, D]


class Branch(nn.Module):
    def __init__(self, in_ch, hidden, kernel_sizes, chunk_len):
        super().__init__()
        self.conv = ConvNet(in_ch, hidden, kernel_sizes)
        self.attn = SelfAttention(self.conv.out_dim, chunk_len)
        self.out_dim = self.conv.out_dim

    def forward(self, x):
        return self.attn(self.conv(x))


class GATv2Dense(nn.Module):
    """DGL GATv2Conv on a dense adjacency; adj_t[b, i, j] = 1 if j -> i. Returns [B, S, heads, out]."""

    def __init__(self, in_dim, out_dim, heads, slope=0.2):
        super().__init__()
        self.heads, self.out_dim, self.slope = heads, out_dim, slope
        self.fs = nn.Linear(in_dim, heads * out_dim)
        self.fd = nn.Linear(in_dim, heads * out_dim)
        self.attn = nn.Parameter(torch.empty(1, 1, 1, heads, out_dim))
        nn.init.xavier_normal_(self.attn, gain=nn.init.calculate_gain("relu"))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))

    def forward(self, adj_t, h):
        B, S, _ = h.shape
        zs = self.fs(h).view(B, S, self.heads, self.out_dim)
        zd = self.fd(h).view(B, S, self.heads, self.out_dim)
        e = (F.leaky_relu(zd[:, :, None] + zs[:, None, :], self.slope) * self.attn).sum(-1)  # [B, i, j, H]
        e = e.masked_fill(adj_t[..., None] == 0, -1e30)
        a = torch.softmax(e, dim=2)
        out = torch.einsum("bijh,bjhd->bihd", a, zs)
        return out + self.bias.view(1, 1, self.heads, self.out_dim)


class Eadro(nn.Module):
    def __init__(self, node_num, metric_ch, trace_ch, chunk_len, log_ch, fuse_dim=128, trace_hidden=(64,),
                 trace_kernel=(2,), metric_hidden=(64,), metric_kernel=(2,), graph_hidden=64, heads=4,
                 detect_hidden=(64,), locate_hidden=(64,), alpha=0.5):
        super().__init__()
        self.node_num, self.alpha = node_num, alpha
        self.trace = Branch(trace_ch, list(trace_hidden), list(trace_kernel), chunk_len)
        self.metric = Branch(metric_ch, list(metric_hidden), list(metric_kernel), chunk_len)
        self.log = Branch(log_ch, list(metric_hidden), list(metric_kernel), chunk_len)
        self.fuse = nn.Linear(self.trace.out_dim + self.metric.out_dim + self.log.out_dim, fuse_dim)
        self.gat = GATv2Dense(fuse_dim // 2, graph_hidden, heads)
        self.pool_gate = nn.Linear(graph_hidden, 1)
        self.detector = self._mlp(graph_hidden, 2, detect_hidden)
        self.localizer = self._mlp(graph_hidden, node_num, locate_hidden)

    @staticmethod
    def _mlp(i, o, hidden):
        layers = []
        for k, h in enumerate(hidden):
            layers += [nn.Linear(i if k == 0 else hidden[k - 1], h), nn.ReLU()]
        return nn.Sequential(*layers, nn.Linear(hidden[-1], o))

    def forward(self, adj_t, metric, trace, log):  # metric: [B, S, T, Cm]; trace: [B, S, T, Ct]; log: [B, S, T, Cl]
        B, S, T, _ = metric.shape
        t = self.trace(trace.reshape(B * S, T, -1))
        m = self.metric(metric.reshape(B * S, T, -1))
        lg = self.log(log.reshape(B * S, T, -1))
        h = F.glu(self.fuse(torch.cat([t, m, lg], dim=-1))).view(B, S, -1)
        h = self.gat(adj_t, h).max(dim=2).values                   # max over heads -> [B, S, D]
        w = torch.softmax(self.pool_gate(h), dim=1)                # global attention pooling over the nodes
        g = (w * h).sum(1)
        return self.detector(g), self.localizer(g)

    def loss(self, detect_logits, locate_logits, y_anom, y_node):
        loc = F.cross_entropy(locate_logits, y_node, ignore_index=-1) if (y_node >= 0).any() else 0.0
        return self.alpha * F.cross_entropy(detect_logits, y_anom) + (1 - self.alpha) * loc
