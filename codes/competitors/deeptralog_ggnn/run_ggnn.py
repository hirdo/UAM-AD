"""
DeepTraLog GGNN baseline (Zhang et al., ICSE 2022), ported to plain torch (no torch_geometric / torch_scatter).
Model and defaults follow FudanSELab/DeepTraLog model/main.py: GatedGraphConv(300, 3 layers) -> global attention
-> Deep SVDD soft-boundary (nu 0.05, warm-up 10), AdamW lr 1e-4, wd 1e-4, 100 epochs, batch 32, lr x0.1 at 60% of the epochs (epoch 60 for 100),
normal traces split 60/10/30 (train/val/test pool).

Protocol of this thesis (differs from the authors' script only here):
  * the GGNN is trained once per seed on its own normal-train split (unsupervised, normals only);
  * for every F-case the test set is ALL anomalous traces of that F-case plus a random sample of the normal test
    pool of the same size as in UAM-AD's scenarios/test_Fxx.pkl (same anomaly rate);
  * threshold = p95 of the normal val scores; primary metric F1 at that threshold; AUROC/AUPRC/oracle F1 and the
    authors' native threshold (score > 0) are reported next to it.

Usage:
  python run_ggnn.py --npz D:/ClaudeWork/competitors/GraphData/compact.npz --out_dir <dir> --seeds 0 1 2 3 4
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

# F-case name in the thesis -> label prefix inside GraphData's error_trace_type (zip names != internal ids)
FCASES = {"F01": "F01", "F02": "F02", "F13": "F13", "F05": "F05", "F09": "F24"}
# normal traces in UAM-AD's test_Fxx.pkl (so that the anomaly rate matches)
N_NORMAL = {"F01": 14371, "F02": 15057, "F13": 12453, "F05": 7500, "F09": 6846}


class GGNN(nn.Module):
    """GatedGraphConv + GlobalAttention readout (PyG semantics: messages flow edge_index[0] -> edge_index[1])."""

    def __init__(self, in_dim, hidden_dim=300, num_layers=3):
        super().__init__()
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.weight = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        for i in range(num_layers):
            nn.init.xavier_uniform_(self.weight.data[i])
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 200), nn.Tanh(), nn.Linear(200, 1))
        self.pad = hidden_dim - in_dim
        assert self.pad >= 0

    def forward(self, x, edge_index, batch, n_graphs):
        if self.pad:
            x = torch.cat([x, x.new_zeros(x.size(0), self.pad)], dim=1)
        src, dst = edge_index
        for i in range(self.num_layers):
            m = x @ self.weight[i]
            agg = torch.zeros_like(m).index_add_(0, dst, m[src])
            x = self.rnn(agg, x)
        g = self.gate(x).view(-1)
        gmax = torch.full((n_graphs,), -1e30, device=x.device).scatter_reduce(0, batch, g, "amax", include_self=True)
        e = torch.exp(g - gmax[batch])
        den = torch.zeros(n_graphs, device=x.device).index_add_(0, batch, e)
        a = e / den[batch]
        return torch.zeros(n_graphs, self.hidden_dim, device=x.device).index_add_(0, batch, a.unsqueeze(1) * x)


class Graphs:
    def __init__(self, npz, device):
        z = np.load(npz)
        self.nodes, self.edges = z["nodes"].astype(np.int64), z["edges"].astype(np.int64)
        self.n_nodes, self.n_edges = z["n_nodes"], z["n_edges"]
        self.node_off = np.concatenate([[0], np.cumsum(self.n_nodes)])
        self.edge_off = np.concatenate([[0], np.cumsum(self.n_edges)])
        self.is_normal, self.error_type = z["is_normal"], z["error_type"]
        self.emb = torch.tensor(z["event_emb"], device=device)
        self.device = device

    def batch(self, idx):
        nid, ei, bt, off = [], [], [], 0
        for k, i in enumerate(idx):
            nid.append(self.nodes[self.node_off[i]:self.node_off[i + 1]])
            ei.append(self.edges[self.edge_off[i]:self.edge_off[i + 1]] + off)
            bt.append(np.full(self.n_nodes[i], k))
            off += self.n_nodes[i]
        d = self.device
        x = self.emb[torch.from_numpy(np.concatenate(nid)).to(d)]
        e = torch.from_numpy(np.concatenate(ei)).to(d).t().contiguous()
        return x, e, torch.from_numpy(np.concatenate(bt)).to(d), len(idx)


def batches(idx, bs, shuffle, rng):
    idx = np.asarray(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for s in range(0, len(idx), bs):
        yield idx[s:s + bs]


@torch.no_grad()
def embed(model, G, idx, bs):
    model.eval()
    out = []
    for b in batches(idx, bs, False, None):
        out.append(model(*G.batch(b)))
    return torch.cat(out)


def metrics(y, s, thr):
    p = (s > thr).astype(int)
    return dict(f1=f1_score(y, p, zero_division=0), precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0))


def oracle_f1(y, s):
    o = np.argsort(-s)
    tp = np.cumsum(y[o])
    k = np.arange(1, len(y) + 1)
    return float(np.max(2 * tp / (k + y.sum())))


def run_seed(G, seed, args, device):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    normal = np.where(G.is_normal)[0]
    normal = normal[rng.permutation(len(normal))]
    n = len(normal)
    tr, va, te = normal[:int(n * .6)], normal[int(n * .6):int(n * .7)], normal[int(n * .7):]
    if args.max_train:
        tr = tr[:args.max_train]
    model = GGNN(G.emb.shape[1], args.hidden, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[round(0.6 * args.epochs)], gamma=0.1)

    c = embed(model, G, tr, args.batch_size).mean(0)
    c[(c.abs() < 0.1) & (c < 0)] = -0.1
    c[(c.abs() < 0.1) & (c > 0)] = 0.1
    R = torch.tensor(0.0, device=device)
    nu = 0.05
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl, nb, dists = 0.0, 0, []
        for b in batches(tr, args.batch_size, True, rng):
            opt.zero_grad()
            dist = ((model(*G.batch(b)) - c) ** 2).sum(1)
            loss = R ** 2 + (1 / nu) * torch.mean(torch.clamp(dist - R ** 2, min=0))
            loss.backward()
            opt.step()
            tl += loss.item()
            nb += 1
            dists.append(dist.detach())
        if ep >= 10:
            R = torch.tensor(float(np.quantile(torch.cat(dists).sqrt().cpu().numpy(), 1 - nu)), device=device)
        sch.step()
        print(f"seed {seed} epoch {ep + 1}/{args.epochs} loss {tl / nb:.6f} R {R.item():.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    train_time = time.time() - t0

    def score(idx):
        return (((embed(model, G, idx, args.batch_size) - c) ** 2).sum(1) - R ** 2).cpu().numpy()

    thr = float(np.percentile(score(va), 95))
    res = {"seed": seed, "train_time_s": train_time, "threshold_p95_val": thr, "R": R.item(), "fcases": {}}
    for name, prefix in FCASES.items():
        anom = np.where((~G.is_normal) & np.array([str(t).split("-")[0] == prefix for t in G.error_type]))[0]
        nrm = te[rng.permutation(len(te))[:N_NORMAL[name]]]
        idx = np.concatenate([anom, nrm])
        y = np.concatenate([np.ones(len(anom), int), np.zeros(len(nrm), int)])
        t1 = time.time()
        s = score(idx)
        r = metrics(y, s, thr)
        r.update(auroc=float(roc_auc_score(y, s)), auprc=float(average_precision_score(y, s)),
                 oracle_f1=oracle_f1(y, s), native_thr0=metrics(y, s, 0.0),
                 n_anomaly=int(len(anom)), n_normal=int(len(nrm)), infer_time_s=time.time() - t1)
        res["fcases"][name] = r
        print(f"seed {seed} {name} F1@val {r['f1']:.4f} AUROC {r['auroc']:.4f} AUPRC {r['auprc']:.4f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_train", type=int, default=0, help="debug only: cap the number of train graphs")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    G = Graphs(args.npz, device)
    for seed in args.seeds:
        p = os.path.join(args.out_dir, f"seed{seed}.json")
        if os.path.exists(p):
            continue
        json.dump(run_seed(G, seed, args, device), open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
