"""
ART baseline on the thesis' pkl datasets (RCAEval RE2/RE3, OB and TT: sample id = <service>_<fault>_<run>_<second>).

ART (ASE'24) learns to predict the next-step features of every service instance from the previous 5 steps and a
call graph; the anomaly score of a step is the system-level deviation (SLD): sum of the squared prediction errors of
the 3 most deviating instances. Adaptation choices (documented in chon_dataset_va_sota.md):
  * instance = service of the trace graph; channels = that service's metric KPIs (name prefix == service name,
    padded with zeros to the largest service) + its 6 trace features + 2 log features (number of log lines and of
    distinct lines the service wrote in that second, service_logs.py; the 'logs' field of the pkls starts every
    line with the service name). KPIs and logs of other containers (databases, istio, ...) are dropped. The first
    version of this script (seed-0 results moved to item9/art_MT_seed0) had no log channels, i.e. M+T only;
  * graph = trace_adj of the target step, binarised, with self loops, edges caller -> callee, messages along edges;
  * order: samples sorted by (run, second); a window of 6 steps never crosses runs and consecutive samples must be
    <= 60 s apart (the authors' max_gap). The pkls hold a random subset of the seconds of each run, so steps are
    not always consecutive seconds;
  * training = authors' loop (Adam 1e-3, batch 4, early stop when the epoch loss improves < 1e-3 for 5 epochs,
    at most 1000 epochs) on train.pkl; the authors' SPOT / interval-level F1 is replaced by this thesis' protocol:
    threshold = p95 of val scores, per-sample F1, no point-adjust. Score = sum over channels of the SLD vector;
  * evaluation batch = training batch (4) because the transformer attends across the samples of a batch;
  * root-cause ranking (authors' RCL): cosine similarity between each instance's error vector and the SLD vector,
    reported as HR@1/3/5 on anomalous samples whose injected service is known.
The first 5 samples of every run have no full window and are not scored.

Usage:
  python run_art.py --data_dir ../../../data/rcaeval_re2_ob --dataset rcaeval_re2_ob --tests test_cpu --out_dir <dir>
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(os.path.dirname(HERE)), "models")]
from art_model import AutoRegressor  # noqa: E402
from eval_utils import evaluate  # noqa: E402
from rca import parse_injected_service  # noqa: E402
from service_logs import service_log_features  # noqa: E402

HIST = 5          # history steps; window = HIST + 1
MAX_GAP = 60
TOP_T = 3


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def metric_index(meta):
    """List (per service) of the KPI column indices whose name prefix is exactly the service name."""
    names = meta["metric_names"]
    return [[j for j, n in enumerate(names) if n.split("_")[0] == s] for s in meta["services"]]


def build(split, kpi_idx, c_metric):
    """-> feats [T, S, C], adj [T, S, S], run id [T], second [T], label [T], keys; ordered by (run, second)."""
    def key(k):
        a, b = k.rsplit("_", 1)
        return a, int(b)
    keys = sorted(split, key=key)
    S = len(kpi_idx)
    feats = np.zeros((len(keys), S, c_metric + 6), dtype=np.float32)
    adj = np.zeros((len(keys), S, S), dtype=np.float32)
    for t, k in enumerate(keys):
        s = split[k]
        kp = np.nan_to_num(np.asarray(s["kpis"], dtype=np.float32))
        for i, idx in enumerate(kpi_idx):
            feats[t, i, :len(idx)] = kp[idx]
        feats[t, :, c_metric:] = np.nan_to_num(np.asarray(s["trace_node_features"], dtype=np.float32))
        adj[t] = (np.asarray(s["trace_adj"]) > 0).astype(np.float32)
    runs = [key(k)[0] for k in keys]
    run_id = np.unique(runs, return_inverse=True)[1]
    sec = np.array([key(k)[1] for k in keys])
    lab = np.array([int(split[k]["label"]) for k in keys])
    return feats, adj, run_id, sec, lab, keys


def add_logs(feats, keys, split, services):
    """Append the 2 per-service log features (see service_logs.py) as extra channels of every instance."""
    index = {s: i for i, s in enumerate(services)}
    lg = np.stack([service_log_features(split[k]["logs"], index) for k in keys])
    return np.concatenate([feats, lg], axis=2)


def shuffled(rng, feats, adj, lab, keys):
    """Diagnostic only (--shuffle_order): random order over the whole split, as UAM-AD sees its (shuffled) samples."""
    p = rng.permutation(len(keys))
    return feats[p], adj[p], np.zeros(len(p), dtype=int), np.arange(len(p)), lab[p], [keys[i] for i in p]


def valid_positions(run_id, sec):
    pos = np.arange(HIST, len(run_id))
    ok = np.ones(len(pos), dtype=bool)
    for d in range(HIST):  # steps pos-HIST .. pos must be in one run and <= MAX_GAP apart
        ok &= run_id[pos - d - 1] == run_id[pos]
        ok &= (sec[pos - d] - sec[pos - d - 1]) <= MAX_GAP
    return pos[ok]


class Data:
    def __init__(self, feats, adj, device, pad):
        eye = np.eye(feats.shape[1], dtype=np.float32)[None]
        adj = np.minimum(adj + eye, 1.0)
        if pad:  # all-zero dummy instance, no edges
            feats = np.pad(feats, ((0, 0), (0, 1), (0, 0)))
            adj = np.pad(adj, ((0, 0), (0, 1), (0, 1)))
        self.x = torch.tensor(feats, device=device)
        self.adj_t = torch.tensor(adj, device=device).transpose(1, 2).contiguous()  # [b,i,j]=1 if edge j->i


def batch(D, idx):
    idx = torch.as_tensor(idx, device=D.x.device)
    hist = D.x[idx[:, None] + torch.arange(-HIST, 0, device=D.x.device)[None]]
    return hist, D.adj_t[idx], D.x[idx]


@torch.no_grad()
def score(net, D, idx, bs, real):
    """Per-sample SLD score [n] and per-instance error vectors [n, S, C]."""
    net.eval()
    scores, losses, slds = [], [], []
    for s in range(0, len(idx), bs):
        hist, adj_t, tgt = batch(D, idx[s:s + bs])
        loss = (net(adj_t, hist) - tgt) ** 2                      # [B, S, C]
        dev = loss.sum(-1)
        dev[:, real:] = -1.0                                     # never pick the dummy instance
        mask = torch.zeros_like(dev).scatter_(1, dev.topk(TOP_T, dim=-1).indices, 1.0)
        sld = (loss * mask.unsqueeze(-1)).sum(1)                 # [B, C]
        scores.append(sld.sum(-1).cpu())
        losses.append(loss[:, :real].cpu())
        slds.append(sld.cpu())
    return torch.cat(scores).numpy(), torch.cat(losses), torch.cat(slds)


def rank_hits(loss, sld, keys_pos, services, dataset):
    hits = np.zeros(3)
    n = 0
    cos = torch.nn.functional.cosine_similarity(loss, sld.unsqueeze(1), dim=-1).numpy()  # [n, S]
    for r, k in zip(cos, keys_pos):
        gt = parse_injected_service(k, dataset, true_label=1)
        if gt not in services:
            continue
        rank = list(np.argsort(-r)).index(services.index(gt))
        n += 1
        for j, kk in enumerate((1, 3, 5)):
            hits[j] += rank < kk
    return {f"hr{kk}": float(h / n) if n else None for kk, h in zip((1, 3, 5), hits)} | {"n_rca": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tests", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_logs", action="store_true", help="diagnostic: drop the per-service log channels (M+T only)")
    ap.add_argument("--shuffle_order", action="store_true",
                    help="diagnostic: random sample order (no temporal context), as UAM-AD sees its shuffled samples")
    ap.add_argument("--clip", type=float, default=0.0,
                    help="diagnostic: clip the train-max-scaled inputs to +-clip (0 = off, the authors' behaviour)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    meta = load(os.path.join(a.data_dir, "meta.pkl"))
    kpi_idx = metric_index(meta)
    c_metric = max(len(i) for i in kpi_idx)
    S = len(kpi_idx)
    pad = S % 2 == 1
    print(f"instances {S} (+{int(pad)} dummy), channels {c_metric} metric + 6 trace; "
          f"KPIs used {sum(len(i) for i in kpi_idx)}/{len(meta['metric_names'])}", flush=True)

    services = list(meta["services"])
    rng = np.random.default_rng(a.seed)

    def prepare(name):
        split = load(os.path.join(a.data_dir, name + ".pkl"))
        f, ad, r, sec, y, keys = build(split, kpi_idx, c_metric)
        if not a.no_logs:
            f = add_logs(f, keys, split, services)
        if a.shuffle_order:
            f, ad, r, sec, y, keys = shuffled(rng, f, ad, y, keys)
        return f, ad, r, sec, y, keys

    def norm(f):
        x = f / scale
        return np.clip(x, -a.clip, a.clip) if a.clip else x

    ftr, atr, rtr, str_, _, _ = prepare("train")
    fva, ava, rva, sva, _, _ = prepare("val")
    scale = np.maximum(np.abs(ftr).max(0, keepdims=True), 1e-8)
    Dtr, Dva = Data(norm(ftr), atr, dev, pad), Data(norm(fva), ava, dev, pad)
    itr, iva = valid_positions(rtr, str_), valid_positions(rva, sva)
    print(f"train windows {len(itr)}/{len(ftr)}, val windows {len(iva)}/{len(fva)}", flush=True)

    C = ftr.shape[2]
    print(f"channels per instance: {c_metric} metric + 6 trace + {C - c_metric - 6} log", flush=True)
    net = AutoRegressor(S + int(pad), C).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    mse = nn.MSELoss()
    prev, stop, t0 = float("inf"), 0, time.time()
    for ep in range(a.epochs):
        net.train()
        perm = np.random.permutation(itr)
        losses = []
        for s in range(0, len(perm), a.batch_size):
            hist, adj_t, tgt = batch(Dtr, perm[s:s + a.batch_size])
            loss = mse(net(adj_t, hist), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        el = float(np.mean(losses))
        print(f"epoch {ep + 1} loss {el:.6f} [{time.time() - t0:.0f}s]", flush=True)
        if prev - el < 1e-3:
            stop += 1
            if stop == 5:
                print("early stopping", flush=True)
                break
        else:
            stop, prev = 0, el
        sch.step(el)
    train_time = time.time() - t0

    val_scores, _, _ = score(net, Dva, iva, a.batch_size, S)
    res = {"seed": a.seed, "epochs_run": ep + 1, "train_time_s": train_time, "tests": {}}
    for name in a.tests:
        f, ad, r, sec, y, keys = prepare(name)
        D = Data(norm(f), ad, dev, pad)
        ite = valid_positions(r, sec)
        t1 = time.time()
        sc, loss, sld = score(net, D, ite, a.batch_size, S)
        out = evaluate(y[ite], sc, val_scores)
        an = y[ite] == 1
        out.update(rca=rank_hits(loss[an], sld[an], [keys[p] for p in ite[an]], list(meta["services"]), a.dataset),
                   n_unscored=int(len(y) - len(ite)), infer_time_s=time.time() - t1)
        res["tests"][name] = out
        print(f"{name}: F1@val {out['f1']:.4f} P {out['precision']:.4f} R {out['recall']:.4f} AUROC {out['auroc']:.4f} "
              f"AUPRC {out['auprc']:.4f} oracleF1 {out['oracle_f1']:.4f} HR@1/3/5 {out['rca']} "
              f"(unscored {out['n_unscored']})", flush=True)
    json.dump(res, open(os.path.join(a.out_dir, f"seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
