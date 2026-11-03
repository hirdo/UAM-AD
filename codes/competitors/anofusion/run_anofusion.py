"""
AnoFusion baseline on the thesis' pkl datasets (RCAEval RE2/RE3, OB and TT: sample id = <service>_<fault>_<run>_<second>).

AnoFusion is a next-step predictor on a multivariate series of one "service instance". Here the whole system is
treated as one instance: channels = all KPIs + log-template counts + the flattened per-service trace features of one
sample (= one second). Adaptation choices (all documented in chon_dataset_va_sota.md):
  * order: samples are sorted by (run, second); a 20-step window never crosses runs. The pkls only hold a random
    subset of the seconds of each run, so consecutive samples are not always consecutive seconds (gaps are ignored);
  * training uses train.pkl only (normal), the model of the epoch with the lowest val MSE is kept (val.pkl = normal);
  * the authors' EVT threshold is replaced by this thesis' protocol (p95 of val scores), no point-adjust;
  * the authors' score is kept: max over channels of (|err| - median |err|) / 25th-percentile |err|;
  * channels are scaled by the train per-channel max (authors scale each split by its own max); NaN KPIs -> 0;
    with --clip C a channel whose train max is 0 keeps scale 1 and the scaled inputs are clipped to +-C (without it
    such a channel is multiplied by 1e8 in val/test: RE2-OB test_disk reached 1e18 and the scores became NaN);
  * mutual-information matrices are kept as floats (authors' integer array truncates almost every value to 0 here).
The first 20 samples of every run have no full window and are not scored.

Usage:
  python run_anofusion.py --data_dir ../../../data/rcaeval_re2_ob --tests test_cpu --out_dir <dir> --seed 0
"""
import argparse
import collections
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mutual_info_score

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from anofusion_model import AnoFusionNet  # noqa: E402
from eval_utils import evaluate  # noqa: E402

WINDOW = 20


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_vocab(train):
    cnt = collections.Counter(t for s in train.values() for t in s["logs"])
    return {t: i for i, t in enumerate(sorted(cnt))}


def build_channels(split, vocab):
    """Return X [T, N] (raw), run id [T], label [T], keys, ordered by (run, second)."""
    def key(k):
        p = k.rsplit("_", 1)
        return p[0], int(p[1])
    keys = sorted(split, key=key)
    rows, run_names, labels = [], [], []
    for k in keys:
        s = split[k]
        lc = np.zeros(len(vocab), dtype=np.float32)
        for t in s["logs"]:
            j = vocab.get(t)
            if j is not None:
                lc[j] += 1
        rows.append(np.concatenate([np.nan_to_num(np.asarray(s["kpis"], dtype=np.float32)), lc,
                                    np.nan_to_num(np.asarray(s["trace_node_features"], dtype=np.float32)).ravel()]))
        run_names.append(key(k)[0])
        labels.append(int(s["label"]))
    run_id = np.unique(run_names, return_inverse=True)[1]
    return np.stack(rows), run_id, np.array(labels), keys


def valid_positions(run_id):
    pos = np.arange(WINDOW, len(run_id))
    return pos[run_id[pos - WINDOW] == run_id[pos]]


def mi_matrices(Xn, n_metric, n_log, n_trace, device):
    """Six MI matrices of the authors: M-M, L-L, T-T, M-L, M-T, L-T (computed on the train series)."""
    N = Xn.shape[1]
    block = np.concatenate([np.zeros(n_metric, int), np.ones(n_log, int), np.full(n_trace, 2)])
    kind = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (1, 0): 3, (0, 2): 4, (2, 0): 4, (1, 2): 5, (2, 1): 5}
    A = np.zeros((N, N, 6), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            v = mutual_info_score(Xn[:, i], Xn[:, j])
            A[i, j, kind[(block[i], block[j])]] = A[j, i, kind[(block[i], block[j])]] = v
    return torch.tensor(A, device=device)


def make_batch(X, idx, device):
    w = torch.as_tensor(idx, device=device)[:, None] + torch.arange(-WINDOW, 0, device=device)[None]
    return X[w].permute(0, 2, 1), X[torch.as_tensor(idx, device=device)]


@torch.no_grad()
def predict_err(net, A, X, idx, bs):
    net.eval()
    graphs = net.graphs(A)  # the same for every sample: computed once per pass
    mse, score = [], []
    for s in range(0, len(idx), bs):
        xb, yb = make_batch(X, idx[s:s + bs], X.device)
        d = net.forward_graphs(xb, graphs) - yb
        err = d.abs()
        mse.append((d ** 2).mean(1))
        med = err.median(dim=1).values
        q25 = torch.quantile(err, 0.25, dim=1)
        score.append(((err - med[:, None]) / (q25[:, None] + 1e-8)).max(dim=1).values)
    return torch.cat(mse).mean().item(), torch.cat(score).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tests", nargs="+", required=True, help="test pkl base names, e.g. test_cpu")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--micro_batch", type=int, default=0,
                    help="split each batch into micro-batches with gradient accumulation (same update); memory of "
                         "the graph layers grows with batch * channels^2, so wide datasets need a small value")
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--clip", type=float, default=0.0,
                    help="clip of the train-max-scaled inputs, and scale 1 for channels with train max 0 (0 = off, the authors' scaling)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    train, val = load(os.path.join(a.data_dir, "train.pkl")), load(os.path.join(a.data_dir, "val.pkl"))
    vocab = build_vocab(train)
    Xtr, rtr, _, _ = build_channels(train, vocab)
    Xva, rva, _, _ = build_channels(val, vocab)
    n_metric = len(next(iter(train.values()))["kpis"])
    n_trace = Xtr.shape[1] - n_metric - len(vocab)
    m = np.abs(Xtr).max(0)
    scale = np.where(m > 0, m, 1.0).astype(np.float32) if a.clip else np.maximum(m, 1e-8)

    def norm(X):
        return np.clip(X / scale, -a.clip, a.clip) if a.clip else X / scale

    Xtr_n, Xva_n = norm(Xtr), norm(Xva)
    print(f"channels: {n_metric} metric + {len(vocab)} log + {n_trace} trace = {Xtr.shape[1]}; "
          f"train {len(Xtr)} val {len(Xva)}", flush=True)

    t0 = time.time()
    A = mi_matrices(Xtr_n, n_metric, len(vocab), n_trace, dev)
    print(f"MI matrices [{time.time() - t0:.0f}s]", flush=True)

    Xtr_t, Xva_t = torch.tensor(Xtr_n, device=dev), torch.tensor(Xva_n, device=dev)
    itr, iva = valid_positions(rtr), valid_positions(rva)
    net = AnoFusionNet(Xtr.shape[1], WINDOW).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=1)
    best, best_state, best_ep = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = np.random.permutation(itr)
        tot, nb = 0.0, 0
        for s in range(0, len(perm) - a.batch_size + 1, a.batch_size):  # drop_last like the authors
            idx = perm[s:s + a.batch_size]
            mb = a.micro_batch or len(idx)
            opt.zero_grad()
            loss_val = 0.0
            graphs = net.graphs(A)  # same for the whole batch: computed once, its gradient is propagated after the micro-batches
            gs = [g.detach().requires_grad_() for g in graphs]
            for m in range(0, len(idx), mb):
                xb, yb = make_batch(Xtr_t, idx[m:m + mb], dev)
                loss = nn.functional.mse_loss(net.forward_graphs(xb, gs), yb) * (len(xb) / len(idx))
                loss.backward()
                loss_val += loss.item()
            torch.autograd.backward(graphs, [g.grad for g in gs])
            opt.step()
            tot += loss_val
            nb += 1
        sch.step(tot / nb)
        vmse, _ = predict_err(net, A, Xva_t, iva, a.eval_batch)
        if vmse < best:
            best, best_ep = vmse, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}  # on the CPU: the 3 GB GPU is nearly full at 1357 channels
        print(f"epoch {ep}/{a.epochs} train mse {tot / nb:.6f} val mse {vmse:.6f} [{time.time() - t0:.0f}s]", flush=True)
    train_time = time.time() - t0
    net.load_state_dict(best_state)

    _, val_scores = predict_err(net, A, Xva_t, iva, a.eval_batch)
    res = {"seed": a.seed, "best_epoch": best_ep, "train_time_s": train_time, "tests": {}}
    for name in a.tests:
        te = load(os.path.join(a.data_dir, name + ".pkl"))
        Xte, rte, yte, _ = build_channels(te, vocab)
        ite = valid_positions(rte)
        t1 = time.time()
        _, sc = predict_err(net, A, torch.tensor(norm(Xte), device=dev), ite, a.eval_batch)
        r = evaluate(yte[ite], sc, val_scores)
        r.update(n_unscored=int(len(Xte) - len(ite)), infer_time_s=time.time() - t1)
        res["tests"][name] = r
        print(f"{name}: F1@val {r['f1']:.4f} P {r['precision']:.4f} R {r['recall']:.4f} AUROC {r['auroc']:.4f} "
              f"AUPRC {r['auprc']:.4f} oracleF1 {r['oracle_f1']:.4f} (unscored {r['n_unscored']})", flush=True)
        json.dump(res, open(os.path.join(a.out_dir, f"seed{a.seed}.partial.json"), "w"), indent=1)  # a crash keeps the finished tests
    json.dump(res, open(os.path.join(a.out_dir, f"seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
