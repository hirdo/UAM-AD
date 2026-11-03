"""
Eadro (supervised reference row) on the thesis' RCAEval pkls, 3-fold cross-validation over runs.

Eadro needs fault labels and the faulty service while training, but train.pkl holds only normal data. Protocol
(agreed): in every test_<fault>.pkl each (service, fault) has 3 runs. For fold k the model is trained on the labelled
samples of the runs != k of that file (normal, anomalous, injected service) and scores the samples of run k; the 3
folds together cover the whole test file, so results are comparable with the unsupervised rows. The model is the one
of the last epoch (the authors select the epoch by test HR@1, which leaks), with the authors' early break on a rising
training loss. A sample is a chunk of the last 10 steps of one run (authors' chunk length), labelled by its last step.
Reported: F1 at the classifier's own decision (argmax), F1 at the p95 threshold of normal val.pkl samples of the
held-out runs, AUROC/AUPRC of P(anomaly), and HR@1/3/5 of the localizer over the truly anomalous samples.
Modalities L+M+T: the KPIs of the service, its 6 trace features and its 2 log features (service_logs.py, as for ART).

Usage:
  python run_eadro.py --data_dir ../../../data/rcaeval_re2_ob --dataset rcaeval_re2_ob --tests test_cpu --out_dir <dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(CODES, "models"), os.path.join(HERE, "..", "art")]
from eadro_model import Eadro  # noqa: E402
from eval_utils import evaluate  # noqa: E402
from rca import parse_injected_service  # noqa: E402
from run_art import add_logs, build, load, metric_index  # noqa: E402

T = 10
N_TRACE = 6          # a service has per step [c_metric KPIs | 6 trace features | 2 log features (service_logs.py)]
MAX_GAP = 60


def valid_positions(run_id, sec):
    pos = np.arange(T - 1, len(run_id))
    ok = np.ones(len(pos), dtype=bool)
    for d in range(T - 1):
        ok &= run_id[pos - d - 1] == run_id[pos]
        ok &= (sec[pos - d] - sec[pos - d - 1]) <= MAX_GAP
    return pos[ok]


def run_number(key):
    return int(key.split("_")[2])


def window(X, idx, dev):
    idx = torch.as_tensor(idx, device=dev)
    return X[idx[:, None] + torch.arange(-(T - 1), 1, device=dev)[None]]  # [B, T, S, C]


def make(split_path, kpi_idx, c_metric, scale, services, dataset, dev):
    split = load(split_path)
    f, ad, run_id, sec, lab, keys = build(split, kpi_idx, c_metric)
    f = add_logs(f, keys, split, services)
    S = f.shape[1]
    adj = np.minimum(ad + np.eye(S, dtype=np.float32)[None], 1.0)
    pos = valid_positions(run_id, sec)
    node = np.array([(services.index(parse_injected_service(keys[p], dataset, true_label=1))
                      if lab[p] == 1 and parse_injected_service(keys[p], dataset, true_label=1) in services else -1)
                     for p in pos])
    return dict(X=torch.tensor(f / scale, device=dev), adj_t=torch.tensor(adj, device=dev).transpose(1, 2).contiguous(),
                pos=pos, y=lab[pos], node=node, run=np.array([run_number(keys[p]) for p in pos]),
                n_total=len(keys))


def batch(D, idx, c_metric):
    w = window(D["X"], D["pos"][idx], D["X"].device)          # [B, T, S, C]
    w = w.permute(0, 2, 1, 3)                                  # [B, S, T, C]
    return (D["adj_t"][torch.as_tensor(D["pos"][idx], device=w.device)], w[..., :c_metric],
            w[..., c_metric:c_metric + N_TRACE], w[..., c_metric + N_TRACE:])


@torch.no_grad()
def predict(net, D, idx, c_metric, bs=512):
    net.eval()
    pa, pn = [], []
    for s in range(0, len(idx), bs):
        dl, ll = net(*batch(D, idx[s:s + bs], c_metric))
        pa.append(torch.softmax(dl, -1)[:, 1].cpu())
        pn.append(torch.softmax(ll, -1).cpu())
    return torch.cat(pa).numpy(), torch.cat(pn).numpy()


def train_fold(D, tr_idx, c_metric, S, a, dev):
    net = Eadro(S, c_metric, N_TRACE, T, D["X"].shape[-1] - c_metric - N_TRACE).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    prev, worse = float("inf"), 0
    for ep in range(a.epochs):
        net.train()
        perm = np.random.permutation(tr_idx)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), a.batch_size):
            b = perm[s:s + a.batch_size]
            if len(b) < 2:
                continue
            dl, ll = net(*batch(D, b, c_metric))
            loss = net.loss(dl, ll, torch.as_tensor(D["y"][b], device=dev),
                            torch.as_tensor(D["node"][b], device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        tot /= nb
        if tot > prev:
            worse += 1
            if worse >= a.patience:
                break
        else:
            worse = 0
        prev = tot
    return net, ep + 1, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tests", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    meta = load(os.path.join(a.data_dir, "meta.pkl"))
    services = list(meta["services"])
    kpi_idx = metric_index(meta)
    c_metric = max(len(i) for i in kpi_idx)
    S = len(services)
    train = load(os.path.join(a.data_dir, "train.pkl"))
    ftr, _, _, _, _, ktr = build(train, kpi_idx, c_metric)
    ftr = add_logs(ftr, ktr, train, services)
    scale = np.maximum(np.abs(ftr).max(0, keepdims=True), 1e-8)
    val = make(os.path.join(a.data_dir, "val.pkl"), kpi_idx, c_metric, scale, services, a.dataset, dev)

    res = {"seed": a.seed, "tests": {}}
    for name in a.tests:
        D = make(os.path.join(a.data_dir, name + ".pkl"), kpi_idx, c_metric, scale, services, a.dataset, dev)
        runs = sorted(set(D["run"]))
        n = len(D["pos"])
        p_anom, p_node = np.zeros(n), np.zeros((n, S))
        val_scores, info = [], []
        t0 = time.time()
        for k in runs:
            held = np.where(D["run"] == k)[0]
            tr = np.where(D["run"] != k)[0]
            net, ep, loss = train_fold(D, tr, c_metric, S, a, dev)
            p_anom[held], p_node[held] = predict(net, D, held, c_metric)
            vh = np.where(val["run"] == k)[0]
            if len(vh):
                val_scores.append(predict(net, val, vh, c_metric)[0])
            info.append(dict(fold=int(k), epochs=ep, train_loss=loss, n_train=int(len(tr)), n_test=int(len(held))))
            print(f"{name} fold {k}: {ep} epochs, loss {loss:.4f}, train {len(tr)}, held-out {len(held)} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        y = D["y"]
        out = evaluate(y, p_anom, np.concatenate(val_scores))
        pred = (p_anom > 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        out["f1_native"] = 2 * tp / max(2 * tp + fp + fn, 1)
        out["precision_native"], out["recall_native"] = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        an = np.where(D["node"] >= 0)[0]
        rank = np.array([list(np.argsort(-p_node[i])).index(D["node"][i]) for i in an])
        out["rca"] = {f"hr{kk}": float((rank < kk).mean()) for kk in (1, 3, 5)} | {"n_rca": int(len(an))}
        out.update(folds=info, n_unscored=int(D["n_total"] - n), train_time_s=time.time() - t0)
        res["tests"][name] = out
        print(f"{name}: F1 native {out['f1_native']:.4f} (P {out['precision_native']:.4f} R {out['recall_native']:.4f}) "
              f"F1@val {out['f1']:.4f} AUROC {out['auroc']:.4f} AUPRC {out['auprc']:.4f} oracleF1 "
              f"{out['oracle_f1']:.4f} HR {out['rca']}", flush=True)
    json.dump(res, open(os.path.join(a.out_dir, f"seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
