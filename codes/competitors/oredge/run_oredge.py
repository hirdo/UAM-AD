"""
OrEdge (semi-supervised reference row) on the thesis' RCAEval pkls, same 3-fold-by-run protocol as Eadro.

OrEdge needs the faulty node of some anomalous samples while training (semi-supervised: part of the culprit labels is
hidden as "unknown"), so train.pkl (normal only) is not enough. In every test_<fault>.pkl each (service, fault) has 3
runs: for fold k the model is trained on the samples of the runs != k (normal samples: all nodes normal; anomalous
samples: the injected service is the culprit, the other nodes stay normal) and scores the samples of run k; the 3 folds
together cover the test file. Label percent 0.5 as in the authors' chunkDataset: the culprit labels of anomalous samples
are kept for the first 5 of every 10 anomalous samples and set to "unknown" for the other 5. A sample is a window of 10
steps (authors' window) labelled by its last step. Sample anomaly score = max over nodes of P(node is the culprit);
ranking of the services for HR@k = the same probabilities. The model of the LAST epoch is used (--select last, as for
Eadro): the authors' "loss" checkpoint (lowest training loss, --select loss) is not usable with a few epochs, because
the loss is (1 - para) * classification + para * reconstruction with para = 1 / (epoch + 1): at epoch 1 the classifier
gets no gradient and the loss is only the (small) reconstruction term, so epoch 1 was chosen in 10 of the 12 folds of the
smoke runs (their "f1" checkpoint is chosen on test data).
Inputs per node and step (L+M+T): metric = the KPIs of that service, trace = its 6 trace features, log = the counts of
the log templates written by that service (the count vector of the second restricted to the templates of the service,
service_logs.entry_service; templates of no service of the system are dropped).
Defaults of the authors' TT parser: 50 epochs, patience 15, AdaBelief lr 1e-3 wd 5e-4, batch 100, abnormal weight 96.

Usage:
  python run_oredge.py --data_dir ../../../data/rcaeval_re2_ob --dataset rcaeval_re2_ob --tests test_cpu --out_dir <dir>
"""
import argparse
import collections
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "eadro"), os.path.join(HERE, "..", "art"), os.path.dirname(HERE)]
from eval_utils import evaluate  # noqa: E402
from oredge_model import AdaBelief, OrEdge  # noqa: E402
from run_art import add_logs, build, load, metric_index  # noqa: E402
from run_eadro import N_TRACE, T, make, run_number  # noqa: E402
from service_logs import entry_service  # noqa: E402


def log_counts(split, vocab):
    """[n, V] template counts in the order of run_art.build."""
    keys = sorted(split, key=lambda k: (k.rsplit("_", 1)[0], int(k.rsplit("_", 1)[1])))
    out = np.zeros((len(keys), len(vocab)), dtype=np.float32)
    for i, k in enumerate(keys):
        for t in split[k]["logs"]:
            j = vocab.get(t)
            if j is not None:
                out[i, j] += 1
    return out


def owner_mask(vocab, services):
    """[S, V]: 1 where template v was written by service s (templates of no service of the system belong to no node)."""
    owner = np.array([services.index(entry_service(t)) if entry_service(t) in services else -1 for t in vocab])
    return (owner[None, :] == np.arange(len(services))[:, None]).astype(np.float32)


CLIP = 100.0


def make_all(path, kpi_idx, c_metric, scale, lscale, vocab, mask, services, dataset, dev):
    """Inputs are scaled by the train max and clipped to +-CLIP: some test KPIs are constant in train, so their scaled
    value reaches 5e13 and the squared reconstruction error overflows (OrEdge squares the error, the others do not)."""
    D = make(path, kpi_idx, c_metric, scale, services, dataset, dev)
    D["X"].clamp_(-CLIP, CLIP)
    D["L"] = torch.tensor(log_counts(load(path), vocab) / lscale, device=dev).clamp_(-CLIP, CLIP)
    D["M"] = torch.tensor(mask, device=dev)
    return D


def batch(D, idx, c_metric):
    p = torch.as_tensor(D["pos"][idx], device=D["X"].device)[:, None] + torch.arange(-(T - 1), 1, device=D["X"].device)[None]
    x, lg = D["X"][p], D["L"][p]                                # [B, T, S, C] (KPIs | 6 trace | 2 log), [B, T, V]
    return x[..., :c_metric], lg[:, :, None] * D["M"], x[..., c_metric:c_metric + N_TRACE]   # node, log of its own templates, edge


@torch.no_grad()
def predict(net, D, idx, c_metric, bs=500):
    net.eval()
    out = []
    for s in range(0, len(idx), bs):
        out.append(torch.softmax(net(*batch(D, idx[s:s + bs], c_metric))[0], -1)[..., 1].cpu())
    return torch.cat(out).numpy()                               # [n, S] P(node is the culprit)


def node_states(D, tr_idx, label_percent=0.5):
    """0 normal, 1 labelled culprit, 2 unknown culprit (first 5 of every 10 anomalous samples labelled)."""
    st = np.zeros((len(D["pos"]), D["X"].shape[1]), dtype=np.int64)
    n = 0
    for i in tr_idx:
        if D["node"][i] >= 0:
            st[i, D["node"][i]] = 1 if (n % 10) < 10 * label_percent else 2
            n += 1
    return st


def train_fold(D, tr_idx, c_metric, a, dev):
    net = OrEdge(c_metric, D["L"].shape[1], N_TRACE, T).to(dev)
    for p in net.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    opt = AdaBelief(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    states = torch.as_tensor(node_states(D, tr_idx), device=dev)
    cw = torch.tensor([1.0, a.abnormal_weight], device=dev)
    best, best_state, best_ep, prev, worse = float("inf"), None, 0, float("inf"), 0
    for ep in range(a.epochs):
        net.train()
        para = max(1.0 / (ep + 1), a.para_low)
        perm = np.random.permutation(tr_idx)
        tot, nb = 0.0, 0
        for s in range(0, len(perm) - a.batch_size + 1, a.batch_size):          # drop_last like the authors
            b = perm[s:s + a.batch_size]
            logits, rec = net(*batch(D, b, c_metric))
            loss = net.loss(logits, rec, states[torch.as_tensor(b, device=dev)], para, cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            opt.step()
            tot += loss.item()
            nb += 1
        tot /= nb
        if a.select == "last":      # fixed number of epochs: the loss is not comparable across epochs (para changes it)
            continue
        if tot <= best:
            best, best_state, best_ep, worse = tot, copy.deepcopy(net.state_dict()), ep, 0
        elif tot > prev:
            worse += 1
            if worse >= a.patience:
                break
        prev = tot
    if a.select == "last":
        return net, ep + 1, ep + 1, tot
    net.load_state_dict(best_state)
    return net, ep + 1, best_ep + 1, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tests", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--abnormal_weight", type=float, default=96.0)
    ap.add_argument("--para_low", type=float, default=1e-2)
    ap.add_argument("--select", choices=["last", "loss"], default="last",
                    help="model of the last epoch (default), or the authors' lowest training loss (with their patience)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    meta = load(os.path.join(a.data_dir, "meta.pkl"))
    services = list(meta["services"])
    kpi_idx = metric_index(meta)
    c_metric = max(len(i) for i in kpi_idx)
    train = load(os.path.join(a.data_dir, "train.pkl"))
    ftr, _, _, _, _, ktr = build(train, kpi_idx, c_metric)
    scale = np.maximum(np.abs(add_logs(ftr, ktr, train, services)).max(0, keepdims=True), 1e-8)
    vocab = {t: i for i, t in enumerate(sorted(collections.Counter(t for s in train.values() for t in s["logs"])))}
    lscale = np.maximum(log_counts(train, vocab).max(0), 1.0)
    mask = owner_mask(vocab, services)
    del train
    args = (kpi_idx, c_metric, scale, lscale, vocab, mask, services, a.dataset, dev)
    val = make_all(os.path.join(a.data_dir, "val.pkl"), *args)

    res = {"seed": a.seed, "tests": {}}
    for name in a.tests:
        D = make_all(os.path.join(a.data_dir, name + ".pkl"), *args)
        S = len(services)
        n = len(D["pos"])
        p_node = np.zeros((n, S))
        val_scores, info = [], []
        t0 = time.time()
        for k in sorted(set(D["run"])):
            held = np.where(D["run"] == k)[0]
            tr = np.where(D["run"] != k)[0]
            net, ep, best_ep, loss = train_fold(D, tr, c_metric, a, dev)
            p_node[held] = predict(net, D, held, c_metric)
            vh = np.where(val["run"] == k)[0]
            if len(vh):
                val_scores.append(predict(net, val, vh, c_metric).max(1))
            info.append(dict(fold=int(k), epochs_run=ep, best_epoch=best_ep, train_loss=loss, n_train=int(len(tr)),
                             n_test=int(len(held))))
            print(f"{name} fold {k}: {ep} epochs (best {best_ep}), loss {loss:.4f}, train {len(tr)}, held-out "
                  f"{len(held)} [{time.time() - t0:.0f}s]", flush=True)
        y = D["y"]
        score = p_node.max(1)
        out = evaluate(y, score, np.concatenate(val_scores))
        pred = (score > 0.5).astype(int)                       # a node with P > 0.5 = argmax of the classifier
        tp, fp, fn = (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
                      int(((pred == 0) & (y == 1)).sum()))
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
