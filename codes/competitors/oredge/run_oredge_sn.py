"""
OrEdge (semi-supervised reference row) on SN (AnoMod): same model, loss and training loop as run_oredge.py, with the
leave-one-scenario-out folds of sn_labeled.py in place of the 3 folds by run (an SN scenario is one session). One model per
scenario file, trained on the labelled windows of the other sessions, the model of the last epoch is used. Fault windows
of the Perf_* sessions (no single injected service, so no culprit node) are not used for training. Label percent 0.5 as in
run_oredge.py. Inputs per node and 30 s window (L+M+T): the 4 container KPIs of the service, its 6 trace features and the
counts of the log templates ("service|template" strings of the normal baseline) written by that service.
Without --history the chunk of a window is 10 copies of the window (sn_labeled.py explains why).

Usage:
  python run_oredge_sn.py --data_dir ../../../data/sn --out_dir <dir> --seed 0
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "eadro"), os.path.dirname(HERE)]
from eval_utils import evaluate  # noqa: E402
from run_eadro import T  # noqa: E402
from run_oredge import CLIP, owner_mask, predict, train_fold  # noqa: E402
from sn_labeled import SNLabeled  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tests", nargs="+", default=None, help="test file stems; default all scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--abnormal_weight", type=float, default=96.0)
    ap.add_argument("--para_low", type=float, default=1e-2)
    ap.add_argument("--select", choices=["last", "loss"], default="last")
    ap.add_argument("--history", action="store_true",
                    help="real chunk of the 10 previous windows (pseudo history at the session start, which marks the fault "
                         "windows: see sn_labeled.py); default off = the chunk is 10 copies of the window")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    L = SNLabeled(a.data_dir, T)
    sn = L.sn
    D = L.arrays(dev, a.history)
    S = len(sn.services)
    ptr = sn.positions("train")
    vocab = {t: i for i, t in enumerate(sorted(collections.Counter(t for p in ptr for t in sn.samples[p]["logs"])))}
    counts = np.zeros((len(sn.samples), len(vocab)), dtype=np.float32)
    for p, w in enumerate(sn.samples):
        for t in w["logs"]:
            j = vocab.get(t)
            if j is not None:
                counts[p, j] += 1
    counts = counts / np.maximum(counts[ptr].max(0), 1.0)
    D["L"] = torch.tensor(counts if a.history else L.tile(counts), device=dev).clamp_(-CLIP, CLIP)
    D["M"] = torch.tensor(owner_mask(vocab, sn.services), device=dev)
    print(f"services {S}, channels {L.c_metric} metric + 6 trace + {len(vocab)} log templates; windows with a full history "
          f"{len(L.cand)} (fault {int(L.y.sum())}), val {len(L.val)}", flush=True)

    res = {"seed": a.seed, "tests": {}}
    for name in a.tests or sn.test_names():
        tr, held = L.fold(name)
        y = D["y"][held]
        if y.sum() in (0, len(y)):
            print(f"{name}: skipped, single class among the {len(held)} scorable windows", flush=True)
            continue
        tr = tr[~((D["y"][tr] == 1) & (D["node"][tr] < 0))]            # fault window without a known culprit: unusable
        t0 = time.time()
        net, ep, best_ep, loss = train_fold(D, tr, L.c_metric, a, dev)
        p_node = predict(net, D, held, L.c_metric)
        score = p_node.max(1)
        out = evaluate(y, score, predict(net, D, L.val, L.c_metric).max(1))
        pred = (score > 0.5).astype(int)
        tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
        out["f1_native"] = 2 * tp / max(2 * tp + fp + fn, 1)
        out["precision_native"], out["recall_native"] = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        an = np.where(D["node"][held] >= 0)[0]
        rank = np.array([list(np.argsort(-p_node[i])).index(D["node"][held][i]) for i in an])
        out["rca"] = {f"hr{k}": float((rank < k).mean()) if len(an) else None for k in (1, 3, 5)} | {"n_rca": int(len(an))}
        out.update(folds=[dict(epochs_run=ep, best_epoch=best_ep, train_loss=loss, n_train=int(len(tr)),
                               n_train_fault=int(D["y"][tr].sum()), n_test=int(len(held)))],
                   n_unscored=int(len(sn.positions(name)) - len(held)), train_time_s=time.time() - t0)
        res["tests"][name] = out
        print(f"{name}: {ep} epochs, loss {loss:.4f}, train {len(tr)} (fault {int(D['y'][tr].sum())}), held-out {len(held)} "
              f"(fault {int(y.sum())}) [{time.time() - t0:.0f}s]\n   F1 native {out['f1_native']:.4f} F1@val {out['f1']:.4f} "
              f"AUROC {out['auroc']:.4f} AUPRC {out['auprc']:.4f} oracleF1 {out['oracle_f1']:.4f} HR {out['rca']}", flush=True)
    json.dump(res, open(os.path.join(a.out_dir, f"seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
