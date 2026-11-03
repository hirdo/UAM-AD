"""
Eadro (supervised reference row) on SN (AnoMod): same model, losses, training loop and metrics as run_eadro.py, with the
leave-one-scenario-out folds of sn_labeled.py in place of the 3 folds by run (an SN scenario is one session). One model per
scenario file, trained on the labelled windows of the other sessions, the model of the last epoch is used. The fault
windows of the Perf_* sessions (no single injected service) train the detector only (the locator loss ignores node -1).
HR@k is computed on the fault windows of the scenarios that target one service. Without --history the chunk of a window
is 10 copies of the window (sn_labeled.py explains why).

Usage:
  python run_eadro_sn.py --data_dir ../../../data/sn --out_dir <dir> --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from eval_utils import evaluate  # noqa: E402
from run_eadro import T, predict, train_fold  # noqa: E402
from sn_labeled import SNLabeled  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tests", nargs="+", default=None, help="test file stems; default all scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--history", action="store_true",
                    help="real chunk of the 10 previous windows (pseudo history at the session start, which marks the fault "
                         "windows: see sn_labeled.py); default off = the chunk is 10 copies of the window")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    L = SNLabeled(a.data_dir, T)
    D = L.arrays(dev, a.history)
    S = len(L.sn.services)
    print(f"services {S}, channels {L.c_metric} metric + 6 trace; windows with a full history {len(L.cand)} "
          f"(fault {int(L.y.sum())}), val {len(L.val)}", flush=True)

    res = {"seed": a.seed, "tests": {}}
    for name in a.tests or L.sn.test_names():
        tr, held = L.fold(name)
        y = D["y"][held]
        if y.sum() in (0, len(y)):
            print(f"{name}: skipped, single class among the {len(held)} scorable windows", flush=True)
            continue
        t0 = time.time()
        net, ep, loss = train_fold(D, tr, L.c_metric, S, a, dev)
        p_anom, p_node = predict(net, D, held, L.c_metric)
        out = evaluate(y, p_anom, predict(net, D, L.val, L.c_metric)[0])
        pred = (p_anom > 0.5).astype(int)
        tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
        out["f1_native"] = 2 * tp / max(2 * tp + fp + fn, 1)
        out["precision_native"], out["recall_native"] = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        an = np.where(D["node"][held] >= 0)[0]
        rank = np.array([list(np.argsort(-p_node[i])).index(D["node"][held][i]) for i in an])
        out["rca"] = {f"hr{k}": float((rank < k).mean()) if len(an) else None for k in (1, 3, 5)} | {"n_rca": int(len(an))}
        out.update(folds=[dict(epochs=ep, train_loss=loss, n_train=int(len(tr)), n_train_fault=int(D["y"][tr].sum()),
                               n_test=int(len(held)))],
                   n_unscored=int(len(L.sn.positions(name)) - len(held)), train_time_s=time.time() - t0)
        res["tests"][name] = out
        print(f"{name}: {ep} epochs, loss {loss:.4f}, train {len(tr)} (fault {int(D['y'][tr].sum())}), held-out {len(held)} "
              f"(fault {int(y.sum())}) [{time.time() - t0:.0f}s]\n   F1 native {out['f1_native']:.4f} F1@val {out['f1']:.4f} "
              f"AUROC {out['auroc']:.4f} AUPRC {out['auprc']:.4f} oracleF1 {out['oracle_f1']:.4f} HR {out['rca']}", flush=True)
    json.dump(res, open(os.path.join(a.out_dir, f"seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
