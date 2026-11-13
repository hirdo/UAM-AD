"""
AnoFusion on SN (AnoMod): same model, score and protocol as run_anofusion.py (RCAEval), with SN's own data layout:
  * the whole system is one instance; channels = the 59 KPIs + the counts of every log template seen in train.pkl (a
    template is the full "<service>|<template>" string of the pkl) + the flattened per-service trace features (12 x 6);
  * a sample is a 30 s window; the 20 windows before a window in the same session (data/sn/sessions.pkl, see
    common/make_sn_sessions.py) are its input whether or not they are in the (sampled) pkl; the windows scored are
    exactly those of train.pkl / val.pkl / scenarios/test_<name>.pkl (the test sets UAM-AD is scored on); the first 20
    windows of every session are not scored;
  * threshold = p95 of the val scores, no point-adjust; score = max over channels of the normalised error (authors').
train.pkl has 39 windows, i.e. 19 training windows with a full history, so the authors' batch of 64 (drop_last) would
give no step at all: the default batch here is 4 and the epochs are an argument; the val MSE of every epoch is printed.

Usage:
  python run_anofusion_sn.py --data_dir ../../../data/sn --out_dir <dir> --seed 0 --epochs 100
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from anofusion_model import AnoFusionNet  # noqa: E402
from eval_utils import evaluate  # noqa: E402
from run_anofusion import WINDOW, make_batch, mi_matrices, predict_err, valid_positions  # noqa: E402
from sn_data import SNData  # noqa: E402


def channels(sn, vocab):
    rows = []
    for w in sn.samples:
        lc = np.zeros(len(vocab), dtype=np.float32)
        for t in w["logs"]:
            j = vocab.get(t)
            if j is not None:
                lc[j] += 1
        rows.append(np.concatenate([np.nan_to_num(np.asarray(w["kpis"], dtype=np.float32)), lc,
                                    np.nan_to_num(np.asarray(w["trace_node_features"], dtype=np.float32)).ravel()]))
    return np.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tests", nargs="+", default=None, help="test file stems; default all scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--clip", type=float, default=100.0, help="clip of the train-max-scaled inputs")
    ap.add_argument("--pad_normal", action="store_true",
                    help="give windows without a full history the last windows of the normal baseline session as "
                         "history (see sn_data.SNData); without it those windows are not scored")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    sn = SNData(a.data_dir, pad=WINDOW if a.pad_normal else 0)
    ptr, pva = sn.positions("train"), sn.positions("val")
    cnt = collections.Counter(t for p in ptr for t in sn.samples[p]["logs"])
    vocab = {t: i for i, t in enumerate(sorted(cnt))}
    X = channels(sn, vocab)
    n_metric = len(sn.meta["metric_names"])
    # Val/test windows come from other sessions than train, so channels that are constant in train (max 0, or a tiny
    # max) would be multiplied by 1e8 or more: such a channel keeps scale 1 and the inputs are clipped to +-clip
    # (the authors scale every split by its own max, which has no such blow-up)
    m = np.abs(X[ptr]).max(0)
    scale = np.where(m > 0, m, 1.0)
    Xn = np.clip(X / scale, -a.clip, a.clip)
    valid = valid_positions(sn.run)
    itr, iva = np.intersect1d(valid, ptr), np.intersect1d(valid, pva)
    print(f"channels: {n_metric} metric + {len(vocab)} log + {X.shape[1] - n_metric - len(vocab)} trace = {X.shape[1]}; "
          f"windows {len(X)}; train {len(itr)}/{len(ptr)}, val {len(iva)}/{len(pva)}", flush=True)

    t0 = time.time()
    A = mi_matrices(Xn[ptr], n_metric, len(vocab), X.shape[1] - n_metric - len(vocab), dev)
    print(f"MI matrices [{time.time() - t0:.0f}s]", flush=True)

    Xt = torch.tensor(Xn, device=dev)
    net = AnoFusionNet(X.shape[1], WINDOW).to(dev)
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
            xb, yb = make_batch(Xt, idx, dev)
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb, A), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step(tot / nb)
        vmse, _ = predict_err(net, A, Xt, iva, a.eval_batch)
        if vmse < best:
            best, best_ep = vmse, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep}/{a.epochs} train mse {tot / nb:.6f} val mse {vmse:.6f} [{time.time() - t0:.0f}s]", flush=True)
    train_time = time.time() - t0
    net.load_state_dict(best_state)

    _, val_scores = predict_err(net, A, Xt, iva, a.eval_batch)
    res = {"seed": a.seed, "best_epoch": best_ep, "train_time_s": train_time, "tests": {}}
    for name in a.tests or sn.test_names():
        pall = sn.positions(name)
        pos = np.intersect1d(valid, pall)
        y = sn.label[pos]
        if y.sum() in (0, len(y)):
            print(f"{name}: skipped, single class among the {len(pos)} scorable windows", flush=True)
            continue
        t1 = time.time()
        _, sc = predict_err(net, A, Xt, pos, a.eval_batch)
        r = evaluate(y, sc, val_scores)
        r.update(n_unscored=int(len(pall) - len(pos)), infer_time_s=time.time() - t1)
        res["tests"][name] = r
        print(f"{name}: F1@val {r['f1']:.4f} P {r['precision']:.4f} R {r['recall']:.4f} AUROC {r['auroc']:.4f} "
              f"AUPRC {r['auprc']:.4f} oracleF1 {r['oracle_f1']:.4f} (scored {len(pos)}/{len(pall)}, "
              f"anomalies {int(y.sum())})", flush=True)
    json.dump(res, open(os.path.join(a.out_dir, f"seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
