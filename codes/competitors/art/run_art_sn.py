"""
ART on SN (AnoMod): same model, training loop, score and protocol as run_art.py (RCAEval), with SN's own data layout:
  * instance = service (12); channels = its 4 container KPIs (cpu, memory, net_rx, net_tx) + 6 trace features + 2 log
    features (service_logs.py). The 11 host-level KPIs (system metrics, jaeger spans) are not attributable to a service
    and are dropped, as the KPIs of other containers are for RCAEval;
  * a sample is a 30 s window; the history of a window = the 5 windows before it in the same session, taken from
    data/sn/sessions.pkl (make_sn_sessions.py) whether or not they are in the (sampled) pkl; the windows scored are
    exactly those of train.pkl / val.pkl / scenarios/test_<name>.pkl, so the test sets are the ones UAM-AD is scored on;
  * the first 5 windows of every session have no full history and are not scored;
  * threshold = p95 of the val scores, no point-adjust; RCA = the authors' cosine ranking, HR@1/3/5 on anomalous
    windows of scenarios that target one service (rca.py).
Train.pkl has 39 windows (one normal session), i.e. 34 training windows with a full history.

Usage:
  python run_art_sn.py --data_dir ../../../data/sn --out_dir <dir> --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(os.path.dirname(HERE)), "models")]
from art_model import AutoRegressor  # noqa: E402
from eval_utils import evaluate  # noqa: E402
from rca import parse_injected_service, scenario_from_test_pkl  # noqa: E402
from run_art import Data, batch, score, valid_positions  # noqa: E402
from service_logs import service_log_features  # noqa: E402
from sn_data import SNData  # noqa: E402


def world(sn, no_logs):
    kpi_idx = sn.kpi_index()
    c_metric = max(len(i) for i in kpi_idx)
    S = len(sn.services)
    index = {s: i for i, s in enumerate(sn.services)}
    nlog = 0 if no_logs else 2
    feats = np.zeros((len(sn.samples), S, c_metric + 6 + nlog), dtype=np.float32)
    adj = np.zeros((len(sn.samples), S, S), dtype=np.float32)
    for t, w in enumerate(sn.samples):
        kp = np.nan_to_num(np.asarray(w["kpis"], dtype=np.float32))
        for i, idx in enumerate(kpi_idx):
            feats[t, i, :len(idx)] = kp[idx]
        feats[t, :, c_metric:c_metric + 6] = np.nan_to_num(np.asarray(w["trace_node_features"], dtype=np.float32))
        if nlog:
            feats[t, :, c_metric + 6:] = service_log_features(w["logs"], index)
        adj[t] = (np.asarray(w["trace_adj"]) > 0).astype(np.float32)
    return feats, adj, c_metric


def rank_hits(loss, sld, keys_pos, services, scenario):
    hits, n = np.zeros(3), 0
    cos = torch.nn.functional.cosine_similarity(loss, sld.unsqueeze(1), dim=-1).numpy()
    for r, k in zip(cos, keys_pos):
        gt = parse_injected_service(k, "sn", scenario, true_label=1)
        if gt not in services:
            continue
        rank = list(np.argsort(-r)).index(services.index(gt))
        n += 1
        hits += [rank < 1, rank < 3, rank < 5]
    return {f"hr{kk}": float(h / n) if n else None for kk, h in zip((1, 3, 5), hits)} | {"n_rca": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tests", nargs="+", default=None, help="test file stems; default all scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_logs", action="store_true")
    ap.add_argument("--clip", type=float, default=100.0, help="clip of the train-max-scaled inputs")
    ap.add_argument("--pad_normal", action="store_true",
                    help="give windows without a full history the last windows of the normal baseline session as "
                         "history (see sn_data.SNData); without it those windows are not scored")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    sn = SNData(a.data_dir, pad=5 if a.pad_normal else 0)        # 5 = ART's history length
    feats, adj, c_metric = world(sn, a.no_logs)
    S = len(sn.services)
    valid = valid_positions(sn.run, sn.sec)
    ptr, pva = sn.positions("train"), sn.positions("val")
    itr, iva = np.intersect1d(valid, ptr), np.intersect1d(valid, pva)
    # Val/test windows come from other sessions than train: a channel that is constant in train (max 0, or a tiny max)
    # would be multiplied by 1e8 or more, so it keeps scale 1 and the inputs are clipped to +-clip
    m = np.abs(feats[ptr]).max(0, keepdims=True)
    scale = np.where(m > 0, m, 1.0)
    D = Data(np.clip(feats / scale, -a.clip, a.clip), adj, dev, pad=S % 2 == 1)
    print(f"instances {S}, channels {c_metric} metric + 6 trace + {feats.shape[2] - c_metric - 6} log; windows "
          f"{len(feats)} in {len(sn.run) and len(set(sn.run))} sessions; train {len(itr)}/{len(ptr)}, val {len(iva)}/{len(pva)}",
          flush=True)

    net = AutoRegressor(S + int(S % 2 == 1), feats.shape[2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    mse = nn.MSELoss()
    prev, stop, t0 = float("inf"), 0, time.time()
    for ep in range(a.epochs):
        net.train()
        perm = np.random.permutation(itr)
        losses = []
        for s in range(0, len(perm), a.batch_size):
            hist, adj_t, tgt = batch(D, perm[s:s + a.batch_size])
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

    val_scores, _, _ = score(net, D, iva, a.batch_size, S)
    res = {"seed": a.seed, "epochs_run": ep + 1, "train_time_s": train_time, "tests": {}}
    for name in a.tests or sn.test_names():
        pall = sn.positions(name)
        pos = np.intersect1d(valid, pall)
        y = sn.label[pos]
        if y.sum() in (0, len(y)):
            print(f"{name}: skipped, single class among the {len(pos)} scorable windows", flush=True)
            continue
        t1 = time.time()
        sc, loss, sld = score(net, D, pos, a.batch_size, S)
        out = evaluate(y, sc, val_scores)
        an = y == 1
        scenario = scenario_from_test_pkl(os.path.join(a.data_dir, "scenarios", name + ".pkl"))
        out.update(rca=rank_hits(loss[an], sld[an], [sn.keys[p] for p in pos[an]], sn.services, scenario),
                   n_unscored=int(len(pall) - len(pos)), infer_time_s=time.time() - t1)
        res["tests"][name] = out
        print(f"{name}: F1@val {out['f1']:.4f} P {out['precision']:.4f} R {out['recall']:.4f} AUROC {out['auroc']:.4f} "
              f"AUPRC {out['auprc']:.4f} oracleF1 {out['oracle_f1']:.4f} HR@1/3/5 {out['rca']} "
              f"(scored {len(pos)}/{len(pall)}, anomalies {int(y.sum())})", flush=True)
    json.dump(res, open(os.path.join(a.out_dir, f"seed{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
