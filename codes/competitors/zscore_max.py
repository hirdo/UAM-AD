"""
No-training anchors for Table B on the thesis' RCAEval pkls (per sample, no temporal context):
the z-score of every channel of every service instance against the train mean/std (channels that are constant in train
are dropped), with the channels ART uses (service KPIs + 6 trace features + 2 log features), aggregated two ways:
  * max  : the largest |z| over all channels            (the aggregation of AnoFusion / ART's top-k)
  * mean : the mean of |z| over all channels            (the aggregation of UAM-AD's KPI reconstruction error)
Same protocol as the other rows: threshold = p95 of the val scores, no point-adjust, AUROC, AUPRC, plain oracle F1.
HR@k = rank of the injected service by the largest |z| of its channels (anomalous samples only).

Usage:
  python zscore_max.py --data_dir ../../data/rcaeval_re2_ob --dataset rcaeval_re2_ob --out_dir <dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "art"), os.path.join(os.path.dirname(HERE), "models")]
from eval_utils import evaluate  # noqa: E402
from rca import parse_injected_service  # noqa: E402
from run_art import add_logs, build, load, metric_index  # noqa: E402


def features(path, kpi_idx, c_metric, services):
    split = load(path)
    f, _, _, _, y, keys = build(split, kpi_idx, c_metric)
    f = add_logs(f, keys, split, services)
    return f, y, keys                                              # [n, S, C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    meta = load(os.path.join(a.data_dir, "meta.pkl"))
    services = list(meta["services"])
    kpi_idx = metric_index(meta)
    c_metric = max(len(i) for i in kpi_idx)
    ftr, _, _ = features(os.path.join(a.data_dir, "train.pkl"), kpi_idx, c_metric, services)
    mu, sd = ftr.mean(0, keepdims=True), ftr.std(0, keepdims=True)
    kk = sd[0] > 0                                                  # [S, C] channel varies in train
    keep = kk.reshape(-1)
    sd = np.where(sd > 0, sd, 1.0)
    print(f"{a.dataset}: channels {keep.size}, constant in train and dropped {int((~keep).sum())}", flush=True)

    def score(path):
        f, y, keys = features(path, kpi_idx, c_metric, services)
        zfull = np.abs(f - mu) / sd                                 # [n, S, C]
        z = zfull.reshape(len(f), -1)[:, keep]
        inst = np.full((len(f), f.shape[1]), -1.0, dtype=np.float32)  # largest |z| of each service's channels
        for s in range(f.shape[1]):
            if kk[s].any():
                inst[:, s] = zfull[:, s, kk[s]].max(1)
        return y, keys, z.max(1), z.mean(1), inst

    yv, _, mxv, meanv, _ = score(os.path.join(a.data_dir, "val.pkl"))
    res = {"channels": int(keep.size), "dropped": int((~keep).sum()), "tests": {}}
    for p in sorted(glob.glob(os.path.join(a.data_dir, "test_*.pkl"))):
        name = os.path.basename(p)[:-4]
        y, keys, mx, mean, inst = score(p)
        out = {"max": evaluate(y, mx, mxv), "mean": evaluate(y, mean, meanv)}
        an = np.where(y == 1)[0]
        hits, n = np.zeros(3), 0
        for i in an:
            gt = parse_injected_service(keys[i], a.dataset, true_label=1)
            if gt not in services:
                continue
            rank = list(np.argsort(-inst[i])).index(services.index(gt))
            n += 1
            hits += [rank < 1, rank < 3, rank < 5]
        out["rca"] = {f"hr{k}": float(h / n) if n else None for k, h in zip((1, 3, 5), hits)} | {"n_rca": n}
        res["tests"][name] = out
        print(f"{name}: max F1 {out['max']['f1']:.3f} AUROC {out['max']['auroc']:.3f} AUPRC {out['max']['auprc']:.3f} | "
              f"mean F1 {out['mean']['f1']:.3f} AUROC {out['mean']['auroc']:.3f} AUPRC {out['mean']['auprc']:.3f} | "
              f"HR {[round(out['rca'][k], 3) for k in ('hr1', 'hr3', 'hr5')]}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, a.dataset + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
