"""
Aggregate the results of rescore_item10.py (item 10) the way the item-8/9 tables do: per seed the mean over the scenarios of a dataset, then mean +- population std over the seeds.
Prints, per dataset, UAM-AD and UAC-AD on the time-ordered test next to the same checkpoints on the stored shuffled test (item 8), and the HR@k of the four service rankings of UAM-AD.

Usage:  python aggregate_item10.py [--out_dir D:/ClaudeWork/phase3/item10] [--item8 D:/ClaudeWork/phase3/item8]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

SEEDS = range(5)
ORDER = ["rcaeval_re2_ob", "rcaeval_re3_ob", "rcaeval_re2_tt", "rcaeval_re3_tt", "sn"]
METRICS = ["f1", "auroc", "auprc", "oracle_f1_nopa", "oracle_f1"]
RANKINGS = ["current", "trace", "kpi", "sum"]


def load_new(out_dir, ds, model):
    """{scenario: {seed: json}}"""
    res = {}
    for f in glob.glob(os.path.join(out_dir, ds, model, "*", "seed*.json")):
        j = json.load(open(f))
        res.setdefault(j["scenario"], {})[j["seed"]] = j
    return res


def load_stored(item8, ds, model):
    res = {}
    for d in glob.glob(os.path.join(item8, ds, model, "*", "*")):
        p, i = os.path.join(d, "params.json"), os.path.join(d, "info_score.txt")
        if not (os.path.isfile(p) and os.path.isfile(i)):
            continue
        t = open(i).read()
        m = re.search(r"\* Test --.*", t)
        if not m:
            continue
        g = lambda k: float(re.search(r"\b" + k + r":([\d.]+)", m.group(0)).group(1))
        res.setdefault(os.path.basename(os.path.dirname(d)), {})[json.load(open(p))["run_times"]] = {k: g(k) for k in ("f1", "auroc", "auprc", "oracle_f1")}
    return res


def dataset_mean(per_scenario, getter):
    """per seed: mean over scenarios; returns (mean, std) over seeds, or None if a seed is missing in every scenario."""
    vals = []
    for s in SEEDS:
        v = [getter(sc[s]) for sc in per_scenario.values() if s in sc and getter(sc[s]) is not None]
        if v:
            vals.append(np.mean(v))
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


def fmt(x):
    return "–" if x is None else f"{x[0]:.3f}±{x[1]:.3f}".replace(".", ",")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=r"D:\ClaudeWork\phase3\item10")
    ap.add_argument("--item8", default=r"D:\ClaudeWork\phase3\item8")
    a = ap.parse_args()
    for ds in ORDER:
        print(f"\n=== {ds}")
        print(f"{'model / protocol':32s} | " + " | ".join(f"{m:>14s}" for m in METRICS) + " | scen | seeds")
        for model in ("uamad", "uacad"):
            new, old = load_new(a.out_dir, ds, model), load_stored(a.item8, ds, model)
            if not new:
                continue
            old = {k: v for k, v in old.items() if k in new}
            for label, src in ((f"{model} time-ordered (new)" if ds != "sn" else f"{model} (as stored)", new), (f"{model} shuffled (stored)", old)):
                if ds == "sn" and src is old:
                    continue
                cells = [dataset_mean(src, lambda j, k=k: j.get(k)) for k in METRICS]
                print(f"{label:32s} | " + " | ".join(f"{fmt(c):>14s}" for c in cells) + f" | {len(src):4d} | {max((c[2] for c in cells if c), default=0)}")
        new = load_new(a.out_dir, ds, "uamad")
        if new:
            print("UAM-AD service ranking (HR@1 / HR@3 / HR@5, all anomalous samples with a known service):")
            for r in RANKINGS:
                cells = [dataset_mean(new, lambda j, k=k, r=r: j["rca"][r][k] if "rca" in j else None) for k in ("hr1", "hr3", "hr5")]
                print(f"   {r:8s} | " + " | ".join(f"{fmt(c):>14s}" for c in cells))
