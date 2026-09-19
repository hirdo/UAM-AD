"""
Re-sample the normal side of selected SN test files to a new anomaly rate.

Keeps every anomaly window and randomly subsamples the file's existing normal
windows (seeded), so only the anomaly fraction changes. Only rates >= the
current file's rate are reachable (it can only remove normal windows).

Usage:
    python codes/common/resample_test_anomaly_rate.py --data data/sn \
        --prefix Code_Stop --rate 0.15 --seed 42
"""

import argparse
import glob
import os
import pickle
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dir containing scenarios/test_*.pkl")
    p.add_argument("--prefix", required=True, help="Scenario-name prefix to resample, e.g. Code_Stop")
    p.add_argument("--rate", required=True, type=float, help="Target anomaly fraction, e.g. 0.15")
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    rng = random.Random(args.seed)
    for fn in sorted(glob.glob(os.path.join(args.data, "scenarios", f"test_{args.prefix}*.pkl"))):
        with open(fn, "rb") as f:
            d = pickle.load(f)
        anom = [k for k, v in d.items() if v["label"] == 1]
        norm = [k for k, v in d.items() if v["label"] == 0]
        n_norm = min(len(norm), round(len(anom) * (1 - args.rate) / args.rate))
        new = {k: d[k] for k in anom + rng.sample(norm, n_norm)}
        with open(fn, "wb") as f:
            pickle.dump(new, f)
        print(f"{os.path.basename(fn)}: {len(anom)} anomaly + {n_norm} normal "
              f"= {len(new)} windows (rate={len(anom) / len(new):.3f})")


if __name__ == "__main__":
    main()
