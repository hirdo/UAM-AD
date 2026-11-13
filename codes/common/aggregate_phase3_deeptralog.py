"""Mean +- std over seeds of the Phase 3 DeepTraLog runs (reads every <config>/<F-case>/<hash>/info_score.txt)."""
import argparse
import glob
import os
import re

import numpy as np

METRICS = ["f1", "auroc", "auprc", "oracle_f1"]


def read(path):
    for line in open(path):
        if line.startswith("* Test --"):
            return {k: float(v) for k, v in re.findall(r"(\w+):([-\d.eE]+|nan)", line)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    print(f"{'F-case':<6} {'config':<6} {'n':>2}  " + "  ".join(f"{m:>15}" for m in METRICS))
    for fc in sorted({os.path.basename(p) for p in glob.glob(os.path.join(a.out_dir, "*", "F*"))}):
        for cfg in ("sync", "async", "full"):
            rows = [read(f) for f in glob.glob(os.path.join(a.out_dir, cfg, fc, "*", "info_score.txt"))]
            rows = [r for r in rows if r]
            if rows:
                cells = [f"{np.mean([r[m] for r in rows]):.4f}+-{np.std([r[m] for r in rows]):.4f}" for m in METRICS]
                print(f"{fc:<6} {cfg:<6} {len(rows):>2}  " + "  ".join(f"{c:>15}" for c in cells))


if __name__ == "__main__":
    main()
