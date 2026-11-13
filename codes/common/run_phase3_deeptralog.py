"""
Phase 3, task 7: DeepTraLog final runs of UAM-AD (3 branch configurations x 5 F-cases x 5 seeds), one python process at
a time, settings read from frozen_config.json. Seed-major order, so a partial run still has every cell for the first
seeds. A (config, F-case, seed) cell that already has a finished info_score.txt is skipped, so the script can simply be
started again after an interruption.

Usage:
  python run_phase3_deeptralog.py --out_dir D:/ClaudeWork/phase3/dtl [--seeds 0 1 2 3 4] [--configs full sync async]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
DATA = os.path.join(os.path.dirname(CODES), "data", "deeptralog")
BRANCHES = {"full": ("True", "True"), "sync": ("True", "False"), "async": ("False", "True")}
FCASES = ["F01", "F02", "F13", "F05", "F09"]


def done(result_dir, seed):
    """run.py names a result folder by a hash of the parameters (which include the run index), without a seed suffix."""
    for f in glob.glob(os.path.join(result_dir, "*", "info_score.txt")):
        try:
            run = json.load(open(os.path.join(os.path.dirname(f), "params.json")))["run_times"]
        except (OSError, KeyError, ValueError):
            continue
        if run == seed and "* Test --" in open(f).read():
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--configs", nargs="+", default=list(BRANCHES))
    ap.add_argument("--fcases", nargs="+", default=FCASES)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "frozen_config.json")))
    c, sh = cfg["per_dataset"]["deeptralog"], cfg["shared"]
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)
    status = open(os.path.join(a.out_dir, "status.txt"), "a", buffering=1)
    fails = 0
    for seed in a.seeds:
        for fc in a.fcases:
            for name in a.configs:
                rd = os.path.join(a.out_dir, name, fc)
                if done(rd, seed):
                    continue
                ot, oa = BRANCHES[name]
                cmd = [sys.executable, os.path.join(CODES, "run.py"), "--data", DATA, "--dataset", "deeptralog",
                       "--data_type", "fuse", "--open_trace", ot, "--open_async_trace", oa,
                       "--score_rule", sh["score_rule"], "--epoches", *map(str, c["epoches"]),
                       "--batch_size", str(c["batch_size"]), "--patience", str(c["patience"]),
                       "--window_size", str(c["window_size"]), "--val_percentile", str(sh["val_percentile"]),
                       "--alpha", str(c["alpha"]), "--open_gan_sep", str(c["open_gan_sep"]),
                       "--open_unmatch_zoomout", str(c["open_unmatch_zoomout"]), "--gate_lambda", str(c["gate_lambda"]),
                       "--run_start", str(seed), "--run_end", str(seed + 1),
                       "--test_pkl", os.path.join(DATA, "scenarios", f"test_{fc}.pkl"), "--result_dir", rd]
                os.makedirs(rd, exist_ok=True)
                t0 = time.time()
                status.write(f"{time.strftime('%m-%d %H:%M:%S')} START {name} {fc} seed{seed}\n")
                with open(os.path.join(a.out_dir, "logs", f"{name}_{fc}_s{seed}.log"), "w") as lg:
                    rc = subprocess.run(cmd, cwd=CODES, stdout=lg, stderr=subprocess.STDOUT).returncode
                ok = rc == 0 and done(rd, seed)
                status.write(f"{time.strftime('%m-%d %H:%M:%S')} {'OK' if ok else 'FAIL rc=%s' % rc} {name} {fc} "
                             f"seed{seed} [{time.time() - t0:.0f}s]\n")
                fails = 0 if ok else fails + 1
                if fails >= 3:
                    status.write("ABORT: 3 consecutive failures\n")
                    return
    status.write("ALL DONE\n")


if __name__ == "__main__":
    main()
