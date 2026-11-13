"""
Phase 3, task 8: UAM-AD / UAC-AD on the five non-DeepTraLog datasets through the eval_per_scenario_* scripts, one
dataset after the other (lightest first), one python process at a time. Those scripts skip seeds already on disk, so the
same command can be repeated later with a wider seed range (e.g. --run_start 0 --run_end 1 now, --run_start 1
--run_end 5 later). UAM-AD = trace on + service ranking (--enable_rca); UAC-AD = trace off.

Usage:
  python run_phase3_item8.py --model uamad --out_dir D:/ClaudeWork/phase3/item8 --run_start 0 --run_end 1
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
# (dataset folder, eval script) from the lightest to the heaviest
DATASETS = [("sn", "sn"), ("rcaeval_re3_tt", "rcaeval_re3_tt"), ("rcaeval_re3_ob", "rcaeval_re3_ob"),
            ("rcaeval_re2_ob", "rcaeval_re2_ob"), ("rcaeval_re2_tt", "rcaeval_re2_tt")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["uamad", "uacad"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_start", type=int, default=0)
    ap.add_argument("--run_end", type=int, default=1)
    ap.add_argument("--datasets", nargs="+", default=[d for d, _ in DATASETS])
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)
    status = open(os.path.join(a.out_dir, "status.txt"), "a", buffering=1)
    uamad = a.model == "uamad"
    for ds, script in DATASETS:
        if ds not in a.datasets:
            continue
        cmd = [sys.executable, os.path.join(HERE, "eval_per_scenario_" + ("sn" if ds == "sn" else script) + ".py"),
               "--data", os.path.join(DATA, ds), "--dataset", ds,
               "--open_trace", str(uamad), "--enable_rca", str(uamad),
               "--result_dir", os.path.join(a.out_dir, ds, a.model),
               "--run_start", str(a.run_start), "--run_end", str(a.run_end)]
        t0 = time.time()
        status.write(f"{time.strftime('%m-%d %H:%M:%S')} START {ds} {a.model} seeds {a.run_start}..{a.run_end - 1}\n")
        with open(os.path.join(a.out_dir, "logs", f"{ds}_{a.model}_s{a.run_start}-{a.run_end - 1}.log"), "w") as lg:
            rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT).returncode
        status.write(f"{time.strftime('%m-%d %H:%M:%S')} {'OK' if rc == 0 else 'FAIL rc=%s' % rc} {ds} {a.model} "
                     f"[{time.time() - t0:.0f}s]\n")
    status.write("ALL DONE\n")


if __name__ == "__main__":
    main()
