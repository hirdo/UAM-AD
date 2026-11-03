"""
Phase 3 item 9: reference rows (ART, AnoFusion, Eadro, OrEdge) on the RCAEval pkl datasets, one seed per invocation.

Each model script of codes/competitors/ is run as its own process (sequentially: the GPU has 3 GB):
  * ART and AnoFusion train once per dataset on train.pkl and score all test files        -> one job per dataset;
  * Eadro and OrEdge train a 3-fold model per test file (labels needed, see their docstrings) -> one job per test file,
    so a killed run loses at most one test file.
Results go to <out_dir>/<model>/<dataset>[/<test>]/seed<k>.json. A job is skipped when its json exists (resumable) and
is reported FAILED when the process ends without writing it (a silent crash must not look like OK, see item 8).

Usage:
  python run_phase3_item9.py --model art --seed 0 --out_dir D:\\ClaudeWork\\phase3\\item9
  python run_phase3_item9.py --model anofusion --seed 0 --datasets rcaeval_re2_ob --out_dir <dir>
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
COMP = os.path.join(CODES, "competitors")
DATA = os.path.join(os.path.dirname(CODES), "data")

SCRIPT = {"art": "art/run_art.py", "anofusion": "anofusion/run_anofusion.py", "eadro": "eadro/run_eadro.py",
          "oredge": "oredge/run_oredge.py"}
SCRIPT_SN = {"art": "art/run_art_sn.py", "anofusion": "anofusion/run_anofusion_sn.py",   # SN has its own runners
             "eadro": "eadro/run_eadro_sn.py", "oredge": "oredge/run_oredge_sn.py"}      # (leave-one-scenario-out, sn_labeled.py)
PER_TEST = {"eadro", "oredge"}
ALL = ["rcaeval_re3_ob", "rcaeval_re2_ob", "rcaeval_re3_tt", "rcaeval_re2_tt"]
DEFAULT_DATASETS = {"art": ALL + ["sn"], "eadro": ALL + ["sn"], "anofusion": ["rcaeval_re2_ob", "rcaeval_re3_ob", "sn"],
                    "oredge": ALL + ["sn"]}
# AnoFusion: the memory of its graph layers grows with batch * channels^2 (RE2-OB 193 channels fits a batch of 64 on a
# 3 GB GPU, RE3-OB with 465 channels needs micro-batches of 8; the update is the same)
# TT (596 and 1357 channels): micro-batches scaled by channels^2 from the RE3-OB case (465 channels -> 8); not yet fitted on the GPU
ANOFUSION_EXTRA = {"rcaeval_re2_ob": ["--clip", "100"], "rcaeval_re3_ob": ["--micro_batch", "8", "--eval_batch", "64", "--clip", "100"],
                   "rcaeval_re2_tt": ["--micro_batch", "4", "--eval_batch", "32", "--clip", "100"],
                   "rcaeval_re3_tt": ["--micro_batch", "1", "--eval_batch", "8", "--clip", "100"]}
# Epochs = those of UAM-AD/UAC-AD (frozen_config.json: 5 on RCAEval, 50 on SN) for every model, so the budget is the same
EPOCHS = {"sn": 50}
EPOCHS_DEFAULT = 5


def tests_of(dataset):
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(DATA, dataset, "test_*.pkl")))


def jobs(model, dataset, out_dir, seed, only=None):
    base = os.path.join(out_dir, model, dataset)
    if dataset == "sn":
        return [("all", [], base)]                  # the SN runners score every scenario file of data/sn/scenarios
    tests = only or tests_of(dataset)
    if model in PER_TEST:
        return [(t, [t], os.path.join(base, t)) for t in tests]
    return [("all", tests, base)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SCRIPT))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--tests", nargs="+", default=None, help="only these test file stems (quick check)")
    ap.add_argument("--epochs", type=int, default=None, help="override of EPOCHS (5 on RCAEval, 50 on SN)")
    a = ap.parse_args()
    datasets = a.datasets or DEFAULT_DATASETS[a.model]
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)
    status = os.path.join(a.out_dir, "status.txt")

    def log(msg):
        with open(status, "a", encoding="utf-8") as f:
            f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for ds in datasets:
        for name, tests, out in jobs(a.model, ds, a.out_dir, a.seed, a.tests):
            result = os.path.join(out, f"seed{a.seed}.json")
            tag = f"{a.model} {ds} {name} seed{a.seed}"
            if os.path.isfile(result):
                continue
            if ds == "sn":
                cmd = [sys.executable, os.path.join(COMP, SCRIPT_SN[a.model]), "--data_dir", os.path.join(DATA, ds),
                       "--out_dir", out, "--seed", str(a.seed)]
                if a.model in ("art", "anofusion"):
                    cmd += ["--pad_normal"]                                     # see sn_data.SNData
                if a.tests:
                    cmd += ["--tests", *a.tests]
            else:
                cmd = [sys.executable, os.path.join(COMP, SCRIPT[a.model]), "--data_dir", os.path.join(DATA, ds),
                       "--tests", *tests, "--out_dir", out, "--seed", str(a.seed)]
                if a.model != "anofusion":
                    cmd += ["--dataset", ds]
                else:
                    cmd += ANOFUSION_EXTRA.get(ds, [])
            cmd += ["--epochs", str(a.epochs if a.epochs is not None else EPOCHS.get(ds, EPOCHS_DEFAULT))]
            logf = os.path.join(a.out_dir, "logs", f"{a.model}_{ds}_{name}_seed{a.seed}.log")
            log(f"START {tag}")
            t0 = time.time()
            with open(logf, "w", encoding="utf-8") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            ok = os.path.isfile(result)
            log(f"{'OK' if ok else 'FAILED'} {tag} [{time.time() - t0:.0f}s] rc={rc}")
    log(f"ALL DONE {a.model} seed{a.seed}")


if __name__ == "__main__":
    main()
