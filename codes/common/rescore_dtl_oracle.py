"""
Phase 3, Table C: oracle F1 of the stored DeepTraLog checkpoints of UAM-AD (3 branch configurations x F-cases x 5 seeds), no training.

info_score.txt holds run.py's oracle_f1: a point-adjusted sweep of the RAW fusion loss over the top 1-19 %, which is not an oracle of the score that
F1@val95, AUROC and AUPRC use (norm_sum); it can even be lower than F1@val95. Here one process per (configuration, F-case) lets run.py (--pre_model) prepare
the data once, then loads the 5 stored checkpoints in turn (as rescore_item10.py does), takes the val and test score components of each
(BaseModel._score_components) and applies the repo's own BaseModel.evaluate_norm_sum to them. The oracle F1 is the best F1 over every threshold of that same
final score (taken from the roc_auc_score call of evaluate_norm_sum), no point-adjust (the competitors' eval_utils definition). The rebuilt F1@val95 / AUROC /
AUPRC are stored next to the values of info_score.txt, so a score that does not reproduce is noticed.

Usage:
  python rescore_dtl_oracle.py [--dtl_dir D:/ClaudeWork/phase3/dtl] [--out_dir D:/ClaudeWork/phase3/dtl_oracle] [--fcases F01 F02 F13] [--configs full sync async] [--seeds 0 1 2 3 4]
Results: <out_dir>/<config>/<F>/seed<k>.json ; progress in <out_dir>/status.txt. Existing results are skipped.
"""
import argparse
import glob
import json
import os
import re
import runpy
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
DATA = os.path.join(os.path.dirname(CODES), "data", "deeptralog")
BRANCHES = {"full": ("True", "True"), "sync": ("True", "False"), "async": ("False", "True")}


def checkpoint_dir(dtl_dir, name, fc, seed):
    for p in glob.glob(os.path.join(dtl_dir, name, fc, "*", "params.json")):
        d = os.path.dirname(p)
        if json.load(open(p))["run_times"] == seed and os.path.isfile(os.path.join(d, "model.ckpt")):
            return d
    return None


def stored_scores(d):
    line = next(l for l in open(os.path.join(d, "info_score.txt")) if l.startswith("* Test --"))
    return {k: float(v) for k, v in re.findall(r"(\w+):([-\d.eE]+)", line)}


def result_path(out_dir, name, fc, seed):
    return os.path.join(out_dir, name, fc, f"seed{seed}.json")


def run_py_args(name, fc, seed, pre_model, result_dir):
    """The arguments of run_phase3_deeptralog.py for this cell, plus --pre_model."""
    cfg = json.load(open(os.path.join(HERE, "frozen_config.json")))
    c, sh = cfg["per_dataset"]["deeptralog"], cfg["shared"]
    ot, oa = BRANCHES[name]
    return ["--data", DATA, "--dataset", "deeptralog", "--data_type", "fuse", "--open_trace", ot, "--open_async_trace", oa,
            "--score_rule", sh["score_rule"], "--epoches", *map(str, c["epoches"]), "--batch_size", str(c["batch_size"]),
            "--patience", str(c["patience"]), "--window_size", str(c["window_size"]), "--val_percentile", str(sh["val_percentile"]),
            "--alpha", str(c["alpha"]), "--open_gan_sep", str(c["open_gan_sep"]), "--open_unmatch_zoomout", str(c["open_unmatch_zoomout"]),
            "--gate_lambda", str(c["gate_lambda"]), "--run_start", str(seed), "--run_end", str(seed + 1),
            "--test_pkl", os.path.join(DATA, "scenarios", f"test_{fc}.pkl"), "--result_dir", result_dir, "--pre_model", pre_model], sh["val_percentile"]


def rebuild(b, val, test, val_percentile):
    """norm_sum of the repo on the score components; returns its metrics and the oracle F1 of the final test score."""
    class Stub:
        ERR_Z_ONE = b.BaseModel.ERR_Z_ONE

        def __init__(self):
            self.val_percentile = val_percentile

        def _score_components(self, loader):
            return loader

    got, orig = {}, b.roc_auc_score
    b.roc_auc_score = lambda y, s: (got.update(s=np.asarray(s, dtype=float)), orig(y, s))[1]
    try:
        cal = b.BaseModel.evaluate_norm_sum(Stub(), val, test)
    finally:
        b.roc_auc_score = orig
    y = test["labels"].astype(int)
    o = np.argsort(-got["s"])
    tp = np.cumsum(y[o])
    oracle = float(np.max(2 * tp / (np.arange(1, len(y) + 1) + y.sum())))
    return dict(f1=float(cal["f1"]), auroc=float(cal["auroc"]), auprc=float(cal["auprc"]), oracle_f1=oracle, n=int(len(y)), n_anomaly=int(y.sum()))


def worker(a):
    """Runs inside the process that run.py (--pre_model) prepares: BaseModel.load_model is replaced by the loop over the stored checkpoints."""
    sys.path.insert(0, CODES)
    os.chdir(CODES)
    import models.basev3 as b
    todo = [(s, checkpoint_dir(a.dtl_dir, a.config, a.fcase, s)) for s in a.seeds]
    orig_load = b.BaseModel.load_model

    def load_all(self, path):
        loc = sys._getframe(1).f_locals
        val_loader, test_loader = loc["val_loader"], loc["test_loader"]
        for seed, ckpt in todo:
            t0 = time.time()
            orig_load(self, os.path.join(ckpt, "model.ckpt"))
            self.model.eval()
            out = rebuild(b, self._score_components(val_loader), self._score_components(test_loader), self.val_percentile)
            st = stored_scores(ckpt)
            out |= dict(config=a.config, fcase=a.fcase, seed=seed, checkpoint=ckpt, stored={k: st[k] for k in ("f1", "auroc", "auprc")})
            out["reproduced"] = all(abs(out[k] - st[k]) < 1e-3 for k in ("f1", "auroc", "auprc"))
            os.makedirs(os.path.dirname(result_path(a.out_dir, a.config, a.fcase, seed)), exist_ok=True)
            json.dump(out, open(result_path(a.out_dir, a.config, a.fcase, seed), "w"), indent=1)
            print(f"{'OK' if out['reproduced'] else 'NOT REPRODUCED'} {a.config} {a.fcase} seed{seed} F1 {out['f1']:.4f}/{st['f1']:.4f} "
                  f"AUROC {out['auroc']:.4f}/{st['auroc']:.4f} AUPRC {out['auprc']:.4f}/{st['auprc']:.4f} oracle {out['oracle_f1']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        raise SystemExit(0)

    b.BaseModel.load_model = load_all
    args, _ = run_py_args(a.config, a.fcase, a.seeds[0], todo[0][1], os.path.join(a.out_dir, "_tmp_result"))
    sys.argv = ["run.py"] + args
    runpy.run_path(os.path.join(CODES, "run.py"), run_name="__main__")


def orchestrate(a):
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)
    status = open(os.path.join(a.out_dir, "status.txt"), "a", buffering=1)
    for fc in a.fcases:
        for name in a.configs:
            seeds = [s for s in a.seeds if not os.path.isfile(result_path(a.out_dir, name, fc, s))]
            missing = [s for s in seeds if checkpoint_dir(a.dtl_dir, name, fc, s) is None]
            if missing:
                status.write(f"{time.strftime('%m-%d %H:%M:%S')} NO CHECKPOINT {name} {fc} seeds {missing}\n")
                seeds = [s for s in seeds if s not in missing]
            if not seeds:
                continue
            t0 = time.time()
            status.write(f"{time.strftime('%m-%d %H:%M:%S')} START {name} {fc} seeds {seeds}\n")
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--config", name, "--fcase", fc, "--seeds", *map(str, seeds),
                   "--dtl_dir", a.dtl_dir, "--out_dir", a.out_dir]
            with open(os.path.join(a.out_dir, "logs", f"{name}_{fc}.log"), "w") as lg:
                rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=CODES).returncode
            bad = [s for s in seeds if not os.path.isfile(result_path(a.out_dir, name, fc, s))]
            status.write(f"{time.strftime('%m-%d %H:%M:%S')} {'OK' if rc == 0 and not bad else 'FAIL rc=%s missing %s' % (rc, bad)} {name} {fc} [{time.time() - t0:.0f}s]\n")
    status.write("ALL DONE\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--config", choices=list(BRANCHES))
    ap.add_argument("--fcase")
    ap.add_argument("--dtl_dir", default=r"D:/ClaudeWork/phase3/dtl")
    ap.add_argument("--out_dir", default=r"D:/ClaudeWork/phase3/dtl_oracle")
    ap.add_argument("--fcases", nargs="+", default=["F01", "F02", "F13"])
    ap.add_argument("--configs", nargs="+", default=list(BRANCHES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    worker(a) if a.worker else orchestrate(a)
