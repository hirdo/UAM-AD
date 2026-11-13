"""
Phase 3, item 10: re-score the stored UAM-AD and UAC-AD checkpoints of item 8 (no training) on the TEST WINDOWS IN TRUE TIME ORDER, and rank the
services with the item-10 design (trace + the service's own KPI channels).

Why: the test pkls of RE2/RE3 are shuffled, so every 30-sample test window mixes experiments and classes (ART, AnoFusion, Eadro and OrEdge read the true order).
Here the SAME samples are sorted by (service_fault, run, second) -- the id has no label -- and each checkpoint is scored again.

One process per (dataset, scenario): run.py prepares the data once (about 90% of the time of a single scoring), then every checkpoint (UAM-AD and UAC-AD, seeds 0-4)
is loaded in turn. Metrics are computed with the repo's own BaseModel.evaluate_norm_sum (F1@val95, AUROC, AUPRC) on the score components of one forward pass; the
oracle F1 is given in both definitions (point-adjusted = BaseModel.evaluate, order dependent; sweep without point-adjust = the competitors' eval_utils.evaluate).
UAM-AD also gets HR@1/3/5 on ALL anomalous test samples with a known injected service (ART's convention) for four rankings: the raw TraceDAE node score
(the previous design), the same score standardised per node on val, the service's KPI channels standardised on val (max over channels), and their sum (adopted).
SN is not re-ordered (its test files carry no time information); with --datasets sn only the UAM-AD HR is refreshed.

Usage:
  python rescore_item10.py [--datasets rcaeval_re3_ob ...] [--scenarios cpu ...] [--protocol time-ordered|shuffled] [--out_dir D:/ClaudeWork/phase3/item10]
Results: <out_dir>/<dataset>/<model>/<scenario>/seed<k>.json ; progress in <out_dir>/status.txt. Existing results are skipped.
"""
import argparse
import glob
import json
import os
import pickle
import re
import runpy
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
DATA = os.path.join(os.path.dirname(CODES), "data")
DATASETS = {                                     # services, scenarios, models
    "rcaeval_re3_ob": (11, ["f1", "f2", "f3", "f4", "f5"], ("uamad", "uacad")),
    "rcaeval_re3_tt": (27, ["f1", "f2", "f3", "f4"], ("uamad", "uacad")),
    "rcaeval_re2_ob": (11, ["cpu", "delay", "disk", "loss", "mem", "socket"], ("uamad", "uacad")),
    "rcaeval_re2_tt": (27, ["cpu", "delay", "disk", "loss", "mem", "socket"], ("uamad", "uacad")),
    "sn": (12, ["Code_Stop_MediaService", "Code_Stop_TextService", "Code_Stop_UserService", "DB_Redis_CacheLimit_HomeTimeline",
                "DB_Redis_CacheLimit_SocialGraph", "DB_Redis_CacheLimit_UserTimeline", "Perf_CPU_Contention", "Perf_Disk_IO_Stress",
                "Perf_Network_Loss", "Svc_Kill_Media", "Svc_Kill_SocialGraph", "Svc_Kill_UserTimeline"], ("uamad",)),
}
SEEDS = range(5)
KEY = re.compile(r"^(.*)_(\d+)_(\d+)$")


def checkpoint_dir(item8, ds, model, sc, seed):
    for d in glob.glob(os.path.join(item8, ds, model, sc, "*")):
        p = os.path.join(d, "params.json")
        if os.path.isfile(p) and json.load(open(p))["run_times"] == seed and os.path.isfile(os.path.join(d, "model.ckpt")):
            return d
    return None


def out_path(out_dir, ds, model, sc, seed):
    return os.path.join(out_dir, ds, model, sc, f"seed{seed}.json")


def run_py_args(ds, ns, test_pkl, result_dir, pre_model):
    common = ["--data", os.path.join(DATA, ds), "--dataset", ds, "--data_type", "fuse", "--open_trace", "True", "--num_services", str(ns), "--trace_c", "6",
              "--val_percentile", "95", "--open_gan_sep", "True", "--run_start", "0", "--run_end", "1", "--test_pkl", test_pkl, "--result_dir", result_dir,
              "--gate_lambda", "0.01", "--score_rule", "norm_sum", "--enable_rca", "False", "--pre_model", pre_model]
    if ds == "sn":
        return common + ["--epoches", "50", "50", "--batch_size", "256", "--patience", "15", "--window_size", "5", "--alpha", "0.16",
                         "--gate_delta_lr_mult", "10", "--activity_penalty_weight", "1.5"]
    return common + ["--epoches", "5", "5", "--batch_size", "128", "--patience", "3", "--window_size", "30", "--alpha", "0.5"]


def worker(a):
    """Runs inside the process that run.py (--pre_model) prepares: BaseModel.evaluate is replaced by the loop over the checkpoints."""
    sys.path.insert(0, CODES)
    os.chdir(CODES)
    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    import models.basev3 as b
    from models.rca import channel_services, hit_rates, parse_injected_service, scenario_from_test_pkl, service_scores
    from models.utils import adjustment_decision

    ns, _, models_ = DATASETS[a.dataset]
    meta = pickle.load(open(os.path.join(DATA, a.dataset, "meta.pkl"), "rb"))
    s2i = meta["service2idx"]
    ch_svc = channel_services(meta["metric_names"], s2i)
    scen_name = scenario_from_test_pkl(a.test_pkl) if a.dataset == "sn" else None
    ckpts = {(m, s): checkpoint_dir(a.item8, a.dataset, m, a.scenario, s) for m in models_ for s in SEEDS}
    todo = [(m, s) for (m, s) in ckpts if not os.path.isfile(out_path(a.out_dir, a.dataset, m, a.scenario, s))]
    if not todo:
        return
    first = ckpts[("uamad", 0)]

    def collect(m, loader, stride):
        """Score components of one pass (BaseModel._score_components, untouched) plus, for UAM-AD, the per-channel KPI error and the per-node trace
        score of every `stride`-th window (hooked on the forward of the same pass)."""
        cap = {"ke": [], "nd": [], "n": 0}
        orig = m.model.forward

        def hooked(inp, *args, **kw):
            res = orig(inp, *args, **kw)
            nb = res["fusion_loss"].shape[0]
            if m.open_trace:
                keep = [i for i in range(nb) if (cap["n"] + i) % stride == 0]
                if keep:
                    kk = torch.as_tensor(keep, device=res["fusion_loss"].device)
                    err = m.model.criterion2(res["output"][1], inp["kpi_features"])
                    ns_ = res["node_scores"]
                    cap["ke"].append(err[kk].reshape(-1, err.shape[-1]).float().cpu().numpy())
                    cap["nd"].append(ns_[kk].reshape(-1, ns_.shape[-1]).float().cpu().numpy())
            cap["n"] += nb
            return res

        m.model.forward = hooked
        try:
            comps = m._score_components(loader)
        finally:
            del m.model.forward
        return comps, (np.concatenate(cap["ke"]) if cap["ke"] else None), (np.concatenate(cap["nd"]) if cap["nd"] else None)

    def rescore(self, test_loader, datatype="Test", threshold=None):
        fr = sys._getframe(1)
        loc, glob_ = fr.f_locals, fr.f_globals
        val_loader = loc["val_loader"]
        W = test_loader.dataset.window_size
        idx2id = test_loader.dataset.idx2id
        params = dict(glob_["params"])
        models = {"uamad": self}
        if "uacad" in models_:
            p2 = dict(params, open_trace=False, enable_rca=False)
            models["uacad"] = b.BaseModel(device=glob_["device"], var_nums=loc["var_nums"], vocab_size=loc["vocab_size"], **p2)
        for name, seed in todo:
            d = ckpts[(name, seed)]
            if d is None:
                print(f"no checkpoint {name} seed {seed}", flush=True)
                continue
            t0 = time.time()
            m = models[name]
            m.load_model(os.path.join(d, "model.ckpt"))
            m.model.eval()
            cv, kv, nv = collect(m, val_loader, W)
            ct, kt, nt = collect(m, test_loader, 1)
            cache = {id(val_loader): cv, id(test_loader): ct}
            m._score_components = lambda loader: cache[id(loader)]
            try:
                cal = m.evaluate_norm_sum(val_loader, test_loader)
            finally:
                del m._score_components
            y = ct["labels"].astype(int)
            out = {"dataset": a.dataset, "scenario": a.scenario, "model": name, "seed": seed, "protocol": a.protocol,
                   "f1": float(cal["f1"]), "pc": float(cal["pc"]), "rc": float(cal["rc"]), "auroc": float(cal["auroc"]), "auprc": float(cal["auprc"]),
                   "threshold": float(cal["threshold"]), "components": cal["components"], "n_test": int(len(y)), "n_anomaly": int(y.sum())}
            # oracle F1, competitors' definition: best F1 over all thresholds of the norm_sum score, no point-adjust
            zt = np.sum([np.clip((ct[k] - cal["median"][k]) / cal["scale_q"][k], 0, None) for k in cal["components"]], axis=0)
            if abs(roc_auc_score(y, zt) - cal["auroc"]) < 1e-9:
                o = np.argsort(-zt)
                tp = np.cumsum(y[o])
                out["oracle_f1_nopa"] = float(np.max(2 * tp / (np.arange(1, len(y) + 1) + y.sum())))
            # oracle F1, repo definition (BaseModel.evaluate): sweep of the fusion loss with point-adjust; depends on the row order
            if m.activity_penalty_weight <= 0:
                loss, true = ct["fusion"], y.tolist()
                best = -1.0
                for ano in range(1, m.anomaly_rate):
                    thr = np.percentile(loss, 100 - ano)
                    pred = adjustment_decision([1 if v > thr else 0 for v in loss], true)
                    best = max(best, f1_score(true, pred))
                out["oracle_f1"] = float(best)
            if name == "uamad" and nt is not None:
                rows = [r for r in np.where(y == 1)[0]]
                gt = [parse_injected_service(idx2id.get(int(r)), a.dataset, scen_name, 1) for r in rows]
                keep = [i for i, g in enumerate(gt) if g in s2i]
                if keep:  # infrastructure scenarios (SN Perf_*) name no service, so there is nothing to localise
                    sel = np.array([rows[i] for i in keep])
                    gt_idx = np.array([s2i[gt[i]] for i in keep])
                    sc = service_scores(nt[sel], nv, kt[sel], kv, ch_svc)
                    out["rca"] = {k: hit_rates(v, gt_idx) for k, v in sc.items()} | {"n_scored": int(len(sel)), "mapped_kpi_channels": int((ch_svc >= 0).sum())}
            out["elapsed_s"] = time.time() - t0
            os.makedirs(os.path.dirname(out_path(a.out_dir, a.dataset, name, a.scenario, seed)), exist_ok=True)
            json.dump(out, open(out_path(a.out_dir, a.dataset, name, a.scenario, seed), "w"), indent=1)
            print(f"{a.dataset} {a.scenario} {name} seed{seed}: F1 {out['f1']:.4f} AUROC {out['auroc']:.4f} AUPRC {out['auprc']:.4f} "
                  f"[{out['elapsed_s']:.0f}s]" + (f" HR@1 sum {out['rca']['sum']['hr1']:.3f}" if "rca" in out else ""), flush=True)
        raise SystemExit(0)

    b.BaseModel.evaluate = rescore
    tmp = os.path.join(a.out_dir, "_tmp_result", a.dataset, a.scenario)
    sys.argv = ["run.py"] + run_py_args(a.dataset, ns, a.test_pkl, tmp, os.path.join(first, "model.ckpt"))
    runpy.run_path(os.path.join(CODES, "run.py"), run_name="__main__")


def orchestrate(a):
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)
    status = open(os.path.join(a.out_dir, "status.txt"), "a", buffering=1)
    for ds in a.datasets:
        ns, scenarios, models_ = DATASETS[ds]
        for sc in (a.scenarios or scenarios):
            if all(os.path.isfile(out_path(a.out_dir, ds, m, sc, s)) for m in models_ for s in SEEDS):
                continue
            if ds == "sn":
                src = glob.glob(os.path.join(DATA, ds, "scenarios", f"test_{sc}_*.pkl"))[0]
            else:
                src = os.path.join(DATA, ds, f"test_{sc}.pkl")
            pkl, protocol = src, ("as-stored" if ds == "sn" else a.protocol)
            if protocol == "time-ordered":
                pkl = os.path.join(a.out_dir, "_tmp", f"test_{ds}_{sc}_ordered.pkl")
                os.makedirs(os.path.dirname(pkl), exist_ok=True)
                data = pickle.load(open(src, "rb"))
                key = lambda k: (KEY.match(k).group(1), int(KEY.match(k).group(2)), int(KEY.match(k).group(3)))
                pickle.dump({k: data[k] for k in sorted(data, key=key)}, open(pkl, "wb"))
                del data
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--dataset", ds, "--scenario", sc, "--test_pkl", pkl, "--protocol", protocol,
                   "--item8", a.item8, "--out_dir", a.out_dir]
            t0 = time.time()
            status.write(f"{time.strftime('%m-%d %H:%M:%S')} START {ds} {sc} ({protocol})\n")
            with open(os.path.join(a.out_dir, "logs", f"{ds}_{sc}.log"), "w", encoding="utf-8") as lg:
                rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=CODES).returncode
            done = sum(os.path.isfile(out_path(a.out_dir, ds, m, sc, s)) for m in models_ for s in SEEDS)
            status.write(f"{time.strftime('%m-%d %H:%M:%S')} {'OK' if rc == 0 else 'FAIL rc=%s' % rc} {ds} {sc} {done}/{len(models_) * len(SEEDS)} results [{time.time() - t0:.0f}s]\n")
            if protocol == "time-ordered" and os.path.isfile(pkl):
                os.remove(pkl)
    status.write("ALL DONE\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["rcaeval_re3_ob", "rcaeval_re3_tt", "rcaeval_re2_ob", "rcaeval_re2_tt"])
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--protocol", choices=["time-ordered", "shuffled", "as-stored"], default="time-ordered")
    ap.add_argument("--item8", default=r"D:\ClaudeWork\phase3\item8")
    ap.add_argument("--out_dir", default=r"D:\ClaudeWork\phase3\item10")
    ap.add_argument("--dataset")
    ap.add_argument("--scenario")
    ap.add_argument("--test_pkl")
    args = ap.parse_args()
    worker(args) if args.worker else orchestrate(args)
