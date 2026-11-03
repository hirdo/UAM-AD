"""
SN (AnoMod): every window of every session in time order, for the time-series competitors (ART, AnoFusion).

The SN pkls hold shuffled 30 s windows and, because preprocess_sn.py::_to_dict drops the fields that start with '_',
neither the session nor the time of a window; the test files also hold only a sample of the normal windows, so most
windows have no neighbours in the file. A next-step predictor needs the windows BEFORE the one it scores, and those
exist in the raw data. This script re-runs the feature extraction of preprocess_sn.py unchanged (same class, same
parameters, same processing order, so the same Drain3 templates) on every session and stores ALL windows of each
session in time order, keyed by the same md5 block id as the pkls. Nothing in data/sn is rewritten, and the competitors
score exactly the windows of the pkls (same samples and labels as UAM-AD) with the true history of the session.

Output: <output_dir>/sessions.pkl = {"sessions": [name, ...], "windows": {name: [{"key", "t", "t_str", "label", "kpis",
"logs", "trace_node_features", "trace_adj"}, ...] sorted by t}}  with t = seconds since the session start.
Then every key of train.pkl, val.pkl and scenarios/test_*.pkl is looked up and its arrays are compared; the script exits
with an error unless all of them are found and identical.

Usage:
  python codes/common/make_sn_sessions.py --sn_data_root D:/AnoMod/SN_data --data_dir data/sn
"""
import argparse
import glob
import logging
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import preprocess_sn as P  # noqa: E402


def build_sessions(sn_data_root, window_sec=30, seed=42):
    pre = P.SNPreprocessor(sn_data_root=sn_data_root, output_dir="", window_sec=window_sec, seed=seed)
    dirs = pre._scenario_dirs()
    normal_key = [k for k in dirs if "Normal_Baseline" in k][0]
    others = [k for k in dirs if "Normal_Baseline" not in k]
    pre._adj = pre._build_static_adj(dirs[normal_key]["trace_dir"])
    pre._build_latency_baseline(dirs[normal_key]["trace_dir"])
    pre._fit_drain3(dirs[normal_key]["log_dir"])
    sessions, windows = [], {}
    for sc in [normal_key] + others:                       # same order as SNPreprocessor.run()
        fw = None if sc == normal_key else P._fault_window_for(sc)
        anom, norm = pre._process_scenario(sc, dirs[sc], fw)
        items = sorted(anom + norm, key=lambda it: it[1]["_t_start"])
        t0 = items[0][1]["_t_start"]
        windows[sc] = [{"key": bid, "t": float((s["_t_start"] - t0).total_seconds()), "t_str": str(s["_t_start"]),
                        "label": s["label"], "kpis": s["kpis"], "logs": s["logs"],
                        "trace_node_features": s["trace_node_features"], "trace_adj": s["trace_adj"]}
                       for bid, s in items]
        sessions.append(sc)
    return {"sessions": sessions, "windows": windows}


def verify(data, data_dir):
    by_key = {w["key"]: (sc, w) for sc in data["sessions"] for w in data["windows"][sc]}
    files = [os.path.join(data_dir, "train.pkl"), os.path.join(data_dir, "val.pkl")]
    files += sorted(glob.glob(os.path.join(data_dir, "scenarios", "test_*.pkl")))
    bad = 0
    for path in files:
        d = pickle.load(open(path, "rb"))
        missing = mismatch = 0
        for k, s in d.items():
            if k not in by_key:
                missing += 1
                continue
            w = by_key[k][1]
            same = (int(s["label"]) == int(w["label"]) and np.array_equal(np.asarray(s["kpis"]), np.asarray(w["kpis"]))
                    and np.array_equal(np.asarray(s["trace_node_features"]), np.asarray(w["trace_node_features"]))
                    and np.array_equal(np.asarray(s["trace_adj"]), np.asarray(w["trace_adj"]))
                    and list(s["logs"]) == list(w["logs"]))
            mismatch += not same
        bad += missing + mismatch
        print(f"{os.path.basename(path):<70} windows {len(d):>4}  missing {missing}  different {mismatch}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn_data_root", default=r"D:\AnoMod\SN_data")
    ap.add_argument("--data_dir", default=r"D:\UAM-AD\data\sn")
    ap.add_argument("--window_sec", type=int, default=30)
    a = ap.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    data = build_sessions(a.sn_data_root, a.window_sec)
    for sc in data["sessions"]:
        w = data["windows"][sc]
        print(f"{sc:<55} windows {len(w):>3}  anomalous {sum(x['label'] for x in w):>3}  span {w[-1]['t']:.0f} s", flush=True)
    bad = verify(data, a.data_dir)
    if bad:
        print(f"NOT SAVED: {bad} windows missing or different")
        sys.exit(1)
    with open(os.path.join(a.data_dir, "sessions.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("saved", os.path.join(a.data_dir, "sessions.pkl"))


if __name__ == "__main__":
    main()
