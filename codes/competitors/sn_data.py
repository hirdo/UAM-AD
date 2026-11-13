"""
SN windows in true time order for the time-series competitors (ART, AnoFusion). Reads data/sn/sessions.pkl written by
codes/common/make_sn_sessions.py (all windows of every session, verified identical to the windows of the pkls).

Global order = sessions sorted by name, windows by time; run id = session, sec = seconds since the session start. A
split (train.pkl, val.pkl or scenarios/test_<name>.pkl) is a set of global positions: the competitor scores exactly
those windows, with the previous windows of the same session as history whether or not they are in the file.
"""
import glob
import os
import pickle

import numpy as np


class SNData:
    """pad = n > 0: every session except Normal_Baseline gets n pseudo windows in front, the last n windows of the
    Normal_Baseline session (key None, label 0, never scored). Needed because AnoMod starts collecting 15 s AFTER the
    fault is injected, so the first windows of a session (where the faults of DB_Redis_*, Perf_*, Svc_Kill_* are) have
    no history in the data; the system before the injection was in its normal state. The same pseudo history is given
    to every window that lacks one, whatever its label. pad = 0: windows without a full history are not scored."""

    def __init__(self, data_dir, pad=0):
        self.data_dir = data_dir
        d = pickle.load(open(os.path.join(data_dir, "sessions.pkl"), "rb"))
        names = self.names = sorted(d["sessions"])
        base = [n for n in names if "Normal_Baseline" in n][0]
        tail = d["windows"][base][len(d["windows"][base]) - pad:] if pad else []
        step = 30.0
        self.samples, run, sec = [], [], []
        for i, sc in enumerate(names):
            wins = d["windows"][sc]
            if pad and sc != base:
                t0 = wins[0]["t"]
                wins = [dict(w, key=None, label=0, t=t0 - step * (pad - j)) for j, w in enumerate(tail)] + wins
            for w in wins:
                self.samples.append(w)
                run.append(i)
                sec.append(w["t"])
        self.run, self.sec = np.array(run), np.array(sec)
        self.label = np.array([int(w["label"]) for w in self.samples])
        self.keys = [w["key"] for w in self.samples]
        self.pos = {k: p for p, k in enumerate(self.keys) if k is not None}
        self.meta = pickle.load(open(os.path.join(data_dir, "meta.pkl"), "rb"))
        self.services = sorted(self.meta["service2idx"], key=self.meta["service2idx"].get)

    def positions(self, name):
        """Global positions (sorted) of the windows of 'train', 'val' or a scenarios/<name> test file."""
        path = os.path.join(self.data_dir, name + ".pkl") if name in ("train", "val") else \
            os.path.join(self.data_dir, "scenarios", name + ".pkl")
        return np.array(sorted(self.pos[k] for k in pickle.load(open(path, "rb"))))

    def test_names(self):
        return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(self.data_dir, "scenarios", "test_*.pkl")))

    def kpi_index(self):
        """Per service: the KPI columns named container_<kind>__<service> (4 each)."""
        names = self.meta["metric_names"]
        return [[j for j, n in enumerate(names) if n.endswith("__" + s)] for s in self.services]
