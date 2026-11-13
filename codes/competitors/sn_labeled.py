"""
SN windows for the labelled references (Eadro: supervised, OrEdge: semi-supervised): the arrays of one dataset dict D
(one entry per window with a full history of T-1 windows) and the leave-one-scenario-out folds.

Why not the RCAEval protocol (3-fold by run inside a test file): it needs several runs of the same fault, an SN scenario
is ONE session. Fold of scenario X = train on the labelled windows of the OTHER sessions (Normal_Baseline included),
score the windows of scenarios/test_X.pkl, i.e. the windows UAM-AD is scored on. Left out of the training set: every window
of session X, and every window of val.pkl (they give the threshold) or test_X.pkl (it holds normal windows of the other
sessions). Labels: window label (0/1) and, for a fault window, the injected service of its session (rca.SCENARIO2SERVICE);
the three Perf_* sessions have no single service, their fault windows have node -1.
Inputs as for ART on SN (run_art_sn.world): 4 container KPIs + 6 trace features + 2 log features per service (L+M+T),
scaled by the train max (scale 1 where it is 0) and clipped to +-clip. Every session except the baseline
gets T-1 pseudo windows in front (the tail of Normal_Baseline, sn_data.SNData), so every real window has a full history.

History (the chunk of T windows) is OFF by default: the fault windows of DB_Redis_*, Perf_* and Svc_Kill_* are the first
windows of their session, exactly those whose history is pseudo history, so a model trained with labels learns "pseudo
history = fault". The rule "the window is among the first 9 of its session" alone gives AUROC 0.91-0.98 on those nine
scenarios (scratchpad/sn_pad_artifact.py). With history off the chunk is T copies of the window itself and no information
about the position in the session enters the input. history=True keeps the real chunk (for the sensitivity row).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "eadro"), os.path.join(HERE, "art"),
                os.path.join(os.path.dirname(os.path.dirname(HERE)), "models")]
from rca import parse_injected_service  # noqa: E402
from run_art_sn import world  # noqa: E402
from run_eadro import valid_positions  # noqa: E402
from sn_data import SNData  # noqa: E402


class SNLabeled:
    def __init__(self, data_dir, T, clip=100.0):
        self.T = T
        self.sn = sn = SNData(data_dir, pad=T - 1)
        feats, adj, self.c_metric = world(sn, no_logs=False)
        m = np.abs(feats[sn.positions("train")]).max(0, keepdims=True)
        self.feats = np.clip(feats / np.where(m > 0, m, 1.0), -clip, clip)
        S = len(sn.services)
        self.adj = np.minimum(adj + np.eye(S, dtype=np.float32)[None], 1.0)
        self.cand = np.array([p for p in valid_positions(sn.run, sn.sec) if sn.keys[p] is not None])
        self.y = sn.label[self.cand]
        self.node = np.full(len(self.cand), -1)
        for i, p in enumerate(self.cand):
            svc = parse_injected_service(sn.keys[p], "sn", sn.names[sn.run[p]], true_label=self.y[i])
            if svc in sn.services:
                self.node[i] = sn.services.index(svc)
        self.val = np.searchsorted(self.cand, np.intersect1d(self.cand, sn.positions("val")))

    def tile(self, a):
        """A per-window array [N_all, ...] in the coordinates of D when history is off: T copies of every window."""
        return np.repeat(a[self.cand], self.T, axis=0)

    def arrays(self, dev, history):
        """The dict D of run_eadro / run_oredge: X, adj_t indexed by position, pos = positions of the windows.
        history False: the chunk of a window is T copies of the window itself (see the module note)."""
        X, adj, pos = self.feats, self.adj, self.cand
        if not history:
            X, adj, pos = self.tile(X), self.tile(adj), np.arange(len(self.cand)) * self.T + self.T - 1
        return dict(X=torch.tensor(X, device=dev), adj_t=torch.tensor(adj, device=dev).transpose(1, 2).contiguous(),
                    pos=pos, y=self.y, node=self.node)

    def fold(self, name):
        """(train idx, held-out idx) into D["pos"] for scenarios/<name>.pkl; the validation idx are self.val."""
        sn = self.sn
        pall = sn.positions(name)
        sx = sn.names.index(name[len("test_"):])
        keep = (sn.run[self.cand] != sx) & ~np.isin(self.cand, np.union1d(pall, sn.positions("val")))
        return np.where(keep)[0], np.searchsorted(self.cand, np.intersect1d(self.cand, pall))
