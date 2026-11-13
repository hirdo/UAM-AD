"""Metrics shared by the competitor runners: same protocol as UAM-AD (threshold = p95 of normal val scores, no
point-adjust; AUROC/AUPRC and oracle F1 reported next to F1@val)."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate(y, s, val_scores, percentile=95.0):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    thr = float(np.percentile(val_scores, percentile))
    p = (s > thr).astype(int)
    o = np.argsort(-s)
    tp = np.cumsum(y[o])
    k = np.arange(1, len(y) + 1)
    return dict(f1=float(f1_score(y, p, zero_division=0)), precision=float(precision_score(y, p, zero_division=0)),
                recall=float(recall_score(y, p, zero_division=0)), auroc=float(roc_auc_score(y, s)),
                auprc=float(average_precision_score(y, s)), oracle_f1=float(np.max(2 * tp / (k + y.sum()))),
                threshold=thr, n=int(len(y)), n_anomaly=int(y.sum()))
