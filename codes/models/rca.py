"""Root Cause Localization — TraceDAE §E.

Ranks service nodes by their per-node reconstruction score (see
`TraceModel.forward` in trace_model_v3.py) and, where derivable, evaluates
the ranking against the service that was actually injected with a fault.

This module only works with plain numpy/python types — no torch — so it can
be exercised independently of the model.
"""

import os
import numpy as np

# Fault-type suffixes used in RCAEval sample ids: f"{service}_{fault}_{run_id}_{i}"
# (see codes/common/preprocess_rcaeval_re2_ob.py / preprocess_rcaeval_re3_ob.py)
RE2_OB_FAULT_TYPES = ("cpu", "delay", "disk", "loss", "mem", "socket")
RE3_OB_FAULT_TYPES = ("f1", "f2", "f3", "f4", "f5")

# SN scenario name (prefix, timestamp suffix stripped) → target service in
# meta.pkl["service2idx"]. Built from real folder names under
# D:/AnoMod/SN_data/trace_data. Scenarios not listed here (Perf_CPU_Contention,
# Perf_Disk_IO_Stress, Perf_Network_Loss, Normal_Baseline) are host/infra-level
# stress that isn't attributable to one service — intentionally absent.
SCENARIO2SERVICE = {
    "Code_Stop_MediaService":           "media-service",
    "Code_Stop_TextService":            "text-service",
    "Code_Stop_UserService":            "user-service",
    "DB_Redis_CacheLimit_HomeTimeline": "home-timeline-service",
    "DB_Redis_CacheLimit_SocialGraph":  "social-graph-service",
    "DB_Redis_CacheLimit_UserTimeline": "user-timeline-service",
    "Svc_Kill_Media":                   "media-service",
    "Svc_Kill_SocialGraph":             "social-graph-service",
    "Svc_Kill_UserTimeline":            "user-timeline-service",
}


def rank_top_k_services(node_scores, idx2service, k=3):
    """Rank service nodes by score, descending.

    Args:
        node_scores: array-like [N] — per-node anomaly score for one (batch,timestep).
        idx2service: dict {node_idx: service_name}.
        k: how many to keep.
    Returns:
        list[(service_name, score)], sorted descending, length <= k.
    """
    scores = np.asarray(node_scores)
    order = np.argsort(-scores)[:k]
    return [(idx2service.get(int(i), f"service_{int(i)}"), float(scores[i])) for i in order]


def scenario_from_test_pkl(test_pkl_path):
    """Recover the SN scenario name from a --test_pkl path.

    SN has no merged test.pkl — every run points --test_pkl at one
    scenarios/test_<scenario_name>.pkl file (preprocess_sn.py::_save_scenario_test),
    so the scenario for the whole run is unambiguous from the filename.
    (preprocess_sn.py's _to_dict strips the per-sample "_scenario" field
    before saving, so it can't be recovered from the sample data itself.)
    """
    if not test_pkl_path:
        return None
    name = os.path.basename(test_pkl_path)
    if name.startswith("test_") and name.endswith(".pkl"):
        return name[len("test_"):-len(".pkl")]
    return None


def parse_injected_service(sample_id, dataset, scenario=None, true_label=1):
    """Recover the ground-truth root-cause service for one sample, if derivable.

    Only meaningful for genuine anomalies (true_label == 1) — a pre-injection
    (normal) sample's id still carries a "{service}_{fault}" prefix naming the
    experiment it was recorded during, but that service isn't actually at
    fault for that (normal) timestep.
    """
    if true_label != 1 or not sample_id:
        return None

    if dataset in ("rcaeval_re2_ob", "rcaeval_re3_ob"):
        fault_types = RE2_OB_FAULT_TYPES if dataset == "rcaeval_re2_ob" else RE3_OB_FAULT_TYPES
        # sid = f"{service}_{fault}_{run_id}_{i}"; service/fault never contain "_".
        parts = sample_id.split("_")
        if len(parts) != 4 or parts[1] not in fault_types:
            return None
        return parts[0]

    if dataset == "sn":
        if not scenario:
            return None
        for prefix, service in SCENARIO2SERVICE.items():
            if scenario.startswith(prefix):
                return service
        return None

    return None


def compute_hit_rate_at_k(records):
    """HR@1/HR@3/HR@5 and MRR over records with a known ground-truth service.

    HR@k = fraction of scored records where gt_service is among the top-k
    predicted services. MRR = mean of 1/rank(gt_service) (0 if not found).
    Returns None if no record has a ground-truth service.
    """
    scored = [r for r in records if r.get("gt_service") is not None]
    if not scored:
        return None

    hr1 = hr3 = hr5 = 0
    rr_sum = 0.0
    for r in scored:
        services = [s for s, _ in r["top_k_services"]]
        gt = r["gt_service"]
        if gt in services:
            rr_sum += 1.0 / (services.index(gt) + 1)
        if gt in services[:1]:
            hr1 += 1
        if gt in services[:3]:
            hr3 += 1
        if gt in services[:5]:
            hr5 += 1

    n = len(scored)
    return {
        "hr1": hr1 / n,
        "hr3": hr3 / n,
        "hr5": hr5 / n,
        "mrr": rr_sum / n,
        "n_scored": n,
    }
