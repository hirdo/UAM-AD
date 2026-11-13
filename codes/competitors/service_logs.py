"""
Per-service log features from the 'logs' field of the thesis' pkl samples, for the competitors that model every service
as an instance (ART, Eadro, OrEdge).

The 'logs' field of a sample (one second, or one 30 s window for SN) is a list of strings that start with the service
that wrote the line:
  * RCAEval RE3-OB, RE3-TT, SN : "<service>|<template>"
  * RCAEval RE2-OB             : "<service> <template or message>"   (no '|')
  * RCAEval RE2-TT             : "<container>"                        (name only, templates were not available)
  * "padding" marks an empty second.
Lines of containers that are not services of the system (ts-ui-dashboard, *-mongo, ...) are dropped, the same way the
KPIs of other containers are dropped for ART.

Features per service (log1p-scaled): [number of log lines, number of distinct lines].
Run as a script to print the coverage of every dataset:  python service_logs.py
"""
import collections
import os
import pickle
import sys

import numpy as np

# CamelCase service names of the SN log files -> canonical service names (from preprocess_sn.py)
SN_NAMES = {
    "ComposePostService": "compose-post-service", "HomeTimelineService": "home-timeline-service",
    "MediaService": "media-service", "NginxThrift": "nginx-web-server", "PostStorageService": "post-storage-service",
    "SocialGraphService": "social-graph-service", "TextService": "text-service", "UniqueIdService": "unique-id-service",
    "UrlShortenService": "url-shorten-service", "UserMentionService": "user-mention-service",
    "UserService": "user-service", "UserTimelineService": "user-timeline-service",
}
N_FEATURES = 2


def entry_service(entry):
    if entry == "padding":
        return None
    head = entry.split("|", 1)[0] if "|" in entry else entry.split(" ", 1)[0]
    return SN_NAMES.get(head, head)


def service_log_features(logs, index):
    """logs: list of strings of one sample; index: {service: row}. -> float32 [n_services, N_FEATURES]."""
    out = np.zeros((len(index), N_FEATURES), dtype=np.float32)
    seen = [set() for _ in index]
    for e in logs:
        i = index.get(entry_service(e))
        if i is None:
            continue
        out[i, 0] += 1
        seen[i].add(e)
    for i, s in enumerate(seen):
        out[i, 1] = len(s)
    return np.log1p(out)


def services_of(meta):
    return list(meta["services"]) if "services" in meta else sorted(meta["service2idx"], key=meta["service2idx"].get)


if __name__ == "__main__":
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    for ds in sys.argv[1:] or ["rcaeval_re2_ob", "rcaeval_re3_ob", "rcaeval_re2_tt", "rcaeval_re3_tt", "sn"]:
        meta = pickle.load(open(os.path.join(data, ds, "meta.pkl"), "rb"))
        services = services_of(meta)
        index = {s: i for i, s in enumerate(services)}
        for split in ("train", "val"):
            d = pickle.load(open(os.path.join(data, ds, split + ".pkl"), "rb"))
            tot = mapped = pad = 0
            dropped = collections.Counter()
            for s in d.values():
                for e in s["logs"]:
                    tot += 1
                    svc = entry_service(e)
                    if svc is None:
                        pad += 1
                    elif svc in index:
                        mapped += 1
                    else:
                        dropped[svc] += 1
            f = np.stack([service_log_features(s["logs"], index) for s in d.values()])
            print(f"{ds:<16}{split:<6} samples {len(d):>6}  lines {tot:>9}  padding {pad:>5}  mapped {mapped / max(tot - pad, 1):.3f}  "
                  f"services with logs {int((f[:, :, 0] > 0).any(0).sum())}/{len(index)}  dropped heads {dropped.most_common(3)}")
