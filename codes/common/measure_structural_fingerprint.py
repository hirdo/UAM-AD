"""Structural fingerprint of DeepTraLog F-cases (plan item 3).

For every F-case, over its *labelled-anomalous* traces, measure the share whose service-level graph has an
edge never seen in normal traces of the same API ("unseen edge"), split by edge kind (S = sync, A = async,
same rule as preprocess_deeptralog.classify_edge). Used to pick the F-cases that count as "structural"
faults (edge set changes) for the sync-vs-async comparison.

Usage:
    python measure_structural_fingerprint.py --dtl_dir D:/ClaudeWork/dtl --fcases F03 F04 F05 F07 F10 F14
Needs <dtl_dir>/normal/*.zip, <dtl_dir>/tld/F*.zip and the label file <dtl_dir>/graphdata/trace_labels.pkl
({trace_id: (trace_bool, error_type)}, True = normal). The normal baseline is cached in
<dtl_dir>/normal_baselines.pkl (same format as the earlier fingerprint run).
"""
import argparse
import collections
import glob
import io
import os
import pickle
import re
import zipfile

import pandas as pd

ASYNC_COMPONENTS = {"rabbitmq-producer", "rabbitmq-consumer", "SpringAsync"}
COLS = ["TraceId", "SpanId", "ParentSpan", "Service", "Component", "SpanType", "StartTime", "EndTime", "URL"]


def root_url(u):
    u = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<id>", str(u))
    return re.sub(r"/\d+(\.\d+)?", "/<n>", u)[:80]


def trace_signatures(zpath):
    """Yield (trace_id, edges, orders, n_spans, api) per trace; one SpanData CSV in memory at a time."""
    z = zipfile.ZipFile(zpath)
    for member in z.namelist():
        if "SpanData" not in member or not member.endswith(".csv") or "__MACOSX" in member:
            continue
        try:
            df = pd.read_csv(io.BytesIO(z.read(member)), usecols=COLS,
                             dtype={"TraceId": "category", "SpanId": str, "ParentSpan": str,
                                    "Service": "category", "Component": "category",
                                    "SpanType": "category", "URL": str})
        except Exception:
            continue
        for tid, g in df.groupby("TraceId", observed=True):
            g = g.set_index("SpanId")
            st, en, sv, co = (g[c].to_dict() for c in ("StartTime", "EndTime", "Service", "Component"))
            edges = set()
            for _, r in g.iterrows():
                p = r.ParentSpan
                if p in sv and sv[p] != r.Service:
                    is_async = r.StartTime > en[p] or r.Component in ASYNC_COMPONENTS or co[p] in ASYNC_COMPONENTS
                    edges.add((sv[p], r.Service, "A" if is_async else "S"))
            lo, hi = {}, {}
            for sid in g.index:
                s = sv[sid]
                lo[s] = min(lo.get(s, st[sid]), st[sid])
                hi[s] = max(hi.get(s, en[sid]), en[sid])
            orders = set()
            for a in lo:
                for b in lo:
                    if a < b and hi[a] < lo[b]:
                        orders.add((a, b, "<"))
                    elif a < b and hi[b] < lo[a]:
                        orders.add((a, b, ">"))
            root = g[g.ParentSpan.astype(str) == "-1"]
            api = root_url(root.URL.mode().iloc[0]) if len(root) and len(root.URL.mode()) else None
            yield str(tid), frozenset(edges), frozenset(orders), len(g), api


def build_baseline(dtl_dir):
    cache = os.path.join(dtl_dir, "normal_baselines.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    base = collections.defaultdict(lambda: dict(edge_union=set(), order_union=set(), ntr=0, span_counts=[]))
    for zp in sorted(glob.glob(os.path.join(dtl_dir, "normal", "*.zip"))):
        for _, e, o, n, api in trace_signatures(zp):
            if api is None:
                continue
            b = base[api]
            b["ntr"] += 1
            b["span_counts"].append(n)
            b["edge_union"] |= e
            b["order_union"] |= o
    base = dict(base)
    pickle.dump(base, open(cache, "wb"))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtl_dir", default="D:/ClaudeWork/dtl")
    ap.add_argument("--fcases", nargs="+", default=["F03", "F04", "F05", "F07", "F10", "F14"])
    args = ap.parse_args()

    base = build_baseline(args.dtl_dir)
    labels = pickle.load(open(os.path.join(args.dtl_dir, "graphdata", "trace_labels.pkl"), "rb"))

    print(f"{'F':4s} {'anom':>5s} {'noAPI%':>7s} {'S-edge%':>8s} {'A-edge%':>8s} {'order%':>7s}   "
          "(% of the F-case's labelled-anomalous traces; S/A = unseen sync/async edge)")
    for f in args.fcases:
        n = no_api = new_s = new_a = new_o = unlabeled = 0
        for tid, e, o, _, api in trace_signatures(os.path.join(args.dtl_dir, "tld", f"{f}.zip")):
            lab = labels.get(tid)
            if lab is None:
                unlabeled += 1
                continue
            if lab[0]:      # trace_bool True = normal
                continue
            n += 1
            if api not in base:
                no_api += 1
                continue
            unseen = e - base[api]["edge_union"]
            new_s += any(k == "S" for _, _, k in unseen)
            new_a += any(k == "A" for _, _, k in unseen)
            new_o += bool(o - base[api]["order_union"])
        pct = lambda x: 100 * x / max(n, 1)
        print(f"{f:4s} {n:5d} {pct(no_api):7.1f} {pct(new_s):8.1f} {pct(new_a):8.1f} {pct(new_o):7.1f}"
              + (f"   [{unlabeled} traces had no label]" if unlabeled else ""))


if __name__ == "__main__":
    main()
