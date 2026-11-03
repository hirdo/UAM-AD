"""
Preprocess DeepTraLog (TrainTicket) -> UAM-AD pkl format. Writes both branches'
inputs: `trace_node_features`/`trace_adj` (sync edges only -- consumed by the
existing, unmodified trace_model_v3.py) and the async graph consumed by
async_trace_model_v3.py: `async_trace_node_features`/`async_msg_count_adj` (async
message edges, built in the same single pass over each trace's spans, see
_build_edges) plus `async_temporal_order_adj` (service-level precedence relation, see
_order_adj). See async_trace_model_v3.py's docstring for the async schema.

Stages (one entry point, like preprocess_sn.py):
  labels  extract per-trace labels from DeepTraLog's GraphData archive -> --label_pkl
  normal  build the normal pool (train/unlabel/val/meta + a cache for `fcase`)
  fcase   process --fcases against that cache
  all     normal + fcase in one process (small/smoke runs only)

Unlike SN/RE2/RE3 (1 step = 1 time bucket), DeepTraLog is trace-native:
**1 step = 1 trace**. Each fault case (F01.zip .. F14.zip) is a set of
"runs" (sub-experiments, e.g. F01-01/, F01-02/, ...), each a SpanData CSV +
a raw log file, repeatedly hitting ONE API. `normal/*.zip` similarly holds
several runs, each also pinned to one API.

Sync vs async edge classification (see docs/preprocess_deeptralog_{en,vi}.md
Sec. 2 for the full rationale and a worked example):
  1. Component in {rabbitmq-producer, rabbitmq-consumer} -> async
  2. Component == SpringAsync                              -> async
  3. child span starts after the parent span already ended -> async
     (impossible for a synchronous call: the parent must be waiting)
  4. otherwise                                              -> sync (default)
Only sync edges go into trace_adj; a same-service parent/child pair is
never an inter-service edge (skipped, self-loops added separately) — same
convention as preprocess_sn.py's _build_static_adj. Node features
(trace_node_features) are computed from ALL spans of that service in the
trace, exactly like SN's formula — the one deliberate difference from the
existing pipeline is which edges the encoder's adjacency mask can route
information through.

Per-trace node feature vector (same 6-dim schema as SN, TRACE_NODE_FEAT_DIM):
  [call_count, avg_dur_us, max_dur_us, error_rate, root_rate, latency_dev]

Labels come from GraphData's `error_trace_type` (ground truth), not from
which zip a trace's SpanData CSV lives in -- see extract_labels below and
docs/preprocess_deeptralog_{en,vi}.md Sec. 3 for why (some F-cases are
internally renumbered, and F12.zip mixes in 298 F13-labelled traces).

Output (OUTPUT_DIR/):
  train.pkl / unlabel.pkl  — a sample of normal traces (for training)
  val.pkl                  — a disjoint sample of normal traces (threshold/model selection)
  meta.pkl                 — dataset metadata
  scenarios/
    test_{fcase}.pkl       — that F-case's anomalous traces + normal traces
                              sampled to reach --target_anomaly_rate, shuffled

Usage:
    python codes/common/preprocess_deeptralog.py --stage labels \\
        --graphdata_dir D:/ClaudeWork/dtl/graphdata \\
        --label_pkl D:/ClaudeWork/dtl/graphdata/trace_labels.pkl

    python codes/common/preprocess_deeptralog.py \\
        --fault_dir D:/ClaudeWork/dtl/tld \\
        --normal_dir D:/ClaudeWork/dtl/normal \\
        --label_pkl D:/ClaudeWork/dtl/graphdata/trace_labels.pkl \\
        --output_dir D:/UAM-AD/data/deeptralog \\
        --fcases F01 F02 F04 F13 \\
        --max_normal_traces 20000
"""

import argparse
import gc
import glob
import hashlib
import json
import logging
import os
import pickle
import random
import re
import struct
import zipfile
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

# 35 canonical TrainTicket services, from GraphData/id_service.csv (fixed
# order 0..34, verified 2026-09-22). Any Service value seen in the data that
# is not in this list is dropped (log a warning once) rather than crashing —
# new services could in principle appear in a run we have not inspected.
SERVICES = [
    "ts-order-service", "ts-station-service", "ts-travel2-service", "ts-ticketinfo-service",
    "ts-basic-service", "ts-route-service", "ts-train-service", "ts-price-service",
    "ts-order-other-service", "ts-seat-service", "ts-config-service", "ts-food-service",
    "ts-food-map-service", "ts-travel-service", "ts-execute-service", "ts-assurance-service",
    "ts-inside-payment-service", "ts-payment-service", "ts-contacts-service", "ts-auth-service",
    "ts-verification-code-service", "ts-preserve-other-service", "ts-security-service",
    "ts-user-service", "ts-notification-service", "ts-delivery-service", "ts-cancel-service",
    "ts-preserve-service", "ts-admin-basic-info-service", "ts-admin-travel-service",
    "ts-consign-service", "ts-consign-price-service", "ts-rebook-service",
    "ts-travel-plan-service", "ts-route-plan-service",
]

TRACE_NODE_FEAT_DIM = 6  # [call_count, avg_dur_us, max_dur_us, error_rate, root_rate, latency_dev]
ASYNC_TRACE_NODE_FEAT_DIM = 3  # [async_out_count, async_in_count, avg_lag] — see async_trace_model_v3.py
ASYNC_COMPONENTS = {"rabbitmq-producer", "rabbitmq-consumer", "SpringAsync"}
SPAN_COLS = ["TraceId", "SpanId", "ParentSpan", "Service", "Component", "StartTime", "EndTime", "IsError"]

VAL_FRACTION = 0.2
TARGET_ANOMALY_RATE = 0.125

# raw log line: "... [SW_CTX:[service,instance,TRACE_ID,segment.span,flag]] ..."
_SW_CTX_RE = re.compile(r"\[SW_CTX:\[([^,\]]*),([^,\]]*),([^,\]]*),([^,\]]*),([^,\]]*)\]\]")
_LOG_PREFIX_RE = re.compile(r"^[\d\-: .]+\[SW_CTX:.*?\]\]\s*\[[^\]]*\]\s*")


# ── Stage `labels`: per-trace labels from GraphData ─────────────────────────────
# GraphData/ (github.com/FudanSELab/DeepTraLog/tree/main/GraphData) ships as an
# old-style split zip: graph_data.z01..z07 + graph_data.zip (last part, holds the
# central directory). Concatenating the parts reproduces the original archive, but
# its central directory's local-header offsets are wrong from the 4th entry on --
# both Python's zipfile and `unzip -FF` misread it (verified 2026-09-22). So the
# central directory is bypassed: scan for each entry's real local header
# (b"PK\x03\x04" + the exact filename) and inflate the data directly with
# zlib.decompress(data, -15) (raw deflate). Each processN.jsons is JSON-Lines,
# one trace per line: {"trace_id", "trace_bool", "error_trace_type"};
# trace_bool=True means NORMAL. 132,485 traces, 23,334 (17.6%) anomalous -- the
# paper's numbers.
_GD_PARTS = [f"graph_data.z0{i}" for i in range(1, 8)] + ["graph_data.zip"]
_GD_ENTRIES = ["id_service.csv", "id_url+temp.csv", "id_url+type.csv"] + [f"process{i}.jsons" for i in range(8)]
_GD_SIG = b"PK\x03\x04"


def _find_local_headers(data: bytes) -> Dict[str, Tuple[int, int]]:
    """Real (data_start, compressed_size) of each entry, verifying the filename
    recorded right after the local-header fixed fields."""
    found: Dict[str, Tuple[int, int]] = {}
    pos = 0
    for name in _GD_ENTRIES:
        name_b = name.encode()
        idx = pos
        while True:
            idx = data.find(_GD_SIG, idx)
            if idx < 0:
                raise ValueError(f"Local header for {name} not found (search started at {pos})")
            if idx + 30 <= len(data):
                fnlen = struct.unpack("<H", data[idx + 26:idx + 28])[0]
                exlen = struct.unpack("<H", data[idx + 28:idx + 30])[0]
                if data[idx + 30:idx + 30 + fnlen] == name_b:
                    csize = struct.unpack("<I", data[idx + 18:idx + 22])[0]
                    method = struct.unpack("<H", data[idx + 8:idx + 10])[0]
                    if method != 8:
                        raise ValueError(f"{name}: unexpected compression method {method} (expected 8=deflate)")
                    dstart = idx + 30 + fnlen + exlen
                    found[name] = (dstart, csize)
                    pos = dstart + csize
                    break
            idx += 1
    return found


def extract_labels(graphdata_dir: str) -> Dict[str, Tuple[bool, str]]:
    """{trace_id: (trace_bool, error_trace_type)} for all 132,485 traces (True = normal)."""
    chunks = []
    for name in _GD_PARTS:
        path = os.path.join(graphdata_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} in {graphdata_dir} (download all of GraphData/ first)")
        with open(path, "rb") as f:
            chunks.append(f.read())
    data = b"".join(chunks)
    del chunks
    logging.info(f"Concatenated {len(_GD_PARTS)} GraphData parts: {len(data):,} bytes")
    headers = _find_local_headers(data)
    labels: Dict[str, Tuple[bool, str]] = {}
    for i in range(8):
        dstart, csize = headers[f"process{i}.jsons"]
        raw = zlib.decompress(data[dstart:dstart + csize], -15)
        n = 0
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            labels[d["trace_id"]] = (bool(d["trace_bool"]), d["error_trace_type"])
            n += 1
        logging.info(f"  process{i}.jsons: {n} traces (running total {len(labels)})")
        del raw
    n_anom = sum(1 for v in labels.values() if not v[0])
    logging.info(f"Done: {len(labels)} traces, {n_anom} anomalous ({100 * n_anom / len(labels):.1f}%)")
    return labels


def classify_edge(parent_row: pd.Series, child_row: pd.Series) -> str:
    """Returns 'A' (async) or 'S' (sync) for the edge parent_row -> child_row."""
    if child_row["Component"] in ASYNC_COMPONENTS or parent_row["Component"] in ASYNC_COMPONENTS:
        return "A"
    if child_row["StartTime"] > parent_row["EndTime"]:
        return "A"
    return "S"


class DeepTraLogPreprocessor:
    def __init__(self, fault_dir, normal_dir, label_pkl, output_dir,
                 fcases=None, max_normal_traces=20000, max_drain3_messages=200_000, seed=42):
        self.fault_dir = fault_dir
        self.normal_dir = normal_dir
        self.output_dir = output_dir
        self.fcases = fcases
        self.max_normal_traces = max_normal_traces
        self.max_drain3_messages = max_drain3_messages
        self.seed = seed
        self.rng = random.Random(seed)

        self.services = SERVICES
        self.num_services = len(SERVICES)
        self.service2idx = {s: i for i, s in enumerate(SERVICES)}

        logging.info(f"Loading labels from {label_pkl} ...")
        with open(label_pkl, "rb") as f:
            self.labels: Dict[str, Tuple[bool, str]] = pickle.load(f)
        logging.info(f"  {len(self.labels)} labelled traces")

        self._miner = None
        self._latency_baseline: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._unknown_services_warned = set()

    # ── Drain3 (fit on normal logs) ──────────────────────────────────────────

    def _fit_drain3(self, log_texts):
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except ImportError:
            logging.warning("drain3 not installed - falling back to regex templates")
            self._miner = None
            return
        config = TemplateMinerConfig()
        config.drain_depth = 4
        config.drain_sim_th = 0.5
        config.drain_max_children = 100
        config.parametrize_numeric_tokens = True
        miner = TemplateMiner(config=config)
        n = 0
        for content in log_texts:
            miner.add_log_message(content)
            n += 1
            if n >= self.max_drain3_messages:
                break
        logging.info(f"  Drain3 fitted on {n:,} messages -> {len(miner.drain.id_to_cluster)} templates")
        self._miner = miner

    def _to_template(self, content: str) -> str:
        if self._miner is not None:
            result = self._miner.add_log_message(content)
            return result["template_mined"] if result else content
        return re.sub(r"[0-9a-f]{8,}", "<*>", re.sub(r"\d+", "<NUM>", content))

    # ── Zip helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _span_members(zpath) -> List[str]:
        z = zipfile.ZipFile(zpath)
        return sorted(n for n in z.namelist() if "SpanData" in n and n.endswith(".csv") and "__MACOSX" not in n)

    @staticmethod
    def _log_members(zpath) -> List[str]:
        z = zipfile.ZipFile(zpath)
        return sorted(n for n in z.namelist() if n.endswith(".log") and "__MACOSX" not in n
                      and "back0729" not in n and "evomaster" not in n.lower())

    def _read_spans(self, zpath, member) -> pd.DataFrame:
        z = zipfile.ZipFile(zpath)
        try:
            # Stream via z.open() rather than io.BytesIO(z.read(member)) -- the
            # latter buffers the whole decompressed CSV as one bytes object
            # before pandas even starts parsing it, which for the largest
            # SpanData files is a sizeable transient spike (see the matching
            # note on _run_folder_logs).
            # usecols as a predicate (not the plain list) so a file missing one
            # of SPAN_COLS (a few raw CSVs lack IsError) is still read instead
            # of raising -- a list usecols requires every name to be present.
            with z.open(member) as f:
                df = pd.read_csv(f, usecols=lambda c: c in SPAN_COLS,
                                  dtype={"TraceId": str, "SpanId": str, "ParentSpan": str,
                                         "Service": "category", "Component": "category"})
        except Exception as e:
            logging.warning(f"  Skipping {member}: {e}")
            return pd.DataFrame(columns=SPAN_COLS)
        if "IsError" not in df.columns:
            logging.warning(f"  {member}: no IsError column, assuming no errors")
            df["IsError"] = False
        return df

    def _run_folder_logs(self, zpath, run_prefix) -> Dict[str, List[str]]:
        """Returns {trace_id: [raw log content strings]} for every .log file
        under run_prefix (e.g. 'F01-01/') in this zip."""
        z = zipfile.ZipFile(zpath)
        out: Dict[str, List[str]] = {}
        for n in self._log_members(zpath):
            if not n.startswith(run_prefix):
                continue
            # Stream line-by-line via z.open() rather than z.read().decode() --
            # some raw logs are hundreds of MB uncompressed (e.g. normal0809_03),
            # and materialising the whole decoded text (~2x the file size once
            # you add .splitlines()'s copy) was a large transient spike that,
            # done twice (once in Pass 1's Drain3 pass, again here in Pass 2),
            # contributed to this process being killed for low memory
            # (2026-09-22). A line iterator keeps this to one line at a time.
            try:
                with z.open(n) as f:
                    for raw_line in f:
                        line = raw_line.decode("utf-8", errors="replace")
                        m = _SW_CTX_RE.search(line)
                        if not m:
                            continue
                        tid = m.group(3)
                        content = _LOG_PREFIX_RE.sub("", line).strip()
                        if content:
                            out.setdefault(tid, []).append(content)
            except Exception:
                continue
        return out

    # ── Per-run feature building ─────────────────────────────────────────────

    def _build_samples_for_run(self, zpath, span_member) -> List[Tuple[str, Dict]]:
        """One SpanData CSV = one run. Returns a list of (trace_id, sample_dict)."""
        df = self._read_spans(zpath, span_member)
        if df.empty:
            return []
        df = df.dropna(subset=["TraceId", "SpanId", "Service"])
        unknown = set(df["Service"].astype(str).unique()) - set(self.services)
        new_unknown = unknown - self._unknown_services_warned
        if new_unknown:
            logging.warning(f"  Unknown services in {span_member} (dropped): {sorted(new_unknown)}")
            self._unknown_services_warned |= new_unknown
        df = df[df["Service"].astype(str).isin(self.services)]

        run_prefix = span_member.rsplit("/", 1)[0] + "/" if "/" in span_member else ""
        run_logs = self._run_folder_logs(zpath, run_prefix)

        results = []
        for tid, g in df.groupby("TraceId", observed=True):
            lbl = self.labels.get(tid)
            if lbl is None:
                continue  # trace not present in GraphData snapshot (see docs Sec.3, F07/F08 gap)
            is_normal, err_type = lbl  # GraphData's trace_bool: True == normal (see dataset.py: y=0 if trace_bool else 1)
            is_anomaly = not is_normal

            node_feat = self._trace_node_features(g)
            sync_adj, async_node, async_adj = self._build_edges(g)
            order_adj = self._order_adj(g)

            msgs_raw = run_logs.get(tid, [])
            msgs = [f"{svc}|{self._to_template(c)}" for svc, c in
                    self._pair_log_content_with_service(g, msgs_raw)]
            if not msgs:
                msgs = ["padding"]

            sample = {
                "label": int(is_anomaly),
                "kpi_label": int(is_anomaly),
                "log_label": int(is_anomaly),
                "kpis": np.zeros(1, dtype=np.float32),           # no metric in this dataset
                "logs": msgs,
                "seqs": msgs,
                "log_features": np.zeros(1, dtype=np.float32),   # overwritten by FeatureExtractor
                "trace_node_features": node_feat,
                "trace_adj": sync_adj,
                "async_trace_node_features": async_node,         # Step 4 (additive)
                "async_msg_count_adj": async_adj,                    # Step 4 (additive)
                "async_temporal_order_adj": order_adj,                    # precedence relation (see _order_adj)
                "_error_trace_type": err_type,
                "_trace_id": tid,
            }
            results.append((tid, sample))
        return results

    @staticmethod
    def _pair_log_content_with_service(spans: pd.DataFrame, msgs_raw: List[str]):
        """Best-effort: we don't know which span emitted which log line, so we
        just tag every log line of this trace with the trace's dominant
        (most-frequent) service. Good enough for a bag-of-templates feature
        (see semantics.py FeatureExtractor: order/attribution within a step
        is discarded anyway)."""
        if not msgs_raw:
            return []
        dominant_svc = spans["Service"].astype(str).mode().iloc[0]
        return [(dominant_svc, c) for c in msgs_raw]

    def _trace_node_features(self, spans: pd.DataFrame) -> np.ndarray:
        N = self.num_services
        result = np.zeros((N, TRACE_NODE_FEAT_DIM), dtype=np.float32)
        spans = spans.copy()
        spans["dur_us"] = (spans["EndTime"] - spans["StartTime"]).clip(lower=0) * 1000.0
        spans["is_root"] = (spans["ParentSpan"].astype(str) == "-1").astype(int)
        spans["is_error"] = spans["IsError"].astype(str).str.lower().isin(["true", "1"]).astype(int)
        spans["svc_idx"] = spans["Service"].map(self.service2idx)

        for svc_idx, sdf in spans.groupby("svc_idx"):
            if pd.isna(svc_idx):
                continue
            svc_idx = int(svc_idx)
            n = len(sdf)
            dur = sdf["dur_us"].values
            result[svc_idx, 0] = float(n)
            result[svc_idx, 1] = float(dur.mean())
            result[svc_idx, 2] = float(dur.max())
            result[svc_idx, 3] = float(sdf["is_error"].sum()) / n
            result[svc_idx, 4] = float(sdf["is_root"].sum()) / n

        active = result[:, 0] > 0  # which services actually appear in this trace

        result[:, 0] = np.log1p(result[:, 0]) / 10.0
        result[:, 1] = result[:, 1] / 1e6
        result[:, 2] = result[:, 2] / 1e6

        if self._latency_baseline is not None:
            # Only for services that participated: col1 (avg_dur) is 0 for an
            # absent service, and unconditionally z-scoring that against its
            # baseline mean/std produces a large, trace-independent constant
            # (verified: -0.4545 for service 20 in every trace that doesn't
            # call it) rather than "no data" -- with ~2/3 of rows inactive in
            # a typical trace, this constant filler dominated the structural
            # signal from the handful of services that actually took part and
            # was very likely why F04 (a real, structural-edge-changing fault)
            # scored near-random (AUROC 0.45) in the first full run
            # (2026-09-22): every trace's feature matrix looked nearly
            # identical regardless of which services it actually touched.
            bl_mean, bl_std = self._latency_baseline
            dev = np.clip((result[:, 1] - bl_mean) / bl_std, -10.0, 10.0)
            result[:, 5] = np.where(active, dev, 0.0)
        return result

    def _build_edges(self, spans: pd.DataFrame):
        """One pass over cross-service parent/child span pairs, classifying
        each as sync or async (module docstring rules 1-4) and building:
          sync_adj    [N,N] symmetric 0/1 -- SYNC edges only (the one
                      deliberate change from preprocess_sn.py's
                      _build_static_adj, which includes every edge)
          async_node  [N,3] [async_out_count, async_in_count, avg_lag],
                      log1p-scaled counts, from ASYNC edges only
          async_adj   [N,N] directed, weighted (log1p(message count) i->j),
                      NOT symmetrized -- see async_trace_model_v3.py's
                      docstring for why direction and count both matter here.
        Async branch (Step 4, additive): computed alongside sync_adj in the
        same loop rather than a second pass over the same spans.
        """
        N = self.num_services
        sync_adj = np.zeros((N, N), dtype=np.float32)
        async_adj_count = np.zeros((N, N), dtype=np.float64)
        out_cnt = np.zeros(N, dtype=np.int64)
        in_cnt = np.zeros(N, dtype=np.int64)
        lag_sum = np.zeros(N, dtype=np.float64)

        by_id = spans.set_index("SpanId")
        for sid, row in spans.iterrows():
            pid = row["ParentSpan"]
            if pid not in by_id.index:
                continue
            prow = by_id.loc[pid]
            if isinstance(prow, pd.DataFrame):  # duplicate SpanId (shouldn't happen, be defensive)
                prow = prow.iloc[0]
            if prow["Service"] == row["Service"]:
                continue  # same-service: not an inter-service edge
            pi = self.service2idx.get(str(prow["Service"]))
            ci = self.service2idx.get(str(row["Service"]))
            if pi is None or ci is None:
                continue

            if classify_edge(prow, row) == "S":
                sync_adj[pi, ci] = 1.0
                sync_adj[ci, pi] = 1.0
            else:
                async_adj_count[pi, ci] += 1.0
                out_cnt[pi] += 1
                in_cnt[ci] += 1
                lag_sum[ci] += max(row["StartTime"] - prow["EndTime"], 0) / 1000.0  # ms -> s

        np.fill_diagonal(sync_adj, 1.0)

        async_node = np.zeros((N, 3), dtype=np.float32)
        async_node[:, 0] = np.log1p(out_cnt)
        async_node[:, 1] = np.log1p(in_cnt)
        avg_lag_sec = np.divide(lag_sum, in_cnt, out=np.zeros(N), where=in_cnt > 0)
        # log1p, not raw seconds: typical lag is ~2.5ms, but a rare slow
        # consumer can be seconds to minutes late (observed 223s in normal
        # data -- a real, legitimate value, not a data bug). Left as raw
        # seconds, that one row alone swamped the async attribute-decoder's
        # MSE (target 223 vs a near-zero prediction -> squared error ~50,000)
        # and was the actual cause of a handful of normal traces getting a
        # pathologically high async_trace_dis (up to 5.6, vs every anomalous
        # F02 trace staying under 0.08) -- diagnosed 2026-09-23 by tracing
        # one such trace back to this exact column. log1p(223)=5.4 keeps it
        # in the same rough range as the other two columns instead of
        # dominating them by 2 orders of magnitude.
        async_node[:, 2] = np.log1p(avg_lag_sec)
        async_adj = np.log1p(async_adj_count).astype(np.float32)

        return sync_adj, async_node, async_adj

    def _order_adj(self, spans: pd.DataFrame) -> np.ndarray:
        """Service-level precedence relation of one trace, the async graph's second
        relation (DeepTraLog's TEG "Sequence" relation at service granularity):
          [i, j] = 1  iff EVERY span of service i ended before the FIRST span of
                      service j started (directed, binary; (0, 0) between two
                      services = they overlap, i.e. ran in parallel)
          [i, i] = 1  iff service i has at least one span in the trace (a service is
                      never "before" itself, so the diagonal marks presence)
        Uses ALL spans (sync and async) and carries no durations, so it is not
        confounded with latency_dev. Consumed by async_trace_model_v3.py.
        """
        N = self.num_services
        idx = spans["Service"].astype(str).map(self.service2idx)
        lo = spans.groupby(idx)["StartTime"].min()
        hi = spans.groupby(idx)["EndTime"].max()
        ids = lo.index.values.astype(int)
        m = np.zeros((N, N), dtype=np.float32)
        m[np.ix_(ids, ids)] = (hi.values[:, None] < lo.values[None, :]).astype(np.float32)
        m[ids, ids] = 1.0
        return m

    # ── Latency baseline (normal traces, per service) ────────────────────────
    # Incremental (running sums), so Pass 1 never holds more than one run's
    # spans in memory at a time -- a first version collected every normal
    # run's span DataFrame in a list before computing this, which peaked at
    # several hundred MB for the full ~93k-trace normal pool (~5M+ span
    # rows) and got the process killed for low memory (2026-09-22).

    def _init_latency_accum(self):
        self._lat_sums = np.zeros(self.num_services, dtype=np.float64)
        self._lat_sq = np.zeros(self.num_services, dtype=np.float64)
        self._lat_cnt = np.zeros(self.num_services, dtype=np.int64)

    def _accumulate_latency(self, spans: pd.DataFrame):
        d = spans[["Service", "StartTime", "EndTime"]]
        dur_sec = (d["EndTime"].values - d["StartTime"].values).clip(min=0) / 1000.0
        svc_idx = d["Service"].map(self.service2idx).values
        for i in range(self.num_services):
            mask = svc_idx == i
            if mask.any():
                v = dur_sec[mask]
                self._lat_sums[i] += v.sum()
                self._lat_sq[i] += (v ** 2).sum()
                self._lat_cnt[i] += len(v)

    def _finalize_latency_baseline(self):
        baseline_mean = np.zeros(self.num_services, dtype=np.float32)
        baseline_std = np.full(self.num_services, 1e-6, dtype=np.float32)
        for i in range(self.num_services):
            if self._lat_cnt[i] > 1:
                mean = self._lat_sums[i] / self._lat_cnt[i]
                var = max(self._lat_sq[i] / self._lat_cnt[i] - mean ** 2, 0.0)
                baseline_mean[i] = mean
                baseline_std[i] = (var ** 0.5) + 1e-6
        self._latency_baseline = (baseline_mean, baseline_std)
        logging.info(f"  Latency baseline from {int(self._lat_cnt.sum()):,} normal spans")
        del self._lat_sums, self._lat_sq, self._lat_cnt

    # ── Reservoir sampling (bounded memory regardless of pool size) ─────────

    @staticmethod
    def _reservoir_add(reservoir: list, item, seen_count: int, capacity: int, rng: random.Random) -> int:
        """Algorithm R. `reservoir` is mutated in place; returns the updated seen_count."""
        seen_count += 1
        if len(reservoir) < capacity:
            reservoir.append(item)
        else:
            j = rng.randrange(seen_count)
            if j < capacity:
                reservoir[j] = item
        return seen_count

    # ── Entry point ───────────────────────────────────────────────────────────
    # Split into run_normal() and run_fcase() rather than one run(): each
    # fault-case zip's processing kept driving memory up on top of whatever
    # the (already large) normal-pool build had retained, and this process
    # got killed for low memory twice reaching F02 even after Pass 1/2 were
    # made streaming/reservoir-bounded (2026-09-22). Running each stage as
    # its own OS process (see main()'s --stage) is the reliable fix: process
    # exit guarantees full memory release in a way gc.collect() within one
    # long-lived process does not.

    def run_normal(self):
        """Pass 1+2 over normal/*.zip; saves train/unlabel/val/meta.pkl and a
        small cache (fitted Drain3 miner + latency baseline) for run_fcase()."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "scenarios"), exist_ok=True)

        normal_zips = sorted(glob.glob(os.path.join(self.normal_dir, "*.zip")))
        logging.info(f"Found {len(normal_zips)} normal zip(s)")

        # Pass 1: latency baseline (running sums, see _accumulate_latency) +
        # a bounded reservoir sample of raw log messages for Drain3 -- one
        # run's span/log data in memory at a time, nothing retained across
        # runs except small running accumulators.
        logging.info("Pass 1/2: reading normal runs for latency baseline + Drain3 fit ...")
        self._init_latency_accum()
        drain3_corpus: List[str] = []
        drain3_seen = 0
        for zpath in normal_zips:
            for member in self._span_members(zpath):
                df = self._read_spans(zpath, member)
                if df.empty:
                    continue
                df = df[df["Service"].astype(str).isin(self.services)]
                if not df.empty:
                    self._accumulate_latency(df)
                run_prefix = member.rsplit("/", 1)[0] + "/" if "/" in member else ""
                run_logs = self._run_folder_logs(zpath, run_prefix)
                for msgs in run_logs.values():
                    for m in msgs:
                        drain3_seen = self._reservoir_add(drain3_corpus, m, drain3_seen, self.max_drain3_messages, self.rng)
                del df, run_logs
            gc.collect()  # zip boundary: encourage the allocator to give large freed CSV/log buffers back

        self._finalize_latency_baseline()
        logging.info(f"  Drain3 corpus: {len(drain3_corpus):,} messages reservoir-sampled from {drain3_seen:,} seen")
        self._fit_drain3(drain3_corpus)
        del drain3_corpus

        # Pass 2: build normal samples, reservoir-sampling down to
        # --max_normal_traces as we go (never materialising the full
        # ~93k-trace pool at once -- that peaked at 1.3+ GB and got this
        # process killed for low memory on this 8 GB machine, 2026-09-22).
        logging.info("Pass 2/2: building normal samples (reservoir-sampled to --max_normal_traces) ...")
        normal_samples_all: List[Tuple[str, Dict]] = []
        normal_seen = 0
        # Which (i,j) service pairs ever carry a real async edge, across every
        # normal trace seen (not just the ones the reservoir happens to keep --
        # a rare-but-real edge type must not depend on sampling luck). Used to
        # restrict AsyncTraceModel's structural loss to cells that can ever be
        # nonzero: averaging MSE over all N*N=1225 pairs when only a handful
        # ever have a real edge diluted the one real signal by ~1225x and let
        # any single rare-edge misprediction swing the trace-level loss --
        # diagnosed 2026-09-23 while investigating why some normal F02 traces
        # scored as more anomalous than every real F02 anomaly.
        self._async_edge_seen = np.zeros((self.num_services, self.num_services), dtype=bool)
        for zpath in normal_zips:
            for member in self._span_members(zpath):
                for tid, sample in self._build_samples_for_run(zpath, member):
                    self._async_edge_seen |= (sample["async_msg_count_adj"] > 0)
                    normal_seen = self._reservoir_add(normal_samples_all, (tid, sample), normal_seen,
                                                        self.max_normal_traces, self.rng)
            gc.collect()
        logging.info(f"  {len(normal_samples_all)} normal samples kept (reservoir-sampled from {normal_seen:,} seen)")
        logging.info(f"  Async edge mask: {int(self._async_edge_seen.sum())} / "
                     f"{self.num_services**2} (service,service) pairs ever seen with an async edge")

        self.rng.shuffle(normal_samples_all)
        n_val = round(len(normal_samples_all) * VAL_FRACTION)
        val_samples = normal_samples_all[:n_val]
        train_samples = normal_samples_all[n_val:]
        logging.info(f"  train/unlabel: {len(train_samples)}   val: {len(val_samples)}")

        self._build_and_save(train_samples, val_samples, fcase_names=[])

        cache = {"miner": self._miner, "latency_baseline": self._latency_baseline}
        with open(os.path.join(self.output_dir, "_cache.pkl"), "wb") as f:
            pickle.dump(cache, f)
        logging.info("  Saved _cache.pkl (fitted Drain3 miner + latency baseline, for run_fcase())")
        logging.info("Normal-pool preprocessing complete. Run --stage fcase for each fault case next.")

    def run_fcase(self, fcode: str):
        """Process one fault-case zip; appends its result to scenarios/ and to
        meta.pkl's scenario_names. Loads the cache run_normal() saved, and
        the already-saved train/val.pkl as the normal pool to sample from
        (their union is exactly the --max_normal_traces reservoir)."""
        scenarios_dir = os.path.join(self.output_dir, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)

        cache_path = os.path.join(self.output_dir, "_cache.pkl")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"{cache_path} not found -- run --stage normal first")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        self._miner = cache["miner"]
        self._latency_baseline = cache["latency_baseline"]

        test_normal_pool: List[Tuple[str, Dict]] = []
        for split in ("train", "val"):
            with open(os.path.join(self.output_dir, f"{split}.pkl"), "rb") as f:
                for block_id, s in pickle.load(f).items():
                    test_normal_pool.append((block_id, s))
        logging.info(f"  Loaded normal pool: {len(test_normal_pool)} traces (train+val)")

        zpath = os.path.join(self.fault_dir, f"{fcode}.zip")
        if not os.path.exists(zpath):
            raise FileNotFoundError(zpath)

        samples = []
        for member in self._span_members(zpath):
            samples.extend(self._build_samples_for_run(zpath, member))
        anom = [(tid, s) for tid, s in samples if s["label"] == 1]
        if not anom:
            logging.warning(f"  {fcode}: no labelled anomalous traces found, nothing saved")
            return
        self._save_scenario_test(fcode, anom, test_normal_pool, scenarios_dir)

        meta_path = os.path.join(self.output_dir, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if fcode not in meta["scenario_names"]:
            meta["scenario_names"].append(fcode)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        logging.info(f"  {fcode} complete.")

    def _save_scenario_test(self, fcode, anom_samples, normal_pool, scenarios_dir):
        n_anom = len(anom_samples)
        n_normal = min(len(normal_pool),
                        round(n_anom * (1 - TARGET_ANOMALY_RATE) / TARGET_ANOMALY_RATE))
        normal_pick = self.rng.sample(normal_pool, n_normal) if n_normal < len(normal_pool) else list(normal_pool)
        combined = list(anom_samples) + list(normal_pick)
        self.rng.shuffle(combined)
        data = self._to_dict(combined)
        path = os.path.join(scenarios_dir, f"test_{fcode}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"  -> scenarios/test_{fcode}.pkl ({len(combined)} traces, "
                     f"{n_anom} anomalous, rate={n_anom/len(combined):.3f})")

    @staticmethod
    def _to_dict(samples: List[Tuple[str, Dict]]) -> Dict:
        d = {}
        for i, (tid, s) in enumerate(samples):
            block_id = hashlib.md5(f"{tid}_{i}".encode()).hexdigest()[:12]
            d[block_id] = {k: v for k, v in s.items() if not k.startswith("_")}
        return d

    def _build_and_save(self, train_samples, val_samples, fcase_names):
        train_data = self._to_dict(train_samples)
        val_data = self._to_dict(val_samples)
        for split, data in (("train", train_data), ("unlabel", train_data)):
            path = os.path.join(self.output_dir, f"{split}.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"  Saved {split}.pkl: {len(data)} normal traces")
        val_path = os.path.join(self.output_dir, "val.pkl")
        with open(val_path, "wb") as f:
            pickle.dump(val_data, f)
        logging.info(f"  Saved val.pkl: {len(val_data)} normal traces")

        meta = {
            "num_services": self.num_services,
            "service2idx": self.service2idx,
            "metric_names": [],
            "kpi_c": 1,
            "log_c": 1,
            "trace_c": TRACE_NODE_FEAT_DIM,
            "async_c": ASYNC_TRACE_NODE_FEAT_DIM,
            "async_order": True,   # samples carry async_temporal_order_adj (see _order_adj)
            "async_edge_mask": getattr(self, "_async_edge_seen", None),
            "n_log_templates": 1,
            "scenario_names": fcase_names,
        }
        with open(os.path.join(self.output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        logging.info(f"  Saved meta.pkl -> {self.output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fault_dir", help="Dir containing F01.zip .. F14.zip (stages normal/fcase/all)")
    p.add_argument("--normal_dir", help="Dir containing normal/*.zip (stages normal/fcase/all)")
    p.add_argument("--label_pkl", required=True,
                    help="Trace-label pickle: OUTPUT of --stage labels, INPUT of the other stages")
    p.add_argument("--graphdata_dir", help="Dir containing graph_data.z01..z07 and graph_data.zip (stage labels)")
    p.add_argument("--output_dir", help="Dataset output dir (stages normal/fcase/all)")
    p.add_argument("--stage", choices=["labels", "normal", "fcase", "all"], default="all",
                    help="'labels': extract per-trace labels from GraphData into --label_pkl. "
                         "'normal': build the normal pool only (train/unlabel/val/meta.pkl + a cache "
                         "for --stage fcase). 'fcase': process --fcases against that cache (run "
                         "--stage normal first). 'all': both, in one process -- fine for small/smoke "
                         "runs, but for the full dataset run each stage as its own process instead "
                         "(one per F-case for 'fcase'): a fault case's processing on top of an "
                         "already-large normal pool got this process killed for low memory twice even "
                         "after Pass 1/2 were made streaming (2026-09-22); a fresh process per stage "
                         "is the reliable fix, since process exit guarantees memory is released in a "
                         "way gc.collect() within one long-lived process does not.")
    p.add_argument("--fcases", nargs="*", default=None,
                    help="F-cases to process (e.g. F01 F02 F04 F13). Required for --stage fcase; "
                         "ignored for --stage normal.")
    p.add_argument("--max_normal_traces", type=int, default=20000)
    p.add_argument("--max_drain3_messages", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.stage == "labels":
        if not args.graphdata_dir:
            raise SystemExit("--graphdata_dir is required for --stage labels")
        labels = extract_labels(args.graphdata_dir)
        os.makedirs(os.path.dirname(os.path.abspath(args.label_pkl)), exist_ok=True)
        with open(args.label_pkl, "wb") as f:
            pickle.dump(labels, f)
        logging.info(f"Saved -> {args.label_pkl}")
        return
    for need in ("fault_dir", "normal_dir", "output_dir"):
        if not getattr(args, need):
            raise SystemExit(f"--{need} is required for --stage {args.stage}")

    pre = DeepTraLogPreprocessor(
        fault_dir=args.fault_dir, normal_dir=args.normal_dir, label_pkl=args.label_pkl,
        output_dir=args.output_dir, fcases=args.fcases,
        max_normal_traces=args.max_normal_traces, max_drain3_messages=args.max_drain3_messages,
        seed=args.seed,
    )
    if args.stage in ("normal", "all"):
        pre.run_normal()
    if args.stage in ("fcase", "all"):
        if not args.fcases:
            raise SystemExit("--fcases is required for --stage fcase")
        for fcode in args.fcases:
            pre.run_fcase(fcode)
        logging.info("All requested F-cases processed.")


if __name__ == "__main__":
    main()
