"""
Preprocess SocialNetwork (AnoMod) → UAM-AD pkl format.

Dataset layout expected under SN_DATA_ROOT/:
  log_data/    {scenario}_logs_{timestamp}/    *.log files per service
  metric_data/ {scenario}_metrics_{timestamp}/ *.csv (system + container + jaeger)
  trace_data/  {scenario}_traces_{timestamp}/  all_traces.csv

Anomaly labeling: each scenario is split into (anomaly, normal) windows using
FAULT_WINDOWS below, derived from AnoMod's actual injection timing rather than a
blanket "skip N minutes, everything after is anomaly" rule (see FAULT_WINDOWS
docstring for the per-fault-type rationale, citing the collection script).

Two separate normal pools:
  - train/unlabel/val: Normal_Baseline only. Kept narrow/homogeneous on purpose
    — pooling in every scenario's recovered/never-faulted windows here taught
    the model that low-activity windows are normal too, which backfires for
    "service went silent" faults (reconstructing a near-empty window is
    *easier* than a busy one, so a dead-service window stopped scoring as
    anomalous at all once quiet-but-normal and quiet-because-dead blurred
    together).
  - scenarios/test_{name}.pkl's normal side: pooled from EVERY scenario
    (Normal_Baseline + each scenario's recovered/never-faulted windows), so
    evaluation still spans many different recording times instead of one
    ~20min slice, without changing what the model was trained to reconstruct.

Output (OUTPUT_DIR/):
  train.pkl              — 80% (shuffled) of Normal_Baseline's windows (for training)
  unlabel.pkl            — same as train.pkl
  val.pkl                — remaining 20% (unseen normal, used to compute anomaly
                           threshold without test leakage)
  meta.pkl               — dataset metadata
  scenarios/
    test_{name}.pkl      — per-scenario test file, one per scenario that has a
                           real fault window (see FAULT_WINDOWS): all of that
                           scenario's anomaly windows + normal windows sampled
                           from the pool of all scenarios (round-robin across
                           source scenarios) to reach target_anomaly_rate
                           (default 0.125), shuffled.

KPI features (59 total):
  10 system   : cpu_usage, disk_io_time, disk_read_bytes, disk_usage_pct,
                disk_write_bytes, load1, memory_usage_pct, network_errors,
                network_receive_bytes, network_transmit_bytes
  48 container: 12 services x 4 metrics (cpu, memory, net_rx, net_tx)
   1 jaeger   : spans_rate (result="ok")

Trace node features per service (5-dim):
  [call_count, avg_dur_us, max_dur_us, error_rate, root_rate]

Static adjacency [12x12]: built from Normal_Baseline traces (bool call graph).

Usage:
    python codes/common/preprocess_sn.py \\
        --sn_data_root D:/AnoMod/SN_data \\
        --output_dir data/sn \\
        --window_sec 30 \\
        --target_anomaly_rate 0.125 --seed 42
"""

import argparse
import hashlib
import logging
import os
import pickle
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Constants ────────────────────────────────────────────────────────────────

# 12 canonical services (from Jaeger available_services.json)
SN_SERVICES = [
    "compose-post-service",
    "home-timeline-service",
    "media-service",
    "nginx-web-server",
    "post-storage-service",
    "social-graph-service",
    "text-service",
    "unique-id-service",
    "url-shorten-service",
    "user-mention-service",
    "user-service",
    "user-timeline-service",
]

# Container metric label for each trace service (most are identical; nginx is different)
_CONTAINER_LABEL = {svc: svc for svc in SN_SERVICES}
_CONTAINER_LABEL["nginx-web-server"] = "nginx-thrift"

# 10 system metric files (one scalar per window)
SYSTEM_METRIC_FILES = [
    "system_cpu_usage.csv",
    "system_disk_io_time.csv",
    "system_disk_read_bytes.csv",
    "system_disk_usage_percent.csv",
    "system_disk_write_bytes.csv",
    "system_load1.csv",
    "system_memory_usage_percent.csv",
    "system_network_errors.csv",
    "system_network_receive_bytes.csv",
    "system_network_transmit_bytes.csv",
]

# 4 container metric files (one value per service per window)
CONTAINER_METRIC_FILES = [
    "socialnet_container_cpu.csv",
    "socialnet_container_memory.csv",
    "socialnet_container_network_receive.csv",
    "socialnet_container_network_transmit.csv",
]
CONTAINER_LABEL_COL = "container_label_com_docker_compose_service"

TRACE_NODE_FEAT_DIM = 6  # [call_count, avg_dur_us, max_dur_us, error_rate, root_rate, latency_dev]

# Per-scenario fault window (start_sec, end_sec) relative to when data collection
# starts for that session. Windows starting inside this range get label=1;
# everything else in that same session (before start_sec, at/after end_sec, or
# the whole session when the scenario has no entry / entry is None) gets label=0
# and is folded into the shared normal pool. Derived from AnoMod's actual
# injection timing (automated_multimodal_collection.sh, github.com/EvoTestOps/AnoMod):
# collect_data() starts 15s AFTER inject_anomaly() (10s ANOMALY_EFFECT_WAIT +
# 5s POST_TEST_WAIT). Matched by scenario-name prefix.
#   - Code_Stop_*: `docker stop <container>`, not restarted until end-of-run
#     cleanup -> fault is active for the entire collected session.
#   - Perf_*/DB_Redis_CacheLimit_*: ChaosBlade `--timeout 300` -> auto-reverts
#     300s after injection = ~285s after collection starts; using a flat 300s
#     (5 min) here is a small conservative margin, not a precise cutoff.
#   - Svc_Kill_*: ChaosBlade `process kill --signal 9`, no --timeout, relies on
#     Docker's own restart policy. Empirically verified directly (not assumed)
#     via the `container_label_restartcount` label in socialnet_container_memory
#     .csv: for all 3 Svc_Kill_* scenarios it flips 0->1 at exactly t=105s into
#     the collected session (Normal_Baseline never gets this label at all, i.e.
#     restartcount stays 0 the whole time -> not a generic artifact). Cross-
#     checked against raw span gaps in all_traces.csv: Svc_Kill_UserTimeline's
#     target service goes fully silent for ~75s (101.8s -> 176.8s), right at
#     the restart marker, versus a normal ~5-10s inter-span gap for that
#     service elsewhere in the session. (90, 210) below is a 2-minute window
#     bracketing that observed gap with margin on both sides.
FAULT_WINDOWS: Dict[str, Optional[Tuple[float, Optional[float]]]] = {
    "Code_Stop_MediaService":           (0.0, None),
    "Code_Stop_TextService":            (0.0, None),
    "Code_Stop_UserService":            (0.0, None),
    "Perf_CPU_Contention":              (0.0, 300.0),
    "Perf_Disk_IO_Stress":              (0.0, 300.0),
    "Perf_Network_Loss":                (0.0, 300.0),
    "DB_Redis_CacheLimit_HomeTimeline": (0.0, 300.0),
    "DB_Redis_CacheLimit_SocialGraph":  (0.0, 300.0),
    "DB_Redis_CacheLimit_UserTimeline": (0.0, 300.0),
    "Svc_Kill_Media":                   (90.0, 210.0),
    "Svc_Kill_SocialGraph":             (90.0, 210.0),
    "Svc_Kill_UserTimeline":            (90.0, 210.0),
}


def _fault_window_for(scenario_name: str) -> Optional[Tuple[float, Optional[float]]]:
    """Look up FAULT_WINDOWS by matching scenario_name's prefix."""
    for prefix, window in FAULT_WINDOWS.items():
        if scenario_name.startswith(prefix):
            return window
    raise KeyError(f"No FAULT_WINDOWS entry matches scenario '{scenario_name}'")

# Log timestamp format: [YYYY-Mon-DD HH:MM:SS.ffffff] <LEVEL>: ...
_LOG_TS_RE  = re.compile(
    r'^\[(\d{4}-\w{3}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\]'
)
_LOG_LVL_RE = re.compile(r'<(info|warn|warning|error|debug|trace)>', re.IGNORECASE)
_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_log_ts(ts_str: str) -> Optional[datetime]:
    """Parse '[YYYY-Mon-DD HH:MM:SS.ffffff]' style timestamp."""
    try:
        date_part, time_part = ts_str.strip().split()
        y, mon, d = date_part.split("-")
        month = _MONTHS.get(mon, 0)
        if month == 0:
            return None
        h, mi, rest = time_part.split(":")
        # rest may be 'SS.ffffff' or just 'SS'
        if "." in rest:
            s, us = rest.split(".")
            us = int(us[:6].ljust(6, "0"))
        else:
            s, us = rest, 0
        return datetime(int(y), month, int(d), int(h), int(mi), int(s), us)
    except Exception:
        return None


# ─── Preprocessor ─────────────────────────────────────────────────────────────

class SNPreprocessor:
    """
    Preprocess SocialNetwork (AnoMod) dataset → UAM-AD pkl format.
    """

    def __init__(
        self,
        sn_data_root: str,
        output_dir: str,
        window_sec: int = 30,
        target_anomaly_rate: float = 0.125,
        seed: int = 42,
    ):
        self.sn_data_root        = sn_data_root
        self.output_dir          = output_dir
        self.window_sec          = window_sec
        self.target_anomaly_rate = target_anomaly_rate
        self.seed                = seed
        self.rng                 = random.Random(seed)

        self.services    = SN_SERVICES
        self.num_services = len(self.services)
        self.service2idx  = {s: i for i, s in enumerate(self.services)}

        self.log_dir    = os.path.join(sn_data_root, "log_data")
        self.metric_dir = os.path.join(sn_data_root, "metric_data")
        self.trace_dir  = os.path.join(sn_data_root, "trace_data")

        self._miner            = None   # Drain3 TemplateMiner fitted on Normal_Baseline
        self._adj              = None   # Static adjacency [12, 12] from Normal_Baseline
        self._latency_baseline = None   # Tuple(mean[N], std[N]) of avg_dur/1e6 from Normal_Baseline

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _scenario_dirs(self) -> Dict[str, Dict[str, str]]:
        """
        Map each scenario name → {log_dir, metric_dir, trace_dir}.
        Scenario name is derived from the folder name up to the first '_metrics' etc.
        Uses the folder name prefix (before _logs/_metrics/_traces).
        """
        def _strip_suffix(name: str, suffix: str) -> str:
            idx = name.find(suffix)
            return name[:idx] if idx != -1 else name

        metrics_folders = {_strip_suffix(d, "_metrics"): d
                           for d in os.listdir(self.metric_dir)
                           if os.path.isdir(os.path.join(self.metric_dir, d))}
        log_folders     = {_strip_suffix(d, "_logs"): d
                           for d in os.listdir(self.log_dir)
                           if os.path.isdir(os.path.join(self.log_dir, d))}
        trace_folders   = {_strip_suffix(d, "_traces"): d
                           for d in os.listdir(self.trace_dir)
                           if os.path.isdir(os.path.join(self.trace_dir, d))}

        common = sorted(set(metrics_folders) & set(log_folders) & set(trace_folders))
        logging.info(f"Found {len(common)} scenarios: {common}")
        return {
            sc: {
                "metric_dir": os.path.join(self.metric_dir, metrics_folders[sc]),
                "log_dir":    os.path.join(self.log_dir,    log_folders[sc]),
                "trace_dir":  os.path.join(self.trace_dir,  trace_folders[sc]),
            }
            for sc in common
        }

    def _window_starts(self, t_min: datetime, t_max: datetime) -> List[datetime]:
        """Window 0 starts exactly at t_min (not floored to a wall-clock grid
        boundary): FAULT_WINDOWS offsets are computed relative to t_min in
        _process_scenario, so flooring here would shift window 0 to *before*
        t_min, giving it a negative offset that incorrectly fails fault-window
        checks like `offset_sec >= fw_start` for fw_start=0.0."""
        delta   = timedelta(seconds=self.window_sec)
        current = t_min.replace(microsecond=0)
        wins = []
        while current < t_max:
            wins.append(current)
            current += delta
        return wins

    # ── Step 1: Build static adjacency from Normal_Baseline ──────────────────

    def _build_static_adj(self, trace_dir: str) -> np.ndarray:
        """Build boolean adjacency matrix [N, N] from trace spans."""
        adj = np.zeros((self.num_services, self.num_services), dtype=np.float32)
        path = os.path.join(trace_dir, "all_traces.csv")
        if not os.path.exists(path):
            logging.warning(f"  No all_traces.csv in {trace_dir}, returning zero adj")
            return adj

        df = pd.read_csv(path, usecols=["span_id", "parent_span_id", "service"])
        span2svc = dict(zip(df["span_id"], df["service"]))

        for _, row in df.iterrows():
            child_svc  = row["service"]
            parent_sid = row["parent_span_id"]
            if pd.isna(parent_sid):
                continue
            parent_svc = span2svc.get(parent_sid)
            if parent_svc is None or parent_svc == child_svc:
                continue
            ci = self.service2idx.get(child_svc,  -1)
            pi = self.service2idx.get(parent_svc, -1)
            if ci >= 0 and pi >= 0:
                adj[pi, ci] = 1.0  # parent → child
                adj[ci, pi] = 1.0  # undirected

        # Self-loops
        np.fill_diagonal(adj, 1.0)
        n_edges = int(adj.sum()) - self.num_services
        logging.info(f"  Static adj built: {n_edges} edges (symmetric) from {len(df)} spans")
        return adj

    # ── Step 1b: Build per-service latency baseline from Normal_Baseline ────────

    def _build_latency_baseline(self, trace_dir: str) -> None:
        """
        Compute per-service mean and std of avg span duration (in seconds) from
        Normal_Baseline all_traces.csv.  Stored in self._latency_baseline as
        (mean [N], std [N]).  Used by _build_trace_node_features to compute
        latency_dev (col 5) as a z-score for every scenario.
        """
        baseline_mean = np.zeros(self.num_services, dtype=np.float32)
        baseline_std  = np.full(self.num_services, 1e-6, dtype=np.float32)

        path = os.path.join(trace_dir, "all_traces.csv")
        if not os.path.exists(path):
            logging.warning("  No all_traces.csv — latency_dev will be zero for all windows")
            self._latency_baseline = (baseline_mean, baseline_std)
            return

        df = pd.read_csv(path, usecols=["service", "duration_us"])
        df = df.dropna(subset=["service", "duration_us"])
        df["service_idx"] = df["service"].map(self.service2idx)
        df = df[df["service_idx"].notna()].copy()
        df["dur_sec"] = pd.to_numeric(df["duration_us"], errors="coerce").fillna(0) / 1e6

        for i in range(self.num_services):
            vals = df[df["service_idx"] == i]["dur_sec"].values
            if len(vals) > 1:
                baseline_mean[i] = float(vals.mean())
                baseline_std[i]  = float(vals.std()) + 1e-6

        logging.info(
            f"  Latency baseline from {len(df):,} spans — "
            f"mean dur/svc: {baseline_mean.mean()*1000:.2f} ms"
        )
        self._latency_baseline = (baseline_mean, baseline_std)

    # ── Step 2: Fit Drain3 on Normal_Baseline logs ───────────────────────────

    def _fit_drain3(self, log_dir: str) -> None:
        """Read all *.log files in log_dir and fit a Drain3 TemplateMiner."""
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except ImportError:
            logging.warning("drain3 not installed — falling back to regex templates")
            self._miner = None
            return

        config = TemplateMinerConfig()
        config.drain_depth          = 4
        config.drain_sim_th         = 0.5
        config.drain_max_children   = 100
        config.parametrize_numeric_tokens = True

        miner = TemplateMiner(config=config)
        n_msgs = 0

        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith(".log"):
                continue
            fpath = os.path.join(log_dir, fname)
            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    content = self._extract_log_content(line)
                    if content:
                        miner.add_log_message(content)
                        n_msgs += 1

        n_tmpl = len(miner.drain.id_to_cluster)
        logging.info(f"  Drain3 fitted on {n_msgs:,} messages → {n_tmpl} templates")
        self._miner = miner

    def _extract_log_content(self, line: str) -> Optional[str]:
        """Extract message content (after '] ') from a log line."""
        # Format: [YYYY-Mon-DD HH:MM:SS.ffffff] <level>: ...
        m = re.match(r'^\[[\d\w\-\s:\.]+\]\s*', line)
        if m:
            return line[m.end():].strip() or None
        return line.strip() or None

    def _to_template(self, line: str) -> str:
        content = self._extract_log_content(line)
        if not content:
            return "padding"
        if self._miner is not None:
            result = self._miner.add_log_message(content)
            return result["template_mined"] if result else content
        # Fallback: basic regex normalisation
        return re.sub(r'[0-9a-f]{8,}', '<*>', re.sub(r'\d+', '<NUM>', content))

    # ── Step 3: Per-window KPI features ──────────────────────────────────────

    def _load_system_metrics(self, metric_dir: str) -> pd.DataFrame:
        """
        Load 10 system metric CSVs, aggregate to 15-s samples, return
        DataFrame with DatetimeIndex and 10 columns.
        """
        dfs = {}
        for fname in SYSTEM_METRIC_FILES:
            path = os.path.join(metric_dir, fname)
            col  = fname.replace("system_", "").replace(".csv", "")
            if not os.path.exists(path):
                logging.warning(f"    Missing system metric: {fname}")
                continue
            df = pd.read_csv(path, usecols=["timestamp", "value"],
                             parse_dates=["timestamp"])
            df = df.dropna(subset=["value"])
            df = df.groupby("timestamp")["value"].mean()  # resolve duplicate timestamps
            df.name = col
            dfs[col] = df

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs.values(), axis=1).sort_index()

    def _load_container_metrics(self, metric_dir: str) -> pd.DataFrame:
        """
        Load 4 container metric CSVs (cpu, memory, net_rx, net_tx),
        filter to 12 SN services, pivot to 48 columns: {metric}_{service}.
        """
        all_frames = []
        metric_names_map = {
            "socialnet_container_cpu.csv":              "container_cpu",
            "socialnet_container_memory.csv":           "container_memory",
            "socialnet_container_network_receive.csv":  "container_net_rx",
            "socialnet_container_network_transmit.csv": "container_net_tx",
        }

        for fname, metric_name in metric_names_map.items():
            path = os.path.join(metric_dir, fname)
            if not os.path.exists(path):
                logging.warning(f"    Missing container metric: {fname}")
                continue
            df = pd.read_csv(path, usecols=["timestamp", "value", CONTAINER_LABEL_COL],
                             parse_dates=["timestamp"])
            df = df.rename(columns={CONTAINER_LABEL_COL: "service"})
            # Map container labels to canonical service names
            inv_map = {v: k for k, v in _CONTAINER_LABEL.items()}
            df["service"] = df["service"].map(lambda s: inv_map.get(s, s))
            # Keep only the 12 canonical services
            df = df[df["service"].isin(self.services)]
            df["col_name"] = metric_name + "__" + df["service"]
            pivot = df.pivot_table(index="timestamp", columns="col_name",
                                   values="value", aggfunc="mean")
            all_frames.append(pivot)

        if not all_frames:
            return pd.DataFrame()
        result = pd.concat(all_frames, axis=1).sort_index()
        return result

    def _load_jaeger_metric(self, metric_dir: str) -> pd.Series:
        """Load jaeger_spans_rate.csv, sum 'ok' result per timestamp."""
        path = os.path.join(metric_dir, "jaeger_spans_rate.csv")
        if not os.path.exists(path):
            logging.warning(f"    Missing jaeger_spans_rate.csv")
            return pd.Series(dtype=float, name="jaeger_spans_ok")
        df = pd.read_csv(path, usecols=["timestamp", "value", "result"],
                         parse_dates=["timestamp"])
        ok = df[df["result"] == "ok"].set_index("timestamp")["value"]
        ok.name = "jaeger_spans_ok"
        return ok

    def _build_kpi_matrix(
        self,
        metric_dir: str,
        win_starts: List[datetime],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build [W, 59] KPI matrix for a scenario's metric directory.
        Returns (matrix, metric_names).
        """
        W = len(win_starts)
        delta = timedelta(seconds=self.window_sec)

        # Load all metric dataframes
        sys_df  = self._load_system_metrics(metric_dir)
        cont_df = self._load_container_metrics(metric_dir)
        jae_sr  = self._load_jaeger_metric(metric_dir)

        # Build a unified time-indexed DataFrame
        parts = []
        if not sys_df.empty:
            parts.append(sys_df)
        if not cont_df.empty:
            parts.append(cont_df)
        if not jae_sr.empty:
            parts.append(jae_sr.to_frame())

        if not parts:
            metric_names = []
            return np.zeros((W, 0), dtype=np.float32), metric_names

        all_metrics = pd.concat(parts, axis=1).sort_index()
        metric_names = list(all_metrics.columns)

        # For each window, compute mean over [t_start, t_start+window_sec)
        matrix = np.zeros((W, len(metric_names)), dtype=np.float32)

        ts_arr = all_metrics.index.values  # datetime64
        vals   = all_metrics.values.astype(np.float32)

        for i, t_start in enumerate(win_starts):
            t_end = t_start + delta
            t0 = np.datetime64(t_start)
            t1 = np.datetime64(t_end)
            mask = (ts_arr >= t0) & (ts_arr < t1)
            if mask.any():
                window_vals = vals[mask]
                # column-wise mean, ignore NaN
                col_means = np.nanmean(window_vals, axis=0)
                # replace remaining NaN with 0
                col_means = np.where(np.isnan(col_means), 0.0, col_means)
                matrix[i] = col_means.astype(np.float32)

        return matrix, metric_names

    # ── Step 4: Per-window trace node features ────────────────────────────────

    def _build_trace_node_features(
        self,
        trace_dir: str,
        win_starts: List[datetime],
    ) -> np.ndarray:
        """
        Build [W, N, 6] trace node features.
        Features per service per window:
          [call_count, avg_dur_us, max_dur_us, error_rate, root_rate, latency_dev]
        latency_dev = z-score of avg_dur vs Normal_Baseline per service.
        """
        W = len(win_starts)
        N = self.num_services
        result = np.zeros((W, N, TRACE_NODE_FEAT_DIM), dtype=np.float32)

        path = os.path.join(trace_dir, "all_traces.csv")
        if not os.path.exists(path):
            return result

        df = pd.read_csv(
            path,
            usecols=["span_id", "parent_span_id", "service",
                     "start_time", "duration_us", "http_status_code"],
            parse_dates=["start_time"],
        )
        df = df.dropna(subset=["service"])
        df["service_idx"] = df["service"].map(self.service2idx)
        df = df[df["service_idx"].notna()].copy()
        df["service_idx"] = df["service_idx"].astype(int)

        # is_error: http_status_code >= 400 or NaN (treat as ok)
        df["is_error"] = df["http_status_code"].apply(
            lambda x: 1 if (pd.notna(x) and float(x) >= 400) else 0
        )
        # is_root: no parent_span_id
        df["is_root"] = df["parent_span_id"].isna().astype(int)

        ts_arr = df["start_time"].values  # datetime64

        delta = timedelta(seconds=self.window_sec)

        for i, t_start in enumerate(win_starts):
            t_end = t_start + delta
            t0 = np.datetime64(t_start)
            t1 = np.datetime64(t_end)
            mask = (ts_arr >= t0) & (ts_arr < t1)
            if not mask.any():
                continue
            win_df = df[mask]

            for svc_idx in range(N):
                sdf = win_df[win_df["service_idx"] == svc_idx]
                if sdf.empty:
                    continue
                n   = len(sdf)
                dur = sdf["duration_us"].fillna(0).values
                result[i, svc_idx, 0] = float(n)                           # call_count
                result[i, svc_idx, 1] = float(dur.mean())                  # avg_dur_us
                result[i, svc_idx, 2] = float(dur.max())                   # max_dur_us
                result[i, svc_idx, 3] = float(sdf["is_error"].sum()) / n   # error_rate
                result[i, svc_idx, 4] = float(sdf["is_root"].sum()) / n    # root_rate

        # Normalise call_count (log1p) and durations (/ 1e6 → seconds)
        result[:, :, 0] = np.log1p(result[:, :, 0]) / 10.0
        result[:, :, 1] = result[:, :, 1] / 1e6
        result[:, :, 2] = result[:, :, 2] / 1e6

        # col 5: latency_dev = z-score of avg_dur vs Normal_Baseline per service.
        # Clipped to [-10, 10] — bl_std is estimated from only ~40 Normal_Baseline
        # windows, so for a low-variance service (e.g. a reverse proxy with a very
        # stable baseline latency) it can be near-zero, and a genuine latency spike
        # during a real anomaly then produces a z-score in the thousands (observed:
        # nginx-web-server hit ~5000 during Code_Stop_* scenarios), which dominates
        # the MSE-based reconstruction loss for that node and drowns out every other
        # service's signal. ±10 comfortably covers the natural range seen on
        # scenarios without this pathology (observed max ~14).
        if self._latency_baseline is not None:
            bl_mean, bl_std = self._latency_baseline
            result[:, :, 5] = np.clip((result[:, :, 1] - bl_mean) / bl_std, -10.0, 10.0)

        return result

    # ── Step 5: Per-window log features ──────────────────────────────────────

    def _load_log_windows(
        self,
        log_dir: str,
        win_starts: List[datetime],
    ) -> List[List[str]]:
        """
        Read all .log files, parse timestamps, assign templates to windows.
        Returns list of length W, each element is a list of template strings.
        """
        W = len(win_starts)
        delta = timedelta(seconds=self.window_sec)

        # Collect (timestamp, template) from all log files
        records: List[Tuple[datetime, str]] = []

        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith(".log"):
                continue
            fpath = os.path.join(log_dir, fname)
            service_name = fname.replace("_.log", "").replace(".log", "")

            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = _LOG_TS_RE.match(line)
                    if not m:
                        continue
                    ts = _parse_log_ts(m.group(1))
                    if ts is None:
                        continue
                    tmpl = self._to_template(line)
                    records.append((ts, f"{service_name}|{tmpl}"))

        if not records:
            return [["padding"]] * W

        records.sort(key=lambda x: x[0])
        ts_list   = [r[0] for r in records]
        tmpl_list = [r[1] for r in records]

        win_logs = []
        for i in range(W):
            t_start = win_starts[i]
            t_end   = t_start + delta
            batch = [tmpl_list[j] for j, ts in enumerate(ts_list)
                     if t_start <= ts < t_end]
            win_logs.append(batch if batch else ["padding"])

        return win_logs

    # ── Step 6: Compute compact log_features vector ───────────────────────────

    _SVC_ABBREVS = {
        "ComposePostService": "compose-post-service",
        "HomeTimelineService": "home-timeline-service",
        "MediaService": "media-service",
        "NginxThrift": "nginx-web-server",
        "PostStorageService": "post-storage-service",
        "SocialGraphService": "social-graph-service",
        "TextService": "text-service",
        "UniqueIdService": "unique-id-service",
        "UrlShortenService": "url-shorten-service",
        "UserMentionService": "user-mention-service",
        "UserService": "user-service",
        "UserTimelineService": "user-timeline-service",
    }

    def _compute_log_features(self, msgs: List[str]) -> np.ndarray:
        """
        6-dim compact log feature vector:
        [error_rate, warn_rate, info_rate, retry_rate,
         service_diversity_norm, log_count_norm]
        """
        N_FEATS = 6
        real = [m for m in msgs if m and m != "padding"]
        if not real:
            return np.zeros(N_FEATS, dtype=np.float32)

        n = len(real)
        error_cnt = warn_cnt = info_cnt = retry_cnt = 0
        services_seen = set()

        for m in real:
            lv = _LOG_LVL_RE.search(m)
            if lv:
                lvl = lv.group(1).lower()
                if lvl == "error":              error_cnt += 1
                elif lvl in ("warn", "warning"): warn_cnt  += 1
                elif lvl == "info":              info_cnt  += 1
            if "retry" in m.lower():
                retry_cnt += 1
            # Service name comes from the prefix before '|'
            parts = m.split("|")
            if parts:
                svc = parts[0].strip()
                if svc:
                    services_seen.add(svc)

        feat = np.array([
            error_cnt / n,
            warn_cnt  / n,
            info_cnt  / n,
            retry_cnt / n,
            len(services_seen) / self.num_services,
            min(np.log1p(n) / 8.0, 1.0),
        ], dtype=np.float32)
        return feat

    # ── Step 7: Process one scenario → windows ────────────────────────────────

    def _process_scenario(
        self,
        scenario_name: str,
        dirs: Dict[str, str],
        fault_window_sec: Optional[Tuple[float, Optional[float]]],
    ) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, Dict]]]:
        """
        Process one scenario and split its windows into (anomaly_samples, normal_samples)
        based on fault_window_sec = (start_sec, end_sec), the fault-active range relative
        to this session's own start (see FAULT_WINDOWS). A window is anomaly=1 iff its
        start falls in [start_sec, end_sec) (end_sec=None means "to session end").
        fault_window_sec=None means no fault window at all -> every window is normal.
        """
        logging.info(f"  Processing scenario: {scenario_name}")
        metric_dir = dirs["metric_dir"]
        log_dir    = dirs["log_dir"]
        trace_dir  = dirs["trace_dir"]

        # Determine time range from system CPU metric (most reliable)
        cpu_path = os.path.join(metric_dir, "system_cpu_usage.csv")
        if not os.path.exists(cpu_path):
            logging.warning(f"    No system_cpu_usage.csv, skipping")
            return [], []

        cpu_df = pd.read_csv(cpu_path, usecols=["timestamp"], parse_dates=["timestamp"])
        t_min  = cpu_df["timestamp"].min()
        t_max  = cpu_df["timestamp"].max()
        logging.info(f"    Time range: {t_min} → {t_max} "
                     f"({(t_max - t_min).total_seconds() / 60:.1f} min)")

        win_starts = self._window_starts(t_min, t_max)
        W = len(win_starts)
        if W == 0:
            logging.warning(f"    No windows, skipping")
            return [], []

        # KPI matrix
        kpi_matrix, metric_names = self._build_kpi_matrix(metric_dir, win_starts)
        if not hasattr(self, "_metric_names"):
            self._metric_names = metric_names
            logging.info(f"    KPI features: {len(metric_names)}")

        # Trace node features
        node_feats = self._build_trace_node_features(trace_dir, win_starts)

        # Log windows
        win_log_lists = self._load_log_windows(log_dir, win_starts)

        # Assemble samples, splitting by fault_window_sec
        anomaly_samples: List[Tuple[str, Dict]] = []
        normal_samples:  List[Tuple[str, Dict]] = []
        fw_start, fw_end = fault_window_sec if fault_window_sec is not None else (None, None)

        for i in range(W):
            t_start  = win_starts[i]
            offset_sec = (t_start - t_min).total_seconds()
            is_anomaly = (
                fault_window_sec is not None
                and offset_sec >= fw_start
                and (fw_end is None or offset_sec < fw_end)
            )
            label = 1 if is_anomaly else 0

            block_id = hashlib.md5(
                f"{scenario_name}_{t_start}".encode()
            ).hexdigest()[:12]

            msgs    = win_log_lists[i]
            log_feat = self._compute_log_features(msgs)

            sample = {
                "label":               label,
                "kpi_label":           label,
                "log_label":           label,
                "kpis":                kpi_matrix[i].copy(),
                "logs":                msgs,
                "seqs":                msgs,
                "log_features":        log_feat,
                "trace_node_features": node_feats[i].copy(),   # [N, 5]
                "trace_adj":           self._adj.copy(),        # [N, N]
                "_scenario":           scenario_name,           # metadata (not used by model)
                "_t_start":            t_start,
            }
            (anomaly_samples if is_anomaly else normal_samples).append((block_id, sample))

        logging.info(f"    → {len(anomaly_samples)} anomaly + {len(normal_samples)} normal windows assembled")
        return anomaly_samples, normal_samples

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        scenario_dirs = self._scenario_dirs()

        # Identify Normal_Baseline
        normal_keys = [k for k in scenario_dirs if "Normal_Baseline" in k]
        other_keys = [k for k in scenario_dirs if "Normal_Baseline" not in k]
        if not normal_keys:
            raise ValueError("No Normal_Baseline scenario found in data root.")
        normal_key = normal_keys[0]
        logging.info(f"Normal scenario: {normal_key}")
        logging.info(f"Other scenarios ({len(other_keys)}): {other_keys}")

        # Step 1: Build static adjacency from Normal_Baseline
        logging.info("Step 1: Building static adjacency from Normal_Baseline traces …")
        self._adj = self._build_static_adj(scenario_dirs[normal_key]["trace_dir"])

        # Step 1b: Build per-service latency baseline from Normal_Baseline traces
        logging.info("Step 1b: Building latency baseline from Normal_Baseline traces …")
        self._build_latency_baseline(scenario_dirs[normal_key]["trace_dir"])

        # Step 2: Fit Drain3 on Normal_Baseline logs
        logging.info("Step 2: Fitting Drain3 on Normal_Baseline logs …")
        self._fit_drain3(scenario_dirs[normal_key]["log_dir"])

        # Step 3: Process every scenario, splitting each into (anomaly, normal)
        # windows per FAULT_WINDOWS. Two separate normal pools are kept:
        #   - baseline_normal_samples: Normal_Baseline only. Used for
        #     train/unlabel/val, i.e. what the model actually learns "normal"
        #     from. Kept narrow/homogeneous on purpose: pooling in the more
        #     varied recovered/never-faulted windows here taught the model
        #     that low-activity windows are normal too, which backfires for
        #     "service went silent" faults (Code_Stop_*) — reconstructing a
        #     near-empty window is *easier* than a busy one, so once the
        #     model treats quiet-but-normal and quiet-because-dead the same
        #     way, the dead-service windows stop scoring as anomalous at all
        #     (verified: their reconstruction loss came out *lower* than
        #     normal windows', producing F1=0 despite a real, strong fault).
        #   - test_normal_pool: every scenario's normal-labeled windows
        #     pooled together. Used only to fill out test_<scenario>.pkl's
        #     normal side, so evaluation still spans many different
        #     recording times/sessions instead of one ~20min slice — this is
        #     what breaks the "which session is this" shortcut a
        #     single-source normal pool would otherwise hand a baseline
        #     model. Test-time-only, so it doesn't affect what the model
        #     itself was trained to reconstruct.
        logging.info("Step 3: Processing all scenarios (anomaly/normal split) …")
        _, baseline_normal_samples = self._process_scenario(normal_key, scenario_dirs[normal_key], None)
        test_normal_pool = list(baseline_normal_samples)

        anomaly_by_scenario: Dict[str, List[Tuple[str, Dict]]] = {}
        for sc in other_keys:
            fw = _fault_window_for(sc)
            anom, norm = self._process_scenario(sc, scenario_dirs[sc], fw)
            test_normal_pool.extend(norm)
            if anom:
                anomaly_by_scenario[sc] = anom
            else:
                logging.info(f"    (no fault window for {sc} -> contributes normal only, no test file)")

        # Step 4: Save each scenario's test file now that the full normal pool
        # (Normal_Baseline + every scenario's recovered/never-faulted windows)
        # is assembled.
        logging.info("Step 4: Saving per-scenario test files …")
        os.makedirs(self.output_dir, exist_ok=True)
        scenarios_dir = os.path.join(self.output_dir, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)
        for sc, samples in anomaly_by_scenario.items():
            self._save_scenario_test(sc, samples, test_normal_pool, scenarios_dir)

        anomaly_samples: List[Tuple[str, Dict]] = [
            s for sc_samples in anomaly_by_scenario.values() for s in sc_samples
        ]

        logging.info(f"\nBaseline (train) normal windows : {len(baseline_normal_samples)}")
        logging.info(f"Test normal pool                : {len(test_normal_pool)}")
        logging.info(f"Anomaly windows : {len(anomaly_samples)} "
                     f"({len(anomaly_by_scenario)} scenarios)")

        # Step 5: Save train / unlabel / meta — from Normal_Baseline only
        logging.info("Step 5: Saving train/unlabel/meta …")
        self._build_and_save(baseline_normal_samples, anomaly_by_scenario)

        logging.info("Preprocessing complete.")

    # ── Save ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(samples: List[Tuple[str, Dict]]) -> Dict:
        d = {}
        for block_id, s in samples:
            d[block_id] = {k: v for k, v in s.items() if not k.startswith("_")}
        return d

    def _save_scenario_test(
        self,
        sc_name: str,
        sc_samples: List[Tuple[str, Dict]],
        normal_samples: List[Tuple[str, Dict]],
        scenarios_dir: str,
    ):
        """Save one per-scenario test file.

        Uses every anomaly window of the scenario and samples just enough
        normal windows from the pooled normal set to reach target_anomaly_rate
        (capped at the pool size). The normal windows are drawn round-robin
        across their source scenarios so the file still mixes many sessions.
        """
        sc_samples_sub = sc_samples
        n_anom = len(sc_samples_sub)
        n_normal = min(len(normal_samples),
                       round(n_anom * (1 - self.target_anomaly_rate) / self.target_anomaly_rate))

        by_source: Dict[str, List[Tuple[str, Dict]]] = {}
        for item in normal_samples:
            by_source.setdefault(item[1]["_scenario"], []).append(item)
        for group in by_source.values():
            self.rng.shuffle(group)
        chosen: List[Tuple[str, Dict]] = []
        while len(chosen) < n_normal:
            for group in by_source.values():
                if group and len(chosen) < n_normal:
                    chosen.append(group.pop())

        combined = chosen + sc_samples_sub
        self.rng.shuffle(combined)
        sc_data = self._to_dict(combined)

        n_anom    = len(sc_samples_sub)
        anom_rate = n_anom / len(combined)
        safe_name = re.sub(r'[^A-Za-z0-9_]', '_', sc_name)
        path = os.path.join(scenarios_dir, f"test_{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(sc_data, f)
        logging.info(f"  -> Saved scenarios/test_{safe_name}.pkl  "
                     f"({len(combined)} windows, {n_anom} anomaly, rate={anom_rate:.2f})")

    def _build_and_save(
        self,
        normal_samples: List[Tuple[str, Dict]],
        anomaly_by_scenario: Dict[str, List[Tuple[str, Dict]]],
    ):
        """Save train.pkl, unlabel.pkl, val.pkl, meta.pkl.

        normal_samples is now pooled from many different scenarios/sessions
        (see run()), not just one chronological recording, so a positional
        first-80%/last-20% split would just split by which scenario happened
        to be processed first/last rather than by anything meaningful. Shuffle
        (seeded) before splitting 80% → train/unlabel, 20% → val (unseen normal).
        Val is used to compute anomaly threshold without test data leakage.
        """
        shuffled = list(normal_samples)
        self.rng.shuffle(shuffled)
        n_val   = max(1, round(len(shuffled) * 0.2))
        n_train = len(shuffled) - n_val
        train_samples = shuffled[:n_train]
        val_samples   = shuffled[n_train:]

        train_data = self._to_dict(train_samples)
        val_data   = self._to_dict(val_samples)

        for split, data in (("train", train_data), ("unlabel", train_data)):
            path = os.path.join(self.output_dir, f"{split}.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"  Saved {split}.pkl: {len(data)} normal windows")

        val_path = os.path.join(self.output_dir, "val.pkl")
        with open(val_path, "wb") as f:
            pickle.dump(val_data, f)
        logging.info(f"  Saved val.pkl: {len(val_data)} normal windows (unseen, for threshold)")

        metric_names = getattr(self, "_metric_names", [])
        meta = {
            "num_services":    self.num_services,
            "service2idx":     self.service2idx,
            "metric_names":    metric_names,
            "kpi_c":           len(metric_names),
            "log_c":           1,
            "trace_c":         TRACE_NODE_FEAT_DIM,
            "window_sec":      self.window_sec,
            "n_log_templates": 1,
            "scenario_names":  list(anomaly_by_scenario.keys()),
        }
        meta_path = os.path.join(self.output_dir, "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        logging.info(f"  Saved meta.pkl → {self.output_dir}")
        logging.info(
            f"\n  Evaluate with:\n"
            f"  python codes/common/eval_per_scenario_sn.py"
            f" --data {self.output_dir} --dataset sn --data_type fuse"
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Preprocess SocialNetwork (AnoMod) → UAM-AD pkl format"
    )
    p.add_argument(
        "--sn_data_root",
        default=r"D:\AnoMod\SN_data",
        help="Root directory containing log_data/, metric_data/, trace_data/",
    )
    p.add_argument(
        "--output_dir",
        default=r"D:\UAM-AD\data\sn",
        help="Output directory for pkl files",
    )
    p.add_argument("--window_sec",           default=30,  type=int,
                   help="Window size in seconds (default: 30s; metrics sampled at 15s)")
    p.add_argument("--target_anomaly_rate",  default=0.125, type=float,
                   help="Anomaly fraction of each scenario test file. All of the scenario's "
                        "anomaly windows are kept; normal windows are sampled to reach this rate "
                        "(capped at the pooled normal size).")
    p.add_argument("--seed",                 default=42,  type=int)
    args = p.parse_args()

    SNPreprocessor(
        sn_data_root        = args.sn_data_root,
        output_dir          = args.output_dir,
        window_sec          = args.window_sec,
        target_anomaly_rate = args.target_anomaly_rate,
        seed                = args.seed,
    ).run()


if __name__ == "__main__":
    main()
