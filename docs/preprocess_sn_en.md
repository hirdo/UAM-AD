# Preprocessing: SocialNetwork (AnoMod) Dataset

## 1. Dataset Overview

The SocialNetwork (SN) dataset is part of the **AnoMod benchmark** for cloud microservice anomaly detection. It was collected from a 12-service microservice application deployed on a real cluster. The dataset captures three modalities:

| Modality    | Source                            | Location       |
|:------------|:----------------------------------|:---------------|
| KPI metrics | System + container + Jaeger spans | `metric_data/` |
| Logs        | Per-service log files             | `log_data/`    |
| Traces      | Distributed traces (Jaeger)       | `trace_data/`  |

### Scenarios

The 13 sessions were recorded back to back in one ~3-hour run (`Normal_Baseline` first). Fault mechanisms come from AnoMod's collection script (`github.com/EvoTestOps/AnoMod`, `automated_multimodal_collection.sh`), where collection starts 15 s after the fault is injected:

| Type                  | Count | Mechanism (per script)                                          | Anomaly window (from session start) |
|:----------------------|------:|:----------------------------------------------------------------|:------------------------------------|
| Normal_Baseline       |     1 | Steady-state traffic, no fault                                  | – (whole session normal)            |
| Code_Stop_*           |     3 | `docker stop` of the service container, not restarted          | whole session                       |
| DB_Redis_CacheLimit_* |     3 | ChaosBlade Redis cache limit, `--timeout 300`                   | 0–300 s                             |
| Perf_CPU_Contention   |     1 | ChaosBlade CPU stress, `--timeout 300`                          | 0–300 s                             |
| Perf_Disk_IO_Stress   |     1 | ChaosBlade disk I/O stress, `--timeout 300`                     | 0–300 s                             |
| Perf_Network_Loss     |     1 | ChaosBlade packet loss, `--timeout 300`                         | 0–300 s                             |
| Svc_Kill_*            |     3 | ChaosBlade process kill (SIGKILL) + Docker auto-restart         | 90–210 s                            |

The `Svc_Kill_*` window is confirmed from the data: the `container_label_restartcount` column flips 0→1 at t=105 s in all 3 scenarios (absent in `Normal_Baseline`), and `user-timeline-service` has a ~75 s silent gap (101.8 s → 176.8 s) in its traces. The rest of each session (after the anomaly window) is treated as normal.

Each session records approximately **19.5–25 minutes**.

---

## 2. Feature Engineering

### 2.1 Windowing

Raw time-series data is divided into **non-overlapping windows** of `window_sec=30` seconds, and each window is one data point. Window 0 starts exactly at the first timestamp of the session; no warm-up period is skipped, because anomaly labels are computed as offsets from the session start (§3.1).

**Why 30 seconds?**
- Short enough to capture transient anomalies (service kills manifest within seconds)
- Long enough to produce stable aggregated statistics (avoid noise from individual metric reads)

`Normal_Baseline` (19.5 min) yields **39 windows**; the other sessions yield about 39–50 windows each.

### 2.2 KPI Features (59 dimensions)

| Group     | Count | Features                                                                                                                                      |
|:----------|------:|:----------------------------------------------------------------------------------------------------------------------------------------------|
| System    |    10 | cpu_usage, disk_io_time, disk_read_bytes, disk_usage_pct, disk_write_bytes, load1, memory_usage_pct, network_errors, network_receive_bytes, network_transmit_bytes |
| Container |    48 | 12 services × 4 metrics: cpu, memory, net_rx, net_tx                                                                                         |
| Jaeger    |     1 | spans_rate (result="ok", normalized per window)                                                                                               |

Each KPI is aggregated per window using the mean value of all samples within that 30-second interval. Missing values (e.g., a container not yet started) are filled with column means computed from non-NaN windows.

### 2.3 Log Features

Logs are parsed using **Drain3** (streaming template miner), fitted on Normal_Baseline logs only:
- Templates learned: ~458 from 317,055 log messages
- Feature type: `template_appear` — binary presence/absence of each template in the window
- New templates encountered in anomaly scenarios are treated as unseen (mapped to a special "unknown" bucket)

**Why fit Drain3 on Normal_Baseline only?**  
To avoid contamination from anomaly log patterns when building the vocabulary. The model should learn to flag unseen templates as anomalous, which is only possible if the template vocabulary was built from normal logs.

### 2.4 Trace Features (optional, `open_trace=True`)

For each service, a 6-dimensional feature vector is computed per window (`trace_c=6`):

```
[call_count, avg_duration_us, max_duration_us, error_rate, root_rate, latency_dev]
```

- `call_count`: number of trace spans involving this service in the window
- `avg_duration_us` / `max_duration_us`: latency statistics (normalized to seconds)
- `error_rate`: fraction of spans with non-OK HTTP status codes
- `root_rate`: fraction of spans that are root spans (entry points)
- `latency_dev`: z-score of `avg_duration` vs per-service baseline from Normal_Baseline
  = `(avg_dur − mean_baseline) / (std_baseline + 1e-6)` — positive means slower than normal, **clipped to [−10, 10]**

`latency_dev` baseline is computed once from **Normal_Baseline** `all_traces.csv` (per-service mean and std of `duration_us / 1e6`), then applied to all scenarios uniformly. The clip is needed because `std_baseline` is estimated from few samples and can be near zero, which would otherwise push a genuine latency spike to a z-score in the thousands and let one node dominate the reconstruction loss.

A **static adjacency matrix** (12×12) is built from Normal_Baseline traces: edge (i, j) = 1 if service i calls service j at least once. This graph is fixed for all scenarios — we assume the call graph topology does not change between experiments.

---

## 3. Labels and Splits

### 3.1 Normal / anomaly labels

Every scenario is split into anomaly and normal windows by the fault windows in §1 (`FAULT_WINDOWS` in `preprocess_sn.py`); the recovered or never-faulted part of a session is normal.

### 3.2 Train / Val split

```
Normal_Baseline (39 windows)                 → train.pkl = unlabel.pkl (all 39)
Every other session's normal windows (281)   → per source session: 20% → val.pkl (57 windows)
                                                                    80% → test normal pool (224 windows)
```

Val is unseen during training and is used for model selection and the threshold (§4). It comes from the same mix of sessions as the test normals but is disjoint from them, and it holds 57 windows → (57 − 5) × 5 = 260 scores (train/val use a sliding window of 5).

Train stays narrow on purpose. Adding low-activity windows from other scenarios to training makes the model treat "low activity" as normal, which breaks detection of "went completely silent" faults (`Code_Stop_*`, `Svc_Kill_*`): an almost-empty window is easier to reconstruct than a busy one, so its reconstruction loss ends up lower than that of normal windows.

### 3.3 Test files (per scenario)

```
test_{scenario}.pkl = the scenario's anomaly windows (at most --max_anomalies = 39, thinned evenly over time)
                    + normal windows drawn from the 224-window test pool
                    → shuffled
```

- **Anomaly**: only that scenario's own anomaly windows; only `Code_Stop_*` (40–50 available) is thinned to 39.
- **Normal**: drawn round-robin across the source sessions, so each test file mixes normal windows from many recording sessions. A single fixed normal session would let a model separate the classes by "which session is this" instead of by the fault.
- **Size**: normals = anomaly × (1 − r) / r with r = `--target_anomaly_rate` (default 0.125), limited by the 224-window pool:

| Scenario | Anomaly | Normal | Total | Rate |
|:--|--:|--:|--:|--:|
| `Code_Stop_*` (3 files) | 39 | 224 | 263 | 14.8% |
| `DB_Redis_*`, `Perf_*` (6 files) | 10 | 70 | 80 | 12.5% |
| `Svc_Kill_*` (3 files) | 4 | 28 | 32 | 12.5% |

**Why shuffle?** The model receives a mixed stream (as in production) and must score each window individually; without shuffling all anomaly windows would sit at the end.

---

## 4. Thresholds and F1

The full protocol is in `docs/evaluation_protocol_en.md`. In short:

- **Model selection**: the epoch with the lowest mean val score (loss, plus the activity term when `--activity_penalty_weight > 0`).
- **Threshold**: `np.percentile(val_scores, 95)` of the selected model (`--val_percentile`, default 95).
- **Primary metric**: precision / recall / F1 at that threshold (no `point_adjust`, no test labels), plus threshold-free AUROC / AUPRC.
- **Oracle F1** (threshold swept on test labels with `point_adjust`) is saved in separate `oracle_*` fields.

`docs/experiment_results_sn_trace_vs_baseline_en.md` reports these results.

---

## 5. Output Structure

```
data/sn/
├── train.pkl              # 39 normal windows (Normal_Baseline)
├── unlabel.pkl            # same as train, for the GAN unlabeled phase
├── val.pkl                # 57 normal windows (other sessions), for model selection and the threshold
├── meta.pkl               # dataset metadata (adj matrix, feature dims, etc.)
└── scenarios/
    ├── test_Code_Stop_MediaService_20251104_024819.pkl   # 263 windows (39 anomaly)
    ├── test_Code_Stop_TextService_20251104_022416.pkl    # 263 (39)
    ├── test_Code_Stop_UserService_20251104_020019.pkl    # 263 (39)
    ├── test_DB_Redis_CacheLimit_HomeTimeline_20251104_004905.pkl   # 80 (10)
    ├── test_DB_Redis_CacheLimit_SocialGraph_20251104_013615.pkl
    ├── test_DB_Redis_CacheLimit_UserTimeline_20251104_011238.pkl
    ├── test_Perf_CPU_Contention_20251103_222601.pkl
    ├── test_Perf_Disk_IO_Stress_20251103_231335.pkl
    ├── test_Perf_Network_Loss_20251103_224954.pkl
    ├── test_Svc_Kill_Media_20251104_000111.pkl           # 32 (4)
    ├── test_Svc_Kill_SocialGraph_20251104_002506.pkl
    └── test_Svc_Kill_UserTimeline_20251103_233717.pkl
```

`*.pkl` files are git-ignored and regenerated with the commands below.

---

## 6. Usage

```bash
python codes/common/preprocess_sn.py \
    --sn_data_root D:/AnoMod/SN_data \
    --output_dir data/sn \
    --window_sec 30 \
    --target_anomaly_rate 0.125 \
    --max_anomalies 39 \
    --seed 42
```

### Key Parameters

| Parameter               |  Value | Rationale                                                                 |
|:------------------------|-------:|:--------------------------------------------------------------------------|
| `--window_sec`          |     30 | 30-second granularity: captures transient anomalies, stable aggregates    |
| `--target_anomaly_rate` |  0.125 | Anomaly fraction of each test file; normal sampled up to the pool size    |
| `--max_anomalies`       |     39 | Cap per test file so the 224-window pool still gives about 15% for `Code_Stop_*` |
| `--seed`                |     42 | Reproducibility: controls shuffle and sampling                            |
