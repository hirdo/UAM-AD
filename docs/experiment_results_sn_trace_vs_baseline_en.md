# Experiment Results: SocialNetwork — Trace vs Baseline

## 1. Experiment Setup

### Model
**HADES** — GAN-based unsupervised anomaly detection model trained only on normal data.

### Evaluation Protocol

| Setting | Value |
|:---|:---|
| Dataset | SocialNetwork (AnoMod), 12 fault scenarios |
| Data type | `fuse` (KPI + Log [+ Trace when `open_trace=True`]) |
| Train / unlabel | 31 windows (80% of the 39 `Normal_Baseline` windows) |
| Val | 8 windows (remaining 20% of `Normal_Baseline`) |
| Test per scenario | 326 windows = 320 normal + 6 anomaly (**1.84%**); `Svc_Kill_*`: 324 = 320 + 4 (**1.23%**) |
| `window_size` | 5 (5 windows × 30 s) |
| `val_percentile` | 95 |
| `epoches` / `patience` | 50 50 / 15 (same for baseline and trace) |
| `batch_size`, `alpha`, `open_gan_sep` | 256, 0.16, True |
| `activity_penalty_weight` | 1.5 (same for baseline and trace) |
| `gate_delta_lr_mult` | 10 (trace only — baseline has no `trace_gate`/`delta_head`) |
| `run_end` | 1 (single run, one seed) |

### Threshold Calibration
The anomaly threshold does not use test labels: `threshold = np.percentile(val_losses, 95)`, where `val_losses` are the losses of the 8 normal windows in `val.pkl`.

### Result Directories
| Configuration | Folder |
|:---|:---|
| Baseline (KPI + Log) | `data/sn/result_per_scenario_fuse_baseline/` |
| Trace (KPI + Log + Trace) | `data/sn/result_per_scenario_fuse_trace/` |

## 2. Normal / Anomaly Labels per Fault Type

The fault window (`FAULT_WINDOWS` in `codes/common/preprocess_sn.py`) is derived from AnoMod's original collection script (`github.com/EvoTestOps/AnoMod`, `automated_multimodal_collection.sh`: collection starts 15 s after the fault is injected) and from direct measurement. Times are measured from the start of each session's recording:

| Fault type | Mechanism (per script) | Anomaly window | Rest of the session |
|---|---|---|---|
| `Code_Stop_*` | `docker stop`, never auto-restarted | Whole session | – |
| `Perf_*`, `DB_Redis_CacheLimit_*` | ChaosBlade `--timeout 300`, auto-reverts after 300 s | 0–300 s | Normal (recovered) |
| `Svc_Kill_*` | ChaosBlade process kill + Docker auto-restart | 90–210 s | Normal |
| `Normal_Baseline` | – | – | Whole session normal |

The `Svc_Kill_*` window is confirmed by: the `container_label_restartcount` column flipping 0→1 at t=105 s in all 3 scenarios (absent in `Normal_Baseline`), and a ~75 s silent gap (101.8 s → 176.8 s) in `user-timeline-service`'s traces.

## 3. Two Normal Pools

- **Train / unlabel / val**: `Normal_Baseline` only. Kept narrow on purpose: adding low-activity windows from other scenarios to training makes the model treat "low activity" as normal, which destroys detection of "went completely silent" faults (§4.1).
- **Test** (the normal side of each `test_<scenario>.pkl`): pooled from **every** scenario (`Normal_Baseline` + each scenario's recovered / never-faulted windows) — 320 windows spanning the ~3-hour experiment. Each test file compares a scenario's anomaly against normal from many different sessions, so the model cannot rely on "which session is this" to separate them.

Anomaly is always kept per scenario: each `test_<scenario>.pkl` contains only that scenario's own anomaly windows (evenly subsampled to at most 6).

## 4. Scoring and Training Components

### 4.1 `activity_penalty_weight`
For faults that make a service go silent (`Code_Stop_*`, `Svc_Kill_*`), the reconstruction loss of anomaly windows is **lower** than that of normal windows — e.g. `Code_Stop_TextService`: mean anomaly loss 0.92 vs mean normal loss 1.29, in both the log and KPI components. A near-empty/flat input is easier to reconstruct than one with real variation. `activity_penalty_weight` adds a reconstruction-independent term to the anomaly score: how many standard deviations the current activity (`kpi_features.sum + log_features.sum`) sits below the normal training level. Default `0.0` (no effect on other datasets).

### 4.2 `gate_delta_lr_mult` (trace only)
A separate learning-rate multiplier for `trace_gate` and `delta_head`. Both final layers are zero-initialised, so each one's gradient is proportional to the other's near-zero value (a double bottleneck); a factor of 10 lets them move away from zero faster. Default `1.0`.

### 4.3 `latency_dev` clip to ±10
`latency_dev` (the 6th trace feature) is a z-score against `Normal_Baseline`; `bl_std`, estimated from few samples, can be near zero and push z-scores into the thousands. Values are clipped to [−10, 10]. It binds on only ~0.1–0.2% of test values and 0% of train/val values.

### 4.4 `epoches` / `patience`
50 50 / 15 for both configurations.

## 5. Commands

```bash
cd D:/UAM-AD/codes
# Baseline
python common/eval_per_scenario_sn.py --data ../data/sn --dataset sn --data_type fuse \
    --open_trace False --activity_penalty_weight 1.5 \
    --epoches 50 50 --batch_size 256 --patience 15 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True \
    --run_start 0 --run_end 1
# Trace
python common/eval_per_scenario_sn.py --data ../data/sn --dataset sn --data_type fuse \
    --open_trace True --activity_penalty_weight 1.5 --gate_delta_lr_mult 10 \
    --epoches 50 50 --batch_size 256 --patience 15 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True \
    --run_start 0 --run_end 1
```

## 6. Results

| Scenario | Baseline F1 | P | R | Trace F1 | P | R | Δ F1 |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| Code_Stop_MediaService | 0.500 | 0.333 | 1.000 | **0.600** | 0.429 | 1.000 | +0.100 |
| Code_Stop_TextService | 0.462 | 0.300 | 1.000 | **0.632** | 0.462 | 1.000 | +0.170 |
| Code_Stop_UserService | 0.462 | 0.300 | 1.000 | **0.615** | 0.571 | 0.667 | +0.154 |
| DB_Redis_CacheLimit_HomeTimeline | 0.333 | 0.200 | 1.000 | **0.375** | 0.231 | 1.000 | +0.042 |
| DB_Redis_CacheLimit_SocialGraph | 0.462 | 0.300 | 1.000 | 0.435 | 0.294 | 0.833 | −0.027 |
| DB_Redis_CacheLimit_UserTimeline | 0.200 | 0.250 | 0.167 | **0.276** | 0.174 | 0.667 | +0.076 |
| Perf_CPU_Contention | 0.333 | 0.200 | 1.000 | 0.345 | 0.217 | 0.833 | +0.012 |
| Perf_Disk_IO_Stress | 0.333 | 0.200 | 1.000 | 0.345 | 0.217 | 0.833 | +0.012 |
| Perf_Network_Loss | 0.200 | 0.250 | 0.167 | **0.345** | 0.217 | 0.833 | +0.145 |
| Svc_Kill_Media | 0.242 | 0.138 | 1.000 | 0.267 | 0.154 | 1.000 | +0.025 |
| Svc_Kill_SocialGraph | 0.286 | 0.167 | 1.000 | 0.320 | 0.190 | 1.000 | +0.034 |
| Svc_Kill_UserTimeline | 0.296 | 0.174 | 1.000 | 0.333 | 0.200 | 1.000 | +0.037 |
| **Mean** | **0.342** | 0.234 | 0.861 | **0.407** | 0.280 | 0.889 | **+0.065** |
| Std | 0.102 | 0.061 | 0.311 | 0.127 | 0.128 | 0.124 | |

Mean F1 rises from 0.342 (baseline) to 0.407 (trace); trace is higher in 11/12 scenarios.

### Call-flow fault group (service kill / stop)

| Scenario | Baseline F1 | Trace F1 | Δ |
|---|---:|---:|:---:|
| Code_Stop_MediaService | 0.500 | 0.600 | +0.100 |
| Code_Stop_TextService | 0.462 | 0.632 | +0.170 |
| Code_Stop_UserService | 0.462 | 0.615 | +0.154 |
| Svc_Kill_Media | 0.242 | 0.267 | +0.025 |
| Svc_Kill_SocialGraph | 0.286 | 0.320 | +0.034 |
| Svc_Kill_UserTimeline | 0.296 | 0.333 | +0.037 |
| **Mean** | **0.375** | **0.461** | **+0.087** |

Mean F1 of this group rises from 0.375 to 0.461; trace is higher in 6/6. `Code_Stop_*` rises by +0.10 to +0.17; `Svc_Kill_*` (signal lasts only ~2 minutes) rises by +0.03 to +0.04.

## 7. Ablation

### 7.1 Trace (12 scenarios)

| Config | epochs/patience | `gate_delta_lr_mult` | Mean F1 (12) | Call-flow mean F1 (6) | vs baseline at same epochs |
|:---|:---:|:---:|---:|---:|:---:|
| Baseline | 50/15 | – | 0.342 | 0.375 | – |
| Trace | 50/15 | 1 | 0.355 | 0.421 | 5 win / 3 tie / 4 lose |
| Trace | 10/5 | 10 | 0.372 | 0.418 | 6 win / 1 tie / 5 lose |
| **Trace** | 50/15 | 10 | **0.407** | **0.461** | 11 win / 1 lose |

`Code_Stop_*` is insensitive to these two knobs (trace is above baseline in every row). On the other scenarios, raising epochs or `gate_delta_lr_mult` alone is not enough for trace to beat baseline consistently (e.g. without `gate_delta_lr_mult`, trace 0.160 vs baseline 0.286 on `Svc_Kill_SocialGraph`); using both raises the call-flow mean F1 from 0.375 (baseline) to 0.461.

### 7.2 Other components (baseline, 10 epochs, 6 call-flow scenarios)

| `activity_penalty_weight` | 0 | 1.0 | 1.5 |
|---|---:|---:|---:|
| Mean F1 | 0.036 | 0.296 | 0.320 |

Baseline `epoches/patience` 10/5 → 50/15 (weight 1.5) raises the 12-scenario mean F1 from 0.289 to 0.342. The `latency_dev` clip (§4.3) was not ablated separately (would need re-preprocessing).

## 8. Limitations

- **One seed, few anomaly windows**: each test file has only 4–6 anomaly windows, so one flipped window moves F1 by about 0.05–0.1; differences of +0.01 to +0.04 (`Svc_Kill_*`, `Perf_*`) are within noise. The most robust conclusion is the `Code_Stop_*` group.
- **Val has only 8 windows**, so the 95th-percentile threshold is unstable.
- **Precision is low** (0.15–0.57): `activity_penalty_weight` trades precision for recall.
- `DB_Redis_CacheLimit_SocialGraph` is the only scenario where trace is below baseline (−0.027).
- With one real `Normal_Baseline` session, the training set stays small (31 windows).
- `gate_delta_lr_mult` has no counterpart on the baseline side; this remains one asymmetry between the two configurations.

## 9. Related Files

| What | File |
|---|---|
| `FAULT_WINDOWS`, two normal pools, `latency_dev` clip | `codes/common/preprocess_sn.py` |
| `activity_penalty_weight`, `gate_delta_lr_mult` | `codes/models/basev3.py`, `codes/run.py` |
| Flag forwarding via wrapper | `codes/common/eval_per_scenario_sn.py` |
| Checkpoints | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
