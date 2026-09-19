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
| Test per scenario | All of the scenario's anomaly windows + sampled normal windows so the anomaly rate is **12.5%** (`--target_anomaly_rate 0.125`) |
| Test sizes | `Code_Stop_MediaService` 370 (50 anomaly + 320 normal, 13.5%); `Code_Stop_TextService`/`UserService` 320 (40 + 280); `Perf_*`, `DB_Redis_*` 80 (10 + 70); `Svc_Kill_*` 32 (4 + 28) |
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
- **Test** (the normal side of each `test_<scenario>.pkl`): sampled from a pool of every scenario (`Normal_Baseline` + each scenario's recovered / never-faulted windows), taken round-robin across the source scenarios so each test file mixes normal from many sessions. The model therefore cannot rely on "which session is this" to separate the classes.

Anomaly is always kept per scenario: each `test_<scenario>.pkl` contains only that scenario's own anomaly windows (all of the available ones).

## 4. Scoring and Training Components

### 4.1 `activity_penalty_weight`
For faults that make a service go silent (`Code_Stop_*`, `Svc_Kill_*`), the reconstruction loss of anomaly windows is **lower** than that of normal windows — e.g. `Code_Stop_TextService`: mean anomaly loss 0.92 vs mean normal loss 1.29, in both the log and KPI components. A near-empty/flat input is easier to reconstruct than one with real variation. `activity_penalty_weight` adds a reconstruction-independent term to the anomaly score: how many standard deviations the current activity (`kpi_features.sum + log_features.sum`) sits below the normal training level. Default `0.0` (no effect on other datasets).

### 4.2 `gate_delta_lr_mult` (trace only)
A separate learning-rate multiplier for `trace_gate` and `delta_head`. Both final layers are zero-initialised, so each one's gradient is proportional to the other's near-zero value (a double bottleneck); a factor of 10 lets them move away from zero faster. Default `1.0`.

### 4.3 `latency_dev` clip to ±10
`latency_dev` (the 6th trace feature) is a z-score against `Normal_Baseline`; `bl_std`, estimated from few samples, can be near zero and push z-scores into the thousands. Values are clipped to [−10, 10]. It binds on 0–1% of test values (mostly the `Code_Stop_*` files) and 0% of train/val values.

### 4.4 `epoches` / `patience`
50 50 / 15 for both configurations.

## 5. Commands

```bash
cd D:/UAM-AD
python codes/common/preprocess_sn.py --sn_data_root D:/AnoMod/SN_data --output_dir data/sn \
    --window_sec 30 --target_anomaly_rate 0.125 --seed 42

cd codes
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
| Code_Stop_MediaService | 0.899 | 0.831 | 0.980 | 0.899 | 0.831 | 0.980 | 0.000 |
| Code_Stop_TextService | 0.867 | 0.765 | 1.000 | **0.897** | 0.812 | 1.000 | +0.030 |
| Code_Stop_UserService | 0.879 | 0.784 | 1.000 | 0.857 | 0.765 | 0.975 | −0.022 |
| DB_Redis_CacheLimit_HomeTimeline | 0.692 | 0.529 | 1.000 | **0.762** | 0.667 | 0.889 | +0.070 |
| DB_Redis_CacheLimit_SocialGraph | 0.900 | 0.818 | 1.000 | 0.900 | 0.818 | 1.000 | 0.000 |
| DB_Redis_CacheLimit_UserTimeline | 0.667 | 0.583 | 0.778 | **0.762** | 0.667 | 0.889 | +0.095 |
| Perf_CPU_Contention | 0.667 | 0.533 | 0.889 | **0.783** | 0.643 | 1.000 | +0.116 |
| Perf_Disk_IO_Stress | 0.769 | 0.625 | 1.000 | **0.833** | 0.714 | 1.000 | +0.064 |
| Perf_Network_Loss | 0.636 | 0.583 | 0.700 | **0.952** | 0.909 | 1.000 | +0.316 |
| Svc_Kill_Media | 0.444 | 0.400 | 0.500 | **0.889** | 0.800 | 1.000 | +0.445 |
| Svc_Kill_SocialGraph | 0.444 | 0.333 | 0.667 | **0.750** | 0.600 | 1.000 | +0.306 |
| Svc_Kill_UserTimeline | 0.800 | 1.000 | 0.667 | **1.000** | 1.000 | 1.000 | +0.200 |
| **Mean** | **0.722** | 0.649 | 0.848 | **0.857** | 0.769 | 0.978 | **+0.135** |
| Std of F1 | 0.154 | | | 0.077 | | | |

Mean F1 rises from 0.722 (baseline) to 0.857 (trace). Trace is higher in 9/12 scenarios, equal in 2 (`Code_Stop_MediaService`, `DB_Redis_CacheLimit_SocialGraph`) and lower in 1 (`Code_Stop_UserService`, −0.022).

### Call-flow fault group (service kill / stop)

| Scenario | Baseline F1 | Trace F1 | Δ |
|---|---:|---:|:---:|
| Code_Stop_MediaService | 0.899 | 0.899 | 0.000 |
| Code_Stop_TextService | 0.867 | 0.897 | +0.030 |
| Code_Stop_UserService | 0.879 | 0.857 | −0.022 |
| Svc_Kill_Media | 0.444 | 0.889 | +0.445 |
| Svc_Kill_SocialGraph | 0.444 | 0.750 | +0.306 |
| Svc_Kill_UserTimeline | 0.800 | 1.000 | +0.200 |
| **Mean** | **0.722** | **0.882** | **+0.160** |

Mean F1 of this group rises from 0.722 to 0.882. `Code_Stop_*` (whole session is anomaly, strong signal): baseline already reaches 0.87–0.90, so trace differs by −0.02 to +0.03. `Svc_Kill_*` (~2-minute signal): baseline 0.44–0.80, trace 0.75–1.00.

## 7. Ablation

### 7.1 Trace (12 scenarios)

| Config | epochs/patience | `gate_delta_lr_mult` | Mean F1 (12) | Call-flow mean F1 (6) | vs baseline at same epochs |
|:---|:---:|:---:|---:|---:|:---:|
| Baseline | 50/15 | – | 0.722 | 0.722 | – |
| Trace | 50/15 | 1 | 0.796 | 0.813 | 10 win / 1 tie / 1 lose |
| Trace | 10/5 | 10 | 0.823 | 0.821 | 10 win / 1 tie / 1 lose |
| **Trace** | 50/15 | 10 | **0.857** | **0.882** | 9 win / 2 tie / 1 lose |

Raising epochs alone (10/5 → 50/15) or `gate_delta_lr_mult` alone (1 → 10) already puts trace above baseline (12-scenario F1 0.796 and 0.823 vs 0.722); using both reaches 0.857.

### 7.2 Other components (baseline, 10 epochs)

| `activity_penalty_weight` | 0 | 1.0 | 1.5 |
|---|---:|---:|---:|
| Call-flow mean F1 (6) | 0.105 | 0.425 | 0.628 |
| Mean F1 (12) | 0.321 | 0.530 | 0.657 |

Baseline (weight 1.5): `epoches/patience` 10/5 → 50/15 raises the 12-scenario mean F1 from 0.657 to 0.722. The `latency_dev` clip (§4.3) was not ablated separately (would need re-preprocessing).

## 8. Limitations

- **Small test files**: 10 of 12 scenarios have only 32–80 windows; `Svc_Kill_*` has just 4 anomaly windows, so one flipped window moves F1 by more than 0.1. F1 also depends on the number of normal windows and the anomaly rate of the test file, so it is comparable only within this setup.
- **One seed** (`run_end 1`); small differences (≤ 0.03 on `Code_Stop_*`) are within noise.
- **Val has only 8 windows**, so the 95th-percentile threshold is unstable.
- **Train/test overlap**: the test normal pool includes a few `Normal_Baseline` windows (which are also in train/val).
- With one real `Normal_Baseline` session, the training set stays small (31 windows).
- `gate_delta_lr_mult` has no counterpart on the baseline side; this remains one asymmetry between the two configurations.

## 9. Related Files

| What | File |
|---|---|
| `FAULT_WINDOWS`, two normal pools, `target_anomaly_rate`, `latency_dev` clip | `codes/common/preprocess_sn.py` |
| `activity_penalty_weight`, `gate_delta_lr_mult` | `codes/models/basev3.py`, `codes/run.py` |
| Flag forwarding via wrapper | `codes/common/eval_per_scenario_sn.py` |
| Checkpoints | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
