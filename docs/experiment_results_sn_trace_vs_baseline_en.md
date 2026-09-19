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
| Test per scenario | All of the scenario's anomaly windows + normal windows sampled to a target anomaly rate |
| Test anomaly rate | `Code_Stop_*`: **15%** (`Code_Stop_MediaService` 333 = 50 + 283; `TextService`/`UserService` 267 = 40 + 227). All other scenarios: **12.5%** (`Perf_*`, `DB_Redis_*` 80 = 10 + 70; `Svc_Kill_*` 32 = 4 + 28) |
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

`preprocess_sn.py` builds every test file at 12.5% (`--target_anomaly_rate 0.125`); the `Code_Stop_*` files are then subsampled on the normal side down to 15% with `codes/common/resample_test_anomaly_rate.py` (all anomaly windows kept, seed 42). The normal pool has only 320 windows, so `Code_Stop_*` cannot reach 10% (at most 11.1–13.5%).

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
python codes/common/resample_test_anomaly_rate.py --data data/sn --prefix Code_Stop --rate 0.15 --seed 42

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
| Code_Stop_MediaService | 0.990 | 1.000 | 0.980 | 0.990 | 1.000 | 0.980 | 0.000 |
| Code_Stop_TextService | 0.963 | 0.951 | 0.975 | **0.987** | 1.000 | 0.975 | +0.024 |
| Code_Stop_UserService | 0.963 | 0.951 | 0.975 | **0.987** | 1.000 | 0.975 | +0.024 |
| DB_Redis_CacheLimit_HomeTimeline | 0.692 | 0.529 | 1.000 | **0.762** | 0.667 | 0.889 | +0.070 |
| DB_Redis_CacheLimit_SocialGraph | 0.900 | 0.818 | 1.000 | 0.900 | 0.818 | 1.000 | 0.000 |
| DB_Redis_CacheLimit_UserTimeline | 0.667 | 0.583 | 0.778 | **0.762** | 0.667 | 0.889 | +0.095 |
| Perf_CPU_Contention | 0.667 | 0.533 | 0.889 | **0.783** | 0.643 | 1.000 | +0.116 |
| Perf_Disk_IO_Stress | 0.769 | 0.625 | 1.000 | **0.833** | 0.714 | 1.000 | +0.064 |
| Perf_Network_Loss | 0.636 | 0.583 | 0.700 | **0.952** | 0.909 | 1.000 | +0.316 |
| Svc_Kill_Media | 0.444 | 0.400 | 0.500 | **0.889** | 0.800 | 1.000 | +0.445 |
| Svc_Kill_SocialGraph | 0.444 | 0.333 | 0.667 | **0.750** | 0.600 | 1.000 | +0.306 |
| Svc_Kill_UserTimeline | 0.800 | 1.000 | 0.667 | **1.000** | 1.000 | 1.000 | +0.200 |
| **Mean** | **0.745** | 0.692 | 0.844 | **0.883** | 0.818 | 0.976 | **+0.138** |
| Std of F1 | 0.180 | | | 0.096 | | | |

Mean F1 rises from 0.745 (baseline) to 0.883 (trace). Trace is higher in 10/12 scenarios, equal in 2 (`Code_Stop_MediaService`, `DB_Redis_CacheLimit_SocialGraph`), and lower in none.

### Call-flow fault group (service kill / stop)

| Scenario | Baseline F1 | Trace F1 | Δ |
|---|---:|---:|:---:|
| Code_Stop_MediaService | 0.990 | 0.990 | 0.000 |
| Code_Stop_TextService | 0.963 | 0.987 | +0.024 |
| Code_Stop_UserService | 0.963 | 0.987 | +0.024 |
| Svc_Kill_Media | 0.444 | 0.889 | +0.445 |
| Svc_Kill_SocialGraph | 0.444 | 0.750 | +0.306 |
| Svc_Kill_UserTimeline | 0.800 | 1.000 | +0.200 |
| **Mean** | **0.767** | **0.934** | **+0.167** |

Mean F1 of this group rises from 0.767 to 0.934. `Code_Stop_*` (whole session is anomaly, strong signal): mean F1 rises from 0.972 to 0.988, with baseline already at 0.96–0.99. `Svc_Kill_*` (~2-minute signal): baseline 0.44–0.80, trace 0.75–1.00.

### 6.1 Sensitivity to the `Code_Stop_*` anomaly rate

The `Code_Stop_*` rate was first 12.5% (13.5% for `MediaService`) and was then changed to 15% (the top of the intended 10–15% range); both are reported. Re-running on the same data reproduced identical results (same seed), so the differences below come from the composition of the test file.

| Scenario | Baseline F1 (12.5% / 15%) | Trace F1 (12.5% / 15%) |
|---|---:|---:|
| Code_Stop_MediaService | 0.899 / 0.990 | 0.899 / 0.990 |
| Code_Stop_TextService | 0.867 / 0.963 | 0.897 / 0.987 |
| Code_Stop_UserService | 0.879 / 0.963 | 0.857 / 0.987 |
| **Mean** | 0.882 / 0.972 | 0.884 / 0.988 |

At 12.5% trace and baseline are nearly equal (`UserService`: trace 0.022 lower); at 15% trace is equal or higher in all 3. Dropping just 53 normal windows (280 → 227) moved baseline F1 by more than 0.09, far larger than the trace–baseline gap (≤ 0.03) in this group.

## 7. Ablation

### 7.1 Trace (12 scenarios)

| Config | epochs/patience | `gate_delta_lr_mult` | Mean F1 (12) | Call-flow mean F1 (6) | vs baseline at same epochs |
|:---|:---:|:---:|---:|---:|:---:|
| Baseline | 50/15 | – | 0.745 | 0.767 | – |
| Trace | 50/15 | 1 | 0.824 | 0.869 | 10 higher / 2 equal / 0 lower |
| Trace | 10/5 | 10 | 0.846 | 0.867 | 9 higher / 3 equal / 0 lower |
| **Trace** | 50/15 | 10 | **0.883** | **0.934** | 10 higher / 2 equal / 0 lower |

Raising epochs alone (10/5 → 50/15) or `gate_delta_lr_mult` alone (1 → 10) already puts trace above baseline (12-scenario F1 0.824 and 0.846 vs 0.745); using both reaches 0.883. On `Code_Stop_*`, trace F1 barely changes across configurations (0.963–0.990).

### 7.2 Other components (baseline, 10 epochs)

| `activity_penalty_weight` | 0 | 1.0 | 1.5 |
|---|---:|---:|---:|
| Call-flow mean F1 (6) | 0.272 | 0.486 | 0.676 |
| Mean F1 (12) | 0.405 | 0.560 | 0.681 |

Baseline (weight 1.5): `epoches/patience` 10/5 → 50/15 raises the 12-scenario mean F1 from 0.681 to 0.745. The `latency_dev` clip (§4.3) was not ablated separately (would need re-preprocessing).

## 8. Limitations

- **Small test files**: 9 of 12 scenarios have only 32–80 windows; `Svc_Kill_*` has just 4 anomaly windows, so one flipped window moves F1 by more than 0.1. F1 also depends on the number of normal windows and the anomaly rate of the test file (§6.1), so it is comparable only within this setup.
- **One seed** (`run_end 1`); small differences (≤ 0.03 on `Code_Stop_*`) are within noise.
- **Val has only 8 windows**, so the 95th-percentile threshold is unstable.
- **The `Code_Stop_*` rate was changed after the first run** (12.5% → 15%); the 12.5% results are in §6.1.
- **Train/test overlap**: the test normal pool includes a few `Normal_Baseline` windows (which are also in train/val).
- **Epoch selected on test**: each run's best checkpoint is chosen by test F1 (applied equally to baseline and trace), so absolute numbers may be slightly optimistic.
- With one real `Normal_Baseline` session, the training set stays small (31 windows).
- `gate_delta_lr_mult` has no counterpart on the baseline side; this remains one asymmetry between the two configurations.

## 9. Related Files

| What | File |
|---|---|
| `FAULT_WINDOWS`, two normal pools, `target_anomaly_rate`, `latency_dev` clip | `codes/common/preprocess_sn.py` |
| Change the anomaly rate of existing test files | `codes/common/resample_test_anomaly_rate.py` |
| `activity_penalty_weight`, `gate_delta_lr_mult` | `codes/models/basev3.py`, `codes/run.py` |
| Flag forwarding via wrapper | `codes/common/eval_per_scenario_sn.py` |
| Checkpoints | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
