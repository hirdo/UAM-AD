# Experiment Results: SocialNetwork — Trace vs Baseline

Evaluated with the standard protocol in [`evaluation_protocol_en.md`](evaluation_protocol_en.md): epoch chosen on val loss, threshold = p95 of val scores, **F1 at the val threshold is the primary metric**, AUROC / AUPRC secondary, oracle F1 reported separately.

## 1. Experiment Setup

### Model
**HADES** — GAN-based unsupervised anomaly detection model trained only on normal data.

### Settings

| Setting                               | Value                                                                                                                                                                  |
| :------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                               | SocialNetwork (AnoMod), 12 fault scenarios                                                                                                                             |
| Data type                             | `fuse` (KPI + Log [+ Trace when `open_trace=True`])                                                                                                                    |
| Train / unlabel                       | 39 windows (all `Normal_Baseline`)                                                                                                                                     |
| Val                                   | 57 normal windows (20% of every other session's normals) → 260 scores                                                                                                  |
| Test per scenario                     | Anomaly windows + normals from the 224-window test pool; `Code_Stop_*` 263 = 39 + 224 (14.8%), `Perf_*`/`DB_Redis_*` 80 = 10 + 70 and `Svc_Kill_*` 32 = 4 + 28 (12.5%) |
| `window_size`                         | 5 (5 windows × 30 s)                                                                                                                                                   |
| `val_percentile`                      | 95                                                                                                                                                                     |
| `epoches` / `patience`                | 50 50 / 15 (same for baseline and trace)                                                                                                                               |
| `batch_size`, `alpha`, `open_gan_sep` | 256, 0.16, True                                                                                                                                                        |
| `activity_penalty_weight`             | 1.5 (same for baseline and trace)                                                                                                                                      |
| `gate_delta_lr_mult`                  | 10 (trace only — baseline has no `trace_gate`/`delta_head`)                                                                                                            |
| `run_end`                             | 1 (single run, one seed)                                                                                                                                               |

### Result Directories
| Configuration             | Folder                                       |
| :------------------------ | :------------------------------------------- |
| Baseline (KPI + Log)      | `data/sn/result_per_scenario_fuse_baseline/` |
| Trace (KPI + Log + Trace) | `data/sn/result_per_scenario_fuse_trace/`    |

## 2. Normal / Anomaly Labels per Fault Type

The fault window (`FAULT_WINDOWS` in `codes/common/preprocess_sn.py`) is derived from AnoMod's original collection script (`github.com/EvoTestOps/AnoMod`, `automated_multimodal_collection.sh`: collection starts 15 s after the fault is injected) and from direct measurement. Times are measured from the start of each session's recording:

| Fault type                        | Mechanism (per script)                               | Anomaly window | Rest of the session  |
| --------------------------------- | ---------------------------------------------------- | -------------- | -------------------- |
| `Code_Stop_*`                     | `docker stop`, never auto-restarted                  | Whole session  | –                    |
| `Perf_*`, `DB_Redis_CacheLimit_*` | ChaosBlade `--timeout 300`, auto-reverts after 300 s | 0–300 s        | Normal (recovered)   |
| `Svc_Kill_*`                      | ChaosBlade process kill + Docker auto-restart        | 90–210 s       | Normal               |
| `Normal_Baseline`                 | –                                                    | –              | Whole session normal |

The `Svc_Kill_*` window is confirmed by: the `container_label_restartcount` column flipping 0→1 at t=105 s in all 3 scenarios (absent in `Normal_Baseline`), and a ~75 s silent gap (101.8 s → 176.8 s) in `user-timeline-service`'s traces.

## 3. Two Normal Pools

- **Train / unlabel**: all 39 `Normal_Baseline` windows. Kept narrow on purpose: adding low-activity windows from other scenarios to training makes the model treat "low activity" as normal, which destroys detection of "went completely silent" faults (§4.1).
- **Val and test normals**: the normal windows of every other session (281 windows), split 20% / 80% per source session: 57 windows to val, 224 to the test pool. Each test file draws its normals round-robin from the test pool, so it mixes many sessions; val comes from the same mix but is disjoint from test.

Each `test_<scenario>.pkl` contains that scenario's own anomaly windows (at most 39, thinned evenly over time; only `Code_Stop_*` exceeds it) plus normals to reach the target rate (`--target_anomaly_rate 0.125`, limited by the 224-window pool, which gives 14.8% for `Code_Stop_*`).

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

### 6.1 Primary: F1 at the val threshold (p95 of val scores)

| Scenario                         | Baseline F1 |     P |     R |  Trace F1 |     P |     R |    Δ F1    |
| :------------------------------- | ----------: | ----: | ----: | --------: | ----: | ----: | :--------: |
| Code_Stop_MediaService           |       0.690 | 0.527 | 1.000 | **0.772** | 0.629 | 1.000 |   +0.082   |
| Code_Stop_TextService            |       0.709 | 0.549 | 1.000 | **0.796** | 0.661 | 1.000 |   +0.087   |
| Code_Stop_UserService            |       0.731 | 0.576 | 1.000 | **0.817** | 0.691 | 1.000 |   +0.086   |
| DB_Redis_CacheLimit_HomeTimeline |       0.364 | 0.333 | 0.400 | **0.800** | 0.667 | 1.000 |   +0.436   |
| DB_Redis_CacheLimit_SocialGraph  |       0.720 | 0.562 | 1.000 | **0.857** | 0.750 | 1.000 |   +0.137   |
| DB_Redis_CacheLimit_UserTimeline |       0.476 | 0.385 | 0.625 | **0.706** | 0.667 | 0.750 |   +0.230   |
| Perf_CPU_Contention              |       0.522 | 0.462 | 0.600 | **0.800** | 0.667 | 1.000 |   +0.278   |
| Perf_Disk_IO_Stress              |       0.552 | 0.400 | 0.889 | **0.615** | 0.471 | 0.889 |   +0.064   |
| Perf_Network_Loss                |       0.400 | 0.400 | 0.400 | **0.857** | 0.818 | 0.900 |   +0.457   |
| Svc_Kill_Media                   |       0.667 | 0.500 | 1.000 | **0.889** | 0.800 | 1.000 |   +0.222   |
| Svc_Kill_SocialGraph             |       0.600 | 0.429 | 1.000 | **0.750** | 0.600 | 1.000 |   +0.150   |
| Svc_Kill_UserTimeline            |       0.727 | 0.571 | 1.000 |     0.727 | 0.571 | 1.000 |   +0.000   |
| **Mean**                         |   **0.596** | 0.475 | 0.826 | **0.782** | 0.666 | 0.962 | **+0.186** |
| Std of F1                        |       0.126 |       |       |     0.072 |       |       |            |

Trace F1 is higher in 11/12 scenarios, equal in 1/12 (`Svc_Kill_UserTimeline`) and lower in none. Recall is 1.000 for both in all `Code_Stop_*` files and most `Svc_Kill_*` files, so precision (false alarms at the p95 threshold) decides F1 there.

### 6.2 Secondary: AUROC, AUPRC and oracle F1 (baseline / trace)

| Scenario                         |     AUROC     |     AUPRC     |   Oracle F1   |
| :------------------------------- | :-----------: | :-----------: | :-----------: |
| Code_Stop_MediaService           | 0.977 / 0.977 | 0.780 / 0.786 | 0.918 / 0.918 |
| Code_Stop_TextService            | 0.974 / 0.978 | 0.713 / 0.773 | 0.907 / 0.929 |
| Code_Stop_UserService            | 0.964 / 0.968 | 0.664 / 0.715 | 0.894 / 0.894 |
| DB_Redis_CacheLimit_HomeTimeline | 0.637 / 0.949 | 0.373 / 0.603 | 0.444 / 0.833 |
| DB_Redis_CacheLimit_SocialGraph  | 0.973 / 0.973 | 0.736 / 0.736 | 0.900 / 0.900 |
| DB_Redis_CacheLimit_UserTimeline | 0.692 / 0.923 | 0.544 / 0.696 | 0.667 / 0.778 |
| Perf_CPU_Contention              | 0.903 / 0.966 | 0.642 / 0.768 | 0.667 / 0.833 |
| Perf_Disk_IO_Stress              | 0.901 / 0.939 | 0.579 / 0.641 | 0.667 / 0.727 |
| Perf_Network_Loss                | 0.625 / 0.977 | 0.508 / 0.906 | 0.571 / 0.909 |
| Svc_Kill_Media                   | 0.962 / 0.962 | 0.679 / 0.679 | 0.889 / 0.889 |
| Svc_Kill_SocialGraph             | 0.926 / 0.926 | 0.478 / 0.478 | 0.750 / 0.750 |
| Svc_Kill_UserTimeline            | 0.962 / 0.962 | 0.679 / 0.679 | 0.889 / 0.889 |
| **Mean**                         | 0.874 / 0.958 | 0.615 / 0.705 | 0.764 / 0.854 |

Oracle F1 (threshold swept on test labels, with `point_adjust`) is optimistic and only for comparison with papers that use a sweep; it is not the headline. AUROC/AUPRC are threshold-free, so they show the quality of the score itself.

### 6.3 How to read these results

- **Where trace helps most**: AUROC rises from 0.637 to 0.949 (`DB_Redis_CacheLimit_HomeTimeline`), 0.692 to 0.923 (`DB_Redis_CacheLimit_UserTimeline`), 0.625 to 0.977 (`Perf_Network_Loss`) and 0.903 to 0.966 (`Perf_CPU_Contention`): the scenarios where baseline is weak. In `Code_Stop_*` baseline AUROC is already 0.96–0.98 and trace adds at most 0.004; the F1 gain there (+0.09 on average) comes from higher precision at the same threshold.
- **Identical scores**: on `Svc_Kill_*`, `DB_Redis_CacheLimit_SocialGraph` and `Code_Stop_MediaService` baseline and trace have (nearly) the same AUROC, so the trace branch changed little in the ranking there; F1 differs where the threshold falls.
- **Small files**: `Svc_Kill_*` has 4 anomaly windows and 32 windows in total, so a single window moves F1 by more than 0.1; treat those rows as anecdotal.
- **Single seed**: small differences (`Perf_Disk_IO_Stress` 0.552 → 0.615, the `Code_Stop_*` gains) are within run-to-run noise (§7). The protocol asks for 3–5 seeds before any claim; this table is a 1-seed verification.
- Per-feature signal analysis and ablations (epochs, `gate_delta_lr_mult`, `activity_penalty_weight`) were not repeated on this split, so they are not reported.

## 7. Limitations

- **Small test files**: 9 of 12 scenarios have only 32–80 windows; `Svc_Kill_*` has just 4 anomaly windows. F1 also depends on the number of normal windows and the anomaly rate of the test file, so it is comparable only within this setup.
- **One seed** (`run_end 1`); small differences are within noise.
- **Not exactly reproducible**: with the same code, data and seed, runs can still differ, so each number is one sample.
- **p95 caps precision**: the threshold flags about 5% of val normal scores by design, which lowers precision when the anomaly rate is low.
- **System KPIs carry fault-unspecific traces**: `load1` is high at the start of every session (stack start-up) and `disk_usage_percent` grows across sessions, separating anomaly from normal with AUC 0.99–1.00 in some scenarios. Baseline and trace both receive these KPIs, so the comparison is like-for-like, but absolute F1 may be lifted by time/session traces rather than the fault alone. No ablation dropping these KPIs has been run.
- **Session shift**: train (`Normal_Baseline`) and val/test normals come from different recording times; val is drawn from the same sessions as test normals by design.
- With one real `Normal_Baseline` session, the training set stays small (39 windows).
- `gate_delta_lr_mult` has no counterpart on the baseline side; this remains one asymmetry between the two configurations.

## 8. Related Files

| What                                                                                                             | File                                                 |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Standard protocol                                                                                                | `docs/evaluation_protocol_en.md`                     |
| `FAULT_WINDOWS`, val/test split, `target_anomaly_rate`, `max_anomalies`, `latency_dev` clip                      | `codes/common/preprocess_sn.py`                      |
| Val-loss model selection, val threshold, AUROC/AUPRC, oracle F1, `activity_penalty_weight`, `gate_delta_lr_mult` | `codes/models/basev3.py`, `codes/run.py`             |
| Wrapper and summary table                                                                                        | `codes/common/eval_per_scenario_sn.py`               |
| Results                                                                                                          | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
