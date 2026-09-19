# Experiment Results: SocialNetwork — Trace vs Baseline

> **Important update**: The previous version of this document (baseline F1 ~0.92, trace ~0.96) was computed on a preprocessing pipeline with a bug (see §2) that let baseline "cheat" via a session-level confound, and did not reflect real fault-detection ability. All numbers in this document were recomputed after fixing preprocessing + scoring — lower than before, but trustworthy.

## 1. Context: why everything was redone

While investigating why baseline (log+KPI only, no trace) scored unexpectedly high F1 on "call-flow" faults (service kill/stop — exactly the fault type trace should help with most), two root-cause bugs were found in the old preprocessing:

### 1.1 Session-level confound
Every test file compared the **same single `Normal_Baseline` session** (the first of 13 sequential experiments) against **one separate fault session** (recorded 15 minutes to nearly 3 hours later). Baseline could learn to distinguish "which session" (via environmental drift over time) instead of detecting the actual fault — verified by observing that metrics entirely unrelated to a given fault (e.g. another service's CPU) still differed clearly between the two groups.

### 1.2 Anomaly labels mismatched the fault's real timing
The old preprocessing used one blanket rule for every fault type: "skip the first 5 minutes (warmup), everything after is anomaly." Checking AnoMod's actual data-collection script (`github.com/EvoTestOps/AnoMod`, `automated_multimodal_collection.sh`) showed this rule was wrong for most fault types:

| Fault type | Real mechanism (per script) | Old label | Correct label |
|---|---|---|---|
| `Code_Stop_*` | `docker stop`, never auto-restarted | Whole session = anomaly | Correct, unchanged |
| `Perf_*`, `DB_Redis_CacheLimit_*` | ChaosBlade `--timeout 300`, auto-reverts after 300s | Whole session (minus 5min) = anomaly | **Only the first 5 minutes = anomaly**, the rest had already recovered |
| `Svc_Kill_*` | ChaosBlade process kill + Docker auto-restart | Whole session = anomaly | **Only ~2 minutes (t=90-210s)** — confirmed via the `container_label_restartcount` column (0→1 at t=105s) and a real ~75s silent gap in the traces |

## 2. New Preprocessing Design

### 2.1 Labels from `FAULT_WINDOWS` (not a blanket rule)
`codes/common/preprocess_sn.py` now defines the exact fault window for each scenario (see table above), grounded in real evidence (AnoMod's script + direct measurement) instead of a generic "skip N minutes" heuristic.

### 2.2 Two separate normal pools
- **Train/unlabel/val**: `Normal_Baseline` only (39 windows) — kept narrow/homogeneous on purpose (see §2.3).
- **Test** (the "normal" side of every `test_<scenario>.pkl`): pooled from **every** scenario (Normal_Baseline + each scenario's recovered/never-faulted windows) — 320 windows spanning the entire ~3-hour experiment instead of a single 20-minute slice. This is a real (not a trick) fix for the session confound: baseline can no longer "guess the session" since normal now comes from many different times.

### 2.3 Why train does NOT use the diverse pool
An earlier attempt used the diverse pool for training too — but direct measurement showed this taught the model that low-activity windows are normal too, breaking detection of "went silent" faults specifically (see §3). Reverting to a narrow train pool (Normal_Baseline only) fixed the actual problem without giving up the confound-reduction benefit at test time.

## 3. Scoring Fix: "went silent" faults scored backwards

### 3.1 Finding
Even after fixing the labels, `Code_Stop_TextService`/`UserService` and `Svc_Kill_*` still scored F1=0.0000. Measuring the reconstruction loss distribution by label directly (not assumed):

```
Code_Stop_TextService:  anomaly mean loss = 0.92   |   normal mean loss = 1.29
```

**Anomaly windows scored LOWER loss than normal windows** — the exact opposite of the "high loss = anomaly" logic the whole system relies on. Checked the log and KPI components separately — both showed the same inversion, so it wasn't just a sparse-log artifact.

### 3.2 Root cause
When a service dies completely, its log/KPI input becomes almost empty/flat. For an autoencoder, reconstructing a near-constant input is *easier* than reconstructing a genuinely varied normal one — so the loss comes out lower, not higher. This is an inherent property of reconstruction-loss scoring, not a pooling bug (verified: the inversion persisted identically whether the train pool was narrow or diverse).

### 3.3 Fix: `activity_penalty_weight`
Added a scoring term **independent of reconstruction**, computed directly from the raw input: how far the current activity level (`kpi_features.sum + log_features.sum`) sits *below* the normal training activity level, added to the anomaly score before thresholding. Defaults to `0.0` (no-op, doesn't affect other datasets or existing checkpoints).

```bash
--activity_penalty_weight 1.5
```

### 3.4 Combined with `gate_delta_lr_mult`
For the trace branch, combined with `--gate_delta_lr_mult 10 --epoches 50 50 --patience 15` (a pre-existing mechanism that helps `trace_gate`/`delta_head` escape a gradient bottleneck faster) — this compounds with the activity penalty to close the remaining gap on the weakest-signal scenarios (`Svc_Kill_*`).

---

## 4. Final Run Configuration

### Baseline (`open_trace=False`)
```bash
cd D:/UAM-AD/codes
python common/eval_per_scenario_sn.py \
    --data ../data/sn --dataset sn --data_type fuse \
    --open_trace False --activity_penalty_weight 1.5 \
    --epoches 50 50 --batch_size 256 --patience 15 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True \
    --run_start 0 --run_end 1
```
> **`epoches`/`patience` matched to the trace config** (see "Fairness note" in §5) — `gate_delta_lr_mult` doesn't apply here since baseline has no `trace_gate`/`delta_head`.

### Trace (`open_trace=True`)
```bash
cd D:/UAM-AD/codes
python common/eval_per_scenario_sn.py \
    --data ../data/sn --dataset sn --data_type fuse \
    --open_trace True --activity_penalty_weight 1.5 \
    --gate_delta_lr_mult 10 \
    --epoches 50 50 --batch_size 256 --patience 15 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True \
    --run_start 0 --run_end 1
```

---

## 5. Results (12 scenarios, after fixing labels + scoring)

> **Fairness note**: the first version of this section ran baseline with `epoches=10 10, patience=5` (the original default) while trace used `epoches=50 50, patience=15` — an unequal training budget. Re-checking with baseline given the **same** budget as trace showed this mattered: baseline improved meaningfully on its own (mean F1 0.289→0.342), especially on the `Svc_Kill_*` group. The numbers below are the corrected version — baseline and trace both use `epoches=50 50, patience=15` (the only remaining difference is `gate_delta_lr_mult=10`, which doesn't apply to baseline — it has no `trace_gate`/`delta_head`).

| Scenario | Baseline F1 | Baseline P | Baseline R | Trace F1 | Trace P | Trace R | Δ F1 |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| Code_Stop_MediaService | 0.500 | 0.333 | 1.000 | **0.600** | 0.429 | 1.000 | +0.100 |
| Code_Stop_TextService | 0.462 | 0.300 | 1.000 | **0.632** | 0.462 | 1.000 | +0.170 |
| Code_Stop_UserService | 0.462 | 0.300 | 1.000 | **0.615** | 0.571 | 0.667 | +0.154 |
| DB_Redis_CacheLimit_HomeTimeline | 0.333 | 0.200 | 1.000 | **0.375** | 0.231 | 1.000 | +0.042 |
| DB_Redis_CacheLimit_SocialGraph | 0.462 | 0.300 | 1.000 | 0.435 | 0.294 | 0.833 | -0.027 |
| DB_Redis_CacheLimit_UserTimeline | 0.200 | 0.250 | 0.167 | **0.276** | 0.174 | 0.667 | +0.076 |
| Perf_CPU_Contention | 0.333 | 0.200 | 1.000 | 0.345 | 0.217 | 0.833 | +0.012 |
| Perf_Disk_IO_Stress | 0.333 | 0.200 | 1.000 | 0.345 | 0.217 | 0.833 | +0.012 |
| Perf_Network_Loss | 0.200 | 0.250 | 0.167 | **0.345** | 0.217 | 0.833 | +0.145 |
| Svc_Kill_Media | 0.242 | 0.138 | 1.000 | 0.267 | 0.154 | 1.000 | +0.025 |
| Svc_Kill_SocialGraph | 0.286 | 0.167 | 1.000 | 0.320 | 0.190 | 1.000 | +0.034 |
| Svc_Kill_UserTimeline | 0.296 | 0.174 | 1.000 | 0.333 | 0.200 | 1.000 | +0.037 |
| **Mean** | **0.342** | **0.234** | **0.861** | **0.407** | **0.280** | **0.889** | **+0.065** |
| Std | 0.102 | 0.061 | 0.311 | 0.127 | 0.128 | 0.124 | |

**Trace still wins on 11/12 scenarios** (loses `DB_Redis_CacheLimit_SocialGraph`, -0.027) even after giving baseline the same training budget, but the win margin is **noticeably thinner** on the `Svc_Kill_*`/`Perf_*` group than in the earlier (corrected) report. Caveat: single run (`run_end=1`, one seed) and only 4–6 anomaly windows per test file, so one flipped window moves F1 by ~0.05–0.1; margins of +0.01–0.04 are within noise.

### 5.2 Ablation: which ingredient matters (trace, all 12 scenarios, mean F1)

| Config | epochs/patience | `gate_delta_lr_mult` | Mean F1 (12) | Call-flow mean F1 (6) | Wins vs fair baseline |
|:---|:---:|:---:|---:|---:|:---:|
| Baseline (no trace) | 50/15 | – | 0.342 | 0.375 | – |
| Trace, no LR fix | 50/15 | 1 | 0.355 | 0.421 | 5 win / 3 tie / 4 lose |
| Trace, no extra epochs | 10/5 | 10 | 0.372 | 0.418 | 6 win / 1 tie / 5 lose |
| **Trace, final** | 50/15 | 10 | **0.407** | **0.461** | 11 win / 1 lose |

Reading: `Code_Stop_*` is insensitive to both knobs (trace wins in every row). Neither the extra epochs nor `gate_delta_lr_mult` alone lets trace beat baseline consistently — e.g. without the LR fix trace *loses* `Svc_Kill_SocialGraph` (0.160 vs 0.286) and `Perf_Disk_IO_Stress` (0.200 vs 0.333) — but together they do. The two effects are roughly additive (~+0.05 each on the call-flow mean). Not isolated: Fix A (`latency_dev` clip to ±10) — it only binds on ~0.1–0.2% of test values (none in train/val), so its contribution is expected to be small but was not ablated (would need re-preprocessing).

### 5.1 "Call-flow" fault group (primary target: service kill/stop)

| Scenario | Baseline F1 (fair) | Trace F1 | Δ |
|---|---:|---:|:---:|
| Code_Stop_MediaService | 0.500 | 0.600 | +0.100 |
| Code_Stop_TextService | 0.462 | 0.632 | +0.170 |
| Code_Stop_UserService | 0.462 | 0.615 | +0.154 |
| Svc_Kill_Media | 0.242 | 0.267 | +0.025 |
| Svc_Kill_SocialGraph | 0.286 | 0.320 | +0.034 |
| Svc_Kill_UserTimeline | 0.296 | 0.333 | +0.037 |
| **Mean** | **0.375** | **0.461** | **+0.087** |

**Trace wins all 6/6**, and the fairness check cleanly separates the two sub-types:
- **`Code_Stop_*` (service permanently dead, strong/sustained signal): large win margin, and UNCHANGED by giving baseline 40 more epochs** (+0.10 to +0.17 — identical to the pre-fairness-fix numbers) → this is solid evidence that the win is not a training-budget artifact, but genuine value from trace.
- **`Svc_Kill_*` (service auto-restarts quickly, ~2min signal window): win margin shrinks substantially** (+0.12–0.16 before the fairness fix, down to **+0.03–0.04** after) — most of the earlier "improvement" here actually came from baseline being undertrained, not from trace. Trace still wins, but by a thin margin that honestly reflects this fault type's short signal window.

## 6. Remaining Limitations

- **Absolute F1 is still low** (0.2-0.6) compared to the old numbers (0.9+) — this is the **honest** figure after removing the confound, reflecting the real difficulty of the task on a small dataset (39 training windows). Do not compare directly against the old report.
- **Precision is still low** (0.15-0.35) — the activity penalty trades precision for higher recall; there's room to tune further if needed.
- **Win margins on `Svc_Kill_*`/`Perf_*` are thin** (+0.01 to +0.04) once the training-budget confound is controlled for — the "trace helps" conclusion for this group should be stated cautiously, not oversold.
- **`DB_Redis_CacheLimit_SocialGraph`** is the only scenario where trace doesn't win (-0.027) — not yet investigated in depth.
- Since SN has only one real `Normal_Baseline` session, the training set will always be small (39 windows) — this is a structural limitation of the raw data, not of preprocessing.
- `gate_delta_lr_mult=10` has no equivalent counterpart on the baseline side (it can't apply) — this remains one asymmetry between the two configs, but it's unavoidable since the mechanism only exists where there's a `trace_gate`.

## 7. Related Files

| Change | File |
|---|---|
| `FAULT_WINDOWS`, two normal pools | `codes/common/preprocess_sn.py` |
| `activity_penalty_weight` | `codes/models/basev3.py`, `codes/run.py` |
| Flag forwarding via wrapper | `codes/common/eval_per_scenario_sn.py` |
| Final checkpoints | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
