# Standard evaluation protocol (SN, RE2-OB, RE3-OB)

One protocol for every dataset. No step uses test labels.

## 1. Steps

1. **Model / epoch selection**: early stopping on the mean anomaly score (fusion loss, plus the activity penalty when enabled) over the **val** set (normal only). Test data and labels are never used to pick the epoch.
2. **Threshold**: the **95th percentile** of the selected model's scores on the val normals (`--val_percentile 95`, same for all datasets).
3. **Primary metric**: precision / recall / **F1 at the val threshold** (no `point_adjust`). Written to `info_score.txt` as `f1`, `pc`, `rc`.
4. **Secondary metrics**: **AUROC** and **AUPRC** (`auroc`, `auprc`). They are threshold-free, so they separate the quality of the score from the quality of the threshold calibration.
5. **Oracle F1** (`oracle_f1`, `oracle_pc`, `oracle_rc`, `oracle_threshold`): the threshold is swept on the test scores (top `--anomaly_rate` percent, with `point_adjust`). It uses test labels and is optimistic, so it is reported only in its own field, for comparison with papers that use a sweep (UAC-AD, TraceDAE). It is never the headline number.
6. **Seeds**: report **3–5 seeds** (mean ± std). Quick verification runs may use 1 seed; differences between methods that are small relative to run-to-run noise are then treated as ties.

## 2. Why not the oracle as the headline

The oracle picks the best threshold, and (previously) the best epoch, by test labels. On SN it gave baseline 0.745 / trace 0.883, against 0.407 / 0.355 at a val threshold. A number that needs test labels cannot be reproduced in deployment, and it hides threshold-calibration error, which is what a practitioner actually faces.

## 3. Split rules

| Dataset | train | val (normal only) | test |
|:--|:--|:--|:--|
| SN | all 39 `Normal_Baseline` windows | 20% of every other session's normal windows (57 windows → (57−5)×5 = 260 scores) | per scenario: anomaly windows + normal windows from the other 80% (224-window pool) |
| RE2-OB / RE3-OB | `unlabel.pkl` / `train.pkl` (80% / 20% of the pre-injection normal windows) | **not produced yet** (`preprocess_rcaeval_re{2,3}_ob.py` write no `val.pkl`); the code falls back to the unlabel normals, which the model has trained on, so its threshold is too low and the F1 is not comparable to SN until a held-out val split is added | post-injection data with labels |

Val is drawn from the same mix of sessions as the test normals, and is disjoint from them. Val must be large enough for a stable p95: with 8 windows (15 scores) one outlier window decided the threshold (bootstrap range 1.81–3.81); with 57 windows the spread is much smaller.

### SN anomaly rate per scenario

Normals per test file = anomaly × (1 − r) / r, limited by the 224-window pool. `Code_Stop_*` anomalies are thinned evenly over time to 39.

| Scenario | Anomaly | Normal | Total | Rate |
|:--|--:|--:|--:|--:|
| Code_Stop_MediaService | 39 | 224 | 263 | 14.8% |
| Code_Stop_TextService | 39 | 224 | 263 | 14.8% |
| Code_Stop_UserService | 39 | 224 | 263 | 14.8% |
| DB_Redis_CacheLimit_{HomeTimeline, SocialGraph, UserTimeline} | 10 | 70 | 80 | 12.5% |
| Perf_{CPU_Contention, Disk_IO_Stress, Network_Loss} | 10 | 70 | 80 | 12.5% |
| Svc_Kill_{Media, SocialGraph, UserTimeline} | 4 | 28 | 32 | 12.5% |

## 4. Limitations

- `Svc_Kill_*` has 4 anomaly windows, so F1 is very noisy whatever the threshold.
- Training is not bit-reproducible: the same code and seed can differ between runs, so use several seeds for any claim.
- Val comes from other sessions than train (a different recording time), so a session shift moves val scores too. This is intended: it makes the threshold representative of test normals.
- p95 fixes the expected false-positive rate at about 5% of normal scores; with a low anomaly rate this caps precision.
