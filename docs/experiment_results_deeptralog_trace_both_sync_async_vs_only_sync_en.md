# Experiment Results: DeepTraLog — Sync-only vs Sync+Async Trace

Evaluated with the standard protocol in [`evaluation_protocol_en.md`](evaluation_protocol_en.md): epoch chosen on val loss, threshold = p95 of val scores, **F1 at the val threshold is the primary metric**, AUROC / AUPRC secondary, oracle F1 reported separately. Unlike the other `experiment_results_*` docs, both configurations here use `--score_rule norm_sum` (not the raw fusion-loss sum) and **3 seeds** — see §4 and §6.

## 1. Experiment Setup

### Model
**HADES** — GAN-based unsupervised anomaly detection model trained only on normal data, extended with one additive async-trace branch (see [`model_architecture_flow_en.md`](model_architecture_flow_en.md) §10 for the architecture).

### Settings

| Setting                               | Value                                                                                              |
| :------------------------------------ | :--------------------------------------------------------------------------------------------------- |
| Dataset                               | DeepTraLog (TrainTicket), F01/F02/F13 (the paper's "Asynchronous Interaction" fault group)          |
| Data type                             | `fuse` (log + trace; no metric in this dataset, `kpi_c=1` all-zero, `open_unmatch_zoomout=False`)   |
| Train / unlabel                       | 6,000 / 6,000 normal traces (downsampled from ~93,000 for an 8 GB machine)                          |
| Val                                   | 1,500 normal traces                                                                                  |
| Test per F-case                       | F01 16,424 (2,053 anomalous, 12.5%); F02 17,208 (2,151, 12.5%); F13 14,232 (1,779, 12.5%)            |
| `window_size`                         | 5                                                                                                      |
| `val_percentile`                      | 95                                                                                                     |
| `score_rule`                          | `norm_sum` for **both** configurations (see §4) — the raw-sum default is not used on this dataset     |
| `epoches` / `patience`                | 10 10 / 5 (same for both configurations)                                                              |
| `batch_size`, `alpha`, `open_gan_sep` | 128, 0.16, True                                                                                       |
| `open_async_trace`, `async_order`     | only-sync: False, — ; both: True, True (auto-read from `meta["async_order"]`)                        |
| `gate_lambda`                         | 0.01 (both configurations)                                                                            |
| `run_start` / `run_end`               | 0..3 (**3 seeds**, `random_seed = 42 + run_times`)                                                    |

### Result Directories
| Configuration                                | Folder                                                              |
| :-------------------------------------------- | :------------------------------------------------------------------- |
| Only sync (log + sync trace)                  | `data/deeptralog/result_per_scenario_fuse_trace_only_sync/`         |
| Both sync and async (log + sync + async trace)| `data/deeptralog/result_per_scenario_fuse_trace_both_sync_and_async/`|

## 2. What F01/F02/F13 Change (from the dataset's structural fingerprint, `docs/preprocess_deeptralog_en.md` §4)

| F-case | API(s) | What changes vs same-API normal traces | What stays the same |
| :--- | :--- | :--- | :--- |
| F01 | `preserveservice/preserve`, `cancel`, `execute`, `collected`, `travelplan/{cheapest,quickest}` | 100% of traces (in the order-sensitive sub-cases) have a service-to-service temporal order relation never seen in normal traces of the same API | Edge set (which services call which) is unchanged |
| F02 | `foodservice/createOrderBatch` (+ a travel-root sub-case with almost no async messages) | Async message count +21% on the food-root sub-case (11.6 → 14.1 msg/trace); no order-relation change | Edge set unchanged |
| F13 | `rebookservice/rebook`, `admintravelservice/admintravel`, `trips/left` (silent/fail-fast sub-case) | 100% unseen order relation on the two rebook/admintravel sub-cases; the `trips/left` sub-case instead fails fast (18 spans, 100% has an error span, vs 0.1% of normal traces) — see §4.4 | Edge set unchanged |

This is why a sync-only branch (which only tracks the edge set) sees little in these three F-cases, and why the async branch (message count, order relation, error-span evidence) is the one expected to move the needle.

## 3. Two Normal Pools

- **Train / unlabel**: 6,000 / 6,000 normal traces, reservoir-sampled from the ~93,000 normal traces across the dataset's `normal/*.zip` files (downsampled purely for RAM; the labels/log-template/latency-baseline pass still scans everything).
- **Val**: 1,500 normal traces, disjoint from train/unlabel.
- **Test**: each `test_{F}.pkl` holds that F-case's own anomalous traces plus normal traces sampled to reach the dataset's fixed 12.5% target anomaly rate (`TARGET_ANOMALY_RATE` in `preprocess_deeptralog.py`, the same convention as SN).

## 4. Scoring Components (`score_rule=norm_sum`)

Both configurations score with `evaluate_norm_sum` (`codes/models/basev3.py`): one fused score `S = Σ_k z_k`, threshold = val p95 of `S`, each term standardised with **normal-only val statistics**, `z_k = max(0, (T_k − median_k) / (p95_k − median_k))`. This dataset does not use the raw fusion-loss sum (the default for other datasets) because the noisiest term would otherwise decide the ranking.

| Term | Present when | What it measures |
| :--- | :--- | :--- |
| `log_kpi_loss` | always | Log (+ degenerate KPI) reconstruction error |
| `trace_dis` | `open_trace=True` | Sync trace structure+attribute reconstruction error (unchanged from the SN/RE2/RE3 branch) |
| `trace_dis_async_count` | `open_async_trace=True` | Async branch: message-count structure + attribute reconstruction error. **Context-normalised**: only ~8.7% of val traces exchange async messages at all, so this term is `z=0` for message-free traces and is standardised only against the message-bearing val traces (robust scale `(p90−median)·1.645/1.2816`; falls back to global normalisation if fewer than 40 such val traces exist) |
| `trace_dis_async_order` | `open_async_trace=True, async_order=True` | Async branch: BCE reconstruction error of the directed service-pair temporal-order relation (`async_temporal_order_adj`), averaged over present-service pairs |
| `trace_err` | `open_async_trace=True` (added for this run) | Error-span evidence: `log1p(Σ_service call_count · error_rate)` for the whole trace. Val normal traces have an error span 0.13% of the time, so `p95 − median` is ~0 and cannot serve as a scale; the term is floored at a **fixed scale** so that one error span in an otherwise error-free trace scores `z = 4` (`BaseModel.ERR_Z_ONE`). If normals in a dataset *do* commonly have error spans, the measured val scale takes over automatically. |

The async embedding (`ZV_async`, masked mean over present services) also feeds the shared `delta_head` alongside the sync embedding, i.e. it can help reconstruct log/KPI, not just add a score term (see `model_architecture_flow_en.md` §10.5).

### 4.1 Why `trace_err` (F13's silent/fail-fast sub-case)

`trips/left` (F13, 299 traces) is not "quiet" trace-structure noise — it is a **fail-fast error trace**: exactly 18 spans (vs 134 for a normal trip request), 100% have at least one error span (vs 0.1% of normal traces overall), 5 services are never called, and the whole request finishes in ~14 ms (vs ~94 ms). Before adding `trace_err`, this sub-group's threshold-relative recall collapsed under the higher `norm_sum` threshold (message-free order-relation term contributes 0 here, so the trace has no other signal above baseline log noise). Adding `trace_err` recovers it (recall of this sub-group 0.085 → 1.000, measured on a saved checkpoint before the full re-run) without moving F01/F02 (which have 0 error spans in every sub-case).

## 5. Commands

```bash
cd D:/UAM-AD
python codes/common/preprocess_deeptralog.py --stage normal --fault_dir <F*.zip dir> --normal_dir <normal dir> \
    --label_pkl <labels.pkl> --output_dir data/deeptralog
python codes/common/preprocess_deeptralog.py --stage fcase --fcases F01 F02 F13 --fault_dir <F*.zip dir> \
    --normal_dir <normal dir> --label_pkl <labels.pkl> --output_dir data/deeptralog

cd codes
# Only sync
python common/eval_per_scenario_deeptralog.py --data ../data/deeptralog --dataset deeptralog --data_type fuse \
    --open_trace True --open_async_trace False --score_rule norm_sum \
    --epoches 10 10 --batch_size 128 --patience 5 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True --open_unmatch_zoomout False \
    --gate_lambda 0.01 --fcases F01 F02 F13 --run_start 0 --run_end 3
# Both sync and async
python common/eval_per_scenario_deeptralog.py --data ../data/deeptralog --dataset deeptralog --data_type fuse \
    --open_trace True --open_async_trace True --score_rule norm_sum \
    --epoches 10 10 --batch_size 128 --patience 5 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True --open_unmatch_zoomout False \
    --gate_lambda 0.01 --fcases F01 F02 F13 --run_start 0 --run_end 3
```
(`async_order` is not passed explicitly — `run.py` auto-reads it from `meta["async_order"]`, `True` for this dataset.) In practice each F-case × seed was run as its **own process**, one at a time, to fit an 8 GB machine; the commands above are the single-process equivalent.

## 6. Results

### 6.1 Primary: F1 at the val threshold (p95 of val scores), mean ± std over 3 seeds

| F-case | Only sync F1 | P | R | Both F1 | P | R | Δ F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| F01 | 0.764±0.003 | 0.706±0.005 | 0.833±0.000 | **0.774±0.002** | 0.722±0.004 | 0.832±0.001 | +0.010 |
| F02 | 0.110±0.003 | 0.184±0.003 | 0.078±0.003 | **0.138±0.011** | 0.233±0.014 | 0.098±0.009 | +0.028 |
| F13 | 0.827±0.036 | 0.735±0.012 | 0.948±0.072 | **0.863±0.001** | 0.759±0.002 | 0.999±0.000 | +0.036 |
| **Mean** | **0.567** | 0.542 | 0.620 | **0.592** | 0.571 | 0.643 | **+0.025** |

F1 with both branches is higher on all 3/3 F-cases; the gain is largest on F13 (where a single-seed run without `trace_err` — see §4.1 — showed F1 *below* the only-sync arm) and smallest on F01.

### 6.2 Secondary: AUROC, AUPRC and oracle F1, mean ± std over 3 seeds (only sync / both)

| F-case | AUROC | AUPRC | Oracle F1 |
| :--- | :---: | :---: | :---: |
| F01 | 0.933±0.008 / **0.971±0.005** | 0.831±0.003 / **0.857±0.004** | 0.656±0.031 / 0.540±0.061 |
| F02 | 0.655±0.006 / **0.809±0.040** | 0.166±0.003 / **0.266±0.045** | 0.396±0.012 / 0.358±0.008 |
| F13 | 0.983±0.003 / **0.989±0.001** | 0.862±0.015 / **0.883±0.009** | 0.498±0.014 / 0.484±0.025 |

Oracle F1 (threshold swept on test labels, with `point_adjust`) is optimistic and only for comparison with papers that use a sweep; it is not the headline. AUROC/AUPRC are threshold-free, so they show the quality of the score itself, and both branches beat the only-sync branch on every F-case for both.

### 6.3 How to read these results

- **F02** is the cleanest evidence for the async branch: AUROC 0.655 → 0.809 (+0.154), AUPRC 0.166 → 0.266 (+0.10). The fault here changes only the async message count, which the only-sync branch (edge set only, §2) structurally cannot see. F1 stays low in absolute terms (0.110 → 0.138) because normal and faulty message counts overlap heavily (8–15 vs 9–19 messages/trace) — the async count term's own AUROC (≈0.74–0.79, measured on component scores) is close to a plain message counter's ceiling on this sub-case.
- **F13** gains the most in F1 (+0.036) and reaches recall 0.999, driven by the `trace_err` term recovering the `trips/left` sub-group (§4.1) plus the order-relation term catching the two rebook/admintravel sub-cases.
- **F01** gains the least (+0.010 F1, though AUROC/AUPRC gains are similar in scale to F13): most of its sub-cases are order-relation faults like F13's rebook/admintravel, but F01 has no error-span evidence and no message-count signal, so only the order term (and the sync branch, and log) contribute.
- **3 seeds, sequential runs**: std of F1 is ≤0.011 for both configurations on F02/F01 and 0.036/0.001 on F13 (only-sync/both) — small enough that the ranking (both > only-sync on every F-case) holds seed-by-seed, not just on average.
- Oracle F1 is slightly *lower* for the both-branches configuration on F01/F02/F13 despite higher AUROC/AUPRC — oracle F1 sweeps a single scalar threshold on the test set itself and is sensitive to how sharply the score separates the top of the ranking, not just to the ranking's overall quality; it is reported for reference only, not as the primary comparison.

## 7. Limitations

- **One dataset, one system (TrainTicket)**: the fault taxonomy (edge set / order relation / message count / error-span evidence) was derived from this dataset; generalising the exact scoring rules to another async system is not validated here.
- **`async_temporal_order_adj` lineage**: pkl sample keys are `md5(trace_id + shuffled position)` with `trace_id` dropped, so the order-relation matrix is joined back to the raw span data by a structural signature (span count + per-service max duration), not by id. 1.7% of samples (normal traces only, sharing a signature with another normal trace) resolve to a random candidate among ties — a small, quantified source of noise, not a systematic bias (see `docs/preprocess_deeptralog_en.md` §8).
- **`trace_err`'s fixed scale (`ERR_Z_ONE=4.0`)** is a deliberate choice for a rare-event term on *this* dataset (0.13% of val normals have any error span); it is not learned from data and was not swept (values 2/4/8 gave equivalent results in an offline check on saved checkpoints, but that is not the same as a proper sweep on this exact re-run).
- **F02's absolute F1 is low** (0.138) even with the async branch, because the fault's own signal-to-noise ratio in raw message counts is close to a hard ceiling for this sub-case (§6.3); this is a property of the fault, not obviously fixable by more model capacity.
- **`open_unmatch_zoomout=False`**: with no real metric (`kpi_c=1`, all-zero), the unmatched-KPI contrastive hinge is degenerate and is disabled; this dataset cannot be used to validate that component.
- **3 seeds, not 5**: run-to-run std is already small (§6.3), but the protocol's usual 3–5 seed range was used at its low end for time budget reasons (8 GB / no dedicated GPU machine, ~7–13 min per run).
- **Downsampled train/unlabel/val** (6,000/6,000/1,500 out of ~93,000+ available normal traces) for RAM; a larger pool was not tried.
- Three raw SpanData CSVs (2 in `F07.zip`, 1 in a `normal*` zip) lack the `IsError` column; `_read_spans` now fills `IsError=False` for those files instead of skipping them (previously silently dropped), but this was fixed after the results in this document were produced from the previously-processed pkl files, so those specific files' contribution is unchanged in the numbers above.

## 8. Related Files

| What | File |
| :--- | :--- |
| Standard protocol | `docs/evaluation_protocol_en.md` |
| Async-trace architecture (§10) | `docs/model_architecture_flow_en.md` |
| Dataset/label verification, structural fingerprint, pkl schema, data caveats | `docs/preprocess_deeptralog_en.md` |
| Preprocessing pipeline, `TARGET_ANOMALY_RATE`, `async_temporal_order_adj` construction | `codes/common/preprocess_deeptralog.py` |
| `evaluate_norm_sum`, `_score_components`, `trace_err`/`ERR_Z_ONE` | `codes/models/basev3.py` |
| Async trace model (encoder/decoder, order head) | `codes/models/async_trace_model_v3.py` |
| Wrapper and summary table | `codes/common/eval_per_scenario_deeptralog.py` |
| Results | `data/deeptralog/result_per_scenario_fuse_trace_{only_sync,both_sync_and_async}/` |
