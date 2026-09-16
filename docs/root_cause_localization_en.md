# Root Cause Localization (TraceDAE §E)

> Adds root-cause candidate ranking on top of the existing trace branch
> (`--open_trace True`). No new model, no new hyperparameter for the
> scoring itself — reuses the dual-autoencoder reconstruction already
> computed by `TraceModel` (`codes/models/trace_model_v3.py`).

---

## 1. Mechanism

TraceDAE §E: when a trace/window is flagged anomalous, compute a
per-node reconstruction score, rank service nodes by it, and report the
top-k as root-cause candidates.

| Paper (Eq. 14) | UAM-AD |
|---|---|
| `S_i = α‖A_i−Â_i‖² + (1-α)‖X_i−X̂_i‖²` | `node_scores[i] = loss_struct_per_node[i] + lambda_lat * loss_latency_per_node[i] + lambda_err * loss_error_per_node[i]` — same weighting `TraceModel` already uses for the aggregate training loss, just kept per-node instead of reduced away. |
| node → microservice | `idx2service`, built from `meta.pkl["service2idx"]` |
| "for each abnormal STG" | reuse the threshold already chosen by `BaseModel.evaluate()` |

Implementation: `TraceModel.forward` (`codes/models/trace_model_v3.py`)
returns a 5th value, `node_scores` `[B, N]`, computed alongside — not
instead of — the existing aggregate `loss` `[B]`. `MultiModel.forward`
(`codes/models/fuse_v3.py`) reshapes it to `[B, W, N]` and adds it to the
result dict as `"node_scores"`.

`BaseModel.localize_root_causes(test_loader, threshold, top_k=3)`
(`codes/models/basev3.py`) then, for every `(window, timestep)` whose
`fusion_loss > threshold`, ranks the `N` service nodes by `node_scores`
and keeps the top-k (`codes/models/rca.py::rank_top_k_services`).

## 2. HR@k and MRR

Standard RCA ranking metrics, computed only over anomalies whose true
root-cause service is known (`gt_service is not None`):

- **HR@k** (Hit Rate@k): fraction of scored anomalies where the true
  service appears in the top-k predicted candidates.
  `HR@k = |{records : gt_service ∈ top_k}| / |scored records|`
- **MRR** (Mean Reciprocal Rank): mean of `1 / rank(gt_service)` (0 if
  the true service isn't in the returned top-k at all).

Implemented in `codes/models/rca.py::compute_hit_rate_at_k`.

## 3. Ground truth per dataset

| Dataset | Ground truth available? | How it's derived |
|---|---|---|
| `rcaeval_re2_ob` | Yes, 100% of anomalies | Sample id is `f"{service}_{fault}_{run_id}_{i}"` — the injected service is parsed straight out of the id (`preprocess_rcaeval_re2_ob.py:358`). |
| `rcaeval_re3_ob` | Yes, 100% of anomalies | Same id scheme as RE2-OB, fault suffixes `f1..f5`. |
| `sn` | Yes, 9 of 12 scenarios | SN has no merged `test.pkl` — every run points `--test_pkl` at one `scenarios/test_<scenario_name>.pkl` file, so the scenario for the whole run is derived once from that filename (`codes/models/rca.py::scenario_from_test_pkl`). (The per-sample `"_scenario"` field written during `_process_scenario` — `preprocess_sn.py:727` — is stripped before the pkl is saved by `_to_dict`, `preprocess_sn.py:796-801`, so it can't be read back from the sample data itself.) `SCENARIO2SERVICE` (`codes/models/rca.py`) maps 9 scenario name prefixes (`Code_Stop_*`, `DB_Redis_CacheLimit_*`, `Svc_Kill_*`) to a specific service. The 3 `Perf_*` scenarios (`Perf_CPU_Contention`, `Perf_Disk_IO_Stress`, `Perf_Network_Loss`) are host/infra-level stress, not attributable to one service — they always resolve to `gt_service = None`. |

`gt_service` is only computed for samples where `true_label == 1` — a
pre-injection (normal) sample's id/scenario still names the experiment it
was recorded during, but that service isn't actually at fault for that
(normal) timestep.

## 4. How to run

**Fast path (recommended for testing) — reuse an existing checkpoint, no
training.** The repo has committed `open_trace=True` checkpoints under
`data/<dataset>/result_per_scenario_fuse_trace/<scenario>/<hash>/model.ckpt`
(check the matching `params.json` next to it for the exact
`window_size`/`hidden_size`/`num_services`/`trace_c`/`fuse_type` used, and
pass those on the command line so the architecture matches the saved
weights — also pass `--num_services`/`--trace_c` explicitly regardless, the
`meta.pkl` auto-load in `run.py` only fires when those flags are left at
their CLI default, which for RE2-OB/RE3-OB/SN never matches the real
value). **Verified working**: all 6 RE2-OB and all 5 RE3-OB checkpoints
load cleanly with the current `GATLayer`. **All 24 SN checkpoints predate
the decomposed-attention `GATLayer` refactor and fail to load**
(`size mismatch`/`missing key` on `gat1.a_l`/`gat1.a_r`) — for SN, train a
fresh (even short, e.g. `--epoches 3 3`) model instead until new
checkpoints are committed.

```bash
python codes/run.py --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob \
    --data_type fuse --open_trace True --window_size 30 --hidden_size 32 \
    --num_services 11 --trace_c 6 \
    --pre_model data/rcaeval_re2_ob/result_per_scenario_fuse_trace/cpu/671af35f/model.ckpt \
    --test_pkl data/rcaeval_re2_ob/test_cpu.pkl \
    --enable_rca True --rca_top_k 3
```

**Full training path**: add `--enable_rca True --rca_top_k K` to a normal
`run.py` training command (requires `--open_trace True`).

`--enable_rca` defaults to `False` — existing runs/results are unaffected
unless explicitly opted in.

## 5. Output

- `rca_results.json` (via `dump_rca_results`, `codes/common/utils.py`) —
  list of records, one per anomalous `(window, timestep)`:
  ```json
  {
    "sample_id": "checkoutservice_cpu_2_812",
    "true_label": 1,
    "top_k_services": [["checkoutservice", 0.41], ["cartservice", 0.18], ["frontend", 0.09]],
    "gt_service": "checkoutservice"
  }
  ```
- `info_score.txt` gets an appended `* RCA -- hr1:.. hr3:.. hr5:.. mrr:.. n_scored:..`
  line whenever at least one record has a known `gt_service`.

## 6. Known caveat — threshold protocol differs by dataset (out of scope here)

`sn` has `val.pkl` (last 20% of `Normal_Baseline`, never used for
training) so it can use the no-leak threshold protocol (`--val_percentile`,
`threshold = percentile(val_losses, val_percentile)`).
`rcaeval_re2_ob`/`rcaeval_re3_ob` don't produce a `val.pkl`, and their
`test_<fault>.pkl` normal samples are randomly drawn from the *entire*
normal pool without excluding ids already used in `train.pkl`/`unlabel.pkl`
(`preprocess_rcaeval_re2_ob.py:440`) — i.e. there's train/test overlap for
the normal class, beyond just the threshold. RCA itself still works
correctly on these two datasets regardless of which threshold protocol
picked the flagged windows, but the reported F1/precision/recall may be
optimistic. Fixing this requires reprocessing raw data and retraining —
tracked as a separate follow-up task, not part of this feature.
