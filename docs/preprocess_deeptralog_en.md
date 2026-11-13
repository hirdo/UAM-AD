# Preprocessing: DeepTraLog (TrainTicket) Dataset

## 1. Dataset Overview

The DeepTraLog dataset (`github.com/FudanSELab/DeepTraLog`, ICSE 2022) was collected from the TrainTicket system, and is used here to answer whether the model handles **synchronous (sync)** and **asynchronous (async)** call flows in a trace. It is the primary evidence for the "async" part of the model, for two reasons already checked (see the design conversation and dataset survey): the other TrainTicket datasets (Eadro, RCAEval RE2/RE3-TT, TrainTicketTrace) contain no real async faults; DeepTraLog does.

| Modality | Source | Location on GitHub |
|:---|:---|:---|
| Trace (span) | SkyWalking-style CSV, 1 row = 1 span | `TraceLogData/F01.zip` .. `F14.zip`, `TraceLogData/normal/*.zip` |
| Log | Raw log + Drain3-parsed log | same zips, `.log` files |
| Metric | **None** | — |

No metric is available — running log+trace requires disabling/faking the KPI branch (see the plan, "Running log+trace without a metric").

**14 fault cases** per the paper's Table 1, in 4 groups:

| Group (per paper) | Fault cases | Meaning |
|:---|:---|:---|
| **Asynchronous Interaction** | **F1, F2, F13** | Faults in the async message send/receive mechanism |
| Multi-Instance | F8, F11, F12 | Multiple instances of the same service with unsynchronized state |
| Configuration | F3, F4, F5, F7 | Wrong/inconsistent configuration |
| Monolithic | F6, F9, F10, F14 | Internal computation/logic faults within one service |

**Only the Asynchronous Interaction group (F01, F02, F13) is reliable async evidence** — the other three groups do not mean "sync fault"; they simply do not specifically target the async mechanism (see §3).

---

## 2. Definitions: sync/async edges, sync/async faults

### 2.1 Why a redefinition is needed

Every trace (including normal ones) is a mix of sync and async calls — TrainTicket uses RabbitMQ between `ts-food-service` → `ts-delivery-service`, and `SpringAsync` inside `ts-cancel-service`; everything else is a synchronous HTTP call. So "an async trace" is not a separate category; what matters is **which edge in a trace is sync vs async**, and **which part a given fault targets**.

### 2.2 Service-level graph of a trace

Each trace is collapsed into a graph: node = service, edge = `(A, B, kind)` meaning "A calls B", `kind ∈ {S, A}`. The rule for classifying the edge for a child span of A handled by B:

1. `Component ∈ {rabbitmq-producer, rabbitmq-consumer}` → **A** (explicit message send/receive)
2. `Component = SpringAsync` → **A** (framework-marked background task)
3. The child span **starts after the parent span has already ended** → **A**. This is a physical signal: in a synchronous call the parent must *wait* for the child, so the child's time interval is always nested inside the parent's; if the child starts after the parent already finished, the parent cannot have been waiting — it can only be "fire-and-forget".
4. None of the above → **S** (default).

Illustrative example (from F02, API `foodservice/createOrderBatch`):

```
ts-preserve-other-service  |----- HTTP POST /foodservice/orders ---------------|
ts-food-service                    |--createFoodOrder--|--(send message)--|
ts-delivery-service                                          |--receive message, save delivery--|
```
`preserve-other-service → food-service`: parent waits for child → **S**.
`food-service → delivery-service`: child starts after the parent (food-service) has already moved on → **A**.

### 2.3 "Unseen edge"

For a given API, pool all **normal** traces for that API into one **edge baseline**. An edge `(A, B, kind)` of a faulty trace is **"unseen"** if it is not in that baseline — i.e. the fault made a service-to-service connection appear (or changed an existing connection's kind) that never occurs normally.

*Current limitation:* only the "new edge appears" direction is measured so far, not "edge disappears". To be added when the real preprocessor is written (Step 3+).

### 2.4 Sync vs async fault (experimental, self-defined)

The paper (Table 1) has no "Synchronous Interaction" label — only the 4 groups in §1. From that, we define our own working split based on which quantity of the trace a fault changes:

| Fault type (self-defined) | Measured signature | Example |
|:---|:---|:---|
| **Structural (sync-type) fault** | Edge set changes: the faulty trace has an edge never seen in normal traces of the same API | F04: 100% of traces have an unseen edge |
| **Async fault (matches Table 1)** | Edge set **unchanged**, but the **count** of async edges or the **relative temporal order** between services changes | F02: edges unchanged, message count +21%. F01/F13: edges unchanged, 100% of traces have an order relation never seen before |

The "relative temporal order" between two services A, B in one trace: if every span of A ends before B's first span starts, `A < B`; the reverse gives `A > B`; if they overlap (parallel calls), no relation is recorded. The set of all such relations for a trace is its "order signature".

---

## 3. Label verification results (Step 1, 2026-09-22)

`GraphData/` (7 parts z01–z07 + zip, used to train DeepTraLog's original GGNN) is a split zip with a broken central-directory offset — standard `unzip` misreads entry 6 onward. The real local headers were located by scanning for the `PK\x03\x04` signature + filename, and decompressed with `zlib.decompress(..., -15)`.

- Exactly **132,485 traces**, **23,334 anomalous (17.6%)** — matches the paper's numbers.
- Joining `TraceId` (SpanData CSV) ↔ `trace_id` (GraphData) for all 14 `F01.zip`..`F14.zip` files: **12/14 match 100%**; F07 matches 84.0%, F08 matches 68.7% (the missing ones sit in the `back0729` subfolder, absent from the GraphData snapshot).
- **The true label comes from GraphData's `error_trace_type`, not the zip filename.** The zip filename is the paper's official numbering, but the internal numbering sometimes differs: F06.zip↔internal "F23", F09.zip↔"F24", F10.zip↔"F25" (pure renumbering). **F12.zip mixes labels:** 1,174/1,472 traces are internally "F12", the remaining 298 carry the "F13" label — using F12 requires filtering by per-trace `error_trace_type`, not by whole file.
- **The async group (F01, F02, F13) has zip-name = internal-label match at 100%**, no renumbering.

---

## 4. Per-F-case structural fingerprint results (Step 2, 2026-09-22)

Built a structural baseline (edge set, order-relation set, mean span count) for **22 APIs** from 9 `normal/*.zip` files (~93,000 normal traces), and compared every trace of the 14 F-cases against the baseline for the same API.

**Async group (F01, F02, F13): edge set unchanged, order/count change sharply — consistent:**

| F-case (API) | n traces | % unseen edges | % unseen order | span-count deviation |
|:---|--:|--:|--:|--:|
| F01 (`preserveservice/preserve`) | 400 | 0% | **100%** | +2% |
| F02 (`foodservice/createOrderBatch`) | 1,199 | 0% | 0% | **+21%** (matches the earlier hand-measured figure: 14.1 vs 11.6 msg/trace) |
| F13 (`rebookservice/rebook`) | 377 | 0% | **100%** | +1% |
| F13 (`admintravelservice/admintravel`) | 350 | 0% | **100%** | +3% |

**Control (genuine structural fault):** F04 (`preserveservice/preserve`, n=330): **100% unseen edges** — sharply different from the async group.

**Implication for model design:** the fault signal for F01/F02/F13 lives entirely in the two quantities (order, count) that the model's async branch is designed to measure (MSE decoder for count, order channel k2); the sync branch (which only tracks edges) sees nothing unusual in these traces — exactly the intended behaviour of the two-branch, non-interacting design.

**Caveats:**
1. Some secondary APIs (admin/config) give suspiciously round `span_dev%` values (-50.0%, -48.4%, -25.0%...) — likely shared "warm-up" requests across many test types, not specific to that F-case. These rows should not be used to draw conclusions.
2. F12.zip mixes in 298 F13-labelled traces (§3), so the "F12 → 66% unseen order" row may be contaminated by that async subset — not yet separated, since this measurement groups by API rather than filtering by per-trace `error_trace_type`.
3. F09, F10, F14 have no full API overlap with the normal set, so their experimental labels are indicative only.

The intermediate data used to compute the tables above lives outside the repo (`D:\ClaudeWork\dtl\`) and is not committed.

---

## 5. F-cases used for experiments

| Purpose | F-case | Note |
|:---|:---|:---|
| Primary evidence for async faults | **F01, F02, F13** | Matches both the paper's Table 1 and the measured structural fingerprint |
| Control example for structural (sync) faults | **F04** | 100% of traces have an unseen edge, the cleanest case |
| Secondary control (structural change, with caveats) | F03, F10, F14 | 34–77% unseen edges, but on small sub-APIs or with incomplete normal coverage |
| Not used to classify sync vs async | F05, F06, F07, F08, F09, F11, F12 | Mixed or too noisy to draw a firm conclusion |

## 6. Pipeline: one entry point, `codes/common/preprocess_deeptralog.py`

Like `preprocess_sn.py`, everything lives in one script with a `--stage` switch (and `eval_per_scenario_deeptralog.py` runs the per-F-case evaluation):

| Stage | What it does | Output |
|:---|:---|:---|
| `labels` | Per-trace labels from DeepTraLog's GraphData archive (a split zip whose central directory is broken, so it is read by scanning local headers and inflating raw deflate) | `--label_pkl` = `{trace_id: (trace_bool, error_trace_type)}`, `True` = normal; 132,485 traces, 23,334 (17.6%) anomalous |
| `normal` | Normal pool: Drain3 log templates, latency baseline, `train`/`unlabel`/`val` samples (reservoir sampling), plus a cache for `fcase` | `train.pkl`, `unlabel.pkl`, `val.pkl`, `meta.pkl`, `_cache.pkl` |
| `fcase` | One F-case against that cache: its anomalous traces + normal traces to reach 12.5% anomalies, shuffled | `scenarios/test_{F}.pkl` |
| `all` | `normal` + `fcase` in one process (smoke runs only) | |

```
python codes/common/preprocess_deeptralog.py --stage labels --graphdata_dir <GraphData dir> --label_pkl <labels.pkl>
python codes/common/preprocess_deeptralog.py --stage normal --fault_dir <F*.zip dir> --normal_dir <normal dir> --label_pkl <labels.pkl> --output_dir data/deeptralog
python codes/common/preprocess_deeptralog.py --stage fcase  --fcases F01 --fault_dir ... --normal_dir ... --label_pkl ... --output_dir data/deeptralog
```
Run each stage (and each F-case) as its own process: on an 8 GB machine a single long-lived process got killed for low memory even with streaming I/O. Evaluation results go to `data/deeptralog/result_per_scenario_*` inside the same dataset folder, as for SN.

## 7. pkl schema (1 step = 1 trace)

| Key | Shape | Meaning |
|:---|:---|:---|
| `label` | int | 1 = anomalous trace |
| `logs`, `log_features` | list / vector | Drain3 templates of the trace's log lines; `log_features` is rebuilt at load time (`semantics.py`) |
| `kpis` | `[1]` | placeholder (no metric in this dataset) |
| `trace_node_features` | `[35, 6]` | sync branch: `[call_count, avg_dur, max_dur, error_rate, root_rate, latency_dev]` per service |
| `trace_adj` | `[35, 35]` | sync branch: binary symmetric call graph (sync edges only) |
| `async_trace_node_features` | `[35, 3]` | async branch: `[log1p(#messages sent), log1p(#messages received), log1p(mean consumer lag, s)]` |
| `async_msg_count_adj` | `[35, 35]` | async branch, relation 1: directed, `log1p(#messages i→j)` |
| `async_temporal_order_adj` | `[35, 35]` | async branch, relation 2 (`_order_adj`): `[i,j]=1` iff every span of service i ended before the first span of service j started; `[i,i]=1` marks a service present in the trace. Uses all spans, carries no durations |

`meta.pkl`: `num_services`, `service2idx`, `trace_c=6`, `async_c=3`, `async_edge_mask` (service pairs that ever carry an async edge in the normal data), `async_order=True`, `scenario_names`, ...

## 8. Data caveats (measured)

- **Lineage of `async_temporal_order_adj` in the current `data/deeptralog`.** The pkl keys are `md5(trace_id + position)` and the raw trace id is not stored, so the key was added afterwards by matching each sample to its raw trace through a signature (per service: number of spans and max span duration). All 61,364 samples matched; 1.7% (normal traces only) had several raw traces with the same signature but different order matrices, and one was picked. The native computation in `_order_adj` was checked to give identical matrices for 1,779/1,779 F13 anomalies and 1,855/1,855 sampled normal traces. A fresh run of the stages writes the key natively (and may draw a different sample of normal traces).
- **Sub-cases.** Each F-case zip holds 5 sub-cases (e.g. F01: cancel, preserve, execute, travel-plan cheapest/quickest), each hitting a different API, and every trace in an F-case zip is an anomaly (normals come from the normal zips). F01-04/05 (travel-plan, 47% of F01's anomalies) use APIs absent from the normal data.
- **What the F01/F13 anomalies look like.** Most carry a ~4 s delay (request duration ≈4.0–4.8 s vs 7–216 ms) plus 2 extra `SpringAsync` spans; on 7 of 9 groups the request duration alone gives AUROC 0.975–1.0 against normals of the same API. In preserve/preserveOther (and rebook, trips/left, admin-travel) the service-level order relation also changes (100% of anomalies vs 0% of normals, including naturally slow ones), which latency cannot explain. F13-02 (trips/left) traces are "silent" (≈18 spans, ≈14 ms vs ≈133 spans).
- **F02** = 1,198 traces rooted at food-service (createOrderBatch, ≈+20% async messages) + 952 rooted at travel-service (almost no async messages). Metrics pooled over an F-case therefore mix fault signal with API composition; compare with normals of the same API when it matters.
- `train`/`unlabel` (6,000) and `val` (1,500) are down-sampled from the 20,000-trace pool for RAM; only about 130 val traces carry async messages.
