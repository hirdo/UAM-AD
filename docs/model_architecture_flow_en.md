# HADES + Trace — Model Architecture Flow

> This document describes the architecture and data flow of the HADES model extended with the Trace branch (GAT Structure Autoencoder).
> **Version**: `fuse_v3.py` with 7 architectural changes (based on TraceDAE).

---

## 1. Project Architecture Overview

```
UAM-AD/
├── codes/
│   ├── run.py                          ← Main entry point
│   ├── run_sequential.py               ← Sequential run (avoids CUDA OOM)
│   ├── common/
│   │   ├── data_loads.py               ← Load & window data → DataLoader
│   │   ├── semantics.py                ← Extract log features (Word2Vec/template)
│   │   ├── utils.py                    ← General utilities (seed, dump results...)
│   │   ├── preprocess_XX.py            ← Build pkl from raw dataset XX
|   |   └── eval_per_scenario_XX.py     ← Eval for dataset XX have many inject fault types/ scenarios
│   └── models/
│       ├── basev3.py                   ← Train/eval loop (BaseModel)
│       ├── fuse_v3.py                  ← Multi-modal model (log+metric+trace)
│       ├── log_model_v3.py             ← Log encoder (Transformer)
│       ├── kpi_model_v3.py             ← Metric encoder (Transformer)
│       ├── trace_model_v3.py           ← Sync trace encoder (GAT) + TraceModel
│       ├── async_trace_model_v3.py     ← Async trace branch (CHANGE 9, §10) — additive, sync branch untouched
│       └── utils.py                    ← Shared modules (Attention, ...)
└── data/
    └── XX/
        ├── train.pkl / unlabel.pkl / test.pkl
        └── meta.pkl
```

---

## 2. Input Data Flow

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INPUT  [B, W, *]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  log_features        [B, W, log_c]          ← log template features
  kpi_features        [B, W, kpi_c]          ← KPI metrics
  trace_node_features [B, W, N, trace_c]     ← service node features (trace_c=6)
  trace_adj           [B, W, N, N]           ← service call graph (STG)
  unmatched_kpi       [B, W, kpi_c]          ← shuffled KPI from other windows

  B = batch size | W = window size (numbers based on dataset) | N = num_services (numbers based on dataset)
  H = hidden_size (32) | trace_c = 6

  Node feature layout (trace_c=6):
    col 0: call_count   — span count, normalized
    col 1: avg_dur_ms   — mean duration, normalized
    col 2: max_dur_ms   — max duration, normalized
    col 3: error_rate   — error fraction ∈ [0,1]
    col 4: root_rate    — root span fraction ∈ [0,1]
    col 5: latency_dev  — z-score(avg_dur vs pre-fault baseline)
```

---

## 3. Generator — `MultiModel.forward()`

### 3.1 MultiEncoder — [CHANGE 1] Trace Separated from Self-Attention

> **Change**:
> Now Self-Attention is applied **only** to log+KPI. Trace encoder runs **separately** → `ZV`.
> (structural info should guide the **decoder**, not encoder attention).

```
  ┌──────────────────────────── MultiEncoder ─────────────────────────────┐
  │                                                                        │
  │  log_features ──► LogEncoder (4-layer Transformer) ──► log_re [B,W,H] │
  │                                                                        │
  │  kpi_features ──► KpiEncoder (Transformer)          ──► kpi_re [B,W,H] │
  │                                                                        │
  │  cat([kpi_re ‖ log_re]) ──► Self-Attention (2H, n_heads=2)            │
  │                                  └──► fused_modal [B, W, 2H]          │
  │                                                                        │
  │  trace_nodes ──► TraceEncoder (2-layer GAT)                            │
  │  trace_adj        reshape [B*W, N, C] → z [B*W, N, H]                │
  │                   mean_pool over N → ZV [B, W, H]     (separate)      │
  └────────────────────────────────────────────────────────────────────────┘

  Returns: (fused_kpi, fused_log, fused_modal [B,W,2H], ZV [B,W,H])
```

> **Without trace** (`open_trace=False`): ZV = None, fused_modal remains `[B,W,2H]`.

---

### 3.2 Reconstruction — [CHANGE 2] ZV Injected into Decoder

> **Change**: Decoder receives `cat([fused_modal, ZV])` instead of only `fused_modal`.
> Rationale: Inspired by TraceDAE (Eq.10) — structural embedding ZV should guide **reconstruction**,
> not fusion attention. Analogous to `X_hat = ZV · ZA^T` in TraceDAE.

```
  fused_modal [B,W,2H] ──┐
                          ├──► cat → [B,W,3H] ──► fuse_decoder (Linear 3H→kpi_c+log_c)
  ZV [B,W,H] ────────────┘                                │
                                               fused_out [B, W, kpi_c+log_c]
                                              ┌────────────┴────────────┐
                                         kpi_out [B,W,kpi_c]    log_out [B,W,log_c]

  kpi_dis = L1(kpi_out, kpi_features).mean(dim=-1)    [B, W]
  log_dis = L1(log_out, log_features).mean(dim=-1)    [B, W]
```

> **Without trace** (ZV=None): fuse_decoder = Linear(2H → kpi_c+log_c) — decoder input = 2H.

---

### 3.3 Trace Autoencoder — [CHANGE 3] adj_hat + [CHANGE 6] Attribute Reconstruction Loss

> **CHANGE 3**: `adj_hat` (reconstructed adjacency matrix) is returned from `MultiModel`
> so the Discriminator can use it as "fake trace adjacency".
>
> **CHANGE 6** : TraceModel adds an **attribute decoder** with two additional reconstruction losses:
> - `loss_latency`: MSE on `latency_dev` (col 5) — detects services slower than their pre-fault baseline
> - `loss_error`: BCE on `error_rate` (col 3) — detects services with elevated error rates
>
> Purpose: forces node embeddings ZV to encode both topology and attribute anomaly information,
> making `trace_dis` more discriminative for delay/loss/mem/socket fault types.

```
  trace_nodes ──► TraceEncoder (2-layer GAT) ──► ZV [B*W, N, H]
                                                       │
                   ┌───────────────────────────────────┼─────────────────────────┐
                   │                                   │                         │
           Structural decoder                  Attribute decoder          adj_hat return
           A_hat = sigmoid(ZV·ZVᵀ)            X_hat = Linear(ZV)
           loss_struct = BCE(A_hat, adj)       loss_lat = MSE(X_hat[:,5], x[:,5])
           .mean(dim=[-2,-1])  [B]             loss_err = BCE(σ(X_hat[:,3]),      [B]
                                                              x[:,3].clamp(0,1))  [B]
                   │
  trace_dis = loss_struct + λ_lat × loss_lat + λ_err × loss_err    [B]
              (λ_lat = λ_err = 0.5 by default)

  adj_hat_4d  = A_hat.reshape(B, W, N, N)      ← CHANGE 3: returned from MultiModel output
  feats_hat_4d = feats_hat[:,:,[3,5]].reshape(B, W, N, 2)  ← CHANGE 7: returned for attr discriminator
```

---

### 3.4 Fusion Loss — [CHANGE 4] Variance-based Alpha (replaces Learnable trace_alpha)

> **Change**: `trace_alpha = nn.Parameter(-2.2)` (learnable as TraceDAE design) → Learnable alpha replaced by **variance-based alpha** — no parameters.
>
> **Problem with learnable alpha**: Gradients push `trace_alpha` to converge so as to balance
> *reconstruction error magnitudes* on normal data — not actual discriminativeness. On datasets
> with weak trace signal (e.g. RE3-OB), `trace_dis` is nearly constant → alpha increases to
> compensate → noise is injected into the anomaly score.
>
> **Variance-based solution**: α directly reflects the relative discriminativeness of the trace signal
> compared to log+KPI. When trace is noisy → var(trace_dis) is low → α → 0 automatically.

```
  log_d   = log_dis  × expand_anomaly_gap(log_dis)
  kpi_d   = kpi_dis  × expand_anomaly_gap(kpi_dis)
  trace_d = trace_dis × expand_anomaly_gap(trace_dis)

  log_kpi_loss = log_d + kpi_d + narrow_modal_gap(|log_d − kpi_d|)

  var_lk    = var(log_kpi_loss.detach())      ← no gradient
  var_trace = var(trace_dis.detach())
  α = var_trace / (var_lk + var_trace + ε)    ← ∈ [0, 1], no learned parameters

  fusion_loss = (1 − α) × log_kpi_loss + α × trace_d    [B, W]
                └─────────────────────────────────────────────────────┘
                         Anomaly Score (used for evaluation)
```

---

### 3.5 Contrastive Loss (Unmatched pairs)

```
  contrastive = max(0,  L1(kpi_features, kpi_out)
                      + unmatch_k
                      − L1(unmatched_kpi, kpi_out_unmatched))
```

---

### 3.6 Residual-Gated Trace Fusion — [CHANGE 8] (active whenever `open_trace=True`)

> **Problem**: Prior decoder always used `cat([fused_modal, ZV])` → when trace is uninformative
> (e.g. RE3-OB code-defect faults), noisy ZV propagated into `kpi_out` / `log_out` and hurt F1
> *below baseline*. The variance-based α in §3.4 only re-weights the **loss**, not the decoder output.
>
> **Fix**: residual form with a per-sample gate that can completely close the trace contribution.
>
> - `base_decoder`: `Linear(2H → H) → ReLU → Linear(H → kpi_c+log_c)` — baseline path on log+KPI only
> - `delta_head`:   `Linear(3H → 2H) → ReLU → Linear(2H → kpi_c+log_c)` — **zero-init final layer**
> - `trace_gate`:   `Linear(6 → 16) → ReLU → Linear(16 → 1) → sigmoid` — bias init to `−2.0` → g₀ ≈ 0.12
> - **Trace-quality features** (per B,W): mean call count, coverage (non-zero spans), mean error_rate,
>   mean |latency_dev|, adjacency density, call-count variance. Log1p-normalized.

```
  Trace-quality features [B, W, 6]
         │
   trace_gate (MLP, bias=-2.0) ──► g ∈ (0, 1)   [B, W, 1]
                                          │
  fused_modal [B,W,2H] ──► base_decoder ──► y_base [B,W,kpi_c+log_c]
                                          │
  cat([fm, ZV]) [B,W,3H] ─► delta_head ──► Δ [B,W,kpi_c+log_c]   (zero-init → Δ≈0 at start)
                                          │
                            fused_out = y_base + g · Δ    ← g=0 ⇒ exact baseline

  fusion_loss = log_kpi_loss + g · trace_d + gate_lambda · g.mean()
                                            └─── L1 reg keeps gate closed by default
```

> **Guarantee**: at initialization `Δ ≈ 0` and `g ≈ 0.12`; gradient only opens the gate if
> `Δ` reduces log+KPI reconstruction loss by more than `gate_lambda` (default 0.01).
> On trace-noisy datasets the gate stays closed and the model is *exactly* equivalent to baseline.
>
> **CLI**: `--gate_lambda 0.01` (auto-applied when `--open_trace True`).
>
> **Validation**: On RE3-OB (5 code-defect fault types) the residual-gated variant recovers
> baseline F1 within 0.015 on every fault type.
> See `experiment_results_re3_ob_trace_vs_baseline_en.md`.

---

## 4. Discriminator — `MultiDiscriminator.get_loss()`

### 4.1 Structural Trace Discrimination — [CHANGE 5] (unchanged)

> **Change**: FAKE pass uses `adj_hat` (adjacency reconstructed by the Structure AE) →
> `trace_re_fake ≠ trace_re` → loss is meaningful.
> Analogous to how log_out/kpi_out are "fake" for log/KPI, `adj_hat` is "fake" for trace structure.
>
> **Note on `trace_nodes`**: `trace_nodes` is identical in both REAL and FAKE passes — it contributes
> zero discriminative signal on its own. It serves only as node feature input for GAT attention
> computation in `encoder_low`. The entire structural discrimination signal comes from `adj` vs `adj_hat`.

```
  MultiEncoder_low (lightweight, 1-layer GAT):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  REAL:  encoder_low(log_x,   kpi_x,   trace_nodes, trace_adj)        │
  │         → log_re, kpi_re, trace_re              [B, W, H]            │
  │                                                                       │
  │  FAKE:  encoder_low(log_out, kpi_out, trace_nodes, adj_hat)  ← CHANGE│
  │         → log_re_fake, kpi_re_fake, trace_re_fake                    │
  │         (adj_hat from Structure AE → trace_re_fake ≠ trace_re  ✅)   │
  └──────────────────────────────────────────────────────────────────────┘

  Discriminator loss (update Discriminator weights):
  disc_loss = CE(pred_kpi,       real=1) + CE(pred_kpi_fake,   fake=0)
            + CE(pred_log,       real=1) + CE(pred_log_fake,   fake=0)
            + CE(pred_trace,     real=1) + CE(pred_trace_fake, fake=0)

  Deceive loss (update Generator weights):
  deceive_loss = MSE(kpi_re, kpi_re_fake)
               + MSE(log_re, log_re_fake)
               + MSE(trace_re, trace_re_fake)
```

---

### 4.2 Attribute Trace Discrimination — [CHANGE 7] (new, additive)

> **Motivation**: The structural head (CHANGE 5) only discriminates via topology (adj vs adj_hat).
> Node attributes are identical in REAL and FAKE → attributes contribute nothing to discrimination.
> Adding a separate attribute head that compares ground-truth vs reconstructed node attributes
> forces the Generator to produce realistic `error_rate` and `latency_dev` values.
>
> **Why only col 3 & col 5?**
> - `error_rate` (col 3) and `latency_dev` (col 5) have explicit supervision in CHANGE 6
>   → `feats_hat[:,:,[3,5]]` quality is reliable.
> - Other 4 cols (call_count, avg_dur_ms, max_dur_ms, root_rate) have no explicit loss
>   → reconstruction quality is poor → using them would inject noise.
>
> **Existing structural code is unchanged** — the attribute head is purely additive.

```
  Attribute head (separate from structural, no encoder_low involved):

  REAL:  trace_nodes[:, :, [3, 5]]            [B, W, N, 2]
         → mean over N → attr_real            [B*W, 2]

  FAKE:  feats_hat[:, :, [3, 5]]              [B, W, N, 2]   ← from TraceModel (CHANGE 7)
         → mean over N → attr_fake            [B*W, 2]

  attr_classifier: Linear(2, H) → ReLU → Linear(H, 1) → sigmoid
                                                          [B*W, 1]

  attr_disc_loss = BCE(attr_classifier(attr_real), real=1)
                 + BCE(attr_classifier(attr_fake), fake=0)

  attr_deceive_loss = MSE(attr_classifier(attr_real),
                          attr_classifier(attr_fake))

  Total losses (combined):
  disc_loss    += attr_disc_loss
  deceive_loss += attr_deceive_loss
```

---

## 5. Training Loop (per epoch)

```
  for batch in unlabel_loader:

    ① Generator step:
        res = model(batch)
        loss_G = res["loss"]
                + λ₁ × discriminator.get_loss(batch, res)["deceive_loss"]
        loss_G.backward() → optimizer_G.step()

    ② Discriminator step:
        loss_D = discriminator.get_loss(batch, res)["loss"]
        loss_D.backward() → optimizer_D.step()

    ③ Evaluation (end of epoch):
        score = fusion_loss [B, W]
        → point_adjustment(pred, gt)  ← if any point in a segment is detected
                                         → entire segment = detected
        → F1 / Recall / Precision
```

---

## 6. Routing by `data_type`

```
  data_type = "fuse"  →  MultiModel   (log + KPI + trace if open_trace=True)
  data_type = "log"   →  LogModel     (log only)
  data_type = "kpi"   →  KpiModel     (KPI only, open_trace ignored)
```

> ⚠️ Only `data_type=fuse` activates the trace branch.

---

## 7. Architecture Comparison: Before vs After (fuse_v3.py)

| #   | Component                   | Before (old)                                       | After (new)                                                                    |
|:---:|:----------------------------|:---------------------------------------------------|:-------------------------------------------------------------------------------|
|  1  | **Self-Attention**          | cat([log‖kpi]) → 2H                               | cat([log‖kpi]) → 2H, trace **separated**                                       |
|  2  | **Decoder input**           | fused_modal [B,W,2H or 3H]                         | cat([fused_modal, ZV]) → [B,W,3H]                                              |
|  3  | **adj_hat**                 | not used                                           | returned from MultiModel for use by Discriminator                              |
|  4  | **trace_alpha**             | `nn.Parameter(−2.2)`, learned ≈ 0.10              | **variance-based**: `α = var(trace) / (var(lk) + var(trace) + ε)`             |
|  5  | **Discriminator FAKE**      | uses `trace_adj` (real) → contradiction            | uses `adj_hat` (reconstructed) → valid loss                                    |
|  6  | **trace_dis**               | BCE(A_hat, adj) — structural loss only             | + λ_lat×MSE(latency_dev) + λ_err×BCE(error_rate)                              |
|  7  | **Attribute Discriminator** | not present — node attributes not discriminated    | separate head: REAL=trace_nodes[:,:,[3,5]], FAKE=feats_hat[:,:,[3,5]] → Linear(2,H)→ReLU→Linear(H,1) |
|  8  | **Decoder fusion mode**     | always `cat([fm, ZV])` → noise leaks into kpi_out / log_out when trace is non-informative | **Residual-gated**: `y_base(fm) + g · delta_head(cat[fm,ZV])`, `g∈[0,1]` per-sample from 6 trace-quality features, `delta_head` zero-init, L1 reg on g (`gate_lambda`) |
|  9  | **Async trace branch**      | not present                                        | **Additive, independent branch** (§10): own encoder/decoder/gate for message-count + temporal-order relations between services; its embedding rides in the SAME `delta_head` (widened by one more H block) alongside the sync embedding; off by default (`open_async_trace=False` ⇒ byte-identical to row 1-8 behaviour) |

---

## 8. End-to-End Overview Diagram (after improvements)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  log_x [B,W,log_c]  ──┐
  kpi_x [B,W,kpi_c]  ──┼──► MultiEncoder ──► fused_modal [B,W,2H]
                        │    (Self-Attn      │
  trace [B,W,N,C]    ──┘     log+KPI only)  └──► ZV [B,W,H]  (GAT)
  adj   [B,W,N,N]    ──────────────────────────────│
                                                   │
                        ┌─── base_decoder(fused_modal) ─────► y_base ─┐
                        │                                                │
                        │  cat([fused_modal, ZV]) [B,W,3H]              │
                        │           │                                    │
                        │    delta_head (zero-init) ─► Δ [B,W,out] ───►  +  ◄── g·Δ
                        │                                    ▲           │
                        │   trace-quality feats [B,W,6] ─► trace_gate → g│
                        └──────────── (g→0 when trace is uninformative → y ≡ baseline) ┘
                                                         │
                                                 fused_out = y_base + g·Δ
                                                         │
                        kpi_out [B,W,kpi_c] + log_out [B,W,log_c]
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
    kpi_dis [B,W]            log_dis [B,W]            trace_dis [B,W]
    L1(kpi_out, kpi_x)       L1(log_out, log_x)       BCE(A_hat, adj)
                                                      + λ_lat×MSE(X_hat[:,5], x[:,5])
                                                      + λ_err×BCE(σ(X_hat[:,3]), x[:,3])
         │                          │                          │
         └──────────────────────────┴──────────────────────────┘
                                    │
                    α = var(trace_dis) / (var(log_kpi) + var(trace_dis) + ε)  ← variance-based
                    fusion_loss = (1-α)×(log+kpi) + α×trace    [B,W]
                             = Anomaly Score

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DISCRIMINATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Structural head (CHANGE 5):
    REAL: encoder_low(log_x,   kpi_x,   trace_nodes, adj)     → trace_re
    FAKE: encoder_low(log_out, kpi_out, trace_nodes, adj_hat) → trace_re_fake
    disc_loss    += CE(trace_re, real=1) + CE(trace_re_fake, fake=0)
    deceive_loss += MSE(trace_re, trace_re_fake)

  Attribute head (CHANGE 7 — new, separate):
    REAL: trace_nodes[:,:,[3,5]].mean(N) → attr_real  [B*W, 2]
    FAKE: feats_hat[:,:,[3,5]].mean(N)  → attr_fake  [B*W, 2]
    attr_classifier: Linear(2,H) → ReLU → Linear(H,1)
    disc_loss    += BCE(attr_classifier(attr_real), 1) + BCE(attr_classifier(attr_fake), 0)
    deceive_loss += MSE(attr_classifier(attr_real), attr_classifier(attr_fake))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT: F1 / Recall / Precision (with point-adjustment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 9. Fault Category Compatibility with Trace Branch

> **When does trace help?** Only when the fault changes **observable service-to-service interaction patterns**
> — i.e., the call graph topology, latency between services, or cross-service error rates.
> When a fault stays **internal to a service** without affecting these patterns, trace is non-discriminative
> and CHANGE 8 (hard gate) ensures it is silently disabled.

| Fault Category                                        | Trace useful? | Reason |
| **Network** (delay, packet loss, bandwidth limit)     | ✅ Strong     | Direct latency spike + error_rate change between services |
| **Resource** (CPU, memory, disk overload)             | ✅ Moderate   | Overload → slow service → latency_dev increases |
| **Service crash / OOM**                               | ✅ Strong     | Error rate spike, call count drops |
| **Code Logic / Defect**                               | ❌ No         | Bug runs internally — no change in call graph or latency |
| **Configuration error**                               | ❌ No         | Misconfigured value, service still responds normally |
| **Database internal** (slow query, deadlock)          | ⚠️ Partial    | Only visible if DB is in the service mesh; latency of the calling service may increase |
| **Business logic** (wrong calculation, wrong output)  | ❌ No         | Output incorrect but performance characteristics unchanged |
| **Security** (auth bypass, injection)                 | ❌ No         | No change in call pattern or latency |

---

## 10. Async Trace Branch — [CHANGE 9] (additive, `open_async_trace=True`)

> Validated on DeepTraLog (TrainTicket) — see [`experiment_results_deeptralog_trace_both_sync_async_vs_only_sync_en.md`](experiment_results_deeptralog_trace_both_sync_async_vs_only_sync_en.md) for the full results, [`preprocess_deeptralog_en.md`](preprocess_deeptralog_en.md) for the dataset/label pipeline.

### 10.0 Why a second branch, not a bigger sync branch

The sync trace branch (§3.3) tracks one thing: an undirected, unweighted service-call **edge set** (`trace_adj`, presence/absence). Measured on DeepTraLog's structural fingerprint (§4 of the preprocess doc), the three async fault cases (F01, F02, F13) leave that edge set **100% unchanged** — the fault instead shifts the **message count** between two services (F02, +21%), the **relative temporal order** in which services are called (F01/F13, 100% of traces get a never-seen-before order relation), or makes the request **fail fast with an error span** (part of F13). None of these three quantities has a representation in the sync branch's node features or adjacency, and a symmetric presence/absence decoder is structurally blind to "the same edge, but the count went up" or "the same two services, but now A happens after B instead of before". So instead of overloading `trace_model_v3.py` (used, unmodified, by every other dataset) with new columns/decoders, the async branch is a **second, independent model** (`async_trace_model_v3.py`) reusing the same `TraceEncoder`/`GATLayer` building block, added purely alongside — off by default, and when off the sync branch's tensors and parameter count are byte-identical to before this change (verified: 2,598 / 48,003 / 129 sync-branch parameters, unchanged with `open_async_trace` True or False).

### 10.1 Input (per trace, node = service, same `N` as the sync branch)

```
  async_trace_node_features [B,W,N,3]   col 0: log1p(# async out-edges, this service as sender)
                                          col 1: log1p(# async in-edges, this service as receiver)
                                          col 2: log1p(mean consumer lag in seconds), 0 if col1==0

  async_msg_count_adj  [B,W,N,N]   directed, weighted: log1p(# async messages i→j) — NOT symmetrized
                                 (who sends to whom is exactly what a message-count fault needs)

  async_temporal_order_adj  [B,W,N,N]   directed, binary: [i,j]=1 iff every span of service i ended
                                 before service j's first span started (service-level version of
                                 DeepTraLog's TEG "Sequence" edge); diagonal [i,i]=1 marks service i
                                 as PRESENT in the trace (this is how "present services" is known
                                 without a separate mask key)
```
`async_msg_count_adj`/`async_temporal_order_adj` come from `preprocess_deeptralog.py`'s `_build_edges`/`_order_adj`; a dataset without them simply has no async branch (`open_async_trace` requires both keys; `async_order` additionally requires `async_temporal_order_adj`, else `async_encoder_inputs` raises `ValueError` rather than silently degrading).

### 10.2 Encoder — `async_encoder_inputs` (shared helper, `async_trace_model_v3.py`)

```
  x [B,N,3] ──┐
              ├─ use_order? ──► cat([x, one_hot(present services)]) [B,N,3+N]  ──► TraceEncoder ──► z [B,N,H]
  order_adj ──┘                  mask = (adj>0) | (present_i AND present_j)     (2-layer GAT, same
                                  ── the ORDER relation itself is NOT in the      class as sync branch)
                                     mask: it is the target to predict, not
                                     something attention may copy from
```
Two independent instantiations of this encoder, unshared weights (exactly mirroring how the sync branch has one `TraceEncoder` inside `AsyncTraceModel` for the autoencoder and one inside `MultiEncoder` for the Fused-decoder embedding):
- `AsyncTraceModel.encoder` — feeds the async autoencoder (§10.3).
- `MultiEncoder.async_trace_encoder` — feeds the embedding that goes into the shared `delta_head` (§10.4).

Without node identity, most async node features are near-zero (a service with no message activity has all three columns ≈0), so services that exchange no message would be indistinguishable and no head could learn *which pair* is unusual; the one-hot identity fixes that. The mask deliberately excludes the order relation itself — including it let early experiments' attention copy the relation through edge-sharing instead of predicting it, which hid genuinely unseen relations (this was diagnosed and fixed during development, see the `async_encoder_inputs` docstring).

### 10.3 Async Autoencoder — `AsyncTraceModel` (`async_trace_model_v3.py`)

```
  z [B,N,H] (from TraceEncoder above)
       │
       ├─────────────────────────────┬─────────────────────────────┐
  Structural (count) decoder   Attribute decoder            Order decoder (if async_order)
  adj_hat = softplus(          feats_hat = Linear(z)        logits = (z·W_order)·zᵀ
    (z·W_edge)·zᵀ )              [B,N,3]                     (directed)
  MSE(adj_hat, async_adj)      MSE(feats_hat, x)            BCE(logits, order_adj)
  × cell_weight [N,N]           [B,N] per-node               only on PRESENT (i,j) pairs, i≠j
       │                             │                             │
  loss_struct [B]              loss_attr [B]                loss_order [B]
       └──────────────┬──────────────┘
              loss = loss_struct + λ_attr · loss_attr   [B]   (λ_attr = 0.5 default)
                      (loss_order kept separate, not folded in — see §10.5)
```
- **Count decoder is `Linear + bmm`, not `nn.Bilinear` over a flattened batch**: an earlier version's `nn.Bilinear(H,H,1)` on `[B*N*N,H]` rows allocated a `[B*N*N,H,H]` gradient intermediate that overflowed a 3 GB GPU (`B*W=640, N=35` ⇒ ≈3.2 GB) — `Linear+bmm` computes the same bilinear form without ever materialising anything larger than `[B,N,N]`.
- **Loss target is MSE against a weighted (log1p count), not BCE against a binary edge** — a presence/absence decoder cannot see "the same edge, count went up".
- **`cell_weight` down-weights the ~1,220/1,225 (service,service) pairs that never carry a real async edge anywhere in the dataset** (`unseen_edge_weight`, default 0.05, from `meta.pkl`'s `async_edge_mask`): averaging uniformly over all N×N cells let one rare cell's misprediction dominate a trace's score, which is what made some normal traces score as *more* anomalous than every real anomaly before this was added (diagnosed 2026-09-23).
- **Order decoder is a separate bilinear head (`W_order ≠ W_edge`) with its own BCE**, averaged only over pairs of *present* services (excluding the diagonal) — kept as its own scalar (`loss_order`), not summed into `loss_struct`, so it can be standardised on its own scale downstream (§10.5).

### 10.4 Async branch's own gate — `async_trace_gate`

Mirrors the sync branch's `trace_gate` (§3.6) exactly, but reads **only** async-branch inputs (`_async_trace_quality_feats`: mean out/in-count, coverage, mean lag, adjacency density, count variance — never `trace_node_features`/`trace_adj`), so the two gates cannot cross-talk: `Linear(6→16) → ReLU → Linear(16→1) → sigmoid`, bias init `0.0` (not `−2.0` like the sync gate — see §10.6 for why).

### 10.5 Embedding into the Fused decoder — same `delta_head`, not a second one

> An earlier version gave the async branch its **own** `delta_head_async` reading the same `fused_out`/`log_out`/`kpi_out` that the sync `delta_head` also writes to — two deltas editing one shared output caused F01's F1 to collapse (0.62 → 0.22 in an early full run) purely from that interference, unrelated to whether the async signal itself was useful. Fixed by **widening the existing `delta_head`** instead of adding a parallel one.

```
  cached_ZV_async [B,W,H] = masked mean, over PRESENT services (diagonal of async_temporal_order_adj,
                             falling back to "has any async message" if no order relation), of
                             MultiEncoder.async_trace_encoder's output — a plain mean over all N
                             services would dilute the few active ones' signal.

  delta_head input:  open_async_trace=False → cat([fm, ZV])          [B,W,3H]   (unchanged from §3.6)
                      open_async_trace=True  → cat([fm, ZV, ZV_async]) [B,W,4H]  (+H·(3H+2H) ≈ 2,048 params)
                      (ZV_async is zeros, not omitted, when a batch/step has no async messages at
                       all — keeps the tensor shape fixed and the delta exactly 0 there)

  fused_out = fuse_decoder(fm) + gate_g · delta_head(cat([fm, ZV, ZV_async]))
              (gate_g is the SYNC gate, §3.6 — one shared gate for the whole delta,
               same as before this change; the async branch's OWN gate, §10.4, only
               weights its autoencoder loss below, not this decoder path)
```
At initialisation `delta_head`'s last layer is zero-init (unchanged), so `fused_out` starts identical whether or not the async branch is present; verified as a no-op at init and confirmed that gradient reaches `async_trace_encoder` through this path.

### 10.6 Score assembly — two rules

**Raw fusion loss** (`--score_rule raw_sum`, the default for every other dataset): the async branch adds one more gated term, following the exact pattern of the sync branch's CHANGE 8:
```
  trace_dis_async_all = trace_dis_async_count + trace_dis_async_order      # one branch, one term for raw-sum purposes
  fusion_loss += gate_g_async · trace_dis_async_all · expand_anomaly_gap(...) + gate_lambda · gate_g_async
  loss        += (gate_g_async · trace_dis_async_all).mean()   + gate_lambda · gate_g_async.mean()
```

**`--score_rule norm_sum`** (used for DeepTraLog — see the experiment-results doc): rather than one raw sum where the noisiest term dominates the ranking, each term is standardised against **val-normal-only statistics** before adding:
```
  S = Σ_k z_k,   z_k = max(0, (T_k − median_k) / (p95_k − median_k))
  k ∈ {log_kpi_loss, trace_dis, trace_dis_async_count, trace_dis_async_order, trace_err}   (whichever are present)
  threshold = val p95 of S
```
- `trace_dis_async_count` (the count+attribute term) is **context-normalised**: only the message-bearing subset of val traces (≈8.7% on DeepTraLog) gets a nonzero `z`; those are standardised against each other with a robust scale, not against the whole (mostly message-free) val set — otherwise that small subset's own outliers would set a much higher threshold than the majority of anomalies (which have no async messages at all) can clear.
- `trace_err` = `log1p(Σ_service call_count·error_rate)`, i.e. log1p of the trace's total error-span evidence — added for the DeepTraLog F13 `trips/left` sub-case, which fails fast with an error span but exchanges no async messages, so neither `trace_dis_async_count` nor `trace_dis_async_order` sees it. Rare-event term: val normals almost never have an error span, so `p95 − median` is floored at a fixed scale (`BaseModel.ERR_Z_ONE = 4.0`, i.e. one error span in an otherwise clean trace scores `z=4`) rather than left at ≈0.
- This is computed in `BaseModel._score_components`/`evaluate_norm_sum` (`codes/models/basev3.py`), reusing the fused-model's per-window component outputs (`log_kpi_loss`, `trace_dis`, `trace_dis_async_count`, `trace_dis_async_order` from the `forward()` dict) — no separate model pass.

### 10.7 Guarantees checked

- `open_async_trace=False`: no async tensors are built, `delta_head` keeps its original 3H input, `fusion_loss`/`loss` are exactly the pre-CHANGE-9 expressions — verified byte-identical scoring on a regression run (sync-only checkpoint re-scored with the cleaned-up code).
- `open_async_trace=True` at initialisation: `delta_head`'s zero-init last layer makes the async contribution to `fused_out` exactly 0 regardless of `ZV_async`; a synthetic zero/shuffle probe on a trained checkpoint confirmed `log_dis` barely moves (0.745→0.747/0.743), ruling out an "explain-away" mechanism.
- Gradient reaches `async_trace_encoder` through the shared `delta_head` path (checked directly on a small batch).

### 10.8 CLI flags

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--open_async_trace` | `False` | Build the async branch (encoder, autoencoder, gate, delta input) |
| `--async_order` | `None` (auto-read from `meta["async_order"]`) | Also build/use the order-relation head; `ValueError` if `True` but the data has no `async_temporal_order_adj` |
| `--async_c` | from `meta["async_c"]` | Number of async node feature columns (3 for DeepTraLog) |
| `--async_gate_init_bias` | `0.0` | Init bias of `async_trace_gate` (sync's `trace_gate` uses `−2.0`; the async gate starts more open since its own autoencoder loss, not the shared decoder, is what it weights) |
| `--score_rule` | `raw_sum` | `norm_sum` for the val-normalised sum (§10.6); `raw_sum` elsewhere is unaffected by any of this section |
| `--dump_components` | off | Dump per-window `_score_components` to an `.npz` instead of running full evaluation — used to re-score a saved checkpoint with a new scoring rule without retraining |

### 10.9 Diagnostics: does this dataset actually support the async branch?

`--open_async_trace True` on a dataset whose pkl has no `async_trace_node_features`/`async_msg_count_adj` used to silently build nothing (§10.1's `has_async_input` check just stays `False` every batch) with no line anywhere saying so. `common/data_loads.py`'s `Process.__init__` now logs this explicitly, right after "Data loaded done!":
- `--open_async_trace True` + the dataset has the keys → `INFO  Async trace branch: ACTIVE -- ... (order relation present/ABSENT).`
- `--open_async_trace True` + the dataset does **not** have the keys → `WARNING  Async trace branch: --open_async_trace True but this dataset has NO async_trace_node_features/async_msg_count_adj -- the branch will NOT be built and this run is equivalent to --open_async_trace False. ...`
- `--open_async_trace False` but the dataset **does** have the keys → an `INFO` note that the branch is available but unused this run.
- `--open_async_trace False` and the dataset has no async keys → silent (the common case for SN/RE2/RE3).
