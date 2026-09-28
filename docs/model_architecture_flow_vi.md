# HADES + Trace — Luồng mô hình (Model Architecture Flow)

> Tài liệu này mô tả kiến trúc và luồng dữ liệu của mô hình HADES được mở rộng với nhánh Trace (GAT Structure Autoencoder).
> **Phiên bản**: `fuse_v3.py` với 7 thay đổi kiến trúc (dựa trên TraceDAE).

---

## 1. Tổng quan kiến trúc dự án

```
UAM-AD/
├── codes/
│   ├── run.py                          ← Điểm vào chính
│   ├── run_sequential.py               ← Chạy tuần tự (tránh CUDA OOM)
│   ├── common/
│   │   ├── data_loads.py               ← Load & window data → DataLoader
│   │   ├── semantics.py                ← Trích xuất log features (Word2Vec/template)
│   │   ├── utils.py                    ← Tiện ích chung (seed, dump results...)
│   │   ├── preprocess_XX.py            ← Build pkl từ raw dataset XX
|   |   └── eval_per_scenario_XX.py     ← Eval cho dataset XX có nhiều fault type/scenario
│   └── models/
│       ├── basev3.py                   ← Vòng lặp train/eval (BaseModel)
│       ├── fuse_v3.py                  ← Model đa phương thức (log+metric+trace)
│       ├── log_model_v3.py             ← Log encoder (Transformer)
│       ├── kpi_model_v3.py             ← Metric encoder (Transformer)
│       ├── trace_model_v3.py           ← Trace encoder sync (GAT) + TraceModel
│       ├── async_trace_model_v3.py     ← Nhánh trace async (CHANGE 9, mục 10) — cộng thêm, không đụng nhánh sync
│       └── utils.py                    ← Các module dùng chung (Attention, ...)
└── data/
    └── XX/
        ├── train.pkl / unlabel.pkl / test.pkl
        └── meta.pkl
```

---

## 2. Luồng dữ liệu đầu vào

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INPUT  [B, W, *]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  log_features        [B, W, log_c]          ← log template features
  kpi_features        [B, W, kpi_c]          ← KPI metrics
  trace_node_features [B, W, N, trace_c]     ← service node features (trace_c=6)
  trace_adj           [B, W, N, N]           ← service call graph (STG)
  unmatched_kpi       [B, W, kpi_c]          ← shuffled KPI từ windows khác

  B = batch size | W = window size (tùy dataset) | N = num_services (tùy dataset)
  H = hidden_size (32) | trace_c = 6

  Node feature layout (trace_c=6):
    col 0: call_count   — số spans, normalized
    col 1: avg_dur_ms   — mean duration, normalized
    col 2: max_dur_ms   — max duration, normalized
    col 3: error_rate   — fraction lỗi ∈ [0,1]
    col 4: root_rate    — fraction root spans ∈ [0,1]
    col 5: latency_dev  — z-score(avg_dur vs pre-fault baseline)
```

---

## 3. Generator — `MultiModel.forward()`

### 3.1 MultiEncoder — [CHANGE 1] Trace tách khỏi Self-Attention

> **Thay đổi**:
> Giờ Self-Attention **chỉ** áp dụng trên log+KPI. Trace encoder chạy **riêng** → `ZV`.
> (structural info nên guide **decoder**, không phải encoder attention).

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
  │                   mean_pool over N → ZV [B, W, H]     (riêng biệt)   │
  └────────────────────────────────────────────────────────────────────────┘

  Returns: (fused_kpi, fused_log, fused_modal [B,W,2H], ZV [B,W,H])
```

> **Không có trace** (`open_trace=False`): ZV = None, fused_modal vẫn là `[B,W,2H]`.

---

### 3.2 Reconstruction — [CHANGE 2] ZV inject vào Decoder

> **Thay đổi**: Decoder nhận `cat([fused_modal, ZV])` thay vì chỉ `fused_modal`.
> Lý do: Terinspirasi từ TraceDAE (Eq.10) — structural embedding ZV nên guide **reconstruction**,
> không phải fusion attention. Giống `X_hat = ZV · ZA^T` trong TraceDAE.

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

> **Không có trace** (ZV=None): fuse_decoder = Linear(2H → kpi_c+log_c) — decoder input = 2H.

---

### 3.3 Trace Autoencoder — [CHANGE 3] adj_hat + [CHANGE 6] Attribute Reconstruction Loss

> **CHANGE 3**: `adj_hat` (adjacency matrix tái tạo) được return ra ngoài `MultiModel`
> để Discriminator có thể dùng làm "fake trace adjacency".
>
> **CHANGE 6**: TraceModel thêm **attribute decoder** và hai reconstruction loss bổ sung:
> - `loss_latency`: MSE trên `latency_dev` (col 5) — phát hiện service chậm hơn baseline
> - `loss_error`: BCE trên `error_rate` (col 3) — phát hiện service tăng lỗi
>
> Mục đích: buộc node embedding ZV encode cả thông tin topology lẫn attribute anomaly,
> làm `trace_dis` discriminative hơn cho delay/loss/mem/socket fault.

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
              (λ_lat = λ_err = 0.5 mặc định)

  adj_hat_4d   = A_hat.reshape(B, W, N, N)          ← CHANGE 3: trả ra MultiModel output
  feats_hat_4d = feats_hat[:,:,[3,5]].reshape(B, W, N, 2)  ← CHANGE 7: trả ra cho attribute discriminator
```

---

### 3.4 Fusion Loss — [CHANGE 4] Variance-based Alpha (thay thế Learnable trace_alpha)

> **Thay đổi**: `trace_alpha = nn.Parameter(-2.2)` (learnable theo thiết kế TraceDAE) → Learnable alpha bị thay bằng **variance-based alpha** — không có parameter.
>
> **Vấn đề với learnable alpha**: Gradient đẩy `trace_alpha` converge để cân bằng *reconstruction error magnitude*
> trên normal data — không phản ánh discriminativeness thực sự. Trên dataset trace yếu (e.g. RE3-OB),
> `trace_dis` gần constant → alpha tăng lên để bù → noise injection.
>
> **Giải pháp variance-based**: α phản ánh trực tiếp discriminativeness của trace signal so với log+KPI.
> Khi trace noise → var(trace_dis) thấp → α → 0 tự động.

```
  log_d   = log_dis  × expand_anomaly_gap(log_dis)
  kpi_d   = kpi_dis  × expand_anomaly_gap(kpi_dis)
  trace_d = trace_dis × expand_anomaly_gap(trace_dis)

  log_kpi_loss = log_d + kpi_d + narrow_modal_gap(|log_d − kpi_d|)

  var_lk    = var(log_kpi_loss.detach())      ← không có gradient
  var_trace = var(trace_dis.detach())
  α = var_trace / (var_lk + var_trace + ε)    ← ∈ [0, 1], không học được

  fusion_loss = (1 − α) × log_kpi_loss + α × trace_d    [B, W]
                └─────────────────────────────────────────────────────┘
                         Anomaly Score (dùng để đánh giá)
```

---

### 3.5 Contrastive Loss (Unmatched pairs)

```
  contrastive = max(0,  L1(kpi_features, kpi_out)
                      + unmatch_k
                      − L1(unmatched_kpi, kpi_out_unmatched))
```

---

### 3.6 Residual-Gated Trace Fusion — [CHANGE 8] (luôn bật khi `open_trace=True`)

> **Vấn đề**: Decoder cũ luôn dùng `cat([fused_modal, ZV])` → khi trace không có thông tin
> (vd RE3-OB code-defect), ZV nhiễu rò vào `kpi_out` / `log_out`, kéo F1 *xuống dưới baseline*.
> Variance-based α ở §3.4 chỉ re-weight **loss**, không chữa được output của decoder.
>
> **Giải pháp**: dạng residual với gate per-sample có thể đóng hoàn toàn đóng góp của trace.
>
> - `base_decoder`: `Linear(2H → H) → ReLU → Linear(H → kpi_c+log_c)` — đường baseline chỉ log+KPI
> - `delta_head`:   `Linear(3H → 2H) → ReLU → Linear(2H → kpi_c+log_c)` — **zero-init lớp cuối**
> - `trace_gate`:   `Linear(6 → 16) → ReLU → Linear(16 → 1) → sigmoid` — bias init `−2.0` → g₀ ≈ 0.12
> - **Trace-quality features** (per B,W): mean call count, coverage (span khác 0), mean error_rate,
>   mean |latency_dev|, adjacency density, call-count variance. Log1p-normalize.

```
  Trace-quality features [B, W, 6]
         │
   trace_gate (MLP, bias=-2.0) ──► g ∈ (0, 1)   [B, W, 1]
                                          │
  fused_modal [B,W,2H] ──► base_decoder ──► y_base [B,W,kpi_c+log_c]
                                          │
  cat([fm, ZV]) [B,W,3H] ─► delta_head ──► Δ [B,W,kpi_c+log_c]   (zero-init → Δ≈0 lúc đầu)
                                          │
                            fused_out = y_base + g · Δ    ← g=0 ⇒ đúng bằng baseline

  fusion_loss = log_kpi_loss + g · trace_d + gate_lambda · g.mean()
                                            └─── L1 reg giữ gate đóng mặc định
```

> **Bảo đảm**: khởi điểm `Δ ≈ 0` và `g ≈ 0.12`; gradient chỉ mở gate nếu
> `Δ` giảm loss log+KPI hơn `gate_lambda` (mặc định 0.01).
> Trên dataset mà trace nhiễu, gate đóng và mô hình *đúng bằng* baseline.
>
> **CLI**: `--gate_lambda 0.01` (tự động áp dụng khi `--open_trace True`).
>
> **Kiểm chứng**: Trên RE3-OB (5 fault type code-defect), residual-gated recover baseline F1
> trong phạm vi 0.015 trên mọi fault type.
> Xem `experiment_results_re3_ob_trace_vs_baseline_vi.md`.

---

## 4. Discriminator — `MultiDiscriminator.get_loss()`

### 4.1 Structural Trace Discrimination — [CHANGE 5] (không thay đổi)

> **Thay đổi**: FAKE pass dùng `adj_hat` (adjacency tái tạo từ Structure AE) →
> `trace_re_fake ≠ trace_re` → loss có ý nghĩa.
> Tương tự như log_out/kpi_out là "fake" cho log/KPI, `adj_hat` là "fake" cho trace structure.
>
> **Lưu ý về `trace_nodes`**: `trace_nodes` giống nhau trong cả REAL và FAKE → không đóng góp
> discriminative signal. Nó chỉ đóng vai trò là input node feature cho GAT attention trong `encoder_low`.
> Toàn bộ signal structural discrimination đến từ sự khác biệt giữa `adj` và `adj_hat`.

```
  MultiEncoder_low (lightweight, 1-layer GAT):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  REAL:  encoder_low(log_x,   kpi_x,   trace_nodes, trace_adj)        │
  │         → log_re, kpi_re, trace_re              [B, W, H]            │
  │                                                                       │
  │  FAKE:  encoder_low(log_out, kpi_out, trace_nodes, adj_hat)  ← CHANGE│
  │         → log_re_fake, kpi_re_fake, trace_re_fake                    │
  │         (adj_hat từ Structure AE → trace_re_fake ≠ trace_re  ✅)     │
  └──────────────────────────────────────────────────────────────────────┘

  Discriminator loss (cập nhật Discriminator weights):
  disc_loss = CE(pred_kpi,       real=1) + CE(pred_kpi_fake,   fake=0)
            + CE(pred_log,       real=1) + CE(pred_log_fake,   fake=0)
            + CE(pred_trace,     real=1) + CE(pred_trace_fake, fake=0)

  Deceive loss (cập nhật Generator weights):
  deceive_loss = MSE(kpi_re, kpi_re_fake)
               + MSE(log_re, log_re_fake)
               + MSE(trace_re, trace_re_fake)
```

---

### 4.2 Attribute Trace Discrimination — [CHANGE 7] (mới, hoàn toàn độc lập)

> **Động lực**: Structural head (CHANGE 5) chỉ discriminate qua topology (adj vs adj_hat).
> Node attributes giống nhau trong cả REAL và FAKE → attributes không đóng góp gì cho discrimination.
> Thêm một attribute head riêng biệt so sánh ground-truth vs reconstructed node attributes
> buộc Generator phải sinh ra `error_rate` và `latency_dev` thực tế hơn.
>
> **Tại sao chỉ dùng col 3 & col 5?**
> - `error_rate` (col 3) và `latency_dev` (col 5) có supervision rõ ràng trong CHANGE 6
>   → chất lượng của `feats_hat[:,:,[3,5]]` đáng tin cậy.
> - 4 col còn lại (call_count, avg_dur_ms, max_dur_ms, root_rate) không có explicit loss
>   → chất lượng reconstruction kém → dùng chúng sẽ inject noise.
>
> **Code structural cũ không cần sửa** — attribute head là hoàn toàn additive.

```
  Attribute head (độc lập với structural, không qua encoder_low):

  REAL:  trace_nodes[:, :, [3, 5]]            [B, W, N, 2]
         → mean over N → attr_real            [B*W, 2]

  FAKE:  feats_hat[:, :, [3, 5]]              [B, W, N, 2]   ← từ TraceModel (CHANGE 7)
         → mean over N → attr_fake            [B*W, 2]

  attr_classifier: Linear(2, H) → ReLU → Linear(H, 1) → sigmoid
                                                          [B*W, 1]

  attr_disc_loss = BCE(attr_classifier(attr_real), real=1)
                 + BCE(attr_classifier(attr_fake), fake=0)

  attr_deceive_loss = MSE(attr_classifier(attr_real),
                          attr_classifier(attr_fake))

  Tổng hợp loss (cộng thêm vào):
  disc_loss    += attr_disc_loss
  deceive_loss += attr_deceive_loss
```

---

## 5. Training Loop (mỗi epoch)

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
        → point_adjustment(pred, gt)  ← nếu phát hiện bất kỳ điểm nào
                                         trong segment → cả segment = detected
        → F1 / Recall / Precision
```

---

## 6. Routing theo `data_type`

```
  data_type = "fuse"  →  MultiModel   (log + KPI + trace nếu open_trace=True)
  data_type = "log"   →  LogModel     (log only)
  data_type = "kpi"   →  KpiModel     (KPI only, bỏ qua open_trace)
```

> ⚠️ Chỉ `data_type=fuse` mới kích hoạt nhánh trace.

---

## 7. So sánh kiến trúc: Trước vs Sau (fuse_v3.py)

| #   | Thành phần                   | Trước (cũ)                                         | Sau (mới)                                                                             |
|:---:|:-----------------------------|:---------------------------------------------------|:--------------------------------------------------------------------------------------|
|  1  | **Self-Attention**           | cat([log‖kpi]) → 2H                               | cat([log‖kpi]) → 2H, trace **tách riêng**                                             |
|  2  | **Decoder input**            | fused_modal [B,W,2H hoặc 3H]                       | cat([fused_modal, ZV]) → [B,W,3H]                                                     |
|  3  | **adj_hat**                  | không dùng                                         | return ra ngoài MultiModel để Discriminator dùng                                      |
|  4  | **trace_alpha**              | `nn.Parameter(−2.2)`, learned ≈ 0.10              | **variance-based**: `α = var(trace) / (var(lk) + var(trace) + ε)`                    |
|  5  | **Discriminator FAKE**       | dùng `trace_adj` (thật) → contradiction            | dùng `adj_hat` (tái tạo) → valid loss                                                 |
|  6  | **trace_dis**                | BCE(A_hat, adj) — chỉ structural loss              | + λ_lat×MSE(latency_dev) + λ_err×BCE(error_rate)                                     |
|  7  | **Attribute Discriminator**  | không có — node attributes không được discriminate | head riêng: REAL=trace_nodes[:,:,[3,5]], FAKE=feats_hat[:,:,[3,5]] → Linear(2,H)→ReLU→Linear(H,1) |
|  8  | **Decoder fusion mode**      | luôn `cat([fm, ZV])` → noise rò vào kpi_out / log_out khi trace không informative | **Residual-gated**: `y_base(fm) + g · delta_head(cat[fm,ZV])`, `g∈[0,1]` per-sample từ 6 trace-quality features, `delta_head` zero-init, L1 reg trên g (`gate_lambda`) |
|  9  | **Nhánh trace async**        | không có                                           | **Nhánh cộng thêm, độc lập** (mục 10): encoder/decoder/gate riêng cho quan hệ số lượng message + thứ tự thời gian giữa service; embedding của nó đi vào CÙNG `delta_head` với embedding sync (nới thêm một khối H); tắt mặc định (`open_async_trace=False` ⇒ giống hệt từng bit hành vi ở dòng 1-8) |

---

## 8. Sơ đồ tổng thể End-to-End (sau khi cải tiến)

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
                        └──────────── (g→0 khi trace không informative → y ≡ baseline) ┘
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

  Attribute head (CHANGE 7 — mới, hoàn toàn độc lập):
    REAL: trace_nodes[:,:,[3,5]].mean(N) → attr_real  [B*W, 2]
    FAKE: feats_hat[:,:,[3,5]].mean(N)  → attr_fake  [B*W, 2]
    attr_classifier: Linear(2,H) → ReLU → Linear(H,1)
    disc_loss    += BCE(attr_classifier(attr_real), 1) + BCE(attr_classifier(attr_fake), 0)
    deceive_loss += MSE(attr_classifier(attr_real), attr_classifier(attr_fake))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT: F1 / Recall / Precision (với point-adjustment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 9. Khả năng đóng góp của Trace theo loại Fault

> **Khi nào trace giúp ích?** Chỉ khi fault thay đổi **hành vi observable ở tầng service-to-service**
> — tức là topology call graph, latency giữa các service, hoặc error_rate qua mạng.
> Khi fault **nội bộ trong service** và không ảnh hưởng đến các pattern này, trace không discriminative
> và CHANGE 8 (hard gate) tự động vô hiệu hoá nó hoàn toàn.

| Loại fault                                        | Trace có ích? | Lý do |
| **Network** (delay, packet loss, bandwidth limit) | ✅ Rõ ràng    | Latency spike trực tiếp + error_rate thay đổi giữa các service |
| **Resource** (CPU, memory, disk overload)         | ✅ Vừa phải   | Overload → service chậm → latency_dev tăng |
| **Service crash / OOM**                           | ✅ Rõ ràng    | Error rate spike, call count giảm |
| **Code Logic / Defect**                           | ❌ Không      | Bug chạy nội bộ — call graph và latency không đổi |
| **Configuration error**                           | ❌ Không      | Sai config nhưng service vẫn respond bình thường |
| **Database internal** (slow query, deadlock)      | ⚠️ Một phần   | Chỉ thấy nếu DB trong service mesh; latency service gọi DB có thể tăng |
| **Business logic** (tính sai, output sai)         | ❌ Không      | Output sai nhưng performance characteristics không đổi |
| **Security** (auth bypass, injection)             | ❌ Không      | Call pattern và latency không thay đổi |

---

## 10. Nhánh Trace Async — [CHANGE 9] (cộng thêm, `open_async_trace=True`)

> Kiểm chứng trên DeepTraLog (TrainTicket) — xem [`experiment_results_deeptralog_trace_both_sync_async_vs_only_sync_vi.md`](experiment_results_deeptralog_trace_both_sync_async_vs_only_sync_vi.md) cho kết quả đầy đủ, [`preprocess_deeptralog_vi.md`](preprocess_deeptralog_vi.md) cho pipeline dataset/nhãn.

### 10.0 Vì sao cần nhánh thứ hai, không mở rộng nhánh sync

Nhánh trace sync (mục 3.3) chỉ theo dõi một thứ: **tập cạnh gọi service** vô hướng, không trọng số (`trace_adj`, có/không). Đo trên dấu vân cấu trúc của DeepTraLog (mục 4 của docs preprocess), ba F-case async (F01, F02, F13) giữ tập cạnh này **không đổi 100%** — lỗi thay vào đó đổi **số lượng message** giữa 2 service (F02, +21%), **thứ tự thời gian tương đối** giữa các service được gọi (F01/F13, 100% trace có quan hệ thứ tự chưa từng thấy), hoặc làm request **lỗi rồi dừng sớm với span lỗi** (một phần F13). Không thứ nào trong 3 đại lượng này có biểu diễn trong node feature hay adjacency của nhánh sync, và decoder có/không đối xứng thì về cấu trúc không thể thấy "vẫn cạnh đó nhưng số lượng tăng" hay "vẫn 2 service đó nhưng giờ A xảy ra sau B thay vì trước". Vì vậy, thay vì nạp thêm cột/decoder mới vào `trace_model_v3.py` (dùng nguyên xi cho mọi dataset khác), nhánh async là một **mô hình thứ hai, độc lập** (`async_trace_model_v3.py`) dùng lại đúng khối `TraceEncoder`/`GATLayer`, cộng thêm song song — tắt theo mặc định, và khi tắt thì tensor và số tham số của nhánh sync giống hệt từng bit trước khi có thay đổi này (đã kiểm: 2.598 / 48.003 / 129 tham số nhánh sync, không đổi cả khi `open_async_trace` là True hay False).

### 10.1 Đầu vào (mỗi trace, node = service, cùng `N` với nhánh sync)

```
  async_trace_node_features [B,W,N,3]   col 0: log1p(số cạnh async đi ra, service này là bên gửi)
                                          col 1: log1p(số cạnh async đi vào, service này là bên nhận)
                                          col 2: log1p(độ trễ trung bình consumer, giây), 0 nếu col1==0

  async_msg_count_adj  [B,W,N,N]   có hướng, có trọng số: log1p(số message async i→j) — KHÔNG đối xứng hoá
                                 (ai gửi cho ai chính là điều lỗi số lượng message cần)

  async_temporal_order_adj  [B,W,N,N]   có hướng, nhị phân: [i,j]=1 nếu mọi span của service i kết thúc
                                 trước khi span đầu tiên của service j bắt đầu (bản mức-service của
                                 cạnh "Sequence" trong TEG của DeepTraLog); đường chéo [i,i]=1 đánh dấu
                                 service i CÓ MẶT trong trace (đây là cách biết "service có mặt" mà
                                 không cần khoá mask riêng)
```
`async_msg_count_adj`/`async_temporal_order_adj` được tính trong `_build_edges`/`_order_adj` của `preprocess_deeptralog.py`; dataset không có 2 khoá này thì đơn giản không có nhánh async (`open_async_trace` cần cả 2 khoá; `async_order` cần thêm `async_temporal_order_adj`, thiếu thì `async_encoder_inputs` báo `ValueError` chứ không lặng lẽ suy giảm).

### 10.2 Encoder — `async_encoder_inputs` (hàm dùng chung, `async_trace_model_v3.py`)

```
  x [B,N,3] ──┐
              ├─ use_order? ──► cat([x, one_hot(service có mặt)]) [B,N,3+N] ──► TraceEncoder ──► z [B,N,H]
  order_adj ──┘                  mask = (adj>0) | (present_i AND present_j)   (2-layer GAT, cùng
                                  ── chính quan hệ THỨ TỰ không nằm trong        class với nhánh sync)
                                     mask: nó là mục tiêu cần dự đoán, không
                                     phải thứ attention được phép chép lại
```
Hai bản khởi tạo độc lập của encoder này, trọng số riêng (giống hệt cách nhánh sync có một `TraceEncoder` trong `AsyncTraceModel` cho autoencoder và một trong `MultiEncoder` cho embedding vào Fused decoder):
- `AsyncTraceModel.encoder` — nuôi autoencoder async (mục 10.3).
- `MultiEncoder.async_trace_encoder` — nuôi embedding đi vào `delta_head` chung (mục 10.4).

Không có danh tính node, hầu hết đặc trưng node async gần như 0 (service không có hoạt động message thì cả 3 cột ≈0), nên các service không trao đổi message sẽ giống hệt nhau và không đầu nào học được *cặp nào* là lạ; one-hot danh tính khắc phục điều này. Mask có chủ đích không chứa chính quan hệ thứ tự — đưa nó vào mask khiến các thí nghiệm đầu tiên attention chép lại quan hệ qua việc chia sẻ cạnh thay vì dự đoán, làm ẩn mất các quan hệ thật sự chưa từng thấy (đã chẩn đoán và sửa trong lúc phát triển, xem docstring của `async_encoder_inputs`).

### 10.3 Autoencoder Async — `AsyncTraceModel` (`async_trace_model_v3.py`)

```
  z [B,N,H] (từ TraceEncoder trên)
       │
       ├─────────────────────────────┬─────────────────────────────┐
  Decoder cấu trúc (đếm)       Decoder thuộc tính           Decoder thứ tự (nếu async_order)
  adj_hat = softplus(          feats_hat = Linear(z)        logits = (z·W_order)·zᵀ
    (z·W_edge)·zᵀ )              [B,N,3]                     (có hướng)
  MSE(adj_hat, async_adj)      MSE(feats_hat, x)            BCE(logits, order_adj)
  × cell_weight [N,N]           [B,N] mỗi node               chỉ trên cặp (i,j) CÓ MẶT, i≠j
       │                             │                             │
  loss_struct [B]              loss_attr [B]                loss_order [B]
       └──────────────┬──────────────┘
              loss = loss_struct + λ_attr · loss_attr   [B]   (λ_attr = 0,5 mặc định)
                      (loss_order giữ riêng, không gộp vào — xem mục 10.5)
```
- **Decoder đếm dùng `Linear + bmm`, không dùng `nn.Bilinear` trên batch đã làm phẳng**: bản đầu dùng `nn.Bilinear(H,H,1)` trên các hàng `[B*N*N,H]` tạo tensor gradient trung gian `[B*N*N,H,H]` tràn RAM của GPU 3 GB (`B*W=640, N=35` ⇒ ≈3,2 GB) — `Linear+bmm` tính đúng dạng bilinear đó mà không bao giờ tạo tensor lớn hơn `[B,N,N]`.
- **Mục tiêu loss là MSE so với giá trị có trọng số (log1p số lượng), không phải BCE so với cạnh nhị phân** — decoder có/không không thể thấy "vẫn cạnh đó, số lượng tăng lên".
- **`cell_weight` giảm trọng số cho ~1.220/1.225 cặp (service,service) không bao giờ có cạnh async thật trong toàn dataset** (`unseen_edge_weight`, mặc định 0,05, từ `async_edge_mask` của `meta.pkl`): lấy trung bình đều trên mọi ô N×N khiến một ô hiếm dự đoán sai lấn át điểm số của cả trace — đây chính là lý do một số trace bình thường có điểm cao hơn mọi trace bất thường thật trước khi thêm cơ chế này (chẩn đoán ngày 23/9).
- **Decoder thứ tự là một đầu bilinear riêng (`W_order ≠ W_edge`) với BCE riêng**, chỉ trung bình trên các cặp service *có mặt* (bỏ đường chéo) — giữ là số vô hướng riêng (`loss_order`), không cộng vào `loss_struct`, để có thể chuẩn hoá theo thang riêng ở bước sau (mục 10.5).

### 10.4 Gate riêng của nhánh async — `async_trace_gate`

Giống hệt `trace_gate` của nhánh sync (mục 3.6), nhưng chỉ đọc đầu vào của nhánh async (`_async_trace_quality_feats`: trung bình số lượng ra/vào, độ phủ, độ trễ trung bình, mật độ adjacency, phương sai số lượng — không bao giờ đọc `trace_node_features`/`trace_adj`), nên hai gate không thể "nói chuyện" với nhau: `Linear(6→16) → ReLU → Linear(16→1) → sigmoid`, bias khởi tạo `0,0` (không phải `−2,0` như gate sync — xem mục 10.6 để biết lý do).

### 10.5 Embedding vào Fused decoder — cùng `delta_head`, không phải cái thứ hai

> Bản đầu cho nhánh async một `delta_head_async` RIÊNG đọc cùng `fused_out`/`log_out`/`kpi_out` mà `delta_head` sync cũng viết vào — hai delta cùng sửa một đầu ra chung làm F1 của F01 sụp (0,62 → 0,22 trong một lần chạy đầy đủ ban đầu) chỉ vì sự can thiệp này, không liên quan gì đến việc tín hiệu async có ích hay không. Sửa bằng cách **nới rộng `delta_head` hiện có** thay vì thêm cái song song.

```
  cached_ZV_async [B,W,H] = trung bình có mask, trên các service CÓ MẶT (đường chéo của
                             async_temporal_order_adj, lùi về "có bất kỳ message async" nếu không có quan hệ
                             thứ tự), của đầu ra MultiEncoder.async_trace_encoder — trung bình đều
                             trên cả N service sẽ làm loãng tín hiệu của vài service đang hoạt động.

  đầu vào delta_head:  open_async_trace=False → cat([fm, ZV])            [B,W,3H]   (không đổi so với mục 3.6)
                       open_async_trace=True  → cat([fm, ZV, ZV_async])  [B,W,4H]   (+H·(3H+2H) ≈ 2.048 tham số)
                       (ZV_async là số 0, không bị bỏ, khi một batch/step không có message async
                        nào — giữ shape tensor cố định và delta đúng bằng 0 ở đó)

  fused_out = fuse_decoder(fm) + gate_g · delta_head(cat([fm, ZV, ZV_async]))
              (gate_g là gate SYNC, mục 3.6 — một gate chung cho cả delta,
               giống trước khi có thay đổi này; gate RIÊNG của nhánh async, mục 10.4,
               chỉ cân trọng số loss autoencoder của nó ở dưới, không phải đường decoder này)
```
Lúc khởi tạo, lớp cuối của `delta_head` khởi tạo bằng 0 (không đổi), nên `fused_out` bắt đầu giống hệt nhau bất kể có nhánh async hay không; đã kiểm là no-op đúng lúc khởi tạo và xác nhận gradient chảy tới `async_trace_encoder` qua đường này.

### 10.6 Cách gộp điểm — hai luật

**Tổng thô fusion loss** (`--score_rule raw_sum`, mặc định cho mọi dataset khác): nhánh async cộng thêm một số hạng có gate, theo đúng khuôn CHANGE 8 của nhánh sync:
```
  trace_dis_async_all = trace_dis_async_count + trace_dis_async_order   # một nhánh, một số hạng cho mục đích raw-sum
  fusion_loss += gate_g_async · trace_dis_async_all · expand_anomaly_gap(...) + gate_lambda · gate_g_async
  loss        += (gate_g_async · trace_dis_async_all).mean()  + gate_lambda · gate_g_async.mean()
```

**`--score_rule norm_sum`** (dùng cho DeepTraLog — xem docs kết quả thực nghiệm): thay vì một tổng thô mà số hạng ồn nhất áp đảo thứ hạng, mỗi số hạng được chuẩn hoá bằng **thống kê chỉ từ normal của val** trước khi cộng:
```
  S = Σ_k z_k,   z_k = max(0, (T_k − median_k) / (p95_k − median_k))
  k ∈ {log_kpi_loss, trace_dis, trace_dis_async_count, trace_dis_async_order, trace_err}   (những cái có mặt)
  ngưỡng = p95 val của S
```
- `trace_dis_async_count` (số hạng đếm+thuộc tính) được **chuẩn hoá theo ngữ cảnh**: chỉ tập con trace val có trao đổi message (≈8,7% trên DeepTraLog) nhận `z` khác 0; chúng được chuẩn hoá với nhau bằng thang bền vững, không so với toàn bộ val (đa số không có message) — nếu không, các giá trị ngoại lai của tập con nhỏ đó sẽ đặt ngưỡng cao hơn nhiều so với mức đa số trace lỗi (không có message async nào) có thể vượt qua.
- `trace_err` = `log1p(Σ_service call_count·error_rate)`, tức log1p của tổng bằng chứng span lỗi của trace — thêm cho tiểu ca `trips/left` của F13 (DeepTraLog), lỗi-dừng-sớm với span lỗi nhưng không trao đổi message async, nên cả `trace_dis_async_count` và `trace_dis_async_order` đều không thấy nó. Số hạng hiếm-gặp: trace normal ở val hầu như không bao giờ có span lỗi, nên `p95 − median` được đặt sàn ở thang cố định (`BaseModel.ERR_Z_ONE = 4,0`, tức một span lỗi trong trace vốn sạch cho `z=4`) thay vì để ≈0.
- Tính trong `BaseModel._score_components`/`evaluate_norm_sum` (`codes/models/basev3.py`), dùng lại các đầu ra thành phần theo từng cửa sổ của model fused (`log_kpi_loss`, `trace_dis`, `trace_dis_async_count`, `trace_dis_async_order` từ dict trả về của `forward()`) — không cần một lượt chạy model riêng.

### 10.7 Các bảo đảm đã kiểm

- `open_async_trace=False`: không dựng tensor async nào, `delta_head` giữ đầu vào 3H gốc, `fusion_loss`/`loss` đúng bằng biểu thức trước CHANGE 9 — đã kiểm điểm số giống hệt từng bit trên một lần chấm lại hồi quy (chấm lại checkpoint chỉ-sync bằng code đã dọn).
- `open_async_trace=True` lúc khởi tạo: lớp cuối zero-init của `delta_head` làm phần đóng góp của async vào `fused_out` đúng bằng 0 bất kể `ZV_async`; một phép thử tổng hợp (thay embedding bằng 0/xáo trộn) trên checkpoint đã train xác nhận `log_dis` gần như không đổi (0,745→0,747/0,743), loại bỏ giả thuyết "giải thích-thay-thế" (explain-away).
- Gradient chảy tới `async_trace_encoder` qua đường `delta_head` chung (đã kiểm trực tiếp trên một batch nhỏ).

### 10.8 Cờ CLI

| Cờ | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--open_async_trace` | `False` | Dựng nhánh async (encoder, autoencoder, gate, đầu vào delta) |
| `--async_order` | `None` (tự đọc từ `meta["async_order"]`) | Dựng/dùng thêm đầu quan hệ thứ tự; báo `ValueError` nếu `True` mà dữ liệu không có `async_temporal_order_adj` |
| `--async_c` | từ `meta["async_c"]` | Số cột đặc trưng node async (3 cho DeepTraLog) |
| `--async_gate_init_bias` | `0,0` | Bias khởi tạo của `async_trace_gate` (gate sync `trace_gate` dùng `−2,0`; gate async khởi đầu mở hơn vì nó chỉ cân loss autoencoder riêng của nó, không phải decoder chung) |
| `--score_rule` | `raw_sum` | `norm_sum` cho tổng đã chuẩn hoá theo val (mục 10.6); `raw_sum` ở dataset khác không bị ảnh hưởng bởi mục này |
| `--dump_components` | tắt | Dump các thành phần `_score_components` theo từng cửa sổ ra `.npz` thay vì chạy đánh giá đầy đủ — dùng để chấm lại checkpoint đã lưu bằng luật điểm mới mà không cần train lại |

### 10.9 Chẩn đoán: dataset này có thực sự hỗ trợ nhánh async không?

Trước đây, `--open_async_trace True` trên dataset mà pkl không có `async_trace_node_features`/`async_msg_count_adj` sẽ âm thầm không dựng gì (kiểm tra `has_async_input` ở mục 10.1 luôn ra `False` mọi batch) mà không có dòng log nào báo. `common/data_loads.py`'s `Process.__init__` giờ log rõ điều này, ngay sau "Data loaded done!":
- `--open_async_trace True` + dataset có 2 khoá → `INFO  Async trace branch: ACTIVE -- ... (order relation present/ABSENT).`
- `--open_async_trace True` + dataset **không** có 2 khoá → `WARNING  Async trace branch: --open_async_trace True but this dataset has NO async_trace_node_features/async_msg_count_adj -- the branch will NOT be built and this run is equivalent to --open_async_trace False. ...`
- `--open_async_trace False` nhưng dataset **có** 2 khoá → ghi `INFO` cho biết nhánh có sẵn nhưng không dùng ở lần chạy này.
- `--open_async_trace False` và dataset không có khoá async → im lặng (trường hợp thường gặp với SN/RE2/RE3).
