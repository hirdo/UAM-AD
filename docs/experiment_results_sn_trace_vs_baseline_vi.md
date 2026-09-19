# Kết quả thí nghiệm: SocialNetwork — Trace so với Baseline

## 1. Thiết lập thí nghiệm

### Mô hình
**HADES** — mô hình phát hiện bất thường không giám sát dựa trên GAN, huấn luyện chỉ trên dữ liệu bình thường.

### Giao thức đánh giá

| Cấu hình | Giá trị |
|:---|:---|
| Tập dữ liệu | SocialNetwork (AnoMod), 12 scenario lỗi |
| Loại dữ liệu | `fuse` (KPI + Nhật ký [+ Trace khi `open_trace=True`]) |
| Train / unlabel | 31 cửa sổ (80% của 39 cửa sổ `Normal_Baseline`) |
| Val | 8 cửa sổ (20% còn lại của `Normal_Baseline`) |
| Test mỗi scenario | 326 cửa sổ = 320 normal + 6 anomaly (**1,84%**); `Svc_Kill_*`: 324 = 320 + 4 (**1,23%**) |
| `window_size` | 5 (5 cửa sổ × 30 s) |
| `val_percentile` | 95 |
| `epoches` / `patience` | 50 50 / 15 (baseline và trace như nhau) |
| `batch_size`, `alpha`, `open_gan_sep` | 256, 0.16, True |
| `activity_penalty_weight` | 1.5 (baseline và trace như nhau) |
| `gate_delta_lr_mult` | 10 (chỉ trace — baseline không có `trace_gate`/`delta_head`) |
| `run_end` | 1 (chạy đơn lần, 1 seed) |

### Hiệu chỉnh ngưỡng
Ngưỡng bất thường không dùng nhãn test: `threshold = np.percentile(val_losses, 95)`, với `val_losses` là loss của 8 cửa sổ normal trong `val.pkl`.

### Thư mục kết quả
| Cấu hình | Thư mục |
|:---|:---|
| Baseline (KPI + Nhật ký) | `data/sn/result_per_scenario_fuse_baseline/` |
| Trace (KPI + Nhật ký + Trace) | `data/sn/result_per_scenario_fuse_trace/` |

## 2. Nhãn normal / anomaly theo từng loại lỗi

Cửa sổ lỗi (`FAULT_WINDOWS` trong `codes/common/preprocess_sn.py`) được xác định từ script thu thập gốc của AnoMod (`github.com/EvoTestOps/AnoMod`, `automated_multimodal_collection.sh`: thu thập bắt đầu 15 s sau khi inject lỗi) và từ dữ liệu đo trực tiếp. Thời điểm tính từ lúc bắt đầu ghi của từng session:

| Loại lỗi | Cơ chế (theo script) | Cửa sổ anomaly | Phần còn lại của session |
|---|---|---|---|
| `Code_Stop_*` | `docker stop`, không tự restart | Cả session | – |
| `Perf_*`, `DB_Redis_CacheLimit_*` | ChaosBlade `--timeout 300`, tự hết hiệu lực sau 300 s | 0–300 s | Normal (đã hồi phục) |
| `Svc_Kill_*` | ChaosBlade kill process + Docker auto-restart | 90–210 s | Normal |
| `Normal_Baseline` | – | – | Cả session normal |

Cửa sổ `Svc_Kill_*` được xác nhận bằng: cột `container_label_restartcount` chuyển 0→1 tại giây 105 ở cả 3 scenario (`Normal_Baseline` không có cột này), và khoảng lặng ~75 s (101,8 s → 176,8 s) trong trace của `user-timeline-service`.

## 3. Hai pool normal

- **Train / unlabel / val**: chỉ từ `Normal_Baseline`. Giữ hẹp có chủ đích: đưa thêm các cửa sổ ít hoạt động từ scenario khác vào train làm model coi "ít hoạt động" là bình thường, làm mất khả năng phát hiện lỗi "im lặng hoàn toàn" (mục 4.1).
- **Test** (phần normal của mỗi `test_<scenario>.pkl`): gộp từ **mọi** scenario (`Normal_Baseline` + phần đã hồi phục / chưa từng lỗi của từng scenario) — 320 cửa sổ trải khắp ~3 giờ thí nghiệm. Mỗi file test luôn so anomaly của scenario đó với normal từ nhiều session khác nhau, nên mô hình không thể dựa vào "đây là session nào" để phân biệt.

Anomaly luôn tách riêng theo scenario: mỗi `test_<scenario>.pkl` chỉ chứa anomaly của đúng scenario đó (subsample đều tối đa 6 cửa sổ).

## 4. Thành phần chấm điểm và huấn luyện

### 4.1 `activity_penalty_weight`
Với lỗi làm service "im lặng" (`Code_Stop_*`, `Svc_Kill_*`), reconstruction loss của cửa sổ anomaly **thấp hơn** cửa sổ normal — ví dụ `Code_Stop_TextService`: loss trung bình anomaly 0,92 so với normal 1,29; cả thành phần log lẫn KPI đều bị đảo như vậy. Input gần như trống/phẳng dễ tái tạo hơn input có biến động thật. `activity_penalty_weight` cộng vào điểm bất thường một số hạng độc lập với reconstruction: mức hoạt động hiện tại (`kpi_features.sum + log_features.sum`) thấp hơn mức normal của train bao nhiêu độ lệch chuẩn. Mặc định `0.0` (không ảnh hưởng dataset khác).

### 4.2 `gate_delta_lr_mult` (chỉ trace)
Learning rate riêng cho `trace_gate` và `delta_head` (nhân với `lr`). Cả hai lớp cuối khởi tạo bằng 0 nên gradient của mỗi bên tỉ lệ với giá trị gần 0 của bên kia (nghẽn nhân đôi); hệ số 10 giúp chúng dịch chuyển khỏi 0 nhanh hơn. Mặc định `1.0`.

### 4.3 `latency_dev` clip ±10
`latency_dev` (đặc trưng trace thứ 6) là z-score so với `Normal_Baseline`; `bl_std` ước lượng từ ít mẫu có thể gần 0 làm z-score lên hàng nghìn. Giá trị được cắt về [−10, 10]. Chỉ chạm ~0,1–0,2% giá trị test và 0% giá trị train/val.

### 4.4 `epoches` / `patience`
50 50 / 15 cho cả hai cấu hình.

## 5. Lệnh chạy

```bash
cd D:/UAM-AD/codes
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

## 6. Kết quả

| Scenario | Baseline F1 | P | R | Trace F1 | P | R | Δ F1 |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| Code_Stop_MediaService | 0,500 | 0,333 | 1,000 | **0,600** | 0,429 | 1,000 | +0,100 |
| Code_Stop_TextService | 0,462 | 0,300 | 1,000 | **0,632** | 0,462 | 1,000 | +0,170 |
| Code_Stop_UserService | 0,462 | 0,300 | 1,000 | **0,615** | 0,571 | 0,667 | +0,154 |
| DB_Redis_CacheLimit_HomeTimeline | 0,333 | 0,200 | 1,000 | **0,375** | 0,231 | 1,000 | +0,042 |
| DB_Redis_CacheLimit_SocialGraph | 0,462 | 0,300 | 1,000 | 0,435 | 0,294 | 0,833 | −0,027 |
| DB_Redis_CacheLimit_UserTimeline | 0,200 | 0,250 | 0,167 | **0,276** | 0,174 | 0,667 | +0,076 |
| Perf_CPU_Contention | 0,333 | 0,200 | 1,000 | 0,345 | 0,217 | 0,833 | +0,012 |
| Perf_Disk_IO_Stress | 0,333 | 0,200 | 1,000 | 0,345 | 0,217 | 0,833 | +0,012 |
| Perf_Network_Loss | 0,200 | 0,250 | 0,167 | **0,345** | 0,217 | 0,833 | +0,145 |
| Svc_Kill_Media | 0,242 | 0,138 | 1,000 | 0,267 | 0,154 | 1,000 | +0,025 |
| Svc_Kill_SocialGraph | 0,286 | 0,167 | 1,000 | 0,320 | 0,190 | 1,000 | +0,034 |
| Svc_Kill_UserTimeline | 0,296 | 0,174 | 1,000 | 0,333 | 0,200 | 1,000 | +0,037 |
| **Trung bình** | **0,342** | 0,234 | 0,861 | **0,407** | 0,280 | 0,889 | **+0,065** |
| Độ lệch chuẩn | 0,102 | 0,061 | 0,311 | 0,127 | 0,128 | 0,124 | |

F1 trung bình tăng từ 0,342 (baseline) lên 0,407 (trace); trace cao hơn ở 11/12 scenario.

### Nhóm lỗi luồng gọi hệ thống (service kill / dừng)

| Scenario | Baseline F1 | Trace F1 | Δ |
|---|---:|---:|:---:|
| Code_Stop_MediaService | 0,500 | 0,600 | +0,100 |
| Code_Stop_TextService | 0,462 | 0,632 | +0,170 |
| Code_Stop_UserService | 0,462 | 0,615 | +0,154 |
| Svc_Kill_Media | 0,242 | 0,267 | +0,025 |
| Svc_Kill_SocialGraph | 0,286 | 0,320 | +0,034 |
| Svc_Kill_UserTimeline | 0,296 | 0,333 | +0,037 |
| **Trung bình** | **0,375** | **0,461** | **+0,087** |

F1 trung bình nhóm này tăng từ 0,375 lên 0,461; trace cao hơn ở cả 6/6. `Code_Stop_*` tăng +0,10 đến +0,17; `Svc_Kill_*` (tín hiệu chỉ ~2 phút) tăng +0,03 đến +0,04.

## 7. Ablation

### 7.1 Trace (12 scenario)

| Cấu hình | epoch/patience | `gate_delta_lr_mult` | F1 TB (12) | F1 TB nhóm luồng gọi (6) | So với baseline cùng epoch |
|:---|:---:|:---:|---:|---:|:---:|
| Baseline | 50/15 | – | 0,342 | 0,375 | – |
| Trace | 50/15 | 1 | 0,355 | 0,421 | 5 thắng / 3 hòa / 4 thua |
| Trace | 10/5 | 10 | 0,372 | 0,418 | 6 thắng / 1 hòa / 5 thua |
| **Trace** | 50/15 | 10 | **0,407** | **0,461** | 11 thắng / 1 thua |

`Code_Stop_*` không nhạy với hai tham số này (trace cao hơn baseline ở mọi dòng). Ở các scenario còn lại, tăng epoch hoặc `gate_delta_lr_mult` riêng lẻ chưa đủ để trace vượt baseline nhất quán (VD không có `gate_delta_lr_mult`, trace 0,160 so với baseline 0,286 ở `Svc_Kill_SocialGraph`); dùng cả hai thì F1 nhóm luồng gọi tăng từ 0,375 (baseline) lên 0,461.

### 7.2 Các thành phần khác (đo trên baseline, 10 epoch, 6 scenario luồng gọi)

| `activity_penalty_weight` | 0 | 1,0 | 1,5 |
|---|---:|---:|---:|
| F1 trung bình | 0,036 | 0,296 | 0,320 |

Baseline: `epoches/patience` 10/5 → 50/15 (weight 1,5) tăng F1 trung bình 12 scenario từ 0,289 lên 0,342. Phần clip `latency_dev` (mục 4.3) chưa được ablation riêng (cần preprocess lại).

## 8. Giới hạn

- **1 seed, ít cửa sổ anomaly**: mỗi file test chỉ có 4–6 cửa sổ anomaly nên lệch 1 cửa sổ đổi F1 khoảng 0,05–0,1; các chênh lệch +0,01 đến +0,04 (`Svc_Kill_*`, `Perf_*`) nằm trong nhiễu. Kết luận vững nhất là nhóm `Code_Stop_*`.
- **Val chỉ 8 cửa sổ** nên ngưỡng percentile 95 kém ổn định.
- **Precision thấp** (0,15–0,57): `activity_penalty_weight` đổi precision lấy recall.
- `DB_Redis_CacheLimit_SocialGraph` là scenario duy nhất trace thấp hơn baseline (−0,027).
- Dataset chỉ có 1 session `Normal_Baseline` thật nên train luôn nhỏ (31 cửa sổ).
- `gate_delta_lr_mult` không có đối tượng tương ứng ở baseline; đây là điểm bất đối xứng còn lại giữa hai cấu hình.

## 9. Tệp liên quan

| Nội dung | File |
|---|---|
| `FAULT_WINDOWS`, hai pool normal, `latency_dev` clip | `codes/common/preprocess_sn.py` |
| `activity_penalty_weight`, `gate_delta_lr_mult` | `codes/models/basev3.py`, `codes/run.py` |
| Chuyển tiếp tham số qua wrapper | `codes/common/eval_per_scenario_sn.py` |
| Checkpoint | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
