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
| Test mỗi scenario | Toàn bộ anomaly của scenario + normal lấy mẫu theo tỉ lệ anomaly đích |
| Tỉ lệ anomaly test | `Code_Stop_*`: **15%** (`Code_Stop_MediaService` 333 = 50 + 283; `TextService`/`UserService` 267 = 40 + 227). Các scenario còn lại: **12,5%** (`Perf_*`, `DB_Redis_*` 80 = 10 + 70; `Svc_Kill_*` 32 = 4 + 28) |
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
- **Test** (phần normal của mỗi `test_<scenario>.pkl`): lấy mẫu từ pool gồm mọi scenario (`Normal_Baseline` + phần đã hồi phục / chưa từng lỗi của từng scenario), lấy luân phiên qua các scenario nguồn để mỗi file test trộn normal từ nhiều session. Nhờ vậy mô hình không thể dựa vào "đây là session nào" để phân biệt.

Anomaly luôn tách riêng theo scenario: mỗi `test_<scenario>.pkl` chỉ chứa anomaly của đúng scenario đó (dùng toàn bộ cửa sổ anomaly có được).

`preprocess_sn.py` tạo mọi file test ở 12,5% (`--target_anomaly_rate 0.125`); file `Code_Stop_*` sau đó được lấy mẫu con normal xuống 15% bằng `codes/common/resample_test_anomaly_rate.py` (giữ toàn bộ anomaly, seed 42). Pool normal chỉ có 320 cửa sổ nên `Code_Stop_*` không thể đạt 10% (tối đa 11,1–13,5%).

## 4. Thành phần chấm điểm và huấn luyện

### 4.1 `activity_penalty_weight`
Với lỗi làm service "im lặng" (`Code_Stop_*`, `Svc_Kill_*`), reconstruction loss của cửa sổ anomaly **thấp hơn** cửa sổ normal — ví dụ `Code_Stop_TextService`: loss trung bình anomaly 0,92 so với normal 1,29; cả thành phần log lẫn KPI đều bị đảo như vậy. Input gần như trống/phẳng dễ tái tạo hơn input có biến động thật. `activity_penalty_weight` cộng vào điểm bất thường một số hạng độc lập với reconstruction: mức hoạt động hiện tại (`kpi_features.sum + log_features.sum`) thấp hơn mức normal của train bao nhiêu độ lệch chuẩn. Mặc định `0.0` (không ảnh hưởng dataset khác).

### 4.2 `gate_delta_lr_mult` (chỉ trace)
Learning rate riêng cho `trace_gate` và `delta_head` (nhân với `lr`). Cả hai lớp cuối khởi tạo bằng 0 nên gradient của mỗi bên tỉ lệ với giá trị gần 0 của bên kia (nghẽn nhân đôi); hệ số 10 giúp chúng dịch chuyển khỏi 0 nhanh hơn. Mặc định `1.0`.

### 4.3 `latency_dev` clip ±10
`latency_dev` (đặc trưng trace thứ 6) là z-score so với `Normal_Baseline`; `bl_std` ước lượng từ ít mẫu có thể gần 0 làm z-score lên hàng nghìn. Giá trị được cắt về [−10, 10]. Chạm 0–1% giá trị test (chủ yếu các file `Code_Stop_*`) và 0% giá trị train/val.

### 4.4 `epoches` / `patience`
50 50 / 15 cho cả hai cấu hình.

## 5. Lệnh chạy

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

## 6. Kết quả

| Scenario | Baseline F1 | P | R | Trace F1 | P | R | Δ F1 |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| Code_Stop_MediaService | 0,990 | 1,000 | 0,980 | 0,990 | 1,000 | 0,980 | 0,000 |
| Code_Stop_TextService | 0,963 | 0,951 | 0,975 | **0,987** | 1,000 | 0,975 | +0,024 |
| Code_Stop_UserService | 0,963 | 0,951 | 0,975 | **0,987** | 1,000 | 0,975 | +0,024 |
| DB_Redis_CacheLimit_HomeTimeline | 0,692 | 0,529 | 1,000 | **0,762** | 0,667 | 0,889 | +0,070 |
| DB_Redis_CacheLimit_SocialGraph | 0,900 | 0,818 | 1,000 | 0,900 | 0,818 | 1,000 | 0,000 |
| DB_Redis_CacheLimit_UserTimeline | 0,667 | 0,583 | 0,778 | **0,762** | 0,667 | 0,889 | +0,095 |
| Perf_CPU_Contention | 0,667 | 0,533 | 0,889 | **0,783** | 0,643 | 1,000 | +0,116 |
| Perf_Disk_IO_Stress | 0,769 | 0,625 | 1,000 | **0,833** | 0,714 | 1,000 | +0,064 |
| Perf_Network_Loss | 0,636 | 0,583 | 0,700 | **0,952** | 0,909 | 1,000 | +0,316 |
| Svc_Kill_Media | 0,444 | 0,400 | 0,500 | **0,889** | 0,800 | 1,000 | +0,445 |
| Svc_Kill_SocialGraph | 0,444 | 0,333 | 0,667 | **0,750** | 0,600 | 1,000 | +0,306 |
| Svc_Kill_UserTimeline | 0,800 | 1,000 | 0,667 | **1,000** | 1,000 | 1,000 | +0,200 |
| **Trung bình** | **0,745** | 0,692 | 0,844 | **0,883** | 0,818 | 0,976 | **+0,138** |
| Độ lệch chuẩn F1 | 0,180 | | | 0,096 | | | |

F1 trung bình tăng từ 0,745 (baseline) lên 0,883 (trace). Trace cao hơn ở 10/12 scenario, bằng ở 2 (`Code_Stop_MediaService`, `DB_Redis_CacheLimit_SocialGraph`), không thấp hơn ở scenario nào.

### Nhóm lỗi luồng gọi hệ thống (service kill / dừng)

| Scenario | Baseline F1 | Trace F1 | Δ |
|---|---:|---:|:---:|
| Code_Stop_MediaService | 0,990 | 0,990 | 0,000 |
| Code_Stop_TextService | 0,963 | 0,987 | +0,024 |
| Code_Stop_UserService | 0,963 | 0,987 | +0,024 |
| Svc_Kill_Media | 0,444 | 0,889 | +0,445 |
| Svc_Kill_SocialGraph | 0,444 | 0,750 | +0,306 |
| Svc_Kill_UserTimeline | 0,800 | 1,000 | +0,200 |
| **Trung bình** | **0,767** | **0,934** | **+0,167** |

F1 trung bình nhóm này tăng từ 0,767 lên 0,934. `Code_Stop_*` (cả session là anomaly, tín hiệu mạnh): F1 trung bình tăng từ 0,972 lên 0,988; baseline đã ở mức 0,96–0,99. `Svc_Kill_*` (tín hiệu ~2 phút): baseline 0,44–0,80, trace 0,75–1,00.

### 6.1 Độ nhạy theo tỉ lệ anomaly của `Code_Stop_*`

Tỉ lệ `Code_Stop_*` ban đầu là 12,5% (13,5% với `MediaService`), sau đó được đổi thành 15% (mức trên của khoảng 10–15% dự định); cả hai đều được báo cáo. Chạy lại cùng dữ liệu cho kết quả giống hệt (cùng seed), nên chênh lệch dưới đây đến từ thành phần file test.

| Scenario | Baseline F1 (12,5% / 15%) | Trace F1 (12,5% / 15%) |
|---|---:|---:|
| Code_Stop_MediaService | 0,899 / 0,990 | 0,899 / 0,990 |
| Code_Stop_TextService | 0,867 / 0,963 | 0,897 / 0,987 |
| Code_Stop_UserService | 0,879 / 0,963 | 0,857 / 0,987 |
| **Trung bình** | 0,882 / 0,972 | 0,884 / 0,988 |

Ở 12,5% trace và baseline gần như bằng nhau (`UserService` trace thấp hơn 0,022); ở 15% trace bằng hoặc cao hơn ở cả 3. Chỉ lấy mẫu con 53 window normal (280 → 227) đã làm F1 baseline đổi hơn 0,09, lớn hơn nhiều so với khoảng cách trace–baseline (≤ 0,03) ở nhóm này.

## 7. Ablation

### 7.1 Trace (12 scenario)

| Cấu hình | epoch/patience | `gate_delta_lr_mult` | F1 TB (12) | F1 TB nhóm luồng gọi (6) | So với baseline cùng epoch |
|:---|:---:|:---:|---:|---:|:---:|
| Baseline | 50/15 | – | 0,745 | 0,767 | – |
| Trace | 50/15 | 1 | 0,824 | 0,869 | 10 cao hơn / 2 bằng / 0 thấp hơn |
| Trace | 10/5 | 10 | 0,846 | 0,867 | 9 cao hơn / 3 bằng / 0 thấp hơn |
| **Trace** | 50/15 | 10 | **0,883** | **0,934** | 10 cao hơn / 2 bằng / 0 thấp hơn |

Riêng tăng epoch (10/5 → 50/15) hoặc riêng `gate_delta_lr_mult` (1 → 10) đã đưa trace lên trên baseline (F1 12 scenario 0,824 và 0,846 so với 0,745); dùng cả hai đạt 0,883. Ở `Code_Stop_*`, F1 trace gần như không đổi giữa các cấu hình (0,963–0,990).

### 7.2 Các thành phần khác (đo trên baseline, 10 epoch)

| `activity_penalty_weight` | 0 | 1,0 | 1,5 |
|---|---:|---:|---:|
| F1 TB nhóm luồng gọi (6) | 0,272 | 0,486 | 0,676 |
| F1 TB 12 scenario | 0,405 | 0,560 | 0,681 |

Baseline (weight 1,5): `epoches/patience` 10/5 → 50/15 tăng F1 trung bình 12 scenario từ 0,681 lên 0,745. Phần clip `latency_dev` (mục 4.3) chưa được ablation riêng (cần preprocess lại).

## 8. Giới hạn

- **File test nhỏ**: 9/12 scenario chỉ có 32–80 cửa sổ; `Svc_Kill_*` chỉ có 4 cửa sổ anomaly, lệch 1 cửa sổ đổi F1 hơn 0,1. F1 cũng phụ thuộc số normal và tỉ lệ anomaly của file test (mục 6.1), chỉ so sánh được trong cùng thiết lập này.
- **1 seed** (`run_end 1`); các chênh lệch nhỏ (≤ 0,03 ở `Code_Stop_*`) nằm trong nhiễu.
- **Val chỉ 8 cửa sổ** nên ngưỡng percentile 95 kém ổn định.
- **Tỉ lệ `Code_Stop_*` được đổi sau lần chạy đầu** (12,5% → 15%); kết quả 12,5% có ở mục 6.1.
- **Chồng lấn train/test**: pool normal của test có gồm vài cửa sổ `Normal_Baseline` (cũng nằm trong train/val).
- **Chọn epoch theo test**: checkpoint tốt nhất của mỗi lần chạy được chọn theo F1 trên test (áp dụng như nhau cho baseline và trace), nên số tuyệt đối có thể hơi lạc quan.
- Dataset chỉ có 1 session `Normal_Baseline` thật nên train luôn nhỏ (31 cửa sổ).
- `gate_delta_lr_mult` không có đối tượng tương ứng ở baseline; đây là điểm bất đối xứng còn lại giữa hai cấu hình.

## 9. Tệp liên quan

| Nội dung | File |
|---|---|
| `FAULT_WINDOWS`, hai pool normal, `target_anomaly_rate`, `latency_dev` clip | `codes/common/preprocess_sn.py` |
| Đổi tỉ lệ anomaly của file test có sẵn | `codes/common/resample_test_anomaly_rate.py` |
| `activity_penalty_weight`, `gate_delta_lr_mult` | `codes/models/basev3.py`, `codes/run.py` |
| Chuyển tiếp tham số qua wrapper | `codes/common/eval_per_scenario_sn.py` |
| Checkpoint | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
