# Tiền xử lý: Tập dữ liệu SocialNetwork (AnoMod)

## 1. Tổng quan tập dữ liệu

Tập dữ liệu SocialNetwork (SN) là một phần của **benchmark AnoMod** dành cho phát hiện bất thường trong kiến trúc microservice đám mây. Dữ liệu được thu thập từ ứng dụng microservice gồm 12 dịch vụ, triển khai trên cụm thực tế. Tập dữ liệu ghi lại ba phương thức:

| Phương thức  | Nguồn                                      | Vị trí         |
|:-------------|:-------------------------------------------|:---------------|
| Chỉ số KPI   | Hệ thống + container + Jaeger spans        | `metric_data/` |
| Nhật ký      | File log theo từng dịch vụ                 | `log_data/`    |
| Trace        | Distributed traces (Jaeger)                | `trace_data/`  |

### Các kịch bản

13 session được ghi liên tiếp trong một lần chạy ~3 giờ (`Normal_Baseline` ghi đầu tiên). Cơ chế lỗi lấy từ script thu thập của AnoMod (`github.com/EvoTestOps/AnoMod`, `automated_multimodal_collection.sh`), trong đó việc thu thập bắt đầu 15 s sau khi inject lỗi:

| Loại                  | Số lượng | Cơ chế (theo script)                                          | Cửa sổ anomaly (tính từ đầu session) |
|:----------------------|---------:|:--------------------------------------------------------------|:-------------------------------------|
| Normal_Baseline       |        1 | Lưu lượng ổn định, không lỗi                                  | – (cả session normal)                |
| Code_Stop_*           |        3 | `docker stop` container dịch vụ, không restart                | cả session                           |
| DB_Redis_CacheLimit_* |        3 | ChaosBlade giới hạn cache Redis, `--timeout 300`              | 0–300 s                              |
| Perf_CPU_Contention   |        1 | ChaosBlade CPU stress, `--timeout 300`                        | 0–300 s                              |
| Perf_Disk_IO_Stress   |        1 | ChaosBlade disk I/O stress, `--timeout 300`                   | 0–300 s                              |
| Perf_Network_Loss     |        1 | ChaosBlade mất gói tin, `--timeout 300`                       | 0–300 s                              |
| Svc_Kill_*            |        3 | ChaosBlade kill process (SIGKILL) + Docker tự restart         | 90–210 s                             |

Cửa sổ `Svc_Kill_*` được xác nhận từ dữ liệu: cột `container_label_restartcount` chuyển 0→1 tại giây 105 ở cả 3 scenario (`Normal_Baseline` không có cột này), và `user-timeline-service` có khoảng lặng ~75 s (101,8 s → 176,8 s) trong trace. Phần còn lại của mỗi session (ngoài cửa sổ anomaly) được coi là normal.

Mỗi session ghi khoảng **19,5–25 phút**.

---

## 2. Kỹ thuật đặc trưng

### 2.1 Phân cửa sổ

Dữ liệu chuỗi thời gian thô được chia thành các **cửa sổ không chồng lấp** dài `window_sec=30` giây, mỗi cửa sổ là một điểm dữ liệu. Cửa sổ 0 bắt đầu đúng tại timestamp đầu tiên của session; không bỏ qua giai đoạn khởi động nào, vì nhãn anomaly được tính theo độ lệch từ đầu session (mục 3.1).

**Tại sao chọn 30 giây?**
- Đủ ngắn để nắm bắt các bất thường thoáng qua (kill dịch vụ xuất hiện trong vài giây)
- Đủ dài để tạo ra các thống kê tổng hợp ổn định (tránh nhiễu từ các lần đọc chỉ số riêng lẻ)

`Normal_Baseline` (19,5 phút) cho **39 cửa sổ**; các session khác cho khoảng 39–50 cửa sổ.

### 2.2 Đặc trưng KPI (59 chiều)

| Nhóm      | Số lượng | Đặc trưng                                                                                                                                                    |
|:----------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hệ thống  |       10 | cpu_usage, disk_io_time, disk_read_bytes, disk_usage_pct, disk_write_bytes, load1, memory_usage_pct, network_errors, network_receive_bytes, network_transmit_bytes |
| Container |       48 | 12 dịch vụ × 4 chỉ số: cpu, memory, net_rx, net_tx                                                                                                          |
| Jaeger    |        1 | spans_rate (result="ok", chuẩn hóa theo cửa sổ)                                                                                                             |

Mỗi KPI được tổng hợp theo cửa sổ bằng giá trị trung bình của tất cả mẫu trong khoảng 30 giây đó. Giá trị thiếu (ví dụ: container chưa khởi động) được điền bằng giá trị trung bình cột tính từ các cửa sổ không có NaN.

### 2.3 Đặc trưng nhật ký

Nhật ký được phân tích bằng **Drain3** (streaming template miner), khớp chỉ trên nhật ký Normal_Baseline:
- Template học được: ~458 từ 317.055 tin nhắn log
- Loại đặc trưng: `template_appear` — sự hiện diện/vắng mặt nhị phân của mỗi template trong cửa sổ
- Template mới gặp trong kịch bản bất thường được coi là chưa thấy (ánh xạ vào bucket "unknown" đặc biệt)

**Tại sao khớp Drain3 chỉ trên Normal_Baseline?**  
Để tránh nhiễm bởi các mẫu log bất thường khi xây dựng từ điển. Mô hình cần học cách đánh dấu các template chưa thấy là bất thường, điều này chỉ khả thi nếu từ điển template được xây dựng từ log bình thường.

### 2.4 Đặc trưng trace (tùy chọn, `open_trace=True`)

Với mỗi dịch vụ, một vector đặc trưng 6 chiều được tính toán theo cửa sổ (`trace_c=6`):

```
[call_count, avg_duration_us, max_duration_us, error_rate, root_rate, latency_dev]
```

- `call_count`: số lượng trace span liên quan đến dịch vụ này trong cửa sổ
- `avg_duration_us` / `max_duration_us`: thống kê độ trễ (chuẩn hoá về giây)
- `error_rate`: tỷ lệ span có HTTP status code không phải OK
- `root_rate`: tỷ lệ span là root span (điểm vào)
- `latency_dev`: z-score của `avg_duration` so với baseline per-service từ Normal_Baseline
  = `(avg_dur − mean_baseline) / (std_baseline + 1e-6)` — dương nghĩa là chậm hơn bình thường, **được cắt về [−10, 10]**

Baseline `latency_dev` được tính một lần từ **Normal_Baseline** `all_traces.csv` (mean và std per-service của `duration_us / 1e6`), sau đó áp dụng đồng nhất cho mọi scenario. Cần cắt giá trị vì `std_baseline` ước lượng từ ít mẫu có thể gần 0, khiến một đợt tăng độ trễ thật có z-score lên hàng nghìn và làm một node lấn át reconstruction loss.

Một **ma trận kề tĩnh** (12×12) được xây dựng từ trace Normal_Baseline: cạnh (i, j) = 1 nếu dịch vụ i gọi dịch vụ j ít nhất một lần. Đồ thị này cố định cho tất cả các kịch bản — chúng ta giả sử topo đồ thị lời gọi không thay đổi giữa các thí nghiệm.

---

## 3. Nhãn và chiến lược phân chia

### 3.1 Nhãn normal / anomaly

Mỗi scenario được tách thành cửa sổ anomaly và normal theo cửa sổ lỗi ở mục 1 (`FAULT_WINDOWS` trong `preprocess_sn.py`); phần đã hồi phục hoặc chưa từng lỗi của session là normal.

### 3.2 Chia train / val

```
Normal_Baseline (39 cửa sổ)                     → train.pkl = unlabel.pkl (cả 39)
Normal của mọi session khác (281 cửa sổ)        → theo từng session nguồn: 20% → val.pkl (57 cửa sổ)
                                                                            80% → pool normal của test (224 cửa sổ)
```

Val không được thấy khi huấn luyện, dùng để chọn model và lấy ngưỡng (mục 4). Val lấy từ cùng hỗn hợp session với normal của test nhưng không trùng với chúng, gồm 57 cửa sổ → (57 − 5) × 5 = 260 điểm số (train/val dùng cửa sổ trượt 5).

Train được giữ hẹp có chủ đích. Đưa thêm các cửa sổ ít hoạt động từ scenario khác vào train làm model coi "ít hoạt động" là bình thường, phá khả năng phát hiện lỗi "im lặng hoàn toàn" (`Code_Stop_*`, `Svc_Kill_*`): cửa sổ gần như trống dễ tái tạo hơn cửa sổ bận rộn nên reconstruction loss của nó thấp hơn cửa sổ normal.

### 3.3 File test (theo từng scenario)

```
test_{scenario}.pkl = cửa sổ anomaly của scenario (tối đa --max_anomalies = 39, lấy đều theo thời gian)
                    + cửa sổ normal lấy từ pool test 224 cửa sổ
                    → xáo trộn
```

- **Anomaly**: chỉ anomaly của đúng scenario đó; chỉ `Code_Stop_*` (có 40–50) bị giảm xuống 39.
- **Normal**: lấy luân phiên qua các session nguồn nên mỗi file test trộn normal từ nhiều session. Nếu normal chỉ đến từ một session cố định, model có thể phân biệt bằng "đây là session nào" thay vì bằng lỗi.
- **Kích thước**: số normal = anomaly × (1 − r) / r với r = `--target_anomaly_rate` (mặc định 0,125), bị giới hạn bởi pool 224 cửa sổ:

| Scenario | Anomaly | Normal | Tổng | Tỉ lệ |
|:--|--:|--:|--:|--:|
| `Code_Stop_*` (3 file) | 39 | 224 | 263 | 14,8% |
| `DB_Redis_*`, `Perf_*` (6 file) | 10 | 70 | 80 | 12,5% |
| `Svc_Kill_*` (3 file) | 4 | 28 | 32 | 12,5% |

**Vì sao xáo trộn?** Model nhận luồng hỗn hợp (như production) và phải chấm điểm từng cửa sổ; nếu không xáo, mọi cửa sổ anomaly sẽ nằm cuối.

---

## 4. Ngưỡng và F1

Giao thức đầy đủ ở `docs/evaluation_protocol_vi.md`. Tóm tắt:

- **Chọn model**: epoch có điểm số val trung bình thấp nhất (loss, cộng số hạng activity khi `--activity_penalty_weight > 0`).
- **Ngưỡng**: `np.percentile(val_scores, 95)` của model đã chọn (`--val_percentile`, mặc định 95).
- **Chỉ số chính**: precision / recall / F1 tại ngưỡng đó (không `point_adjust`, không nhãn test), kèm AUROC / AUPRC không phụ thuộc ngưỡng.
- **F1 oracle** (quét ngưỡng theo nhãn test, có `point_adjust`) lưu ở các trường `oracle_*` riêng.

Kết quả nằm ở `docs/experiment_results_sn_trace_vs_baseline_vi.md`.

---

## 5. Cấu trúc đầu ra

```
data/sn/
├── train.pkl              # 39 cửa sổ normal (Normal_Baseline)
├── unlabel.pkl            # giống train, cho pha unlabeled của GAN
├── val.pkl                # 57 cửa sổ normal (session khác), để chọn model và lấy ngưỡng
├── meta.pkl               # metadata dataset (ma trận kề, số chiều đặc trưng, ...)
└── scenarios/
    ├── test_Code_Stop_MediaService_20251104_024819.pkl   # 263 cửa sổ (39 anomaly)
    ├── test_Code_Stop_TextService_20251104_022416.pkl    # 263 (39)
    ├── test_Code_Stop_UserService_20251104_020019.pkl    # 263 (39)
    ├── test_DB_Redis_CacheLimit_HomeTimeline_20251104_004905.pkl   # 80 (10)
    ├── test_DB_Redis_CacheLimit_SocialGraph_20251104_013615.pkl
    ├── test_DB_Redis_CacheLimit_UserTimeline_20251104_011238.pkl
    ├── test_Perf_CPU_Contention_20251103_222601.pkl
    ├── test_Perf_Disk_IO_Stress_20251103_231335.pkl
    ├── test_Perf_Network_Loss_20251103_224954.pkl
    ├── test_Svc_Kill_Media_20251104_000111.pkl           # 32 (4)
    ├── test_Svc_Kill_SocialGraph_20251104_002506.pkl
    └── test_Svc_Kill_UserTimeline_20251103_233717.pkl
```

Các file `*.pkl` bị git-ignore và được tạo lại bằng lệnh bên dưới.

---

## 6. Cách dùng

```bash
python codes/common/preprocess_sn.py \
    --sn_data_root D:/AnoMod/SN_data \
    --output_dir data/sn \
    --window_sec 30 \
    --target_anomaly_rate 0.125 \
    --max_anomalies 39 \
    --seed 42
```

### Tham số chính

| Tham số                 | Giá trị | Lý do                                                                     |
|:------------------------|-------:|:--------------------------------------------------------------------------|
| `--window_sec`          |     30 | Độ mịn 30 giây: bắt được bất thường thoáng qua, tổng hợp ổn định          |
| `--target_anomaly_rate` |  0.125 | Tỉ lệ anomaly của mỗi file test; normal lấy tới tối đa kích thước pool    |
| `--max_anomalies`       |     39 | Giới hạn mỗi file test để pool 224 cửa sổ vẫn cho khoảng 15% ở `Code_Stop_*` |
| `--seed`                |     42 | Tái lập: điều khiển xáo trộn và lấy mẫu                                   |
