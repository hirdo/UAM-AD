# Kết quả thực nghiệm: SocialNetwork — Trace so với Baseline

Đánh giá theo giao thức chuẩn trong [`evaluation_protocol_vi.md`](evaluation_protocol_vi.md): chọn epoch theo val loss, ngưỡng = p95 điểm số val, **F1 tại ngưỡng val là chỉ số chính**, AUROC / AUPRC là phụ, F1 oracle ghi riêng.

## 1. Thiết lập thực nghiệm

### Mô hình
**HADES** — mô hình phát hiện bất thường không giám sát dựa trên GAN, chỉ huấn luyện trên dữ liệu normal.

### Cấu hình

| Thiết lập                             | Giá trị                                                                                                                                                       |
| :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset                               | SocialNetwork (AnoMod), 12 scenario lỗi                                                                                                                       |
| Loại dữ liệu                          | `fuse` (KPI + Nhật ký [+ Trace khi `open_trace=True`])                                                                                                        |
| Train / unlabel                       | 39 cửa sổ (toàn bộ `Normal_Baseline`)                                                                                                                         |
| Val                                   | 57 cửa sổ normal (20% normal của mỗi session khác) → 260 điểm số                                                                                              |
| Test mỗi scenario                     | Cửa sổ anomaly + normal từ pool test 224 cửa sổ; `Code_Stop_*` 263 = 39 + 224 (14,8%), `Perf_*`/`DB_Redis_*` 80 = 10 + 70 và `Svc_Kill_*` 32 = 4 + 28 (12,5%) |
| `window_size`                         | 5 (5 cửa sổ × 30 s)                                                                                                                                           |
| `val_percentile`                      | 95                                                                                                                                                            |
| `epoches` / `patience`                | 50 50 / 15 (giống nhau cho baseline và trace)                                                                                                                 |
| `batch_size`, `alpha`, `open_gan_sep` | 256, 0.16, True                                                                                                                                               |
| `activity_penalty_weight`             | 1.5 (giống nhau cho baseline và trace)                                                                                                                        |
| `gate_delta_lr_mult`                  | 10 (chỉ trace — baseline không có `trace_gate`/`delta_head`)                                                                                                  |
| `run_end`                             | 1 (một lần chạy, một seed)                                                                                                                                    |

### Thư mục kết quả
| Cấu hình                      | Thư mục                                      |
| :---------------------------- | :------------------------------------------- |
| Baseline (KPI + Nhật ký)      | `data/sn/result_per_scenario_fuse_baseline/` |
| Trace (KPI + Nhật ký + Trace) | `data/sn/result_per_scenario_fuse_trace/`    |

## 2. Nhãn normal / anomaly theo từng loại lỗi

Cửa sổ lỗi (`FAULT_WINDOWS` trong `codes/common/preprocess_sn.py`) được xác định từ script thu thập gốc của AnoMod (`github.com/EvoTestOps/AnoMod`, `automated_multimodal_collection.sh`: thu thập bắt đầu 15 s sau khi inject lỗi) và từ dữ liệu đo trực tiếp. Thời điểm tính từ lúc bắt đầu ghi của từng session:

| Loại lỗi                          | Cơ chế (theo script)                                  | Cửa sổ anomaly | Phần còn lại của session |
| --------------------------------- | ----------------------------------------------------- | -------------- | ------------------------ |
| `Code_Stop_*`                     | `docker stop`, không tự restart                       | Cả session     | –                        |
| `Perf_*`, `DB_Redis_CacheLimit_*` | ChaosBlade `--timeout 300`, tự hết hiệu lực sau 300 s | 0–300 s        | Normal (đã hồi phục)     |
| `Svc_Kill_*`                      | ChaosBlade kill process + Docker auto-restart         | 90–210 s       | Normal                   |
| `Normal_Baseline`                 | –                                                     | –              | Cả session normal        |

Cửa sổ `Svc_Kill_*` được xác nhận bằng: cột `container_label_restartcount` chuyển 0→1 tại giây 105 ở cả 3 scenario (`Normal_Baseline` không có cột này), và khoảng lặng ~75 s (101,8 s → 176,8 s) trong trace của `user-timeline-service`.

## 3. Hai pool normal

- **Train / unlabel**: toàn bộ 39 cửa sổ `Normal_Baseline`. Giữ hẹp có chủ đích: đưa thêm các cửa sổ ít hoạt động từ scenario khác vào train làm model coi "ít hoạt động" là bình thường, làm mất khả năng phát hiện lỗi "im lặng hoàn toàn" (mục 4.1).
- **Val và normal của test**: các cửa sổ normal của mọi session khác (281 cửa sổ), chia 20% / 80% theo từng session nguồn: 57 cửa sổ cho val, 224 cho pool test. Mỗi file test lấy normal luân phiên từ pool test nên trộn nhiều session; val lấy từ cùng hỗn hợp nhưng không trùng với test.

Mỗi `test_<scenario>.pkl` chứa anomaly của chính scenario đó (tối đa 39, lấy đều theo thời gian; chỉ `Code_Stop_*` vượt mức này) cùng normal để đạt tỉ lệ đích (`--target_anomaly_rate 0.125`, bị giới hạn bởi pool 224 cửa sổ nên `Code_Stop_*` là 14,8%).

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

### 6.1 Chỉ số chính: F1 tại ngưỡng val (p95 điểm số val)

| Scenario                         | Baseline F1 |     P |     R |  Trace F1 |     P |     R |    Δ F1    |
| :------------------------------- | ----------: | ----: | ----: | --------: | ----: | ----: | :--------: |
| Code_Stop_MediaService           |       0,690 | 0,527 | 1,000 | **0,772** | 0,629 | 1,000 |   +0,082   |
| Code_Stop_TextService            |       0,709 | 0,549 | 1,000 | **0,796** | 0,661 | 1,000 |   +0,087   |
| Code_Stop_UserService            |       0,731 | 0,576 | 1,000 | **0,817** | 0,691 | 1,000 |   +0,086   |
| DB_Redis_CacheLimit_HomeTimeline |       0,364 | 0,333 | 0,400 | **0,800** | 0,667 | 1,000 |   +0,436   |
| DB_Redis_CacheLimit_SocialGraph  |       0,720 | 0,562 | 1,000 | **0,857** | 0,750 | 1,000 |   +0,137   |
| DB_Redis_CacheLimit_UserTimeline |       0,476 | 0,385 | 0,625 | **0,706** | 0,667 | 0,750 |   +0,230   |
| Perf_CPU_Contention              |       0,522 | 0,462 | 0,600 | **0,800** | 0,667 | 1,000 |   +0,278   |
| Perf_Disk_IO_Stress              |       0,552 | 0,400 | 0,889 | **0,615** | 0,471 | 0,889 |   +0,064   |
| Perf_Network_Loss                |       0,400 | 0,400 | 0,400 | **0,857** | 0,818 | 0,900 |   +0,457   |
| Svc_Kill_Media                   |       0,667 | 0,500 | 1,000 | **0,889** | 0,800 | 1,000 |   +0,222   |
| Svc_Kill_SocialGraph             |       0,600 | 0,429 | 1,000 | **0,750** | 0,600 | 1,000 |   +0,150   |
| Svc_Kill_UserTimeline            |       0,727 | 0,571 | 1,000 |     0,727 | 0,571 | 1,000 |   +0,000   |
| **Trung bình**                   |   **0,596** | 0,475 | 0,826 | **0,782** | 0,666 | 0,962 | **+0,186** |
| Độ lệch chuẩn F1                 |       0,126 |       |       |     0,072 |       |       |            |

F1 của trace cao hơn ở 11/12 scenario, bằng ở 1/12 (`Svc_Kill_UserTimeline`) và thấp hơn ở 0. Recall đều bằng 1,000 cho cả hai ở toàn bộ file `Code_Stop_*` và hầu hết `Svc_Kill_*`, nên độ chính xác (cảnh báo giả tại ngưỡng p95) quyết định F1 ở đó.

### 6.2 Chỉ số phụ: AUROC, AUPRC và F1 oracle (baseline / trace)

| Scenario                         |     AUROC     |     AUPRC     |   F1 oracle   |
| :------------------------------- | :-----------: | :-----------: | :-----------: |
| Code_Stop_MediaService           | 0,977 / 0,977 | 0,780 / 0,786 | 0,918 / 0,918 |
| Code_Stop_TextService            | 0,974 / 0,978 | 0,713 / 0,773 | 0,907 / 0,929 |
| Code_Stop_UserService            | 0,964 / 0,968 | 0,664 / 0,715 | 0,894 / 0,894 |
| DB_Redis_CacheLimit_HomeTimeline | 0,637 / 0,949 | 0,373 / 0,603 | 0,444 / 0,833 |
| DB_Redis_CacheLimit_SocialGraph  | 0,973 / 0,973 | 0,736 / 0,736 | 0,900 / 0,900 |
| DB_Redis_CacheLimit_UserTimeline | 0,692 / 0,923 | 0,544 / 0,696 | 0,667 / 0,778 |
| Perf_CPU_Contention              | 0,903 / 0,966 | 0,642 / 0,768 | 0,667 / 0,833 |
| Perf_Disk_IO_Stress              | 0,901 / 0,939 | 0,579 / 0,641 | 0,667 / 0,727 |
| Perf_Network_Loss                | 0,625 / 0,977 | 0,508 / 0,906 | 0,571 / 0,909 |
| Svc_Kill_Media                   | 0,962 / 0,962 | 0,679 / 0,679 | 0,889 / 0,889 |
| Svc_Kill_SocialGraph             | 0,926 / 0,926 | 0,478 / 0,478 | 0,750 / 0,750 |
| Svc_Kill_UserTimeline            | 0,962 / 0,962 | 0,679 / 0,679 | 0,889 / 0,889 |
| **Trung bình**                   | 0,874 / 0,958 | 0,615 / 0,705 | 0,764 / 0,854 |

F1 oracle (quét ngưỡng theo nhãn test, có `point_adjust`) lạc quan và chỉ để so với các bài báo dùng cách quét; không phải số chính. AUROC/AUPRC không phụ thuộc ngưỡng nên thể hiện chất lượng của chính điểm số.

### 6.3 Cách đọc kết quả

- **Trace giúp nhiều nhất**: AUROC tăng từ 0,637 lên 0,949 (`DB_Redis_CacheLimit_HomeTimeline`), 0,692 lên 0,923 (`DB_Redis_CacheLimit_UserTimeline`), 0,625 lên 0,977 (`Perf_Network_Loss`) và 0,903 lên 0,966 (`Perf_CPU_Contention`): những scenario baseline yếu. Ở `Code_Stop_*` AUROC baseline đã 0,96–0,98 và trace chỉ thêm tối đa 0,004; mức tăng F1 ở đó (+0,09 trung bình) đến từ precision cao hơn tại cùng ngưỡng.
- **Điểm số giống nhau**: ở `Svc_Kill_*`, `DB_Redis_CacheLimit_SocialGraph` và `Code_Stop_MediaService`, baseline và trace có AUROC (gần như) bằng nhau, nghĩa là nhánh trace ít làm đổi thứ hạng ở đó; F1 khác nhau tùy vị trí ngưỡng.
- **File nhỏ**: `Svc_Kill_*` chỉ có 4 cửa sổ anomaly, tổng 32 cửa sổ, nên một cửa sổ làm F1 đổi hơn 0,1; xem các dòng này như giai thoại.
- **Một seed**: chênh lệch nhỏ (`Perf_Disk_IO_Stress` 0,552 → 0,615, mức tăng của `Code_Stop_*`) nằm trong nhiễu giữa các lần chạy (mục 7). Giao thức yêu cầu 3–5 seed trước khi kết luận; bảng này chỉ là kiểm chứng 1 seed.
- Phân tích tín hiệu theo từng đặc trưng và các ablation (epoch, `gate_delta_lr_mult`, `activity_penalty_weight`) chưa được chạy lại trên cách chia này nên không báo cáo.

## 7. Hạn chế

- **File test nhỏ**: 9/12 scenario chỉ có 32–80 cửa sổ; `Svc_Kill_*` chỉ có 4 cửa sổ anomaly. F1 còn phụ thuộc số cửa sổ normal và tỉ lệ anomaly của file test nên chỉ so sánh được trong cùng thiết lập này.
- **Một seed** (`run_end 1`); chênh lệch nhỏ nằm trong nhiễu.
- **Không tái lập hoàn toàn**: cùng code, dữ liệu, seed vẫn có thể ra khác nhau, nên mỗi số là một mẫu.
- **p95 chặn trên precision**: ngưỡng theo thiết kế báo động khoảng 5% điểm số normal của val, làm giảm precision khi tỉ lệ anomaly thấp.
- **KPI hệ thống mang dấu vết không đặc trưng cho lỗi**: `load1` cao ở đầu mọi session (khởi động stack) và `disk_usage_percent` tăng dần qua các session, tách anomaly khỏi normal với AUC 0,99–1,00 ở một số scenario. Baseline và trace đều nhận các KPI này nên so sánh vẫn công bằng, nhưng F1 tuyệt đối có thể được nâng bởi dấu vết thời gian/session chứ không chỉ do lỗi. Chưa chạy ablation bỏ các KPI này.
- **Lệch session**: train (`Normal_Baseline`) và normal của val/test đến từ các thời điểm ghi khác nhau; val được lấy từ cùng các session với normal của test theo chủ ý.
- Chỉ có một session `Normal_Baseline` thật nên tập train vẫn nhỏ (39 cửa sổ).
- `gate_delta_lr_mult` không có đối ứng ở phía baseline; đây vẫn là một điểm bất đối xứng giữa hai cấu hình.

## 8. File liên quan

| Nội dung                                                                                                      | File                                                 |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Giao thức chuẩn                                                                                               | `docs/evaluation_protocol_vi.md`                     |
| `FAULT_WINDOWS`, chia val/test, `target_anomaly_rate`, `max_anomalies`, clip `latency_dev`                    | `codes/common/preprocess_sn.py`                      |
| Chọn model theo val loss, ngưỡng val, AUROC/AUPRC, F1 oracle, `activity_penalty_weight`, `gate_delta_lr_mult` | `codes/models/basev3.py`, `codes/run.py`             |
| Wrapper và bảng tổng kết                                                                                      | `codes/common/eval_per_scenario_sn.py`               |
| Kết quả                                                                                                       | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
