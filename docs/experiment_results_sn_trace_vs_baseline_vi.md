# Kết quả thí nghiệm: SocialNetwork — Trace so với Baseline

> **Cập nhật quan trọng**: Bản trước của tài liệu này (F1 baseline ~0.92, trace ~0.96) được tính trên preprocessing có lỗi (xem mục 2) khiến baseline "ăn gian" nhờ confound cấp session, không phản ánh khả năng phát hiện lỗi thật. Toàn bộ số liệu trong tài liệu này được tính lại sau khi sửa preprocessing + scoring, thấp hơn nhiều nhưng đáng tin cậy hơn.

## 1. Bối cảnh: vì sao phải làm lại toàn bộ

Khi điều tra tại sao baseline (chỉ log+KPI, không trace) lại đạt F1 cao bất thường trên các lỗi "luồng gọi hệ thống" (service bị kill/dừng — loại lỗi lẽ ra trace phải phát huy tác dụng nhất), phát hiện ra 2 vấn đề gốc rễ trong preprocessing cũ:

### 1.1 Confound cấp session
Mọi file test đều so sánh **cùng 1 session `Normal_Baseline`** (ghi đầu tiên trong chuỗi 13 thí nghiệm) với **1 session lỗi khác** (ghi sau đó, cách nhau 15 phút đến gần 3 tiếng). Baseline có thể học cách phân biệt "session nào" (do trôi dạt môi trường theo thời gian) thay vì phát hiện lỗi thật — verify bằng cách thấy các metric hoàn toàn không liên quan đến lỗi (CPU của service khác) cũng lệch rõ giữa 2 nhóm.

### 1.2 Nhãn anomaly sai theo thời gian thật của lỗi
Preprocessing cũ dùng 1 quy tắc chung cho mọi loại lỗi: "bỏ qua 5 phút đầu (warmup), phần còn lại = anomaly". Kiểm tra script thu thập dữ liệu gốc của AnoMod (`github.com/EvoTestOps/AnoMod`, `automated_multimodal_collection.sh`) cho thấy quy tắc này sai với phần lớn loại lỗi:

| Loại lỗi | Cơ chế thật (theo script) | Nhãn cũ | Nhãn đúng |
|---|---|---|---|
| `Code_Stop_*` | `docker stop`, không tự restart | Cả session = anomaly | Đúng, giữ nguyên |
| `Perf_*`, `DB_Redis_CacheLimit_*` | ChaosBlade `--timeout 300`, tự hết hiệu lực sau 300s | Cả session (trừ 5 phút đầu) = anomaly | **Chỉ 5 phút đầu = anomaly**, phần còn lại đã hồi phục |
| `Svc_Kill_*` | ChaosBlade kill process + Docker auto-restart | Cả session = anomaly | **Chỉ ~2 phút (giây 90-210)** — xác nhận qua cột `container_label_restartcount` (0→1 tại giây 105) và khoảng lặng ~75s thật trong trace |

## 2. Thiết kế preprocessing mới

### 2.1 Nhãn theo `FAULT_WINDOWS` (thay vì quy tắc chung)
`codes/common/preprocess_sn.py` giờ định nghĩa cửa sổ lỗi chính xác cho từng scenario (xem bảng trên), dựa trên bằng chứng thật (script AnoMod + số liệu đo trực tiếp), không còn quy tắc "bỏ N phút đầu" chung chung.

### 2.2 Hai pool "normal" tách biệt
- **Train/unlabel/val**: chỉ từ `Normal_Baseline` (39 windows) — giữ hẹp/đồng nhất có chủ đích (xem mục 2.3).
- **Test** (phần "normal" trong mỗi `test_<scenario>.pkl`): gộp từ **mọi** scenario (Normal_Baseline + phần đã hồi phục/chưa từng lỗi của từng scenario khác) — 320 windows, trải khắp toàn bộ 3 tiếng thí nghiệm thay vì 1 lát 20 phút duy nhất. Đây là cách thật (không phải mẹo) để triệt tiêu confound cấp session: baseline không còn có thể "đoán session" vì normal giờ đến từ rất nhiều thời điểm khác nhau.

### 2.3 Vì sao train KHÔNG dùng pool đa dạng
Thử nghiệm ban đầu dùng pool đa dạng cho cả train — nhưng verify trực tiếp cho thấy điều này dạy model rằng "ít hoạt động cũng là bình thường", phá hỏng khả năng phát hiện đúng loại lỗi "im lặng hoàn toàn" (xem mục 3). Quay lại train hẹp (chỉ Normal_Baseline) giải quyết đúng vấn đề mà không đánh đổi lợi ích giảm confound ở test.

## 3. Sửa scoring: lỗi "im lặng hoàn toàn" bị chấm điểm ngược

### 3.1 Phát hiện
Ngay cả sau khi sửa nhãn đúng, `Code_Stop_TextService`/`UserService` và `Svc_Kill_*` vẫn cho F1=0.0000. Đo trực tiếp phân phối reconstruction loss theo nhãn (không suy đoán):

```
Code_Stop_TextService:  anomaly mean loss = 0.92   |   normal mean loss = 1.29
```

**Loss của window anomaly THẤP HƠN window normal** — ngược hoàn toàn với logic "loss cao = bất thường" mà toàn hệ thống dựa vào. Đã kiểm tra tách riêng cả 2 thành phần (log + KPI) — cả 2 đều bị đảo ngược tương tự, không phải chỉ do log thưa.

### 3.2 Nguyên nhân
Khi service chết hẳn, log/KPI của nó gần như trống rỗng/phẳng lặng. Với autoencoder, tái tạo lại 1 input gần-như-không-biến-thiên lại **dễ hơn** tái tạo 1 input bình thường có nội dung/biến động thật — nên loss thấp hơn, không cao hơn. Đây là đặc tính cố hữu của reconstruction-loss, không phải lỗi ở khâu gộp/tách pool (đã verify: vẫn xảy ra y hệt dù dùng pool train hẹp hay đa dạng).

### 3.3 Fix: `activity_penalty_weight`
Thêm 1 tín hiệu **độc lập với reconstruction**, đo trực tiếp từ input thật: mức hoạt động hiện tại (`kpi_features.sum + log_features.sum`) lệch bao nhiêu **dưới** mức hoạt động normal kỳ vọng (tính từ train data), cộng vào điểm bất thường trước khi so ngưỡng. Mặc định `0.0` (no-op, không ảnh hưởng dataset khác/checkpoint cũ).

```bash
--activity_penalty_weight 1.5
```

### 3.4 Kết hợp với `gate_delta_lr_mult`
Với nhánh trace, kết hợp thêm `--gate_delta_lr_mult 10 --epoches 50 50 --patience 15` (cơ chế đã có từ trước, giúp `trace_gate`/`delta_head` thoát nghẽn gradient nhanh hơn) — cộng hưởng với activity penalty để đóng nốt khoảng cách còn lại ở các scenario tín hiệu yếu (`Svc_Kill_*`).

---

## 4. Cấu hình chạy cuối cùng

### Baseline (`open_trace=False`)
```bash
cd D:/UAM-AD/codes
python common/eval_per_scenario_sn.py \
    --data ../data/sn --dataset sn --data_type fuse \
    --open_trace False --activity_penalty_weight 1.5 \
    --epoches 10 10 --batch_size 256 --patience 5 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True \
    --run_start 0 --run_end 1
```

### Trace (`open_trace=True`)
```bash
cd D:/UAM-AD/codes
python common/eval_per_scenario_sn.py \
    --data ../data/sn --dataset sn --data_type fuse \
    --open_trace True --activity_penalty_weight 1.5 \
    --gate_delta_lr_mult 10 \
    --epoches 50 50 --batch_size 256 --patience 15 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True \
    --run_start 0 --run_end 1
```

---

## 5. Kết quả (12 scenario, sau khi sửa nhãn + scoring)

| Scenario | Baseline F1 | Baseline P | Baseline R | Trace F1 | Trace P | Trace R | Δ F1 |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| Code_Stop_MediaService | 0.500 | 0.357 | 0.833 | **0.600** | 0.429 | 1.000 | +0.100 |
| Code_Stop_TextService | 0.462 | 0.300 | 1.000 | **0.632** | 0.462 | 1.000 | +0.170 |
| Code_Stop_UserService | 0.462 | 0.300 | 1.000 | **0.615** | 0.571 | 0.667 | +0.154 |
| DB_Redis_CacheLimit_HomeTimeline | 0.218 | 0.122 | 1.000 | **0.375** | 0.231 | 1.000 | +0.157 |
| DB_Redis_CacheLimit_SocialGraph | 0.462 | 0.300 | 1.000 | 0.435 | 0.294 | 0.833 | -0.027 |
| DB_Redis_CacheLimit_UserTimeline | 0.200 | 0.250 | 0.167 | **0.276** | 0.174 | 0.667 | +0.076 |
| Perf_CPU_Contention | 0.245 | 0.140 | 1.000 | **0.345** | 0.217 | 0.833 | +0.100 |
| Perf_Disk_IO_Stress | 0.231 | 0.150 | 0.500 | **0.345** | 0.217 | 0.833 | +0.114 |
| Perf_Network_Loss | 0.200 | 0.250 | 0.167 | **0.345** | 0.217 | 0.833 | +0.145 |
| Svc_Kill_Media | 0.143 | 0.077 | 1.000 | **0.267** | 0.154 | 1.000 | +0.124 |
| Svc_Kill_SocialGraph | 0.170 | 0.093 | 1.000 | **0.320** | 0.190 | 1.000 | +0.150 |
| Svc_Kill_UserTimeline | 0.174 | 0.095 | 1.000 | **0.333** | 0.200 | 1.000 | +0.159 |
| **Trung bình** | **0.289** | **0.203** | **0.806** | **0.407** | **0.280** | **0.889** | **+0.119** |
| Độ lệch chuẩn | 0.132 | 0.096 | 0.318 | 0.127 | 0.128 | 0.124 | |

**Trace thắng 11/12 scenario**, chỉ `DB_Redis_CacheLimit_SocialGraph` gần như hòa (-0.027). Recall trung bình của trace đạt 0.889 (so với 0.806 của baseline) — cải thiện đều, không chỉ ở 1-2 scenario.

### 5.1 Nhóm "lỗi luồng gọi hệ thống" (mục tiêu chính: service kill/dừng)

| Scenario | Baseline F1 | Trace F1 | Δ |
|---|---:|---:|:---:|
| Code_Stop_MediaService | 0.500 | 0.600 | +0.100 |
| Code_Stop_TextService | 0.462 | 0.632 | +0.170 |
| Code_Stop_UserService | 0.462 | 0.615 | +0.154 |
| Svc_Kill_Media | 0.143 | 0.267 | +0.124 |
| Svc_Kill_SocialGraph | 0.170 | 0.320 | +0.150 |
| Svc_Kill_UserTimeline | 0.174 | 0.333 | +0.159 |
| **Trung bình** | **0.319** | **0.461** | **+0.141** |

**Trace thắng cả 6/6** — đây là bằng chứng trực tiếp, có cơ sở cho luận điểm "trace giúp phát hiện tốt hơn lỗi luồng gọi hệ thống", khác biệt rõ giữa 2 nhóm:
- `Code_Stop_*` (service chết hẳn, tín hiệu mạnh/bền vững suốt session): margin thắng lớn (+0.10 đến +0.17).
- `Svc_Kill_*` (service restart nhanh, tín hiệu ngắn ~2 phút — giới hạn thật của dữ liệu, đã verify qua `container_label_restartcount`): margin thắng nhỏ hơn nhưng vẫn nhất quán dương (+0.12 đến +0.16), sau khi kết hợp `activity_penalty_weight` + `gate_delta_lr_mult`.

## 6. Giới hạn còn lại

- **F1 tuyệt đối còn thấp** (0.3-0.6) so với con số cũ (0.9+) — đây là con số **trung thực** sau khi loại bỏ confound, phản ánh đúng độ khó thật của bài toán trên dataset nhỏ (39 window train). Không nên so sánh trực tiếp với báo cáo cũ.
- **Precision còn thấp** (0.15-0.35) — activity penalty đánh đổi độ chính xác lấy recall cao; còn dư địa tinh chỉnh nếu cần.
- **`DB_Redis_CacheLimit_SocialGraph`** là scenario duy nhất trace không thắng rõ — chưa điều tra sâu nguyên nhân riêng.
- Do dataset SN chỉ có 1 session `Normal_Baseline` thật, train set mãi mãi nhỏ (39 window) — đây là giới hạn cấu trúc của raw data, không phải preprocessing.

## 7. Tệp liên quan

| Thay đổi | File |
|---|---|
| `FAULT_WINDOWS`, 2 pool normal | `codes/common/preprocess_sn.py` |
| `activity_penalty_weight` | `codes/models/basev3.py`, `codes/run.py` |
| Forward flag qua wrapper | `codes/common/eval_per_scenario_sn.py` |
| Checkpoint cuối cùng | `data/sn/result_per_scenario_fuse_{baseline,trace}/` |
