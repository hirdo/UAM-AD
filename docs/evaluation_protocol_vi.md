# Giao thức đánh giá chuẩn (SN, RE2-OB, RE3-OB)

Một giao thức cho mọi dataset. Không bước nào dùng nhãn test.

## 1. Các bước

1. **Chọn model/epoch**: early stopping theo điểm số trung bình (fusion loss, cộng activity penalty nếu bật) trên tập **val** (chỉ normal). Không dùng test hay nhãn test để chọn epoch.
2. **Ngưỡng**: **percentile 95** của điểm số val normal của model đã chọn (`--val_percentile 95`, giống nhau cho mọi dataset).
3. **Chỉ số chính**: precision / recall / **F1 tại ngưỡng val** (không `point_adjust`). Ghi vào `info_score.txt` với khóa `f1`, `pc`, `rc`.
4. **Chỉ số phụ**: **AUROC** và **AUPRC** (`auroc`, `auprc`). Không phụ thuộc ngưỡng nên tách được chất lượng điểm số khỏi chất lượng hiệu chỉnh ngưỡng.
5. **F1 oracle** (`oracle_f1`, `oracle_pc`, `oracle_rc`, `oracle_threshold`): quét ngưỡng trên điểm số test (top `--anomaly_rate` phần trăm, có `point_adjust`). Dùng nhãn test nên lạc quan; chỉ ghi ở trường riêng để so với các bài báo dùng cách quét (UAC-AD, TraceDAE), không bao giờ là số chính.
6. **Số seed**: báo cáo **3–5 seed** (trung bình ± độ lệch chuẩn). Lần kiểm chứng nhanh có thể chạy 1 seed; khi đó chênh lệch nhỏ so với nhiễu giữa các lần chạy xem là hòa.

## 2. Vì sao không dùng oracle làm số chính

Oracle chọn ngưỡng (và trước đây cả epoch) theo nhãn test. Trên SN, oracle cho baseline 0.745 / trace 0.883, còn ở ngưỡng val là 0.407 / 0.355. Số cần nhãn test thì không tái lập được khi triển khai, và che mất sai số hiệu chỉnh ngưỡng, thứ mà người vận hành thực sự gặp.

## 3. Quy tắc chia dữ liệu

| Dataset | train | val (chỉ normal) | test |
|:--|:--|:--|:--|
| SN | toàn bộ 39 cửa sổ `Normal_Baseline` | 20% cửa sổ normal của mỗi session khác (57 cửa sổ → (57−5)×5 = 260 điểm số) | mỗi scenario: cửa sổ anomaly + normal lấy từ 80% còn lại (pool 224 cửa sổ) |
| RE2-OB / RE3-OB | `unlabel.pkl` / `train.pkl` (80% / 20% cửa sổ normal trước inject) | **chưa có** (`preprocess_rcaeval_re{2,3}_ob.py` không ghi `val.pkl`); code dùng normal của unlabel (model đã train trên đó) nên ngưỡng quá thấp và F1 chưa so được với SN cho đến khi thêm tập val riêng | dữ liệu sau inject có nhãn |

Val lấy từ cùng hỗn hợp session với normal của test và không trùng với chúng. Val phải đủ lớn để p95 ổn định: với 8 cửa sổ (15 điểm số) một cửa sổ ngoại lai quyết định ngưỡng (bootstrap 1.81–3.81); với 57 cửa sổ độ lệch nhỏ hơn nhiều.

### %anomaly từng scenario của SN

Số normal mỗi file test = anomaly × (1 − r) / r, bị giới hạn bởi pool 224 cửa sổ. Anomaly của `Code_Stop_*` được lấy đều theo thời gian xuống 39.

| Scenario | Anomaly | Normal | Tổng | Tỉ lệ |
|:--|--:|--:|--:|--:|
| Code_Stop_MediaService | 39 | 224 | 263 | 14.8% |
| Code_Stop_TextService | 39 | 224 | 263 | 14.8% |
| Code_Stop_UserService | 39 | 224 | 263 | 14.8% |
| DB_Redis_CacheLimit_{HomeTimeline, SocialGraph, UserTimeline} | 10 | 70 | 80 | 12.5% |
| Perf_{CPU_Contention, Disk_IO_Stress, Network_Loss} | 10 | 70 | 80 | 12.5% |
| Svc_Kill_{Media, SocialGraph, UserTimeline} | 4 | 28 | 32 | 12.5% |

## 4. Giới hạn

- `Svc_Kill_*` chỉ có 4 cửa sổ anomaly nên F1 rất nhiễu với mọi ngưỡng.
- Huấn luyện không tái lập từng bit: cùng code và seed vẫn có thể ra khác nhau, nên mọi kết luận cần nhiều seed.
- Val lấy từ session khác train (thời điểm ghi khác), nên lệch session cũng làm dịch điểm số val. Chủ ý như vậy để ngưỡng đại diện cho normal của test.
- p95 cố định tỉ lệ dương tính giả kỳ vọng khoảng 5% điểm số normal; với tỉ lệ anomaly thấp, điều này chặn trên độ chính xác.
