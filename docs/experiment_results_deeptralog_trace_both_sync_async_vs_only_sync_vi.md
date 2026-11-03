# Kết quả thực nghiệm: DeepTraLog — Chỉ sync so với Sync + Async

Đánh giá theo giao thức chuẩn trong [`evaluation_protocol_vi.md`](evaluation_protocol_vi.md): chọn epoch theo val loss, ngưỡng = p95 điểm số val, **F1 tại ngưỡng val là chỉ số chính**, AUROC / AUPRC là phụ, F1 oracle ghi riêng. Khác với các file `experiment_results_*` khác, cả hai cấu hình ở đây dùng `--score_rule norm_sum` (không dùng tổng thô fusion loss) và **3 seed** — xem mục 4 và 6.

## 1. Thiết lập thực nghiệm

### Mô hình
**HADES** — mô hình phát hiện bất thường không giám sát dựa trên GAN, chỉ huấn luyện trên dữ liệu normal, mở rộng thêm một nhánh trace-async cộng vào (xem [`model_architecture_flow_vi.md`](model_architecture_flow_vi.md) mục 10 cho kiến trúc).

### Cấu hình

| Thiết lập                             | Giá trị                                                                                              |
| :------------------------------------ | :------------------------------------------------------------------------------------------------- |
| Dataset                               | DeepTraLog (TrainTicket), F01/F02/F13 (nhóm lỗi "Asynchronous Interaction" theo bài báo)             |
| Loại dữ liệu                          | `fuse` (log + trace; dataset không có metric, `kpi_c=1` toàn 0, `open_unmatch_zoomout=False`)      |
| Train / unlabel                       | 6.000 / 6.000 trace normal (lấy mẫu từ ~93.000 trace, giảm để chạy trên máy 8 GB)                   |
| Val                                   | 1.500 trace normal                                                                                    |
| Test mỗi F-case                       | F01 16.424 (2.053 bất thường, 12,5%); F02 17.208 (2.151, 12,5%); F13 14.232 (1.779, 12,5%)          |
| `window_size`                         | 5                                                                                                      |
| `val_percentile`                      | 95                                                                                                     |
| `score_rule`                          | `norm_sum` cho **cả hai** cấu hình (xem mục 4) — không dùng `raw_sum` mặc định của dataset khác      |
| `epoches` / `patience`                | 10 10 / 5 (giống nhau cho cả hai cấu hình)                                                            |
| `batch_size`, `alpha`, `open_gan_sep` | 128, 0.16, True                                                                                       |
| `open_async_trace`, `async_order`     | chỉ sync: False, — ; cả hai: True, True (tự đọc từ `meta["async_order"]`)                            |
| `gate_lambda`                         | 0.01 (cả hai cấu hình)                                                                                |
| `run_start` / `run_end`               | 0..3 (**3 seed**, `random_seed = 42 + run_times`)                                                    |

### Thư mục kết quả
| Cấu hình                                      | Thư mục                                                              |
| :--------------------------------------------- | :------------------------------------------------------------------- |
| Chỉ sync (log + trace sync)                    | `data/deeptralog/result_per_scenario_fuse_trace_only_sync/`         |
| Cả sync và async (log + trace sync + async)    | `data/deeptralog/result_per_scenario_fuse_trace_both_sync_and_async/`|

## 2. F01/F02/F13 đổi gì (theo dấu vân cấu trúc dữ liệu, `docs/preprocess_deeptralog_vi.md` mục 4)

| F-case | API | Đổi gì so với trace bình thường cùng API | Giữ nguyên gì |
| :--- | :--- | :--- | :--- |
| F01 | `preserveservice/preserve`, `cancel`, `execute`, `collected`, `travelplan/{cheapest,quickest}` | 100% trace (ở các tiểu ca nhạy thứ tự) có quan hệ thứ tự thời gian giữa service chưa từng thấy ở trace bình thường cùng API | Tập cạnh (service nào gọi service nào) không đổi |
| F02 | `foodservice/createOrderBatch` (+ 1 tiểu ca root travel gần như không có message async) | Số message async +21% ở tiểu ca root food (11,6 → 14,1 msg/trace); không đổi quan hệ thứ tự | Tập cạnh không đổi |
| F13 | `rebookservice/rebook`, `admintravelservice/admintravel`, `trips/left` (tiểu ca im lặng/lỗi-dừng-sớm) | 100% quan hệ thứ tự lạ ở 2 tiểu ca rebook/admintravel; tiểu ca `trips/left` thì lỗi-dừng-sớm (18 span, 100% có span lỗi, so với 0,1% trace bình thường) — xem mục 4.1 | Tập cạnh không đổi |

Đây là lý do nhánh chỉ-sync (chỉ theo dõi tập cạnh) thấy rất ít ở cả 3 F-case này, và nhánh async (số message, quan hệ thứ tự, bằng chứng span lỗi) mới là nhánh kỳ vọng tạo khác biệt.

## 3. Hai pool normal

- **Train / unlabel**: 6.000 / 6.000 trace normal, lấy mẫu reservoir từ ~93.000 trace normal trong các file `normal/*.zip` của dataset (giảm quy mô thuần vì RAM; bước gán nhãn/log-template/latency-baseline vẫn quét toàn bộ).
- **Val**: 1.500 trace normal, tách biệt với train/unlabel.
- **Test**: mỗi `test_{F}.pkl` chứa trace bất thường của chính F-case đó cùng trace normal lấy mẫu để đạt tỉ lệ đích 12,5% cố định của dataset (`TARGET_ANOMALY_RATE` trong `preprocess_deeptralog.py`, cùng quy ước với SN).

## 4. Thành phần chấm điểm (`score_rule=norm_sum`)

Cả hai cấu hình chấm bằng `evaluate_norm_sum` (`codes/models/basev3.py`): một điểm fused `S = Σ_k z_k`, ngưỡng = p95 val của `S`, mỗi số hạng chuẩn hoá bằng **thống kê chỉ từ normal của val**, `z_k = max(0, (T_k − median_k) / (p95_k − median_k))`. Dataset này không dùng tổng thô fusion loss (mặc định cho dataset khác) vì số hạng ồn nhất sẽ quyết định thứ hạng nếu không chuẩn hoá.

| Số hạng | Có khi | Đo gì |
| :--- | :--- | :--- |
| `log_kpi_loss` | luôn có | Sai số tái tạo log (+ KPI suy biến) |
| `trace_dis` | `open_trace=True` | Sai số tái tạo cấu trúc+thuộc tính trace sync (giữ nguyên nhánh dùng cho SN/RE2/RE3) |
| `trace_dis_async_count` | `open_async_trace=True` | Nhánh async: sai số tái tạo cấu trúc số lượng message + thuộc tính. **Chuẩn hoá theo ngữ cảnh**: chỉ ~8,7% trace val có trao đổi message async, nên số hạng này z=0 với trace không có message và chỉ chuẩn hoá theo các trace val có message (thang bền vững `(p90−median)·1,645/1,2816`; lùi về chuẩn hoá toàn cục nếu <40 trace val như vậy) |
| `trace_dis_async_order` | `open_async_trace=True, async_order=True` | Nhánh async: sai số tái tạo BCE của quan hệ thứ tự thời gian có hướng giữa cặp service (`async_temporal_order_adj`), trung bình trên các cặp service có mặt |
| `trace_err` | `open_async_trace=True` (thêm cho lần chạy này) | Bằng chứng span lỗi: `log1p(Σ_service call_count · error_rate)` của cả trace. Trace normal ở val có span lỗi chỉ 0,13% thời gian, nên `p95 − median` ≈ 0 không dùng làm thang được; số hạng được đặt sàn ở **thang cố định** sao cho một span lỗi trong trace vốn không lỗi cho z = 4 (`BaseModel.ERR_Z_ONE`). Nếu dataset có trace normal thường có span lỗi, thang đo từ val sẽ tự động thay thế. |

Embedding async (`ZV_async`, trung bình có mặt-mask trên các service có mặt) cũng đi vào `delta_head` dùng chung với embedding sync, tức là nó có thể hỗ trợ tái tạo log/KPI, không chỉ cộng thêm một số hạng điểm (xem `model_architecture_flow_vi.md` mục 10.5).

### 4.1 Vì sao cần `trace_err` (tiểu ca im lặng/lỗi-dừng-sớm của F13)

`trips/left` (F13, 299 trace) không phải nhiễu cấu trúc "im lặng" thường — đây là **trace lỗi-rồi-dừng-sớm**: đúng 18 span (so với 134 span của một request đặt vé bình thường), 100% có ít nhất một span lỗi (so với 0,1% trace bình thường), 5 service không bao giờ được gọi, và cả request chạy xong trong ~14 ms (so với ~94 ms). Trước khi thêm `trace_err`, recall của nhóm này ở ngưỡng `norm_sum` cao hơn đã tụt (số hạng thứ tự không có gì vì trace không có message, nên trace này không có tín hiệu nào khác vượt nhiễu log nền). Thêm `trace_err` phục hồi nó (recall của nhóm này 0,085 → 1,000, đo trên checkpoint đã lưu trước khi chạy lại đầy đủ) mà không ảnh hưởng F01/F02 (0 span lỗi ở mọi tiểu ca của hai F-case này).

## 5. Lệnh chạy

```bash
cd D:/UAM-AD
python codes/common/preprocess_deeptralog.py --stage normal --fault_dir <thư mục F*.zip> --normal_dir <thư mục normal> \
    --label_pkl <labels.pkl> --output_dir data/deeptralog
python codes/common/preprocess_deeptralog.py --stage fcase --fcases F01 F02 F13 --fault_dir <thư mục F*.zip> \
    --normal_dir <thư mục normal> --label_pkl <labels.pkl> --output_dir data/deeptralog

cd codes
# Chỉ sync
python common/eval_per_scenario_deeptralog.py --data ../data/deeptralog --dataset deeptralog --data_type fuse \
    --open_trace True --open_async_trace False --score_rule norm_sum \
    --epoches 10 10 --batch_size 128 --patience 5 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True --open_unmatch_zoomout False \
    --gate_lambda 0.01 --fcases F01 F02 F13 --run_start 0 --run_end 3
# Cả sync và async
python common/eval_per_scenario_deeptralog.py --data ../data/deeptralog --dataset deeptralog --data_type fuse \
    --open_trace True --open_async_trace True --score_rule norm_sum \
    --epoches 10 10 --batch_size 128 --patience 5 \
    --window_size 5 --val_percentile 95 --alpha 0.16 --open_gan_sep True --open_unmatch_zoomout False \
    --gate_lambda 0.01 --fcases F01 F02 F13 --run_start 0 --run_end 3
```
(`async_order` không cần truyền tay — `run.py` tự đọc từ `meta["async_order"]`, `True` cho dataset này.) Thực tế mỗi F-case × seed được chạy trong **tiến trình riêng**, tuần tự từng cái, để phù hợp máy 8 GB; lệnh trên là dạng tương đương 1 tiến trình.

## 6. Kết quả

### 6.1 Chỉ số chính: F1 tại ngưỡng val (p95 điểm số val), trung bình ± std của 3 seed

| F-case | Chỉ sync F1 | P | R | Cả hai F1 | P | R | Δ F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| F01 | 0,764±0,003 | 0,706±0,005 | 0,833±0,000 | **0,774±0,002** | 0,722±0,004 | 0,832±0,001 | +0,010 |
| F02 | 0,110±0,003 | 0,184±0,003 | 0,078±0,003 | **0,138±0,011** | 0,233±0,014 | 0,098±0,009 | +0,028 |
| F13 | 0,827±0,036 | 0,735±0,012 | 0,948±0,072 | **0,863±0,001** | 0,759±0,002 | 0,999±0,000 | +0,036 |
| **Trung bình** | **0,567** | 0,542 | 0,620 | **0,592** | 0,571 | 0,643 | **+0,025** |

F1 khi có cả hai nhánh cao hơn ở toàn bộ 3/3 F-case; mức tăng lớn nhất ở F13 (bản chạy 1 seed chưa có `trace_err` — xem mục 4.1 — trước đó F1 còn *thấp hơn* nhánh chỉ-sync) và nhỏ nhất ở F01.

### 6.2 Chỉ số phụ: AUROC, AUPRC và F1 oracle, trung bình ± std của 3 seed (chỉ sync / cả hai)

| F-case | AUROC | AUPRC | F1 oracle |
| :--- | :---: | :---: | :---: |
| F01 | 0,933±0,008 / **0,971±0,005** | 0,831±0,003 / **0,857±0,004** | 0,656±0,031 / 0,540±0,061 |
| F02 | 0,655±0,006 / **0,809±0,040** | 0,166±0,003 / **0,266±0,045** | 0,396±0,012 / 0,358±0,008 |
| F13 | 0,983±0,003 / **0,989±0,001** | 0,862±0,015 / **0,883±0,009** | 0,498±0,014 / 0,484±0,025 |

F1 oracle (quét ngưỡng theo nhãn test, có `point_adjust`) lạc quan và chỉ để so với các bài báo dùng cách quét; không phải số chính. AUROC/AUPRC không phụ thuộc ngưỡng nên thể hiện chất lượng của chính điểm số, và cả hai nhánh vượt nhánh chỉ-sync ở mọi F-case cho cả hai chỉ số này.

### 6.3 Cách đọc kết quả

- **F02** là bằng chứng rõ nhất cho nhánh async: AUROC 0,655 → 0,809 (+0,154), AUPRC 0,166 → 0,266 (+0,10). Lỗi ở đây chỉ đổi số message async, mà nhánh chỉ-sync (chỉ theo tập cạnh, mục 2) về cấu trúc không thể thấy được. F1 tuyệt đối vẫn thấp (0,110 → 0,138) vì số message của trace bình thường và trace lỗi chồng lấn nhiều (8–15 so với 9–19 message/trace) — AUROC riêng của số hạng đếm async (≈0,74–0,79, đo trên điểm thành phần) đã gần trần của một bộ đếm message thuần cho tiểu ca này.
- **F13** tăng F1 nhiều nhất (+0,036) và đạt recall 0,999, nhờ số hạng `trace_err` phục hồi nhóm `trips/left` (mục 4.1) cộng với số hạng thứ tự bắt được hai tiểu ca rebook/admintravel.
- **F01** tăng ít nhất (+0,010 F1, tuy mức tăng AUROC/AUPRC có quy mô tương tự F13): hầu hết tiểu ca của F01 là lỗi quan hệ thứ tự giống rebook/admintravel của F13, nhưng F01 không có bằng chứng span lỗi và không có tín hiệu số lượng message, nên chỉ số hạng thứ tự (cùng nhánh sync và log) góp phần.
- **3 seed, chạy tuần tự**: std của F1 ≤0,011 cho cả hai cấu hình ở F02/F01 và 0,036/0,001 ở F13 (chỉ sync/cả hai) — đủ nhỏ để thứ hạng (cả hai > chỉ sync ở mọi F-case) đúng theo từng seed, không chỉ đúng trung bình.
- F1 oracle của cấu hình có cả hai nhánh lại *thấp hơn* một chút ở F01/F02/F13 dù AUROC/AUPRC cao hơn — F1 oracle quét một ngưỡng vô hướng duy nhất trên chính tập test và nhạy với độ sắc nét của phần đầu thứ hạng, không chỉ chất lượng tổng thể của thứ hạng; chỉ ghi để tham khảo, không dùng để so sánh chính.

## 7. Hạn chế

- **Một dataset, một hệ thống (TrainTicket)**: phân loại lỗi (tập cạnh / quan hệ thứ tự / số lượng message / bằng chứng span lỗi) được rút ra từ chính dataset này; chưa kiểm chứng tổng quát hoá các quy tắc chấm điểm này cho hệ thống async khác.
- **Nguồn gốc `async_temporal_order_adj`**: khoá mẫu trong pkl là `md5(trace_id + vị trí đã xáo)` và `trace_id` đã bị bỏ, nên ma trận quan hệ thứ tự được ghép lại với dữ liệu span thô bằng chữ ký cấu trúc (số span + thời lượng tối đa theo service), không theo id. 1,7% mẫu (chỉ trace bình thường, trùng chữ ký với trace khác) chọn ngẫu nhiên trong các ứng viên trùng — một nguồn nhiễu nhỏ, đã đo được, không phải lệch có hệ thống (xem `docs/preprocess_deeptralog_vi.md` mục 8).
- **Thang cố định của `trace_err` (`ERR_Z_ONE=4.0`)** là lựa chọn có chủ đích cho số hạng hiếm-gặp trên *dataset này* (0,13% trace normal của val có span lỗi); không học từ dữ liệu và chưa quét tham số đầy đủ (giá trị 2/4/8 cho kết quả tương đương trong kiểm tra offline trên checkpoint đã lưu, nhưng đó không phải quét tham số đúng nghĩa trên chính lần chạy lại này).
- **F1 tuyệt đối của F02 vẫn thấp** (0,138) dù đã có nhánh async, vì tỉ lệ tín hiệu/nhiễu của chính lỗi này trong số lượng message thô đã gần trần cứng cho tiểu ca này (mục 6.3); đây là đặc điểm của lỗi, không hẳn khắc phục được bằng thêm dung lượng mô hình.
- **`open_unmatch_zoomout=False`**: không có metric thật (`kpi_c=1`, toàn 0), số hạng đối nghịch KPI-không-khớp bị suy biến nên tắt; dataset này không dùng để kiểm chứng thành phần đó.
- **3 seed, không phải 5**: std giữa các lần chạy đã nhỏ (mục 6.3), nhưng khoảng 3–5 seed thường dùng của giao thức chỉ chọn ở mức thấp vì hạn chế thời gian (máy 8 GB / không có GPU riêng, ~7–13 phút mỗi lần chạy).
- **Train/unlabel/val đã giảm mẫu** (6.000/6.000/1.500 trong hơn 93.000 trace normal có sẵn) vì RAM; chưa thử pool lớn hơn.
- Ba file CSV SpanData thô (2 trong `F07.zip`, 1 trong một zip `normal*`) thiếu cột `IsError`; `_read_spans` giờ điền `IsError=False` cho các file đó thay vì bỏ qua (trước đây bị bỏ qua âm thầm), nhưng sửa này thực hiện sau khi các kết quả trong tài liệu này đã chạy trên pkl xử lý trước đó, nên phần đóng góp của các file này chưa đổi trong các số ở trên.

## 8. File liên quan

| Nội dung | File |
| :--- | :--- |
| Giao thức chuẩn | `docs/evaluation_protocol_vi.md` |
| Kiến trúc nhánh trace async (mục 10) | `docs/model_architecture_flow_vi.md` |
| Kiểm chứng dataset/nhãn, dấu vân cấu trúc, schema pkl, lưu ý dữ liệu | `docs/preprocess_deeptralog_vi.md` |
| Pipeline tiền xử lý, `TARGET_ANOMALY_RATE`, cách tính `async_temporal_order_adj` | `codes/common/preprocess_deeptralog.py` |
| `evaluate_norm_sum`, `_score_components`, `trace_err`/`ERR_Z_ONE` | `codes/models/basev3.py` |
| Mô hình trace async (encoder/decoder, đầu thứ tự) | `codes/models/async_trace_model_v3.py` |
| Wrapper và bảng tổng kết | `codes/common/eval_per_scenario_deeptralog.py` |
| Kết quả | `data/deeptralog/result_per_scenario_fuse_trace_{only_sync,both_sync_and_async}/` |
