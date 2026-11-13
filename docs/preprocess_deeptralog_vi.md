# Tiền xử lý: Tập dữ liệu DeepTraLog (TrainTicket)

## 1. Tổng quan tập dữ liệu

Tập dữ liệu DeepTraLog (`github.com/FudanSELab/DeepTraLog`, ICSE 2022) thu thập từ hệ thống TrainTicket, dùng để trả lời câu hỏi: mô hình có xử lý được luồng gọi **đồng bộ (sync)** và **bất đồng bộ (async)** trong trace hay không. Đây là bằng chứng chính cho phần "async" của mô hình vì hai lý do đã kiểm chứng (xem `docs/experiment_results_*`, hội thoại thiết kế): các dataset TrainTicket khác (Eadro, RCAEval RE2/RE3-TT, TrainTicketTrace) không có lỗi async thật; DeepTraLog thì có.

| Phương thức | Nguồn | Vị trí trên GitHub |
|:---|:---|:---|
| Trace (span) | SkyWalking-style CSV, 1 dòng = 1 span | `TraceLogData/F01.zip` .. `F14.zip`, `TraceLogData/normal/*.zip` |
| Log | Log thô + log đã parse (Drain3) | cùng zip, các file `.log` |
| Metric | **Không có** | — |

Không có metric — khi chạy log+trace phải tắt/giả lập nhánh KPI (xem kế hoạch, mục "Chạy log+trace khi không có metric").

**14 fault case** theo Table 1 của bài báo, chia 4 nhóm:

| Nhóm (theo bài báo) | Fault case | Ý nghĩa |
|:---|:---|:---|
| **Asynchronous Interaction** | **F1, F2, F13** | Lỗi ở cơ chế gửi/nhận message bất đồng bộ |
| Multi-Instance | F8, F11, F12 | Nhiều instance của cùng service không đồng bộ trạng thái |
| Configuration | F3, F4, F5, F7 | Cấu hình sai/không nhất quán |
| Monolithic | F6, F9, F10, F14 | Lỗi tính toán/logic nội bộ 1 service |

**Chỉ nhóm Asynchronous Interaction (F01, F02, F13) là bằng chứng async đáng tin cậy** — ba nhóm còn lại không đồng nghĩa "lỗi sync", chỉ đơn giản là không nhắm riêng vào cơ chế async (xem mục 3).

---

## 2. Định nghĩa: cạnh sync/async, lỗi sync/async

### 2.1 Vì sao cần định nghĩa lại

Mọi trace (kể cả trace bình thường) đều là hỗn hợp sync + async — TrainTicket dùng RabbitMQ giữa `ts-food-service` → `ts-delivery-service`, và `SpringAsync` trong `ts-cancel-service`, phần còn lại là gọi HTTP đồng bộ. Vì vậy "trace async" không tồn tại như một khái niệm tách biệt; cái cần phân biệt là **cạnh nào trong trace là sync, cạnh nào là async**, và **lỗi nhắm vào phần nào**.

### 2.2 Đồ thị mức-service của một trace

Mỗi trace được gộp thành đồ thị: node = service, cạnh = `(A, B, kind)` nghĩa là "A gọi B", `kind ∈ {S, A}`. Quy tắc gán `kind` cho cạnh ứng với span con của A do B xử lý:

1. `Component ∈ {rabbitmq-producer, rabbitmq-consumer}` → **A** (gửi/nhận message tường minh)
2. `Component = SpringAsync` → **A** (framework đánh dấu tác vụ nền)
3. Span con **bắt đầu sau khi span cha đã kết thúc** → **A**. Đây là dấu hiệu vật lý: gọi đồng bộ thì cha phải *chờ* con, nên khoảng thời gian của con luôn nằm lọt trong khoảng thời gian của cha; nếu con bắt đầu sau khi cha đã xong, cha không thể đang chờ — chỉ có thể là "gọi xong rồi bỏ đi" (fire-and-forget).
4. Không rơi vào 3 trường hợp trên → **S** (mặc định).

Ví dụ minh hoạ (rút từ F02, API `foodservice/createOrderBatch`):

```
ts-preserve-other-service  |----- HTTP POST /foodservice/orders ---------------|
ts-food-service                    |--createFoodOrder--|--(gửi message)--|
ts-delivery-service                                          |--nhận message, lưu delivery--|
```
`preserve-other-service → food-service`: cha đợi con → **S**.
`food-service → delivery-service`: con bắt đầu sau khi cha (food-service) đã đi tiếp việc khác → **A**.

### 2.3 "Cạnh lạ" (unseen edge)

Với một API cho trước, gộp tất cả trace **bình thường** cùng API lại thành một **tập cạnh nền** (edge baseline). Một cạnh `(A, B, kind)` của một trace lỗi là **"lạ"** nếu nó không nằm trong tập cạnh nền đó — tức lỗi làm xuất hiện một kết nối service-tới-service (hoặc đổi kind của kết nối cũ) mà bình thường không có.

*Giới hạn hiện tại:* mới đo chiều "cạnh mới xuất hiện", chưa đo chiều "cạnh biến mất". Sẽ bổ sung khi viết preprocessor thật (Bước 3+).

### 2.4 Định nghĩa lỗi sync vs lỗi async (thực nghiệm, tự đặt ra)

Bài báo (Table 1) không có nhãn "Synchronous Interaction" — chỉ có 4 nhóm ở mục 1. Từ đó, ta tự định nghĩa dựa trên đại lượng nào của trace bị lỗi làm thay đổi:

| Loại lỗi (tự đặt) | Biểu hiện đo được | Ví dụ |
|:---|:---|:---|
| **Lỗi cấu trúc (sync-type)** | Tập cạnh đổi: trace lỗi có cạnh chưa từng thấy ở normal cùng API | F04: 100% trace có cạnh lạ |
| **Lỗi async (đúng Table 1)** | Tập cạnh **không đổi**, nhưng **số lượng** cạnh async hoặc **thứ tự thời gian tương đối** giữa các service đổi | F02: cạnh không đổi, số message +21%. F01/F13: cạnh không đổi, 100% trace có quan hệ thứ tự chưa từng thấy |

Quan hệ "thứ tự thời gian tương đối" giữa 2 service A, B trong 1 trace: nếu mọi span của A kết thúc trước khi span đầu tiên của B bắt đầu → `A < B`; ngược lại → `A > B`; nếu chồng lấn (gọi song song) thì không ghi quan hệ. Tập tất cả quan hệ này của 1 trace là "chữ ký thứ tự" (order signature).

---

## 3. Kết quả xác minh nhãn (Bước 1, 2026-09-22)

`GraphData/` (7 phần z01–z07 + zip, dùng để huấn luyện GGNN gốc của DeepTraLog) là split-zip lỗi offset ở central directory — `unzip` chuẩn đọc sai entry thứ 6 trở đi. Đã tự tìm local header thật bằng chữ ký `PK\x03\x04` + tên file, giải nén bằng `zlib.decompress(..., -15)`.

- Đúng **132.485 trace**, **23.334 anomaly (17,6%)** — khớp số trong bài báo.
- Join `TraceId` (SpanData CSV) ↔ `trace_id` (GraphData) cho 14 file `F01.zip`..`F14.zip`: **12/14 khớp 100%**; F07 khớp 84,0%, F08 khớp 68,7% (phần thiếu nằm ở thư mục con `back0729`, không có trong snapshot GraphData).
- **Nhãn thật lấy từ `error_trace_type` của GraphData, không phải tên file zip.** Tên file zip là số thứ tự chính thức theo bài báo, nhưng nội bộ đôi khi đánh số khác: F06.zip↔nội bộ "F23", F09.zip↔"F24", F10.zip↔"F25" (đổi số thuần tuý). **F12.zip lẫn nhãn:** 1.174/1.472 trace nội bộ là "F12", 298 trace còn lại mang nhãn "F13" — khi dùng F12 phải lọc theo `error_trace_type` từng trace, không gộp cả file.
- **Nhóm async (F01, F02, F13) khớp tên-file = nhãn-nội-bộ 100%**, không bị đổi số.

---

## 4. Kết quả dấu vân cấu trúc theo F-case (Bước 2, 2026-09-22)

Dựng baseline cấu trúc (tập cạnh, tập quan hệ thứ tự, số span trung bình) cho **22 API** từ 9 file `normal/*.zip` (~93.000 trace bình thường), so từng trace của 14 F-case với baseline cùng API.

**Nhóm async (F01, F02, F13): tập cạnh không đổi, thứ tự/số lượng đổi mạnh — nhất quán:**

| F-case (API) | n trace | % cạnh lạ | % thứ tự lạ | lệch số span |
|:---|--:|--:|--:|--:|
| F01 (`preserveservice/preserve`) | 400 | 0% | **100%** | +2% |
| F02 (`foodservice/createOrderBatch`) | 1.199 | 0% | 0% | **+21%** (khớp số đo tay: 14,1 vs 11,6 msg/trace) |
| F13 (`rebookservice/rebook`) | 377 | 0% | **100%** | +1% |
| F13 (`admintravelservice/admintravel`) | 350 | 0% | **100%** | +3% |

**Đối chứng (lỗi cấu trúc thật):** F04 (`preserveservice/preserve`, n=330): **100% cạnh lạ** — khác hẳn nhóm async.

**Ý nghĩa cho thiết kế mô hình:** tín hiệu lỗi của F01/F02/F13 nằm hoàn toàn ở hai đại lượng (thứ tự, số lượng) mà nhánh async của mô hình được thiết kế để đo (decoder MSE cho count, kênh thứ tự k2); nhánh sync (chỉ theo dõi cạnh) sẽ không thấy gì bất thường ở các trace này — đúng ý đồ kiến trúc hai nhánh tách biệt (không cross-talk).

**Caveat:**
1. Một số API phụ (admin/config) cho `span_dev%` tròn số bất thường (-50,0%, -48,4%, -25,0%...) — nghi do các request "khởi động" chung giữa nhiều loại test, không đặc trưng riêng cho F-case đó. Không dùng các dòng này để kết luận.
2. F12.zip lẫn 298 trace nhãn F13 (mục 3) nên dòng "F12 → 66% thứ tự lạ" có thể bị nhiễm bởi phần async trộn vào — chưa tách được vì phép đo này chạy theo API chứ không lọc theo `error_trace_type` từng trace.
3. F09, F10, F14 không có API trùng normal đầy đủ nên nhãn thực nghiệm chỉ mang tính tham khảo.

Dữ liệu trung gian dùng để tính các bảng trên nằm ngoài repo (`D:\ClaudeWork\dtl\`), không commit.

---

## 5. F-case dùng cho thực nghiệm

| Mục đích | F-case | Ghi chú |
|:---|:---|:---|
| Bằng chứng chính cho lỗi async | **F01, F02, F13** | Khớp cả tên bài báo (Table 1) lẫn số đo cấu trúc |
| Ví dụ đối chứng cho lỗi cấu trúc (sync) | **F04** | 100% trace có cạnh lạ, sạch nhất |
| Đối chứng phụ (cấu trúc đổi, có caveat) | F03, F10, F14 | Edge lạ 34–77%, nhưng API con nhỏ hoặc thiếu normal đầy đủ |
| Không dùng để phân loại sync/async | F05, F06, F07, F08, F09, F11, F12 | Tín hiệu hỗn hợp hoặc quá nhiễu để kết luận chắc |

## 6. Pipeline: một điểm vào duy nhất, `codes/common/preprocess_deeptralog.py`

Giống `preprocess_sn.py`, mọi thứ nằm trong một script có công tắc `--stage` (còn `eval_per_scenario_deeptralog.py` chạy đánh giá theo F-case):

| Stage | Việc làm | Đầu ra |
|:---|:---|:---|
| `labels` | Nhãn từng trace từ kho GraphData của DeepTraLog (zip chia nhỏ bị hỏng central directory nên đọc bằng cách quét local header và giải nén raw deflate) | `--label_pkl` = `{trace_id: (trace_bool, error_trace_type)}`, `True` = bình thường; 132.485 trace, 23.334 (17,6%) bất thường |
| `normal` | Kho trace bình thường: template Drain3, baseline độ trễ, mẫu `train`/`unlabel`/`val` (reservoir sampling), kèm cache cho `fcase` | `train.pkl`, `unlabel.pkl`, `val.pkl`, `meta.pkl`, `_cache.pkl` |
| `fcase` | Một F-case so với cache đó: trace bất thường của nó + trace bình thường cho đủ 12,5% bất thường, xáo trộn | `scenarios/test_{F}.pkl` |
| `all` | `normal` + `fcase` trong một tiến trình (chỉ để chạy thử) | |

```
python codes/common/preprocess_deeptralog.py --stage labels --graphdata_dir <thư mục GraphData> --label_pkl <labels.pkl>
python codes/common/preprocess_deeptralog.py --stage normal --fault_dir <thư mục F*.zip> --normal_dir <thư mục normal> --label_pkl <labels.pkl> --output_dir data/deeptralog
python codes/common/preprocess_deeptralog.py --stage fcase  --fcases F01 --fault_dir ... --normal_dir ... --label_pkl ... --output_dir data/deeptralog
```
Chạy mỗi stage (và mỗi F-case) trong một tiến trình riêng: trên máy 8 GB, một tiến trình sống lâu vẫn bị dừng vì thiếu RAM dù đã đọc dạng streaming. Kết quả đánh giá nằm ở `data/deeptralog/result_per_scenario_*` ngay trong thư mục dataset, như SN.

## 7. Schema pkl (1 step = 1 trace)

| Khoá | Kích thước | Ý nghĩa |
|:---|:---|:---|
| `label` | int | 1 = trace bất thường |
| `logs`, `log_features` | danh sách / vector | Template Drain3 của các dòng log của trace; `log_features` được dựng lại lúc nạp (`semantics.py`) |
| `kpis` | `[1]` | placeholder (dataset không có metric) |
| `trace_node_features` | `[35, 6]` | nhánh sync: `[call_count, avg_dur, max_dur, error_rate, root_rate, latency_dev]` mỗi service |
| `trace_adj` | `[35, 35]` | nhánh sync: đồ thị lời gọi nhị phân đối xứng (chỉ cạnh sync) |
| `async_trace_node_features` | `[35, 3]` | nhánh async: `[log1p(#message gửi), log1p(#message nhận), log1p(độ trễ consumer trung bình, giây)]` |
| `async_msg_count_adj` | `[35, 35]` | nhánh async, quan hệ 1: có hướng, `log1p(#message i→j)` |
| `async_temporal_order_adj` | `[35, 35]` | nhánh async, quan hệ 2 (`_order_adj`): `[i,j]=1` nếu mọi span của service i kết thúc trước khi span đầu tiên của service j bắt đầu; `[i,i]=1` đánh dấu service có mặt trong trace. Dùng mọi span, không chứa thời lượng |

`meta.pkl`: `num_services`, `service2idx`, `trace_c=6`, `async_c=3`, `async_edge_mask` (các cặp service từng có cạnh async ở dữ liệu bình thường), `async_order=True`, `scenario_names`, ...

## 8. Lưu ý về dữ liệu (đã đo)

- **Nguồn gốc `async_temporal_order_adj` trong `data/deeptralog` hiện tại.** Khoá pkl là `md5(trace_id + vị trí)` và không lưu id trace thô, nên khoá này được thêm sau bằng cách ghép mỗi mẫu với trace thô qua chữ ký (mỗi service: số span và thời lượng span lớn nhất). Cả 61.364 mẫu đều ghép được; 1,7% (chỉ trace bình thường) có nhiều trace thô trùng chữ ký nhưng khác ma trận thứ tự, và một trong số đó được chọn. Cách tính native trong `_order_adj` đã được kiểm cho ma trận giống hệt ở 1.779/1.779 trace lỗi F13 và 1.855/1.855 trace bình thường đã lấy mẫu. Chạy lại các stage sẽ ghi khoá này native (và có thể lấy mẫu trace bình thường khác).
- **Tiểu ca.** Mỗi zip F-case gồm 5 tiểu ca (ví dụ F01: cancel, preserve, execute, travel-plan cheapest/quickest), mỗi tiểu ca gọi một API khác nhau, và mọi trace trong zip F-case đều bất thường (trace bình thường lấy từ các zip normal). F01-04/05 (travel-plan, 47% trace lỗi F01) dùng API không có trong dữ liệu bình thường.
- **Trace lỗi F01/F13 trông thế nào.** Phần lớn kèm độ trễ ~4 s (thời lượng request ≈4,0–4,8 s so với 7–216 ms) và thêm 2 span `SpringAsync`; ở 7/9 nhóm, chỉ riêng thời lượng request đã cho AUROC 0,975–1,0 so với trace bình thường cùng API. Ở preserve/preserveOther (và rebook, trips/left, admin-travel), quan hệ thứ tự mức service cũng đổi (100% trace lỗi so với 0% trace bình thường, kể cả trace chậm tự nhiên), điều mà độ trễ không giải thích được. Trace F13-02 (trips/left) là trace "im lặng" (≈18 span, ≈14 ms so với ≈133 span).
- **F02** = 1.198 trace có root là food-service (createOrderBatch, ≈+20% message async) + 952 trace có root là travel-service (gần như không có message async). Vì vậy các số đo gộp theo F-case trộn tín hiệu lỗi với cấu thành API; khi cần chính xác hãy so với trace bình thường cùng API.
- `train`/`unlabel` (6.000) và `val` (1.500) được lấy mẫu xuống từ kho 20.000 trace vì RAM; chỉ khoảng 130 trace val có message async.
