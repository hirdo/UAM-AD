# Chọn dataset và bài SOTA để so sánh cho UAM-AD

Ghi chú thiết kế thí nghiệm và **kết quả hiện tại** cho thesis UAM-AD. Ngày: 2026-10-04, cập nhật lần cuối 2026-10-31 (ở mục 1–4, các điểm đổi ngày 10-13 đánh dấu **[CN 10-13]**). Phạm vi: chọn dataset và bài so sánh, thiết kế bảng, kết quả. Ký hiệu: **[KC]** = đã kiểm chứng từ nguồn gốc (link ở cuối), **[Repo]** = kiểm chứng từ tài liệu/code nhánh `develop`, **[SL]** = suy luận, chưa kiểm chứng.

**File này chỉ giữ thiết kế đã chốt và kết quả hiện tại.** Tiến trình làm việc theo ngày (chạy gì lúc nào, lỗi và cách sửa, chạy thử, ước tính giờ, kế hoạch cũ, số liệu trung gian đã bị thay) nằm ở `working_history.md` cùng thư mục; không cần đọc để lấy kết quả. Khi hai file khác số, số ở file này là số hiện tại.

**Kết quả hiện tại nằm ở mục 5, Giai đoạn 3:**
- **Việc 7, DeepTraLog** (Bảng C): chỉ-sync, chỉ-async, đủ bộ, và GGNN, trên F01, F02, F13.
- **Việc 8 – SUMMARY experiments between UAC-AD and UAM-AD**: S1 chi phí, S2 theo dataset, S3 theo scenario, S5 điều số liệu đỡ được, S6 luật chấm.
- **Việc 9 – SUMMARY experiments between UAM-AD and the other models** (so với ART, AnoFusion, Eadro, OrEdge): T1 bảng so sánh và nhận xét theo từng mô hình, T2 mức 80%, T3 cấu hình chạy của các đối thủ.
- **Việc 10, 11, 12, Giai đoạn 4:** Bảng E (chi phí: tham số, thời gian suy luận CPU, RAM, thời gian huấn luyện), Bảng D (HR@k của ART, Eadro, OrEdge, UAM-AD) và Việc 12: Bảng A–E tổng hợp theo bố cục luận văn.
- **Phụ lục** (cuối file, tra cứu): A chi tiết từng seed của UAC-AD và UAM-AD, B bảng sinh tự động của các đối thủ, C chẩn đoán phụ (seed 0).

---

## 1. Đánh giá từng dataset

| Dataset | L+M+T đủ? | Async? | Loại lỗi | Nhãn | Chi phí với code hiện tại | Khả dụng | Kết luận |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **AnoMod-SN** | Có | Hầu như không (Jaeger, gọi Thrift sync) [Repo/SL] | Perf, Service, DB, "Code"(=stop) — 12 case | Theo cửa sổ thời gian từ script inject [Repo] | **0** (đã có preprocess + kết quả) | Zenodo, CC-BY-4.0 [KC] | **Giữ (chính)** |
| **AnoMod-TT** | Không đủ trên thực tế | Có thể một phần (SkyWalking) [SL] | 12 case, code-level JVM thật [KC] | Như SN | Cao: trace chỉ phủ đoạn workload (3–21 phút, ≤5000 trace/case, case Normal 3,4 phút), log chỉ là 1000 dòng cuối mỗi pod, nên phần normal có đủ 3 modality chỉ ~14 cửa sổ 15 s [KC: script thu thập AnoMod + đo trên dữ liệu] | Như SN | **Bỏ [CN 10-13]** (dùng RE2-TT, RE3-TT làm hệ TrainTicket) |
| **RCAEval RE2-OB** | Có | Không [memory dự án] | 6 lỗi hạ tầng × 5 service | Thời điểm inject [KC] | **Thấp**: chỉ thiếu `val.pkl` | MIT [KC] | **Giữ** |
| **RCAEval RE3-OB** | Có | Không | 5 lỗi code | Như RE2 | Thấp: thiếu `val.pkl` | MIT | **Giữ** (chứng minh trace không làm hại: Δ −0,006 ≈ hòa) |
| RCAEval RE2/RE3-TT | Có | Không [memory dự án] | Như OB | Như OB | Trung bình (TT 64 service, trace lớn hơn) | MIT | **Giữ [CN 10-13]** (là hệ TrainTicket duy nhất của bộ chính, vì AnoMod-TT đã bỏ) |
| **DeepTraLog** | Không có metric | **Có thật** (RabbitMQ, SpringAsync) [Repo] | 14 F-case; dùng F01/F02/F13 (nhóm async của bài; mỗi F vừa có dấu hiệu async vừa có dấu hiệu nhánh sync thấy được, mục 2.3) [Repo] [CN 10-31] | Từng trace [KC/Repo] | **0** (đã chạy 5 seed) | Public trên GitHub FudanSELab [KC] | **Giữ (bằng chứng async)** |
| AIOps'21 | Có | Không rõ | Performance (7) | Không rõ cách inject | Cao | Bản thô chưa thấy link công khai; repo ART có kèm D2 "hệ ngân hàng, 01–06/2021", nhiều khả năng chính là bộ này nhưng ở dạng đặc trưng đã trích, không phải log/trace thô [SL] | **Bỏ** (UAM-AD cần log/trace thô để dựng đồ thị trace) |
| Multi-source (MSDS) | Có | Không (OpenStack osprofiler) [SL] | Service (3) | Rally report + fault script | Cao: hệ khác hẳn, trace OpenStack, lệch múi giờ | Zenodo [KC] | **Bỏ**, trừ khi hội đồng yêu cầu so số trực tiếp với MSTGAD |
| Eadro (TT/SN) | Có | Không [notes của bạn] | CPU, network jam, packet loss (3) [KC] | Có | Trung bình | Zenodo 7615393 [KC] | Không cần như dataset; dùng **Eadro như phương pháp so sánh** |
| GAIA (CloudWise) | Có | Không rõ | Nhiều lỗi ứng dụng | Có | Trung bình | Public [KC qua AnoFusion] | Tùy chọn: chỉ chạy nếu muốn so số với AnoFusion/CMT-AD trên sân nhà của họ |

## 2. Bộ dataset đã chốt (cập nhật 2026-10-13)

Quyết định của bạn (cập nhật 2026-10-13): chạy AnoMod chỉ với **SN**; RCAEval (RE2, RE3) cho cả OB và TT (chấp nhận TT chạy hai lần vì dạng lỗi khác nhau); và DeepTraLog. **AnoMod-TT đã bỏ** (lý do ở hàng 2 của bảng dưới).

### 2.1 Bảng dataset

| # | Dataset | Hệ | Scenario chạy | Modality | Loại trace / lỗi | Trạng thái | Việc còn lại trước khi chạy |
|:-:|:--|:--|:--|:--|:--|:--|:--|
| 1 | AnoMod-SN | SocialNetwork, 12 service [Repo] | 12 (3 Code_Stop, 3 DB_Redis, 3 Perf, 3 Svc_Kill) | L+M+T | Sync (Jaeger). Perf / Service / DB / "Code" (thực chất stop container) | Đã chạy 1 seed | Chạy 5 seed |
| 2 | ~~AnoMod-TT~~ | – | – | – | – | **Bỏ [CN 10-13]** | Lý do (từ script thu thập `TT_collection-scripts` của AnoMod, đối chiếu dữ liệu thật): workload chạy trước, rồi mới thu log/metric/trace. Trace chỉ phủ đoạn workload; log là `kubectl logs --tail 1000` mỗi pod (pod bận chỉ phủ ~9 s cuối); metric bước 15 s gồm cả đoạn hệ thống rảnh trước khi bật chaos. Phần normal có đủ 3 modality chỉ ~14 cửa sổ 15 s (SN: 39 cửa sổ 30 s). Thay bằng RE2-TT/RE3-TT |
| 3 | RCAEval RE2-OB | Online Boutique, 11 service | 6 (cpu, delay, disk, loss, mem, socket) [Repo] | L+M+T | Sync, lỗi hạ tầng | Đã chạy 1 seed | `val.pkl` đã có [CN 10-13] (commit `f7c9e2b`, nhánh local `Reval-re2-re3`); chạy 5 seed (số epoch: xem mục 2.2) |
| 4 | RCAEval RE2-TT | TrainTicket, 27 service trong trace [KC: đo trên dữ liệu; không phải 41 hay 64] | 6 loại lỗi (cpu, delay, disk, loss, mem, socket) × 5 service (auth, order, route, train, travel), 3 run | L+M+T | Sync, lỗi hạ tầng | **Đã preprocess [CN 10-13]** (`data/rcaeval_re2_tt`, script `preprocess_rcaeval_re2_tt.py`) | Chạy 5 seed; thêm script eval riêng khi vào Giai đoạn 3 |
| 5 | RCAEval RE3-OB | Online Boutique | 5 (F1–F5) [Repo] | L+M+T | Sync, lỗi code | Đã chạy 1 seed | `val.pkl` đã có [CN 10-13] (như RE2-OB); chạy 5 seed |
| 6 | RCAEval RE3-TT | TrainTicket, 27 service | 4 (f1–f4) × 2 service (auth, route); `ts-route-service_f3_1` và run 4 của auth không dùng | L+M+T | Sync, lỗi code | **Đã preprocess [CN 10-13]** (`data/rcaeval_re3_tt`, script `preprocess_rcaeval_re3_tt.py`; log dùng Drain3 như RE3-OB) | Chạy 5 seed; thêm script eval riêng khi vào Giai đoạn 3 |
| 7 | DeepTraLog | TrainTicket | F01, F02, F13 (async + sync, mục 2.3) [CN 10-31] | **L+T** (không có metric) | Async thật; mỗi F-case cũng có dấu hiệu nhánh sync thấy được | **Đã chạy 5 seed**, 3 cấu hình (chỉ-sync, chỉ-async, đủ bộ) và GGNN | Không còn. F05, F09 đã chạy nhưng không đưa vào Bảng C (mục 2.3) |

Chi phí tiền xử lý lớn nhất là 2 dòng "Chưa" (RE2-TT, RE3-TT). Cả hai là TrainTicket (RCAEval), nên **nên viết một bộ đọc trace TT dùng chung** rồi chỉnh theo từng nguồn dữ liệu.

### 2.2 Số seed và tham số

**Số seed: 5 cho mọi dataset.** `Svc_Kill_*` của SN chỉ có 4 cửa sổ anomaly nên nhiễu [Repo]; xử lý bằng cách báo trung bình qua các scenario kèm std qua seed, không tăng seed riêng cho SN.

**Chi phí** (từ docs [Repo]): RE2-OB ≈ 1761 s/scenario × 6 ≈ 3 giờ cho mỗi cấu hình mỗi seed; RE3-OB (trace) ≈ 5045 s cho 5 scenario ≈ 1,4 giờ; DeepTraLog ≈ 7–13 phút mỗi F-case mỗi seed; các dataset TT chưa đo, nhiều khả năng nặng hơn [SL].

**Phần cứng [CN 10-13]:** GTX 1050 3GB, ~8GB RAM; mỗi run cần ~2GB RAM nên chỉ chạy tuần tự từng run. Ước tính riêng UAM-AD + UAC-AD 5 seed: RE2-OB ≈ 28 giờ (2 cấu hình × 5 seed × ~2,8 giờ), RE3-OB ≈ 12 giờ, DeepTraLog ≈ 15 giờ (90 run × ~10 phút); chưa tính 2 dataset TT và các đối thủ, tổng dự kiến trên 100 giờ máy.

**Tham số: không tối ưu.** Chọn tham số theo kết quả test thì ít giá trị (dữ liệu test dùng để chọn rồi lại dùng để báo cáo), và chia thêm tập để chọn thì dữ liệu quá ít. Mục tiêu của luận văn là so sánh công bằng, không phải tìm kết quả tốt nhất có thể, nên:

| Nguyên tắc | Nội dung |
|:--|:--|
| Cùng cấu hình cố định | UAM-AD và UAC-AD dùng chung một cấu hình (cùng code gốc); mỗi đối thủ dùng cấu hình mặc định của tác giả |
| Cùng điều kiện chạy | cùng split, cùng ngưỡng p95-val, cùng 5 seed, cùng dữ liệu đầu vào |
| Chốt trước, không chỉnh sau | ghi cấu hình ra trước khi chạy 5 seed cuối; không đổi sau khi thấy kết quả |
| Nói thẳng trong luận văn | kết quả không phải kết quả tốt nhất có thể của từng mô hình; chỉ so sánh tương đối trong cùng điều kiện |

Hai điểm nên xử lý trung thực:
1. **Ngân sách huấn luyện đang lệch giữa dataset:** RE2-OB và RE3-OB chạy 5 epoch, SN 50 epoch (patience 15), DeepTraLog 10 [Repo]. Chạy lệch giữa các dataset không làm hỏng so sánh trong cùng một dataset, nhưng nên thống nhất (ví dụ tối đa 50 epoch, dừng sớm theo val loss, patience 15) nếu thời gian cho phép. Đây chỉ dùng val (chỉ normal), không dùng nhãn test. **[CN 10-13] Quyết định:** chạy thử 1 scenario RE2-OB với 5 so với ~20 epoch, chỉ nhìn val loss, rồi mới chốt số epoch chung; chạy 50 epoch toàn bộ có thể tốn gần 10 lần thời gian (RE2-OB ≈ 350 s/epoch/scenario).
2. **Một số tham số riêng của UAM-AD đã được chọn khi nhìn kết quả** (`activity_penalty_weight` 1,5 và `gate_delta_lr_mult` 10 trên SN; `score_rule=norm_sum` trên DeepTraLog [Repo]). Đây là thành phần của mô hình đề xuất, không phải quét tham số, nhưng nên nêu rõ trong luận văn, và các đối thủ chạy theo mặc định của họ.

**Cấu hình đóng băng [CN 10-14]** (file `codes/common/frozen_config.json`; UAM-AD và UAC-AD dùng chung, không đổi sau khi thấy kết quả):

| Dataset | epoch | patience | window | batch | Tham số khác |
|:--|:-:|:-:|:-:|:-:|:--|
| RE2-OB, RE3-OB, RE2-TT, RE3-TT | 5 5 | 3 | 30 | 128 | `trace_c` 6, `gate_lambda` 0,01 |
| SN | 50 50 | 15 | 5 | 256 | `alpha` 0,16, `open_gan_sep` True |
| DeepTraLog | 10 10 | 5 | 5 | 128 | `alpha` 0,16, `open_gan_sep` True, `open_unmatch_zoomout` False, `gate_lambda` 0,01 |

Dùng chung: 5 seed (run 0–4), `val_percentile` 95, `score_rule=norm_sum`, không point-adjust, `learning_rate` 0,001. Ghi chú: (1) 5 epoch cho RE2/RE3 vì thử 15 epoch trên RE2-OB `cpu` (1 seed) chỉ nâng AUROC 0,860→0,880 và F1@val 0,504→0,523 với thời gian gấp 3; (2) `window_size` tính theo đơn vị mẫu của từng dataset (OB/TT: 30 mẫu 1 giây = 30 giây; SN: 5 cửa sổ 30 giây = 150 giây; DeepTraLog: 5 trace liên tiếp), nên khác nhau giữa dataset nhưng giống nhau giữa hai mô hình; (3) batch của SN (39 mẫu train) thực tế là chạy cả tập một lượt nên 256 hay 128 như nhau [suy ra từ số mẫu, chưa chạy kiểm]; (4) các lần chạy cũ của SN/OB dùng `raw_sum`, cấu hình này dùng `norm_sum` nên Giai đoạn 3 chạy lại toàn bộ; (5) các giá trị lấy từ các lần chạy trước, chưa rõ đã tinh chỉnh hay đặt tay.

### 2.3 DeepTraLog: F-case dùng và mệnh đề cần chứng minh

**F-case dùng [CN 10-31]: F01, F02, F13.** Đây là nhóm "Asynchronous Interaction" của bài DeepTraLog (bảng 1) [Repo: `docs/preprocess_deeptralog_en.md`]. Đo theo từng API (trace bất thường so với normal cùng API; nền là khoảng 93 000 trace normal; `docs/preprocess_deeptralog_en.md` §4, §8 và `D:\ClaudeWork\dtl\fcase_fingerprints.pkl`) [KC], cả ba **vừa có dấu hiệu async vừa có dấu hiệu mà nhánh sync nhìn thấy**, nên mỗi F-case được coi là *async + sync*, không tách được thành lỗi async thuần hay sync thuần:

| F-case | Dấu hiệu async (thứ tự service, số tin nhắn) | Dấu hiệu nhánh sync thấy được (thời lượng, cạnh sync, số span) |
|:--|:--|:--|
| F01 | thứ tự đổi 100% ở `preserve` (400 trace); số tin nhắn async không đổi ở các nhóm so được | trễ khoảng 4 giây: thời lượng request tách trace bất thường khỏi normal cùng nhóm (AUROC 0,97 và 1,00 ở hai nhóm so được); số span +64% ở `cancel` (340), +83% ở `execute` (350). 47% trace bất thường (travel-plan) thuộc API không có normal để so |
| F02 | số tin nhắn async +21% ở `createOrderBatch` (1 199/2 151 trace); 952 trace `trips/left_parallel` không đổi cấu trúc | ở nhóm `createOrderBatch`: thời lượng request dài hơn (0,1 → 0,2 giây, AUROC 1,00), số dòng log +44% [SL: nhóm lấy theo tập service, xấp xỉ API] |
| F13 | thứ tự đổi 100% ở `rebook` (377), `trips/left` (299), `admintravel` (350), `preserveOther` (350); số tin nhắn async không đổi | trễ khoảng 4 giây (thời lượng request AUROC 1,00 ở 3 nhóm so được); 37,7% cạnh sync lạ ở `preserveOther` (350/1 779 trace) |

Các số về thời lượng request lấy từ đo lại (10-31) trên pkl đầu vào, nhóm theo tập service (xấp xỉ API) và so với normal train cùng nhóm; tài liệu repo §8 ghi cùng kết luận (AUROC 0,975–1,0 ở 7/9 nhóm). Nghĩa là các lỗi này được tiêm vào cơ chế async (theo bài), nhưng dấu vết đo được nằm ở cả hai nhánh; không dùng cách nói "lỗi async thuần" trong luận văn.

**Không có F-case nào là lỗi sync thuần** [KC]. Dấu vân cấu trúc (2026-10-13), tính trên các trace có nhãn bất thường của từng F-case: tỉ lệ trace có cạnh "lạ" so với trace bình thường cùng API; cạnh sync (S) và async (A) tách riêng theo luật `classify_edge` của `preprocess_deeptralog.py`.

| F-case | Trace bất thường | % cạnh S lạ | % cạnh A lạ | % thứ tự lạ |
|:--|--:|--:|--:|--:|
| F03 | 2564 | 12,9 | 0 | 25,8 |
| F04 | 2546 | 13,0 | 0 | 26,2 |
| F05 | 1665 | 0 | 2,3 | 6,4 |
| F09 | 978 | 0 | 0 | 0 |
| F10 | 900 | 0 | 84,4 | 0 |
| F14 | 870 | 33,7 | 0 | 33,7 |

- Không F-case nào đạt "≥ 80% trace có cạnh sync lạ". F04: 100% chỉ đúng với API `preserve` (330/2546 trace); mỗi F-case chỉ có **một** sub-case có cạnh sync lạ (F04-01 330 trace, F03-03 330, F14-03 293), cùng một kiểu lỗi (`preserve`/`preserveOther` gọi thêm `consign`); các sub-case còn lại không có cạnh lạ (đổi số span; F04 còn đổi thứ tự ở `trips/left`, xem dưới).
- Ở F04 và F14, thứ tự lạ đi cùng cạnh lạ (F04-`preserve`: cả hai 100%; F14: cả hai 33,7%), và F04 còn đổi thứ tự ở `trips/left` (69% và 48%). Vì vậy F03, F04, F14 không phải bằng chứng "lỗi sync thuần".
- F10: 84,4% "cạnh A lạ" là các cặp service đã có dạng S trong dữ liệu normal, bị heuristic thời gian (con bắt đầu sau khi cha kết thúc) đổi thành A; không phải cạnh mới, không dùng.

**F05, F09 không dùng [CN 10-31]** (nhóm Configuration và Monolithic của bài). Cạnh lạ và thứ tự lạ ≤ 10% (F09: 0%), tức không có cấu trúc async hay sync đổi. Gần như không ghép được trace bất thường với normal cùng nhóm tập service (F05: 1 624/1 665 trace, F09: 978/978 nằm ngoài các nhóm có ít nhất 40 normal train), nên AUROC mức file của hai F-case này chưa tách được tín hiệu lỗi khỏi khác biệt API. Hai F-case đã chạy 5 seed (số ở S6) nhưng không đưa vào Bảng C.

**Hai mệnh đề cần chứng minh** (vì F01, F02, F13 không tách được lỗi async và lỗi sync riêng):

1. **Đủ bộ tốt hơn từng nhánh lẻ:** ở mỗi F-case và mỗi chỉ số (F1@val, AUROC, AUPRC), đủ bộ không thấp hơn chỉ-sync và chỉ-async ("≥"), và hơn (">") ở phần lớn các cặp.
2. **So các nhánh lẻ với nhau:** báo nhánh lẻ nào hơn ở F-case nào, kèm giải thích từ đầu vào (bảng trên).

Quy ước với 5 seed: "≥" nghĩa là trung bình đủ bộ không thấp hơn trung bình nhánh lẻ quá một độ lệch chuẩn; ">" nghĩa là hơn quá một độ lệch chuẩn (lấy độ lệch chuẩn lớn hơn của hai cấu hình so sánh). Báo cả F1@val lẫn AUROC/AUPRC; kết luận về F1 thường yếu hơn AUROC (F02, F1 ≈ 0,1). **Không phát biểu** "lỗi sync chỉ tốt khi chạy sync" hay "lỗi async chỉ tốt khi chạy async": không F-case nào thuần, và chưa có cấu hình log-only để tách phần của log.

**Cấu hình chạy trên mỗi F-case** (cùng split, `norm_sum`, 5 seed): chỉ-sync (`--open_trace True --open_async_trace False`), chỉ-async (`--open_trace False --open_async_trace True`), đủ bộ (cả hai `True`), và DeepTraLog GGNN (code gốc). Hai cấu hình chỉ-sync và chỉ-async vẫn giữ phần tái tạo log; nhánh async gồm số hạng đếm tin nhắn và số hạng thứ tự. AnoFusion, ART, Eadro không chạy được vì dataset không có metric [Repo].

## 3. Bài SOTA

### 3.1 Danh sách đã kiểm chứng

| Phương pháp | Venue | Modality | Học | Dataset gốc | Code | Ghi chú |
|:--|:--|:--|:--|:--|:--|:--|
| **UAC-AD** (Liu et al.) | IEEE TSC 2024 [KC] | L+M | Không giám sát | Open-source + dữ liệu viễn thông [KC] | Là code gốc của repo | **Baseline chính** |
| **Hades** (Lee et al.) | ICSE 2023 [KC] | L+M | Bán giám sát | Mô phỏng + Huawei Cloud [KC] | Có, tác giả tuyên bố public [KC]; repo GitHub BEbillionaireUSD [SL] | Repo của bạn gọi model là "HADES" trong docs — nên thống nhất tên khi viết |
| **Eadro** (Lee et al.) | ICSE 2023 [KC] | L+M+T | **Có giám sát** | TT, SN tự thu (Zenodo) [KC] | github.com/BEbillionaireUSD/Eadro [KC] | AD + RCA. Chạy được trên SN/RE2 vì cùng 3 modality; ghi rõ "supervised" |
| **AnoFusion** (Zhao et al.) | KDD 2023 [KC] | L+M+T | Không giám sát | GAIA + ngân hàng (bí mật) [KC] | github.com/AIOps-Lab-NKU/AnoFusion [KC; link `zcyyc/AnoFusion` cũ trả 404, đã sửa 2026-10-13] | **Đối thủ công bằng nhất** (cùng 3 modality, cùng không giám sát). Ngưỡng EVT |
| **ART** | ASE 2024 [KC] | L+M+T (đặc trưng metric/trace/log theo pod + metric node) [KC] | **Không giám sát** [KC] | D1: e-commerce mô phỏng 46 instance (2022); D2: hệ ngân hàng thương mại 18 instance (01–06/2021) [KC] | github.com/bbyldebb/ART, kèm cả 2 dataset [KC] | **Đối thủ L+M+T không giám sát thứ 2**. Làm AD + failure triage + RCA. Trong TrioXpert (ASE'25), ART đạt AD F1 0,683/0,737, Hades 0,865/0,868 [KC] |
| **OrEdge** (Zaki et al.) | arXiv 07/2026, chưa bình duyệt [KC] | L+M+T [KC] | **Bán giám sát** (loss tái tạo + loss phân loại) [KC] | MSDS, SN + TT của Eadro [KC] | github.com/theamrzaki/MicroService_Twin_Original (dựa trên code MSTGAD + Eadro) [KC] | Bài L+M+T mới nhất có code; đã tự so với Eadro, AnoFusion, ART, MSTGAD. Nice-to-have, ghi rõ bán giám sát. Cùng họ với MSTGAD (xây trên code MSTGAD + Eadro [KC README]; khác ở chỗ làm phẳng đồ thị thành chuỗi độc lập và tái tạo tuyến tính trên miền đa thức Legendre, ~9,6K tham số, nhắm triển khai ở edge [KC bài]); chưa so sánh từng dòng code [SL] |
| Medicine (Tao et al.) | ASE 2024 [KC] | L+M+T [KC] | **Có giám sát** [KC] | AIOps22, GAIA [KC] | github.com/AIOps-Lab-NKU/Medicine [KC] | Bài toán là **phân loại loại lỗi** (failure diagnosis), không phải anomaly detection → không so ở Bảng B, chỉ trích dẫn |
| **MSTGAD** (Huang et al.) | arXiv 2310.04701, 2023 [KC]; venue chưa xác minh | L+M+T | **Bán giám sát** (50% train có nhãn, loss tái tạo + loss phân loại) [KC] | MSDS + AIOps-Challenge [KC] | github.com/alipay/microservice_system_twin_graph_based_anomaly_detection (bản chính thức của Alipay) [KC], chỉ có script cho MSDS | arXiv 10/2023, chưa thấy bản đăng hội nghị/tạp chí [KC]. Cùng họ với OrEdge; chỉ dùng thay OrEdge nếu OrEdge không dựng được. Là bán giám sát nên chỉ làm hàng tham chiếu, không thay được hàng không giám sát (ART) [SL] |
| **DeepTraLog** (Zhang et al.) | ICSE 2022 [KC] | T+L | Không giám sát (GGNN + Deep SVDD) | DeepTraLog/TT; P 0,93, R 0,97 [KC] | Repo FudanSELab (có data; code chưa xác minh) [SL] | **Đối thủ chính trên DeepTraLog** |
| **TraceVAE** (Xie et al.) | WWW 2023 [KC] | T | Không giám sát | TT + dữ liệu công nghiệp | Có trong TADBench [KC] | Chỉ trích dẫn ở related work; không chạy lại (mô hình multimodal) |
| **GTrace**, TraceCRL, CRISP, TraceAnomaly, PUTraceAD, MultimodalLSTM | FSE'23, FSE'22, ATC'22, ISSRE'20, ISSRE'22, CLOUD'19 [KC] | T | – | 5 dataset trace (TT, GAIA, AIOps20/22/23) | Cả 7 trong **TADBench** (github.com/nkalgo/TADBench) [KC] | Nguồn tiện nhất để chạy nhiều baseline trace |
| **TraceDAE** | IEEE, 2025 (doc 11052877) [KC tiêu đề] | T | Không giám sát | Chưa xác minh | Chưa tìm thấy | Nguồn cảm hứng thiết kế của bạn: trích dẫn, không cần chạy lại |
| SCWarn | ISSTA 2022 [SL] | L+M | Không giám sát | – | Có [SL] | Là baseline trong MSTGAD/AnoFusion [KC] |
| KANAD | JNSM 2025 [KC tên] | L+M(+T?) | – | – | Chưa xác minh | Chỉ trích dẫn |
| AMulSys | Information Fusion 2025 [KC] | L+M+T | – | 2 dataset thật | Không thấy code [KC] | So với UAC-AD, Hades, DeepTraLog, AnoFusion, MSTGAD → chỉ trích dẫn ở related work |
| CMT-AD | arXiv 2609.25856, 2026 (chưa bình duyệt) [KC] | L+M | Clustering | GAIA, Spark, Stack | github.com/wpp33669-hub/CMT-AD [KC] | Bài mới nhất có so với UAC-AD (UAC-AD F1 0,951/0,891/0,939). Tùy chọn |

RCA-only (TORAI, BARO, Nezha, MULAN, DiagFusion): không so ở phần AD. Nếu muốn so phần RCA top-k, RCAEval đã cài sẵn BARO, RCD, CIRCA, MicroRank, TraceRCA, PDiagnose... [KC].

### 3.2 Bài nào chạy được cùng giao thức (so sánh công bằng)

So số từ bài báo khác **không công bằng**: split, cửa sổ, ngưỡng và point-adjust khác nhau. Cách đúng là chạy lại họ trên chính split của bạn, cùng ngưỡng p95 val, rồi báo cả F1@val, AUROC, AUPRC; F1 oracle chỉ dùng để đối chiếu với số trong bài gốc.

Hai chỗ khác nhau về giao thức (cột trái [Repo]; cột phải là thông lệ chung, riêng EVT của AnoFusion là [KC], từng bài khác chưa kiểm [SL]):

| | Giao thức của bạn | Thường gặp ở bài khác |
|:--|:--|:--|
| Chọn ngưỡng | Cố định = phân vị 95 của điểm trên val (chỉ normal), **không dùng nhãn test** | Quét ngưỡng trên điểm test rồi lấy F1 cao nhất (chính là F1 oracle), hoặc EVT (AnoFusion) |
| Cách đếm đúng | **Không point-adjust**: mỗi cửa sổ/trace tính riêng | **Point-adjust**: chỉ cần bắt trúng 1 điểm trong cả đoạn lỗi thì cả đoạn được tính là bắt đúng |

Cả quét ngưỡng lẫn point-adjust đều làm F1 cao hơn thực tế (quét ngưỡng dùng nhãn test để chọn ngưỡng; point-adjust thưởng cho việc bắt trúng 1 điểm). Nên số F1 trong bài gốc thường cao hơn số chạy lại, và không đặt cạnh số của bạn được. Vì vậy bộ 4 chỉ số của bạn chia hai nhóm: F1@val là chỉ số chính theo giao thức thật; AUROC/AUPRC không cần ngưỡng nên không bị ảnh hưởng bởi chuyện chọn ngưỡng; F1 oracle chỉ để nhìn cùng kiểu với các bài quét ngưỡng (UAC-AD, TraceDAE).

Ưu tiên theo giá trị / công sức:

| # | Phương pháp | Học | Vai trò trong Bảng B | Ghi chú |
|:-:|:--|:--|:--|:--|
| 1 | **AnoFusion** (KDD'23) | Không giám sát | Đối thủ công bằng | L+M+T, code public → SN, RE2-OB, RE3-OB (và các TT). Phép so quan trọng nhất cho "mô hình đa modality có trace" |
| 2 | **ART** (ASE'24) | Không giám sát | Đối thủ công bằng | L+M+T, code public kèm dữ liệu. Đầu vào là đặc trưng theo instance (metric, số lượng/độ trễ trace, số log theo pod), nên cần script chuyển pkl của bạn sang định dạng của ART [SL]. Có ART là có 2 đối thủ L+M+T không giám sát |
| 3 | **Eadro** (ICSE'23) | **Có giám sát** | Hàng tham chiếu (tùy chọn) | L+M+T, code public; lợi thế nhãn nên nếu UAM-AD gần bằng là đã tốt |
| 4 | **OrEdge** (arXiv 07/2026) | **Bán giám sát** | Hàng tham chiếu (tùy chọn) | Mới nhất có code. MSTGAD cùng họ (OrEdge xây trên code MSTGAD + Eadro): chỉ chạy một; MSTGAD thay thế khi OrEdge không dựng được |
| 5 | **DeepTraLog (GGNN)** | Không giám sát | Bảng C (chỉ trên DeepTraLog) | Cùng split F01/F02/F13 |

Đã tìm kỹ các bài 2025–2026 nhưng **không thấy mô hình L+M+T không giám sát nào mới hơn ART có code public**; các bài mới đều bán giám sát (OrEdge), few-shot, dùng LLM (TrioXpert), không có code (AMulSys, KANAD) hoặc chỉ dùng L+M (CMT-AD). Vì vậy hai đối thủ không giám sát là AnoFusion (2023) và ART (2024); nếu ART không dựng được thì không có phương án thay trong cùng nhóm không giám sát (MSTGAD/OrEdge dùng nhãn nên không thay được), mục tiêu hiện tại là chỉ còn AnoFusion + hàng tham chiếu.

**Có nên đưa mô hình bán giám sát / có giám sát vào so sánh?** Không cùng điều kiện (Eadro, OrEdge, MSTGAD dùng nhãn lỗi khi train, UAM-AD thì không), nên không dùng làm mốc "công bằng". Nhưng vẫn đáng đưa vào Bảng B như **hàng tham chiếu**, tách riêng và ghi rõ cột "Học", vì: (a) cho thấy mô hình không giám sát cách mô hình có nhãn bao xa; (b) nếu UAM-AD gần bằng hoặc hơn thì là điểm mạnh (không cần nhãn mà vẫn tương đương); (c) nếu thấp hơn thì khoảng cách đó có lời giải thích sẵn. Điều kiện: nêu rõ lượng nhãn Eadro/OrEdge được dùng (cùng split train, nhãn từ phần nào), và kết luận chính của luận văn chỉ dựa trên so sánh với các hàng không giám sát (UAC-AD, AnoFusion, ART).

## 4. Bảng thí nghiệm gợi ý

Thứ tự bảng (đã bỏ ablation nên đánh số lại): **A** thống kê dataset, **B** kết quả chính phát hiện bất thường, **C** sync vs async trên DeepTraLog, **D** xác định service lỗi (RCA), **E** chi phí.

### Bảng A – Thống kê dataset (số lấy từ pkl đã tạo và từ docs tiền xử lý [Repo]; cập nhật 2026-10-13)

| Dataset | Hệ / số service | Modality | Đơn vị mẫu | Train (normal) | Val (normal) | Test | Tỉ lệ anomaly trong test | Async |
|:--|:--|:--|:--|--:|--:|:--|:--|:--|
| AnoMod-SN | SocialNetwork / 12 | L+M+T | cửa sổ 30 s | 39 | 57 | 12 file: 3 `Code_Stop` 263 cửa sổ (39 anomaly), 6 `DB_Redis` + `Perf` 80 (10), 3 `Svc_Kill` 32 (4) | 12,5% – 14,8% | Không (Jaeger, sync) |
| RE2-OB | Online Boutique / 11 | L+M+T | 1 giây = 1 mẫu | 11.104 (+ 44.414 `unlabel` cho GAN) | 9.796 (`val.pkl`, đã có) | 6 file theo loại lỗi (cpu, delay, disk, loss, mem, socket), 51.520 – 54.075 mẫu | ≈ 20% (tối đa 20%) | Không |
| RE2-TT | TrainTicket / 27 | L+M+T | 1 giây = 1 mẫu | 11.016 (+ 44.064 `unlabel`) | 9.720 (`val.pkl`) | 6 file theo loại lỗi, 54.075 mẫu mỗi file | 20% (10.815 anomaly/file) | Không |
| RE3-OB | Online Boutique / 11 | L+M+T | 1 giây = 1 mẫu | 3.672 (+ 14.688 `unlabel`) | 3.240 (`val.pkl`, đã có) | 5 file (f1–f5) | tối đa 20% | Không |
| RE3-TT | TrainTicket / 27 | L+M+T | 1 giây = 1 mẫu | 3.672 (+ 14.688 `unlabel`) | 3.240 (`val.pkl`) | 4 file (f1–f4), 26.999 mẫu mỗi file | 20% (5.399 anomaly/file) | Không |
| DeepTraLog | TrainTicket / 35 node | L+T (không có metric) | 1 trace | 6.000 (lấy mẫu xuống từ kho 20.000) | 1.500 | mỗi F-case một file (anomaly của F-case + normal cho đủ 12,5%) | 12,5% mỗi file; toàn dataset 132.485 trace, 23.334 bất thường (17,6%) | **Có** (RabbitMQ, SpringAsync) |

Số scenario/run: AnoMod-SN 12 scenario + 1 `Normal_Baseline`, mỗi session 19,5–25 phút; RE2-OB và RE2-TT 30 scenario × 3 run (pre-injection 720 s/run); RE3-OB 10 scenario × 3 run; RE3-TT 8 scenario dùng được (auth/route × f1–f4) × 3 run, 1801 s/run (lỗi ở giữa); DeepTraLog 14 F-case, dùng F01, F02, F13 (file test: F01 16 424 trace, 2 053 bất thường; F02 17 208 trace, 2 151 bất thường; F13 14 232 trace, 1 779 bất thường; 12,5% mỗi file; normal lấy từ pool 7500 trace train+val). Cột "Đơn vị mẫu" khác nhau giữa các dataset (cửa sổ 30 s, 1 giây, 1 trace), nên không so trực tiếp số mẫu giữa các hàng.

**Giới hạn đã biết của RE2/RE3 [CN 10-13]:** `val.pkl` không trùng `train`/`unlabel`, nhưng các file `test_*.pkl` lấy normal từ **toàn bộ** pool normal (gồm cả train, unlabel, val): khoảng 2/3 normal trong test là cửa sổ mô hình đã thấy khi train. Commit `f7c9e2b` ghi đây là giới hạn cố ý vì ở anomaly_rate 0,20 không đủ normal để tách. Cần ghi vào luận văn; kết quả vẫn so sánh được giữa các phương pháp trên cùng split.

### Bảng B – Kết quả phát hiện bất thường (5 seed, mean±std)

Bố cục: **hàng = phương pháp, cột = dataset** (AnoMod-SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT). Làm **4 khối cùng bố cục**, mỗi khối một chỉ số: **B1 F1@val** (chỉ số chính), **B2 AUROC**, **B3 AUPRC**, **B4 F1 oracle** (ghi chú: quét ngưỡng trên test, có point-adjust, chỉ để đối chiếu với bài quét ngưỡng; không dùng để kết luận). DeepTraLog nằm riêng ở Bảng C.

| Nhóm | Phương pháp | Modality | Học |
|:--|:--|:--|:--|
| Không giám sát (so sánh công bằng) | UAC-AD (baseline) | L+M | Unsup |
| | AnoFusion | L+M+T | Unsup |
| | ART | L+M+T | Unsup |
| | **UAM-AD** | L+M+T | Unsup |
| Tham chiếu có nhãn (tùy chọn) | OrEdge | L+M+T | Semi-sup |
| | Eadro | L+M+T | **Sup** |

Chỉ nhóm không giám sát dùng để kết luận "UAM-AD hơn". Hai hàng tham chiếu là tùy chọn: có thì cho thấy khoảng cách so với mô hình có nhãn, không có cũng không ảnh hưởng kết luận chính.

### Bảng C – Sync vs async (DeepTraLog; xem mục 2.3)

Theo F-case: F01, F02, F13 (mỗi F vừa có dấu hiệu async vừa có dấu hiệu nhánh sync thấy được). Mỗi ô báo F1@val / AUROC / AUPRC (cột F1 oracle của `run.py` không dùng cho DeepTraLog, xem kết quả).

| Cấu hình | F01 | F02 | F13 |
|:--|:--|:--|:--|
| Chỉ sync | | | |
| Chỉ async | | | |
| Đủ bộ sync + async (UAM-AD) | | | |
| DeepTraLog GGNN | | | |

Kỳ vọng cần chứng minh (hai mệnh đề ở mục 2.3): (1) hàng đủ bộ không thua hàng chỉ-sync và chỉ-async ở mỗi F-case, và hơn từng nhánh lẻ ở phần lớn các cặp; (2) so các nhánh lẻ với nhau, nhánh nào hơn ở F-case nào.

### Bảng D – Xác định đúng service lỗi (RCA top-k)

**Bảng cần làm.** Chỉ hỏi một câu: khi có lỗi, mô hình chỉ ra đúng service lỗi bao nhiêu phần trăm.

**So với hai bài (chốt):** **ART** (ASE'24, không giám sát) và **Eadro** (ICSE'23, có giám sát, hàng tham chiếu). Lý do: cả hai dùng L+M+T như UAM-AD và cùng làm xác định service lỗi, đồng thời đã nằm trong Bảng B nên không phải dựng thêm mô hình [KC]. Không đưa thêm các phương pháp RCA khác (BARO, RCD... trong RCAEval): chúng chỉ làm RCA, không phải mô hình đa modality để so với UAM-AD, và điều kiện đầu vào khác (được cho sẵn thời điểm lỗi). UAC-AD không có hàng ở bảng này vì không xuất được xếp hạng service (điểm theo từng service chỉ do nhánh trace tính ra [Repo]).

**Chỉ số: HR@k** = tỉ lệ anomaly mà service lỗi thật nằm trong top-k dự đoán, k = 1, 3, 5. Bỏ MRR cho gọn (code vẫn tính sẵn nếu cần). **Cách tính (đã đưa về cùng tập mẫu):** mọi mẫu bất thường của file test mà service lỗi biết trước; ART, Eadro, OrEdge và UAM-AD đều dùng tập này. Đơn vị chấm vẫn khác (UAM-AD cửa sổ 30 mẫu; ART từng mẫu giây; Eadro, OrEdge cửa sổ 10), nên chỉ dùng để thấy khoảng cách cỡ lớn. OrEdge được thêm làm hàng tham chiếu thứ hai vì đã có sẵn từ việc 9.

**Kết quả: xem mục 5, Giai đoạn 4, Việc 11.** **Dataset tính được (cả 5 dataset):** RE2-OB, RE3-OB, RE2-TT, RE3-TT (100% mẫu bất thường có service lỗi; OB 11 service, TT 27 service) và SN (9/12 scenario, 154/183 cửa sổ bất thường; 3 scenario `Perf_*` là lỗi mức host, không quy về một service nên bị loại) [Repo]. DeepTraLog không nằm trong bảng này.

### Bảng E – Chi phí

**Bảng cần làm.** Tham khảo bộ chỉ số của OrEdge (tên cột và đơn vị theo ý bạn; định nghĩa chính xác trong bài OrEdge mình chưa đọc lại). Đo trên cùng máy cho mọi mô hình:

| Tiêu chí | Đo như thế nào | Chốt |
|:--|:--|:--|
| Training time (s) | Tổng giây huấn luyện đến khi dừng, ghi rõ thiết bị (GPU hay CPU) | Giữ |
| Số tham số | Tham số **dùng lúc suy luận** (không tính phần chỉ dùng khi train như discriminator của GAN, nếu có); ghi thêm tổng nếu muốn | Giữ (tương ứng "ReconstructionParams" của OrEdge) |
| Inference time (CPU, ms) | ms trên mỗi cửa sổ/trace, chạy bằng CPU, cùng batch size | Giữ, đổi sang đo trên CPU |
| Max mem (CPU, MB) | Đỉnh RAM của tiến trình khi suy luận | **Thêm** (đo theo cách bên dưới) |

**Cách đo RAM cho công bằng** (vì file dữ liệu nạp vào RAM dễ làm lệch số): (1) nạp dữ liệu test vào RAM trước, đọc RSS lúc đó làm mốc (`psutil.Process().memory_info().rss`); (2) chạy suy luận trên toàn bộ test với batch size cố định, theo dõi RSS cao nhất trong lúc chạy (lấy mẫu định kỳ, hoặc `resource.getrusage(...).ru_maxrss` ở một tiến trình chạy riêng); (3) ghi **đỉnh RSS − mốc** (phần RAM do mô hình tăng thêm) và đỉnh RSS tuyệt đối; **bảng chỉ đưa cột đỉnh tuyệt đối** (đỉnh − mốc nằm trong file json từng lượt đo); (4) mỗi mô hình chạy trong một tiến trình mới, cùng dữ liệu, cùng batch size, để mô hình trước không để lại bộ nhớ. Số tuyệt đối vẫn chịu ảnh hưởng của thư viện (PyTorch, CUDA...) nên so sánh tương đối giữa các mô hình trên cùng máy, không so với số trong bài khác [SL].

Hàng: UAC-AD, UAM-AD, AnoFusion, ART, Eadro, OrEdge. Báo trên cả 5 dataset (RE2-OB, RE2-TT, RE3-OB, RE3-TT, SN), đo trên mọi file test của dataset. SN có bộ đo riêng vì runner SN của các đối thủ khác runner RCAEval. Đã bỏ: số epoch đến mô hình tốt nhất, bộ nhớ GPU, FLOPs, chi phí dựng đặc trưng trace, và tỉ số so baseline (chỉ có nghĩa giữa UAM-AD và UAC-AD). Số đã có [Repo]: RE2-OB trace ≈ 1761 s/scenario so với baseline 1620 s; RE3-OB 5045 s so với 3579 s cho 5 scenario; DeepTraLog ≈ 7–13 phút mỗi F-case mỗi seed. Số tham số, thời gian suy luận và RAM đã đo ngày 10-30: kết quả ở mục 5, Giai đoạn 4, Việc 10.

## 5. Việc cần làm theo thứ tự

Thứ tự theo phụ thuộc thật: dữ liệu → chốt cấu hình → chạy → đo thêm → tổng hợp. Các việc cùng giai đoạn chạy song song được.

### Giai đoạn 1 – Chuẩn bị dữ liệu (chưa chạy kết quả cuối)

| # | Việc | Phụ thuộc | Ra gì |
|:-:|:--|:--|:--|
| 1 | Tách `val.pkl` cho RE2-OB, RE3-OB (chỉ cửa sổ normal, không trùng train/test) | – | **Xong**: val = 15% cuối theo thời gian của mỗi run, không trùng train/unlabel. Giới hạn test normal: xem dưới Bảng A |
| 2 | Viết **một bộ đọc trace TrainTicket dùng chung**, rồi preprocess RE2-TT, RE3-TT (kèm `val.pkl`). Kiểm số service của TT và ground-truth service lỗi cho từng nguồn. AnoMod-TT đã bỏ [CN 10-13] | – | **Xong**: hai script `preprocess_rcaeval_re{2,3}_tt.py` tái dùng lớp của bản OB, chỉ đổi danh sách service/loại lỗi. Số service TT = 27 (cả hai nguồn). Ground-truth service lấy từ id mẫu như OB (đã thêm tên TT vào `rca.py`, `run.py`): 100% anomaly có GT, nên TT dùng được cho Bảng D. `trace_node_features[:, 3]` (error_rate) luôn 0 vì `statusCode` rỗng ở mọi run TT. |
| 3 | Đo "dấu vân cấu trúc" trên dữ liệu thô DeepTraLog | – | **Xong**: kết quả ở mục 2.3 (script `codes/common/measure_structural_fingerprint.py`). Không F-case nào có đa số lỗi cấu trúc. Ban đầu thêm F05, F09 làm đối chứng "không có async" (đã tạo `scenarios/test_F05.pkl`, `test_F09.pkl`); **10-31 bỏ khỏi Bảng C**, lý do ở mục 2.3 |
| 4 | Điền số liệu thật vào Bảng A | 1, 2, 3 | **Xong**: Bảng A đã đủ số liệu cho cả 6 dataset |

Việc 1, 2, 3 độc lập, làm song song. Việc 2 nặng nhất (2 dataset mới).

### Giai đoạn 2 – Chốt cấu hình (phải xong trước khi chạy kết quả cuối)

| # | Việc | Phụ thuộc | Ra gì |
|:-:|:--|:--|:--|
| 5 | Chốt **một cấu hình cố định** cho UAM-AD và UAC-AD (mục 2.2): thống nhất ngân sách huấn luyện (số epoch chốt sau khi chạy thử 1 scenario RE2-OB với 5 so với ~20 epoch và chỉ nhìn val loss; 50 epoch toàn bộ có thể tốn gần 10 lần thời gian), `val_percentile=95`, 5 seed. Ghi cấu hình ra file rồi **đóng băng** | 1 | File cấu hình, không đổi sau khi thấy kết quả | **Xong:** chốt 5 epoch cho RE2/RE3 (patience 3, window 30, batch 128), vì thử 15 epoch trên RE2-OB `cpu` (1 seed) chỉ nâng AUROC 0,860→0,880 và F1@val 0,504→0,523 mà tốn gấp 3 lần thời gian; khớp các lần chạy RE2/RE3-OB trước đó. Cấu hình ở `codes/common/frozen_config.json`. DeepTraLog giữ thiết lập lần chạy trước (10 epoch, patience 5); SN giữ thiết lập lần chạy trước của SN là 50 epoch, patience 15 (theo `docs/experiment_results_sn_trace_vs_baseline_vi.md`), chưa chỉnh lại
| 6 | Dựng các đối thủ trên cùng split (cấu hình mặc định của tác giả): AnoFusion, ART (viết script chuyển pkl sang đặc trưng của ART), Eadro, DeepTraLog GGNN (xác minh code chạy được); OrEdge nếu còn thời gian. Chạy thử 1 seed trên SN để kiểm đầu vào/đầu ra | 1 (SN và OB trước; TT thêm khi có việc 2) | Mỗi đối thủ chạy được và xuất điểm để tính F1@val, AUROC, AUPRC, F1 oracle |

Việc 6 có thể bắt đầu ngay trên SN, không cần chờ việc 1, 2.

### Giai đoạn 3 – Chạy kết quả cuối (5 seed)

| # | Việc | Phụ thuộc | Ra gì |
|:-:|:--|:--|:--|
| 7 | **DeepTraLog trước** (là luận điểm chính về async): 3 cấu hình (chỉ-sync, chỉ-async, đủ bộ) × F01, F02, F13 × 5 seed; DeepTraLog GGNN cùng split. **Xong**; F05, F09 đã chạy nhưng không đưa vào Bảng C (10-31) | 3, 5, 6 | Bảng C (hai mệnh đề ở mục 2.3). Báo đúng số đo, không chỉnh tham số theo kết quả |
| 8 | UAM-AD và UAC-AD, 5 seed, 5 dataset còn lại (SN, RE2-OB, RE2-TT, RE3-OB, RE3-TT). **Bật `--enable_rca` ngay trong lần chạy của UAM-AD** để khỏi chạy lại | 1, 2, 5 | Bảng B (hai hàng), và HR@k của UAM-AD cho Bảng D |
| 9 | AnoFusion, ART, Eadro (và OrEdge), 5 seed, cùng 5 dataset; lấy HR@k của ART và Eadro | 5, 6 (và 2 cho các TT) | Bảng B (các hàng còn lại), HR@k cho Bảng D |
| 10 | **Chấm lại UAM-AD và UAC-AD theo thứ tự thời gian thật trên cả bốn dataset RCAEval** (**xong 10-30 20:56**; người dùng đã đồng ý chấm lại hết cùng cách, 10-30): dùng lại checkpoint của việc 8 (mỗi mô hình: RE2-OB 30, RE3-OB 25, RE2-TT 30, RE3-TT 20), không huấn luyện lại; `run.py --pre_model` với file test sắp theo (thí nghiệm, giây) như ART và AnoFusion đang dùng. Dựng dữ liệu một lần cho mỗi scenario rồi chấm cả 10 checkpoint (5 seed × 2 mô hình) trong cùng tiến trình; bốn dataset RCAEval mất khoảng 1 giờ 20 phút máy. Trình điều khiển: `codes/common/rescore_item10.py`, tổng hợp: `codes/common/aggregate_item10.py`; hàm xếp hạng mới được thêm vào `codes/models/rca.py`, hàm cũ giữ nguyên. Đã kiểm tái tạo số đã lưu ở test xáo trên 10 checkpoint RE3-OB `f1` (lệch tối đa 0,00005 ở F1@val95, AUROC, AUPRC, F1 oracle) và kiểm UAM-AD seed 0 sắp thứ tự trùng tuyệt đối (lệch 0) với các lần kiểm trước ở 21 scenario. Kết quả ghi vào `D:\ClaudeWork\phase3\item10\`, không ghi đè kết quả cũ. F1 oracle xuất hai định nghĩa: point-adjust của repo (phụ thuộc thứ tự dòng, nên không so được giữa test xáo và test sắp thứ tự) và quét ngưỡng không point-adjust như các đối thủ; bảng dùng bản không point-adjust cho RE2. SN không chấm lại theo thứ tự (file test SN dùng id băm và nhãn xen kẽ; đã thử sắp theo `sessions.pkl` với checkpoint seed 0 và không giúp: AUROC 0,956 → 0,942, AUPRC 0,705 → 0,669, F1@val95 0,734 → 0,691); với SN chỉ làm lại HR cho UAM-AD. **Xếp hạng service mới** (cài bằng hàm mới trong `rca.py`, không sửa `localize_root_causes`): điểm service = trace (Eq. 14) chuẩn hoá theo node + KPI của chính service chuẩn hoá theo kênh (cả hai bằng trung vị và p95 − trung vị trên val, như `norm_sum`), HR tính trên mọi mẫu bất thường biết service lỗi; xuất thêm các biến thể chỉ-KPI, chỉ-trace và cách cũ cho bảng bỏ từng phần (ở T1, phần HR). **Đã đưa vào bảng (10-30):** RE2-OB, RE2-TT (F1@val95, AUROC, AUPRC của cả hai mô hình, F1 oracle không point-adjust) và HR@k của UAM-AD ở cả 5 dataset, ở S2–S5 và T1–T3; RE3-OB, RE3-TT giữ số đã lưu (sắp lại đổi tối đa 0,011, số sắp lại vẫn nằm trong `item10`). | 8 | Bảng B: hàng UAM-AD và UAC-AD của RE2-OB, RE2-TT; cột HR@k của UAM-AD ở cả 5 dataset. Số đo thử seed 0 và diễn biến: `working_history.md`, mục 2026-10-30 |

Việc 7, 8, 9 độc lập nhau, chạy theo khả năng máy. Nếu thiếu thời gian: chạy đủ 5 dataset cho UAM-AD, UAC-AD, AnoFusion, ART; hàng Eadro/OrEdge (chỉ tham chiếu) có thể làm ít dataset hơn.

#### Việc 7: kết quả DeepTraLog (Bảng C)

**Việc 7, UAM-AD trên DeepTraLog** (45 ô = 3 cấu hình × F01, F02, F13 × 5 seed, theo `frozen_config.json`). Trung bình ± độ lệch chuẩn (độ lệch chuẩn của quần thể, 5 seed). F1 ở ngưỡng p95 của val; "chỉ-sync" và "chỉ-async" vẫn giữ phần tái tạo log, chỉ khác nhánh trace nào bật:

| F-case | Cấu hình | F1@val | AUROC | AUPRC |
|:--|:--|--:|--:|--:|
| F01 | chỉ-sync | 0,765 ± 0,003 | 0,935 ± 0,007 | 0,830 ± 0,002 |
| F01 | chỉ-async | 0,258 ± 0,000 | 0,855 ± 0,006 | 0,345 ± 0,005 |
| F01 | đủ bộ | **0,775 ± 0,003** | **0,971 ± 0,004** | **0,859 ± 0,005** |
| F02 | chỉ-sync | 0,107 ± 0,001 | 0,664 ± 0,031 | 0,170 ± 0,014 |
| F02 | chỉ-async | 0,105 ± 0,000 | 0,803 ± 0,002 | 0,285 ± 0,002 |
| F02 | đủ bộ | 0,133 ± 0,014 | 0,803 ± 0,033 | 0,256 ± 0,037 |
| F13 | chỉ-sync | 0,852 ± 0,004 | 0,989 ± 0,000 | 0,892 ± 0,008 |
| F13 | chỉ-async | 0,434 ± 0,001 | 0,805 ± 0,002 | 0,467 ± 0,002 |
| F13 | đủ bộ | 0,866 ± 0,002 | 0,990 ± 0,001 | 0,893 ± 0,014 |

**Đọc theo quy ước ở mục 2.3:**
- **Mệnh đề 1 (đủ bộ ≥ từng nhánh lẻ): đúng ở cả 3 F-case và cả 3 chỉ số.** Trong 18 cặp (3 F-case × 3 chỉ số × 2 nhánh lẻ), đủ bộ hơn quá một độ lệch chuẩn ở 14 cặp (F1: cả 6 cặp; AUROC và AUPRC: cả hai nhánh ở F01, nhánh sync ở F02, nhánh async ở F13) và ngang ở 4 cặp: AUROC và AUPRC của F02 so với chỉ-async (0,803 so với 0,803; 0,256 so với 0,285), AUROC và AUPRC của F13 so với chỉ-sync (0,990 so với 0,989; 0,893 so với 0,892). Không cặp nào thấp hơn quá một độ lệch chuẩn.
- **Mệnh đề 2 (so các nhánh lẻ với nhau):** ở F01 và F13, chỉ-sync hơn chỉ-async ở cả ba chỉ số (F01: F1 0,765 so với 0,258, AUROC 0,935 so với 0,855, AUPRC 0,830 so với 0,345; F13: 0,852 so với 0,434, 0,989 so với 0,805, 0,892 so với 0,467). Ở F02, chỉ-async hơn chỉ-sync về AUROC (0,803 so với 0,664) và AUPRC (0,285 so với 0,170); F1 ngang và đều thấp (0,105 và 0,107). **Giải thích (đo trên đầu vào, không phải ablation):** F01 và F13 mang trễ khoảng 4 giây mà nhánh sync thấy được (thời lượng request, AUROC 0,97–1,00 so với normal cùng nhóm), còn số tin nhắn async không đổi ở các nhóm so được, nên chỉ-sync mạnh hơn dù lỗi được tiêm vào cơ chế async; F02 là F-case có số tin nhắn async đổi rõ nhất (+21%), nên chỉ-async hơn. Chưa giải thích hết vì sao chỉ-sync yếu ở F02 dù thời lượng nhóm `createOrderBatch` dài hơn [SL: 0,1–0,2 giây có thể nằm trong dải thời lượng của các API khác, vì file test trộn nhiều API; chưa kiểm].
- **Điều cần nói rõ khi viết:** (1) chỉ-sync và chỉ-async đều giữ phần tái tạo log, nên AUROC của chỉ-async (0,80–0,86) không phải thuần của nhánh async; chưa có cấu hình log-only để tách phần của log. (2) Mỗi file test trộn nhiều API (F02: 56% `createOrderBatch`, 44% `trips/left_parallel`; F01: 47% trace thuộc API không có normal), nên metric mức file lẫn tín hiệu lỗi với thành phần API [Repo]. (3) F1 ở ngưỡng p95 của val rất thấp ở F02 (≈ 0,1) dù AUROC cao hơn mức ngẫu nhiên rõ, tức ngưỡng từ val không chuyển sang test tốt, đã thấy ở các lần chạy trước. (4) **Cột `oracle_f1` do `run.py` ghi cho DeepTraLog không tin được**: tính với `point_adjust` (dòng log ghi rõ) và mâu thuẫn với AUROC (ví dụ F05 chỉ-async, một F-case nay ngoài Bảng C: AUROC 0,9994 nhưng oracle F1 chỉ 0,019). Chưa điều tra; không dùng cột này cho DeepTraLog, và không so với `oracle_f1` của GGNN vì cột đó là quét ngưỡng thường, không point-adjust.

**Việc 7, DeepTraLog GGNN** (5 seed, 30 epoch, `run_ggnn.py`). Trung bình ± độ lệch chuẩn (quần thể) qua 5 seed; ngưỡng p95 điểm val của chính GGNN:

| F-case | F1@val | AUROC | AUPRC | Precision | Recall | F1 oracle (quét ngưỡng, không point-adjust) |
|:--|--:|--:|--:|--:|--:|--:|
| F01 | 0,793 ± 0,003 | 0,936 ± 0,014 | 0,898 ± 0,006 | 0,723 | 0,879 | 0,933 |
| F02 | 0,856 ± 0,007 | 0,998 ± 0,001 | 0,981 ± 0,010 | 0,748 | 1,000 | 0,964 |
| F13 | 0,745 ± 0,008 | 0,943 ± 0,016 | 0,863 ± 0,012 | 0,699 | 0,799 | 0,880 |

**So với UAM-AD đủ bộ (cùng F-case, 5 seed):**

| F-case | F1@val UAM-AD / GGNN | AUROC UAM-AD / GGNN | AUPRC UAM-AD / GGNN |
|:--|--:|--:|--:|
| F01 | 0,775 / **0,793** | **0,971** / 0,936 | 0,859 / **0,898** |
| F02 | 0,133 / **0,856** | 0,803 / **0,998** | 0,257 / **0,981** |
| F13 | **0,866** / 0,745 | **0,990** / 0,943 | **0,893** / 0,863 |

**Kết quả không có lợi cho UAM-AD, báo đúng số đo:** GGNN thắng rõ ở F02 (mọi chỉ số, F1 chênh 0,72) và thắng F1 và AUPRC ở F01 (0,793 so với 0,775; 0,898 so với 0,859); UAM-AD thắng ở F13 (cả ba chỉ số) và AUROC của F01 (0,971 so với 0,936). Mệnh đề 1 ("đủ bộ ≥ nhánh lẻ") là so sánh nội bộ UAM-AD; so với GGNN thì UAM-AD **không** vượt trội trên DeepTraLog, nhất là ở F02. **Điều kiện không ngang nhau, phải ghi trong luận văn:** (1) GGNN chấm từng trace riêng, UAM-AD chấm cửa sổ 5 trace; (2) GGNN huấn luyện trên ≈ 65.000 trace normal của riêng nó (chia 60/10/30), UAM-AD trên `train.pkl`; (3) embedding sự kiện của GGNN do ta dựng lại từ nội dung sự kiện (TF-IDF + SVD, xem `working_history.md`, mục "DeepTraLog GGNN") vì tác giả không công bố file embedding, nên có thể mang thông tin nội dung mà UAM-AD không thấy; chưa kiểm xem phần này có làm điểm GGNN cao hơn thực tế không; (4) 30 epoch thay vì 100 của tác giả. Ngưỡng p95 của GGNN rất sát 0 (mỗi seed từ −1,3e-6 đến +3,2e-7) và R ≈ 0,003, vì Deep SVDD co hình cầu về gần một điểm; xếp hạng vẫn tốt (AUROC ≥ 0,94), nhưng điểm số tuyệt đối rất nhỏ. Cột oracle F1 của GGNN là quét ngưỡng thường, nhất quán với AUROC (AUROC 0,9998 ↔ oracle 0,994), khác cột oracle của `run.py` (xem cảnh báo ở trên).

#### Việc 8 – SUMMARY experiments between UAC-AD and UAM-AD

Tổng hợp việc 8 (UAM-AD so với UAC-AD): 5 seed (0–4) × 33 scenario × 2 mô hình = 330 lần chạy, cấu hình đóng băng (`codes/common/frozen_config.json`), luật chấm `norm_sum`. Mọi số **sinh tự động từ thư mục kết quả** `D:\ClaudeWork\phase3\item8\`, không gõ tay.

**Cách đọc.**
- **F1@val95**: F1 ở ngưỡng = phân vị 95 của điểm trên val (chỉ normal), không point-adjust, không dùng nhãn test. Đây là chỉ số chính. **AUROC, AUPRC**: không cần ngưỡng.
- **F1 oracle**: cột `oracle_f1` do `run.py` ghi: quét ngưỡng trên test lấy F1 cao nhất **và có point-adjust**. Chỉ để đối chiếu, không dùng để kết luận, không đặt cạnh F1@val95 (luôn cao hơn), và không so với cột oracle của DeepTraLog GGNN (quét ngưỡng thường). **Ngoại lệ (việc 10, 10-30): ở RE2-OB và RE2-TT cột này là bản không point-adjust**, vì point-adjust phụ thuộc thứ tự dòng của test (sau khi sắp theo thời gian nó nhảy lên khoảng 0,95); nên cột F1 oracle không so được giữa RE2 và các dataset còn lại.
- **HR@1/3/5** (xác định đúng service lỗi): chỉ có ở **UAM-AD**; cột HR của UAC-AD ghi "–". **Số HR là của việc 10 (10-30): xếp hạng trace + KPI** (điểm trace chuẩn hoá của node + điểm KPI của các kênh thuộc service; mô tả ở T1), chấm lại bằng đúng các checkpoint đã lưu, trên mọi cửa sổ bất thường có service lỗi, đủ 5 seed kể cả HR@5; thay cho HR cũ (trace thô, chỉ các cửa sổ vượt ngưỡng `raw_sum`). HR của dataset = trung bình qua các scenario có service lỗi (SN có 3 scenario hiệu năng hệ thống không gắn với một service nên không có HR). Mức ngẫu nhiên k/N: SN (12 service) 8% / 25% / 42%; RE3-OB và RE2-OB (11) 9% / 27% / 45%; RE3-TT và RE2-TT (27) 4% / 11% / 19%.
- **Việc 10 (10-30): F1@val95, AUROC, AUPRC của UAM-AD và UAC-AD ở RE2-OB và RE2-TT là số chấm lại trên test sắp theo thứ tự thời gian** (cùng checkpoint, không huấn luyện lại; test đã lưu bị xáo, và xáo làm UAM-AD kém đi ở RE2-OB). Các dataset còn lại giữ số đã lưu (RE3: sắp lại chỉ đổi tối đa 0,011; SN không có thông tin thời gian). Cột giờ/phút vẫn là thời gian của các lần chạy gốc. Bảng phụ `raw_sum` ở S2 chưa chấm lại nên RE2 vẫn theo thứ tự xáo.
- Mỗi dòng của dataset = **trung bình qua các scenario**; "a±b" = trung bình±độ lệch chuẩn tổng thể (`np.std`) qua 5 seed. Cột cuối là thời gian thật của lần chạy (gồm nạp dữ liệu, huấn luyện, đánh giá, và RCA với UAM-AD): S2 là **giờ** trung bình mỗi seed, S3 là **phút** trung bình mỗi lần chạy, S4 là **phút** của lần chạy đó. Số thập phân dùng dấu phẩy.

##### S1. Chi phí chạy: UAM-AD so với UAC-AD

**Tổng giờ chạy (330 lần chạy thành công, tính từ log): UAM-AD 34,7 giờ, UAC-AD 29,9 giờ, tỉ lệ UAM-AD / UAC-AD = 34,7 / 29,9 ≈ 1,16.**

Số 1,16 là tỉ lệ thô và bị nhiễu bởi một ô của UAC-AD: rcaeval_re2_tt / `socket` / seed 2 mất 12137 giây so với trung vị 1430 giây của các seed còn lại (máy bị dùng cùng lúc). Thay ô đó bằng trung vị thì UAC-AD ≈ 26,9 giờ và **tỉ lệ UAM-AD / UAC-AD = 34,7 / 26,9 ≈ 1,29**, phản ánh chi phí thêm nhánh trace tốt hơn. Khi viết luận văn ghi cả hai con số, kèm lý do.

Theo dataset (giờ = tổng các lần chạy; "giây/epoch" = trung vị `Train time/epoch` trong `info_score.txt`, không chịu ảnh hưởng của việc nạp dữ liệu hay RCA; "epoch chạy" = số epoch thực chạy trung bình mỗi lần chạy, vì dừng sớm làm giờ thấp đi):

| Dataset | Số lần chạy/mô hình | Epoch tối đa | Epoch chạy UAC-AD / UAM-AD | UAC-AD giờ | UAM-AD giờ | UAM / UAC (giờ) | UAC-AD giây/epoch | UAM-AD giây/epoch | UAM / UAC (giây/epoch) |
|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|
| SN | 60 | 50 | 37,4 / 50,0 | 0,29 | 0,36 | 1,25 | 0,27 | 0,28 | 1,02 |
| RE3-TT | 20 | 5 | 4,8 / 5,0 | 3,10 | 4,23 | 1,37 | 88,0 | 121,9 | 1,39 |
| RE3-OB | 25 | 5 | 4,4 / 5,0 | 2,45 | 3,18 | 1,30 | 63,9 | 75,4 | 1,18 |
| RE2-OB | 30 | 5 | 5,0 / 5,0 | 9,68 | 11,16 | 1,15 | 187,2 | 216,7 | 1,16 |
| RE2-TT | 30 | 5 | 4,9 / 5,0 | 14,34 | 15,77 | 1,10 | 220,0 | 319,6 | 1,45 |
| **Tổng** | 165 | | | **29,85** | **34,70** | **1,16** | | | |

RE2-TT sau khi thay ô nhiễu: UAC-AD 11,36 giờ, tỉ lệ UAM / UAC (giờ) = 1,39 (thô: 1,10), khớp với tỉ lệ giây/epoch của RE2-TT.

Giờ chạy theo seed (tổng các scenario của dataset trong seed đó; tỉ lệ = UAM-AD / UAC-AD):

| Dataset | Mô hình | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Tổng giờ |
|:--|:--|--:|--:|--:|--:|--:|--:|
| SN | UAC-AD | 0,05 | 0,07 | 0,07 | 0,03 | 0,06 | 0,29 |
| SN | UAM-AD | 0,07 | 0,08 | 0,08 | 0,07 | 0,06 | 0,36 |
| SN | tỉ lệ UAM/UAC | 1,24 | 1,08 | 1,21 | 2,24 | 1,03 | 1,25 |
| RE3-TT | UAC-AD | 0,54 | 0,63 | 0,65 | 0,67 | 0,60 | 3,10 |
| RE3-TT | UAM-AD | 0,80 | 0,89 | 0,90 | 0,82 | 0,82 | 4,23 |
| RE3-TT | tỉ lệ UAM/UAC | 1,48 | 1,41 | 1,38 | 1,22 | 1,37 | 1,37 |
| RE3-OB | UAC-AD | 0,45 | 0,45 | 0,46 | 0,58 | 0,51 | 2,45 |
| RE3-OB | UAM-AD | 0,60 | 0,67 | 0,67 | 0,62 | 0,62 | 3,18 |
| RE3-OB | tỉ lệ UAM/UAC | 1,35 | 1,50 | 1,45 | 1,07 | 1,20 | 1,30 |
| RE2-OB | UAC-AD | 1,94 | 1,92 | 1,97 | 2,04 | 1,79 | 9,68 |
| RE2-OB | UAM-AD | 2,11 | 2,38 | 2,35 | 2,17 | 2,16 | 11,16 |
| RE2-OB | tỉ lệ UAM/UAC | 1,09 | 1,24 | 1,19 | 1,06 | 1,20 | 1,15 |
| RE2-TT | UAC-AD | 2,34 | 2,38 | 5,27 | 2,44 | 1,91 | 14,34 |
| RE2-TT | UAM-AD | 3,07 | 3,24 | 3,35 | 3,02 | 3,08 | 15,77 |
| RE2-TT | tỉ lệ UAM/UAC | 1,31 | 1,36 | 0,64 | 1,24 | 1,61 | 1,10 |
| **Cả 5 dataset** | UAC-AD | 5,32 | 5,45 | 8,43 | 5,77 | 4,88 | **29,85** |
| **Cả 5 dataset** | UAM-AD | 6,65 | 7,25 | 7,36 | 6,70 | 6,74 | **34,70** |
| **Cả 5 dataset** | tỉ lệ UAM/UAC | 1,25 | 1,33 | 0,87 | 1,16 | 1,38 | **1,16** |

Tỉ lệ 0,64 (RE2-TT) và 0,87 (cả 5 dataset) ở seed 2 là do ô nhiễu; thay bằng trung vị thì tỉ lệ cả 5 dataset ở seed 2 là 1,35.

Ghi chú về chi phí:
- Chi phí tập trung ở hai dataset RE2: RE2-OB + RE2-TT chiếm 80% tổng giờ của UAC-AD và 78% của UAM-AD (cửa sổ nhiều, RE2-TT có 383 kênh KPI). SN chỉ vài phút mỗi seed.
- **Số epoch thực chạy không bằng nhau, nên giờ không so thẳng được ở SN**: UAM-AD luôn chạy đủ 50 epoch, còn UAC-AD chỉ chạy trung bình 19,0 epoch (seed 0) và 18,0 epoch (seed 3), đúng hai seed UAC-AD sập (AUROC 0,551 và 0,564), vì dừng sớm (patience 15). Ở RE3-OB UAC-AD cũng dừng ở 4/5 epoch ở seed 0–2, và RE3-TT ở seed 0. Việc dừng sớm trùng với việc sập ở SN là một quan sát, chưa phải chẩn đoán nguyên nhân.
- Số tham số, thời gian suy luận và RAM không đo ở đây: xem Bảng E (mục 4).

##### S2. Kết quả 5 seed theo dataset (`norm_sum`, trung bình qua scenario, trung bình±độ lệch chuẩn qua 5 seed)

| Dataset | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | giờ/seed |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SN | UAC-AD | 0,523±0,185 | 0,795±0,194 | 0,594±0,173 | 0,758±0,043 | – | – | – | 0,06 |
| SN | UAM-AD | 0,728±0,004 | 0,956±0,002 | 0,716±0,012 | 0,849±0,009 | 0,185±0,017 | 0,347±0,055 | 0,397±0,036 | 0,07 |
| SN | Δ UAM−UAC | +0,205 (+39,1%) | +0,161 (+20,3%) | +0,123 (+20,6%) | +0,091 (+12,0%) | – | – | – | – |
| RE3-TT | UAC-AD | 0,540±0,006 | 0,745±0,008 | 0,432±0,010 | 0,645±0,006 | – | – | – | 0,62 |
| RE3-TT | UAM-AD | 0,513±0,002 | 0,702±0,003 | 0,347±0,003 | 0,633±0,004 | 0,251±0,006 | 0,332±0,017 | 0,417±0,022 | 0,85 |
| RE3-TT | Δ UAM−UAC | −0,026 (−4,9%) | −0,042 (−5,7%) | −0,086 (−19,8%) | −0,012 (−1,8%) | – | – | – | – |
| RE3-OB | UAC-AD | 0,647±0,007 | 0,858±0,001 | 0,696±0,004 | 0,761±0,002 | – | – | – | 0,49 |
| RE3-OB | UAM-AD | 0,560±0,019 | 0,835±0,005 | 0,615±0,012 | 0,755±0,007 | 0,824±0,024 | 0,902±0,022 | 0,959±0,019 | 0,64 |
| RE3-OB | Δ UAM−UAC | −0,087 (−13,5%) | −0,023 (−2,7%) | −0,080 (−11,5%) | −0,005 (−0,7%) | – | – | – | – |
| RE2-OB | UAC-AD | 0,611±0,084 | 0,880±0,026 | 0,750±0,057 | 0,713±0,039 | – | – | – | 1,94 |
| RE2-OB | UAM-AD | 0,681±0,076 | 0,917±0,016 | 0,823±0,025 | 0,764±0,020 | 0,759±0,022 | 0,949±0,008 | 0,976±0,004 | 2,23 |
| RE2-OB | Δ UAM−UAC | +0,070 (+11,4%) | +0,037 (+4,2%) | +0,073 (+9,7%) | +0,051 (+7,1%) | – | – | – | – |
| RE2-TT | UAC-AD | 0,345±0,026 | 0,670±0,007 | 0,431±0,021 | 0,533±0,003 | – | – | – | 2,87 |
| RE2-TT | UAM-AD | 0,597±0,007 | 0,809±0,002 | 0,680±0,006 | 0,624±0,006 | 0,721±0,027 | 0,871±0,029 | 0,908±0,032 | 3,15 |
| RE2-TT | Δ UAM−UAC | +0,251 (+72,8%) | +0,139 (+20,7%) | +0,249 (+57,7%) | +0,092 (+17,3%) | – | – | – | – |

HR@1/3/5 là xếp hạng trace + KPI của việc 10, 5 seed. Hai dòng RE2-OB, RE2-TT: số của việc 10 (test sắp theo thời gian; F1 oracle không point-adjust). Dòng Δ = trung bình qua 5 seed của (UAM-AD − UAC-AD) tính theo từng seed. Trong ngoặc: % so với giá trị trung bình 5 seed của UAC-AD ở cùng cột (Δ chia cho UAC-AD; dương = UAM-AD cao hơn).

Bảng phụ `raw_sum` (luật chấm có cổng, F1@val95 / AUROC / AUPRC; nguồn cùng các lần chạy; HR@k của luật này không lưu). **Hai dòng RE2-OB, RE2-TT vẫn là số trên test đã xáo (chưa chấm lại), nên không đặt cạnh cột `norm_sum` mới của S2:**

| Dataset | UAC-AD | UAM-AD |
|:--|:--|:--|
| SN | 0,612±0,061 / 0,890±0,054 / 0,621±0,041 | 0,743±0,039 / 0,959±0,002 / 0,720±0,015 |
| RE3-TT | 0,540±0,006 / 0,753±0,009 / 0,440±0,010 | 0,532±0,003 / 0,747±0,004 / 0,420±0,006 |
| RE3-OB | 0,659±0,003 / 0,864±0,002 / 0,716±0,002 | 0,662±0,002 / 0,865±0,003 / 0,713±0,005 |
| RE2-OB | 0,523±0,079 / 0,872±0,024 / 0,709±0,060 | 0,519±0,051 / 0,892±0,021 / 0,738±0,048 |
| RE2-TT | 0,256±0,002 / 0,678±0,003 / 0,450±0,001 | 0,412±0,055 / 0,732±0,020 / 0,547±0,035 |

##### S3. Kết quả từng scenario trong từng dataset (trung bình±độ lệch chuẩn qua 5 seed)

**SN** (12 scenario)

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút/lần chạy |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `Code_Stop_MediaService` | UAC-AD | 0,680±0,117 | 0,891±0,111 | 0,784±0,138 | 0,899±0,018 | – | – | – | 0,4 |
| `Code_Stop_MediaService` | UAM-AD | 0,759±0,018 | 0,966±0,002 | 0,827±0,010 | 0,911±0,012 | 0,708±0,038 | 0,846±0,043 | 0,867±0,041 | 0,4 |
| `Code_Stop_TextService` | UAC-AD | 0,694±0,061 | 0,890±0,105 | 0,832±0,087 | 0,911±0,009 | – | – | – | 0,3 |
| `Code_Stop_TextService` | UAM-AD | 0,699±0,004 | 0,976±0,002 | 0,903±0,007 | 0,929±0,000 | 0,492±0,163 | 0,728±0,180 | 0,810±0,158 | 0,4 |
| `Code_Stop_UserService` | UAC-AD | 0,722±0,016 | 0,934±0,053 | 0,882±0,055 | 0,903±0,011 | – | – | – | 0,3 |
| `Code_Stop_UserService` | UAM-AD | 0,748±0,006 | 0,983±0,001 | 0,947±0,003 | 0,894±0,000 | 0,305±0,266 | 0,742±0,119 | 0,800±0,106 | 0,4 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAC-AD | 0,623±0,223 | 0,882±0,118 | 0,641±0,160 | 0,671±0,185 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAM-AD | 0,803±0,025 | 0,965±0,002 | 0,751±0,020 | 0,833±0,000 | 0,160±0,136 | 0,360±0,049 | 0,400±0,063 | 0,4 |
| `DB_Redis_CacheLimit_SocialGraph` | UAC-AD | 0,711±0,099 | 0,893±0,127 | 0,870±0,134 | 0,883±0,021 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_SocialGraph` | UAM-AD | 0,857±0,000 | 0,984±0,001 | 0,835±0,034 | 0,891±0,017 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 | 0,4 |
| `DB_Redis_CacheLimit_UserTimeline` | UAC-AD | 0,319±0,262 | 0,684±0,279 | 0,336±0,195 | 0,613±0,065 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_UserTimeline` | UAM-AD | 0,556±0,000 | 0,938±0,004 | 0,608±0,057 | 0,738±0,028 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 | 0,3 |
| `Perf_CPU_Contention` | UAC-AD | 0,517±0,234 | 0,848±0,139 | 0,546±0,167 | 0,552±0,113 | – | – | – | 0,3 |
| `Perf_CPU_Contention` | UAM-AD | 0,775±0,012 | 0,962±0,004 | 0,711±0,036 | 0,841±0,015 | – | – | – | 0,4 |
| `Perf_Disk_IO_Stress` | UAC-AD | 0,512±0,150 | 0,819±0,148 | 0,443±0,145 | 0,466±0,089 | – | – | – | 0,3 |
| `Perf_Disk_IO_Stress` | UAM-AD | 0,651±0,013 | 0,912±0,004 | 0,460±0,025 | 0,727±0,000 | – | – | – | 0,4 |
| `Perf_Network_Loss` | UAC-AD | 0,522±0,252 | 0,814±0,205 | 0,643±0,342 | 0,673±0,083 | – | – | – | 0,3 |
| `Perf_Network_Loss` | UAM-AD | 0,879±0,022 | 0,983±0,002 | 0,790±0,007 | 0,935±0,021 | – | – | – | 0,4 |
| `Svc_Kill_Media` | UAC-AD | 0,393±0,335 | 0,635±0,370 | 0,425±0,253 | 0,889±0,000 | – | – | – | 0,3 |
| `Svc_Kill_Media` | UAM-AD | 0,750±0,000 | 0,963±0,004 | 0,704±0,050 | 0,889±0,000 | 0,000±0,000 | 0,450±0,367 | 0,650±0,339 | 0,3 |
| `Svc_Kill_SocialGraph` | UAC-AD | 0,206±0,183 | 0,590±0,317 | 0,237±0,124 | 0,750±0,000 | – | – | – | 0,3 |
| `Svc_Kill_SocialGraph` | UAM-AD | 0,525±0,040 | 0,899±0,009 | 0,427±0,035 | 0,733±0,033 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 | 0,3 |
| `Svc_Kill_UserTimeline` | UAC-AD | 0,379±0,316 | 0,663±0,361 | 0,484±0,301 | 0,889±0,000 | – | – | – | 0,3 |
| `Svc_Kill_UserTimeline` | UAM-AD | 0,733±0,033 | 0,946±0,005 | 0,632±0,012 | 0,871±0,036 | 0,000±0,000 | 0,000±0,000 | 0,050±0,100 | 0,3 |

**RE3-TT** (4 scenario)

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút/lần chạy |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,542±0,008 | 0,759±0,012 | 0,444±0,011 | 0,653±0,005 | – | – | – | 9,3 |
| `f1` | UAM-AD | 0,520±0,002 | 0,718±0,003 | 0,358±0,003 | 0,644±0,003 | 0,309±0,007 | 0,383±0,016 | 0,456±0,027 | 12,7 |
| `f2` | UAC-AD | 0,535±0,005 | 0,731±0,005 | 0,425±0,008 | 0,642±0,008 | – | – | – | 9,2 |
| `f2` | UAM-AD | 0,508±0,003 | 0,688±0,003 | 0,341±0,004 | 0,628±0,005 | 0,281±0,005 | 0,365±0,016 | 0,431±0,023 | 12,6 |
| `f3` | UAC-AD | 0,551±0,009 | 0,764±0,014 | 0,442±0,014 | 0,648±0,008 | – | – | – | 9,3 |
| `f3` | UAM-AD | 0,518±0,003 | 0,721±0,003 | 0,353±0,004 | 0,638±0,004 | 0,131±0,004 | 0,219±0,024 | 0,361±0,021 | 12,8 |
| `f4` | UAC-AD | 0,531±0,007 | 0,725±0,006 | 0,418±0,008 | 0,637±0,005 | – | – | – | 9,2 |
| `f4` | UAM-AD | 0,508±0,002 | 0,682±0,005 | 0,334±0,003 | 0,623±0,003 | 0,282±0,014 | 0,361±0,016 | 0,420±0,027 | 12,6 |

**RE3-OB** (5 scenario)

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút/lần chạy |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,848±0,004 | 0,961±0,001 | 0,905±0,003 | 0,915±0,002 | – | – | – | 5,9 |
| `f1` | UAM-AD | 0,754±0,023 | 0,938±0,005 | 0,816±0,016 | 0,907±0,007 | 0,383±0,010 | 0,632±0,009 | 0,883±0,012 | 7,6 |
| `f2` | UAC-AD | 0,888±0,007 | 0,967±0,001 | 0,916±0,002 | 0,945±0,000 | – | – | – | 5,7 |
| `f2` | UAM-AD | 0,848±0,024 | 0,950±0,004 | 0,831±0,017 | 0,942±0,003 | 0,959±0,023 | 0,980±0,008 | 0,987±0,009 | 7,5 |
| `f3` | UAC-AD | 0,492±0,032 | 0,825±0,004 | 0,515±0,012 | 0,664±0,005 | – | – | – | 5,9 |
| `f3` | UAM-AD | 0,276±0,035 | 0,770±0,010 | 0,381±0,017 | 0,658±0,009 | 0,958±0,007 | 0,988±0,002 | 0,992±0,002 | 7,7 |
| `f4` | UAC-AD | 0,157±0,007 | 0,576±0,003 | 0,250±0,002 | 0,365±0,006 | – | – | – | 6,1 |
| `f4` | UAM-AD | 0,160±0,005 | 0,575±0,003 | 0,241±0,002 | 0,361±0,011 | 0,864±0,104 | 0,929±0,103 | 0,947±0,090 | 7,8 |
| `f5` | UAC-AD | 0,852±0,004 | 0,959±0,001 | 0,892±0,003 | 0,915±0,002 | – | – | – | 5,8 |
| `f5` | UAM-AD | 0,761±0,016 | 0,940±0,004 | 0,807±0,011 | 0,909±0,007 | 0,956±0,023 | 0,981±0,001 | 0,987±0,001 | 7,6 |

**RE2-OB** (6 scenario)

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút/lần chạy |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,507±0,068 | 0,819±0,033 | 0,561±0,097 | 0,571±0,030 | – | – | – | 19,4 |
| `cpu` | UAM-AD | 0,624±0,067 | 0,895±0,027 | 0,752±0,045 | 0,655±0,051 | 0,861±0,033 | 0,971±0,013 | 0,989±0,002 | 22,3 |
| `delay` | UAC-AD | 0,664±0,104 | 0,917±0,042 | 0,801±0,090 | 0,714±0,093 | – | – | – | 19,3 |
| `delay` | UAM-AD | 0,723±0,087 | 0,956±0,014 | 0,900±0,022 | 0,840±0,012 | 0,543±0,050 | 0,918±0,036 | 0,982±0,012 | 22,3 |
| `disk` | UAC-AD | 0,756±0,092 | 0,996±0,001 | 0,994±0,002 | 0,988±0,000 | – | – | – | 19,3 |
| `disk` | UAM-AD | 0,818±0,085 | 0,997±0,001 | 0,995±0,001 | 0,991±0,000 | 0,983±0,002 | 0,988±0,002 | 0,992±0,001 | 22,2 |
| `loss` | UAC-AD | 0,651±0,075 | 0,921±0,022 | 0,830±0,047 | 0,744±0,043 | – | – | – | 19,3 |
| `loss` | UAM-AD | 0,744±0,084 | 0,945±0,015 | 0,873±0,026 | 0,804±0,025 | 0,635±0,035 | 0,913±0,020 | 0,951±0,015 | 22,3 |
| `mem` | UAC-AD | 0,747±0,097 | 0,957±0,023 | 0,891±0,044 | 0,812±0,045 | – | – | – | 19,3 |
| `mem` | UAM-AD | 0,772±0,096 | 0,971±0,013 | 0,917±0,023 | 0,841±0,025 | 0,680±0,056 | 0,957±0,025 | 0,980±0,013 | 22,2 |
| `socket` | UAC-AD | 0,343±0,080 | 0,673±0,040 | 0,427±0,070 | 0,448±0,033 | – | – | – | 19,5 |
| `socket` | UAM-AD | 0,405±0,045 | 0,741±0,029 | 0,505±0,037 | 0,450±0,020 | 0,855±0,071 | 0,945±0,035 | 0,962±0,024 | 22,7 |

**RE2-TT** (6 scenario)

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút/lần chạy |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,080±0,007 | 0,484±0,005 | 0,186±0,002 | 0,338±0,002 | – | – | – | 23,1 |
| `cpu` | UAM-AD | 0,408±0,022 | 0,688±0,012 | 0,466±0,030 | 0,424±0,009 | 0,836±0,035 | 0,968±0,024 | 0,983±0,009 | 32,0 |
| `delay` | UAC-AD | 0,197±0,111 | 0,647±0,017 | 0,298±0,063 | 0,417±0,007 | – | – | – | 23,1 |
| `delay` | UAM-AD | 0,692±0,012 | 0,851±0,014 | 0,790±0,010 | 0,767±0,000 | 0,714±0,046 | 0,907±0,041 | 0,964±0,042 | 31,7 |
| `disk` | UAC-AD | 0,756±0,032 | 0,945±0,022 | 0,902±0,036 | 0,880±0,023 | – | – | – | 21,7 |
| `disk` | UAM-AD | 0,765±0,008 | 0,924±0,003 | 0,859±0,003 | 0,813±0,023 | 0,986±0,001 | 0,989±0,001 | 0,991±0,001 | 31,2 |
| `loss` | UAC-AD | 0,084±0,021 | 0,476±0,003 | 0,193±0,003 | 0,346±0,003 | – | – | – | 22,9 |
| `loss` | UAM-AD | 0,512±0,011 | 0,727±0,010 | 0,589±0,011 | 0,518±0,009 | 0,442±0,035 | 0,699±0,060 | 0,788±0,061 | 31,8 |
| `mem` | UAC-AD | 0,845±0,006 | 0,964±0,005 | 0,812±0,045 | 0,870±0,014 | – | – | – | 21,6 |
| `mem` | UAM-AD | 0,755±0,025 | 0,957±0,008 | 0,853±0,021 | 0,773±0,029 | 0,643±0,100 | 0,755±0,101 | 0,783±0,096 | 30,5 |
| `socket` | UAC-AD | 0,111±0,022 | 0,503±0,009 | 0,195±0,004 | 0,344±0,008 | – | – | – | 59,6 |
| `socket` | UAM-AD | 0,449±0,006 | 0,704±0,001 | 0,522±0,007 | 0,451±0,006 | 0,705±0,106 | 0,907±0,040 | 0,939±0,039 | 32,1 |

Ghi chú cho bảng từng scenario:
- Cột cuối là **phút** của một lần chạy, trung bình qua 5 seed.
- Scenario không có service lỗi (SN: `Perf_CPU_Contention`, `Perf_Disk_IO_Stress`, `Perf_Network_Loss`) có cột HR ghi "–".
- **HR của từng scenario SN rất nhiễu** vì chỉ có vài cửa sổ bất thường có service lỗi: `Svc_Kill_*` 3–4 cửa sổ, `DB_Redis_*` 8–10, `Code_Stop_*` 38–39. Giá trị 0,000 hoặc 1,000 ở `Svc_Kill_*` và `DB_Redis_*` không đáng tin; HR trung bình theo scenario của SN bị sáu scenario nhỏ này (`Svc_Kill_*`, `DB_Redis_*`) kéo xuống (xem T1, phần HR).

##### S5. Điều số liệu trên đỡ được, và cách tái tạo

Chỉ những điều sau (số lấy từ S1 và S2):
- **RE2-TT: UAM-AD hơn rõ** (F1 +0,251, AUROC +0,139, AUPRC +0,249; hơn ở cả 5 seed và cả ba chỉ số), với chi phí thêm cỡ 1,4 lần.
- **SN: UAM-AD ổn định, UAC-AD sập ở seed 0 và 3.** Mức hơn trung bình chủ yếu do hai seed sập; chỉ tính các seed không sập (1, 2, 4) thì AUROC, AUPRC ngang nhau (xem S4, từng seed).
- **RE2-OB:** hơn ở trung bình (F1 +0,070, AUROC +0,037, AUPRC +0,073) nhưng không đồng đều theo seed: AUROC và AUPRC hơn ở 4/5 seed, F1 hơn ở 3/5 seed, và seed 4 UAM-AD thấp hơn ở cả ba chỉ số (F1 −0,189). Độ lệch chuẩn F1 khoảng 0,08 ở cả hai mô hình (cả hai có seed "nhảy"), nên F1 chưa phân biệt được.
- **RE3-OB, RE3-TT: UAM-AD thấp hơn UAC-AD dưới `norm_sum`**, hệ thống theo seed; dưới `raw_sum` chênh lệch gần 0 (bảng phụ ở S2; cơ chế ở S6).
- **F1 oracle** (có point-adjust; ở RE2 là bản không point-adjust) cùng chiều với AUROC nhưng khoảng cách nhỏ hơn nhiều và bị thổi phồng: UAC-AD ở SN seed 0 có AUROC 0,551 (gần ngẫu nhiên) mà F1 oracle vẫn 0,706. Chỉ để đối chiếu, không dùng để kết luận.
- **HR@k** chỉ có ở UAM-AD (xếp hạng trace + KPI, việc 10): RE2-OB, RE3-OB, RE2-TT hơn hẳn ngẫu nhiên (HR@1 0,759; 0,824; 0,721); RE3-TT hơn ngẫu nhiên (HR@1 0,251 so với 4%) nhưng thấp hơn đoán cố định; SN hơn ngẫu nhiên ở HR@1 (0,185 so với 8%) và HR@3 (0,347 so với 25%) nhưng HR@5 (0,397) thấp hơn ngẫu nhiên (42%). UAC-AD không có RCA nên không có hàng so sánh HR giữa hai mô hình (chưa chạy RCA cho UAC-AD).

**Cách tái tạo.** Hai script nằm ngoài vùng git, cạnh kết quả: `D:\ClaudeWork\phase3\item8\summary_tools\load_step8.py` (đọc 330 lần chạy) và `build_summary.py` (sinh `summary_part1.md` đến `summary_part4.md`). HR@k tính lại từ `rca_results.json` khớp với các cột HR@k của S2 (kiểm trên cả 5 dataset). Mọi tổ hợp (mô hình, dataset, scenario) có đúng seed 0–4.

##### S6. Luật chấm: `norm_sum` hay `raw_sum`

- **`raw_sum`** là luật gốc: điểm = `fusion_loss`, tức `log_kpi_loss + g·trace_d + λ·g` (trace nhân với cổng `g`; SN còn cộng số hạng phạt hoạt động). **`norm_sum`** được thêm khi làm nhánh async trên DeepTraLog: mỗi số hạng (`log_kpi_loss`, `trace_dis`, `trace_err`) được chuẩn hóa theo trung vị và độ lệch p95 của val rồi cộng với trọng số 1; `trace_dis` và `trace_err` **không nhân cổng**, và không có số hạng phạt hoạt động. Cấu hình đóng băng chọn `norm_sum` cho mọi dataset. Cả hai điểm đều có trong `info_score.txt` của mọi lần chạy (cùng checkpoint).
- **Vì sao không thể đổi sang `raw_sum` làm luật duy nhất: DeepTraLog.** Chấm 75 ô DeepTraLog (bước 7; F05, F09 đã chạy nhưng từ 10-31 không còn trong Bảng C) bằng `raw_sum`, ở 4 trong 5 F-case `raw_sum` kém hơn `norm_sum` (nhiều ở F01, F13, F05; ít hơn ở F02), chỉ F09 là tốt hơn (5 seed, trung bình±độ lệch chuẩn):

| F-case | Cấu hình | `norm_sum` F1 / AUROC / AUPRC | `raw_sum` F1 / AUROC / AUPRC |
|:--|:--|:--|:--|
| F01 | chỉ-sync | 0,765±0,003 / 0,935±0,007 / 0,830±0,002 | 0,235±0,042 / 0,885±0,008 / 0,429±0,031 |
| F01 | chỉ-async | 0,258±0,000 / 0,855±0,006 / 0,345±0,005 | 0,208±0,014 / 0,838±0,004 / 0,309±0,007 |
| F01 | đủ bộ | 0,775±0,003 / 0,971±0,004 / 0,859±0,005 | 0,151±0,022 / 0,861±0,014 / 0,339±0,031 |
| F02 | chỉ-sync | 0,107±0,001 / 0,664±0,031 / 0,170±0,014 | 0,000±0,000 / 0,713±0,007 / 0,189±0,004 |
| F02 | chỉ-async | 0,105±0,000 / 0,803±0,002 / 0,285±0,002 | 0,018±0,037 / 0,733±0,008 / 0,212±0,019 |
| F02 | đủ bộ | 0,133±0,014 / 0,803±0,033 / 0,256±0,037 | 0,020±0,031 / 0,726±0,010 / 0,196±0,007 |
| F13 | chỉ-sync | 0,852±0,004 / 0,989±0,000 / 0,892±0,008 | 0,250±0,022 / 0,725±0,007 / 0,340±0,043 |
| F13 | chỉ-async | 0,434±0,001 / 0,805±0,002 / 0,467±0,002 | 0,204±0,003 / 0,696±0,019 / 0,249±0,004 |
| F13 | đủ bộ | 0,866±0,002 / 0,990±0,001 / 0,893±0,014 | 0,263±0,001 / 0,832±0,009 / 0,351±0,016 |
| F05 | chỉ-sync | 0,903±0,003 / 0,994±0,000 / 0,952±0,001 | 0,064±0,001 / 0,554±0,126 / 0,230±0,033 |
| F05 | chỉ-async | 0,905±0,001 / 0,999±0,000 / 0,990±0,000 | 0,007±0,009 / 0,296±0,004 / 0,129±0,004 |
| F05 | đủ bộ | 0,910±0,003 / 0,994±0,000 / 0,940±0,007 | 0,084±0,016 / 0,489±0,024 / 0,207±0,008 |
| F09 | chỉ-sync | 0,216±0,006 / 0,923±0,003 / 0,438±0,009 | 0,425±0,107 / 0,958±0,004 / 0,571±0,025 |
| F09 | chỉ-async | 0,241±0,066 / 0,951±0,001 / 0,533±0,006 | 0,400±0,016 / 0,960±0,001 / 0,582±0,007 |
| F09 | đủ bộ | 0,187±0,047 / 0,941±0,004 / 0,494±0,015 | 0,560±0,025 / 0,964±0,000 / 0,612±0,006 |

**Ngoại lệ:** F09, nơi `raw_sum` tốt hơn. Nguyên nhân `raw_sum` hỏng ở DeepTraLog chưa được chẩn đoán; chỉ biết đó là số đo. Vì kết luận chính về async (bước 7) dựa trên `norm_sum`, dùng `raw_sum` cho bước 8 và `norm_sum` cho bước 7 nghĩa là mỗi dataset một luật chấm, chọn sau khi đã thấy kết quả test. Không làm vậy.

- **Hai điều cần nói thật khi viết luận văn.** (1) Chọn `norm_sum` cho DeepTraLog được làm khi đã nhìn kết quả (mục 2.2, điểm 2), nên nó là một lựa chọn của mô hình đề xuất, không phải luật được định trước hoàn toàn; ghi rõ. (2) Phần giảm ở RE3 dưới `norm_sum` **không phải nhiễu nhẹ của mã trace**: nó hệ thống (RE3-TT: 20/20 cặp scenario × seed đều thấp hơn ở cả ba chỉ số) và có cơ chế: dưới `norm_sum` cổng không còn chặn số hạng trace, mà `trace_dis` riêng lẻ ở RE3-TT có AUROC 0,370 (đảo chiều) và ở RE3-OB 0,544 (gần ngẫu nhiên), nên cộng vào chỉ làm nhiễu điểm. Dưới `raw_sum` cổng làm đúng việc của nó.

- **So sánh UAM-AD − UAC-AD dưới hai luật (trung bình qua 5 seed của hiệu theo từng seed; F1 / AUROC / AUPRC; trong ngoặc: % so với giá trị trung bình 5 seed của UAC-AD dưới cùng luật):**

| Dataset | `norm_sum` (luật đóng băng) | `raw_sum` | Hai luật có cùng dấu? |
|:--|:--|:--|:--|
| SN | +0,205 (+39,1%) / +0,161 (+20,3%) / +0,123 (+20,6%) | +0,131 (+21,4%) / +0,068 (+7,7%) / +0,098 (+15,8%) | cùng dương; phần lớn do UAC-AD sập ở 2 seed; `raw_sum` cho UAC-AD điểm cao hơn nhiều nên mức hơn nhỏ hơn (chưa tách được nguyên nhân: số hạng phạt hoạt động hay phép cắt dưới median) |
| RE3-TT | −0,026 (−4,9%) / −0,042 (−5,7%) / −0,086 (−19,8%) | −0,007 (−1,4%) / −0,005 (−0,7%) / −0,020 (−4,5%) | cùng âm; dưới `raw_sum` nhỏ nhưng nhất quán (AUPRC âm ở 20/20 cặp) |
| RE3-OB | −0,087 (−13,5%) / −0,023 (−2,7%) / −0,080 (−11,5%) | +0,003 (+0,4%) / +0,001 (+0,2%) / −0,003 (−0,4%) | dưới `raw_sum` ≈ 0 (không còn mức giảm của `norm_sum`) |
| RE2-OB | +0,070 (+11,4%) / +0,037 (+4,2%) / +0,073 (+9,7%) | −0,004 (−0,7%) / +0,020 (+2,2%) / +0,029 (+4,1%) | cùng dương ở AUROC, AUPRC (dưới `norm_sum` hơn ở 4/5 seed, seed 4 âm); F1 không phân biệt được. `raw_sum` ở hàng này chưa chấm lại theo thứ tự thời gian |
| RE2-TT | +0,251 (+72,8%) / +0,139 (+20,7%) / +0,249 (+57,7%) | +0,156 (+60,8%) / +0,054 (+7,9%) / +0,098 (+21,7%) | cùng dương, rõ. `raw_sum` ở hàng này chưa chấm lại theo thứ tự thời gian |

**Đề xuất cách trình bày:** giữ `norm_sum` làm luật chính duy nhất cho mọi dataset (đã đóng băng, đồng nhất với bước 7), in `raw_sum` thành bảng song song (S2 đã có), và chỉ khẳng định điều đúng ở cả hai luật: UAM-AD hơn UAC-AD ở RE2-TT, RE2-OB (AUROC, AUPRC) và ổn định hơn ở SN; không có lợi ở RE3 (≈ 0 hoặc thấp hơn tùy luật, vì nhánh trace không mang tín hiệu hữu ích ở lỗi mức mã). Mức hơn ở SN nên nói là độ ổn định, không phải điểm cao hơn. Một biến thể `norm_sum` có nhân cổng cho số hạng trace có thể làm như ablation, nhưng phải ghi là post-hoc và không thay luật chính; chưa làm.

#### Việc 9 – SUMMARY experiments between UAM-AD and the other models (ART, AnoFusion, Eadro, OrEdge)

Tổng hợp việc 9 (UAM-AD so với ART, AnoFusion, Eadro, OrEdge). Hai bảng T1 và T2 **sinh tự động** bởi `summary_tools/item9_tables.py` từ cùng dữ liệu với khối "Kết quả 5 seed" ở Phụ lục B; phần chữ viết tay dùng số của hai bảng này.

**Cách đọc.**
- T1 là bảng "Theo dataset (5 seed)" của Phụ lục B, đã bỏ cột "seed". Ô "a±b" = trung bình±độ lệch chuẩn tổng thể (`np.std`) qua 5 seed của trung bình theo scenario.
- **AnoFusion chỉ có 1 seed (seed 0)**, vì mỗi seed tốn 298,5 phút không kể RE3-TT (riêng RE2-TT 220 phút); không sao chép seed 0 sang seed khác. **Theo yêu cầu của người dùng, độ lệch chuẩn của AnoFusion ghi ±0,000 trong T1; đó là quy ước trình bày, không phải số đo.** Khi viết thesis phải ghi "AnoFusion: 1 seed". Dòng AnoFusion RE3-TT chỉ có `test_f1` (dừng sớm), dấu * và không vào dòng Trung bình; % của nó (+348%) phình ra vì nền 0,116 rất thấp.
- **Δ** = UAM-AD trừ mô hình ở dòng đó, trong ngoặc là % giá trị của mô hình đó (dương = UAM-AD cao hơn); so với AnoFusion, Δ = trung bình 5 seed của UAM-AD trừ seed 0 của AnoFusion. Dòng Trung bình: % = chênh lệch trung bình chia giá trị trung bình của mô hình (không phải trung bình các %).
- Ba chỉ số ưu tiên: F1@val95, AUROC, AUPRC. **F1 oracle chỉ so được với các mô hình khác ở RE2-OB, RE2-TT** (ở đó UAM-AD dùng bản không point-adjust như các mô hình kia; ở SN, RE3-OB, RE3-TT UAM-AD có point-adjust); **HR@k khác đơn vị chấm** nên không so thẳng được. Bảng giữ các cột đó để tra, không dùng để kết luận.
- **Việc 10 (cập nhật 10-30):** UAM-AD ở **RE2-OB và RE2-TT** là số chấm lại bằng đúng các checkpoint cũ trên test sắp theo thứ tự thời gian của file gốc (UAC-AD ở hai dataset này cũng vậy; test đã lưu bị xáo, và xáo làm UAM-AD kém đi ở RE2-OB). RE3-OB, RE3-TT giữ số đã lưu vì sắp lại chỉ đổi tối đa 0,011; SN không có thông tin thời gian. **HR@k của UAM-AD ở cả 5 dataset** là xếp hạng trace + KPI của việc 10 (thay cách cũ chỉ dùng trace thô), tính trên mọi cửa sổ bất thường có service lỗi, 5 seed đủ cả HR@5.
- Cấu hình chạy của từng mô hình và vai trò dòng tham chiếu có nhãn (Eadro, OrEdge): xem T3.

##### T1. Bảng so sánh theo dataset (5 seed)

<!-- SUMMARY9 T1 BEGIN -->

| Dataset | Mô hình | F1@val95 | Δ F1 | AUROC | Δ AUROC | AUPRC | Δ AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | UAM-AD | 0,681±0,076 | – | 0,917±0,016 | – | 0,823±0,025 | – | 0,764±0,020 | 0,759±0,022 | 0,949±0,008 | 0,976±0,004 | – | 134,0 |
| RE2-OB | ART | 0,880±0,001 | −0,199 (−22,6%) | 0,981±0,002 | −0,064 (−6,5%) | 0,957±0,004 | −0,134 (−14,0%) | 0,923±0,004 | 0,753±0,004 | 0,844±0,004 | 0,933±0,000 | – | 6,3 |
| RE2-OB | AnoFusion | 0,587±0,000 | +0,094 (+16,0%) | 0,819±0,000 | +0,098 (+12,0%) | 0,693±0,000 | +0,130 (+18,8%) | 0,654±0,000 | – | – | – | – | 18,8 |
| RE2-OB | Eadro (có giám sát) | 0,711±0,016 | −0,030 (−4,3%) | 0,867±0,005 | +0,051 (+5,8%) | 0,791±0,011 | +0,033 (+4,2%) | 0,749±0,016 | 0,667±0,016 | 0,880±0,003 | 0,996±0,002 | 0,696±0,007 | 12,6 |
| RE2-OB | OrEdge (bán giám sát) | 0,867±0,012 | −0,186 (−21,4%) | 0,973±0,003 | −0,056 (−5,7%) | 0,959±0,008 | −0,135 (−14,1%) | 0,945±0,008 | 0,927±0,003 | 0,946±0,003 | 0,952±0,004 | 0,895±0,010 | 30,2 |
| RE3-OB | UAM-AD | 0,560±0,019 | – | 0,835±0,005 | – | 0,615±0,012 | – | 0,755±0,007 | 0,824±0,024 | 0,902±0,022 | 0,959±0,019 | – | 38,2 |
| RE3-OB | ART | 0,523±0,058 | +0,037 (+7,1%) | 0,893±0,005 | −0,059 (−6,6%) | 0,746±0,015 | −0,131 (−17,5%) | 0,756±0,017 | 0,727±0,006 | 0,788±0,007 | 0,827±0,006 | – | 2,2 |
| RE3-OB | AnoFusion | 0,366±0,000 | +0,194 (+53,1%) | 0,630±0,000 | +0,204 (+32,4%) | 0,447±0,000 | +0,169 (+37,8%) | 0,460±0,000 | – | – | – | – | 48,1 |
| RE3-OB | Eadro (có giám sát) | 0,645±0,026 | −0,085 (−13,2%) | 0,817±0,022 | +0,018 (+2,2%) | 0,727±0,024 | −0,111 (−15,3%) | 0,679±0,024 | 0,853±0,022 | 1,000±0,000 | 1,000±0,000 | 0,648±0,015 | 4,5 |
| RE3-OB | OrEdge (bán giám sát) | 0,762±0,009 | −0,202 (−26,5%) | 0,892±0,003 | −0,058 (−6,5%) | 0,832±0,005 | −0,217 (−26,1%) | 0,815±0,009 | 0,941±0,009 | 0,994±0,002 | 0,999±0,001 | 0,763±0,010 | 10,9 |
| RE2-TT | UAM-AD | 0,597±0,007 | – | 0,809±0,002 | – | 0,680±0,006 | – | 0,624±0,006 | 0,721±0,027 | 0,871±0,029 | 0,908±0,032 | – | 189,2 |
| RE2-TT | ART | 0,583±0,002 | +0,014 (+2,4%) | 0,833±0,002 | −0,025 (−3,0%) | 0,688±0,003 | −0,008 (−1,1%) | 0,677±0,001 | 0,447±0,003 | 0,511±0,004 | 0,590±0,007 | – | 7,3 |
| RE2-TT | AnoFusion | 0,584±0,000 | +0,013 (+2,2%) | 0,773±0,000 | +0,035 (+4,5%) | 0,672±0,000 | +0,008 (+1,2%) | 0,668±0,000 | – | – | – | – | 220,3 |
| RE2-TT | Eadro (có giám sát) | 0,641±0,020 | −0,044 (−6,8%) | 0,838±0,014 | −0,029 (−3,5%) | 0,728±0,021 | −0,048 (−6,6%) | 0,691±0,020 | 0,539±0,015 | 0,780±0,007 | 0,993±0,003 | 0,621±0,016 | 34,0 |
| RE2-TT | OrEdge (bán giám sát) | 0,831±0,002 | −0,234 (−28,2%) | 0,967±0,005 | −0,159 (−16,4%) | 0,949±0,003 | −0,269 (−28,4%) | 0,926±0,002 | 0,954±0,007 | 0,982±0,004 | 0,994±0,001 | 0,905±0,004 | 31,9 |
| RE3-TT | UAM-AD | 0,513±0,002 | – | 0,702±0,003 | – | 0,347±0,003 | – | 0,633±0,004 | 0,251±0,006 | 0,332±0,017 | 0,417±0,022 | – | 50,8 |
| RE3-TT | ART | 0,322±0,012 | +0,191 (+59,3%) | 0,684±0,013 | +0,019 (+2,8%) | 0,448±0,014 | −0,102 (−22,7%) | 0,473±0,007 | 0,344±0,053 | 0,413±0,032 | 0,478±0,023 | – | 2,4 |
| RE3-TT | AnoFusion (CHỈ f1, dừng sớm) | 0,116±0,000 | +0,404 (+348,2%)* | 0,502±0,000 | +0,217 (+43,2%)* | 0,206±0,000 | +0,152 (+73,9%)* | 0,351±0,000 | – | – | – | – | – |
| RE3-TT | Eadro (có giám sát) | 0,365±0,028 | +0,148 (+40,4%) | 0,606±0,020 | +0,097 (+16,0%) | 0,443±0,017 | −0,096 (−21,7%) | 0,443±0,009 | 0,621±0,005 | 1,000±0,000 | 1,000±0,000 | 0,225±0,004 | 11,6 |
| RE3-TT | OrEdge (bán giám sát) | 0,631±0,006 | −0,117 (−18,6%) | 0,764±0,004 | −0,062 (−8,1%) | 0,690±0,003 | −0,344 (−49,8%) | 0,666±0,003 | 0,905±0,011 | 0,995±0,001 | 0,997±0,000 | 0,619±0,009 | 33,5 |
| SN | UAM-AD | 0,728±0,004 | – | 0,956±0,002 | – | 0,716±0,012 | – | 0,849±0,009 | 0,185±0,017 | 0,347±0,055 | 0,397±0,036 | – | 4,3 |
| SN | ART | 0,553±0,034 | +0,174 (+31,5%) | 0,890±0,019 | +0,067 (+7,5%) | 0,590±0,038 | +0,126 (+21,5%) | 0,664±0,028 | 0,058±0,018 | 0,161±0,025 | 0,345±0,049 | – | 0,4 |
| SN | AnoFusion | 0,408±0,000 | +0,320 (+78,5%) | 0,731±0,000 | +0,225 (+30,8%) | 0,455±0,000 | +0,261 (+57,4%) | 0,575±0,000 | – | – | – | – | 11,3 |
| SN | Eadro (có giám sát) | 0,820±0,016 | −0,092 (−11,2%) | 0,980±0,003 | −0,024 (−2,4%) | 0,857±0,013 | −0,141 (−16,4%) | 0,883±0,010 | 0,074±0,052 | 0,266±0,027 | 0,498±0,053 | 0,809±0,021 | 1,1 |
| SN | OrEdge (bán giám sát) | 0,416±0,108 | +0,312 (+74,9%) | 0,844±0,039 | +0,112 (+13,3%) | 0,546±0,061 | +0,170 (+31,1%) | 0,654±0,054 | 0,262±0,077 | 0,506±0,011 | 0,724±0,039 | 0,029±0,037 | 2,0 |
| **Trung bình** | UAM-AD | 0,616±0,018 | – | 0,844±0,004 | – | 0,636±0,005 | – | – | – | – | – | – | – |
| **Trung bình** | ART | 0,572±0,013 | +0,044 (+7,6%) | 0,856±0,004 | −0,012 (−1,4%) | 0,686±0,008 | −0,050 (−7,2%) | – | – | – | – | – | – |
| **Trung bình** | AnoFusion (4 dataset) | 0,486±0,000 | +0,155 (+32,0%) | 0,739±0,000 | +0,141 (+19,0%) | 0,567±0,000 | +0,142 (+25,1%) | – | – | – | – | – | – |
| **Trung bình** | Eadro (có giám sát) | 0,637±0,007 | −0,021 (−3,3%) | 0,821±0,005 | +0,022 (+2,7%) | 0,709±0,005 | −0,073 (−10,2%) | – | – | – | – | – | – |
| **Trung bình** | OrEdge (bán giám sát) | 0,701±0,020 | −0,086 (−12,2%) | 0,888±0,008 | −0,044 (−5,0%) | 0,795±0,013 | −0,159 (−20,0%) | – | – | – | – | – | – |

*Đếm ô (3 chỉ số F1@val95, AUROC, AUPRC × các dataset đủ kết quả của mô hình): UAM-AD hơn hoặc kém ở **mọi seed** có kết quả của mô hình đó; "hoà" = dấu của chênh lệch lẫn giữa các seed. AnoFusion chỉ 1 seed nên không có ô hoà.*

| Mô hình | số ô | UAM-AD hơn ở mọi seed | lẫn dấu giữa các seed (hoà) | UAM-AD kém ở mọi seed |
|:--|--:|--:|--:|--:|
| ART | 15 | 5 | 3 | 7 |
| AnoFusion (1 seed, 4 dataset) | 12 | 12 | 0 | 0 |
| Eadro (có giám sát) | 15 | 4 | 2 | 9 |
| OrEdge (bán giám sát) | 15 | 3 | 0 | 12 |

<!-- SUMMARY9 T1 END -->

**Nhận xét** (Δ% lấy từ T1; "cả 5 seed cùng chiều" = chênh lệch theo từng seed luôn cùng dấu, "hoà" = dấu lẫn giữa các seed). Số từng ô xem bảng; ở đây chỉ ghi dạng hình và điều cần cẩn thận khi đọc.

**Ba nhận định đầu tiên của người dùng, kiểm với số liệu:** (i) "UAM-AD tốt hơn AnoFusion": **đúng**, hơn ở 12/12 ô trên 4 dataset AnoFusion có đủ, và hơn ở cả ba chỉ số khi lấy trung bình (+19,0% đến +32,0%), với điều kiện AnoFusion chỉ có 1 seed. (ii) "không tốt bằng ART": **đúng một phần**. Kém ART rõ ở AUPRC (trung bình −7,2%) và hơi kém ở AUROC (−1,4%), nhưng F1@val95 trung bình hơn (+7,6%), và UAM-AD hơn ART ở SN (cả ba chỉ số) và ở F1 của RE2-TT và RE3-TT. Trong 15 ô: hơn 5, hoà 3, kém 7. (iii) "chắc chắn thua mô hình có nhãn": **đúng với OrEdge ở RCAEval** (kém ở cả 12 ô, cả 5 seed cùng chiều) và **đúng với Eadro về F1@val95 và AUPRC** (trung bình kém ở 4/5 dataset mỗi chỉ số; trong 15 ô chung ba chỉ số: kém 9, hoà 2, hơn 4), **không đúng với AUROC của Eadro**: trung bình UAM-AD hơn 2,7%, hơn ở RE2-OB (+5,8%) và RE3-TT (+16,0%), hoà ở RE3-OB (+2,2%), kém nhẹ ở RE2-TT (−3,5%) và SN (−2,4%).

**So với ART (không giám sát).**
- F1 kém nhất ở RE2-OB (−22,6%): kém ở cả ba chỉ số (AUROC −6,5%, AUPRC −14,0%), cả 5 seed cùng chiều; ART gần như không đổi giữa các seed (sd F1 0,001), còn UAM-AD 0,681±0,076.
- Gần ngang ở RE2-TT (F1 +2,4%, AUROC −3,0%, AUPRC −1,1%). RE3-OB: F1 hoà; RE3-TT: F1 hơn rõ (0,513 so với 0,322). AUPRC kém ở cả hai dataset RE3 (−17,5% và −22,7%).
- Hơn ở SN cả ba chỉ số, cả 5 seed cùng chiều.
- AUROC: kém ở RE2-OB (−6,5%), RE3-OB (−6,6%), RE2-TT (−3,0%) cả 5 seed cùng chiều, hoà ở RE3-TT (+2,8%), hơn ở SN (+7,5%).
- Trung bình 5 dataset: F1 +7,6%, AUROC −1,4%, AUPRC −7,2%. Chỗ ART hơn rõ nhất là AUPRC ở các dataset OB và RE3-TT.

**So với AnoFusion (không giám sát, 1 seed).**
- RE2-OB: hơn ở cả ba chỉ số (F1 +16,0%, AUROC +12,0%, AUPRC +18,8%). RE2-TT: sát nhau (F1 +2,2%, AUROC +4,5%, AUPRC +1,2%); AnoFusion mạnh nhất ở dataset này nhưng cũng tốn 220 phút cho một seed.
- RE3-OB và SN: UAM-AD hơn rõ ở cả ba chỉ số (AnoFusion yếu ở RE3-OB, AUROC 0,630).
- RE3-TT: AnoFusion chỉ có `test_f1` nên so trên một file (dấu *); không dùng để kết luận.
- Trung bình 4 dataset: +32,0% F1, +19,0% AUROC, +25,1% AUPRC; hơn ở cả 12 ô. Vì chỉ 1 seed, chưa biết nhiễu seed của AnoFusion, nên "hơn" ở các ô chênh dưới vài % (RE2-TT) không nên khẳng định.

**So với Eadro (có giám sát).**
- AUROC là chỉ số UAM-AD cạnh tranh được: hơn ở RE2-OB (+5,8%) và RE3-TT (+16,0%, vì Eadro yếu ở lỗi mức mã, AUROC 0,606), hoà ở RE3-OB, kém nhẹ ở RE2-TT (−3,5%) và SN (−2,4%; Eadro đạt AUROC 0,980 ở SN).
- F1@val95 và AUPRC: trung bình kém ở 4/5 dataset mỗi chỉ số; ngoại lệ là F1 ở RE3-TT (+40,4%) và AUPRC ở RE2-OB (+4,2%, cả 5 seed cùng chiều); F1 ở RE2-OB (−4,3%) hoà giữa các seed.
- Trong 15 ô: hơn 4, hoà 2, kém 9. Trung bình: F1 −3,3%, AUROC +2,7%, AUPRC −10,2%.

**So với OrEdge (bán giám sát).**
- RCAEval (RE2-OB, RE3-OB, RE2-TT, RE3-TT): kém ở cả 12 ô, cả 5 seed cùng chiều. Chênh lớn nhất ở AUPRC (−14,1% đến −49,8%) và F1 (−18,6% đến −28,2%); AUROC −5,7% đến −16,4%. OrEdge đạt AUROC 0,76–0,97 và AUPRC 0,69–0,96 ở RCAEval.
- SN: hơn ở cả ba chỉ số, cả 5 seed cùng chiều. Không nên đọc đây là ưu thế của mô hình: OrEdge SN có độ lệch chuẩn rất lớn (F1 0,416±0,108, AUROC 0,844±0,039), và dùng giao thức khác (xem đoạn giải thích SN bên dưới).
- Trong 15 ô: hơn 3 (cả ba ở SN), hoà 0, kém 12. Trung bình: F1 −12,2%, AUROC −5,0%, AUPRC −20,0%.

**Vì sao ở SN Eadro hơn UAM-AD còn OrEdge kém UAM-AD.** Hai mô hình có nhãn dùng cùng giao thức leave-one-scenario-out, huấn luyện trên tập rất nhỏ (152–399 cửa sổ). Eadro có đầu phát hiện riêng ở mức cửa sổ và thấy sẵn hai scenario cùng họ lỗi trong huấn luyện (bỏ cả họ lỗi làm AUROC của một bộ phân loại có nhãn đơn giản giảm 0,985 → 0,924), nên phát hiện tốt và khá ổn định (AUROC 0,980), nhưng định vị kém (HR@1 0,074). OrEdge thiết kế để định vị nút lỗi (HR@1 0,262, cao nhất bốn mô hình ở SN); điểm phát hiện của nó là xác suất nút lớn nhất. Huấn luyện từ vài trăm cửa sổ làm kết quả của OrEdge phụ thuộc mạnh vào khởi tạo ngẫu nhiên: AUROC của cùng một scenario dao động trung bình 0,226 giữa các seed (ví dụ 0,48–0,96 ở `Code_Stop_MediaService`), so với 0,009 ở UAM-AD; trung bình 5 seed là AUROC 0,844±0,039 và F1@val95 0,416±0,108. Ngay cả seed tốt nhất của OrEdge cũng chỉ vượt seed tốt nhất của UAM-AD ở 2/12 scenario. UAM-AD không dùng nhãn lỗi nên không chịu bất ổn này. Kết quả cho thấy mô hình có nhãn kém ổn định khi nhãn ít, không cho phép kết luận UAM-AD hơn OrEdge nói chung (ở RCAEval OrEdge hơn UAM-AD ở cả 12 ô).

**HR@k (định vị service lỗi; đơn vị chấm khác nên không tính %).** Số từng mô hình ở các cột HR của T1.
- **Cách UAM-AD xếp hạng (việc 10; cấu hình trace + KPI được khai báo trước khi chạy).** Điểm của một service = điểm trace của node đó (Eq. 14 của TraceDAE, nhưng chuẩn hoá theo mức bình thường của chính node trên val: trung vị và p95 − trung vị, cắt ở 0) + điểm KPI của các kênh thuộc service (sai số tái tạo bình phương, chuẩn hoá theo từng kênh trên val, lấy lớn nhất trong các kênh của service). HR tính trên mọi cửa sổ bất thường có service lỗi, như ART, Eadro, OrEdge. Cách cũ (điểm trace thô, `rca.py:37-49`) hỏng vì một service trung tâm chiếm top-1 bất kể lỗi ở đâu (đo ở seed 1: RE3-OB `frontend` 57% số mẫu, không bao giờ là service lỗi; RE2-OB `recommendationservice` 56%; SN `nginx-web-server` 58%).
- **Kết quả (HR@1 / HR@3 / HR@5, 5 seed):** RE2-OB 0,759 / 0,949 / 0,976; RE3-OB 0,824 / 0,902 / 0,959; RE2-TT 0,721 / 0,871 / 0,908; RE3-TT 0,251 / 0,332 / 0,417; SN 0,185 / 0,347 / 0,397. HR@1 so với ART: hơn ở RE3-OB, RE2-TT, SN; ngang ở RE2-OB (0,759 so với 0,753); kém ở RE3-TT (0,344). So với Eadro: hơn ở RE2-OB, RE2-TT, SN; kém ở RE3-OB (0,853) và RE3-TT (0,621). Kém OrEdge ở cả 5 dataset (0,262 đến 0,954).
- **Hai dataset chưa đạt:** RE3-TT (HR@1 0,251) và SN (0,185) đều thấp hơn đoán cố định service hay gặp nhất (khoảng 0,50 ở RE3-TT; 0,27–0,34 ở SN tuỳ tập mẫu), nên ở hai dataset này chưa nói được UAM-AD định vị. Ở SN, trung bình theo scenario bị kéo xuống bởi sáu scenario chỉ có 3–10 mẫu bất thường (HR gần như chỉ là 0 hoặc 1); ba scenario `Code_Stop_*` (38–39 mẫu mỗi scenario) lên rõ: HR@1 0,708 / 0,492 / 0,305 so với 0,046 / 0,067 / 0,037 của cách cũ.
- **Đóng góp của từng phần (ablation, UAM-AD, 5 seed, HR@1 / HR@3 / HR@5):**

| Dataset | Cách cũ (trace thô) | Chỉ trace chuẩn hoá | Chỉ KPI | **Trace + KPI (dùng ở T1)** |
|:--|:--|:--|:--|:--|
| RE2-OB | 0,355 / 0,613 / 0,759 | 0,318 / 0,471 / 0,570 | 0,817 / 0,929 / 0,969 | **0,759 / 0,949 / 0,976** |
| RE3-OB | 0,022 / 0,219 / 0,529 | 0,114 / 0,376 / 0,565 | 0,837 / 0,893 / 0,925 | **0,824 / 0,902 / 0,959** |
| RE2-TT | 0,315 / 0,510 / 0,599 | 0,350 / 0,541 / 0,619 | 0,670 / 0,834 / 0,884 | **0,721 / 0,871 / 0,908** |
| RE3-TT | 0,227 / 0,286 / 0,393 | 0,068 / 0,184 / 0,307 | 0,253 / 0,323 / 0,400 | **0,251 / 0,332 / 0,417** |
| SN | 0,189 / 0,286 / 0,520 | 0,037 / 0,095 / 0,322 | 0,292 / 0,392 / 0,440 | **0,185 / 0,347 / 0,397** |

  Phần lớn tín hiệu đến từ KPI của chính service; trace chuẩn hoá một mình vẫn yếu. Chỉ KPI cao hơn trace + KPI ở HR@1 của RE2-OB (0,817 so với 0,759), RE3-OB (0,837 so với 0,824) và SN (0,292 so với 0,185); trace + KPI hơn ở RE2-TT (0,721 so với 0,670) và ở HR@3, HR@5 của bốn dataset trừ SN. Biến thể "chỉ KPI" được xem sau khi đã thấy nhãn test nên chỉ là ablation, không thay cấu hình đã khai báo. Cột "Cách cũ" tính trên cùng tập mẫu với ba cột kia nên khác chút so với HR cũ của S2 (ví dụ RE2-OB HR@1 0,355 so với 0,352).
- **AnoFusion không có định vị.** Trong bản chuyển thể của luận văn, cả hệ thống được xem là một "instance" (kênh = mọi KPI + số lần xuất hiện mẫu log + đặc trưng trace dàn phẳng của mọi service), điểm = max theo kênh, mỗi mẫu chỉ có **một điểm, không có điểm theo service** (`run_anofusion.py` đầu file), nên cột HR của nó là "–" (kế hoạch ban đầu cũng chỉ lấy HR@k của ART và Eadro). Theo tóm tắt bài gốc (arXiv 2305.18985) AnoFusion là phương pháp phát hiện sự cố; mình chưa đọc toàn văn nên chỉ khẳng định điều này ở mức tóm tắt. Muốn có HR cho AnoFusion phải tự thêm một bước quy kênh có sai số lớn nhất về service; đó là phần thêm của luận văn, không thuộc mô hình gốc.
- **Các mô hình khác định vị bằng gì:** ART (không giám sát) xếp service theo độ tương tự cosine giữa vector sai số của từng service và vector sai số tổng (`run_art.py` `rank_hits`), mỗi service có KPI, log, trace riêng của nó; Eadro và OrEdge có đầu phân loại service huấn luyện bằng nhãn service lỗi (`eadro_model.py:7`). Đơn vị chấm của HR cũng khác (UAM-AD cửa sổ 30 mẫu, các đối thủ từng mẫu giây hoặc cửa sổ 10), nên so HR trực tiếp chỉ nên dùng để thấy khoảng cách cỡ lớn.

**Điểm mạnh của UAM-AD.** SN (hơn ART, AnoFusion, OrEdge; chỉ kém Eadro); hơn AnoFusion ở cả 12 ô; AUROC hơn Eadro ở RE2-OB và RE3-TT; HR@1 hơn ART ở RE3-OB, RE2-TT, SN và ngang ở RE2-OB (HR@1 ở RE3-TT và SN vẫn là điểm yếu, xem phần HR@k); và độ ổn định giữa các seed ở AUROC và AUPRC: độ lệch chuẩn chia trung bình, lấy trung bình qua 5 dataset, là 0,6% và 1,7% ở UAM-AD, so với 1,0% và 2,5% (ART), 1,7% và 2,6% (Eadro), 1,3% và 2,7% (OrEdge); ở F1@val95 (UAM-AD 3,3%, ART 4,3%, Eadro 3,8%, OrEdge 5,9%) UAM-AD thấp nhất, nhưng RE2-OB vẫn 11,1%.

##### T2. Mức 80% so với hai mô hình có nhãn

<!-- SUMMARY9 T2 BEGIN -->

*Mức 80% so với hai mô hình có nhãn (UAM-AD / mô hình, trung bình 5 seed; ✓ = đạt, ≥ 80%):*

| Dataset | F1@val95 / Eadro | F1@val95 / OrEdge | AUROC / Eadro | AUROC / OrEdge | AUPRC / Eadro | AUPRC / OrEdge |
|:--|--:|--:|--:|--:|--:|--:|
| RE2-OB | 95,7% ✓ | 78,6% ✗ | 105,8% ✓ | 94,3% ✓ | 104,2% ✓ | 85,9% ✓ |
| RE3-OB | 86,8% ✓ | 73,5% ✗ | 102,2% ✓ | 93,5% ✓ | 84,7% ✓ | 73,9% ✗ |
| RE2-TT | 93,2% ✓ | 71,8% ✗ | 96,5% ✓ | 83,6% ✓ | 93,4% ✓ | 71,6% ✗ |
| RE3-TT | 140,4% ✓ | 81,4% ✓ | 116,0% ✓ | 91,9% ✓ | 78,3% ✗ | 50,2% ✗ |
| SN | 88,8% ✓ | 174,9% ✓ | 97,6% ✓ | 113,3% ✓ | 83,6% ✓ | 131,1% ✓ |
| **Trung bình** | 96,7% ✓ | 87,8% ✓ | 102,7% ✓ | 95,0% ✓ | 89,8% ✓ | 80,0% ✗ |

Trên 5 dataset (không tính dòng Trung bình): **UAM-AD đạt 23/30 ô**. F1@val95 7/10 (Eadro 5/5, OrEdge 2/5, cả hai cùng lúc 2/5); AUROC 10/10 (Eadro 5/5, OrEdge 5/5, cả hai cùng lúc 5/5); AUPRC 6/10 (Eadro 4/5, OrEdge 2/5, cả hai cùng lúc 2/5).

<!-- SUMMARY9 T2 END -->

**Nhận xét mức 80%.** (1) **Mức 80% do người dùng chọn ngày 10-27, sau khi đã thấy kết quả** (trước đó thử mức 90%, UAM-AD đạt 16/30 ô, rồi bỏ mức này). Đây là mức chấp nhận của người viết, không lấy từ tài liệu nào đã kiểm chứng; thesis nên ghi đúng như vậy và báo kèm Δ% từng chỉ số (bảng T1) thay vì chỉ báo đạt hay không. Mức áp chung cho cả ba chỉ số và cả hai mô hình có nhãn. (2) **UAM-AD đạt 23/30 ô**: Eadro 14/15, OrEdge 9/15. Bảy ô trượt đều là F1@val95 hoặc AUPRC (xem bảng); **chỉ SN đạt cả sáu ô**. (3) Dòng Trung bình: 5/6 ô đạt, ô trượt là AUPRC so với OrEdge (79,9986%, làm tròn hiện 80,0% trong bảng nhưng vẫn trượt, thiếu 0,0014 điểm %).

**Mức độ nhạy với vạch 80%.** Ba ô nằm trong ±2 điểm % của vạch: 78,6% (trượt), 78,3% (trượt) và 81,4% (đạt); dòng Trung bình AUPRC so với OrEdge 79,9986% cũng sát vạch. Hạ vạch xuống 78% thì thêm 2 ô, nâng lên 82% thì mất 1 ô. Nhiễu seed của UAM-AD nhỏ hơn khoảng cách tới vạch ở hầu hết các ô; F1 ở RE2-OB (sd/trung bình 11,1%) là ngoại lệ, nơi 78,6% có thể lên xuống quanh vạch giữa các seed.

**Giới hạn khi đọc.** (a) AUROC có sàn 0,5 và AUPRC có sàn bằng tỉ lệ cửa sổ lỗi (khoảng 13–20%), nên 10/10 ở AUROC không nói UAM-AD "ngang" mô hình có nhãn, mà nói điểm xếp hạng của nó không thấp hơn nhiều. (b) Tiêu chí chỉ xét phát hiện, không xét định vị: HR@1 của UAM-AD ở RCAEval (0,25–0,82) thấp hơn OrEdge (0,91–0,95) ở cả bốn dataset và thấp hơn Eadro (0,54–0,85) ở RE3-OB và RE3-TT (RE3-TT chỉ 0,25). (c) Eadro và OrEdge có nhãn và dùng giao thức riêng ở SN (leave-one-scenario-out); đây là dòng tham chiếu, không phải đối thủ cùng điều kiện. (d) Cấu hình của chúng theo quy tắc epoch chung (xem T3), không tinh chỉnh riêng; riêng đặc trưng log theo node của OrEdge ở SN được chọn sau khi đã xem kết quả SN.

**Câu đỡ được trong thesis.** "Ở mức chấp nhận 80%, UAM-AD (không dùng nhãn lỗi khi huấn luyện) giữ ≥ 80% AUROC của cả Eadro và OrEdge ở 5/5 dataset, ≥ 80% F1@val95 của Eadro ở 5/5 dataset và AUPRC của Eadro ở 4/5 dataset, nhưng chỉ ở 2/5 dataset với OrEdge cho cả F1@val95 và AUPRC; chỉ SN đạt cả sáu chỉ tiêu." Không nên viết "UAM-AD ngang mô hình có giám sát".

##### T3. Cấu hình chạy của các đối thủ (điều kiện phải ghi khi báo kết quả)

- **Số epoch = của UAM-AD cho mọi mô hình, không thêm gì cả:** 5 epoch trên RCAEval (RE2/RE3, OB/TT), 50 epoch trên SN; không tinh chỉnh riêng cho từng mô hình.
- **Số seed:** UAM-AD, ART, Eadro, OrEdge có 5 seed (0–4); AnoFusion chỉ seed 0, ghi là kết quả 1 seed.
- **Modality:** cả bốn đối thủ dùng L+M+T. ART chỉ báo bản L+M+T; các số ART M+T chạy trước đó đã bỏ khỏi mọi bảng.
- **RCAEval:** ART và AnoFusion huấn luyện một lần cho cả dataset rồi chấm mọi file test; Eadro và OrEdge huấn luyện một lần cho mỗi file test (3-fold theo run).
- **SN, ART và AnoFusion:** đệm lịch sử bằng đuôi phiên Normal_Baseline (`--pad_normal`), vì hai mô hình cần chuỗi cửa sổ liền trước mà mỗi phiên SN bắt đầu ngay lúc có lỗi. Chỉ dùng dữ liệu train (chính các cửa sổ Normal_Baseline mà UAM-AD cũng học), không dùng test hay nhãn. Hai giới hạn: đệm chỉ giúp vài cửa sổ đầu phiên (5 cho ART, 20 cho AnoFusion), nên với lỗi kéo dài (nhất là 3 phiên `Code_Stop_*`) điểm của chúng có thể thấp vì bản chất mô hình (chưa đo); và chưa có dòng ART không đệm.
- **SN, Eadro và OrEdge:** leave-one-scenario-out (huấn luyện trên các cửa sổ có nhãn của 11 phiên khác, gồm Normal_Baseline; chấm đúng tập test của UAM-AD), chạy không lịch sử. Hai mô hình này dùng nhãn lỗi nên là dòng tham chiếu.
- **Đặc trưng log:** Eadro dùng 2 đặc trưng log mỗi service (log1p số dòng và số dòng khác nhau). OrEdge dùng vector đếm template theo từng nút như tác giả; đổi này được chọn sau khi đã thấy kết quả SN của bản 2 đặc trưng.
- **AnoFusion trên RCAEval:** runner dùng thang của tác giả (chia mỗi kênh cho max train), nên kênh có max train bằng 0 mà khác 0 lúc test bị nhân tới 1e8. Đã cắt ±100 (`--clip 100`) ở RE2-OB, RE3-OB, RE2-TT; so bản cũ không cắt, AUROC theo scenario đổi rất ít (ví dụ RE3-OB `f1` 0,831 → 0,829, RE2-TT `cpu` 0,856 → 0,855).
- **AnoFusion RE3-TT:** chỉ có `test_f1`; `f2`–`f4` không chạy (hết thời gian chạy AnoFusion). Có một ước lượng thống kê trong `working_history.md`, **không phải số đo**, không vào bảng nào.
- Nhật ký chạy, lỗi và cách sửa của từng mô hình: `working_history.md`.

### Giai đoạn 4 – Đo thêm và tổng hợp

| # | Việc | Phụ thuộc | Ra gì |
|:-:|:--|:--|:--|
| 10 | **Xong 10-30** (việc 10 của Giai đoạn 4; khác việc 10 chấm lại ở Giai đoạn 3). Đo chi phí (Bảng E): số tham số, thời gian suy luận (CPU), đỉnh RAM, mỗi mô hình trong một tiến trình mới, cùng máy và batch size; thời gian train lấy từ log các lần chạy. Đã đo mọi file test của cả 5 dataset (165 lượt) cho UAM-AD, UAC-AD, ART, Eadro, OrEdge; AnoFusion đo suy luận ở SN (12 file, đo thêm ngày 10-31), RE2-OB (1 file, cả file) và RE3-OB, RE2-TT (2 560 mẫu đầu của 1 file, đo ngày 11-01); RE3-TT không đo (số tham số thì có đủ cả 5 dataset); điều kiện và lưu ý ở Bảng E | 8, 9 | Bảng E (ngay dưới bảng này) |
| 11 | **Xong 10-30.** Làm Bảng D từ kết quả việc 8, 9: HR@k của ART, Eadro, OrEdge (việc 9) và UAM-AD (xếp hạng trace + KPI của việc 10 ở Giai đoạn 3) trên cả 5 dataset, tính trên cùng tập mẫu bất thường có service lỗi | 8, 9, kiểm ground-truth ở việc 2 | Bảng D (ngay dưới bảng này) |
| 12 | **Xong 10-31 (bảng):** Bảng A–E tổng hợp ngay dưới (Việc 12); phần chữ của chương kết quả chưa viết. Tổng hợp Bảng A–E, viết chương kết quả. Ghi rõ trong luận văn: cấu hình cố định, không tối ưu trên test, các tham số riêng đã chọn lúc nhìn kết quả (mục 2.2), và đối thủ có nhãn chỉ là hàng tham chiếu | 4, 7–11 | Chương kết quả |

#### Việc 10 (Giai đoạn 4) – Bảng E: chi phí của UAM-AD và các mô hình

Điều kiện cần đo và cách đo RAM: mục 4, Bảng E. **Kết quả theo dataset (đo 10-30 và 10-31 bằng `codes/common/measure_cost.py` trên mọi file test; số đo từng file ở `D:\ClaudeWork\phase3\cost_all\<dataset>\<file test>\<mô hình>.json`; bảng sinh bởi `summary_tools/cost_table.py`).**

<!-- BANGE BEGIN -->

**RE2-OB** (6 file test: cpu, delay, disk, loss, mem, socket; 319 709 mẫu)

| Mô hình | Huấn luyện (phút / lần, GPU) | Tham số lúc suy luận | Suy luận CPU (ms / 1000 mẫu) | RAM đỉnh (MB) |
|:--|--:|--:|--:|--:|
| UAC-AD | 15,6 | 607 870 | 93,1 | 1 128 |
| **UAM-AD** | 18,1 | 627 395 | 110,7 | 1 183 |
| AnoFusion | 6,5 | 309 814 | 17 635,6* | 1 695* |
| ART | 1,7 | 10 612 | 168,1 | 787 |
| Eadro (có giám sát) | 0,6 | 72 300 | 250,5 | 607 |
| OrEdge (bán giám sát) | 1,5 | 11 900 | 257,5 | 608 |

*dấu \* (AnoFusion): chỉ đo 1 file test (`test_cpu`, toàn bộ file, 876,8 giây) trong 6 file, nên hai ô này là của file đó*

**RE2-TT** (6 file test: cpu, delay, disk, loss, mem, socket; 324 450 mẫu)

| Mô hình | Huấn luyện (phút / lần, GPU) | Tham số lúc suy luận | Suy luận CPU (ms / 1000 mẫu) | RAM đỉnh (MB) |
|:--|--:|--:|--:|--:|
| UAC-AD | 18,3 | 647 473 | 108,7 | 1 917 |
| **UAM-AD** | 26,6 | 686 953 | 161,1 | 2 069 |
| AnoFusion | 76,6 | 2 637 542 | 311 556,3* | 5 062* |
| ART | 1,8 | 15 908 | 217,1 | 1 071 |
| Eadro (có giám sát) | 1,7 | 73 340 | 916,8 | 1 012 |
| OrEdge (bán giám sát) | 1,6 | 13 830 | 545,0 | 874 |

*dấu \* (AnoFusion): chỉ đo 1 file test (`test_cpu`, 2 560 mẫu đầu, 797,6 giây) trong 6 file, nên hai ô này là của file đó*

**RE3-OB** (5 file test: f1, f2, f3, f4, f5; 102 704 mẫu)

| Mô hình | Huấn luyện (phút / lần, GPU) | Tham số lúc suy luận | Suy luận CPU (ms / 1000 mẫu) | RAM đỉnh (MB) |
|:--|--:|--:|--:|--:|
| UAC-AD | 4,4 | 643 732 | 87,5 | 1 136 |
| **UAM-AD** | 6,3 | 681 327 | 105,2 | 1 173 |
| AnoFusion | 14,7 | 1 631 462 | 143 313,1* | 3 569* |
| ART | 0,6 | 10 612 | 121,6 | 751 |
| Eadro (có giám sát) | 0,2 | 72 300 | 197,9 | 569 |
| OrEdge (bán giám sát) | 0,7 | 61 115 | 658,9 | 741 |

*dấu \* (AnoFusion): chỉ đo 1 file test (`test_f1`, 2 560 mẫu đầu, 366,9 giây) trong 5 file, nên hai ô này là của file đó*

**RE3-TT** (4 file test: f1, f2, f3, f4; 107 996 mẫu)

| Mô hình | Huấn luyện (phút / lần, GPU) | Tham số lúc suy luận | Suy luận CPU (ms / 1000 mẫu) | RAM đỉnh (MB) |
|:--|--:|--:|--:|--:|
| UAC-AD | 7,2 | 809 110 | 148,7 | 2 216 |
| **UAM-AD** | 10,2 | 930 035 | 209,3 | 2 338 |
| AnoFusion | 1 212,3 | 13 233 706 | không đo | – |
| ART | 0,6 | 15 779 | 207,8 | 951 |
| Eadro (có giám sát) | 0,9 | 73 212 | 896,5 | 881 |
| OrEdge (bán giám sát) | 2,7 | 171 366 | 4 431,2 | 1 651 |

**SN** (12 file test: Code_Stop_MediaService, Code_Stop_TextService, Code_Stop_UserService, DB_Redis_CacheLimit_HomeTimeline, DB_Redis_CacheLimit_SocialGraph, DB_Redis_CacheLimit_UserTimeline, Perf_CPU_Contention, Perf_Disk_IO_Stress, Perf_Network_Loss, Svc_Kill_Media, Svc_Kill_SocialGraph, Svc_Kill_UserTimeline; 1 279 mẫu)

| Mô hình | Huấn luyện (phút / lần, GPU) | Tham số lúc suy luận | Suy luận CPU (ms / 1000 mẫu) | RAM đỉnh (MB) |
|:--|--:|--:|--:|--:|
| UAC-AD | 0,2 | 657 406 | 214,4 | 796 |
| **UAM-AD** | 0,2 | 701 891 | 241,1 | 801 |
| AnoFusion | 3,5 | 2 526 964 | 318 456,3 | 4 620 |
| ART | 0,1 | 10 096 | 104,6 | 1 101 |
| Eadro (có giám sát) | 0,1 | 71 853 | 246,5 | 1 145 |
| OrEdge (bán giám sát) | 0,1 | 90 643 | 1 064,1 | 1 327 |

<!-- BANGE END -->

**Điều kiện đo và cách đọc.**
- **Cách đo:** CPU, 6 luồng (mặc định của torch), batch 128 cho mọi mô hình, **mọi file test của dataset** (RE2-OB 319 709 mẫu, RE2-TT 324 450, RE3-OB 102 704, RE3-TT 107 996; SN 12 file, 1 279 cửa sổ ART chấm được); mỗi cặp (mô hình, file test) một tiến trình mới, 179 lượt đo (RE2: 61, RE3-OB: 26, RE3-TT: 20, SN: 72, trong đó 12 lượt AnoFusion ở SN đo thêm ngày 10-31 và 2 lượt AnoFusion rút gọn ở RE2-TT, RE3-OB đo ngày 11-01), không lượt nào lỗi (một lượt RE3-TT `test_f3` bị hệ thống dừng do thiếu RAM và được đo lại). Dữ liệu nạp và mạng dựng xong rồi mới bấm giờ; RAM (working set của Windows, đọc bằng `ctypes` vì máy không có `psutil`) được lấy mẫu mỗi 5 ms trong lúc suy luận. Trong bảng: "Suy luận CPU" = tổng thời gian suy luận của mọi file / tổng số mẫu (ms trên 1 000 mẫu); "RAM đỉnh" là đỉnh working set, **lớn nhất qua các file** (file json từng lượt đo còn giữ cả mốc trước khi suy luận và phần RAM thêm = đỉnh − mốc); "Huấn luyện" là thời gian **một lần huấn luyện** (đúng định nghĩa "giây huấn luyện đến khi dừng"), lấy từ log việc 8, 9 (chạy trên GPU GTX 1050).
- **UAM-AD, UAC-AD:** checkpoint seed 0 đã lưu của từng scenario, nạp qua `run.py --pre_model`. Mã gốc gọi `.cuda()` cứng ở `models/utils.py:170` (tạo một LayerNorm mới mỗi lần gọi attention), nên trên CPU lệnh này được làm thành no-op trong tiến trình đo; mã repo không đổi và phép tính giống hệt.
- **ART, AnoFusion, Eadro, OrEdge:** việc 9 không lưu checkpoint, nên mạng được dựng bằng code của chính runner với **trọng số khởi tạo**. Thời gian và RAM suy luận chỉ phụ thuộc kích thước mạng và dữ liệu, không phụ thuộc giá trị trọng số, nhưng đây không phải đầu ra của mô hình đã huấn luyện. Ma trận thông tin tương hỗ của AnoFusion được thay bằng ma trận ngẫu nhiên cùng kích thước (nó là dữ liệu, không tính vào tham số).
- **Đơn vị chấm khác nhau:** UAM-AD, UAC-AD chấm theo cửa sổ (RCAEval: 30 mẫu, RE2-OB khoảng 1 700 cửa sổ mỗi file; SN: 5 cửa sổ 30 giây), đối thủ chấm từng mẫu. Vì vậy cột so được là "ms / 1000 mẫu" (tính trên số mẫu, không trên số lượt forward). UAM-AD chấm một cửa sổ bằng một lượt forward nên số lượt forward ít hơn nhiều; con số thấp của nó không có nghĩa mạng nhanh hơn.
- **Huấn luyện không cùng giao thức:** UAM-AD, UAC-AD chạy mỗi file test bằng một tiến trình `run.py` riêng nên huấn luyện một lần cho mỗi file test (RE2: 6, RE3-OB: 5, RE3-TT: 4, SN: 12 lần mỗi seed), nhưng các lần của một dataset dùng **cùng** train, unlabel, val và cùng seed; ở RE2-OB (11 104 + 44 414 + 9 796 mẫu trong pkl) đường cong huấn luyện của hai file test khác nhau gần như trùng (val score epoch 0: 0,1948 và 0,1923, lệch do GPU không tất định). Đó là cách bộ chạy được viết, không phải mỗi file test cần một mô hình riêng, nên **bảng chỉ báo thời gian một lần huấn luyện và không cộng các lần**. ART, AnoFusion huấn luyện một lần cho cả dataset; Eadro, OrEdge 3 fold × số file test (18 lần ở RE2, 15 ở RE3-OB, 12 ở RE3-TT) và 12 lần ở SN (một fold cho mỗi scenario). "Huấn luyện (phút / lần)": UAM/UAC-AD = **trung vị** của mọi lần huấn luyện (số file test × 5 seed: 30, 30, 25, 20, 60), mỗi lần là `Train time/epoch` × số epoch từ log; dùng trung vị vì một lần chạy UAC-AD ở RE2-TT (file `socket`, seed 2) bị đứng máy (epoch 0 mất 7 534 giây thay vì khoảng 230 giây), kéo trung bình thêm 5 phút. Đối thủ = trung bình 5 seed của tổng `train_time_s` chia cho số lần huấn luyện (độ lệch chuẩn giữa các seed dưới 10% giá trị trung bình; của Eadro, OrEdge gồm cả dự đoán các fold; AnoFusion chỉ seed 0).
- **AnoFusion:** suy luận đo ở **SN** (cả 12 file; mỗi file chấm lặp 10 lần vì AnoFusion chậm; chấm được 750 cửa sổ, ít hơn 1 279 của ART vì 20 cửa sổ đầu mỗi phiên không đủ lịch sử), ở **RE2-OB** trên một file (`test_cpu`, 877 giây, cả file) và ở **RE3-OB**, **RE2-TT** trên **2 560 mẫu đầu của một file** (`test_f1` 367 giây; `test_cpu` 798 giây; đo 11-01, `measure_cost.py --limit 2560`). Các ô dấu * là của file đó, không phải trung bình các file test, vì đo cả file quá lâu (RE2-OB 6 file khoảng 90 phút; ngoại suy từ số đo: khoảng 1 giờ cho một file RE3-OB, khoảng 4,5 giờ cho một file RE2-TT). Giới hạn 2 560 mẫu không đổi số ms / 1 000 mẫu (chi phí mỗi mẫu như nhau) và RAM đỉnh đã đạt từ các batch đầu. **RE3-TT không đo** (1 357 kênh): RAM thêm tăng ít nhất tuyến tính theo N (5,5; 6,2; 6,0; 7,3 MB mỗi kênh ở N = 193; 465; 583; 596), nên ở 1 357 kênh, batch 128 ước tính cần hơn 7 GB thêm và vượt RAM 7,9 GB của máy; ở việc 9 (GPU) riêng chấm `test_f1` đã mất 72 545 giây (khoảng 20 giờ) và lần chạy dừng sau file đó. Ở việc 9 mô hình đã chấm xong cả 6 file RE2-TT (khoảng 1 308 giây mỗi file) và cả 5 file RE3-OB (183–471 giây); thời gian GPU đó không thay được số đo CPU của Bảng E (RE2-OB, file `test_cpu`: GPU 100 giây, CPU 877 giây). Số tham số không cần suy luận nên đếm được ở cả 5 dataset (`summary_tools/anofusion_params.py`). Thời gian huấn luyện lấy từ việc 9: 6,5 (RE2-OB), 76,6 (RE2-TT; thời gian chạy cả lượt ở T1 là 220 phút vì gồm cả dựng ma trận thông tin tương hỗ và nạp dữ liệu), 14,7 (RE3-OB), 3,5 (SN) và **1 212,3 phút (khoảng 20 giờ) ở RE3-TT**: 5 epoch, mỗi epoch khoảng 4 giờ; lần chạy đó chỉ chấm file `test_f1` rồi dừng, nhưng huấn luyện đã đủ 5 epoch.
- **AnoFusion chậm do mô hình, không do bản dựng (kiểm 11-01).** (1) *Cùng hàm:* với cùng trọng số ngẫu nhiên và cùng đầu vào, đầu ra của `anofusion_model.py` giống hệt mã tác giả (`AIOps-Lab-NKU/AnoFusion`, thư mục `model/`): chênh lệch tối đa bằng 0 ở N = 12 và N = 30 (chế độ eval, CPU). Số tham số của các lớp mà forward gọi tới bằng nhau (309 814; 2 526 964; 13 233 706 ở N = 193; 583; 1 357); mã tác giả còn dựng thêm các lớp forward không gọi (`linT`, `lin1`, `lin2`, BatchNorm: 247 486; 746 686; 1 737 406 tham số) mà Bảng E không tính. Cách tạo cửa sổ (20 bước, đích là bước kế) và công thức điểm theo mã tác giả. Bản dựng chỉ nhanh hơn mã tác giả (GTN chạy một lần thay vì cho từng mẫu trong batch; h·hᵀ dùng chung cho 6 loại cạnh thay vì tính lại 6 lần). (2) *Chi phí tăng theo N³:* theo mã tác giả, độ rộng ẩn của GAT bằng số kênh N, nên mỗi mẫu cần khoảng 42·N³ phép nhân-cộng (6 head × (1 + 6 loại cạnh) × N³). Số đo: 17,6; 143,3; 318,5; 311,6 ms / mẫu ở N = 193; 465; 583; 596, tức 34–59 GFLOP/s, ổn định khi N tăng hơn gấp ba, nên thời gian khớp với mức tính toán tăng theo N³ (không thấy phần chi phí thừa nào tăng nhanh hơn). (3) *Khác tác giả ở đơn vị mô hình:* tác giả chạy cho một service instance (N là số kênh của một service); ta coi cả hệ thống là một instance (N = 193–1 357), là khác biệt (1) trong danh sách thích nghi ở Giai đoạn 2 và là lựa chọn của ta, nên N, và chi phí, lớn hơn nhiều so với cách dùng của tác giả.
- **SN và RE3:** RE3 dùng đúng bộ đo của RE2. SN có worker riêng cho ART, Eadro, OrEdge, AnoFusion (dữ liệu, tỉ lệ chuẩn hoá và cửa sổ chấm lấy từ `run_art_sn.py`, `run_eadro_sn.py`, `run_oredge_sn.py`, `run_anofusion_sn.py`, mạng ở trọng số khởi tạo), còn UAM-AD, UAC-AD dùng checkpoint seed 0 của từng scenario trong `item8/sn`. Mẫu ở SN là một cửa sổ 30 giây; mỗi file test chỉ có 32–263 cửa sổ, nên mỗi file được chấm lặp 30 lần trong đồng hồ (10 lần với AnoFusion vì nó chậm hơn nhiều; cột "ms / 1000 mẫu" chia cho cả số lần lặp) và batch nhỏ hơn nhiều so với RE: **số ms của SN không so được với số của RE**, chỉ so các mô hình với nhau trong SN. Ở SN, ART, Eadro, OrEdge nạp `sessions.pkl` (mọi cửa sổ của mọi phiên) nên RAM đỉnh của chúng cao hơn UAM-AD, UAC-AD, ngược với RE.
- **Độ tin cậy:** mỗi ô thời gian suy luận gộp nhiều lượt đo (4–12 file) nên nhiễu từng lượt được trung bình đi; ở thử nghiệm đầu, hai lần đo cùng một file lệch nhau tới 25% (UAC-AD, RE2-OB, `test_cpu`: 4,9 và 6,4 giây), nên chênh dưới khoảng 10–25% giữa hai mô hình không đáng kể (ví dụ ở RE3-TT, UAM-AD 209,3 và ART 207,8 là ngang nhau). RAM là giá trị lớn nhất nên nhạy với một lượt; đừng đọc kỹ các chênh dưới khoảng 100 MB. **"RAM đỉnh" của UAM-AD, UAC-AD bất lợi cho hai mô hình này ở RCAEval:** chúng chạy qua `run.py`, chương trình nạp luôn dữ liệu train, unlabel, val vào RAM trước khi suy luận (các mô hình đối thủ chỉ nạp train và file test); phần RAM do riêng suy luận (đỉnh − mốc) của UAM-AD ở RE2-OB, RE2-TT là 158 và 308 MB. Checkpoint là seed 0, mỗi mô hình một seed.
- **Đọc kết quả.** (a) **Tham số:** UAM-AD lớn nhất trong UAC-AD, ART, Eadro, OrEdge ở cả 5 dataset (627, 687, 681, 930 và 702 nghìn ở RE2-OB, RE2-TT, RE3-OB, RE3-TT, SN): gấp 9–13 lần Eadro, 43–70 lần ART, 5–53 lần OrEdge (số tham số của OrEdge tăng theo số mẫu log nên khoảng cách nhỏ nhất ở RE3-TT và SN); so với UAC-AD thêm 3,2%, 6,1%, 5,8%, 14,9% và 6,8%, do nhánh trace. **AnoFusion lớn hơn UAM-AD ở bốn dataset:** 2,64; 1,63; 13,23 và 2,53 triệu ở RE2-TT, RE3-OB, RE3-TT, SN (gấp 3,8; 2,4; 14,2 và 3,6 lần UAM-AD); chỉ ở RE2-OB nó nhỏ hơn (310 nghìn, bằng 0,49 lần UAM-AD). Số tham số của AnoFusion tăng theo số kênh (193 kênh ở RE2-OB đến 1 357 ở RE3-TT). (b) **Suy luận CPU:** UAM-AD đắt hơn UAC-AD 1,2 lần (RE2-OB), 1,5 (RE2-TT), 1,2 (RE3-OB), 1,4 (RE3-TT) và 1,1 (SN). So với ART, Eadro, OrEdge: thấp hơn cả ba ở RE2-OB (110,7 so với 168,1; 250,5; 257,5), RE2-TT (161,1 so với 217,1; 916,8; 545,0) và RE3-OB (105,2 so với 121,6; 197,9; 658,9); ngang ART ở RE3-TT (209,3 so với 207,8) và thấp hơn Eadro (896,5), OrEdge (4 431,2); **ở SN cao hơn ART** (241,1 so với 104,6), ngang Eadro (246,5) và thấp hơn OrEdge (1 064,1). **AnoFusion** chậm hơn rất nhiều ở cả bốn nơi đã đo, và chậm dần theo số kênh: 17 636 ms* ở RE2-OB (gấp khoảng 160 lần UAM-AD), 143 313 ms* ở RE3-OB (gấp khoảng 1 360 lần), 311 556 ms* ở RE2-TT (gấp khoảng 1 930 lần) và 318 456 ms / 1 000 cửa sổ ở SN (gấp khoảng 1 320 lần UAM-AD và 3 040 lần ART). Cần đọc kèm lưu ý đơn vị chấm và lưu ý SN ở trên. (c) **RAM đỉnh:** UAM-AD cao nhất trong năm mô hình không kể AnoFusion (UAM-AD, UAC-AD, ART, Eadro, OrEdge) ở bốn dataset RCAEval (1 183, 2 069, 1 173, 2 338 MB; UAC-AD 1 128, 1 917, 1 136, 2 216), một phần do dữ liệu huấn luyện mà `run.py` nạp (xem Độ tin cậy); ở SN UAM-AD 801 MB, ngang UAC-AD (796) và thấp hơn ART (1 101), Eadro (1 145), OrEdge (1 327). **AnoFusion cao hơn UAM-AD ở cả bốn nơi đã đo:** 4 620 MB ở SN (gấp 5,8 lần), 1 695 MB* ở RE2-OB (UAM-AD 1 183), 3 569 MB* ở RE3-OB (UAM-AD 1 173; gấp 3,0 lần) và 5 062 MB* ở RE2-TT (UAM-AD 2 069; gấp 2,4 lần). (d) **Huấn luyện:** mỗi lần UAM-AD mất 18,1, 26,6, 6,3 và 10,2 phút ở RE2-OB, RE2-TT, RE3-OB, RE3-TT (khoảng 217 và 320 giây mỗi epoch ở RE2-OB và RE2-TT, 5 epoch) và 14 giây ở SN; nặng hơn ART, Eadro, OrEdge (0,2–2,7 phút) ở cả bốn dataset RCAEval, nhẹ hơn AnoFusion ở bốn trên năm dataset (trừ RE2-OB: 18,1 so với 6,5 phút); so với UAC-AD thêm 16%, 45%, 43%, 40% và 13% cho mỗi lần. Chưa tách được nguyên nhân khoảng cách lớn với ART (khoảng 21 giây mỗi epoch ở RE2-OB): log cho thấy mỗi epoch của UAM-AD có hai bước (generator, discriminator) mỗi bước khoảng 100 giây ở RE2-OB, nhưng chưa đo riêng ảnh hưởng của lượng dữ liệu hay kích thước mạng.

#### Việc 11 – Bảng D: xác định đúng service lỗi (HR@k)

Định nghĩa, tập mẫu và dataset của bảng: mục 4, Bảng D. Bố cục như T1 của việc 9 (sinh tự động bởi `summary_tools/item9_tables.py`): ô "a±b" = trung bình±độ lệch chuẩn tổng thể qua 5 seed; **Δ = UAM-AD trừ mô hình ở dòng đó, trong ngoặc là % giá trị của mô hình đó (dương = UAM-AD cao hơn)**. AnoFusion không có xếp hạng service; UAC-AD không xuất được.

<!-- BANGD BEGIN -->

| Dataset | Mô hình | HR@1 | Δ HR@1 | HR@3 | Δ HR@3 | HR@5 | Δ HR@5 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| RE2-OB | UAM-AD | 0,759±0,022 | – | 0,949±0,008 | – | 0,976±0,004 | – |
| RE2-OB | ART | 0,753±0,004 | +0,006 (+0,8%) | 0,844±0,004 | +0,104 (+12,4%) | 0,933±0,000 | +0,044 (+4,7%) |
| RE2-OB | Eadro (có giám sát) | 0,667±0,016 | +0,092 (+13,8%) | 0,880±0,003 | +0,068 (+7,8%) | 0,996±0,002 | −0,020 (−2,0%) |
| RE2-OB | OrEdge (bán giám sát) | 0,927±0,003 | −0,168 (−18,1%) | 0,946±0,003 | +0,002 (+0,3%) | 0,952±0,004 | +0,025 (+2,6%) |
| RE3-OB | UAM-AD | 0,824±0,024 | – | 0,902±0,022 | – | 0,959±0,019 | – |
| RE3-OB | ART | 0,727±0,006 | +0,097 (+13,3%) | 0,788±0,007 | +0,114 (+14,5%) | 0,827±0,006 | +0,132 (+15,9%) |
| RE3-OB | Eadro (có giám sát) | 0,853±0,022 | −0,028 (−3,3%) | 1,000±0,000 | −0,098 (−9,8%) | 1,000±0,000 | −0,041 (−4,1%) |
| RE3-OB | OrEdge (bán giám sát) | 0,941±0,009 | −0,117 (−12,4%) | 0,994±0,002 | −0,092 (−9,3%) | 0,999±0,001 | −0,040 (−4,0%) |
| RE2-TT | UAM-AD | 0,721±0,027 | – | 0,871±0,029 | – | 0,908±0,032 | – |
| RE2-TT | ART | 0,447±0,003 | +0,274 (+61,2%) | 0,511±0,004 | +0,360 (+70,4%) | 0,590±0,007 | +0,318 (+53,9%) |
| RE2-TT | Eadro (có giám sát) | 0,539±0,015 | +0,182 (+33,7%) | 0,780±0,007 | +0,091 (+11,6%) | 0,993±0,003 | −0,085 (−8,6%) |
| RE2-TT | OrEdge (bán giám sát) | 0,954±0,007 | −0,233 (−24,4%) | 0,982±0,004 | −0,111 (−11,3%) | 0,994±0,001 | −0,086 (−8,6%) |
| RE3-TT | UAM-AD | 0,251±0,006 | – | 0,332±0,017 | – | 0,417±0,022 | – |
| RE3-TT | ART | 0,344±0,053 | −0,093 (−27,0%) | 0,413±0,032 | −0,081 (−19,6%) | 0,478±0,023 | −0,061 (−12,7%) |
| RE3-TT | Eadro (có giám sát) | 0,621±0,005 | −0,371 (−59,6%) | 1,000±0,000 | −0,668 (−66,8%) | 1,000±0,000 | −0,583 (−58,3%) |
| RE3-TT | OrEdge (bán giám sát) | 0,905±0,011 | −0,654 (−72,3%) | 0,995±0,001 | −0,663 (−66,6%) | 0,997±0,000 | −0,581 (−58,2%) |
| SN | UAM-AD | 0,185±0,017 | – | 0,347±0,055 | – | 0,397±0,036 | – |
| SN | ART | 0,058±0,018 | +0,127 (+219,3%) | 0,161±0,025 | +0,187 (+116,3%) | 0,345±0,049 | +0,052 (+15,2%) |
| SN | Eadro (có giám sát) | 0,074±0,052 | +0,111 (+148,5%) | 0,266±0,027 | +0,082 (+30,8%) | 0,498±0,053 | −0,101 (−20,2%) |
| SN | OrEdge (bán giám sát) | 0,262±0,077 | −0,077 (−29,3%) | 0,506±0,011 | −0,158 (−31,3%) | 0,724±0,039 | −0,327 (−45,1%) |
| **Trung bình 5 dataset** | UAM-AD | 0,548±0,009 | – | 0,680±0,009 | – | 0,731±0,011 | – |
| **Trung bình 5 dataset** | ART | 0,466±0,012 | +0,082 (+17,7%) | 0,543±0,006 | +0,137 (+25,2%) | 0,635±0,013 | +0,097 (+15,3%) |
| **Trung bình 5 dataset** | Eadro (có giám sát) | 0,551±0,015 | −0,003 (−0,5%) | 0,785±0,006 | −0,105 (−13,4%) | 0,897±0,010 | −0,166 (−18,5%) |
| **Trung bình 5 dataset** | OrEdge (bán giám sát) | 0,798±0,017 | −0,250 (−31,3%) | 0,885±0,002 | −0,205 (−23,1%) | 0,933±0,008 | −0,202 (−21,6%) |

*Đếm ô (5 dataset): UAM-AD hơn hoặc kém mô hình ở **mọi seed**; "hoà" = dấu của chênh lệch lẫn giữa các seed.*

| Mô hình | HR@1 | HR@3 | HR@5 |
|:--|:--|:--|:--|
| ART | hơn 3 / hoà 1 / kém 1 | hơn 4 / hoà 0 / kém 1 | hơn 3 / hoà 1 / kém 1 |
| Eadro (có giám sát) | hơn 3 / hoà 1 / kém 1 | hơn 2 / hoà 1 / kém 2 | hơn 0 / hoà 1 / kém 4 |
| OrEdge (bán giám sát) | hơn 0 / hoà 1 / kém 4 | hơn 0 / hoà 1 / kém 4 | hơn 1 / hoà 0 / kém 4 |

<!-- BANGD END -->

**Cách đọc.** (1) **Trung bình 5 dataset:** UAM-AD đạt HR@1 0,548, HR@3 0,680, HR@5 0,731. So với ART hơn ở cả ba k (+17,7%, +25,2%, +15,3%); so với Eadro ngang ở HR@1 (−0,5%) rồi kém dần ở HR@3 (−13,4%) và HR@5 (−18,5%); so với OrEdge kém ở cả ba k (−31,3%, −23,1%, −21,6%). (2) **Đếm theo dataset (mọi seed cùng dấu), bảng đếm ô ở dưới:** ART: HR@1 hơn 3, hoà 1, kém 1; HR@3 hơn 4, kém 1; HR@5 hơn 3, hoà 1, kém 1. Eadro: HR@1 hơn 3, hoà 1, kém 1; HR@3 hơn 2, hoà 1, kém 2; HR@5 hơn 0, hoà 1, kém 4 (khi k lớn Eadro tốt hơn vì HR@3 và HR@5 của nó gần như bằng 1,000 ở RE3-OB và RE3-TT). OrEdge: kém ở 4/5 dataset ở cả ba k. (3) **RE3-TT và SN là chỗ yếu** (mốc tham chiếu ngoài bảng: đoán cố định một service; ngẫu nhiên là k/N): HR@1 của UAM-AD (0,251 và 0,185) thấp hơn đoán cố định (0,50 và 0,27–0,34); ở RE3-TT còn kém ART (−27,0% ở HR@1); HR@5 của SN (0,397) thấp hơn ngẫu nhiên (0,417). Ở SN cả bốn mô hình đều có HR@1 thấp hơn đoán cố định (0,06 đến 0,26). (4) Eadro và OrEdge dùng nhãn service lỗi khi huấn luyện nên chỉ là hàng tham chiếu; đơn vị chấm khác nhau (mục 4) nên chỉ đọc khoảng cách cỡ lớn.

#### Việc 12 – Tổng hợp Bảng A–E (chương kết quả)

Năm bảng của luận văn, theo bố cục mục 4: **hàng = phương pháp, cột = dataset** (SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT). Bảng B, C, D, E **sinh tự động** bởi `summary_tools/final_tables.py` từ cùng dữ liệu với việc 7–11 (số trùng với T1, với kết quả Việc 7, với Bảng D và E ở trên); Bảng A chép từ mục 4 (train và val đã đếm lại bằng pkl ngày 10-31). Ô "a±b" = trung bình±độ lệch chuẩn tổng thể (`np.std`) qua 5 seed của trung bình theo scenario. Chi tiết, cách tính Δ và các lưu ý từng bảng: T1–T3 (việc 9), Việc 7, Việc 10, Việc 11.

**Bảng A – Thống kê dataset**

| Dataset | Hệ / số service | Modality | Đơn vị mẫu | Train (normal) | Val (normal) | Test | Tỉ lệ anomaly trong test | Async |
|:--|:--|:--|:--|--:|--:|:--|:--|:--|
| AnoMod-SN | SocialNetwork / 12 | L+M+T | cửa sổ 30 s | 39 | 57 | 12 file: 3 `Code_Stop` 263 cửa sổ (39 anomaly), 6 `DB_Redis` + `Perf` 80 (10), 3 `Svc_Kill` 32 (4) | 12,5% – 14,8% | Không (Jaeger, sync) |
| RE2-OB | Online Boutique / 11 | L+M+T | 1 giây = 1 mẫu | 11.104 (+ 44.414 `unlabel` cho GAN) | 9.796 (`val.pkl`, đã có) | 6 file theo loại lỗi (cpu, delay, disk, loss, mem, socket), 51.520 – 54.075 mẫu | ≈ 20% (tối đa 20%) | Không |
| RE2-TT | TrainTicket / 27 | L+M+T | 1 giây = 1 mẫu | 11.016 (+ 44.064 `unlabel`) | 9.720 (`val.pkl`) | 6 file theo loại lỗi, 54.075 mẫu mỗi file | 20% (10.815 anomaly/file) | Không |
| RE3-OB | Online Boutique / 11 | L+M+T | 1 giây = 1 mẫu | 3.672 (+ 14.688 `unlabel`) | 3.240 (`val.pkl`, đã có) | 5 file (f1–f5) | tối đa 20% | Không |
| RE3-TT | TrainTicket / 27 | L+M+T | 1 giây = 1 mẫu | 3.672 (+ 14.688 `unlabel`) | 3.240 (`val.pkl`) | 4 file (f1–f4), 26.999 mẫu mỗi file | 20% (5.399 anomaly/file) | Không |
| DeepTraLog | TrainTicket / 35 node | L+T (không có metric) | 1 trace | 6.000 (lấy mẫu xuống từ kho 20.000) | 1.500 | 3 file, mỗi F-case một file: F01 16.424 trace (2.053 bất thường), F02 17.208 (2.151), F13 14.232 (1.779) | 12,5% mỗi file; toàn dataset 132.485 trace, 23.334 bất thường (17,6%) | **Có** (RabbitMQ, SpringAsync) |

Giới hạn của RE2/RE3 (normal trong test gồm cả cửa sổ đã thấy khi train) và các ghi chú còn lại của Bảng A: mục 4.

**Bảng B – Kết quả phát hiện bất thường (5 seed)**

Nhóm không giám sát dùng để kết luận; OrEdge (bán giám sát) và Eadro (có giám sát) chỉ là hàng tham chiếu. Bảng chia hai phần: **phần a so UAC-AD với UAM-AD** (in đậm = trung bình cao hơn trong hai mô hình ở cột đó); **phần b so UAM-AD với các mô hình khác**, không có UAC-AD (in đậm ở B1–B3 = trung bình cao nhất trong AnoFusion, ART, UAM-AD ở cột đó). Chưa xét độ lệch chuẩn khi in đậm.

**Δ và Δ% (hai hàng cuối mỗi bảng).** Phần a: Δ = UAM-AD − UAC-AD, Δ% = Δ / UAC-AD. Phần b: Δ = UAM-AD − mô hình khác, Δ% = Δ / mô hình khác (cùng quy ước với T1, Việc 9). Dương = UAM-AD cao hơn, âm = UAM-AD thấp hơn. Δ là hiệu số tuyệt đối của chỉ số; Δ% là hiệu số tương đối so với giá trị của mô hình đứng sau dấu trừ (không phải điểm phần trăm). Cả hai tính từ trung bình qua seed chưa làm tròn, nên có thể lệch 0,001 so với hiệu của hai ô hiển thị, và chưa xét độ lệch chuẩn. Hai chỗ so không cùng tập: (1) cột Trung bình của AnoFusion (‡) so với trung bình của UAM-AD trên đúng 4 dataset không có RE3-TT (F1 0,641; AUROC 0,879; AUPRC 0,709; khác trung bình 5 dataset ở hàng UAM-AD); (2) Δ ở RE3-TT của AnoFusion (†) tính trên riêng `test_f1`, UAM-AD cũng lấy riêng `test_f1` (như T1), nên khác hiệu của hai ô hiển thị ở hàng đó; vì chỉ một file nên chỉ mang tính tham khảo. Ở B4b, Δ chỉ tính ở RE2-OB và RE2-TT (cùng định nghĩa oracle); ba dataset còn lại ghi –.

**AnoFusion chỉ có 1 seed** (±0,000 là quy ước trình bày, không phải số đo); † = chỉ có `test_f1`, dừng sớm; ‡ = trung bình 4 dataset, không gồm RE3-TT. Cột "Trung bình" = trung bình theo seed của trung bình 5 dataset. UAM-AD và UAC-AD ở RE2-OB, RE2-TT là số chấm lại trên test sắp theo thứ tự thời gian (việc 10, Giai đoạn 3). B4 (F1 oracle): giữa UAC-AD và UAM-AD so được ở mọi dataset (cùng định nghĩa); với đối thủ chỉ so được ở RE2-OB, RE2-TT, vì ở SN, RE3-OB, RE3-TT cột oracle của UAM-AD và UAC-AD có point-adjust còn của đối thủ thì không.

<!-- FINAL B BEGIN -->

**Phần a – UAC-AD so với UAM-AD**

**B1a. F1@val95 (chỉ số chính)**

| Phương pháp | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|--:|--:|--:|--:|--:|--:|
| UAC-AD | 0,523±0,185 | 0,611±0,084 | 0,345±0,026 | **0,647±0,007** | **0,540±0,006** | 0,533±0,052 |
| **UAM-AD** | **0,728±0,004** | **0,681±0,076** | **0,597±0,007** | 0,560±0,019 | 0,513±0,002 | 0,616±0,018 |
| Δ = UAM-AD − UAC-AD | +0,205 | +0,070 | +0,251 | −0,087 | −0,026 | +0,082 |
| Δ% = Δ / UAC-AD | +39,1% | +11,4% | +72,8% | −13,5% | −4,9% | +15,4% |

**B2a. AUROC**

| Phương pháp | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|--:|--:|--:|--:|--:|--:|
| UAC-AD | 0,795±0,194 | 0,880±0,026 | 0,670±0,007 | **0,858±0,001** | **0,745±0,008** | 0,790±0,044 |
| **UAM-AD** | **0,956±0,002** | **0,917±0,016** | **0,809±0,002** | 0,835±0,005 | 0,702±0,003 | 0,844±0,004 |
| Δ = UAM-AD − UAC-AD | +0,161 | +0,037 | +0,139 | −0,023 | −0,042 | +0,054 |
| Δ% = Δ / UAC-AD | +20,3% | +4,2% | +20,7% | −2,7% | −5,7% | +6,9% |

**B3a. AUPRC**

| Phương pháp | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|--:|--:|--:|--:|--:|--:|
| UAC-AD | 0,594±0,173 | 0,750±0,057 | 0,431±0,021 | **0,696±0,004** | **0,432±0,010** | 0,581±0,045 |
| **UAM-AD** | **0,716±0,012** | **0,823±0,025** | **0,680±0,006** | 0,615±0,012 | 0,347±0,003 | 0,636±0,005 |
| Δ = UAM-AD − UAC-AD | +0,123 | +0,073 | +0,249 | −0,080 | −0,086 | +0,056 |
| Δ% = Δ / UAC-AD | +20,6% | +9,7% | +57,7% | −11,5% | −19,8% | +9,6% |

**B4a. F1 oracle (chỉ để đối chiếu, không dùng để kết luận)**

| Phương pháp | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|--:|--:|--:|--:|--:|
| UAC-AD | 0,758±0,043 | 0,713±0,039 | 0,533±0,003 | 0,761±0,002 | 0,645±0,006 |
| **UAM-AD** | 0,849±0,009 | 0,764±0,020 | 0,624±0,006 | 0,755±0,007 | 0,633±0,004 |
| Δ = UAM-AD − UAC-AD | +0,091 | +0,051 | +0,092 | −0,005 | −0,012 |
| Δ% = Δ / UAC-AD | +12,0% | +7,1% | +17,3% | −0,7% | −1,8% |

**Phần b – UAM-AD so với các mô hình khác**

**B1b. F1@val95 (chỉ số chính)**

| Nhóm | Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| Không giám sát | AnoFusion | Unsup | 0,408±0,000 | 0,587±0,000 | 0,584±0,000 | 0,366±0,000 | 0,116±0,000† | 0,486±0,000‡ |
|  | ART | Unsup | 0,553±0,034 | **0,880±0,001** | 0,583±0,002 | 0,523±0,058 | 0,322±0,012 | 0,572±0,013 |
|  | **UAM-AD** | Unsup | **0,728±0,004** | 0,681±0,076 | **0,597±0,007** | **0,560±0,019** | **0,513±0,002** | 0,616±0,018 |
| Tham chiếu có nhãn | OrEdge | Semi-sup | 0,416±0,108 | 0,867±0,012 | 0,831±0,002 | 0,762±0,009 | 0,631±0,006 | 0,701±0,020 |
|  | Eadro | Sup | 0,820±0,016 | 0,711±0,016 | 0,641±0,020 | 0,645±0,026 | 0,365±0,028 | 0,637±0,007 |
| Δ = UAM-AD − mô hình | AnoFusion |  | +0,320 | +0,094 | +0,013 | +0,194 | +0,404† | +0,155‡ |
|  | ART |  | +0,174 | −0,199 | +0,014 | +0,037 | +0,191 | +0,044 |
|  | OrEdge |  | +0,312 | −0,186 | −0,234 | −0,202 | −0,117 | −0,086 |
|  | Eadro |  | −0,092 | −0,030 | −0,044 | −0,085 | +0,148 | −0,021 |
| Δ% = Δ / mô hình | AnoFusion |  | +78,5% | +16,0% | +2,2% | +53,1% | +348,2%† | +32,0%‡ |
|  | ART |  | +31,5% | −22,6% | +2,4% | +7,1% | +59,3% | +7,6% |
|  | OrEdge |  | +74,9% | −21,4% | −28,2% | −26,5% | −18,6% | −12,2% |
|  | Eadro |  | −11,2% | −4,3% | −6,8% | −13,2% | +40,4% | −3,3% |

**B2b. AUROC**

| Nhóm | Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| Không giám sát | AnoFusion | Unsup | 0,731±0,000 | 0,819±0,000 | 0,773±0,000 | 0,630±0,000 | 0,502±0,000† | 0,739±0,000‡ |
|  | ART | Unsup | 0,890±0,019 | **0,981±0,002** | **0,833±0,002** | **0,893±0,005** | 0,684±0,013 | 0,856±0,004 |
|  | **UAM-AD** | Unsup | **0,956±0,002** | 0,917±0,016 | 0,809±0,002 | 0,835±0,005 | **0,702±0,003** | 0,844±0,004 |
| Tham chiếu có nhãn | OrEdge | Semi-sup | 0,844±0,039 | 0,973±0,003 | 0,967±0,005 | 0,892±0,003 | 0,764±0,004 | 0,888±0,008 |
|  | Eadro | Sup | 0,980±0,003 | 0,867±0,005 | 0,838±0,014 | 0,817±0,022 | 0,606±0,020 | 0,821±0,005 |
| Δ = UAM-AD − mô hình | AnoFusion |  | +0,225 | +0,098 | +0,035 | +0,204 | +0,217† | +0,141‡ |
|  | ART |  | +0,067 | −0,064 | −0,025 | −0,059 | +0,019 | −0,012 |
|  | OrEdge |  | +0,112 | −0,056 | −0,159 | −0,058 | −0,062 | −0,044 |
|  | Eadro |  | −0,024 | +0,051 | −0,029 | +0,018 | +0,097 | +0,022 |
| Δ% = Δ / mô hình | AnoFusion |  | +30,8% | +12,0% | +4,5% | +32,4% | +43,2%† | +19,0%‡ |
|  | ART |  | +7,5% | −6,5% | −3,0% | −6,6% | +2,8% | −1,4% |
|  | OrEdge |  | +13,3% | −5,7% | −16,4% | −6,5% | −8,1% | −5,0% |
|  | Eadro |  | −2,4% | +5,8% | −3,5% | +2,2% | +16,0% | +2,7% |

**B3b. AUPRC**

| Nhóm | Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| Không giám sát | AnoFusion | Unsup | 0,455±0,000 | 0,693±0,000 | 0,672±0,000 | 0,447±0,000 | 0,206±0,000† | 0,567±0,000‡ |
|  | ART | Unsup | 0,590±0,038 | **0,957±0,004** | **0,688±0,003** | **0,746±0,015** | **0,448±0,014** | 0,686±0,008 |
|  | **UAM-AD** | Unsup | **0,716±0,012** | 0,823±0,025 | 0,680±0,006 | 0,615±0,012 | 0,347±0,003 | 0,636±0,005 |
| Tham chiếu có nhãn | OrEdge | Semi-sup | 0,546±0,061 | 0,959±0,008 | 0,949±0,003 | 0,832±0,005 | 0,690±0,003 | 0,795±0,013 |
|  | Eadro | Sup | 0,857±0,013 | 0,791±0,011 | 0,728±0,021 | 0,727±0,024 | 0,443±0,017 | 0,709±0,005 |
| Δ = UAM-AD − mô hình | AnoFusion |  | +0,261 | +0,130 | +0,008 | +0,169 | +0,152† | +0,142‡ |
|  | ART |  | +0,126 | −0,134 | −0,008 | −0,131 | −0,102 | −0,050 |
|  | OrEdge |  | +0,170 | −0,135 | −0,269 | −0,217 | −0,344 | −0,159 |
|  | Eadro |  | −0,141 | +0,033 | −0,048 | −0,111 | −0,096 | −0,073 |
| Δ% = Δ / mô hình | AnoFusion |  | +57,4% | +18,8% | +1,2% | +37,8% | +73,9%† | +25,1%‡ |
|  | ART |  | +21,5% | −14,0% | −1,1% | −17,5% | −22,7% | −7,2% |
|  | OrEdge |  | +31,1% | −14,1% | −28,4% | −26,1% | −49,8% | −20,0% |
|  | Eadro |  | −16,4% | +4,2% | −6,6% | −15,3% | −21,7% | −10,2% |

**B4b. F1 oracle (chỉ để đối chiếu, không dùng để kết luận)**

| Nhóm | Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|:--|:--|--:|--:|--:|--:|--:|
| Không giám sát | AnoFusion | Unsup | 0,575±0,000 | 0,654±0,000 | 0,668±0,000 | 0,460±0,000 | 0,351±0,000† |
|  | ART | Unsup | 0,664±0,028 | 0,923±0,004 | 0,677±0,001 | 0,756±0,017 | 0,473±0,007 |
|  | **UAM-AD** | Unsup | 0,849±0,009 | 0,764±0,020 | 0,624±0,006 | 0,755±0,007 | 0,633±0,004 |
| Tham chiếu có nhãn | OrEdge | Semi-sup | 0,654±0,054 | 0,945±0,008 | 0,926±0,002 | 0,815±0,009 | 0,666±0,003 |
|  | Eadro | Sup | 0,883±0,010 | 0,749±0,016 | 0,691±0,020 | 0,679±0,024 | 0,443±0,009 |
| Δ = UAM-AD − mô hình | AnoFusion |  | – | +0,109 | −0,044 | – | – |
|  | ART |  | – | −0,159 | −0,052 | – | – |
|  | OrEdge |  | – | −0,181 | −0,302 | – | – |
|  | Eadro |  | – | +0,015 | −0,067 | – | – |
| Δ% = Δ / mô hình | AnoFusion |  | – | +16,7% | −6,6% | – | – |
|  | ART |  | – | −17,2% | −7,7% | – | – |
|  | OrEdge |  | – | −19,1% | −32,6% | – | – |
|  | Eadro |  | – | +2,0% | −9,7% | – | – |

<!-- FINAL B END -->

**Bảng C – Sync vs async trên DeepTraLog (F01, F02, F13; 5 seed)**

Mỗi dòng là một cấu hình, mỗi cột một chỉ số. Mỗi F-case vừa có dấu hiệu async vừa có dấu hiệu nhánh sync thấy được (mục 2.3), nên đọc theo hai mệnh đề: đủ bộ không thua nhánh lẻ nào và hơn ở phần lớn các cặp; so các nhánh lẻ với nhau. "Chỉ sync" và "chỉ async" vẫn giữ phần tái tạo log. GGNN chấm từng trace và huấn luyện trên dữ liệu riêng, nên không cùng điều kiện với UAM-AD (xem Việc 7).

**Cặp cần so:** hai dòng in đậm (UAM-AD đủ bộ và DeepTraLog GGNN) ở mỗi F-case; trong mỗi cột, số in đậm là của bên cao hơn trong cặp đó (so ở 3 chữ số thập phân, chưa xét độ lệch chuẩn). Hai dòng chỉ-sync, chỉ-async dùng để so nội bộ UAM-AD (mệnh đề 1). Đếm 12 ô (3 F-case × 4 chỉ số): GGNN cao hơn ở 7 ô, UAM-AD đủ bộ cao hơn ở 5 ô. F01: GGNN hơn ở F1, AUPRC, F1 oracle, UAM-AD hơn ở AUROC; F02: GGNN hơn cả bốn; F13: UAM-AD hơn cả bốn.

**F1 oracle** = F1 tốt nhất khi quét mọi ngưỡng trên chính điểm số của dòng đó, không point-adjust (định nghĩa của đối thủ); chỉ để đối chiếu, không dùng để kết luận. Với ba cấu hình UAM-AD, F1 oracle tính lại từ checkpoint đã lưu bằng `codes/common/rescore_dtl_oracle.py` (không huấn luyện lại): F1@val95, AUROC, AUPRC dựng lại khớp `info_score.txt` ở cả 45 checkpoint (lệch dưới 1e-3) và oracle quét trên đúng điểm norm_sum đó. Không dùng `oracle_f1` ghi sẵn trong `info_score.txt`, vì đó là oracle có point-adjust của điểm thô (raw), có thể thấp hơn cả F1@val95 (F01 đủ bộ seed 0: 0,631 so với 0,772). GGNN lấy `oracle_f1` của chính nó. Ở cột F1 oracle, đủ bộ cao hơn nhánh lẻ ở 4 trong 6 cặp và ngang ở 2 (F01 so với chỉ-sync: 0,800 so với 0,809; F02 so với chỉ-async: 0,468 so với 0,492; cả hai chênh dưới một độ lệch chuẩn của đủ bộ), không cặp nào thấp hơn quá một độ lệch chuẩn; kết luận của mệnh đề 1 không đổi.

<!-- FINAL C BEGIN -->

| F-case | Cấu hình | F1@val95 | AUROC | AUPRC | F1 oracle |
|:--|:--|--:|--:|--:|--:|
| F01 | Chỉ sync | 0,765±0,003 | 0,935±0,007 | 0,830±0,002 | 0,809±0,003 |
|  | Chỉ async | 0,258±0,000 | 0,855±0,006 | 0,345±0,005 | 0,585±0,003 |
|  | **Đủ bộ sync + async (UAM-AD)** | 0,775±0,003 | **0,971±0,004** | 0,859±0,005 | 0,800±0,017 |
|  | **DeepTraLog GGNN** | **0,793±0,003** | 0,936±0,014 | **0,898±0,006** | **0,933±0,001** |
| F02 | Chỉ sync | 0,107±0,001 | 0,664±0,031 | 0,170±0,014 | 0,338±0,018 |
|  | Chỉ async | 0,105±0,000 | 0,803±0,002 | 0,285±0,002 | 0,492±0,002 |
|  | **Đủ bộ sync + async (UAM-AD)** | 0,133±0,014 | 0,803±0,033 | 0,256±0,037 | 0,468±0,028 |
|  | **DeepTraLog GGNN** | **0,856±0,007** | **0,998±0,001** | **0,981±0,010** | **0,964±0,024** |
| F13 | Chỉ sync | 0,852±0,004 | 0,989±0,000 | 0,892±0,008 | 0,874±0,012 |
|  | Chỉ async | 0,434±0,001 | 0,805±0,002 | 0,467±0,002 | 0,506±0,010 |
|  | **Đủ bộ sync + async (UAM-AD)** | **0,866±0,002** | **0,990±0,001** | **0,893±0,014** | **0,898±0,021** |
|  | **DeepTraLog GGNN** | 0,745±0,008 | 0,943±0,016 | 0,863±0,012 | 0,880±0,001 |

<!-- FINAL C END -->

**Bảng D – Xác định đúng service lỗi (HR@k, 5 seed)**

UAM-AD: xếp hạng trace + KPI (việc 10, Giai đoạn 3), tính trên cùng tập mẫu bất thường có service lỗi với ART, Eadro, OrEdge. Đơn vị chấm khác nhau (UAM-AD cửa sổ 30 mẫu; ART từng mẫu; Eadro, OrEdge cửa sổ 10) nên chỉ đọc khoảng cách cỡ lớn. AnoFusion và UAC-AD không xuất được xếp hạng service. Chi tiết (Δ, đếm hơn/hoà/kém theo k): Việc 11. Dưới mỗi bảng có Δ = UAM-AD − mô hình khác và Δ% = Δ / mô hình khác (dương = UAM-AD cao hơn; cách tính như Bảng B). Chỗ UAM-AD thấp rõ nhất là RE3-TT so với OrEdge và Eadro (HR@1 0,251 so với 0,905 và 0,621).

<!-- FINAL D BEGIN -->

**D1. HR@1**

| Nhóm | Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| Không giám sát | **UAM-AD** | Unsup | 0,185±0,017 | 0,759±0,022 | 0,721±0,027 | 0,824±0,024 | 0,251±0,006 | 0,548±0,009 |
|  | ART | Unsup | 0,058±0,018 | 0,753±0,004 | 0,447±0,003 | 0,727±0,006 | 0,344±0,053 | 0,466±0,012 |
| Tham chiếu có nhãn | OrEdge | Semi-sup | 0,262±0,077 | 0,927±0,003 | 0,954±0,007 | 0,941±0,009 | 0,905±0,011 | 0,798±0,017 |
|  | Eadro | Sup | 0,074±0,052 | 0,667±0,016 | 0,539±0,015 | 0,853±0,022 | 0,621±0,005 | 0,551±0,015 |
| Δ = UAM-AD − mô hình | ART |  | +0,127 | +0,006 | +0,274 | +0,097 | −0,093 | +0,082 |
|  | OrEdge |  | −0,077 | −0,168 | −0,233 | −0,117 | −0,654 | −0,250 |
|  | Eadro |  | +0,111 | +0,092 | +0,182 | −0,028 | −0,371 | −0,003 |
| Δ% = Δ / mô hình | ART |  | +219,3% | +0,8% | +61,2% | +13,3% | −27,0% | +17,7% |
|  | OrEdge |  | −29,3% | −18,1% | −24,4% | −12,4% | −72,3% | −31,3% |
|  | Eadro |  | +148,5% | +13,8% | +33,7% | −3,3% | −59,6% | −0,5% |

**D2. HR@3**

| Nhóm | Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| Không giám sát | **UAM-AD** | Unsup | 0,347±0,055 | 0,949±0,008 | 0,871±0,029 | 0,902±0,022 | 0,332±0,017 | 0,680±0,009 |
|  | ART | Unsup | 0,161±0,025 | 0,844±0,004 | 0,511±0,004 | 0,788±0,007 | 0,413±0,032 | 0,543±0,006 |
| Tham chiếu có nhãn | OrEdge | Semi-sup | 0,506±0,011 | 0,946±0,003 | 0,982±0,004 | 0,994±0,002 | 0,995±0,001 | 0,885±0,002 |
|  | Eadro | Sup | 0,266±0,027 | 0,880±0,003 | 0,780±0,007 | 1,000±0,000 | 1,000±0,000 | 0,785±0,006 |
| Δ = UAM-AD − mô hình | ART |  | +0,187 | +0,104 | +0,360 | +0,114 | −0,081 | +0,137 |
|  | OrEdge |  | −0,158 | +0,002 | −0,111 | −0,092 | −0,663 | −0,205 |
|  | Eadro |  | +0,082 | +0,068 | +0,091 | −0,098 | −0,668 | −0,105 |
| Δ% = Δ / mô hình | ART |  | +116,3% | +12,4% | +70,4% | +14,5% | −19,6% | +25,2% |
|  | OrEdge |  | −31,3% | +0,3% | −11,3% | −9,3% | −66,6% | −23,1% |
|  | Eadro |  | +30,8% | +7,8% | +11,6% | −9,8% | −66,8% | −13,4% |

**D3. HR@5**

| Nhóm | Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT | Trung bình |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| Không giám sát | **UAM-AD** | Unsup | 0,397±0,036 | 0,976±0,004 | 0,908±0,032 | 0,959±0,019 | 0,417±0,022 | 0,731±0,011 |
|  | ART | Unsup | 0,345±0,049 | 0,933±0,000 | 0,590±0,007 | 0,827±0,006 | 0,478±0,023 | 0,635±0,013 |
| Tham chiếu có nhãn | OrEdge | Semi-sup | 0,724±0,039 | 0,952±0,004 | 0,994±0,001 | 0,999±0,001 | 0,997±0,000 | 0,933±0,008 |
|  | Eadro | Sup | 0,498±0,053 | 0,996±0,002 | 0,993±0,003 | 1,000±0,000 | 1,000±0,000 | 0,897±0,010 |
| Δ = UAM-AD − mô hình | ART |  | +0,052 | +0,044 | +0,318 | +0,132 | −0,061 | +0,097 |
|  | OrEdge |  | −0,327 | +0,025 | −0,086 | −0,040 | −0,581 | −0,202 |
|  | Eadro |  | −0,101 | −0,020 | −0,085 | −0,041 | −0,583 | −0,166 |
| Δ% = Δ / mô hình | ART |  | +15,2% | +4,7% | +53,9% | +15,9% | −12,7% | +15,3% |
|  | OrEdge |  | −45,1% | +2,6% | −8,6% | −4,0% | −58,2% | −21,6% |
|  | Eadro |  | −20,2% | −2,0% | −8,6% | −4,1% | −58,3% | −18,5% |

<!-- FINAL D END -->

**Bảng E – Chi phí**

Đo trên mọi file test của dataset, CPU 6 luồng, batch 128, mỗi (mô hình, file test) một tiến trình mới (Việc 10). Huấn luyện = một lần huấn luyện trên GPU GTX 1050: UAM-AD, UAC-AD là trung vị của mọi lần huấn luyện; đối thủ là trung bình 5 seed (AnoFusion 1 seed; RE3-TT: 5 epoch, lần chạy chỉ chấm `test_f1`). Suy luận và RAM đỉnh: đối thủ ở trọng số khởi tạo (việc 9 không lưu checkpoint); RAM đỉnh của UAM-AD, UAC-AD ở RCAEval gồm cả dữ liệu huấn luyện do `run.py` nạp; số ms của SN không so được với RE (file test chỉ 32–263 cửa sổ). `*` = AnoFusion chỉ đo 1 file test (RE2-OB: cả file; RE3-OB, RE2-TT: 2 560 mẫu đầu); `–` = không đo (AnoFusion ở RE3-TT, 1 357 kênh: ước tính vượt RAM của máy, và ở việc 9 chấm riêng `test_f1` đã mất khoảng 20 giờ trên GPU, xem Việc 10; số tham số của nó đếm được ở cả 5 dataset). AnoFusion ở SN: 750 cửa sổ chấm được, mỗi file chấm lặp 10 lần (các mô hình khác 30 lần). Bảng E chia hai phần như Bảng B: phần a UAC-AD so với UAM-AD, phần b UAM-AD so với các mô hình khác.

<!-- FINAL E BEGIN -->

**Phần a – UAC-AD so với UAM-AD**

**E1a. Huấn luyện, phút / lần (GPU GTX 1050)**

| Phương pháp | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|--:|--:|--:|--:|--:|
| UAC-AD | 0,21 | 15,6 | 18,3 | 4,4 | 7,2 |
| **UAM-AD** | 0,23 | 18,1 | 26,6 | 6,3 | 10,2 |

**E2a. Số tham số lúc suy luận**

| Phương pháp | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|--:|--:|--:|--:|--:|
| UAC-AD | 657 406 | 607 870 | 647 473 | 643 732 | 809 110 |
| **UAM-AD** | 701 891 | 627 395 | 686 953 | 681 327 | 930 035 |

**E3a. Suy luận CPU, ms / 1000 mẫu**

| Phương pháp | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|--:|--:|--:|--:|--:|
| UAC-AD | 214,4 | 93,1 | 108,7 | 87,5 | 148,7 |
| **UAM-AD** | 241,1 | 110,7 | 161,1 | 105,2 | 209,3 |

**E4a. RAM đỉnh khi suy luận, MB**

| Phương pháp | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|--:|--:|--:|--:|--:|
| UAC-AD | 796 | 1 128 | 1 917 | 1 136 | 2 216 |
| **UAM-AD** | 801 | 1 183 | 2 069 | 1 173 | 2 338 |

**Phần b – UAM-AD so với các mô hình khác**

**E1b. Huấn luyện, phút / lần (GPU GTX 1050)**

| Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|:--|--:|--:|--:|--:|--:|
| AnoFusion | Unsup | 3,5 | 6,5 | 76,6 | 14,7 | 1 212,3 |
| ART | Unsup | 0,07 | 1,7 | 1,8 | 0,61 | 0,63 |
| **UAM-AD** | Unsup | 0,23 | 18,1 | 26,6 | 6,3 | 10,2 |
| OrEdge | Semi-sup | 0,14 | 1,5 | 1,6 | 0,66 | 2,7 |
| Eadro | Sup | 0,07 | 0,60 | 1,7 | 0,24 | 0,88 |

**E2b. Số tham số lúc suy luận**

| Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|:--|--:|--:|--:|--:|--:|
| AnoFusion | Unsup | 2 526 964 | 309 814 | 2 637 542 | 1 631 462 | 13 233 706 |
| ART | Unsup | 10 096 | 10 612 | 15 908 | 10 612 | 15 779 |
| **UAM-AD** | Unsup | 701 891 | 627 395 | 686 953 | 681 327 | 930 035 |
| OrEdge | Semi-sup | 90 643 | 11 900 | 13 830 | 61 115 | 171 366 |
| Eadro | Sup | 71 853 | 72 300 | 73 340 | 72 300 | 73 212 |

**E3b. Suy luận CPU, ms / 1000 mẫu**

| Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|:--|--:|--:|--:|--:|--:|
| AnoFusion | Unsup | 318 456,3 | 17 635,6* | 311 556,3* | 143 313,1* | – |
| ART | Unsup | 104,6 | 168,1 | 217,1 | 121,6 | 207,8 |
| **UAM-AD** | Unsup | 241,1 | 110,7 | 161,1 | 105,2 | 209,3 |
| OrEdge | Semi-sup | 1 064,1 | 257,5 | 545,0 | 658,9 | 4 431,2 |
| Eadro | Sup | 246,5 | 250,5 | 916,8 | 197,9 | 896,5 |

**E4b. RAM đỉnh khi suy luận, MB**

| Phương pháp | Học | SN | RE2-OB | RE2-TT | RE3-OB | RE3-TT |
|:--|:--|--:|--:|--:|--:|--:|
| AnoFusion | Unsup | 4 620 | 1 695* | 5 062* | 3 569* | – |
| ART | Unsup | 1 101 | 787 | 1 071 | 751 | 951 |
| **UAM-AD** | Unsup | 801 | 1 183 | 2 069 | 1 173 | 2 338 |
| OrEdge | Semi-sup | 1 327 | 608 | 874 | 741 | 1 651 |
| Eadro | Sup | 1 145 | 607 | 1 012 | 569 | 881 |

<!-- FINAL E END -->

**Điều phải ghi trong luận văn** (yêu cầu của việc 12):
- **Cấu hình cố định, không tối ưu trên test:** UAM-AD và UAC-AD dùng chung `frozen_config.json`, chốt trước khi chạy 5 seed cuối; đối thủ chạy cấu hình mặc định của tác giả. Kết quả không phải kết quả tốt nhất có thể của từng mô hình (mục 2.2).
- **Tham số riêng của UAM-AD chọn khi đã nhìn kết quả:** `activity_penalty_weight` 1,5 và `gate_delta_lr_mult` 10 (SN); `score_rule=norm_sum` (chọn khi làm DeepTraLog; S6). Đây là thành phần của mô hình đề xuất, không phải quét tham số.
- **Đối thủ có nhãn chỉ là hàng tham chiếu:** Eadro (có giám sát), OrEdge (bán giám sát); kết luận "UAM-AD hơn" chỉ dựa trên nhóm không giám sát.
- **Giới hạn của số liệu:** AnoFusion 1 seed, RE3-TT chỉ `test_f1`, suy luận AnoFusion đo ở SN (12 file) và trên 1 file của RE2-OB, RE3-OB, RE2-TT (RE3-TT không đo); ngân sách huấn luyện khác nhau giữa dataset (RE: 5 epoch, SN: 50, DeepTraLog: 10); đơn vị chấm khác nhau giữa các mô hình; test DeepTraLog trộn nhiều API và chưa có cấu hình log-only; normal trong test RE2/RE3 gồm cả cửa sổ đã thấy khi train.

### Rủi ro và cách xử lý

- **ART khó dựng**: trong nhóm không giám sát không có phương án thay (MSTGAD/OrEdge dùng nhãn). Khi đó chỉ còn AnoFusion làm đối thủ không giám sát, nên nên dựng thử ART sớm (việc 6).
- **Khối lượng tính toán** [CN 10-13]: GTX 1050 3GB, ~8GB RAM, chạy tuần tự, tổng dự kiến trên 100 giờ máy chưa tính đối thủ. Cách giảm: chốt số epoch bằng chạy thử (mục 2.2), chạy theo thứ tự ưu tiên của Giai đoạn 3, hàng tham chiếu (Eadro, OrEdge) chạy ít dataset hơn.
- **Mệnh đề của Bảng C không đứng ở một F-case hay một chỉ số**: báo đúng kết quả đo, không chỉnh tham số theo kết quả test.

## Phụ lục (tra cứu)

Ba phụ lục dưới đây là số liệu chi tiết và chẩn đoán phụ; kết luận đã nằm ở các mục trên.

### Phụ lục A. UAM-AD và UAC-AD: chi tiết từng seed → dataset → scenario (S4)

##### S4. Chi tiết từng seed → từng dataset → từng scenario (một lần chạy mỗi ô, `norm_sum`)

Mỗi seed có một bảng tổng theo dataset (trung bình qua scenario; cột cuối là **tổng phút** của dataset trong seed đó) rồi đến bảng từng scenario của từng dataset (cột cuối là **phút** của lần chạy). Thứ tự dataset: SN, RE3-TT, RE3-OB, RE2-OB, RE2-TT. Ở RE2-OB, RE2-TT, F1@val95, AUROC, AUPRC là số chấm lại theo thứ tự thời gian và F1 oracle là bản không point-adjust (việc 10); HR@1/3/5 của UAM-AD ở mọi dataset là xếp hạng trace + KPI (việc 10).

###### Seed 0

**Seed 0: theo dataset**

| Dataset | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SN | UAC-AD | 0,303 | 0,551 | 0,382 | 0,706 | – | – | – | 3,2 |
| SN | UAM-AD | 0,734 | 0,956 | 0,705 | 0,850 | 0,196 | 0,375 | 0,403 | 4,0 |
| RE3-TT | UAC-AD | 0,530 | 0,732 | 0,416 | 0,635 | – | – | – | 32,5 |
| RE3-TT | UAM-AD | 0,515 | 0,705 | 0,350 | 0,631 | 0,244 | 0,318 | 0,392 | 48,1 |
| RE3-OB | UAC-AD | 0,640 | 0,857 | 0,691 | 0,760 | – | – | – | 26,9 |
| RE3-OB | UAM-AD | 0,556 | 0,832 | 0,611 | 0,750 | 0,778 | 0,859 | 0,921 | 36,3 |
| RE2-OB | UAC-AD | 0,519 | 0,845 | 0,663 | 0,682 | – | – | – | 116,3 |
| RE2-OB | UAM-AD | 0,713 | 0,923 | 0,828 | 0,760 | 0,752 | 0,947 | 0,975 | 126,6 |
| RE2-TT | UAC-AD | 0,321 | 0,661 | 0,413 | 0,535 | – | – | – | 140,6 |
| RE2-TT | UAM-AD | 0,589 | 0,810 | 0,685 | 0,627 | 0,687 | 0,835 | 0,872 | 184,1 |

**Seed 0: SN, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `Code_Stop_MediaService` | UAC-AD | 0,519 | 0,741 | 0,602 | 0,918 | – | – | – | 0,4 |
| `Code_Stop_MediaService` | UAM-AD | 0,750 | 0,965 | 0,822 | 0,897 | 0,718 | 0,846 | 0,872 | 0,3 |
| `Code_Stop_TextService` | UAC-AD | 0,643 | 0,760 | 0,727 | 0,907 | – | – | – | 0,3 |
| `Code_Stop_TextService` | UAM-AD | 0,703 | 0,975 | 0,900 | 0,929 | 0,667 | 0,744 | 0,923 | 0,3 |
| `Code_Stop_UserService` | UAC-AD | 0,705 | 0,869 | 0,815 | 0,916 | – | – | – | 0,3 |
| `Code_Stop_UserService` | UAM-AD | 0,745 | 0,982 | 0,945 | 0,894 | 0,079 | 0,632 | 0,684 | 0,3 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAC-AD | 0,353 | 0,736 | 0,444 | 0,444 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAM-AD | 0,833 | 0,965 | 0,727 | 0,833 | 0,300 | 0,400 | 0,400 | 0,3 |
| `DB_Redis_CacheLimit_SocialGraph` | UAC-AD | 0,600 | 0,735 | 0,707 | 0,857 | – | – | – | 0,2 |
| `DB_Redis_CacheLimit_SocialGraph` | UAM-AD | 0,857 | 0,985 | 0,832 | 0,900 | 0,000 | 0,000 | 0,000 | 0,3 |
| `DB_Redis_CacheLimit_UserTimeline` | UAC-AD | 0,000 | 0,347 | 0,101 | 0,533 | – | – | – | 0,2 |
| `DB_Redis_CacheLimit_UserTimeline` | UAM-AD | 0,556 | 0,942 | 0,583 | 0,706 | 0,000 | 0,000 | 0,000 | 0,3 |
| `Perf_CPU_Contention` | UAC-AD | 0,235 | 0,669 | 0,337 | 0,400 | – | – | – | 0,2 |
| `Perf_CPU_Contention` | UAM-AD | 0,769 | 0,960 | 0,724 | 0,833 | – | – | – | 0,3 |
| `Perf_Disk_IO_Stress` | UAC-AD | 0,353 | 0,635 | 0,267 | 0,400 | – | – | – | 0,2 |
| `Perf_Disk_IO_Stress` | UAM-AD | 0,640 | 0,916 | 0,444 | 0,727 | – | – | – | 0,4 |
| `Perf_Network_Loss` | UAC-AD | 0,222 | 0,555 | 0,224 | 0,571 | – | – | – | 0,2 |
| `Perf_Network_Loss` | UAM-AD | 0,909 | 0,985 | 0,798 | 0,952 | – | – | – | 0,4 |
| `Svc_Kill_Media` | UAC-AD | 0,000 | 0,154 | 0,133 | 0,889 | – | – | – | 0,6 |
| `Svc_Kill_Media` | UAM-AD | 0,750 | 0,962 | 0,679 | 0,889 | 0,000 | 0,750 | 0,750 | 0,3 |
| `Svc_Kill_SocialGraph` | UAC-AD | 0,000 | 0,204 | 0,100 | 0,750 | – | – | – | 0,2 |
| `Svc_Kill_SocialGraph` | UAM-AD | 0,545 | 0,889 | 0,387 | 0,750 | 0,000 | 0,000 | 0,000 | 0,3 |
| `Svc_Kill_UserTimeline` | UAC-AD | 0,000 | 0,211 | 0,133 | 0,889 | – | – | – | 0,2 |
| `Svc_Kill_UserTimeline` | UAM-AD | 0,750 | 0,942 | 0,622 | 0,889 | 0,000 | 0,000 | 0,000 | 0,3 |

**Seed 0: RE3-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,528 | 0,739 | 0,425 | 0,645 | – | – | – | 8,4 |
| `f1` | UAM-AD | 0,523 | 0,720 | 0,362 | 0,642 | 0,304 | 0,368 | 0,433 | 12,0 |
| `f2` | UAC-AD | 0,530 | 0,726 | 0,415 | 0,631 | – | – | – | 8,0 |
| `f2` | UAM-AD | 0,508 | 0,690 | 0,345 | 0,624 | 0,275 | 0,347 | 0,399 | 12,0 |
| `f3` | UAC-AD | 0,538 | 0,745 | 0,420 | 0,635 | – | – | – | 8,1 |
| `f3` | UAM-AD | 0,520 | 0,723 | 0,356 | 0,637 | 0,126 | 0,203 | 0,337 | 12,1 |
| `f4` | UAC-AD | 0,523 | 0,720 | 0,406 | 0,629 | – | – | – | 8,0 |
| `f4` | UAM-AD | 0,510 | 0,686 | 0,337 | 0,620 | 0,272 | 0,355 | 0,398 | 12,0 |

**Seed 0: RE3-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,843 | 0,960 | 0,901 | 0,913 | – | – | – | 5,3 |
| `f1` | UAM-AD | 0,737 | 0,934 | 0,802 | 0,900 | 0,374 | 0,639 | 0,893 | 7,2 |
| `f2` | UAC-AD | 0,891 | 0,966 | 0,914 | 0,945 | – | – | – | 5,2 |
| `f2` | UAM-AD | 0,866 | 0,953 | 0,841 | 0,939 | 0,951 | 0,966 | 0,970 | 7,1 |
| `f3` | UAC-AD | 0,463 | 0,823 | 0,504 | 0,664 | – | – | – | 5,6 |
| `f3` | UAM-AD | 0,247 | 0,756 | 0,362 | 0,649 | 0,946 | 0,986 | 0,989 | 7,3 |
| `f4` | UAC-AD | 0,154 | 0,576 | 0,249 | 0,363 | – | – | – | 5,5 |
| `f4` | UAM-AD | 0,161 | 0,574 | 0,241 | 0,356 | 0,656 | 0,723 | 0,767 | 7,4 |
| `f5` | UAC-AD | 0,851 | 0,958 | 0,889 | 0,913 | – | – | – | 5,3 |
| `f5` | UAM-AD | 0,767 | 0,941 | 0,809 | 0,903 | 0,962 | 0,979 | 0,986 | 7,2 |

**Seed 0: RE2-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,433 | 0,789 | 0,392 | 0,574 | – | – | – | 19,5 |
| `cpu` | UAM-AD | 0,631 | 0,901 | 0,756 | 0,652 | 0,889 | 0,978 | 0,993 | 21,1 |
| `delay` | UAC-AD | 0,549 | 0,857 | 0,688 | 0,636 | – | – | – | 19,2 |
| `delay` | UAM-AD | 0,764 | 0,960 | 0,901 | 0,840 | 0,532 | 0,848 | 0,959 | 21,1 |
| `disk` | UAC-AD | 0,649 | 0,996 | 0,994 | 0,988 | – | – | – | 19,4 |
| `disk` | UAM-AD | 0,846 | 0,997 | 0,994 | 0,991 | 0,981 | 0,988 | 0,992 | 21,1 |
| `loss` | UAC-AD | 0,570 | 0,892 | 0,774 | 0,724 | – | – | – | 19,4 |
| `loss` | UAM-AD | 0,781 | 0,948 | 0,873 | 0,793 | 0,603 | 0,949 | 0,976 | 21,1 |
| `mem` | UAC-AD | 0,636 | 0,915 | 0,813 | 0,754 | – | – | – | 19,4 |
| `mem` | UAM-AD | 0,820 | 0,976 | 0,922 | 0,826 | 0,748 | 0,986 | 0,993 | 21,0 |
| `socket` | UAC-AD | 0,278 | 0,622 | 0,314 | 0,414 | – | – | – | 19,5 |
| `socket` | UAM-AD | 0,438 | 0,759 | 0,523 | 0,460 | 0,756 | 0,933 | 0,939 | 21,2 |

**Seed 0: RE2-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,087 | 0,478 | 0,182 | 0,341 | – | – | – | 23,6 |
| `cpu` | UAM-AD | 0,427 | 0,677 | 0,486 | 0,428 | 0,775 | 0,924 | 0,969 | 31,8 |
| `delay` | UAC-AD | 0,072 | 0,624 | 0,237 | 0,424 | – | – | – | 23,9 |
| `delay` | UAM-AD | 0,667 | 0,879 | 0,811 | 0,767 | 0,654 | 0,841 | 0,879 | 29,3 |
| `disk` | UAC-AD | 0,707 | 0,930 | 0,879 | 0,868 | – | – | – | 22,1 |
| `disk` | UAM-AD | 0,749 | 0,918 | 0,863 | 0,856 | 0,986 | 0,990 | 0,991 | 29,4 |
| `loss` | UAC-AD | 0,060 | 0,475 | 0,196 | 0,349 | – | – | – | 25,2 |
| `loss` | UAM-AD | 0,521 | 0,737 | 0,606 | 0,529 | 0,396 | 0,635 | 0,711 | 32,3 |
| `mem` | UAC-AD | 0,845 | 0,964 | 0,787 | 0,878 | – | – | – | 22,7 |
| `mem` | UAM-AD | 0,715 | 0,943 | 0,817 | 0,724 | 0,552 | 0,677 | 0,717 | 29,2 |
| `socket` | UAC-AD | 0,151 | 0,498 | 0,194 | 0,349 | – | – | – | 23,2 |
| `socket` | UAM-AD | 0,456 | 0,703 | 0,529 | 0,458 | 0,760 | 0,939 | 0,964 | 32,0 |

###### Seed 1

**Seed 1: theo dataset**

| Dataset | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SN | UAC-AD | 0,611 | 0,945 | 0,704 | 0,782 | – | – | – | 4,3 |
| SN | UAM-AD | 0,724 | 0,954 | 0,705 | 0,857 | 0,199 | 0,409 | 0,415 | 4,6 |
| RE3-TT | UAC-AD | 0,545 | 0,754 | 0,444 | 0,651 | – | – | – | 37,8 |
| RE3-TT | UAM-AD | 0,513 | 0,703 | 0,347 | 0,635 | 0,248 | 0,322 | 0,428 | 53,2 |
| RE3-OB | UAC-AD | 0,642 | 0,856 | 0,692 | 0,760 | – | – | – | 26,8 |
| RE3-OB | UAM-AD | 0,577 | 0,841 | 0,628 | 0,763 | 0,839 | 0,914 | 0,973 | 40,1 |
| RE2-OB | UAC-AD | 0,560 | 0,865 | 0,728 | 0,677 | – | – | – | 115,5 |
| RE2-OB | UAM-AD | 0,733 | 0,930 | 0,840 | 0,769 | 0,789 | 0,962 | 0,983 | 142,6 |
| RE2-TT | UAC-AD | 0,322 | 0,667 | 0,421 | 0,529 | – | – | – | 142,6 |
| RE2-TT | UAM-AD | 0,607 | 0,809 | 0,680 | 0,630 | 0,701 | 0,841 | 0,867 | 194,5 |

**Seed 1: SN, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `Code_Stop_MediaService` | UAC-AD | 0,772 | 0,981 | 0,889 | 0,878 | – | – | – | 0,4 |
| `Code_Stop_MediaService` | UAM-AD | 0,737 | 0,962 | 0,814 | 0,897 | 0,744 | 0,923 | 0,923 | 0,5 |
| `Code_Stop_TextService` | UAC-AD | 0,681 | 0,972 | 0,890 | 0,907 | – | – | – | 0,4 |
| `Code_Stop_TextService` | UAM-AD | 0,696 | 0,972 | 0,891 | 0,929 | 0,692 | 1,000 | 1,000 | 0,3 |
| `Code_Stop_UserService` | UAC-AD | 0,729 | 0,973 | 0,917 | 0,894 | – | – | – | 0,4 |
| `Code_Stop_UserService` | UAM-AD | 0,745 | 0,982 | 0,945 | 0,894 | 0,053 | 0,605 | 0,658 | 0,4 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAC-AD | 0,769 | 0,977 | 0,766 | 0,800 | – | – | – | 0,4 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAM-AD | 0,783 | 0,965 | 0,754 | 0,833 | 0,300 | 0,400 | 0,400 | 0,4 |
| `DB_Redis_CacheLimit_SocialGraph` | UAC-AD | 0,750 | 0,995 | 0,968 | 0,900 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_SocialGraph` | UAM-AD | 0,857 | 0,983 | 0,777 | 0,900 | 0,000 | 0,000 | 0,000 | 0,4 |
| `DB_Redis_CacheLimit_UserTimeline` | UAC-AD | 0,476 | 0,903 | 0,465 | 0,667 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_UserTimeline` | UAM-AD | 0,556 | 0,937 | 0,584 | 0,750 | 0,000 | 0,000 | 0,000 | 0,4 |
| `Perf_CPU_Contention` | UAC-AD | 0,667 | 0,959 | 0,678 | 0,632 | – | – | – | 0,4 |
| `Perf_CPU_Contention` | UAM-AD | 0,769 | 0,955 | 0,661 | 0,870 | – | – | – | 0,4 |
| `Perf_Disk_IO_Stress` | UAC-AD | 0,522 | 0,936 | 0,556 | 0,444 | – | – | – | 0,4 |
| `Perf_Disk_IO_Stress` | UAM-AD | 0,640 | 0,912 | 0,440 | 0,727 | – | – | – | 0,4 |
| `Perf_Network_Loss` | UAC-AD | 0,667 | 0,979 | 0,915 | 0,737 | – | – | – | 0,3 |
| `Perf_Network_Loss` | UAM-AD | 0,857 | 0,982 | 0,784 | 0,952 | – | – | – | 0,4 |
| `Svc_Kill_Media` | UAC-AD | 0,500 | 0,913 | 0,501 | 0,889 | – | – | – | 0,3 |
| `Svc_Kill_Media` | UAM-AD | 0,750 | 0,971 | 0,804 | 0,889 | 0,000 | 0,750 | 0,750 | 0,4 |
| `Svc_Kill_SocialGraph` | UAC-AD | 0,222 | 0,802 | 0,261 | 0,750 | – | – | – | 0,3 |
| `Svc_Kill_SocialGraph` | UAM-AD | 0,545 | 0,889 | 0,387 | 0,750 | 0,000 | 0,000 | 0,000 | 0,4 |
| `Svc_Kill_UserTimeline` | UAC-AD | 0,571 | 0,952 | 0,646 | 0,889 | – | – | – | 0,3 |
| `Svc_Kill_UserTimeline` | UAM-AD | 0,750 | 0,942 | 0,622 | 0,889 | 0,000 | 0,000 | 0,000 | 0,4 |

**Seed 1: RE3-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,551 | 0,768 | 0,456 | 0,657 | – | – | – | 9,6 |
| `f1` | UAM-AD | 0,518 | 0,722 | 0,358 | 0,646 | 0,310 | 0,379 | 0,474 | 13,1 |
| `f2` | UAC-AD | 0,538 | 0,740 | 0,438 | 0,652 | – | – | – | 9,4 |
| `f2` | UAM-AD | 0,506 | 0,690 | 0,341 | 0,629 | 0,278 | 0,356 | 0,445 | 13,2 |
| `f3` | UAC-AD | 0,559 | 0,777 | 0,456 | 0,656 | – | – | – | 9,5 |
| `f3` | UAM-AD | 0,518 | 0,721 | 0,354 | 0,640 | 0,128 | 0,193 | 0,346 | 13,6 |
| `f4` | UAC-AD | 0,532 | 0,733 | 0,426 | 0,640 | – | – | – | 9,3 |
| `f4` | UAM-AD | 0,507 | 0,681 | 0,334 | 0,625 | 0,278 | 0,358 | 0,447 | 13,2 |

**Seed 1: RE3-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,847 | 0,961 | 0,903 | 0,916 | – | – | – | 5,4 |
| `f1` | UAM-AD | 0,778 | 0,945 | 0,834 | 0,917 | 0,378 | 0,629 | 0,889 | 8,0 |
| `f2` | UAC-AD | 0,894 | 0,966 | 0,915 | 0,945 | – | – | – | 5,2 |
| `f2` | UAM-AD | 0,865 | 0,954 | 0,848 | 0,945 | 0,975 | 0,984 | 0,995 | 7,8 |
| `f3` | UAC-AD | 0,474 | 0,820 | 0,507 | 0,661 | – | – | – | 5,4 |
| `f3` | UAM-AD | 0,311 | 0,780 | 0,399 | 0,667 | 0,955 | 0,991 | 0,994 | 8,1 |
| `f4` | UAC-AD | 0,150 | 0,575 | 0,248 | 0,363 | – | – | – | 5,7 |
| `f4` | UAM-AD | 0,158 | 0,580 | 0,244 | 0,371 | 0,917 | 0,984 | 0,996 | 8,3 |
| `f5` | UAC-AD | 0,847 | 0,958 | 0,888 | 0,912 | – | – | – | 5,2 |
| `f5` | UAM-AD | 0,774 | 0,944 | 0,818 | 0,916 | 0,973 | 0,983 | 0,988 | 8,0 |

**Seed 1: RE2-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,476 | 0,788 | 0,541 | 0,538 | – | – | – | 19,1 |
| `cpu` | UAM-AD | 0,650 | 0,914 | 0,774 | 0,675 | 0,879 | 0,978 | 0,989 | 23,6 |
| `delay` | UAC-AD | 0,598 | 0,895 | 0,745 | 0,637 | – | – | – | 19,2 |
| `delay` | UAM-AD | 0,797 | 0,969 | 0,917 | 0,837 | 0,597 | 0,930 | 0,991 | 23,7 |
| `disk` | UAC-AD | 0,703 | 0,997 | 0,995 | 0,988 | – | – | – | 19,1 |
| `disk` | UAM-AD | 0,888 | 0,998 | 0,996 | 0,991 | 0,984 | 0,990 | 0,992 | 23,5 |
| `loss` | UAC-AD | 0,601 | 0,909 | 0,798 | 0,692 | – | – | – | 19,3 |
| `loss` | UAM-AD | 0,799 | 0,956 | 0,887 | 0,805 | 0,696 | 0,917 | 0,951 | 23,5 |
| `mem` | UAC-AD | 0,701 | 0,958 | 0,891 | 0,788 | – | – | – | 19,3 |
| `mem` | UAM-AD | 0,836 | 0,979 | 0,931 | 0,842 | 0,657 | 0,977 | 0,988 | 23,4 |
| `socket` | UAC-AD | 0,282 | 0,645 | 0,401 | 0,421 | – | – | – | 19,5 |
| `socket` | UAM-AD | 0,430 | 0,764 | 0,533 | 0,465 | 0,922 | 0,980 | 0,985 | 24,9 |

**Seed 1: RE2-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,069 | 0,478 | 0,188 | 0,338 | – | – | – | 24,4 |
| `cpu` | UAM-AD | 0,420 | 0,683 | 0,468 | 0,424 | 0,850 | 0,977 | 0,985 | 32,5 |
| `delay` | UAC-AD | 0,114 | 0,636 | 0,249 | 0,421 | – | – | – | 23,4 |
| `delay` | UAM-AD | 0,698 | 0,842 | 0,784 | 0,768 | 0,725 | 0,950 | 0,983 | 33,8 |
| `disk` | UAC-AD | 0,751 | 0,937 | 0,885 | 0,859 | – | – | – | 23,0 |
| `disk` | UAM-AD | 0,767 | 0,926 | 0,862 | 0,809 | 0,985 | 0,988 | 0,990 | 32,4 |
| `loss` | UAC-AD | 0,069 | 0,470 | 0,196 | 0,349 | – | – | – | 24,7 |
| `loss` | UAM-AD | 0,528 | 0,738 | 0,596 | 0,528 | 0,456 | 0,701 | 0,756 | 32,5 |
| `mem` | UAC-AD | 0,838 | 0,965 | 0,807 | 0,875 | – | – | – | 22,5 |
| `mem` | UAM-AD | 0,787 | 0,963 | 0,861 | 0,808 | 0,510 | 0,594 | 0,626 | 31,0 |
| `socket` | UAC-AD | 0,089 | 0,515 | 0,200 | 0,333 | – | – | – | 24,5 |
| `socket` | UAM-AD | 0,444 | 0,704 | 0,509 | 0,446 | 0,678 | 0,835 | 0,861 | 32,3 |

###### Seed 2

**Seed 2: theo dataset**

| Dataset | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SN | UAC-AD | 0,646 | 0,949 | 0,712 | 0,785 | – | – | – | 4,2 |
| SN | UAM-AD | 0,726 | 0,959 | 0,715 | 0,831 | 0,202 | 0,384 | 0,393 | 5,1 |
| RE3-TT | UAC-AD | 0,545 | 0,753 | 0,440 | 0,649 | – | – | – | 39,2 |
| RE3-TT | UAM-AD | 0,510 | 0,697 | 0,341 | 0,627 | 0,247 | 0,317 | 0,395 | 54,2 |
| RE3-OB | UAC-AD | 0,641 | 0,857 | 0,693 | 0,759 | – | – | – | 27,8 |
| RE3-OB | UAM-AD | 0,576 | 0,839 | 0,626 | 0,763 | 0,844 | 0,917 | 0,970 | 40,3 |
| RE2-OB | UAC-AD | 0,691 | 0,904 | 0,798 | 0,745 | – | – | – | 118,4 |
| RE2-OB | UAM-AD | 0,660 | 0,909 | 0,813 | 0,752 | 0,782 | 0,951 | 0,973 | 141,1 |
| RE2-TT | UAC-AD | 0,370 | 0,676 | 0,447 | 0,534 | – | – | – | 316,0 |
| RE2-TT | UAM-AD | 0,602 | 0,811 | 0,688 | 0,626 | 0,712 | 0,877 | 0,922 | 200,9 |

**Seed 2: SN, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `Code_Stop_MediaService` | UAC-AD | 0,757 | 0,979 | 0,878 | 0,878 | – | – | – | 0,4 |
| `Code_Stop_MediaService` | UAM-AD | 0,760 | 0,965 | 0,821 | 0,929 | 0,744 | 0,821 | 0,821 | 0,5 |
| `Code_Stop_TextService` | UAC-AD | 0,681 | 0,974 | 0,896 | 0,907 | – | – | – | 0,3 |
| `Code_Stop_TextService` | UAM-AD | 0,696 | 0,977 | 0,902 | 0,929 | 0,282 | 0,692 | 0,769 | 0,5 |
| `Code_Stop_UserService` | UAC-AD | 0,729 | 0,975 | 0,918 | 0,894 | – | – | – | 0,4 |
| `Code_Stop_UserService` | UAM-AD | 0,745 | 0,984 | 0,946 | 0,894 | 0,789 | 0,895 | 0,895 | 0,4 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAC-AD | 0,769 | 0,977 | 0,766 | 0,833 | – | – | – | 0,4 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAM-AD | 0,833 | 0,968 | 0,785 | 0,833 | 0,000 | 0,300 | 0,300 | 0,4 |
| `DB_Redis_CacheLimit_SocialGraph` | UAC-AD | 0,750 | 0,995 | 0,968 | 0,900 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_SocialGraph` | UAM-AD | 0,857 | 0,985 | 0,832 | 0,857 | 0,000 | 0,000 | 0,000 | 0,4 |
| `DB_Redis_CacheLimit_UserTimeline` | UAC-AD | 0,545 | 0,909 | 0,476 | 0,667 | – | – | – | 0,4 |
| `DB_Redis_CacheLimit_UserTimeline` | UAM-AD | 0,556 | 0,937 | 0,577 | 0,706 | 0,000 | 0,000 | 0,000 | 0,4 |
| `Perf_CPU_Contention` | UAC-AD | 0,667 | 0,957 | 0,674 | 0,632 | – | – | – | 0,3 |
| `Perf_CPU_Contention` | UAM-AD | 0,769 | 0,962 | 0,676 | 0,833 | – | – | – | 0,4 |
| `Perf_Disk_IO_Stress` | UAC-AD | 0,583 | 0,933 | 0,550 | 0,444 | – | – | – | 0,4 |
| `Perf_Disk_IO_Stress` | UAM-AD | 0,667 | 0,916 | 0,444 | 0,727 | – | – | – | 0,4 |
| `Perf_Network_Loss` | UAC-AD | 0,667 | 0,975 | 0,909 | 0,737 | – | – | – | 0,3 |
| `Perf_Network_Loss` | UAM-AD | 0,870 | 0,985 | 0,798 | 0,909 | – | – | – | 0,4 |
| `Svc_Kill_Media` | UAC-AD | 0,667 | 0,933 | 0,585 | 0,889 | – | – | – | 0,4 |
| `Svc_Kill_Media` | UAM-AD | 0,750 | 0,962 | 0,679 | 0,889 | 0,000 | 0,750 | 0,750 | 0,4 |
| `Svc_Kill_SocialGraph` | UAC-AD | 0,364 | 0,840 | 0,306 | 0,750 | – | – | – | 0,3 |
| `Svc_Kill_SocialGraph` | UAM-AD | 0,545 | 0,914 | 0,476 | 0,667 | 0,000 | 0,000 | 0,000 | 0,4 |
| `Svc_Kill_UserTimeline` | UAC-AD | 0,571 | 0,942 | 0,622 | 0,889 | – | – | – | 0,4 |
| `Svc_Kill_UserTimeline` | UAM-AD | 0,667 | 0,952 | 0,646 | 0,800 | 0,000 | 0,000 | 0,000 | 0,4 |

**Seed 2: RE3-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,549 | 0,772 | 0,452 | 0,654 | – | – | – | 9,8 |
| `f1` | UAM-AD | 0,518 | 0,713 | 0,353 | 0,639 | 0,299 | 0,368 | 0,435 | 13,9 |
| `f2` | UAC-AD | 0,527 | 0,732 | 0,428 | 0,648 | – | – | – | 9,8 |
| `f2` | UAM-AD | 0,502 | 0,682 | 0,336 | 0,621 | 0,288 | 0,352 | 0,408 | 13,4 |
| `f3` | UAC-AD | 0,561 | 0,776 | 0,453 | 0,653 | – | – | – | 9,9 |
| `f3` | UAM-AD | 0,513 | 0,715 | 0,345 | 0,631 | 0,134 | 0,211 | 0,354 | 13,6 |
| `f4` | UAC-AD | 0,543 | 0,731 | 0,426 | 0,642 | – | – | – | 9,7 |
| `f4` | UAM-AD | 0,505 | 0,675 | 0,328 | 0,618 | 0,267 | 0,336 | 0,383 | 13,3 |

**Seed 2: RE3-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,848 | 0,961 | 0,904 | 0,915 | – | – | – | 5,6 |
| `f1` | UAM-AD | 0,778 | 0,944 | 0,833 | 0,912 | 0,380 | 0,644 | 0,889 | 8,1 |
| `f2` | UAC-AD | 0,893 | 0,966 | 0,914 | 0,946 | – | – | – | 5,4 |
| `f2` | UAM-AD | 0,861 | 0,952 | 0,839 | 0,945 | 0,976 | 0,987 | 0,991 | 7,9 |
| `f3` | UAC-AD | 0,461 | 0,826 | 0,506 | 0,661 | – | – | – | 5,6 |
| `f3` | UAM-AD | 0,303 | 0,781 | 0,397 | 0,671 | 0,967 | 0,986 | 0,990 | 8,1 |
| `f4` | UAC-AD | 0,150 | 0,576 | 0,248 | 0,358 | – | – | – | 5,7 |
| `f4` | UAM-AD | 0,163 | 0,577 | 0,244 | 0,375 | 0,926 | 0,984 | 0,991 | 8,3 |
| `f5` | UAC-AD | 0,853 | 0,959 | 0,890 | 0,914 | – | – | – | 5,6 |
| `f5` | UAM-AD | 0,775 | 0,944 | 0,818 | 0,915 | 0,969 | 0,982 | 0,988 | 7,9 |

**Seed 2: RE2-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,550 | 0,837 | 0,611 | 0,572 | – | – | – | 19,9 |
| `cpu` | UAM-AD | 0,595 | 0,879 | 0,730 | 0,629 | 0,879 | 0,979 | 0,989 | 23,7 |
| `delay` | UAC-AD | 0,760 | 0,957 | 0,887 | 0,781 | – | – | – | 19,8 |
| `delay` | UAM-AD | 0,740 | 0,956 | 0,904 | 0,835 | 0,594 | 0,933 | 0,988 | 23,5 |
| `disk` | UAC-AD | 0,876 | 0,996 | 0,994 | 0,988 | – | – | – | 19,7 |
| `disk` | UAM-AD | 0,787 | 0,998 | 0,996 | 0,991 | 0,980 | 0,988 | 0,991 | 23,4 |
| `loss` | UAC-AD | 0,715 | 0,940 | 0,868 | 0,770 | – | – | – | 19,6 |
| `loss` | UAM-AD | 0,720 | 0,942 | 0,873 | 0,814 | 0,655 | 0,910 | 0,955 | 23,6 |
| `mem` | UAC-AD | 0,869 | 0,980 | 0,942 | 0,870 | – | – | – | 19,6 |
| `mem` | UAM-AD | 0,737 | 0,966 | 0,905 | 0,820 | 0,719 | 0,969 | 0,983 | 23,3 |
| `socket` | UAC-AD | 0,376 | 0,716 | 0,489 | 0,486 | – | – | – | 19,8 |
| `socket` | UAM-AD | 0,379 | 0,714 | 0,473 | 0,422 | 0,862 | 0,929 | 0,930 | 23,6 |

**Seed 2: RE2-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,079 | 0,491 | 0,188 | 0,335 | – | – | – | 23,6 |
| `cpu` | UAM-AD | 0,426 | 0,705 | 0,502 | 0,438 | 0,859 | 0,986 | 0,992 | 34,4 |
| `delay` | UAC-AD | 0,334 | 0,668 | 0,388 | 0,416 | – | – | – | 22,3 |
| `delay` | UAM-AD | 0,698 | 0,844 | 0,785 | 0,767 | 0,745 | 0,912 | 0,981 | 33,3 |
| `disk` | UAC-AD | 0,752 | 0,944 | 0,897 | 0,871 | – | – | – | 22,3 |
| `disk` | UAM-AD | 0,768 | 0,923 | 0,855 | 0,799 | 0,986 | 0,989 | 0,990 | 33,0 |
| `loss` | UAC-AD | 0,092 | 0,476 | 0,191 | 0,342 | – | – | – | 23,4 |
| `loss` | UAM-AD | 0,513 | 0,728 | 0,589 | 0,518 | 0,413 | 0,636 | 0,753 | 33,5 |
| `mem` | UAC-AD | 0,853 | 0,968 | 0,822 | 0,887 | – | – | – | 22,1 |
| `mem` | UAM-AD | 0,754 | 0,961 | 0,868 | 0,777 | 0,756 | 0,848 | 0,869 | 32,9 |
| `socket` | UAC-AD | 0,110 | 0,512 | 0,197 | 0,355 | – | – | – | 202,3 |
| `socket` | UAM-AD | 0,454 | 0,704 | 0,529 | 0,458 | 0,511 | 0,893 | 0,946 | 33,8 |

###### Seed 3

**Seed 3: theo dataset**

| Dataset | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SN | UAC-AD | 0,306 | 0,564 | 0,386 | 0,709 | – | – | – | 1,8 |
| SN | UAM-AD | 0,725 | 0,956 | 0,717 | 0,855 | 0,163 | 0,260 | 0,334 | 4,0 |
| RE3-TT | UAC-AD | 0,543 | 0,744 | 0,432 | 0,645 | – | – | – | 40,5 |
| RE3-TT | UAM-AD | 0,516 | 0,705 | 0,349 | 0,635 | 0,253 | 0,360 | 0,453 | 49,3 |
| RE3-OB | UAC-AD | 0,656 | 0,860 | 0,702 | 0,765 | – | – | – | 34,9 |
| RE3-OB | UAM-AD | 0,566 | 0,834 | 0,614 | 0,748 | 0,826 | 0,909 | 0,968 | 37,3 |
| RE2-OB | UAC-AD | 0,553 | 0,871 | 0,739 | 0,688 | – | – | – | 122,7 |
| RE2-OB | UAM-AD | 0,755 | 0,934 | 0,854 | 0,799 | 0,741 | 0,946 | 0,978 | 130,1 |
| RE2-TT | UAC-AD | 0,383 | 0,679 | 0,465 | 0,536 | – | – | – | 146,1 |
| RE2-TT | UAM-AD | 0,592 | 0,804 | 0,674 | 0,614 | 0,748 | 0,895 | 0,934 | 181,4 |

**Seed 3: SN, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `Code_Stop_MediaService` | UAC-AD | 0,557 | 0,770 | 0,631 | 0,918 | – | – | – | 0,2 |
| `Code_Stop_MediaService` | UAM-AD | 0,755 | 0,967 | 0,833 | 0,916 | 0,641 | 0,795 | 0,821 | 0,4 |
| `Code_Stop_TextService` | UAC-AD | 0,651 | 0,763 | 0,727 | 0,907 | – | – | – | 0,1 |
| `Code_Stop_TextService` | UAM-AD | 0,696 | 0,977 | 0,905 | 0,929 | 0,359 | 0,436 | 0,538 | 0,3 |
| `Code_Stop_UserService` | UAC-AD | 0,705 | 0,870 | 0,815 | 0,916 | – | – | – | 0,1 |
| `Code_Stop_UserService` | UAM-AD | 0,745 | 0,983 | 0,947 | 0,894 | 0,263 | 0,711 | 0,895 | 0,3 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAC-AD | 0,353 | 0,740 | 0,447 | 0,444 | – | – | – | 0,1 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAM-AD | 0,783 | 0,963 | 0,739 | 0,833 | 0,200 | 0,400 | 0,500 | 0,3 |
| `DB_Redis_CacheLimit_SocialGraph` | UAC-AD | 0,600 | 0,740 | 0,707 | 0,857 | – | – | – | 0,1 |
| `DB_Redis_CacheLimit_SocialGraph` | UAM-AD | 0,857 | 0,985 | 0,859 | 0,900 | 0,000 | 0,000 | 0,000 | 0,3 |
| `DB_Redis_CacheLimit_UserTimeline` | UAC-AD | 0,000 | 0,336 | 0,100 | 0,533 | – | – | – | 0,1 |
| `DB_Redis_CacheLimit_UserTimeline` | UAM-AD | 0,556 | 0,933 | 0,573 | 0,750 | 0,000 | 0,000 | 0,000 | 0,3 |
| `Perf_CPU_Contention` | UAC-AD | 0,235 | 0,687 | 0,346 | 0,429 | – | – | – | 0,2 |
| `Perf_CPU_Contention` | UAM-AD | 0,769 | 0,968 | 0,751 | 0,833 | – | – | – | 0,3 |
| `Perf_Disk_IO_Stress` | UAC-AD | 0,353 | 0,641 | 0,263 | 0,400 | – | – | – | 0,1 |
| `Perf_Disk_IO_Stress` | UAM-AD | 0,640 | 0,904 | 0,466 | 0,727 | – | – | – | 0,3 |
| `Perf_Network_Loss` | UAC-AD | 0,222 | 0,572 | 0,225 | 0,571 | – | – | – | 0,1 |
| `Perf_Network_Loss` | UAM-AD | 0,857 | 0,982 | 0,784 | 0,952 | – | – | – | 0,3 |
| `Svc_Kill_Media` | UAC-AD | 0,000 | 0,211 | 0,133 | 0,889 | – | – | – | 0,1 |
| `Svc_Kill_Media` | UAM-AD | 0,750 | 0,962 | 0,679 | 0,889 | 0,000 | 0,000 | 0,000 | 0,3 |
| `Svc_Kill_SocialGraph` | UAC-AD | 0,000 | 0,204 | 0,100 | 0,750 | – | – | – | 0,2 |
| `Svc_Kill_SocialGraph` | UAM-AD | 0,545 | 0,901 | 0,443 | 0,750 | 0,000 | 0,000 | 0,000 | 0,3 |
| `Svc_Kill_UserTimeline` | UAC-AD | 0,000 | 0,231 | 0,133 | 0,889 | – | – | – | 0,1 |
| `Svc_Kill_UserTimeline` | UAM-AD | 0,750 | 0,942 | 0,622 | 0,889 | 0,000 | 0,000 | 0,250 | 0,3 |

**Seed 3: RE3-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,546 | 0,754 | 0,439 | 0,650 | – | – | – | 10,0 |
| `f1` | UAM-AD | 0,520 | 0,717 | 0,358 | 0,642 | 0,313 | 0,411 | 0,501 | 12,4 |
| `f2` | UAC-AD | 0,541 | 0,731 | 0,425 | 0,641 | – | – | – | 10,1 |
| `f2` | UAM-AD | 0,512 | 0,690 | 0,345 | 0,633 | 0,279 | 0,385 | 0,461 | 12,3 |
| `f3` | UAC-AD | 0,553 | 0,773 | 0,450 | 0,654 | – | – | – | 10,3 |
| `f3` | UAM-AD | 0,520 | 0,725 | 0,356 | 0,642 | 0,130 | 0,261 | 0,398 | 12,4 |
| `f4` | UAC-AD | 0,534 | 0,720 | 0,414 | 0,634 | – | – | – | 10,2 |
| `f4` | UAM-AD | 0,512 | 0,688 | 0,338 | 0,623 | 0,289 | 0,383 | 0,452 | 12,2 |

**Seed 3: RE3-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,846 | 0,961 | 0,905 | 0,914 | – | – | – | 7,1 |
| `f1` | UAM-AD | 0,757 | 0,937 | 0,815 | 0,898 | 0,382 | 0,618 | 0,884 | 7,4 |
| `f2` | UAC-AD | 0,878 | 0,968 | 0,918 | 0,945 | – | – | – | 6,8 |
| `f2` | UAM-AD | 0,847 | 0,950 | 0,828 | 0,939 | 0,975 | 0,985 | 0,990 | 7,3 |
| `f3` | UAC-AD | 0,536 | 0,832 | 0,535 | 0,673 | – | – | – | 7,0 |
| `f3` | UAM-AD | 0,300 | 0,769 | 0,384 | 0,653 | 0,959 | 0,990 | 0,993 | 7,5 |
| `f4` | UAC-AD | 0,167 | 0,581 | 0,254 | 0,376 | – | – | – | 7,1 |
| `f4` | UAM-AD | 0,166 | 0,574 | 0,241 | 0,355 | 0,902 | 0,970 | 0,986 | 7,6 |
| `f5` | UAC-AD | 0,855 | 0,959 | 0,895 | 0,915 | – | – | – | 6,9 |
| `f5` | UAM-AD | 0,760 | 0,939 | 0,802 | 0,897 | 0,911 | 0,982 | 0,986 | 7,4 |

**Seed 3: RE2-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,457 | 0,807 | 0,574 | 0,548 | – | – | – | 20,4 |
| `cpu` | UAM-AD | 0,723 | 0,927 | 0,816 | 0,737 | 0,797 | 0,944 | 0,987 | 21,7 |
| `delay` | UAC-AD | 0,598 | 0,902 | 0,759 | 0,652 | – | – | – | 20,3 |
| `delay` | UAM-AD | 0,761 | 0,967 | 0,919 | 0,863 | 0,528 | 0,925 | 0,981 | 21,6 |
| `disk` | UAC-AD | 0,696 | 0,994 | 0,990 | 0,988 | – | – | – | 20,3 |
| `disk` | UAM-AD | 0,901 | 0,997 | 0,995 | 0,991 | 0,987 | 0,991 | 0,995 | 21,6 |
| `loss` | UAC-AD | 0,603 | 0,912 | 0,810 | 0,718 | – | – | – | 20,4 |
| `loss` | UAM-AD | 0,828 | 0,961 | 0,904 | 0,843 | 0,610 | 0,898 | 0,933 | 21,7 |
| `mem` | UAC-AD | 0,671 | 0,955 | 0,884 | 0,791 | – | – | – | 20,3 |
| `mem` | UAM-AD | 0,866 | 0,984 | 0,947 | 0,888 | 0,586 | 0,926 | 0,983 | 21,7 |
| `socket` | UAC-AD | 0,293 | 0,657 | 0,418 | 0,431 | – | – | – | 20,8 |
| `socket` | UAM-AD | 0,448 | 0,769 | 0,545 | 0,472 | 0,939 | 0,989 | 0,991 | 21,8 |

**Seed 3: RE2-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,088 | 0,488 | 0,186 | 0,341 | – | – | – | 25,5 |
| `cpu` | UAM-AD | 0,402 | 0,676 | 0,463 | 0,412 | 0,822 | 0,964 | 0,975 | 30,3 |
| `delay` | UAC-AD | 0,327 | 0,665 | 0,358 | 0,403 | – | – | – | 24,1 |
| `delay` | UAM-AD | 0,698 | 0,845 | 0,785 | 0,767 | 0,778 | 0,949 | 0,989 | 31,1 |
| `disk` | UAC-AD | 0,809 | 0,987 | 0,973 | 0,925 | – | – | – | 23,9 |
| `disk` | UAM-AD | 0,769 | 0,924 | 0,855 | 0,791 | 0,985 | 0,989 | 0,991 | 30,6 |
| `loss` | UAC-AD | 0,120 | 0,479 | 0,192 | 0,342 | – | – | – | 22,6 |
| `loss` | UAM-AD | 0,498 | 0,722 | 0,579 | 0,506 | 0,452 | 0,726 | 0,848 | 30,1 |
| `mem` | UAC-AD | 0,848 | 0,969 | 0,890 | 0,867 | – | – | – | 24,2 |
| `mem` | UAM-AD | 0,746 | 0,954 | 0,840 | 0,762 | 0,644 | 0,812 | 0,840 | 28,8 |
| `socket` | UAC-AD | 0,107 | 0,489 | 0,188 | 0,338 | – | – | – | 25,8 |
| `socket` | UAM-AD | 0,443 | 0,706 | 0,520 | 0,445 | 0,805 | 0,928 | 0,960 | 30,5 |

###### Seed 4

**Seed 4: theo dataset**

| Dataset | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SN | UAC-AD | 0,751 | 0,967 | 0,784 | 0,809 | – | – | – | 3,7 |
| SN | UAM-AD | 0,730 | 0,958 | 0,738 | 0,854 | 0,166 | 0,309 | 0,443 | 3,9 |
| RE3-TT | UAC-AD | 0,536 | 0,739 | 0,429 | 0,644 | – | – | – | 35,8 |
| RE3-TT | UAM-AD | 0,513 | 0,702 | 0,347 | 0,638 | 0,262 | 0,343 | 0,416 | 48,9 |
| RE3-OB | UAC-AD | 0,656 | 0,858 | 0,700 | 0,762 | – | – | – | 30,8 |
| RE3-OB | UAM-AD | 0,525 | 0,828 | 0,596 | 0,753 | 0,833 | 0,912 | 0,964 | 37,0 |
| RE2-OB | UAC-AD | 0,733 | 0,916 | 0,824 | 0,774 | – | – | – | 107,7 |
| RE2-OB | UAM-AD | 0,543 | 0,890 | 0,782 | 0,738 | 0,734 | 0,938 | 0,972 | 129,4 |
| RE2-TT | UAC-AD | 0,332 | 0,666 | 0,411 | 0,528 | – | – | – | 114,9 |
| RE2-TT | UAM-AD | 0,593 | 0,808 | 0,672 | 0,625 | 0,758 | 0,907 | 0,945 | 185,1 |

**Seed 4: SN, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `Code_Stop_MediaService` | UAC-AD | 0,796 | 0,986 | 0,920 | 0,902 | – | – | – | 0,4 |
| `Code_Stop_MediaService` | UAM-AD | 0,792 | 0,970 | 0,843 | 0,916 | 0,692 | 0,846 | 0,897 | 0,3 |
| `Code_Stop_TextService` | UAC-AD | 0,812 | 0,983 | 0,923 | 0,929 | – | – | – | 0,3 |
| `Code_Stop_TextService` | UAM-AD | 0,703 | 0,979 | 0,914 | 0,929 | 0,462 | 0,769 | 0,821 | 0,3 |
| `Code_Stop_UserService` | UAC-AD | 0,745 | 0,982 | 0,942 | 0,894 | – | – | – | 0,3 |
| `Code_Stop_UserService` | UAM-AD | 0,761 | 0,985 | 0,954 | 0,894 | 0,342 | 0,868 | 0,868 | 0,3 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAC-AD | 0,870 | 0,982 | 0,784 | 0,833 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAM-AD | 0,783 | 0,965 | 0,748 | 0,833 | 0,000 | 0,300 | 0,400 | 0,3 |
| `DB_Redis_CacheLimit_SocialGraph` | UAC-AD | 0,857 | 1,000 | 1,000 | 0,900 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_SocialGraph` | UAM-AD | 0,857 | 0,985 | 0,876 | 0,900 | 0,000 | 0,000 | 0,000 | 0,3 |
| `DB_Redis_CacheLimit_UserTimeline` | UAC-AD | 0,571 | 0,923 | 0,540 | 0,667 | – | – | – | 0,3 |
| `DB_Redis_CacheLimit_UserTimeline` | UAM-AD | 0,556 | 0,942 | 0,722 | 0,778 | 0,000 | 0,000 | 0,000 | 0,3 |
| `Perf_CPU_Contention` | UAC-AD | 0,783 | 0,968 | 0,696 | 0,667 | – | – | – | 0,3 |
| `Perf_CPU_Contention` | UAM-AD | 0,800 | 0,965 | 0,741 | 0,833 | – | – | – | 0,3 |
| `Perf_Disk_IO_Stress` | UAC-AD | 0,750 | 0,950 | 0,578 | 0,640 | – | – | – | 0,3 |
| `Perf_Disk_IO_Stress` | UAM-AD | 0,667 | 0,912 | 0,508 | 0,727 | – | – | – | 0,3 |
| `Perf_Network_Loss` | UAC-AD | 0,833 | 0,988 | 0,941 | 0,750 | – | – | – | 0,3 |
| `Perf_Network_Loss` | UAM-AD | 0,900 | 0,982 | 0,784 | 0,909 | – | – | – | 0,3 |
| `Svc_Kill_Media` | UAC-AD | 0,800 | 0,962 | 0,771 | 0,889 | – | – | – | 0,3 |
| `Svc_Kill_Media` | UAM-AD | 0,750 | 0,962 | 0,679 | 0,889 | 0,000 | 0,000 | 1,000 | 0,3 |
| `Svc_Kill_SocialGraph` | UAC-AD | 0,444 | 0,901 | 0,421 | 0,750 | – | – | – | 0,3 |
| `Svc_Kill_SocialGraph` | UAM-AD | 0,444 | 0,901 | 0,443 | 0,750 | 0,000 | 0,000 | 0,000 | 0,3 |
| `Svc_Kill_UserTimeline` | UAC-AD | 0,750 | 0,981 | 0,887 | 0,889 | – | – | – | 0,3 |
| `Svc_Kill_UserTimeline` | UAM-AD | 0,750 | 0,952 | 0,646 | 0,889 | 0,000 | 0,000 | 0,000 | 0,3 |

**Seed 4: RE3-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,539 | 0,762 | 0,448 | 0,657 | – | – | – | 8,9 |
| `f1` | UAM-AD | 0,518 | 0,718 | 0,361 | 0,649 | 0,318 | 0,389 | 0,437 | 12,2 |
| `f2` | UAC-AD | 0,538 | 0,726 | 0,419 | 0,636 | – | – | – | 8,9 |
| `f2` | UAM-AD | 0,509 | 0,688 | 0,340 | 0,633 | 0,285 | 0,384 | 0,440 | 12,2 |
| `f3` | UAC-AD | 0,543 | 0,750 | 0,429 | 0,641 | – | – | – | 9,0 |
| `f3` | UAM-AD | 0,517 | 0,723 | 0,352 | 0,643 | 0,138 | 0,226 | 0,369 | 12,2 |
| `f4` | UAC-AD | 0,524 | 0,720 | 0,420 | 0,640 | – | – | – | 9,0 |
| `f4` | UAM-AD | 0,506 | 0,679 | 0,333 | 0,627 | 0,305 | 0,375 | 0,418 | 12,2 |

**Seed 4: RE3-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAC-AD | 0,854 | 0,964 | 0,910 | 0,919 | – | – | – | 6,2 |
| `f1` | UAM-AD | 0,720 | 0,931 | 0,796 | 0,910 | 0,402 | 0,631 | 0,860 | 7,4 |
| `f2` | UAC-AD | 0,882 | 0,968 | 0,919 | 0,946 | – | – | – | 6,0 |
| `f2` | UAM-AD | 0,803 | 0,942 | 0,800 | 0,943 | 0,916 | 0,980 | 0,988 | 7,3 |
| `f3` | UAC-AD | 0,526 | 0,827 | 0,526 | 0,661 | – | – | – | 6,1 |
| `f3` | UAM-AD | 0,221 | 0,762 | 0,360 | 0,653 | 0,963 | 0,986 | 0,993 | 7,4 |
| `f4` | UAC-AD | 0,163 | 0,573 | 0,249 | 0,364 | – | – | – | 6,4 |
| `f4` | UAM-AD | 0,151 | 0,573 | 0,238 | 0,346 | 0,920 | 0,984 | 0,993 | 7,5 |
| `f5` | UAC-AD | 0,857 | 0,961 | 0,896 | 0,918 | – | – | – | 6,0 |
| `f5` | UAM-AD | 0,731 | 0,933 | 0,788 | 0,911 | 0,965 | 0,981 | 0,988 | 7,4 |

**Seed 4: RE2-OB, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,620 | 0,874 | 0,686 | 0,624 | – | – | – | 18,0 |
| `cpu` | UAM-AD | 0,518 | 0,852 | 0,681 | 0,584 | 0,861 | 0,974 | 0,989 | 21,4 |
| `delay` | UAC-AD | 0,816 | 0,971 | 0,925 | 0,865 | – | – | – | 17,9 |
| `delay` | UAM-AD | 0,553 | 0,929 | 0,858 | 0,827 | 0,462 | 0,955 | 0,991 | 21,5 |
| `disk` | UAC-AD | 0,854 | 0,997 | 0,995 | 0,988 | – | – | – | 17,9 |
| `disk` | UAM-AD | 0,668 | 0,996 | 0,994 | 0,991 | 0,981 | 0,986 | 0,992 | 21,6 |
| `loss` | UAC-AD | 0,763 | 0,952 | 0,900 | 0,814 | – | – | – | 17,9 |
| `loss` | UAM-AD | 0,592 | 0,918 | 0,827 | 0,768 | 0,613 | 0,892 | 0,940 | 21,6 |
| `mem` | UAC-AD | 0,857 | 0,976 | 0,925 | 0,860 | – | – | – | 17,9 |
| `mem` | UAM-AD | 0,600 | 0,948 | 0,881 | 0,831 | 0,691 | 0,929 | 0,954 | 21,6 |
| `socket` | UAC-AD | 0,487 | 0,724 | 0,513 | 0,491 | – | – | – | 18,0 |
| `socket` | UAM-AD | 0,329 | 0,697 | 0,450 | 0,429 | 0,796 | 0,894 | 0,967 | 21,7 |

**Seed 4: RE2-TT, từng scenario**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAC-AD | 0,079 | 0,486 | 0,187 | 0,337 | – | – | – | 18,6 |
| `cpu` | UAM-AD | 0,368 | 0,700 | 0,412 | 0,416 | 0,875 | 0,988 | 0,992 | 31,2 |
| `delay` | UAC-AD | 0,138 | 0,643 | 0,255 | 0,420 | – | – | – | 21,6 |
| `delay` | UAM-AD | 0,698 | 0,844 | 0,786 | 0,767 | 0,670 | 0,883 | 0,986 | 30,8 |
| `disk` | UAC-AD | 0,763 | 0,927 | 0,878 | 0,874 | – | – | – | 17,4 |
| `disk` | UAM-AD | 0,770 | 0,927 | 0,859 | 0,811 | 0,987 | 0,990 | 0,991 | 30,5 |
| `loss` | UAC-AD | 0,077 | 0,480 | 0,190 | 0,349 | – | – | – | 18,4 |
| `loss` | UAM-AD | 0,501 | 0,712 | 0,576 | 0,510 | 0,493 | 0,796 | 0,873 | 30,5 |
| `mem` | UAC-AD | 0,839 | 0,956 | 0,756 | 0,845 | – | – | – | 16,7 |
| `mem` | UAM-AD | 0,775 | 0,964 | 0,877 | 0,795 | 0,751 | 0,841 | 0,864 | 30,5 |
| `socket` | UAC-AD | 0,095 | 0,503 | 0,197 | 0,345 | – | – | – | 22,2 |
| `socket` | UAM-AD | 0,448 | 0,702 | 0,522 | 0,449 | 0,772 | 0,940 | 0,964 | 31,7 |

### Phụ lục B. Các đối thủ: bảng sinh tự động (5 seed theo dataset và theo scenario, chi tiết từng seed)

Khối dưới do `summary_tools/item9_tables.py` ghi giữa hai dấu ITEM9 RESULTS BEGIN và END; không sửa tay trong khối.

<!-- ITEM9 RESULTS BEGIN -->

**Kết quả 5 seed (0–4) của UAM-AD, ART, Eadro, OrEdge; AnoFusion chỉ seed 0 (sinh tự động bởi `summary_tools/item9_tables.py`)**

Cách đọc: mỗi ô "a±b" là trung bình±độ lệch chuẩn tổng thể (`np.std`, như các bảng bước 8) qua các seed của trung bình theo scenario; ô không có ± là một seed. Cột "seed" = số seed có kết quả. UAM-AD lấy từ bước 8 (`norm_sum`), cùng 5 seed 0–4; **riêng RE2-OB và RE2-TT là số chấm lại theo thứ tự thời gian (việc 10), F1 oracle của hai dataset này là bản không point-adjust**; HR@k của UAM-AD ở cả 5 dataset là xếp hạng trace + KPI chấm lại ở việc 10 (đủ 5 seed, kể cả HR@5). **AnoFusion chỉ có seed 0** (chi phí 298,5 phút mỗi seed không tính RE3-TT, riêng RE2-TT 220 phút), **không sao chép seed 0 sang seed khác, không có độ lệch chuẩn**; Δ so với AnoFusion = trung bình 5 seed của UAM-AD trừ seed 0 của AnoFusion. Δ = UAM-AD trừ mô hình ở dòng đó, trong ngoặc là % giá trị trung bình của mô hình đó (dương = UAM-AD cao hơn), chỉ tính trên scenario cả hai cùng có; dấu * = mô hình dừng sớm (AnoFusion RE3-TT chỉ có `f1`), không vào dòng Trung bình. Dòng Trung bình: trung bình các dataset của từng seed rồi lấy trung bình±độ lệch chuẩn qua seed; % = chênh lệch trung bình chia giá trị trung bình của mô hình (không phải trung bình của các %). F1 oracle của UAM-AD chỉ so được với các mô hình khác ở RE2-OB, RE2-TT (cùng không point-adjust); ở các dataset còn lại UAM-AD có point-adjust. HR@k xem ghi chú ở bảng từng seed bên dưới.

*Theo dataset (5 seed):*

| Dataset | Mô hình | seed | F1@val95 | Δ F1 | AUROC | Δ AUROC | AUPRC | Δ AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | UAM-AD | 5 | 0,681±0,076 | – | 0,917±0,016 | – | 0,823±0,025 | – | 0,764±0,020 | 0,759±0,022 | 0,949±0,008 | 0,976±0,004 | – | 134,0 |
| RE2-OB | ART | 5 | 0,880±0,001 | −0,199 (−22,6%) | 0,981±0,002 | −0,064 (−6,5%) | 0,957±0,004 | −0,134 (−14,0%) | 0,923±0,004 | 0,753±0,004 | 0,844±0,004 | 0,933±0,000 | – | 6,3 |
| RE2-OB | AnoFusion | 1 | 0,587 | +0,094 (+16,0%) | 0,819 | +0,098 (+12,0%) | 0,693 | +0,130 (+18,8%) | 0,654 | – | – | – | – | 18,8 |
| RE2-OB | Eadro (có giám sát) | 5 | 0,711±0,016 | −0,030 (−4,3%) | 0,867±0,005 | +0,051 (+5,8%) | 0,791±0,011 | +0,033 (+4,2%) | 0,749±0,016 | 0,667±0,016 | 0,880±0,003 | 0,996±0,002 | 0,696±0,007 | 12,6 |
| RE2-OB | OrEdge (bán giám sát) | 5 | 0,867±0,012 | −0,186 (−21,4%) | 0,973±0,003 | −0,056 (−5,7%) | 0,959±0,008 | −0,135 (−14,1%) | 0,945±0,008 | 0,927±0,003 | 0,946±0,003 | 0,952±0,004 | 0,895±0,010 | 30,2 |
| RE3-OB | UAM-AD | 5 | 0,560±0,019 | – | 0,835±0,005 | – | 0,615±0,012 | – | 0,755±0,007 | 0,824±0,024 | 0,902±0,022 | 0,959±0,019 | – | 38,2 |
| RE3-OB | ART | 5 | 0,523±0,058 | +0,037 (+7,1%) | 0,893±0,005 | −0,059 (−6,6%) | 0,746±0,015 | −0,131 (−17,5%) | 0,756±0,017 | 0,727±0,006 | 0,788±0,007 | 0,827±0,006 | – | 2,2 |
| RE3-OB | AnoFusion | 1 | 0,366 | +0,194 (+53,1%) | 0,630 | +0,204 (+32,4%) | 0,447 | +0,169 (+37,8%) | 0,460 | – | – | – | – | 48,1 |
| RE3-OB | Eadro (có giám sát) | 5 | 0,645±0,026 | −0,085 (−13,2%) | 0,817±0,022 | +0,018 (+2,2%) | 0,727±0,024 | −0,111 (−15,3%) | 0,679±0,024 | 0,853±0,022 | 1,000±0,000 | 1,000±0,000 | 0,648±0,015 | 4,5 |
| RE3-OB | OrEdge (bán giám sát) | 5 | 0,762±0,009 | −0,202 (−26,5%) | 0,892±0,003 | −0,058 (−6,5%) | 0,832±0,005 | −0,217 (−26,1%) | 0,815±0,009 | 0,941±0,009 | 0,994±0,002 | 0,999±0,001 | 0,763±0,010 | 10,9 |
| RE2-TT | UAM-AD | 5 | 0,597±0,007 | – | 0,809±0,002 | – | 0,680±0,006 | – | 0,624±0,006 | 0,721±0,027 | 0,871±0,029 | 0,908±0,032 | – | 189,2 |
| RE2-TT | ART | 5 | 0,583±0,002 | +0,014 (+2,4%) | 0,833±0,002 | −0,025 (−3,0%) | 0,688±0,003 | −0,008 (−1,1%) | 0,677±0,001 | 0,447±0,003 | 0,511±0,004 | 0,590±0,007 | – | 7,3 |
| RE2-TT | AnoFusion | 1 | 0,584 | +0,013 (+2,2%) | 0,773 | +0,035 (+4,5%) | 0,672 | +0,008 (+1,2%) | 0,668 | – | – | – | – | 220,3 |
| RE2-TT | Eadro (có giám sát) | 5 | 0,641±0,020 | −0,044 (−6,8%) | 0,838±0,014 | −0,029 (−3,5%) | 0,728±0,021 | −0,048 (−6,6%) | 0,691±0,020 | 0,539±0,015 | 0,780±0,007 | 0,993±0,003 | 0,621±0,016 | 34,0 |
| RE2-TT | OrEdge (bán giám sát) | 5 | 0,831±0,002 | −0,234 (−28,2%) | 0,967±0,005 | −0,159 (−16,4%) | 0,949±0,003 | −0,269 (−28,4%) | 0,926±0,002 | 0,954±0,007 | 0,982±0,004 | 0,994±0,001 | 0,905±0,004 | 31,9 |
| RE3-TT | UAM-AD | 5 | 0,513±0,002 | – | 0,702±0,003 | – | 0,347±0,003 | – | 0,633±0,004 | 0,251±0,006 | 0,332±0,017 | 0,417±0,022 | – | 50,8 |
| RE3-TT | ART | 5 | 0,322±0,012 | +0,191 (+59,3%) | 0,684±0,013 | +0,019 (+2,8%) | 0,448±0,014 | −0,102 (−22,7%) | 0,473±0,007 | 0,344±0,053 | 0,413±0,032 | 0,478±0,023 | – | 2,4 |
| RE3-TT | AnoFusion (CHỈ f1, dừng sớm) | 1 | 0,116 | +0,404 (+348,2%)* | 0,502 | +0,217 (+43,2%)* | 0,206 | +0,152 (+73,9%)* | 0,351 | – | – | – | – | – |
| RE3-TT | Eadro (có giám sát) | 5 | 0,365±0,028 | +0,148 (+40,4%) | 0,606±0,020 | +0,097 (+16,0%) | 0,443±0,017 | −0,096 (−21,7%) | 0,443±0,009 | 0,621±0,005 | 1,000±0,000 | 1,000±0,000 | 0,225±0,004 | 11,6 |
| RE3-TT | OrEdge (bán giám sát) | 5 | 0,631±0,006 | −0,117 (−18,6%) | 0,764±0,004 | −0,062 (−8,1%) | 0,690±0,003 | −0,344 (−49,8%) | 0,666±0,003 | 0,905±0,011 | 0,995±0,001 | 0,997±0,000 | 0,619±0,009 | 33,5 |
| SN | UAM-AD | 5 | 0,728±0,004 | – | 0,956±0,002 | – | 0,716±0,012 | – | 0,849±0,009 | 0,185±0,017 | 0,347±0,055 | 0,397±0,036 | – | 4,3 |
| SN | ART | 5 | 0,553±0,034 | +0,174 (+31,5%) | 0,890±0,019 | +0,067 (+7,5%) | 0,590±0,038 | +0,126 (+21,5%) | 0,664±0,028 | 0,058±0,018 | 0,161±0,025 | 0,345±0,049 | – | 0,4 |
| SN | AnoFusion | 1 | 0,408 | +0,320 (+78,5%) | 0,731 | +0,225 (+30,8%) | 0,455 | +0,261 (+57,4%) | 0,575 | – | – | – | – | 11,3 |
| SN | Eadro (có giám sát) | 5 | 0,820±0,016 | −0,092 (−11,2%) | 0,980±0,003 | −0,024 (−2,4%) | 0,857±0,013 | −0,141 (−16,4%) | 0,883±0,010 | 0,074±0,052 | 0,266±0,027 | 0,498±0,053 | 0,809±0,021 | 1,1 |
| SN | OrEdge (bán giám sát) | 5 | 0,416±0,108 | +0,312 (+74,9%) | 0,844±0,039 | +0,112 (+13,3%) | 0,546±0,061 | +0,170 (+31,1%) | 0,654±0,054 | 0,262±0,077 | 0,506±0,011 | 0,724±0,039 | 0,029±0,037 | 2,0 |
| **Trung bình** | UAM-AD | 5 | 0,616±0,018 | – | 0,844±0,004 | – | 0,636±0,005 | – | – | – | – | – | – | – |
| **Trung bình** | ART | 5 | 0,572±0,013 | +0,044 (+7,6%) | 0,856±0,004 | −0,012 (−1,4%) | 0,686±0,008 | −0,050 (−7,2%) | – | – | – | – | – | – |
| **Trung bình** | AnoFusion (4 dataset) | 1 | 0,486 | +0,155 (+32,0%) | 0,739 | +0,141 (+19,0%) | 0,567 | +0,142 (+25,1%) | – | – | – | – | – | – |
| **Trung bình** | Eadro (có giám sát) | 5 | 0,637±0,007 | −0,021 (−3,3%) | 0,821±0,005 | +0,022 (+2,7%) | 0,709±0,005 | −0,073 (−10,2%) | – | – | – | – | – | – |
| **Trung bình** | OrEdge (bán giám sát) | 5 | 0,701±0,020 | −0,086 (−12,2%) | 0,888±0,008 | −0,044 (−5,0%) | 0,795±0,013 | −0,159 (−20,0%) | – | – | – | – | – | – |

*Theo scenario (5 seed):*

**RE2-OB**

| Scenario | Mô hình | seed | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAM-AD | 5 | 0,624±0,067 | 0,895±0,027 | 0,752±0,045 | 0,655±0,051 | 0,861±0,033 | 0,971±0,013 | 0,989±0,002 |
| `cpu` | ART | 5 | 0,846±0,011 | 0,978±0,005 | 0,942±0,011 | 0,856±0,008 | 0,934±0,012 | 0,965±0,005 | 0,981±0,003 |
| `cpu` | AnoFusion | 1 | 0,349 | 0,663 | 0,455 | 0,419 | – | – | – |
| `cpu` | Eadro (có giám sát) | 5 | 0,749±0,012 | 0,870±0,018 | 0,822±0,015 | 0,783±0,008 | 0,673±0,045 | 0,944±0,023 | 0,996±0,000 |
| `cpu` | OrEdge (bán giám sát) | 5 | 0,883±0,009 | 0,991±0,002 | 0,987±0,001 | 0,973±0,009 | 0,976±0,008 | 0,990±0,002 | 0,993±0,001 |
| `delay` | UAM-AD | 5 | 0,723±0,087 | 0,956±0,014 | 0,900±0,022 | 0,840±0,012 | 0,543±0,050 | 0,918±0,036 | 0,982±0,012 |
| `delay` | ART | 5 | 0,930±0,008 | 0,991±0,000 | 0,989±0,000 | 0,986±0,001 | 0,632±0,028 | 0,917±0,020 | 0,986±0,002 |
| `delay` | AnoFusion | 1 | 0,728 | 0,882 | 0,805 | 0,729 | – | – | – |
| `delay` | Eadro (có giám sát) | 5 | 0,886±0,024 | 0,983±0,010 | 0,965±0,015 | 0,911±0,018 | 0,851±0,054 | 0,966±0,015 | 0,997±0,002 |
| `delay` | OrEdge (bán giám sát) | 5 | 0,783±0,056 | 0,934±0,020 | 0,883±0,050 | 0,840±0,061 | 0,872±0,014 | 0,893±0,024 | 0,901±0,027 |
| `disk` | UAM-AD | 5 | 0,818±0,085 | 0,997±0,001 | 0,995±0,001 | 0,991±0,000 | 0,983±0,002 | 0,988±0,002 | 0,992±0,001 |
| `disk` | ART | 5 | 0,931±0,008 | 0,995±0,000 | 0,993±0,000 | 0,992±0,000 | 0,400±0,000 | 0,406±0,000 | 0,799±0,000 |
| `disk` | AnoFusion | 1 | 0,879 | 0,992 | 0,989 | 0,990 | – | – | – |
| `disk` | Eadro (có giám sát) | 5 | 0,799±0,010 | 0,924±0,012 | 0,892±0,008 | 0,879±0,004 | 0,924±0,030 | 0,983±0,019 | 1,000±0,000 |
| `disk` | OrEdge (bán giám sát) | 5 | 0,895±0,010 | 0,993±0,001 | 0,991±0,001 | 0,989±0,002 | 0,988±0,002 | 0,994±0,001 | 0,996±0,001 |
| `loss` | UAM-AD | 5 | 0,744±0,084 | 0,945±0,015 | 0,873±0,026 | 0,804±0,025 | 0,635±0,035 | 0,913±0,020 | 0,951±0,015 |
| `loss` | ART | 5 | 0,914±0,008 | 0,986±0,001 | 0,978±0,001 | 0,965±0,002 | 0,803±0,003 | 0,913±0,003 | 0,938±0,003 |
| `loss` | AnoFusion | 1 | 0,616 | 0,802 | 0,699 | 0,623 | – | – | – |
| `loss` | Eadro (có giám sát) | 5 | 0,449±0,092 | 0,648±0,040 | 0,503±0,073 | 0,501±0,091 | 0,257±0,012 | 0,656±0,003 | 0,998±0,002 |
| `loss` | OrEdge (bán giám sát) | 5 | 0,856±0,017 | 0,947±0,010 | 0,929±0,011 | 0,919±0,011 | 0,771±0,025 | 0,825±0,035 | 0,839±0,036 |
| `mem` | UAM-AD | 5 | 0,772±0,096 | 0,971±0,013 | 0,917±0,023 | 0,841±0,025 | 0,680±0,056 | 0,957±0,025 | 0,980±0,013 |
| `mem` | ART | 5 | 0,929±0,008 | 0,995±0,000 | 0,992±0,000 | 0,987±0,001 | 0,835±0,007 | 0,915±0,001 | 0,925±0,001 |
| `mem` | AnoFusion | 1 | 0,696 | 0,868 | 0,789 | 0,711 | – | – | – |
| `mem` | Eadro (có giám sát) | 5 | 0,766±0,009 | 0,896±0,019 | 0,810±0,038 | 0,776±0,009 | 0,688±0,037 | 0,955±0,006 | 0,989±0,012 |
| `mem` | OrEdge (bán giám sát) | 5 | 0,884±0,014 | 0,989±0,002 | 0,985±0,003 | 0,982±0,002 | 0,980±0,004 | 0,987±0,003 | 0,991±0,003 |
| `socket` | UAM-AD | 5 | 0,405±0,045 | 0,741±0,029 | 0,505±0,037 | 0,450±0,020 | 0,855±0,071 | 0,945±0,035 | 0,962±0,024 |
| `socket` | ART | 5 | 0,729±0,026 | 0,943±0,006 | 0,847±0,015 | 0,751±0,019 | 0,917±0,016 | 0,951±0,002 | 0,966±0,002 |
| `socket` | AnoFusion | 1 | 0,253 | 0,708 | 0,422 | 0,453 | – | – | – |
| `socket` | Eadro (có giám sát) | 5 | 0,619±0,010 | 0,879±0,021 | 0,751±0,023 | 0,642±0,014 | 0,610±0,029 | 0,778±0,021 | 0,996±0,002 |
| `socket` | OrEdge (bán giám sát) | 5 | 0,899±0,016 | 0,986±0,008 | 0,978±0,014 | 0,965±0,018 | 0,978±0,011 | 0,988±0,005 | 0,990±0,005 |

**RE3-OB**

| Scenario | Mô hình | seed | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAM-AD | 5 | 0,754±0,023 | 0,938±0,005 | 0,816±0,016 | 0,907±0,007 | 0,383±0,010 | 0,632±0,009 | 0,883±0,012 |
| `f1` | ART | 5 | 0,866±0,043 | 0,975±0,003 | 0,945±0,009 | 0,896±0,017 | 0,632±0,006 | 0,650±0,002 | 0,668±0,002 |
| `f1` | AnoFusion | 1 | 0,773 | 0,829 | 0,775 | 0,775 | – | – | – |
| `f1` | Eadro (có giám sát) | 5 | 0,914±0,035 | 0,978±0,010 | 0,961±0,018 | 0,921±0,033 | 0,980±0,003 | 1,000±0,000 | 1,000±0,000 |
| `f1` | OrEdge (bán giám sát) | 5 | 0,764±0,013 | 0,898±0,011 | 0,845±0,014 | 0,799±0,011 | 0,955±0,023 | 0,991±0,007 | 0,996±0,003 |
| `f2` | UAM-AD | 5 | 0,848±0,024 | 0,950±0,004 | 0,831±0,017 | 0,942±0,003 | 0,959±0,023 | 0,980±0,008 | 0,987±0,009 |
| `f2` | ART | 5 | 0,699±0,156 | 0,969±0,007 | 0,893±0,032 | 0,874±0,031 | 0,928±0,009 | 0,945±0,007 | 0,963±0,003 |
| `f2` | AnoFusion | 1 | 0,386 | 0,627 | 0,456 | 0,405 | – | – | – |
| `f2` | Eadro (có giám sát) | 5 | 0,630±0,127 | 0,855±0,084 | 0,745±0,132 | 0,707±0,154 | 1,000±0,000 | 1,000±0,000 | 1,000±0,000 |
| `f2` | OrEdge (bán giám sát) | 5 | 0,893±0,022 | 0,977±0,003 | 0,973±0,003 | 0,977±0,001 | 0,995±0,006 | 1,000±0,000 | 1,000±0,000 |
| `f3` | UAM-AD | 5 | 0,276±0,035 | 0,770±0,010 | 0,381±0,017 | 0,658±0,009 | 0,958±0,007 | 0,988±0,002 | 0,992±0,002 |
| `f3` | ART | 5 | 0,146±0,002 | 0,824±0,015 | 0,524±0,015 | 0,606±0,019 | 0,635±0,024 | 0,773±0,030 | 0,842±0,032 |
| `f3` | AnoFusion | 1 | 0,121 | 0,515 | 0,251 | 0,341 | – | – | – |
| `f3` | Eadro (có giám sát) | 5 | 0,504±0,079 | 0,633±0,057 | 0,545±0,068 | 0,520±0,074 | 0,593±0,068 | 1,000±0,000 | 1,000±0,000 |
| `f3` | OrEdge (bán giám sát) | 5 | 0,871±0,009 | 0,974±0,004 | 0,957±0,009 | 0,924±0,025 | 0,967±0,012 | 0,992±0,003 | 0,998±0,002 |
| `f4` | UAM-AD | 5 | 0,160±0,005 | 0,575±0,003 | 0,241±0,002 | 0,361±0,011 | 0,864±0,104 | 0,929±0,103 | 0,947±0,090 |
| `f4` | ART | 5 | 0,170±0,004 | 0,731±0,005 | 0,466±0,005 | 0,519±0,005 | 0,495±0,010 | 0,605±0,007 | 0,684±0,011 |
| `f4` | AnoFusion | 1 | 0,117 | 0,527 | 0,264 | 0,341 | – | – | – |
| `f4` | Eadro (có giám sát) | 5 | 0,610±0,026 | 0,820±0,066 | 0,709±0,057 | 0,627±0,034 | 0,878±0,063 | 1,000±0,000 | 1,000±0,000 |
| `f4` | OrEdge (bán giám sát) | 5 | 0,380±0,026 | 0,630±0,014 | 0,407±0,017 | 0,394±0,023 | 0,806±0,053 | 0,991±0,005 | 0,999±0,001 |
| `f5` | UAM-AD | 5 | 0,761±0,016 | 0,940±0,004 | 0,807±0,011 | 0,909±0,007 | 0,956±0,023 | 0,981±0,001 | 0,987±0,001 |
| `f5` | ART | 5 | 0,731±0,100 | 0,968±0,004 | 0,904±0,020 | 0,884±0,022 | 0,945±0,008 | 0,966±0,004 | 0,980±0,002 |
| `f5` | AnoFusion | 1 | 0,431 | 0,654 | 0,487 | 0,437 | – | – | – |
| `f5` | Eadro (có giám sát) | 5 | 0,569±0,123 | 0,798±0,056 | 0,672±0,086 | 0,620±0,072 | 0,812±0,057 | 1,000±0,000 | 1,000±0,000 |
| `f5` | OrEdge (bán giám sát) | 5 | 0,902±0,017 | 0,984±0,001 | 0,981±0,001 | 0,981±0,001 | 0,982±0,002 | 0,997±0,002 | 1,000±0,000 |

**RE2-TT**

| Scenario | Mô hình | seed | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAM-AD | 5 | 0,408±0,022 | 0,688±0,012 | 0,466±0,030 | 0,424±0,009 | 0,836±0,035 | 0,968±0,024 | 0,983±0,009 |
| `cpu` | ART | 5 | 0,260±0,007 | 0,725±0,001 | 0,390±0,004 | 0,442±0,002 | 0,443±0,008 | 0,523±0,007 | 0,665±0,014 |
| `cpu` | AnoFusion | 1 | 0,608 | 0,855 | 0,698 | 0,626 | – | – | – |
| `cpu` | Eadro (có giám sát) | 5 | 0,616±0,045 | 0,797±0,023 | 0,691±0,043 | 0,640±0,038 | 0,593±0,041 | 0,762±0,032 | 0,998±0,001 |
| `cpu` | OrEdge (bán giám sát) | 5 | 0,830±0,011 | 0,993±0,003 | 0,987±0,004 | 0,971±0,003 | 0,991±0,002 | 0,998±0,001 | 0,999±0,000 |
| `delay` | UAM-AD | 5 | 0,692±0,012 | 0,851±0,014 | 0,790±0,010 | 0,767±0,000 | 0,714±0,046 | 0,907±0,041 | 0,964±0,042 |
| `delay` | ART | 5 | 0,700±0,001 | 0,849±0,005 | 0,779±0,003 | 0,770±0,000 | 0,542±0,009 | 0,674±0,010 | 0,755±0,011 |
| `delay` | AnoFusion | 1 | 0,561 | 0,652 | 0,606 | 0,612 | – | – | – |
| `delay` | Eadro (có giám sát) | 5 | 0,590±0,022 | 0,801±0,032 | 0,656±0,041 | 0,641±0,033 | 0,386±0,020 | 0,683±0,011 | 0,996±0,003 |
| `delay` | OrEdge (bán giám sát) | 5 | 0,873±0,008 | 0,974±0,018 | 0,966±0,014 | 0,948±0,005 | 0,962±0,015 | 0,992±0,007 | 0,998±0,002 |
| `disk` | UAM-AD | 5 | 0,765±0,008 | 0,924±0,003 | 0,859±0,003 | 0,813±0,023 | 0,986±0,001 | 0,989±0,001 | 0,991±0,001 |
| `disk` | ART | 5 | 0,890±0,001 | 0,992±0,000 | 0,987±0,000 | 0,989±0,000 | 0,979±0,001 | 0,986±0,000 | 0,989±0,000 |
| `disk` | AnoFusion | 1 | 0,892 | 0,988 | 0,982 | 0,974 | – | – | – |
| `disk` | Eadro (có giám sát) | 5 | 0,891±0,019 | 0,987±0,005 | 0,977±0,013 | 0,963±0,041 | 0,753±0,058 | 0,965±0,019 | 1,000±0,000 |
| `disk` | OrEdge (bán giám sát) | 5 | 0,889±0,011 | 0,993±0,001 | 0,990±0,001 | 0,989±0,001 | 0,989±0,001 | 0,993±0,001 | 0,995±0,002 |
| `loss` | UAM-AD | 5 | 0,512±0,011 | 0,727±0,010 | 0,589±0,011 | 0,518±0,009 | 0,442±0,035 | 0,699±0,060 | 0,788±0,061 |
| `loss` | ART | 5 | 0,388±0,002 | 0,669±0,004 | 0,480±0,003 | 0,397±0,003 | 0,291±0,004 | 0,372±0,011 | 0,485±0,013 |
| `loss` | AnoFusion | 1 | 0,138 | 0,418 | 0,239 | 0,343 | – | – | – |
| `loss` | Eadro (có giám sát) | 5 | 0,376±0,034 | 0,674±0,014 | 0,470±0,020 | 0,422±0,009 | 0,331±0,029 | 0,632±0,011 | 0,997±0,001 |
| `loss` | OrEdge (bán giám sát) | 5 | 0,706±0,008 | 0,865±0,022 | 0,787±0,015 | 0,724±0,010 | 0,843±0,042 | 0,949±0,025 | 0,988±0,010 |
| `mem` | UAM-AD | 5 | 0,755±0,025 | 0,957±0,008 | 0,853±0,021 | 0,773±0,029 | 0,643±0,100 | 0,755±0,101 | 0,783±0,096 |
| `mem` | ART | 5 | 0,894±0,001 | 0,995±0,000 | 0,990±0,000 | 0,988±0,000 | 0,025±0,000 | 0,030±0,000 | 0,032±0,000 |
| `mem` | AnoFusion | 1 | 0,835 | 0,954 | 0,915 | 0,892 | – | – | – |
| `mem` | Eadro (có giám sát) | 5 | 0,889±0,015 | 0,990±0,002 | 0,984±0,004 | 0,974±0,007 | 0,777±0,064 | 0,951±0,033 | 1,000±0,000 |
| `mem` | OrEdge (bán giám sát) | 5 | 0,819±0,008 | 0,990±0,002 | 0,982±0,005 | 0,963±0,005 | 0,989±0,002 | 0,995±0,001 | 0,997±0,001 |
| `socket` | UAM-AD | 5 | 0,449±0,006 | 0,704±0,001 | 0,522±0,007 | 0,451±0,006 | 0,705±0,106 | 0,907±0,040 | 0,939±0,039 |
| `socket` | ART | 5 | 0,367±0,009 | 0,768±0,005 | 0,500±0,009 | 0,473±0,005 | 0,404±0,009 | 0,482±0,007 | 0,615±0,018 |
| `socket` | AnoFusion | 1 | 0,468 | 0,774 | 0,592 | 0,564 | – | – | – |
| `socket` | Eadro (có giám sát) | 5 | 0,483±0,067 | 0,777±0,036 | 0,589±0,058 | 0,509±0,049 | 0,396±0,019 | 0,687±0,006 | 0,967±0,016 |
| `socket` | OrEdge (bán giám sát) | 5 | 0,871±0,007 | 0,990±0,001 | 0,983±0,002 | 0,962±0,005 | 0,951±0,002 | 0,968±0,003 | 0,986±0,005 |

**RE3-TT**

| Scenario | Mô hình | seed | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAM-AD | 5 | 0,520±0,002 | 0,718±0,003 | 0,358±0,003 | 0,644±0,003 | 0,309±0,007 | 0,383±0,016 | 0,456±0,027 |
| `f1` | ART | 5 | 0,222±0,014 | 0,655±0,016 | 0,366±0,016 | 0,403±0,008 | 0,436±0,072 | 0,522±0,038 | 0,586±0,029 |
| `f1` | AnoFusion | 1 | 0,116 | 0,502 | 0,206 | 0,351 | – | – | – |
| `f1` | Eadro (có giám sát) | 5 | 0,285±0,081 | 0,529±0,033 | 0,362±0,044 | 0,352±0,018 | 0,521±0,006 | 1,000±0,001 | 1,000±0,001 |
| `f1` | OrEdge (bán giám sát) | 5 | 0,688±0,007 | 0,804±0,003 | 0,750±0,003 | 0,736±0,001 | 0,941±0,009 | 0,995±0,001 | 0,996±0,001 |
| `f2` | UAM-AD | 5 | 0,508±0,003 | 0,688±0,003 | 0,341±0,004 | 0,628±0,005 | 0,281±0,005 | 0,365±0,016 | 0,431±0,023 |
| `f2` | ART | 5 | 0,209±0,010 | 0,644±0,017 | 0,356±0,016 | 0,399±0,008 | 0,419±0,083 | 0,511±0,050 | 0,586±0,037 |
| `f2` | Eadro (có giám sát) | 5 | 0,281±0,086 | 0,558±0,034 | 0,361±0,045 | 0,365±0,015 | 0,527±0,011 | 1,000±0,000 | 1,000±0,000 |
| `f2` | OrEdge (bán giám sát) | 5 | 0,689±0,016 | 0,799±0,005 | 0,746±0,004 | 0,731±0,007 | 0,938±0,009 | 0,997±0,002 | 0,998±0,000 |
| `f3` | UAM-AD | 5 | 0,518±0,003 | 0,721±0,003 | 0,353±0,004 | 0,638±0,004 | 0,131±0,004 | 0,219±0,024 | 0,361±0,021 |
| `f3` | ART | 5 | 0,630±0,002 | 0,777±0,002 | 0,700±0,001 | 0,687±0,000 | 0,076±0,003 | 0,102±0,003 | 0,157±0,009 |
| `f3` | Eadro (có giám sát) | 5 | 0,597±0,027 | 0,761±0,013 | 0,669±0,011 | 0,671±0,003 | 0,913±0,008 | 1,000±0,000 | 1,000±0,000 |
| `f3` | OrEdge (bán giám sát) | 5 | 0,456±0,011 | 0,651±0,011 | 0,519±0,009 | 0,468±0,007 | 0,801±0,038 | 0,997±0,002 | 1,000±0,000 |
| `f4` | UAM-AD | 5 | 0,508±0,002 | 0,682±0,005 | 0,334±0,003 | 0,623±0,003 | 0,282±0,014 | 0,361±0,016 | 0,420±0,027 |
| `f4` | ART | 5 | 0,228±0,024 | 0,658±0,018 | 0,371±0,022 | 0,403±0,013 | 0,443±0,055 | 0,516±0,037 | 0,581±0,025 |
| `f4` | Eadro (có giám sát) | 5 | 0,299±0,105 | 0,575±0,029 | 0,377±0,055 | 0,383±0,008 | 0,525±0,009 | 0,999±0,001 | 0,999±0,001 |
| `f4` | OrEdge (bán giám sát) | 5 | 0,690±0,003 | 0,802±0,006 | 0,746±0,005 | 0,730±0,003 | 0,942±0,014 | 0,993±0,001 | 0,995±0,001 |

**SN**

| Scenario | Mô hình | seed | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `Code_Stop_MediaService` | UAM-AD | 5 | 0,759±0,018 | 0,966±0,002 | 0,827±0,010 | 0,911±0,012 | 0,708±0,038 | 0,846±0,043 | 0,867±0,041 |
| `Code_Stop_MediaService` | ART | 5 | 0,772±0,022 | 0,972±0,008 | 0,852±0,037 | 0,807±0,034 | 0,082±0,086 | 0,185±0,071 | 0,313±0,038 |
| `Code_Stop_MediaService` | AnoFusion | 1 | 0,427 | 0,730 | 0,557 | 0,582 | – | – | – |
| `Code_Stop_MediaService` | Eadro (có giám sát) | 5 | 0,767±0,075 | 0,967±0,008 | 0,754±0,048 | 0,836±0,032 | 0,000±0,000 | 0,000±0,000 | 0,185±0,157 |
| `Code_Stop_MediaService` | OrEdge (bán giám sát) | 5 | 0,492±0,384 | 0,749±0,192 | 0,564±0,328 | 0,655±0,251 | 0,651±0,317 | 0,862±0,075 | 0,887±0,045 |
| `Code_Stop_TextService` | UAM-AD | 5 | 0,699±0,004 | 0,976±0,002 | 0,903±0,007 | 0,929±0,000 | 0,492±0,163 | 0,728±0,180 | 0,810±0,158 |
| `Code_Stop_TextService` | ART | 5 | 0,782±0,015 | 0,987±0,004 | 0,933±0,019 | 0,880±0,024 | 0,154±0,104 | 0,303±0,034 | 0,323±0,035 |
| `Code_Stop_TextService` | AnoFusion | 1 | 0,595 | 0,777 | 0,723 | 0,781 | – | – | – |
| `Code_Stop_TextService` | Eadro (có giám sát) | 5 | 0,834±0,012 | 0,994±0,001 | 0,958±0,013 | 0,937±0,012 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 |
| `Code_Stop_TextService` | OrEdge (bán giám sát) | 5 | 0,323±0,384 | 0,747±0,165 | 0,532±0,237 | 0,639±0,200 | 0,651±0,247 | 0,913±0,050 | 0,969±0,019 |
| `Code_Stop_UserService` | UAM-AD | 5 | 0,748±0,006 | 0,983±0,001 | 0,947±0,003 | 0,894±0,000 | 0,305±0,266 | 0,742±0,119 | 0,800±0,106 |
| `Code_Stop_UserService` | ART | 5 | 0,756±0,036 | 0,986±0,006 | 0,944±0,016 | 0,886±0,000 | 0,026±0,023 | 0,118±0,048 | 0,149±0,019 |
| `Code_Stop_UserService` | AnoFusion | 1 | 0,689 | 0,911 | 0,861 | 0,886 | – | – | – |
| `Code_Stop_UserService` | Eadro (có giám sát) | 5 | 0,739±0,158 | 0,965±0,033 | 0,769±0,143 | 0,850±0,079 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 |
| `Code_Stop_UserService` | OrEdge (bán giám sát) | 5 | 0,639±0,287 | 0,847±0,085 | 0,744±0,143 | 0,803±0,075 | 0,554±0,226 | 0,738±0,030 | 0,800±0,094 |
| `DB_Redis_CacheLimit_HomeTimeline` | UAM-AD | 5 | 0,803±0,025 | 0,965±0,002 | 0,751±0,020 | 0,833±0,000 | 0,160±0,136 | 0,360±0,049 | 0,400±0,063 |
| `DB_Redis_CacheLimit_HomeTimeline` | ART | 5 | 0,576±0,100 | 0,917±0,025 | 0,619±0,081 | 0,675±0,036 | 0,140±0,080 | 0,400±0,110 | 0,640±0,080 |
| `DB_Redis_CacheLimit_HomeTimeline` | AnoFusion | 1 | 0,667 | 0,870 | 0,459 | 0,696 | – | – | – |
| `DB_Redis_CacheLimit_HomeTimeline` | Eadro (có giám sát) | 5 | 0,810±0,023 | 0,979±0,003 | 0,851±0,021 | 0,848±0,018 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 |
| `DB_Redis_CacheLimit_HomeTimeline` | OrEdge (bán giám sát) | 5 | 0,381±0,230 | 0,821±0,068 | 0,475±0,106 | 0,589±0,096 | 0,100±0,000 | 0,100±0,000 | 0,220±0,040 |
| `DB_Redis_CacheLimit_SocialGraph` | UAM-AD | 5 | 0,857±0,000 | 0,984±0,001 | 0,835±0,034 | 0,891±0,017 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 |
| `DB_Redis_CacheLimit_SocialGraph` | ART | 5 | 0,593±0,037 | 0,883±0,044 | 0,506±0,090 | 0,652±0,069 | 0,020±0,040 | 0,080±0,040 | 0,200±0,110 |
| `DB_Redis_CacheLimit_SocialGraph` | AnoFusion | 1 | 0,636 | 0,869 | 0,789 | 0,824 | – | – | – |
| `DB_Redis_CacheLimit_SocialGraph` | Eadro (có giám sát) | 5 | 0,862±0,014 | 0,973±0,002 | 0,730±0,031 | 0,877±0,016 | 0,000±0,000 | 0,120±0,117 | 0,700±0,000 |
| `DB_Redis_CacheLimit_SocialGraph` | OrEdge (bán giám sát) | 5 | 0,779±0,105 | 0,970±0,004 | 0,685±0,014 | 0,881±0,037 | 0,100±0,000 | 0,140±0,080 | 0,540±0,215 |
| `DB_Redis_CacheLimit_UserTimeline` | UAM-AD | 5 | 0,556±0,000 | 0,938±0,004 | 0,608±0,057 | 0,738±0,028 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 |
| `DB_Redis_CacheLimit_UserTimeline` | ART | 5 | 0,344±0,039 | 0,861±0,018 | 0,412±0,019 | 0,532±0,045 | 0,100±0,089 | 0,160±0,150 | 0,480±0,160 |
| `DB_Redis_CacheLimit_UserTimeline` | AnoFusion | 1 | 0,000 | 0,527 | 0,138 | 0,265 | – | – | – |
| `DB_Redis_CacheLimit_UserTimeline` | Eadro (có giám sát) | 5 | 0,822±0,048 | 0,984±0,007 | 0,895±0,044 | 0,865±0,029 | 0,120±0,117 | 0,420±0,194 | 0,800±0,000 |
| `DB_Redis_CacheLimit_UserTimeline` | OrEdge (bán giám sát) | 5 | 0,284±0,055 | 0,839±0,046 | 0,472±0,051 | 0,524±0,081 | 0,100±0,200 | 0,800±0,063 | 1,000±0,000 |
| `Perf_CPU_Contention` | UAM-AD | 5 | 0,775±0,012 | 0,962±0,004 | 0,711±0,036 | 0,841±0,015 | – | – | – |
| `Perf_CPU_Contention` | ART | 5 | 0,672±0,035 | 0,964±0,004 | 0,794±0,010 | 0,804±0,032 | – | – | – |
| `Perf_CPU_Contention` | AnoFusion | 1 | 0,667 | 0,863 | 0,418 | 0,667 | – | – | – |
| `Perf_CPU_Contention` | Eadro (có giám sát) | 5 | 0,852±0,035 | 0,985±0,003 | 0,893±0,017 | 0,870±0,024 | – | – | – |
| `Perf_CPU_Contention` | OrEdge (bán giám sát) | 5 | 0,474±0,234 | 0,835±0,080 | 0,591±0,136 | 0,614±0,137 | – | – | – |
| `Perf_Disk_IO_Stress` | UAM-AD | 5 | 0,651±0,013 | 0,912±0,004 | 0,460±0,025 | 0,727±0,000 | – | – | – |
| `Perf_Disk_IO_Stress` | ART | 5 | 0,311±0,032 | 0,796±0,046 | 0,343±0,036 | 0,451±0,046 | – | – | – |
| `Perf_Disk_IO_Stress` | AnoFusion | 1 | 0,545 | 0,806 | 0,354 | 0,571 | – | – | – |
| `Perf_Disk_IO_Stress` | Eadro (có giám sát) | 5 | 0,758±0,014 | 0,960±0,008 | 0,751±0,059 | 0,765±0,008 | – | – | – |
| `Perf_Disk_IO_Stress` | OrEdge (bán giám sát) | 5 | 0,393±0,143 | 0,845±0,069 | 0,411±0,093 | 0,541±0,102 | – | – | – |
| `Perf_Network_Loss` | UAM-AD | 5 | 0,879±0,022 | 0,983±0,002 | 0,790±0,007 | 0,935±0,021 | – | – | – |
| `Perf_Network_Loss` | ART | 5 | 0,517±0,047 | 0,878±0,026 | 0,521±0,052 | 0,617±0,073 | – | – | – |
| `Perf_Network_Loss` | AnoFusion | 1 | 0,667 | 0,889 | 0,624 | 0,667 | – | – | – |
| `Perf_Network_Loss` | Eadro (có giám sát) | 5 | 0,870±0,024 | 0,982±0,003 | 0,876±0,012 | 0,877±0,016 | – | – | – |
| `Perf_Network_Loss` | OrEdge (bán giám sát) | 5 | 0,617±0,113 | 0,949±0,013 | 0,705±0,033 | 0,762±0,056 | – | – | – |
| `Svc_Kill_Media` | UAM-AD | 5 | 0,750±0,000 | 0,963±0,004 | 0,704±0,050 | 0,889±0,000 | 0,000±0,000 | 0,450±0,367 | 0,650±0,339 |
| `Svc_Kill_Media` | ART | 5 | 0,492±0,047 | 0,823±0,021 | 0,458±0,051 | 0,571±0,000 | 0,000±0,000 | 0,200±0,187 | 0,800±0,187 |
| `Svc_Kill_Media` | AnoFusion | 1 | 0,000 | 0,509 | 0,195 | 0,286 | – | – | – |
| `Svc_Kill_Media` | Eadro (có giám sát) | 5 | 0,818±0,036 | 0,995±0,007 | 0,968±0,044 | 0,956±0,054 | 0,000±0,000 | 0,000±0,000 | 0,800±0,400 |
| `Svc_Kill_Media` | OrEdge (bán giám sát) | 5 | 0,328±0,186 | 0,841±0,160 | 0,536±0,196 | 0,602±0,185 | 0,000±0,000 | 0,000±0,000 | 0,100±0,122 |
| `Svc_Kill_SocialGraph` | UAM-AD | 5 | 0,525±0,040 | 0,899±0,009 | 0,427±0,035 | 0,733±0,033 | 0,000±0,000 | 0,000±0,000 | 0,000±0,000 |
| `Svc_Kill_SocialGraph` | ART | 5 | 0,490±0,107 | 0,809±0,051 | 0,342±0,110 | 0,582±0,097 | 0,000±0,000 | 0,000±0,000 | 0,100±0,122 |
| `Svc_Kill_SocialGraph` | AnoFusion | 1 | 0,000 | 0,420 | 0,129 | 0,273 | – | – | – |
| `Svc_Kill_SocialGraph` | Eadro (có giám sát) | 5 | 0,889±0,000 | 0,991±0,014 | 0,926±0,125 | 0,956±0,054 | 0,000±0,000 | 1,000±0,000 | 1,000±0,000 |
| `Svc_Kill_SocialGraph` | OrEdge (bán giám sát) | 5 | 0,283±0,348 | 0,902±0,033 | 0,488±0,104 | 0,690±0,053 | 0,000±0,000 | 0,000±0,000 | 1,000±0,000 |
| `Svc_Kill_UserTimeline` | UAM-AD | 5 | 0,733±0,033 | 0,946±0,005 | 0,632±0,012 | 0,871±0,036 | 0,000±0,000 | 0,000±0,000 | 0,050±0,100 |
| `Svc_Kill_UserTimeline` | ART | 5 | 0,337±0,098 | 0,800±0,068 | 0,353±0,051 | 0,517±0,073 | 0,000±0,000 | 0,000±0,000 | 0,100±0,200 |
| `Svc_Kill_UserTimeline` | AnoFusion | 1 | 0,000 | 0,607 | 0,214 | 0,400 | – | – | – |
| `Svc_Kill_UserTimeline` | Eadro (có giám sát) | 5 | 0,818±0,036 | 0,989±0,014 | 0,913±0,125 | 0,956±0,054 | 0,550±0,400 | 0,850±0,200 | 1,000±0,000 |
| `Svc_Kill_UserTimeline` | OrEdge (bán giám sát) | 5 | 0,000±0,000 | 0,786±0,101 | 0,352±0,123 | 0,552±0,116 | 0,200±0,245 | 1,000±0,000 | 1,000±0,000 |

**Kết quả seed 0 của ART, AnoFusion, Eadro và OrEdge (chi tiết từng seed; sinh tự động bởi `summary_tools/item9_tables.py`)**

Cách đọc: trung bình qua scenario của từng dataset; mọi mô hình trong bảng dùng đủ ba modality L+M+T. UAM-AD lấy cùng seed từ bước 8 (`norm_sum`). **F1 oracle của UAM-AD ở SN, RE3-OB, RE3-TT là quét ngưỡng CÓ point-adjust (`run.py`); của ART, AnoFusion, Eadro, OrEdge là quét ngưỡng thường, KHÔNG point-adjust, nên ở ba dataset đó cột này không so được. Ở RE2-OB, RE2-TT, F1 oracle của UAM-AD (và UAC-AD) là bản không point-adjust của việc 10, cùng định nghĩa với đối thủ.** HR@k của UAM-AD (việc 10: điểm trace chuẩn hoá cộng điểm KPI của service) và của ART, Eadro, OrEdge đều tính trên mọi mẫu bất thường có service lỗi; đơn vị chấm vẫn khác (UAM-AD cửa sổ 30 mẫu, các đối thủ từng mẫu giây hoặc cửa sổ 10), nên chưa so thẳng được. "F1 argmax" chỉ có ở Eadro, OrEdge (quyết định của bộ phân loại). Cột cuối là phút (UAM-AD: tổng thời gian log của các scenario; đối thủ: tổng thời gian chạy ô).

Cột **Δ** (chỉ cho F1@val95, AUROC, AUPRC) = UAM-AD trừ mô hình ở dòng đó; trong ngoặc là chênh lệch tính theo % giá trị của mô hình ở dòng đó (dương = UAM-AD cao hơn). Chỉ tính trên các scenario cả hai cùng có. Dấu * = mô hình dừng sớm (AnoFusion RE3-TT chỉ có `f1`): so trên scenario đã chấm và không vào dòng Trung bình. Dòng **Trung bình** của một mô hình = trung bình các dataset mô hình đó có đủ; % = chênh lệch trung bình chia cho giá trị trung bình của mô hình đó (không phải trung bình của các %, vì nó bị dataset có nền thấp kéo lệch dấu).

*Theo dataset:*

| Dataset | Mô hình | F1@val95 | Δ F1 | AUROC | Δ AUROC | AUPRC | Δ AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | UAM-AD | 0,713 | – | 0,923 | – | 0,828 | – | 0,760 | 0,752 | 0,947 | 0,975 | – | 126,6 |
| RE2-OB | ART | 0,880 | −0,167 (−19,0%) | 0,982 | −0,059 (−6,0%) | 0,958 | −0,130 (−13,5%) | 0,921 | 0,751 | 0,840 | 0,932 | – | 6,0 |
| RE2-OB | AnoFusion | 0,587 | +0,126 (+21,5%) | 0,819 | +0,104 (+12,7%) | 0,693 | +0,135 (+19,5%) | 0,654 | – | – | – | – | 18,8 |
| RE2-OB | Eadro (có giám sát) | 0,698 | +0,015 (+2,2%) | 0,864 | +0,060 (+6,9%) | 0,788 | +0,040 (+5,1%) | 0,731 | 0,666 | 0,885 | 0,998 | 0,683 | 12,7 |
| RE2-OB | OrEdge (bán giám sát) | 0,846 | −0,133 (−15,7%) | 0,969 | −0,045 (−4,7%) | 0,944 | −0,116 (−12,3%) | 0,930 | 0,932 | 0,951 | 0,957 | 0,886 | 31,1 |
| RE3-OB | UAM-AD | 0,556 | – | 0,832 | – | 0,611 | – | 0,750 | 0,778 | 0,859 | 0,921 | – | 36,3 |
| RE3-OB | ART | 0,548 | +0,008 (+1,4%) | 0,892 | −0,061 (−6,8%) | 0,747 | −0,136 (−18,3%) | 0,747 | 0,733 | 0,787 | 0,820 | – | 2,0 |
| RE3-OB | AnoFusion | 0,366 | +0,190 (+51,9%) | 0,630 | +0,201 (+31,9%) | 0,447 | +0,164 (+36,8%) | 0,460 | – | – | – | – | 48,1 |
| RE3-OB | Eadro (có giám sát) | 0,668 | −0,112 (−16,8%) | 0,842 | −0,010 (−1,2%) | 0,758 | −0,147 (−19,4%) | 0,698 | 0,889 | 1,000 | 1,000 | 0,666 | 4,5 |
| RE3-OB | OrEdge (bán giám sát) | 0,767 | −0,212 (−27,6%) | 0,891 | −0,059 (−6,6%) | 0,830 | −0,219 (−26,4%) | 0,827 | 0,930 | 0,994 | 0,999 | 0,770 | 11,1 |
| RE2-TT | UAM-AD | 0,589 | – | 0,810 | – | 0,685 | – | 0,627 | 0,687 | 0,835 | 0,872 | – | 184,1 |
| RE2-TT | ART | 0,580 | +0,009 (+1,5%) | 0,831 | −0,021 (−2,6%) | 0,685 | +0,000 (+0,1%) | 0,675 | 0,444 | 0,511 | 0,586 | – | 6,8 |
| RE2-TT | AnoFusion | 0,584 | +0,005 (+0,9%) | 0,773 | +0,036 (+4,7%) | 0,672 | +0,013 (+2,0%) | 0,668 | – | – | – | – | 220,3 |
| RE2-TT | Eadro (có giám sát) | 0,620 | −0,031 (−4,9%) | 0,825 | −0,016 (−1,9%) | 0,698 | −0,013 (−1,8%) | 0,659 | 0,523 | 0,768 | 0,991 | 0,596 | 34,5 |
| RE2-TT | OrEdge (bán giám sát) | 0,833 | −0,244 (−29,3%) | 0,968 | −0,159 (−16,4%) | 0,950 | −0,265 (−27,8%) | 0,924 | 0,946 | 0,979 | 0,994 | 0,899 | 32,8 |
| RE3-TT | UAM-AD | 0,515 | – | 0,705 | – | 0,350 | – | 0,631 | 0,244 | 0,318 | 0,392 | – | 48,1 |
| RE3-TT | ART | 0,317 | +0,198 (+62,4%) | 0,686 | +0,019 (+2,8%) | 0,447 | −0,097 (−21,8%) | 0,478 | 0,291 | 0,361 | 0,437 | – | 2,3 |
| RE3-TT | AnoFusion (CHỈ f1, dừng sớm) | 0,116 | +0,407 (+351,5%)* | 0,502 | +0,218 (+43,5%)* | 0,206 | +0,156 (+75,8%)* | 0,351 | – | – | – | – | – |
| RE3-TT | Eadro (có giám sát) | 0,350 | +0,165 (+47,1%) | 0,574 | +0,131 (+22,9%) | 0,427 | −0,077 (−18,0%) | 0,427 | 0,622 | 1,000 | 1,000 | 0,227 | 11,8 |
| RE3-TT | OrEdge (bán giám sát) | 0,624 | −0,109 (−17,4%) | 0,766 | −0,061 (−7,9%) | 0,690 | −0,340 (−49,3%) | 0,661 | 0,899 | 0,995 | 0,997 | 0,615 | 33,2 |
| SN | UAM-AD | 0,734 | – | 0,956 | – | 0,705 | – | 0,850 | 0,196 | 0,375 | 0,403 | – | 4,0 |
| SN | ART | 0,531 | +0,203 (+38,1%) | 0,870 | +0,086 (+9,9%) | 0,546 | +0,159 (+29,2%) | 0,638 | 0,048 | 0,168 | 0,294 | – | 0,4 |
| SN | AnoFusion | 0,408 | +0,326 (+80,0%) | 0,731 | +0,224 (+30,7%) | 0,455 | +0,250 (+55,0%) | 0,575 | – | – | – | – | 11,3 |
| SN | Eadro (có giám sát) | 0,811 | −0,077 (−9,5%) | 0,979 | −0,024 (−2,4%) | 0,851 | −0,146 (−17,1%) | 0,866 | 0,028 | 0,217 | 0,500 | 0,805 | 1,1 |
| SN | OrEdge (bán giám sát) | 0,345 | +0,389 (+112,5%) | 0,813 | +0,143 (+17,6%) | 0,480 | +0,225 (+46,9%) | 0,592 | 0,269 | 0,496 | 0,707 | 0,013 | 1,9 |
| **Trung bình** | UAM-AD | 0,621 | – | 0,845 | – | 0,636 | – | – | – | – | – | – | – |
| **Trung bình** | ART | 0,572 | +0,050 (+8,7%) | 0,852 | −0,007 (−0,8%) | 0,677 | −0,041 (−6,0%) | – | – | – | – | – | – |
| **Trung bình** | AnoFusion (4 dataset) | 0,486 | +0,162 (+33,3%) | 0,739 | +0,141 (+19,2%) | 0,567 | +0,141 (+24,8%) | – | – | – | – | – | – |
| **Trung bình** | Eadro (có giám sát) | 0,629 | −0,008 (−1,2%) | 0,817 | +0,028 (+3,5%) | 0,705 | −0,069 (−9,7%) | – | – | – | – | – | – |
| **Trung bình** | OrEdge (bán giám sát) | 0,683 | −0,062 (−9,0%) | 0,881 | −0,036 (−4,1%) | 0,779 | −0,143 (−18,3%) | – | – | – | – | – | – |

*Theo scenario:*

**RE2-OB**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAM-AD | 0,631 | 0,901 | 0,756 | 0,652 | 0,889 | 0,978 | 0,993 | – | 21,1 |
| `cpu` | ART | 0,846 | 0,979 | 0,943 | 0,850 | 0,925 | 0,957 | 0,977 | – | – |
| `cpu` | AnoFusion | 0,349 | 0,663 | 0,455 | 0,419 | – | – | – | – | – |
| `cpu` | Eadro (có giám sát) | 0,738 | 0,875 | 0,817 | 0,771 | 0,637 | 0,913 | 0,996 | 0,764 | – |
| `cpu` | OrEdge (bán giám sát) | 0,869 | 0,991 | 0,987 | 0,979 | 0,981 | 0,989 | 0,992 | 0,968 | – |
| `delay` | UAM-AD | 0,764 | 0,960 | 0,901 | 0,840 | 0,532 | 0,848 | 0,959 | – | 21,1 |
| `delay` | ART | 0,934 | 0,991 | 0,989 | 0,985 | 0,626 | 0,906 | 0,987 | – | – |
| `delay` | AnoFusion | 0,728 | 0,882 | 0,805 | 0,729 | – | – | – | – | – |
| `delay` | Eadro (có giám sát) | 0,857 | 0,963 | 0,937 | 0,883 | 0,888 | 0,981 | 1,000 | 0,881 | – |
| `delay` | OrEdge (bán giám sát) | 0,671 | 0,897 | 0,784 | 0,729 | 0,860 | 0,878 | 0,882 | 0,655 | – |
| `disk` | UAM-AD | 0,846 | 0,997 | 0,994 | 0,991 | 0,981 | 0,988 | 0,992 | – | 21,1 |
| `disk` | ART | 0,934 | 0,995 | 0,994 | 0,991 | 0,401 | 0,406 | 0,799 | – | – |
| `disk` | AnoFusion | 0,879 | 0,992 | 0,989 | 0,990 | – | – | – | – | – |
| `disk` | Eadro (có giám sát) | 0,796 | 0,929 | 0,897 | 0,882 | 0,980 | 0,999 | 1,000 | 0,879 | – |
| `disk` | OrEdge (bán giám sát) | 0,894 | 0,992 | 0,990 | 0,989 | 0,987 | 0,993 | 0,994 | 0,989 | – |
| `loss` | UAM-AD | 0,781 | 0,948 | 0,873 | 0,793 | 0,603 | 0,949 | 0,976 | – | 21,1 |
| `loss` | ART | 0,917 | 0,986 | 0,979 | 0,962 | 0,804 | 0,908 | 0,936 | – | – |
| `loss` | AnoFusion | 0,616 | 0,802 | 0,699 | 0,623 | – | – | – | – | – |
| `loss` | Eadro (có giám sát) | 0,417 | 0,602 | 0,463 | 0,427 | 0,254 | 0,655 | 1,000 | 0,231 | – |
| `loss` | OrEdge (bán giám sát) | 0,876 | 0,957 | 0,940 | 0,933 | 0,809 | 0,884 | 0,901 | 0,786 | – |
| `mem` | UAM-AD | 0,820 | 0,976 | 0,922 | 0,826 | 0,748 | 0,986 | 0,993 | – | 21,0 |
| `mem` | ART | 0,932 | 0,995 | 0,992 | 0,986 | 0,832 | 0,916 | 0,926 | – | – |
| `mem` | AnoFusion | 0,696 | 0,868 | 0,789 | 0,711 | – | – | – | – | – |
| `mem` | Eadro (có giám sát) | 0,765 | 0,916 | 0,842 | 0,766 | 0,618 | 0,952 | 0,997 | 0,730 | – |
| `mem` | OrEdge (bán giám sát) | 0,870 | 0,986 | 0,981 | 0,978 | 0,974 | 0,982 | 0,987 | 0,976 | – |
| `socket` | UAM-AD | 0,438 | 0,759 | 0,523 | 0,460 | 0,756 | 0,933 | 0,939 | – | 21,2 |
| `socket` | ART | 0,719 | 0,946 | 0,849 | 0,753 | 0,915 | 0,950 | 0,967 | – | – |
| `socket` | AnoFusion | 0,253 | 0,708 | 0,422 | 0,453 | – | – | – | – | – |
| `socket` | Eadro (có giám sát) | 0,615 | 0,897 | 0,773 | 0,656 | 0,622 | 0,810 | 0,995 | 0,615 | – |
| `socket` | OrEdge (bán giám sát) | 0,898 | 0,989 | 0,982 | 0,973 | 0,980 | 0,983 | 0,984 | 0,942 | – |

**RE3-OB**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAM-AD | 0,737 | 0,934 | 0,802 | 0,900 | 0,374 | 0,639 | 0,893 | – | 7,2 |
| `f1` | ART | 0,879 | 0,976 | 0,946 | 0,893 | 0,634 | 0,650 | 0,667 | – | – |
| `f1` | AnoFusion | 0,773 | 0,829 | 0,775 | 0,775 | – | – | – | – | – |
| `f1` | Eadro (có giám sát) | 0,875 | 0,965 | 0,943 | 0,884 | 0,978 | 1,000 | 1,000 | 0,869 | – |
| `f1` | OrEdge (bán giám sát) | 0,769 | 0,892 | 0,834 | 0,794 | 0,969 | 0,994 | 0,997 | 0,773 | – |
| `f2` | UAM-AD | 0,866 | 0,953 | 0,841 | 0,939 | 0,951 | 0,966 | 0,970 | – | 7,1 |
| `f2` | ART | 0,755 | 0,971 | 0,902 | 0,860 | 0,933 | 0,945 | 0,961 | – | – |
| `f2` | AnoFusion | 0,386 | 0,627 | 0,456 | 0,405 | – | – | – | – | – |
| `f2` | Eadro (có giám sát) | 0,698 | 0,908 | 0,818 | 0,770 | 1,000 | 1,000 | 1,000 | 0,761 | – |
| `f2` | OrEdge (bán giám sát) | 0,864 | 0,976 | 0,969 | 0,975 | 1,000 | 1,000 | 1,000 | 0,974 | – |
| `f3` | UAM-AD | 0,247 | 0,756 | 0,362 | 0,649 | 0,946 | 0,986 | 0,989 | – | 7,3 |
| `f3` | ART | 0,147 | 0,820 | 0,517 | 0,593 | 0,656 | 0,771 | 0,822 | – | – |
| `f3` | AnoFusion | 0,121 | 0,515 | 0,251 | 0,341 | – | – | – | – | – |
| `f3` | Eadro (có giám sát) | 0,421 | 0,592 | 0,482 | 0,456 | 0,636 | 1,000 | 1,000 | 0,374 | – |
| `f3` | OrEdge (bán giám sát) | 0,888 | 0,974 | 0,963 | 0,956 | 0,983 | 0,997 | 1,000 | 0,954 | – |
| `f4` | UAM-AD | 0,161 | 0,574 | 0,241 | 0,356 | 0,656 | 0,723 | 0,767 | – | 7,4 |
| `f4` | ART | 0,173 | 0,725 | 0,459 | 0,512 | 0,490 | 0,599 | 0,669 | – | – |
| `f4` | AnoFusion | 0,117 | 0,527 | 0,264 | 0,341 | – | – | – | – | – |
| `f4` | Eadro (có giám sát) | 0,631 | 0,921 | 0,783 | 0,666 | 0,976 | 1,000 | 1,000 | 0,633 | – |
| `f4` | OrEdge (bán giám sát) | 0,407 | 0,628 | 0,401 | 0,432 | 0,718 | 0,982 | 1,000 | 0,171 | – |
| `f5` | UAM-AD | 0,767 | 0,941 | 0,809 | 0,903 | 0,962 | 0,979 | 0,986 | – | 7,2 |
| `f5` | ART | 0,786 | 0,970 | 0,912 | 0,875 | 0,953 | 0,969 | 0,981 | – | – |
| `f5` | AnoFusion | 0,431 | 0,654 | 0,487 | 0,437 | – | – | – | – | – |
| `f5` | Eadro (có giám sát) | 0,713 | 0,824 | 0,765 | 0,715 | 0,854 | 1,000 | 1,000 | 0,696 | – |
| `f5` | OrEdge (bán giám sát) | 0,909 | 0,984 | 0,981 | 0,980 | 0,982 | 0,996 | 1,000 | 0,979 | – |

**RE2-TT**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `cpu` | UAM-AD | 0,427 | 0,677 | 0,486 | 0,428 | 0,775 | 0,924 | 0,969 | – | 31,8 |
| `cpu` | ART | 0,255 | 0,725 | 0,388 | 0,443 | 0,435 | 0,521 | 0,652 | – | – |
| `cpu` | AnoFusion | 0,608 | 0,855 | 0,698 | 0,626 | – | – | – | – | – |
| `cpu` | Eadro (có giám sát) | 0,567 | 0,796 | 0,647 | 0,598 | 0,569 | 0,747 | 0,997 | 0,596 | – |
| `cpu` | OrEdge (bán giám sát) | 0,849 | 0,987 | 0,980 | 0,966 | 0,990 | 0,996 | 0,999 | 0,956 | – |
| `delay` | UAM-AD | 0,667 | 0,879 | 0,811 | 0,767 | 0,654 | 0,841 | 0,879 | – | 29,3 |
| `delay` | ART | 0,699 | 0,844 | 0,776 | 0,770 | 0,545 | 0,676 | 0,768 | – | – |
| `delay` | AnoFusion | 0,561 | 0,652 | 0,606 | 0,612 | – | – | – | – | – |
| `delay` | Eadro (có giám sát) | 0,576 | 0,786 | 0,603 | 0,610 | 0,392 | 0,671 | 0,995 | 0,389 | – |
| `delay` | OrEdge (bán giám sát) | 0,873 | 0,990 | 0,979 | 0,944 | 0,962 | 0,999 | 1,000 | 0,921 | – |
| `disk` | UAM-AD | 0,749 | 0,918 | 0,863 | 0,856 | 0,986 | 0,990 | 0,991 | – | 29,4 |
| `disk` | ART | 0,890 | 0,992 | 0,987 | 0,989 | 0,978 | 0,986 | 0,989 | – | – |
| `disk` | AnoFusion | 0,892 | 0,988 | 0,982 | 0,974 | – | – | – | – | – |
| `disk` | Eadro (có giám sát) | 0,858 | 0,978 | 0,950 | 0,881 | 0,674 | 0,946 | 1,000 | 0,869 | – |
| `disk` | OrEdge (bán giám sát) | 0,894 | 0,991 | 0,989 | 0,988 | 0,988 | 0,994 | 0,996 | 0,987 | – |
| `loss` | UAM-AD | 0,521 | 0,737 | 0,606 | 0,529 | 0,396 | 0,635 | 0,711 | – | 32,3 |
| `loss` | ART | 0,386 | 0,667 | 0,478 | 0,394 | 0,290 | 0,373 | 0,476 | – | – |
| `loss` | AnoFusion | 0,138 | 0,418 | 0,239 | 0,343 | – | – | – | – | – |
| `loss` | Eadro (có giám sát) | 0,398 | 0,660 | 0,457 | 0,415 | 0,320 | 0,643 | 0,996 | 0,270 | – |
| `loss` | OrEdge (bán giám sát) | 0,695 | 0,860 | 0,780 | 0,715 | 0,798 | 0,921 | 0,982 | 0,603 | – |
| `mem` | UAM-AD | 0,715 | 0,943 | 0,817 | 0,724 | 0,552 | 0,677 | 0,717 | – | 29,2 |
| `mem` | ART | 0,895 | 0,995 | 0,990 | 0,988 | 0,025 | 0,030 | 0,032 | – | – |
| `mem` | AnoFusion | 0,835 | 0,954 | 0,915 | 0,892 | – | – | – | – | – |
| `mem` | Eadro (có giám sát) | 0,885 | 0,990 | 0,983 | 0,973 | 0,778 | 0,913 | 1,000 | 0,973 | – |
| `mem` | OrEdge (bán giám sát) | 0,812 | 0,990 | 0,985 | 0,965 | 0,989 | 0,994 | 0,997 | 0,963 | – |
| `socket` | UAM-AD | 0,456 | 0,703 | 0,529 | 0,458 | 0,760 | 0,939 | 0,964 | – | 32,0 |
| `socket` | ART | 0,357 | 0,764 | 0,492 | 0,468 | 0,392 | 0,479 | 0,597 | – | – |
| `socket` | AnoFusion | 0,468 | 0,774 | 0,592 | 0,564 | – | – | – | – | – |
| `socket` | Eadro (có giám sát) | 0,435 | 0,741 | 0,550 | 0,478 | 0,406 | 0,686 | 0,957 | 0,477 | – |
| `socket` | OrEdge (bán giám sát) | 0,876 | 0,992 | 0,987 | 0,963 | 0,948 | 0,968 | 0,990 | 0,962 | – |

**RE3-TT**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `f1` | UAM-AD | 0,523 | 0,720 | 0,362 | 0,642 | 0,304 | 0,368 | 0,433 | – | 12,0 |
| `f1` | ART | 0,215 | 0,659 | 0,366 | 0,411 | 0,373 | 0,457 | 0,533 | – | – |
| `f1` | AnoFusion | 0,116 | 0,502 | 0,206 | 0,351 | – | – | – | – | – |
| `f1` | Eadro (có giám sát) | 0,318 | 0,508 | 0,376 | 0,336 | 0,521 | 1,000 | 1,000 | 0,081 | – |
| `f1` | OrEdge (bán giám sát) | 0,692 | 0,806 | 0,753 | 0,735 | 0,958 | 0,994 | 0,996 | 0,716 | – |
| `f2` | UAM-AD | 0,508 | 0,690 | 0,345 | 0,624 | 0,275 | 0,347 | 0,399 | – | 12,0 |
| `f2` | ART | 0,205 | 0,649 | 0,356 | 0,409 | 0,331 | 0,429 | 0,515 | – | – |
| `f2` | Eadro (có giám sát) | 0,113 | 0,497 | 0,283 | 0,337 | 0,528 | 1,000 | 1,000 | 0,099 | – |
| `f2` | OrEdge (bán giám sát) | 0,665 | 0,795 | 0,739 | 0,719 | 0,929 | 0,998 | 0,999 | 0,709 | – |
| `f3` | UAM-AD | 0,520 | 0,723 | 0,356 | 0,637 | 0,126 | 0,203 | 0,337 | – | 12,1 |
| `f3` | ART | 0,633 | 0,776 | 0,698 | 0,688 | 0,075 | 0,100 | 0,164 | – | – |
| `f3` | Eadro (có giám sát) | 0,641 | 0,761 | 0,664 | 0,669 | 0,918 | 1,000 | 1,000 | 0,657 | – |
| `f3` | OrEdge (bán giám sát) | 0,453 | 0,662 | 0,522 | 0,462 | 0,778 | 0,994 | 0,999 | 0,346 | – |
| `f4` | UAM-AD | 0,510 | 0,686 | 0,337 | 0,620 | 0,272 | 0,355 | 0,398 | – | 12,0 |
| `f4` | ART | 0,217 | 0,659 | 0,367 | 0,405 | 0,387 | 0,459 | 0,536 | – | – |
| `f4` | Eadro (có giám sát) | 0,329 | 0,528 | 0,385 | 0,367 | 0,521 | 0,998 | 0,998 | 0,072 | – |
| `f4` | OrEdge (bán giám sát) | 0,688 | 0,799 | 0,744 | 0,728 | 0,932 | 0,992 | 0,995 | 0,689 | – |

**SN**

| Scenario | Mô hình | F1@val95 | AUROC | AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `Code_Stop_MediaService` | UAM-AD | 0,750 | 0,965 | 0,822 | 0,897 | 0,718 | 0,846 | 0,872 | – | 0,3 |
| `Code_Stop_MediaService` | ART | 0,750 | 0,962 | 0,812 | 0,763 | 0,000 | 0,128 | 0,359 | – | – |
| `Code_Stop_MediaService` | AnoFusion | 0,427 | 0,730 | 0,557 | 0,582 | – | – | – | – | – |
| `Code_Stop_MediaService` | Eadro (có giám sát) | 0,651 | 0,963 | 0,712 | 0,804 | 0,000 | 0,000 | 0,000 | 0,713 | – |
| `Code_Stop_MediaService` | OrEdge (bán giám sát) | 0,816 | 0,962 | 0,841 | 0,849 | 0,897 | 0,923 | 0,923 | 0,000 | – |
| `Code_Stop_TextService` | UAM-AD | 0,703 | 0,975 | 0,900 | 0,929 | 0,667 | 0,744 | 0,923 | – | 0,3 |
| `Code_Stop_TextService` | ART | 0,788 | 0,985 | 0,920 | 0,864 | 0,308 | 0,359 | 0,359 | – | – |
| `Code_Stop_TextService` | AnoFusion | 0,595 | 0,777 | 0,723 | 0,781 | – | – | – | – | – |
| `Code_Stop_TextService` | Eadro (có giám sát) | 0,857 | 0,995 | 0,969 | 0,949 | 0,000 | 0,000 | 0,000 | 0,780 | – |
| `Code_Stop_TextService` | OrEdge (bán giám sát) | 0,033 | 0,515 | 0,158 | 0,275 | 0,179 | 0,821 | 0,974 | 0,000 | – |
| `Code_Stop_UserService` | UAM-AD | 0,745 | 0,982 | 0,945 | 0,894 | 0,079 | 0,632 | 0,684 | – | 0,3 |
| `Code_Stop_UserService` | ART | 0,710 | 0,974 | 0,917 | 0,886 | 0,026 | 0,026 | 0,128 | – | – |
| `Code_Stop_UserService` | AnoFusion | 0,689 | 0,911 | 0,861 | 0,886 | – | – | – | – | – |
| `Code_Stop_UserService` | Eadro (có giám sát) | 0,791 | 0,971 | 0,770 | 0,805 | 0,000 | 0,000 | 0,000 | 0,792 | – |
| `Code_Stop_UserService` | OrEdge (bán giám sát) | 0,077 | 0,734 | 0,505 | 0,684 | 0,641 | 0,718 | 0,718 | 0,000 | – |
| `DB_Redis_CacheLimit_HomeTimeline` | UAM-AD | 0,833 | 0,965 | 0,727 | 0,833 | 0,300 | 0,400 | 0,400 | – | 0,3 |
| `DB_Redis_CacheLimit_HomeTimeline` | ART | 0,444 | 0,886 | 0,509 | 0,636 | 0,000 | 0,300 | 0,500 | – | – |
| `DB_Redis_CacheLimit_HomeTimeline` | AnoFusion | 0,667 | 0,870 | 0,459 | 0,696 | – | – | – | – | – |
| `DB_Redis_CacheLimit_HomeTimeline` | Eadro (có giám sát) | 0,783 | 0,977 | 0,843 | 0,833 | 0,000 | 0,000 | 0,000 | 0,783 | – |
| `DB_Redis_CacheLimit_HomeTimeline` | OrEdge (bán giám sát) | 0,500 | 0,793 | 0,462 | 0,609 | 0,100 | 0,100 | 0,200 | 0,000 | – |
| `DB_Redis_CacheLimit_SocialGraph` | UAM-AD | 0,857 | 0,985 | 0,832 | 0,900 | 0,000 | 0,000 | 0,000 | – | 0,3 |
| `DB_Redis_CacheLimit_SocialGraph` | ART | 0,571 | 0,817 | 0,389 | 0,583 | 0,000 | 0,000 | 0,000 | – | – |
| `DB_Redis_CacheLimit_SocialGraph` | AnoFusion | 0,636 | 0,869 | 0,789 | 0,824 | – | – | – | – | – |
| `DB_Redis_CacheLimit_SocialGraph` | Eadro (có giám sát) | 0,870 | 0,974 | 0,764 | 0,870 | 0,000 | 0,100 | 0,700 | 0,870 | – |
| `DB_Redis_CacheLimit_SocialGraph` | OrEdge (bán giám sát) | 0,909 | 0,973 | 0,696 | 0,909 | 0,100 | 0,100 | 0,300 | 0,154 | – |
| `DB_Redis_CacheLimit_UserTimeline` | UAM-AD | 0,556 | 0,942 | 0,583 | 0,706 | 0,000 | 0,000 | 0,000 | – | 0,3 |
| `DB_Redis_CacheLimit_UserTimeline` | ART | 0,333 | 0,839 | 0,396 | 0,483 | 0,100 | 0,200 | 0,300 | – | – |
| `DB_Redis_CacheLimit_UserTimeline` | AnoFusion | 0,000 | 0,527 | 0,138 | 0,265 | – | – | – | – | – |
| `DB_Redis_CacheLimit_UserTimeline` | Eadro (có giám sát) | 0,857 | 0,984 | 0,897 | 0,857 | 0,000 | 0,100 | 0,800 | 0,800 | – |
| `DB_Redis_CacheLimit_UserTimeline` | OrEdge (bán giám sát) | 0,333 | 0,856 | 0,481 | 0,538 | 0,500 | 0,800 | 1,000 | 0,000 | – |
| `Perf_CPU_Contention` | UAM-AD | 0,769 | 0,960 | 0,724 | 0,833 | – | – | – | – | 0,3 |
| `Perf_CPU_Contention` | ART | 0,667 | 0,961 | 0,788 | 0,778 | – | – | – | – | – |
| `Perf_CPU_Contention` | AnoFusion | 0,667 | 0,863 | 0,418 | 0,667 | – | – | – | – | – |
| `Perf_CPU_Contention` | Eadro (có giám sát) | 0,783 | 0,980 | 0,873 | 0,833 | – | – | – | 0,783 | – |
| `Perf_CPU_Contention` | OrEdge (bán giám sát) | 0,154 | 0,711 | 0,373 | 0,435 | – | – | – | 0,000 | – |
| `Perf_Disk_IO_Stress` | UAM-AD | 0,640 | 0,916 | 0,444 | 0,727 | – | – | – | – | 0,4 |
| `Perf_Disk_IO_Stress` | ART | 0,286 | 0,726 | 0,299 | 0,417 | – | – | – | – | – |
| `Perf_Disk_IO_Stress` | AnoFusion | 0,545 | 0,806 | 0,354 | 0,571 | – | – | – | – | – |
| `Perf_Disk_IO_Stress` | Eadro (có giám sát) | 0,741 | 0,949 | 0,674 | 0,750 | – | – | – | 0,741 | – |
| `Perf_Disk_IO_Stress` | OrEdge (bán giám sát) | 0,400 | 0,886 | 0,458 | 0,552 | – | – | – | 0,000 | – |
| `Perf_Network_Loss` | UAM-AD | 0,909 | 0,985 | 0,798 | 0,952 | – | – | – | – | 0,4 |
| `Perf_Network_Loss` | ART | 0,471 | 0,831 | 0,429 | 0,483 | – | – | – | – | – |
| `Perf_Network_Loss` | AnoFusion | 0,667 | 0,889 | 0,624 | 0,667 | – | – | – | – | – |
| `Perf_Network_Loss` | Eadro (có giám sát) | 0,909 | 0,984 | 0,872 | 0,909 | – | – | – | 0,909 | – |
| `Perf_Network_Loss` | OrEdge (bán giám sát) | 0,588 | 0,949 | 0,679 | 0,692 | – | – | – | 0,000 | – |
| `Svc_Kill_Media` | UAM-AD | 0,750 | 0,962 | 0,679 | 0,889 | 0,000 | 0,750 | 0,750 | – | 0,3 |
| `Svc_Kill_Media` | ART | 0,500 | 0,795 | 0,417 | 0,571 | 0,000 | 0,500 | 1,000 | – | – |
| `Svc_Kill_Media` | AnoFusion | 0,000 | 0,509 | 0,195 | 0,286 | – | – | – | – | – |
| `Svc_Kill_Media` | Eadro (có giám sát) | 0,800 | 1,000 | 1,000 | 1,000 | 0,000 | 0,000 | 1,000 | 0,800 | – |
| `Svc_Kill_Media` | OrEdge (bán giám sát) | 0,333 | 0,866 | 0,531 | 0,545 | 0,000 | 0,000 | 0,250 | 0,000 | – |
| `Svc_Kill_SocialGraph` | UAM-AD | 0,545 | 0,889 | 0,387 | 0,750 | 0,000 | 0,000 | 0,000 | – | 0,3 |
| `Svc_Kill_SocialGraph` | ART | 0,571 | 0,795 | 0,287 | 0,571 | 0,000 | 0,000 | 0,000 | – | – |
| `Svc_Kill_SocialGraph` | AnoFusion | 0,000 | 0,420 | 0,129 | 0,273 | – | – | – | – | – |
| `Svc_Kill_SocialGraph` | Eadro (có giám sát) | 0,889 | 0,991 | 0,950 | 0,889 | 0,000 | 1,000 | 1,000 | 0,889 | – |
| `Svc_Kill_SocialGraph` | OrEdge (bán giám sát) | 0,000 | 0,866 | 0,381 | 0,615 | 0,000 | 0,000 | 1,000 | 0,000 | – |
| `Svc_Kill_UserTimeline` | UAM-AD | 0,750 | 0,942 | 0,622 | 0,889 | 0,000 | 0,000 | 0,000 | – | 0,3 |
| `Svc_Kill_UserTimeline` | ART | 0,286 | 0,866 | 0,388 | 0,615 | 0,000 | 0,000 | 0,000 | – | – |
| `Svc_Kill_UserTimeline` | AnoFusion | 0,000 | 0,607 | 0,214 | 0,400 | – | – | – | – | – |
| `Svc_Kill_UserTimeline` | Eadro (có giám sát) | 0,800 | 0,982 | 0,887 | 0,889 | 0,250 | 0,750 | 1,000 | 0,800 | – |
| `Svc_Kill_UserTimeline` | OrEdge (bán giám sát) | 0,000 | 0,643 | 0,197 | 0,400 | 0,000 | 1,000 | 1,000 | 0,000 | – |

**Kết quả seed 1 của ART, AnoFusion, Eadro và OrEdge (chi tiết từng seed; sinh tự động bởi `summary_tools/item9_tables.py`)**

Cách đọc: trung bình qua scenario của từng dataset; mọi mô hình trong bảng dùng đủ ba modality L+M+T. UAM-AD lấy cùng seed từ bước 8 (`norm_sum`). **F1 oracle của UAM-AD ở SN, RE3-OB, RE3-TT là quét ngưỡng CÓ point-adjust (`run.py`); của ART, AnoFusion, Eadro, OrEdge là quét ngưỡng thường, KHÔNG point-adjust, nên ở ba dataset đó cột này không so được. Ở RE2-OB, RE2-TT, F1 oracle của UAM-AD (và UAC-AD) là bản không point-adjust của việc 10, cùng định nghĩa với đối thủ.** HR@k của UAM-AD (việc 10: điểm trace chuẩn hoá cộng điểm KPI của service) và của ART, Eadro, OrEdge đều tính trên mọi mẫu bất thường có service lỗi; đơn vị chấm vẫn khác (UAM-AD cửa sổ 30 mẫu, các đối thủ từng mẫu giây hoặc cửa sổ 10), nên chưa so thẳng được. "F1 argmax" chỉ có ở Eadro, OrEdge (quyết định của bộ phân loại). Cột cuối là phút (UAM-AD: tổng thời gian log của các scenario; đối thủ: tổng thời gian chạy ô).

Cột **Δ** (chỉ cho F1@val95, AUROC, AUPRC) = UAM-AD trừ mô hình ở dòng đó; trong ngoặc là chênh lệch tính theo % giá trị của mô hình ở dòng đó (dương = UAM-AD cao hơn). Chỉ tính trên các scenario cả hai cùng có. Dấu * = mô hình dừng sớm (AnoFusion RE3-TT chỉ có `f1`): so trên scenario đã chấm và không vào dòng Trung bình. Dòng **Trung bình** của một mô hình = trung bình các dataset mô hình đó có đủ; % = chênh lệch trung bình chia cho giá trị trung bình của mô hình đó (không phải trung bình của các %, vì nó bị dataset có nền thấp kéo lệch dấu).

*Theo dataset:*

| Dataset | Mô hình | F1@val95 | Δ F1 | AUROC | Δ AUROC | AUPRC | Δ AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | UAM-AD | 0,733 | – | 0,930 | – | 0,840 | – | 0,769 | 0,789 | 0,962 | 0,983 | – | 142,6 |
| RE2-OB | ART | 0,881 | −0,147 (−16,7%) | 0,984 | −0,054 (−5,5%) | 0,964 | −0,125 (−12,9%) | 0,931 | 0,749 | 0,847 | 0,933 | – | 6,7 |
| RE2-OB | Eadro (có giám sát) | 0,729 | +0,005 (+0,7%) | 0,866 | +0,064 (+7,4%) | 0,800 | +0,039 (+4,9%) | 0,769 | 0,638 | 0,884 | 0,994 | 0,700 | 12,7 |
| RE2-OB | OrEdge (bán giám sát) | 0,863 | −0,130 (−15,0%) | 0,971 | −0,041 (−4,2%) | 0,958 | −0,119 (−12,4%) | 0,949 | 0,925 | 0,943 | 0,947 | 0,888 | 30,1 |
| RE3-OB | UAM-AD | 0,577 | – | 0,841 | – | 0,628 | – | 0,763 | 0,839 | 0,914 | 0,973 | – | 40,1 |
| RE3-OB | ART | 0,429 | +0,148 (+34,6%) | 0,885 | −0,045 (−5,1%) | 0,722 | −0,093 (−12,9%) | 0,734 | 0,718 | 0,778 | 0,821 | – | 2,3 |
| RE3-OB | Eadro (có giám sát) | 0,684 | −0,107 (−15,7%) | 0,819 | +0,022 (+2,6%) | 0,744 | −0,116 (−15,6%) | 0,705 | 0,837 | 1,000 | 1,000 | 0,644 | 4,6 |
| RE3-OB | OrEdge (bán giám sát) | 0,776 | −0,199 (−25,6%) | 0,899 | −0,058 (−6,5%) | 0,841 | −0,212 (−25,3%) | 0,821 | 0,933 | 0,991 | 0,997 | 0,761 | 10,7 |
| RE2-TT | UAM-AD | 0,607 | – | 0,809 | – | 0,680 | – | 0,630 | 0,701 | 0,841 | 0,867 | – | 194,5 |
| RE2-TT | ART | 0,581 | +0,026 (+4,5%) | 0,831 | −0,022 (−2,6%) | 0,685 | −0,005 (−0,7%) | 0,675 | 0,450 | 0,514 | 0,595 | – | 7,6 |
| RE2-TT | Eadro (có giám sát) | 0,627 | −0,019 (−3,1%) | 0,818 | −0,009 (−1,1%) | 0,710 | −0,030 (−4,2%) | 0,684 | 0,526 | 0,787 | 0,998 | 0,611 | 33,9 |
| RE2-TT | OrEdge (bán giám sát) | 0,834 | −0,226 (−27,1%) | 0,965 | −0,156 (−16,1%) | 0,949 | −0,269 (−28,3%) | 0,928 | 0,961 | 0,984 | 0,994 | 0,903 | 31,4 |
| RE3-TT | UAM-AD | 0,513 | – | 0,703 | – | 0,347 | – | 0,635 | 0,248 | 0,322 | 0,428 | – | 53,2 |
| RE3-TT | ART | 0,316 | +0,196 (+62,0%) | 0,679 | +0,025 (+3,6%) | 0,442 | −0,096 (−21,6%) | 0,471 | 0,274 | 0,389 | 0,469 | – | 2,6 |
| RE3-TT | Eadro (có giám sát) | 0,392 | +0,121 (+30,8%) | 0,600 | +0,103 (+17,1%) | 0,450 | −0,103 (−22,9%) | 0,439 | 0,618 | 0,999 | 0,999 | 0,220 | 11,6 |
| RE3-TT | OrEdge (bán giám sát) | 0,639 | −0,127 (−19,8%) | 0,761 | −0,057 (−7,5%) | 0,689 | −0,342 (−49,6%) | 0,667 | 0,900 | 0,996 | 0,997 | 0,609 | 33,6 |
| SN | UAM-AD | 0,724 | – | 0,954 | – | 0,705 | – | 0,857 | 0,199 | 0,409 | 0,415 | – | 4,6 |
| SN | ART | 0,611 | +0,113 (+18,4%) | 0,922 | +0,033 (+3,6%) | 0,633 | +0,072 (+11,3%) | 0,709 | 0,070 | 0,199 | 0,408 | – | 0,4 |
| SN | Eadro (có giám sát) | 0,796 | −0,072 (−9,1%) | 0,976 | −0,022 (−2,3%) | 0,849 | −0,143 (−16,9%) | 0,879 | 0,000 | 0,256 | 0,397 | 0,772 | 1,1 |
| SN | OrEdge (bán giám sát) | 0,266 | +0,458 (+172,2%) | 0,786 | +0,168 (+21,4%) | 0,473 | +0,233 (+49,2%) | 0,589 | 0,139 | 0,493 | 0,679 | 0,103 | 1,9 |
| **Trung bình** | UAM-AD | 0,631 | – | 0,848 | – | 0,640 | – | – | – | – | – | – | – |
| **Trung bình** | ART | 0,564 | +0,067 (+11,9%) | 0,860 | −0,013 (−1,5%) | 0,689 | −0,049 (−7,2%) | – | – | – | – | – | – |
| **Trung bình** | Eadro (có giám sát) | 0,645 | −0,015 (−2,3%) | 0,816 | +0,031 (+3,8%) | 0,711 | −0,071 (−9,9%) | – | – | – | – | – | – |
| **Trung bình** | OrEdge (bán giám sát) | 0,676 | −0,045 (−6,6%) | 0,876 | −0,029 (−3,3%) | 0,782 | −0,142 (−18,1%) | – | – | – | – | – | – |
**Kết quả seed 2 của ART, AnoFusion, Eadro và OrEdge (chi tiết từng seed; sinh tự động bởi `summary_tools/item9_tables.py`)**

Cách đọc: trung bình qua scenario của từng dataset; mọi mô hình trong bảng dùng đủ ba modality L+M+T. UAM-AD lấy cùng seed từ bước 8 (`norm_sum`). **F1 oracle của UAM-AD ở SN, RE3-OB, RE3-TT là quét ngưỡng CÓ point-adjust (`run.py`); của ART, AnoFusion, Eadro, OrEdge là quét ngưỡng thường, KHÔNG point-adjust, nên ở ba dataset đó cột này không so được. Ở RE2-OB, RE2-TT, F1 oracle của UAM-AD (và UAC-AD) là bản không point-adjust của việc 10, cùng định nghĩa với đối thủ.** HR@k của UAM-AD (việc 10: điểm trace chuẩn hoá cộng điểm KPI của service) và của ART, Eadro, OrEdge đều tính trên mọi mẫu bất thường có service lỗi; đơn vị chấm vẫn khác (UAM-AD cửa sổ 30 mẫu, các đối thủ từng mẫu giây hoặc cửa sổ 10), nên chưa so thẳng được. "F1 argmax" chỉ có ở Eadro, OrEdge (quyết định của bộ phân loại). Cột cuối là phút (UAM-AD: tổng thời gian log của các scenario; đối thủ: tổng thời gian chạy ô).

Cột **Δ** (chỉ cho F1@val95, AUROC, AUPRC) = UAM-AD trừ mô hình ở dòng đó; trong ngoặc là chênh lệch tính theo % giá trị của mô hình ở dòng đó (dương = UAM-AD cao hơn). Chỉ tính trên các scenario cả hai cùng có. Dấu * = mô hình dừng sớm (AnoFusion RE3-TT chỉ có `f1`): so trên scenario đã chấm và không vào dòng Trung bình. Dòng **Trung bình** của một mô hình = trung bình các dataset mô hình đó có đủ; % = chênh lệch trung bình chia cho giá trị trung bình của mô hình đó (không phải trung bình của các %, vì nó bị dataset có nền thấp kéo lệch dấu).

*Theo dataset:*

| Dataset | Mô hình | F1@val95 | Δ F1 | AUROC | Δ AUROC | AUPRC | Δ AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | UAM-AD | 0,660 | – | 0,909 | – | 0,813 | – | 0,752 | 0,782 | 0,951 | 0,973 | – | 141,1 |
| RE2-OB | ART | 0,881 | −0,221 (−25,1%) | 0,981 | −0,072 (−7,3%) | 0,956 | −0,142 (−14,9%) | 0,921 | 0,752 | 0,845 | 0,932 | – | 6,2 |
| RE2-OB | Eadro (có giám sát) | 0,731 | −0,071 (−9,7%) | 0,877 | +0,032 (+3,6%) | 0,805 | +0,009 (+1,1%) | 0,768 | 0,669 | 0,878 | 0,998 | 0,698 | 12,6 |
| RE2-OB | OrEdge (bán giám sát) | 0,873 | −0,213 (−24,4%) | 0,975 | −0,066 (−6,8%) | 0,961 | −0,148 (−15,4%) | 0,947 | 0,929 | 0,947 | 0,954 | 0,892 | 29,7 |
| RE3-OB | UAM-AD | 0,576 | – | 0,839 | – | 0,626 | – | 0,763 | 0,844 | 0,917 | 0,970 | – | 40,3 |
| RE3-OB | ART | 0,484 | +0,092 (+18,9%) | 0,890 | −0,051 (−5,7%) | 0,739 | −0,113 (−15,3%) | 0,746 | 0,721 | 0,782 | 0,826 | – | 2,1 |
| RE3-OB | Eadro (có giám sát) | 0,621 | −0,045 (−7,2%) | 0,834 | +0,005 (+0,6%) | 0,731 | −0,105 (−14,3%) | 0,690 | 0,849 | 1,000 | 1,000 | 0,664 | 4,5 |
| RE3-OB | OrEdge (bán giám sát) | 0,764 | −0,188 (−24,6%) | 0,889 | −0,050 (−5,6%) | 0,826 | −0,200 (−24,2%) | 0,802 | 0,952 | 0,995 | 0,999 | 0,744 | 10,9 |
| RE2-TT | UAM-AD | 0,602 | – | 0,811 | – | 0,688 | – | 0,626 | 0,712 | 0,877 | 0,922 | – | 200,9 |
| RE2-TT | ART | 0,587 | +0,015 (+2,6%) | 0,835 | −0,025 (−2,9%) | 0,692 | −0,004 (−0,5%) | 0,677 | 0,451 | 0,513 | 0,598 | – | 7,4 |
| RE2-TT | Eadro (có giám sát) | 0,664 | −0,062 (−9,3%) | 0,851 | −0,040 (−4,7%) | 0,751 | −0,062 (−8,3%) | 0,709 | 0,537 | 0,781 | 0,991 | 0,629 | 33,8 |
| RE2-TT | OrEdge (bán giám sát) | 0,830 | −0,228 (−27,4%) | 0,973 | −0,162 (−16,7%) | 0,952 | −0,264 (−27,7%) | 0,925 | 0,948 | 0,978 | 0,993 | 0,910 | 31,2 |
| RE3-TT | UAM-AD | 0,510 | – | 0,697 | – | 0,341 | – | 0,627 | 0,247 | 0,317 | 0,395 | – | 54,2 |
| RE3-TT | ART | 0,314 | +0,195 (+62,2%) | 0,666 | +0,031 (+4,6%) | 0,433 | −0,093 (−21,4%) | 0,463 | 0,358 | 0,434 | 0,488 | – | 2,3 |
| RE3-TT | Eadro (có giám sát) | 0,333 | +0,177 (+53,2%) | 0,607 | +0,089 (+14,7%) | 0,427 | −0,086 (−20,2%) | 0,442 | 0,623 | 1,000 | 1,000 | 0,228 | 11,6 |
| RE3-TT | OrEdge (bán giám sát) | 0,625 | −0,116 (−18,5%) | 0,758 | −0,061 (−8,1%) | 0,685 | −0,345 (−50,3%) | 0,666 | 0,927 | 0,996 | 0,997 | 0,623 | 33,5 |
| SN | UAM-AD | 0,726 | – | 0,959 | – | 0,715 | – | 0,831 | 0,202 | 0,384 | 0,393 | – | 5,1 |
| SN | ART | 0,509 | +0,217 (+42,7%) | 0,884 | +0,075 (+8,4%) | 0,565 | +0,151 (+26,7%) | 0,648 | 0,045 | 0,143 | 0,394 | – | 0,4 |
| SN | Eadro (có giám sát) | 0,819 | −0,093 (−11,3%) | 0,980 | −0,021 (−2,2%) | 0,841 | −0,126 (−15,0%) | 0,884 | 0,122 | 0,278 | 0,543 | 0,818 | 1,1 |
| SN | OrEdge (bán giám sát) | 0,515 | +0,211 (+41,0%) | 0,878 | +0,081 (+9,2%) | 0,593 | +0,122 (+20,6%) | 0,697 | 0,334 | 0,524 | 0,794 | 0,015 | 2,2 |
| **Trung bình** | UAM-AD | 0,615 | – | 0,843 | – | 0,637 | – | – | – | – | – | – | – |
| **Trung bình** | ART | 0,555 | +0,060 (+10,8%) | 0,851 | −0,008 (−1,0%) | 0,677 | −0,040 (−5,9%) | – | – | – | – | – | – |
| **Trung bình** | Eadro (có giám sát) | 0,634 | −0,019 (−3,0%) | 0,830 | +0,013 (+1,6%) | 0,711 | −0,074 (−10,4%) | – | – | – | – | – | – |
| **Trung bình** | OrEdge (bán giám sát) | 0,721 | −0,107 (−14,8%) | 0,895 | −0,052 (−5,8%) | 0,804 | −0,167 (−20,8%) | – | – | – | – | – | – |
**Kết quả seed 3 của ART, AnoFusion, Eadro và OrEdge (chi tiết từng seed; sinh tự động bởi `summary_tools/item9_tables.py`)**

Cách đọc: trung bình qua scenario của từng dataset; mọi mô hình trong bảng dùng đủ ba modality L+M+T. UAM-AD lấy cùng seed từ bước 8 (`norm_sum`). **F1 oracle của UAM-AD ở SN, RE3-OB, RE3-TT là quét ngưỡng CÓ point-adjust (`run.py`); của ART, AnoFusion, Eadro, OrEdge là quét ngưỡng thường, KHÔNG point-adjust, nên ở ba dataset đó cột này không so được. Ở RE2-OB, RE2-TT, F1 oracle của UAM-AD (và UAC-AD) là bản không point-adjust của việc 10, cùng định nghĩa với đối thủ.** HR@k của UAM-AD (việc 10: điểm trace chuẩn hoá cộng điểm KPI của service) và của ART, Eadro, OrEdge đều tính trên mọi mẫu bất thường có service lỗi; đơn vị chấm vẫn khác (UAM-AD cửa sổ 30 mẫu, các đối thủ từng mẫu giây hoặc cửa sổ 10), nên chưa so thẳng được. "F1 argmax" chỉ có ở Eadro, OrEdge (quyết định của bộ phân loại). Cột cuối là phút (UAM-AD: tổng thời gian log của các scenario; đối thủ: tổng thời gian chạy ô).

Cột **Δ** (chỉ cho F1@val95, AUROC, AUPRC) = UAM-AD trừ mô hình ở dòng đó; trong ngoặc là chênh lệch tính theo % giá trị của mô hình ở dòng đó (dương = UAM-AD cao hơn). Chỉ tính trên các scenario cả hai cùng có. Dấu * = mô hình dừng sớm (AnoFusion RE3-TT chỉ có `f1`): so trên scenario đã chấm và không vào dòng Trung bình. Dòng **Trung bình** của một mô hình = trung bình các dataset mô hình đó có đủ; % = chênh lệch trung bình chia cho giá trị trung bình của mô hình đó (không phải trung bình của các %, vì nó bị dataset có nền thấp kéo lệch dấu).

*Theo dataset:*

| Dataset | Mô hình | F1@val95 | Δ F1 | AUROC | Δ AUROC | AUPRC | Δ AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | UAM-AD | 0,755 | – | 0,934 | – | 0,854 | – | 0,799 | 0,741 | 0,946 | 0,978 | – | 130,1 |
| RE2-OB | ART | 0,877 | −0,123 (−14,0%) | 0,978 | −0,044 (−4,5%) | 0,951 | −0,097 (−10,2%) | 0,919 | 0,757 | 0,849 | 0,933 | – | 6,5 |
| RE2-OB | Eadro (có giám sát) | 0,690 | +0,064 (+9,3%) | 0,864 | +0,070 (+8,1%) | 0,784 | +0,070 (+8,9%) | 0,741 | 0,680 | 0,878 | 0,997 | 0,703 | 12,6 |
| RE2-OB | OrEdge (bán giám sát) | 0,880 | −0,125 (−14,2%) | 0,976 | −0,042 (−4,3%) | 0,966 | −0,112 (−11,6%) | 0,954 | 0,927 | 0,947 | 0,952 | 0,915 | 29,8 |
| RE3-OB | UAM-AD | 0,566 | – | 0,834 | – | 0,614 | – | 0,748 | 0,826 | 0,909 | 0,968 | – | 37,3 |
| RE3-OB | ART | 0,562 | +0,004 (+0,7%) | 0,899 | −0,065 (−7,2%) | 0,760 | −0,146 (−19,2%) | 0,779 | 0,731 | 0,797 | 0,835 | – | 2,2 |
| RE3-OB | Eadro (có giám sát) | 0,627 | −0,061 (−9,7%) | 0,811 | +0,023 (+2,9%) | 0,706 | −0,092 (−13,0%) | 0,652 | 0,864 | 1,000 | 1,000 | 0,626 | 4,4 |
| RE3-OB | OrEdge (bán giám sát) | 0,753 | −0,187 (−24,8%) | 0,893 | −0,059 (−6,6%) | 0,836 | −0,221 (−26,5%) | 0,810 | 0,938 | 0,995 | 0,999 | 0,766 | 10,8 |
| RE2-TT | UAM-AD | 0,592 | – | 0,804 | – | 0,674 | – | 0,614 | 0,748 | 0,895 | 0,934 | – | 181,4 |
| RE2-TT | ART | 0,583 | +0,009 (+1,6%) | 0,835 | −0,031 (−3,7%) | 0,688 | −0,015 (−2,1%) | 0,677 | 0,443 | 0,503 | 0,580 | – | 7,6 |
| RE2-TT | Eadro (có giám sát) | 0,666 | −0,074 (−11,1%) | 0,856 | −0,051 (−6,0%) | 0,752 | −0,078 (−10,4%) | 0,717 | 0,545 | 0,788 | 0,993 | 0,642 | 33,8 |
| RE2-TT | OrEdge (bán giám sát) | 0,831 | −0,238 (−28,7%) | 0,972 | −0,168 (−17,3%) | 0,951 | −0,277 (−29,1%) | 0,927 | 0,963 | 0,988 | 0,995 | 0,905 | 31,2 |
| RE3-TT | UAM-AD | 0,516 | – | 0,705 | – | 0,349 | – | 0,635 | 0,253 | 0,360 | 0,453 | – | 49,3 |
| RE3-TT | ART | 0,317 | +0,199 (+62,7%) | 0,681 | +0,024 (+3,5%) | 0,444 | −0,095 (−21,4%) | 0,470 | 0,414 | 0,439 | 0,493 | – | 2,6 |
| RE3-TT | Eadro (có giám sát) | 0,405 | +0,111 (+27,5%) | 0,611 | +0,094 (+15,4%) | 0,471 | −0,122 (−25,9%) | 0,454 | 0,630 | 1,000 | 1,000 | 0,231 | 11,6 |
| RE3-TT | OrEdge (bán giám sát) | 0,637 | −0,121 (−19,0%) | 0,770 | −0,065 (−8,4%) | 0,694 | −0,345 (−49,7%) | 0,668 | 0,898 | 0,994 | 0,998 | 0,613 | 33,5 |
| SN | UAM-AD | 0,725 | – | 0,956 | – | 0,717 | – | 0,855 | 0,163 | 0,260 | 0,334 | – | 4,0 |
| SN | ART | 0,553 | +0,172 (+31,1%) | 0,900 | +0,056 (+6,2%) | 0,638 | +0,079 (+12,3%) | 0,685 | 0,039 | 0,168 | 0,288 | – | 0,3 |
| SN | Eadro (có giám sát) | 0,836 | −0,111 (−13,3%) | 0,983 | −0,027 (−2,8%) | 0,866 | −0,149 (−17,2%) | 0,893 | 0,089 | 0,289 | 0,511 | 0,830 | 1,1 |
| SN | OrEdge (bán giám sát) | 0,396 | +0,329 (+83,1%) | 0,853 | +0,103 (+12,0%) | 0,558 | +0,159 (+28,5%) | 0,672 | 0,219 | 0,510 | 0,705 | 0,000 | 1,9 |
| **Trung bình** | UAM-AD | 0,631 | – | 0,847 | – | 0,642 | – | – | – | – | – | – | – |
| **Trung bình** | ART | 0,579 | +0,052 (+9,0%) | 0,859 | −0,012 (−1,4%) | 0,696 | −0,055 (−7,9%) | – | – | – | – | – | – |
| **Trung bình** | Eadro (có giám sát) | 0,645 | −0,014 (−2,2%) | 0,825 | +0,022 (+2,6%) | 0,716 | −0,074 (−10,4%) | – | – | – | – | – | – |
| **Trung bình** | OrEdge (bán giám sát) | 0,699 | −0,068 (−9,8%) | 0,893 | −0,046 (−5,2%) | 0,801 | −0,159 (−19,9%) | – | – | – | – | – | – |
**Kết quả seed 4 của ART, AnoFusion, Eadro và OrEdge (chi tiết từng seed; sinh tự động bởi `summary_tools/item9_tables.py`)**

Cách đọc: trung bình qua scenario của từng dataset; mọi mô hình trong bảng dùng đủ ba modality L+M+T. UAM-AD lấy cùng seed từ bước 8 (`norm_sum`). **F1 oracle của UAM-AD ở SN, RE3-OB, RE3-TT là quét ngưỡng CÓ point-adjust (`run.py`); của ART, AnoFusion, Eadro, OrEdge là quét ngưỡng thường, KHÔNG point-adjust, nên ở ba dataset đó cột này không so được. Ở RE2-OB, RE2-TT, F1 oracle của UAM-AD (và UAC-AD) là bản không point-adjust của việc 10, cùng định nghĩa với đối thủ.** HR@k của UAM-AD (việc 10: điểm trace chuẩn hoá cộng điểm KPI của service) và của ART, Eadro, OrEdge đều tính trên mọi mẫu bất thường có service lỗi; đơn vị chấm vẫn khác (UAM-AD cửa sổ 30 mẫu, các đối thủ từng mẫu giây hoặc cửa sổ 10), nên chưa so thẳng được. "F1 argmax" chỉ có ở Eadro, OrEdge (quyết định của bộ phân loại). Cột cuối là phút (UAM-AD: tổng thời gian log của các scenario; đối thủ: tổng thời gian chạy ô).

Cột **Δ** (chỉ cho F1@val95, AUROC, AUPRC) = UAM-AD trừ mô hình ở dòng đó; trong ngoặc là chênh lệch tính theo % giá trị của mô hình ở dòng đó (dương = UAM-AD cao hơn). Chỉ tính trên các scenario cả hai cùng có. Dấu * = mô hình dừng sớm (AnoFusion RE3-TT chỉ có `f1`): so trên scenario đã chấm và không vào dòng Trung bình. Dòng **Trung bình** của một mô hình = trung bình các dataset mô hình đó có đủ; % = chênh lệch trung bình chia cho giá trị trung bình của mô hình đó (không phải trung bình của các %, vì nó bị dataset có nền thấp kéo lệch dấu).

*Theo dataset:*

| Dataset | Mô hình | F1@val95 | Δ F1 | AUROC | Δ AUROC | AUPRC | Δ AUPRC | F1 oracle | HR@1 | HR@3 | HR@5 | F1 argmax (chính nó) | phút |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | UAM-AD | 0,543 | – | 0,890 | – | 0,782 | – | 0,738 | 0,734 | 0,938 | 0,972 | – | 129,4 |
| RE2-OB | ART | 0,880 | −0,337 (−38,3%) | 0,982 | −0,092 (−9,4%) | 0,957 | −0,175 (−18,3%) | 0,921 | 0,759 | 0,840 | 0,933 | – | 6,1 |
| RE2-OB | Eadro (có giám sát) | 0,708 | −0,165 (−23,3%) | 0,862 | +0,028 (+3,2%) | 0,775 | +0,007 (+0,8%) | 0,735 | 0,683 | 0,878 | 0,993 | 0,697 | 12,6 |
| RE2-OB | OrEdge (bán giám sát) | 0,872 | −0,328 (−37,7%) | 0,976 | −0,086 (−8,8%) | 0,963 | −0,181 (−18,8%) | 0,942 | 0,924 | 0,943 | 0,948 | 0,896 | 30,1 |
| RE3-OB | UAM-AD | 0,525 | – | 0,828 | – | 0,596 | – | 0,753 | 0,833 | 0,912 | 0,964 | – | 37,0 |
| RE3-OB | ART | 0,590 | −0,065 (−11,0%) | 0,900 | −0,072 (−8,0%) | 0,762 | −0,165 (−21,7%) | 0,774 | 0,731 | 0,796 | 0,835 | – | 2,1 |
| RE3-OB | Eadro (có giám sát) | 0,627 | −0,102 (−16,3%) | 0,778 | +0,050 (+6,4%) | 0,693 | −0,096 (−13,9%) | 0,650 | 0,824 | 1,000 | 1,000 | 0,638 | 4,5 |
| RE3-OB | OrEdge (bán giám sát) | 0,751 | −0,226 (−30,1%) | 0,891 | −0,063 (−7,0%) | 0,830 | −0,234 (−28,2%) | 0,815 | 0,952 | 0,996 | 0,999 | 0,774 | 10,9 |
| RE2-TT | UAM-AD | 0,593 | – | 0,808 | – | 0,672 | – | 0,625 | 0,758 | 0,907 | 0,945 | – | 185,1 |
| RE2-TT | ART | 0,584 | +0,010 (+1,6%) | 0,833 | −0,025 (−3,0%) | 0,688 | −0,017 (−2,4%) | 0,678 | 0,449 | 0,515 | 0,592 | – | 7,0 |
| RE2-TT | Eadro (có giám sát) | 0,627 | −0,034 (−5,4%) | 0,839 | −0,031 (−3,7%) | 0,728 | −0,056 (−7,7%) | 0,689 | 0,565 | 0,777 | 0,993 | 0,628 | 33,9 |
| RE2-TT | OrEdge (bán giám sát) | 0,828 | −0,235 (−28,4%) | 0,958 | −0,150 (−15,6%) | 0,944 | −0,272 (−28,8%) | 0,927 | 0,953 | 0,982 | 0,993 | 0,908 | 32,9 |
| RE3-TT | UAM-AD | 0,513 | – | 0,702 | – | 0,347 | – | 0,638 | 0,262 | 0,343 | 0,416 | – | 48,9 |
| RE3-TT | ART | 0,346 | +0,167 (+48,3%) | 0,707 | −0,005 (−0,7%) | 0,474 | −0,128 (−26,9%) | 0,483 | 0,379 | 0,440 | 0,501 | – | 2,2 |
| RE3-TT | Eadro (có giám sát) | 0,348 | +0,165 (+47,4%) | 0,636 | +0,066 (+10,4%) | 0,438 | −0,091 (−20,8%) | 0,451 | 0,614 | 1,000 | 1,000 | 0,220 | 11,6 |
| RE3-TT | OrEdge (bán giám sát) | 0,628 | −0,115 (−18,3%) | 0,766 | −0,064 (−8,4%) | 0,693 | −0,347 (−50,0%) | 0,668 | 0,903 | 0,996 | 0,997 | 0,633 | 33,6 |
| SN | UAM-AD | 0,730 | – | 0,958 | – | 0,738 | – | 0,854 | 0,166 | 0,309 | 0,443 | – | 3,9 |
| SN | ART | 0,563 | +0,167 (+29,7%) | 0,873 | +0,085 (+9,8%) | 0,566 | +0,172 (+30,4%) | 0,642 | 0,087 | 0,124 | 0,341 | – | 0,4 |
| SN | Eadro (có giám sát) | 0,837 | −0,107 (−12,8%) | 0,984 | −0,025 (−2,6%) | 0,878 | −0,139 (−15,9%) | 0,891 | 0,133 | 0,289 | 0,540 | 0,823 | 1,1 |
| SN | OrEdge (bán giám sát) | 0,558 | +0,172 (+30,9%) | 0,891 | +0,068 (+7,6%) | 0,627 | +0,111 (+17,7%) | 0,721 | 0,348 | 0,507 | 0,735 | 0,015 | 1,9 |
| **Trung bình** | UAM-AD | 0,581 | – | 0,837 | – | 0,627 | – | – | – | – | – | – | – |
| **Trung bình** | ART | 0,593 | −0,012 (−2,0%) | 0,859 | −0,022 (−2,5%) | 0,689 | −0,062 (−9,1%) | – | – | – | – | – | – |
| **Trung bình** | Eadro (có giám sát) | 0,630 | −0,049 (−7,7%) | 0,820 | +0,018 (+2,1%) | 0,702 | −0,075 (−10,7%) | – | – | – | – | – | – |
| **Trung bình** | OrEdge (bán giám sát) | 0,727 | −0,146 (−20,1%) | 0,896 | −0,059 (−6,6%) | 0,812 | −0,185 (−22,7%) | – | – | – | – | – | – |
<!-- ITEM9 RESULTS END -->

### Phụ lục C. Chẩn đoán phụ cho các so sánh với đối thủ (seed 0, tham khảo)

Số liệu seed 0, chạy để hiểu kết quả, không phải kết quả cuối; một số con số ở đây khác 5 seed ở T1.

**Vì sao ART cho số tốt trên RCAEval [CN 10-20, giải thích có đo; người dùng hỏi "sao ART kết quả tốt quá"].** Hai kiểm tra dùng nhãn test **chỉ để hiểu**, không chọn gì theo đó (`D:\ClaudeWork\phase3\item9_diag\single_channel.py`, `per_service.py`), trên đúng 4 scenario của lần chạy thử nhanh, z-score của từng kênh so với train:
1. **Từng kênh riêng lẻ gần như không phân biệt được** khi gộp mọi lỗi của file test: kênh tốt nhất có AUROC 0,767 (RE2-OB `cpu`), 0,978 (RE3-OB `f2`, vì file này chỉ tiêm vào `emailservice`), 0,749 (RE2-TT `cpu`), 0,911 (RE3-TT `f1`). Lý do: mỗi file test gồm các lần tiêm lỗi vào **nhiều service khác nhau** (RE2-OB `cpu`: 5 service; RE2-TT `cpu`: 5 service), nên một kênh chỉ lệch ở một phần các mẫu lỗi.
2. **Nhưng kênh của chính service bị tiêm lệch rất mạnh:** lấy z lớn nhất trong các kênh của service bị tiêm, AUROC theo từng service bị tiêm là **trung bình 0,982 (RE2-OB, nhỏ nhất 0,934), 0,976 (RE3-OB), 0,986 (RE2-TT, nhỏ nhất 0,969)**. RE3-TT `f1` khó hơn: `ts-auth-service` 0,977 nhưng `ts-route-service` chỉ 0,583 (trung bình 0,780).
3. **Diễn giải (suy từ hai số trên và từ số z-score đã có, không phải thí nghiệm riêng):** lỗi của RCAEval làm lệch mạnh, cục bộ các kênh của đúng service bị tiêm. ART có một nút mỗi service, mỗi nút có kênh riêng, và điểm bất thường là bình phương sai số của **3 service lệch nhiều nhất**, tức là kiểu "chỉ cần một service bất thường là báo", cùng loại với z-score lấy max (không huấn luyện, AUROC 0,984 ở RE2-OB). Điểm kiểu **trung bình toàn kênh** (như số hạng KPI của UAM-AD, trung bình trên mọi kênh) pha loãng tín hiệu cục bộ đó, nên thua. Phù hợp với số đo trước đó: z-score thua UAM-AD không phải vì mô hình, mà vì cách gộp kênh và chuẩn hoá từng kênh.
4. **Vì sao ART giảm ở TT:** hệ có nhiều service và kênh hơn (TT 27 service, 333–374 kênh dùng được, OB 11 service, 111–113 kênh), nên lấy max qua nhiều kênh thì càng dễ gặp một kênh bình thường nhưng đột biến, và tín hiệu của service bị tiêm bị lẫn: RE2-TT `cpu` AUROC của ART chỉ 0,725 dù từng service bị tiêm có AUROC 0,986 khi biết trước service nào. Khớp với bảng chạy thử (OB 0,979 và 0,971; TT 0,725 và 0,659).5. **Không phải rò rỉ hay cài đặt thiên vị:** ART chỉ huấn luyện trên `train.pkl` (bình thường), ngưỡng từ p95 điểm val bình thường, nhãn chỉ dùng để tính chỉ số. Nên đây là dataset "dễ với bộ phát hiện theo service", không phải ART gian lận.
6. **Ý nghĩa cho luận văn:** trên OB, một bộ phát hiện có cấu trúc theo service gần như đạt trần; UAM-AD không có cấu trúc đó (ghi trong hạn chế). Chỗ UAM-AD còn cạnh tranh là khi nhiều service/kênh (TT) và SN.

<!-- ITEM9 DIAG BEGIN -->

**Chẩn đoán vì sao các đối thủ hơn UAM-AD trên RCAEval [CN 10-19, chưa phải kết quả cuối; số liệu sinh bởi `summary_tools/diag_tables.py`]**

Câu hỏi của người dùng: ART và AnoFusion có "được cài để thắng" không? Đã kiểm bốn điều, tất cả trên seed 0, không dùng nhãn test để chọn gì. Kết quả ở `D:\ClaudeWork\phase3\item9_diag\`.

1. **Log theo service (sửa nhận định cũ).** Ghi chú cũ (trong `working_history.md`) và docstring ART bản đầu nói log "không gán được cho service" là **sai**: trường `logs` của pkl bắt đầu mỗi dòng bằng tên service (RE3-OB, RE3-TT, SN: `service|template`; RE2-OB: `service template`; RE2-TT: chỉ tên container). `codes/competitors/service_logs.py` tách ra 2 đặc trưng mỗi service (số dòng log, số dòng khác nhau), độ phủ 100% (RE2-OB, RE3-OB), 97% (RE2-TT), 95% (RE3-TT, phần còn lại là `ts-ui-dashboard` và các `*-mongo` không thuộc danh sách service), 100% (SN). ART nay mặc định là L+M+T (`--no_logs` để tắt). Bản ART M+T (không log) từng chạy trước đó đã bỏ theo yêu cầu người dùng [CN 10-20]; ART chỉ báo L+M+T.
2. **Điểm z-score không huấn luyện** (`codes/competitors/zscore_max.py`): độ lệch |x−μ|/σ của từng kênh (KPI theo service, 6 đặc trưng trace, 2 log; bỏ kênh hằng trong train) so với train, gộp bằng **max** hoặc **trung bình** qua kênh; ngưỡng p95 val; không mô hình nào được học.
3–4. Hai kiểm tra trên ART M+T (xáo thứ tự thời gian `--shuffle_order`; cắt đầu vào ±100 `--clip 100`) đã bỏ cùng bản M+T [CN 10-20]; chưa chạy lại trên L+M+T (hai cờ vẫn có trong `run_art.py`, mỗi lần 1–3 phút mỗi dataset).

Trung bình qua scenario, seed 0 (UAM-AD và UAC-AD cùng seed 0 từ bước 8, `norm_sum`):

| Dataset | Phương pháp | F1@val95 | AUROC | AUPRC | HR@1 | HR@3 |
|:--|:--|--:|--:|--:|--:|--:|
| RE2-OB | z-score max theo kênh (không huấn luyện) | 0,885 | 0,984 | 0,959 | 0,884 | 0,959 |
| RE2-OB | z-score trung bình theo kênh (không huấn luyện) | 0,844 | 0,962 | 0,906 | 0,884 | 0,959 |
| RE2-OB | UAM-AD (seed 0) | 0,495 | 0,887 | 0,775 | 0,352 | 0,622 |
| RE2-OB | UAC-AD (seed 0) | 0,480 | 0,854 | 0,661 | – | – |
| RE3-OB | z-score max theo kênh (không huấn luyện) | 0,683 | 0,892 | 0,772 | 0,703 | 0,827 |
| RE3-OB | z-score trung bình theo kênh (không huấn luyện) | 0,613 | 0,868 | 0,718 | 0,703 | 0,827 |
| RE3-OB | UAM-AD (seed 0) | 0,556 | 0,832 | 0,611 | 0,022 | 0,240 |
| RE3-OB | UAC-AD (seed 0) | 0,640 | 0,857 | 0,691 | – | – |
| RE2-TT | z-score max theo kênh (không huấn luyện) | 0,723 | 0,924 | 0,794 | 0,683 | 0,785 |
| RE2-TT | z-score trung bình theo kênh (không huấn luyện) | 0,532 | 0,831 | 0,629 | 0,683 | 0,785 |
| RE2-TT | UAM-AD (seed 0) | 0,540 | 0,809 | 0,692 | 0,421 | 0,604 |
| RE2-TT | UAC-AD (seed 0) | 0,258 | 0,672 | 0,445 | – | – |
| RE3-TT | z-score max theo kênh (không huấn luyện) | 0,379 | 0,710 | 0,421 | 0,419 | 0,554 |
| RE3-TT | z-score trung bình theo kênh (không huấn luyện) | 0,537 | 0,756 | 0,411 | 0,419 | 0,554 |
| RE3-TT | UAM-AD (seed 0) | 0,515 | 0,705 | 0,350 | 0,063 | 0,095 |
| RE3-TT | UAC-AD (seed 0) | 0,530 | 0,732 | 0,416 | – | – |

Đọc kết quả (chỉ những điều số liệu đỡ được):
- **Một điểm z-score không cần mô hình ngang hoặc hơn ART (L+M+T) và hơn UAM-AD ở cả 4 dataset** (AUROC z-max: RE2-OB 0,984, RE3-OB 0,892, RE2-TT 0,924, RE3-TT 0,710, tức so với ART L+M+T +0,002, 0,000, +0,093, +0,024; so với UAM-AD seed 0 là +0,097, +0,060, +0,115, +0,005; riêng RE3-TT z-trung bình 0,756 hơn UAM-AD 0,705 rõ hơn). Vậy phần lớn "ưu thế" của ART ở RE2-OB và RE3-OB đến từ việc dataset dễ khi xét từng kênh theo service, không phải từ cách tôi cài ART: ART L+M+T gần như trùng điểm z-score ở hai dataset OB (0,982 so với 0,984; 0,892 so với 0,892) và **thấp hơn rõ ở RE2-TT (0,831 so với 0,924)**.
- **HR@k của ART trong bảng kết quả là bản chưa cắt đầu vào (theo tác giả).** Trên bản ART M+T cũ (đã bỏ), cắt ±100 không đổi điểm phát hiện nhưng đổi HR@1 ở RE2-OB (0,768 → 0,863); chưa đo lại trên L+M+T, nên HR@k của ART phải ghi là chưa cắt.
- **Giả thuyết "trung bình làm loãng tín hiệu" chỉ đúng một phần.** Ở RE2-TT (432 kênh) z-trung bình thấp hơn z-max nhiều (AUROC 0,831 so với 0,924; scenario `cpu` 0,686 so với 0,919), nhưng ở RE2-OB z-trung bình vẫn đạt AUROC 0,962, **cao hơn UAM-AD (0,887)** dù cũng là trung bình. Nên hàm chấm của UAM-AD là một phần nguyên nhân (nhất là ở dataset nhiều kênh), còn lại là bản thân mô hình tái tạo kém hơn điểm chuẩn hoá từng kênh. Nguyên nhân cụ thể chưa chẩn đoán (nghi: chuẩn hoá MinMax chung, nút cổ chai dùng chung cho mọi kênh, huấn luyện trên mẫu bị xáo); chưa thử.
- **HR@k:** điểm z-score tự nó xếp hạng service lỗi tốt (HR@1 RE2-OB 0,884; RE3-OB 0,703) vì mỗi kênh gắn với một service; UAM-AD không có cấu trúc theo service nên HR@1 chỉ 0,35 (RE2-OB), 0,02 (RE3-OB). Đây là hạn chế cấu trúc của UAM-AD, không phải do đối thủ.
- **Đề xuất cho Bảng B:** đưa hai hàng z-score (max, trung bình) làm mốc cạnh UAC-AD và UAM-AD, để người đọc thấy RCAEval dễ đến đâu với điểm không học. Chưa kiểm cho AnoFusion (chưa có số).

<!-- ITEM9 DIAG END -->

**Độ tin cậy của AUROC so với F1 [CN 10-24; đọc lại json seed 0 bằng `scratchpad/f1_vs_auroc.py`, không chạy thêm gì].** F1@val95 (ngưỡng = p95 điểm bình thường của val, không point-adjust, cùng luật cho cả sáu mô hình) so được giữa các mô hình, nhưng ngoài độ tách bình thường/bất thường nó còn đo việc ngưỡng val có chuyển sang test hay không, nên ồn hơn AUROC. Thứ hạng hai chỉ số khớp nhau (Spearman theo scenario, gộp mô hình: RE2-OB 0,74; RE3-OB 0,91; RE2-TT 0,88; RE3-TT 0,97; SN 0,91). Chỗ lệch là do ngưỡng: tỉ lệ báo động giả thật trên test ở ngưỡng val95 (đích 5%) của ART ở RE3-OB chỉ 1,7% (recall 0,47, nên F1 0,548 dù AUROC 0,892 cao nhất dataset), AnoFusion RE3-OB 1,4%, ART SN trung vị 7,9% (tối đa 21%), OrEdge tối đa 15,5% (RE3-OB). UAM-AD ở RE2-OB có precision 0,39 và recall 0,78 (nhiều báo động giả); không tính được tỉ lệ thật vì json của UAM-AD không lưu số mẫu, và chưa kiểm tỉ lệ cửa sổ lỗi của UAM-AD có bằng ≈ 20% của đối thủ không. Hai chỗ F1 đảo thứ tự so với AUROC: AnoFusion hơn UAM-AD về F1@val95 ở RE2-OB (0,587 so với 0,495) và RE2-TT (0,584 so với 0,540) dù AUROC thấp hơn (0,819 so với 0,887; 0,773 so với 0,809). F1 oracle của UAM-AD có point-adjust nên không so với nhóm còn lại. F1 argmax của OrEdge ở SN là 0,013 trong khi AUROC 0,813. SN mỗi file chỉ 4 đến 39 cửa sổ lỗi (trung vị 10): một cửa sổ đổi recall khoảng 10%, nên F1 từng scenario ồn. F1 còn phụ thuộc tỉ lệ lỗi (đối thủ: RCAEval ≈ 20%, SN ≈ 13%) nên không so được giữa dataset. **Đề xuất:** AUROC (kèm AUPRC) là chỉ số chính cho khả năng tách; F1@val95 là chỉ số phụ cho "ngưỡng triển khai thật", ghi kèm ghi chú trên; F1 oracle chỉ so trong cùng nhóm.

**F1@val95 của UAM-AD so với từng mô hình [CN 10-24, seed 0, `scratchpad/f1_wins.py`, đọc json].** Chênh = UAM-AD trừ mô hình kia, trong ngoặc là số scenario UAM-AD thắng (hơn trên 0,02) – hoà (trong 0,02) – thua. Cột theo thứ tự RE2-OB, RE3-OB, RE2-TT, RE3-TT, SN. ART: −0,385 (0-0-6), +0,008 (2-2-1), −0,040 (3-1-2), +0,198 (3-0-1), +0,203 (9-1-2). AnoFusion: −0,092 (2-0-4), +0,190 (4-0-1), −0,044 (2-1-3), RE3-TT chỉ `f1` (0,523 so với 0,116), +0,326 (12-0-0). Eadro: −0,203 (0-1-5), −0,112 (2-0-3), −0,080 (3-0-3), +0,165 (3-0-1), −0,077 (2-3-7). OrEdge: −0,351 (0-0-6), −0,211 (0-1-4), −0,293 (0-0-6), −0,109 (1-0-3), +0,389 (10-0-2). Trung bình 5 dataset của F1@val95 theo dataset: UAM-AD 0,568; ART 0,572; Eadro 0,629; OrEdge 0,683; trên 4 dataset (không RE3-TT) UAM-AD 0,581, AnoFusion 0,486. Điểm z-score không huấn luyện có F1@val95 cao hơn UAM-AD ở RE2-OB (0,885), RE3-OB (0,683) và RE2-TT (0,723); chưa đo ở SN. **Đọc:** UAM-AD không phải là mô hình tốt nhất về F1@val95 ở RCAEval; điểm mạnh là SN (hơn mọi mô hình không nhãn và hơn OrEdge) và hơn AnoFusion về trung bình; ngang ART về trung bình nhưng khác hình dạng (ART hơn hẳn ở RE2-OB, UAM-AD hơn ở RE3-TT và SN). Một seed.

**Vì sao UAM-AD thua ART và điểm z-score ở RE2-OB [CN 10-24; đọc log, mã và json của bước 8, không chạy thêm gì; nguyên nhân gốc CHƯA kiểm].** (1) *ART không có dấu hiệu bị cài để thắng:* AUROC của ART (0,982) gần bằng điểm z-score không huấn luyện (0,984), và AnoFusion thấp hơn UAM-AD về AUROC ở cả 4 dataset có đủ số. (2) *Seed 0 là seed xấu nhất của UAM-AD ở RE2-OB, nhưng không giải thích hết khoảng cách.* UAM-AD đã có 5 seed (bước 8). RE2-OB, AUROC theo seed 0 đến 4: 0,887 / 0,925 / 0,894 / 0,906 / 0,905 (trung bình 0,903, độ lệch 0,014); F1@val95: 0,495 / 0,677 / 0,496 / 0,631 / 0,523 (trung bình 0,564, độ lệch 0,084). Bốn dataset còn lại seed 0 sát trung bình (độ lệch AUROC ≤ 0,005). Trung bình 5 seed vẫn kém ART (0,903 so với 0,982; 0,564 so với 0,880). Đối thủ mới có 1 seed. (3) *F1@val95 của UAM-AD ở RE2-OB chủ yếu là lỗi ngưỡng:* scenario `disk` có AUROC 0,994 đến 0,995 ở cả 5 seed nhưng F1@val95 là 0,453 / 0,628 / 0,338 / 0,781 / 0,344, với recall 0,99 đến 1,00 và precision 0,20 đến 0,64 (precision 0,20 trùng tỉ lệ lỗi ≈ 20% của đối thủ, nên có vẻ gần như mọi cửa sổ test bị báo lỗi, nghĩa là ngưỡng p95 của val thấp hơn điểm của cửa sổ bình thường ở test; giả định tỉ lệ lỗi của UAM-AD cũng ≈ 20%, chưa kiểm). (4) *Điểm nghi về AUROC, từ mã:* `fuse_v3.py:638-639` tính `kpi_dis` và `log_dis` là sai số tái dựng **lấy trung bình trên mọi kênh** (`.mean(dim=-1)`), trên dữ liệu **MinMax theo khoảng của train** (`data_loads.py::normalize_data`); lỗi trong một service bị chia cho số kênh và bị nén theo khoảng min–max, còn điểm z-score chia cho độ lệch chuẩn từng kênh rồi lấy max. Chưa đủ để kết luận vì z-score lấy trung bình cũng đạt AUROC 0,962 ở RE2-OB. Nhánh trace riêng lẻ (`trace_scores.npz`) chỉ có AUROC 0,66 đến 0,83, thấp hơn điểm đầy đủ (0,74 đến 0,99) ở mọi scenario, nên điểm yếu không nằm ở việc nhánh trace kéo xuống. (5) *Cách kiểm đề xuất (chưa chạy, cần lệnh):* huấn luyện lại 1 scenario của RE2-OB (khoảng 20 đến 25 phút) với thêm lưu sai số từng kênh, rồi so AUROC khi gộp bằng trung bình, bằng max, và bằng trung bình sau khi chuẩn hoá từng kênh theo trung vị và p95 của val (cùng kiểu `norm_sum` đang chuẩn hoá từng thành phần). Nếu đúng thì sửa nằm ở hàm chấm điểm, không đụng kiến trúc hợp nhất, và phải kiểm lại A/B nhiều seed trên cả 5 dataset (SN không được tụt).

**SN là dataset lỗi làm đứt luồng gọi [CN 10-25; ý của người dùng: `Code_Stop`, `Svc_Kill` làm service dừng hoặc chết, đổi luồng gọi, còn RE2 (tài nguyên, hiệu năng) và RE3 (lỗi mã) không đổi luồng gọi nên trace ít giúp. Lần đầu tôi hiểu sai và đo điểm riêng của nhánh trace trên cả dataset; việc đó không kiểm ý này].** Đã kiểm lại bằng cách tách SN theo họ lỗi (tính một lần bằng lệnh Python đọc json seed 0 và `trace_scores.npz`, không lưu thành file; mỗi họ 3 scenario, 1 seed; AUROC trung bình; cột cuối là điểm riêng của nhánh trace `trace_dis` của UAM-AD):

| Họ lỗi SN | UAM-AD | ART | AnoFusion | Eadro | OrEdge | Nhánh trace riêng |
|:--|--:|--:|--:|--:|--:|--:|
| `Code_Stop_*` | 0,974 | 0,974 | 0,806 | 0,976 | 0,737 | 0,910 |
| `Svc_Kill_*` | 0,931 | 0,818 | 0,512 | 0,991 | 0,792 | 0,628 |
| Luồng gọi (hai họ trên, 6 scenario) | 0,952 | 0,896 | 0,659 | 0,984 | 0,764 | 0,769 |
| `DB_Redis_*` | 0,964 | 0,847 | 0,755 | 0,979 | 0,874 | 0,690 |
| `Perf_*` | 0,953 | 0,840 | 0,852 | 0,971 | 0,849 | 0,700 |
| Khác (hai họ trên, 6 scenario) | 0,959 | 0,843 | 0,804 | 0,975 | 0,861 | 0,695 |

Đọc: **ủng hộ ý người dùng** ở ba chỗ: (i) nhánh trace đứng riêng phát hiện `Code_Stop` rất tốt (0,910, so với 0,69–0,70 ở `DB_Redis`, `Perf`); (ii) AnoFusion và OrEdge kém rõ nhất đúng ở họ luồng gọi (AnoFusion 0,659 so với 0,804; OrEdge 0,764 so với 0,861); (iii) khoảng cách UAM-AD hơn hai mô hình đó lớn hơn ở họ luồng gọi (hơn AnoFusion 0,293 so với 0,155; hơn OrEdge 0,188 so với 0,098). **Không ủng hộ** ở hai chỗ: (i) `Svc_Kill` nhánh trace riêng chỉ 0,628; (ii) ưu thế so với ART **không** tập trung ở họ luồng gọi, mà ngược lại (hơn ART 0,056 ở luồng gọi, 0,116 ở họ khác; ở `Code_Stop` UAM-AD và ART bằng nhau, 0,974). Điểm hợp nhất của UAM-AD cao đều ở cả bốn họ (0,93–0,97), tức SN dễ với UAM-AD nói chung. Cách viết đỡ được: "trên SN UAM-AD vượt AnoFusion và OrEdge rõ nhất ở lỗi làm đứt luồng gọi, và nhánh trace tự phát hiện tốt `Code_Stop`"; chưa đỡ được câu "UAM-AD hơn mọi mô hình khác nhờ lỗi luồng gọi". Giới hạn: mỗi họ 3 scenario, 1 seed, `Svc_Kill` chỉ có 4 cửa sổ lỗi mỗi file. Muốn khẳng định cơ chế cần ablation tắt nhánh trace trên SN (UAM-AD chạy 12 scenario SN khoảng 4 phút mỗi seed; **chưa chạy, cần lệnh**).

## Nguồn

- AnoMod (Table 1, mô tả case): https://arxiv.org/html/2601.22881v1 ; Zenodo: https://zenodo.org/records/18342898 ; scripts: https://github.com/EvoTestOps/AnoMod
- RCAEval: https://github.com/phamquiluan/RCAEval
- DeepTraLog: https://github.com/FudanSELab/DeepTraLog ; https://www.microsoft.com/en-us/research/publication/deeptralog-trace-log-combined-microservice-anomaly-detection-through-graph-based-deep-learning/
- MSDS (Multi-source): https://zenodo.org/record/3549604
- MSTGAD: https://arxiv.org/abs/2310.04701 ; https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection
- UAC-AD: DOI 10.1109/TSC.2024.3411481
- Hades: https://arxiv.org/abs/2302.06914
- Eadro: https://arxiv.org/abs/2302.05092 ; https://github.com/BEbillionaireUSD/Eadro
- AnoFusion: https://arxiv.org/abs/2305.18985 ; code: https://github.com/AIOps-Lab-NKU/AnoFusion
- ART: https://conf.researchr.org/details/ase-2024/ase-2024-research/95/ART-A-Unified-Unsupervised-Framework-for-Incident-Management-in-Microservice-Systems ; code: https://github.com/bbyldebb/ART
- OrEdge: https://pith.science/paper/2608.00309 ; code: https://github.com/theamrzaki/MicroService_Twin_Original
- TrioXpert (bảng so ART/Eadro/Hades): https://arxiv.org/html/2506.10043v2
- UAC-AD code: https://github.com/lhysgithub/UAC-AD
- TADBench (TraceVAE, GTrace, ...): https://github.com/nkalgo/TADBench
- TraceDAE: https://ieeexplore.ieee.org/document/11052877/
- AMulSys: https://www.sciencedirect.com/science/article/abs/pii/S1566253525000909
- CMT-AD: https://arxiv.org/html/2609.25856v1
- TORAI: https://arxiv.org/abs/2604.13522
- AnoMod scripts thu thập TT (chaos, `run_experiment.sh`, `log_collector.py`): https://github.com/EvoTestOps/AnoMod/tree/main/TT_collection-scripts
- RCAEval datasets (RE1/RE2/RE3, gồm RE2-TT 2,8GB, RE3-TT 0,24GB): https://zenodo.org/records/14590730
