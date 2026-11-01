# Lịch sử làm việc (working_history)

Chuyển nguyên văn từ `chon_dataset_va_sota.md` ngày 2026-10-30: mọi đoạn ghi tiến trình theo ngày (chạy gì lúc nào, lỗi và cách sửa, chạy thử, ước tính giờ, kế hoạch cũ, quyết định lúc đó, số liệu trung gian). **Kết quả hiện tại không nằm ở đây mà ở `chon_dataset_va_sota.md`.** Số trong file này có thể đã bị thay bằng số mới hơn (3 seed → 4 seed → 5 seed, seed 0 → 5 seed); khi khác nhau, lấy số ở `chon_dataset_va_sota.md`. Các chỗ "phía trên/bên dưới/xem mục" trong đoạn được giữ như lúc viết, tính theo bố cục file cũ.

Quy ước: tiến trình mới ghi thêm vào cuối file này, mỗi mục một ngày, kèm giờ; kết quả và nhận xét hiện tại ghi ở `chon_dataset_va_sota.md`.


## Tiêu đề và lời dẫn đầu file (bản cũ)

Ngày: 2026-10-04, cập nhật 2026-10-13 (các điểm đổi đánh dấu **[CN 10-13]**). Phạm vi: nghiên cứu, kèm chuẩn bị dữ liệu ở giai đoạn 1. Ký hiệu: **[KC]** = đã kiểm chứng từ nguồn gốc (link ở cuối), **[Repo]** = kiểm chứng từ tài liệu/code nhánh `develop`, **[SL]** = suy luận, chưa kiểm chứng.

## Giai đoạn 1: chuẩn bị dữ liệu (ghi hoàn thành)

**Giai đoạn 1 đã hoàn thành (2026-10-13).** Thay đổi trong cây làm việc của `total_experiment` (chưa commit): code `val.pkl` cho RE2/RE3-OB (từ `f7c9e2b`), sửa NaN cho pandas 3 ở hai script OB, `preprocess_rcaeval_re{2,3}_tt.py`, `measure_structural_fingerprint.py`, tên dataset TT trong `run.py`/`rca.py`. Dữ liệu mới (git-ignored): `data/rcaeval_re2_tt`, `data/rcaeval_re3_tt`, `data/deeptralog/scenarios/test_F05.pkl`, `test_F09.pkl`, kết quả chạy thử đã chuyển ra ngoài vùng git, ở `D:\ClaudeWork\dtl\pilot_F05_F09_only_sync_seed0\`. Còn thiếu cho Giai đoạn 3: script eval theo scenario cho TT (copy từ bản OB), cấu hình đóng băng (Giai đoạn 2).

## Giai đoạn 2: khảo sát đối thủ, dựng đối thủ, chạy thử, quyết định giao thức

**Khảo sát đối thủ (2026-10-13)** [KC: đọc mã nguồn, chưa chạy]. Môi trường máy: Python 3.11, torch 2.5.1, pandas 3.0.1, không có DGL/PyG; chỉ có một bản Python. Cả bốn repo đều ghim phiên bản cũ và không có license, chỉ dùng để so sánh nghiên cứu và phải trích dẫn.

| Đối thủ | Phụ thuộc cũ | Điểm lệch so với giao thức của ta | Ước lượng dựng + chạy thử SN |
|:--|:--|:--|--:|
| AnoFusion | torch 1.6, pandas 1.1.5 (dùng `DataFrame.append`, `cuda_device=1` cố định) | Mô hình cho **một service instance** trên chuỗi thời gian (cửa sổ 20 mẫu liên tiếp, dự đoán bước kế); đồ thị từ thông tin tương hỗ `mutual_info_score` giữa mọi kênh (rất chậm với 300+ kênh liên tục). Ngưỡng EVT nằm ngoài repo | 5–8 giờ |
| ART | DGL 0.9.0, torch 1.12.1, Python 3.9 | Đơn vị mẫu là (timestamp, đồ thị, đặc trưng nút) theo từng sự cố; pre-train mặc định 1000 epoch; kết quả ở mức sự cố qua SPOT, không phải F1 theo cửa sổ | 8–12 giờ |
| Eadro | DGL 0.9.1 (`GATv2Conv`, `GlobalAttentionPooling`), torch 1.12.1; không dùng `tick` trong code | Có giám sát: cần nhãn lỗi và nhãn service lỗi khi train | 5–8 giờ |
| DeepTraLog GGNN | torch 1.8, PyG 1.6.1 | Đọc `GraphData` riêng của nó, chia train/test 6:4; cần dựng đồ thị đúng split F-case của ta | 5–7 giờ |

Tổng ước lượng 23–35 giờ làm việc (3–4,5 ngày), chủ yếu là viết bộ chuyển dữ liệu và port lớp đồ thị sang torch 2.5 (cài `torch_geometric` trong một venv riêng, kế thừa gói hệ thống, để không đụng môi trường chính). **DeepTraLog GGNN cụ thể [KC: đọc `model/*.py`]:** đầu vào là `GraphData` gốc của tác giả (mỗi trace một đồ thị gồm `node_info`, `edge_index`, `edge_attr`, `trace_bool`, `trace_id` và embedding sự kiện); mô hình là GatedGraphConv + GlobalAttention + Deep SVDD (soft-boundary, 100 epoch, batch 32, hidden 300), tự chia 60% normal để train, 10% val, 30% normal cộng toàn bộ bất thường để test. `GraphData` (khoảng 1GB) hiện không còn trên máy, cần tải lại. Mã dùng `torch_scatter` và PyG 1.6 nên phải port. Vì pkl của ta lưu id dạng băm md5 nên không dựng lại được chính xác tập normal train của ta; phương án đề xuất: **cùng tập trace bất thường của từng F-case, normal theo cách chia của chính GGNN**, ngưỡng p95 trên val của nó, rồi báo F1@val, AUROC, AUPRC; ghi rõ khác biệt này trong luận văn. 

**DeepTraLog GGNN: đã dựng và chạy thử [CN 10-14]** (`codes/competitors/deeptralog_ggnn/prepare_graphs.py`, `run_ggnn.py`). `GraphData` tải từ GitHub của tác giả (khoảng 180 MB nén, không phải 1 GB; để ở `D:\ClaudeWork\competitors\GraphData`), 132.485 trace khớp số liệu nhãn đã có. Cài `torch_geometric` không cần: GatedGraphConv và global attention viết lại bằng torch thuần (cùng ngữ nghĩa PyG), giữ mặc định tác giả (hidden 300, 3 lớp, Deep SVDD soft-boundary nu 0,05, AdamW 1e-4, batch 32, 100 epoch, chia normal 60/10/30). **Khác biệt cần ghi trong luận văn:** (1) tác giả không công bố file embedding sự kiện (`embedding_id_url+type.csv`), nên embedding 300 chiều được dựng lại từ nội dung sự kiện (TF-IDF + SVD, 1878 sự kiện); (2) mỗi seed huấn luyện một lần trên normal rồi chấm cả 5 F-case, tập test của F-case = toàn bộ trace bất thường của F-case + normal lấy từ tập test của GGNN với số lượng bằng tập test của UAM-AD (cùng tỉ lệ bất thường), ngưỡng p95 điểm val. Chạy thử rút gọn (15 epoch, 8.000 trace train, 1 seed, **chưa phải kết quả cuối**): AUROC F01 0,82, F02 0,94, F13 0,93, F05 0,996, F09 0,29 (F09 thấp hơn ngẫu nhiên, nhất quán với việc F09 không có tín hiệu cấu trúc). **Chi phí:** khoảng 156 giây mỗi epoch ở 65.490 trace train (tính toán trên GPU là điểm nghẽn), 100 epoch ≈ 4,3 giờ mỗi seed, 5 seed ≈ 22 giờ. **Quyết định (2026-10-14): Giai đoạn 3 chạy 30 epoch** (`--epochs 30`, mốc giảm lr ở epoch 18 = 60% như mặc định; ≈ 6,5 giờ cho 5 seed), khác mặc định 100 epoch của tác giả và phải ghi trong luận văn. Giai đoạn 2 chỉ dựng và chạy thử 1 seed.

**AnoFusion: đã dựng, chạy thử RE2-OB `cpu` [CN 10-14]** (`codes/competitors/anofusion/anofusion_model.py`, `run_anofusion.py`, `codes/competitors/eval_utils.py`). Bản tải trước đó thiếu 4 file mô hình (GAT, GATGRU, GTblock, GTlayer), đã tải bổ sung từ GitHub; kiến trúc và siêu tham số theo tác giả (GTN 5 lớp, 6 loại cạnh, GAT 6 head, GRU, cửa sổ 20, batch 64, lr 1e-5, MSE). **Thích nghi và khác biệt phải ghi trong luận văn:** (1) cả hệ thống coi là một "instance": kênh = toàn bộ KPI + đếm template log + đặc trưng trace của các service (RE2-OB: 86 + 41 + 66 = 193 kênh); (2) mẫu xếp theo (run, giây) và cửa sổ không băng qua run, nhưng pkl chỉ giữ một phần ngẫu nhiên số giây của mỗi run (train: khoảng 107/127 chỗ nhảy), nên các mẫu liền kề không phải lúc nào cũng liền giây; 20 mẫu đầu mỗi run không có đủ cửa sổ nên không được chấm (1.800 trong 51.520 mẫu của test_cpu, đều là normal); (3) ngưỡng EVT của tác giả thay bằng p95 điểm val, điểm số giữ như tác giả (max qua kênh của sai số chuẩn hoá); (4) chọn epoch theo MSE nhỏ nhất trên val (normal); (5) chuẩn hoá theo max của train (tác giả chuẩn hoá từng tập theo max riêng); NaN trong KPI gán 0 (62/86 cột KPI của RE2-OB có NaN); (6) ma trận thông tin tương hỗ giữ dạng số thực (mảng nguyên của tác giả cắt gần hết giá trị về 0 với dữ liệu này); (7) tham số `GTAggregation.weight` tác giả không khởi tạo, ở đây khởi tạo 0; GTN chạy một lần cho cả batch vì ma trận kề giống nhau giữa các mẫu (kết quả tương đương). Kết quả chạy thử (1 seed, 2 epoch, chưa phải kết quả cuối): F1@val 0,749, AUROC 0,849, AUPRC 0,817 trên RE2-OB `cpu`. **Chi phí:** khoảng 100 giây mỗi epoch trên RE2-OB, MI mất khoảng 45 giây. Lần chạy 10 epoch bị hệ thống dừng ở epoch 6 do thiếu RAM (không chạy lại); đến lúc đó val MSE đã đi ngang: 0,0794 (epoch 2), 0,0598 (4), 0,0589 (5), 0,0588 (6). Mặc định của tác giả là 100 epoch (≈ 2,8 giờ mỗi seed trên RE2-OB, chỉ cho một lần huấn luyện dùng chung cho 6 test); đề xuất chạy Giai đoạn 3 với **tối đa 20 epoch** và ghi rõ, vì val MSE đã hội tụ sớm.

**ART: đã dựng, chạy thử RE2-OB `cpu` [CN 10-14]** (`codes/competitors/art/art_model.py`, `run_art.py`). Bản tải trước đó chỉ có `main.py` và một file cấu hình hỏng; đã tải đủ mã từ `bbyldebb/ART`. Port sang torch thuần (DGL `GraphConv(norm='none')` thành tích ma trận kề dày), giữ kiến trúc và vòng huấn luyện của tác giả (Transformer, GRU 5 bước lịch sử, GraphConv 2 lớp, Adam 1e-3, batch 4, dừng sớm khi loss cải thiện dưới 1e-3 trong 5 epoch). **Khác biệt phải ghi trong luận văn:** (1) instance = service của đồ thị trace; kênh = KPI có tiền tố trùng tên service (RE2-OB: 78/86 KPI, đệm 0 tới 8 kênh) + 6 đặc trưng trace; KPI của container khác (DB, istio) bị bỏ; **không dùng log** vì pkl chỉ có danh sách template theo giây, không gán được cho service, nên ART ở đây là M+T chứ không phải L+M+T [nhận định sai, đã sửa [CN 10-20]: log gán được cho service, ART nay là L+M+T, xem khối chẩn đoán bên dưới]; (2) đồ thị lấy từ `trace_adj` của bước đích (nhị phân hoá, thêm self-loop); (3) mẫu xếp theo (run, giây), cửa sổ 6 bước không băng qua run và hai mẫu liền nhau cách tối đa 60 giây (như tác giả), nhưng pkl chỉ giữ một phần số giây nên bước không phải lúc nào cũng liền giây; 5 mẫu đầu mỗi run không được chấm (455/51.520 ở test_cpu); (4) SPOT và F1 theo khoảng sự cố của tác giả được thay bằng giao thức của ta: điểm = tổng sai số bình phương của 3 instance lệch nhiều nhất (SLD của tác giả), ngưỡng p95 val, F1 theo từng mẫu; (5) Transformer của tác giả chạy với `batch_first=False` nên attention đi qua các mẫu trong cùng batch; giữ nguyên và chấm với batch 4 bằng batch huấn luyện (tác giả chấm với 128); (6) GraphConv cuối của tác giả nằm trong list Python nên không được đăng ký tham số (không bao giờ được huấn luyện); ở đây đã đăng ký và huấn luyện; (7) 11 service là số lẻ nên thêm 1 instance giả toàn 0 để 2 head attention chia hết (không bao giờ được chọn vào top-3). Xếp hạng service lỗi theo cách RCL của tác giả (cosine giữa vector sai số của instance và SLD), HR@k tính theo từng mẫu bất thường. **Kết quả chạy thử (1 seed, dừng sớm ở epoch 10, ≈ 3 phút, chưa phải kết quả cuối):** RE2-OB `cpu` F1@val 0,897, P 0,840, R 0,963, AUROC 0,9875, AUPRC 0,967, HR@1/3/5 = 0,962/0,976/0,990. **Cần cẩn trọng:** số này cao hơn nhiều so với UAM-AD 5 epoch trên cùng scenario (F1 0,504, AUROC 0,860) và AnoFusion (AUROC 0,85 sau 2 epoch). Chưa kiểm được rò rỉ; lý do khả dĩ là ART mô hình hoá từng service riêng với KPI cpu/mem của chính service đó nên lỗi `cpu` lộ rõ, trong khi UAM-AD gộp mọi KPI thành một vector. Cần xem lại các loại lỗi khác (`delay`, `loss`) và kiểm lại công bằng trước khi dùng số này trong Bảng B.

**Eadro (hàng tham chiếu có giám sát): đã dựng, chạy thử RE2-OB `cpu` [CN 10-14]** (`codes/competitors/eadro/eadro_model.py`, `run_eadro.py`). Bản mã tải trước đó không chạy được (tên biến chưa định nghĩa: `trace_dropout`, `metric_dropout`, `self.locator` thay vì `self.localizer`; đối tượng đồ thị DGL), nên viết lại bằng torch thuần theo cấu trúc của nó: nhánh trace và metric (Conv1d giãn nở + self-attention theo chunk), fuse Linear + GLU, GATv2 4 head (max qua head), global attention pooling, detector 2 lớp và localizer theo service, loss 0,5·detect + 0,5·locate; siêu tham số mặc định của tác giả (50 epoch, batch 256, lr 1e-3, dừng khi loss tăng 10 lần, chunk 10 bước). **Giao thức nhãn (bạn đã chọn, 2026-10-14): 3-fold theo run.** Trong mỗi `test_<lỗi>.pkl`, mỗi (service, lỗi) có 3 run; fold k huấn luyện trên nhãn (bình thường/bất thường và service lỗi) của các run khác k trong cùng file, chấm run k; ba fold gộp phủ toàn bộ test nên so được với các hàng không giám sát. Mỗi test file × seed là 3 lần huấn luyện. Dùng mô hình ở epoch cuối (tác giả chọn epoch theo HR@1 trên test, là rò rỉ). **Khác biệt phải ghi trong luận văn:** (1) không có nhánh log (pkl không có đếm log theo service); kênh trace là 6 đặc trưng (bản gốc 1 kênh độ trễ); (2) đồ thị service là `trace_adj` của từng mẫu thay vì một đồ thị cố định; (3) nhãn mỗi chunk là nhãn bước cuối; (4) Eadro có giám sát: F1 báo theo quyết định argmax của chính nó (chỉ số chính cho hàng này), kèm F1@val (p95 điểm val của các run giữ lại), AUROC, AUPRC; HR@k tính trên mọi mẫu thật sự bất thường, không điều kiện theo detector; (5) mô hình thấy các run khác của đúng cùng (service, lỗi), nên đây là điều kiện dễ hơn nhiều so với các hàng không giám sát. **Kết quả chạy thử (1 seed, 50 epoch, 3 fold, ≈ 12 phút, chưa phải kết quả cuối):** F1 argmax 0,707 (P 0,887, R 0,588), F1@val 0,719, AUROC 0,832, AUPRC 0,787, HR@1/3/5 = 0,479/0,788/0,994. Chạy 2 epoch cho AUROC 0,850, HR@1 0,396, HR@3 0,983 (không tệ hơn đáng kể), nên 50 epoch mặc định có dấu hiệu quá khớp các run huấn luyện; giữ mặc định của tác giả theo nguyên tắc 'không chỉnh sau khi thấy kết quả'.

**OrEdge (hàng tham chiếu bán giám sát): đã dựng, mới chạy thử 1 epoch [CN 10-14]** (`codes/competitors/oredge/oredge_model.py`, `run_oredge.py`; mã gốc ở `D:\ClaudeWork\competitors\OrEdge`, nhánh `(pushing-recal-to-1.0)`, clone bị lỗi checkout do tên file `result/` không hợp lệ trên Windows nên chỉ khôi phục `src/`, `util/`, `main.py`). Mô hình chính của bài là `FREQ_DOMAIN=FITS_Legendre`: mỗi nút (service) và mỗi modality (metric, log, trace) qua embedding + mã hoá vị trí, một bộ mã hoá thời gian FITS-Legendre dùng chung (RevIN, cơ sở Legendre bậc 5, NormLin, lọc thông thấp, up-sampler tuyến tính), linear attention giữa token metric và log, tái dựng đầu vào; sai số tái dựng (thời gian + 0,2 × sai số hệ số Legendre) của bước cuối từng nút đi vào bộ phân loại D→128→2 cho biết nút đó có phải nút lỗi không. Huấn luyện bán giám sát với 3 trạng thái nút (bình thường / lỗi đã gán nhãn / chưa biết), loss `(1−para)·BCE có trọng số 96 + para·rec`, `para = max(1/(epoch+1), 0,01)`, AdaBelief lr 1e-3, wd 5e-4, batch 100, cửa sổ 10, 50 epoch, patience 15 (mặc định parser TT của tác giả). **Giao thức nhãn:** như Eadro, 3-fold theo run trong từng `test_<lỗi>.pkl`; nhãn nút lỗi của mẫu bất thường chỉ giữ cho 5 trong mỗi 10 mẫu bất thường (label_percent 0,5 như `chunkDataset` của tác giả), 5 mẫu còn lại là "chưa biết". Điểm của mẫu = max qua nút của P(nút là nút lỗi); xếp hạng service cho HR@k = chính các xác suất đó; chọn mô hình theo loss huấn luyện nhỏ nhất (checkpoint `loss` của tác giả, checkpoint `f1` chọn theo dữ liệu test nên bỏ). **Khác biệt phải ghi trong luận văn:** (1) log không gán được cho service nên mọi nút nhận cùng vector đếm template của hệ thống; metric của nút = KPI có tiền tố trùng tên service (như ART/Eadro); trace = 6 đặc trưng của service; (2) `legendre_encode` của tác giả gọi `reshape(B*C, T)` không hoán vị nên trộn giá trị thời gian với kênh, bản này lấy hệ số Legendre dọc trục thời gian; (3) AdaBelief viết lại (rectified, decoupled weight decay) để khỏi cài thêm gói; (4) đầu vào bị chặn ở ±100 lần max của train vì có KPI hằng số trong train nhưng đổi trong test, giá trị chuẩn hoá lên tới 5,6e13 làm bình phương sai số tràn thành NaN (chỉ OrEdge bình phương sai số nên bị; **xem ghi chú ART ở dưới: cùng hiện tượng có thể làm điểm ART bị chi phối bởi vài kênh**). **Chi phí:** khoảng 13 giây mỗi epoch mỗi fold trên RE2-OB, 50 epoch × 3 fold ≈ 30 phút mỗi test file mỗi seed. **Chạy thử (1 seed, chỉ 3 epoch, 3 fold, ≈ 2 phút, RE2-OB `cpu`, chưa phải kết quả cuối):** F1 argmax 0,598 (P 0,437, R 0,946), F1@val 0,811, AUROC 0,955, AUPRC 0,909, F1 oracle 0,875, HR@1/3/5 = 0,788/0,801/0,804. Giai đoạn 2 chỉ chạy thử vài epoch; số epoch cho Giai đoạn 3 (mặc định tác giả 50) chưa chốt. Lần chạy thử 50 epoch chạy nền bị hệ thống dừng vì thiếu RAM, chạy thử ở foreground thì ổn.

**Script eval theo scenario cho TT [CN 10-14]:** `codes/common/eval_per_scenario_rcaeval_re2_tt.py` và `eval_per_scenario_rcaeval_re3_tt.py` (sao từ bản OB; `num_services` mặc định 27, loại lỗi RE3-TT là f1–f4, mặc định RE2-TT đặt theo cấu hình đóng băng 5 5 / batch 128 / patience 3 / window 30). Chưa chạy thử vì mỗi lần chạy UAM-AD đủ mất nhiều giờ.

**SN và các đối thủ cần thứ tự thời gian [CN 10-14, đã kiểm trên pkl]:** `preprocess_sn.py` có tạo `_scenario` và `_t_start` cho từng cửa sổ (dòng ~818), nhưng các file pkl hiện có **không còn hai trường này** (chỉ có `label`, `kpis`, `logs`, `log_features`, `trace_node_features`, `trace_adj`) và id là băm md5, nên với pkl hiện tại thứ tự thời gian của SN không khôi phục được. Thêm nữa: train SN chỉ có 39 cửa sổ 30 giây, mỗi file test chỉ có 32–263 cửa sổ rời rạc và đúng một phiên bất thường (không chia 3-fold theo run được). Kết luận tạm: AnoFusion, ART, Eadro, OrEdge **không áp dụng trên SN** với pkl hiện tại; muốn chạy phải sinh lại pkl từ dữ liệu thô AnoMod có giữ `_t_start` (việc này không đổi nội dung các file đang dùng, nhưng cần dữ liệu thô). Chờ bạn quyết định.

**Kiểm đường dữ liệu của đối thủ trên dataset khác [CN 10-14, 1 seed, vài epoch, chỉ kiểm chạy được, không phải kết quả]:**

| Đối thủ | Dataset (test) | Cấu hình thử | Kết quả | Chi phí / ghi chú |
|:--|:--|:--|:--|:--|
| OrEdge | RE3-TT f1 (27 service) | 3 epoch | AUROC 0,490 (≈ ngẫu nhiên) | ≈ 36 giây/epoch/fold, 50 epoch ≈ 90 phút mỗi fold; nặng |
| Eadro | RE3-TT f1 | 3 epoch | AUROC 0,432, HR@3 = 1,0 | ≈ 27 giây/fold ở 3 epoch |
| ART | RE3-TT f1 | 3 epoch | F1@val 0,512, AUROC 0,727 | ≈ 8 giây/epoch; 27 service là số lẻ nên tự thêm instance giả |
| AnoFusion | RE3-OB f1 (465 kênh) | 1 epoch, `--micro_batch 8` | F1@val 0,766, AUROC 0,810 | MI 151 giây, 263 giây/epoch |
| AnoFusion | RE2-TT (596 kênh), RE3-TT (1357 kênh) | **chưa chạy thử** | – | xem dưới |

Với RE3-TT f1 (lỗi mức mã nguồn), ba mô hình đều gần ngẫu nhiên sau vài epoch, nhất quán với việc tập này khó phát hiện bằng đặc trưng hiện có. **AnoFusion coi mỗi kênh là một nút đồ thị**: số kênh RE2-OB 193, RE3-OB 465, RE2-TT 596, RE3-TT 1357; bộ nhớ lớp đồ thị tăng theo batch × kênh², batch 64 mặc định tràn GPU 3 GB từ RE3-OB trở lên (đã thêm `--micro_batch`, gradient tích luỹ, cập nhật tương đương vì mô hình không có BatchNorm, và `--eval_batch`). Ngoại suy theo kênh² (chưa đo): RE3-TT cần MI khoảng 20 phút và khoảng 35 phút mỗi epoch, 20 epoch ≈ 12 giờ mỗi seed, **không khả thi cho 5 seed**; RE2-TT khoảng 7 phút mỗi epoch. Cần quyết định cho Giai đoạn 3: bỏ AnoFusion trên RE3-TT (ghi rõ lý do), hoặc giảm kênh (ví dụ chỉ giữ top-K template log) và ghi là khác tác giả.

**Kiểm số cao của ART [CN 10-14]:** chạy thêm ART (1 seed) trên RE2-OB `delay` và `loss`: F1@val 0,911 / 0,898, AUROC 0,993 / 0,987, nên cao ở cả các loại lỗi khác, không riêng `cpu`. Điểm z-score đơn giản không huấn luyện (độ lệch tuyệt đối lớn nhất của một kênh KPI/trace so với trung bình và độ lệch chuẩn của train, đặc trưng theo service như ART) cho RE2-OB `cpu` / `delay` / `loss`: AUROC 0,964 / 0,986 / 0,986, F1@val 0,813 / 0,899 / 0,906. Vậy ART cao không phải do rò rỉ: RE2-OB dễ tách khi xét từng kênh theo service, và một điểm số không học đã vượt UAM-AD 5 epoch (AUROC 0,860 trên `cpu`). Nên đưa hàng "z-score tối đa theo kênh" làm mốc kiểm tra vào Bảng B để người đọc thấy mức dễ của dataset.

**Quyết định giao thức cần chốt trước:** file test của ta xáo trộn mẫu (UAM-AD chia cửa sổ trên thứ tự đã xáo), trong khi AnoFusion và ART cần thứ tự thời gian; id mẫu RE2/RE3 có chỉ số giây nên khôi phục được thứ tự, nhưng normal trong test lấy rời rạc từ pool nên có khoảng trống.

## Việc 7: ghi chạy (lời dẫn cũ của các bảng DeepTraLog)

**Việc 7, phần UAM-AD trên DeepTraLog: đã chạy xong 75 ô [CN 10-15]** (3 cấu hình × F01, F02, F13, F05, F09 × 5 seed; `codes/common/run_phase3_deeptralog.py` chạy từng ô một theo `frozen_config.json`, `aggregate_phase3_deeptralog.py` gộp; kết quả ở `D:\ClaudeWork\phase3\dtl\`, ngoài vùng git; ≈ 7,8 giờ). Một ô (`full F02 seed3`) bị tiến trình Python sập lúc khởi động (mã `0xC0000409`, chưa rõ nguyên nhân, 1/76 lần chạy) và được chạy lại đúng cấu hình và seed, ra kết quả bình thường. Trung bình ± độ lệch chuẩn (độ lệch chuẩn của quần thể, 5 seed). F1 ở ngưỡng p95 của val; "chỉ-sync" và "chỉ-async" vẫn giữ phần tái tạo log, chỉ khác nhánh trace nào bật:

**Việc 7, phần DeepTraLog GGNN: đã chạy xong 5 seed [CN 10-15]** (`run_ggnn.py --epochs 30 --seeds 0 1 2 3 4`, kết quả ở `D:\ClaudeWork\phase3\ggnn\seed{0..4}.json`; ≈ 81 phút mỗi seed, tổng ≈ 6,8 giờ). Trung bình ± độ lệch chuẩn (quần thể) qua 5 seed; ngưỡng p95 điểm val của chính GGNN:

## Việc 8: tiến trình UAM-AD và UAC-AD (lỗi bộ nhớ TT, ước tính, sửa script eval, seed 0 → 5, kết quả trung gian 3 và 4 seed)

**Việc 8: lỗi bộ nhớ trên TT đã sửa, và ước lượng thời gian [CN 10-15].** Chạy thử UAM-AD (và UAC-AD) trên RE2-TT, RE3-TT ban đầu **không chạy được**: tiến trình Python vọt từ 2,6 GB lên khoảng 11,7 GB bộ nhớ riêng ngay đầu epoch đầu tiên rồi lỗi `CUDA error: out of memory` ở `pin_memory` (lặp lại 3 lần: RE2-TT trace, RE3-TT trace, RE3-TT baseline). Nguyên nhân: `codes/models/basev3.py` giữ trong RAM đầu vào và đầu ra của mọi batch trong một epoch (`kpi_inputs`, `kpi_outputs`, `log_*`, ở vòng huấn luyện và ở `evaluate`) rồi `np.concatenate` (nhân đôi), chỉ để vẽ đồ thị đã bị comment; với 44.000 cửa sổ × 30 bước × 383 kênh KPI thì thành vài GB (OB chỉ 86 kênh nên không lộ). **Sửa tối thiểu:** với `data_type == "fuse"` bỏ phần tích luỹ này (các biến không được dùng ở đâu khác; `git diff` còn 16 dòng thêm, 16 dòng bớt; chưa commit). Sau sửa, đỉnh bộ nhớ riêng còn ≈ 5,0–5,6 GB và cả hai TT chạy qua. Chưa chạy lại so sánh số trước/sau trên OB (kết quả không đổi về nguyên tắc vì các mảng chỉ để vẽ); nên chạy kiểm một scenario OB khi có dịp.

Đo hôm nay (UAM-AD có trace, 1 seed, 1 epoch, `norm_sum`): RE3-TT f1: 119 giây/epoch, cả lần chạy 3 phút 48 giây (phần nạp dữ liệu và tiền xử lý ≈ 110 giây); RE2-TT cpu: 321 giây/epoch, cả lần chạy 9 phút 35 giây (phần cố định ≈ 254 giây). Đây chỉ là chạy thử để đo giờ, không phải kết quả (RE3-TT F1 0,507, AUROC 0,703; RE2-TT F1 0,432, AUROC 0,690 sau 1 epoch).

**Ước lượng cho 5 epoch (cấu hình đóng băng), mỗi seed, cộng mọi scenario của dataset.** OB và SN lấy từ các lần chạy trước (tháng 5, 1 seed, cùng 5 epoch; sai số khoảng ±20%), TT lấy từ số đo hôm nay; baseline TT chưa đo, ước trong khoảng tỉ lệ baseline/trace của OB (0,71 ở RE3-OB, 0,96 ở RE2-OB; lấy giá trị giữa cho RE3-TT):

| Dataset (số scenario) | UAM-AD mỗi seed | UAC-AD mỗi seed | UAM-AD 5 seed | UAC-AD 5 seed |
|:--|--:|--:|--:|--:|
| SN (12) | 6,6 phút | 5,4 phút | 0,6 giờ | 0,5 giờ |
| RE3-TT (4) | 47 phút | ≈ 40 phút | 3,9 giờ | ≈ 3,3 giờ |
| RE3-OB (5) | 1,4 giờ | 1,0 giờ | 7,0 giờ | 5,0 giờ |
| RE2-OB (6) | 2,9 giờ | 2,8 giờ | 14,7 giờ | 14,1 giờ |
| RE2-TT (6) | ≈ 3,1 giờ | ≈ 3,0 giờ | ≈ 15,5 giờ | ≈ 15 giờ |
| **Cộng** | **≈ 8,3 giờ** | **≈ 7,6 giờ** | **≈ 41,7 giờ** | **≈ 38 giờ** |

Tổng việc 8 ≈ **80 giờ máy liên tục** cho 5 seed (≈ 16 giờ cho mỗi "vòng seed" gồm cả 5 dataset và cả 2 mô hình; 3 seed ≈ 48 giờ). Chưa tính: `--enable_rca` (thêm bước tính xếp hạng service, chưa đo, dự kiến nhỏ), dừng sớm do patience 3 (có thể bớt một ít), và sự chậm đi khi RAM bị tranh chấp. Mỗi ô (scenario × seed) dài từ 6 phút (SN) đến khoảng 31 phút (RE2-TT), nên bị ngắt giữa chừng chỉ mất tối đa một ô nếu dùng bộ chạy có khả năng tiếp tục như bộ của DeepTraLog. **Lưu ý khi lập bộ chạy cho việc 8:** các script `eval_per_scenario_*` của OB/TT/SN không truyền `--score_rule norm_sum` (mặc định `run.py` là `raw_sum`) và không truyền `--enable_rca`, đồng thời `_scan_latest_results` lấy F1 cao nhất giữa các seed; nên cần bộ chạy gọi thẳng `run.py` theo `frozen_config.json` như bộ của DeepTraLog.

**Đã sửa các script eval cho việc 8 [CN 10-15]** (`eval_per_scenario_{rcaeval_re2_ob, rcaeval_re3_ob, rcaeval_re2_tt, rcaeval_re3_tt, sn}.py` và module mới `eval_seeds.py`; chưa commit): (1) truyền `--score_rule norm_sum` (mặc định mới, đúng cấu hình đóng băng) và `--enable_rca` (mặc định False; bật True cho UAM-AD); (2) mỗi seed là một tiến trình `run.py` riêng và **seed đã có kết quả trên đĩa được bỏ qua**, nên chạy `--run_start 0 --run_end 1` hôm nay rồi `--run_start 1 --run_end 5` sau này cho đúng 5 seed như chạy một lần (mỗi seed đã độc lập: `random_seed + chỉ số seed`; mỗi lần chạy `run.py` nạp lại dữ liệu nên chạy tách không tốn thêm); (3) bảng kết quả là trung bình ± độ lệch chuẩn **qua các seed** (không còn lấy F1 cao nhất), kèm AUROC, AUPRC, không in cột `oracle_f1` vì cột đó point-adjust; (4) thư mục kết quả mặc định đổi thành `result_phase3_*` để không lẫn với kết quả cũ trong `result_per_scenario_*`; (5) mặc định của script RE2-OB đổi sang giá trị đóng băng (5 5 / batch 128 / patience 3 / window 30; trước đó 10 10 / 256 / 5 / 60); (6) mặc định của script SN đổi sang 50 50 / patience 15 / `activity_penalty_weight` 1,5 / `gate_delta_lr_mult` 10. **Sửa thiếu sót trong `frozen_config.json`:** mục SN thiếu hai tham số mà lần chạy SN trước đã dùng (`activity_penalty_weight` 1,5 cho cả UAM-AD và UAC-AD, `gate_delta_lr_mult` 10 chỉ cho trace; xem `docs/experiment_results_sn_trace_vs_baseline_*.md`), đã bổ sung trước khi có bất kỳ kết quả SN cuối nào. Kiểm trên SN: seed 0 chạy qua script mới (UAM-AD, 12 scenario, 307 giây), rồi gọi lại với `--run_end 2` thì 12 scenario đều bỏ qua seed 0 và chỉ chạy seed 1; `rca_results.json` được ghi cho mỗi seed. Bộ chạy `run_phase3_item8.py` chạy lần lượt SN, RE3-TT, RE3-OB, RE2-OB, RE2-TT cho UAM-AD hoặc UAC-AD; kết quả ở `D:\ClaudeWork\phase3\item8\<dataset>\<model>\`.

**Việc 8, trạng thái [CN 10-17 13:30]: đã xong seed 0 và 1 cho cả UAM-AD và UAC-AD trên 5 dataset; đã dừng**, để xem lại số liệu trước khi chạy tiếp (seed 2–4 chưa chạy; chuỗi seed 2 đã hủy). Một ô (UAC-AD RE3-OB f4 seed 1) bị tiến trình sập (mã `0xC0000409`, lần thứ hai thấy lỗi này) nhưng bộ chạy vẫn báo OK cho cả dataset; phát hiện khi gộp, chạy lại ô đó (6 phút) ra kết quả bình thường. Khi gộp phải kiểm số seed của từng scenario, không tin trạng thái OK của bộ chạy. Trung bình qua scenario, rồi trung bình ± độ lệch chuẩn qua 2 seed:

| Dataset | UAM-AD F1@val | UAC-AD F1@val | UAM-AD AUROC | UAC-AD AUROC | UAM-AD AUPRC | UAC-AD AUPRC | UAM-AD thắng AUROC |
|:--|--:|--:|--:|--:|--:|--:|--:|
| SN | **0,729 ± 0,005** | 0,457 ± 0,154 | **0,955 ± 0,001** | 0,748 ± 0,197 | **0,705** | 0,543 ± 0,161 | 12/12 |
| RE2-TT | **0,561 ± 0,020** | 0,257 ± 0,001 | **0,807 ± 0,002** | 0,670 ± 0,002 | **0,689 ± 0,003** | 0,447 ± 0,002 | 4/6 |
| RE2-OB | **0,586 ± 0,091** | 0,481 ± 0,001 | **0,906 ± 0,019** | 0,851 ± 0,004 | **0,806 ± 0,031** | 0,656 ± 0,006 | 6/6 |
| RE3-OB | 0,566 ± 0,011 | **0,641 ± 0,001** | 0,836 ± 0,005 | **0,856 ± 0,000** | 0,620 ± 0,009 | **0,692 ± 0,000** | 1/5 |
| RE3-TT | 0,514 ± 0,001 | **0,537 ± 0,008** | 0,704 ± 0,001 | **0,743 ± 0,011** | 0,348 ± 0,002 | **0,430 ± 0,014** | 0/4 |

Trace giúp rõ ở SN và RE2-TT (đặc biệt lỗi làm service im lặng ở SN: `Svc_Kill_*` UAC-AD AUROC 0,15–0,21 so với UAM-AD 0,89–0,96), giúp vừa ở RE2-OB, và **không giúp, hơi kém hơn, ở hai tập lỗi mức mã nguồn RE3-OB, RE3-TT**. UAC-AD ở SN rất không ổn định giữa hai seed (F1 ± 0,154; train SN chỉ 39 cửa sổ). F1@val của UAM-AD ở RE2-OB dao động mạnh (± 0,091) trong khi AUROC ổn định hơn (± 0,019). HR@k của UAM-AD: xem bảng ngay dưới (bản trước ghi HR@5 của SN là 0,24; đó thực chất là HR@3, SN chưa có HR@5).

Ghi chú kế hoạch cũ: lần chạy chuỗi seed 2 đã hủy. Các seed mới của UAM-AD dùng `--rca_top_k 5` (mặc định mới của script eval, kể cả SN), seed 0 của UAM-AD chỉ có top-3. Thời gian đo thật mỗi seed: UAM-AD 6,7 giờ, UAC-AD 5,4 giờ.

**HR@k của UAM-AD, 3 seed [CN 10-18]** (trung bình qua scenario có service lỗi; mỗi ô là seed 0 / seed 1 / seed 2; cột cuối là trung bình±độ lệch chuẩn qua seed):

| Dataset | Số service | HR@1 | HR@3 | HR@5 (chỉ seed 1 / 2) | Mức ngẫu nhiên HR@1/3/5 |
|:--|--:|:--|:--|:--|:--|
| SN | 12 | 0,178 / 0,173 / 0,250 → **0,200±0,035** | 0,252 / 0,240 / 0,326 → **0,273±0,038** | 0,513 / 0,616 → **0,565±0,052** | 8% / 25% / 42% |
| RE3-TT | 27 | 0,063 / 0,082 / 0,069 → **0,071±0,008** | 0,095 / 0,113 / 0,104 → **0,104±0,008** | 0,220 / 0,209 → **0,215±0,006** | 4% / 11% / 19% |
| RE3-OB | 11 | 0,022 / 0,022 / 0,024 → **0,023±0,001** | 0,240 / 0,235 / 0,188 → **0,221±0,023** | 0,563 / 0,478 → **0,521±0,042** | 9% / 27% / 45% |
| RE2-OB | 11 | 0,352 / 0,360 / 0,347 → **0,353±0,005** | 0,622 / 0,608 / 0,606 → **0,612±0,007** | 0,735 / 0,726 → **0,731±0,004** | 9% / 27% / 45% |
| RE2-TT | 27 | 0,421 / 0,469 / 0,370 → **0,420±0,041** | 0,604 / 0,634 / 0,531 → **0,590±0,043** | 0,699 / 0,626 → **0,662±0,036** | 4% / 11% / 19% |

HR@1 và HR@3 có đủ 3 seed. HR@5 thật chỉ có ở seed 1 và 2, vì seed 0 chạy trước khi thêm `--rca_top_k` (lưu top-3, HR@5 trong `rca.py` chỉ nhìn danh sách nên bằng HR@3). Vì vậy HR@5 báo là trung bình của seed 1–2 (hoặc 1–4 nếu chạy tiếp), không trộn seed 0. SN seed 1 ban đầu cũng là kết quả cũ (top-3; bước SN lần sau chỉ chạy 1 giây vì bỏ qua seed đã có), nên đã chạy lại [CN 10-17 15:53, 355 giây, đủ 12 scenario] với top-5: kết quả trùng bản cũ (F1 0,724, AUROC 0,954, AUPRC 0,705, HR@1/3 0,173/0,240) và HR@5 = 0,513. **Không tính lại seed 0 từ checkpoint được:** đường `--pre_model` của `run.py` dùng ngưỡng quét trên test thay vì ngưỡng p95 của val, và chú thích ở `basev3.py` (dòng 877) cho biết chỉ số đặc trưng log đổi theo từng tiến trình Python; muốn có HR@5 của seed 0 phải chạy lại (khoảng 6,7 giờ).

Cách đọc khi viết (đối chiếu với mức ngẫu nhiên k/N, giả định xếp hạng ngẫu nhiên N service; **sửa lại nhận xét ngày 10-17**, khi đó ghi RE3-TT "cả ba mức" ≤ ngẫu nhiên là chưa đúng ở HR@1):
- **RE2-OB, RE2-TT:** hơn hẳn ngẫu nhiên ở mọi mức (RE2-OB HR@1 0,353 so với 0,091; RE2-TT HR@1 0,420 so với 0,037). RE2-TT dao động theo seed nhiều nhất (HR@1 0,370–0,469).
- **SN:** HR@1 0,200 và HR@5 0,565 cao hơn ngẫu nhiên (0,083 và 0,417) nhưng HR@3 0,273 ≈ ngẫu nhiên (0,250); phụ thuộc seed (seed 2 cao hơn hẳn: 0,250 / 0,326 / 0,616), nên SN cần cả 5 seed mới nói chắc.
- **RE3-TT:** HR@1 0,071 gấp khoảng 2 lần ngẫu nhiên (0,037) nhưng tuyệt đối vẫn thấp; HR@3 0,104 ≈ ngẫu nhiên (0,111); HR@5 0,215 cao hơn ngẫu nhiên nhẹ (0,185).
- **RE3-OB:** HR@1 0,023 và HR@3 0,221 **thấp hơn** ngẫu nhiên (0,091 và 0,273); chỉ HR@5 0,521 cao hơn (0,455). Đây là kết quả RCA yếu nhất, cần ghi rõ như một hạn chế.
- Lưu ý chung: RCA chỉ xếp hạng các cửa sổ có `fusion_loss` > ngưỡng p95 của luật `raw_sum`, không phải bộ phát hiện `norm_sum` dùng để báo F1, nên HR@k có điều kiện theo bộ phát hiện thô.

**Chẩn đoán phần giảm của UAM-AD so với UAC-AD ở RE3, và hai lưu ý ở SN [CN 10-17]** (số liệu từ 2 seed, chỉ phân tích, không đổi cấu hình đóng băng):

- Hiệu từng cặp (UAM-AD − UAC-AD, cùng scenario, cùng seed): RE3-TT AUROC −0,019 … −0,056 ở cả 8 ô (trung bình −0,039), AUPRC −0,082 (8/8 ô âm), F1 −0,023 (8/8 ô âm). RE3-OB AUROC trung bình −0,020 (9/10 ô âm; `f3` tệ nhất, `f4` ≈ 0), AUPRC −0,072 (10/10), F1 −0,075 (8/10). Độ lệch giữa hai seed của trung bình dataset chỉ ≈ 0,00–0,02, nên phần giảm là hệ thống, không phải nhiễu.
- Nguyên nhân nằm ở luật `norm_sum`, không phải ở cổng. Trong `fusion_loss` thô, trace bị nhân với cổng (`log_kpi_loss + g·trace_d + λ·g`). Còn `norm_sum` lấy `trace_dis` và `trace_err` chưa nhân cổng, chuẩn hóa theo độ lệch của val rồi cộng với trọng số 1, nên cổng không còn chặn trace trong điểm số. AUROC của riêng `trace_dis`: RE3-OB 0,49–0,61 (trung bình 0,544, gần ngẫu nhiên); RE3-TT 0,29–0,50 (trung bình 0,370, bị đảo chiều: bất thường có lỗi tái tạo trace thấp hơn bình thường); RE2-OB 0,77, RE2-TT 0,73, SN 0,73.
- Cùng các checkpoint đó nhưng chấm bằng luật thô (`raw_sum_*` có sẵn trong `info_score.txt`, không tốn thêm lần chạy), cổng làm đúng việc: UAM-AD − UAC-AD ở RE3-OB là AUROC +0,002, AUPRC −0,001, F1 +0,005; ở RE3-TT là −0,005, −0,018, −0,004.

| UAM-AD − UAC-AD (trung bình 2 seed) | `norm_sum` AUROC / AUPRC / F1 | `raw_sum` AUROC / AUPRC / F1 |
|:--|:--|:--|
| SN | +0,207 / +0,162 / +0,272 | +0,082 / +0,092 / +0,173 |
| RE2-OB | +0,055 / +0,150 / +0,105 | +0,043 / +0,100 / +0,060 |
| RE2-TT | +0,137 / +0,242 / +0,303 | +0,076 / +0,132 / +0,185 |
| RE3-OB | −0,020 / −0,072 / −0,075 | +0,002 / −0,001 / +0,005 |
| RE3-TT | −0,039 / −0,082 / −0,023 | −0,005 / −0,018 / −0,004 |

- **SN, phần lớn mức tăng phụ thuộc seed:** UAC-AD seed 0 sập (AUROC 0,551, F1 0,303), seed 1 phục hồi (AUROC 0,945, F1 0,611). Chỉ tính seed 1, UAM-AD hơn UAC-AD +0,009 AUROC, +0,001 AUPRC, +0,113 F1. Cần các seed 2–4 mới kết luận được mức tăng ở SN. [CN 10-18: seed 2 của cả hai mô hình đã có, UAC-AD seed 2 phục hồi như seed 1, xem phần "Kết quả 3 seed" bên dưới; các số trong khối chẩn đoán này là 2 seed, giữ nguyên làm lịch sử.]
- **SN, `norm_sum` không có số hạng phạt hoạt động:** `activity_penalty_weight` 1,5 chỉ cộng vào `fusion` (luật thô), còn tập số hạng của `norm_sum` là `log_kpi_loss`, `trace_dis`, `trace_err`. UAC-AD đi từ AUROC 0,877 (`raw_sum`) xuống 0,748 (`norm_sum`), UAM-AD gần như không đổi (0,959 → 0,955); UAM-AD còn `trace_err` bù cho lỗi làm service im lặng. Chưa tách được phần do bỏ số hạng phạt với phần do cắt về 0 dưới median của val.
- **Quyết định [CN 10-17]:** giữ `norm_sum` làm kết quả chính (đã đóng băng), đồng thời báo thêm cột `raw_sum` như bảng phụ (số đã có sẵn trong `info_score.txt` của cả 5 dataset và 75 ô DeepTraLog, không chạy thêm); khi viết nói rõ rằng dưới `norm_sum` cổng không chặn số hạng trace trong điểm số.

**Đã xong [CN 10-18 00:24]:** (1) chạy lại UAM-AD SN seed 1 với top-5 để HR@5 đồng bộ với các dataset khác (bản cũ top-3 giữ ở `D:\ClaudeWork\phase3\item8_backup\sn_uamad_seed1_top3`; lần chạy đầu bị hệ thống dừng lúc 15:42 vì thiếu RAM, mới xong 1/12 scenario, lần sau chạy tiếp các ô còn lại); (2) UAC-AD seed 2 trên cả 5 dataset, đã kiểm tra mỗi scenario đủ đúng seed {0, 1, 2}. Chuỗi chạy: `D:\ClaudeWork\phase3\item8\chain_seed2.cmd`, khởi chạy tách khỏi phiên để không bị dừng cùng phiên. Thời gian thật: SN 6 phút, RE3-TT 2398 s, RE3-OB 1717 s, RE2-OB 7165 s, RE2-TT 19037 s. RE2-TT lâu bất thường vì scenario cuối (`socket`) tranh RAM và GPU với game từ 21:14 đến khoảng 23:30 (epoch 0 mất 7534 s thay vì khoảng 5 phút); đây chỉ là số đo thời gian bị nhiễu, kết quả không bị ảnh hưởng.

**Đã xong [CN 10-18 07:52]:** UAM-AD seed 2 trên cả 5 dataset, đã kiểm tra mỗi scenario của cả hai mô hình có đúng seed {0, 1, 2}. Chuỗi chạy `D:\ClaudeWork\phase3\item8\chain_seed2_uamad.cmd` (tự chờ chuỗi UAC-AD xong rồi chạy, 00:24 → 07:52). Thời gian thật: SN 413 s, RE3-TT 3318 s, RE3-OB 2463 s, RE2-OB 8526 s, RE2-TT 12147 s (tổng khoảng 7,4 giờ). Không scenario nào phải chạy lại.

**Kết quả 3 seed (0, 1, 2), cấu hình đóng băng [CN 10-18]:** trung bình qua scenario của từng seed, rồi trung bình±độ lệch chuẩn (tổng thể, `np.std`) qua 3 seed. Đây là kết quả khung chính cho item 8 khi chưa chạy seed 3–4.

`norm_sum` (kết quả chính), F1 / AUROC / AUPRC:

| Dataset | UAC-AD | UAM-AD |
|---|---|---|
| SN | 0,520±0,154 / 0,815±0,187 / 0,600±0,154 | 0,728±0,004 / 0,956±0,002 / 0,709±0,005 |
| RE3-TT | 0,540±0,007 / 0,747±0,010 / 0,433±0,012 | 0,513±0,002 / 0,702±0,004 / 0,346±0,004 |
| RE3-OB | 0,641±0,001 / 0,857±0,001 / 0,692±0,000 | 0,570±0,010 / 0,837±0,004 / 0,622±0,008 |
| RE2-OB | 0,483±0,003 / 0,853±0,005 / 0,667±0,017 | 0,556±0,086 / 0,902±0,016 / 0,797±0,028 |
| RE2-TT | 0,258±0,001 / 0,671±0,002 / 0,447±0,001 | 0,564±0,017 / 0,807±0,001 / 0,688±0,003 |

Bảng phụ `raw_sum` (cùng cách tính; số có sẵn trong `info_score.txt`), F1 / AUROC / AUPRC:

| Dataset | UAC-AD | UAM-AD |
|---|---|---|
| SN | 0,616±0,056 / 0,896±0,051 / 0,624±0,038 | 0,768±0,028 / 0,958±0,002 / 0,711±0,012 |
| RE3-TT | 0,540±0,007 / 0,755±0,010 / 0,441±0,012 | 0,531±0,002 / 0,745±0,003 / 0,416±0,005 |
| RE3-OB | 0,657±0,002 / 0,863±0,001 / 0,715±0,000 | 0,662±0,001 / 0,866±0,004 / 0,715±0,006 |
| RE2-OB | 0,483±0,003 / 0,858±0,005 / 0,669±0,016 | 0,520±0,061 / 0,892±0,027 / 0,736±0,062 |
| RE2-TT | 0,258±0,001 / 0,677±0,004 / 0,449±0,001 | 0,426±0,031 / 0,737±0,021 / 0,561±0,028 |

Hiệu từng cặp UAM-AD − UAC-AD (cùng scenario, cùng seed), trung bình qua mọi cặp; "âm" là số cặp có hiệu < 0:

| Dataset | `norm_sum` F1 / AUROC / AUPRC | số cặp âm (F1 / AUROC / AUPRC) | `raw_sum` F1 / AUROC / AUPRC |
|:--|:--|:--|:--|
| SN | +0,208 / +0,141 / +0,109 | 1/36, 10/36, 11/36 | +0,152 / +0,062 / +0,087 |
| RE2-OB | +0,073 / +0,049 / +0,130 | 4/18, 1/18, 0/18 | +0,037 / +0,034 / +0,067 |
| RE2-TT | +0,306 / +0,135 / +0,241 | 3/18, 6/18, 6/18 | +0,168 / +0,060 / +0,112 |
| RE3-OB | −0,072 / −0,019 / −0,070 | 12/15, 13/15, 15/15 | +0,005 / +0,003 / 0,000 |
| RE3-TT | −0,027 / −0,045 / −0,088 | 12/12, 12/12, 12/12 | −0,009 / −0,010 / −0,025 |

Đọc kết quả (chỉ nêu điều số liệu đỡ được):
- **RE2-OB và RE2-TT: UAM-AD hơn UAC-AD ở cả hai luật.** RE2-TT chắc nhất (AUROC +0,135, AUPRC +0,241; UAC-AD gần như không dao động theo seed, độ lệch chuẩn ≤ 0,002). RE2-OB: AUROC cao hơn ở cả 3 seed (UAM-AD 0,887 / 0,925 / 0,894, UAC-AD 0,854 / 0,847 / 0,859), nhưng F1 của UAM-AD phụ thuộc seed: seed 1 là 0,677, hai seed còn lại 0,495 và 0,496, chỉ hơn UAC-AD (0,480–0,487) khoảng 0,01. Vì vậy F1 của RE2-OB ở UAM-AD (độ lệch chuẩn 0,086) không nên trình bày như mức tăng chắc chắn; AUROC/AUPRC thì chắc hơn.
- **SN: mức tăng phần lớn đến từ một seed của UAC-AD.** UAC-AD sập ở seed 0 (AUROC 0,551, F1 0,303) và chạy bình thường ở seed 1 và 2 (AUROC 0,945 / 0,949). Chỉ tính seed 1–2, UAM-AD hơn UAC-AD +0,097 F1, +0,010 AUROC, +0,002 AUPRC. UAM-AD ổn định (AUROC 0,954–0,959). Mức tăng AUROC +0,141 ở bảng trên bị seed 0 kéo lên; nên báo cả hai cách. Hiện tượng sập này là của UAC-AD (không trace) trên SN, chưa tìm ra nguyên nhân; 5 seed sẽ cho biết tần suất.
- **RE3-OB và RE3-TT: UAM-AD thấp hơn UAC-AD dưới `norm_sum`, ổn định theo seed (độ lệch chuẩn ≤ 0,010), tức là hệ thống chứ không phải nhiễu.** Dưới `raw_sum` (cổng còn tác dụng) chênh lệch gần 0: RE3-OB +0,003 AUROC, 0,000 AUPRC; RE3-TT −0,010, −0,025. Cơ chế đã chẩn đoán ở phần trên (số hạng trace chưa nhân cổng được cộng vào `norm_sum`), số liệu 3 seed không đổi kết luận.
- Quyết định giữ nguyên: `norm_sum` là kết quả chính, `raw_sum` là bảng phụ; luật đóng băng không đổi sau khi xem kết quả test.
- Nhiễu giữa các seed (3 seed 0–2): UAC-AD ổn định ở 4 dataset RCAEval (độ lệch chuẩn AUROC ≤ 0,010); UAM-AD ổn định ở SN, RE3-TT, RE3-OB, RE2-TT (độ lệch chuẩn ≤ 0,017), riêng RE2-OB dao động mạnh hơn (F1 0,086, AUPRC 0,028). **[CN 10-18 14:54] Nhận xét về UAC-AD ở RE2-OB không còn đúng khi có seed 3, xem khối bên dưới.**

**UAC-AD seed 3 đã xong [CN 10-18 14:54]** (08:54 → 14:54, chuỗi `chain_seed3_uacad.cmd`; UAC-AD không có RCA nên không có HR@k). Hai ô bị dừng im lặng giữa chừng (driver vẫn báo `OK`): SN `Svc_Kill_SocialGraph` (08:57, ngay sau dòng "Activity baseline") và RE2-TT `loss` (13:39, sau dòng "Known word number"); cả hai đã chạy lại đúng ô đó (21 s và 1365 s) bằng `chain_seed3_uacad_fix.cmd` và `chain_seed3_uacad_fix2.cmd`. Đã kiểm tra cả 5 dataset có đúng seed {0, 1, 2, 3} ở mọi scenario. **Lưu ý khi chạy các seed sau: luôn đếm seed theo scenario sau mỗi dataset; đây là lần thứ hai trong hai lượt (seed 2, seed 3) gặp ô dừng im lặng.**

Số liệu từng seed (trung bình qua scenario; F1 / AUROC / AUPRC), `norm_sum`:

| Dataset | UAC-AD seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| SN | 0,303 / 0,551 / 0,382 | 0,611 / 0,945 / 0,704 | 0,646 / 0,949 / 0,712 | **0,306 / 0,564 / 0,386** |
| RE3-TT | 0,530 / 0,732 / 0,416 | 0,545 / 0,754 / 0,444 | 0,545 / 0,753 / 0,440 | 0,543 / 0,744 / 0,432 |
| RE3-OB | 0,640 / 0,857 / 0,691 | 0,642 / 0,856 / 0,692 | 0,641 / 0,857 / 0,693 | 0,656 / 0,860 / 0,702 |
| RE2-OB | 0,480 / 0,854 / 0,661 | 0,481 / 0,847 / 0,650 | 0,487 / 0,859 / 0,689 | **0,681 / 0,915 / 0,817** |
| RE2-TT | 0,258 / 0,672 / 0,445 | 0,257 / 0,668 / 0,448 | 0,258 / 0,674 / 0,448 | 0,257 / 0,674 / 0,448 |

UAC-AD 4 seed (0–3), trung bình±độ lệch chuẩn, F1 / AUROC / AUPRC:

| Dataset | `norm_sum` (chính) | `raw_sum` (bảng phụ) |
|---|---|---|
| SN | 0,466±0,162 / 0,752±0,195 / 0,546±0,162 | 0,597±0,060 / 0,879±0,054 / 0,611±0,040 |
| RE3-TT | 0,541±0,006 / 0,746±0,009 / 0,433±0,011 | 0,541±0,006 / 0,754±0,009 / 0,441±0,011 |
| RE3-OB | 0,645±0,007 / 0,858±0,002 / 0,694±0,004 | 0,658±0,003 / 0,864±0,002 / 0,716±0,002 |
| RE2-OB | 0,533±0,086 / 0,869±0,027 / 0,705±0,067 | 0,533±0,086 / 0,874±0,027 / 0,707±0,067 |
| RE2-TT | 0,258±0,001 / 0,672±0,002 / 0,447±0,001 | 0,258±0,001 / 0,678±0,003 / 0,449±0,001 |

Hai điều seed 3 làm thay đổi cách đọc (chưa so UAM-AD ở seed 3 vì UAM-AD chưa chạy seed 3; **không so UAM-AD 3 seed với UAC-AD 4 seed**, chỉ so cùng tập seed):
- **SN: UAC-AD sập ở 2 trong 4 seed (seed 0 và seed 3, AUROC 0,551 và 0,564), bình thường ở seed 1–2 (0,945 / 0,949).** Đây là hiện tượng lặp lại, không phải một lần ngẫu nhiên; UAM-AD ở SN ổn định ở cả 3 seed đã chạy (AUROC 0,954–0,959). Bimodal nên trung bình±độ lệch chuẩn của UAC-AD ở SN (0,752±0,195) không mô tả tốt; nên báo thêm từng seed hoặc trung vị. Nguyên nhân sập chưa tìm ra, chỉ biết nó nằm ở UAC-AD (không trace) trên SN.
- **RE2-OB: UAC-AD cũng có seed "nhảy" như UAM-AD.** Seed 3 của UAC-AD đạt F1 0,681 / AUROC 0,915 / AUPRC 0,817, so với 0,48 / 0,85 / 0,65–0,69 ở seed 0–2; UAM-AD seed 1 cũng nhảy (0,677 / 0,925 / 0,836). Vậy dao động theo seed ở RE2-OB là đặc điểm chung của dataset/nền mô hình, không riêng UAM-AD; lập luận ở khối trên rằng F1 của RE2-OB "phụ thuộc seed" ở UAM-AD cần nói lại là phụ thuộc seed ở cả hai mô hình. Mức tăng của trace ở RE2-OB cần UAM-AD seed 3–4 mới so được công bằng: cùng seed 0–2, UAM-AD hơn UAC-AD +0,049 AUROC, +0,130 AUPRC, +0,073 F1; nếu lấy trung bình 4 seed của UAC-AD (0,869 AUROC) thì mức hơn thu hẹp, nhưng đó là so lệch tập seed.
- RE3-TT, RE3-OB, RE2-TT: seed 3 của UAC-AD khớp seed 0–2 (RE3 độ lệch chuẩn ≤ 0,009, RE2-TT ≤ 0,003), nên kết luận về các dataset này không đổi.

**Chưa làm [CN 10-18 14:54] (đã lỗi thời: UAM-AD seed 3 và seed 4 của cả hai mô hình đã chạy xong ngày 10-19, xem dòng "Bước 8 đã xong đủ 5 seed" ngay dưới; chỉ item 9 còn lại):** UAM-AD seed 3 (chưa chạy, khoảng 7,4 giờ), seed 4 của cả hai mô hình, và item 9. Chờ quyết định của người dùng.

**Bước 8 đã xong đủ 5 seed (0–4) [CN 10-19 12:13]:** UAM-AD seed 3–4 rồi UAC-AD seed 4 (18:35 ngày 10-18 → 12:13 ngày 10-19, chuỗi `D:\ClaudeWork\phase3\item8\chain_seed34.cmd`; mỗi lệnh lặp 3 lần để tự lấp ô chết im lặng, driver bỏ qua ô đã xong). Thời gian thật: UAM-AD seed 3–4: SN 650 s, RE3-TT 5998 s, RE3-OB 4548 s, RE2-OB 15690 s, RE2-TT 22157 s; UAC-AD seed 4: SN 320 s, RE3-TT 2191 s, RE3-OB 1573 s (+394 s chạy bù ô `f4`, chết im lặng lần đầu), RE2-OB 6521 s, RE2-TT 6992 s. **Đã kiểm tra: cả 10 tổ hợp (5 dataset × 2 mô hình) có đúng seed {0, 1, 2, 3, 4} ở mọi scenario, không còn ô thiếu.**

**KẾT QUẢ CUỐI BƯỚC 8, 5 seed [CN 10-19]** (bản tra cứu đầy đủ theo từng seed → dataset → scenario, kèm giờ chạy và HR@1/3/5, nằm ở mục "SUMMARY experiments between UAC-AD and UAM-AD" ở cuối bước 8, ngay trước Giai đoạn 4; cấu hình đóng băng; trung bình qua scenario của từng seed, rồi trung bình±độ lệch chuẩn tổng thể `np.std` qua 5 seed; F1 / AUROC / AUPRC):

`norm_sum` (kết quả chính):

| Dataset | UAC-AD | UAM-AD |
|---|---|---|
| SN | 0,523±0,185 / 0,795±0,194 / 0,594±0,173 | 0,728±0,004 / 0,956±0,002 / 0,716±0,012 |
| RE3-TT | 0,540±0,006 / 0,745±0,008 / 0,432±0,010 | 0,513±0,002 / 0,702±0,003 / 0,347±0,003 |
| RE3-OB | 0,647±0,007 / 0,858±0,001 / 0,696±0,004 | 0,560±0,019 / 0,835±0,005 / 0,615±0,012 |
| RE2-OB | 0,523±0,079 / 0,868±0,025 / 0,707±0,060 | 0,564±0,075 / 0,903±0,013 / 0,799±0,022 |
| RE2-TT | 0,256±0,002 / 0,673±0,003 / 0,448±0,002 | 0,567±0,025 / 0,808±0,002 / 0,688±0,003 |

`raw_sum` (bảng phụ):

| Dataset | UAC-AD | UAM-AD |
|---|---|---|
| SN | 0,612±0,061 / 0,890±0,054 / 0,621±0,041 | 0,743±0,039 / 0,959±0,002 / 0,720±0,015 |
| RE3-TT | 0,540±0,006 / 0,753±0,009 / 0,440±0,010 | 0,532±0,003 / 0,747±0,004 / 0,420±0,006 |
| RE3-OB | 0,659±0,003 / 0,864±0,002 / 0,716±0,002 | 0,662±0,002 / 0,865±0,003 / 0,713±0,005 |
| RE2-OB | 0,523±0,079 / 0,872±0,024 / 0,709±0,060 | 0,519±0,051 / 0,892±0,021 / 0,738±0,048 |
| RE2-TT | 0,256±0,002 / 0,678±0,003 / 0,450±0,001 | 0,412±0,055 / 0,732±0,020 / 0,547±0,035 |

Hiệu từng cặp UAM-AD − UAC-AD (cùng scenario, cùng seed; trung bình | số cặp âm trên tổng; thứ tự F1, AUROC, AUPRC):

| Dataset | `norm_sum` | `raw_sum` |
|:--|:--|:--|
| SN (60 cặp) | +0,205 (7/60), +0,161 (18/60), +0,123 (19/60) | +0,131 (7/60), +0,068 (4/60), +0,098 (6/60) |
| RE2-OB (30) | +0,041 (8/30), +0,036 (4/30), +0,093 (4/30) | −0,004 (14/30), +0,020 (8/30), +0,029 (8/30) |
| RE2-TT (30) | +0,310 (5/30), +0,134 (10/30), +0,240 (10/30) | +0,156 (2/30), +0,054 (8/30), +0,098 (8/30) |
| RE3-OB (25) | −0,087 (22/25), −0,023 (22/25), −0,080 (25/25) | +0,003 (6/25), +0,001 (7/25), −0,003 (13/25) |
| RE3-TT (20) | −0,026 (20/20), −0,042 (20/20), −0,086 (20/20) | −0,007 (16/20), −0,005 (11/20), −0,020 (20/20) |

Điểm số từng seed (trung bình qua scenario, F1 / AUROC / AUPRC), để thấy hai hiện tượng bimodal:

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| SN UAC-AD | 0,303 / 0,551 / 0,382 | 0,611 / 0,945 / 0,704 | 0,646 / 0,949 / 0,712 | 0,306 / 0,564 / 0,386 | 0,751 / 0,967 / 0,784 |
| SN UAM-AD | 0,734 / 0,956 / 0,705 | 0,724 / 0,954 / 0,705 | 0,726 / 0,959 / 0,715 | 0,725 / 0,956 / 0,717 | 0,730 / 0,958 / 0,738 |
| RE2-OB UAC-AD | 0,480 / 0,854 / 0,661 | 0,481 / 0,847 / 0,650 | 0,487 / 0,859 / 0,689 | 0,681 / 0,915 / 0,817 | 0,485 / 0,862 / 0,714 |
| RE2-OB UAM-AD | 0,495 / 0,887 / 0,775 | 0,677 / 0,925 / 0,836 | 0,496 / 0,894 / 0,778 | 0,631 / 0,906 / 0,798 | 0,523 / 0,905 / 0,809 |

Kết luận số liệu đỡ được (chỉ những điều dưới đây, không suy rộng):
- **RE2-TT: UAM-AD hơn UAC-AD rõ và ổn định** ở cả hai luật (`norm_sum` AUROC +0,134, AUPRC +0,240; `raw_sum` +0,054, +0,098; UAC-AD gần như không dao động theo seed).
- **SN: UAM-AD ổn định, UAC-AD dao động hai nhóm.** UAC-AD sập ở seed 0 và 3 (AUROC 0,551 / 0,564, F1 khoảng 0,30), bình thường ở seed 1, 2, 4 (AUROC 0,945–0,967). UAM-AD gần như không đổi (AUROC 0,954–0,959, độ lệch chuẩn 0,002). Mức hơn +0,161 AUROC trung bình **chủ yếu** do hai seed sập của UAC-AD: chỉ tính 3 seed UAC-AD không sập (1, 2, 4), UAM-AD hơn UAC-AD +0,058 F1, +0,003 AUROC, −0,014 AUPRC, tức hai mô hình ngang nhau về AUROC/AUPRC. Giá trị của trace ở SN vì vậy nên diễn đạt là **độ ổn định** (không sập), không phải điểm cao hơn khi UAC-AD chạy bình thường. Nguyên nhân sập của UAC-AD chưa tìm ra.
- **RE2-OB: hơn nhẹ ở AUROC/AUPRC, F1 không phân biệt được.** `norm_sum` AUROC +0,036, AUPRC +0,093 (chỉ 4/30 cặp âm); nhưng F1 +0,041 với độ lệch chuẩn 0,075–0,079 ở cả hai mô hình (cả hai có seed "nhảy" F1 ≈ 0,63–0,68: UAC-AD seed 3, UAM-AD seed 1 và 3), và dưới `raw_sum` F1 −0,004. Không nên khẳng định cải thiện F1 ở RE2-OB.
- **RE3-OB, RE3-TT: UAM-AD thấp hơn UAC-AD dưới `norm_sum`** (RE3-OB AUROC −0,023, AUPRC −0,080; RE3-TT −0,042, −0,086; hầu hết mọi cặp đều âm), ổn định theo seed nên là hệ thống. Dưới `raw_sum` (cổng còn tác dụng) chênh lệch gần 0 ở RE3-OB (+0,001 AUROC, −0,003 AUPRC) và nhỏ ở RE3-TT (−0,005, −0,020). Cơ chế đã chẩn đoán ở phần trên.
- Tóm lại: trace giúp rõ ở RE2-TT, giúp độ ổn định ở SN, giúp nhẹ AUROC/AUPRC ở RE2-OB, và làm giảm điểm ở RE3 dưới luật `norm_sum` đóng băng. Luật đóng băng không đổi sau khi xem kết quả test.

HR@k của UAM-AD, 5 seed (trung bình qua scenario có service lỗi; HR@1 và HR@3 trên seed 0–4, HR@5 chỉ trên seed 1–4 vì seed 0 lưu top-3; trung bình±độ lệch chuẩn qua seed; mức ngẫu nhiên k/N):

| Dataset | Số service | HR@1 | HR@3 | HR@5 (seed 1–4) | Ngẫu nhiên HR@1/3/5 |
|:--|--:|:--|:--|:--|:--|
| SN | 12 | 0,194±0,029 | 0,275±0,033 | 0,528±0,052 | 8% / 25% / 42% |
| RE3-TT | 27 | 0,079±0,015 | 0,112±0,018 | 0,225±0,023 | 4% / 11% / 19% |
| RE3-OB | 11 | 0,023±0,001 | 0,224±0,019 | 0,531±0,032 | 9% / 27% / 45% |
| RE2-OB | 11 | 0,352±0,005 | 0,610±0,007 | 0,736±0,014 | 9% / 27% / 45% |
| RE2-TT | 27 | 0,413±0,040 | 0,583±0,037 | 0,660±0,028 | 4% / 11% / 19% |

Cách đọc HR@k không đổi so với bản 3 seed: RE2-OB và RE2-TT hơn hẳn ngẫu nhiên; SN hơn ngẫu nhiên ở HR@1 và HR@5 nhưng HR@3 ≈ ngẫu nhiên; RE3-TT HR@1 gấp khoảng 2 lần ngẫu nhiên nhưng tuyệt đối thấp, HR@3 ≈ ngẫu nhiên; RE3-OB HR@1 và HR@3 **thấp hơn** ngẫu nhiên (chỉ HR@5 cao hơn), cần ghi như hạn chế. Các bảng 3 seed và 4 seed ở trên giữ làm lịch sử, số cuối là bảng 5 seed này.

Ước tính thời gian (đã lỗi thời, giữ làm lịch sử; số thật ở mục S1 của SUMMARY): UAM-AD khoảng 7,4 giờ mỗi seed, UAC-AD khoảng 6 giờ mỗi seed (seed 3 thật: 6 giờ gồm hai lần chạy lại ô lỗi). Việc còn lại là item 9 gồm AnoFusion, ART, Eadro, OrEdge.

**Việc 8, UAM-AD seed 0 trên 5 dataset: đã chạy xong [CN 10-16]** (1 seed, cấu hình đóng băng, `norm_sum`, `--enable_rca`; trung bình qua scenario của từng dataset; **chưa phải kết quả cuối**, mới 1 seed):

| Dataset | F1@val | Precision | Recall | AUROC | AUPRC | HR@1 | HR@3 | Thời gian |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SN (12 scenario; 9 có service lỗi) | 0,734 | 0,639 | 0,890 | 0,956 | 0,705 | 0,178 | 0,252 | 5 phút |
| RE3-TT (4) | 0,515 | 0,515 | 0,516 | 0,705 | 0,350 | 0,063 | 0,095 | 49 phút |
| RE3-OB (5) | 0,556 | 0,593 | 0,553 | 0,832 | 0,611 | 0,022 | 0,240 | 37 phút |
| RE2-OB (6) | 0,495 | 0,394 | 0,775 | 0,887 | 0,775 | 0,352 | 0,622 | 2,1 giờ |
| RE2-TT (6) | 0,540 | 0,587 | 0,606 | 0,809 | 0,692 | 0,421 | 0,604 | 3,1 giờ |

Tổng 6,7 giờ (24.193 giây), ngắn hơn ước tính 8,3 giờ vì OB chạy nhanh hơn lần chạy tháng 5; không ô nào lỗi. Số seed còn lại của UAM-AD (4 seed) ≈ 27 giờ. Quan sát (1 seed): F1@val ở RE2-OB thấp vì precision thấp (ngưỡng p95 của val đặt thấp, nhiều cảnh báo sai) dù AUROC 0,887; RE3-TT gần ngẫu nhiên (AUROC 0,705, khớp việc đối thủ cũng yếu ở tập lỗi mức mã); xếp hạng service lỗi (HR@k) của UAM-AD thấp, nhất là ở RE3 và SN (HR@1 0,02–0,18), so với số chạy thử của ART trên RE2-OB `cpu` (HR@1 0,96) và Eadro (0,48); hai số này khác phạm vi (chạy thử chỉ một scenario) nên chưa so được. **Hạn chế cần quyết định:** `rca_results.json` chỉ lưu top-3 service (mặc định `--rca_top_k 3`) nên HR@5 của các lần chạy này **không có** (hàm tính HR@5 trong `rca.py` chỉ nhìn danh sách top-k nên sẽ bằng HR@3). Muốn có HR@5 thì phải truyền `--rca_top_k 5` (đã thêm vào các script eval, mặc định 5, áp dụng từ seed 1 trở đi; riêng SN seed 1 là kết quả cũ nên vẫn chỉ có top-3); seed 0 không tính lại được từ checkpoint (xem mục HR@k bên dưới) nên phải chạy lại (≈ 6,7 giờ) hoặc báo HR@5 trên các seed có top-5. Điểm z-score không huấn luyện đạt AUROC 0,96–0,99 trên RE2-OB `cpu/delay/loss` vẫn cao hơn UAM-AD (0,878/0,944/0,857) ở cùng scenario.

## Việc 8: lời dẫn cũ của mục SUMMARY

Mục này là bản tổng hợp để tra cứu của bước 8, đặt ở cuối bước 8 vì các đoạn phía trên ghi theo trình tự làm việc nên khó nhìn tổng thể. Mọi số trong mục này **sinh tự động từ thư mục kết quả** `D:\ClaudeWork\phase3\item8\<dataset>\{uacad,uamad}\<scenario>\<hash>\` (`info_score.txt`, `params.json`, `running.log`, `rca_results.json`), 5 seed (0–4) × 33 scenario × 2 mô hình = 330 lần chạy, không gõ tay. Cấu hình đóng băng (`codes/common/frozen_config.json`), luật chấm `norm_sum`.

## Việc 9: tiến trình các đối thủ (seed 0, chạy thử, sửa mã, quyết định, seed 1–4, kết quả seed 0 và đoạn 5 seed đã thay bằng T1–T4)

#### Việc 9: các đối thủ (ART, AnoFusion, Eadro, OrEdge), chạy từng seed

**Bắt đầu seed 0 [CN 10-19 18:06]** theo yêu cầu "chạy từng seed, trước mắt 1 seed để biết khoảng cách". Bộ chạy `codes/common/run_phase3_item9.py` (chưa commit): chạy tuần tự từng mô hình qua các script ở `codes/competitors/`, ART và AnoFusion một lần huấn luyện cho cả dataset (chấm mọi file test), Eadro và OrEdge một lần cho mỗi file test (3-fold theo run); ô đã có `seed<k>.json` thì bỏ qua; ô kết thúc mà không ghi json được báo `FAILED` (không báo OK nhầm như bộ chạy item 8). Kết quả ở `D:\ClaudeWork\phase3\item9\<mô hình>\<dataset>[\<test>]\seed<k>.json`, nhật ký ở `...\item9\status.txt` và `...\item9\logs\`. Chuỗi `D:\ClaudeWork\phase3\item9\chain_seed0.cmd`: ART (RE3-OB, RE2-OB, RE3-TT, RE2-TT) → AnoFusion (RE2-OB, RE3-OB, 20 epoch) → Eadro (4 dataset) → OrEdge (RE2-OB, RE3-OB), rồi lặp lại cả 4 lệnh một lần để lấp ô chết im lặng.

**Quyết định của người dùng [CN 10-19 tối]:** (1) **Tập trung ART và AnoFusion trước**, Eadro và OrEdge tính sau; (2) chạy cả **SN** cho hai mô hình này (sinh lại dữ liệu có thứ tự nếu cần); (3) **đủ ba modality L+M+T** (viết script log theo service); (4) làm kiểm chẩn đoán trước; (5) **chỉ chạy seed khi người dùng ra lệnh** (đã dừng chuỗi seed 0 lúc 18:38; chẩn đoán và chạy thử nhanh không tính là chạy seed). Mặc định cũ "SN không áp dụng" của tôi **bị thay thế**: đó là hạn chế kỹ thuật đã giải quyết được cho ART và AnoFusion (xem khối SN bên dưới); với Eadro và OrEdge vẫn chưa làm (SN mỗi scenario chỉ có một phiên lỗi, không chia 3-fold theo run được). Mặc định còn lại chưa đổi: AnoFusion RE2-OB, RE3-OB tối đa 20 epoch, **chưa chạy hai TT**; OrEdge chỉ hai OB.

**Ước tính giờ cho 1 seed** (đo = lấy từ lần chạy thử đã ghi ở trên; ngoại suy = tự tính, độ tin thấp hơn):

| Mô hình | Dataset | Cơ sở | Ước tính |
|:--|:--|:--|--:|
| ART | 4 dataset RCAEval | **đã chạy xong seed 0 [CN 10-19 18:29]: RE3-OB 3,5 phút, RE2-OB 7,7, RE3-TT 2,9, RE2-TT 8,6; tổng ≈ 23 phút** (huấn luyện 9–16 epoch, chấm mỗi file test 7–36 giây) | ≈ 0,4 giờ (đã đo) |
| AnoFusion | RE2-OB | đo: 100 s/epoch, MI 45 s; 20 epoch; chấm 6 file test chưa đo | ≈ 0,8 giờ |
| AnoFusion | RE3-OB | đo: 263 s/epoch, MI 151 s (`--micro_batch 8`); 20 epoch | ≈ 2 giờ |
| AnoFusion | RE2-TT (596 kênh) | ngoại suy theo kênh² × số cửa sổ huấn luyện từ RE2-OB: ≈ 16 phút/epoch, 20 epoch ≈ 5,3 giờ + MI ≈ 7 phút + chấm | ≈ 6–8 giờ |
| AnoFusion | RE3-TT (1357 kênh) | ngoại suy như trên: ≈ 27 phút/epoch, 20 epoch ≈ 9 giờ; chưa biết có vừa GPU 3 GB không | ≈ 9–11 giờ |
| Eadro | RE2-OB (6 file) | đo: ≈ 12 phút mỗi file (50 epoch × 3 fold) | ≈ 1,2 giờ |
| Eadro | RE3-OB (5 file) | ngoại suy theo số mẫu | ≈ 0,4 giờ |
| Eadro | RE3-TT (4 file) | ngoại suy từ 9 s/epoch/fold đo ở 3 epoch: ≈ 22 phút mỗi file | ≈ 1,5 giờ |
| Eadro | RE2-TT (6 file) | ngoại suy: ≈ 45 phút mỗi file | ≈ 4,5 giờ |
| OrEdge | RE2-OB (6 file) | đo: 13 s/epoch/fold, 50 epoch × 3 fold ≈ 32 phút mỗi file (dừng sớm patience 15 có thể ngắn hơn) | ≈ 3,2 giờ |
| OrEdge | RE3-OB (5 file) | ngoại suy theo số mẫu | ≈ 1 giờ |
| OrEdge | RE3-TT (4 file) | đo 36 s/epoch/fold: 36 × 50 = 30 phút mỗi fold, **≈ 90 phút mỗi file** (ghi chú cũ ở trên viết "90 phút mỗi fold", có vẻ nhầm đơn vị; sẽ biết chắc khi chạy) | ≈ 6 giờ |
| OrEdge | RE2-TT (6 file) | ngoại suy: ≈ 3 giờ mỗi file | ≈ 18 giờ |

Chuỗi đang chạy (phần mặc định ở trên) khoảng **16 giờ**: ART ≈ 0,7 + AnoFusion OB ≈ 3 + Eadro ≈ 7,6 + OrEdge OB ≈ 4,3. Hai TT của AnoFusion và OrEdge, nếu chạy đủ, thêm khoảng 40 giờ cho một seed, nên chờ quyết định (bỏ, hoặc giảm kênh/epoch và ghi rõ khác tác giả). Sửa lại một ước lượng cũ ở trên: "RE2-TT khoảng 7 phút mỗi epoch" của AnoFusion đã bỏ qua việc RE2-TT có số cửa sổ huấn luyện gấp 3 lần RE3-OB; tính lại là ≈ 16 phút/epoch.

**Kỳ vọng cần nói thật trước khi có số cuối:** các lần chạy thử trên RE2-OB `cpu` cho ART AUROC 0,987–0,993 (cao hơn UAM-AD 0,887 ở cùng scenario) và một điểm z-score không huấn luyện cũng đạt 0,96–0,986; AnoFusion sau 2 epoch AUROC 0,849. Nên khả năng cao ART hơn UAM-AD ở RE2-OB; cần xem lại tính tương đương trước khi diễn giải (ART chấm theo từng mẫu giây; AnoFusion coi cả hệ thống là một instance; oracle F1 của các đối thủ là quét ngưỡng thường, **không point-adjust**, khác cột F1 oracle của UAM-AD).

**SN cho ART và AnoFusion [CN 10-19]: đã dựng dữ liệu có thứ tự và hai runner, chưa chạy seed nào.**
- **Vì sao trước đó không chạy được:** pkl SN không còn thứ tự (`preprocess_sn.py::_to_dict` bỏ các trường bắt đầu bằng `_`, id là md5) và file test chỉ chứa một phần mẫu normal rời rạc, nên đọc theo file thì hầu hết cửa sổ không có cửa sổ liền trước. Dữ liệu thô vẫn ở `D:\AnoMod\SN_data`.
- **Cách làm (không sinh lại pkl, không đổi tập test của UAM-AD):** `codes/common/make_sn_sessions.py` chạy lại đúng lớp `SNPreprocessor` (cùng tham số, cùng thứ tự xử lý nên cùng template Drain3) cho cả 13 phiên và lưu **mọi cửa sổ theo thứ tự thời gian** vào `data/sn/sessions.pkl`, khoá bằng chính md5 của pkl. **Đã kiểm đối chiếu 100%:** train (39), val (57) và 12 file test (32–263 cửa sổ): 0 cửa sổ thiếu, 0 cửa sổ khác (nhãn, KPI, đặc trưng trace, ma trận kề, danh sách log đều giống hệt). Mỗi phiên 39–50 cửa sổ 30 giây; `Code_Stop_*` toàn bộ phiên là lỗi (40–50 cửa sổ), `DB_Redis_*` và `Perf_*` 10 cửa sổ lỗi đầu phiên, `Svc_Kill_*` 4 cửa sổ.
- **Runner:** `codes/competitors/sn_data.py`, `art/run_art_sn.py`, `anofusion/run_anofusion_sn.py`; bộ chạy item 9 gọi chúng khi dataset là `sn` (cần `--datasets sn` hoặc mặc định của ART/AnoFusion). Cùng mô hình, hàm chấm, ngưỡng p95 val, không point-adjust. ART trên SN: instance = 12 service, mỗi instance 4 KPI container + 6 đặc trưng trace + 2 đặc trưng log (11 KPI mức máy chủ bị bỏ, như KPI container khác bị bỏ ở RCAEval; **ART gốc có dùng metric mức node, đây là hạn chế**). AnoFusion trên SN: 59 KPI + 452 template log (chuỗi `service|template` trong train) + 72 đặc trưng trace = 583 kênh.
- **Phát hiện cần quyết định: lỗi được tiêm ngay đầu mỗi phiên.** AnoMod bắt đầu thu 15 giây SAU khi tiêm lỗi, nên các cửa sổ lỗi của `DB_Redis_*`, `Perf_*`, `Svc_Kill_*` nằm ở những cửa sổ đầu phiên và **không có lịch sử trước đó trong dữ liệu**. Nếu không đệm, ART (lịch sử 5 cửa sổ) chỉ chấm được 5 trong 10 cửa sổ lỗi của `DB_Redis_*` và `Perf_*` và 2 trong 4 của `Svc_Kill_*`; AnoFusion (lịch sử 20 cửa sổ) **không chấm được cửa sổ lỗi nào** của 9 trong 12 scenario. Tuỳ chọn `--pad_normal`: đệm lịch sử bằng các cửa sổ cuối của phiên `Normal_Baseline` (hệ thống trước khi tiêm lỗi vốn bình thường), đệm giống nhau cho mọi cửa sổ thiếu lịch sử bất kể nhãn; khi đó chấm đủ mọi cửa sổ như UAM-AD. Nhược điểm: lịch sử giả này là "bình thường" nên có thể giúp đối thủ thấy thay đổi rõ hơn UAM-AD (UAM-AD không có lịch sử). **Đề xuất: dùng `--pad_normal` làm cấu hình chính cho AnoFusion (bắt buộc, nếu không thì không báo được), và báo ART cả hai cách; ghi rõ trong luận văn. Chờ quyết định.**
- **Thang đo:** val/test SN lấy từ phiên khác train nên kênh hằng trong train (max 0) bị nhân tới 1e8: lần chạy thử đầu AnoFusion cho val MSE 1,9e14. Hai runner SN dùng `scale = 1` cho kênh max 0 và cắt đầu vào ±100 (`--clip 100`, như OrEdge); khác tác giả (họ chuẩn hoá mỗi tập theo max riêng). Sau sửa val MSE 0,42 (bình thường).
- **Ngân sách huấn luyện AnoFusion trên SN:** train chỉ có 19 cửa sổ có đủ lịch sử, batch mặc định 64 của tác giả không cho bước nào, nên dùng batch 4 (4 bước mỗi epoch); 100 epoch ≈ 400 bước với lr 1e-5, có nguy cơ chưa hội tụ; cần chốt số epoch (xem đường val MSE khi chạy). **Chạy thử (2 và 3 epoch, không phải kết quả):** cả hai runner chạy hết 12 scenario; MI của AnoFusion mất 146 giây; ước tính chạy đủ: ART SN ≈ 1–2 phút, AnoFusion SN ≈ 12 phút (MI 2,5 phút + 100 epoch × 4,5 giây + chấm).

**Quyết định về SN và số epoch [CN 10-19 đêm, người dùng chốt; thay các đề xuất cũ ở các gạch đầu dòng "Phát hiện cần quyết định" và "Ngân sách huấn luyện AnoFusion" ngay trên và bảng ước tính giờ 20 epoch ở trên].**
- **Epoch = của UAM-AD cho mọi mô hình, "không thêm gì cả":** 5 epoch trên RCAEval (RE2/RE3 OB/TT), 50 epoch trên SN (`frozen_config.json`). Không còn chạy thêm bản 20 epoch của AnoFusion để kiểm độ nhạy. Bộ chạy `run_phase3_item9.py` đặt sẵn `EPOCHS = {"sn": 50}`, mặc định 5, áp cho cả ART và AnoFusion (`--epochs` vẫn ghi đè được). **Hệ quả cho ART:** các số ART M+T chạy trước đó (9–16 epoch do dừng sớm theo tác giả, không log) đã **bỏ khỏi mọi bảng theo yêu cầu người dùng [CN 10-20]**; ART chỉ báo bản L+M+T, 5 epoch (50 trên SN). Hệ quả cho AnoFusion: giờ một seed giảm còn khoảng 1/4 so với bảng 20 epoch (RE2-OB ≈ 0,2 giờ, RE3-OB ≈ 0,5 giờ; hai TT ước ≈ 1,5 và ≈ 2,5 giờ, chưa rõ RE3-TT 1357 kênh có vừa GPU 3 GB không; chưa quyết có chạy TT).
- **Vì sao ART và AnoFusion "chạy không được" trên SN trong khi UAM-AD chạy được (nguyên nhân là dữ liệu, không phải cách xử lý):** UAM-AD chấm từng cửa sổ riêng lẻ, không cần lịch sử. ART (5 cửa sổ trước) và AnoFusion (20 cửa sổ trước) là mô hình **dự đoán** cửa sổ hiện tại từ lịch sử, nên cần các cửa sổ liền trước. Mỗi phiên SN là một lần thu riêng **bắt đầu sau khi lỗi đã được tiêm** (AnoMod thu sau tiêm 15 giây), nên không có đoạn bình thường trước lỗi như trong các chuỗi liên tục mà hai mô hình này được thiết kế cho (và như RCAEval). Nhãn theo thời gian (đếm từ `sessions.pkl`):

| Nhóm scenario (phiên) | Số cửa sổ/phiên | Cửa sổ lỗi | Chấm được nếu không đệm: ART (cần 5) | AnoFusion (cần 20) |
|:--|--:|:--|--:|--:|
| `Code_Stop_*` (3 phiên) | 50, 40, 40 | **cả phiên** (50, 40, 40) | 45, 35, 35 | 30, 20, 20 |
| `DB_Redis_*` (3), `Perf_*` (3) | 39–40 | 10 cửa sổ đầu | 5 trên 10 | 0 trên 10 |
| `Svc_Kill_*` (3) | 39 | cửa sổ 4–7 (3 cửa sổ bình thường ở trước) | 2 trên 4 | 0 trên 4 |

  Hệ quả: nếu không đệm thì AnoFusion không có cửa sổ lỗi nào để chấm ở 9 trong 12 scenario.
- **Cách xử lý chốt: đệm lịch sử bằng đuôi phiên Normal_Baseline (`--pad_normal`), bật mặc định cho SN trong bộ chạy.** Lý do: (1) đó là điều xảy ra trong thực tế và trong giao thức gốc của hai mô hình, hệ thống bình thường ngay trước khi lỗi xuất hiện; (2) chỉ dùng **dữ liệu train** (chính 39 cửa sổ Normal_Baseline mà UAM-AD cũng học), không dùng thông tin test hay nhãn, và mọi cửa sổ thiếu lịch sử nhận cùng một kiểu đệm; (3) mọi cửa sổ trong file test của UAM-AD đều được chấm, nên hai bên chấm đúng cùng một tập test và không phải chạy lại UAM-AD. **Sửa một nhận định trước đó:** tôi từng nói đệm "không công bằng với UAM-AD"; nói quá, vì đệm không đưa thêm thông tin ngoài dữ liệu train mà UAM-AD cũng có. Phần còn đúng: đối thủ được dùng dữ liệu bình thường đó **dưới dạng chuỗi liền trước cửa sổ**, UAM-AD thì dùng nó dưới dạng phân phối đã học.
- **Hai giới hạn phải ghi khi báo kết quả SN của hai mô hình này:** (a) đệm chỉ giúp vài cửa sổ đầu phiên (5 cho ART, 20 cho AnoFusion); từ đó trở đi lịch sử là chính các cửa sổ lỗi của phiên, và mô hình dự đoán đo "bất ngờ" chứ không đo "bất thường", nên với lỗi kéo dài (nhất là 3 phiên `Code_Stop_*` mà cả phiên là lỗi) kết quả của chúng có thể thấp vì bản chất mô hình, không phải vì cài đặt (chưa đo); (b) **đề xuất, chưa được người dùng duyệt:** với ART báo thêm bản không đệm (chỉ chấm các cửa sổ có đủ lịch sử thật) như một dòng độ nhạy, vì rẻ (1–2 phút) và cho thấy việc đệm có làm số đổi không; AnoFusion không có bản không đệm vì không đủ cửa sổ.

**Chạy thử nhanh [CN 10-19 22:59–23:53, KHÔNG phải kết quả cuối]: 1 scenario nhỏ mỗi dataset, seed 0.** Theo yêu cầu "test thử nhanh 1 cái nhỏ trong mỗi dataset". Scenario chọn theo kích thước file test nhỏ nhất, không nhìn kết quả: RE2-OB `cpu`, RE3-OB `f2`, RE2-TT `cpu` (cả 6 file bằng nhau, lấy cái đầu), RE3-TT `f1` (cả 4 file bằng nhau), SN `Svc_Kill_Media` (32 cửa sổ, chỉ 4 cửa sổ lỗi). Cấu hình: ART L+M+T và AnoFusion theo quyết định ở trên (5 epoch RCAEval, 50 epoch SN, SN có `--pad_normal`); AnoFusion chưa chạy trên hai TT. Kết quả ở `D:\ClaudeWork\phase3\item9_quick\` (chuỗi `chain.cmd`, `status.txt` toàn OK). UAM-AD và UAC-AD là seed 0 của bước 8 (`norm_sum`) cùng scenario. "Cả lần chạy" = thời gian thật của lần chạy (UAM-AD/UAC-AD: log bước 8; ART/AnoFusion: bộ chạy, gồm nạp dữ liệu, huấn luyện, MI và chấm đúng 1 file test).

| Dataset / scenario | Mô hình | F1@val95 | AUROC | AUPRC | HR@1 / HR@3 / HR@5 | Cả lần chạy (giây) |
|:--|:--|--:|--:|--:|:--|--:|
| RE2-OB `cpu` | UAM-AD | 0,557 | 0,878 | 0,703 | 0,226 / 0,557 / – | 1266 |
| | UAC-AD | 0,494 | 0,849 | 0,486 | – | 1168 |
| | ART | 0,846 | 0,979 | 0,943 | 0,925 / 0,957 / 0,977 | 182 |
| | AnoFusion | 0,355 | 0,672 | 0,463 | – | 675 |
| RE3-OB `f2` | UAM-AD | 0,866 | 0,953 | 0,841 | 0,000 / 0,269 / – | 427 |
| | UAC-AD | 0,891 | 0,966 | 0,914 | – | 311 |
| | ART | 0,755 | 0,971 | 0,902 | 0,933 / 0,945 / 0,961 | 61 |
| | AnoFusion | 0,386 | 0,626 | 0,456 | – | 1643 |
| RE2-TT `cpu` | UAM-AD | 0,430 | 0,717 | 0,508 | 0,296 / 0,527 / – | 1909 |
| | UAC-AD | 0,078 | 0,493 | 0,192 | – | 1418 |
| | ART | 0,255 | 0,725 | 0,388 | 0,435 / 0,521 / 0,652 | 186 |
| | AnoFusion | chưa chạy | | | | |
| RE3-TT `f1` | UAM-AD | 0,523 | 0,720 | 0,362 | 0,065 / 0,104 / – | 721 |
| | UAC-AD | 0,528 | 0,739 | 0,425 | – | 504 |
| | ART | 0,215 | 0,659 | 0,366 | 0,373 / 0,457 / 0,533 | 75 |
| | AnoFusion | chưa chạy | | | | |
| SN `Svc_Kill_Media` | UAM-AD | 0,750 | 0,962 | 0,679 | 0,000 / 0,000 / – | 20 |
| | UAC-AD | 0,000 | 0,154 | 0,133 | – | 36 |
| | ART (pad) | 0,500 | 0,795 | 0,417 | 0,000 / 0,500 / 1,000 (4 cửa sổ) | 23 |
| | AnoFusion (pad) | 0,000 | 0,509 | 0,195 | – | 365 |

Đọc kết quả (chỉ những điều số liệu đỡ được; một scenario, một seed, SN chỉ 4 cửa sổ lỗi nên rất nhiễu):
- **ART L+M+T hơn UAM-AD rõ ở RE2-OB** (AUROC 0,979 so với 0,878; AUPRC 0,943 so với 0,703), **hơn nhẹ ở RE3-OB về AUROC** (0,971 so với 0,953) nhưng F1@val thấp hơn (0,755 so với 0,866), **ngang ở RE2-TT về AUROC** (0,725 so với 0,717) nhưng F1 và AUPRC thấp hơn (0,255 và 0,388 so với 0,430 và 0,508), **thấp hơn ở RE3-TT** (AUROC 0,659 so với 0,720; AUPRC 0,366 so với 0,362 là ngang) và **thấp hơn ở SN** (0,795 so với 0,962). Tức là bức tranh **không phải "ART hơn UAM-AD ở mọi dataset"**. **HR@k của ART cao hơn nhiều ở mọi dataset RCAEval** (HR@1 0,925 so với 0,226 ở RE2-OB; 0,373 so với 0,065 ở RE3-TT), đúng như hạn chế cấu trúc của UAM-AD đã nêu.
- **AnoFusion thấp hơn UAM-AD ở cả 3 scenario đã chạy** (AUROC 0,672 so với 0,878; 0,626 so với 0,953; 0,509 so với 0,962 ở SN, tức ngang ngẫu nhiên). **Không thể quy cho "huấn luyện thiếu epoch" một cách đơn giản:** ở RE2-OB `cpu`, val MSE vẫn giảm đến epoch 5 (0,0589, trùng với lần chạy cũ), nhưng lần chạy thử cũ dừng ở epoch 2 cho AUROC 0,849 và F1@val 0,749 (ghi ở dòng "AnoFusion: đã dựng" ở trên) còn lần này chọn epoch có val MSE nhỏ nhất (epoch 5) cho AUROC 0,672. Tức **huấn luyện lâu hơn làm sai số trên val giảm nhưng phát hiện xấu đi** ở scenario này. Không chọn epoch theo kết quả test (sẽ là chỉnh theo test); nguyên nhân chưa chẩn đoán (nghi: điểm "max qua kênh" bị chi phối bởi vài kênh nhiễu khi mô hình khớp tốt phần còn lại). Trên SN mô hình chỉ có 19 cửa sổ huấn luyện và epoch tốt nhất là 50 (cuối), val MSE gần phẳng (0,3315).
- **Chi phí:** ART rẻ hơn UAM-AD khoảng 7–10 lần ở bốn dataset RCAEval (cả lần chạy 61–186 giây so với 427–1909 giây; huấn luyện riêng 38–128 giây), ngang ở SN (23 so với 20 giây); AnoFusion trên RE3-OB (1643 giây) tốn gấp gần 4 lần UAM-AD (427 giây). Lưu ý UAM-AD/UAC-AD ở đây gồm cả bước RCA (UAM-AD) và chấm đúng một scenario, còn ART/AnoFusion chấm một file test.
- **Chưa làm / chưa quyết:** AnoFusion trên hai TT (ước ≈ 1,5 giờ RE2-TT, ≈ 2,5 giờ RE3-TT, RE3-TT có thể không vừa GPU 3 GB); chưa chạy seed 0 đầy đủ (5 hay 6 scenario mỗi dataset) cho cả hai mô hình.

**Seed 0 đầy đủ của ART, AnoFusion, Eadro và OrEdge: tiến độ và cách đọc [CN 10-23 09:00; AnoFusion RE3-TT chỉ có `test_f1`, đã dừng; Eadro và OrEdge xong lúc 02:33].** Chuỗi `D:\ClaudeWork\phase3\item9\chain_seed0_full.cmd` (ART, AnoFusion) chạy từ 00:14; chuỗi `chain_eadro_oredge.cmd` (Eadro, OrEdge) chạy 23:39 ngày 10-22 đến 02:33 ngày 10-23. Cấu hình: 5 epoch RCAEval, 50 epoch SN; SN có `--pad_normal` cho ART và AnoFusion, Eadro và OrEdge chạy SN không lịch sử. Bảng ngay dưới do `summary_tools/item9_tables.py` sinh từ các json đã có.

| Mô hình | Dataset | Trạng thái | Giờ chạy |
|:--|:--|:--|--:|
| ART | RE3-OB, RE2-OB, RE3-TT, RE2-TT, SN | xong, đủ mọi scenario | 2,0 / 6,0 / 2,3 / 6,8 / 0,4 phút |
| AnoFusion | SN | xong (12 scenario) | 11 phút |
| AnoFusion | RE2-OB | xong (6 scenario) với `--clip 100` và mã tăng tốc, 10:50–11:09. Lần đầu (không clip) **LỖI** `ValueError: Input contains NaN` ở `test_disk` (3 kênh có max train 0 bị nhân 1e8, giá trị tới 1,3e18); json mất, hai số in trong log (`cpu` 0,6724, `delay` 0,8873) không dùng. Lần chạy lại 10:15 bị hệ thống dừng ở epoch 2 vì thiếu RAM. Val MSE epoch 1 và 2 của lần dừng (mã cũ) và lần xong (mã mới) khớp: 0,139253 / 0,0791, nên mã tăng tốc cho cùng kết quả trên dữ liệu thật | 18,8 phút (MI 47 giây, 5 epoch 6,5 phút, chấm 6 file) |
| AnoFusion | RE3-OB | xong (5 scenario) chạy lại với `--clip 100` và mã tăng tốc, 11:51–12:39 (bản cũ không clip, 56 phút, giữ ở `anofusion_noclip\`) | 48 phút (MI 151 giây, huấn luyện 14,7 phút) |
| AnoFusion | RE2-TT | xong (6 scenario) chạy lại với `--clip 100` và mã tăng tốc, 12:39–16:20, `--micro_batch 4 --eval_batch 32` (bản cũ không clip, 5,5 giờ, giữ ở `anofusion_noclip\`) | 3,7 giờ (MI 8,5 phút, huấn luyện 76,6 phút, mỗi file test khoảng 22 phút) |
| AnoFusion | RE3-TT | **ĐÃ DỪNG theo lệnh người dùng lúc 19:14 ngày 10-22; chỉ có `test_f1` (1 trong 4 scenario)**. Cấu hình: 1357 kênh, `--micro_batch 1 --eval_batch 8 --clip 100`, 5 epoch, chuỗi `chain_re3tt.cmd`, bắt đầu 17:19 ngày 10-20. Thời gian đo thật: MI 23 phút; huấn luyện 5 epoch 20,2 giờ (13068, 13844, 14667, 15650, 15511 giây mỗi epoch; val MSE 0,0151, 0,0067, 0,0063, 0,0059, 0,0058); val cuối 1,7 giờ; `test_f1` 20,2 giờ (≈ 2,9 giây/mẫu); `test_f2` chạy được 9736/≈25000 mẫu (2,7 giây/mẫu) thì dừng. Tổng 2,1 ngày chạy. So với ước 14 giờ ban đầu (benchmark dữ liệu giả): huấn luyện chậm 3,4 lần, chấm chậm 11 lần; nguyên nhân chưa rõ (nghi bộ nhớ GPU gần đầy, 2967/3072 MiB, chưa kiểm). Ước còn lại nếu chạy tiếp: `test_f2` xong sáng 10-23, `test_f4` xong đầu giờ 10-25 (≈ 4,3 ngày tổng). **Kết quả `test_f1`: F1@val95 0,116, AUROC 0,502, AUPRC 0,206, F1 oracle 0,351** (`item9\anofusion\rcaeval_re3_tt\seed0.partial.json`, bản sao `seed0.f1_only.backup.json`). Mô hình chưa lưu ra đĩa nên `test_f2`, `test_f3`, `test_f4` không chấm lại được nếu không huấn luyện lại 20 giờ. **Không suy `test_f2`–`test_f4` từ riêng `test_f1`** (AUROC giữa các scenario cùng dataset lệch rất xa: RE2-OB 0,663–0,992, RE3-OB 0,515–0,829, RE2-TT 0,418–0,988). Người dùng chọn (10-22) không chạy thêm AnoFusion, và dùng một **ước lượng thống kê, không phải số đo**, ghi riêng ở đoạn "Ước lượng thống kê cho AnoFusion RE3-TT" ngay dưới bảng này; ước lượng đó không vào các bảng kết quả. Bảng sinh tự động ghi hàng này là "CHỈ f1, dừng sớm" | chỉ f1 |
| Eadro (có giám sát) | RE2-OB, RE3-OB, RE3-TT, RE2-TT, SN | xong, đủ mọi scenario, mọi ô `rc=0` (23:39 ngày 10-22 đến 00:43 ngày 10-23); mô hình epoch cuối | 12,7 / 4,5 / 11,8 / 34,5 / 1,1 phút (tổng 64,6) |
| OrEdge (bán giám sát) | RE2-OB, RE3-OB, RE3-TT, RE2-TT, SN | xong, đủ mọi scenario, mọi ô `rc=0` (00:43 đến 02:33 ngày 10-23); mô hình epoch cuối, log đếm template theo nút | 31,1 / 11,1 / 33,2 / 32,8 / 1,9 phút (tổng 110,1) |

**Ước lượng thống kê cho AnoFusion RE3-TT `f2`–`f4` (KHÔNG phải số đo; người dùng chọn 10-22 vì hết thời gian chạy AnoFusion; mã `scratchpad/estimate_re3tt_anofusion.py`, chỉ đọc json).** Hồi quy tuyến tính AUROC của AnoFusion theo AUROC của ART và của UAM-AD trên cùng scenario, học từ 17 scenario seed 0 của RE2-OB, RE3-OB, RE2-TT (hệ số: chặn −0,064, ART 0,599, UAM-AD 0,321; tương quan riêng với ART 0,60, với UAM-AD 0,58). Sai số bỏ-một-ra (leave-one-out) RMSE **0,162 AUROC**, gần bằng việc đoán bằng trung bình (0,176), tức hồi quy hầu như không hơn đoán trung bình. Áp vào RE3-TT: `f2` ≈ 0,547 (ART 0,649; UAM-AD 0,690), `f3` ≈ 0,633 (ART 0,776; UAM-AD 0,723), `f4` ≈ 0,552 (ART 0,659; UAM-AD 0,686), mỗi số ±0,16. Cùng với `f1` đã đo 0,502: trung bình 4 scenario ≈ 0,558 (nếu ba sai số độc lập thì ±0,07; thực tế sai số cùng dataset có thể tương quan nên rộng hơn). **Cách dùng:** chỉ ghi trong phần thảo luận hoặc hạn chế, dạng "ước lượng ngoại suy, không phải đo, sai số khoảng ±0,16 mỗi scenario"; không đặt vào bảng kết quả cạnh số đo, không báo F1@val95, AUPRC, HR (không ước lượng được). Kết luận duy nhất đỡ được: AnoFusion trên RE3-TT nhiều khả năng ở mức 0,5 đến 0,65, thấp hơn UAM-AD (0,705) và ART (0,686) trung bình; điều này khớp với `f1` đã đo (0,502 so với 0,720 và 0,659). Giới hạn: chỉ 17 điểm; ba dataset dùng để học đều có số kênh thấp hơn RE3-TT; AnoFusion đủ kênh có thể kém hơn bản hồi quy nếu 5 epoch chưa đủ cho 1357 kênh (val MSE epoch 5 vẫn giảm 1,5%).

**Lỗi thang đo của AnoFusion trên RCAEval [CN 10-20, đã đo bằng `scratchpad/check_scale_re2ob.py`, chỉ đọc dữ liệu]: có ở cả 4 dataset, mới sửa cho RE2-OB.** Runner dùng thang của tác giả: chia mỗi kênh cho max train, sàn 1e-8. Kênh có max train bằng 0 mà khác 0 ở test thì bị nhân 1e8 (val không bị). Số kênh max train bằng 0: RE2-OB 55/193 (13 kênh KPI), RE3-OB 40/465 (5), RE2-TT 59/596 (19), RE3-TT 56/1357 (16). Giá trị lớn nhất sau chuẩn hoá trong các file test: RE2-OB 1,3e18 (`test_disk`, 3 kênh có giá trị thô tới 1,28e10; các file khác 1e13), RE3-OB 2,5e8, RE2-TT 5e7, RE3-TT 1,8e8. Số giá trị vượt 100 trong một file test, lớn nhất: RE2-OB 10.563, RE3-OB 1.710, RE2-TT 6.277, RE3-TT 10.697. Chỉ RE2-OB tràn float32 thành NaN. **Hệ quả (sửa lại [CN 10-20, sau khi đo mức ảnh hưởng]; câu cũ "có thể thấp hơn giá trị thật" là suy đoán, không có căn cứ về chiều):** các giá trị phình to hầu như chỉ rơi vào cửa sổ **bất thường** (kênh luôn bằng 0 trong train bắt đầu hoạt động khi có lỗi), nên cắt ±100 chỉ đổi đúng phần tín hiệu dễ phát hiện nhất; chiều ảnh hưởng lên chỉ số **chưa biết**. Tỉ lệ cửa sổ được chấm có chứa mẫu bị đổi (cửa sổ = 20 mẫu trước + mẫu đích; `scratchpad/clip_reach.py`), trong ngoặc là tỉ lệ trong cửa sổ bất thường: RE2-OB `cpu` 0,23% (1,1%), `delay` 0,12% (0,6%), `disk` **20,2% (97,7%)**, `loss` 7,7% (37,3%), `mem` 7,6% (36,8%), `socket` 0,24% (1,1%); RE3-OB `f1` 6,7% (32,8%), `f2` 0, `f3` 0,64% (3,1%), `f4` 0,67% (3,2%), `f5` 3,6% (17,3%); RE2-TT `cpu` 0,22% (0,65%), `delay` 0,32% (1,3%), `disk` **12,1% (58,0%)**, `loss` 0,16% (0,4%), `mem` 0,12% (0,2%), `socket` 1,06% (4,7%); RE3-TT `f1` 1,3% (6,3%), `f2` 1,4% (6,8%), `f3` **10,6% (52,0%)**, `f4` 1,5% (7,4%). Cửa sổ bình thường bị đổi: 0,00% (RE2-OB, RE3-OB, RE3-TT), 0,05 đến 0,10% (RE2-TT); val gần như 0. Đo duy nhất có cả hai bản (RE2-OB, mã cũ không clip so với mã mới có clip): `cpu` AUROC 0,6724 → 0,6634 và `delay` 0,8873 → 0,8818, tức thấp hơn nhẹ, cùng bậc với 1,1% và 0,6% cửa sổ bất thường bị đổi; chưa tách được ảnh hưởng của clip và sai số làm tròn của mã mới (log lần chạy đầu đã bị ghi đè). Bản sửa: `run_anofusion.py --clip 100` (kênh max 0 giữ thang 1, cắt ±100, cùng cách với runner SN và `run_art_sn.py`), mặc định tắt (để tái lập bản cũ); `run_phase3_item9.py` mới bật cho RE2-OB, RE3-OB, RE2-TT (RE3-TT chưa chạy nên chưa bật). **Đã chạy lại RE3-OB và RE2-TT với clip (xem đoạn "Chạy lại" bên dưới): AUROC trung bình đổi không quá 0,002 (RE3-OB 0,6317 → 0,6304; RE2-TT 0,7748 → 0,7735), từng scenario lệch tối đa 0,009 và 0,003, có cái lên có cái xuống.** Vậy với seed 0 clip chỉ chặn tràn số, không đổi kết luận. ART cũng chạy không cắt (theo tác giả); với bản M+T cũ, cắt không đổi điểm phát hiện (đã đo ở RE2-OB, RE3-OB), nhưng chưa kiểm trên L+M+T.

**Tăng tốc AnoFusion, kết quả toán học không đổi [CN 10-20, đã sửa mã `anofusion_model.py`, `run_anofusion.py`; đã đo tốc độ trên GPU ở bảng dưới].** Mô hình gốc tính lại cho từng mẫu những thứ không phụ thuộc mẫu: (1) GTN và chuẩn hoá đồ thị chỉ phụ thuộc ma trận MI và trọng số, nay tính một lần mỗi bước tối ưu (huấn luyện: lan truyền gradient về đồ thị một lần sau các micro-batch) hoặc mỗi lượt chấm; (2) ma trận năng lượng attention `h·hᵀ` của mỗi head (tích N³, phần đắt nhất) không phụ thuộc loại cạnh, trước tính lại 6 lần, nay một lần. Kiểm trên CPU (mô hình 40 kênh, trọng số GTN khởi tạo ngẫu nhiên, `scratchpad/check_graph_cache.py`): gradient lệch tương đối tối đa 5,5e-6, đầu ra lệch 4,8e-7, tức sai số làm tròn. Lần chạy SN dùng mã cũ; RE2-OB, RE3-OB, RE2-TT dùng mã mới (hai bản cũ của RE3-OB và RE2-TT giữ ở `anofusion_noclip\`); mã mới tương đương tới sai số làm tròn. Thời gian thật của hai lần chạy lại (có cả clip): RE3-OB 48 phút (ước ≈ 42), RE2-TT 3,7 giờ (ước ≈ 4,2); huấn luyện RE2-TT 76,6 phút so với 128 phút ở mã cũ. **Đo trên GTX 1050 3 GB, dữ liệu giả cùng cỡ, mỗi chế độ một tiến trình mới** (`scratchpad/bench_anofusion.py`; chạy nối tiếp trong cùng tiến trình cho số sai vì bộ nhớ GPU đã đầy sau lần huấn luyện). Cấu hình micro-batch và batch chấm như bộ chạy:

| Dataset (số kênh) | Giây mỗi mẫu huấn luyện, cũ → mới | Giây mỗi mẫu chấm, cũ → mới | Một seed: cũ → mới |
|:--|:--|:--|:--|
| RE3-OB (465) | 0,063 → 0,042 | 0,019 → 0,014 | 56 phút (đo thật) → ước ≈ 42 phút |
| RE2-TT (596) | 0,132 → 0,079 | 0,034 → 0,026 | 5,5 giờ (đo thật) → ước ≈ 4,2 giờ |
| RE3-TT (1357, micro-batch 1) | **8,73 → 0,90** | 0,361 → 0,251 | ước ≈ 58 giờ → ước ≈ 14 giờ |

Ở RE3-TT bản cũ chạm đỉnh 3035/3072 MiB bộ nhớ GPU và chậm gấp khoảng 10 lần so với số phép tính; bản mới 2655 MiB. Nguyên nhân chậm là bộ nhớ gần đầy (nghi, suy từ đỉnh bộ nhớ, chưa kiểm riêng). Ước một seed RE3-TT mới: huấn luyện 4,6 giờ (55 phút mỗi epoch), val 1,4 giờ (6 lượt), chấm 4 file test khoảng 7,5 giờ (108 nghìn mẫu), MI 0,4 giờ; tổng ≈ 14 giờ, **chưa chạy**. Hai ước lượng trước của tôi (10–13 giờ, rồi 20–30 giờ) đều quá thấp vì ngoại suy theo số phép tính chứ chưa đo. Chấm điểm (7,5 giờ) là phần lớn, và tôi chưa thấy cách tăng tốc chính xác nào thêm (phép nhân ma trận đo được khoảng 1,4 TFLOPS trên GPU này; chưa thử cách nào khác); muốn dưới vài giờ phải giảm số kênh (ví dụ chỉ giữ K template log phổ biến nhất trong train, 868 kênh log là phần lớn 1357 kênh), đổi so với tác giả nên cần ghi rõ và cần người dùng duyệt; chi phí tỉ lệ khoảng N³ nên N ≈ 600 cho khoảng 1 đến 2 giờ.

**Chạy lại AnoFusion RE3-OB rồi RE2-TT với `--clip 100` và mã tăng tốc [CN 10-20 11:51–16:20, người dùng ra lệnh; XONG, cả hai lần chạy đều `rc=0`].** Chuỗi `D:\ClaudeWork\phase3\item9\chain_clip_rerun.cmd` (log `logs\anofusion_rcaeval_re3_ob_all_seed0.log`, `..._re2_tt_...`). RE3-OB 11:51–12:39 (48 phút), RE2-TT 12:39–16:20 (3,7 giờ). **Kết quả cũ không clip (mã cũ) được giữ ở `item9\anofusion_noclip\` (json và log, đã kiểm hash giống bản gốc)**. So bản cũ và mới, trung bình qua scenario (cả hai cùng `best_epoch` 5):

| Dataset | AUROC cũ → mới | F1@val95 cũ → mới | AUPRC cũ → mới | Lệch lớn nhất theo scenario (AUROC) |
|:--|:--|:--|:--|--:|
| RE3-OB | 0,6317 → 0,6304 | 0,3686 → 0,3658 | 0,4525 → 0,4466 | 0,0088 |
| RE2-TT | 0,7748 → 0,7735 | 0,5851 → 0,5839 | 0,6723 → 0,6719 | 0,0026 |

AUROC theo scenario (cũ → mới): RE3-OB `f1` 0,831 → 0,829, `f2` 0,626 → 0,627, `f3` 0,518 → 0,515, `f4` 0,521 → 0,527, `f5` 0,663 → 0,654; RE2-TT `cpu` 0,856 → 0,855, `delay` 0,655 → 0,652, `disk` 0,988 → 0,988, `loss` 0,420 → 0,418, `mem` 0,954 → 0,954, `socket` 0,775 → 0,774. Chiều lệch không nhất quán (RE3-OB `f2`, `f4` tăng; các cái khác giảm hoặc giữ), và độ lớn cùng bậc với sai số làm tròn của mã mới, nên **tôi không tách được ảnh hưởng của clip khỏi sai số làm tròn; kết luận duy nhất đỡ được là clip không đổi điểm đáng kể trên hai dataset này, dù 6,7% (RE3-OB `f1`) đến 12% (RE2-TT `disk`) cửa sổ được chấm có chứa mẫu bị cắt**. Hai điều chưa kiểm: thang của chính tác giả (mỗi split chia cho max của chính nó) có thể cho kết quả khác cả hai bản này; điểm RE3-OB thấp (0,630) có thể do luật 5 epoch (val MSE vẫn giảm 17% ở epoch 5: 0,0305 → 0,0252), chưa chạy thêm epoch để kiểm vì người dùng cố định 5 epoch. Thay đổi công cụ để chạy lại an toàn: `item9_tables.py::seconds` lấy giờ của lần chạy sau cùng cho mỗi job thay vì cộng dồn; `run_anofusion.py` ghi `seed<k>.partial.json` sau mỗi file test (chạy hỏng giữa chừng vẫn giữ các file đã chấm xong) và chỉ ghi `seed<k>.json` khi xong hết (vì bộ chạy bỏ qua job đã có `seed<k>.json`); lần gọi đầu của chuỗi này kết thúc ngay vì json cũ còn đó (dòng `ALL DONE` 11:50 trong `status.txt` là vì vậy, không có gì bị ghi đè).

**Kiểm nhanh Eadro và OrEdge trước khi chạy seed 0 [CN 10-22 21:22–21:47, người dùng ra lệnh; chạy thử 1 file test mỗi dataset, 5 epoch (luật của người dùng), ghi ở thư mục tạm, KHÔNG phải kết quả seed 0, không vào `item9`].** Cả 8 lần chạy kết thúc `rc=0`, không lỗi; **không cần tiền xử lý thêm** (pkl và `meta.pkl` của 4 dataset RCAEval đã đủ; `run_art.build` đổi ngày 10-19 vẫn trả 6 giá trị như cũ nên Eadro và OrEdge không bị ảnh hưởng). SN: vẫn **không áp dụng** (mỗi scenario chỉ có một phiên lỗi, không chia 3-fold theo run được). Thời gian đo thật (cả tải dữ liệu, 3 fold, 5 epoch; OrEdge ở RE3-TT và RE2-TT đo ở 2 epoch rồi nhân lên 5 epoch):

| Mô hình | Dataset (số file test) | Giây mỗi file | Cả dataset |
|:--|:--|--:|--:|
| Eadro | RE2-OB (6) | 119 | 12 phút |
| Eadro | RE3-OB (5) | 54 | 4,5 phút |
| Eadro | RE3-TT (4) | 141 | 9,4 phút |
| Eadro | RE2-TT (6) | 280 | 28 phút |
| OrEdge | RE2-OB (6) | 306 | 31 phút |
| OrEdge | RE3-OB (5) | 179 | 15 phút |
| OrEdge | RE3-TT (4) | ≈ 610 (ngoại suy từ 40 giây/epoch/fold) | ≈ 41 phút |
| OrEdge | RE2-TT (6) | ≈ 340 (ngoại suy từ 21 giây/epoch/fold) | ≈ 34 phút |

Một seed: Eadro ≈ 54 phút; OrEdge ≈ 46 phút (chỉ hai OB) hoặc ≈ 2,0 giờ (cả bốn); sai số ±10% giữa các file. Ước 50 epoch cũ (Eadro 7,6 giờ, OrEdge OB 4,3 giờ) không còn dùng. **Phát hiện cần quyết định:** OrEdge chọn mô hình theo loss huấn luyện nhỏ nhất (checkpoint `loss` của tác giả), nhưng loss = (1−para)·BCE + para·rec với para = 1/(epoch+1), nên **epoch 1 có para = 1 (loss chỉ còn phần tái dựng, đầu phân loại chưa được huấn luyện) và loss không so được giữa các epoch**. Trong 5 epoch, epoch được chọn: RE2-OB `cpu` fold (5, 1, 5), RE3-OB `f1` (1, 1, 1); trong 2 epoch: RE3-TT `f1` và RE2-TT `cpu` đều (1, 1, 1), kết quả AUROC 0,491 và 0,520 (≈ ngẫu nhiên). Với 50 epoch của tác giả para giảm tới 0,01 nên lỗi này ít xảy ra. Nếu giữ 5 epoch, chọn theo loss làm hỏng hàng OrEdge; Eadro đang dùng mô hình epoch cuối (đã ghi từ trước). **Người dùng duyệt (10-22): OrEdge lấy mô hình epoch cuối** (chọn theo nguyên tắc chứ không theo kết quả test); đã sửa `run_oredge.py` (`--select last` mặc định, không còn dừng sớm theo patience; `--select loss` giữ cách cũ). Kiểm lại RE3-OB `f2` 2 epoch sau sửa: chạy bình thường. Kết quả chạy thử (1 file mỗi dataset, chỉ để kiểm chạy được): Eadro AUROC 0,847 (RE2-OB `cpu`), 0,974 (RE3-OB `f1`), 0,431 (RE3-TT `f1`), 0,807 (RE2-TT `cpu`); OrEdge 0,954, 0,638, 0,491 (2 epoch), 0,520 (2 epoch).

**Eadro và OrEdge trên SN [CN 10-22 21:50–22:20, người dùng yêu cầu "phải chạy được trên cả SN"; chạy được; kết quả seed 0 nằm trong bảng sinh tự động bên dưới (chuỗi `chain_eadro_oredge.cmd`, xong 02:33 ngày 10-23)].** Giao thức 3-fold theo run không áp dụng được (mỗi scenario SN là một phiên). **Giao thức SN (mới): leave-one-scenario-out** (`codes/competitors/sn_labeled.py`, `eadro/run_eadro_sn.py`, `oredge/run_oredge_sn.py`): với scenario X, huấn luyện trên các cửa sổ có nhãn của 11 phiên khác (gồm Normal_Baseline), chấm đúng các cửa sổ của `scenarios/test_X.pkl` (cùng tập UAM-AD được chấm). Loại khỏi huấn luyện: mọi cửa sổ của phiên X, và mọi cửa sổ của `val.pkl` (dùng cho ngưỡng) hoặc `test_X.pkl` (chứa cửa sổ bình thường của các phiên khác). Nhãn: nhãn cửa sổ và service lỗi của phiên (`rca.SCENARIO2SERVICE`); 3 phiên `Perf_*` không có service riêng nên cửa sổ lỗi của chúng chỉ dạy bộ phát hiện của Eadro (loss định vị bỏ qua nhãn -1) và không dùng cho OrEdge. Mỗi fold huấn luyện khoảng 180 đến 400 cửa sổ (150 đến 200 cửa sổ lỗi). Cùng epoch với các mô hình khác (50 trên SN), mô hình epoch cuối. Đầu vào như ART SN (4 KPI container + 6 đặc trưng trace + 2 đặc trưng log theo service; xem đoạn L+M+T bên dưới). HR@k tính trên cửa sổ lỗi của 9 scenario có một service mục tiêu. Hạn chế: service lỗi của scenario bị giữ lại có thể xuất hiện làm service lỗi ở phiên khác (ví dụ `media-service` ở `Code_Stop_Media` và `Svc_Kill_Media`), còn `text-service`, `user-service`, `home-timeline-service` chỉ có một phiên nên không bao giờ thấy làm service lỗi lúc huấn luyện.

**Phát hiện: nhiễu "đầu phiên" khi mô hình có nhãn dùng lịch sử** [số trong đoạn này và bảng ngay dưới là của bản M+T (Eadro) và log mức hệ thống (OrEdge) chạy trước khi sửa; đã thay bằng bản L+M+T ở đoạn "Eadro và OrEdge nay là L+M+T" bên dưới; kết luận về nhiễu đầu phiên không đổi; file cũ ở `item9\sn_sensitivity_old_mt_and_system_log`]**.** Cửa sổ lỗi của `DB_Redis_*`, `Perf_*`, `Svc_Kill_*` nằm ở những cửa sổ đầu phiên, đúng nơi lịch sử là lịch sử đệm (đuôi Normal_Baseline, xem `--pad_normal`). Mô hình có nhãn học được "có lịch sử đệm = lỗi". Đo bằng quy tắc không mô hình "cửa sổ nằm trong 9 cửa sổ đầu của phiên" (`scratchpad/sn_pad_artifact.py`): AUROC **0,907 đến 0,982** trên chín scenario không phải `Code_Stop_*` (và 0,574 đến 0,586 ở `Code_Stop_*`). Vì vậy **cấu hình chính của hai mô hình trên SN không dùng lịch sử**: chunk 10 bước của một cửa sổ là 10 bản sao của chính cửa sổ đó (`--history` mới giữ chunk thật, dùng làm hàng độ nhạy). Chạy thử 1 seed (thư mục tạm; bản sao kết quả ở `item9\sn_sensitivity\`; AUROC trung bình 12 scenario / UAM-AD seed 0 = 0,956):

| Mô hình | Có lịch sử (bị nhiễu) | Không lịch sử (cấu hình chính) |
|:--|--:|--:|
| Eadro | 0,979 | **0,986** (HR@1 0,00, HR@3 0,18, HR@5 0,51) |
| OrEdge | 0,917 | **0,557** (HR@1 0,22, HR@3 0,52, HR@5 0,64) |

Eadro gần như không đổi giữa hai cách (0,979 và 0,986), nên điểm cao của Eadro là tín hiệu thật trong nội dung cửa sổ chứ không phải nhiễu lịch sử đệm. OrEdge thì khác hẳn: 0,917 khi có lịch sử, 0,557 khi không (chín scenario `DB_Redis_*`, `Perf_*`, `Svc_Kill_*` rơi xuống 0,12 đến 0,61, năm trong chín cái dưới 0,5, nghĩa là điểm bất thường đảo chiều). Hai lý do không tách được: (a) với lịch sử, OrEdge dùng được nhiễu đầu phiên (0,727 đến 0,982 ở chín scenario đó, cùng mức với quy tắc đầu phiên thuần 0,907 đến 0,982); (b) OrEdge là mô hình tái dựng theo thời gian (bộ lọc FITS-Legendre), nên chuỗi 10 bản sao giống hệt là tín hiệu phẳng, mất đúng phần mô hình dùng. **Con số OrEdge không lịch sử vì vậy là cận dưới bất công với OrEdge, con số có lịch sử là cận trên bị nhiễu; thesis phải ghi cả hai và không rút kết luận mạnh về OrEdge trên SN.** Cả hai chỉ chạy thử 1 seed. Tôi chọn cấu hình chính trước khi biết cái nào cho điểm cao hơn ở OrEdge dựa trên nguyên tắc "không có thông tin vị trí trong phiên" (nhưng tôi đã thấy cả hai kết quả trước khi chốt chạy seed 0, vì Eadro có sẵn cả hai).

**Chuỗi seed 0 Eadro rồi OrEdge bắt đầu 22:20 ngày 10-22 đã bị DỪNG lúc 22:31 theo yêu cầu đổi sang L+M+T** (chạy xong 8 file test RCAEval của Eadro bản M+T; kết quả chuyển sang `item9\eadro_mt_old`, log sang `eadro_mt_old_logs`, không dùng). Đã chạy lại 23:39 với bản đã sửa (xem đoạn "Sửa OrEdge ... chạy seed 0" bên dưới).

**Eadro và OrEdge nay là L+M+T [CN 10-22 22:31–23:00, người dùng yêu cầu]; chạy thử 1 file test mỗi dataset (5 epoch) và SN (50 epoch), thư mục tạm, KHÔNG phải kết quả seed 0.** [Phần OrEdge (log 2 đặc trưng, cột OrEdge của bảng, các dòng SN của OrEdge) đã bị thay bằng bản log theo từng nút ở đoạn "Sửa OrEdge" ngay dưới; phần Eadro vẫn đúng.] Cùng đặc trưng log theo service với ART (`service_logs.py`: log1p số dòng và số dòng khác nhau của service). **Eadro:** thêm nhánh log (cùng cấu trúc nhánh metric: Conv1d giãn nở + self-attention) vào `eadro_model.py`; `run_eadro.py` thêm 2 kênh log vào đặc trưng và vào phép chuẩn hoá theo max train. **OrEdge:** log của mỗi nút là 2 đặc trưng của service đó, thay vì vector đếm template mức hệ thống (giống nhau cho mọi nút) trước đây; bỏ `log_counts` và từ vựng template. Hai runner SN dùng cùng đặc trưng (`sn_labeled.py` gọi `world(sn, no_logs=False)`). Lưu ý đây là bản log gọn (2 đặc trưng) để đồng nhất giữa ART, Eadro, OrEdge; bản gốc của Eadro dùng vector đếm sự kiện theo service, của OrEdge vector đếm template theo nút. Kết quả chạy thử (1 file mỗi dataset, chỉ để kiểm chạy được):

| Dataset (file) | Eadro: AUROC / F1 argmax / HR@1 / giây | OrEdge: AUROC / F1 argmax / HR@1 / giây |
|:--|:--|:--|
| RE2-OB `cpu` | 0,867 / 0,774 / 0,720 / 122 | 0,989 / 0,941 / 0,927 / 301 |
| RE3-OB `f1` | 0,964 / 0,878 / 0,961 / 68 | 0,857 / 0,700 / 0,935 / 163 |
| RE3-TT `f1` | 0,487 / 0,070 / 0,512 / 171 | 0,700 (2 epoch) / 0,174 / 0,611 / 79 (2 epoch) |
| RE2-TT `cpu` | 0,792 / 0,606 / 0,562 / 338 | 0,935 (2 epoch) / 0,918 / 0,927 / 146 (2 epoch) |

(OrEdge trước khi sửa, M+T ngoài log mức hệ thống và chọn theo loss: 0,954, 0,638, 0,491, 0,520; sau sửa tốt hơn rõ ở cả bốn, hai thay đổi cùng lúc nên không tách được phần của từng thay đổi.) **Ước giờ một seed (L+M+T, 5 epoch RCAEval, 50 SN; OrEdge TT ngoại suy từ 2 epoch):** Eadro RE2-OB 6 × 122 s = 12 phút, RE3-OB 5 × 68 s = 6 phút, RE3-TT 4 × 171 s = 11 phút, RE2-TT 6 × 338 s = 34 phút, SN 1 phút: **≈ 64 phút**. OrEdge RE2-OB 6 × 301 s = 30 phút, RE3-OB 5 × 163 s = 14 phút, RE3-TT 4 × ≈ 175 s = 12 phút, RE2-TT 6 × ≈ 330 s = 33 phút, SN 2 phút: **≈ 91 phút**. Tổng **≈ 2,6 giờ** (sai số ±15%).

SN, L+M+T, 1 seed, 12 scenario (AUROC trung bình; UAM-AD seed 0: 0,956):

| Mô hình | Không lịch sử (cấu hình chính) | Có lịch sử (độ nhạy) |
|:--|--:|--:|
| Eadro | **0,979** (F1@val 0,805, HR@1/3/5 0,06/0,22/0,50) | 0,983 |
| OrEdge | **0,533** (F1@val 0,299, F1 argmax 0,000, HR@1/3/5 0,26/0,57/0,68) | 0,436 |

Eadro không đổi giữa hai cách, nên điểm cao là tín hiệu thật của nội dung cửa sổ. **OrEdge trên SN kém ở cả hai cách, và F1 argmax = 0 ở mọi scenario** (không cửa sổ nào có P > 0,5): các scenario `Svc_Kill_*` có AUROC 0,03 đến 0,08 (điểm lỗi thấp hơn điểm bình thường), `Code_Stop_Media/Text` 0,99/0,98. Nguyên nhân khả dĩ, chưa kiểm: (a) huấn luyện ít bước: khoảng 370 cửa sổ huấn luyện, batch 100, 3 bước mỗi epoch, 50 epoch = 150 bước (RCAEval: 5 epoch × hàng trăm bước); (b) bản log 2 đặc trưng mất thông tin so với đếm template (bản log mức hệ thống từng cho AUROC 0,917 khi có lịch sử, 0,557 khi không, ở chế độ trước); (c) lỗi `Svc_Kill_*` làm dịch vụ biến mất nên sai số tái dựng nhỏ. Kết quả file: `item9\sn_lmt_smoke\`.

**Sửa OrEdge (log theo từng nút, như tác giả) và chạy seed 0 [CN 10-22 23:00–23:40, người dùng: "sửa luôn cho hoàn chỉnh rồi chạy seed 0"].** Bản 2 đặc trưng log làm OrEdge trên SN rơi xuống AUROC 0,533 (F1 argmax 0; `Svc_Kill_*` 0,03 đến 0,08). Sửa: log của mỗi nút = vector đếm template của giây (hoặc cửa sổ 30 s) giới hạn trong các template do chính service đó ghi (`run_oredge.owner_mask`, `service_logs.entry_service`; template không thuộc service nào của hệ thống bị bỏ). Dùng cho cả RCAEval và SN (hai runner). **Đây là thay đổi chọn sau khi thấy kết quả SN của bản 2 đặc trưng**, nhưng dựa trên nguyên tắc "giữ biểu diễn log của tác giả (đếm template theo nút)"; Eadro vẫn dùng 2 đặc trưng log/service (bộ nhớ: vector đếm theo service quá lớn cho RE3-TT).

Kết quả chạy thử bản mới (1 seed, thư mục tạm): **SN** AUROC trung bình **0,813** không lịch sử (cấu hình chính; F1@val 0,345, HR@1/3/5 0,27/0,50/0,71), **0,804** có lịch sử (độ nhạy; gần nhau, nghĩa là OrEdge không còn dùng nhiễu đầu phiên); mọi scenario trên 0,5 (thấp nhất `Code_Stop_Text` 0,515); F1 argmax vẫn gần 0 (xác suất ít khi vượt 0,5: SN chỉ có 150 bước huấn luyện). RE3-TT `f1` 2 epoch: AUROC 0,777 (bản 2 đặc trưng 0,700; bản cũ 0,491), HR@1 0,83. Kết quả SN ở `item9\sn_sensitivity\` (Eadro có lịch sử, OrEdge hai chế độ) và `item9\sn_smoke_oredge_2feat_eadro_final\` (Eadro không lịch sử cuối cùng, hai số cũ của OrEdge 2 đặc trưng bị bỏ).

**Chuỗi seed 0 chạy lại từ 23:39 ngày 10-22** (`D:\ClaudeWork\phase3\item9\chain_eadro_oredge.cmd`: Eadro rồi OrEdge; RE3-OB, RE2-OB, RE3-TT, RE2-TT rồi SN; 5 epoch RCAEval, 50 epoch SN; OrEdge lấy epoch cuối). **Ước giờ:** Eadro ≈ 64 phút (RE2-OB 12, RE3-OB 6, RE3-TT 11, RE2-TT 34, SN 1); OrEdge ≈ 2,0 giờ (RE2-OB 6 × ≈ 306 s = 31, RE3-OB 5 × ≈ 179 s = 15, RE3-TT 4 × ≈ 615 s = 41, RE2-TT 6 × ≈ 340 s = 34, SN 2 phút; RE2-OB, RE3-OB, RE2-TT lấy từ lần đo trước khi đổi log, RE3-TT từ 40 giây/epoch/fold đo lại) ; **tổng ≈ 3,1 giờ, xong khoảng 02:45 ngày 10-23**, sai số ±15%. **Kết quả thật: XONG 02:33:38 ngày 10-23, mọi ô `rc=0`, không ô nào bị hệ thống dừng hay lỗi.** Tổng giờ chạy ô 174,7 phút (2,9 giờ), thấp hơn ước 6%, trong sai số. Ước → thật (phút): Eadro RE2-OB 12 → 12,7, RE3-OB 6 → 4,5, RE3-TT 11 → 11,8, RE2-TT 34 → 34,5, SN 1 → 1,1 (tổng 64 → 64,6); OrEdge RE2-OB 31 → 31,1, RE3-OB 15 → 11,1, RE3-TT 41 → 33,2, RE2-TT 34 → 32,8, SN 2 → 1,9 (tổng ≈ 120 → 110,1). Hai hàng độ nhạy "có lịch sử" của SN chỉ có từ lần chạy thử (`item9\sn_sensitivity\`), chưa có trong seed 0.

Đọc số seed 0 (trung bình qua scenario, AUROC; ART, AnoFusion, Eadro và OrEdge so với UAM-AD cùng seed; chỉ những điều số liệu đỡ được):
- **UAM-AD hơn AnoFusion** ở SN (0,956 so với 0,731; thắng 12/12 scenario về AUROC) và RE3-OB (0,832 so với 0,630, chạy `--clip 100`; UAM-AD thắng 5/5 về AUROC; F1@val95 0,556 so với 0,366, AnoFusion chỉ thắng 1/5: `f1`). Ở RE2-TT (chạy `--clip 100`) hai bên gần nhau: AnoFusion 0,773 so với UAM-AD 0,809 về AUROC (AnoFusion thắng 3/6 scenario: `cpu`, `disk`, `socket`), còn F1@val95 AnoFusion 0,584 cao hơn UAM-AD 0,540 (thắng 4/6). RE2-OB (chạy với `--clip 100`): AnoFusion kém UAM-AD về AUROC, 0,819 so với 0,887 (thắng 0/6 scenario), nhưng hơn về F1@val95, 0,587 so với 0,495 (thắng 4/6: `delay`, `disk`, `loss`, `mem`). RE3-TT: AnoFusion chỉ có `test_f1` (đã dừng vì 20 giờ mỗi scenario): AUROC 0,502, F1@val95 0,116, thấp hơn UAM-AD (0,720; 0,523) và ART (0,659; 0,215) trên cùng scenario `f1`. Một scenario thì không kết luận được cho cả dataset. `f2`–`f4` chỉ có ước lượng thống kê (không phải số đo), xem đoạn "Ước lượng thống kê cho AnoFusion RE3-TT".
- **ART hơn UAM-AD** ở RE2-OB (0,982 so với 0,887; thắng 6/6 scenario cả AUROC lẫn F1@val), RE3-OB (0,892 so với 0,832; 5/5 AUROC, nhưng F1@val chỉ thắng 3/5 và trung bình 0,548 so với 0,556) và RE2-TT (0,831 so với 0,809; 5/6 AUROC, F1@val chỉ thắng 2/6). **UAM-AD hơn ART** ở RE3-TT (0,705 so với 0,686; ART thắng 1/4) và SN (0,956 so với 0,870; ART thắng 2/12).
- **HR@k:** ART cao hơn UAM-AD rất rõ ở RE2-OB (HR@1 0,751 so với 0,352), RE3-OB (0,733 so với 0,022), RE3-TT (0,291 so với 0,063), gần ở RE2-TT (0,444 so với 0,421), và **thấp hơn ở SN (0,048 so với 0,178)**. Đơn vị chấm HR của hai bên khác nhau (xem "Cách đọc" ngay dưới).
- **OrEdge (bán giám sát) hơn UAM-AD ở cả bốn dataset RCAEval, kém xa ở SN.** AUROC: RE2-OB 0,969 so với 0,887 (hơn 4/6 scenario), RE3-OB 0,891 so với 0,832 (4/5), RE2-TT 0,968 so với 0,809 (6/6), RE3-TT 0,766 so với 0,705 (3/4); F1@val95 0,846 / 0,767 / 0,833 / 0,624 so với 0,495 / 0,556 / 0,540 / 0,515. SN: 0,813 so với 0,956, hơn 0/12 scenario. OrEdge học từ nhãn lỗi của 2/3 số run ngay trong file test (3-fold theo run), còn UAM-AD không dùng nhãn nào, nên OrEdge hơn là điều có thể đoán trước và đây là hàng tham chiếu có nhãn, không phải đối thủ cùng điều kiện. F1 argmax (quyết định của bộ phân loại): 0,886 / 0,770 / 0,899 / 0,615 ở RCAEval nhưng **0,013 ở SN** (xác suất hầu như không vượt 0,5; nghi do chỉ 150 bước huấn luyện, chưa kiểm).
- **Eadro (có giám sát) lẫn lộn trên RCAEval, tốt trên SN.** AUROC thấp hơn UAM-AD ở RE2-OB (0,864 so với 0,887) và RE3-TT (0,574 so với 0,705), cao hơn ở RE3-OB (0,842 so với 0,832) và RE2-TT (0,825 so với 0,809); F1@val95 cao hơn UAM-AD ở RE2-OB, RE3-OB, RE2-TT, SN (0,698 / 0,668 / 0,620 / 0,811 so với 0,495 / 0,556 / 0,540 / 0,734), thấp hơn ở RE3-TT (0,350 so với 0,515). **Thấp hơn ART (không giám sát) ở mọi dataset RCAEval** (0,864 / 0,842 / 0,825 / 0,574 so với 0,982 / 0,892 / 0,831 / 0,686) nhưng **cao hơn ART ở SN** (0,979 so với 0,870; 11/12 scenario) và cao hơn UAM-AD (0,979 so với 0,956; 8/12). Nguyên nhân Eadro không hơn ART ở RCAEval chưa kiểm (ứng viên: luật 5 epoch chung, log 2 đặc trưng thay vì vector đếm sự kiện của tác giả, chỉ 2/3 số run để huấn luyện). F1 argmax 0,683 / 0,666 / 0,596 / 0,227 ở RCAEval (RE3-TT thấp nhất) và 0,805 ở SN.
- **HR@1 của hai mô hình có nhãn** cao hơn UAM-AD ở mọi dataset RCAEval: Eadro 0,666 / 0,889 / 0,523 / 0,622 (RE2-OB, RE3-OB, RE2-TT, RE3-TT), OrEdge 0,932 / 0,930 / 0,946 / 0,899, UAM-AD 0,352 / 0,022 / 0,421 / 0,063. Ở SN thì Eadro 0,028, OrEdge 0,269, UAM-AD 0,178. Không so thẳng được: hai mô hình có nhãn học cách định vị từ nhãn service lỗi, HR@k tính trên mọi mẫu bất thường, đơn vị chấm khác UAM-AD (xem "Cách đọc" ngay dưới).
- **Chi phí:** ART 0,4–6,8 phút mỗi dataset (cả lần chạy), UAM-AD 4–184 phút; AnoFusion 11 phút (SN), 19 phút (RE2-OB), 48 phút (RE3-OB), 220 phút (RE2-TT; mã tăng tốc và clip), nhiều hơn UAM-AD ở RE3-OB (36 phút) và RE2-TT (184 phút). Eadro 1,1–34,5 phút và OrEdge 1,9–33,2 phút mỗi dataset, ít hơn UAM-AD ở cả năm dataset (tổng một seed: Eadro 65 phút, OrEdge 110 phút).
- **Lưu ý:** một seed, nên chưa có độ lệch chuẩn; SN chỉ có 4 đến 39 cửa sổ lỗi mỗi file test. Trên SN, ART và AnoFusion chạy với lịch sử đệm từ Normal_Baseline (xem khối quyết định ở trên); Eadro và OrEdge chạy **không lịch sử** (cấu hình chính) vì hai mô hình có nhãn học được nhiễu đầu phiên. **Giao thức SN của Eadro và OrEdge khác các hàng còn lại** (leave-one-scenario-out, huấn luyện bằng nhãn của 11 phiên khác), và `text-service`, `user-service`, `home-timeline-service` không bao giờ là service lỗi lúc huấn luyện. Eadro dùng log 2 đặc trưng mỗi service (không phải vector đếm sự kiện của tác giả). Hàng OrEdge dùng log đếm template theo nút, một thay đổi chọn sau khi thấy kết quả SN của bản 2 đặc trưng (đã nêu ở trên). Cả hai là hàng tham chiếu có nhãn, ghi rõ trong thesis.

**Chốt lại bảng so sánh seed 0 [CN 10-25, theo góp ý của người dùng].** (1) **Bỏ UAC-AD khỏi mọi bảng và đoạn so sánh seed 0**: UAC-AD chỉ có L+M, không cùng hạng với các mô hình đủ L+M+T; UAC-AD vẫn là baseline của UAM-AD ở các mục bước 8 phía trên (đó là so sánh với mô hình gốc, không phải so với đối thủ), nên không xoá ở đó. (2) **Bỏ chú thích "(L+M+T)" ở ART** (và ở Eadro, OrEdge): mọi mô hình đưa vào so sánh đều đủ ba modality; các đoạn nhật ký phát triển cũ giữ nguyên chữ L+M+T vì chúng ghi lại việc đổi từ M+T sang L+M+T. (3) **Bảng "Theo dataset" có thêm cột Δ** sau mỗi chỉ số ưu tiên F1@val95, AUROC, AUPRC (Δ = UAM-AD trừ mô hình ở dòng đó, trong ngoặc là % so với giá trị của mô hình đó; bảng chênh lệch riêng lần đầu đã bỏ) và dòng Trung bình. % của dòng Trung bình = chênh lệch trung bình chia cho giá trị trung bình của mô hình kia, không phải trung bình của các % (lần đầu tính kiểu sau, F1 so với ART ra −0,003 nhưng +10,2%, sai dấu vì dataset nền thấp kéo lệch). Dòng AnoFusion chỉ có 4 dataset (RE3-TT chỉ có `f1`) và so với UAM-AD cũng trên 4 dataset đó. (4) **Hai bảng Δ UAM−UAC cuối cùng của bước 8 cũng có %** (bảng S2 và bảng so hai luật `norm_sum`/`raw_sum`; % = Δ chia cho giá trị trung bình 5 seed của UAC-AD cùng cột); các bảng nháp cũ cùng loại (trung bình 2 seed, trung bình theo cặp) giữ nguyên số cũ. HR@k không tính %, vì đơn vị chấm khác UAM-AD.

**So với các mô hình không giám sát (ART, AnoFusion) [CN 10-25; seed 0, số lấy từ bảng "Theo dataset" bên dưới].** Nhận định của người dùng ("SN ta cao nhất, OB và TT không thua quá nhiều") **khớp số liệu về AUROC**: UAM-AD cao nhất nhóm ở SN (0,956 so với ART 0,870 và AnoFusion 0,731; +9,9% và +30,7%) và ở RE3-TT (0,705 so với ART 0,686, +2,8%; AnoFusion chỉ có `f1`); ở RE2-OB, RE3-OB, RE2-TT UAM-AD đứng thứ hai, sau ART và trên AnoFusion, thấp hơn ART 9,7%, 6,8%, 2,7%. Trung bình 5 dataset UAM-AD kém ART 1,7% về AUROC. **F1@val95** cao nhất nhóm ở RE3-OB (+1,4% so với ART), RE3-TT (+62,4%) và SN (+38,1%), nhưng **thứ ba ở RE2-OB (−43,8% so với ART, −15,6% so với AnoFusion) và RE2-TT (−6,9% và −7,5%)**. RE2-OB là ngoại lệ lớn duy nhất: dùng trung bình 5 seed của UAM-AD (F1 0,564, AUROC 0,903) thì vẫn kém ART 35,9% về F1 và 8,0% về AUROC (so nhóm 5 seed với 1 seed, chỉ để biết cỡ). Nên câu "không quá thua thiệt" đỡ được với AUROC ở cả bốn dataset RCAEval và với F1 ở ba trên bốn, **không đỡ được với F1@val95 ở RE2-OB**.

**So với các mô hình có nhãn (Eadro, OrEdge): mức chấp nhận [CN 10-25; người dùng chọn F1@val95, AUROC, AUPRC làm ba chỉ số ưu tiên và chốt giữ mức 90% cho cả ba chỉ số].** Chênh lệch % của UAM-AD (thứ tự RE2-OB, RE3-OB, RE2-TT, RE3-TT, SN). **AUROC:** Eadro +2,7%, −1,2%, −2,0%, +22,9%, −2,4% (trung bình +2,5%); OrEdge −8,4%, −6,6%, −16,5%, −7,9%, +17,6% (trung bình −5,0%). **AUPRC:** Eadro −1,7%, −19,4%, −0,9%, −18,0%, −17,1% (trung bình −11,1%); OrEdge −17,9%, −26,4%, −27,1%, −49,3%, +46,9% (trung bình −19,5%). **F1@val95:** Eadro −29,1%, −16,8%, −12,8%, +47,1%, −9,5% (trung bình −9,7%); OrEdge −41,5%, −27,6%, −35,2%, −17,4%, +112,5% (trung bình −16,9%). Áp mức "**UAM-AD giữ ≥ 90% giá trị của mô hình có nhãn**" cho cả ba chỉ số, seed 0 **đạt 15/30 ô** (2 mô hình × 3 chỉ số × 5 dataset): AUROC 9/10 (Eadro 5/5; OrEdge 4/5, trừ RE2-TT với 83,5%), F1@val95 3/10 (Eadro: RE3-TT, SN; OrEdge: SN), AUPRC 3/10 (Eadro: RE2-OB, RE2-TT; OrEdge: SN). Đạt cùng lúc cả hai mô hình: AUROC 4/5 dataset, F1@val95 1/5 (SN), AUPRC 0/5. Thay seed 0 của UAM-AD bằng trung bình 5 seed vẫn 15/30. Theo dòng Trung bình: đạt ở AUROC (cả hai mô hình) và F1@val95 so với Eadro (90,3%); không đạt ở AUPRC (88,9% so với Eadro, 80,5% so với OrEdge) và F1@val95 so với OrEdge (83,1%). Ở các ô không đạt, UAM-AD kém Eadro 12,8–29,1% và kém OrEdge 16,5–49,3%.

**Kết luận đỡ được:** "cứ ≥ 90% là được" chỉ đúng với AUROC; với AUPRC và F1@val95 thì UAM-AD **không** đạt mức đó ở đa số dataset RCAEval, nên câu trong thesis phải là "UAM-AD giữ ≥ 90% AUROC của cả hai mô hình có nhãn ở 4/5 dataset, nhưng AUPRC và F1@val95 còn thấp hơn rõ ở RCAEval; ở SN vượt OrEdge ở cả ba chỉ số". **Không nên hạ mức riêng cho AUPRC và F1 cho vừa số** (đó là chỉnh tiêu chuẩn theo kết quả); nếu muốn đổi mức thì chọn một mức chung và ghi lý do trước khi có số 5 seed. Mức 90% không "đều tay" giữa các chỉ số: sàn của AUROC là 0,5, sàn của AUPRC là tỉ lệ lỗi (khoảng 13–20%), nên cùng một bảng điểm thì % của AUROC nhỏ hơn nhiều (OrEdge RE2-OB: −8,4% AUROC nhưng −17,9% AUPRC); đạt 90% AUROC là phát biểu yếu hơn đạt 90% AUPRC. **Người dùng đã chốt (10-25): giữ mức 90% cho cả ba chỉ số và báo kết quả đúng như trên**, không đổi mức theo từng chỉ số; áp nguyên cho kết quả 5 seed. **90% là con số tôi đề xuất, không lấy từ tài liệu nào tôi đã kiểm chứng**, đặt sau khi đã thấy số seed 0. Độ lệch chuẩn giữa 5 seed của UAM-AD (theo % giá trị trung bình): AUROC 0,2–1,6%; AUPRC 0,5–3,1%; F1@val95 0,5–14,9% (lớn nhất ở RE2-OB), nên dung sai 10% lớn hơn nhiễu seed ở AUROC và AUPRC; riêng F1@val95 ở RE2-OB nhiễu seed đã gần 15%.

**Hạn chế khi đọc các so sánh với mô hình có nhãn:** (1) đối thủ mới có 1 seed, nên nhiễu của chúng chưa biết, chênh vài % ở từng dataset không nên đọc thành hơn hay kém; (2) SN: Eadro và OrEdge dùng giao thức khác (leave-one-scenario-out), OrEdge SN chỉ 150 bước huấn luyện, Eadro dùng log 2 đặc trưng; (3) dòng Trung bình trộn dataset khó và dễ, nên báo kèm từng dataset; (4) HR@1 của UAM-AD ở RCAEval 0,02–0,42, so với Eadro 0,52–0,89 và OrEdge 0,90–0,95 (đơn vị chấm khác, không tính %): định vị là chỗ thua rõ nhất, vì hai mô hình có nhãn học định vị từ nhãn service lỗi.

**Kế hoạch chạy đủ seed 1–4 cho bốn mô hình [CN 10-25 21:55; người dùng đề nghị chạy đủ để có số liệu cuối; ước tính cho cả bốn mô hình; người dùng sau đó chỉ cho chạy ba mô hình, xem đoạn "Quyết định và khởi chạy" ngay dưới].** Phạm vi: ART, AnoFusion, Eadro, OrEdge × seed 1, 2, 3, 4 × 5 dataset, **trừ AnoFusion RE3-TT (không chạy, theo người dùng)**: AnoFusion còn 4 dataset, `f1` của seed 0 và ước lượng thống kê vẫn nằm ngoài bảng. Cấu hình y hệt seed 0 (5 epoch RCAEval, 50 epoch SN, AnoFusion `--clip 100`, SN có `--pad_normal` cho ART và AnoFusion, Eadro và OrEdge lấy mô hình epoch cuối và không dùng lịch sử trên SN). Lệnh dạng `run_phase3_item9.py --model <m> --seed <k> --out_dir D:\ClaudeWork\phase3\item9`; riêng AnoFusion phải truyền `--datasets rcaeval_re2_ob rcaeval_re3_ob rcaeval_re2_tt sn` vì mặc định của bộ chạy chưa gồm RE2-TT. Bộ chạy bỏ qua ô đã có `seed<k>.json`, nên chạy lại được sau khi bị dừng. **Giờ đo thật của seed 0** (phút, lần chạy thành công cuối của từng ô; lấy từ `status.txt`): ART 17,6 (RE2-OB 6,0; RE3-OB 2,0; RE2-TT 6,8; RE3-TT 2,3; SN 0,4); Eadro 64,4 (12,7; 4,5; 34,5; 11,8; 1,1); OrEdge 110,0 (31,1; 11,1; 32,8; 33,2; 1,9); AnoFusion 298,5 (RE2-OB 18,8; RE3-OB 48,1; RE2-TT 220,3; SN 11,3; không có RE3-TT). **Một seed của cả bốn mô hình = 490,6 phút = 8,18 giờ; bốn seed = 32,7 giờ ≈ 1,36 ngày**, trong đó AnoFusion 19,9 giờ (riêng RE2-TT 14,7 giờ) và ba mô hình còn lại 12,8 giờ. Nếu bắt đầu 22:15 ngày 10-25 và chạy theo từng seed (mỗi seed đủ bốn mô hình theo thứ tự ART, Eadro, OrEdge, AnoFusion): xong seed 1 khoảng 06:25 ngày 10-26, seed 2 khoảng 14:35, seed 3 khoảng 22:45, **seed 4 khoảng 06:55 ngày 10-27**; khoảng sai số −10% đến +15%, tức xong trong khoảng 03:40 đến 11:50 ngày 10-27 (sai số của ước tính giờ Eadro và OrEdge lần trước là −6%; giờ AnoFusion RE2-TT dao động theo áp lực RAM của máy, chưa đo riêng). Chọn chạy theo từng seed (không theo từng mô hình) để nếu phải dừng giữa chừng thì vẫn có đủ số seed hoàn chỉnh cho cả bốn mô hình. Giả định: máy rảnh (một việc một lúc, GPU 3 GB) và không bị hệ thống dừng vì thiếu RAM; nếu có ô bị dừng thì báo lại, không tự chạy lại. Việc phải làm sau khi chạy xong: mở rộng `item9_tables.py` để báo trung bình±độ lệch chuẩn 5 seed và Δ, % tính trên trung bình 5 seed (hiện công cụ chỉ sinh bảng theo từng seed); UAM-AD đã có sẵn 5 seed từ bước 8.

**Quyết định và khởi chạy seed 1–4 [CN 10-25 22:02, người dùng ra lệnh].** Người dùng chọn: (1) chạy trước **ART, Eadro, OrEdge** cho seed 1–4; (2) **AnoFusion không chạy thêm seed**, giữ seed 0 ("cứ cho là kết quả 5 seed như seed 0"). **Cách ghi: AnoFusion là kết quả 1 seed (seed 0) ở mọi bảng, không sao chép thành seed 1–4 và không có độ lệch chuẩn**; trong thesis phải ghi rõ AnoFusion chỉ 1 seed vì chi phí (298,5 phút mỗi seed, riêng RE2-TT 220 phút) và RE3-TT không có số đo đủ. Chuỗi `D:\ClaudeWork\phase3\item9\chain_seeds1to4_art_eadro_oredge.cmd` khởi chạy lúc 22:02 ngày 10-25, chạy theo từng seed (ART, Eadro, OrEdge của seed 1, rồi seed 2, 3, 4); log ở `item9\logs\`, tiến độ ở `status.txt` (dòng `SEED <k> DONE` và `CHAIN ... DONE`). Một seed của ba mô hình = 192,0 phút (3,2 giờ), bốn seed = **12,8 giờ**: dự kiến xong seed 1 khoảng 01:15, seed 2 khoảng 04:25, seed 3 khoảng 07:40, **seed 4 khoảng 10:50 ngày 10-26** (sai số −10% đến +15%: 09:35 đến 12:45). Lúc khởi chạy RAM trống 2,35/7,88 GB, GPU rảnh (263/3072 MiB), không còn tiến trình python nào chạy dở. Nếu có ô bị hệ thống dừng: báo, không tự chạy lại.

**Chạy xong seed 1–4 và kết quả 5 seed [CN 10-26 12:46].** Chuỗi chạy từ 22:02 ngày 25-10 đến 10:44 ngày 26-10 (12,7 giờ, sát ước lượng 12,8 giờ). Mọi ô seed 1–4 của ART, Eadro, OrEdge đều `rc=0`, không ô nào bị dừng; mỗi mô hình có đủ 5 seed (0–4). Phút mỗi seed: ART 17,8–19,6; Eadro 63,5–63,8; OrEdge 107,4–109,5. **AnoFusion giữ 1 seed (seed 0)** theo quyết định của người dùng, ghi là kết quả 1 seed, không sao chép sang seed khác, không có độ lệch chuẩn. Bảng đầy đủ nằm ở khối tự sinh bên dưới ("Theo dataset (5 seed)", "Mức 90%", "Theo scenario (5 seed)"); độ lệch chuẩn là `np.std` (tổng thể) như các bảng bước 8, nên khoảng "độ lệch chuẩn / trung bình" ghi ở các đoạn seed 0 phía trên (tính bằng `ddof=1`) lệch nhẹ so với con số dưới đây.

**Trung bình 5 dataset, 5 seed (trung bình±độ lệch chuẩn qua seed của trung bình các dataset)** — F1@val95 / AUROC / AUPRC: UAM-AD 0,586±0,016 / 0,841±0,003 / 0,633±0,005; ART 0,572±0,013 / 0,856±0,004 / 0,686±0,008; Eadro 0,637±0,007 / 0,821±0,005 / 0,709±0,005; OrEdge 0,701±0,020 / 0,888±0,008 / 0,795±0,013; AnoFusion (1 seed, 4 dataset) 0,486 / 0,739 / 0,567. Chênh lệch của UAM-AD (% giá trị của mô hình kia, F1 / AUROC / AUPRC): ART +2,5% / −1,8% / −7,7%; AnoFusion +24,4% / +18,5% / +24,3%; Eadro −7,9% / +2,4% / −10,7%; OrEdge −16,4% / −5,3% / −20,4%. So với seed 0, dòng Trung bình gần như không đổi (lệch tối đa gần 5 điểm %: F1 so với ART từ −0,6% thành +2,5%, F1 so với AnoFusion từ +19,6% thành +24,4%).

**HR@1 (đơn vị chấm khác, không tính %)**, 5 seed, thứ tự RE2-OB, RE3-OB, RE2-TT, RE3-TT: UAM-AD 0,352 / 0,023 / 0,413 / 0,079; ART 0,753 / 0,727 / 0,447 / 0,344; Eadro 0,667 / 0,853 / 0,539 / 0,621; OrEdge 0,927 / 0,941 / 0,954 / 0,905. Định vị vẫn là chỗ thua rõ nhất ở RCAEval, kể cả với ART (không giám sát). Ở SN HR@1: UAM-AD 0,194; ART 0,058; Eadro 0,074; OrEdge 0,262.

## Việc 9: lời dẫn cũ của mục SUMMARY và tiêu đề

Mục này là bản tổng hợp tra cứu của việc 9 (so UAM-AD với ART, AnoFusion, Eadro, OrEdge), đặt ở cuối giai đoạn 3, sau phần ghi theo trình tự làm việc của mục việc 9 ở trên. Hai bảng T1 và T2 **sinh tự động** bởi `summary_tools/item9_tables.py` từ cùng dữ liệu với khối "Kết quả 5 seed" ở mục việc 9 phía trên; phần nhận xét viết tay [CN 10-27], dùng số của hai bảng này.

## Giai đoạn 1 và 2: các dòng bảng việc 1–5 (bản cũ, trước khi rút gọn ngày 2026-10-30)

Ở `chon_dataset_va_sota.md` chỉ bỏ nhãn ngày khỏi chữ "Xong" (cả năm dòng), cắt câu về commit và nhánh ở dòng 1, và cắt đoạn "Lỗi môi trường đã sửa" (pandas 3.0.1) ở dòng 2. Nguyên văn các dòng 1 đến 5 lúc trước:

| 1 | Tách `val.pkl` cho RE2-OB, RE3-OB (chỉ cửa sổ normal, không trùng train/test) | – | **Xong [CN 10-13]**: code đã có ở commit `f7c9e2b` (nhánh local `Reval-re2-re3`, chưa vào `develop`), đã đưa sang `total_experiment` (chưa commit); pkl có sẵn trong `data/`. val = 15% cuối theo thời gian của mỗi run, không trùng train/unlabel. Giới hạn test normal: xem dưới Bảng A |

| 2 | Viết **một bộ đọc trace TrainTicket dùng chung**, rồi preprocess RE2-TT, RE3-TT (kèm `val.pkl`). Kiểm số service của TT và ground-truth service lỗi cho từng nguồn. AnoMod-TT đã bỏ [CN 10-13] | – | **Xong [CN 10-13]**: hai script `preprocess_rcaeval_re{2,3}_tt.py` tái dùng lớp của bản OB, chỉ đổi danh sách service/loại lỗi. Số service TT = 27 (cả hai nguồn). Ground-truth service lấy từ id mẫu như OB (đã thêm tên TT vào `rca.py`, `run.py`): 100% anomaly có GT, nên TT dùng được cho Bảng D. `trace_node_features[:, 3]` (error_rate) luôn 0 vì `statusCode` rỗng ở mọi run TT. **Lỗi môi trường đã sửa:** pandas 3.0.1 làm `astype(str)` giữ NaN nên mọi span bị tính là lỗi (error_rate = 1) và span gốc không nhận ra; đã sửa ở cả hai script OB (kết quả giữ nguyên với pandas cũ), pkl OB đã có không đổi |

| 3 | Đo "dấu vân cấu trúc" trên dữ liệu thô DeepTraLog | – | **Xong [CN 10-13]**: kết quả ở mục 2.3 (script `codes/common/measure_structural_fingerprint.py`). Không F-case nào có đa số lỗi cấu trúc, nên vế 2 đã đổi. Đã tạo `scenarios/test_F05.pkl`, `test_F09.pkl` và chạy thử chỉ-sync 1 seed: cả hai đạt (AUROC 0,994 và 0,925), chốt F05, F09 |

| 4 | Điền số liệu thật vào Bảng A | 1, 2, 3 | **Xong [CN 10-13]**: Bảng A đã đủ số liệu cho cả 6 dataset |

| 5 | Chốt **một cấu hình cố định** cho UAM-AD và UAC-AD (mục 2.2): thống nhất ngân sách huấn luyện (số epoch chốt sau khi chạy thử 1 scenario RE2-OB với 5 so với ~20 epoch và chỉ nhìn val loss; 50 epoch toàn bộ có thể tốn gần 10 lần thời gian), `val_percentile=95`, 5 seed. Ghi cấu hình ra file rồi **đóng băng** | 1 | File cấu hình, không đổi sau khi thấy kết quả | **Xong [CN 10-14]:** chốt 5 epoch cho RE2/RE3 (patience 3, window 30, batch 128), vì thử 15 epoch trên RE2-OB `cpu` (1 seed) chỉ nâng AUROC 0,860→0,880 và F1@val 0,504→0,523 mà tốn gấp 3 lần thời gian; khớp các lần chạy RE2/RE3-OB trước đó. Cấu hình ở `codes/common/frozen_config.json`. DeepTraLog giữ thiết lập lần chạy trước (10 epoch, patience 5); SN giữ thiết lập lần chạy trước của SN là 50 epoch, patience 15 (theo `docs/experiment_results_sn_trace_vs_baseline_vi.md`), chưa chỉnh lại

## Rút gọn hai mục SUMMARY ngày 2026-10-30: nguyên văn các đoạn trước khi rút gọn

Ở `chon_dataset_va_sota.md` các đoạn dưới đây được rút gọn hoặc bỏ (chỉ phần chữ, không đụng bảng); số liệu giữ nguyên, chỉ bớt chỗ nhắc lại số có sẵn trong bảng, chi tiết tiến trình và danh sách ô. Hai tiêu đề được thêm "Việc 8 –" và "Việc 9 –". Thứ tự theo vị trí trong file.

- **SUMMARY experiments between UAC-AD and UAM-AD** (việc 8): S1 chi phí, S2 theo dataset, S3 theo scenario, S5 điều số liệu đỡ được, S6 luật chấm.

- **SUMMARY experiments between UAM-AD and the other models** (việc 9, so với ART, AnoFusion, Eadro, OrEdge): T1 bảng so sánh và nhận xét theo từng mô hình, T2 mức 80%, T3 mức 90% và nhiễu seed, T4 cấu hình chạy của các đối thủ.

#### SUMMARY experiments between UAC-AD and UAM-AD

Mục này là bản tổng hợp kết quả của bước 8 (UAM-AD so với UAC-AD), để tra cứu. Mọi số trong mục này **sinh tự động từ thư mục kết quả** `D:\ClaudeWork\phase3\item8\<dataset>\{uacad,uamad}\<scenario>\<hash>\` (`info_score.txt`, `params.json`, `running.log`, `rca_results.json`), 5 seed (0–4) × 33 scenario × 2 mô hình = 330 lần chạy, không gõ tay. Cấu hình đóng băng (`codes/common/frozen_config.json`), luật chấm `norm_sum`.

- **F1 oracle**: cột `oracle_f1` do `run.py` ghi: quét ngưỡng trên test lấy F1 cao nhất **và có point-adjust**. Chỉ để đối chiếu với các bài quét ngưỡng (UAC-AD, TraceDAE), không dùng để kết luận, không đặt cạnh F1@val95 (số này luôn cao hơn), và không so với cột oracle của DeepTraLog GGNN (quét ngưỡng thường, không point-adjust).

- **HR@1/3/5** (xác định đúng service lỗi): chỉ có ở **UAM-AD** (chạy với `--enable_rca`). UAC-AD không có bước RCA nên các cột HR của UAC-AD ghi "–". HR của một scenario = tỉ lệ cửa sổ có service lỗi mà service đó nằm trong top-k xếp hạng; HR của dataset = trung bình qua các scenario có service lỗi (SN có 3 scenario hiệu năng hệ thống không gắn với một service nên không có HR). **HR@5 của seed 0 ghi "–"** vì `rca_results.json` của seed 0 chỉ lưu top-3 (từ seed 1 trở đi lưu top-5); các số "5 seed" của HR@5 vì vậy là trung bình±độ lệch chuẩn qua seed 1–4. Mức ngẫu nhiên k/N: SN (12 service) 8% / 25% / 42%; RE3-OB và RE2-OB (11) 9% / 27% / 45%; RE3-TT và RE2-TT (27) 4% / 11% / 19%.

- Mỗi dòng của dataset = **trung bình qua các scenario** của dataset đó; dạng "a±b" = trung bình±độ lệch chuẩn tổng thể (`np.std`) qua 5 seed. Cột cuối là thời gian thật của lần chạy (từ dòng log đầu đến dòng log cuối, gồm nạp dữ liệu, huấn luyện, đánh giá, và RCA với UAM-AD): ở S2 là **giờ** trung bình mỗi seed (tổng các scenario của dataset); ở S3 là **phút** trung bình mỗi lần chạy; ở S4 là **phút** của lần chạy đó (dòng dataset là tổng các scenario của seed đó).

- Số thập phân dùng dấu phẩy. Các bảng kết quả S2, S3, S4 dùng cùng một bố cục cột; bảng chi phí S1 có bố cục riêng.

Số 1,16 là tỉ lệ thô và bị nhiễu bởi một ô của UAC-AD: rcaeval_re2_tt / `socket` / seed 2 mất 12137 giây trong khi trung vị các seed còn lại của ô đó là 1430 giây (người dùng chơi game cùng lúc, epoch 0 mất 7534 giây). Thay ô đó bằng trung vị thì UAC-AD ≈ 26,9 giờ và **tỉ lệ UAM-AD / UAC-AD = 34,7 / 26,9 ≈ 1,29**, là con số phản ánh chi phí thêm nhánh trace tốt hơn (không ô nào khác, của cả hai mô hình, vượt 3 lần trung vị các seed còn lại của chính nó). Khi viết luận văn ghi cả hai con số, kèm lý do.

Theo dataset (giờ = tổng các lần chạy; "giây/epoch" = trung vị của `Train time/epoch` trong `info_score.txt`, không chịu ảnh hưởng của việc nạp dữ liệu hay RCA; "epoch chạy" = số epoch thực chạy trung bình mỗi lần chạy, đếm từ log, vì dừng sớm làm giờ thấp đi):

- Tỉ lệ giây/epoch (cột thuần huấn luyện) là 1,02 ở SN, 1,18 ở RE3-OB, 1,16 ở RE2-OB, 1,39 ở RE3-TT, 1,45 ở RE2-TT: nhánh trace tốn thêm cỡ 2–45% mỗi epoch, nhiều nhất ở hai dataset TT. Tỉ lệ giờ còn gồm bước nạp dữ liệu trace và (với UAM-AD) bước RCA.

- **Số epoch thực chạy không bằng nhau, nên giờ không so thẳng được ở SN**: UAM-AD luôn chạy đủ 50 epoch ở SN, còn UAC-AD chỉ chạy trung bình 19,0 epoch (seed 0) và 18,0 epoch (seed 3) — đúng hai seed UAC-AD sập (AUROC 0,551 và 0,564) — dừng sớm vì điểm val không tốt hơn trong 15 epoch (patience 15); ở RE3-OB, UAC-AD cũng dừng ở 4/5 epoch ở seed 0–2 và RE3-TT seed 0. Đó là lý do giờ của UAC-AD thấp hơn ở những ô này. Việc dừng sớm trùng với việc sập ở SN là một quan sát, chưa phải chẩn đoán nguyên nhân.

- Thời gian log chỉ tính các lần chạy **thành công**, lấy từ lần khởi động cuối của `running.log` (ô chạy lại sau khi chết im lặng thì chỉ tính lần chạy lại). Tổng thời gian của driver (`status.txt`, gồm cả lần chạy chết và khởi động tiến trình) là 35,1 giờ cho UAM-AD và 30,5 giờ cho UAC-AD, nhỉnh hơn tổng log 0,4 và 0,7 giờ. Các ô có hai lần khởi động trong log: UAC-AD SN `Svc_Kill_SocialGraph` seed 3, RE3-OB `f4` seed 1 và seed 4, RE2-TT `loss` seed 3; UAM-AD SN `Code_Stop_TextService` seed 1.

- **HR của từng scenario SN rất nhiễu vì chỉ có vài cửa sổ được xếp hạng** (cửa sổ có service lỗi và vượt ngưỡng `raw_sum` p95; số cửa sổ min–max qua 5 seed): `Code_Stop_MediaService` 31–35; `Code_Stop_TextService` 29–30; `Code_Stop_UserService` 17–33; `DB_Redis_CacheLimit_HomeTimeline` 10–10; `DB_Redis_CacheLimit_SocialGraph` 9–9; `DB_Redis_CacheLimit_UserTimeline` 4–6; `Svc_Kill_Media` 4–4; `Svc_Kill_SocialGraph` 1–2; `Svc_Kill_UserTimeline` 2–3. Giá trị 0,000 hoặc 1,000 ở các scenario `Svc_Kill_*` chỉ dựa trên 2–4 cửa sổ; chỉ HR cấp dataset mới đáng tin hơn.

- **Chi phí:** UAM-AD tốn 34,7 giờ so với 29,9 giờ của UAC-AD cho 5 seed × 33 scenario, tỉ lệ thô 1,16, và 1,29 sau khi thay ô UAC-AD bị tranh tài nguyên. Chi phí huấn luyện mỗi epoch chỉ cao hơn 2–45% (SN 1,02; RE3-OB 1,18; RE2-OB 1,16; RE3-TT 1,39; RE2-TT 1,45). Tỉ lệ giờ ở SN và RE3 còn bị lệch bởi việc UAC-AD dừng sớm (S1).

- **RE2-TT: UAM-AD hơn rõ** (F1 +0,310, AUROC +0,134, AUPRC +0,240), ổn định theo seed (S2), với giá chi phí thêm cỡ 1,4 lần.

- **SN: UAM-AD ổn định, UAC-AD sập ở seed 0 và 3.** Mức hơn trung bình (F1 +0,205, AUROC +0,161, AUPRC +0,123) chủ yếu do hai seed sập; chỉ tính các seed không sập (1, 2, 4) thì AUROC, AUPRC ngang nhau (xem S4, từng seed).

- **RE2-OB:** AUROC +0,036, AUPRC +0,093, F1 +0,041 nhưng độ lệch chuẩn F1 khoảng 0,08 ở cả hai mô hình (cả hai có seed "nhảy"), nên F1 không phân biệt được.

- **RE3-OB, RE3-TT: UAM-AD thấp hơn UAC-AD dưới `norm_sum`** (RE3-OB F1 −0,087, AUROC −0,023, AUPRC −0,080; RE3-TT F1 −0,026, AUROC −0,042, AUPRC −0,086), hệ thống theo seed; dưới `raw_sum` chênh lệch gần 0 (bảng phụ ở S2).

- **F1 oracle (quét ngưỡng trên test, có point-adjust)** cho cùng chiều với AUROC ở cả 5 dataset nhưng khoảng cách nhỏ hơn nhiều: SN +0,091, RE3-TT −0,012, RE3-OB −0,005, RE2-OB +0,011, RE2-TT +0,091. Số này bị thổi phồng bởi point-adjust: UAC-AD ở SN seed 0 có AUROC 0,551 (gần ngẫu nhiên) mà F1 oracle vẫn 0,706. Vì vậy chỉ dùng để đối chiếu, không dùng để kết luận.

- **HR@k** chỉ có ở UAM-AD (S2): RE2-OB và RE2-TT hơn hẳn ngẫu nhiên; SN hơn ở HR@1 và HR@5 nhưng HR@3 ≈ ngẫu nhiên; RE3-TT HR@1 gấp khoảng 2 lần ngẫu nhiên nhưng tuyệt đối thấp; RE3-OB HR@1 và HR@3 thấp hơn ngẫu nhiên (hạn chế cần ghi trong luận văn). UAC-AD không có RCA nên không có hàng so sánh HR giữa hai mô hình; muốn có thì phải chạy thêm bước RCA cho UAC-AD (chưa làm).

**Cách tái tạo.** Hai script nằm ngoài vùng git, cạnh kết quả: `D:\ClaudeWork\phase3\item8\summary_tools\load_step8.py` (đọc `info_score.txt`, `params.json`, `running.log`, `rca_results.json` của 330 lần chạy) và `build_summary.py` (sinh bốn tệp `summary_part1.md` đến `summary_part4.md` cùng thư mục; mục này là bốn tệp đó ghép lại theo thứ tự). HR@k tính lại từ `rca_results.json` bằng cùng cách của `hr_table.py` và khớp với các cột HR@k của S2 (kiểm trên cả 5 dataset). Mỗi ô lỗi chết im lặng đã được chạy lại nên mọi tổ hợp (mô hình, dataset, scenario) có đúng seed 0–4.

- **`raw_sum`** là luật gốc: điểm = `fusion_loss` của mô hình, tức `log_kpi_loss + g·trace_d + λ·g` (trace nhân với cổng `g`; SN còn cộng số hạng phạt hoạt động). Các lần chạy cũ của SN và OB dùng luật này. **`norm_sum`** được thêm khi làm nhánh async trên DeepTraLog: mỗi số hạng (`log_kpi_loss`, `trace_dis`, `trace_err`) được chuẩn hóa theo trung vị và độ lệch p95 của val rồi cộng với trọng số 1; `trace_dis` và `trace_err` **không nhân cổng**, và không có số hạng phạt hoạt động. Cấu hình đóng băng ngày 10-14 chọn `norm_sum` cho mọi dataset. Cả hai điểm đều có trong `info_score.txt` của mọi lần chạy (cùng checkpoint, không tốn thêm lần chạy).

Ví dụ: F05 đủ bộ AUROC 0,994±0,000 (`norm_sum`) so với 0,489±0,024 (`raw_sum`, gần ngẫu nhiên; chỉ-async còn 0,296±0,004, dưới mức ngẫu nhiên); F13 chỉ-sync 0,989±0,000 so với 0,725±0,007; F01 đủ bộ F1 0,775±0,003 so với 0,151±0,022. **Ngoại lệ:** F09, nơi `raw_sum` tốt hơn (đủ bộ F1 0,560±0,025 so với 0,187±0,047, AUROC 0,964±0,000 so với 0,941±0,004). Nguyên nhân `raw_sum` hỏng ở DeepTraLog chưa được chẩn đoán; chỉ biết đó là số đo. Vì kết luận chính về async (bước 7) dựa trên `norm_sum`, dùng `raw_sum` cho bước 8 và `norm_sum` cho bước 7 nghĩa là mỗi dataset một luật chấm, chọn sau khi đã thấy kết quả test. Không làm vậy.

**Đề xuất cách trình bày:** giữ `norm_sum` làm luật chính duy nhất cho mọi dataset (đã đóng băng, đồng nhất với bước 7, và ở F01, F13, F05 của DeepTraLog cho AUROC đủ bộ 0,97–0,99 so với 0,49–0,86 của `raw_sum`), in `raw_sum` thành bảng song song đầy đủ ở cùng chỗ (S2 đã có), và chỉ khẳng định điều đúng ở cả hai luật: UAM-AD hơn UAC-AD ở RE2-TT, RE2-OB (AUROC, AUPRC) và ổn định hơn ở SN; không có lợi ở RE3 (≈ 0 hoặc thấp hơn tùy luật, vì nhánh trace không mang tín hiệu hữu ích ở lỗi mức mã). Mức hơn ở SN nên nói là độ ổn định, không phải điểm cao hơn. Một biến thể `norm_sum` có nhân cổng cho số hạng trace có thể làm như ablation, nhưng nó sẽ được đề ra sau khi đã thấy kết quả test nên phải ghi là post-hoc và không thay luật chính; chưa làm.

#### SUMMARY experiments between UAM-AD and the other models (ART, AnoFusion, Eadro, OrEdge)

Mục này là bản tổng hợp kết quả của việc 9 (so UAM-AD với ART, AnoFusion, Eadro, OrEdge), để tra cứu. Hai bảng T1 và T2 **sinh tự động** bởi `summary_tools/item9_tables.py` từ cùng dữ liệu với khối "Kết quả 5 seed" ở Phụ lục B; phần nhận xét viết tay, dùng số của hai bảng này.

**Cách đọc.** (1) Bảng T1 là bảng "Theo dataset (5 seed)" của khối tự sinh ở Phụ lục B, **đã bỏ cột "seed"**. (2) **Theo yêu cầu của người dùng, độ lệch chuẩn của AnoFusion ghi ±0,000 trong bảng này.** Đó là quy ước trình bày, **không phải số đo**: AnoFusion chỉ có 1 seed (seed 0), vì mỗi seed tốn 298,5 phút không kể RE3-TT (riêng RE2-TT 220 phút); không sao chép seed 0 sang seed khác. Khi viết thesis phải ghi "AnoFusion: 1 seed". (3) Mỗi ô "a±b" của UAM-AD, ART, Eadro, OrEdge là trung bình±độ lệch chuẩn tổng thể (`np.std`, như bước 8) qua 5 seed của trung bình theo scenario. (4) **Δ** = UAM-AD trừ mô hình ở dòng đó, trong ngoặc là % giá trị của mô hình đó (dương = UAM-AD cao hơn); so với AnoFusion thì Δ = trung bình 5 seed của UAM-AD trừ seed 0 của AnoFusion. Dòng AnoFusion RE3-TT chỉ có `test_f1` (dừng sớm), dấu * và không vào dòng Trung bình; % của nó (+348%) phình ra vì nền 0,116 rất thấp. Dòng Trung bình: % = chênh lệch trung bình chia giá trị trung bình của mô hình (không phải trung bình các %). (5) Ba chỉ số ưu tiên là F1@val95, AUROC, AUPRC. **F1 oracle, HR@k không so được giữa UAM-AD và các mô hình khác** (F1 oracle của UAM-AD có point-adjust, của các mô hình kia không; HR@k khác đơn vị chấm) nên bảng giữ các cột đó để tra, không dùng để kết luận. (6) Mọi mô hình trong bảng dùng đủ ba modality L+M+T. Eadro (có giám sát) và OrEdge (bán giám sát) là dòng tham chiếu có nhãn; ở SN chúng chạy leave-one-scenario-out, OrEdge SN chỉ khoảng 150 bước huấn luyện, Eadro dùng log 2 đặc trưng; ART và AnoFusion ở SN dùng `--pad_normal`.

**Nhận xét theo từng mô hình, từng dataset** (số trong ngoặc là Δ% của UAM-AD so với mô hình, lấy từ bảng T1; "cả 5 seed cùng chiều" nghĩa là chênh lệch theo từng seed luôn cùng dấu, "hoà" là dấu lẫn giữa các seed).

- RE2-OB: kém ở cả ba (F1 −35,9%, AUROC −8,0%, AUPRC −16,5%), cả 5 seed cùng chiều. ART F1 0,880 gần như không đổi giữa các seed (sd 0,001), còn UAM-AD 0,564±0,075. Đây là dataset UAM-AD yếu nhất so với ART.

- RE3-OB: F1 hoà (+7,1%, 4/5 seed dương); kém AUROC 6,6% và AUPRC 17,5%.

- RE2-TT: gần ngang. F1 hoà (−2,8%), AUPRC hoà (0,0%), AUROC kém nhẹ 3,1% (cả 5 seed cùng chiều).

- RE3-TT: F1 hơn rõ (+59,3%: 0,513 so với 0,322), AUROC hoà (+2,8%, 4/5 seed dương), AUPRC kém 22,7%.

- SN: hơn ở cả ba (+31,5%, +7,5%, +21,5%), cả 5 seed cùng chiều.

- RE2-OB: hơn AUROC 10,3% và AUPRC 15,3%; F1 kém 3,8% (0,564 so với 0,587), nhỏ hơn độ lệch chuẩn F1 của UAM-AD ở đây (0,075) nên chưa phân biệt được.

- RE3-OB: hơn rõ ở cả ba (F1 +53,1%, AUROC +32,4%, AUPRC +37,8%); AnoFusion yếu ở đây (AUROC 0,630).

- RE2-TT: sát nhau. AUROC +4,4%, AUPRC +2,4%, F1 kém 2,9% (nhỏ hơn sd 0,025 của UAM-AD). AnoFusion mạnh nhất ở dataset này nhưng cũng tốn 220 phút cho một seed.

- RE3-TT: AnoFusion chỉ có `test_f1` nên so trên một file (F1 +348%, AUROC +43,2%, AUPRC +73,9%, dấu *); không dùng để kết luận.

- SN: hơn rất rõ (F1 +78,5%, AUROC +30,8%, AUPRC +57,4%).

- Trung bình 4 dataset: +24,4% F1, +18,5% AUROC, +24,3% AUPRC. Hai ô kém đều là F1 và đều nhỏ. Vì chỉ 1 seed, chưa biết nhiễu seed của AnoFusion, nên "hơn" ở các ô chênh dưới vài % (RE2-TT) không nên khẳng định.

- RE2-OB: F1 kém 20,6% (0,564 so với 0,711), AUROC hơn 4,2% (cả 5 seed cùng chiều), AUPRC hoà (+1,1%).

- RE3-OB: F1 kém 13,2%, AUROC hoà (+2,2%, 4/5 seed dương), AUPRC kém 15,3%.

- RE2-TT: kém cả ba nhưng nhẹ (F1 −11,5%, AUROC −3,6%, AUPRC −5,5%).

- RE3-TT: hơn F1 (+40,4%) và AUROC (+16,0%) vì Eadro yếu ở lỗi mức mã (AUROC 0,606); kém AUPRC 21,7%.

- SN: kém ở cả ba (−11,2%, −2,4%, −16,4%); Eadro đạt AUROC 0,980 ở SN.

- RCAEval (RE2-OB, RE3-OB, RE2-TT, RE3-TT): kém ở cả 12 ô, cả 5 seed cùng chiều. F1 −34,9%, −26,5%, −31,8%, −18,6%; AUROC −7,2%, −6,5%, −16,5%, −8,1%; AUPRC −16,6%, −26,1%, −27,5%, −49,8%. OrEdge đạt AUROC 0,76–0,97 và AUPRC 0,69–0,96 ở RCAEval.

- SN: hơn ở cả ba (F1 +74,9%, AUROC +13,3%, AUPRC +31,1%), cả 5 seed cùng chiều. Không nên đọc đây là ưu thế của mô hình: OrEdge SN có độ lệch chuẩn rất lớn (F1 0,416±0,108, AUROC 0,844±0,039), chỉ khoảng 150 bước huấn luyện và dùng giao thức khác.

**HR@1 (định vị service lỗi; đơn vị chấm khác nên không tính %).** Ở RCAEval UAM-AD thấp nhất: 0,352 / 0,023 / 0,413 / 0,079 (RE2-OB, RE3-OB, RE2-TT, RE3-TT), so với ART 0,753 / 0,727 / 0,447 / 0,344, Eadro 0,667 / 0,853 / 0,539 / 0,621, OrEdge 0,927 / 0,941 / 0,954 / 0,905. Ở SN: UAM-AD 0,194, ART 0,058, Eadro 0,074, OrEdge 0,262. Định vị là chỗ yếu rõ nhất của UAM-AD, kể cả so với ART không giám sát.

**Tóm lại.** Phát hiện bất thường: UAM-AD hơn AnoFusion, xấp xỉ ART (hơn ở SN và RE3-TT F1, kém ở RE2-OB và ở AUPRC của OB và RE3-TT), ngang Eadro về AUROC nhưng kém về F1 và AUPRC, và kém OrEdge ở RCAEval. Điểm mạnh nhất của UAM-AD là SN (hơn ART, AnoFusion, OrEdge; chỉ kém Eadro) và độ ổn định giữa các seed ở AUROC và AUPRC: độ lệch chuẩn chia trung bình, lấy trung bình qua 5 dataset, là 0,6% và 1,5% ở UAM-AD, so với 1,0% và 2,5% (ART), 1,7% và 2,6% (Eadro), 1,3% và 2,7% (OrEdge); ở F1@val95 không nổi bật (UAM-AD 4,4%, ART 4,3%, Eadro 3,8%, OrEdge 5,9%).

**Nhận xét mức 80%.** (1) **Mức 80% do người dùng chọn ngày 10-27, sau khi đã thấy kết quả ở mức 90%** (14/30 ô, ở T3 phía trên). Đây là mức chấp nhận của người viết, không lấy từ tài liệu nào đã kiểm chứng, và thesis nên ghi đúng như vậy và báo kèm Δ% từng chỉ số (bảng T1) thay vì chỉ báo đạt hay không. Mức áp chung cho cả ba chỉ số và cả hai mô hình có nhãn, không đổi theo chỉ số. (2) **UAM-AD đạt 22/30 ô** (ở mức 90% là 14/30): Eadro 13/15, OrEdge 9/15. AUROC 10/10 (thấp nhất là OrEdge ở RE2-TT, 83,5%); F1@val95 6/10 (Eadro 4/5, OrEdge 2/5); AUPRC 6/10 (Eadro 4/5, OrEdge 2/5). (3) **Tám ô không đạt:** F1@val95 so với Eadro ở RE2-OB (79,4%); F1@val95 so với OrEdge ở RE2-OB (65,1%), RE3-OB (73,5%), RE2-TT (68,2%); AUPRC so với Eadro ở RE3-TT (78,3%); AUPRC so với OrEdge ở RE3-OB (73,9%), RE2-TT (72,5%), RE3-TT (50,2%). (4) Đạt cùng lúc cả hai mô hình có nhãn: AUROC 5/5 dataset, F1@val95 2/5 (RE3-TT, SN), AUPRC 2/5 (RE2-OB, SN). **Chỉ SN đạt cả sáu ô**; RE3-TT trượt AUPRC, ba dataset còn lại trượt F1 so với ít nhất một trong hai mô hình. (5) Dòng Trung bình: 5/6 ô đạt, ô trượt là AUPRC so với OrEdge (79,6%, thiếu 0,4 điểm %).

**Mức độ nhạy với vạch 80%.** Ba ô nằm trong ±2 điểm % của vạch: 79,4% (trượt), 78,3% (trượt) và 81,4% (đạt); dòng Trung bình AUPRC so với OrEdge 79,6% cũng sát vạch. Kết quả đếm vì vậy nhạy với vạch ở vùng 78–82%: hạ vạch xuống 78% thì thêm 2 ô, nâng lên 82% thì mất 1 ô. Nhiễu seed của UAM-AD (sd/trung bình: F1 tới 13,3% ở RE2-OB, AUROC tối đa 1,4%, AUPRC tối đa 2,8%) nhỏ hơn khoảng cách tới vạch ở hầu hết các ô; RE2-OB F1 là ngoại lệ, nơi 79,4% có thể lên xuống quanh vạch giữa các seed.

**Giới hạn khi đọc.** (a) AUROC có sàn 0,5, nên đạt 80% AUROC là phát biểu yếu: 80% của AUROC 0,967 (OrEdge RE2-TT) là 0,774; AUPRC có sàn bằng tỉ lệ cửa sổ lỗi (khoảng 13–20%). Vì vậy 10/10 ở AUROC không nói UAM-AD "ngang" mô hình có nhãn, mà nói điểm xếp hạng của nó không thấp hơn nhiều. (b) Tiêu chí chỉ xét phát hiện, không xét định vị: HR@1 của UAM-AD ở RCAEval (0,02–0,41) thấp hơn nhiều so với Eadro (0,54–0,85) và OrEdge (0,91–0,95). (c) Eadro và OrEdge có nhãn và dùng giao thức riêng ở SN (leave-one-scenario-out); đây là dòng tham chiếu, không phải đối thủ cùng điều kiện. (d) Cấu hình của Eadro và OrEdge theo quy tắc epoch chung cho mọi mô hình (RCAEval 5 epoch, SN 50 epoch), không tinh chỉnh riêng; riêng đặc trưng log theo node của OrEdge ở SN được chọn sau khi đã xem kết quả SN.

**Mức 90%** (chốt trước mức 80% của T2): "UAM-AD giữ ≥ 90% giá trị của mô hình có nhãn", áp chung cho F1@val95, AUROC, AUPRC. **90% là con số được đề xuất sau khi đã thấy số seed 0, không lấy từ tài liệu nào đã kiểm chứng**; người dùng chốt giữ mức này cho cả ba chỉ số (10-25), không hạ riêng cho từng chỉ số. Sàn của AUROC là 0,5 và sàn của AUPRC là tỉ lệ lỗi (khoảng 13–20%), nên cùng một mức % thì đạt AUROC là phát biểu yếu hơn đạt AUPRC.

**Mức 90% so với Eadro và OrEdge, trung bình 5 seed của cả hai phía: UAM-AD đạt 14/30 ô** (seed 0: 15/30; ô đổi là F1@val95 của SN so với Eadro, 90,5% ở seed 0, 88,8% ở trung bình 5 seed). AUROC 9/10 (Eadro 5/5; OrEdge 4/5, trừ RE2-TT với 83,5%); F1@val95 2/10 (Eadro: RE3-TT 140,4%; OrEdge: SN 174,9%); AUPRC 3/10 (Eadro: RE2-OB 101,1%, RE2-TT 94,5%; OrEdge: SN 131,1%). Đạt cả hai mô hình cùng lúc: AUROC 4/5 dataset, F1@val95 0/5, AUPRC 0/5. Dòng Trung bình: AUROC đạt cả hai (102,4% và 94,7%); F1@val95 đạt so với Eadro (92,1%, nhờ RE3-TT hơn 40%) và không đạt so với OrEdge (83,6%); AUPRC không đạt (89,3% so với Eadro, 79,6% so với OrEdge). Mức 90% không đổi và không hạ riêng cho từng chỉ số. **Câu đỡ được trong thesis:** "UAM-AD giữ ≥ 90% AUROC của cả hai mô hình có nhãn ở 4/5 dataset; ở F1@val95 và AUPRC còn thấp hơn rõ ở RCAEval (ở các ô không đạt, kém Eadro 11,2–21,7% và kém OrEdge 16,5–49,8%); ở SN vượt OrEdge ở cả ba chỉ số nhưng kém Eadro".

**Nhiễu giữa seed (độ lệch chuẩn / trung bình, khoảng qua 5 dataset; lần đầu có cho cả đối thủ).** UAM-AD: F1@val95 0,4–13,3% (RE2-OB lớn nhất), AUROC 0,2–1,4%, AUPRC 0,4–2,8%. ART: 0,2–11,2%, 0,2–2,2%, 0,4–6,5%. Eadro: 1,9–7,6%, 0,3–3,3%, 1,3–3,8%. OrEdge: 0,2–25,9%, 0,3–4,7%, 0,3–11,2% (lớn nhất ở SN: F1 0,416±0,108, AUROC 0,844±0,039, nên dòng OrEdge SN không ổn định, nhưng UAM-AD vẫn hơn nó ở cả 5 seed). **Chênh UAM-AD trừ mô hình, tính theo từng seed, cùng dấu ở cả 5 seed** trong 28/30 ô so với Eadro và OrEdge (hai ô lẫn dấu: Eadro RE2-OB AUPRC 3 dương/2 âm; Eadro RE3-OB AUROC 4 dương/1 âm), nên các khoảng cách với hai mô hình có nhãn **không** do nhiễu seed. So với ART có 11/15 ô cùng dấu; **4 ô coi là hoà**: F1 ở RE3-OB (+7,1%, 4/5 seed) và RE2-TT (−2,8%, 4/5 seed âm), AUROC ở RE3-TT (+2,8%, 4/5 seed), AUPRC ở RE2-TT (0,0%).

**Nhận định "SN ta cao nhất, OB và TT không thua quá nhiều" với 5 seed (so ART và AnoFusion).** *AUROC:* UAM-AD cao nhất nhóm ở SN (+7,5% so với ART, +30,8% so với AnoFusion); RE3-TT hoà với ART (+2,8%); ở RE2-TT, RE3-OB, RE2-OB kém ART 3,1%, 6,6%, 8,0% (cả năm seed cùng dấu) và hơn AnoFusion 4,4–32,4%. *F1@val95:* cao nhất nhóm ở SN (+31,5% so với ART), RE3-TT (+59,3%); hoà với ART ở RE3-OB, RE2-TT; **kém ART 35,9% ở RE2-OB** (cả năm seed), chỗ yếu rõ nhất. *AUPRC:* chỉ SN cao nhất (+21,5% so với ART); kém ART 16,5% (RE2-OB), 17,5% (RE3-OB), 22,7% (RE3-TT), hoà ở RE2-TT; trung bình kém ART 7,7%. Vậy câu "không thua quá nhiều" đỡ được cho AUROC (không dataset nào kém ART quá 8%) và cho F1@val95 trừ RE2-OB, **không đỡ được cho AUPRC** ở RE2-OB, RE3-OB, RE3-TT. UAM-AD hơn AnoFusion ở trung bình cả ba chỉ số (+18,5% đến +24,4%) nhưng AnoFusion chỉ có 1 seed và RE3-TT chỉ có `f1`.

- **SN, ART và AnoFusion:** đệm lịch sử bằng đuôi phiên Normal_Baseline (`--pad_normal`), vì hai mô hình cần chuỗi cửa sổ liền trước mà mỗi phiên SN bắt đầu ngay lúc có lỗi. Chỉ dùng dữ liệu train (chính các cửa sổ Normal_Baseline mà UAM-AD cũng học), không dùng test hay nhãn, nên hai bên chấm đúng cùng tập test. Hai giới hạn: đệm chỉ giúp vài cửa sổ đầu phiên (5 cho ART, 20 cho AnoFusion), từ đó trở đi lịch sử là chính các cửa sổ lỗi, nên với lỗi kéo dài (nhất là 3 phiên `Code_Stop_*`) điểm của chúng có thể thấp vì bản chất mô hình (chưa đo); và chưa có dòng ART không đệm để thấy đệm có làm số đổi không.

## 2026-10-30 12:15–12:45: chẩn đoán vì sao UAM-AD kém ART ở RE2-OB (chỉ suy luận từ checkpoint đã lưu, không huấn luyện)

Người dùng hỏi vì sao kết quả so với mô hình khác lẫn lộn trên RCAEval, rồi ra lệnh thử thứ tự thời gian và đóng bớt chương trình để đủ RAM (đã đóng Steam helper và Zalo; phép thử bị hệ thống dọn một tác vụ chờ nền do thiếu RAM, tác vụ chính vẫn chạy xong).

- **Phép đo không mô hình** (`scratchpad/kpi_aggregation_probe.py`, `modality_probe.py`, `rcaeval_by_fault.py`): AUROC theo từng mẫu của độ lệch KPI so với trung bình train. RE2-OB: KPI trung bình MinMax 0,971, z-max 0,982 (UAM-AD 0,903, ART 0,981). RE3-OB 0,876 / 0,866 (UAM-AD 0,835, ART 0,893). RE2-TT 0,755 / 0,903 (UAM-AD 0,808, ART 0,833, OrEdge 0,967). RE3-TT 0,768 / 0,779 (UAM-AD 0,702, OrEdge 0,764). Kết luận: câu "trung bình toàn kênh làm loãng tín hiệu" ở Phụ lục C chỉ đúng ở TT (383 kênh), không đúng ở OB.
- **Thử thứ tự thời gian** (`scratchpad/order_probe_run.py`, `order_probe_analyze.py`, `order_probe_combos.py`): checkpoint seed 0 của RE2-OB `socket`, `cpu`, `loss`, chấm lại đúng các mẫu test theo hai thứ tự bằng `run.py --pre_model --dump_components`. Thứ tự xáo tái tạo đúng số đã lưu (AUROC 0,749 / 0,878 / 0,857 so với 0,7492 / 0,8783 / 0,8568). Thứ tự thời gian thật (sắp theo thí nghiệm rồi giây): 0,759 / 0,901 / 0,948. Tỉ lệ cửa sổ lẫn cả hai lớp: 100% (xáo) so với 2% (thứ tự thời gian).
- **Từng số hạng của điểm** (AUROC, xáo): KPI một mình 0,846 / 0,879 / 0,794; log một mình 0,506 / 0,593 / 0,543; trace một mình 0,646 / 0,778 / 0,788; KPI+log 0,806 / 0,861 / 0,782; KPI+trace 0,828 / 0,916 / 0,844; điểm thật (`norm_sum`) 0,749 / 0,878 / 0,857. Thứ tự thời gian: KPI một mình 0,848 / 0,907 / 0,966.
- Giới hạn: một seed, ba scenario của RE2-OB; các tổ hợp số hạng chỉ để chẩn đoán (chọn tổ hợp theo nhãn test là không hợp lệ); chưa đo TT và RE3 bằng checkpoint.

## 2026-10-30 13:00–15:00: kiểm tra nhanh thứ tự thời gian trên cả 4 dataset RCAEval (chỉ suy luận từ checkpoint seed 0, không huấn luyện)

Người dùng ra lệnh kiểm tra nhanh để quyết định có đáng chạy lại không; cho phép đóng chương trình để đủ RAM (đã đóng Steam helper, Zalo, Edge). Mã: `scratchpad/order_all_run.py`, `order_all_analyze.py`. Mỗi scenario chấm lại đúng các mẫu test theo thứ tự (thí nghiệm, run, giây) bằng `run.py --pre_model --dump_components`; so với số đã lưu của cùng checkpoint. Thứ tự xáo tái tạo đúng số đã lưu ở RE2-OB (cpu, loss, socket), RE3-OB f1, RE3-TT f1 (sai khác < 0,0001).

AUROC xáo → thứ tự thời gian (F1@val95 trong ngoặc), trung bình qua scenario, seed 0:
- RE2-OB: 0,887 → 0,923 (0,495 → 0,713). Cả 6 lỗi đều tăng AUROC (+0,002 đến +0,091) và F1 (+0,023 đến +0,393).
- RE3-OB: 0,832 → 0,829 (0,556 → 0,544). Không đổi.
- RE2-TT: 0,809 → 0,810 (0,540 → 0,589). AUROC tăng ở delay, disk, loss, socket, giảm ở cpu (−0,039) và mem (−0,024); F1 tăng chủ yếu do disk (0,337 → 0,749).
- RE3-TT: 0,705 → 0,704 (0,515 → 0,515). Không đổi.

Từng số hạng của điểm trên test theo thứ tự thời gian (AUROC, trung bình qua scenario): RE2-OB KPI 0,947, log 0,696, trace 0,768, KPI+trace 0,954, điểm thật 0,923. RE3-OB KPI 0,863, log 0,510, trace 0,536, điểm thật 0,829. RE2-TT KPI 0,752, log 0,590, trace 0,731, KPI+trace 0,825, điểm thật 0,810 (KPI tốt ở disk, mem; trace tốt ở cpu, loss, socket). RE3-TT KPI 0,745, log 0,364, trace 0,382, điểm thật 0,704. Các tổ hợp chỉ để chẩn đoán (nhìn bằng nhãn test).

Ghi chú: lần chấm RE2-TT `disk` mất 2699 giây (thường 190–420 giây) do thiếu RAM; kết quả không bị ảnh hưởng. Tác vụ chờ nền bị hệ thống dọn một lần lúc 12:20 do thiếu RAM, không chạy lại.

## 2026-10-30 chiều: quyết định chạy lại RE2-OB theo thứ tự thời gian; kiểm tra đề xuất đổi cách gộp kênh KPI (chỉ phân tích dữ liệu, không chạy mô hình)

- **Quyết định của người dùng:** chạy lại UAM-AD trên RE2-OB để lấy số mới (chấm lại 30 checkpoint theo thứ tự thời gian thật, không huấn luyện lại). Làm sau, không chạy trong lúc đang thảo luận. Chấp nhận nếu vẫn thua ART mà không quá xa ở các dataset RE2 và RE3. Đã thêm Việc 10 vào mục 5 của `chon_dataset_va_sota.md`. Căn cứ: seed 0 đo thử RE2-OB AUROC 0,887 → 0,923, F1@val 0,495 → 0,713, cả 6 lỗi đều tăng (xem mục 13:00–15:00); ước 5 seed AUROC khoảng 0,94 (ước, chưa đo), ART 0,981.
- **Cần quyết khi chạy:** UAC-AD RE2-OB chấm lại cùng cách (chưa kiểm có checkpoint); ba dataset RCAEval còn lại không đổi AUROC khi sắp thứ tự (RE3-OB 0,832 → 0,829; RE2-TT 0,809 → 0,810; RE3-TT 0,705 → 0,704) nhưng đang chấm trên test xáo, nên bảng sẽ có hai giao thức nếu chỉ chấm lại RE2-OB.
- **Kiểm tra đề xuất "chuẩn hoá từng kênh + lấy vài kênh lệch nhất"** (`scratchpad/kpi_aggregation_probe.py` chạy lại, `range_std_probe.py`; theo mẫu, chỉ KPI, trung bình các lỗi, AUROC; nhìn bằng nhãn test nên chỉ để chẩn đoán):

| Dataset | MinMax, TB (như UAM-AD) | z, TB | z, top-3 | z, max | PCA16 MinMax, TB | PCA16 z, TB | PCA16 z, max | Số hạng KPI của UAM-AD (thứ tự thời gian) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | 0,971 | 0,977 | 0,980 | 0,982 | 0,988 | 0,987 | 0,985 | 0,947 |
| RE3-OB | 0,876 | 0,864 | 0,866 | 0,866 | 0,885 | 0,880 | 0,875 | 0,863 |
| RE2-TT | 0,755 | 0,867 | 0,902 | 0,903 | 0,835 | 0,901 | 0,908 | 0,752 |
| RE3-TT | 0,768 | 0,800 | 0,786 | 0,779 | 0,797 | 0,791 | 0,790 | 0,745 |

  - Phần lớn lợi ở RE2-TT đến từ **chuẩn hoá từng kênh** (MinMax → z: 0,755 → 0,867; PCA 0,835 → 0,901), phần lấy top-3 thêm khoảng +0,03 đến +0,04. Câu "trung bình 0,755 so với top-3 0,90" đã nói trước đó gộp hai thay đổi làm một.
  - Lợi chỉ có ở RE2-TT (cpu 0,590 → 0,794 → 0,928; socket 0,623 → 0,874 → 0,947; delay 0,842 → 0,926 → 0,961; loss không đổi quanh 0,6; disk, mem đã ≈ 0,99). RE2-OB đã ≥ 0,97 ở mọi cách gộp; RE3-OB và RE3-TT đổi trong ±0,03.
  - Cột PCA (sai số tái tạo tuyến tính, thay thế gần đúng cho mô hình tái tạo) cũng cao hơn số hạng KPI của UAM-AD ở cả bốn dataset (+0,02 đến +0,08). Chưa so cùng điều kiện (PCA chỉ nhìn một bước thời gian, chỉ KPI; số hạng của UAM-AD đi qua cửa sổ và kết hợp đa phương thức), nên chỉ coi là dấu hiệu phù hợp với nghi ngờ "bộ tự mã hoá tái tạo cả bất thường", chưa kiểm.
  - **Giả thuyết cơ chế bị loại:** khoảng (max−min)/std của kênh KPI trong train, trung vị: RE2-OB 14,5; RE3-OB 7,7; RE2-TT 8,4; RE3-TT 9,0. RE2-TT không phình hơn RE2-OB, nên "thang MinMax bị giá trị cực đoan làm phình" không giải thích vì sao MinMax kém ở RE2-TT. Cơ chế chưa xác định; kết luận chỉ dựa vào số đo.
  - Chưa kiểm trên sai số thật của bộ tự mã hoá (sai số từng kênh chưa được lưu). Nếu áp dụng thì phải áp dụng cho mọi dataset (kể cả SN, nơi UAM-AD đã hơn ART), nên cần chấm lại toàn bộ checkpoint để nhất quán. Chưa chạy, chưa đưa vào kế hoạch.
- **Người dùng không duyệt đề xuất chuẩn hoá kênh** ("nghe không đúng lắm"), và đồng ý **chấm lại hết cùng cách** theo thứ tự thời gian (UAM-AD và UAC-AD, cả bốn dataset RCAEval); đã sửa Việc 10 trong `chon_dataset_va_sota.md`.

## 2026-10-30 chiều: chẩn đoán HR@k thấp của UAM-AD (chỉ đọc code, đọc `rca_results.json` seed 0 đã lưu và dữ liệu thô; không chạy mô hình)

Người dùng thấy HR@1 của UAM-AD quá thấp (RE3-OB 0,023; RE3-TT 0,079) so với mô hình khác và yêu cầu tìm nguyên nhân. Mã: `scratchpad/rca_probe.py`, `rca_probe2.py`, `rca_probe3.py`. Mẫu dùng: các bản ghi của `rca_results.json` seed 0 (cửa sổ vượt ngưỡng phát hiện, có service lỗi thật). HR@1 tính lại từ bản ghi seed 0: RE2-OB 0,362; RE3-OB 0,037; RE2-TT 0,389; RE3-TT 0,063 (gần số 5 seed ở T1: 0,352; 0,023; 0,413; 0,079).

- **Loại trừ lỗi ghép nhãn/ánh xạ.** 0 trên 139.109 (RE2-OB), 19.454 (RE3-OB), 85.747 (RE2-TT), 23.081 (RE3-TT) bản ghi có `true_label` khác nhãn trong pkl của đúng `sample_id` đó. Cửa sổ k của test là `data[kW:(k+1)W]` và `flat_idx = k·W + t` khớp. Hàng i của `trace_node_features` ứng với `SERVICES[i]`, cùng `service2idx` trong `meta.pkl`. Một luật không mô hình dùng đúng ánh xạ này (`max |latency_dev|`) đạt HR@1 0,39–0,47 ở RE2, cao hơn hẳn ngẫu nhiên, nên ánh xạ đúng.
- **Số service lỗi thật rất ít**, nên mức ngẫu nhiên k/N (ghi ở mục kết quả) không phải mốc đúng; mốc đúng là đoán hằng service lỗi hay gặp nhất: RE2-OB 5/11 service, đoán hằng HR@1 0,248; RE3-OB 4/11, 0,557 (emailservice 56%); RE2-TT 5/27, 0,259; RE3-TT 2/27, 0,861 (ts-auth-service 86%, ts-route-service 14%). Mô hình có nhãn học được tỉ lệ này (HR@3 = 1,000 ở RE3-OB và RE3-TT của Eadro). Đặt cạnh mốc này: UAM-AD RE3-OB 0,023 thấp hơn cả ngẫu nhiên (0,091), RE3-TT 0,079 thấp hơn nhiều so với 0,861; ART RE3-TT 0,344 và Eadro RE3-TT 0,621 cũng thấp hơn mốc này.
- **Thứ hạng của UAM-AD chỉ lấy từ điểm tái tạo thô theo node của nhánh trace** (`trace_model_v3.py:197-199`; KPI và log không tham gia), nên bị một vài node chiếm top-1 bất kể lỗi ở đâu (top-1 chiếm nhiều nhất): RE2-OB recommendationservice 57%, RE3-OB frontend 57% (cộng recommendationservice 19% và checkoutservice 16% là 92%, không node nào trong ba node này từng là service lỗi nên HR@1 tối đa khoảng 0,08), RE2-TT phân tán (ts-train-service 18%), RE3-TT ts-route-service 49% (service lỗi chỉ 14% mẫu).
- **Hai cơ chế đo được (không phải một):**
  - OB, phần cấu trúc: ma trận kề được chuẩn hoá theo hàng (giá trị phân số, không nhị phân như chú thích trong `trace_model_v3.py`), nên `BCE(adj_hat, adj)` có sàn H(adj) khác 0 và chỉ phụ thuộc vào hàng: sàn cao nhất ở recommendationservice (0,125) và checkoutservice (0,073) ở RE2-OB; frontend (0,166), recommendationservice (0,121) ở RE3-OB, đúng ba node chiếm top-1. Top-1 của mô hình nằm trong top-3 của xếp hạng chỉ bằng sàn ở 64% (RE2-OB) và 84% (RE3-OB) mẫu; xếp hạng theo bậc node trùng top-1 của mô hình ở 49% và 67%. Ở TT sàn nhỏ và không giải thích (trùng 12% và 8%).
  - RE3-TT: xếp hạng theo độ lệch thô của hàng kề so với hàng trung bình bình thường cho ts-route-service ở top-1 trong 86% mẫu và trùng top-1 của mô hình ở 46,5%; chuẩn hoá độ lệch đó theo từng node (z) thì HR@1 từ 0,028 lên 0,481. Ở RE2-OB và RE2-TT xếp hạng của mô hình bám luật `max |latency_dev|` (trùng top-1 85,9% và 71,5%), tức hoạt động đúng thiết kế nhưng không hơn luật đơn giản đó (HR@1 0,362 so với 0,394; 0,389 so với 0,472).
- **Luật không mô hình trên chính sáu đặc trưng trace** (z theo từng node với thống kê train, không dùng nhãn): HR@1 RE2-OB 0,38–0,41; RE2-TT 0,45–0,47; RE3-TT 0,42–0,51; **RE3-OB 0,00–0,08**. Vậy ở RE3-OB (lỗi mức mã) đặc trưng trace gần như không chứa tín hiệu định vị; không cách chấm nào dựa trên chúng sửa được. Ở RE3-TT các luật này (≈0,48) vẫn thấp hơn đoán hằng 0,861, nên không coi là có khả năng định vị.
- **Đơn vị so sánh khác nhau:** UAM-AD chỉ tính HR trên cửa sổ vượt ngưỡng (`basev3.py:530`), ART tính trên mọi mẫu bất thường (`run_art.py`, `rank_hits`), Eadro/OrEdge phân loại service bằng đầu có nhãn. Khác biệt này có xu hướng có lợi cho UAM-AD, nên không giải thích được chỗ thấp.
- **Chưa kiểm:** chuẩn hoá điểm theo từng node bằng mức của chính nó trên val bình thường (cùng kiểu `norm_sum`, không dùng nhãn) rồi xếp hạng lại bằng `model.ckpt` đã lưu (chỉ suy luận; cần mã xuất điểm theo node, hiện `trace_scores.npz` chỉ lưu điểm tổng mỗi mẫu). Chưa chạy, chờ lệnh. Các luật không mô hình chỉ cho biết cận trên có thể đạt từ đặc trưng trace, không bảo đảm điểm thật của mô hình đạt được.
- Giới hạn: seed 0, mẫu là cửa sổ vượt ngưỡng; các luật z và độ lệch nhìn bằng nhãn test (để chấm HR) nên chỉ để chẩn đoán.

## 2026-10-30 chiều (tiếp): đề xuất thiết kế RCA mới và kiểm tra bằng dữ liệu thô (không chạy mô hình)

- **Yêu cầu của người dùng:** RCA hiện tại làm theo ý tưởng TraceDAE; chuẩn hoá điểm từng node (mức 1) chỉ tăng RE3-TT nên không ổn, phải tăng cho tất cả dataset; chấp nhận sửa thiết kế RCA; nếu kiểm chứng được nhanh thì gộp vào Việc 10 như bước cải thiện.
- **Cơ sở thiết kế hiện tại:** `docs/root_cause_localization_vi.md` ghi RCA là cài đặt đúng công thức Eq. 14 của TraceDAE (điểm tái tạo thô theo node, chỉ từ trace). Chưa đọc bài gốc; chỉ dựa vào tài liệu của repo.
- **Phép đo dữ liệu thô** (`scratchpad/rca_design_probe.py`). Bằng chứng theo service = đặc trưng trace của node + kênh KPI của chính service (kênh `<service>_<metric>`; SN `container_<metric>__<service>`; số kênh gắn được: RE2-OB 78/86, RE3-OB 69/103, RE2-TT 214/383, RE3-TT 176/327, SN 48/59). Mỗi số hạng chuẩn hoá theo service bằng trung vị và (p95 − trung vị) trên mẫu bình thường của train. Số hạng trace = max |z| của 6 đặc trưng; số hạng KPI = max |z| qua các kênh của service; "tổng" = hai số hạng cộng lại. HR@1 trên mọi mẫu bất thường biết service lỗi (quy ước của ART), seed không liên quan vì không dùng mô hình:

| Dataset | Đoán hằng | Trace | KPI theo service | Tổng | UAM-AD hiện tại (T1) | ART | Eadro | OrEdge |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| RE2-OB | 0,203 | 0,346 | 0,906 | 0,890 | 0,352 | 0,753 | 0,667 | 0,927 |
| RE3-OB | 0,509 | 0,024 | 0,637 | 0,554 | 0,023 | 0,727 | 0,853 | 0,941 |
| RE2-TT | 0,200 | 0,365 | 0,687 | 0,680 | 0,413 | 0,447 | 0,539 | 0,954 |
| RE3-TT | 0,500 | 0,198 | 0,404 | 0,448 | 0,079 | 0,344 | 0,621 | 0,905 |
| SN | 0,270 | 0,006 | 0,535 | 0,214 | 0,194 | 0,058 | 0,074 | 0,262 |

  - HR@3 của KPI theo service: RE2-OB 0,955; RE3-OB 0,725; RE2-TT 0,784; RE3-TT 0,586; SN 0,767. SN chỉ có 159 mẫu bất thường biết service (9 scenario), nên nhiễu lớn.
  - Cấu hình "tổng" được khai báo trước; "KPI theo service" (bỏ trace) là lựa chọn sau khi đã xem số này, tức chọn bằng nhãn test (một lựa chọn).
  - Trace kéo xuống rõ ở SN (0,535 → 0,214) và RE3-OB (0,637 → 0,554); chỉ RE3-TT tăng (0,404 → 0,448); RE2-OB, RE2-TT gần như hoà.
- **Đính chính mốc "đoán hằng":** mục chẩn đoán HR ở trên tính mốc này trên các cửa sổ UAM-AD gắn cờ (RE3-TT: 0,861, vì `ts-auth-service` chiếm 86% cửa sổ gắn cờ). Trên mọi mẫu bất thường (tập mẫu của ART, Eadro, OrEdge) mốc là 0,203 / 0,509 / 0,200 / 0,500 (RE2-OB / RE3-OB / RE2-TT / RE3-TT) và 0,270 (SN). So với ART, Eadro, OrEdge phải dùng mốc sau.
- **Chưa kiểm:** sai số tái tạo thật của mô hình. Số liệu trên là z-score của dữ liệu thô, chưa phải điểm của UAM-AD. Cần suy luận từ checkpoint: lấy `result["output"][1]` (kpi_out) so với `kpi_features` theo từng kênh, và `result["node_scores"]`, chuẩn hoá theo val. Trước đó có dấu hiệu bộ tự mã hoá KPI kém PCA 0,02–0,08 AUROC phát hiện, nên điểm thật có thể thấp hơn số dữ liệu thô. Chưa chạy, chờ lệnh.

- **Hướng đã nói với người dùng (chờ duyệt bước kiểm bằng checkpoint):** trace phải tham gia xếp hạng, nên cấu hình chính là "tổng" (khai báo trước): điểm service i = trace (Eq. 14 như cũ, chuẩn hoá theo node trên val) + KPI của chính service (sai số tái tạo từng kênh, chuẩn hoá theo kênh trên val, lấy max trong service), chuẩn hoá bằng trung vị và (p95 − trung vị) rồi cắt âm về 0 (cùng kiểu `norm_sum`). Không huấn luyện lại; chỉ đổi cách đọc điểm từ checkpoint, nên F1, AUROC, AUPRC không đổi. Số liệu dữ liệu thô của "tổng" so với UAM-AD hiện tại (HR@1; hai tập mẫu khác nhau nên chỉ so xu hướng): tăng ở RE2-OB, RE3-OB, RE2-TT, RE3-TT; SN chỉ 0,214 so với 0,194 (159 mẫu, trong vùng nhiễu). Cái giá so với "chỉ KPI": kém ở RE3-OB (0,554 so với 0,637) và SN (0,214 so với 0,535).

## 2026-10-30 chiều (tiếp): kiểm nhanh thiết kế RCA mới bằng checkpoint seed 0 (chỉ suy luận, không huấn luyện, không sửa code repo)

Người dùng duyệt bước kiểm. Mã (scratchpad): `rca_model_dump.py` (bọc `run.py --pre_model`, thay `BaseModel.evaluate` để xuất sai số KPI từng kênh `criterion2(kpi_out, kpi_features)`, `node_scores` Eq. 14, `fusion_loss`, nhãn, id mẫu của val và test), `rca_model_run.py` (điều khiển), `rca_model_analyze.py`, `rca_model_diag.py`. Mỗi dataset một scenario, chốt trước: RE2-OB `cpu`, RE3-OB `f1`, RE2-TT `cpu`, RE3-TT `f1`, SN ba scenario `Code_Stop_*`. Test giữ nguyên thứ tự xáo của kết quả đã lưu. Thời gian xuất: SN 10 giây mỗi scenario, RE3-OB 51 giây, RE3-TT 113 giây, RE2-OB 177 giây, RE2-TT 282 giây.

- **Phép tái tạo:** với cùng `sample_id`, top-1 từ dump trùng 100% với `rca_results.json` đã lưu (RE2-OB 10598/10598; RE3-OB 6463/6463; RE2-TT 2969/2969; RE3-TT 5754/5754; SN 113/113). Nên số liệu xuất ra là của đúng mô hình đã lưu.
- **Thiết kế (khai báo trước):** điểm service = trace (Eq. 14, mỗi node chuẩn hoá theo trung vị và p95 − trung vị trên các khối không chồng của val, cắt âm về 0) + KPI (sai số bình phương từng kênh chuẩn hoá cùng cách, lấy max qua kênh của service). Điều kiện qua, khai báo trước: HR@1 của "tổng" cao hơn cách hiện tại (cùng mẫu) ở mọi dataset.
- **Kết quả (HR@1 trên mọi mẫu bất thường biết service lỗi; HR@3 của tổng trong ngoặc):**

| Dataset (scenario) | Số mẫu | Đoán hằng | Hiện tại (Eq. 14 thô) | Chỉ trace chuẩn hoá | Chỉ KPI | **Tổng** | Qua? |
|:--|--:|--:|--:|--:|--:|--:|:-:|
| RE2-OB `cpu` | 10303 | 0,210 | 0,390 | 0,342 | 0,895 | **0,860** (0,951) | ✓ |
| RE3-OB `f1` | 5392 | 0,334 | 0,097 | 0,312 | 0,372 | **0,374** (0,639) | ✓ |
| RE2-TT `cpu` | 10813 | 0,200 | 0,254 | 0,290 | 0,955 | **0,848** (0,985) | ✓ |
| RE3-TT `f1` | 5391 | 0,500 | 0,237 | 0,031 | 0,315 | **0,304** (0,368) | ✓ |
| SN (3 `Code_Stop_*`) | 116 | 0,336 | 0,009 | 0,043 | 0,819 | **0,491** (0,741) | ✓ |

  - Số đã lưu của cách hiện tại trên cửa sổ vượt ngưỡng (seed 0, cùng scenario): 0,226; 0,101; 0,296; 0,065; SN 0,000 / 0,000 / 0,048.
  - Mẫu SN: ba scenario lần lượt tổng 0,718 / 0,667 / 0,079 (chỉ KPI 0,897 / 0,897 / 0,658).
  - Chỉ chuẩn hoá trace (không thêm KPI) không tăng đồng đều: tăng ở RE3-OB, RE2-TT, SN; giảm ở RE2-OB và RE3-TT.
- **Trace kéo "tổng" xuống so với "chỉ KPI" ở SN (0,819 → 0,491) và RE2-TT (0,955 → 0,848)**; ngang ở RE3-OB, RE3-TT; có ích ở thứ hạng sâu hơn ở RE3-OB (HR@5 0,892 so với 0,723). Chẩn đoán (`rca_model_diag.py`): thang của trace không nổ (thang node trung vị 0,03–0,6, không node nào < 1e-4); nhưng ở chính service lỗi, trace_z trung vị chỉ 0,0 (RE2-OB), 0,0 (RE3-OB), 0,9 (RE2-TT), 0,1 (RE3-TT), 0,3 đến 6,3 (SN), nên điểm node Eq. 14 hầu như không nhô lên ở service lỗi. Khi trace quyết định top-1 của tổng (RE2-OB 11% mẫu, RE3-OB 4,5%, RE2-TT 32,5%, RE3-TT 8,6%, SN 18–84%) thì thường chọn sai: HR@1 trong các mẫu đó 0,46 so với 0,91 (RE2-OB), 0,60 so với 0,97 (RE2-TT), 0,00 so với 0,88 và 0,90 (SN `Code_Stop_Media` và `_Text`), 0,03 so với 0,33 (SN `Code_Stop_User`).
- **Giới hạn:** seed 0, một scenario mỗi dataset (SN chỉ ba scenario `Code_Stop_*`, 116 mẫu), test xáo như kết quả đã lưu. Ở RE3-OB (0,374 so với đoán hằng 0,334) và RE3-TT (0,304 so với đoán hằng 0,500) tổng chỉ hơn cách cũ, chưa hơn mốc đoán hằng ở RE3-TT. Số một scenario không so trực tiếp được với số trung bình dataset 5 seed ở T1 (ART, Eadro, OrEdge).
- **Quan sát phụ về ngưỡng cắt cửa sổ của HR hiện tại:** trong đường huấn luyện, `localize_root_causes` nhận `val_threshold` là phân vị p95 của `fusion_loss` thô trên val (cùng thang với `fusion_loss`), tức cửa sổ được chọn theo quy tắc tổng thô, không phải `norm_sum` mà bảng F1 dùng; ở SN ngưỡng còn cộng phạt hoạt động trong khi `fusion_loss` trong `localize_root_causes` thì không.

## 2026-10-30 chiều (tiếp): ước tính thời gian làm xong Việc 10 (từ log; chưa chạy gì thêm)

Người dùng đồng ý gộp bước cải thiện RCA vào Việc 10 và hỏi tổng thời gian. Số đo (từ log và mtime của các lần chạy hôm nay; `scratchpad/stage_times.py`):

| Dataset | Đọc và biến đổi | Dựng dataset (unlabel, test, val) | Dựng mô hình + suy luận val/test + xuất | Cả tiến trình | Tổng thời gian chạy (có khởi động) |
|:--|--:|--:|--:|--:|--:|
| RE2-OB `cpu` | 8 s | 142 s | 15 s | 165 s | 177 s |
| RE3-OB `f1` | 4 s | 33 s | 6 s | 43 s | 51 s |
| RE2-TT `cpu` | 14 s | 235 s | 22 s | 271 s | 282 s |
| RE3-TT `f1` | 5 s | 79 s | 11 s | 95 s | 113 s |
| SN (ba scenario) | 1 s | 0 s | 1–2 s | 2–3 s | 10 s |

- **Khoảng 90% thời gian mỗi tiến trình là dựng dữ liệu, không phụ thuộc checkpoint.** Dữ liệu dựng chỉ phụ thuộc vào file dữ liệu, nên cùng một lần dựng có thể dùng cho cả 5 seed và cả UAM-AD lẫn UAC-AD (chưa kiểm; sẽ kiểm bằng cách tái tạo số đã lưu).
- **Cách A (mỗi checkpoint một tiến trình, như các lần kiểm hôm nay):** UAM-AD: RE2-OB 30 × 177 s = 88,5 phút; RE3-OB 25 × 51 s = 21 phút; RE2-TT 30 × 282 s = 141 phút; RE3-TT 20 × 113 s = 38 phút; tổng 4,8 giờ. UAC-AD cùng khối lượng dựng dữ liệu, suy luận nhẹ hơn (chưa đo): khoảng 4,5–4,8 giờ. Tổng khoảng 9,5 giờ. Số lần chạy RE2-TT trước đó dao động 193–645 giây, một lần 2699 giây khi thiếu RAM, nên có thể dài hơn.
- **Cách B (dựng dữ liệu một lần cho mỗi scenario, chấm 10 checkpoint trong cùng tiến trình):** dựng dữ liệu RE2-OB 6 × 150 s, RE3-OB 5 × 37 s, RE2-TT 6 × 249 s, RE3-TT 4 × 84 s ≈ 49 phút; suy luận 210 checkpoint × (15 / 6 / 22 / 11 s theo dataset) ≈ 49 phút; tổng khoảng 1,6–1,7 giờ. Cần viết trình điều khiển trong tiến trình (ước 2–3 giờ làm cùng kiểm tra bằng tái tạo số đã lưu).
- **SN:** test SN là id băm (`6d8f7b9e6f72`...), 38–67% cửa sổ 5 mẫu lẫn cả hai lớp, không có thông tin thời gian trong file → không chấm lại theo thứ tự; chỉ làm lại HR cho UAM-AD (60 checkpoint × khoảng 10 s ≈ 10 phút). Chưa kiểm `data/sn/sessions.pkl`.
- **Chưa tính:** viết mã chấm điểm đầy đủ (F1@val95, AUROC, AUPRC, F1 oracle, HR@1/3/5) từ dump và kiểm lại bằng số đã lưu; cập nhật các bảng T1–T4 và file md (ước 1 giờ).

## 2026-10-30 chiều (tiếp): người dùng chọn cách B cho Việc 10

- Người dùng chọn **cách B** (dựng dữ liệu một lần cho mỗi scenario, chấm 10 checkpoint trong cùng tiến trình; khoảng 1,7 giờ máy) và dặn chỉ viết mã và chạy khi họ ra lệnh. Đã ghi vào hàng Việc 10 của `chon_dataset_va_sota.md`. Chưa viết mã, chưa chạy.
- Việc phải làm khi được lệnh: (1) trình điều khiển trong tiến trình, sắp test theo (thí nghiệm, run, giây), xuất sai số KPI từng kênh, `node_scores`, `fusion_loss`; (2) sửa `localize_root_causes` khoảng 30 dòng (chuẩn hoá theo val, điểm service = trace + KPI, HR trên mọi mẫu bất thường biết service, xuất thêm chỉ-KPI, chỉ-trace và cách cũ); (3) mã chấm điểm từ dump (F1@val95, AUROC, AUPRC, F1 oracle, HR@1/3/5) và tái tạo số đã lưu ở test xáo cho vài checkpoint trước khi chạy hàng loạt; (4) HR cho SN (UAM-AD, 60 checkpoint); (5) cập nhật T1–T4 và md.

## 2026-10-30 chiều (tiếp): kiểm tra kết quả Eadro và OrEdge có chuẩn không, vì sao cao hơn, và mục tiêu 80–90% (đọc mã, đọc kết quả đã lưu, đo bằng bộ phân loại có nhãn đơn giản; không chạy lại Eadro/OrEdge)

Mã (scratchpad): `labelled_baseline_probe.py`, `labelled_stored_summary.py`, `dup_check.py`.

- **Giao thức (đọc `run_eadro.py`, `run_oredge.py`):** với mỗi `test_<lỗi>.pkl`, huấn luyện trên các run khác run k (có nhãn, cùng loại lỗi, cùng service), chấm run k; mô hình ở epoch cuối; thang lấy từ `train.pkl`; ngưỡng F1@val95 từ `val.pkl` của run k. Đúng quy tắc 5 epoch: mỗi fold chạy 5 epoch (khoảng 34 nghìn mẫu huấn luyện, khoảng 17 nghìn mẫu chấm), mặc định của tác giả là 50. Cả hai chấm cửa sổ 10 bước liên tiếp theo thời gian thật trong một run. "Bán giám sát" của OrEdge chỉ là một nửa nhãn service lỗi bị ẩn; nhãn bất thường/bình thường của mọi mẫu huấn luyện vẫn đủ, nên nhãn của OrEdge không ít hơn Eadro về phát hiện.
- **Không có rò rỉ trong giao thức chia theo run:** 0 mẫu trùng giữa các run trong cùng một file (RE2-OB, RE3-OB, RE3-TT; so bằng vector KPI).
- **Mẫu bình thường lặp giữa các file test:** khoảng 75% mẫu bình thường của mỗi dataset có bản sao ở file test khác (RE2-OB 191.409/255.767; RE3-OB 60.564/82.164; RE3-TT 64.800/86.400); khoảng 32–33% mẫu bình thường của test cũng nằm trong `train.pkl`/`val.pkl` (cùng với `unlabel` là khoảng 2/3, đúng như ghi chú đã có ở mục giới hạn RE2/RE3). Hệ quả: các mô hình không giám sát (UAM-AD, ART, AnoFusion) được chấm trên mẫu bình thường mà 2/3 là mẫu đã thấy khi huấn luyện, trong khi Eadro/OrEdge chấm run bị giữ ra hoàn toàn, nên cách chấm này nghiêng về phía các mô hình không giám sát, không phải về phía Eadro/OrEdge.
- **Phép "loại lỗi lạ" của bộ phân loại đơn giản là KHÔNG hợp lệ và đã bị rút lại** (huấn luyện trên file lỗi khác lại chứa chính các mẫu bình thường lặp của file chấm, nên AUROC 0,94–0,997 bị thổi phồng). Chỉ số liệu "cùng loại lỗi khác run" là sạch.
- **Bộ phân loại có nhãn đơn giản** (cây tăng cường độ dốc, một bước thời gian, z-score KPI + 6 đặc trưng trace mỗi service, không chuỗi, không đồ thị, mặc định; RE2-TT, RE3-TT dùng cấu hình nhẹ hơn: 15.000 mẫu, 50 vòng), chia 3 fold theo run như Eadro/OrEdge, AUROC / AUPRC trung bình các lỗi:

| Dataset | Bộ phân loại đơn giản | OrEdge | Eadro | ART | UAM-AD | z-score tối đa không nhãn (theo mẫu) |
|:--|--:|--:|--:|--:|--:|--:|
| RE2-OB | 0,937 / 0,757 | 0,973 / 0,959 | 0,867 / 0,791 | 0,981 / 0,957 | 0,903 / 0,799 | 0,984 (AUROC) |
| RE3-OB | 0,932 / 0,842 | 0,892 / 0,832 | 0,817 / 0,727 | 0,893 / 0,746 | 0,835 / 0,615 | 0,892 |
| RE2-TT | 0,924 / 0,855 | 0,967 / 0,949 | 0,838 / 0,728 | 0,833 / 0,688 | 0,808 / 0,688 | 0,924 |
| RE3-TT | 0,851 / 0,710 | 0,764 / 0,690 | 0,606 / 0,443 | 0,684 / 0,448 | 0,702 / 0,347 | 0,710 |

  - Bộ phân loại đơn giản có nhãn vượt Eadro ở cả bốn dataset (AUROC +0,07 đến +0,25), nên Eadro (bản chuyển, 5 epoch) **không phải cận trên**. Nó vượt OrEdge ở RE3-OB và RE3-TT, thấp hơn OrEdge ở RE2.
  - Ở RE2 nhãn không thêm gì so với z-score tối đa không nhãn (RE2-OB 0,984; RE2-TT 0,924); ở RE3 (lỗi mức mã) nhãn thêm nhiều (RE3-OB 0,892 → 0,932; RE3-TT 0,710 → 0,851).
- **Eadro theo từng lỗi (5 seed):** rất thất thường: RE2-OB `loss` AUROC 0,648; RE3-OB `f3` 0,633 (OrEdge 0,974 ở `f3`, nhưng 0,630 ở `f4`); RE3-TT `f1`, `f2`, `f4` AUROC 0,529 / 0,558 / 0,575 và F1 theo quyết định riêng chỉ 0,077 / 0,092 / 0,077 (gần như luôn dự đoán bình thường). OrEdge thấp ở RE3-OB `f4` (0,630) và RE3-TT `f3` (0,651).
- **Vì sao OrEdge > Eadro (quan sát, chưa kiểm):** OrEdge phân loại từng nút từ vector sai số tái tạo, có trọng số lớp bất thường 96, vector đếm template log theo từng nút, mã hoá chuỗi riêng; Eadro dùng entropy chéo không trọng số lớp, gộp chú ý toàn đồ thị trước khi phát hiện, 2 đặc trưng log mỗi service. Chưa chạy Eadro với 50 epoch để tách ảnh hưởng của quy tắc 5 epoch.
- **So với mục tiêu 80–90%:** ART (không giám sát tốt nhất) so với OrEdge đạt 80% ở F1 tại 2/5 dataset (101%, 69%, 70%, 51%, 133%) và AUPRC tại 3/5 (100%, 90%, 72,5%, 65%, 108%). UAM-AD so với OrEdge ở F1: 65%, 73,5%, 68%, 81%, 175%; AUPRC: 83%, 74%, 72,5%, 50%, 131%; AUROC: 93%, 93,5%, 83,5%, 92%, 113%. Ước tính sau Việc 10 (chỉ RE2-OB F1 chuyển sang đạt, theo số đo seed 0: 0,495 → 0,713): các ô còn trượt cần thêm +0,05 đến +0,2 tuyệt đối.

## 2026-10-30 chiều (tiếp): chốt số liệu Eadro và OrEdge

- Người dùng không muốn chạy thêm để làm số của hai mô hình này tăng (số hiện đã quá lớn so với UAM-AD) và cần biết có chốt được không để so sánh mục tiêu 80–90%. Không chạy thêm gì.
- Kiểm bổ sung: `dup_check.py` ở RE2-TT cho 0 mẫu trùng giữa các run trong cùng file (normal 0, anomalous 0), nên cả bốn dataset RCAEval đã xác minh giao thức chia theo run sạch (RE2-OB, RE3-OB, RE3-TT, RE2-TT). Mã SN (`sn_labeled.py`): fold của phiên X loại khỏi huấn luyện mọi cửa sổ của phiên X, của `val.pkl` và của `test_X.pkl`; đã tắt lịch sử để tránh dấu hiệu giả của phần đệm (ghi trong mã).
- Kết luận để chốt: số của Eadro và OrEdge đo đúng theo giao thức đã mô tả, không thấy lỗi hay rò rỉ; chưa đối chiếu với số trong bài gốc; hai mô hình là bản tự viết lại. Eadro thấp hơn mức mà dữ liệu có nhãn cho phép (bộ phân loại đơn giản vượt nó ở 4/4 dataset RCAEval) nên không gọi là cận trên.
- Số ô đạt (UAM-AD / mô hình, từ T2 và T3): mức 80%: Eadro 13/15, OrEdge 9/15; mức 90%: Eadro 8/15, OrEdge 6/15 (tổng 22/30 và 14/30). Khoảng cách tuyệt đối còn thiếu để đạt 80% so với OrEdge: F1 RE2-OB +0,13 (Việc 10 ước tính đủ), RE3-OB +0,05, RE2-TT +0,10; AUPRC RE3-OB +0,05, RE2-TT +0,07, RE3-TT +0,21. So với Eadro ở mức 90%: F1 thiếu +0,01 đến +0,02 (RE2-TT, SN, RE3-OB), AUPRC thiếu +0,04 đến +0,055 (RE3-OB, RE3-TT, SN).

## 2026-10-30 chiều (tiếp): vì sao Eadro vượt UAM-AD ở SN (số liệu đã lưu, chấm lại inference seed 0, bộ phân loại có nhãn đơn giản; không chạy lại Eadro)

Mã (scratchpad): `sn_per_scenario.py`, `sn_order_run.py`, `sn_family_probe.py`, `sn_family_debug.py`.

- **Theo từng scenario (5 seed; AUROC / AUPRC / F1@val95):** trung bình UAM-AD 0,956 / 0,716 / 0,728; Eadro 0,980 / 0,857 / 0,820; OrEdge 0,844 / 0,546 / 0,416; ART 0,890 / 0,590 / 0,553. Ba scenario lớn `Code_Stop_*` (263 cửa sổ, 39 lỗi): AUROC Media 0,966 so với 0,967, Text 0,976 so với 0,994, User 0,983 so với 0,965 (UAM-AD / Eadro). Lợi thế của Eadro nằm ở chín scenario nhỏ (32–80 cửa sổ, 4–10 cửa sổ lỗi): AUPRC `Svc_Kill_*` 0,704 / 0,427 / 0,632 (UAM-AD) so với 0,968 / 0,926 / 0,913 (Eadro); `DB_Redis_UserTimeline` 0,608 so với 0,895; `Perf_CPU` 0,711 so với 0,893; `Perf_Disk` 0,460 so với 0,751.
- **Không phải do test bị xáo:** chấm lại SN theo thứ tự `sessions.pkl` (seed 0, 12 scenario) không giúp UAM-AD: AUROC 0,956 → 0,942, AUPRC 0,705 → 0,669, F1@val95 0,734 → 0,691. Chấm lại ở thứ tự xáo tái tạo đúng cả 12 số đã lưu. Cửa sổ 5 mẫu lẫn cả hai lớp: 33–67% (xáo) so với 0–33% (thứ tự). Vì vậy SN không cần chấm lại theo thứ tự ở Việc 10.
- **Cấu trúc tập test SN:** mỗi file test gồm cửa sổ lỗi của đúng phiên X và cửa sổ bình thường lấy từ các phiên khác (ví dụ `Code_Stop_Media`: 39 lỗi từ phiên đó và 224 bình thường từ chín phiên còn lại). Hai file `Code_Stop_*` dùng chung 224/263 cửa sổ. Fold của Eadro/OrEdge loại khỏi huấn luyện mọi cửa sổ của test_X và val, nên tập huấn luyện nhỏ và nghiêng về lỗi (khoảng 182–192 cửa sổ, trong đó 152–162 là lỗi, ở các fold `Code_Stop_*`).
- **Hiệu ứng "anh em cùng họ lỗi"** (bộ phân loại cây tăng cường, mỗi cửa sổ một bước, trên đúng các fold của Eadro/OrEdge; so với cùng fold nhưng bỏ cả họ lỗi của scenario bị giữ ra): trung bình AUROC 0,985 → 0,924, AUPRC 0,881 → 0,742. `Code_Stop_*`: AUROC 0,969 → 0,881 / 0,623 / 0,732; `Svc_Kill_*`: AUROC 1,000 → 0,964 / 0,982 / 0,964, AUPRC 1,000 → 0,667 / 0,800 / 0,667; `DB_Redis_*` và `Perf_*` gần như không đổi (≥ 0,97 ở cả hai chế độ). Trong fold gốc bộ phân loại cho p = 1,0 cho mọi cửa sổ lỗi (chỉ hai giá trị dự đoán), nên ba `Code_Stop_*` có cùng AUROC/AUPRC (0,969 / 0,736) và ba `Svc_Kill_*` cùng 1,000; đây là hiện tượng thật (đã kiểm, không phải lỗi phép đo).
- **Giới hạn:** OrEdge cùng giao thức SN lại kém (AUROC 0,844), nên giao thức không tự bảo đảm mô hình có nhãn thắng; ngân sách 50 epoch ở SN so với 5 epoch ở RCAEval có thể là một phần lý do Eadro mạnh ở SN nhưng yếu ở RCAEval (giả thuyết, chưa kiểm). Eadro chưa được chạy với fold bỏ cả họ lỗi.

## 2026-10-30 chiều (tiếp): vì sao ở SN Eadro hơn UAM-AD nhưng UAM-AD hơn OrEdge (số liệu đã lưu, không chạy gì mới)

Mã (scratchpad): `sn_sensitivity_summary.py`, `sn_steps.py`. Dữ liệu: `item9/eadro|oredge/sn/seed0.json`, `item9/sn_sensitivity/*`, `item9/*/rcaeval_*`.

- **Số bước tối ưu hoá mỗi fold (seed 0):** SN: Eadro 50–100 bước (182–399 cửa sổ huấn luyện), OrEdge 50–150 bước (152–369 cửa sổ). RCAEval: RE2-OB `cpu` Eadro khoảng 660 bước, OrEdge khoảng 1.700 bước; RE3-TT `f1` Eadro 350 bước, OrEdge 890 bước.
- **Quyết định riêng của mô hình (trung bình các scenario SN, seed 0):** Eadro recall 0,947, precision 0,705; **OrEdge recall 0,008, precision 0,028** (F1 theo quyết định riêng 0,013), tức bộ phân loại của OrEdge gần như không bao giờ dự đoán lỗi ở SN; xếp hạng điểm của nó vẫn cho AUROC 0,813 (seed 0; 5 seed là 0,844). Ở RE2-OB `cpu` OrEdge recall 0,939, precision 0,999.
- **Lịch sử thật không giải thích:** SN có lịch sử 10 cửa sổ thật (seed 0): OrEdge AUROC 0,813 → 0,804, AUPRC 0,480 → 0,536, F1@val95 0,345 → 0,488; Eadro AUROC 0,979 → 0,983, AUPRC 0,851 → 0,890, F1@val95 0,811 → 0,812. Giả thuyết "tắt lịch sử làm hỏng bộ mã hoá chuỗi của OrEdge" không được ủng hộ.
- **Khác biệt giao thức giữa hai mô hình ở SN:** OrEdge loại cửa sổ lỗi của ba phiên `Perf_*` khỏi huấn luyện (không có service lỗi; `run_oredge_sn.py`), Eadro vẫn dùng chúng cho bộ phát hiện. OrEdge `Perf_CPU` AUROC 0,711 so với Eadro 0,980.
- **Giải thích đề xuất (dựa trên mã và số đo, chưa kiểm bằng chạy thêm):** bộ phân loại của OrEdge nhận vector sai số tái tạo và chỉ được đưa vào dần (trọng số 1 − 1/(epoch + 1), bằng 0 ở epoch đầu), nên cần nhiều bước; ở SN chỉ có 50–150 bước nên nó chưa học. Bộ phát hiện của Eadro học trực tiếp từ nhãn bằng entropy chéo nên đủ với 50–100 bước vì lỗi SN dễ tách. Giải thích này chưa phải chứng minh: Eadro ở RE3-TT `f1` cũng không học được với 350 bước (recall 0,042), nên số bước không là nguyên nhân duy nhất.

## 2026-10-30 chiều (tiếp): sửa cách giải thích OrEdge ở SN (số liệu đã lưu, không chạy gì mới)

Người dùng thấy cách giải thích "OrEdge thiếu bước huấn luyện" như đổ lỗi cho thiết lập và yêu cầu cách hiểu hợp lý hơn. Mã: `oredge_sn_profile.py`, kèm đoạn so sánh loss huấn luyện cuối.

- **OrEdge không hội tụ ở SN:** loss huấn luyện cuối trung bình 3,76 (khoảng 1,99–25,0) so với 0,47 (RE2-OB), 0,45 (RE3-OB), 0,10 (RE2-TT), 0,36 (RE3-TT); Eadro ở SN 0,25. Ngưỡng val-p95 của OrEdge ở SN (0,018) cùng cỡ với RE2-OB (0,022) nên ngưỡng không phải vấn đề. Quyết định riêng ở SN: recall 0,022, precision 0,081; ở RCAEval recall 0,47–0,86, precision 0,97–0,99.
- **Điểm mạnh thật của OrEdge ở SN là định vị:** HR@1 / HR@3 / HR@5 = 0,262 / 0,506 / 0,724, cao nhất trong bốn mô hình (UAM-AD 0,194 / 0,275 / 0,528; Eadro 0,074 / 0,266 / 0,498; ART 0,058 / 0,161 / 0,345); chênh với UAM-AD ở HR@1 nằm trong độ lệch chuẩn của OrEdge (0,077). Phát hiện của OrEdge là phụ phẩm (điểm = max xác suất nút là culprit), không có đầu phát hiện riêng; recall ở ngưỡng val-p95 theo scenario dao động 0,00–1,00.
- **Cách đọc đã đề xuất:** ở SN OrEdge cần nhiều dữ liệu có nhãn hơn SN cung cấp (152–369 cửa sổ huấn luyện mỗi fold); kết quả này không cho phép kết luận UAM-AD hơn OrEdge nói chung. Ba ô "UAM-AD hơn OrEdge" ở bảng đếm ô đều là SN; nếu coi hàng OrEdge-SN là không so sánh được thì UAM-AD hơn OrEdge ở 0/12 ô RCAEval, và số ô đạt 80% so với OrEdge còn 6/12 (từ 9/15), 90% còn 3/12 (từ 6/15). Chưa kiểm xem thêm dữ liệu hay bước huấn luyện có làm OrEdge hội tụ không (không chạy thêm theo ý người dùng).

## 2026-10-30 chiều (tiếp): OrEdge ở SN không ổn định giữa các seed; không cần chạy lại (chẩn đoán trên hai fold + 5 seed đã lưu)

Mã (scratchpad): `sn_seed_spread.py`; chẩn đoán chạy `run_oredge_sn.py` (không sửa mã repo) ghi vào `scratchpad/oredge_diag/`, không vào bảng nào.

- **Số đã lưu tái lập đúng:** chạy lại seed 0, 50 epoch, theo thứ tự fold mặc định (Media rồi Text): `Code_Stop_TextService` AUROC 0,515, F1@val 0,033, loss 6,976 và `Code_Stop_MediaService` AUROC 0,962, loss 9,206, trùng số đã lưu.
- **Cùng fold, cùng seed 0, thứ tự fold khác:** `Code_Stop_TextService` chạy đầu tiên cho AUROC 0,966, F1@val 0,761 (so với 0,515 và 0,033). 250 epoch (Text chạy đầu): loss giảm (9,42 → 3,16; `Svc_Kill_UserTimeline` 2,34 → 1,35) nhưng kết quả không đơn điệu: Text AUROC 0,966 → 0,701, `Svc_Kill_UserTimeline` AUROC 0,911 → 1,000. Vậy "chưa hội tụ" không phải nguyên nhân chính; kết quả phụ thuộc mạnh vào trạng thái ngẫu nhiên khi huấn luyện từ 152–369 cửa sổ.
- **Dao động giữa 5 seed đã lưu, AUROC mỗi scenario (trung bình độ lệch chuẩn / khoảng min–max trên 12 scenario):** UAM-AD 0,003 / 0,009; Eadro 0,009 / 0,023; **OrEdge 0,085 / 0,226**. Ví dụ OrEdge `Code_Stop_MediaService` 0,481–0,962, `Svc_Kill_Media` 0,527–0,964, `Code_Stop_TextService` 0,515–0,968.
- **Kết luận UAM-AD > OrEdge ở SN không chỉ do seed xấu:** seed tốt nhất của OrEdge chỉ vượt seed tốt nhất của UAM-AD (theo AUROC) ở 2/12 scenario (`Perf_Disk_IO_Stress` 0,933 so với 0,916; `Svc_Kill_SocialGraph` 0,946 so với 0,914).
- **Quyết định đề xuất:** không chạy lại OrEdge ở SN (số hiện tại là trung bình hợp lệ, tái lập được; chạy lại hay thêm seed chỉ làm trung bình đổi cỡ ± độ lệch chuẩn của trung bình, không đổi kết luận). Cách giải thích thay cho "chưa khớp được tập nhỏ": mô hình có nhãn huấn luyện từ vài trăm cửa sổ rất không ổn định (OrEdge), trong khi UAM-AD ổn định (không dùng nhãn lỗi).

## 2026-10-30 19:27: bắt đầu chạy Việc 10 (chấm lại UAM-AD và UAC-AD theo thứ tự thời gian + xếp hạng RCA mới)

Người dùng ra lệnh bắt đầu chạy Việc 10 và duyệt đoạn giải thích SN (đã chèn vào SUMMARY, sau các gạch đầu dòng "So với OrEdge").

- **Mã:** `codes/common/rescore_item10.py` (một tiến trình cho mỗi scenario: `run.py --pre_model` dựng dữ liệu một lần, thay `BaseModel.evaluate` bằng vòng lặp qua 10 checkpoint = UAM-AD và UAC-AD, seed 0–4; F1@val95/AUROC/AUPRC bằng `BaseModel.evaluate_norm_sum` của repo trên các thành phần điểm của một lượt chạy xuôi; F1 oracle xuất hai định nghĩa) và hàm mới trong `codes/models/rca.py` (`channel_services`, `standardise_on_val`, `service_scores`, `hit_rates`); `localize_root_causes` cũ giữ nguyên, nên kết quả cũ vẫn tái lập được. Kết quả ghi vào `D:\ClaudeWork\phase3\item10\<dataset>\<model>\<scenario>\seed<k>.json`, tiến độ ở `item10\status.txt`, log ở `item10\logs`.
- **Kiểm tái tạo:** chạy ở thứ tự xáo trên RE3-OB `f1` (10 checkpoint, 88 giây, ghi vào `item10_check`): lệch tối đa so với `info_score.txt` đã lưu là 0,000040 (F1@val95), 0,000044 (AUROC), 0,000042 (AUPRC), 0,000047 (F1 oracle). F1 oracle point-adjust 0,9000 so với oracle không point-adjust 0,7442 (UAM-AD seed 0, RE3-OB `f1`): hai định nghĩa khác nhau rất nhiều.
- **Thứ tự chạy:** RE3-OB (5 scenario), RE3-TT (4), RE2-OB (6), RE2-TT (6), rồi SN (12 scenario, chỉ UAM-AD, giữ thứ tự đã lưu, chỉ để làm lại HR). Thời gian dự kiến từ các lần đo hôm nay: khoảng 2 giờ (RE3-OB 8 phút, RE3-TT 12, RE2-OB 36, RE2-TT 54, SN 6); sẽ cập nhật theo số đo thật.
- **Đối chiếu có sẵn:** số chấm theo thứ tự thời gian của UAM-AD seed 0 từ các lần kiểm trước (`scratchpad/order_all`, `order_probe`) phải trùng với kết quả mới của cùng scenario.

## 2026-10-30 19:40: tiến độ Việc 10 và số đầu tiên (RE3-OB đầy đủ)

- **Tiến độ:** RE3-OB xong (5 scenario × 10 checkpoint, 466 giây: 90 / 83 / 95 / 107 / 91 giây); RE3-TT `f1` 165 giây; đang chạy RE3-TT `f2`. Ước tính hoàn thành: RE3-TT khoảng 19:49, RE2-OB khoảng 20:22 (6 × khoảng 330 giây), RE2-TT khoảng 21:10 (6 × khoảng 480 giây), SN khoảng 21:17; tổng khoảng 1 giờ 50 phút kể từ 19:27 (có thể dài hơn nếu thiếu RAM; các lần chạy RE2-TT trước đó từng dao động 193–2699 giây).
- **Đối chiếu:** UAM-AD seed 0 sắp thứ tự của RE3-OB `f1`–`f3` trùng tuyệt đối (lệch 0) với các lần kiểm trước.
- **RE3-OB (5 seed, trung bình ± độ lệch chuẩn), UAM-AD sắp thứ tự so với đã lưu (xáo):** F1@val95 0,549±0,020 so với 0,560±0,019; AUROC 0,832±0,005 so với 0,835±0,005; AUPRC 0,605±0,013 so với 0,615±0,012. Sắp thứ tự không giúp ở RE3-OB (đúng như phép kiểm seed 0). UAC-AD: F1 0,643 so với 0,647, AUROC 0,857 so với 0,858, AUPRC 0,692 so với 0,696. F1 oracle point-adjust của UAM-AD tăng 0,755 → 0,989 vì phụ thuộc thứ tự dòng, không so được với kiểu xáo; oracle không point-adjust 0,642.
- **Xếp hạng service RE3-OB (UAM-AD, HR@1 / HR@3 / HR@5, mọi mẫu bất thường biết service):** cách cũ 0,022 / 0,219 / 0,529; chỉ trace chuẩn hoá 0,114 / 0,376 / 0,565; chỉ KPI 0,837 / 0,893 / 0,925; **trace + KPI 0,824 / 0,902 / 0,959**. (T1 hiện tại: UAM-AD 0,023 / 0,224 / 0,531; ART 0,727 / 0,788 / 0,827; Eadro 0,853 / 1,000 / 1,000; OrEdge 0,941 / 0,994 / 0,999.)

## 2026-10-30 20:57: Việc 10 xong, hai lỗi của chính mã chấm lại đã sửa

- **Kết thúc:** 4 dataset RCAEval chạy xong lúc 20:47 (RE3-OB 19:35, RE3-TT 19:46, RE2-OB 20:13, RE2-TT 20:47; sớm hơn dự kiến khoảng 20 phút). Phần SN (chỉ làm lại HR của UAM-AD) lỗi ngay lúc đầu, sửa xong và chạy lại lúc 20:54–20:56 (7 giây mỗi scenario).
- **Hai lỗi của `rescore_item10.py`, cả hai chỉ làm SN không chạy, không ảnh hưởng số RCAEval:**
  - `--protocol` của tiến trình con không nhận giá trị `as-stored` mà tiến trình chính truyền cho SN. Đã thêm giá trị này vào danh sách.
  - Ba scenario `Perf_*` của SN là lỗi hạ tầng, không gắn service nào, nên danh sách mẫu có nhãn service rỗng và chỉ số mảng bị lỗi kiểu dữ liệu. Đã bỏ qua phần RCA khi không có mẫu nào; F1/AUROC/AUPRC của ba scenario này vẫn ghi bình thường. HR ở SN vì vậy tính trên 9/12 scenario.
- **Đối chiếu:** UAM-AD seed 0 sắp thứ tự trùng tuyệt đối (lệch 0) với các lần kiểm trước ở 21 scenario. SN AUROC 0,956 trùng số đã lưu.
- **Kết quả sắp thứ tự, UAM-AD (F1@val95 / AUROC / AUPRC, 5 seed, mới so với đã lưu xáo):**
  - RE2-OB: 0,681 / 0,917 / 0,823 so với 0,564 / 0,903 / 0,799.
  - RE3-OB: 0,549 / 0,832 / 0,605 so với 0,560 / 0,835 / 0,615.
  - RE2-TT: 0,597 / 0,809 / 0,680 so với 0,567 / 0,808 / 0,688.
  - RE3-TT: 0,512 / 0,701 / 0,344 so với 0,513 / 0,702 / 0,347.
  - SN giữ nguyên thứ tự đã lưu: 0,728 / 0,956 / 0,716.
- **UAC-AD sắp thứ tự:** RE2-OB 0,611 / 0,880 / 0,750 (đã lưu 0,523 / 0,868 / 0,707); RE3-OB 0,643 / 0,857 / 0,692; RE2-TT 0,345 / 0,670 / 0,431 (đã lưu 0,256 / 0,673 / 0,448); RE3-TT 0,540 / 0,745 / 0,432.
- **HR@1 / HR@3 / HR@5 của UAM-AD, trace + KPI (cách đã chọn trước):** RE2-OB 0,759 / 0,949 / 0,976; RE3-OB 0,824 / 0,902 / 0,959; RE2-TT 0,721 / 0,871 / 0,908; RE3-TT 0,251 / 0,332 / 0,417; SN 0,185 / 0,347 / 0,397. Cách cũ: 0,355 / 0,613 / 0,759; 0,022 / 0,219 / 0,529; 0,315 / 0,510 / 0,599; 0,227 / 0,286 / 0,393; 0,189 / 0,286 / 0,520.
- **SN không tăng ở HR@1:** trung bình theo scenario bị 6 scenario rất nhỏ (3–10 mẫu bất thường, HR gần như 0 hoặc 1) kéo xuống. Ba scenario `Code_Stop_*` (38–39 mẫu mỗi scenario) lên rõ: trace + KPI 0,708 / 0,492 / 0,305 so với cách cũ 0,046 / 0,067 / 0,037; chỉ KPI 0,897 / 0,913 / 0,658. Cần quyết định cách gộp (theo scenario hay gộp theo mẫu) trước khi điền T1.

## 2026-10-30 21:40: đưa kết quả Việc 10 vào file md chính (RE2-OB, RE2-TT và HR của UAM-AD)

- **Yêu cầu của người dùng:** cập nhật RE2-OB, RE2-TT cho file md, kèm HR@k mới của UAM-AD, và cả UAC-AD (vì cũng chấm lại).
- **Cách làm:** thêm `summary_tools/item10_override.py`, được `load_step8.load()` gọi ở cuối, nên mọi bộ sinh bảng (`item9_tables.py`, `build_summary.py`) đọc cùng số. Sau đó:
  - `item9_tables.py` sinh lại T1, T2 và khối Phụ lục B (ghi chú trong bộ sinh đã sửa cho khớp).
  - Các hàng S2, S3, S4 của Việc 8 viết lại bằng script (không dùng `assemble_into_md.py` vì nó ghi đè phần chữ đã sửa tay). 173 hàng không đổi trùng khít với bộ sinh, 288 hàng đổi đều nằm đúng ô được phép (cột chỉ số của RE2, cột HR của UAM-AD); không có thay đổi ngoài dự kiến.
  - Chữ viết tay của Việc 8 (cách đọc, ghi chú S2, S3, S4, S5, S6) và Việc 9 (nhận xét T1, T2, T3) sửa số theo bảng mới.
  - Phần HR@k của T1 viết lại theo cách xếp hạng mới, kèm bảng ablation bốn cách xếp hạng.
  - Hàng Việc 10 của mục 5 đổi từ "đang chạy" sang "xong 20:56".
- **Quyết định trình bày, người dùng chưa duyệt riêng (cần báo lại):**
  - Chỉ RE2-OB, RE2-TT đổi F1@val95, AUROC, AUPRC; RE3-OB, RE3-TT giữ số đã lưu (chênh tối đa 0,011); SN không đổi.
  - HR@k của UAM-AD đổi ở cả 5 dataset, vì cách xếp hạng là một thiết kế chung.
  - F1 oracle ở RE2 là bản không point-adjust (bản point-adjust phụ thuộc thứ tự dòng, nhảy lên khoảng 0,95); cột này vì vậy không so được giữa RE2 và dataset khác.
  - Bảng phụ `raw_sum` và hai hàng RE2 của S6 (cột `raw_sum`) vẫn theo test xáo, đã ghi chú trong bảng.
  - HR của SN gộp theo scenario như các bảng khác (trung bình qua 9 scenario có service lỗi), nên bị sáu scenario nhỏ kéo xuống.
- **Thay đổi kết luận so với trước:**
  - RE2-OB: UAM-AD hơn UAC-AD ở trung bình (F1 +0,070, AUROC +0,037, AUPRC +0,073) nhưng không đồng đều: AUROC và AUPRC hơn ở 4/5 seed, F1 hơn ở 3/5 seed, và seed 4 thấp hơn ở cả ba chỉ số.
  - RE2-TT: UAM-AD hơn UAC-AD ở cả 5 seed và cả ba chỉ số (F1 +0,251, AUROC +0,139, AUPRC +0,249).
  - So với đối thủ: UAM-AD hơn AnoFusion ở 12/12 ô (trước 10/12); T2 đạt 23/30 ô (trước 22/30); T3 đạt 16/30 ô (trước 14/30). Ô AUPRC trung bình so với OrEdge là 79,9986%, hiện 80,0% trong bảng nhưng vẫn trượt vạch.
  - HR@1 của UAM-AD: RE2-OB 0,759 (trước 0,352), RE3-OB 0,824 (0,023), RE2-TT 0,721 (0,413), RE3-TT 0,251 (0,079), SN 0,185 (0,194). RE3-TT và SN vẫn thấp hơn đoán cố định, SN HR@5 (0,397) thấp hơn ngẫu nhiên (0,42).
- **Chưa đụng:** khối chẩn đoán seed 0 ở Phụ lục C và các đoạn `[CN 10-24]` (số trung gian theo test xáo, đã ghi nhãn là chưa phải kết quả cuối). Chưa commit gì.
- **21:55: rà lại nhận xét Việc 9 theo từng mô hình:** đối chiếu từng số với T1/T2 mới. Một câu sai so với số mới ("Yếu nhất ở RE2-OB" khi so với ART; AUPRC của RE3-TT và RE3-OB còn kém hơn) đã sửa thành "F1 kém nhất ở RE2-OB". Thêm dòng AUROC so với ART và bổ sung đoạn điểm mạnh (hơn AnoFusion 12/12 ô, AUROC hơn Eadro ở RE2-OB và RE3-TT, HR@1 hơn ART ở 3/5 dataset).


## 2026-10-30 22:05: xoá T3 (mức 90% và nhiễu giữa seed) khỏi file md chính

- **Người dùng quyết định:** bỏ mức 90%, giữ mức 80% (T2) làm mức chính. T3 bị xoá cả khối, T4 đánh số lại thành T3. Nội dung cuối cùng của T3 (số mới sau việc 10) giữ nguyên dưới đây để tra cứu; không còn trong file md chính.
- Ba phần của T3: kết quả mức 90% (16/30 ô), nhiễu giữa seed (sd/trung bình, số ô cùng dấu), và kiểm nhận định "SN ta cao nhất, OB và TT không thua quá nhiều".

**T3. Mức 90% và nhiễu giữa seed (số liệu bổ sung, 5 seed)**

**Mức 90%** (chốt trước mức 80% của T2): "UAM-AD giữ ≥ 90% giá trị của mô hình có nhãn", áp chung cho F1@val95, AUROC, AUPRC. **90% là con số được đề xuất sau khi đã thấy số seed 0, không lấy từ tài liệu nào đã kiểm chứng**; người dùng chốt giữ mức này cho cả ba chỉ số (10-25). Sàn của AUROC là 0,5 và của AUPRC là tỉ lệ lỗi (khoảng 13–20%), nên đạt AUROC là phát biểu yếu hơn đạt AUPRC.

**Kết quả mức 90% (trung bình 5 seed của cả hai phía): UAM-AD đạt 16/30 ô** (seed 0: 17/30). AUROC 9/10 (trừ OrEdge ở RE2-TT, 83,6%); F1@val95 4/10 (Eadro: RE2-OB, RE2-TT, RE3-TT; OrEdge: SN); AUPRC 3/10 (Eadro: RE2-OB, RE2-TT; OrEdge: SN). Đạt cả hai mô hình cùng lúc: AUROC 4/5 dataset, F1@val95 0/5, AUPRC 0/5. **Câu đỡ được trong thesis:** "UAM-AD giữ ≥ 90% AUROC của cả hai mô hình có nhãn ở 4/5 dataset; ở F1@val95 và AUPRC còn thấp hơn rõ ở RCAEval (ở các ô không đạt, kém Eadro 11,2–21,7% và kém OrEdge 14,1–49,8%); ở SN vượt OrEdge ở cả ba chỉ số nhưng kém Eadro".

**Nhiễu giữa seed (độ lệch chuẩn / trung bình, khoảng qua 5 dataset).** UAM-AD: F1@val95 0,4–11,1% (RE2-OB lớn nhất), AUROC 0,2–1,8%, AUPRC 0,9–3,0%. ART: 0,2–11,2%, 0,2–2,2%, 0,4–6,5%. Eadro: 1,9–7,6%, 0,3–3,3%, 1,3–3,8%. OrEdge: 0,2–25,9%, 0,3–4,7%, 0,3–11,2% (lớn nhất ở SN: F1 0,416±0,108, AUROC 0,844±0,039, nên dòng OrEdge SN không ổn định, nhưng UAM-AD vẫn hơn nó ở cả 5 seed). **Chênh UAM-AD trừ mô hình, tính theo từng seed, cùng dấu ở cả 5 seed** trong 28/30 ô so với Eadro và OrEdge (hai ô lẫn dấu: Eadro RE2-OB F1, Eadro RE3-OB AUROC), nên các khoảng cách với hai mô hình có nhãn **không** do nhiễu seed. So với ART có 12/15 ô cùng dấu; 3 ô coi là hoà: F1 ở RE3-OB, AUROC ở RE3-TT, AUPRC ở RE2-TT.

**Nhận định "SN ta cao nhất, OB và TT không thua quá nhiều" (so ART và AnoFusion, 5 seed).** UAM-AD cao nhất nhóm ở SN. Câu "không thua quá nhiều" đỡ được cho AUROC (không dataset nào kém ART quá 7%) và cho F1@val95 trừ RE2-OB (kém ART 22,6%, chỗ yếu rõ nhất), **không đỡ được cho AUPRC** ở RE2-OB, RE3-OB, RE3-TT (kém ART 14,0–22,7%). UAM-AD hơn AnoFusion ở trung bình cả ba chỉ số nhưng AnoFusion chỉ có 1 seed và RE3-TT chỉ có `f1`.

## 2026-10-30 23:05: Bảng D và Bảng E (việc 11 và 10 của Giai đoạn 4) đã điền vào file md chính

- **Người dùng yêu cầu:** xoá T3, giữ T2 (mức 80%) làm mức chính; điền Việc 10 và 11 của Giai đoạn 4 vì đã làm được. Việc 10 của Giai đoạn 4 (đo chi phí) khác việc 10 của Giai đoạn 3 (chấm lại theo thứ tự thời gian); trùng số, đã ghi chú ở hàng đó.
- **T3:** xoá cả khối khỏi md (nội dung cuối ở mục 22:05 phía trên), T4 đổi thành T3, bỏ bảng mức 90% khỏi khối Phụ lục B (sửa `item9_tables.py`). T2 ghi rõ mức 80% được chọn sau khi đã thấy kết quả và đã thử mức 90% (16/30 ô) trước.
- **Bảng D:** điền bằng `item9_tables.py` (khối `BANGD`), 5 dataset, hàng ART, Eadro, OrEdge, UAM-AD, kèm hàng ngẫu nhiên và đoán cố định. Bỏ ý cũ "ground-truth service của TT không có" (sai: cả bốn dataset RCAEval có 100% mẫu bất thường có service lỗi).
- **Bảng E: đo thật, `codes/common/measure_cost.py` (mới, chưa commit), kết quả ở `D:\ClaudeWork\phase3\cost\`.** CPU 6 luồng, batch 128, cả file `test_cpu`, một tiến trình mới cho mỗi mô hình, RAM đọc working set qua `ctypes` (máy không có `psutil`; không cài thêm gói). Các vướng mắc đã gặp:
  - Mã gốc gọi `.cuda()` cứng ở `models/utils.py:170`, UAM-AD, UAC-AD không chạy được trên CPU; sửa bằng no-op `torch.nn.Module.cuda` trong tiến trình đo (repo không đổi). `run.py` đặt lại `CUDA_VISIBLE_DEVICES` từ `--gpu_device`, nên phải truyền `--gpu_device -1`.
  - Đối thủ không lưu checkpoint ở việc 9, nên đo với mạng dựng từ code runner, trọng số khởi tạo (ghi rõ trong md).
  - Lần đầu AnoFusion trên CPU chạy hơn 12 phút và chiếm 4,4 GB RAM (máy còn 1,1 GB trống), nên mình tự dừng tiến trình đo đó; chạy lại sau trên toàn bộ file test: 877 giây, đỉnh RSS 1 695 MB (không lặp lại mức 4,4 GB). Chưa rõ nguyên nhân chênh RAM giữa hai lần.
  - AnoFusion trên RE2-TT không đo (1 357 kênh, nguy cơ hết RAM); chỉ có thời gian huấn luyện từ việc 9.
  - Chạy lặp 2 lần cho 5 mô hình ở RE2-OB (ngoại trừ AnoFusion): thời gian suy luận lệch tới 25% giữa hai lần, RAM thêm gần như không đổi trừ UAM-AD (100 và 157 MB). RE2-TT một lần đo.
  - Lần đầu của UAM-AD, UAC-AD ở RE2-OB (`cost\run1_ours`) lưu trước khi thêm trường `samples_done`, `ms_per_1000_samples`; bảng tính lại hai trường này từ `infer_s` và `test_samples` (đều chạy cả file).
- **Kết quả chính:** UAM-AD nhiều tham số nhất (627 nghìn ở RE2-OB, 687 nghìn ở RE2-TT), suy luận CPU 1,2–1,5 lần UAC-AD, RAM thêm 128 và 322 MB, huấn luyện nhiều nhất cả dataset (111,5 và 159,9 phút, vì huấn luyện lại 6 lần). Chưa đo SN.
- **23:20: dời kết quả Bảng D và Bảng E sang Giai đoạn 4.** Người dùng không thấy hai bảng ở Giai đoạn 4 (chúng nằm ở mục 4, phần thiết kế). Kết quả (bảng sinh tự động và chú thích) chuyển thành "Việc 10 (Giai đoạn 4) – Bảng E" và "Việc 11 – Bảng D" ngay dưới bảng Giai đoạn 4, theo cách kết quả Việc 8, 9 nằm ở Giai đoạn 3; mục 4 giữ định nghĩa, tiêu chí và cách đo, kèm dòng trỏ sang. Các dấu `BANGD`, `BANGE` đi theo khối nên `item9_tables.py` và `cost_table.py` vẫn ghi đúng chỗ. Bản md trước khi dời: `scratchpad/backup/chon_dataset_va_sota.before_move_DE.md`.
- **23:50: Bảng D viết lại theo bố cục T1 của Việc 9.** Mỗi dataset có hàng UAM-AD, ART, Eadro, OrEdge, ngẫu nhiên (k/N), đoán cố định; cột HR@1, Δ HR@1, HR@3, Δ HR@3, HR@5, Δ HR@5 (Δ = UAM-AD trừ mô hình, trong ngoặc là % giá trị của mô hình); thêm dòng Trung bình 5 dataset và bảng đếm hơn/hoà/kém theo từng k. Trung bình 5 dataset của UAM-AD: HR@1 0,548, HR@3 0,680, HR@5 0,731. Một chi tiết được sửa: so với Eadro ở HR@1, RE3-OB là "hoà" (−3,3%, dấu lẫn giữa các seed), không phải "kém" như câu cũ.
- **Hỏi của người dùng về Bảng E (chỉ có một file test mỗi dataset):** không phải quên lưu. Các lần chạy việc 8, 9 chạy trên GPU và không ghi thời gian suy luận CPU, số tham số hay RAM; checkpoint của ART, AnoFusion, Eadro, OrEdge không được lưu. Bảng E hiện đo trên đúng một file test (`test_cpu`) mỗi dataset vì mình chọn đo nhanh; muốn theo dataset phải đo lại mọi file test.
- **01:00 (10-31): Bảng E đo lại theo dataset, mọi file test.** Người dùng chọn "đo hết, AnoFusion ước tính". `measure_cost.py` đổi sang lưu theo file test (`D:\ClaudeWork\phase3\cost_all\<dataset>\<file test>\<mô hình>.json`); chạy 23:29–00:57, 60/60 lượt (5 mô hình × 12 file test của RE2-OB và RE2-TT), không lỗi. Số đo thử nghiệm ban đầu (1 file mỗi dataset) giữ ở `D:\ClaudeWork\phase3\cost` (và `cost_rep`, `cost\run1_ours`); bảng md chỉ dùng `cost_all`, riêng AnoFusion RE2-OB lấy từ file đo thử (877 giây, 1 067 MB) và nhân lên 6 file thành 5 638 giây (ước tính, có dấu *). Bảng đổi thành: thời gian suy luận cộng qua các file, RAM lấy giá trị lớn nhất qua các file. Kết quả: UAM-AD 35,4 giây (RE2-OB) và 52,3 giây (RE2-TT) so với UAC-AD 29,8 và 35,2; RAM thêm 158 và 308 MB; chưa đo SN.
- **Hỏi của người dùng sau đó:** Bảng D có cần hai hàng "Ngẫu nhiên (k/N)" và "Đoán cố định một service" không; Bảng E sao nhiều cột; sao chỉ có RE2-OB và RE2-TT.
- **10-31: rút gọn Bảng D, Bảng E (không chạy thêm gì).** (1) Bảng D bỏ hai hàng tham chiếu (không phải mô hình, và không có trong định nghĩa Bảng D ở mục 4); mốc đoán cố định và ngẫu nhiên chỉ còn nhắc một câu trong "Cách đọc" (RE3-TT và SN yếu). `item9_tables.py` bỏ `N_SERVICES`, `CONSTANT_GUESS`. (2) Bảng E giảm từ 9 cột còn 6 cột, đúng bốn tiêu chí của định nghĩa mục 4: huấn luyện (phút / lần), tham số, suy luận CPU (ms / 1 000 mẫu), RAM thêm, RAM đỉnh. Bỏ khỏi bảng: số lần huấn luyện, huấn luyện cả dataset (nay là một câu trong ghi chú), suy luận cả dataset (giây). AnoFusion RE2-OB không còn "ước tính 5 638 giây": ba ô ms / 1 000 mẫu, RAM thêm, RAM đỉnh là của một file `test_cpu`, đánh dấu *. Số 1,2 lần ở thời gian huấn luyện cả dataset RE2-TT sửa thành 1,1 lần (159,9 / 139,5). (3) Phạm vi dataset: định nghĩa ban đầu ghi "2–3 dataset đại diện (ví dụ SN, RE2-OB)"; mình chọn RE2-OB và RE2-TT, không có lý do riêng cho việc bỏ RE3-OB, RE3-TT (cùng runner RCAEval, checkpoint seed 0 của UAM/UAC-AD có sẵn); SN cần bộ đo riêng. Chưa đo thêm, chờ người dùng quyết.
- **10-31: người dùng chọn C (đo thêm RE3-OB, RE3-TT, SN) và hỏi vì sao huấn luyện UAC-AD, UAM-AD lâu và "6 lần".** Đã kiểm: `eval_per_scenario_*.py` chạy một `run.py` cho mỗi file test, cùng `--data` (cùng train, unlabel, val; RE2-OB 11 104 + 44 414 + 9 796 cửa sổ) và cùng seed, nên 6 lần huấn luyện là 6 bản của cùng một mô hình (val score epoch 0: 0,1948 ở `cpu`, 0,1923 ở `delay`; khác do GPU không tất định). Bảng E chỉ giữ thời gian một lần; đã xoá các số "huấn luyện cả dataset" (111,5 và 159,9 phút...) và câu "UAM-AD mất nhiều nhất cả dataset" vì chúng chỉ phản ánh số tiến trình của bộ chạy. Một lần UAM-AD: 5 epoch × khoảng 211 giây (RE2-OB), mỗi epoch gồm bước generator (khoảng 100 giây), discriminator (khoảng 100 giây) và val (10 giây) trên 55 518 mẫu trong pkl, GPU GTX 1050.
- **10-31 09:00: chuẩn bị đo Bảng E cho RE3-OB, RE3-TT, SN (lựa chọn C của người dùng), chưa chạy bản đo đầy đủ.** `codes/common/measure_cost.py` thêm: `--repeat` (số lượt chấm trong đồng hồ; file test SN chỉ có 32–263 mẫu nên một lượt quá ngắn), `worker_art_sn`, `worker_eadro_sn`, `worker_oredge_sn` (dựng dữ liệu và cửa sổ chấm giống `run_*_sn.py`, mạng ở trọng số khởi tạo), và đường UAM-AD/UAC-AD cho SN (file test ở `data/sn/scenarios/`, checkpoint seed 0 ở `item8/sn/<mô hình>/<scenario>/`). `cost_table.py` tính ms / 1 000 mẫu theo `samples_done × repeat`. Chạy thử vào thư mục tạm (không ghi vào `cost_all`): SN `test_Svc_Kill_Media` (5 mô hình, repeat 20) và RE3-OB `test_f2` (5 mô hình), đều chạy xong, không lỗi. AnoFusion không đo ở RE3 và SN (RE2-OB đã mất 877 giây cho một file; RE3-OB có 2,6 lần thời gian mỗi epoch). Máy lúc này còn 2,3 GB RAM trống trong 7,9 GB, thấp so với đỉnh khoảng 2,1 GB của UAM-AD ở dataset TT, nên chờ người dùng tắt bớt ứng dụng trước khi chạy đầy đủ.
- **10-31 (tiếp): sửa thời gian huấn luyện và bỏ cột RAM thêm.** (1) Người dùng hỏi "mỗi lần huấn luyện thật sự 18,6 phút?": đã kiểm log, UAM-AD 211–249 giây/epoch × 5 epoch ở cả 30 lần của RE2-OB (trung vị 217 giây); ART 104,7 giây cho 5 epoch (seed 0, RE2-OB), tức khoảng 21 giây/epoch. Con số đúng, nguyên nhân khoảng cách chưa tách được. (2) Phát hiện một lần chạy UAC-AD ở RE2-TT (`socket`, seed 2) bị đứng máy: epoch 0 mất 7 534 giây, epoch 1 mất 2 476 giây (các epoch khác khoảng 230–257 giây), kéo trung bình thêm 5 phút. `cost_table.py` đổi sang **trung vị** của 30 lần huấn luyện cho UAM/UAC-AD: UAM-AD 18,1 (RE2-OB) và 26,6 (RE2-TT), UAC-AD 15,6 và 18,3 (trước đây 23,2). Vậy UAM-AD đắt hơn UAC-AD 16% và 45% mỗi lần, không phải 19% và 15%. (3) Theo ý người dùng bỏ cột "RAM thêm"; bảng còn "RAM đỉnh" (đỉnh − mốc vẫn lưu trong json). Ghi chú trong md nói rõ RAM đỉnh của UAM/UAC-AD cao hơn một phần vì `run.py` nạp thêm dữ liệu huấn luyện. (4) Đối thủ: độ lệch chuẩn thời gian huấn luyện giữa các seed dưới 10%. AnoFusion ở RE3-TT trong item9 ghi 1 212,3 phút (khoảng 20 giờ), cao bất thường so với ước tính 6–8 giờ, chưa kiểm xem có đứng máy không. (5) Đã báo người dùng RAM trống 2,8 GB, chờ người dùng tắt bớt ứng dụng để chạy đo RE3-OB, RE3-TT, SN.
- **10-31 09:19: bắt đầu đo Bảng E cho SN, RE3-OB, RE3-TT (việc C).** Người dùng cho phép dọn RAM; đã đóng Edge (RAM trống 2,8 → 3,4 GB, không đụng tiến trình `claude` và tiến trình hệ thống). Chạy nền một lệnh nối tiếp: SN (12 file test, `--repeat 30`), RE3-OB (5 file), RE3-TT (4 file), 5 mô hình (uacad, uamad, art, eadro, oredge), ghi vào `D:\ClaudeWork\phase3\cost_all\<dataset>\<file test>\<mô hình>.json`, trạng thái ở `cost_all\status.txt` (đầu/cuối ở `c_phase_start.txt`). Ước tính dự kiến: SN 15 phút, RE3-OB 15–25 phút, RE3-TT 35 phút, tổng khoảng 70 phút (60–80). Ước tính dựa trên thời gian từng lượt của RE2 (UAM/UAC-AD khoảng 140 giây ở RE2-OB, 160–210 giây ở RE2-TT, đa phần không phải suy luận) và hai lượt thử nhỏ.
- **10-31 09:49: lần đo C bị hệ thống dừng vì thiếu RAM (không lỗi của lệnh), không tự chạy lại.** Chạy từ 09:19, đến 09:49 đã xong: SN 12 file (60 lượt), RE3-OB 5 file (25 lượt), RE3-TT `test_f1` và `test_f2` (10 lượt); đang chạy dở `test_f3` (uacad), không có file kết quả. Còn thiếu RE3-TT `test_f3` và `test_f4` (10 lượt, ước tính 20–25 phút). Không có lượt nào FAIL. RAM trống sau khi bị dừng: 3,3 GB. Bảng md (Bảng E) nay có RE2-OB, RE2-TT, RE3-OB, SN; RE3-TT chưa đưa vào. Kết quả đáng chú ý: RE3-OB UAM-AD huấn luyện 6,3 phút/lần (UAC-AD 4,4), suy luận 105,2 ms/1000 mẫu (ART 121,6, Eadro 197,9, OrEdge 658,9); SN UAM-AD 241,1 ms/1000 mẫu, **cao hơn ART (104,6)** và gần Eadro (246,5), nên ở SN UAM-AD không thấp hơn mọi đối thủ về suy luận. Phần "Đọc kết quả" của Bảng E chưa viết lại cho 5 dataset (chờ RE3-TT).
- **10-31 10:07: xong đo Bảng E cho cả 5 dataset (165 lượt, không lượt nào FAIL).** Người dùng chọn chạy tiếp RE3-TT và đã đóng ứng dụng; mình đóng thêm `PhoneExperienceHost`, `NVIDIA Overlay`, `msedgewebview2` (RAM trống 3,4 GB) và chạy foreground từng file (không dùng nền để tránh bị dừng): `test_f3` 09:55–10:01, `test_f4` 10:01–10:07, mỗi file khoảng 5,6 phút. Bảng md nay đủ RE2-OB, RE2-TT, RE3-OB, RE3-TT, SN; phần "Đọc kết quả" đã viết lại cho 5 dataset. Số chính (UAM-AD): tham số lớn nhất ở cả 5 dataset; suy luận thấp hơn ART, Eadro, OrEdge ở RE2-OB, RE2-TT, RE3-OB, ngang ART ở RE3-TT (209,3 so với 207,8), **cao hơn ART ở SN (241,1 so với 104,6)**; huấn luyện mỗi lần 18,1 / 26,6 / 6,3 / 10,2 phút và 14 giây (SN), đắt hơn UAC-AD 16% / 45% / 43% / 40% / 13%; RAM đỉnh cao nhất ở 4 dataset RCAEval (một phần do `run.py` nạp dữ liệu huấn luyện) và thấp ở SN. AnoFusion RE3-TT: huấn luyện 1 212,3 phút (20 giờ, 5 epoch) lấy từ file `seed0.f1_only.backup.json` của item9, đã ghi chú trong md. Mã mới chưa commit: `codes/common/measure_cost.py` (thêm `--repeat`, worker SN).
- **10-31 (tối): quay lại Việc 7 (async/sync), phân tích dữ liệu có sẵn, không chạy thêm gì.** Người dùng hỏi F nào là async-only/sync-only, vì sao F01, F13 sync > async còn F05, F09 async > sync, và những mệnh đề nào chứng minh được. Đọc `trace_scores.npz` của các lần chạy DeepTraLog (`D:\ClaudeWork\phase3\dtl\{sync,async,full}\<F>\<hash>\`; chỉ lưu số hạng sync `trace_dis` và số hạng đếm tin nhắn async `trace_dis_async_count`, **không lưu số hạng thứ tự async**), tính AUROC mức cửa sổ của từng số hạng đứng một mình, trung bình 5 seed: F01 sync 0,92–0,93, async 0,43; F02 sync 0,63, async 0,63; F13 sync 0,97–0,98, async 0,41–0,44; F05 sync 0,99, async 0,59–0,64; F09 sync 0,25 (đảo, full run 0,57±0,30), async 0,79–0,84. Cùng dấu vân cấu trúc từ `D:\ClaudeWork\dtl\fcase_fingerprints.pkl`: cạnh lạ / thứ tự lạ F01 0/36,6%, F02 0/0, F13 7,4/77,3%, F05 2,3/6,4%, F09 0/0. Và `order_vs_latency.pkl`: trace bất thường (cancel, preserve, preserveOther) dài 4,7–5,0 giây so với 0,5–0,8 giây bình thường, độ trễ async ≈ 4 giây. Kết luận: nhãn "async" của F01/F02/F13 là cách tiêm lỗi; F01, F13 bị nhánh sync bắt qua đặc trưng thời gian. Chưa có tham chiếu log-only (cả hai nhánh trace tắt) nên chưa tách được phần do log và phần do trace.
- **10-31 (tối, tiếp): kiểm nhanh không dùng mô hình trên đầu vào của 5 F-case DeepTraLog** (`scratchpad\dtl_input_check.py`, đọc `data/deeptralog/scenarios/test_F*.pkl`; AUROC của thống kê thô, trace bất thường so với bình thường cùng file). Cột: sync tốt nhất / async đếm tin nhắn tốt nhất / tỉ lệ trace có ma trận thứ tự chưa thấy trong train (bất thường vs bình thường) / mẫu log lạ. F01: 0,992 (max_dur) / 0,565 / 72% vs 0,5% / 0,999. F02: 0,776 (latency_dev) / 0,789 / 0% vs 0,5% / 1,000. F13: 0,904 / 0,566 / 77% vs 0,5% / 0,913. F05: 1,000 (error_rate) / 0,475 / 99,8% vs 0,3% / 0,803. F09: 0,971 (số cạnh sync) / 0,463 / 100% vs 0,2% / 1,000. Kết luận: không F nào là async-only; cả 5 F có tín hiệu sync (F02 yếu 0,78); đếm tin nhắn async chỉ có tín hiệu ở F02 (0,79); ma trận thứ tự đổi ở 4/5 F nhưng là hệ quả của thời gian/lỗi chứ không riêng lỗi async; log tách gần hoàn hảo ở cả 5 F. Đính chính câu trả lời trước: AUROC số hạng async 0,43 ở F01/F13 chỉ là phần đếm tin nhắn; nhánh async còn số hạng thứ tự (không được lưu trong `trace_scores.npz`), có tín hiệu ở 4/5 F. Fingerprint cũ chỉ đếm cạnh lạ nên bỏ sót F09 (số cạnh sync khác bình thường, AUROC 0,971).
- **10-31 (tối, đính chính hai mục ngay trên): các kết luận "không F nào là async-only", "ma trận thứ tự đổi chỉ là hệ quả của thời gian hoặc lỗi", "log tách gần hoàn hảo ở cả 5 F", "F05, F09 có tín hiệu sync mạnh" là SAI hoặc chưa kiểm chứng.** Nguyên nhân: kiểm nhanh tính trên toàn file test (trộn nhiều API, bất thường và bình thường thuộc API khác nhau), trong khi `docs/preprocess_deeptralog_en.md` §4 và §8 đã ghi phải so với normal cùng API. Kiểm lại theo nhóm tập service (xấp xỉ API; `scratchpad\dtl_input_check3.py`, so trace bất thường với normal train cùng nhóm) và đọc lại dấu vân theo API (`fcase_fingerprints.pkl`, 93 nghìn normal): theo bài (bảng 1) F01, F02, F13 là nhóm Asynchronous Interaction, và đo theo API xác nhận: F01 thứ tự đổi 100% ở preserve (400), số span +64% ở cancel, +83% ở execute; F02 số tin nhắn async +21% ở createOrderBatch (1 199/2 151 trace), 952 trace còn lại (trips/left_parallel) không đổi cấu trúc; F13 thứ tự đổi 100% ở rebook (377), trips/left (299), admintravel (350), preserveOther (350, thêm 37,7% cạnh lạ). F01/F13 còn có trễ khoảng 4 giây: thời lượng request tách hoàn toàn trace bất thường khỏi normal cùng nhóm (AUROC 0,97–1,00), còn số tin nhắn async không đổi trong các nhóm khớp được, nên nhánh sync bắt được các lỗi async này. F02 nhóm food-service: tin nhắn async 11,5 → 14,0 (AUROC 0,75), thời lượng AUROC 1,00, số dòng log +44%. F05 (nhóm Configuration): cạnh lạ ≤ 5%, thứ tự lạ ≤ 10%, số span giảm 45–98%; F09 (Monolithic): cạnh và thứ tự lạ 0%, span −2 đến −3%. Với F05 và F09 gần như không ghép được trace bất thường với normal cùng nhóm tập service (1 624/1 665 và 978/978 nằm ngoài nhóm có ≥ 40 normal train), nên các AUROC mức file của hai F này chưa tách được tín hiệu lỗi khỏi khác biệt API.

- **10-31: cập nhật Việc 7 trong md; người dùng chốt chỉ dùng F01, F02, F13 (async + sync) và trường hợp "không tách được".** Các đoạn bị thay được giữ nguyên văn dưới đây.

<details><summary>Bảng C (định nghĩa, mục 4)</summary>

### Bảng C – Sync vs async (DeepTraLog; xem mục 2.3)

Theo F-case: F01, F02, F13 (async) và F-case không có async (F05, F09). Mỗi ô báo F1@val / AUROC / AUPRC / F1 oracle.

| Cấu hình | Lỗi async | Lỗi không có async |
|:--|:--|:--|
| Chỉ sync | | |
| Chỉ async | | |
| Đủ bộ sync + async (UAM-AD) | | |
| DeepTraLog GGNN | | |

Kỳ vọng cần chứng minh: hàng đủ bộ không thua hàng chỉ-sync và chỉ-async ở cả hai cột, và hơn từng nhánh lẻ ở loại lỗi mà nhánh đó mù.

</details>

<details><summary>Mục 2.3 (cũ)</summary>

### 2.3 DeepTraLog: chạy những F-case nào để chứng minh "trace loại nào cũng detect tốt"

**Đo lại dấu vân cấu trúc (2026-10-13)** [KC: `codes/common/measure_structural_fingerprint.py`, dữ liệu ở `D:\ClaudeWork\dtl`]. Cách tính: trên **các trace có nhãn bất thường** của từng F-case (đúng tập sẽ đem đi đánh giá), tỉ lệ trace có cạnh "lạ" so với trace bình thường cùng API; cạnh sync (S) và async (A) tách riêng theo cùng luật `classify_edge` của `preprocess_deeptralog.py`.

| F-case | Trace bất thường | % cạnh S lạ | % cạnh A lạ | % thứ tự lạ |
|:--|--:|--:|--:|--:|
| F03 | 2564 | 12,9 | 0 | 25,8 |
| F04 | 2546 | 13,0 | 0 | 26,2 |
| F05 | 1665 | 0 | 2,3 | 6,4 |
| F09 | 978 | 0 | 0 | 0 |
| F10 | 900 | 0 | 84,4 | 0 |
| F14 | 870 | 33,7 | 0 | 33,7 |

Kết luận từ số đo:

- **Không F-case nào đạt "≥80% trace có cạnh sync lạ".** Con số "F04: 100%" trước đây chỉ đúng với API `preserve` (330/2546 trace).
- Nhãn trace có mức **sub-case** (`F04-01` … `F04-08`; mỗi sub-case là một API/workload chạy khi lỗi bật). Mỗi F-case chỉ có **một** sub-case thật sự đổi cấu trúc (100% trace có cạnh sync lạ), các sub-case còn lại 0%: F04-01 (330 trace), F03-03 (330), F14-03 (293). Cả ba là cùng một kiểu lỗi (`preserve`/`preserveOther` gọi thêm `consign`): một loại lỗi gặp ba lần, không phải ba lỗi độc lập.
- F10: 84,4% "cạnh A lạ" là các cặp service đã có dạng S trong dữ liệu normal, bị heuristic thời gian (con bắt đầu sau khi cha kết thúc) đổi thành A. Không phải cạnh mới, không dùng làm lỗi cấu trúc.
- Nhãn trace dùng đánh số khác tên file: `F06.zip`, `F09.zip`, `F10.zip` lần lượt mang nhãn `F23`, `F24`, `F25`. Suy từ số trace bất thường khớp đúng (1500 = 5×300, 978 = 330+326+322, 900 = 3×300); chưa có tài liệu xác nhận.
- F-case không có dấu hiệu async (cạnh A lạ ≈ 0, thứ tự lạ thấp): F09, F05. **Chưa biết chỉ-sync có phát hiện được các F-case này không**, cần chạy thử.

**Quyết định (2026-10-13): đổi vế 2** từ "bắt được lỗi cấu trúc" sang "thêm nhánh async không làm hại khi lỗi không có async". F-case chọn theo dấu vân async bằng 0 (dự kiến F05, F09). Vẫn chạy đủ 3 cấu hình trên mọi F-case. Trước khi chốt, chạy thử chỉ-sync 1 seed (~10 phút/F-case) trên F05, F09 và chỉ giữ F-case có AUROC rõ trên 0,5, vì nếu cả ba cấu hình đều xấp xỉ ngẫu nhiên thì kết quả hòa không có ý nghĩa. Sub-case F04-01 / F14-03 là bằng chứng phụ tùy chọn (ghi rõ chỉ là một loại lỗi). **Kết quả chạy thử (chỉ-sync, 1 seed, `norm_sum`, 10 epoch):** F05 F1@val 0,906, AUROC 0,994, AUPRC 0,952; F09 F1@val 0,223, AUROC 0,925, AUPRC 0,444. Cả hai AUROC rõ trên 0,5 nên **giữ cả F05 và F09**. Lưu ý F05 gần chạm trần (AUROC 0,994), nên so sánh "không thấp hơn" ở F05 ít phân biệt được, F09 còn dư địa hơn.

**Mệnh đề có 3 vế**, cùng một nguyên tắc "đủ bộ không thua từng phần lẻ":

| Vế | Phát biểu | F-case | Ghi chú |
|:--|:--|:--|:--|
| 1 | Đủ bộ **>** chỉ-sync | F01, F02, F13 (lỗi async) | Đã có, F1 tăng +0,010 đến +0,036; AUROC rõ hơn (F02 0,655 → 0,809) [Repo] |
| 2 | Đủ bộ **≥** chỉ-sync | F05, F09 (lỗi không có async; chốt sau chạy thử) [CN 10-13] | Thêm nhánh async không làm kết quả tệ đi khi lỗi không liên quan async. Chưa chạy; điều kiện: chỉ-sync có AUROC rõ trên 0,5 ở F-case đó (chạy thử 1 seed). Sub-case F04-01 / F14-03: bằng chứng phụ tùy chọn |
| 3 | Đủ bộ **≥** chỉ-async | cả hai nhóm | Chưa chạy |

**Về vế 3 (đủ bộ ≥ chỉ-async): được, và nên đưa vào.** Code hỗ trợ sẵn cấu hình chỉ-async (`--open_trace False --open_async_trace True`; chú thích trong `fuse_v3.py` ghi "async-only experiment also works" nhưng chưa ai chạy [Repo]). Hai nhánh gần như độc lập về mô hình, cổng và số hạng điểm (mỗi nhánh có trọng số riêng, nhánh async "chỉ cộng vào điểm" [Repo]), nhưng **không hoàn toàn độc lập**: embedding async cũng đi vào `delta_head` dùng chung (`fuse_v3.py:467`), và tất cả cùng chung phần tái tạo log và tổng điểm. Nên "hai nhánh tốt hơn một nhánh" là kết quả đo được, không phải hiển nhiên.

Kỳ vọng và cách đọc:

| Loại lỗi | Chỉ-sync | Chỉ-async | Đủ bộ |
|:--|:--|:--|:--|
| Async (F01, F02, F13) | thấp (không thấy tập cạnh đổi) | cao | ≈ hoặc > chỉ-async |
| Không có async (F05, F09) | cao nếu chỉ-sync phát hiện được | thấp | ≈ chỉ-sync (không thấp hơn quá một độ lệch chuẩn) |

Nghĩa là mỗi nhánh đơn lẻ chỉ bắt được một loại lỗi, còn đủ bộ bắt được cả hai; đó là điểm cần chứng minh. Quy ước so sánh với 5 seed: "≥" nghĩa là trung bình đủ bộ không thấp hơn trung bình nhánh đơn quá một độ lệch chuẩn; ">" nghĩa là hơn quá độ lệch chuẩn. Báo cả F1@val lẫn AUROC/AUPRC; kết luận về F1 thường yếu hơn AUROC (đã thấy ở F02, F1 0,138).

**Cấu hình chạy trên mỗi F-case** (cùng split, `norm_sum`, 5 seed): chỉ-sync (`--open_trace True --open_async_trace False`), chỉ-async (`--open_trace False --open_async_trace True`), đủ bộ (cả hai `True`), và DeepTraLog GGNN (code gốc). AnoFusion, ART, Eadro không chạy được vì dataset không có metric [Repo].

</details>

<details><summary>Việc 7 kết quả (cũ, gồm F05, F09)</summary>

#### Việc 7: kết quả DeepTraLog (Bảng C)

**Việc 7, UAM-AD trên DeepTraLog** (75 ô = 3 cấu hình × F01, F02, F13, F05, F09 × 5 seed, theo `frozen_config.json`). Trung bình ± độ lệch chuẩn (độ lệch chuẩn của quần thể, 5 seed). F1 ở ngưỡng p95 của val; "chỉ-sync" và "chỉ-async" vẫn giữ phần tái tạo log, chỉ khác nhánh trace nào bật:

| F-case | Cấu hình | F1@val | AUROC | AUPRC |
|:--|:--|--:|--:|--:|
| F01 | chỉ-sync | 0,765 ± 0,003 | 0,935 ± 0,007 | 0,830 ± 0,002 |
| F01 | chỉ-async | 0,258 ± 0,000 | 0,855 ± 0,006 | 0,345 ± 0,006 |
| F01 | đủ bộ | **0,775 ± 0,003** | **0,971 ± 0,004** | **0,859 ± 0,005** |
| F02 | chỉ-sync | 0,107 ± 0,001 | 0,664 ± 0,031 | 0,170 ± 0,014 |
| F02 | chỉ-async | 0,105 ± 0,000 | 0,804 ± 0,002 | 0,285 ± 0,002 |
| F02 | đủ bộ | 0,133 ± 0,014 | 0,803 ± 0,033 | 0,257 ± 0,037 |
| F13 | chỉ-sync | 0,852 ± 0,005 | 0,989 ± 0,000 | 0,892 ± 0,008 |
| F13 | chỉ-async | 0,434 ± 0,001 | 0,805 ± 0,003 | 0,467 ± 0,002 |
| F13 | đủ bộ | 0,866 ± 0,002 | 0,990 ± 0,001 | 0,893 ± 0,014 |
| F05 | chỉ-sync | 0,903 ± 0,003 | 0,994 ± 0,000 | 0,952 ± 0,001 |
| F05 | chỉ-async | 0,905 ± 0,001 | 0,999 ± 0,000 | 0,990 ± 0,000 |
| F05 | đủ bộ | 0,910 ± 0,003 | 0,994 ± 0,000 | 0,940 ± 0,007 |
| F09 | chỉ-sync | 0,216 ± 0,006 | 0,923 ± 0,003 | 0,438 ± 0,009 |
| F09 | chỉ-async | 0,241 ± 0,066 | 0,952 ± 0,001 | 0,533 ± 0,006 |
| F09 | đủ bộ | 0,187 ± 0,047 | 0,941 ± 0,004 | 0,494 ± 0,015 |

**Đọc theo quy ước ở mục 2.3** (">" hơn quá một độ lệch chuẩn, "≥" không thấp hơn quá một độ lệch chuẩn; ở đây lấy độ lệch chuẩn lớn hơn của hai cấu hình so sánh):
- **Vế 1 (đủ bộ > chỉ-sync, F01, F02, F13):** đúng cho F1 ở cả ba F-case; đúng cho AUROC và AUPRC ở F01 và F02 (F02 AUROC 0,664 → 0,803); F13 ngang nhau về AUROC và AUPRC (0,989 và 0,990; sync đã gần trần).
- **Vế 2 (đủ bộ ≥ chỉ-sync, F05, F09):** F09 đúng (AUROC 0,923 → 0,941, AUPRC 0,438 → 0,494; F1 thấp hơn 0,029 nhưng trong một độ lệch chuẩn). F05 đúng về F1 nhưng **không đúng chặt** về AUROC (thấp hơn 0,0007, ≈ 2 độ lệch chuẩn, rất nhỏ về số tuyệt đối) và AUPRC (0,952 → 0,940, ≈ 1,6 độ lệch chuẩn).
- **Vế 3 (đủ bộ ≥ chỉ-async, cả hai nhóm):** đúng cho F1 ở cả 5 F-case, đúng mọi chỉ số ở F01, F02, F13 (F02 AUROC và AUPRC ngang trong sai số). **Không đúng** về AUROC và AUPRC ở F05 (AUROC 0,994 so với 0,999, AUPRC 0,940 so với 0,990) và F09 (AUROC 0,941 so với 0,952, AUPRC 0,494 so với 0,533), khoảng cách lớn hơn nhiều lần độ lệch chuẩn. Theo nguyên tắc đã ghi, báo đúng số đo, không chỉnh tham số: phát biểu của vế 3 phải thu hẹp lại thành "đủ bộ không thua nhánh đơn về F1; về xếp hạng (AUROC, AUPRC), đủ bộ vượt chỉ-sync ở lỗi async nhưng kém chỉ-async một chút ở F05, F09".
- **Điều cần nói rõ khi viết:** (1) ở F05 và F09 (không có tín hiệu async), "chỉ-async" vẫn đạt rất cao vì nó giữ phần tái tạo log, nên điểm cao đó nhiều khả năng do log chứ không do nhánh async; (2) F1 ở ngưỡng p95 của val rất thấp ở F02 và F09 (0,1–0,3) dù AUROC cao hơn mức ngẫu nhiên rõ, tức ngưỡng từ val không chuyển sang test tốt, đã thấy ở các lần chạy trước; (3) **cột `oracle_f1` do `run.py` ghi cho DeepTraLog không tin được**: tính với `point_adjust` (dòng log ghi rõ) và mâu thuẫn với AUROC (F05 chỉ-async AUROC 0,9994 nhưng oracle F1 chỉ 0,019). Chưa điều tra; không dùng cột này cho DeepTraLog cho đến khi kiểm, và không so với `oracle_f1` của GGNN vì cột đó là quét ngưỡng thường, không point-adjust.

**Việc 7, DeepTraLog GGNN** (5 seed, 30 epoch, `run_ggnn.py`). Trung bình ± độ lệch chuẩn (quần thể) qua 5 seed; ngưỡng p95 điểm val của chính GGNN:

| F-case | F1@val | AUROC | AUPRC | Precision | Recall | F1 oracle (quét ngưỡng, không point-adjust) |
|:--|--:|--:|--:|--:|--:|--:|
| F01 | 0,793 ± 0,003 | 0,936 ± 0,014 | 0,898 ± 0,006 | 0,723 | 0,879 | 0,933 |
| F02 | 0,856 ± 0,007 | 0,998 ± 0,001 | 0,981 ± 0,010 | 0,748 | 1,000 | 0,964 |
| F13 | 0,745 ± 0,008 | 0,943 ± 0,016 | 0,863 ± 0,012 | 0,699 | 0,799 | 0,880 |
| F05 | 0,900 ± 0,004 | 1,000 ± 0,000 | 0,999 ± 0,000 | 0,818 | 1,000 | 0,994 |
| F09 | 0,853 ± 0,008 | 0,998 ± 0,001 | 0,967 ± 0,013 | 0,744 | 1,000 | 0,978 |

**So với UAM-AD đủ bộ (cùng F-case, 5 seed):**

| F-case | F1@val UAM-AD / GGNN | AUROC UAM-AD / GGNN | AUPRC UAM-AD / GGNN |
|:--|--:|--:|--:|
| F01 | 0,775 / **0,793** | **0,971** / 0,936 | 0,859 / **0,898** |
| F02 | 0,133 / **0,856** | 0,803 / **0,998** | 0,257 / **0,981** |
| F13 | **0,866** / 0,745 | **0,990** / 0,943 | **0,893** / 0,863 |
| F05 | **0,910** / 0,900 | 0,994 / **1,000** | 0,940 / **0,999** |
| F09 | 0,187 / **0,853** | 0,941 / **0,998** | 0,494 / **0,967** |

**Kết quả không có lợi cho UAM-AD, báo đúng số đo:** GGNN thắng rõ ở F02 và F09 (mọi chỉ số, F1 chênh 0,72 và 0,67), và thắng AUPRC ở F05 (0,999 so với 0,940) và F01 (0,898 so với 0,859); UAM-AD chỉ thắng ở F13 (cả ba chỉ số), AUROC của F01, và F1 của F05 (0,910 so với 0,900, chênh nhỏ). Luận điểm "đủ bộ ≥ nhánh đơn" (bảng trên) vẫn là so sánh nội bộ UAM-AD; so với GGNN thì UAM-AD **không** vượt trội trên DeepTraLog, nhất là ở F02 và F09. **Điều kiện không ngang nhau, phải ghi trong luận văn:** (1) GGNN chấm từng trace riêng, UAM-AD chấm cửa sổ 5 trace; (2) GGNN huấn luyện trên ≈ 65.000 trace normal của riêng nó (chia 60/10/30), UAM-AD trên `train.pkl`; (3) embedding sự kiện của GGNN do ta dựng lại từ nội dung sự kiện (TF-IDF + SVD, xem `working_history.md`, mục "DeepTraLog GGNN") vì tác giả không công bố file embedding, nên có thể mang thông tin nội dung mà UAM-AD không thấy; chưa kiểm xem phần này có làm điểm GGNN cao hơn thực tế không; (4) 30 epoch thay vì 100 của tác giả. Ngưỡng p95 của GGNN rất sát 0 (mỗi seed từ −1,3e-6 đến +3,2e-7) và R ≈ 0,003, vì Deep SVDD co hình cầu về gần một điểm; xếp hạng vẫn tốt (AUROC ≥ 0,94), nhưng điểm số tuyệt đối rất nhỏ. Cột oracle F1 của GGNN là quét ngưỡng thường, nhất quán với AUROC (AUROC 0,9998 ↔ oracle 0,994), khác cột oracle của `run.py` (xem cảnh báo ở trên).

</details>
- **10-31: Việc 12 (Giai đoạn 4), tổng hợp Bảng A–E, phần bảng đã xong.** Thêm mục `#### Việc 12 – Tổng hợp Bảng A–E` trong md (sau Việc 11): Bảng A chép từ mục 4 (đếm lại train, val bằng pkl: SN 39/57, RE2-OB 11 104/9 796, RE2-TT 11 016/9 720, RE3-OB và RE3-TT 3 672/3 240, DeepTraLog 6 000/1 500; hàng DeepTraLog đổi thành 3 file F01, F02, F13); Bảng B (B1 F1@val95, B2 AUROC, B3 AUPRC có cột Trung bình, B4 F1 oracle), Bảng C (F01, F02, F13 × 4 cấu hình), Bảng D (HR@1/3/5), Bảng E (E1 huấn luyện, E2 tham số, E3 suy luận, E4 RAM đỉnh) **sinh bởi mới** `D:\ClaudeWork\phase3\item8\summary_tools\final_tables.py` (chạy `WRITE=1 python final_tables.py`, ghi giữa các marker `<!-- FINAL B/C/D/E BEGIN/END -->`). Kiểm chéo bằng script (`scratchpad\crosscheck_viec12.py`): 335 ô trùng đúng chuỗi với T1, S2, bảng Việc 7, Bảng D và Bảng E của Việc 10, 11; 4 ô khác chỉ do E1 hiển thị 2 chữ số thập phân khi dưới 1 phút (SN và vài ô nhỏ: 0,07 so với 0,1). **Sửa số bảng Việc 7**: năm ô lệch chữ số cuối vì bảng cũ gõ từ đầu ra 4 chữ số (F01 chỉ-async AUPRC ±0,006 → ±0,005; F02 chỉ-async AUROC 0,804 → 0,803; F02 đủ bộ AUPRC 0,257 → 0,256; F13 chỉ-sync F1 ±0,005 → ±0,004; F13 chỉ-async AUROC ±0,003 → ±0,002) và hai câu trích số đó; kết luận (14 cặp hơn, 4 cặp ngang) không đổi. Danh sách "điều phải ghi trong luận văn" đã đưa vào mục này. Chưa viết phần chữ của chương kết quả. Chưa commit gì (mã mới chưa commit: `codes/common/measure_cost.py`; công cụ bảng ở `D:\ClaudeWork` không thuộc repo).
- **10-31: AnoFusion đo suy luận ở SN; sửa Bảng E và mô tả; Bảng B, E chia hai phần.** Người dùng hỏi vì sao các ô AnoFusion của Bảng E còn "–" (nhớ là chỉ kẹt ở TT) và SN nhỏ mà vẫn không có số. Nguyên nhân thật: (1) SN không đo vì `measure_cost.py` chưa có worker AnoFusion cho SN (đã thêm `worker_anofusion_sn`, dữ liệu và cửa sổ chấm lấy từ `run_anofusion_sn.py`), không phải vì chậm; (2) lời giải thích cũ trong md sai ở hai chỗ: ghi "RE2-TT có 1 357 kênh" trong khi 1 357 kênh là của **RE3-TT** (RE2-TT có 596, RE3-OB 465, RE2-OB 193, SN 583), và lý do bỏ RE3 và SN là suy từ thời gian huấn luyện, chưa đo. Đã đo AnoFusion ở SN (12 file, mỗi file lặp 10 lần): 318 456 ms / 1 000 cửa sổ, RAM đỉnh 4 620 MB, 2 526 964 tham số; số tham số đếm được cho cả 5 dataset bằng `summary_tools/anofusion_params.py` (309 814; 2 637 542; 1 631 462; 13 233 706; 2 526 964 ở RE2-OB, RE2-TT, RE3-OB, RE3-TT, SN). **Hệ quả:** câu cũ "UAM-AD lớn nhất trong các mô hình đã đo ở cả 5 dataset" sai, vì AnoFusion lớn hơn UAM-AD ở bốn dataset; đã sửa. RE2-TT, RE3-OB, RE3-TT vẫn chưa đo suy luận (chưa chạy thử). Bảng B và E ở Việc 12 chia hai phần: a (UAC-AD so với UAM-AD) và b (UAM-AD so với các mô hình khác). Đoạn bị thay, giữ nguyên văn:

<details><summary>Các đoạn cũ của Việc 10, 12 (trước khi đo AnoFusion ở SN)</summary>

> AnoFusion chỉ đo suy luận 1 file ở RE2-OB, các dataset khác chỉ có thời gian huấn luyện; điều kiện và lưu ý ở Bảng E

> 165 lượt đo (RE2: 60, RE3-OB: 25, RE3-TT: 20, SN: 60), không lượt nào lỗi

> - **AnoFusion:** suy luận chỉ đo ở RE2-OB, trên một file (`test_cpu`, 877 giây, toàn bộ file); hai ô dấu * (ms / 1000 mẫu, RAM đỉnh) là của file đó, không phải trung bình 6 file, vì đo 6 file mất khoảng 90 phút. Không đo ở bốn dataset còn lại: RE2-TT có 1 357 kênh (tensor chú ý ước tính 128 × 1 357² × 4 byte ≈ 0,9 GB mỗi cái, nguy cơ hết RAM 7,9 GB của máy); RE3 và SN không đo vì RE3-OB đã nặng gấp 2,6 lần RE2-OB mỗi epoch ở lần huấn luyện việc 9 (263 so với 100 giây). 

> SN có worker riêng cho ART, Eadro, OrEdge (dữ liệu, tỉ lệ chuẩn hoá và cửa sổ chấm lấy từ `run_art_sn.py`, `run_eadro_sn.py`, `run_oredge_sn.py`, mạng ở trọng số khởi tạo)

> nên mỗi file được chấm lặp 30 lần trong đồng hồ (cột "ms / 1000 mẫu" chia cho cả số lần lặp)

> (a) **Tham số:** UAM-AD lớn nhất trong các mô hình đã đo ở cả 5 dataset (627, 687, 681, 930 và 702 nghìn ở RE2-OB, RE2-TT, RE3-OB, RE3-TT, SN): gấp khoảng 2 lần AnoFusion (RE2-OB), 9–13 lần Eadro, 43–70 lần ART, 5–53 lần OrEdge (số tham số của OrEdge tăng theo số mẫu log nên khoảng cách nhỏ nhất ở RE3-TT và SN); so với UAC-AD thêm 3,2%, 6,1%, 5,8%, 14,9% và 6,8%, do nhánh trace.

> Cần đọc kèm lưu ý đơn vị chấm và lưu ý SN ở trên.

> UAM-AD cao nhất trong năm mô hình ở bốn dataset RCAEval

> ở SN UAM-AD 801 MB, ngang UAC-AD (796) và thấp hơn ART (1 101), Eadro (1 145), OrEdge (1 327).

> In đậm ở B1–B3 = trung bình cao nhất trong nhóm không giám sát ở cột đó (chưa xét độ lệch chuẩn).

> `*` = chỉ đo 1 file test (AnoFusion, RE2-OB); `–` = không đo.

> AnoFusion 1 seed, RE3-TT chỉ `test_f1`, suy luận chỉ đo 1 file ở RE2-OB;

</details>
- **10-31: Bảng B, C, E của Việc 12 đổi cách trình bày; F1 oracle của DeepTraLog tính lại từ checkpoint.** (1) Bảng B và E chia hai phần: a = UAC-AD so với UAM-AD (hai hàng, không cột Nhóm/Học), b = UAM-AD so với AnoFusion, ART, OrEdge, Eadro (không có UAC-AD); các bảng gán nhãn B1a..B4a, B1b..B4b và E1a..E4a, E1b..E4b. Mã sinh bảng: `final_tables.py` (`PARTS`, `table_bd(grouped=...)`). (2) Bảng C: mỗi dòng một cấu hình, cột F1@val95, AUROC, AUPRC, F1 oracle, bốn dòng (chỉ sync, chỉ async, đủ bộ, GGNN) cho mỗi F-case. (3) **`oracle_f1` trong `info_score.txt` của DeepTraLog không dùng được**: đó là oracle có point-adjust của điểm thô (`raw_sum`), quét top 1–19%, và có thể thấp hơn F1@val95 (F01 đủ bộ seed 0: 0,6306 so với F1 0,7723; ngưỡng oracle 0,2345 khác thang của ngưỡng F1 3,1776). `trace_scores.npz` cũng không đủ để dựng điểm norm_sum (chỉ có `trace_dis`, đếm async và nhãn). Cách làm: `codes/common/rescore_dtl_oracle.py` nạp lại 45 checkpoint (3 cấu hình × F01, F02, F13 × 5 seed; một tiến trình `run.py --pre_model` cho mỗi (cấu hình, F-case) rồi lần lượt 5 checkpoint, giống `rescore_item10.py`), lấy thành phần điểm val và test bằng `BaseModel._score_components`, gọi `BaseModel.evaluate_norm_sum` của repo và lấy điểm cuối từ lời gọi `roc_auc_score` của nó; oracle = F1 tốt nhất qua mọi ngưỡng, không point-adjust. F1@val95, AUROC, AUPRC dựng lại **khớp `info_score.txt` ở cả 45 checkpoint** (lệch dưới 1e-3; seed 0 của F01 đủ bộ: 0,7723 / 0,9784 / 0,8525 cả hai phía), nên việc nạp checkpoint ở tiến trình mới là an toàn (sửa `sorted()` của `semantics.py` đã có trong HEAD từ 09-28; ghi chú cũ trong bộ nhớ nói chưa sửa là lỗi thời, đã cập nhật). Kết quả ở `D:\ClaudeWork\phase3\dtl_oracle\<cấu hình>\<F>\seed<k>.json`. F1 oracle (trung bình±độ lệch chuẩn): F01 chỉ-sync 0,809±0,003, chỉ-async 0,585±0,003, đủ bộ 0,800±0,017, GGNN 0,933±0,001; F02 0,338±0,018, 0,492±0,002, 0,468±0,028, GGNN 0,964±0,024; F13 0,874±0,012, 0,506±0,010, 0,898±0,021, GGNN 0,880±0,001. Đủ bộ so với nhánh lẻ: hơn ở 4 trong 6 cặp, ngang ở 2 (F01 so với chỉ-sync −0,009; F02 so với chỉ-async −0,024), không cặp nào thấp hơn quá một độ lệch chuẩn. (4) Kiểm chéo `scratchpad\crosscheck_v2.py` sau khi sinh lại: 414 ô trùng, 4 ô khác chỉ do E1 hiển thị 2 chữ số thập phân khi dưới 1 phút (0,07 so với 0,1; 0,66 so với 0,7; 0,88 so với 0,9). Cột AnoFusion của E2 (số tham số), E3, E4 được so với file json. Chưa commit gì (mã mới chưa commit: `codes/common/measure_cost.py`, `codes/common/rescore_dtl_oracle.py`; công cụ bảng ở `D:\ClaudeWork` không thuộc repo).
- **11-01: Bảng B, D thêm Δ và Δ%; Bảng C tô đậm cặp UAM-AD đủ bộ so với GGNN; làm rõ vì sao AnoFusion thiếu số CPU ở RE2-TT, RE3-OB.** Người dùng yêu cầu: B phần a có Δ giữa UAM-AD và UAC-AD, phần b có Δ giữa mô hình khác và UAM-AD; D tương tự; C không cần Δ nhưng muốn dễ so UAM-AD đủ bộ với GGNN mà không tách bảng. Quy ước: Δ = số trước − số sau, Δ% = Δ / số bị trừ (phần a: UAC-AD; phần b và D: UAM-AD); tính từ trung bình chưa làm tròn (`final_tables.py`: `delta_rows`). Cột Trung bình của AnoFusion so với UAM-AD trên đúng 4 dataset (không RE3-TT). B4b chỉ tính Δ ở RE2-OB, RE2-TT. Bảng C: in đậm hai dòng của cặp và số cao hơn trong cặp theo từng cột (GGNN hơn 7/12 ô, UAM-AD đủ bộ 5/12). Về Bảng E: người dùng nhớ AnoFusion chỉ kẹt ở RE3-TT. Đúng: file kết quả việc 9 cho thấy AnoFusion chấm xong cả 6 file RE2-TT (khoảng 1 308 giây mỗi file, GPU) và cả 5 file RE3-OB (183–471 giây); RE3-TT chấm `test_f1` mất 72 545 giây (khoảng 20 giờ) rồi dừng. Ô "–" ở E3, E4 của RE2-TT, RE3-OB là do phép đo CPU (`measure_cost.py`) chưa chạy ở đó, không phải do mô hình kẹt; đã ghi rõ trong Việc 10 và phần dẫn của Bảng E. Đoạn bị thay, giữ nguyên văn:

<details><summary>Câu cũ trong phần dẫn của Bảng E (trước 11-01)</summary>

> AnoFusion ở RE2-TT, RE3-OB, RE3-TT chưa đo suy luận và RAM, xem Việc 10;

</details>
- **11-01 (tiếp): đổi dấu Δ; đo AnoFusion trên CPU ở RE3-OB, RE2-TT; kiểm bản dựng AnoFusion.** (1) Dấu Δ: người dùng chỉ ra phần b và Bảng D phải là UAM-AD − mô hình khác (cho biết UAM-AD hơn hay thua bao nhiêu), Δ% chia cho mô hình khác; đây đúng là quy ước của T1 (Việc 9) và Bảng D (Việc 11) nên mình đã hiểu sai ở lần trước; đã đổi (`final_tables.py`: `delta_rows(lead, others)`). Ô † của AnoFusion ở RE3-TT so UAM-AD trên riêng `test_f1`, như T1 (`series_on`). Kiểm chéo: 126 ô Δ (Δ%) của phần b và Bảng D trùng từng ký tự với T1 và Bảng D cũ; 308 ô Δ còn lại nằm trong khoảng làm tròn của các ô hiển thị. (2) Đo CPU (người dùng đồng ý 11-01, đóng Edge, Steam để lấy RAM; `measure_cost.py --all --models anofusion --limits anofusion=2560 --repeat 1`, từng tiến trình riêng): RE3-OB (`test_f1`, N = 465) 143 313,1 ms / 1 000 mẫu, RAM đỉnh 3 569 MB, 366,9 giây; RE2-TT (`test_cpu`, N = 596) 311 556,3 ms / 1 000 mẫu, RAM đỉnh 5 062 MB, 797,6 giây; số tham số khớp `anofusion_params.json`. RE3-TT không đo (người dùng đồng ý). Kết quả ở `D:\ClaudeWork\phase3\cost_all\<dataset>\<test>\anofusion.json`; `cost_table.py` và `final_tables.py` gắn dấu * cho ô đo trên một file. (3) Người dùng hỏi AnoFusion của ta có dựng đúng không, vì sợ bị hỏi vì sao nó chậm. Kiểm: đọc từng khối mã so với mã tác giả (GTN, GAT, GAT_GRU, cách tạo cửa sổ, công thức điểm), rồi chạy so số học trên CPU với cùng trọng số ngẫu nhiên: chênh lệch tối đa 0 (N = 12, 30). Mô hình chậm vì độ rộng ẩn của GAT bằng số kênh, chi phí khoảng 42·N³ mỗi mẫu; số đo khớp (34–59 GFLOP/s). Khác biệt với cách dùng của tác giả là ta coi cả hệ thống là một instance nên N rất lớn. Đã ghi vào Việc 10. Đoạn bị thay, giữ nguyên văn:

<details><summary>Các đoạn cũ của Việc 10, Bảng E (trước khi đo AnoFusion ở RE3-OB, RE2-TT)</summary>

> AnoFusion đo suy luận ở SN (12 file, đo thêm ngày 10-31) và RE2-OB (1 file); RE2-TT, RE3-OB, RE3-TT chưa đo suy luận và RAM (số tham số thì có đủ cả 5 dataset);

> 177 lượt đo (RE2: 60, RE3-OB: 25, RE3-TT: 20, SN: 72, trong đó 12 lượt AnoFusion ở SN đo thêm ngày 10-31), không lượt nào lỗi

> - **AnoFusion:** suy luận đo ở **SN** (cả 12 file; mỗi file chấm lặp 10 lần vì AnoFusion chậm; chấm được 750 cửa sổ, ít hơn 1 279 của ART vì 20 cửa sổ đầu mỗi phiên không đủ lịch sử) và ở **RE2-OB** trên một file (`test_cpu`, 877 giây, toàn bộ file; hai ô dấu * là của file đó, không phải trung bình 6 file, vì đo cả 6 file mất khoảng 90 phút). **RE2-TT, RE3-OB, RE3-TT chưa đo suy luận và RAM.** Đây là phép đo chi phí CPU (`measure_cost.py`) chưa chạy cho ba dataset đó; bản thân mô hình đã chạy ở việc 9 (trên GPU): chấm xong cả 6 file RE2-TT (khoảng 1 308 giây mỗi file, 52 275 mẫu được chấm) và cả 5 file RE3-OB (183–471 giây mỗi file), còn ở RE3-TT chỉ riêng `test_f1` đã mất 72 545 giây (khoảng 20 giờ) và lần chạy dừng sau file đó. Thời gian GPU này không thay được số đo CPU của Bảng E (ở RE2-OB, file `test_cpu`: GPU 100 giây, CPU 877 giây). Chi phí của AnoFusion tăng mạnh theo số kênh N: RE2-OB (N = 193) 17 636 ms / 1 000 mẫu, RAM thêm 1 067 MB; SN (N = 583) 318 456 ms / 1 000 cửa sổ, RAM thêm 3 525 MB (đỉnh 4 620 MB). RE3-OB có N = 465, RE2-TT 596, RE3-TT 1 357. Chưa chạy thử ba dataset này; nếu RAM thêm tăng gần tuyến tính theo N (ước tính từ hai điểm đo, không phải số đo) thì RE3-TT ở batch 128 vượt RAM 7,9 GB của máy. Số tham số không cần suy luận nên đếm được ở cả 5 dataset (`summary_tools/anofusion_params.py`). 

> - **SN và RE3:** RE3 dùng đúng bộ đo của RE2.

> **AnoFusion** chậm hơn rất nhiều ở hai nơi đã đo: 318 456 ms / 1 000 cửa sổ ở SN (gấp khoảng 1 320 lần UAM-AD và 3 040 lần ART) và 17 636 ms ở RE2-OB trên file `test_cpu` (gấp khoảng 160 lần UAM-AD).

> **AnoFusion cao hơn UAM-AD ở hai nơi đã đo:** 4 620 MB ở SN (gấp 5,8 lần) và 1 695 MB* ở RE2-OB (UAM-AD 1 183).

> `*` = chỉ đo 1 file test (AnoFusion, RE2-OB); `–` = không đo (AnoFusion ở RE2-TT, RE3-OB, RE3-TT chưa đo suy luận và RAM trên CPU, dù ở việc 9 mô hình đã chấm xong RE2-TT và RE3-OB (RE3-TT chỉ chấm `test_f1`), xem Việc 10; số tham số của nó đếm được ở cả 5 dataset).

> suy luận AnoFusion chỉ đo ở SN và 1 file của RE2-OB (RE2-TT, RE3-OB, RE3-TT chưa đo);

</details>
