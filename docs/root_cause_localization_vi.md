# Root Cause Localization (theo TraceDAE §E)

> Thêm khả năng xếp hạng service nghi ngờ là root cause, dựa trên nhánh
> trace đã có sẵn (`--open_trace True`). Không thêm model mới, không thêm
> hyperparameter mới cho phần tính score — tái sử dụng nguyên vẹn
> reconstruction của dual-autoencoder mà `TraceModel`
> (`codes/models/trace_model_v3.py`) đã tính sẵn.

---

## 1. Cơ chế

TraceDAE §E: khi một trace/window bị gắn cờ bất thường, tính **anomaly
score theo từng node**, xếp hạng các service node theo score đó, rồi báo
cáo top-k service nghi ngờ là root cause.

| Paper (Eq. 14) | UAM-AD |
|---|---|
| `S_i = α‖A_i−Â_i‖² + (1-α)‖X_i−X̂_i‖²` | `node_scores[i] = loss_struct_per_node[i] + lambda_lat * loss_latency_per_node[i] + lambda_err * loss_error_per_node[i]` — dùng lại đúng trọng số `TraceModel` đã dùng cho loss training tổng hợp, chỉ khác là giữ lại chiều node thay vì rút gọn đi. |
| node → microservice | `idx2service`, dựng từ `meta.pkl["service2idx"]` |
| "for each abnormal STG" | tái dùng threshold mà `BaseModel.evaluate()` đã chọn |

Cài đặt: `TraceModel.forward` (`codes/models/trace_model_v3.py`) trả thêm
giá trị thứ 5, `node_scores` `[B, N]`, tính song song — không thay thế —
`loss` `[B]` tổng hợp hiện có. `MultiModel.forward`
(`codes/models/fuse_v3.py`) reshape nó thành `[B, W, N]` và thêm vào dict
kết quả với key `"node_scores"`.

`BaseModel.localize_root_causes(test_loader, threshold, top_k=3)`
(`codes/models/basev3.py`) sau đó, với mỗi `(window, timestep)` có
`fusion_loss > threshold`, xếp hạng `N` service node theo `node_scores` và
giữ lại top-k (`codes/models/rca.py::rank_top_k_services`).

## 2. HR@k và MRR

Hai chỉ số chuẩn đánh giá xếp hạng RCA, chỉ tính trên các anomaly biết
trước ground-truth service thật (`gt_service is not None`):

- **HR@k** (Hit Rate@k): tỉ lệ anomaly (trong số có ground-truth) mà
  service đúng nằm trong top-k dự đoán.
  `HR@k = |{record : gt_service ∈ top_k}| / |record có ground-truth|`
- **MRR** (Mean Reciprocal Rank): trung bình `1 / rank(gt_service)` (bằng
  0 nếu service đúng không xuất hiện trong top-k trả về).

Cài đặt trong `codes/models/rca.py::compute_hit_rate_at_k`.

## 3. Ground-truth theo từng dataset

| Dataset | Có ground-truth không? | Suy ra bằng cách nào |
|---|---|---|
| `rcaeval_re2_ob` | Có, 100% anomaly | Sample id dạng `f"{service}_{fault}_{run_id}_{i}"` — tách thẳng service bị inject lỗi từ id (`preprocess_rcaeval_re2_ob.py:358`). |
| `rcaeval_re3_ob` | Có, 100% anomaly | Cùng cấu trúc id với RE2-OB, hậu tố fault `f1..f5`. |
| `sn` | Có, 9/12 scenario | SN không có `test.pkl` gộp — mỗi lần chạy đều trỏ `--test_pkl` vào đúng 1 file `scenarios/test_<scenario_name>.pkl`, nên scenario cho cả lần chạy được suy ra 1 lần từ tên file đó (`codes/models/rca.py::scenario_from_test_pkl`). (Field `"_scenario"` ghi lúc `_process_scenario` — `preprocess_sn.py:727` — bị strip trước khi lưu pkl bởi `_to_dict`, `preprocess_sn.py:796-801`, nên **không** đọc lại được từ chính sample data.) `SCENARIO2SERVICE` (`codes/models/rca.py`) map 9 tiền tố tên scenario (`Code_Stop_*`, `DB_Redis_CacheLimit_*`, `Svc_Kill_*`) sang 1 service cụ thể. 3 scenario `Perf_*` (`Perf_CPU_Contention`, `Perf_Disk_IO_Stress`, `Perf_Network_Loss`) là stress mức host/hạ tầng, không quy về 1 service — luôn ra `gt_service = None`. |

`gt_service` chỉ được tính khi `true_label == 1` — id/scenario của một
sample normal (chưa inject lỗi) vẫn mang tên thí nghiệm nó được ghi lại,
nhưng service đó không thực sự có lỗi ở timestep normal đó.

## 4. Cách chạy

**Đường nhanh (khuyến nghị để test) — dùng checkpoint có sẵn, không cần
train.** Repo đã có sẵn checkpoint `open_trace=True` commit sẵn dưới
`data/<dataset>/result_per_scenario_fuse_trace/<scenario>/<hash>/model.ckpt`
(xem `params.json` cạnh đó để lấy đúng
`window_size`/`hidden_size`/`num_services`/`trace_c`/`fuse_type` đã dùng,
truyền lại trên command line để khớp kiến trúc với weight đã lưu — luôn
truyền tường minh `--num_services`/`--trace_c` vì cơ chế auto-load từ
`meta.pkl` trong `run.py` chỉ kích hoạt khi 2 flag này còn ở giá trị mặc
định CLI, mà với RE2-OB/RE3-OB/SN giá trị mặc định đó không bao giờ khớp
giá trị thật). **Đã kiểm chứng**: cả 6 checkpoint RE2-OB và cả 5 checkpoint
RE3-OB load sạch với `GATLayer` hiện tại. **Cả 24 checkpoint của SN đều
thuộc bản trước khi refactor decomposed-attention cho `GATLayer` nên load
lỗi** (`size mismatch`/`missing key` ở `gat1.a_l`/`gat1.a_r`) — với SN, hãy
train mới (kể cả train ngắn, vd `--epoches 3 3`) thay vì dùng checkpoint có
sẵn, cho tới khi có checkpoint mới được commit.

```bash
python codes/run.py --data data/rcaeval_re2_ob --dataset rcaeval_re2_ob \
    --data_type fuse --open_trace True --window_size 30 --hidden_size 32 \
    --num_services 11 --trace_c 6 \
    --pre_model data/rcaeval_re2_ob/result_per_scenario_fuse_trace/cpu/671af35f/model.ckpt \
    --test_pkl data/rcaeval_re2_ob/test_cpu.pkl \
    --enable_rca True --rca_top_k 3
```

**Đường train từ đầu**: thêm `--enable_rca True --rca_top_k K` vào lệnh
`run.py` train bình thường (yêu cầu `--open_trace True`).

`--enable_rca` mặc định `False` — các run/kết quả hiện tại không đổi trừ
khi chủ động bật.

## 5. Output

- `rca_results.json` (qua `dump_rca_results`, `codes/common/utils.py`) —
  list record, mỗi record ứng với 1 `(window, timestep)` bị gắn cờ bất
  thường:
  ```json
  {
    "sample_id": "checkoutservice_cpu_2_812",
    "true_label": 1,
    "top_k_services": [["checkoutservice", 0.41], ["cartservice", 0.18], ["frontend", 0.09]],
    "gt_service": "checkoutservice"
  }
  ```
- `info_score.txt` được thêm dòng
  `* RCA -- hr1:.. hr3:.. hr5:.. mrr:.. n_scored:..` bất cứ khi nào có ít
  nhất 1 record biết `gt_service`.

## 6. Lưu ý biết trước — protocol threshold khác nhau theo dataset (không thuộc phạm vi sửa ở đây)

`sn` có `val.pkl` (20% cuối của `Normal_Baseline`, chưa từng dùng để
train) nên dùng được protocol threshold không leak (`--val_percentile`,
`threshold = percentile(val_losses, val_percentile)`).
`rcaeval_re2_ob`/`rcaeval_re3_ob` không tạo `val.pkl`, và phần "normal"
trong mỗi `test_<fault>.pkl` được random sample từ **toàn bộ** normal pool,
không loại trừ id đã dùng trong `train.pkl`/`unlabel.pkl`
(`preprocess_rcaeval_re2_ob.py:440`) — tức có leak giữa train và test cho
lớp normal, không riêng threshold. RCA vẫn hoạt động đúng trên 2 dataset
này bất kể threshold do protocol nào chọn ra các window bị gắn cờ, nhưng
F1/precision/recall báo cáo có thể lạc quan hơn thực tế. Sửa triệt để cần
chạy lại preprocessing trên raw data và train lại — đã tách thành task
riêng, không thuộc phạm vi tính năng này.
