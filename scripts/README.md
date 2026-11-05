# scripts – chạy hàng loạt

Chạy từ **thư mục gốc repo**: `python -m scripts.<tên_không_có_.py> [tham số]`. Đường dẫn mặc định lấy từ `src/config.py`.

| Script | Người viết | Việc | Mục trong `docs/PHAN_VIEC_v2.md` |
|---|---|---|---|
| `check_inputs.py` | có sẵn | kiểm tra vai trò của bạn đã đủ input chưa | – |
| `check_dataset.py` | PV2 | đếm file, kích thước ảnh, chiều mask → `docs/dataset_report.txt` | 2.2 |
| `make_eval_pairs.py` | PV2 | tạo `data/processed/eval_pairs.csv` | 2.3 |
| `run_baseline_batch.py` | PV2 | baseline cho 200 cặp → `outputs/baseline/` | 2.7 |
| `run_batch.py` | PV1 | CatVTON cho 200 cặp → `outputs/catvton/` (chạy tiếp được khi bị ngắt) | 1.4 |
| `make_cache.py` | PV1 | sinh sẵn 60 cặp demo → `data/processed/cache/` | 1.5 |

Script của PV3 nằm ở `eval/`: `python -m eval.run_eval ...`.
