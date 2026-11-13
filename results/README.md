# results – kết quả thí nghiệm (CSV, biểu đồ; commit lên Git, trừ đáp án khảo sát)

| File / thư mục | Người tạo | Nội dung |
|---|---|---|
| `results_per_pair.csv` | PV3 | `id,mode,method,region,mse,rmse,mae,ssim,lpips` |
| `results_summary.csv` | PV3 | `method,mode,region,metric,mean,std,n` (kèm FID, KID cho unpaired) |
| `survey_summary.csv` | PV3 | ma trận nhầm lẫn, accuracy, precision, recall, F1, tỷ lệ chọn CatVTON |
| `survey_key.csv`, `ab_key.csv` | PV3 | đáp án bộ khảo sát: **không commit** (`.gitignore` chặn), để trên Drive `tryon_project/results/` |
| `timing_log_catvton.csv`, `errors_catvton.csv` | PV1 | `id,seconds,peak_vram_gb` / `id,error` |
| `errors_baseline.csv` | PV2 | `id,error` |
| `external_notes.csv` | PV3 | nhận xét định tính ảnh ngoài dataset |
| `figs/` | PV3 | biểu đồ so sánh (PNG, nhỏ) |
| `failure_cases/` | PV3 | 5 ca lỗi (ảnh): để trên Drive, `.gitignore` chặn ảnh |

Mọi thí nghiệm cố định `seed` (xem `src/config.py`). Định nghĩa chỉ số: `docs/metrics_definition.md`.
