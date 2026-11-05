# eval – đánh giá mô hình (PV3)

Chạy từ thư mục gốc repo: `python -m eval.<tên_không_có_.py>`. Cài thư viện: `pip install -r requirements-eval.txt`.

| Script | Việc | Mục trong `docs/PHAN_VIEC_v2.md` |
|---|---|---|
| `run_eval.py` | MSE/RMSE/MAE, SSIM, LPIPS (paired) và FID/KID (unpaired) → `results/results_*.csv` | 3.2 |
| `make_survey_set.py` | chọn và ẩn danh ảnh khảo sát → `outputs/survey/`, `results/survey_key.csv` | 3.3 |
| `analyze_survey.py` | ma trận nhầm lẫn, accuracy, precision, recall, F1 → `results/survey_summary.csv` | 3.4 |
| `make_report_assets.py` | biểu đồ và 5 ca lỗi → `results/figs/`, `results/failure_cases/` | 3.5 |

Trước khi chạy với dữ liệu thật, thử hai phép kiểm tra trong mục 3.2: ảnh dự đoán = ảnh thật phải cho MSE = 0, SSIM = 1; ảnh đen phải cho số xấu rõ rệt.
