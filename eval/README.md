# eval – đánh giá mô hình (PV3)

Chạy từ thư mục gốc repo: `python -m eval.<tên_không_có_.py>`. Cài thư viện: `pip install -r requirements-eval.txt`.

| Script | Việc | Mục trong `docs/PHAN_VIEC_v2.md` |
|---|---|---|
| `selftest.py` | hai phép thử bắt buộc trước khi chạy dữ liệu thật: pred = gt và pred = ảnh đen | 3.2 |
| `run_eval.py` | MSE/RMSE/MAE, SSIM, LPIPS (paired) và FID/KID (unpaired) → `results/results_*.csv` | 3.2 |
| `make_survey_set.py` | chọn và ẩn danh ảnh khảo sát, ghép ảnh A/B cho Form → `outputs/survey/`, `results/survey_key.csv` | 3.3 |
| `analyze_survey.py` | ma trận nhầm lẫn, accuracy, precision, recall, F1 → `results/survey_summary.csv` | 3.4 |
| `make_report_assets.py` | biểu đồ và 5 ca lỗi → `results/figs/`, `results/failure_cases/` | 3.5 |

Trước khi chạy với dữ liệu thật, thử hai phép kiểm tra trong mục 3.2: ảnh dự đoán = ảnh thật phải cho MSE = 0, SSIM = 1; ảnh đen phải cho số xấu rõ rệt.

Thứ tự chạy 3.2: `python -m eval.selftest --gt data/raw/test/image` (phải đạt, kể cả LPIPS) rồi `python -m eval.run_eval ... --method baseline` và `--method catvton`. Hai lần chạy ghi chung vào `results/results_per_pair.csv` và `results/results_summary.csv`; cặp thiếu file được liệt kê ở `results/eval_missing_<method>.csv`.

Khảo sát: dựng bằng `python -m eval.make_survey_set` (bộ đã có từ trước thì chỉ chạy thêm `--ab-sheets` để ghép ảnh A/B), dựng Form theo `docs/survey_form_guide.md`, phân tích bằng `python -m eval.analyze_survey --responses outputs/survey_responses.csv`.

Báo cáo: `python -m eval.make_report_assets` (3.5, cần 3.2 chạy thật cho cả hai method) vẽ `results/figs/bar_*.png` và xuất 5 ca lỗi vào `results/failure_cases/`. Ảnh ngoài dataset (3.6): `python -m eval.external_notes init`, điền `results/external_notes.csv`, rồi `python -m eval.external_notes summary`.
