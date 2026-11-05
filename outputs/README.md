# outputs – ảnh kết quả (để trên Drive, KHÔNG commit)

Tên file luôn là `{tên_ảnh_người}__{tên_ảnh_áo}.png`, ví dụ `00006_00__00012_00.png`. Cặp `paired` có hai tên giống nhau.

| Thư mục | Nội dung | Người giao → người nhận |
|---|---|---|
| `catvton/` | ảnh CatVTON cho 200 cặp (`scripts/run_batch.py`) | PV1 → PV3 |
| `baseline/` | ảnh baseline cho 200 cặp (`scripts/run_baseline_batch.py`) | PV2 → PV3 |
| `masks/` | `{tên_ảnh_người}.png`, đúng mask đã dùng khi sinh ảnh | PV1 → PV3 |
| `external/` | kết quả cho ảnh ngoài dataset | PV1 → PV3 |
| `survey/` | ảnh bộ khảo sát `q01.png`… (`eval/make_survey_set.py`) | PV3 |

Ảnh: RGB, 768×1024, PNG. Mask: 'L', 0/255, 255 = vùng áo. Nhật ký chạy nằm ở `results/` (được commit).
