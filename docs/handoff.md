# Bàn giao giữa các phân việc: cái gì qua Git, cái gì qua Drive

Chi tiết từng việc nằm ở `docs/PHAN_VIEC_v2.md`. Tài liệu này trả lời ba câu hỏi: **file của tôi để ở đâu, người khác lấy ở đâu, và tôi kiểm tra mình đã có đủ input chưa.**

## 1. Hai đường bàn giao

| Đường | Dùng cho | Lý do |
|---|---|---|
| **Git** (Pull Request) | Code, CSV, bảng kết quả, biểu đồ, tài liệu | File nhỏ, cần lịch sử và người duyệt |
| **Google Drive chung** | Mọi **ảnh** (dữ liệu VITON-HD, ảnh kết quả, cache, ảnh ngoài dataset) | Nặng (hàng trăm ảnh PNG); repo này **công khai** nên không đưa ảnh dataset (chỉ dùng cho nghiên cứu) và ảnh khuôn mặt thật lên GitHub |

`.gitignore` đã chặn ảnh trong `data/raw`, `data/processed`, `outputs/` và `results/failure_cases/`, nên lỡ tay `git add .` cũng không đẩy nhầm ảnh lên.

## 2. Thiết lập một lần

**a) Mọi người**
```bash
git clone https://github.com/chauthao1603-dotcom/AI-Business-Project.git
cd AI-Business-Project
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                             # phải đạt hết
```

**b) Một người tạo thư mục Drive chung** `tryon_project/` có đúng cấu trúc dưới đây, chia sẻ quyền *Người chỉnh sửa* cho cả nhóm, rồi dán link vào mục 6.
```
tryon_project/
├── data/raw/                  # VITON-HD: test/{image,cloth,agnostic-mask}/, test_pairs.txt
├── data/processed/
│   ├── garments/  demo/persons/  demo/garments/
│   ├── external/persons/  external/garments/
│   └── cache/
├── outputs/                   # catvton/  baseline/  masks/  external/  survey/
└── results/failure_cases/
```

**c) Lấy file từ Drive về máy**
- *Laptop:* tải thư mục cần dùng từ Drive và đặt **đúng đường dẫn tương đối** trong repo (ví dụ ảnh CatVTON vào `outputs/catvton/`). Không đổi tên thư mục.
- *Colab (PV1, PV3 khi chạy nặng):*
```python
from google.colab import drive
drive.mount('/content/drive')
%env TRYON_ROOT=/content/drive/MyDrive/tryon_project
!git clone https://github.com/chauthao1603-dotcom/AI-Business-Project.git
%cd AI-Business-Project
!python -m scripts.check_inputs PV1
```
`TRYON_ROOT` cho `src/config.py` biết nơi chứa ảnh và `outputs/`; code và file CSV vẫn lấy từ bản clone.

## 3. Ai giao gì, qua đâu, cho ai

| Ai giao | Giao gì | Đường dẫn | Qua | Cho ai |
|---|---|---|---|---|
| PV2 | Báo cáo kiểm tra dữ liệu (kích thước, chiều mask) | `docs/dataset_report.txt` | Git | PV1, PV3 |
| PV2 | Dữ liệu VITON-HD (thư mục test) | `data/raw/` | Drive | PV1, PV3 |
| PV2 | Danh sách 200 cặp đánh giá | `data/processed/eval_pairs.csv` | Git | PV1, PV3 |
| PV2 | Danh mục sản phẩm và ảnh áo | `data/processed/catalog.csv` (Git), `data/processed/garments/` (Drive) | Git + Drive | PV4, PV5 |
| PV2 | Bộ demo (6 người, 10 áo) | `data/processed/demo/` | Drive | PV1 |
| PV2 | Data card (nguồn, giấy phép, mô tả cột) | `docs/data_card.md` | Git | Báo cáo |
| PV1 | `run_tryon` bản đầu | `src/tryon.py` | Git | Cả nhóm, PV4 |
| PV1 | Ảnh mẫu đầu tiên (3 ảnh) | `outputs/catvton/` | Drive | Cả nhóm |
| PV1 | Cache demo (60 cặp) | `data/processed/cache/` | Drive | PV2, PV4 |
| PV1 | 200 ảnh CatVTON và mask đã dùng | `outputs/catvton/`, `outputs/masks/` | Drive | PV3 |
| PV1 | Nhật ký thời gian và lỗi | `results/timing_log_catvton.csv`, `results/errors_catvton.csv` | Git | PV3 |
| PV2 | `run_baseline` bản đầu | `src/baseline.py` | Git | PV4 |
| PV2 | 200 ảnh baseline | `outputs/baseline/` | Drive | PV3 |
| PV3 | Bảng chỉ số và biểu đồ | `results/results_summary.csv`, `results/figs/` | Git | PV4, báo cáo |
| PV3 | Kết quả khảo sát | `results/survey_summary.csv` | Git | Báo cáo |
| PV4 | App chạy được | `app.py` | Git | PV1 (đưa lên Colab), PV5 |
| PV5 | Ảnh ngoài dataset | `data/processed/external/` (Drive), `data/processed/external_pairs.csv` (Git) | Git + Drive | PV1 |
| PV1 | Ảnh CatVTON ngoài dataset | `outputs/external/` | Drive | PV3 |

## 4. Giao một file: làm theo thứ tự

1. **Qua Git:** tạo nhánh `feat/pvX-<việc>` → commit → push → mở Pull Request (dùng mẫu có sẵn, gắn `Closes #<số issue>`) → một bạn duyệt → merge. Người nhận chạy `git pull` sau khi merge.
2. **Qua Drive:** upload đúng đường dẫn như bảng trên (đừng để trong thư mục tạm).
3. **Báo trong nhóm chat:** tên file, vị trí, cách mở thử, bước tiếp theo của người nhận.
4. **Người nhận kiểm tra:** `python -m scripts.check_inputs PV3` (thay PV3 bằng vai trò của mình). Dấu ✓ là đủ, ✗ là còn thiếu, ○ là chưa bắt buộc.

## 5. Lưu ý về triển khai demo

README mẫu của môn học nhắc deploy lên Streamlit Community Cloud, nhưng CatVTON cần GPU còn Streamlit Cloud thì không có. Demo của dự án chạy qua **Colab + đường link công khai** (`cloudflared`, xem mục 1.6 trong `docs/PHAN_VIEC_v2.md`). Cũng không nên deploy bản chỉ hiển thị ảnh sinh sẵn lên Streamlit Cloud, vì ảnh cache không được đưa lên repo công khai. Nhớ chỉnh dòng "Demo" ở đầu `README.md` cho khớp.

## 6. Liên kết của nhóm (điền)

- Drive chung: <dán link>
- Nhóm chat: <dán link>
- Bảng lỗi kiểm thử (PV5): <dán link Google Sheet>
