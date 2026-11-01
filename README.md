# <Tên đề tài>

Đồ án cuối kỳ **ELC3008 – Trí tuệ nhân tạo trong kinh doanh** • Nhóm <số> • GVHD: <tên>

> **Demo:** <dán link Streamlit Cloud> • **Video:** <link> • **Báo cáo:** <link>

## 1. Bài toán
<2–3 câu: ai gặp vấn đề gì, mô hình AI giúp gì, KPI chính>

## 2. Cấu trúc thư mục
```
├── app.py              # Giao diện Streamlit (điểm vào khi deploy)
├── requirements.txt    # Thư viện cần cài
├── src/                # Mã nguồn mô hình AI (module .py, có test)
├── notebooks/          # Notebook thực nghiệm (Colab)
├── data/raw/           # Dữ liệu gốc (KHÔNG commit dữ liệu nhạy cảm)
├── data/processed/     # Dữ liệu đã làm sạch
├── results/            # Kết quả thí nghiệm (CSV, biểu đồ)
├── docs/               # Tài liệu: data card, mẫu SUS/UAT, wireframe
├── tests/              # Unit test (pytest)
└── .github/            # Mẫu issue và pull request
```

## 3. Cài đặt và chạy
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                        # kiểm tra
streamlit run app.py             # mở http://localhost:8501
```
Python khuyến nghị: **3.12** (dùng cùng phiên bản khi deploy).

## 4. Dữ liệu
Nguồn, giấy phép, mô tả cột: xem `docs/data_card.md` (điền khi làm bước 2.2).

## 5. Quy ước làm việc
- Nhánh `main` chỉ nhận code đã review qua Pull Request (≥ 1 người duyệt).
- Mỗi việc một nhánh: `feat/<tên>`, `fix/<tên>`, `docs/<tên>`; gắn với một Issue.
- Commit message bắt đầu bằng động từ: `add astar heuristic`, `fix distance bug`.
- Mọi thí nghiệm cố định `seed`; kết quả lưu ở `results/`.
- Không commit khóa API, mật khẩu, dữ liệu cá nhân. Khóa bí mật đặt trong *Secrets* của Streamlit Cloud.

## 6. Thành viên
| Họ tên | MSSV | Vai trò |
|---|---|---|
| | | |

## 7. Giấy phép và trích dẫn
Dữ liệu và thư viện bên thứ ba thuộc giấy phép của từng nguồn (xem docs/data_card.md).
