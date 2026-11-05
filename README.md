# Virtual Try On - Thử quần áp bằng AI

Đồ án cuối kỳ **ELC3008 – Trí tuệ nhân tạo trong kinh doanh** • Nhóm <số> • GVHD: <tên>

> **Demo:** <dán link Streamlit Cloud> • **Video:** <link> • **Báo cáo:** <link>

## 1. Bài toán
<2–3 câu: ai gặp vấn đề gì, mô hình AI giúp gì, KPI chính>

## 2. Cấu trúc thư mục
tryon_project/              	# ROOT. Trên Colab: /content/drive/MyDrive/tryon_project
 ├── config.py               	# đọc ROOT từ biến môi trường TRYON_ROOT, có giá trị mặc định
 ├── app.py                  	# giao diện Streamlit (PV4)
 ├── core/
 │   ├── tryon.py            	# run_tryon (PV1)
 │   ├── tryon_mock.py       	# run_tryon giả để PV4 làm việc khi chưa có GPU (PV4)
 │   ├── baseline.py             # run_baseline, remove_bg (PV2)
 │   └── utils.py            	# hàm dùng chung: chuẩn hóa ảnh, đọc/ghi mask
 ├── data/
 │   ├── test/               	# VITON-HD: image/, cloth/, agnostic-mask/ ...
 │   ├── test_pairs.txt
 │   ├── eval_pairs.csv          # PV2 tạo
 │   ├── catalog.csv         	# PV2 tạo
 │   ├── garments/           	# ảnh áo cho cửa hàng
 │   ├── demo/persons/  demo/garments/
 │   ├── external/persons/  external/garments/   # PV5
 │   └── cache/              	# kết quả sinh sẵn cho demo
 ├── outputs/
 │   ├── catvton/  baseline/  masks/  external/
 ├── eval/                   	# PV3
 ├── CatVTON/                	# repo clone, không sửa
 └── requirements.txt

## 3. Cài đặt và chạy
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                        # kiểm tra
streamlit run app.py             # mở http://localhost:8501
```
Python khuyến nghị: **3.12** (dùng cùng phiên bản khi deploy).



