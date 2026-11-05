# Virtual Try-On – Thử quần áo ảo bằng AI

Đồ án cuối kỳ **ELC3008 – Trí tuệ nhân tạo trong kinh doanh** • Nhóm <số> • GVHD: <tên>

> **Demo:** <link demo – chạy qua Colab, xem docs/handoff.md mục 5> • **Video:** <link> • **Báo cáo:** <link>

## 1. Bài toán
Khách mua thời trang online khó hình dung món đồ khi mặc lên người nên do dự, dễ chọn sai kiểu hoặc không vừa ý, dẫn đến tỷ lệ chuyển đổi thấp và tỷ lệ hoàn trả cao. Hệ thống dùng mô hình diffusion **CatVTON** (thử đồ ảo) để sinh ảnh người mặc đúng chiếc áo đang xem, và so sánh với một baseline "dán áo lên người". Chất lượng ảnh sinh được đánh giá bằng MSE/RMSE/MAE, SSIM, LPIPS, FID/KID và khảo sát phân loại "ảnh thật hay AI". Phạm vi: áo phần thân trên, ảnh người đứng thẳng nhìn trực diện (đúng dữ liệu VITON-HD).

## 2. Cấu trúc thư mục
```
├── app.py              # Giao diện Streamlit (PV4)
├── requirements.txt    # Thư viện cho app, baseline, test (laptop không GPU)
├── requirements-model.txt  # Thư viện chạy CatVTON trên Colab (PV1)
├── requirements-eval.txt   # Thư viện tính chỉ số đánh giá (PV3)
├── src/                # Mã nguồn: config.py, utils.py (chung), tryon.py (PV1), baseline.py (PV2), tryon_mock.py (PV4)
├── scripts/            # Chạy hàng loạt: run_batch, make_cache, check_dataset, make_eval_pairs, run_baseline_batch, check_inputs
├── eval/               # Tính chỉ số, khảo sát, biểu đồ (PV3)
├── notebooks/          # Notebook Colab (PV1)
├── data/raw/           # [Drive] Dữ liệu gốc VITON-HD
├── data/processed/     # CSV (Git); ảnh áo, bộ demo, ảnh ngoài dataset, cache (Drive)
├── outputs/            # [Drive] Ảnh kết quả: catvton, baseline, masks, external, survey
├── results/            # Bảng chỉ số, nhật ký, biểu đồ (Git)
├── docs/               # PHAN_VIEC_v2.md, handoff.md, data card, mẫu SUS/UAT
├── tests/              # Unit test (pytest)
└── .github/            # Mẫu issue và pull request
```
Ảnh không đi qua GitHub (repo công khai; VITON-HD chỉ dùng cho nghiên cứu): xem `docs/handoff.md` để biết file nào qua Git, file nào qua Drive, và ai giao cho ai.

## 3. Cài đặt và chạy
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                        # kiểm tra
streamlit run app.py             # mở http://localhost:8501
```
Python khuyến nghị: **3.12** (dùng cùng phiên bản khi deploy).

Laptop không có GPU: chạy với hàm try-on giả để làm giao diện (`USE_MOCK=1`).
```bash
USE_MOCK=1 streamlit run app.py                 # Windows PowerShell: $env:USE_MOCK="1"; streamlit run app.py
python -m scripts.check_inputs PV4              # kiểm tra đã đủ input của vai trò mình chưa (PV1…PV5)
```
Mô hình thật chạy trên Google Colab (GPU T4): xem `docs/PHAN_VIEC_v2.md` mục 1.1 và 1.6.

## 4. Dữ liệu
**VITON-HD** (https://github.com/shadow2496/VITON-HD), chỉ dùng phần `test`, cho mục đích nghiên cứu. Nguồn, giấy phép, mô tả cột: xem `docs/data_card.md` (PV2 điền ở bước 2.1–2.2). Thông tin sản phẩm trong `data/processed/catalog.csv` là dữ liệu mô phỏng do nhóm tự đặt.

## 5. Quy ước làm việc
- Nhánh `main` chỉ nhận code đã review qua Pull Request (≥ 1 người duyệt).
- Mỗi việc một nhánh: `feat/<tên>`, `fix/<tên>`, `docs/<tên>`; gắn với một Issue.
- Commit message bắt đầu bằng động từ: `add astar heuristic`, `fix distance bug`.
- Mọi thí nghiệm cố định `seed`; kết quả lưu ở `results/`.
- Không commit khóa API, mật khẩu, dữ liệu cá nhân. Khóa bí mật đặt trong *Secrets* của Streamlit Cloud.
- Chữ ký hai hàm lõi `run_tryon` và `run_baseline` được `tests/test_contract.py` canh giữ: muốn đổi phải báo cả nhóm.

## 6. Thành viên
| Họ tên | MSSV | Vai trò |
|---|---|---|
| | | |

## 7. Giấy phép và trích dẫn
Dữ liệu và thư viện bên thứ ba thuộc giấy phép của từng nguồn (xem docs/data_card.md).
