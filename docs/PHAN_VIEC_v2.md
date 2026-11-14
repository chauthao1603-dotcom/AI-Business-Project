# PHÂN VIỆC KỸ THUẬT – VIRTUAL TRY-ON (bản 2, chi tiết)

> Bản đã chuẩn hóa đường dẫn theo cấu trúc repo `AI-Business-Project` (xem `docs/handoff.md`). Mọi lệnh chạy từ thư mục gốc repo; script chạy bằng `python -m scripts.<tên>` hoặc `python -m eval.<tên>`.

Mô hình: **CatVTON**, chạy trên Google Colab (GPU T4). Phạm vi tài liệu này: chỉ phần để ứng dụng chạy được (không gồm báo cáo, slide, video).

Quy ước gọi tên: PV1 = Kỹ thuật mô hình, PV2 = Dữ liệu và baseline, PV3 = Đánh giá, PV4 = Giao diện Streamlit, PV5 = Tích hợp và ảnh ngoài dataset (việc trước đây chưa có người nhận, xem PV5 ở mục 2).

---

## 0. Việc cần chốt trong buổi họp đầu tiên

| # | Câu hỏi | Đề xuất |
|---|---|---|
| 1 | Đánh giá baseline và CatVTON bằng mask nào? Hai phương pháp **phải dùng cùng một mask**, nếu không so sánh không công bằng | Dùng `agnostic-mask` có sẵn của VITON-HD cho cả hai khi chạy 200 cặp đánh giá. Khi demo với ảnh upload thì cả hai dùng mask do AutoMasker tạo |
| 2 | Ai nhận PV5 (ảnh ngoài dataset, kiểm thử tích hợp)? | 1 người trong nhóm, hoặc 1 người ở nhóm làm báo cáo/video kiêm nhiệm |
| 3 | Tài khoản Google nào chứa Drive chung và chạy Colab? | 1 tài khoản duy nhất; chỉ 1 người chạy mô hình tại một thời điểm để không tranh GPU |
| 4 | PV1 là điểm nghẽn lớn nhất, ai hỗ trợ? | Nên có 2 người: PV1A làm mục 1.1–1.3, PV1B làm 1.4–1.6 và chạy batch |

---

## 1. Quy ước chung

### 1.1 Cấu trúc thư mục

Repo: https://github.com/chauthao1603-dotcom/AI-Business-Project. Code và file nhỏ đi qua **Git**; ảnh và dữ liệu nặng đi qua **Google Drive chung** với **cùng đường dẫn tương đối**. Ai giao gì cho ai, lấy ở đâu: xem `docs/handoff.md`. Mọi lệnh chạy từ thư mục gốc repo.

```
AI-Business-Project/
├── app.py                  # Streamlit (PV4)
├── requirements*.txt       # requirements.txt (app, baseline, test) | -model (PV1, Colab) | -eval (PV3)
├── src/                    # config.py, utils.py (chung) | tryon.py (PV1) | baseline.py (PV2) | tryon_mock.py (PV4)
├── scripts/                # run_batch.py, make_cache.py (PV1) | check_dataset.py, make_eval_pairs.py,
│                           # run_baseline_batch.py (PV2) | check_inputs.py (kiểm tra input theo vai trò)
├── eval/                   # PV3: run_eval.py, make_survey_set.py, analyze_survey.py, make_report_assets.py
├── notebooks/              # PV1: 01_setup.ipynb (Colab)
├── data/raw/               # [Drive] VITON-HD: test/{image,cloth,agnostic-mask}, test_pairs.txt
├── data/processed/         # [Git] eval_pairs.csv, catalog.csv, ...csv
│                           # [Drive] garments/, demo/, external/, cache/ (ảnh)
├── outputs/                # [Drive] catvton/, baseline/, masks/, external/, survey/
├── results/                # [Git] results_*.csv, survey_*.csv, figs/ | [Drive] failure_cases/ (ảnh)
├── docs/                   # PHAN_VIEC_v2.md, handoff.md, dataset_report.txt, metrics_definition.md, README_RUN.md
└── tests/                  # test_contract.py: chữ ký hàm, chuẩn ảnh/mask (chạy `pytest -q`)
```

- **Không đưa lên GitHub:** ảnh từ VITON-HD (dữ liệu chỉ dùng cho nghiên cứu, repo công khai), ảnh khuôn mặt thật của thành viên, ảnh kết quả nặng. `.gitignore` đã chặn các ảnh này; chỉ cần để đúng thư mục.
- Đường dẫn luôn lấy từ `src/config.py`. Trên Colab đặt `TRYON_ROOT` = thư mục Drive chung (có cùng cấu trúc `data/` và `outputs/`); trên laptop để mặc định và tải thư mục Drive về đúng vị trí trong repo.
- PV4 làm trên laptop không GPU: `USE_MOCK=1` và bản mẫu `data/processed/catalog.sample.csv` là đủ để dựng giao diện.

### 1.2 Định danh và tên file

- `stem` = tên file bỏ đuôi, ví dụ `00006_00`.
- Ảnh kết quả: `{person_stem}__{garment_stem}.png` (hai dấu gạch dưới), ví dụ `00006_00__00012_00.png`. Cặp paired có hai stem giống nhau: `00006_00__00006_00.png`.
- `id` của một cặp = phần tên file bỏ `.png`, ví dụ `00006_00__00012_00`. Mọi file CSV dùng cột `id` này để nối với nhau.
- Mask: `outputs/masks/{person_stem}.png` (mask đúng bản đã dùng để sinh ảnh).
- Cache demo: `data/processed/cache/{id}.png` và `data/processed/cache/{id}_mask.png`.
- Ảnh ngoài dataset: `ext_p01`, `ext_p02`… (người), `ext_g01`… (áo).
- Tên file không dấu tiếng Việt, không dấu cách. Ảnh kết quả luôn là **PNG**.

### 1.3 Chuẩn ảnh và mask

- Ảnh trong code: `PIL.Image` chế độ `RGB`, kích thước `(768, 1024)` (rộng × cao, nên `img.size == (768, 1024)`).
- Ảnh đầu vào bất kỳ được đưa về chuẩn này bằng cùng một hàm (`src.utils.standardize_image`, chạy không cần GPU; riêng khâu nạp vào mô hình CatVTON, PV1 có thể dùng `resize_and_padding` của repo CatVTON).
- Mask: `PIL.Image` chế độ `L`, `(768, 1024)`, chỉ gồm 0 và 255. **255 (trắng) = vùng áo sẽ bị thay, 0 (đen) = giữ nguyên.**
- `agnostic-mask` của VITON-HD phải được PV2 kiểm tra xem vùng áo là trắng hay đen và đảo lại nếu cần, trước khi ai dùng.

### 1.4 Hai hàm lõi (chữ ký đầy đủ)

```python
# src/tryon.py   (PV1)
def run_tryon(person_img: Image.Image,
              garment_img: Image.Image,
              mask: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    """
    Vào : person_img, garment_img (PIL RGB, kích thước bất kỳ, sẽ được chuẩn hóa)
          mask: None -> tự tạo bằng AutoMasker (loại "upper"); có giá trị -> dùng mask này
    Ra  : (ảnh_kết_quả RGB 768x1024, ảnh_mask L 768x1024, 0/255)
    Quy tắc: không ghi file, không print; lỗi đầu vào thì raise ValueError với thông báo tiếng Việt.
    """

# src/baseline.py   (PV2)
def run_baseline(person_img: Image.Image,
                 garment_img: Image.Image,
                 mask: Image.Image) -> Image.Image:
    """Ra: ảnh RGB 768x1024. Mask theo chuẩn 1.3."""
```

### 1.5 Định dạng các file CSV

```
eval_pairs.csv          id,person,cloth,mode
                        00006_00__00006_00,00006_00,00006_00,paired
                        00006_00__00012_00,00006_00,00012_00,unpaired

catalog.csv             product_id,name,category,color,price_vnd,image_file
                        SP001,Áo thun trơn trắng,Áo thun,Trắng,199000,00012_00.jpg

timing_log_catvton.csv  id,seconds,peak_vram_gb
errors_catvton.csv      id,error   (baseline: errors_baseline.csv)

results_per_pair.csv    id,mode,method,region,mse,rmse,mae,ssim,lpips
results_summary.csv     method,mode,region,metric,mean,std,n
```

- `mode` ∈ {`paired`, `unpaired`, `external`}. `method` ∈ {`baseline`, `catvton`}. `region` ∈ {`full`, `mask`}.
- Paired = người mặc lại **chính chiếc áo gốc** (có ảnh đáp án để so). Unpaired = người mặc áo khác (không có đáp án).

### 1.6 Tham số cố định

| Mục | Giá trị |
|---|---|
| Sinh ảnh | 50 bước, guidance 2.5, seed 42, kích thước 1024×768 |
| Chọn mẫu ngẫu nhiên (eval_pairs, bộ khảo sát) | seed 42 |
| MSE, RMSE, MAE | tính trên thang 0–255; RMSE = √MSE; càng thấp càng tốt |
| SSIM | `data_range=255`; càng cao càng tốt |
| LPIPS | mạng `alex`; ảnh đưa vào đổi về [-1, 1]; càng thấp càng tốt |
| FID, KID | so tập ảnh sinh với tập ảnh thật trong `data/raw/test/image`; càng thấp càng tốt |

### 1.7 Cách làm việc chung

| File / thư mục | Chủ sở hữu (chỉ người này sửa) |
|---|---|
| `src/tryon.py`, `scripts/run_batch.py`, `scripts/make_cache.py`, `notebooks/`, `requirements-model.txt` | PV1 |
| `src/baseline.py`, `scripts/check_dataset.py`, `scripts/make_eval_pairs.py`, `scripts/run_baseline_batch.py`, `data/processed/*.csv` (trừ `external_pairs.csv`) | PV2 |
| `eval/`, `results/` (riêng `results/timing_log_*.csv`, `results/errors_*.csv`: PV1 và PV2 ghi vào), `requirements-eval.txt`, `docs/metrics_definition.md` | PV3 |
| `app.py`, `src/tryon_mock.py` | PV4 |
| `data/processed/external_pairs.csv` | PV5 |
| `src/config.py`, `src/utils.py`, `tests/`, `.gitignore`, `requirements.txt` | Chung: sửa qua Pull Request và báo trong nhóm chat |

- Chuyển giữa hàm thật và hàm giả bằng biến môi trường `USE_MOCK=1` (đọc trong `src/config.py`). `app.py` lấy hàm qua `src.config.get_run_tryon()`, không import trực tiếp `src/tryon.py`.
- **Mỗi việc (1.1, 2.3, …) là một Issue** (mẫu "Công việc" trong `.github/ISSUE_TEMPLATE`), làm trên **một nhánh** `feat/pv1-run-tryon`, `feat/pv2-baseline`, `docs/...`, rồi mở Pull Request, ≥ 1 người duyệt (quy ước trong README mục 5).
- Chữ ký hàm ở mục 1.4 được `tests/test_contract.py` canh giữ: nếu phải đổi, báo cả nhóm và sửa test cùng lúc.
- Commit message bắt đầu bằng động từ, ví dụ `add run_tryon with AutoMasker`.
- Mỗi lần giao file (đặc biệt qua Drive), ghi vào nhóm chat: tên file, vị trí, cách mở thử. Kiểm tra mình đã có đủ input chưa: `python -m scripts.check_inputs PV3`.

---

## 2. Phân việc chi tiết

Mỗi việc có: **Cần trước** (phải có gì mới làm được), **Làm gì**, **Output**, **Xong khi** (tiêu chí để người khác tin là đã xong).

### PV1 – Kỹ thuật mô hình (Colab)

**1.1 Cài đặt CatVTON trên Colab**
- Cần trước: không.
- Làm gì: Runtime → GPU (T4). Chạy lần lượt: `from google.colab import drive; drive.mount('/content/drive')`; `!git clone https://github.com/Zheng-Chong/CatVTON.git`; `%cd CatVTON`; `!pip install -r requirements.txt`. Kiểm tra `torch.cuda.is_available()` trả về `True`. Repo yêu cầu Python 3.9 nên Colab (bản Python mới hơn) có thể báo xung đột thư viện: ghi nguyên văn từng lỗi. **Nếu sau khoảng 4 giờ vẫn không chạy được, chuyển sang phương án B:** gọi Space demo của tác giả (link trên trang dự án CatVTON) bằng `gradio_client`, và báo cả nhóm.
- Output: `notebooks/01_setup.ipynb` và `requirements-model.txt` (các gói đã chạy được trên Colab).
- Xong khi: Restart runtime rồi chạy lại từ đầu notebook, không lỗi.

**1.2 Chạy thử một ảnh**
- Cần trước: 1.1; 1 ảnh người và 1 ảnh áo trong `data/raw/test` (lấy tạm 2 file bất kỳ nếu PV2 chưa xong).
- Làm gì: chạy giao diện Gradio hoặc `inference.py` có sẵn của repo theo README, đưa ảnh vào, lấy ảnh kết quả.
- Output: 1 ảnh try-on đầu tiên, gửi vào nhóm.
- Xong khi: nhìn thấy đúng người mặc đúng áo (màu, họa tiết).

**1.3 Viết `src/tryon.py` với `run_tryon`**
- Cần trước: 1.2.
- Làm gì: copy cấu trúc từ `app.py` của repo CatVTON (đối chiếu tên lớp thật trong repo, đừng tự đoán). Các bước trong hàm:
  1. `load_models()` chỉ chạy **một lần** (biến toàn cục hoặc `functools.lru_cache`), nạp pipeline và AutoMasker lên GPU.
  2. Đưa `person_img` và `garment_img` về `(768, 1024)` bằng `resize_and_padding`.
  3. Nếu `mask is None`: `mask = automasker(person_img, "upper")['mask']`.
  4. Gọi `pipeline(image=person_img, condition_image=garment_img, mask=mask, num_inference_steps=50, guidance_scale=2.5, seed=42, height=1024, width=768)[0]`.
  5. Trả về `(ảnh_kết_quả, mask)` theo chuẩn 1.3.
  - Thêm khối `if __name__ == "__main__":` nhận `--person`, `--garment`, `--out` để chạy thử bằng dòng lệnh.
- Output: `src/tryon.py`.
- Xong khi: chạy 3 cặp khác nhau cho 3 ảnh `768×1024`, mask chỉ có giá trị 0/255, màn hình in ra giây/ảnh và `torch.cuda.max_memory_allocated()` (số này dùng cho báo cáo). Gửi 3 ảnh mẫu và `tryon.py` cho cả nhóm.

**1.4 Viết `scripts/run_batch.py`**
- Cần trước: 1.3 và `eval_pairs.csv` của PV2.
- Làm gì: dòng lệnh
  `python -m scripts.run_batch --pairs data/processed/eval_pairs.csv --person_dir data/raw/test/image --cloth_dir data/raw/test/cloth --out outputs/catvton --masks outputs/masks --mask_source dataset`
  (`--mask_source dataset` = đọc `data/raw/test/agnostic-mask/{person}.png` bằng `src.utils.load_dataset_mask` rồi truyền vào `run_tryon(..., mask=...)`; `auto` = để `run_tryon` tự tạo). Với mỗi dòng CSV: nếu file `{out}/{id}.png` đã tồn tại thì bỏ qua (để chạy tiếp được khi Colab ngắt); chạy `run_tryon`; lưu ảnh và mask; ghi `results/timing_log_catvton.csv`; nếu lỗi thì ghi `results/errors_catvton.csv` rồi chạy tiếp, không dừng cả batch.
- Output: 200 ảnh trong `outputs/catvton/`, `outputs/masks/`, `results/timing_log_catvton.csv`, `results/errors_catvton.csv`.
- Xong khi: số ảnh = số dòng CSV trừ số dòng trong `results/errors_catvton.csv`; mở ngẫu nhiên 10 ảnh thấy bình thường. Thời gian ước tính = 200 × giây/ảnh (khoảng 100–200 phút nếu 30–60 giây/ảnh), nên chia thành nhiều phiên.

**1.5 Sinh sẵn cache cho bộ demo**
- Cần trước: 1.3 và bộ demo của PV2 (6 người × 10 áo).
- Làm gì: `python -m scripts.make_cache --persons data/processed/demo/persons --garments data/processed/demo/garments --out data/processed/cache` chạy mọi tổ hợp (60 cặp), dùng `mask=None`.
- Output: `data/processed/cache/{id}.png` và `{id}_mask.png` cho 60 cặp.
- Xong khi: đủ 60 cặp, giao cho PV2 xem và lọc cặp lỗi (xem 2.5).

**1.6 Chạy Streamlit trên Colab**
- Cần trước: `app.py` bản chạy được của PV4.
- Làm gì: cài `streamlit`; chạy `streamlit run app.py` ở nền; mở ra ngoài bằng `cloudflared tunnel --url http://localhost:8501` (hoặc `ngrok`).
- Output: `docs/README_RUN.md` ghi từng bước chạy từ đầu và cách lấy link.
- Xong khi: người khác làm theo README mở được app qua link mà không cần hỏi lại.

**1.7 Chạy ảnh ngoài dataset** (khi PV5 giao): dùng lại `scripts/run_batch.py` với `--pairs data/processed/external_pairs.csv --person_dir data/processed/external/persons --cloth_dir data/processed/external/garments --out outputs/external --mask_source auto`.

### PV2 – Dữ liệu và baseline

**2.1 Tải và đặt dữ liệu**
- Cần trước: không.
- Làm gì: tải VITON-HD từ link Google Drive trong README của repo VITON-HD, chỉ cần phần `test` và `test_pairs.txt`. Đọc điều khoản sử dụng dữ liệu trong README và ghi nguồn. Đặt vào `data/raw/test/` và `data/raw/test_pairs.txt`.
- Output: `data/raw/test/image`, `cloth`, `agnostic-mask` và các thư mục còn lại.
- Xong khi: các thư mục đúng tên như quy ước 1.1.

**2.2 Kiểm tra dữ liệu: `scripts/check_dataset.py`**
- Cần trước: 2.1.
- Làm gì: đếm số file trong `image/`, `cloth/`, `agnostic-mask/` (phải bằng nhau); liệt kê các kích thước ảnh gặp (mong đợi `768×1024`); thử mở từng file để tìm file hỏng; lấy 5 mask và ghi vùng áo là **trắng hay đen**; mở `test_pairs.txt` xem mỗi dòng gồm gì.
- Output: `docs/dataset_report.txt`.
- Xong khi: report ghi rõ 3 điều: số file khớp, kích thước thật sự, chiều của mask. Gửi cho PV1, PV3 vì cả hai cần.

**2.3 Tạo `eval_pairs.csv`: `scripts/make_eval_pairs.py`**
- Cần trước: 2.2.
- Làm gì: seed 42. **Paired:** chọn ngẫu nhiên 100 `person` trong `data/raw/test/image`, `cloth` = cùng stem. **Unpaired:** chọn ngẫu nhiên 100 dòng từ `test_pairs.txt` sao cho stem người ≠ stem áo. Gán `id = {person}__{cloth}`. Kiểm tra mọi file đều tồn tại.
- Output: `data/processed/eval_pairs.csv` (200 dòng, schema như 1.5).
- Xong khi: không dòng nào trỏ tới file không tồn tại; giao cho PV1 và PV3 ngay.

**2.4 Tạo `catalog.csv` và thư mục `garments/`**
- Cần trước: 2.1.
- Làm gì: chọn 20 ảnh áo trong `data/raw/test/cloth/` (đa dạng loại, màu, họa tiết), copy sang `data/processed/garments/` giữ nguyên tên. Xem từng ảnh để đặt tên Việt, phân loại (ví dụ Áo thun, Áo sơ mi, Áo len), gán màu bằng tên tiếng Việt, giá giả định. Đây là dữ liệu mô phỏng, báo cáo cần ghi rõ.
- Output: `data/processed/catalog.csv`, `data/processed/garments/`.
- Xong khi: mọi `image_file` trong CSV đều tồn tại. Giao cho PV4 (dùng thay file mẫu) và PV5.

**2.5 Bộ demo**
- Cần trước: 2.4 và `scripts/make_cache.py` của PV1 chạy xong (1.5).
- Làm gì: chọn 6 ảnh người và 10 áo (có thể lấy từ 20 áo ở 2.4) copy vào `data/processed/demo/persons` và `data/processed/demo/garments`. Giao PV1 sinh 60 cặp, xem từng ảnh trong `data/processed/cache/`, ghi `data/processed/demo_review.csv` (`id,ok,ghi_chu`); loại cặp lỗi (áo méo, mất tay, mặt lỗi…).
- Output: `data/processed/demo/`, `data/processed/demo_review.csv`.
- Xong khi: còn ít nhất 4 người × 8 áo cặp "ok" để quay demo.

**2.6 Viết `src/baseline.py`**
- Cần trước: 2.2 (biết chiều mask).
- Làm gì: hàm `run_baseline(person_img, garment_img, mask)`:
  1. Chuẩn hóa ảnh về `768×1024`.
  2. Tìm hình chữ nhật bao quanh mask: `ys, xs = np.where(mask_array == 255)` rồi lấy min/max.
  3. Xóa nền trắng của áo: điểm ảnh có cả ba kênh > 240 thì coi là nền, cắt khung theo phần còn lại.
  4. Resize phần áo vừa khớp hình chữ nhật ở bước 2.
  5. Dán đè lên ảnh người, chỉ dán ở điểm ảnh không phải nền; (tùy chọn) làm mờ viền vài điểm ảnh.
- Output: `src/baseline.py`.
- Xong khi: chạy thử 10 cặp, ảnh baseline có áo nằm đúng vùng ngực–thân, không tràn ra nền; gửi 3 ảnh mẫu cho nhóm.

**2.7 Chạy baseline hàng loạt: `scripts/run_baseline_batch.py`**
- Cần trước: 2.3, 2.6.
- Làm gì: giống `scripts/run_batch.py` về tham số (`--pairs --person_dir --cloth_dir --out outputs/baseline --mask_source dataset`), cùng quy tắc bỏ qua file đã có; lỗi ghi vào `results/errors_baseline.csv`.
- Output: 200 ảnh trong `outputs/baseline/`.
- Xong khi: số ảnh = số dòng CSV, tên file đúng quy ước 1.2.

**2.8 `remove_bg(garment_img) -> Image` (tùy chọn)**
- Làm gì: dùng thư viện `rembg` để xóa nền, ghép lên nền trắng, trả `RGB`. Dùng cho ảnh áo upload có nền lộn xộn.
- Xong khi: thử 3 ảnh áo có nền màu và nền họa tiết, kết quả là áo trên nền trắng sạch.

### PV3 – Đánh giá mô hình

**3.1 Trang định nghĩa chỉ số: `docs/metrics_definition.md`**
- Cần trước: file Chapter 2.
- Làm gì: với mỗi chỉ số (MSE, RMSE, MAE, SSIM, LPIPS, FID, KID, ma trận nhầm lẫn, accuracy, precision, recall, F1): công thức, cách đọc (cao hay thấp là tốt), dùng ở đâu trong dự án. Đánh dấu rõ chỉ số nào có trong Chapter 2 (để báo cáo dẫn đúng chương), chỉ số nào là chuẩn ngành thử đồ ảo.
- Xong khi: thang tính khớp bảng 1.6.

**3.2 Viết `eval/run_eval.py`**
- Cần trước: không cần chờ output thật, nhưng phải có `eval_pairs.csv` (2.3) để biết cột.
- Làm gì: dòng lệnh
  `python -m eval.run_eval --pairs data/processed/eval_pairs.csv --gt data/raw/test/image --pred outputs/catvton --method catvton --masks outputs/masks`
  - **Paired:** với mỗi dòng, so ảnh sinh với ảnh thật `data/raw/test/image/{person}.jpg`. Tính MSE, RMSE, MAE (tự tính bằng numpy), SSIM (`skimage.metrics.structural_similarity(..., channel_axis=2, data_range=255)`), LPIPS (`lpips.LPIPS(net='alex')`). Tính **hai phiên bản**: `region=full` (toàn ảnh) và `region=mask` (chỉ vùng mask: MSE/RMSE/MAE trên điểm ảnh trong mask; SSIM/LPIPS trên khung bao quanh mask). Lý do: phần ngoài mask của ảnh sinh giống hệt ảnh gốc, nên số trên toàn ảnh sẽ cao ảo, cần số vùng mask để thấy sự khác biệt thật.
  - **Unpaired:** tính FID và KID giữa tập ảnh sinh và tập ảnh thật trong `data/raw/test/image`. Với khoảng 100 ảnh thì FID bị lệch, chỉ dùng để so tương đối giữa baseline và CatVTON trên cùng mẫu; KID cần đặt `subset_size` không lớn hơn số ảnh.
  - Chạy lần lượt cho `--method baseline` và `--method catvton`.
- Output: `results/results_per_pair.csv`, `results/results_summary.csv` (schema như 1.5).
- Xong khi qua hai phép thử: (a) cho `pred = gt` thì MSE = 0, SSIM = 1, LPIPS ≈ 0; (b) cho `pred` = ảnh đen toàn bộ thì cả ba chỉ số đều xấu rõ rệt. Cả hai đạt thì mới chạy với dữ liệu thật.

**3.3 Bộ khảo sát: `eval/make_survey_set.py`**
- Cần trước: `outputs/catvton/` đủ cả cặp unpaired và paired (từ PV1).
- Làm gì: seed 42. Chọn 10 ảnh thật (từ `data/raw/test/image`) và 10 ảnh CatVTON (lấy ngẫu nhiên, **không chọn lọc ảnh đẹp**), đưa về cùng định dạng và kích thước để người xem không đoán qua định dạng. Đổi tên `q01.png`…`q20.png`, xáo thứ tự, lưu đáp án vào `results/survey_key.csv` (`qid,label,source_id`, label ∈ {real, ai}). Phần so sánh (đổi theo quyết định của nhóm ngày 14/11: so baseline với CatVTON quá dễ nhận ra): 10 cặp ảnh chụp thật và ảnh CatVTON paired của cùng người, cùng áo, không trùng người/áo với phần trên; ảnh thật bên trái ở đúng 5 câu; câu hỏi "Ảnh nào là ảnh chụp thật?"; đáp án lưu `results/ab_key.csv`.
- Output: ảnh khảo sát trong `outputs/survey/` (để trên Drive), `results/survey_key.csv`, `results/ab_key.csv`. PV3 tự dựng Google Form từ đó.
- Xong khi: một người ngoài nhóm xem thử không đoán được qua tên file hay định dạng.

**3.4 Phân tích khảo sát: `eval/analyze_survey.py`**
- Cần trước: file trả lời Form dạng CSV, `results/survey_key.csv`, `results/ab_key.csv`.
- Làm gì: lớp dương = "AI". Mỗi lượt trả lời tính vào một trong bốn ô: TP (ảnh AI, chọn "AI"), FN (ảnh AI, chọn "Thật"), FP (ảnh thật, chọn "AI"), TN (ảnh thật, chọn "Thật"). Tính accuracy = (TP+TN)/tổng, precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2·P·R/(P+R). Accuracy gần 50% nghĩa là người xem khó phân biệt, tức ảnh AI thuyết phục. Phần A/B: tính tỷ lệ chọn đúng ảnh chụp thật, khoảng tin cậy 95% và p-value so với 50% (gần 50% = không phân biệt được).
- Output: `results/survey_summary.csv`, `results/figs/confusion_matrix.png`.
- Xong khi: tổng TP+FN+FP+TN = số người × 20.

**3.5 Biểu đồ và ca lỗi: `eval/make_report_assets.py`**
- Cần trước: 3.2 chạy xong thật.
- Làm gì: vẽ biểu đồ cột so sánh baseline và CatVTON cho từng chỉ số, lưu `results/figs/`; chọn 5 cặp có LPIPS tệ nhất của CatVTON, copy ảnh gốc, baseline, CatVTON vào `results/failure_cases/{id}/`.
- Output: `results/figs/*.png`, `results/failure_cases/`.

**3.6 Nhận xét ảnh ngoài dataset**
- Cần trước: `outputs/external/` từ PV1 (1.7).
- Làm gì: không có ảnh đáp án nên không tính chỉ số; lập bảng định tính (`id`, kết quả tốt/chấp nhận được/lỗi, loại lỗi: sai tay áo, mất họa tiết, méo cơ thể…).
- Output: `results/external_notes.csv`.

### PV4 – Giao diện Streamlit

**4.1 Khung app với hàm giả**
- Cần trước: không.
- Làm gì: cài `streamlit`; dùng `src/config.py` và `src/tryon_mock.py` đã có sẵn trong repo (hàm giả phủ màu áo lên vùng thân và trả mask hình chữ nhật, đúng chuẩn 1.3; chỉnh nếu cần); `app.py` với `st.set_page_config(layout="wide")` và `st.tabs(["Cửa hàng", "Thử đồ", "Kết quả", "Đánh giá"])`. Chạy bằng `USE_MOCK=1 streamlit run app.py`.
- Output: `app.py` (nhánh `feat/pv4-skeleton`).
- Xong khi: 4 tab hiện ra, bấm nút Thử đồ trả về ảnh giả.

**4.2 Tab Cửa hàng**
- Cần trước: `catalog.csv` (dùng file mẫu 3 dòng tới khi PV2 giao 2.4).
- Làm gì: đọc CSV bằng `pandas`, hiển thị lưới 4 cột (`st.columns(4)`) gồm ảnh, tên, danh mục, màu, giá; nút "Chọn thử" lưu `st.session_state["product_id"]`.
- Xong khi: đổi file CSV thì lưới đổi theo, không sửa code.

**4.3 Tab Thử đồ**
- Cần trước: 4.1.
- Làm gì: `st.file_uploader` cho ảnh người (jpg/png); nguồn ảnh áo gồm hai lựa chọn: lấy từ catalog (đang chọn ở 4.2) hoặc upload; kiểm tra đầu vào (đúng định dạng, kích thước tối thiểu, đã đủ hai ảnh); nút "Thử đồ"; `st.spinner("Đang sinh ảnh, vui lòng chờ…")`; kết quả (`result`, `baseline`, `mask`, `elapsed`) lưu vào `st.session_state`.
- Xong khi: không upload mà bấm nút thì hiện thông báo lỗi tiếng Việt, không văng traceback.

**4.4 Cache kết quả**
- Cần trước: 4.3.
- Làm gì: hàm `get_result(person_img, garment_img)` tính khóa `sha1` từ nội dung hai ảnh; nếu có `data/processed/cache/{khóa hoặc id}.png` thì đọc luôn, không gọi mô hình; ảnh người/áo của bộ demo dùng thẳng `id` có sẵn từ PV1 (1.5). Nạp mô hình bằng `@st.cache_resource` (chỉ nạp một lần cho cả phiên).
- Xong khi: cặp đã chạy rồi thì lần sau hiện kết quả trong chưa đầy 1 giây.

**4.5 Tab Kết quả**
- Cần trước: 4.3, 4.4.
- Làm gì: ba cột Ảnh gốc | Baseline | CatVTON; `st.expander("Xem mask vùng áo")` hiện ảnh mask; khung thông tin sản phẩm (từ `catalog.csv`) nếu áo chọn từ catalog; `st.selectbox("Thử áo khác")` để đổi áo mà **giữ nguyên ảnh người**.
- Xong khi: chạy được đủ 4 bước demo (chọn ảnh → xử lý → xem kết quả → đổi áo) mà không cần tải lại trang.

**4.6 Tab Đánh giá**
- Cần trước: `results/results_summary.csv` và `results/figs/` của PV3 (3.2, 3.5).
- Làm gì: đọc CSV hiển thị bảng, nhúng biểu đồ; nếu chưa có file thì hiện "Chưa có kết quả đánh giá" thay vì lỗi.
- Xong khi: thay file CSV thì bảng đổi theo.

**4.7 Ghép hàm thật**
- Cần trước: `src/tryon.py` (1.3), `src/baseline.py` (2.6).
- Làm gì: chỉ đổi `USE_MOCK=0`; nếu phải sửa `app.py` nhiều hơn vài dòng thì chữ ký hàm chưa đúng chuẩn 1.4, báo PV1/PV2 sửa.
- Xong khi: chạy được toàn bộ quy trình trên Colab qua đường link (cùng PV1 ở 1.6).

**4.8 Xử lý lỗi và hoàn thiện**
- Làm gì: bắt `ValueError` từ `run_tryon` và hiện thông báo; xử lý ảnh quá lớn, sai định dạng, GPU hết bộ nhớ; thống nhất nhãn tiếng Việt, bố cục các tab.
- Xong khi: PV5 (4 mục kiểm thử ở 5.2) không còn lỗi mức "văng ứng dụng".

### PV5 – Ảnh ngoài dataset và kiểm thử tích hợp (đề xuất, chưa có chủ)

**5.1 Thu thập ảnh ngoài dataset**
- Làm gì: chụp 10 ảnh người (đứng thẳng, nhìn trực diện, nền đơn giản, có sự đồng ý của người chụp) và 10 ảnh áo trên nền trắng (áo của thành viên). Đặt tên `ext_p01.jpg`, `ext_g01.jpg`; lập `data/processed/external_pairs.csv` (`id,person,cloth,mode=external`).
- Output: `data/processed/external/`, `external_pairs.csv`. Giao cho PV1 (1.7).

**5.2 Kiểm thử tích hợp**
- Làm gì: chạy app qua link Colab, thực hiện đủ 4 bước demo với 5 cặp khác nhau; thử các tình huống sai (không upload, ảnh quá lớn, file `.pdf`); ghi lỗi vào bảng chung (`ai báo`, `màn hình nào`, `làm gì`, `thấy gì`) gửi PV4/PV1.
- Output: bảng lỗi và xác nhận "đạt" cho từng bước.

---

## 3. Bảng bàn giao (bản đã sửa)

| Ai giao | Giao gì | Cho ai | Dùng để |
|---|---|---|---|
| PV2 | `docs/dataset_report.txt` (kích thước, chiều mask) | PV1, PV3 | Biết chuẩn dữ liệu |
| PV2 | `eval_pairs.csv` | PV1, PV3 | Chạy batch, đánh giá |
| PV1 | 3 ảnh mẫu và `tryon.py` bản đầu | Cả nhóm, PV4 | Thay hàm giả |
| PV2 | `catalog.csv`, `garments/` | PV4, PV5 | Tab Cửa hàng |
| PV2 | Bộ demo (6 người, 10 áo) | PV1 | Sinh cache 60 cặp |
| PV1 | `data/processed/cache/` (60 cặp) | PV2, PV4 | Lọc cặp lỗi, chạy demo nhanh |
| PV1 | `outputs/catvton/` (200 ảnh) | PV3 | Chạy đánh giá |
| PV2 | `outputs/baseline/` (200 ảnh) | PV3 | Chạy đánh giá |
| PV3 | `results_summary.csv`, `figs/` | PV4 | Tab Đánh giá |
| PV4 | `app.py` bản chạy được | PV1 | Triển khai trên Colab |
| PV5 | `external/`, `external_pairs.csv` | PV1 | Chạy ảnh ngoài |
| PV1 | `outputs/external/` | PV3 | Nhận xét tổng quát hóa |
| PV5 | Bảng lỗi kiểm thử | PV4, PV1 | Sửa lỗi |

## 4. Thứ tự phụ thuộc (không gắn ngày)

```
PV2: 2.1 → 2.2 → 2.3 (eval_pairs) ─────────────────────────┐
PV2: 2.4 (catalog) ──► PV4: 4.2                            │
PV1: 1.1 → 1.2 → 1.3 (run_tryon) ──► 1.4 (batch) ◄─────────┘
PV2: 2.2 → 2.6 (baseline) → 2.7 (batch baseline)
PV1: 1.4 + PV2: 2.7 ──► PV3: 3.2 chạy thật → 3.5 → PV4: 4.6
PV4: 4.1 → 4.3 → 4.4 → 4.5 (làm sớm với hàm giả) → 4.7 (ghép hàm thật sau 1.3)
```

Việc có thể làm ngay không chờ ai: 1.1, 2.1, 3.1, 4.1. Việc PV3 viết script (3.2) cũng làm sớm được bằng ảnh thử.
