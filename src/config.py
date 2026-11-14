"""Đường dẫn và hằng số dùng chung của dự án Virtual Try-On.

Mọi script/app lấy đường dẫn từ đây, KHÔNG tự gõ đường dẫn trong code.

Có hai "gốc" thư mục:
- REPO_ROOT: thư mục repo. Chứa code và các file nhỏ được commit (CSV, results/, docs/).
- ROOT: nơi chứa dữ liệu nặng (data/raw, ảnh trong data/processed, outputs/).
  * Laptop: mặc định ROOT = REPO_ROOT (tải thư mục từ Drive về đúng vị trí trong repo).
  * Colab: đặt biến môi trường TRYON_ROOT = thư mục Drive chung, ví dụ
    /content/drive/MyDrive/tryon_project (cùng cấu trúc data/ và outputs/).
Xem docs/handoff.md.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.getenv("TRYON_ROOT", str(REPO_ROOT))).expanduser().resolve()


def use_mock() -> bool:
    """USE_MOCK=1 -> dùng hàm try-on giả (không cần GPU). Đọc lại biến môi trường mỗi lần gọi."""
    return os.getenv("USE_MOCK", "0") == "1"


USE_MOCK = use_mock()

# ---- Chuẩn ảnh và tham số cố định (docs/PHAN_VIEC_v2.md mục 1.3, 1.6) ----
IMG_W, IMG_H = 768, 1024
IMG_SIZE = (IMG_W, IMG_H)      # PIL dùng (rộng, cao)
MIN_SIDE = 256                 # cạnh ngắn nhất cho phép với ảnh người dùng upload (có thể chỉnh)
SEED = 42
NUM_STEPS = 50
GUIDANCE = 2.5
# Vùng áo của agnostic-mask VITON-HD là màu trắng hay đen? PV2 xác nhận bằng scripts/check_dataset.py
# rồi sửa đúng MỘT chỗ này (True = vùng áo trắng). Mọi code khác đọc mask qua utils.load_dataset_mask().
AGNOSTIC_REGION_IS_WHITE = True

# ---- Dữ liệu nặng: để trên Drive, nằm dưới ROOT, KHÔNG commit ----
RAW = ROOT / "data" / "raw"
DATA = REPO_ROOT / "data"
TEST_DIR = RAW / "test"                       # VITON-HD: image/, cloth/, agnostic-mask/ ...
TEST_IMAGE = TEST_DIR / "image"
TEST_CLOTH = TEST_DIR / "cloth"
TEST_AGNOSTIC_MASK = TEST_DIR / "agnostic-mask"
TEST_PAIRS_TXT = RAW / "test_pairs.txt"

PROCESSED = ROOT / "data" / "processed"
GARMENTS = PROCESSED / "garments"             # ảnh áo của cửa hàng (catalog.csv trỏ tới đây)
DEMO_PERSONS = PROCESSED / "demo" / "persons"
DEMO_GARMENTS = PROCESSED / "demo" / "garments"
EXTERNAL_PERSONS = PROCESSED / "external" / "persons"
EXTERNAL_GARMENTS = PROCESSED / "external" / "garments"
CACHE = PROCESSED / "cache"                   # kết quả sinh sẵn cho demo: {id}.png, {id}_mask.png

OUTPUTS = ROOT / "outputs"
OUT_CATVTON = OUTPUTS / "catvton"             # {id}.png
OUT_BASELINE = OUTPUTS / "baseline"           # {id}.png
OUT_MASKS = OUTPUTS / "masks"                 # {person_stem}.png
OUT_EXTERNAL = OUTPUTS / "external"
OUT_SURVEY = OUTPUTS / "survey"               # ảnh bộ khảo sát q01.png ...

# ---- File nhỏ: nằm trong repo, được commit ----
PROCESSED_CSV_DIR = REPO_ROOT / "data" / "processed"
EVAL_PAIRS = PROCESSED_CSV_DIR / "eval_pairs.csv"
CATALOG = PROCESSED_CSV_DIR / "catalog.csv"
CATALOG_SAMPLE = PROCESSED_CSV_DIR / "catalog.sample.csv"
EXTERNAL_PAIRS = PROCESSED_CSV_DIR / "external_pairs.csv"
DEMO_REVIEW = PROCESSED_CSV_DIR / "demo_review.csv"
RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"

# ---- CatVTON (PV1). Nguồn tham chiếu: CatVTON/app.py và CatVTON/model/pipeline.py ----
CATVTON_DIR = REPO_ROOT / "CatVTON"            # git clone Zheng-Chong/CatVTON vào đây (đã có trong .gitignore)
CATVTON_BASE = "booksforcharlie/stable-diffusion-inpainting"   # mặc định của app.py (bản sao vì runwayml đã xóa repo gốc)
CATVTON_CKPT_REPO = "zhengchong/CatVTON"
CATVTON_CKPT_VERSION = "mix"                   # thư mục mix-48k-1024
# Chỉ tải phần cần dùng thay vì cả repo checkpoint (repo còn chứa nhánh FLUX, mask-free...).
CATVTON_PATTERNS = ["DensePose/*", "SCHP/*", "mix-48k-1024/*"]
# T4 (Colab/Kaggle) không hỗ trợ bf16 tốt -> mặc định fp16. GPU Ampere trở lên (A100, L4, RTX 30/40): đặt TRYON_PRECISION=bf16.
PRECISION = os.getenv("TRYON_PRECISION", "fp16")      # "no" | "fp16" | "bf16"
# Dán lại điểm ảnh gốc ra ngoài mask (như `inference.py --repaint` của CatVTON, cũng là cách bài báo đánh giá).
# Nhờ đó vùng ngoài mask của ảnh CatVTON giống hệt ảnh gốc, giống baseline -> so sánh công bằng.
REPAINT = True

def get_run_tryon():
    """Trả về hàm run_tryon thật hoặc giả tùy USE_MOCK. app.py chỉ import hàm này qua config."""
    if use_mock():
        from src.tryon_mock import run_tryon
    else:
        from src.tryon import run_tryon
    return run_tryon
