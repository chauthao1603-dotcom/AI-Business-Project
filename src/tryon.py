"""PV1 – hàm try-on thật dùng CatVTON (chạy trên Colab có GPU, môi trường Python 3.9).

Chữ ký cố định (docs/PHAN_VIEC_v2.md mục 1.4): run_tryon(person_img, garment_img, mask=None) -> (ảnh RGB, mask 'L').
app.py, scripts/run_batch.py, scripts/make_cache.py, scripts/smoke_tryon.py và tests/test_contract.py đều dựa vào chữ ký này.

Cách làm theo CatVTON/app.py và CatVTON/inference.py (tên lớp và tham số đã đối chiếu với mã nguồn CatVTON):
  - nạp mô hình MỘT lần (CatVTONPipeline + AutoMasker), tải checkpoint bằng huggingface_hub;
  - chuẩn hóa ảnh bằng src.utils (đệm trắng về 768x1024) để ảnh gốc, baseline và kết quả luôn khớp từng điểm ảnh;
  - mask None -> AutoMasker "upper"; mask có sẵn -> dùng mask đó (255 = vùng áo sẽ thay);
  - mask đưa vào mô hình được làm mờ (GaussianBlur 9) như app.py; mask TRẢ VỀ là mask nhị phân 0/255;
  - sinh ảnh với 50 bước, guidance 2.5, seed 42 (src/config.py); nếu config.REPAINT thì dán lại điểm ảnh gốc
    ngoài mask như `inference.py --repaint`.
Kiểm tra an toàn (safety checker) mặc định TẮT, giống inference.py của tác giả khi đánh giá: bộ kiểm tra thỉnh thoảng
đánh dấu nhầm ảnh bình thường và thay bằng ảnh cảnh báo, làm sai số liệu. Đặt TRYON_SKIP_SAFETY=0 để bật lại.
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
from PIL import Image, ImageFilter

from src import config
from src.utils import standardize_image, standardize_mask, validate_input_image

SKIP_SAFETY = os.getenv("TRYON_SKIP_SAFETY", "1") != "0"
PIPELINE_BLUR = 9            # độ mờ mask đưa vào mô hình (giá trị của CatVTON/app.py)

_LOCK = threading.Lock()
_ENGINE = None


class _Engine:
    """Gói mô hình đã nạp. Tách riêng để kiểm thử được bằng đối tượng giả (không cần GPU)."""

    def __init__(self, pipeline, automasker, torch_module):
        self.pipeline = pipeline
        self.automasker = automasker
        self._torch = torch_module

    def generator(self, seed: int):
        return self._torch.Generator(device="cuda").manual_seed(seed)

    @property
    def oom_error(self):
        return self._torch.cuda.OutOfMemoryError

    def free_cache(self) -> None:
        self._torch.cuda.empty_cache()


def _build_engine() -> _Engine:
    catvton_dir = config.CATVTON_DIR
    if not (catvton_dir / "model" / "pipeline.py").is_file():
        raise RuntimeError(
            f"Chưa thấy mã nguồn CatVTON trong {catvton_dir}. Trên Colab: chạy ô clone Zheng-Chong/CatVTON "
            f"vào thư mục này (ô 4 của quy trình)."
        )
    if str(catvton_dir) not in sys.path:
        sys.path.insert(0, str(catvton_dir))        # để import được `model` và `utils` của CatVTON
    try:
        import torch
        from huggingface_hub import snapshot_download
        from model.cloth_masker import AutoMasker
        from model.pipeline import CatVTONPipeline
        from utils import init_weight_dtype
    except ImportError as e:
        raise RuntimeError(
            f"Thiếu thư viện của CatVTON ({e}). Hãy chạy bằng /content/venv39/bin/python (Python 3.9) "
            f"đã cài requirements-model.txt, không dùng python mặc định của Colab."
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy GPU. Colab: Runtime > Change runtime type > T4 GPU. (Không có GPU thì đặt USE_MOCK=1.)")

    repo_path = snapshot_download(repo_id=config.CATVTON_CKPT_REPO, allow_patterns=config.CATVTON_PATTERNS)
    pipeline = CatVTONPipeline(
        base_ckpt=config.CATVTON_BASE,
        attn_ckpt=repo_path,
        attn_ckpt_version=config.CATVTON_CKPT_VERSION,
        weight_dtype=init_weight_dtype(config.PRECISION),
        device="cuda",
        skip_safety_check=SKIP_SAFETY,
        use_tf32=True,
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device="cuda",
    )
    return _Engine(pipeline, automasker, torch)


def _load_engine() -> _Engine:
    """Nạp mô hình đúng một lần cho cả tiến trình (an toàn khi nhiều luồng cùng gọi)."""
    global _ENGINE
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                _ENGINE = _build_engine()
    return _ENGINE


def _repaint(person: Image.Image, result: Image.Image, mask: Image.Image) -> Image.Image:
    """Dán lại điểm ảnh gốc ra ngoài mask, làm mờ viền mask như CatVTON/inference.py."""
    kernel = result.size[1] // 50
    if kernel % 2 == 0:
        kernel += 1
    soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(kernel)), dtype=np.float64)[..., None] / 255.0
    out = np.asarray(person, dtype=np.float64) * (1 - soft) + np.asarray(result, dtype=np.float64) * soft
    return Image.fromarray(out.astype(np.uint8))


def run_tryon(person_img: Image.Image,
              garment_img: Image.Image,
              mask: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    """Thử áo lên người.

    Vào : person_img, garment_img – PIL, kích thước bất kỳ (chuẩn hóa về RGB 768x1024 bằng src.utils, đệm trắng).
          mask – None: tự tạo bằng AutoMasker (loại "upper"); có giá trị: dùng mask này
                 (PIL 'L', 255 = vùng áo sẽ thay).
    Ra  : (ảnh_kết_quả RGB 768x1024, ảnh_mask 'L' 768x1024 chỉ gồm 0/255).
    Quy tắc: không ghi file, không print; đầu vào lỗi thì ném ValueError (thông báo tiếng Việt).
    Ảnh người nên có tỉ lệ gần 3:4, nhìn thẳng, thấy rõ thân trên; ảnh lệch tỉ lệ sẽ bị đệm viền trắng.
    """
    validate_input_image(person_img, "ảnh người")
    validate_input_image(garment_img, "ảnh áo")
    if mask is not None and not isinstance(mask, Image.Image):
        raise ValueError("Mask không hợp lệ (cần ảnh PIL, 255 = vùng áo).")

    engine = _load_engine()
    person = standardize_image(person_img)
    garment = standardize_image(garment_img)

    if mask is None:
        used_mask = standardize_mask(engine.automasker(person, "upper")["mask"])
        empty_msg = "Không nhận ra vùng thân trên trong ảnh người. Hãy dùng ảnh nhìn thẳng, thấy rõ thân người."
    else:
        used_mask = standardize_mask(mask)
        empty_msg = "Mask trống (không có điểm ảnh nào bằng 255)."
    if not np.asarray(used_mask).any():
        raise ValueError(empty_msg)

    try:
        result = engine.pipeline(
            image=person,
            condition_image=garment,
            mask=used_mask.filter(ImageFilter.GaussianBlur(PIPELINE_BLUR)),
            num_inference_steps=config.NUM_STEPS,
            guidance_scale=config.GUIDANCE,
            height=config.IMG_H,
            width=config.IMG_W,
            generator=engine.generator(config.SEED),
        )[0]
    except engine.oom_error as e:
        engine.free_cache()
        raise RuntimeError("GPU hết bộ nhớ. Hãy chạy lại với GPU T4 trống (Runtime > Restart session) rồi thử lại.") from e

    result = result.convert("RGB")
    if result.size != config.IMG_SIZE:
        result = result.resize(config.IMG_SIZE, Image.LANCZOS)
    if config.REPAINT:
        result = _repaint(person, result, used_mask)
    return result, used_mask
