"""PV4 – hàm try-on GIẢ, cùng chữ ký với src/tryon.py, để làm giao diện khi chưa có GPU/mô hình.

Bật bằng biến môi trường USE_MOCK=1. Hàm giả KHÔNG phải AI: nó chỉ phủ màu chủ đạo của áo lên vùng thân
để nhìn thấy "có thay đổi". Phần ngoài mask giữ nguyên ảnh người, giống hành vi của mô hình thật.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src import config
from src.utils import standardize_image, standardize_mask, validate_input_image


def _torso_mask() -> Image.Image:
    """Mask hình chữ nhật xấp xỉ vùng thân, dùng khi không truyền mask."""
    w, h = config.IMG_SIZE
    m = Image.new("L", config.IMG_SIZE, 0)
    ImageDraw.Draw(m).rectangle([int(0.22 * w), int(0.22 * h), int(0.78 * w), int(0.62 * h)], fill=255)
    return m


def _dominant_color(img: Image.Image) -> tuple[int, int, int]:
    """Màu trung bình của các điểm ảnh không phải nền trắng (nếu toàn trắng thì trung bình tất cả)."""
    px = np.asarray(img).reshape(-1, 3)
    body = px[(px < 240).any(axis=1)]
    src = body if len(body) else px
    return tuple(int(v) for v in src.mean(axis=0))


def run_tryon(person_img: Image.Image,
              garment_img: Image.Image,
              mask: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    validate_input_image(person_img, "ảnh người")
    validate_input_image(garment_img, "ảnh áo")
    person = standardize_image(person_img)
    garment = standardize_image(garment_img)
    mask = _torso_mask() if mask is None else standardize_mask(mask)
    solid = Image.new("RGB", person.size, _dominant_color(garment))
    result = Image.composite(Image.blend(person, solid, 0.65), person, mask)
    return result, mask
