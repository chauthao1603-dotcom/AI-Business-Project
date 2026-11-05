"""Hàm dùng chung: tên cặp, chuẩn hóa ảnh và mask theo docs/PHAN_VIEC_v2.md mục 1.2–1.3.

Chỉ phụ thuộc Pillow và numpy để chạy được trên laptop không GPU.
(PV1 có thể dùng resize_and_padding của repo CatVTON cho mô hình; hàm ở đây là bản chung cho baseline và hàm giả.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src import config


# ---------- Tên file / định danh cặp ----------
def stem(name) -> str:
    """'00006_00.jpg' -> '00006_00'."""
    return Path(str(name)).stem


def pair_id(person, cloth) -> str:
    """Định danh cặp = '{stem người}__{stem áo}', cũng là tên file ảnh kết quả (không đuôi)."""
    return f"{stem(person)}__{stem(cloth)}"


def split_pair_id(pid: str) -> tuple[str, str]:
    parts = pid.split("__")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"id không đúng mẫu '<người>__<áo>': {pid!r}")
    return parts[0], parts[1]


# ---------- Chuẩn hóa ảnh ----------
def to_rgb(img: Image.Image) -> Image.Image:
    """Đưa về RGB; ảnh có nền trong suốt được ghép lên nền trắng."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def resize_and_pad(img: Image.Image, size=config.IMG_SIZE, fill=(255, 255, 255),
                   resample=Image.BICUBIC) -> Image.Image:
    """Resize giữ tỉ lệ rồi đệm cho vừa `size` (rộng, cao); ảnh đặt ở giữa."""
    w, h = img.size
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), resample)
    canvas = Image.new(img.mode, size, fill)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def standardize_image(img: Image.Image) -> Image.Image:
    """Ảnh bất kỳ -> PIL RGB (768, 1024)."""
    return resize_and_pad(to_rgb(img))


def standardize_mask(mask: Image.Image) -> Image.Image:
    """Mask bất kỳ -> PIL 'L' (768, 1024), chỉ gồm 0 và 255 (255 = vùng áo sẽ bị thay)."""
    m = mask.convert("L")
    if m.size != config.IMG_SIZE:
        m = resize_and_pad(m, fill=0, resample=Image.NEAREST)
    return Image.fromarray(((np.asarray(m) > 127) * 255).astype("uint8"))


def load_dataset_mask(path) -> Image.Image:
    """Đọc agnostic-mask của VITON-HD và trả mask chuẩn (255 = vùng áo), bất kể file gốc trắng hay đen."""
    m = standardize_mask(Image.open(path))
    return m if config.AGNOSTIC_REGION_IS_WHITE else ImageOps.invert(m)


def validate_input_image(img, name: str = "ảnh") -> None:
    """Kiểm tra ảnh người dùng đưa vào; lỗi thì ném ValueError với thông báo tiếng Việt cho giao diện hiển thị."""
    if img is None:
        raise ValueError(f"Chưa có {name}.")
    if not isinstance(img, Image.Image):
        raise ValueError(f"{name} không đọc được (không phải file ảnh hợp lệ).")
    w, h = img.size
    if min(w, h) < config.MIN_SIDE:
        raise ValueError(f"{name} quá nhỏ ({w}x{h}); cạnh ngắn nhất cần ≥ {config.MIN_SIDE}px.")
