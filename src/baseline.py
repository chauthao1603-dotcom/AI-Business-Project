"""PV2 – baseline "dán áo lên người" và hàm xóa nền ảnh áo (docs/PHAN_VIEC_v2.md mục 2.6 và 2.8).

Thuật toán do PV2 viết (trước đây ở core/baseline.py), chuyển về đúng vị trí src/baseline.py theo quy ước v2.
KHÔNG đổi chữ ký hàm: app.py, scripts/run_baseline_batch.py và tests/test_contract.py dựa vào nó.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.utils import standardize_image, standardize_mask

BG_THRESHOLD = 240   # điểm ảnh có cả 3 kênh > 240 được coi là nền trắng của ảnh áo


def run_baseline(person_img: Image.Image,
                 garment_img: Image.Image,
                 mask: Image.Image) -> Image.Image:
    """Dán ảnh áo lên vùng mask của ảnh người.

    Vào : person_img, garment_img – PIL, kích thước bất kỳ; mask – PIL 'L', 255 = vùng áo (chuẩn ở src/utils.py).
    Ra  : ảnh PIL RGB 768x1024.

    Các bước: chuẩn hóa ảnh -> hình chữ nhật bao quanh mask -> xóa nền trắng của áo và cắt sát ->
    resize vừa hình chữ nhật -> dán đè, chỉ dán điểm ảnh không phải nền và nằm trong mask.
    Mask rỗng hoặc ảnh áo toàn nền trắng -> trả lại ảnh người đã chuẩn hóa.
    """
    # Bước 1: chuẩn hóa ảnh người, ảnh áo và mask
    person = standardize_image(person_img)
    garment = standardize_image(garment_img)
    mask_array = np.array(standardize_mask(mask))

    person_array = np.array(person).copy()
    garment_array = np.array(garment)

    # Bước 2: hình chữ nhật bao quanh vùng mask
    ys, xs = np.where(mask_array == 255)
    if len(xs) == 0:
        return person
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1

    # Bước 3: xóa nền trắng của ảnh áo và cắt sát phần áo
    non_background = np.any(garment_array <= BG_THRESHOLD, axis=2)
    cloth_ys, cloth_xs = np.where(non_background)
    if len(cloth_xs) == 0:
        return person
    cloth_crop = garment.crop((cloth_xs.min(), cloth_ys.min(), cloth_xs.max() + 1, cloth_ys.max() + 1))

    # Bước 4: resize áo vừa hình chữ nhật mask
    cloth_resized = cloth_crop.resize((int(x_max - x_min), int(y_max - y_min)), Image.Resampling.LANCZOS)
    resized_array = np.array(cloth_resized)

    # Bước 5: chỉ dán điểm ảnh không phải nền và nằm trong mask
    paste_mask = np.any(resized_array <= BG_THRESHOLD, axis=2) & (mask_array[y_min:y_max, x_min:x_max] == 255)
    region = person_array[y_min:y_max, x_min:x_max]
    region[paste_mask] = resized_array[paste_mask]
    person_array[y_min:y_max, x_min:x_max] = region

    return Image.fromarray(person_array).convert("RGB")


def remove_bg(garment_img: Image.Image) -> Image.Image:
    """(Tùy chọn) Xóa nền ảnh áo bằng thư viện rembg, ghép lên nền trắng, trả PIL RGB."""
    raise NotImplementedError("remove_bg chưa được cài đặt (PV2, mục 2.8, tùy chọn).")
