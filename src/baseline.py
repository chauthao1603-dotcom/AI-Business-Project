"""PV2 – baseline "dán áo lên người" và hàm xóa nền ảnh áo.

HIỆN LÀ KHUNG: PV2 điền phần thân hàm (docs/PHAN_VIEC_v2.md mục 2.6 và 2.8). KHÔNG đổi chữ ký.
"""
from __future__ import annotations

from PIL import Image


def run_baseline(person_img: Image.Image,
                 garment_img: Image.Image,
                 mask: Image.Image) -> Image.Image:
    """Dán ảnh áo lên vùng mask của ảnh người.

    Vào : person_img, garment_img – PIL, kích thước bất kỳ; mask – PIL 'L', 255 = vùng áo (chuẩn ở src/utils.py).
    Ra  : ảnh PIL RGB 768x1024.

    Các bước: chuẩn hóa ảnh (src.utils.standardize_image) -> hình chữ nhật bao quanh mask
    (np.where(mask == 255)) -> xóa nền trắng của áo (cả 3 kênh > 240 là nền) và cắt sát -> resize vừa
    hình chữ nhật -> dán đè, chỉ dán điểm ảnh không phải nền.
    """
    raise NotImplementedError("run_baseline chưa được cài đặt (PV2, mục 2.6 trong docs/PHAN_VIEC_v2.md).")


def remove_bg(garment_img: Image.Image) -> Image.Image:
    """(Tùy chọn) Xóa nền ảnh áo bằng thư viện rembg, ghép lên nền trắng, trả PIL RGB."""
    raise NotImplementedError("remove_bg chưa được cài đặt (PV2, mục 2.8, tùy chọn).")
