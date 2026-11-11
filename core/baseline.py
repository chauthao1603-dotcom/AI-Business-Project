
import numpy as np
from PIL import Image

from src.utils import standardize_image, standardize_mask


def run_baseline(person_img: Image.Image,
                 garment_img: Image.Image,
                 mask: Image.Image) -> Image.Image:
    """
    Dán ảnh áo lên vùng mask của ảnh người.

    Input:
        person_img: Ảnh người.
        garment_img: Ảnh áo.
        mask: Mask, 255 là vùng cần thay áo.

    Output:
        Ảnh RGB kích thước 768x1024.
    """

    # Bước 1: Chuẩn hóa ảnh người, ảnh áo và mask
    person = standardize_image(person_img)
    garment = standardize_image(garment_img)
    mask_img = standardize_mask(mask)

    # Chuyển ảnh thành mảng numpy
    person_array = np.array(person).copy()
    garment_array = np.array(garment)
    mask_array = np.array(mask_img)

    # Bước 2: Tìm hình chữ nhật bao quanh vùng mask
    ys, xs = np.where(mask_array == 255)

    if len(xs) == 0 or len(ys) == 0:
        return person

    x_min = xs.min()
    x_max = xs.max() + 1
    y_min = ys.min()
    y_max = ys.max() + 1

    # Bước 3: Xóa nền trắng của ảnh áo
    # Pixel có cả 3 kênh RGB > 240 được xem là nền trắng
    non_background = np.any(garment_array <= 240, axis=2)

    cloth_ys, cloth_xs = np.where(non_background)

    if len(cloth_xs) == 0 or len(cloth_ys) == 0:
        return person

    # Cắt sát phần áo, loại bỏ nền trắng xung quanh
    cloth_crop = garment.crop((
        cloth_xs.min(),
        cloth_ys.min(),
        cloth_xs.max() + 1,
        cloth_ys.max() + 1
    ))

    # Bước 4: Resize áo vừa với hình chữ nhật mask
    target_width = x_max - x_min
    target_height = y_max - y_min

    cloth_resized = cloth_crop.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS
    )

    resized_array = np.array(cloth_resized)

    # Xác định pixel không phải nền trắng
    resized_non_background = np.any(
        resized_array <= 240,
        axis=2
    )

    # Chỉ dán trong vùng mask
    mask_crop = mask_array[y_min:y_max, x_min:x_max] == 255

    paste_mask = resized_non_background & mask_crop

    # Bước 5: Dán áo lên ảnh người
    person_region = person_array[y_min:y_max, x_min:x_max]

    person_region[paste_mask] = resized_array[paste_mask]

    person_array[y_min:y_max, x_min:x_max] = person_region

    return Image.fromarray(person_array).convert("RGB")
