"""PV1 – hàm try-on thật dùng CatVTON (chạy trên Colab có GPU).

HIỆN LÀ KHUNG: PV1 điền phần thân hàm (docs/PHAN_VIEC_v2.md mục 1.3). KHÔNG đổi tên hàm, tên tham số
hay kiểu trả về: app.py, scripts/run_batch.py và tests/test_contract.py đều dựa vào chữ ký này.
"""
from __future__ import annotations

from PIL import Image


def run_tryon(person_img: Image.Image,
              garment_img: Image.Image,
              mask: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    """Thử áo lên người.

    Vào : person_img, garment_img – PIL, kích thước bất kỳ (sẽ chuẩn hóa về RGB 768x1024).
          mask – None: tự tạo bằng AutoMasker (loại "upper"); có giá trị: dùng mask này
                 (PIL 'L', 255 = vùng áo sẽ thay).
    Ra  : (ảnh_kết_quả RGB 768x1024, ảnh_mask 'L' 768x1024 chỉ gồm 0/255).
    Quy tắc: không ghi file, không print; đầu vào lỗi thì ném ValueError (thông báo tiếng Việt,
    có thể dùng src.utils.validate_input_image).

    Gợi ý cài đặt (đối chiếu tên lớp thật trong app.py của repo CatVTON):
      1. nạp mô hình một lần duy nhất (biến toàn cục hoặc functools.lru_cache);
      2. src.utils.validate_input_image rồi chuẩn hóa ảnh về 768x1024;
      3. nếu mask is None: tạo mask bằng AutoMasker ("upper");
      4. gọi pipeline với 50 bước, guidance 2.5, seed 42 (hằng số trong src/config.py);
      5. trả (kết quả, mask).
    """
    raise NotImplementedError(
        "run_tryon chưa được cài đặt (PV1, mục 1.3 trong docs/PHAN_VIEC_v2.md). "
        "Đặt USE_MOCK=1 để chạy với hàm giả."
    )
