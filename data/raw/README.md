# data/raw – dữ liệu gốc VITON-HD (để trên Drive, KHÔNG commit)

- **Người giao:** PV2 (việc 2.1). **Người dùng:** PV1, PV3.
- Nguồn: https://github.com/shadow2496/VITON-HD (dữ liệu chỉ dùng cho nghiên cứu; đọc điều khoản trong README của họ và ghi nguồn trong báo cáo).
- Chỉ cần phần `test`. Đặt đúng cấu trúc:
  ```
  data/raw/
  ├── test/
  │   ├── image/           # ảnh người, 768×1024
  │   ├── cloth/           # ảnh áo
  │   └── agnostic-mask/   # mask vùng áo của VITON-HD
  └── test_pairs.txt       # mỗi dòng: <ảnh người> <ảnh áo>
  ```
- Mọi thứ trong thư mục này (trừ file README này và `.gitkeep`) bị `.gitignore` chặn. Lấy dữ liệu từ Drive chung: xem `docs/handoff.md`.
- Chiều của `agnostic-mask` (vùng áo trắng hay đen) do PV2 xác nhận bằng `scripts/check_dataset.py`, rồi sửa `AGNOSTIC_REGION_IS_WHITE` trong `src/config.py`.
