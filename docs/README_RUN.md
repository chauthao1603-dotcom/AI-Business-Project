# Chạy dự án trên Colab (PV1, việc 1.6)

Tài liệu này dành cho bất kỳ thành viên nào cần chạy mô hình hoặc mở app bằng link, không cần hỏi lại PV1.
Mọi bước đều nằm trong `notebooks/01_setup.ipynb`; bạn chỉ cần làm đúng 4 bước dưới đây.

## Bước 1. Chuẩn bị một lần cho tài khoản Google của bạn

1. Mở thư mục Drive chung `tryon_project` bằng link nhóm đã gửi, đăng nhập đúng tài khoản Google bạn sẽ dùng cho Colab.
2. Chuột phải `tryon_project`, chọn **Sắp xếp (Organize) → Thêm lối tắt (Add shortcut) → Drive của tôi (My Drive) → Thêm**.
3. Nếu trong "Drive của tôi" đã có một thư mục khác cũng tên `tryon_project` (do bạn tự tạo trước đây), đổi tên nó thành `tryon_project_cu` để khỏi nhầm.
4. Nhờ chủ thư mục kiểm tra bạn có quyền **Người chỉnh sửa (Editor)**. Chỉ có quyền xem thì notebook sẽ báo lỗi ở ô 1.

## Bước 2. Mở notebook trên Colab

1. Vào https://colab.research.google.com, chọn **File → Open notebook → GitHub**.
2. Dán `chauthao1603-dotcom/AI-Business-Project`, chọn `notebooks/01_setup.ipynb`.
3. Chọn **Runtime → Change runtime type → T4 GPU → Save**.

## Bước 3. Chọn việc cần chạy

Ở **ô 0**, bật hoặc tắt các công tắc:

| Muốn làm | Đặt |
|---|---|
| Chỉ mở app để demo | `RUN_TRY6 = RUN_EVAL200 = RUN_CACHE = RUN_EXTERNAL = False`, `RUN_APP = True` |
| Sinh ảnh 200 cặp đánh giá | `RUN_EVAL200 = True` |
| Sinh cache demo 60 cặp | `RUN_CACHE = True` |
| Chạy ảnh ngoài dataset (khi PV5 giao) | `RUN_EXTERNAL = True` |

Sau đó chọn **Runtime → Run all**. Khi Colab hỏi quyền truy cập Google Drive, chọn **Cho phép** bằng đúng tài khoản ở bước 1.

## Bước 4. Lấy link app

Ô 11 in dòng `LINK APP: https://....trycloudflare.com`. Mở link đó trên điện thoại hoặc máy khác.

- Link chỉ sống khi tab Colab còn mở. Đóng tab hoặc Colab ngắt phiên thì phải Run all lại, và link mới sẽ khác.
- Cặp ảnh có trong cache (6 người demo × 10 áo đầu của cửa hàng) hiện ngay. Ảnh tự tải lên lần đầu mất 1–2 phút để nạp mô hình; các lần sau mỗi ảnh mất khoảng bằng số giây/ảnh in ở ô 8.
- Không chạy ô 7 hoặc ô 9 trong lúc app đang mở: GPU T4 không đủ bộ nhớ cho hai mô hình. Chạy ô 12 để tắt app trước.

## Khi gặp lỗi

Mỗi ô tự kiểm tra và in ra nguyên nhân kèm cách sửa (dòng bắt đầu bằng `DungLai:`). Các lỗi hay gặp:

| Thông báo | Cách sửa |
|---|---|
| `Chưa có GPU` | Runtime → Change runtime type → T4 GPU. Nếu Colab báo hết hạn mức GPU, đổi tài khoản hoặc chờ hôm sau |
| `Không thấy thư mục Drive CHUNG` | Làm lại bước 1 (lối tắt), rồi Runtime → Restart session → Run all |
| `Không ghi được vào ...` | Xin quyền Người chỉnh sửa |
| `Sai phiên bản ...` | Chạy `!rm -rf /content/venv39` trong một ô mới, rồi Run all |
| `run_batch dừng vì nhiều cặp liên tiếp lỗi` | Đọc 5 dòng lỗi in ngay phía trên; gửi chúng cho PV1 |
| Colab ngắt giữa chừng | Mở lại và Run all; ảnh đã có trên Drive được bỏ qua |

## Kết quả nằm ở đâu (trên Drive `tryon_project`)

| Thư mục | Nội dung | Ai dùng |
|---|---|---|
| `outputs/catvton/` | Ảnh CatVTON của 200 cặp đánh giá, tên `{người}__{áo}.png` | PV3 |
| `outputs/masks/` | Mask đã dùng, tên `{người}.png` | PV2, PV3 |
| `data/processed/cache/` | Cache demo 60 cặp (ảnh và `_mask.png`) | PV2 lọc lỗi, PV4 dùng trong app |
| `outputs/external/` | Ảnh ngoài dataset | PV3 |
| `results/` | `timing_log_*.csv`, `errors_*.csv`, `pv1_timing_summary.csv`, `pv1_missing_inputs.csv` | PV3 (báo cáo) |
