# data/processed – dữ liệu đã xử lý

**File nhỏ (commit lên Git):** các file `.csv` ở thư mục này.

| File | Người tạo | Nội dung (cột) |
|---|---|---|
| `eval_pairs.csv` | PV2 (2.3) | `id,person,cloth,mode` – 200 cặp: 100 `paired`, 100 `unpaired` |
| `catalog.csv` | PV2 (2.4) | `product_id,name,category,color,price_vnd,image_file` (`image_file` nằm trong `garments/`) |
| `catalog.sample.csv` | có sẵn | Bản mẫu 3 dòng để PV4 dựng giao diện khi chưa có `catalog.csv` |
| `demo_review.csv` | PV2 (2.5) | `id,ok,ghi_chu` – cặp demo nào đạt |
| `external_pairs.csv` | PV5 (5.1) | `id,person,cloth,mode` với `mode=external` |

**Ảnh (để trên Drive, KHÔNG commit; `.gitignore` chặn `.png/.jpg/.jpeg/.webp`):**

| Thư mục | Nội dung | Người giao |
|---|---|---|
| `garments/` | ảnh áo của cửa hàng | PV2 |
| `demo/persons/`, `demo/garments/` | bộ demo: 6 người, 10 áo | PV2 |
| `external/persons/`, `external/garments/` | ảnh ngoài dataset (có sự đồng ý của người chụp) | PV5 |
| `cache/` | kết quả sinh sẵn `{id}.png`, `{id}_mask.png` | PV1 |

Chuẩn tên file và định dạng CSV: `docs/PHAN_VIEC_v2.md` mục 1.2 và 1.5. Cách lấy ảnh: `docs/handoff.md`.
