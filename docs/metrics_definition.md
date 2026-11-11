# Định nghĩa các chỉ số đánh giá (PV3, mục 3.1)

Trang này chốt cách tính và cách đọc từng chỉ số dùng trong dự án Virtual Try-On, và cho biết chỉ số nào đã có trong slide Chapter 2 (Evaluating ML Performance) để báo cáo dẫn đúng chương. Thang tính khớp bảng 1.6 trong `docs/PHAN_VIEC_v2.md`.

## Bảng tóm tắt

| Chỉ số | Đo cái gì | Tốt khi | Dùng ở đâu trong dự án | Thang tính (bảng 1.6) | Nguồn |
|---|---|---|---|---|---|
| MSE | Sai số bình phương trung bình từng điểm ảnh | Thấp | 3.2, cặp paired, `region=full` và `mask` → `results_per_pair.csv`, biểu đồ 3.5 | Thang 0–255 | Chuẩn chung, không có trong Chapter 2 |
| RMSE | Căn bậc hai của MSE | Thấp | Như MSE | Thang 0–255, RMSE = √MSE | Chuẩn chung, không có trong Chapter 2 |
| MAE | Sai số tuyệt đối trung bình từng điểm ảnh | Thấp | Như MSE | Thang 0–255 | Chuẩn chung, không có trong Chapter 2 |
| SSIM | Độ giống về độ sáng, độ tương phản, cấu trúc | Cao (tối đa 1) | 3.2, cặp paired, hai region; phép thử `selftest` | `data_range=255` | Chuẩn ngành thử đồ ảo |
| LPIPS | Khoảng cách cảm nhận trong không gian đặc trưng mạng nơ-ron | Thấp (tối thiểu 0) | 3.2, cặp paired, hai region; 3.5 dùng LPIPS vùng mask để chọn 5 ca lỗi | Mạng `alex`, ảnh đưa vào đổi về [-1, 1] | Chuẩn ngành thử đồ ảo |
| FID | Khoảng cách giữa phân phối cả bộ ảnh sinh và bộ ảnh thật | Thấp | 3.2, cặp unpaired (không có đáp án) → `results_summary.csv`, biểu đồ 3.5 | So tập ảnh sinh với tập ảnh thật trong `data/raw/test/image` | Chuẩn ngành thử đồ ảo |
| KID | Như FID nhưng không chệch khi mẫu nhỏ | Thấp | Như FID | Như FID, `subset_size` ≤ số ảnh | Chuẩn ngành thử đồ ảo |
| Ma trận nhầm lẫn | Bảng đếm TP, FN, FP, TN | Không có "tốt/xấu", dùng để tính 4 chỉ số dưới | 3.4, khảo sát "thật hay AI" → `figs/confusion_matrix.png` | Lớp dương = "AI" | **Có trong Chapter 2** |
| Accuracy | Tỷ lệ đoán đúng | Gần 50% là tốt cho mô hình (xem mục khảo sát) | 3.4 → `survey_summary.csv` | — | **Có trong Chapter 2** |
| Precision | Trong số ảnh bị gọi là "AI", bao nhiêu thật sự là AI | Xem mục khảo sát | 3.4 → `survey_summary.csv` | — | **Có trong Chapter 2** |
| Recall | Trong số ảnh AI, bao nhiêu bị bắt được | Thấp là tốt cho mô hình | 3.4 → `survey_summary.csv` | — | **Có trong Chapter 2** (tên khác: Sensitivity) |
| F1 | Trung bình điều hòa của precision và recall | Xem mục khảo sát | 3.4 → `survey_summary.csv` | — | Chuẩn chung, **không có trong slide Chapter 2** |

Ngoài các chỉ số trên, phần A/B của khảo sát (3.4) chỉ tính tỷ lệ người chọn ảnh CatVTON thay vì baseline, và 3.6 đánh giá định tính ảnh ngoài dataset, không có chỉ số.

Slide Chapter 2 còn có specificity, lỗi loại I và loại II, đường cong ROC. Dự án không dùng các chỉ số này nên không đưa vào bảng.

## Nhóm 1. Sai số điểm ảnh (so ảnh sinh với ảnh thật)

Gọi x là ảnh thật và y là ảnh do mô hình sinh, cùng kích thước 768×1024, giá trị điểm ảnh từ 0 đến 255, N là tổng số giá trị (điểm ảnh × 3 kênh màu).

```
MSE  = (1/N) · Σ (x − y)²
RMSE = √MSE
MAE  = (1/N) · Σ |x − y|
```

Ba chỉ số này đều bằng 0 khi hai ảnh giống hệt nhau và càng lớn khi hai ảnh càng lệch. RMSE và MAE có cùng đơn vị với điểm ảnh nên dễ đọc hơn MSE: RMSE bằng 10 nghĩa là trung bình mỗi điểm ảnh lệch khoảng 10 trên thang 255. MSE phạt nặng các điểm lệch lớn hơn MAE. Nhược điểm chung là các chỉ số này không nhìn thấy cấu trúc: dịch cả hình đi một điểm ảnh có thể làm sai số tăng mạnh dù mắt người thấy hai ảnh gần như giống nhau, vì vậy cần kết hợp với SSIM và LPIPS.

## Nhóm 2. Độ giống về cấu trúc và cảm nhận

**SSIM** so sánh hai ảnh theo ba mặt là độ sáng, độ tương phản và cấu trúc, tính trên từng cửa sổ nhỏ rồi lấy trung bình.

```
SSIM(x, y) = ((2·μx·μy + C1)(2·σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2))
C1 = (0.01·L)², C2 = (0.03·L)², L = 255
```

Trong đó μ là trung bình, σ² là phương sai, σxy là hiệp phương sai trong cửa sổ. SSIM bằng 1 khi hai ảnh giống hệt, càng thấp càng khác. Dự án gọi `skimage.metrics.structural_similarity(..., channel_axis=2, data_range=255)`.

**LPIPS** (Learned Perceptual Image Patch Similarity) đưa cả hai ảnh qua một mạng nơ-ron đã huấn luyện (ở đây là AlexNet), lấy đặc trưng ở nhiều tầng, chuẩn hóa, rồi tính khoảng cách có trọng số giữa hai bộ đặc trưng. Giá trị bằng 0 khi hai ảnh giống hệt và càng lớn khi mắt người thấy hai ảnh càng khác nhau. LPIPS bám sát đánh giá của con người hơn MSE, nên thường là chỉ số chính khi so các mô hình sinh ảnh. Ảnh phải được đổi về khoảng [-1, 1] trước khi đưa vào mạng.

## Nhóm 3. Chất lượng cả bộ ảnh (không cần ảnh đáp án)

FID và KID so sánh phân phối của tập ảnh sinh với tập ảnh thật, nên dùng được cho cả cặp unpaired là cặp không có ảnh đáp án. Cả hai lấy đặc trưng 2048 chiều của mạng Inception-v3.

```
FID = ‖μr − μg‖² + Tr(Σr + Σg − 2·(Σr·Σg)^½)
KID = MMD² giữa hai tập đặc trưng, nhân đa thức k(a, b) = (aᵀb/d + 1)³
```

Trong đó r là tập ảnh thật, g là tập ảnh sinh, μ là vector trung bình và Σ là ma trận hiệp phương sai của đặc trưng. Cả hai càng thấp càng tốt. Với khoảng 100 ảnh, ma trận hiệp phương sai 2048×2048 không đủ dữ liệu để ước lượng ổn định nên FID bị lệch lên, vì vậy chỉ so FID tương đối giữa baseline và CatVTON trên cùng một mẫu, không so với con số trong các bài báo. KID là ước lượng không chệch nên đáng tin hơn khi mẫu nhỏ, nhưng tham số `subset_size` phải không lớn hơn số ảnh.

## Nhóm 4. Khảo sát "thật hay AI" (có trong Chapter 2)

Người xem được cho 20 ảnh và chọn "Thật" hoặc "AI". Lớp dương là "AI". Mỗi lượt trả lời rơi vào đúng một trong bốn ô của ma trận nhầm lẫn.

| | Người xem chọn "AI" | Người xem chọn "Thật" |
|---|---|---|
| Ảnh thật là AI | TP | FN |
| Ảnh thật là ảnh thật | FP | TN |

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · Precision · Recall / (Precision + Recall)
```

Ví dụ lấy từ slide Chapter 2 (giao dịch gian lận, lớp dương là "gian lận"): TP = 5, FP = 2, FN = 1, TN = 2 cho Accuracy = 7/10 = 0.70, Precision = 5/7 = 0.71, Recall = 5/6 = 0.83. Tên "Sensitivity" trong slide chính là Recall.

Điểm khác biệt khi dùng trong dự án: ở bài toán phân loại thông thường, các chỉ số này càng cao càng tốt. Ở đây người xem là "bộ phân loại" còn mô hình sinh ảnh là bên muốn đánh lừa, nên ý nghĩa đảo ngược. Accuracy gần 50% nghĩa là người xem không phân biệt được ảnh AI với ảnh thật, tức ảnh AI thuyết phục. Accuracy gần 100% nghĩa là ảnh AI dễ bị nhận ra. Recall thấp nghĩa là nhiều ảnh AI lọt qua mà không bị phát hiện, và đó là kết quả tốt cho mô hình. Phần so sánh A/B giữa baseline và CatVTON không dùng các chỉ số này mà chỉ tính tỷ lệ người chọn CatVTON.

## Hai lưu ý khi đọc kết quả

Thứ nhất, các chỉ số theo từng cặp được tính ở hai phiên bản. Bản `region=full` tính trên toàn ảnh và thường cao ảo, vì phần ngoài mask của ảnh sinh giống hệt ảnh gốc. Bản `region=mask` chỉ tính trong vùng mask (với SSIM và LPIPS là khung chữ nhật bao quanh mask) nên phản ánh đúng phần mô hình thực sự phải sinh. Mask của VITON-HD không chỉ là thân áo mà còn gồm cổ và hai cánh tay, nên số `region=mask` có cả phần da tay, và báo cáo cần nói rõ điều này.

Thứ hai, chỉ có cặp paired mới có ảnh đáp án để tính nhóm 1 và nhóm 2. Cặp unpaired chỉ đánh giá được bằng FID, KID và khảo sát. Khi chạy nhóm 1 và 2 trên cặp paired, `pred = gt` phải cho MSE = 0, SSIM = 1 và LPIPS gần 0 (đây là phép thử bắt buộc của `eval/run_eval.py`).
