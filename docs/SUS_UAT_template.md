# Mẫu kiểm thử người dùng (UAT) và bảng hỏi SUS

## A. Lời mở đầu đọc cho người tham gia
"Chúng tôi đang kiểm tra **sản phẩm**, không kiểm tra bạn. Không có câu trả lời đúng hay sai.
Bạn có thể dừng bất cứ lúc nào. Thông tin được ẩn danh và chỉ dùng cho báo cáo môn học.
Bạn đồng ý tham gia và cho phép ghi chú/ghi hình màn hình không? (Có / Không)"

## B. Thông tin người tham gia (ẩn danh)
- Mã: P01, P02, ...
- Nhóm (chủ shop / shipper / sinh viên / khác):
- Mức quen dùng ứng dụng web (1–5):

## C. Nhiệm vụ (đưa từng nhiệm vụ, KHÔNG hướng dẫn trước)
| # | Nhiệm vụ | Thành công? (Y/N) | Thời gian (giây) | Số lỗi/nhầm | Ghi chú quan sát |
|---|---|---|---|---|---|
| 1 | (ví dụ) Nhập danh sách đơn hàng và chạy tối ưu | | | | |
| 2 | (ví dụ) Tìm tổng quãng đường tiết kiệm được | | | | |
| 3 | (ví dụ) Đổi số shipper và so sánh kết quả | | | | |

Chỉ số: tỷ lệ hoàn thành, thời gian trung bình mỗi nhiệm vụ, số lỗi.

## D. Bảng hỏi SUS (System Usability Scale – Brooke, 1996)
Thang 1 = Hoàn toàn không đồng ý … 5 = Hoàn toàn đồng ý. Cho điền ngay sau khi làm nhiệm vụ.

| # | Câu hỏi | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1 | Tôi nghĩ rằng tôi sẽ muốn sử dụng hệ thống này thường xuyên. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | Tôi thấy hệ thống này phức tạp một cách không cần thiết. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | Tôi thấy hệ thống này dễ sử dụng. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | Tôi nghĩ rằng tôi sẽ cần sự hỗ trợ của một người có chuyên môn kỹ thuật để dùng được hệ thống này. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | Tôi thấy các chức năng trong hệ thống này được tích hợp tốt với nhau. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | Tôi thấy hệ thống này có quá nhiều điểm thiếu nhất quán. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 7 | Tôi hình dung hầu hết mọi người sẽ học cách dùng hệ thống này rất nhanh. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | Tôi thấy hệ thống này rất cồng kềnh, khó thao tác. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9 | Tôi cảm thấy rất tự tin khi sử dụng hệ thống này. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | Tôi cần học rất nhiều thứ trước khi có thể bắt đầu dùng hệ thống này. | ☐ | ☐ | ☐ | ☐ | ☐ |

## E. Câu hỏi mở (3 câu)
1. Điều bạn thích nhất ở sản phẩm là gì?
2. Điều gây khó chịu hoặc khó hiểu nhất là gì?
3. Bạn mong muốn bổ sung hoặc thay đổi điều gì?

## F. Cách chấm điểm SUS
- Câu lẻ (1, 3, 5, 7, 9): điểm đóng góp = câu trả lời − 1.
- Câu chẵn (2, 4, 6, 8, 10): điểm đóng góp = 5 − câu trả lời.
- Tổng 10 câu × 2,5 = điểm SUS (0–100). Lấy trung bình trên tất cả người tham gia.
- Mốc tham khảo: trung bình ngành khoảng 68; từ 80 trở lên là rất tốt; dưới 51 là kém.
- Lưu ý: SUS đo mức độ dễ dùng chung, **không phải phần trăm**. Với 5–10 người, hãy báo cáo kèm độ lệch chuẩn và nói rõ cỡ mẫu nhỏ.
- Tính tự động: lưu kết quả vào CSV (cột q1..q10) rồi chạy `python -m src.sus ket_qua.csv`.

## G. Tổng hợp
| Chỉ số | Giá trị |
|---|---|
| Số người tham gia | |
| Tỷ lệ hoàn thành nhiệm vụ | |
| Thời gian TB/nhiệm vụ | |
| Điểm SUS trung bình ± SD | |
| 3 vấn đề được nêu nhiều nhất | |
| Thay đổi đã thực hiện sau UAT | |
