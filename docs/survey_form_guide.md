# Hướng dẫn dựng Google Form khảo sát (PV3, mục 3.3 và 3.4)

Form phải dựng đúng quy ước dưới đây thì `eval/analyze_survey.py` mới đọc được file trả lời mà không cần sửa tay.

## Trước khi dựng

1. Chạy `python -m eval.make_survey_set` (cần PV1 và PV2 đã giao ảnh). Lệnh tạo 20 ảnh `q01.png`…`q20.png` và 10 cặp `ab01_A.png`, `ab01_B.png`… trong `outputs/survey/`, cùng hai file đáp án `results/survey_key.csv` và `results/ab_key.csv`.
2. Chỉ nhóm giữ file đáp án. Không đưa hai file này và không đưa tên file gốc cho người làm khảo sát.
3. Không chạy lại lệnh sau khi đã dựng Form: bộ ảnh sẽ đổi và đáp án sẽ sai. Lệnh có chặn việc này (cần `--force` mới ghi đè).

## Phần 1: 20 câu "thật hay AI"

Tạo 20 câu hỏi dạng **Trắc nghiệm** (một lựa chọn), theo thứ tự `q01` đến `q20`. Mỗi câu:

| Mục | Giá trị |
|---|---|
| Tiêu đề câu hỏi | Bắt đầu bằng mã, ví dụ `q01 - Ảnh này là ảnh thật hay do AI tạo ra?` |
| Hình ảnh | Chèn ảnh `outputs/survey/q01.png` (đúng số thứ tự) |
| Lựa chọn | Đúng hai lựa chọn: `Thật` và `AI` |
| Bắt buộc | Bật |

Phần chữ sau mã được bỏ qua, nhưng **mã ở đầu tiêu đề phải đúng** (`q01`, không phải `Q1` hay `Câu 1`).

## Phần 2: 10 câu so sánh A/B

Tạo 10 câu **Trắc nghiệm**, theo thứ tự `ab01` đến `ab10`. Mỗi câu:

| Mục | Giá trị |
|---|---|
| Tiêu đề | `ab01 - Ảnh nào trông thật hơn?` |
| Hình ảnh | Chèn `ab01_A.png` và `ab01_B.png`, theo thứ tự A rồi B, ghi chú dưới mỗi ảnh là "Ảnh A", "Ảnh B" |
| Lựa chọn | Đúng hai lựa chọn: `Ảnh A` và `Ảnh B` |
| Bắt buộc | Bật |

## Cài đặt Form

Không thu email và không bật "giới hạn một phản hồi" nếu nhóm không cần, để tránh dữ liệu cá nhân trong file. Xáo thứ tự câu hỏi của Form ở chế độ tắt: thứ tự đã được xáo sẵn bởi script. Câu mở đầu nên ghi rõ đây là khảo sát cho bài tập môn học và không thu thông tin cá nhân.

## Sau khi có người trả lời

1. Trong Form, mở tab **Câu trả lời**, chọn **Tải xuống câu trả lời (.csv)**.
2. Lưu file vào `outputs/survey_responses.csv`. Không commit file này lên GitHub vì nó có thể chứa thông tin người trả lời.
3. Chạy `python -m eval.analyze_survey --responses outputs/survey_responses.csv`.
4. Lệnh in ma trận nhầm lẫn và bốn chỉ số, đồng thời ghi `results/survey_summary.csv` và `results/figs/confusion_matrix.png`.

Kiểm tra khi xong: tổng TP + FN + FP + TN phải bằng **số người trả lời × 20**. Nếu lệch, lệnh sẽ in dòng `CHÚ Ý` kèm số ô trống và số giá trị lạ. Nguyên nhân thường gặp là ô trống do câu hỏi không bắt buộc, hoặc tên lựa chọn khác `Thật` và `AI`.

## Cách đọc kết quả

Lớp dương là "AI". Accuracy gần 50% nghĩa là người xem không phân biệt được ảnh AI với ảnh thật, tức ảnh AI thuyết phục. Recall thấp nghĩa là nhiều ảnh AI lọt qua mà không bị nhận ra. Mục `per_question` trong `survey_summary.csv` cho biết tỷ lệ trả lời đúng từng câu: nếu một ảnh thật bị gần như tất cả gọi là "AI" (hoặc ngược lại), hãy xem lại ảnh đó vì có thể nó có dấu hiệu lạ. Phần A/B chỉ báo tỷ lệ người chọn ảnh của CatVTON; trên 50% nghĩa là CatVTON trông thật hơn baseline.
