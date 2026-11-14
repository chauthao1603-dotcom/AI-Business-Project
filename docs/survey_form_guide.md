# Hướng dẫn dựng Google Form khảo sát (PV3, mục 3.3 và 3.4)

Form phải dựng đúng quy ước dưới đây thì `eval/analyze_survey.py` mới đọc được file trả lời mà không cần sửa tay.

## Trước khi dựng

1. Chạy `python -m eval.make_survey_set` (cần PV1 và PV2 đã giao ảnh). Lệnh tạo 20 ảnh `q01.png`…`q20.png` và 10 cặp `ab01_A.png`, `ab01_B.png`… trong `outputs/survey/`, 10 ảnh ghép `ab01.png`…`ab10.png` trong `outputs/survey/ab_sheets/`, cùng hai file đáp án `results/survey_key.csv` và `results/ab_key.csv`. Nếu bộ khảo sát đã tạo theo thiết kế cũ (A/B là baseline – CatVTON), chạy `python -m eval.make_survey_set --redo-ab`: lệnh này chỉ làm lại phần 2 theo thiết kế mới, giữ nguyên 20 ảnh và đáp án phần 1. Lệnh cho kết quả giống nhau mỗi lần chạy (seed cố định).
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

## Phần 2: 10 câu "ảnh nào là ảnh chụp thật?"

Mỗi câu đặt cạnh nhau ảnh chụp thật của một người mẫu và ảnh CatVTON của chính người đó mặc chính chiếc áo đó (cặp paired). CatVTON chạy với REPAINT nên ngoài vùng mask hai ảnh gần như trùng nhau, người xem chỉ còn phần áo (đôi khi cả tay) để phân biệt. Người và áo ở phần này không xuất hiện ở phần 1, và ảnh thật nằm bên trái ở đúng 5 câu. Thiết kế cũ so baseline với CatVTON đã bỏ vì ảnh baseline dán áo quá dễ nhận ra.

Google Forms chỉ cho gắn **một** ảnh vào mỗi câu hỏi, nên mỗi câu dùng một ảnh ghép sẵn: ảnh A bên trái, ảnh B bên phải, chữ "A" và "B" in phía trên từng ảnh. Vị trí trái/phải đúng với `left_method`/`right_method` (`real` hoặc `catvton`) trong `results/ab_key.csv`.

Tạo 10 câu **Trắc nghiệm**, theo thứ tự `ab01` đến `ab10`. Mỗi câu:

| Mục | Giá trị |
|---|---|
| Tiêu đề | `ab01 - Ảnh nào là ảnh chụp thật?` |
| Hình ảnh | Chèn **một** ảnh `outputs/survey/ab_sheets/ab01.png` (đúng số thứ tự), để căn giữa và kéo rộng hết khung |
| Lựa chọn | Đúng hai lựa chọn: `Ảnh A` và `Ảnh B` |
| Bắt buộc | Bật |

Không dùng `ab01_A.png` và `ab01_B.png` riêng lẻ trong Form. Hai file này chỉ là nguyên liệu để ghép. Trước khi gửi, mở **Xem trước** trên điện thoại một lần: ảnh ghép nằm ngang nên trên màn hình hẹp mỗi ảnh khá nhỏ, cần chắc chắn vẫn nhìn rõ được chi tiết áo.

## Cài đặt Form

Không thu email và không bật "giới hạn một phản hồi" nếu nhóm không cần, để tránh dữ liệu cá nhân trong file. Xáo thứ tự câu hỏi của Form ở chế độ tắt: thứ tự đã được xáo sẵn bởi script. Câu mở đầu nên ghi rõ đây là khảo sát cho bài tập môn học và không thu thông tin cá nhân.

## Sau khi có người trả lời

1. Trong Form, mở tab **Câu trả lời**, chọn **Tải xuống câu trả lời (.csv)**.
2. Lưu file vào `outputs/survey_responses.csv`. Không commit file này lên GitHub vì nó có thể chứa thông tin người trả lời.
3. Chạy `python -m eval.analyze_survey --responses outputs/survey_responses.csv`.
4. Lệnh in ma trận nhầm lẫn và bốn chỉ số, đồng thời ghi `results/survey_summary.csv` và `results/figs/confusion_matrix.png`.

Kiểm tra khi xong: tổng TP + FN + FP + TN phải bằng **số người trả lời × 20**. Nếu lệch, lệnh sẽ in dòng `CHÚ Ý` kèm số ô trống và số giá trị lạ. Nguyên nhân thường gặp là ô trống do câu hỏi không bắt buộc, hoặc tên lựa chọn khác `Thật` và `AI`.

## Cách đọc kết quả

Lớp dương là "AI". Accuracy gần 50% nghĩa là người xem không phân biệt được ảnh AI với ảnh thật, tức ảnh AI thuyết phục. Recall thấp nghĩa là nhiều ảnh AI lọt qua mà không bị nhận ra. Mục `per_question` trong `survey_summary.csv` cho biết tỷ lệ trả lời đúng từng câu: nếu một ảnh thật bị gần như tất cả gọi là "AI" (hoặc ngược lại), hãy xem lại ảnh đó vì có thể nó có dấu hiệu lạ. Phần A/B báo tỷ lệ người xem chọn đúng ảnh chụp thật (`real_chosen_rate`), khoảng tin cậy 95% và p-value so với 50%. Tỷ lệ gần 50% (khoảng tin cậy chứa 50%) nghĩa là người xem không phân biệt được ảnh CatVTON với ảnh chụp thật của cùng người, cùng áo; tỷ lệ cao rõ rệt nghĩa là vẫn nhận ra được ảnh AI. Mỗi người trả lời cả 10 câu nên các lượt trả lời không hoàn toàn độc lập, khoảng tin cậy chỉ là ước lượng gần đúng. Mục `ab_pair_catvton_chosen` cho biết từng cặp bị nhầm nhiều hay ít.
