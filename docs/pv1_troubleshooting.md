# PV1 – Lỗi khi cài CatVTON trên Colab / Kaggle: nguyên nhân và cách sửa

Tài liệu này trả lời bản tóm tắt lỗi của nhóm (Colab, Kaggle, Hugging Face Space). Cách làm theo từng bước nằm trong `notebooks/01_setup.ipynb`.

## 1. Ba nguyên nhân gốc

1. **Python 3.13.** CatVTON cần **Python 3.9**: `torch==2.4.0` chỉ có bản dựng cho Python 3.8–3.12, và repo CatVTON kèm sẵn file biên dịch `detectron2/_C.cpython-39-x86_64-linux-gnu.so` (README của họ cũng ghim `python==3.9.0`). Python 3.10 trở lên không nạp được file này.
2. **`requirements.txt` của CatVTON cài `diffusers` từ nhánh main** (`git+https://github.com/huggingface/diffusers.git`, không ghim phiên bản). Nhánh này nay đòi Python ≥ 3.11 và lệch với `transformers==4.46.3` / `huggingface_hub`, nên ngay cả Python 3.10 cũng không cài được file gốc. Đây là nguồn của lỗi `resolve_revision`, xung đột `huggingface_hub`, và `BloomPreTrainedModel`.
3. **Space chính thức `zhengchong/CatVTON` đã đổi sang bản CatVTON-FLUX** (README của CatVTON ghi rõ), cần model gated `FLUX.1-Fill-dev`. Lỗi 401 là lỗi của chính Space, không phải do nhóm thiếu token.

Kết luận của nhóm "Python 3.13 là nguyên nhân gốc" **đúng**, nhưng kết luận "không khả thi" **chưa đúng**: chỉ cần chạy CatVTON trong môi trường Python 3.9 riêng và ghim lại phiên bản thư viện (đã làm trong `requirements-model.txt`).

## 2. Đã kiểm chứng gì, chưa kiểm chứng gì

| Đã kiểm chứng (bằng `uv pip compile`, đọc mã nguồn CatVTON) | Chưa kiểm chứng |
|---|---|
| File gốc của CatVTON trên Python 3.13: **thất bại** vì `torch==2.4.0` không có bản `cp313` (đúng lỗi của nhóm) | Cài và chạy thật trên GPU T4 (người soạn không có GPU; lần thử cài đầy đủ trong môi trường kiểm tra bị chặn nên không chạy) |
| File gốc trên Python 3.10: **thất bại** vì `diffusers` nhánh main đòi Python ≥ 3.11 | Việc nạp `detectron2._C` thật sự thành công trên Python 3.9 trong Colab |
| `requirements-model.txt` (kèm Streamlit và các gói của nhóm) trên Python 3.9: **giải quyết được**, ra `torch 2.4.0, diffusers 0.31.0, transformers 4.46.3, huggingface-hub 0.25.2, streamlit 1.50.0` | Chất lượng ảnh và thời gian thật trên T4 |
| Repo kèm file `.so` cho cp39; Python 3.10 chỉ nhận đuôi `.cpython-310-…so`, `.abi3.so`, `.so` | `src/tryon.py` chạy đúng (viết theo `app.py`, chưa chạy GPU) |
| Chuỗi import của phần tạo mask chạm `from detectron2 import _C` ở cấp module (phân tích tĩnh) | |

PV1 là người xác nhận phần bên phải bằng `notebooks/01_setup.ipynb`. Nếu một gói cần đổi phiên bản để chạy được, cập nhật `requirements-model.txt` và ghi vào đây.

## 3. Từng lỗi trong báo cáo của nhóm

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `torch==2.4.0` không tìm thấy | Python 3.13 không có bản dựng | Môi trường Python 3.9 (`uv venv --python 3.9`) |
| `torchvision==0.19.0` không tìm thấy | Như trên | Như trên |
| `Failed to build 'pillow'` | `pillow==10.3.0` không có bản dựng sẵn cho 3.13, pip phải build từ mã nguồn | Python 3.9 có bản dựng sẵn |
| Xung đột `huggingface_hub` / `gradio` / `transformers` / `diffusers` | Mỗi gói đòi một khoảng phiên bản `huggingface_hub` khác nhau; `gradio` chỉ `app.py` cần | Ghim cả bộ (`huggingface_hub==0.25.2`, `diffusers==0.31.0`, `transformers==4.46.3`), bỏ `gradio` |
| `ImportError: resolve_revision` | `diffusers` nhánh main và `huggingface_hub` lệch phiên bản | `diffusers==0.31.0` thay cho nhánh main |
| PyTorch CUDA 13.2 so với TorchAudio CUDA 13.0 | Colab cài sẵn `torchaudio` bản khác `torch` | CatVTON không dùng `torchaudio`; môi trường mới không có nó |
| `BloomPreTrainedModel` (PEFT/Transformers) | `peft` mới lệch `transformers` cũ; `peft` chỉ nhánh FLUX cần | Bỏ `peft`, ghim `transformers==4.46.3` |
| `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value` khi import CatVTON | Colab đặt biến `MPLBACKEND` cho notebook, môi trường 3.9 không có `matplotlib_inline` | `os.environ['MPLBACKEND'] = 'Agg'` (đã có trong notebook và `src/tryon.py`) |
| Kaggle không thấy GPU | Tài khoản chưa xác minh số điện thoại | *Settings → Phone verification*, sau đó chọn Accelerator = GPU |
| Kaggle `Could not resolve host: github.com` | Internet đang tắt | Bật *Internet* trong Session options (cũng cần xác minh điện thoại) |
| Kaggle Python 3.13 | Như Colab | Cùng cách: `uv venv --python 3.9` (cần Internet bật) |
| Space báo 401 `FLUX.1-Fill-dev` | Space chính thức nay là bản FLUX gated, lỗi nằm ở Space | Không dùng Space làm backend; chạy mã nguồn SD1.5 tại chỗ |

## 4. Vì sao không nên chuyển sang IDM-VTON hoặc OOTDiffusion

- **Cùng loại rào cản, nặng hơn.** README của IDM-VTON hướng dẫn tạo môi trường bằng `conda env create -f environment.yaml`, dùng nền SDXL và cần tải thêm checkpoint DensePose, human parsing, OpenPose. Bạn sẽ lại gặp đúng bài toán "cần môi trường cũ", với mô hình lớn hơn CatVTON nhiều (CatVTON dưới 8 GB VRAM). OOTDiffusion tôi chưa kiểm tra.
- **Báo cáo và demo sẽ mâu thuẫn.** Đề bài yêu cầu trình bày workflow của mô hình và demo ứng dụng của chính mô hình đó. Phân tích, so sánh với baseline và KPI đang đo trên CatVTON; demo bằng mô hình khác thì số liệu không áp dụng cho ảnh trong video.
- **Chi phí đổi.** Viết lại `run_tryon`, chạy lại 200 cặp, làm lại cả phần phân tích. Nếu buộc phải đổi, đổi **toàn bộ dự án** (báo cáo, KPI, demo) sang một mô hình, không trộn.

## 5. Thứ tự dự phòng (thử lần lượt, mỗi bước tối đa 2–3 giờ)

1. **Colab + môi trường Python 3.9** (`notebooks/01_setup.ipynb`).
2. **Nếu chỉ phần DensePose/AutoMasker lỗi** (import `detectron2._C`): vẫn làm được việc 1.2, 1.3, 1.4 bằng mask sẵn của VITON-HD (`scripts.smoke_tryon --mask …`; `AutoMasker` được nạp muộn nên không chặn). Chỉ chức năng "người dùng upload ảnh bất kỳ" cần AutoMasker; khi đó tạm demo bằng các ảnh của VITON-HD, và ghi rõ giới hạn trong báo cáo.
3. **Kaggle** (sau khi xác minh điện thoại, bật GPU và Internet), cùng notebook, đổi đường dẫn Drive.
4. **Máy của thành viên có GPU NVIDIA từ 8 GB VRAM.** Lưu ý file `.so` của CatVTON là Linux x86_64: Windows thuần không nạp được, hãy dùng WSL2 (Ubuntu) hoặc Linux.
5. **Chuyển mô hình**, chỉ khi các bước trên thất bại và phải đổi toàn bộ dự án như mục 4.

Hai chi tiết dễ vấp khi chạy được:
- **T4 không hỗ trợ bf16 tốt:** `app.py` mặc định `bf16`; `src/config.py` đã đặt mặc định `fp16`. GPU Ampere trở lên (A100, L4, RTX 30/40) đặt `TRYON_PRECISION=bf16`.
- **Colab ngắt phiên:** `HF_HOME` đặt ở đĩa Colab (cache mô hình tải lại sau mỗi phiên, vài GB); kết quả `outputs/` ghi thẳng lên Drive qua `TRYON_ROOT`.
