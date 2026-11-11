"""PV1 – chạy thử MỘT cặp ảnh qua run_tryon thật, đo thời gian và VRAM (việc 1.2 và 1.3).

Chạy trong môi trường Python 3.9 đã cài requirements-model.txt, ở thư mục gốc repo:
    python -m scripts.smoke_tryon --person <ảnh người> --garment <ảnh áo> [--mask <mask>] [--out outputs/smoke]
Ví dụ với dữ liệu VITON-HD:
    python -m scripts.smoke_tryon --person data/raw/test/image/00006_00.jpg --garment data/raw/test/cloth/00012_00.jpg

Kết quả: {out}/{id}.png (ảnh try-on), {out}/{id}_mask.png (mask), và in ra số liệu dùng cho báo cáo
(thời gian nạp mô hình, giây/ảnh, VRAM đỉnh). Lần gọi thứ hai mới phản ánh tốc độ thật, vì lần đầu gồm cả nạp mô hình.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from src import config
from src.utils import load_dataset_mask, pair_id


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True)
    ap.add_argument("--garment", required=True)
    ap.add_argument("--mask", default=None, help="mask có sẵn (mặc định: tự tạo bằng AutoMasker)")
    ap.add_argument("--out", default=str(config.OUTPUTS / "smoke"))
    args = ap.parse_args(argv)

    from src.tryon import run_tryon          # import muộn: máy không GPU vẫn đọc được --help
    import torch

    person, garment = Image.open(args.person), Image.open(args.garment)
    mask = load_dataset_mask(args.mask) if args.mask else None

    t0 = time.time()
    result, used_mask = run_tryon(person, garment, mask)
    t_first = time.time() - t0                      # gồm cả nạp mô hình
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result, used_mask = run_tryon(person, garment, mask)
    t_steady = time.time() - t0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pid = pair_id(args.person, args.garment)
    result.save(out / f"{pid}.png")
    used_mask.save(out / f"{pid}_mask.png")
    print(f"Đã lưu {out / (pid + '.png')}")
    print(f"Lần 1 (gồm nạp mô hình): {t_first:.1f}s | lần 2 (tốc độ thật): {t_steady:.1f}s/ảnh | "
          f"VRAM đỉnh: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | GPU: {torch.cuda.get_device_name(0)} | "
          f"độ chính xác: {config.PRECISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
