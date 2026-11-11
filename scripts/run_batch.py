"""PV1 – sinh ảnh CatVTON hàng loạt cho một danh sách cặp (việc 1.4, và 1.7 với ảnh ngoài dataset).

Chạy từ thư mục gốc repo, trong môi trường Python 3.9 đã cài requirements-model.txt (GPU):
    python -m scripts.run_batch --limit 5                      # thử 5 dòng đầu trước
    python -m scripts.run_batch                                # chạy toàn bộ eval_pairs.csv
    python -m scripts.run_batch --pairs data/processed/external_pairs.csv \\
        --person_dir data/processed/external/persons --cloth_dir data/processed/external/garments \\
        --out outputs/external --mask_source auto --tag catvton_external

Mặc định (lấy từ src/config.py, nên đặt TRYON_ROOT trên Colab): đọc data/processed/eval_pairs.csv, ảnh trong
data/raw/test/{image,cloth}, mask sẵn của VITON-HD trong data/raw/test/agnostic-mask/{tên}_mask.png.

Đầu ra:  {out}/{id}.png       ảnh try-on (id = "{tên người}__{tên áo}")
         {masks}/{tên người}.png   mask đã dùng
         results/timing_log_{tag}.csv   id,seconds,peak_vram_gb
         results/errors_{tag}.csv       id,error
Chạy lại lệnh cũ sẽ BỎ QUA cặp đã có ảnh (an toàn khi Colab ngắt phiên). Ảnh được ghi qua file tạm rồi đổi tên, và ảnh
kết quả được ghi SAU cùng nên cặp dở dang không bị tính là xong. Dừng hẳn nếu có N cặp liên tiếp lỗi (thường là lỗi môi trường).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from PIL import Image

from src import config
from src.utils import load_dataset_mask, stem

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG")


def find_image(directory, name) -> Path:
    """Tìm ảnh theo tên (có hoặc không có đuôi) trong thư mục."""
    d = Path(directory)
    direct = d / str(name)
    if direct.suffix and direct.is_file():
        return direct
    for ext in IMG_EXTS:
        p = d / f"{name}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Không thấy ảnh '{name}' trong {d}")


def append_csv(path, header, row) -> None:
    """Ghi thêm một dòng vào CSV (tạo file và tiêu đề nếu chưa có); ghi ngay nên không mất khi bị ngắt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def atomic_save(img: Image.Image, path) -> None:
    """Ghi PNG qua file tạm rồi đổi tên: file đích hoặc có đủ hoặc không có."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def _vram_reset() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _vram_peak_gb() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.max_memory_allocated() / 1e9:.2f}"
    except ImportError:
        pass
    return ""


def read_pairs(path, mode=None, limit=None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"id", "person", "cloth"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} thiếu cột {sorted(missing)}; cần các cột id,person,cloth,mode")
        rows = list(reader)
    if mode:
        rows = [r for r in rows if r.get("mode") == mode]
    return rows[:limit] if limit else rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(config.EVAL_PAIRS))
    ap.add_argument("--person_dir", default=str(config.TEST_IMAGE))
    ap.add_argument("--cloth_dir", default=str(config.TEST_CLOTH))
    ap.add_argument("--out", default=str(config.OUT_CATVTON))
    ap.add_argument("--masks", default=str(config.OUT_MASKS), help="nơi lưu mask đã dùng")
    ap.add_argument("--mask_source", choices=["dataset", "auto"], default="dataset",
                    help="dataset: mask sẵn của VITON-HD (200 cặp đánh giá); auto: AutoMasker tự tạo (ảnh ngoài dataset)")
    ap.add_argument("--mask_dir", default=str(config.TEST_AGNOSTIC_MASK))
    ap.add_argument("--log_dir", default=str(config.RESULTS))
    ap.add_argument("--tag", default="catvton", help="tên nhật ký: timing_log_<tag>.csv, errors_<tag>.csv")
    ap.add_argument("--mode", choices=["paired", "unpaired", "external"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N dòng đầu (để thử)")
    ap.add_argument("--max_consecutive_errors", type=int, default=5)
    args = ap.parse_args(argv)

    rows = read_pairs(args.pairs, args.mode, args.limit)
    out_dir, masks_dir, mask_dir = Path(args.out), Path(args.masks), Path(args.mask_dir)
    timing_log = Path(args.log_dir) / f"timing_log_{args.tag}.csv"
    errors_log = Path(args.log_dir) / f"errors_{args.tag}.csv"
    run_tryon = config.get_run_tryon()
    print(f"{len(rows)} cặp | out={out_dir} | mask_source={args.mask_source} | USE_MOCK={config.use_mock()}", flush=True)

    done = skipped = errors = consecutive = 0
    times = []
    for i, row in enumerate(rows, 1):
        pid, out_path = row["id"], out_dir / f"{row['id']}.png"
        if out_path.exists():
            skipped += 1
            continue
        try:
            person = Image.open(find_image(args.person_dir, row["person"]))
            cloth = Image.open(find_image(args.cloth_dir, row["cloth"]))
            mask = None
            if args.mask_source == "dataset":
                mp = mask_dir / f"{stem(row['person'])}_mask.png"
                if not mp.is_file():
                    raise FileNotFoundError(f"Không thấy mask {mp}")
                mask = load_dataset_mask(mp)
            _vram_reset()
            t0 = time.time()
            result, used_mask = run_tryon(person, cloth, mask)
            secs = time.time() - t0
            mask_path = masks_dir / f"{stem(row['person'])}.png"
            if not mask_path.exists():
                atomic_save(used_mask, mask_path)
            atomic_save(result, out_path)                 # ghi sau cùng: có ảnh kết quả mới tính là xong
            append_csv(timing_log, ["id", "seconds", "peak_vram_gb"], [pid, f"{secs:.2f}", _vram_peak_gb()])
            times.append(secs)
            done += 1
            consecutive = 0
            eta_min = (len(rows) - i) * (sum(times) / len(times)) / 60
            print(f"[{i}/{len(rows)}] {pid}  {secs:.1f}s  (còn ~{eta_min:.0f} phút)", flush=True)
        except KeyboardInterrupt:
            print("Đã dừng theo yêu cầu. Chạy lại cùng lệnh để chạy tiếp.", flush=True)
            raise
        except Exception as e:                             # lỗi của MỘT cặp: ghi lại rồi chạy tiếp
            errors += 1
            consecutive += 1
            append_csv(errors_log, ["id", "error"], [pid, f"{type(e).__name__}: {e}"[:500]])
            print(f"[{i}/{len(rows)}] LỖI {pid}: {type(e).__name__}: {e}", flush=True)
            if consecutive >= args.max_consecutive_errors:
                print(f"DỪNG: {consecutive} cặp liên tiếp lỗi, thường là lỗi môi trường (GPU, thư viện, đường dẫn). "
                      f"Xem {errors_log}.", flush=True)
                return 1
    print(f"Xong: mới sinh {done}, bỏ qua {skipped} (đã có), lỗi {errors}. Nhật ký: {timing_log}, {errors_log}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
