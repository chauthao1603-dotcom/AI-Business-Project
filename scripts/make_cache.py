"""PV1 – sinh sẵn kết quả cho bộ demo để quay video không phải chờ mô hình (việc 1.5).

Chạy từ thư mục gốc repo, trong môi trường Python 3.9 (GPU):
    python -m scripts.make_cache                   # mọi người × mọi áo trong data/processed/demo/
    python -m scripts.make_cache --limit 6         # thử trước

Với mỗi cặp (người, áo) sinh {out}/{id}.png và {out}/{id}_mask.png (mask do AutoMasker tạo, như khi người dùng
upload ảnh trong app). Chạy lại sẽ bỏ qua cặp đã có. Lỗi ghi vào results/errors_cache.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from scripts.run_batch import IMG_EXTS, append_csv, atomic_save
from src import config
from src.utils import pair_id


def list_images(directory) -> list:
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix in IMG_EXTS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persons", default=str(config.DEMO_PERSONS))
    ap.add_argument("--garments", default=str(config.DEMO_GARMENTS))
    ap.add_argument("--out", default=str(config.CACHE))
    ap.add_argument("--log_dir", default=str(config.RESULTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_consecutive_errors", type=int, default=5)
    args = ap.parse_args(argv)

    persons, garments = list_images(args.persons), list_images(args.garments)
    if not persons or not garments:
        print(f"Chưa có ảnh: {len(persons)} ảnh người trong {args.persons}, {len(garments)} ảnh áo trong {args.garments}.")
        return 1
    combos = [(p, g) for p in persons for g in garments]
    if args.limit:
        combos = combos[: args.limit]
    out_dir, errors_log = Path(args.out), Path(args.log_dir) / "errors_cache.csv"
    run_tryon = config.get_run_tryon()
    print(f"{len(persons)} người × {len(garments)} áo = {len(combos)} cặp | out={out_dir}", flush=True)

    done = skipped = errors = consecutive = 0
    for i, (p, g) in enumerate(combos, 1):
        pid = pair_id(p.name, g.name)
        if (out_dir / f"{pid}.png").exists():
            skipped += 1
            continue
        try:
            t0 = time.time()
            result, mask = run_tryon(Image.open(p), Image.open(g), None)
            atomic_save(mask, out_dir / f"{pid}_mask.png")
            atomic_save(result, out_dir / f"{pid}.png")      # ghi sau cùng
            done += 1
            consecutive = 0
            print(f"[{i}/{len(combos)}] {pid}  {time.time() - t0:.1f}s", flush=True)
        except KeyboardInterrupt:
            print("Đã dừng. Chạy lại cùng lệnh để chạy tiếp.", flush=True)
            raise
        except Exception as e:
            errors += 1
            consecutive += 1
            append_csv(errors_log, ["id", "error"], [pid, f"{type(e).__name__}: {e}"[:500]])
            print(f"[{i}/{len(combos)}] LỖI {pid}: {type(e).__name__}: {e}", flush=True)
            if consecutive >= args.max_consecutive_errors:
                print(f"DỪNG: {consecutive} cặp liên tiếp lỗi. Xem {errors_log}.", flush=True)
                return 1
    print(f"Xong: mới sinh {done}, bỏ qua {skipped}, lỗi {errors}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
