"""PV1 – gộp các bản trùng tên của nhật ký CatVTON tải từ Drive thành một file, và dựng lại errors_catvton.csv.

Vì sao cần: khi Colab ghi nối (append) vào CSV nằm trên Drive qua lối tắt, Drive có thể tạo NHIỀU file cùng tên
`timing_log_catvton.csv`, mỗi file chỉ có tiêu đề và một dòng. Tải cả thư mục `results/` của Drive về (Drive tự đổi tên
thành `timing_log_catvton (1).csv`, `(2)`...), giải nén vào một thư mục, rồi chạy từ thư mục gốc repo:

    python -m scripts.merge_drive_logs --src D:/tmp/drive_results --done D:/tmp/pv1_final/pv1_done_list.txt

Đầu ra (ghi đè): results/timing_log_catvton.csv (id,seconds,peak_vram_gb, theo thứ tự eval_pairs.csv)
                 results/errors_catvton.csv     (id,error: các cặp không có ảnh kết quả; chỉ ghi khi có --done)
Mọi file có tên bắt đầu bằng `timing_log_catvton` hoặc `timing_log_try6` trong --src đều được đọc (kể cả thư mục con).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from src import config

PREFIXES = ("timing_log_catvton", "timing_log_try6")
HEADER = ["id", "seconds", "peak_vram_gb"]


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("id") and r.get("seconds"):
                yield {k: (r.get(k) or "").strip() for k in HEADER}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="thư mục chứa các CSV tải từ Drive (đã giải nén)")
    ap.add_argument("--pairs", default=str(config.EVAL_PAIRS))
    ap.add_argument("--done", default=None, help="file txt liệt kê tên ảnh có trong outputs/catvton (mỗi dòng một tên)")
    ap.add_argument("--out", default=str(config.RESULTS))
    args = ap.parse_args(argv)

    src = Path(args.src)
    files = sorted(p for p in src.rglob("*.csv") if p.name.startswith(PREFIXES))
    if not files:
        print(f"Không thấy file timing_log_catvton*.csv / timing_log_try6*.csv trong {src}")
        return 1

    with open(args.pairs, newline="", encoding="utf-8-sig") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    wanted = set(ids)

    timing: dict[str, dict] = {}
    for p in files:
        for r in read_rows(p):
            if r["id"] in wanted and r["id"] not in timing:   # giữ lần ghi đầu tiên gặp
                timing[r["id"]] = r

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "timing_log_catvton.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for i in ids:
            if i in timing:
                w.writerow(timing[i])

    secs = [float(r["seconds"]) for r in timing.values()]
    vram = [float(r["peak_vram_gb"]) for r in timing.values() if r["peak_vram_gb"]]
    print(f"Đọc {len(files)} file -> {len(timing)}/{len(ids)} cặp có thời gian.")
    if secs:
        print(f"Giây/ảnh: trung vị {statistics.median(secs):.1f}, trung bình {statistics.mean(secs):.1f}, "
              f"min {min(secs):.1f}, max {max(secs):.1f}. VRAM đỉnh: {max(vram) if vram else 'n/a'} GB")

    if args.done:
        done = {Path(line.strip()).stem for line in open(args.done, encoding="utf-8") if line.strip().endswith(".png")}
        missing = [i for i in ids if i not in done]
        with open(out / "errors_catvton.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "error"])
            for i in missing:
                w.writerow([i, "không có ảnh kết quả sau khi chạy batch"])
        print(f"Ảnh có: {len(ids) - len(missing)}/{len(ids)}; ghi {len(missing)} dòng vào errors_catvton.csv")
        no_time = [i for i in ids if i in done and i not in timing]
        if no_time:
            print(f"Lưu ý: {len(no_time)} ảnh có kết quả nhưng thiếu dòng thời gian, ví dụ {no_time[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
