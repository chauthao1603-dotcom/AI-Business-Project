"""Dùng: python scripts/make_subset.py <eval_pairs.csv> <thư mục data/raw> <thư mục xuất>"""
import csv, shutil, sys
from pathlib import Path

pairs_csv, raw, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = list(csv.DictReader(open(pairs_csv, encoding="utf-8-sig")))


def find(folder, stem, names):
    for n in names:
        p = folder / n.format(stem=stem)
        if p.exists():
            return p
    return None


need = {"image": set(), "cloth": set(), "agnostic-mask": set()}
for r in rows:
    need["image"].add(r["person"]);
    need["cloth"].add(r["cloth"]);
    need["agnostic-mask"].add(r["person"])

missing = []
for sub, stems in need.items():
    dst = out / "test" / sub
    dst.mkdir(parents=True, exist_ok=True)
    pats = ["{stem}_mask.png"] if sub == "agnostic-mask" else ["{stem}.jpg", "{stem}.jpeg", "{stem}.png"]
    for s in sorted(stems):
        src = find(raw / "test" / sub, s, pats)
        if src is None:
            missing.append((sub, s));
            continue
        shutil.copy2(src, dst / src.name)
    print(sub, ":", len(list(dst.iterdir())), "file")
print("THIẾU:", missing if missing else "không")
if not missing:
    print("Đã tạo", shutil.make_archive(str(out), "zip", out))