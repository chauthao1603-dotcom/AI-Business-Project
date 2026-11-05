"""Kiểm tra vai trò của bạn đã có đủ INPUT chưa (tải thiếu gì từ Git/Drive).

Chạy từ thư mục gốc repo:
    python -m scripts.check_inputs PV3
    python -m scripts.check_inputs all
Mỗi dòng: ✓ có đủ, ✗ thiếu hoặc chưa đủ số lượng, ○ chưa bắt buộc ở giai đoạn này.
Nguồn lấy file (Git hay Drive) và người giao: docs/handoff.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config as c

IGNORED = {".gitkeep", "README.md"}


def _count(p: Path) -> int:
    if p.is_file():
        return 1
    if p.is_dir():
        return sum(1 for f in p.rglob("*") if f.is_file() and f.name not in IGNORED)
    return 0


# (nhãn, đường dẫn, số file tối thiểu, bắt buộc?, ai giao, lấy từ đâu)
REQUIRE: dict[str, list[tuple]] = {
    "PV1": [
        ("Ảnh người VITON-HD", c.TEST_IMAGE, 1, True, "PV2 (2.1)", "Drive"),
        ("Ảnh áo VITON-HD", c.TEST_CLOTH, 1, True, "PV2 (2.1)", "Drive"),
        ("agnostic-mask", c.TEST_AGNOSTIC_MASK, 1, True, "PV2 (2.1)", "Drive"),
        ("eval_pairs.csv", c.EVAL_PAIRS, 1, True, "PV2 (2.3)", "Git"),
        ("Bộ demo: ảnh người (≥6)", c.DEMO_PERSONS, 6, False, "PV2 (2.5)", "Drive"),
        ("Bộ demo: ảnh áo (≥10)", c.DEMO_GARMENTS, 10, False, "PV2 (2.5)", "Drive"),
        ("Cặp ảnh ngoài dataset", c.EXTERNAL_PAIRS, 1, False, "PV5 (5.1)", "Git"),
    ],
    "PV2": [
        ("Cache demo (để lọc cặp lỗi)", c.CACHE, 1, False, "PV1 (1.5)", "Drive"),
        ("Mask do PV1 đã dùng", c.OUT_MASKS, 1, False, "PV1 (1.4)", "Drive"),
    ],
    "PV3": [
        ("eval_pairs.csv", c.EVAL_PAIRS, 1, True, "PV2 (2.3)", "Git"),
        ("Ảnh thật (đáp án paired)", c.TEST_IMAGE, 1, True, "PV2 (2.1)", "Drive"),
        ("Ảnh CatVTON (200)", c.OUT_CATVTON, 200, True, "PV1 (1.4)", "Drive"),
        ("Ảnh baseline (200)", c.OUT_BASELINE, 200, True, "PV2 (2.7)", "Drive"),
        ("Mask đã dùng", c.OUT_MASKS, 1, True, "PV1 (1.4)", "Drive"),
        ("Ảnh CatVTON ngoài dataset", c.OUT_EXTERNAL, 1, False, "PV1 (1.7)", "Drive"),
    ],
    "PV4": [
        ("catalog.csv (hoặc bản mẫu)", c.CATALOG if c.CATALOG.exists() else c.CATALOG_SAMPLE, 1, True, "PV2 (2.4)", "Git"),
        ("Ảnh áo cửa hàng", c.GARMENTS, 1, True, "PV2 (2.4)", "Drive"),
        ("Bộ demo: ảnh người", c.DEMO_PERSONS, 1, True, "PV2 (2.5)", "Drive"),
        ("Cache demo", c.CACHE, 1, False, "PV1 (1.5)", "Drive"),
        ("results_summary.csv", c.RESULTS / "results_summary.csv", 1, False, "PV3 (3.2)", "Git"),
    ],
    "PV5": [
        ("Ảnh người ngoài dataset", c.EXTERNAL_PERSONS, 1, False, "chính PV5 (5.1)", "Drive"),
        ("Ảnh áo ngoài dataset", c.EXTERNAL_GARMENTS, 1, False, "chính PV5 (5.1)", "Drive"),
        ("Ảnh kết quả ngoài dataset", c.OUT_EXTERNAL, 1, False, "PV1 (1.7)", "Drive"),
    ],
}


def check(role: str) -> int:
    missing = 0
    print(f"\n== {role} ==  (ROOT = {c.ROOT})")
    for label, path, need, required, giver, where in REQUIRE[role]:
        n = _count(Path(path))
        ok = n >= need
        mark = "✓" if ok else ("✗" if required else "○")
        missing += (not ok) and required
        try:
            shown = Path(path).relative_to(c.ROOT)
        except ValueError:
            shown = Path(path).relative_to(c.REPO_ROOT)
        print(f" {mark} {label:<32} {str(shown):<42} có {n}/{need}  <- {giver}, lấy từ {where}")
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=[*REQUIRE, "all"])
    args = ap.parse_args(argv)
    roles = list(REQUIRE) if args.role == "all" else [args.role]
    missing = sum(check(r) for r in roles)
    print(f"\n{'Đủ input bắt buộc.' if not missing else f'Còn thiếu {missing} mục bắt buộc (✗).'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
