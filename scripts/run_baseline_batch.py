
"""PV2 - Chạy baseline hàng loạt trên eval_pairs.csv."""

from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path

from PIL import Image

from src import config
from src.baseline import run_baseline
from src.utils import stem, load_dataset_mask

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(folder: Path, name: str) -> Path:
    """Trả đường dẫn ảnh cho tên có hoặc không có đuôi; không thấy thì trả folder/name để báo lỗi rõ ràng."""
    direct = folder / name
    if direct.is_file():
        return direct
    for ext in IMG_EXTS:
        for cand in (folder / f"{stem(name)}{ext}", folder / f"{stem(name)}{ext.upper()}"):
            if cand.is_file():
                return cand
    return direct


def run_batch(pairs_file, person_dir, cloth_dir, out_dir,
              mask_dir, error_file, mask_source="dataset", limit=None):

    pairs_file = Path(pairs_file)
    person_dir = Path(person_dir)
    cloth_dir = Path(cloth_dir)
    out_dir = Path(out_dir)
    mask_dir = Path(mask_dir)
    error_file = Path(error_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    errors = []

    # Đọc danh sách cặp ảnh
    with open(pairs_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"person", "cloth"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "eval_pairs.csv phải có các cột person và cloth."
            )

        pairs = list(reader)

    if limit:
        pairs = pairs[:limit]

    print(f"Tổng số cặp cần xử lý: {len(pairs)}")

    for index, pair in enumerate(pairs, start=1):
        person_name = pair["person"].strip()
        cloth_name = pair["cloth"].strip()

        person_stem = stem(person_name)
        cloth_stem = stem(cloth_name)

        pid = (pair.get("id") or "").strip() or f"{person_stem}__{cloth_stem}"
        output_name = f"{pid}.png"
        output_path = out_dir / output_name

        # Bỏ qua ảnh đã có
        if output_path.exists():
            skipped += 1
            print(f"[{index}/{len(pairs)}] Đã tồn tại: {output_name}")
            continue

        try:
            # eval_pairs.csv ghi tên không kèm đuôi (09204_00) -> tìm file theo stem, đuôi nào cũng được
            person_path = find_image(person_dir, person_name)
            cloth_path = find_image(cloth_dir, cloth_name)

            # Kiểm tra file đầu vào
            if not person_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy ảnh người: {person_path}"
                )

            if not cloth_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy ảnh áo: {cloth_path}"
                )

            # Lấy mask theo nguồn dữ liệu
            if mask_source == "dataset":
                mask_path = mask_dir / f"{person_stem}_mask.png"

                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Không tìm thấy mask: {mask_path}"
                    )

                mask = load_dataset_mask(mask_path)

            else:
                raise ValueError(
                    "Hiện tại chỉ hỗ trợ mask_source='dataset'."
                )

            # Đọc ảnh
            with Image.open(person_path) as img:
                person_img = img.copy()

            with Image.open(cloth_path) as img:
                garment_img = img.copy()

            # Chạy baseline
            result = run_baseline(
                person_img,
                garment_img,
                mask
            )

            # Lưu kết quả
            result.convert("RGB").save(output_path, format="PNG")

            success += 1
            print(f"[{index}/{len(pairs)}] Thành công: {output_name}")

        except Exception as exc:
            failed += 1

            errors.append({
                "id": pid,
                "person": person_name,
                "cloth": cloth_name,
                "error": str(exc),
                "traceback": traceback.format_exc()
            })

            print(f"[{index}/{len(pairs)}] Lỗi: {output_name}")
            print(f"    {exc}")

    # Ghi danh sách lỗi
    with open(error_file, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "id", "person", "cloth", "error", "traceback"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    print("\n========== KẾT QUẢ BASELINE ==========")
    print(f"Tổng số cặp: {len(pairs)}")
    print(f"Thành công: {success}")
    print(f"Đã bỏ qua: {skipped}")
    print(f"Thất bại: {failed}")
    print(f"Thư mục kết quả: {out_dir}")
    print(f"File ghi lỗi: {error_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Chạy baseline hàng loạt."
    )

    parser.add_argument(
        "--pairs",
        default=str(config.EVAL_PAIRS)
    )

    parser.add_argument(
        "--person_dir",
        default=str(config.TEST_IMAGE)
    )

    parser.add_argument(
        "--cloth_dir",
        default=str(config.TEST_CLOTH)
    )

    parser.add_argument(
        "--out",
        default=str(config.OUT_BASELINE)
    )

    parser.add_argument(
        "--mask_dir",
        default=str(config.TEST_AGNOSTIC_MASK)
    )

    parser.add_argument(
        "--error_file",
        default=str(config.RESULTS / "errors_baseline.csv")
    )

    parser.add_argument(
        "--mask_source",
        choices=["dataset"],
        default="dataset"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="chỉ chạy N dòng đầu (để thử)"
    )

    args = parser.parse_args()

    run_batch(
        limit=args.limit,
        pairs_file=args.pairs,
        person_dir=args.person_dir,
        cloth_dir=args.cloth_dir,
        out_dir=args.out,
        mask_dir=args.mask_dir,
        error_file=args.error_file,
        mask_source=args.mask_source
    )


if __name__ == "__main__":
    main()
