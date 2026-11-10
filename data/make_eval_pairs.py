
from pathlib import Path
import csv
import random

# Đường dẫn dự án
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
TEST_DIR = RAW_DIR / "test"

IMAGE_DIR = TEST_DIR / "image"
CLOTH_DIR = TEST_DIR / "cloth"
PAIRS_FILE = RAW_DIR / "test_pairs.txt"

OUTPUT_FILE = PROJECT_DIR / "data" / "eval_pairs.csv"

# Cấu hình
SEED = 42
NUM_PAIRED = 100
NUM_UNPAIRED = 100

random.seed(SEED)


def get_stems(folder):
    return {
        file.stem
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }


def main():
    # Kiểm tra thư mục
    for folder in [IMAGE_DIR, CLOTH_DIR]:
        if not folder.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc: {folder}"
            )

    image_stems = get_stems(IMAGE_DIR)
    cloth_stems = get_stems(CLOTH_DIR)

    # Tạo các cặp paired: cùng stem
    paired_stems = sorted(image_stems & cloth_stems)

    if len(paired_stems) < NUM_PAIRED:
        raise ValueError("Khong du anh de tao 100 cap paired.")

    selected_paired = random.sample(
        paired_stems, NUM_PAIRED
    )

    rows = []

    for stem in selected_paired:
        rows.append({
            "id": f"{stem}__{stem}",
            "person": stem,
            "cloth": stem,
            "mode": "paired"
        })

    # Đọc test_pairs.txt để tạo cặp unpaired
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"Khong tim thay: {PAIRS_FILE}"
        )

    unpaired_candidates = []

    with PAIRS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            person_stem = Path(parts[0]).stem
            cloth_stem = Path(parts[1]).stem

            # Chỉ nhận cặp có hai stem khác nhau
            if (
                person_stem != cloth_stem
                and person_stem in image_stems
                and cloth_stem in cloth_stems
            ):
                unpaired_candidates.append(
                    (person_stem, cloth_stem)
                )

    # Loại bỏ cặp trùng
    unpaired_candidates = sorted(
        set(unpaired_candidates)
    )

    if len(unpaired_candidates) < NUM_UNPAIRED:
        raise ValueError(
            "Khong du 100 cap unpaired hop le."
        )

    selected_unpaired = random.sample(
        unpaired_candidates, NUM_UNPAIRED
    )

    for person_stem, cloth_stem in selected_unpaired:
        rows.append({
            "id": f"{person_stem}__{cloth_stem}",
            "person": person_stem,
            "cloth": cloth_stem,
            "mode": "unpaired"
        })

    # Kiểm tra ID không trùng
    ids = [row["id"] for row in rows]

    if len(ids) != len(set(ids)):
        raise ValueError("Phat hien ID bi trung!")

    # Ghi CSV
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "person", "cloth", "mode"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("TAO EVAL PAIRS THANH CONG!")
    print(f"Tong so cap: {len(rows)}")
    print(f"Paired: {NUM_PAIRED}")
    print(f"Unpaired: {NUM_UNPAIRED}")
    print(f"Seed: {SEED}")
    print(f"File da luu: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
