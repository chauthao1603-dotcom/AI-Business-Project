"""PV3 – 3.3: dựng bộ ảnh khảo sát "thật hay AI" và bộ so sánh A/B.

Chạy từ thư mục gốc repo (sau khi PV1 và PV2 đã giao ảnh):

    python -m eval.make_survey_set

Phần 1 (q01–q20): 10 ảnh thật + 10 ảnh CatVTON, 20 người khác nhau, xáo thứ tự.
  - Ảnh AI lấy ngẫu nhiên (seed 42) từ các cặp unpaired có ảnh CatVTON, KHÔNG chọn lọc ảnh đẹp.
  - Mỗi người chỉ xuất hiện một lần trong 20 ảnh, và người ở ảnh thật khác người ở ảnh AI,
    để người xem không so được ảnh thật với ảnh AI của cùng một người.
  - Mọi ảnh đi qua cùng một quy trình (về 768×1024, nén JPEG rồi lưu PNG, bỏ metadata) để không đoán được
    qua định dạng. Ảnh thật của VITON-HD vốn là JPEG còn ảnh CatVTON là PNG, nên bước nén JPEG chung
    là để xóa dấu hiệu đó.
Phần 2 (ab01–ab10): 10 cặp baseline và CatVTON cùng id, vị trí A/B ngẫu nhiên.
  Google Forms chỉ gắn được một ảnh vào mỗi câu hỏi, nên mỗi cặp còn được ghép thành một ảnh
  outputs/survey/ab_sheets/abXX.png: ảnh A bên trái, ảnh B bên phải, nhãn "A" và "B" ở phía trên.

Đầu ra:
    outputs/survey/q01.png … q20.png, ab01_A.png, ab01_B.png …   [Drive, không commit]
    outputs/survey/ab_sheets/ab01.png … ab10.png                  [Drive, không commit]
    results/survey_key.csv   qid,label,source_id        label ∈ {real, ai}
    results/ab_key.csv       ab_id,id,left_method,right_method   (A = trái = left_method, B = phải)

Chạy lại sẽ thay bộ ảnh; nếu Google Form đã dựng từ bộ cũ thì đáp án sẽ sai, nên lệnh từ chối ghi đè
khi results/survey_key.csv đã có, trừ khi thêm --force.

Bộ khảo sát đã dựng từ trước (chưa có ab_sheets/) thì chỉ cần ghép ảnh, không chọn lại ảnh và không đổi đáp án:

    python -m eval.make_survey_set --ab-sheets
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from eval.run_eval import IMG_EXTS, find_image, read_pairs
from src import config, utils

N_REAL = N_AI = 10
N_AB = 10
AB_SHEET_DIR = "ab_sheets"   # thư mục con của outputs/survey/ chứa ảnh ghép A|B
SHEET_GAP = 40               # khoảng trắng giữa ảnh A và ảnh B (px)
SHEET_LABEL_H = 120          # dải trắng phía trên để ghi nhãn A, B (px)


def normalize(img: Image.Image, jpeg_quality: int) -> Image.Image:
    """Về 768×1024 RGB; nếu jpeg_quality > 0 thì nén-giải nén JPEG một lần để mọi ảnh cùng "lịch sử nén"."""
    img = utils.standardize_image(img)
    if jpeg_quality:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=jpeg_quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def save_clean(img: Image.Image, path: Path) -> None:
    """Lưu PNG không kèm metadata (ảnh mới không thừa hưởng thông tin từ file gốc)."""
    clean = Image.new("RGB", img.size)
    clean.paste(img)
    clean.save(path, "PNG")


def _label_font(size: int):
    """Font đậm cho nhãn A/B: Arial trên Windows, DejaVu trên Linux/Colab, không có thì dùng font mặc định."""
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def make_ab_sheets(out) -> list:
    """Ghép abXX_A.png (trái) và abXX_B.png (phải) thành outputs/survey/ab_sheets/abXX.png.

    Chỉ đọc ảnh A/B đã có, không chọn lại ảnh và không chạm vào file đáp án, nên chạy lại bao nhiêu lần
    cũng không làm sai Google Form. Hai ảnh giữ nguyên kích thước và điểm ảnh, chỉ thêm dải nhãn và khoảng trắng.
    """
    out = Path(out)
    a_paths = sorted(out.glob("ab[0-9][0-9]_A.png"))
    if not a_paths:
        raise SystemExit(f"Không thấy ảnh ab01_A.png … trong {out}. Chạy python -m eval.make_survey_set trước.")
    dest_dir = out / AB_SHEET_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    font = _label_font(int(SHEET_LABEL_H * 0.7))
    written = []
    for a_path in a_paths:
        ab_id = a_path.stem[:-2]                      # "ab01_A" -> "ab01"
        b_path = out / f"{ab_id}_B.png"
        if not b_path.exists():
            raise SystemExit(f"Thiếu {b_path.name} đi cùng {a_path.name}.")
        with Image.open(a_path) as a, Image.open(b_path) as b:
            a, b = a.convert("RGB"), b.convert("RGB")
        if a.size != b.size:
            raise SystemExit(f"{a_path.name} và {b_path.name} khác kích thước: {a.size} và {b.size}.")
        w, h = a.size
        sheet = Image.new("RGB", (2 * w + SHEET_GAP, h + SHEET_LABEL_H), "white")
        sheet.paste(a, (0, SHEET_LABEL_H))
        sheet.paste(b, (w + SHEET_GAP, SHEET_LABEL_H))
        draw = ImageDraw.Draw(sheet)
        for text, cx in (("A", w / 2), ("B", w + SHEET_GAP + w / 2)):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            draw.text((cx - (left + right) / 2, SHEET_LABEL_H / 2 - (top + bottom) / 2), text,
                      fill="black", font=font)
        dest = dest_dir / f"{ab_id}.png"
        save_clean(sheet, dest)
        written.append(dest)
    return written


def choose(args) -> dict:
    pairs = read_pairs(Path(args.pairs))
    unpaired = pairs[pairs["mode"] == "unpaired"].sort_values("id")
    cat_dir, base_dir, gt_dir = Path(args.catvton), Path(args.baseline), Path(args.gt)
    rng = random.Random(config.SEED)

    # ---- ảnh AI: mỗi người một ảnh
    pool = [(r.id, r.person) for r in unpaired.itertuples(index=False) if find_image(cat_dir, r.id)]
    rng.shuffle(pool)
    ai, used_persons = [], set()
    for pid, person in pool:
        if person not in used_persons:
            ai.append((pid, person))
            used_persons.add(person)
        if len(ai) == N_AI:
            break
    if len(ai) < N_AI:
        raise SystemExit(f"Cần {N_AI} ảnh CatVTON của {N_AI} người khác nhau (cặp unpaired) trong {cat_dir}, "
                         f"hiện chỉ có {len(ai)}.")

    # ---- ảnh thật: người khác với các người ở ảnh AI
    gt_stems = sorted({p.stem for p in gt_dir.glob("*") if p.suffix.lower() in IMG_EXTS} - used_persons)
    if len(gt_stems) < N_REAL:
        raise SystemExit(f"Cần {N_REAL} ảnh thật khác người trong {gt_dir}, hiện chỉ có {len(gt_stems)}.")
    real = rng.sample(gt_stems, N_REAL)

    items = [("real", s, find_image(gt_dir, s)) for s in real] + \
            [("ai", pid, find_image(cat_dir, pid)) for pid, _ in ai]
    rng.shuffle(items)

    # ---- A/B: cặp có cả hai ảnh, tránh trùng id với phần 1
    quiz_ids = {pid for pid, _ in ai}
    both = [r.id for r in unpaired.itertuples(index=False)
            if r.id not in quiz_ids and find_image(cat_dir, r.id) and find_image(base_dir, r.id)]
    if len(both) < N_AB:
        raise SystemExit(f"Cần {N_AB} cặp unpaired có cả ảnh baseline và CatVTON "
                         f"(trong {base_dir} và {cat_dir}), hiện chỉ có {len(both)}.")
    ab_ids = rng.sample(both, N_AB)
    ab = []
    for k, pid in enumerate(ab_ids, 1):
        cat_left = rng.random() < 0.5
        left, right = ("catvton", "baseline") if cat_left else ("baseline", "catvton")
        ab.append({"ab_id": f"ab{k:02d}", "id": pid, "left_method": left, "right_method": right})
    return {"items": items, "ab": ab, "cat_dir": cat_dir, "base_dir": base_dir}


def build(args) -> dict:
    results, out = Path(args.results), Path(args.out)
    key_path, ab_path = results / "survey_key.csv", results / "ab_key.csv"
    if key_path.exists() and not args.force:
        raise SystemExit(f"{key_path} đã có. Chạy lại sẽ đổi bộ ảnh và làm sai Google Form đã dựng. "
                         f"Nếu chắc chắn, thêm --force.")
    sel = choose(args)
    out.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    key_rows, sizes = [], {"real": [], "ai": []}
    for k, (label, source_id, path) in enumerate(sel["items"], 1):
        qid = f"q{k:02d}"
        dest = out / f"{qid}.png"
        save_clean(normalize(Image.open(path), args.jpeg_quality), dest)
        key_rows.append({"qid": qid, "label": label, "source_id": source_id})
        sizes[label].append(dest.stat().st_size)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)

    for row in sel["ab"]:
        for side, method in (("A", row["left_method"]), ("B", row["right_method"])):
            src_dir = sel["cat_dir"] if method == "catvton" else sel["base_dir"]
            save_clean(normalize(Image.open(find_image(src_dir, row["id"])), args.jpeg_quality),
                       out / f"{row['ab_id']}_{side}.png")
    pd.DataFrame(sel["ab"]).to_csv(ab_path, index=False)
    sheets = make_ab_sheets(out)
    return {"key": pd.DataFrame(key_rows), "ab": pd.DataFrame(sel["ab"]), "sizes": sizes, "sheets": sheets}


def check_uniform(out: Path) -> dict:
    """Mọi ảnh khảo sát phải cùng chế độ màu và kích thước."""
    seen = set()
    for p in sorted(Path(out).glob("*.png")):
        with Image.open(p) as im:
            seen.add((im.mode, im.size))
    return {"formats": seen, "uniform": len(seen) == 1}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Dựng bộ ảnh khảo sát và bộ so sánh A/B.")
    ap.add_argument("--pairs", default=str(config.EVAL_PAIRS))
    ap.add_argument("--gt", default=str(config.TEST_IMAGE))
    ap.add_argument("--catvton", default=str(config.OUT_CATVTON))
    ap.add_argument("--baseline", default=str(config.OUT_BASELINE))
    ap.add_argument("--out", default=str(config.OUT_SURVEY))
    ap.add_argument("--results", default=str(config.RESULTS))
    ap.add_argument("--jpeg-quality", type=int, default=90, help="0 = không nén JPEG chung")
    ap.add_argument("--force", action="store_true", help="cho phép ghi đè bộ khảo sát đã có")
    ap.add_argument("--ab-sheets", action="store_true",
                    help="chỉ ghép ảnh A|B cho bộ đã có (không chọn lại ảnh, không đổi đáp án)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.ab_sheets:
        sheets = make_ab_sheets(args.out)
        print(f"Đã ghép {len(sheets)} ảnh A|B vào {Path(args.out) / AB_SHEET_DIR} "
              f"(A = trái, B = phải; đáp án giữ nguyên trong results/ab_key.csv)")
        return 0
    res = build(args)
    chk = check_uniform(Path(args.out))
    key = res["key"]
    print(f"Đã tạo {len(key)} ảnh q01–q{len(key):02d} ({(key.label == 'real').sum()} thật, "
          f"{(key.label == 'ai').sum()} AI) và {len(res['ab'])} cặp A/B trong {args.out} "
          f"(ảnh ghép A|B cho Google Form: {Path(args.out) / AB_SHEET_DIR})")
    print(f"Định dạng ảnh: {sorted(chk['formats'])} -> {'ĐỒNG NHẤT' if chk['uniform'] else 'KHÔNG ĐỒNG NHẤT'}")
    for label, sz in res["sizes"].items():
        print(f"  dung lượng file trung bình, ảnh {label}: {sum(sz) / len(sz) / 1024:.0f} KB")
    print("Nếu hai mức dung lượng chênh nhau nhiều, thử chạy lại với --jpeg-quality khác rồi so lại.")
    print(f"Đáp án: {Path(args.results) / 'survey_key.csv'}, {Path(args.results) / 'ab_key.csv'} (chỉ nhóm giữ, không đưa cho người làm khảo sát)")
    return 0 if chk["uniform"] else 1


if __name__ == "__main__":
    sys.exit(main())
