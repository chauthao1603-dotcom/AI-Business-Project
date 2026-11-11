"""PV3 – 3.6: bảng nhận xét định tính ảnh ngoài dataset (không có ảnh đáp án nên không tính chỉ số).

Bước 1, tạo bảng mẫu từ ảnh PV1 đã sinh trong outputs/external/ (mỗi ảnh một dòng, cột nhận xét để trống):
    python -m eval.external_notes init
Bước 2, mở results/external_notes.csv bằng Excel, xem từng ảnh rồi điền:
    ket_qua   một trong: tot, chap_nhan_duoc, loi
    loai_loi  để trống nếu tot; nếu không, một hoặc nhiều mã cách nhau bằng dấu ;
              sai_tay_ao, mat_hoa_tiet, meo_co_the, sai_mau, loi_tay, loi_mat, lem_vien, khac
    ghi_chu   mô tả ngắn (tiếng Việt có dấu được)
Bước 3, kiểm tra và tổng hợp để đưa vào báo cáo:
    python -m eval.external_notes summary

Chạy lại `init` khi PV1 thêm ảnh: dòng đã điền được giữ nguyên, chỉ thêm dòng cho ảnh mới.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import config

COLS = ["id", "ket_qua", "loai_loi", "ghi_chu"]
KET_QUA = ("tot", "chap_nhan_duoc", "loi")
LOAI_LOI = ("sai_tay_ao", "mat_hoa_tiet", "meo_co_the", "sai_mau", "loi_tay", "loi_mat", "lem_vien", "khac")


def init(external_dir: Path, out: Path) -> pd.DataFrame:
    ids = sorted(p.stem for p in Path(external_dir).glob("*.png") if not p.stem.endswith("_mask"))
    if not ids:
        raise SystemExit(f"Không có ảnh .png nào trong {external_dir}. PV1 cần chạy mục 1.7 trước.")
    old = pd.read_csv(out, dtype=str).fillna("") if out.is_file() else pd.DataFrame(columns=COLS)
    new_ids = [i for i in ids if i not in set(old["id"])]
    table = pd.concat([old, pd.DataFrame({"id": new_ids, "ket_qua": "", "loai_loi": "", "ghi_chu": ""})],
                      ignore_index=True)[COLS]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")  # utf-8-sig để Excel hiện đúng tiếng Việt
    print(f"{out}: {len(table)} dòng ({len(new_ids)} dòng mới).")
    return table


def validate(table: pd.DataFrame) -> list:
    """Danh sách lỗi điền bảng (dòng trống, giá trị lạ, thiếu loại lỗi)."""
    problems = []
    for r in table.fillna("").itertuples(index=False):
        if r.ket_qua not in KET_QUA:
            problems.append(f"{r.id}: ket_qua = {r.ket_qua!r}, cần một trong {KET_QUA}")
            continue
        codes = [c.strip() for c in r.loai_loi.split(";") if c.strip()]
        bad = [c for c in codes if c not in LOAI_LOI]
        if bad:
            problems.append(f"{r.id}: loai_loi lạ {bad}")
        if r.ket_qua != "tot" and not codes:
            problems.append(f"{r.id}: ket_qua = {r.ket_qua} nhưng chưa ghi loai_loi")
    return problems


def summarize(table: pd.DataFrame) -> dict:
    t = table.fillna("")
    counts = t["ket_qua"].value_counts().reindex(KET_QUA, fill_value=0)
    errors = pd.Series([c.strip() for v in t["loai_loi"] for c in v.split(";") if c.strip()], dtype=str)
    return {"n": len(t), "ket_qua": counts.to_dict(),
            "loai_loi": errors.value_counts().to_dict() if len(errors) else {}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bảng nhận xét định tính ảnh ngoài dataset (3.6).")
    ap.add_argument("action", choices=["init", "summary"])
    ap.add_argument("--external", default=str(config.OUT_EXTERNAL))
    ap.add_argument("--out", default=str(config.RESULTS / "external_notes.csv"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.action == "init":
        init(Path(a.external), out)
        return 0
    if not out.is_file():
        raise SystemExit(f"Chưa có {out}. Chạy `python -m eval.external_notes init` trước.")
    table = pd.read_csv(out, dtype=str)
    problems = validate(table)
    for p in problems:
        print("CẦN SỬA:", p)
    s = summarize(table)
    print(f"\n{s['n']} ảnh: " + ", ".join(f"{k} = {v}" for k, v in s["ket_qua"].items()))
    if s["loai_loi"]:
        print("Loại lỗi gặp: " + ", ".join(f"{k} ({v})" for k, v in s["loai_loi"].items()))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
