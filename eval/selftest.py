"""PV3 – phép thử bắt buộc trước khi chạy run_eval với dữ liệu thật (docs/PHAN_VIEC_v2.md mục 3.2).

    python -m eval.selftest --gt data/raw/test/image --n 8

(a) pred = gt           -> MSE = RMSE = MAE = 0, SSIM = 1, LPIPS ≈ 0
(b) pred = ảnh đen      -> cả ba nhóm chỉ số đều xấu rõ rệt
Chỉ dùng thư mục tạm, không ghi vào results/. Thoát với mã 0 nếu cả hai phép thử đạt, 1 nếu không.
Mask: dùng file trong --mask-dir nếu có ({stem}.png hoặc {stem}_mask.png), nếu không dùng mask hình chữ nhật giả
phủ vùng thân (chỉ để thử, không ảnh hưởng kết luận của hai phép thử).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from eval import run_eval as ev
from src import config


def _mask_for(stem: str, mask_dir: Optional[Path]) -> Image.Image:
    if mask_dir:
        for name in (f"{stem}.png", f"{stem}_mask.png"):
            p = mask_dir / name
            if p.is_file():
                return Image.open(p)
    m = np.zeros((config.IMG_H, config.IMG_W), dtype=np.uint8)
    m[180:930, 85:590] = 255
    return Image.fromarray(m)


def prepare(gt_dir: Path, n: int, mask_dir: Optional[Path], work: Path):
    """Tạo eval_pairs.csv, masks/, pred_gt/, pred_black/ trong thư mục tạm."""
    files = sorted(p for p in gt_dir.glob("*") if p.suffix.lower() in ev.IMG_EXTS)[:n]
    if len(files) < 2:
        raise SystemExit(f"Cần ít nhất 2 ảnh trong {gt_dir} (tìm thấy {len(files)}).")
    for sub in ("masks", "pred_gt", "pred_black"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    black = Image.new("RGB", config.IMG_SIZE, (0, 0, 0))
    for p in files:
        stem = p.stem
        pid = f"{stem}__{stem}"
        rows.append({"id": pid, "person": stem, "cloth": stem, "mode": "paired"})
        Image.fromarray(ev.load_rgb(p)).save(work / "pred_gt" / f"{pid}.png")
        black.save(work / "pred_black" / f"{pid}.png")
        ev.utils.standardize_mask(_mask_for(stem, mask_dir)).save(work / "masks" / f"{stem}.png")
    pairs = work / "eval_pairs.csv"
    pd.DataFrame(rows).to_csv(pairs, index=False)
    return pairs


def _get(summary: pd.DataFrame, region: str, metric: str) -> float:
    row = summary[(summary.region == region) & (summary.metric == metric)]
    return float(row["mean"].iloc[0]) if len(row) else float("nan")


def evaluate(gt_dir, n=8, mask_dir=None, skip_lpips=False, device="cpu", lpips_fn=None) -> list:
    """Chạy hai phép thử, trả về danh sách (tên kiểm tra, giá trị, điều kiện, đạt?)."""
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        pairs = prepare(Path(gt_dir), n, Path(mask_dir) if mask_dir else None, work)
        # Dùng chung một mô hình LPIPS cho cả hai phép thử để khỏi nạp hai lần.
        if lpips_fn is None and not skip_lpips:
            lpips_fn = ev.make_lpips_fn(device)

        def go(pred, method):
            args = Namespace(pairs=str(pairs), gt=str(gt_dir), pred=str(work / pred), masks=str(work / "masks"),
                             out=str(work / "out"), method=method, device=device, kid_subset_size=50,
                             limit=0, skip_lpips=skip_lpips, skip_fid=True)
            return ev.run(args, lpips_fn=lpips_fn)["summary"]

        a = go("pred_gt", "catvton")
        b = go("pred_black", "baseline")
        for region in ("full", "mask"):
            for m in ("mse", "rmse", "mae"):
                v = _get(a, region, m)
                checks.append((f"(a) pred=gt, {region}, {m}", v, "= 0", v == 0.0))
            v = _get(a, region, "ssim")
            checks.append((f"(a) pred=gt, {region}, ssim", v, "≥ 0.9999", v >= 0.9999))
            v = _get(b, region, "mse")
            checks.append((f"(b) ảnh đen, {region}, mse", v, "> 1000", v > 1000))
            v = _get(b, region, "ssim")
            checks.append((f"(b) ảnh đen, {region}, ssim", v, "< 0.5", v < 0.5))
            if not skip_lpips:
                v = _get(a, region, "lpips")
                checks.append((f"(a) pred=gt, {region}, lpips", v, "< 0.001", v < 1e-3))
                v = _get(b, region, "lpips")
                checks.append((f"(b) ảnh đen, {region}, lpips", v, "> 0.2", v > 0.2))
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hai phép thử trước khi chạy run_eval với dữ liệu thật.")
    ap.add_argument("--gt", default=str(config.TEST_IMAGE))
    ap.add_argument("--n", type=int, default=8, help="số ảnh dùng để thử")
    ap.add_argument("--mask-dir", default=None, help="thư mục agnostic-mask thật (không bắt buộc)")
    ap.add_argument("--skip-lpips", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    checks = evaluate(a.gt, a.n, a.mask_dir, a.skip_lpips, a.device)
    width = max(len(c[0]) for c in checks)
    for name, v, cond, ok in checks:
        print(f"{'ĐẠT ' if ok else 'LỖI '} {name:<{width}}  = {v:.6f}   (cần {cond})")
    passed = all(c[3] for c in checks)
    if not passed:
        print("\n=> CHƯA ĐẠT, không chạy dữ liệu thật; xem các dòng LỖI.")
    elif a.skip_lpips:
        print("\n=> Các chỉ số điểm ảnh và SSIM đạt, nhưng LPIPS CHƯA được thử (đang --skip-lpips). "
              "Chạy lại không kèm --skip-lpips trên máy có torch trước khi chạy dữ liệu thật.")
    else:
        print("\n=> CẢ HAI PHÉP THỬ ĐẠT, có thể chạy dữ liệu thật.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
