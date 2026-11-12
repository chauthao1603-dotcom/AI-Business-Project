"""PV3 – 3.2: tính chỉ số định lượng cho MỘT phương pháp (baseline hoặc catvton).

Chạy từ thư mục gốc repo:

    python -m eval.run_eval --pairs data/processed/eval_pairs.csv --gt data/raw/test/image \
        --pred outputs/catvton --method catvton --masks outputs/masks

Chạy lần lượt cho --method baseline và --method catvton: hai lần chạy ghi chung vào cùng hai file CSV
(mỗi lần chỉ thay các dòng của chính method đó, không xóa dòng của method kia).

Paired (có ảnh đáp án): MSE, RMSE, MAE, SSIM, LPIPS, mỗi cặp tính hai bản:
    region=full  toàn ảnh
    region=mask  MSE/RMSE/MAE trên điểm ảnh trong mask; SSIM/LPIPS trên khung chữ nhật bao quanh mask
Unpaired (không có đáp án): FID và KID giữa tập ảnh sinh và tập ảnh thật trong --gt.

Đầu ra (schema như docs/PHAN_VIEC_v2.md mục 1.5):
    results/results_per_pair.csv   id,mode,method,region,mse,rmse,mae,ssim,lpips
    results/results_summary.csv    method,mode,region,metric,mean,std,n
    results/eval_missing_<method>.csv   id,reason   (cặp bị bỏ qua vì thiếu file)

Thang tính theo bảng 1.6: MSE/RMSE/MAE trên thang 0–255; SSIM data_range=255; LPIPS mạng 'alex' với ảnh
đổi về [-1, 1]. Phép thử bắt buộc trước khi chạy dữ liệu thật: python -m eval.selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

from src import config, utils

PER_PAIR_COLS = ["id", "mode", "method", "region", "mse", "rmse", "mae", "ssim", "lpips"]
SUMMARY_COLS = ["method", "mode", "region", "metric", "mean", "std", "n"]
PAIR_METRICS = ["mse", "rmse", "mae", "ssim", "lpips"]
IMG_EXTS = (".png", ".jpg", ".jpeg")
METHODS = ("baseline", "catvton")

LpipsFn = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------- đọc file
def find_image(folder: Path, stem: str) -> Optional[Path]:
    """Tìm {folder}/{stem}.(png|jpg|jpeg); trả None nếu không có."""
    for ext in IMG_EXTS:
        p = Path(folder) / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def load_rgb(path) -> np.ndarray:
    """Ảnh bất kỳ -> uint8 (1024, 768, 3) theo chuẩn 1.3."""
    return np.asarray(utils.standardize_image(Image.open(path)), dtype=np.uint8)


def load_mask(path) -> np.ndarray:
    """Mask do PV1 lưu (255 = vùng áo đã bị thay) -> bool (1024, 768)."""
    return np.asarray(utils.standardize_mask(Image.open(path))) == 255


# ---------------------------------------------------------------- chỉ số
def pixel_errors(gt: np.ndarray, pred: np.ndarray, sel: Optional[np.ndarray] = None) -> dict:
    """MSE, RMSE, MAE trên thang 0–255. `sel` (bool HxW) chỉ giữ điểm ảnh trong mask."""
    a = gt.astype(np.float64)
    b = pred.astype(np.float64)
    if sel is not None:
        a, b = a[sel], b[sel]
    if a.size == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan}
    d = a - b
    mse = float(np.mean(d * d))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(d)))}


def bbox(mask: np.ndarray):
    """Khung chữ nhật bao quanh vùng True: (y0, y1, x0, x1), cận trên không gồm; None nếu mask rỗng."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def ssim_value(gt: np.ndarray, pred: np.ndarray) -> float:
    """SSIM, data_range=255, ảnh màu (channel_axis=2). Tự thu nhỏ cửa sổ nếu ảnh cắt quá nhỏ."""
    short = min(gt.shape[0], gt.shape[1])
    if short < 3:
        return float("nan")
    win = min(7, short if short % 2 == 1 else short - 1)
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255, win_size=win))


def make_lpips_fn(device: str = "cpu") -> LpipsFn:
    """LPIPS mạng 'alex'. Ảnh uint8 HxWx3 được đổi về [-1, 1] (bảng 1.6). Cần torch và gói lpips."""
    import lpips
    import torch

    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def to_tensor(arr: np.ndarray):
        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    def fn(gt: np.ndarray, pred: np.ndarray) -> float:
        with torch.no_grad():
            return float(model(to_tensor(gt), to_tensor(pred)).item())

    return fn


def pair_metrics(gt: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray],
                 lpips_fn: Optional[LpipsFn] = None) -> dict:
    """Trả {region: {mse, rmse, mae, ssim, lpips}}. Không có mask thì chỉ có region 'full'."""
    out = {}

    def one(g, p, sel=None):
        m = pixel_errors(g, p, sel)
        m["ssim"] = ssim_value(g, p)
        m["lpips"] = float(lpips_fn(g, p)) if lpips_fn is not None else np.nan
        return m

    full = one(gt, pred)
    out["full"] = full
    if mask is not None:
        box = bbox(mask)
        if box is None:  # mask rỗng
            out["mask"] = {k: np.nan for k in PAIR_METRICS}
        else:
            y0, y1, x0, x1 = box
            crop = one(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1])
            # MSE/RMSE/MAE chỉ tính trên điểm ảnh trong mask (không phải cả khung bao)
            crop.update(pixel_errors(gt, pred, mask))
            out["mask"] = crop
    return out


def fid_kid(real_files, gen_files, device: str = "cpu", subset_size: int = 50,
            batch_size: int = 32) -> dict:
    """FID và KID (Inception-v3, đặc trưng 2048 chiều) giữa ảnh thật và ảnh sinh. Cần torchmetrics, torch-fidelity."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    torch.manual_seed(config.SEED)
    subset = max(2, min(subset_size, len(real_files), len(gen_files)))
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(feature=2048, subset_size=subset, subsets=100).to(device)
    for files, is_real in ((real_files, True), (gen_files, False)):
        for i in range(0, len(files), batch_size):
            arr = np.stack([load_rgb(f) for f in files[i:i + batch_size]])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)  # uint8, N×3×H×W
            fid.update(x, real=is_real)
            kid.update(x, real=is_real)
    kid_mean, kid_std = kid.compute()
    return {"fid": float(fid.compute()), "kid_mean": float(kid_mean), "kid_std": float(kid_std),
            "kid_subset_size": subset}


# ---------------------------------------------------------------- ghép bảng
def summarize(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Gộp results_per_pair thành mean/std/n theo (method, mode, region, metric). std lấy mẫu (ddof=1)."""
    rows = []
    for (method, mode, region), g in per_pair.groupby(["method", "mode", "region"], sort=False):
        for metric in PAIR_METRICS:
            v = g[metric].dropna().to_numpy(dtype=float)
            if v.size == 0:
                continue
            rows.append([method, mode, region, metric, float(v.mean()),
                         float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)])
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


def merge_write(path: Path, new: pd.DataFrame, cols: list, method: str) -> pd.DataFrame:
    """Ghi `new` vào `path`; giữ nguyên dòng của method khác, thay dòng của `method`."""
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        old = pd.read_csv(path, dtype={"id": str})
        old = old[old["method"] != method]
        new = pd.concat([old, new], ignore_index=True)
    new = new[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6f")
    return new


# ---------------------------------------------------------------- chạy
def read_pairs(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise SystemExit(f"Không tìm thấy {path}. File này do PV2 tạo (mục 2.3): "
                         f"chạy git pull hoặc lấy file từ PV2 rồi đặt vào data/processed/.")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")  # PV2 ghi CSV có BOM
    need = {"id", "person", "cloth", "mode"}
    if not need <= set(df.columns):
        raise SystemExit(f"{path} thiếu cột {sorted(need - set(df.columns))}; cần đủ id,person,cloth,mode.")
    bad = sorted(set(df["mode"]) - {"paired", "unpaired", "external"})
    if bad:
        raise SystemExit(f"Cột mode có giá trị lạ {bad}; chỉ chấp nhận paired, unpaired, external.")
    return df


def run(args, lpips_fn: Optional[LpipsFn] = None) -> dict:
    pairs = read_pairs(Path(args.pairs))
    if args.limit:
        pairs = pairs.head(args.limit)
    gt_dir, pred_dir, mask_dir = Path(args.gt), Path(args.pred), Path(args.masks)
    out_dir = Path(args.out)
    missing = []

    if lpips_fn is None and not args.skip_lpips:
        lpips_fn = make_lpips_fn(args.device)

    # ---- paired: so ảnh sinh với ảnh thật
    rows = []
    paired = pairs[pairs["mode"] == "paired"]
    for r in paired.itertuples(index=False):
        gt_path = find_image(gt_dir, r.person)
        pred_path = find_image(pred_dir, r.id)
        if gt_path is None:
            missing.append((r.id, f"thiếu ảnh thật {r.person} trong {gt_dir}"))
            continue
        if pred_path is None:
            missing.append((r.id, f"thiếu ảnh sinh {r.id} trong {pred_dir}"))
            continue
        mask_path = find_image(mask_dir, r.person)
        if mask_path is None:
            missing.append((r.id, f"thiếu mask {r.person} trong {mask_dir} (chỉ tính region=full)"))
        mask = load_mask(mask_path) if mask_path else None
        res = pair_metrics(load_rgb(gt_path), load_rgb(pred_path), mask, lpips_fn)
        for region, m in res.items():
            rows.append({"id": r.id, "mode": "paired", "method": args.method, "region": region, **m})
    per_pair = pd.DataFrame(rows, columns=PER_PAIR_COLS)

    # ---- unpaired: FID, KID theo tập
    summary_extra = []
    fid_info = None
    if not args.skip_fid:
        gen_files, real_files = [], []
        for r in pairs[pairs["mode"] == "unpaired"].itertuples(index=False):
            p = find_image(pred_dir, r.id)
            if p is None:
                missing.append((r.id, f"thiếu ảnh sinh {r.id} trong {pred_dir}"))
            else:
                gen_files.append(p)
        real_files = sorted(p for p in gt_dir.glob("*") if p.suffix.lower() in IMG_EXTS) if gt_dir.is_dir() else []
        if len(gen_files) >= 2 and len(real_files) >= 2:
            fid_info = fid_kid(real_files, gen_files, args.device, args.kid_subset_size)
            n = len(gen_files)
            summary_extra.append([args.method, "unpaired", "full", "fid", fid_info["fid"], np.nan, n])
            summary_extra.append([args.method, "unpaired", "full", "kid", fid_info["kid_mean"],
                                  fid_info["kid_std"], n])
        else:
            print(f"[bỏ qua FID/KID] cần ≥ 2 ảnh sinh unpaired và ≥ 2 ảnh thật "
                  f"(hiện {len(gen_files)} và {len(real_files)}).", file=sys.stderr)

    summary = pd.concat([summarize(per_pair), pd.DataFrame(summary_extra, columns=SUMMARY_COLS)],
                        ignore_index=True)

    # ---- ghi file
    out_dir.mkdir(parents=True, exist_ok=True)
    merge_write(out_dir / "results_per_pair.csv", per_pair, PER_PAIR_COLS, args.method)
    merge_write(out_dir / "results_summary.csv", summary, SUMMARY_COLS, args.method)
    miss_df = pd.DataFrame(missing, columns=["id", "reason"])
    miss_df.to_csv(out_dir / f"eval_missing_{args.method}.csv", index=False)
    return {"per_pair": per_pair, "summary": summary, "missing": miss_df, "fid_info": fid_info}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tính MSE/RMSE/MAE/SSIM/LPIPS (paired) và FID/KID (unpaired).")
    ap.add_argument("--pairs", default=str(config.EVAL_PAIRS), help="eval_pairs.csv (id,person,cloth,mode)")
    ap.add_argument("--gt", default=str(config.TEST_IMAGE), help="thư mục ảnh thật, tên {person}.jpg")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--pred", default=None, help="thư mục ảnh sinh {id}.png (mặc định theo --method)")
    ap.add_argument("--masks", default=str(config.OUT_MASKS), help="thư mục mask {person}.png do PV1 lưu")
    ap.add_argument("--out", default=str(config.RESULTS), help="thư mục ghi results_*.csv")
    ap.add_argument("--device", default=None, help="cpu hoặc cuda (mặc định tự chọn)")
    ap.add_argument("--kid-subset-size", type=int, default=50, help="subset_size của KID (tự giảm nếu mẫu ít hơn)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N dòng đầu (chạy thử)")
    ap.add_argument("--skip-lpips", action="store_true", help="bỏ LPIPS (cột lpips để trống)")
    ap.add_argument("--skip-fid", action="store_true", help="bỏ FID/KID")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.pred is None:
        args.pred = str(config.OUT_CATVTON if args.method == "catvton" else config.OUT_BASELINE)
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    res = run(args)
    s, miss = res["summary"], res["missing"]
    print(f"\n== {args.method}: {s[s.metric.isin(PAIR_METRICS)].n.max() if len(s) else 0} cặp paired đã tính, "
          f"{len(miss)} mục thiếu file ==")
    if len(s):
        print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(miss):
        print(f"\nCó {len(miss)} mục bị bỏ qua, xem {Path(args.out) / f'eval_missing_{args.method}.csv'}")
    return 0 if len(res["per_pair"]) else 1


if __name__ == "__main__":
    sys.exit(main())
