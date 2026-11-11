"""PV3 – 3.5: biểu đồ KPI và ca lỗi cho báo cáo, sau khi eval.run_eval đã chạy thật cho cả hai method.

    python -m eval.make_report_assets

Đầu vào: results/results_per_pair.csv, results/results_summary.csv (từ 3.2) và ảnh trong
data/raw/test/image, outputs/baseline, outputs/catvton.

Đầu ra:
    results/figs/bar_<chỉ số>.png        biểu đồ cột baseline vs CatVTON cho từng chỉ số
                                          (mse, rmse, mae, ssim, lpips theo region; fid, kid cho unpaired)
    results/failure_cases/<id>/           5 cặp có LPIPS tệ nhất của CatVTON (ưu tiên region=mask):
        original.<đuôi>, baseline.png, catvton.png, compare.png (ba ảnh cạnh nhau, dùng cho slide)
    results/failure_cases/failure_cases.csv   id, region, lpips/ssim/mse của hai method để viết phần hạn chế
Ảnh trong results/failure_cases/ bị .gitignore chặn (VITON-HD không được phát tán); CSV thì commit được.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from eval.run_eval import PAIR_METRICS, find_image, load_rgb  # noqa: E402
from src import config  # noqa: E402

# Màu theo đối tượng, cố định: CatVTON luôn xanh, baseline luôn cam (bảng màu tham chiếu, slot 1 và 2).
COLORS = {"baseline": "#eb6834", "catvton": "#2a78d6"}
LABELS = {"baseline": "Baseline (dán áo)", "catvton": "CatVTON"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REGION_LABEL = {"full": "Toàn ảnh", "mask": "Vùng mask"}
DIRECTION = {"mse": "thấp hơn là tốt", "rmse": "thấp hơn là tốt", "mae": "thấp hơn là tốt",
             "ssim": "cao hơn là tốt", "lpips": "thấp hơn là tốt", "fid": "thấp hơn là tốt",
             "kid": "thấp hơn là tốt"}
N_FAIL = 5


# ---------------------------------------------------------------- biểu đồ
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.3f}" if a >= 0.01 else f"{v:.4f}"


def bar_chart(summary: pd.DataFrame, metric: str, path: Path) -> bool:
    """Một biểu đồ cột: nhóm theo region (hoặc 'unpaired' với FID/KID), hai cột baseline và CatVTON."""
    d = summary[summary.metric == metric]
    if d.empty:
        return False
    groups = [g for g in ("full", "mask") if g in set(d.region)] if metric in PAIR_METRICS else ["full"]
    methods = [m for m in ("baseline", "catvton") if m in set(d.method)]
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.6 if len(groups) > 1 else 4.2, 3.8), facecolor=SURFACE)
    _style(ax)
    tops = []
    for k, method in enumerate(methods):
        vals, errs = [], []
        for g in groups:
            row = d[(d.method == method) & (d.region == g)]
            vals.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
            e = float(row["std"].iloc[0]) if len(row) and pd.notna(row["std"].iloc[0]) else 0.0
            errs.append(e)
        pos = x + (k - (len(methods) - 1) / 2) * (width + 0.02)  # khe hở nhỏ giữa hai cột
        ax.bar(pos, vals, width, color=COLORS[method], label=LABELS[method], zorder=2)
        if any(e > 0 for e in errs):
            ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        for p, v, e in zip(pos, vals, errs):
            ax.text(p, (v if np.isfinite(v) else 0) + e, _fmt(v), ha="center", va="bottom",
                    fontsize=9, color=INK)
            tops.append((v if np.isfinite(v) else 0) + e)
    names = [REGION_LABEL[g] for g in groups] if metric in PAIR_METRICS else ["Cặp unpaired"]
    ax.set_xticks(x, names, color=INK)
    if tops:
        ax.set_ylim(min(0, min(tops)) * 1.15, max(tops) * 1.18 if max(tops) > 0 else 1)
    n = int(d["n"].max())
    ax.set_title(f"{metric.upper()} ({DIRECTION[metric]})", color=INK, fontsize=12, loc="left", pad=22)
    ax.text(0, 1.03, f"Trung bình ± độ lệch chuẩn, n = {n}" if metric != "fid" else f"n = {n} ảnh sinh",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- ca lỗi
def pick_failures(per_pair: pd.DataFrame, n: int = N_FAIL) -> tuple[pd.DataFrame, str]:
    """n cặp CatVTON có LPIPS cao nhất; ưu tiên region=mask vì phản ánh đúng phần mô hình phải sinh."""
    cat = per_pair[(per_pair.method == "catvton") & per_pair.lpips.notna()]
    if cat.empty:
        raise SystemExit("Không có giá trị LPIPS của CatVTON trong results_per_pair.csv "
                         "(có thể 3.2 đã chạy với --skip-lpips). Chạy lại 3.2 có LPIPS.")
    region = "mask" if (cat.region == "mask").any() else "full"
    return cat[cat.region == region].nlargest(n, "lpips"), region


def compare_strip(paths: list, captions: list, out: Path, height: int = 512) -> None:
    """Ghép các ảnh cạnh nhau (cùng chiều cao), chú thích phía trên."""
    imgs = [Image.fromarray(load_rgb(p)) for p in paths]
    imgs = [im.resize((int(im.width * height / im.height), height)) for im in imgs]
    pad, top = 8, 28
    w = sum(im.width for im in imgs) + pad * (len(imgs) + 1)
    canvas = Image.new("RGB", (w, height + top + pad), SURFACE)
    draw = ImageDraw.Draw(canvas)
    x = pad
    for im, cap in zip(imgs, captions):
        canvas.paste(im, (x, top))
        draw.text((x, 8), cap, fill=INK)
        x += im.width + pad
    canvas.save(out)


def export_failures(per_pair: pd.DataFrame, gt_dir: Path, base_dir: Path, cat_dir: Path,
                    out_dir: Path, n: int = N_FAIL) -> pd.DataFrame:
    top, region = pick_failures(per_pair, n)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for r in top.itertuples(index=False):
        person = r.id.split("__")[0]
        src = {"original": find_image(gt_dir, person), "baseline": find_image(base_dir, r.id),
               "catvton": find_image(cat_dir, r.id)}
        dest = out_dir / r.id
        dest.mkdir(parents=True, exist_ok=True)
        for name, p in src.items():
            if p is None:
                problems.append(f"{r.id}: thiếu ảnh {name}")
            else:
                shutil.copy2(p, dest / f"{name}{p.suffix.lower()}")
        have = [(p, cap) for p, cap in ((src["original"], "Anh goc"), (src["baseline"], "Baseline"),
                                        (src["catvton"], f"CatVTON (LPIPS {r.lpips:.3f})")) if p is not None]
        if have:
            compare_strip([p for p, _ in have], [c for _, c in have], dest / "compare.png")
        other = per_pair[(per_pair.id == r.id) & (per_pair.region == region) & (per_pair.method == "baseline")]
        rows.append({"id": r.id, "region": region,
                     "lpips_catvton": r.lpips, "lpips_baseline": other.lpips.iloc[0] if len(other) else np.nan,
                     "ssim_catvton": r.ssim, "ssim_baseline": other.ssim.iloc[0] if len(other) else np.nan,
                     "mse_catvton": r.mse, "mse_baseline": other.mse.iloc[0] if len(other) else np.nan,
                     "loai_loi": "", "ghi_chu": ""})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "failure_cases.csv", index=False, float_format="%.6f")
    stale = sorted(p.name for p in out_dir.iterdir() if p.is_dir() and p.name not in set(table.id))
    if stale:
        print(f"CHÚ Ý: results/failure_cases/ còn thư mục cũ không thuộc 5 ca mới: {stale}. Xóa tay nếu không dùng.")
    for msg in problems:
        print("THIẾU:", msg)
    return table


# ---------------------------------------------------------------- chạy
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Biểu đồ KPI và 5 ca lỗi tệ nhất của CatVTON.")
    ap.add_argument("--results", default=str(config.RESULTS))
    ap.add_argument("--gt", default=str(config.TEST_IMAGE))
    ap.add_argument("--baseline", default=str(config.OUT_BASELINE))
    ap.add_argument("--catvton", default=str(config.OUT_CATVTON))
    ap.add_argument("--n-fail", type=int, default=N_FAIL)
    ap.add_argument("--skip-failures", action="store_true", help="chỉ vẽ biểu đồ")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    res = Path(a.results)
    for f in ("results_per_pair.csv", "results_summary.csv"):
        if not (res / f).is_file():
            raise SystemExit(f"Không thấy {res / f}. Chạy eval.run_eval cho cả --method baseline và catvton trước.")
    per_pair = pd.read_csv(res / "results_per_pair.csv", dtype={"id": str})
    summary = pd.read_csv(res / "results_summary.csv")
    missing_methods = {"baseline", "catvton"} - set(summary.method)
    if missing_methods:
        print(f"CHÚ Ý: results_summary.csv chưa có method {sorted(missing_methods)}, biểu đồ sẽ thiếu cột đó.")
    drawn = [m for m in PAIR_METRICS + ["fid", "kid"] if bar_chart(summary, m, res / "figs" / f"bar_{m}.png")]
    print(f"Đã vẽ {len(drawn)} biểu đồ vào {res / 'figs'}: {', '.join(drawn)}")
    if not a.skip_failures:
        t = export_failures(per_pair, Path(a.gt), Path(a.baseline), Path(a.catvton),
                            res / "failure_cases", a.n_fail)
        print(f"Đã xuất {len(t)} ca lỗi (region={t.region.iloc[0]}) vào {res / 'failure_cases'}; "
              f"điền cột loai_loi và ghi_chu trong failure_cases.csv sau khi xem ảnh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
