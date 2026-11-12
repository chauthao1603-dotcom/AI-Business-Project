"""Test cho eval/run_eval.py (PV3). Không cần torch: LPIPS được thay bằng hàm giả khi thử logic cắt khung."""
import numpy as np
import pandas as pd
import pytest
from argparse import Namespace
from PIL import Image

from eval import run_eval as ev
from eval import selftest
from src import config

H, W = config.IMG_H, config.IMG_W


def _photo(seed: int) -> np.ndarray:
    """Ảnh giả có cấu trúc (dải màu + nhiễu nhẹ), uint8 H×W×3."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W]
    base = np.stack([x / W * 255, y / H * 255, (x + y) / (W + H) * 255], axis=2)
    return np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)


def _torso_mask() -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[200:900, 100:600] = True
    return m


# ---------- chỉ số điểm ảnh
def test_pixel_errors_identical_is_zero():
    a = _photo(0)
    r = ev.pixel_errors(a, a)
    assert r == {"mse": 0.0, "rmse": 0.0, "mae": 0.0}


def test_pixel_errors_known_value():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full((4, 4, 3), 10, dtype=np.uint8)
    r = ev.pixel_errors(a, b)
    assert r["mse"] == pytest.approx(100.0)
    assert r["rmse"] == pytest.approx(10.0)
    assert r["mae"] == pytest.approx(10.0)


def test_pixel_errors_uint8_does_not_wrap():
    a = np.full((2, 2, 3), 5, dtype=np.uint8)
    b = np.full((2, 2, 3), 250, dtype=np.uint8)  # 5 - 250 sẽ tràn nếu giữ uint8
    assert ev.pixel_errors(a, b)["mae"] == pytest.approx(245.0)


def test_pixel_errors_empty_selection_is_nan():
    a = _photo(0)
    r = ev.pixel_errors(a, a, np.zeros((H, W), dtype=bool))
    assert np.isnan(r["mse"])


# ---------- khung bao và vùng mask
def test_bbox():
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 3:8] = True
    assert ev.bbox(m) == (2, 5, 3, 8)
    assert ev.bbox(np.zeros((10, 10), dtype=bool)) is None


def test_full_region_is_inflated_by_unchanged_background():
    """Ảnh sinh chỉ khác trong mask: số toàn ảnh phải nhỏ hơn số vùng mask (lý do cần region=mask)."""
    gt = np.minimum(_photo(1), 200)  # chặn trên 200 để +30 không bị cắt ở 255
    mask = _torso_mask()
    pred = gt.copy()
    pred[mask] = np.clip(pred[mask].astype(int) + 30, 0, 255).astype(np.uint8)
    res = ev.pair_metrics(gt, pred, mask)
    assert res["mask"]["mse"] == pytest.approx(900.0)
    assert res["full"]["mse"] == pytest.approx(900.0 * mask.mean())
    assert res["full"]["mse"] < res["mask"]["mse"]
    assert res["full"]["ssim"] > res["mask"]["ssim"]


def test_lpips_gets_bbox_crop_on_mask_region():
    gt, mask = _photo(2), _torso_mask()
    seen = []

    def fake(g, p):
        seen.append(g.shape)
        return 0.5

    res = ev.pair_metrics(gt, gt, mask, lpips_fn=fake)
    assert seen == [(H, W, 3), (700, 500, 3)]  # full rồi khung bao của mask
    assert res["mask"]["lpips"] == 0.5


def test_ssim_identical_and_black():
    a = _photo(3)
    assert ev.ssim_value(a, a) == pytest.approx(1.0)
    assert ev.ssim_value(a, np.zeros_like(a)) < 0.5


def test_tiny_crop_does_not_crash():
    a = np.random.default_rng(0).integers(0, 255, (5, 5, 3), dtype=np.uint8)
    assert ev.ssim_value(a, a) == pytest.approx(1.0)


# ---------- chạy trọn quy trình trên thư mục tạm
def _setup(tmp_path, n=4):
    gt = tmp_path / "gt"
    gt.mkdir()
    rows = []
    for i in range(n):
        stem = f"{i:05d}_00"
        Image.fromarray(_photo(i)).save(gt / f"{stem}.jpg", quality=95)
        rows.append({"id": f"{stem}__{stem}", "person": stem, "cloth": stem, "mode": "paired"})
    rows.append({"id": "00000_00__00001_00", "person": "00000_00", "cloth": "00001_00", "mode": "unpaired"})
    pairs = tmp_path / "eval_pairs.csv"
    pd.DataFrame(rows).to_csv(pairs, index=False)
    masks = tmp_path / "masks"
    masks.mkdir()
    for i in range(n):
        Image.fromarray((_torso_mask() * 255).astype(np.uint8)).save(masks / f"{i:05d}_00.png")
    return gt, pairs, masks


def _args(tmp_path, gt, pairs, masks, pred, method):
    return Namespace(pairs=str(pairs), gt=str(gt), pred=str(pred), masks=str(masks), out=str(tmp_path / "out"),
                     method=method, device="cpu", kid_subset_size=50, limit=0, skip_lpips=True, skip_fid=True)


def _make_pred(folder, gt, n, brightness=0):
    folder.mkdir(exist_ok=True)
    for i in range(n):
        stem = f"{i:05d}_00"
        arr = np.asarray(Image.open(gt / f"{stem}.jpg").convert("RGB")).astype(int) + brightness
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(folder / f"{stem}__{stem}.png")


def test_two_methods_share_csv_and_schema(tmp_path):
    gt, pairs, masks = _setup(tmp_path)
    _make_pred(tmp_path / "p_cat", gt, 4, 0)
    _make_pred(tmp_path / "p_base", gt, 4, 20)
    ev.run(_args(tmp_path, gt, pairs, masks, tmp_path / "p_cat", "catvton"), lpips_fn=None)
    ev.run(_args(tmp_path, gt, pairs, masks, tmp_path / "p_base", "baseline"), lpips_fn=None)
    per = pd.read_csv(tmp_path / "out" / "results_per_pair.csv", dtype={"id": str})
    summ = pd.read_csv(tmp_path / "out" / "results_summary.csv")
    assert list(per.columns) == ev.PER_PAIR_COLS
    assert list(summ.columns) == ev.SUMMARY_COLS
    assert set(per.method) == {"catvton", "baseline"}
    assert len(per) == 4 * 2 * 2  # 4 cặp × 2 method × 2 region
    cat = summ[(summ.method == "catvton") & (summ.region == "full") & (summ.metric == "mse")]
    base = summ[(summ.method == "baseline") & (summ.region == "full") & (summ.metric == "mse")]
    assert cat["mean"].iloc[0] < base["mean"].iloc[0]
    assert int(cat["n"].iloc[0]) == 4
    assert set(per["mode"]) == {"paired"}  # cặp unpaired không có đáp án nên không có dòng


def test_rerun_replaces_only_own_method(tmp_path):
    gt, pairs, masks = _setup(tmp_path)
    _make_pred(tmp_path / "p", gt, 4, 0)
    a = _args(tmp_path, gt, pairs, masks, tmp_path / "p", "catvton")
    ev.run(a)
    ev.run(a)
    per = pd.read_csv(tmp_path / "out" / "results_per_pair.csv")
    assert len(per) == 4 * 2  # chạy lại không nhân đôi dòng


def test_missing_files_are_reported_not_fatal(tmp_path):
    gt, pairs, masks = _setup(tmp_path)
    _make_pred(tmp_path / "p", gt, 3, 0)  # thiếu ảnh sinh của cặp thứ 4
    res = ev.run(_args(tmp_path, gt, pairs, masks, tmp_path / "p", "catvton"))
    assert len(res["missing"]) == 1
    assert "00003_00__00003_00" in res["missing"]["id"].iloc[0]
    assert res["per_pair"]["id"].nunique() == 3


def test_selftest_passes_without_lpips(tmp_path):
    gt, _, _ = _setup(tmp_path)
    checks = selftest.evaluate(gt, n=4, skip_lpips=True)
    assert all(c[3] for c in checks), [c for c in checks if not c[3]]


def test_selftest_with_fake_lpips(tmp_path):
    gt, _, _ = _setup(tmp_path)

    def fake(g, p):  # L1 chuẩn hóa: 0 khi giống hệt, lớn khi khác
        return float(np.abs(g.astype(float) - p.astype(float)).mean() / 255)

    checks = selftest.evaluate(gt, n=4, skip_lpips=False, lpips_fn=fake)
    assert all(c[3] for c in checks), [c for c in checks if not c[3]]


def test_bad_mode_value_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame([{"id": "a__b", "person": "a", "cloth": "b", "mode": "weird"}]).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        ev.read_pairs(p)
