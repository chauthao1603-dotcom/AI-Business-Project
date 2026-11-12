"""Test cho eval/make_report_assets.py (3.5) và eval/external_notes.py (3.6) bằng dữ liệu giả."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eval import external_notes as en
from eval import make_report_assets as ra
from eval import run_eval as ev


def _fake_results(tmp_path, n=8):
    """results_per_pair/summary giả cho 2 method, 2 region; CatVTON tốt hơn baseline. Kèm ảnh giả."""
    rng = np.random.default_rng(0)
    gt, base, cat = tmp_path / "gt", tmp_path / "base", tmp_path / "cat"
    for d in (gt, base, cat):
        d.mkdir()
    rows = []
    for i in range(n):
        stem = f"{i:05d}_00"
        pid = f"{stem}__{stem}"
        for d, name in ((gt, f"{stem}.jpg"), (base, f"{pid}.png"), (cat, f"{pid}.png")):
            Image.fromarray(rng.integers(0, 255, (128, 96, 3), dtype=np.uint8)).save(d / name)
        for method, k in (("baseline", 2.0), ("catvton", 1.0)):
            for region in ("full", "mask"):
                mse = k * (100 + 10 * i) * (2 if region == "mask" else 1)
                rows.append({"id": pid, "mode": "paired", "method": method, "region": region, "mse": mse,
                             "rmse": mse ** 0.5, "mae": mse / 20, "ssim": 0.9 - 0.05 * k,
                             "lpips": 0.05 * k + 0.01 * i})
    per = pd.DataFrame(rows, columns=ev.PER_PAIR_COLS)
    summ = ev.summarize(per)
    extra = pd.DataFrame([["baseline", "unpaired", "full", "fid", 80.0, np.nan, 100],
                          ["catvton", "unpaired", "full", "fid", 30.0, np.nan, 100],
                          ["baseline", "unpaired", "full", "kid", 0.05, 0.01, 100],
                          ["catvton", "unpaired", "full", "kid", 0.01, 0.005, 100]], columns=ev.SUMMARY_COLS)
    summ = pd.concat([summ, extra], ignore_index=True)
    res = tmp_path / "results"
    res.mkdir()
    per.to_csv(res / "results_per_pair.csv", index=False)
    summ.to_csv(res / "results_summary.csv", index=False)
    return res, gt, base, cat


# ---------------------------------------------------------------- 3.5
def test_one_chart_per_metric(tmp_path):
    res, gt, base, cat = _fake_results(tmp_path)
    code = ra.main(["--results", str(res), "--gt", str(gt), "--baseline", str(base), "--catvton", str(cat)])
    assert code == 0
    for m in ["mse", "rmse", "mae", "ssim", "lpips", "fid", "kid"]:
        f = res / "figs" / f"bar_{m}.png"
        assert f.exists() and f.stat().st_size > 5000, m


def test_failures_are_worst_catvton_lpips_on_mask(tmp_path):
    res, gt, base, cat = _fake_results(tmp_path)
    per = pd.read_csv(res / "results_per_pair.csv", dtype={"id": str})
    top, region = ra.pick_failures(per)
    assert region == "mask"
    # LPIPS tăng theo i nên 5 ca tệ nhất là i = 7, 6, 5, 4, 3
    assert list(top.id) == [f"{i:05d}_00__{i:05d}_00" for i in (7, 6, 5, 4, 3)]


def test_failure_folders_contain_three_images_and_csv(tmp_path):
    res, gt, base, cat = _fake_results(tmp_path)
    per = pd.read_csv(res / "results_per_pair.csv", dtype={"id": str})
    t = ra.export_failures(per, gt, base, cat, res / "failure_cases")
    assert len(t) == 5
    for pid in t.id:
        d = res / "failure_cases" / pid
        assert (d / "original.jpg").exists() and (d / "baseline.png").exists() and (d / "catvton.png").exists()
        assert (d / "compare.png").exists()
    csv = pd.read_csv(res / "failure_cases" / "failure_cases.csv")
    assert {"lpips_catvton", "lpips_baseline", "loai_loi"} <= set(csv.columns)
    assert (csv.lpips_catvton < csv.lpips_baseline).all()  # dữ liệu giả: baseline tệ hơn


def test_failures_need_lpips(tmp_path):
    res, *_ = _fake_results(tmp_path)
    per = pd.read_csv(res / "results_per_pair.csv", dtype={"id": str})
    per["lpips"] = np.nan  # như khi 3.2 chạy với --skip-lpips
    with pytest.raises(SystemExit):
        ra.pick_failures(per)


def test_missing_results_message(tmp_path):
    with pytest.raises(SystemExit):
        ra.main(["--results", str(tmp_path / "khong_co")])


# ---------------------------------------------------------------- 3.6
def _external(tmp_path, ids):
    d = tmp_path / "external"
    d.mkdir(exist_ok=True)
    for i in ids:
        Image.new("RGB", (8, 8)).save(d / f"{i}.png")
    return d


def test_init_creates_template_and_keeps_filled_rows(tmp_path):
    out = tmp_path / "external_notes.csv"
    d = _external(tmp_path, ["ext_p01__ext_g01", "ext_p02__ext_g01"])
    t = en.init(d, out)
    assert list(t.columns) == en.COLS and len(t) == 2
    filled = pd.read_csv(out, dtype=str).fillna("")
    filled.loc[0, ["ket_qua", "loai_loi"]] = ["loi", "sai_tay_ao"]
    filled.to_csv(out, index=False)
    _external(tmp_path, ["ext_p03__ext_g02"])  # PV1 thêm ảnh
    t2 = en.init(d, out)
    assert len(t2) == 3
    assert t2.loc[t2.id == "ext_p01__ext_g01", "ket_qua"].iloc[0] == "loi"  # không mất phần đã điền


def test_validate_and_summary():
    t = pd.DataFrame({"id": ["a", "b", "c", "d"], "ket_qua": ["tot", "loi", "chap_nhan_duoc", "hong"],
                      "loai_loi": ["", "sai_tay_ao;mat_hoa_tiet", "", ""], "ghi_chu": ["", "", "", ""]})
    problems = en.validate(t)
    assert any("c:" in p for p in problems)   # chấp nhận được nhưng thiếu loại lỗi
    assert any("d:" in p for p in problems)   # giá trị ket_qua lạ
    s = en.summarize(t[t.id.isin(["a", "b"])])
    assert s["ket_qua"] == {"tot": 1, "chap_nhan_duoc": 0, "loi": 1}
    assert s["loai_loi"] == {"sai_tay_ao": 1, "mat_hoa_tiet": 1}


def test_init_without_images_is_reported(tmp_path):
    (tmp_path / "external").mkdir()
    with pytest.raises(SystemExit):
        en.init(tmp_path / "external", tmp_path / "x.csv")
