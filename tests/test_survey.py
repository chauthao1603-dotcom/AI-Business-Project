"""Test cho eval/make_survey_set.py và eval/analyze_survey.py (PV3) bằng dữ liệu giả."""
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eval import analyze_survey as an
from eval import make_survey_set as ms

N = 30  # số người giả


def _img(seed, w=96, h=128):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _dataset(tmp_path, n=N, with_baseline=True):
    gt, cat, base = tmp_path / "gt", tmp_path / "cat", tmp_path / "base"
    for d in (gt, cat, base):
        d.mkdir()
    rows = []
    for i in range(n):
        person, cloth = f"{i:05d}_00", f"{(i + 1) % n:05d}_00"
        pid = f"{person}__{cloth}"
        rows.append({"id": pid, "person": person, "cloth": cloth, "mode": "unpaired"})
        _img(i).save(gt / f"{person}.jpg", quality=95)
        _img(1000 + i).save(cat / f"{pid}.png")
        if with_baseline:
            _img(2000 + i).save(base / f"{pid}.png")
    pairs = tmp_path / "eval_pairs.csv"
    pd.DataFrame(rows).to_csv(pairs, index=False)
    return Namespace(pairs=str(pairs), gt=str(gt), catvton=str(cat), baseline=str(base),
                     out=str(tmp_path / "survey"), results=str(tmp_path / "res"),
                     jpeg_quality=90, force=False)


# ---------------------------------------------------------------- 3.3
def test_survey_set_structure(tmp_path):
    args = _dataset(tmp_path)
    res = ms.build(args)
    key, ab = res["key"], res["ab"]
    assert list(key.columns) == ["qid", "label", "source_id"]
    assert list(key.qid) == [f"q{i:02d}" for i in range(1, 21)]
    assert (key.label == "real").sum() == 10 and (key.label == "ai").sum() == 10
    assert list(ab.columns) == ["ab_id", "id", "left_method", "right_method"]
    assert len(ab) == 10
    assert set(zip(ab.left_method, ab.right_method)) <= {("catvton", "baseline"), ("baseline", "catvton")}
    assert ms.check_uniform(args.out)["uniform"]
    # 20 người khác nhau: ảnh thật và ảnh AI không chung người
    ai_people = [s.split("__")[0] for s in key[key.label == "ai"].source_id]
    real_people = list(key[key.label == "real"].source_id)
    assert len(set(ai_people)) == 10 and not set(ai_people) & set(real_people)
    # phần A/B không dùng lại ảnh của phần 1
    assert not set(ab.id) & set(key[key.label == "ai"].source_id)
    for k in range(1, 11):
        assert (tmp_path / "survey" / f"ab{k:02d}_A.png").exists()
        assert (tmp_path / "survey" / f"ab{k:02d}_B.png").exists()


def test_survey_images_are_768x1024_rgb_png_without_metadata(tmp_path):
    args = _dataset(tmp_path)
    ms.build(args)
    for p in (tmp_path / "survey").glob("*.png"):
        with Image.open(p) as im:
            assert im.size == (768, 1024) and im.mode == "RGB" and im.format == "PNG"
            assert not im.info.get("exif")


def test_survey_is_deterministic_with_seed_42(tmp_path):
    a = _dataset(tmp_path)
    k1 = ms.build(a)["key"]
    a.force, a.out, a.results = True, str(tmp_path / "s2"), str(tmp_path / "r2")
    k2 = ms.build(a)["key"]
    pd.testing.assert_frame_equal(k1, k2)


def test_survey_refuses_overwrite_without_force(tmp_path):
    a = _dataset(tmp_path)
    ms.build(a)
    with pytest.raises(SystemExit):
        ms.build(a)
    a.force = True
    ms.build(a)  # có --force thì chạy được


def test_ab_sheets_put_A_left_and_B_right(tmp_path):
    a = _dataset(tmp_path)
    res = ms.build(a)
    out = tmp_path / "survey"
    assert [p.name for p in res["sheets"]] == [f"ab{k:02d}.png" for k in range(1, 11)]
    with Image.open(out / "ab03_A.png") as im_a, Image.open(out / "ab03_B.png") as im_b, \
            Image.open(out / ms.AB_SHEET_DIR / "ab03.png") as sheet:
        w, h = im_a.size
        assert sheet.size == (2 * w + ms.SHEET_GAP, h + ms.SHEET_LABEL_H) and sheet.mode == "RGB"
        top = ms.SHEET_LABEL_H
        left = np.asarray(sheet.crop((0, top, w, top + h)))
        right = np.asarray(sheet.crop((w + ms.SHEET_GAP, top, 2 * w + ms.SHEET_GAP, top + h)))
        assert np.array_equal(left, np.asarray(im_a)) and np.array_equal(right, np.asarray(im_b))
        label_band = np.asarray(sheet.crop((0, 0, sheet.width, top)))
        assert label_band.min() < 128          # có chữ A, B trong dải nhãn
        assert not sheet.info.get("exif")
    # ảnh ghép nằm ở thư mục con nên không làm hỏng phép kiểm tra đồng nhất của 30 ảnh khảo sát
    assert ms.check_uniform(out)["uniform"]


def test_ab_sheets_flag_keeps_existing_answer_keys(tmp_path):
    a = _dataset(tmp_path)
    ms.build(a)
    res_dir, out = tmp_path / "res", tmp_path / "survey"
    before = {f: (res_dir / f).read_bytes() for f in ("survey_key.csv", "ab_key.csv")}
    q_before = (out / "q01.png").read_bytes()
    for p in (out / ms.AB_SHEET_DIR).glob("*.png"):
        p.unlink()
    # bộ đã có đáp án, không cần --force vẫn ghép được ảnh
    assert ms.main(["--ab-sheets", "--out", str(out)]) == 0
    assert len(list((out / ms.AB_SHEET_DIR).glob("ab*.png"))) == 10
    assert {f: (res_dir / f).read_bytes() for f in before} == before
    assert (out / "q01.png").read_bytes() == q_before


def test_ab_sheets_without_survey_set_is_reported(tmp_path):
    with pytest.raises(SystemExit):
        ms.make_ab_sheets(tmp_path / "empty")


def test_survey_needs_enough_images(tmp_path):
    a = _dataset(tmp_path, n=12)  # 12 người: đủ 10 ảnh AI nhưng không đủ 10 ảnh thật + 10 cặp A/B riêng
    with pytest.raises(SystemExit):
        ms.build(a)


def test_survey_needs_baseline_images(tmp_path):
    a = _dataset(tmp_path, with_baseline=False)
    with pytest.raises(SystemExit):
        ms.build(a)


# ---------------------------------------------------------------- 3.4
def _keys():
    labels = ["real", "ai"] * 10  # q01 real, q02 ai, ...
    key = pd.DataFrame({"qid": [f"q{i:02d}" for i in range(1, 21)], "label": labels,
                        "source_id": [f"s{i}" for i in range(20)]})
    ab = pd.DataFrame({"ab_id": [f"ab{i:02d}" for i in range(1, 11)], "id": [f"x{i}" for i in range(10)],
                       "left_method": ["catvton", "baseline"] * 5, "right_method": ["baseline", "catvton"] * 5})
    return key, ab


def _responses(rows):
    """rows: list dict {code: câu trả lời}; tiêu đề cột giống Google Form (mã + phần chữ)."""
    heads = {f"q{i:02d}": f"q{i:02d} - Ảnh này là thật hay AI?" for i in range(1, 21)}
    heads.update({f"ab{i:02d}": f"ab{i:02d} - Ảnh nào trông thật hơn?" for i in range(1, 11)})
    df = pd.DataFrame([{heads[k]: v for k, v in r.items()} for r in rows])
    df.insert(0, "Dấu thời gian", "2026/11/09")
    return df


def test_analyze_known_counts():
    key, ab = _keys()
    # Người 1 đoán đúng toàn bộ; người 2 luôn chọn "AI"; người 3 luôn chọn "Thật"
    perfect = {f"q{i:02d}": ("Thật" if i % 2 else "AI") for i in range(1, 21)}
    always_ai = {f"q{i:02d}": "AI" for i in range(1, 21)}
    always_real = {f"q{i:02d}": "Thật" for i in range(1, 21)}
    abs_ = {f"ab{i:02d}": "Ảnh A" for i in range(1, 11)}  # A = catvton ở các câu lẻ
    rows = [{**perfect, **abs_}, {**always_ai, **abs_}, {**always_real, **abs_}]
    r = an.analyze(_responses(rows), key, ab)
    c = r["counts"]
    # mỗi người có 10 ảnh AI và 10 ảnh thật
    assert c == {"TP": 10 + 10 + 0, "FN": 0 + 0 + 10, "FP": 0 + 10 + 0, "TN": 10 + 0 + 10}
    assert sum(c.values()) == r["expected"] == 3 * 20  # tổng = số người × 20
    m = r["metrics"]
    assert m["accuracy"] == pytest.approx(40 / 60)
    assert m["precision"] == pytest.approx(20 / 30)
    assert m["recall"] == pytest.approx(20 / 30)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))
    # A/B: 3 người chọn "Ảnh A"; A = catvton ở 5 câu trong 10 => 50%
    assert r["cat_rate"] == pytest.approx(0.5)
    assert r["ab_n"] == 30


def test_analyze_example_from_chapter2_slide():
    """TP=5, FP=2, FN=1, TN=2 (slide Chapter 2): accuracy 0.70, precision 5/7, recall 5/6."""
    m = an.classification_metrics(5, 1, 2, 2)
    assert m["accuracy"] == pytest.approx(0.70)
    assert m["precision"] == pytest.approx(5 / 7)
    assert m["recall"] == pytest.approx(5 / 6)


def test_analyze_handles_blank_and_invalid():
    key, ab = _keys()
    row = {f"q{i:02d}": ("Thật" if i % 2 else "AI") for i in range(1, 21)}
    row["q01"] = None          # bỏ trống
    row["q02"] = "không biết"  # giá trị lạ
    r = an.analyze(_responses([row]), key, ab)
    assert r["total"] == 18 and r["expected"] == 20
    assert r["blank"] >= 1 and r["invalid"] == 1


def test_analyze_zero_division_is_nan():
    m = an.classification_metrics(0, 0, 0, 5)  # không ai bị gọi là AI, không có ảnh AI
    assert np.isnan(m["precision"]) and np.isnan(m["recall"]) and np.isnan(m["f1"])
    assert m["accuracy"] == 1.0


def test_analyze_missing_question_column_is_reported():
    key, ab = _keys()
    df = _responses([{f"q{i:02d}": "AI" for i in range(1, 20)}])  # thiếu q20
    with pytest.raises(SystemExit):
        an.analyze(df, key, ab)


@pytest.mark.parametrize("raw,expected", [("Thật", "real"), (" THẬT ", "real"), ("Real", "real"),
                                          ("AI", "ai"), ("ai", "ai"), ("xyz", "invalid"), (None, None)])
def test_parse_class(raw, expected):
    assert an.parse_class(raw) == expected


@pytest.mark.parametrize("raw,expected", [("Ảnh A", "A"), ("ảnh b", "B"), ("A", "A"), ("C", "invalid"), (None, None)])
def test_parse_side(raw, expected):
    assert an.parse_side(raw) == expected


def test_confusion_matrix_figure_is_written(tmp_path):
    out = tmp_path / "figs" / "confusion_matrix.png"
    an.plot_confusion({"TP": 8, "FN": 2, "FP": 3, "TN": 7}, out)
    assert out.exists() and out.stat().st_size > 1000


def test_cli_end_to_end(tmp_path):
    key, ab = _keys()
    key.to_csv(tmp_path / "key.csv", index=False)
    ab.to_csv(tmp_path / "ab.csv", index=False)
    rows = [{**{f"q{i:02d}": ("Thật" if i % 2 else "AI") for i in range(1, 21)},
             **{f"ab{i:02d}": "Ảnh B" for i in range(1, 11)}} for _ in range(4)]
    _responses(rows).to_csv(tmp_path / "resp.csv", index=False)
    code = an.main(["--responses", str(tmp_path / "resp.csv"), "--survey-key", str(tmp_path / "key.csv"),
                    "--ab-key", str(tmp_path / "ab.csv"), "--out", str(tmp_path / "out")])
    assert code == 0
    s = pd.read_csv(tmp_path / "out" / "survey_summary.csv")
    assert list(s.columns) == an.SUMMARY_COLS
    acc = s[(s.section == "classification") & (s.item == "accuracy")].value.iloc[0]
    assert acc == pytest.approx(1.0)
    assert (tmp_path / "out" / "figs" / "confusion_matrix.png").exists()
