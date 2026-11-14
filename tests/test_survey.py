"""Test cho eval/make_survey_set.py và eval/analyze_survey.py (PV3) bằng dữ liệu giả."""
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eval import analyze_survey as an
from eval import make_survey_set as ms

N = 60  # số người giả: phần 1 dùng tối đa 30 người/áo, phần 2 cần thêm 10 cặp paired khác


def _img(seed, w=96, h=128):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _dataset(tmp_path, n=N, with_paired=True):
    gt, cat = tmp_path / "gt", tmp_path / "cat"
    for d in (gt, cat):
        d.mkdir()
    rows = []
    for i in range(n):
        person, cloth = f"{i:05d}_00", f"{(i + 1) % n:05d}_00"
        pid = f"{person}__{cloth}"
        rows.append({"id": pid, "person": person, "cloth": cloth, "mode": "unpaired"})
        _img(i).save(gt / f"{person}.jpg", quality=95)
        _img(1000 + i).save(cat / f"{pid}.png")
        if with_paired:
            same = f"{person}__{person}"
            rows.append({"id": same, "person": person, "cloth": person, "mode": "paired"})
            _img(3000 + i).save(cat / f"{same}.png")
    pairs = tmp_path / "eval_pairs.csv"
    pd.DataFrame(rows).to_csv(pairs, index=False)
    return Namespace(pairs=str(pairs), gt=str(gt), catvton=str(cat),
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
    assert set(zip(ab.left_method, ab.right_method)) <= {("catvton", "real"), ("real", "catvton")}
    assert (ab.left_method == "real").sum() == 5          # đúng 5/10 câu ảnh thật bên trái
    assert ms.check_uniform(args.out)["uniform"]
    # 20 người khác nhau: ảnh thật và ảnh AI không chung người
    ai_people = [s.split("__")[0] for s in key[key.label == "ai"].source_id]
    real_people = list(key[key.label == "real"].source_id)
    assert len(set(ai_people)) == 10 and not set(ai_people) & set(real_people)
    # phần A/B: cặp paired (cùng người, cùng áo), không đụng người hay áo nào của phần 1
    part1 = {x for s in key.source_id for x in s.split("__")}
    for pid in ab.id:
        person, cloth = pid.split("__")
        assert person == cloth and person not in part1
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


def test_survey_needs_paired_catvton_images(tmp_path):
    a = _dataset(tmp_path, with_paired=False)
    with pytest.raises(SystemExit):
        ms.build(a)


def test_ab_images_are_the_real_photo_and_the_catvton_output(tmp_path):
    a = _dataset(tmp_path)
    ab = ms.build(a)["ab"]
    out = tmp_path / "survey"
    r = ab.iloc[0]
    person = r.id.split("__")[0]
    expect = {"real": tmp_path / "gt" / f"{person}.jpg", "catvton": tmp_path / "cat" / f"{r.id}.png"}
    for side, src in (("A", r.left_method), ("B", r.right_method)):
        want = np.asarray(ms.normalize(Image.open(expect[src]), a.jpeg_quality))
        with Image.open(out / f"{r.ab_id}_{side}.png") as got:
            assert np.array_equal(np.asarray(got), want)


def test_redo_ab_keeps_part1_and_is_deterministic(tmp_path):
    a = _dataset(tmp_path)
    ms.build(a)
    res_dir, out = tmp_path / "res", tmp_path / "survey"
    key_before = (res_dir / "survey_key.csv").read_bytes()
    q_before = {p.name: p.read_bytes() for p in out.glob("q*.png")}
    ab_before = (res_dir / "ab_key.csv").read_bytes()
    # giả lập bộ A/B cũ (baseline) còn sót: phải bị xóa và thay hết
    (res_dir / "ab_key.csv").write_text("ab_id,id,left_method,right_method\nab01,x,baseline,catvton\n")
    _img(9).save(out / "ab11_A.png")
    assert ms.main(["--redo-ab", "--pairs", a.pairs, "--gt", a.gt, "--catvton", a.catvton,
                    "--out", a.out, "--results", a.results]) == 0
    assert (res_dir / "survey_key.csv").read_bytes() == key_before
    assert {p.name: p.read_bytes() for p in out.glob("q*.png")} == q_before
    assert (res_dir / "ab_key.csv").read_bytes() == ab_before      # cùng seed => cùng bộ A/B
    assert not (out / "ab11_A.png").exists()
    assert len(list((out / ms.AB_SHEET_DIR).glob("ab*.png"))) == 10


def test_redo_ab_needs_existing_part1(tmp_path):
    a = _dataset(tmp_path)
    with pytest.raises(SystemExit):
        ms.redo_ab(a)


# ---------------------------------------------------------------- 3.4
def _keys(design="real"):
    """design='real': A/B là ảnh thật – CatVTON (thiết kế hiện tại); 'baseline': thiết kế cũ."""
    other = design
    labels = ["real", "ai"] * 10  # q01 real, q02 ai, ...
    key = pd.DataFrame({"qid": [f"q{i:02d}" for i in range(1, 21)], "label": labels,
                        "source_id": [f"s{i}" for i in range(20)]})
    ab = pd.DataFrame({"ab_id": [f"ab{i:02d}" for i in range(1, 11)], "id": [f"x{i}" for i in range(10)],
                       "left_method": ["catvton", other] * 5, "right_method": [other, "catvton"] * 5})
    return key, ab


def _responses(rows):
    """rows: list dict {code: câu trả lời}; tiêu đề cột giống Google Form (mã + phần chữ)."""
    heads = {f"q{i:02d}": f"q{i:02d} - Ảnh này là thật hay AI?" for i in range(1, 21)}
    heads.update({f"ab{i:02d}": f"ab{i:02d} - Ảnh nào là ảnh chụp thật?" for i in range(1, 11)})
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
    assert r["ab_rates"]["real"] == pytest.approx(0.5)
    assert r["ab_n"] == 30
    assert r["ab_test"]["p_value"] == pytest.approx(1.0)


def test_analyze_ab_real_vs_catvton():
    key, ab = _keys()   # câu lẻ: A = catvton, B = thật; câu chẵn: A = thật, B = catvton
    q = {f"q{i:02d}": "AI" for i in range(1, 21)}
    always_real = {f"ab{i:02d}": ("Ảnh B" if i % 2 else "Ảnh A") for i in range(1, 11)}   # luôn chọn đúng ảnh thật
    fooled = {f"ab{i:02d}": ("Ảnh A" if i % 2 else "Ảnh B") for i in range(1, 11)}         # luôn chọn ảnh CatVTON
    r = an.analyze(_responses([{**q, **always_real}] * 3 + [{**q, **fooled}]), key, ab)
    assert r["ab_n"] == 40
    assert r["ab_rates"] == {"catvton": pytest.approx(0.25), "real": pytest.approx(0.75)}
    t = r["ab_test"]
    assert t["real_chosen"] == 30 and t["ci_low"] < 0.75 < t["ci_high"]
    assert t["p_value"] == pytest.approx(an.binom_test_half(30, 40)) and t["p_value"] < 0.01
    s = r["summary"]
    assert set(s[s.section == "ab"].item) == {"catvton_chosen_rate", "real_chosen_rate", "real_chosen_ci95_low",
                                              "real_chosen_ci95_high", "p_value_vs_50pct"}
    assert (s[s.section == "ab_pair_catvton_chosen"].value == 0.25).all()


def test_analyze_old_baseline_design_still_reads():
    key, ab = _keys("baseline")
    row = {**{f"q{i:02d}": "AI" for i in range(1, 21)}, **{f"ab{i:02d}": "Ảnh A" for i in range(1, 11)}}
    r = an.analyze(_responses([row]), key, ab)
    assert r["ab_rates"] == {"baseline": pytest.approx(0.5), "catvton": pytest.approx(0.5)}
    assert r["ab_test"] is None


@pytest.mark.parametrize("k,n,p", [(15, 30, 1.0), (20, 30, 0.098737), (25, 30, 0.000325), (7, 10, 0.34375)])
def test_binomial_test_matches_reference(k, n, p):
    assert an.binom_test_half(k, n) == pytest.approx(p, abs=1e-6)   # giá trị tham chiếu từ scipy.stats.binomtest


def test_wilson_ci_reference():
    lo, hi = an.wilson_ci(20, 30)
    assert lo == pytest.approx(0.487801, abs=1e-6) and hi == pytest.approx(0.807695, abs=1e-6)


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


# ---------------------------------------------------------------- 3.4: cấp người trả lời và thiên lệch vị trí
@pytest.mark.parametrize("df,ref", [(1, 12.706205), (2, 4.302653), (9, 2.262157), (29, 2.045230), (100, 1.983972)])
def test_t_ppf_matches_reference(df, ref):
    assert an.t_ppf(0.975, df) == pytest.approx(ref, abs=1e-5)   # giá trị tham chiếu từ scipy.stats.t.ppf
    assert an.t_ppf(0.025, df) == pytest.approx(-ref, abs=1e-5)


def test_mean_ci_t_known_values():
    st = an.mean_ci_t([0.2, 0.4, 0.6])
    assert st["mean"] == pytest.approx(0.4) and st["sd"] == pytest.approx(0.2) and st["n"] == 3
    half = 4.302653 * 0.2 / np.sqrt(3)
    assert st["ci_low"] == pytest.approx(0.4 - half, abs=1e-5) and st["ci_high"] == pytest.approx(0.4 + half, abs=1e-5)
    assert (st["min"], st["max"]) == (pytest.approx(0.2), pytest.approx(0.6))


def test_mean_ci_t_ignores_nan_and_needs_two_values():
    st = an.mean_ci_t([0.5, np.nan])
    assert st["n"] == 1 and st["mean"] == 0.5 and np.isnan(st["ci_low"]) and np.isnan(st["sd"])


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    v = np.linspace(0.2, 0.9, 30)
    a, b = an.bootstrap_mean_ci(v), an.bootstrap_mean_ci(v)
    assert a == b                                   # seed 42 cố định => tái lập được
    assert a[0] < v.mean() < a[1]
    assert an.bootstrap_mean_ci([0.5] * 10) == (pytest.approx(0.5), pytest.approx(0.5))
    assert all(np.isnan(x) for x in an.bootstrap_mean_ci([0.5]))


def test_per_respondent_part1_accuracy():
    key, ab = _keys()
    perfect = {f"q{i:02d}": ("Thật" if i % 2 else "AI") for i in range(1, 21)}
    always_ai = {f"q{i:02d}": "AI" for i in range(1, 21)}
    abs_ = {f"ab{i:02d}": "Ảnh A" for i in range(1, 11)}
    r = an.analyze(_responses([{**perfect, **abs_}, {**always_ai, **abs_}, {**always_ai, **abs_}]), key, ab)
    st = r["per_respondent"]["part1_accuracy"]
    assert st["n"] == 3 and st["mean"] == pytest.approx(2 / 3)       # 1.0, 0.5, 0.5
    assert st["sd"] == pytest.approx(np.std([1, 0.5, 0.5], ddof=1))
    assert (st["min"], st["max"]) == (pytest.approx(0.5), pytest.approx(1.0))
    assert r["part1_bias"]["ai_answer_rate"] == pytest.approx(50 / 60)   # 10 + 20 + 20 lượt chọn "AI"
    assert r["part1_bias"]["real_image_correct_rate"] == pytest.approx(10 / 30)


def test_per_respondent_ab_uses_each_person_as_one_unit():
    key, ab = _keys()   # câu lẻ: ảnh thật bên phải (B); câu chẵn: ảnh thật bên trái (A)
    q = {f"q{i:02d}": "AI" for i in range(1, 21)}
    always_real = {f"ab{i:02d}": ("Ảnh B" if i % 2 else "Ảnh A") for i in range(1, 11)}
    fooled = {f"ab{i:02d}": ("Ảnh A" if i % 2 else "Ảnh B") for i in range(1, 11)}
    r = an.analyze(_responses([{**q, **always_real}] * 3 + [{**q, **fooled}]), key, ab)
    st = r["per_respondent"]["ab_real_chosen"]
    assert st["n"] == 4 and st["mean"] == pytest.approx(0.75)
    assert st["sd"] == pytest.approx(0.5)                             # 1, 1, 1, 0
    assert st["ci_low"] < 0.75 < st["ci_high"]
    assert st["boot"][0] <= 0.75 <= st["boot"][1]


def test_per_respondent_blank_answers_shrink_the_denominator():
    key, ab = _keys()
    row = {**{f"q{i:02d}": ("Thật" if i % 2 else "AI") for i in range(1, 21)},
           **{f"ab{i:02d}": ("Ảnh B" if i % 2 else "Ảnh A") for i in range(1, 11)}}
    row["q01"] = None
    row["ab01"] = None
    r = an.analyze(_responses([row]), key, ab)
    assert r["per_respondent"]["part1_accuracy"]["mean"] == pytest.approx(1.0)    # 19/19, không phải 19/20
    assert r["per_respondent"]["ab_real_chosen"]["mean"] == pytest.approx(1.0)    # 9/9


def test_position_bias_always_right():
    key, ab = _keys()
    q = {f"q{i:02d}": "AI" for i in range(1, 21)}
    always_b = {f"ab{i:02d}": "Ảnh B" for i in range(1, 11)}
    r = an.analyze(_responses([{**q, **always_b}] * 4), key, ab)
    pos = r["position"]
    assert pos["right_rate"] == pytest.approx(1.0) and pos["right_chosen"] == 40
    assert pos["real_right_pairs"] == 5 and pos["real_left_pairs"] == 5
    assert pos["real_right_rate"] == pytest.approx(1.0) and pos["real_left_rate"] == pytest.approx(0.0)
    assert pos["real_right_votes"] == pos["real_left_votes"] == 20
    assert r["ab_test"]["rate"] == pytest.approx(0.5)     # thiết kế cân bằng 5/5 triệt tiêu thiên lệch trong số gộp
    s = r["summary"]
    items = set(s[s.section == "ab_position"].item)
    assert {"right_chosen_rate", "real_left_correct_rate", "real_right_correct_rate"} <= items


def test_position_bias_old_design_has_no_real_split():
    key, ab = _keys("baseline")
    row = {**{f"q{i:02d}": "AI" for i in range(1, 21)}, **{f"ab{i:02d}": "Ảnh B" for i in range(1, 11)}}
    r = an.analyze(_responses([row]), key, ab)
    assert r["position"]["right_rate"] == pytest.approx(1.0)
    assert "real_left_rate" not in r["position"] and "ab_real_chosen" not in r["per_respondent"]


def test_cli_writes_per_respondent_and_position_rows(tmp_path):
    key, ab = _keys()
    key.to_csv(tmp_path / "key.csv", index=False)
    ab.to_csv(tmp_path / "ab.csv", index=False)
    rows = [{**{f"q{i:02d}": ("Thật" if (i + j) % 2 else "AI") for i in range(1, 21)},
             **{f"ab{i:02d}": ("Ảnh B" if (i + j) % 3 else "Ảnh A") for i in range(1, 11)}} for j in range(5)]
    _responses(rows).to_csv(tmp_path / "resp.csv", index=False)
    assert an.main(["--responses", str(tmp_path / "resp.csv"), "--survey-key", str(tmp_path / "key.csv"),
                    "--ab-key", str(tmp_path / "ab.csv"), "--out", str(tmp_path / "out")]) == 0
    s = pd.read_csv(tmp_path / "out" / "survey_summary.csv")
    pr = s[s.section == "per_respondent"].set_index("item").value
    assert pr["ab_real_chosen_ci95_low_t"] <= pr["ab_real_chosen_mean"] <= pr["ab_real_chosen_ci95_high_t"]
    assert (s[s.section == "per_respondent"].n == 5).all()
    assert "right_chosen_rate" in set(s[s.section == "ab_position"].item)
