"""PV3 – 3.4: biến câu trả lời Google Form thành ma trận nhầm lẫn và các chỉ số.

    python -m eval.analyze_survey --responses <file CSV tải từ Google Form>

Quy ước (docs/survey_form_guide.md): tiêu đề mỗi câu hỏi trong Form bắt đầu bằng mã q01…q20 (câu "thật hay AI")
và ab01…ab10 (câu "ảnh nào là ảnh chụp thật"). Phần chữ sau mã được bỏ qua. Mỗi dòng của CSV là một người trả lời.

Lớp dương = "AI". Mỗi câu trả lời rơi vào đúng một ô:
    TP  ảnh AI,   chọn "AI"      FN  ảnh AI,   chọn "Thật"
    FP  ảnh thật, chọn "AI"      TN  ảnh thật, chọn "Thật"
Phần A/B (mỗi câu một ảnh chụp thật và một ảnh CatVTON của cùng người, cùng áo): tỷ lệ người xem chọn đúng
ảnh thật, kèm khoảng tin cậy 95% (Wilson) và p-value của kiểm định nhị thức hai phía so với 50%.
Gần 50% nghĩa là người xem không phân biệt được ảnh CatVTON với ảnh chụp thật. Mỗi người trả lời 10 câu nên các
lượt trả lời không hoàn toàn độc lập; khoảng tin cậy và p-value tính trên lượt chỉ là ước lượng gần đúng.
Vì vậy script báo thêm ước lượng ở cấp người trả lời: tính tỷ lệ đúng của từng người rồi lấy trung bình, khoảng tin
cậy t và khoảng bootstrap lấy mẫu lại theo người (10 000 lần, seed 42). Phần A/B còn báo thiên lệch vị trí: tỷ lệ
chọn ảnh bên phải (B) và tỷ lệ chọn đúng tách theo vị trí của ảnh thật.
File ab_key.csv theo thiết kế cũ (baseline – CatVTON) vẫn đọc được: khi đó chỉ báo tỷ lệ chọn từng phương pháp.

Đầu ra:
    results/survey_summary.csv       section,item,value,n
        respondents, confusion, classification, part1_bias, per_question,
        ab, ab_position, ab_pair_catvton_chosen, per_respondent
    results/figs/confusion_matrix.png
File trả lời Form có thể chứa email hoặc thông tin cá nhân, không commit lên GitHub.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

CODE_RE = re.compile(r"^\s*(q\d{2}|ab\d{2})\b", re.IGNORECASE)
SUMMARY_COLS = ["section", "item", "value", "n"]


def canon(v) -> Optional[str]:
    """Bỏ dấu, hạ chữ thường, cắt khoảng trắng; ô trống -> None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().strip().lower()
    return s or None


def parse_class(v) -> Optional[str]:
    """'Thật' -> 'real', 'AI' -> 'ai'; giá trị lạ -> 'invalid'; trống -> None."""
    s = canon(v)
    if s is None:
        return None
    if s in {"that", "anh that", "real", "thuc"}:
        return "real"
    if s in {"ai", "anh ai"}:
        return "ai"
    return "invalid"


def parse_side(v) -> Optional[str]:
    """'Ảnh A' -> 'A', 'Ảnh B' -> 'B'; giá trị lạ -> 'invalid'; trống -> None."""
    s = canon(v)
    if s is None:
        return None
    if s in {"a", "anh a"}:
        return "A"
    if s in {"b", "anh b"}:
        return "B"
    return "invalid"


def find_question_columns(df: pd.DataFrame) -> dict:
    """{'q01': tên cột, ..., 'ab01': tên cột}. Báo lỗi nếu một mã xuất hiện ở hai cột."""
    cols = {}
    for c in df.columns:
        m = CODE_RE.match(str(c))
        if m:
            code = m.group(1).lower()
            if code in cols:
                raise SystemExit(f"Mã câu hỏi {code} xuất hiện ở hai cột: {cols[code]!r} và {c!r}.")
            cols[code] = c
    return cols


def classification_metrics(tp: int, fn: int, fp: int, tn: int) -> dict:
    total = tp + fn + fp + tn
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else float("nan")
    return {"accuracy": div(tp + tn, total), "precision": precision, "recall": recall, "f1": f1}


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple:
    """Khoảng tin cậy 95% Wilson cho tỷ lệ k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_test_half(k: int, n: int) -> float:
    """p-value hai phía của kiểm định nhị thức chính xác, H0: tỷ lệ = 0,5."""
    if n == 0:
        return float("nan")
    combs = [math.comb(n, i) for i in range(n + 1)]   # số nguyên: không tràn số khi n lớn
    return min(1.0, sum(c for c in combs if c <= combs[k]) / 2 ** n)


def _t_cdf(x: float, df: int, steps: int = 4000) -> float:
    """Hàm phân phối Student t (tích phân Simpson của hàm mật độ từ 0 đến x); đủ chính xác cho khoảng tin cậy."""
    if x == 0:
        return 0.5
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    xs = np.linspace(0.0, abs(x), steps + 1)
    f = c * (1 + xs * xs / df) ** (-(df + 1) / 2)
    area = (abs(x) / steps / 3) * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
    return 0.5 + area if x > 0 else 0.5 - area


def t_ppf(q: float, df: int) -> float:
    """Phân vị q của phân phối Student t với df bậc tự do (chia đôi trên _t_cdf, không cần scipy)."""
    if not 0 < q < 1 or df < 1:
        return float("nan")
    if q < 0.5:
        return -t_ppf(1 - q, df)
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < q:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2


def mean_ci_t(values, level: float = 0.95) -> dict:
    """Trung bình, độ lệch chuẩn mẫu (ddof=1), min, max và khoảng tin cậy t của trung bình."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    n = len(v)
    out = {"mean": float("nan"), "sd": float("nan"), "min": float("nan"), "max": float("nan"),
           "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    if n == 0:
        return out
    out.update(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    if n >= 2:
        sd = float(v.std(ddof=1))
        half = t_ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n)
        out.update(sd=sd, ci_low=out["mean"] - half, ci_high=out["mean"] + half)
    return out


def bootstrap_mean_ci(values, n_boot: int = 10_000, seed: int = 42, level: float = 0.95) -> tuple:
    """Khoảng bootstrap phân vị cho trung bình: lấy mẫu lại có hoàn lại các giá trị (mỗi giá trị = một người)."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def analyze(responses: pd.DataFrame, survey_key: pd.DataFrame, ab_key: pd.DataFrame) -> dict:
    cols = find_question_columns(responses)
    truth = dict(zip(survey_key["qid"].str.lower(), survey_key["label"].str.lower()))
    missing_q = sorted(set(truth) - set(cols))
    if missing_q:
        raise SystemExit(f"File trả lời thiếu cột cho các câu: {missing_q}. Kiểm tra tiêu đề câu hỏi trong Form "
                         f"có bắt đầu bằng mã q01…q20 không.")

    respondents = len(responses)
    # ---- thật hay AI
    counts = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    per_q, invalid, blank = [], 0, 0
    p1_ok, p1_n = np.zeros(respondents), np.zeros(respondents)     # số câu đúng / số câu hợp lệ của từng người
    for qid, label in truth.items():
        ok = n = 0
        for i, v in enumerate(responses[cols[qid]]):
            p = parse_class(v)
            if p is None:
                blank += 1
                continue
            if p == "invalid":
                invalid += 1
                continue
            n += 1
            ok += p == label
            p1_n[i] += 1
            p1_ok[i] += p == label
            if label == "ai":
                counts["TP" if p == "ai" else "FN"] += 1
            else:
                counts["FP" if p == "ai" else "TN"] += 1
        per_q.append((qid, ok / n if n else float("nan"), n))
    metrics = classification_metrics(counts["TP"], counts["FN"], counts["FP"], counts["TN"])
    total = sum(counts.values())
    expected = respondents * len(truth)
    n_ai_answers = counts["TP"] + counts["FP"]
    part1_bias = {"ai_answer_rate": (n_ai_answers / total if total else float("nan"), total),
                  "real_image_correct_rate": (counts["TN"] / (counts["TN"] + counts["FP"])
                                              if counts["TN"] + counts["FP"] else float("nan"),
                                              counts["TN"] + counts["FP"])}

    # ---- A/B: đếm số lượt chọn từng nguồn (real / catvton, hoặc baseline / catvton ở thiết kế cũ)
    methods = sorted(set(ab_key["left_method"].str.lower()) | set(ab_key["right_method"].str.lower()))
    chosen_total = dict.fromkeys(methods, 0)
    ab_rows, ab_n = [], 0
    ab_ok, ab_nr = np.zeros(respondents), np.zeros(respondents)    # số lần chọn đúng ảnh thật / số câu hợp lệ
    right_chosen = 0
    # theo vị trí của ảnh thật: [số lượt hợp lệ, số lượt chọn đúng ảnh thật, số câu]
    by_pos = {"left": [0, 0, 0], "right": [0, 0, 0]}
    for r in ab_key.itertuples(index=False):
        code = r.ab_id.lower()
        if code not in cols:
            continue
        left, right = r.left_method.lower(), r.right_method.lower()
        real_pos = "left" if left == "real" else "right" if right == "real" else None
        if real_pos:
            by_pos[real_pos][2] += 1
        votes, cat = 0, 0
        for i, v in enumerate(responses[cols[code]]):
            s = parse_side(v)
            if s is None:
                blank += 1
                continue
            if s == "invalid":
                invalid += 1
                continue
            chosen = left if s == "A" else right
            votes += 1
            cat += chosen == "catvton"
            chosen_total[chosen] += 1
            right_chosen += s == "B"
            if real_pos:
                ab_nr[i] += 1
                ab_ok[i] += chosen == "real"
                by_pos[real_pos][0] += 1
                by_pos[real_pos][1] += chosen == "real"
        ab_rows.append((r.ab_id, cat / votes if votes else float("nan"), votes))
        ab_n += votes
    ab_rates = {m: (c / ab_n if ab_n else float("nan")) for m, c in chosen_total.items()}
    cat_rate = ab_rates.get("catvton", float("nan"))
    ab_test = None
    if "real" in chosen_total:
        k = chosen_total["real"]
        lo, hi = wilson_ci(k, ab_n)
        ab_test = {"real_chosen": k, "rate": ab_rates["real"], "ci_low": lo, "ci_high": hi,
                   "p_value": binom_test_half(k, ab_n)}

    # ---- thiên lệch vị trí ở phần A/B
    position = None
    if ab_n:
        lo, hi = wilson_ci(right_chosen, ab_n)
        position = {"right_chosen": right_chosen, "right_rate": right_chosen / ab_n, "ci_low": lo, "ci_high": hi,
                    "p_value": binom_test_half(right_chosen, ab_n)}
        if ab_test:
            for pos, (votes, ok, n_pairs) in by_pos.items():
                position[f"real_{pos}_rate"] = ok / votes if votes else float("nan")
                position[f"real_{pos}_votes"] = votes
                position[f"real_{pos}_pairs"] = n_pairs

    # ---- ước lượng ở cấp người trả lời (mỗi người là một đơn vị độc lập)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1_rate = np.where(p1_n > 0, p1_ok / np.where(p1_n > 0, p1_n, 1), np.nan)
        ab_rate = np.where(ab_nr > 0, ab_ok / np.where(ab_nr > 0, ab_nr, 1), np.nan)
    per_resp = {"part1_accuracy": {**mean_ci_t(p1_rate), "boot": bootstrap_mean_ci(p1_rate)}}
    if ab_test:
        per_resp["ab_real_chosen"] = {**mean_ci_t(ab_rate), "boot": bootstrap_mean_ci(ab_rate)}

    rows = [("respondents", "so_nguoi_tra_loi", respondents, respondents),
            ("respondents", "cau_tra_loi_hop_le_phan_that_hay_ai", total, expected),
            ("respondents", "o_trong_hoac_gia_tri_la", blank + invalid, np.nan)]
    rows += [("confusion", k, v, total) for k, v in counts.items()]
    rows += [("classification", k, v, total) for k, v in metrics.items()]
    rows += [("part1_bias", k, v, n) for k, (v, n) in part1_bias.items()]
    rows += [("per_question", q, a, n) for q, a, n in per_q]
    rows += [("ab", f"{m}_chosen_rate", rate, ab_n) for m, rate in ab_rates.items()]
    if ab_test:
        rows += [("ab", "real_chosen_ci95_low", ab_test["ci_low"], ab_n),
                 ("ab", "real_chosen_ci95_high", ab_test["ci_high"], ab_n),
                 ("ab", "p_value_vs_50pct", ab_test["p_value"], ab_n)]
    if position:
        rows += [("ab_position", "right_chosen_rate", position["right_rate"], ab_n),
                 ("ab_position", "right_chosen_ci95_low", position["ci_low"], ab_n),
                 ("ab_position", "right_chosen_ci95_high", position["ci_high"], ab_n),
                 ("ab_position", "right_chosen_p_value_vs_50pct", position["p_value"], ab_n)]
        for pos in ("left", "right"):
            if f"real_{pos}_rate" in position:
                rows += [("ab_position", f"real_{pos}_pairs", position[f"real_{pos}_pairs"], len(ab_rows)),
                         ("ab_position", f"real_{pos}_correct_rate", position[f"real_{pos}_rate"],
                          position[f"real_{pos}_votes"])]
    rows += [("ab_pair_catvton_chosen", a, c, n) for a, c, n in ab_rows]
    for name, st in per_resp.items():
        rows += [("per_respondent", f"{name}_{k}", st[k], st["n"]) for k in ("mean", "sd", "min", "max")]
        rows += [("per_respondent", f"{name}_ci95_low_t", st["ci_low"], st["n"]),
                 ("per_respondent", f"{name}_ci95_high_t", st["ci_high"], st["n"]),
                 ("per_respondent", f"{name}_ci95_low_bootstrap", st["boot"][0], st["n"]),
                 ("per_respondent", f"{name}_ci95_high_bootstrap", st["boot"][1], st["n"])]
    return {"summary": pd.DataFrame(rows, columns=SUMMARY_COLS), "counts": counts, "metrics": metrics,
            "total": total, "expected": expected, "respondents": respondents,
            "blank": blank, "invalid": invalid, "cat_rate": cat_rate, "ab_rates": ab_rates,
            "ab_test": ab_test, "ab_n": ab_n, "part1_bias": {k: v for k, (v, _) in part1_bias.items()},
            "position": position, "per_respondent": per_resp}


def plot_confusion(counts: dict, path: Path) -> None:
    """Ma trận 2×2: hàng = thực tế (AI, Thật), cột = người xem chọn (AI, Thật)."""
    m = np.array([[counts["TP"], counts["FN"]], [counts["FP"], counts["TN"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{['TP', 'FN', 'FP', 'TN'][i * 2 + j]}\n{v}", ha="center", va="center",
                color="white" if v > m.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Chọn “AI”", "Chọn “Thật”"])
    ax.set_yticks([0, 1], ["Ảnh là AI", "Ảnh là thật"])
    ax.set_xlabel("Người xem trả lời")
    ax.set_ylabel("Thực tế")
    ax.set_title("Ma trận nhầm lẫn (lớp dương = AI)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phân tích kết quả khảo sát Google Form.")
    ap.add_argument("--responses", required=True, help="file CSV tải từ Google Form")
    ap.add_argument("--survey-key", default=str(config.RESULTS / "survey_key.csv"))
    ap.add_argument("--ab-key", default=str(config.RESULTS / "ab_key.csv"))
    ap.add_argument("--out", default=str(config.RESULTS))
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    res = analyze(pd.read_csv(a.responses, dtype=str, encoding="utf-8-sig"), pd.read_csv(a.survey_key, dtype=str, encoding="utf-8-sig"),
                  pd.read_csv(a.ab_key, dtype=str, encoding="utf-8-sig"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res["summary"].to_csv(out / "survey_summary.csv", index=False, float_format="%.6f")
    plot_confusion(res["counts"], out / "figs" / "confusion_matrix.png")
    c, m = res["counts"], res["metrics"]
    print(f"{res['respondents']} người trả lời | TP={c['TP']} FN={c['FN']} FP={c['FP']} TN={c['TN']} "
          f"(tổng {res['total']}, kỳ vọng {res['expected']})")
    print(f"accuracy={m['accuracy']:.3f}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")
    t = res["ab_test"]
    if t:
        print(f"A/B: người xem chọn đúng ảnh chụp thật {t['rate']:.1%} ({t['real_chosen']}/{res['ab_n']} lượt), "
              f"CI95 [{t['ci_low']:.1%}; {t['ci_high']:.1%}], p = {t['p_value']:.3f} so với 50% "
              f"(50% = không phân biệt được ảnh CatVTON với ảnh thật)")
    else:
        print(f"A/B: {res['cat_rate']:.1%} lượt chọn CatVTON (trên {res['ab_n']} lượt)")
    b = res["part1_bias"]
    print(f"Phần 1: người xem chọn \"AI\" ở {b['ai_answer_rate']:.1%} số lượt; "
          f"nhận đúng ảnh chụp thật {b['real_image_correct_rate']:.1%}")
    labels = {"part1_accuracy": "Phần 1 accuracy", "ab_real_chosen": "A/B chọn đúng ảnh thật"}
    for name, st in res["per_respondent"].items():
        print(f"Theo người ({labels[name]}, n = {st['n']}): trung bình {st['mean']:.1%}, SD {st['sd']:.1%}, "
              f"[{st['min']:.0%} – {st['max']:.0%}], CI95 t [{st['ci_low']:.1%}; {st['ci_high']:.1%}], "
              f"bootstrap [{st['boot'][0]:.1%}; {st['boot'][1]:.1%}]")
    pos = res["position"]
    if pos:
        line = (f"Vị trí: người xem chọn ảnh bên phải (B) {pos['right_rate']:.1%}, "
                f"CI95 [{pos['ci_low']:.1%}; {pos['ci_high']:.1%}], p = {pos['p_value']:.3f}")
        if "real_left_rate" in pos:
            line += (f"; ảnh thật bên trái: chọn đúng {pos['real_left_rate']:.1%} ({pos['real_left_pairs']} câu), "
                     f"bên phải: {pos['real_right_rate']:.1%} ({pos['real_right_pairs']} câu)")
        print(line)
    if res["total"] != res["expected"] or res["invalid"]:
        print(f"CHÚ Ý: tổng ô ({res['total']}) khác số người × số câu ({res['expected']}); "
              f"{res['blank']} ô trống, {res['invalid']} giá trị lạ. Kiểm tra đáp án trong Form.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
