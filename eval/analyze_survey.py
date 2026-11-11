"""PV3 – 3.4: biến câu trả lời Google Form thành ma trận nhầm lẫn và các chỉ số.

    python -m eval.analyze_survey --responses <file CSV tải từ Google Form>

Quy ước (docs/survey_form_guide.md): tiêu đề mỗi câu hỏi trong Form bắt đầu bằng mã q01…q20 (câu "thật hay AI")
và ab01…ab10 (câu "ảnh nào trông thật hơn"). Phần chữ sau mã được bỏ qua. Mỗi dòng của CSV là một người trả lời.

Lớp dương = "AI". Mỗi câu trả lời rơi vào đúng một ô:
    TP  ảnh AI,   chọn "AI"      FN  ảnh AI,   chọn "Thật"
    FP  ảnh thật, chọn "AI"      TN  ảnh thật, chọn "Thật"
Phần A/B: tỷ lệ người chọn ảnh của CatVTON.

Đầu ra:
    results/survey_summary.csv       section,item,value,n
    results/figs/confusion_matrix.png
File trả lời Form có thể chứa email hoặc thông tin cá nhân, không commit lên GitHub.
"""
from __future__ import annotations

import argparse
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


def analyze(responses: pd.DataFrame, survey_key: pd.DataFrame, ab_key: pd.DataFrame) -> dict:
    cols = find_question_columns(responses)
    truth = dict(zip(survey_key["qid"].str.lower(), survey_key["label"].str.lower()))
    missing_q = sorted(set(truth) - set(cols))
    if missing_q:
        raise SystemExit(f"File trả lời thiếu cột cho các câu: {missing_q}. Kiểm tra tiêu đề câu hỏi trong Form "
                         f"có bắt đầu bằng mã q01…q20 không.")

    # ---- thật hay AI
    counts = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    per_q, invalid, blank = [], 0, 0
    for qid, label in truth.items():
        ok = n = 0
        for v in responses[cols[qid]]:
            p = parse_class(v)
            if p is None:
                blank += 1
                continue
            if p == "invalid":
                invalid += 1
                continue
            n += 1
            ok += p == label
            if label == "ai":
                counts["TP" if p == "ai" else "FN"] += 1
            else:
                counts["FP" if p == "ai" else "TN"] += 1
        per_q.append((qid, ok / n if n else float("nan"), n))
    metrics = classification_metrics(counts["TP"], counts["FN"], counts["FP"], counts["TN"])
    total = sum(counts.values())
    respondents = len(responses)
    expected = respondents * len(truth)

    # ---- A/B
    ab_rows, cat_votes, ab_n = [], 0, 0
    for r in ab_key.itertuples(index=False):
        code = r.ab_id.lower()
        if code not in cols:
            continue
        votes = cat = 0
        for v in responses[cols[code]]:
            s = parse_side(v)
            if s is None:
                blank += 1
                continue
            if s == "invalid":
                invalid += 1
                continue
            chosen = r.left_method if s == "A" else r.right_method
            votes += 1
            cat += chosen == "catvton"
        ab_rows.append((r.ab_id, cat / votes if votes else float("nan"), votes))
        cat_votes += cat
        ab_n += votes
    cat_rate = cat_votes / ab_n if ab_n else float("nan")

    rows = [("respondents", "so_nguoi_tra_loi", respondents, respondents),
            ("respondents", "cau_tra_loi_hop_le_phan_that_hay_ai", total, expected),
            ("respondents", "o_trong_hoac_gia_tri_la", blank + invalid, np.nan)]
    rows += [("confusion", k, v, total) for k, v in counts.items()]
    rows += [("classification", k, v, total) for k, v in metrics.items()]
    rows += [("per_question", q, a, n) for q, a, n in per_q]
    rows += [("ab", "catvton_rate", cat_rate, ab_n),
             ("ab", "baseline_rate", 1 - cat_rate if ab_n else float("nan"), ab_n)]
    rows += [("ab_pair", a, c, n) for a, c, n in ab_rows]
    return {"summary": pd.DataFrame(rows, columns=SUMMARY_COLS), "counts": counts, "metrics": metrics,
            "total": total, "expected": expected, "respondents": respondents,
            "blank": blank, "invalid": invalid, "cat_rate": cat_rate, "ab_n": ab_n}


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
    print(f"A/B: {res['cat_rate']:.1%} lượt chọn CatVTON (trên {res['ab_n']} lượt)")
    if res["total"] != res["expected"] or res["invalid"]:
        print(f"CHÚ Ý: tổng ô ({res['total']}) khác số người × số câu ({res['expected']}); "
              f"{res['blank']} ô trống, {res['invalid']} giá trị lạ. Kiểm tra đáp án trong Form.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
