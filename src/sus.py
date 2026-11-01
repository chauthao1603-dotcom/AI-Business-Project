"""Tính điểm SUS (System Usability Scale, Brooke 1996).

File CSV đầu vào: mỗi dòng 1 người tham gia, cột q1..q10 với giá trị 1-5.
Chạy:  python -m src.sus docs/sus_responses_sample.csv
"""
from __future__ import annotations

import sys

import pandas as pd


def sus_score(answers: list[int]) -> float:
    """Câu lẻ (1,3,5,7,9): điểm-1. Câu chẵn (2,4,6,8,10): 5-điểm. Tổng x 2.5 -> thang 0-100."""
    if len(answers) != 10 or any(a < 1 or a > 5 for a in answers):
        raise ValueError("Cần đúng 10 câu trả lời, mỗi câu từ 1 đến 5.")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5


def grade(score: float) -> str:
    """Mốc tham khảo: trung bình ngành ~68."""
    if score >= 80.3: return "Rất tốt (A)"
    if score >= 68: return "Đạt/khá (≥ trung bình)"
    if score >= 51: return "Dưới trung bình (cần cải thiện)"
    return "Kém (cần thiết kế lại)"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"q{i}" for i in range(1, 11)]
    out = df.copy()
    out["SUS"] = out[cols].apply(lambda r: sus_score(r.tolist()), axis=1)
    return out


if __name__ == "__main__":
    res = summarize(pd.read_csv(sys.argv[1]))
    print(res[["SUS"]].round(1).to_string())
    m = res["SUS"].mean()
    print(f"\nn = {len(res)} | SUS trung bình = {m:.1f} ± {res['SUS'].std():.1f} | {grade(m)}")
