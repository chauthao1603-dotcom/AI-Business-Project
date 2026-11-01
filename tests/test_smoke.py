import pandas as pd
import pytest

from src.sus import grade, summarize, sus_score


def test_sus_best_and_worst():
    assert sus_score([5, 1, 5, 1, 5, 1, 5, 1, 5, 1]) == 100
    assert sus_score([1, 5, 1, 5, 1, 5, 1, 5, 1, 5]) == 0
    assert sus_score([3] * 10) == 50


def test_sus_validates_input():
    with pytest.raises(ValueError):
        sus_score([3] * 9)
    with pytest.raises(ValueError):
        sus_score([6] + [3] * 9)


def test_sus_sample_file():
    res = summarize(pd.read_csv("docs/sus_responses_sample.csv"))
    assert res["SUS"].between(0, 100).all() and len(res) == 5
    assert grade(res["SUS"].mean())
