import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.metrics import (
    _sorted_for_ap,
    _weighted_ap,
    bootstrap_cost_reduction,
    bootstrap_pr_auc,
    pr_auc,
    precision_at_k,
)


def test_fast_weighted_ap_matches_scikit_learn():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    score = np.round(rng.random(2000) + y * 0.5, 2)  # rounding creates ties
    weights = rng.integers(0, 3, 2000).astype(float)
    order, y_sorted, last = _sorted_for_ap(y, score)
    fast = _weighted_ap(y_sorted, weights[order], last)
    assert fast == pytest.approx(average_precision_score(y, score, sample_weight=weights))


def test_perfect_ranking_has_pr_auc_one():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    score = rng.random(5000) + y
    ci = bootstrap_pr_auc(y, score, n_resamples=200, seed=0)
    assert ci["lo"] <= ci["value"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.3


def test_precision_at_k():
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], k=2) == 0.5
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], k=0) == 0.0


def test_cost_reduction_point_estimate():
    ref = np.array([10.0, 10.0, 10.0, 10.0])
    new = np.array([5.0, 5.0, 5.0, 5.0])
    out = bootstrap_cost_reduction(ref, new, n_resamples=50)
    assert out["value"] == pytest.approx(0.5)
    assert out["lo"] == pytest.approx(0.5)
