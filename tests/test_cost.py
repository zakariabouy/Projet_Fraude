import numpy as np
import pandas as pd
import pytest

from src.cost import (
    CostParams,
    cost_summary,
    enforce_capacity,
    expected_net_benefit,
    optimise_threshold,
    threshold_curve,
    topk_policy,
    total_cost,
    transaction_costs,
)

PARAMS = CostParams(c_inv=5, c_fp=10, recovery_rate=0.8, capacity_per_hour=None)


def test_the_four_cases_by_hand():
    y = [1, 1, 0, 0]
    alert = [0, 1, 1, 0]
    amount = [100, 50, 30, 20]
    # FN: 100 | TP: 5 + 0.2 * 50 = 15 | FP: 5 + 10 = 15 | TN: 0
    np.testing.assert_allclose(transaction_costs(y, alert, amount, PARAMS), [100, 15, 15, 0])
    assert total_cost(y, alert, amount, PARAMS) == pytest.approx(130)


def test_total_cost_formula():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.1
    alert = rng.random(500) < 0.2
    a = rng.lognormal(3, 1, 500)
    expected = (
        a[y & ~alert].sum()
        + PARAMS.c_inv * alert.sum()
        + PARAMS.c_fp * (alert & ~y).sum()
        + (1 - PARAMS.recovery_rate) * a[y & alert].sum()
    )
    assert total_cost(y, alert, a, PARAMS) == pytest.approx(expected)


def test_cost_summary_counts_and_amount_recall():
    s = cost_summary([1, 1, 0, 0], [0, 1, 1, 0], [100, 50, 30, 20], PARAMS)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 1)
    assert s["amount_recall"] == pytest.approx(50 / 150)
    assert s["fraud_amount_lost"] == pytest.approx(100 + 0.2 * 50)


def test_no_alert_costs_all_the_fraud():
    assert total_cost([1, 0, 1], [0, 0, 0], [10, 99, 5], PARAMS) == pytest.approx(15)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        CostParams(c_inv=-1)
    with pytest.raises(ValueError):
        CostParams(recovery_rate=1.5)


def test_capacity_first_come_first_served():
    alert = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
    hour = np.array([0, 0, 0, 1, 1, 1])
    kept = enforce_capacity(alert, hour, capacity=2)
    np.testing.assert_array_equal(kept, [1, 1, 0, 1, 1, 0])


def test_capacity_by_priority():
    alert = np.ones(4, dtype=bool)
    hour = np.zeros(4, dtype=int)
    kept = enforce_capacity(alert, hour, capacity=2, priority=[1, 3, 2, 0])
    np.testing.assert_array_equal(kept, [0, 1, 1, 0])


def test_fast_and_general_capacity_paths_agree():
    rng = np.random.default_rng(1)
    hour = np.sort(rng.integers(0, 20, 2000))
    alert = rng.random(2000) < 0.3
    fast = enforce_capacity(alert, hour, capacity=7)
    shuffled = rng.permutation(2000)
    general = np.empty_like(alert)  # unsorted hours force the general (lexsort) path
    general[shuffled] = enforce_capacity(
        alert[shuffled], hour[shuffled], capacity=7, priority=-shuffled.astype(float)
    )
    np.testing.assert_array_equal(fast, general)


def test_topk_never_exceeds_capacity():
    rng = np.random.default_rng(2)
    n = 3000
    hour = np.sort(rng.integers(0, 10, n))
    alert = topk_policy(rng.random(n), rng.lognormal(3, 1, n), hour, capacity=5)
    per_hour = pd.Series(alert).groupby(hour).sum()
    assert per_hour.max() == 5


def test_topk_ranks_by_expected_loss():
    proba = np.array([0.9, 0.5, 0.1])
    amount = np.array([10.0, 100.0, 1000.0])  # expected loss: 9, 50, 100
    alert = topk_policy(proba, amount, np.zeros(3, dtype=int), capacity=1)
    np.testing.assert_array_equal(alert, [0, 0, 1])


def test_profitable_filter_skips_small_amounts():
    params = CostParams(c_inv=5, c_fp=10, recovery_rate=0.8, capacity_per_hour=10)
    # 0.9 * 0.8 * 1 - 5 - 0.1 * 10 < 0: not worth an investigation
    assert expected_net_benefit([0.9], [1.0], params)[0] < 0
    alert = topk_policy([0.9, 0.9], [1.0, 500.0], [0, 0], 10, params, only_profitable=True)
    np.testing.assert_array_equal(alert, [0, 1])


def test_optimal_threshold_separates_perfectly_ranked_scores():
    y = np.array([0] * 50 + [1] * 5)
    proba = np.r_[np.full(50, 0.1), np.full(5, 0.9)]
    amount = np.full(55, 1000.0)
    params = CostParams(c_inv=5, c_fp=10, recovery_rate=0.8, capacity_per_hour=100)
    curve = threshold_curve(y, proba, amount, np.zeros(55, dtype=int), params)
    t = optimise_threshold(curve)
    assert 0.1 < t <= 0.9
    assert cost_summary(y, proba >= t, amount, params)["fp"] == 0


def test_threshold_must_respect_capacity():
    # Frauds ranked 1st, 6th and 11th: catching them all needs 11 alerts in the same hour,
    # cheap without a constraint (11 EUR vs 100 EUR per missed fraud) but above K = 3.
    y = np.zeros(20, dtype=int)
    y[[0, 5, 10]] = 1
    proba = np.linspace(0.99, 0.01, 20)
    params = CostParams(c_inv=1, c_fp=0, recovery_rate=1.0, capacity_per_hour=3)
    curve = threshold_curve(y, proba, np.full(20, 100.0), np.zeros(20, dtype=int), params)
    t = optimise_threshold(curve)
    assert 1 <= (proba >= t).sum() <= 3
    unconstrained = threshold_curve(
        y, proba, np.full(20, 100.0), np.zeros(20, dtype=int), CostParams(1, 0, 1.0, None)
    )
    assert (proba >= optimise_threshold(unconstrained)).sum() == 11
