"""Evaluation metrics and bootstrap confidence intervals.

With ~100 frauds in the test set, a 2-point PR-AUC gap between two models can be noise:
every headline number is reported with a 95% bootstrap interval (1,000 resamples).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def pr_auc(y_true, score) -> float:
    """Average precision (area under the precision-recall curve). Random model = fraud rate."""
    return float(average_precision_score(y_true, score))


def roc_auc(y_true, score) -> float:
    return float(roc_auc_score(y_true, score))


def brier(y_true, proba) -> float:
    return float(brier_score_loss(y_true, proba))


def logloss(y_true, proba) -> float:
    return float(log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1]))


def precision_at_k(y_true, score, k: int) -> float:
    """Share of true frauds among the k highest scores."""
    y = np.asarray(y_true)
    k = min(int(k), len(y))
    if k <= 0:
        return 0.0
    top = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:k]
    return float(y[top].mean())


def _sorted_for_ap(y_true, score):
    """Sort once by decreasing score; return labels and the last index of each score tie."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = np.asarray(y_true, dtype=float)[order]
    s = np.asarray(score, dtype=float)[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    return order, y, last_of_tie


def _weighted_ap(y_sorted, w_sorted, last_of_tie) -> float:
    """Average precision with sample weights, same definition as scikit-learn."""
    tps = np.cumsum(w_sorted * y_sorted)[last_of_tie]
    fps = np.cumsum(w_sorted * (1 - y_sorted))[last_of_tie]
    if tps[-1] == 0:
        return np.nan
    precision = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) > 0)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def bootstrap_pr_auc(y_true, score, n_resamples=1000, alpha=0.05, seed=0) -> dict:
    """PR-AUC with a percentile bootstrap interval.

    A bootstrap resample is equivalent to integer weights on the original rows, so the
    scores are sorted once and each resample costs O(n) instead of a full re-sort.
    """
    rng = np.random.default_rng(seed)
    order, y_sorted, last_of_tie = _sorted_for_ap(y_true, score)
    n = len(y_sorted)
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        weights = np.bincount(rng.integers(0, n, n), minlength=n)[order]
        values[b] = _weighted_ap(y_sorted, weights.astype(float), last_of_tie)
    values = values[~np.isnan(values)]
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {"value": pr_auc(y_true, score), "lo": float(lo), "hi": float(hi)}


def bootstrap_cost_reduction(
    cost_reference, cost_new, n_resamples=1000, alpha=0.05, seed=0
) -> dict:
    """Relative cost reduction 1 - C_new / C_ref, paired bootstrap over transactions.

    The decisions are held fixed and the transactions are resampled. This is an
    approximation for top-K (whose decisions depend on the other rows of the hour).
    """
    ref = np.asarray(cost_reference, dtype=float)
    new = np.asarray(cost_new, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(ref)
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        values[b] = 1 - (w @ new) / (w @ ref)
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {"value": float(1 - new.sum() / ref.sum()), "lo": float(lo), "hi": float(hi)}


def ranking_metrics(y_true, score, k: int) -> dict:
    return {
        "pr_auc": pr_auc(y_true, score),
        "roc_auc": roc_auc(y_true, score),
        "precision_at_k": precision_at_k(y_true, score, k),
    }
