"""Probability calibration of the frozen (already trained) model.

Class weights (scale_pos_weight) distort the scores: the model over-estimates the risk.
The decision uses p * amount, so p must be a real probability, not only a good ranking.

Calibrating and choosing the threshold on the same rows overfits, so validation is
cross-fitted: it is cut in chronological blocks, each block is scored by a calibrator
fitted on the other blocks. Those out-of-fold probabilities are used to compare
isotonic vs Platt (sigmoid) and to choose the threshold. The calibrator shipped to
production is then refitted on the whole validation period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.metrics import brier, logloss, pr_auc


def fit_calibrator(model, X_cal: pd.DataFrame, y_cal, method: str) -> CalibratedClassifierCV:
    """Calibrate an already fitted model without refitting it (FrozenEstimator)."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_cal, y_cal)


def chronological_blocks(n: int, n_blocks: int) -> list[np.ndarray]:
    return np.array_split(np.arange(n), n_blocks)


def cross_fit_probabilities(
    model, X_valid: pd.DataFrame, y_valid, method: str, n_blocks: int = 2
) -> np.ndarray:
    """Out-of-fold calibrated probabilities on the validation period."""
    y_valid = np.asarray(y_valid)
    oof = np.empty(len(y_valid), dtype=float)
    for block in chronological_blocks(len(y_valid), n_blocks):
        rest = np.setdiff1d(np.arange(len(y_valid)), block)
        calibrator = fit_calibrator(model, X_valid.iloc[rest], y_valid[rest], method)
        oof[block] = calibrator.predict_proba(X_valid.iloc[block])[:, 1]
    return oof


def compare_calibrations(
    model,
    X_valid: pd.DataFrame,
    y_valid,
    methods=("isotonic", "sigmoid"),
    n_blocks: int = 2,
    tie_tolerance: float = 0.01,
) -> tuple[str, dict[str, np.ndarray], pd.DataFrame]:
    """Score raw vs calibrated probabilities out-of-fold and pick a method.

    Primary criterion: Brier score. Methods within `tie_tolerance` (relative) of the best
    Brier are tied and the lowest log-loss wins: it punishes near-zero probabilities given
    to frauds, which matter when the decision uses p * amount.
    """
    y_valid = np.asarray(y_valid)
    probas = {"raw": model.predict_proba(X_valid)[:, 1]}
    for method in methods:
        probas[method] = cross_fit_probabilities(model, X_valid, y_valid, method, n_blocks)
    table = pd.DataFrame(
        [
            {
                "method": name,
                "brier": brier(y_valid, p),
                "log_loss": logloss(y_valid, p),
                "pr_auc": pr_auc(y_valid, p),
                "mean_proba": float(np.mean(p)),
                "frauds_below_1e-4": int(((y_valid == 1) & (p < 1e-4)).sum()),
            }
            for name, p in probas.items()
        ]
    )
    candidates = table[table["method"] != "raw"]
    tied = candidates[candidates["brier"] <= candidates["brier"].min() * (1 + tie_tolerance)]
    best = tied.sort_values("log_loss").iloc[0]["method"]
    return str(best), probas, table


def base_pipeline(calibrated: CalibratedClassifierCV):
    """The fitted features + XGBoost pipeline inside a calibrated model (for SHAP)."""
    inner = calibrated.calibrated_classifiers_[0].estimator
    return inner.estimator if isinstance(inner, FrozenEstimator) else inner
