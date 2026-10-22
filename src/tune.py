"""XGBoost hyper-parameter search: Optuna (TPE) + chronological cross-validation.

The feature step is refitted inside every fold, so the per-hour amount statistics never
see the fold being scored. Objective: mean PR-AUC across TimeSeriesSplit folds.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.metrics import pr_auc
from src.models import xgb_pipeline

log = logging.getLogger(__name__)


def default_params(scale_pos_weight: float) -> dict:
    return {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    }


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    cv_splits: int = 4,
    seed: int = 42,
    timeout: float | None = None,
    night_hours=(0, 6),
) -> tuple[dict, optuna.Study]:
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))  # n_neg / n_pos
    folds = list(TimeSeriesSplit(n_splits=cv_splits).split(X))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 1.2 * ratio, log=True),
        }
        scores = []
        for step, (tr, va) in enumerate(folds):
            if y.iloc[va].sum() == 0:
                continue
            model = xgb_pipeline(params, seed, night_hours).fit(X.iloc[tr], y.iloc[tr])
            scores.append(pr_auc(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_pr_auc", scores)
        return float(np.mean(scores))

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % 10 == 0:
            log.info(
                "Optuna trial %d/%d - best CV PR-AUC %.4f",
                trial.number + 1,
                n_trials,
                study.best_value,
            )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name="xgboost_pr_auc",
    )
    # First trial = the documented default (scale_pos_weight = n_neg / n_pos).
    study.enqueue_trial(default_params(ratio))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[progress])
    best = {**study.best_params}
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    return best, study
