"""Model zoo. Every model is a full pipeline (features + model) fitted on raw transactions."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FraudFeatureEngineer


def build_pipeline(model, scale: bool = False, night_hours=(0, 6)) -> Pipeline:
    steps = [("features", FraudFeatureEngineer(*night_hours))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def baseline_pipelines(seed: int, night_hours=(0, 6)) -> dict[str, Pipeline]:
    return {
        # Floor: predicts the fraud rate for everyone, PR-AUC = prevalence.
        "dummy": build_pipeline(DummyClassifier(strategy="prior"), night_hours=night_hours),
        # Interpretable baseline.
        "logistic_regression": build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=5000),
            scale=True,
            night_hours=night_hours,
        ),
        # Robust non-linear baseline.
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            ),
            night_hours=night_hours,
        ),
    }


def xgb_classifier(params: dict, seed: int) -> XGBClassifier:
    return XGBClassifier(
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=seed,
        **params,
    )


def xgb_pipeline(params: dict, seed: int, night_hours=(0, 6)) -> Pipeline:
    return build_pipeline(xgb_classifier(params, seed), night_hours=night_hours)
