"""Explainability with SHAP (TreeSHAP on the XGBoost model).

SHAP splits "this prediction - the average prediction" between the features, additively
(Shapley values). Values are in log-odds of the *uncalibrated* model: isotonic / Platt
calibration is monotone, so the direction and ranking of the contributions still hold.
SHAP explains the model, not causality in the real world.

The API uses XGBoost's native TreeSHAP (`pred_contribs=True`): same values as
shap.TreeExplainer (checked in tests), without shipping the shap library in the image.
Plots live in `src.plots`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import FEATURE_LABELS


def native_contributions(pipeline, X_raw: pd.DataFrame):
    """TreeSHAP values computed by XGBoost: (contributions, bias, transformed features)."""
    X_t = pipeline.named_steps["features"].transform(X_raw)
    booster = pipeline.named_steps["model"].get_booster()
    contrib = booster.predict(xgb.DMatrix(X_t), pred_contribs=True)
    return pd.DataFrame(contrib[:, :-1], columns=X_t.columns, index=X_t.index), contrib[:, -1], X_t


def reason_codes(contrib_row: pd.Series, values_row: pd.Series, top: int = 3) -> list[dict]:
    """The `top` features that push this transaction the most towards fraud."""
    ranked = contrib_row.sort_values(ascending=False).head(top)
    return [
        {
            "feature": name,
            "label": FEATURE_LABELS.get(name, f"PCA component {name}"),
            "value": float(values_row[name]),
            "contribution": float(value),
            "direction": "increases_risk" if value > 0 else "decreases_risk",
        }
        for name, value in ranked.items()
    ]


def shap_explanation(pipeline, X_raw: pd.DataFrame):
    """shap.Explanation for raw transactions (shap.TreeExplainer on the fitted XGBoost)."""
    import shap

    X_t = pipeline.named_steps["features"].transform(X_raw)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    return explainer(X_t)


def global_importance(explanation) -> pd.DataFrame:
    values = np.abs(np.asarray(explanation.values)).mean(axis=0)
    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": values})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
