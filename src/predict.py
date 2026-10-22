"""The production artefact and the scorer shared by the API and the dashboard.

The bundle holds one scikit-learn object (features -> XGBoost -> calibrator) plus the
decision parameters chosen on validation. Training, API and demo all use this object.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.calibration import base_pipeline
from src.cost import CostParams, expected_loss, expected_net_benefit, topk_policy
from src.data import RAW_FEATURES
from src.explain import native_contributions, reason_codes
from src.features import hour_index


def make_bundle(
    calibrated_model,
    threshold: float,
    cost_params: CostParams,
    calibration_method: str,
    version: str,
    metadata: dict | None = None,
) -> dict:
    return {
        "model": calibrated_model,
        "threshold": float(threshold),
        "cost_params": cost_params.to_dict(),
        "calibration_method": calibration_method,
        "version": version,
        "raw_features": list(RAW_FEATURES),
        "metadata": metadata or {},
    }


def save_bundle(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)


class FraudScorer:
    """Calibrated probability, cost-based decision and SHAP reason codes."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.pipeline = base_pipeline(self.model)
        self.threshold = float(bundle["threshold"])
        self.cost_params = CostParams(**bundle["cost_params"])
        self.version = bundle["version"]
        self.calibration_method = bundle["calibration_method"]

    @classmethod
    def from_path(cls, path: str | Path) -> FraudScorer:
        return cls(load_bundle(path))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[RAW_FEATURES])[:, 1]

    def explain(self, X: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        contrib, _, X_t = native_contributions(self.pipeline, X[RAW_FEATURES])
        return [reason_codes(contrib.iloc[i], X_t.iloc[i], top) for i in range(len(X))]

    def score(self, X: pd.DataFrame, with_reasons: bool = True) -> pd.DataFrame:
        """One row per transaction: probability, expected loss, threshold decision, reasons."""
        X = X.reset_index(drop=True)
        proba = self.predict_proba(X)
        out = pd.DataFrame(
            {
                "fraud_probability": proba,
                "expected_loss": expected_loss(proba, X["Amount"]),
                "expected_net_benefit": expected_net_benefit(proba, X["Amount"], self.cost_params),
                "alert": proba >= self.threshold,
            }
        )
        if with_reasons:
            out["reason_codes"] = self.explain(X)
        return out

    def prioritise(self, X: pd.DataFrame, capacity: int | None = None) -> np.ndarray:
        """Top-K by expected loss within each hour (only profitable investigations)."""
        X = X.reset_index(drop=True)
        capacity = self.cost_params.capacity_per_hour if capacity is None else capacity
        order = np.argsort(X["Time"].to_numpy(), kind="mergesort")
        proba = self.predict_proba(X)
        selected_sorted = topk_policy(
            proba[order],
            X["Amount"].to_numpy()[order],
            hour_index(X["Time"].to_numpy()[order]),
            capacity,
            self.cost_params,
            only_profitable=True,
        )
        selected = np.zeros(len(X), dtype=bool)
        selected[order] = selected_sorted
        return selected
