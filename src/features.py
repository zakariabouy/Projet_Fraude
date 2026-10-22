"""Feature engineering packaged as a scikit-learn transformer.

The same fitted object is used for training, the API and the dashboard: no train/serve
skew. The per-hour amount statistics are learned in `fit`, i.e. on the training data only.
Raw `Time` is dropped: future values are never seen during training, so it cannot
generalise. Only its hour-of-day projection is kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.data import RAW_FEATURES, V_COLUMNS

SECONDS_PER_HOUR = 3600
ENGINEERED = ["log_amount", "hour_sin", "hour_cos", "is_night", "amount_zscore_hour"]
FEATURE_NAMES = [*ENGINEERED, *V_COLUMNS]

FEATURE_LABELS = {
    "log_amount": "Transaction amount (log)",
    "hour_sin": "Hour of day (sin)",
    "hour_cos": "Hour of day (cos)",
    "is_night": "Night-time transaction",
    "amount_zscore_hour": "Amount unusual for this hour",
}

# Per-hour statistics need enough rows to be reliable; otherwise use the global ones.
_MIN_ROWS_PER_HOUR = 30


def hour_of_day(time) -> np.ndarray:
    """Hour of day in [0, 24). Assumes Time = 0 is midnight (traffic dips around t = 2h-7h)."""
    return (np.asarray(time, dtype=float) / SECONDS_PER_HOUR) % 24


def hour_index(time) -> np.ndarray:
    """Absolute hour since the first transaction: the unit of the analysts' capacity."""
    return (np.asarray(time, dtype=float) // SECONDS_PER_HOUR).astype(int)


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw transaction (Time, V1..V28, Amount) -> model features (see FEATURE_NAMES)."""

    def __init__(self, night_start: int = 0, night_end: int = 6):
        self.night_start = night_start
        self.night_end = night_end

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_columns(X)
        log_amount = np.log1p(X["Amount"].to_numpy(dtype=float))
        hour = np.floor(hour_of_day(X["Time"])).astype(int) % 24

        self.global_mean_ = float(log_amount.mean())
        self.global_std_ = float(log_amount.std()) or 1.0
        stats = (
            pd.DataFrame({"hour": hour, "log_amount": log_amount})
            .groupby("hour")["log_amount"]
            .agg(["mean", "std", "count"])
            .reindex(range(24))
        )
        reliable = stats["count"].fillna(0) >= _MIN_ROWS_PER_HOUR
        std_ok = reliable & (stats["std"] > 0)
        self.hour_mean_ = np.where(reliable, stats["mean"], self.global_mean_)
        self.hour_std_ = np.where(std_ok, stats["std"], self.global_std_)
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "hour_mean_")
        X = self._check_columns(X)
        hour = hour_of_day(X["Time"])
        hour_int = np.floor(hour).astype(int) % 24
        log_amount = np.log1p(X["Amount"].to_numpy(dtype=float))
        angle = 2 * np.pi * hour / 24

        engineered = pd.DataFrame(
            {
                "log_amount": log_amount,
                "hour_sin": np.sin(angle),
                "hour_cos": np.cos(angle),
                "is_night": ((hour_int >= self.night_start) & (hour_int < self.night_end)).astype(
                    float
                ),
                "amount_zscore_hour": (log_amount - self.hour_mean_[hour_int])
                / self.hour_std_[hour_int],
            },
            index=X.index,
        )
        return pd.concat([engineered, X[V_COLUMNS].astype(float)], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES, dtype=object)

    @staticmethod
    def _check_columns(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FraudFeatureEngineer expects a pandas DataFrame")
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return X
