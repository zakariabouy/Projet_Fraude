"""Exploratory figures and the EDA numbers quoted in the report."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data import TARGET, V_COLUMNS
from src.features import hour_of_day


def eda_summary(df: pd.DataFrame, night_hours=(0, 6)) -> dict:
    fraud = df[TARGET] == 1
    hour = np.floor(hour_of_day(df["Time"])).astype(int)
    night = (hour >= night_hours[0]) & (hour < night_hours[1])
    separation = sorted(
        (
            (col, float(ks_2samp(df.loc[fraud, col], df.loc[~fraud, col]).statistic))
            for col in V_COLUMNS
        ),
        key=lambda item: -item[1],
    )
    return {
        "rows": len(df),
        "frauds": int(fraud.sum()),
        "fraud_rate": float(fraud.mean()),
        "accuracy_never_fraud": float(1 - fraud.mean()),
        "duration_hours": float(df["Time"].max() / 3600),
        "duplicate_rows": int(df.duplicated().sum()),
        "amount_median_fraud": float(df.loc[fraud, "Amount"].median()),
        "amount_median_legit": float(df.loc[~fraud, "Amount"].median()),
        "amount_mean_fraud": float(df.loc[fraud, "Amount"].mean()),
        "amount_mean_legit": float(df.loc[~fraud, "Amount"].mean()),
        "fraud_rate_night": float(fraud[night].mean()),
        "fraud_rate_day": float(fraud[~night].mean()),
        "share_transactions_night": float(night.mean()),
        "top_separating_components_ks": [{"feature": col, "ks": ks} for col, ks in separation[:6]],
    }


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    hour = np.floor(hour_of_day(df["Time"])).astype(int)
    return (
        df.assign(hour=hour)
        .groupby("hour")
        .agg(transactions=(TARGET, "size"), frauds=(TARGET, "sum"))
        .assign(fraud_rate=lambda d: d["frauds"] / d["transactions"])
    )
