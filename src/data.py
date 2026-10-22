"""Load, validate and split the credit card fraud dataset (Kaggle mlg-ulb/creditcardfraud).

The split is chronological: in production the model scores the future after being trained
on the past. A random split would leak future fraud patterns into training and overstate
performance.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", *V_COLUMNS, "Amount"]
TARGET = "Class"


def download(url: str, dest: str | Path) -> Path:
    """Download the CSV once; later calls reuse the local file."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def validate(
    df: pd.DataFrame, expected_rows: int | None = None, expected_frauds: int | None = None
) -> None:
    missing = [c for c in [*RAW_FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[RAW_FEATURES].isna().any().any():
        raise ValueError("The dataset contains missing values")
    if not set(np.unique(df[TARGET])) <= {0, 1}:
        raise ValueError(f"{TARGET} must be binary (0/1)")
    if (df["Amount"] < 0).any():
        raise ValueError("Negative amounts found")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
    if expected_frauds is not None and int(df[TARGET].sum()) != expected_frauds:
        raise ValueError(f"Expected {expected_frauds} frauds, got {int(df[TARGET].sum())}")


def load_raw(
    path: str | Path, expected_rows: int | None = None, expected_frauds: int | None = None
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    validate(df, expected_rows, expected_frauds)
    return df


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    fractions: tuple[float, float, float]


def _cut_index(time: np.ndarray, i: int) -> int:
    """Move a cut point forward so that rows sharing the same timestamp stay together."""
    while 0 < i < len(time) and time[i] == time[i - 1]:
        i += 1
    return i


def temporal_split(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    min_frauds: int | None = None,
    fallback_fractions: tuple[float, float, float] | None = None,
) -> TemporalSplit:
    """Sort by `Time` and cut into train / validation / test without shuffling."""
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Split fractions must sum to 1")
    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    time = df["Time"].to_numpy()
    n = len(df)
    i1 = _cut_index(time, int(n * fractions[0]))
    i2 = _cut_index(time, int(n * (fractions[0] + fractions[1])))
    split = TemporalSplit(df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:], tuple(fractions))

    if min_frauds is not None and fallback_fractions is not None:
        frauds = min(split.valid[TARGET].sum(), split.test[TARGET].sum())
        if frauds < min_frauds:
            log.warning(
                "Only %d frauds in validation/test with %s, falling back to %s",
                frauds,
                fractions,
                fallback_fractions,
            )
            return temporal_split(df, tuple(fallback_fractions))
    return split


def split_in_time(df: pd.DataFrame, share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut a time-sorted frame in two chronological parts (`share` of the rows first)."""
    i = _cut_index(df["Time"].to_numpy(), int(len(df) * share))
    return df.iloc[:i], df.iloc[i:]


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[RAW_FEATURES], df[TARGET]


def describe_part(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "frauds": int(df[TARGET].sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "time_start_h": float(df["Time"].min() / 3600),
        "time_end_h": float(df["Time"].max() / 3600),
        "fraud_amount": float(df.loc[df[TARGET] == 1, "Amount"].sum()),
    }
