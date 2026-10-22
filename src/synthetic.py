"""Synthetic transactions with the real schema, for tests and CI smoke runs only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TARGET, V_COLUMNS

# Components that separate the classes in the real data; shifted for synthetic frauds.
_FRAUD_SHIFT = {"V14": -3.0, "V17": -2.5, "V12": -2.0, "V10": -1.5, "V4": 1.5}


def make_synthetic_transactions(
    n: int = 100_000, fraud_rate: float = 0.005, hours: int = 48, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Fewer transactions at night: accept arrival times with a day/night profile.
    times = rng.uniform(0, hours * 3600, 3 * n)
    hour = (times / 3600) % 24
    keep_prob = np.where((hour >= 1) & (hour < 7), 0.25, 1.0)
    times = np.sort(times[rng.random(times.size) < keep_prob][:n])
    n = len(times)

    y = rng.random(n) < fraud_rate
    V = rng.normal(size=(n, len(V_COLUMNS)))
    df = pd.DataFrame(V, columns=V_COLUMNS)
    for col, shift in _FRAUD_SHIFT.items():
        df.loc[y, col] += shift + rng.normal(scale=1.0, size=y.sum())
    amount = np.where(y, rng.lognormal(2.0, 1.8, n), rng.lognormal(3.0, 1.2, n))
    df.insert(0, "Time", np.round(times))
    df["Amount"] = np.round(amount, 2)
    df[TARGET] = y.astype(int)
    return df
