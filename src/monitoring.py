"""Drift monitoring with the Population Stability Index.

    PSI = sum_b (p_b - q_b) * ln(p_b / q_b)

q = reference distribution (training period), p = current distribution, b = bins built on
the reference quantiles. Usual reading: < 0.1 stable, 0.1-0.25 watch, > 0.25 major drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STABLE, WATCH = 0.10, 0.25


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    uniques = np.unique(reference)
    if len(uniques) <= bins:  # discrete feature: one bin per value
        edges = np.r_[-np.inf, (uniques[:-1] + uniques[1:]) / 2, np.inf]
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    q = np.histogram(reference, edges)[0] / len(reference)
    p = np.histogram(current, edges)[0] / len(current)
    q, p = np.clip(q, eps, None), np.clip(p, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))


def psi_status(value: float) -> str:
    if value < STABLE:
        return "stable"
    return "watch" if value < WATCH else "drift"


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = [
        {"feature": col, "psi": psi(reference[col], current[col], bins)}
        for col in reference.columns
    ]
    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    report["status"] = report["psi"].map(psi_status)
    return report
