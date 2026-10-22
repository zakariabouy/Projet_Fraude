"""Business cost model and alerting policies.

For a transaction of amount a:

    fraud, not alerted  (FN) -> a                      the money is lost
    fraud, alerted      (TP) -> c_inv + (1 - r) * a    investigation, most of the money saved
    legit, alerted      (FP) -> c_inv + c_fp           investigation + customer friction
    legit, not alerted  (TN) -> 0

    C = sum_FN a + c_inv * #alerts + c_fp * #FP + (1 - r) * sum_TP a

The fraud team can investigate at most K alerts per hour. Every policy is evaluated at
that constant capacity: when a policy raises more than K alerts in an hour, the extra
alerts are not investigated (no cost, but the fraud goes through).

All arrays are expected in chronological order (the temporal split guarantees it), so
"first come, first served" is the array order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostParams:
    c_inv: float = 5.0
    c_fp: float = 10.0
    recovery_rate: float = 0.8
    capacity_per_hour: int | None = 10

    def __post_init__(self):
        if self.c_inv < 0 or self.c_fp < 0:
            raise ValueError("Costs must be non-negative")
        if not 0 <= self.recovery_rate <= 1:
            raise ValueError("recovery_rate must be in [0, 1]")
        if self.capacity_per_hour is not None and self.capacity_per_hour < 0:
            raise ValueError("capacity_per_hour must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: dict) -> CostParams:
        return cls(
            c_inv=float(cfg["c_inv"]),
            c_fp=float(cfg["c_fp"]),
            recovery_rate=float(cfg["recovery_rate"]),
            capacity_per_hour=cfg["capacity_per_hour"],
        )


# --------------------------------------------------------------------------- costs


def transaction_costs(y_true, alert, amount, params: CostParams) -> np.ndarray:
    """Cost of each transaction given the decision (see module docstring)."""
    y = np.asarray(y_true).astype(bool)
    alert = np.asarray(alert).astype(bool)
    amount = np.asarray(amount, dtype=float)
    cost = np.zeros(len(y), dtype=float)
    cost[y & ~alert] = amount[y & ~alert]
    cost[y & alert] = params.c_inv + (1 - params.recovery_rate) * amount[y & alert]
    cost[~y & alert] = params.c_inv + params.c_fp
    return cost


def total_cost(y_true, alert, amount, params: CostParams) -> float:
    return float(transaction_costs(y_true, alert, amount, params).sum())


def cost_summary(y_true, alert, amount, params: CostParams) -> dict:
    """Cost and operational KPIs for one set of decisions."""
    y = np.asarray(y_true).astype(bool)
    alert = np.asarray(alert).astype(bool)
    amount = np.asarray(amount, dtype=float)
    tp, fp = int((y & alert).sum()), int((~y & alert).sum())
    fn, tn = int((y & ~alert).sum()), int((~y & ~alert).sum())
    fraud_amount = float(amount[y].sum())
    caught_amount = float(amount[y & alert].sum())
    return {
        "cost": total_cost(y, alert, amount, params),
        "n_alerts": int(alert.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "amount_recall": caught_amount / fraud_amount if fraud_amount else 0.0,
        "fraud_amount_lost": float(amount[y & ~alert].sum())
        + (1 - params.recovery_rate) * caught_amount,
        "fp_per_1000": 1000 * fp / len(y) if len(y) else 0.0,
    }


def expected_loss(proba, amount) -> np.ndarray:
    """p * a: the amount we expect to lose if the transaction is not investigated."""
    return np.asarray(proba, dtype=float) * np.asarray(amount, dtype=float)


def expected_net_benefit(proba, amount, params: CostParams) -> np.ndarray:
    """Expected saving of investigating a transaction minus its expected cost."""
    p = np.asarray(proba, dtype=float)
    a = np.asarray(amount, dtype=float)
    return p * params.recovery_rate * a - params.c_inv - (1 - p) * params.c_fp


# ------------------------------------------------------------------------ capacity


def enforce_capacity(alert, hour, capacity: int | None, priority=None) -> np.ndarray:
    """Keep at most `capacity` alerts per hour.

    Without `priority`, alerts are served first come, first served (array order).
    With `priority`, the highest-priority alerts of each hour are kept.
    """
    alert = np.asarray(alert).astype(bool)
    if capacity is None:
        return alert.copy()
    hour = np.asarray(hour)
    if priority is None and np.all(np.diff(hour) >= 0):
        return _first_come_first_served(alert, hour, capacity)
    idx = np.flatnonzero(alert)
    if idx.size == 0:
        return alert.copy()
    hour = hour[idx]
    key = idx if priority is None else -np.asarray(priority, dtype=float)[idx]
    order = np.lexsort((idx, key, hour))  # by hour, then key, then arrival
    idx, hour = idx[order], hour[order]
    new_hour = np.r_[True, hour[1:] != hour[:-1]]
    first_of_hour = np.maximum.accumulate(np.where(new_hour, np.arange(idx.size), 0))
    rank_in_hour = np.arange(idx.size) - first_of_hour
    kept = np.zeros_like(alert)
    kept[idx[rank_in_hour < capacity]] = True
    return kept


def _first_come_first_served(alert: np.ndarray, hour: np.ndarray, capacity: int) -> np.ndarray:
    """O(n) version of `enforce_capacity` when rows are already sorted by hour."""
    if alert.size == 0:
        return alert.copy()
    counts = alert.astype(np.int64)
    cumulative = np.cumsum(counts)
    starts = np.r_[0, np.flatnonzero(np.diff(hour)) + 1]
    sizes = np.diff(np.r_[starts, len(hour)])
    before_hour = np.repeat((cumulative - counts)[starts], sizes)
    return alert & (cumulative - before_hour <= capacity)


def max_alerts_per_hour(alert, hour) -> int:
    alert = np.asarray(alert).astype(bool)
    if not alert.any():
        return 0
    return int(np.bincount(pd.factorize(np.asarray(hour)[alert])[0]).max())


# ------------------------------------------------------------------------ policies


def threshold_policy(proba, threshold: float, hour, capacity: int | None) -> np.ndarray:
    """Alert when p >= threshold, analysts serve the alerts in arrival order."""
    return enforce_capacity(np.asarray(proba) >= threshold, hour, capacity)


def topk_policy(
    proba,
    amount,
    hour,
    capacity: int | None,
    params: CostParams | None = None,
    only_profitable: bool = False,
) -> np.ndarray:
    """Each hour, investigate the K transactions with the largest expected loss p * a.

    With `only_profitable`, a transaction is only a candidate if investigating it has a
    positive expected net benefit, so capacity is not spent on near-certain legit traffic.
    """
    candidates = np.ones(len(np.asarray(proba)), dtype=bool)
    if only_profitable:
        if params is None:
            raise ValueError("params are required with only_profitable=True")
        candidates = expected_net_benefit(proba, amount, params) > 0
    return enforce_capacity(candidates, hour, capacity, priority=expected_loss(proba, amount))


# ----------------------------------------------------------------- threshold search


def threshold_grid(size: int = 999) -> np.ndarray:
    return np.linspace(0, 1, size + 2)[1:-1]


def threshold_curve(y_true, proba, amount, hour, params: CostParams, grid=None) -> pd.DataFrame:
    """Cost and KPIs of the threshold policy for every threshold of the grid."""
    grid = threshold_grid() if grid is None else np.asarray(grid)
    y = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for t in grid:
        raised = proba >= t
        alert = enforce_capacity(raised, hour, params.capacity_per_hour)
        s = cost_summary(y, alert, amount, params)
        tp_raw = int((y & raised).sum())
        n_raised = int(raised.sum())
        rows.append(
            {
                "threshold": float(t),
                "cost": s["cost"],
                "n_raised": n_raised,
                "n_alerts": s["n_alerts"],
                "tp": s["tp"],
                "fp": s["fp"],
                "recall": s["recall"],
                "precision": s["precision"],
                "amount_recall": s["amount_recall"],
                "f1_raw": 2 * tp_raw / (n_raised + int(y.sum())) if n_raised + y.sum() else 0.0,
                "max_per_hour": max_alerts_per_hour(raised, hour),
            }
        )
    curve = pd.DataFrame(rows)
    capacity = params.capacity_per_hour
    curve["feasible"] = True if capacity is None else curve["max_per_hour"] <= capacity
    return curve


def _plateau_midpoint(grid: np.ndarray, best: np.ndarray) -> float:
    """Middle of the first run of consecutive best thresholds (max-margin choice)."""
    i = int(np.flatnonzero(best)[0])
    j = i
    while j + 1 < len(best) and best[j + 1]:
        j += 1
    return float((grid[i] + grid[j]) / 2)


def optimise_threshold(curve: pd.DataFrame) -> float:
    """t* = argmin cost among thresholds that never exceed the hourly capacity.

    If no threshold is feasible, fall back to the cheapest one (extra alerts dropped).
    """
    candidates = curve[curve["feasible"]] if curve["feasible"].any() else curve
    best = (curve["cost"] <= candidates["cost"].min() + 1e-9) & curve.index.isin(candidates.index)
    return _plateau_midpoint(curve["threshold"].to_numpy(), best.to_numpy())


def capacity_is_binding(curve: pd.DataFrame) -> bool:
    """True when a threshold that overloads some hour would have been cheaper."""
    if not curve["feasible"].any():
        return True
    return bool(curve["cost"].min() < curve.loc[curve["feasible"], "cost"].min() - 1e-9)


def f1_optimal_threshold(curve: pd.DataFrame) -> float:
    best = curve["f1_raw"] >= curve["f1_raw"].max() - 1e-12
    return _plateau_midpoint(curve["threshold"].to_numpy(), best.to_numpy())
