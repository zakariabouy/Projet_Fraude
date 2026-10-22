"""All report figures (matplotlib, saved as PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src.data import TARGET  # noqa: E402

BLUE, RED, GREY, GREEN, ORANGE = "#1f4e79", "#c0392b", "#7f8c8d", "#2e8b57", "#d68910"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eda(df: pd.DataFrame, hourly: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-1, np.log10(df["Amount"].max() + 1), 50)
    for label, color, name in [(0, BLUE, "Legitimate"), (1, RED, "Fraud")]:
        amounts = df.loc[df[TARGET] == label, "Amount"] + 0.1
        ax.hist(amounts, bins=bins, density=True, alpha=0.55, color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Amount (EUR, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Amount distribution by class")
    ax.legend()
    _save(fig, figures / "eda_amount_by_class.png")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(hourly.index, hourly["transactions"], color=GREY, alpha=0.5, label="Transactions")
    ax1.set_xlabel("Hour of day (assuming Time = 0 is midnight)")
    ax1.set_ylabel("Transactions")
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, 100 * hourly["fraud_rate"], color=RED, marker="o", label="Fraud rate")
    ax2.set_ylabel("Fraud rate (%)")
    ax1.set_title("Volume and fraud rate by hour of day")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.92))
    _save(fig, figures / "eda_fraud_rate_by_hour.png")


def plot_pr_curves(y_true, scores: dict[str, np.ndarray], labels: dict[str, str], path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax.step(recall, precision, where="post", label=labels.get(name, name))
    ax.axhline(np.mean(y_true), color=GREY, ls="--", lw=1, label="Random (fraud rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the test period")
    ax.legend(fontsize=8)
    _save(fig, path)


def reliability_table(y_true, proba, edges=None) -> pd.DataFrame:
    """Observed fraud frequency vs mean predicted probability, on log-spaced bins."""
    edges = np.array([0, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 0.8, 1.0001]) if edges is None else edges
    bins = np.digitize(proba, edges[1:-1])
    frame = pd.DataFrame({"bin": bins, "p": proba, "y": np.asarray(y_true)})
    table = frame.groupby("bin").agg(
        mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return table[table["n"] > 0]


def plot_calibration(y_true, probas: dict[str, np.ndarray], labels: dict[str, str], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    floor = 1e-5  # where bins without any observed fraud are drawn (log scale has no 0)
    ax.plot([floor, 1], [floor, 1], color=GREY, ls="--", lw=1, label="Perfect calibration")
    for name, p in probas.items():
        table = reliability_table(y_true, p)
        observed = table["observed"].where(table["observed"] > 0)
        (line,) = ax.plot(
            table["mean_predicted"], observed, marker="o", label=labels.get(name, name)
        )
        empty = table[table["observed"] == 0]
        ax.scatter(
            empty["mean_predicted"], np.full(len(empty), floor), marker="x", color=line.get_color()
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted probability (log-spaced bins)")
    ax.set_ylabel("Observed fraud frequency")
    ax.set_title("Calibration on the test period (x at the bottom: bin with no fraud)", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_cost_curve(curve: pd.DataFrame, t_star: float, t_f1: float, capacity, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["threshold"], curve["cost"], color=BLUE, label="Total cost (validation)")
    infeasible = curve.loc[~curve["feasible"], "threshold"]
    if len(infeasible):
        ax.axvspan(
            curve["threshold"].min(),
            infeasible.max(),
            color=RED,
            alpha=0.08,
            label=f"> {capacity} alerts in some hour",
        )
    ax.axvline(t_star, color=GREEN, ls="-", label=f"Cost-optimal t* = {t_star:.3f}")
    ax.axvline(t_f1, color=ORANGE, ls="--", label=f"F1-optimal = {t_f1:.3f}")
    ax.axvline(0.5, color=GREY, ls=":", label="Default 0.5")
    ax.set_xscale("logit")
    ax.set_xlabel("Threshold on calibrated probability (logit scale)")
    ax.set_ylabel("Cost (EUR)")
    ax.set_title("Cost of the threshold policy at constant analyst capacity")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_policies(policies: pd.DataFrame, path) -> None:
    frame = policies.sort_values("cost", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(frame) + 1.2))
    colors = [GREEN if i == 0 else BLUE for i in range(len(frame))]
    ax.barh(frame["label"], frame["cost"], color=colors)
    for y, (cost, recall) in enumerate(zip(frame["cost"], frame["amount_recall"], strict=True)):
        ax.text(
            cost,
            y,
            f"  {cost:,.0f} EUR | {100 * recall:.0f}% of fraud EUR caught",
            va="center",
            fontsize=8,
        )
    ax.invert_yaxis()
    ax.set_xlabel("Total cost on the test period (EUR)")
    ax.set_title("Decision policies at constant capacity (test)")
    ax.set_xlim(0, frame["cost"].max() * 1.6)
    _save(fig, path)


def plot_sensitivity(costs: pd.DataFrame, capacity: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, column, title, fmt in [
        (axes[0], "t_star", "Cost-optimal threshold t*", "{:.3f}"),
        (axes[1], "test_cost", "Test cost of the t* policy (EUR)", "{:,.0f}"),
    ]:
        pivot = costs.pivot(index="c_inv", columns="c_fp", values=column)
        im = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
        ax.set_xlabel("c_fp: customer friction (EUR)")
        ax.set_ylabel("c_inv: investigation cost (EUR)")
        ax.set_title(title)
        for (i, j), value in np.ndenumerate(pivot.to_numpy()):
            ax.text(
                j,
                i,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > np.nanmean(pivot.to_numpy()) else "black",
            )
        fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2]
    ax.plot(
        capacity["capacity"],
        capacity["test_cost"],
        marker="o",
        color=BLUE,
        label="Cost, threshold t*",
    )
    ax.plot(
        capacity["capacity"], capacity["topk_cost"], marker="s", color=GREEN, label="Cost, top-K"
    )
    ax.set_xlabel("K: alerts investigated per hour")
    ax.set_ylabel("Test cost (EUR)")
    ax2 = ax.twinx()
    ax2.plot(
        capacity["capacity"],
        100 * capacity["test_amount_recall"],
        ls="--",
        color=RED,
        label="Fraud EUR caught, t*",
    )
    ax2.set_ylabel("Fraud amount caught (%)")
    ax.set_title("Effect of the analysts' capacity")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper center", fontsize=8, framealpha=0.9)
    _save(fig, path)


def plot_psi(report: pd.DataFrame, path, top: int = 15) -> None:
    frame = report.head(top).iloc[::-1]
    colors = frame["status"].map({"stable": GREEN, "watch": ORANGE, "drift": RED})
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(frame) + 1.2))
    ax.barh(frame["feature"], frame["psi"], color=colors)
    ax.axvline(0.10, color=ORANGE, ls="--", lw=1)
    ax.axvline(0.25, color=RED, ls="--", lw=1)
    ax.set_xlabel("PSI (train -> test)")
    ax.set_title("Population Stability Index, highest features")
    _save(fig, path)


def plot_beeswarm(explanation, path, max_display: int = 15) -> None:
    import shap

    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP values on the test period (log-odds, uncalibrated model)", fontsize=10)
    _save(fig, path)


def plot_waterfall(explanation_row, title: str, path, max_display: int = 12) -> None:
    import shap

    shap.plots.waterfall(explanation_row, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    _save(fig, path)
