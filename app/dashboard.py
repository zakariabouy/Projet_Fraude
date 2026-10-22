"""Decision dashboard for non-technical stakeholders.

    streamlit run app/dashboard.py        (after `make train`)

1. Decision: change the cost hypotheses and the analysts' capacity; the cost-optimal
   threshold is re-chosen on validation and applied to the test period, like in training.
2. Alerts: the alerts of the test period with their SHAP reason codes.
3. Model: headline results and figures of the last training run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cost import (  # noqa: E402
    CostParams,
    cost_summary,
    optimise_threshold,
    threshold_curve,
    threshold_grid,
    threshold_policy,
    topk_policy,
)
from src.data import RAW_FEATURES  # noqa: E402
from src.predict import FraudScorer  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Fraud decision dashboard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_scorer() -> FraudScorer:
    return FraudScorer.from_path(ROOT / "models" / "fraud_model.joblib")


@st.cache_data
def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(PROCESSED / "predictions_valid.parquet"),
        pd.read_parquet(PROCESSED / "predictions_test.parquet"),
    )


@st.cache_data
def load_results() -> dict:
    return json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))


@st.cache_data
def choose_threshold(c_inv: float, c_fp: float, r: float, k: int) -> tuple[float, pd.DataFrame]:
    valid, _ = load_predictions()
    params = CostParams(c_inv, c_fp, r, k)
    curve = threshold_curve(
        valid["Class"], valid["p_oof"], valid["Amount"], valid["hour"], params, threshold_grid(299)
    )
    return optimise_threshold(curve), curve


missing = [
    p
    for p in (
        ROOT / "models" / "fraud_model.joblib",
        PROCESSED / "predictions_test.parquet",
        REPORTS / "results.json",
    )
    if not p.exists()
]
if missing:
    st.error(
        "Run `make train` first. Missing: " + ", ".join(str(p.relative_to(ROOT)) for p in missing)
    )
    st.stop()

scorer = load_scorer()
valid, test = load_predictions()
results = load_results()
base = scorer.cost_params

st.title("Fraud detection: from score to decision")
fraud_eur = test.loc[test["Class"] == 1, "Amount"].sum()
st.markdown(
    f"A fraud team can only investigate **K alerts per hour**. Which transactions should it "
    f"look at? Three policies are replayed on a held-out period of **{len(test):,} real card "
    f"transactions** ({int(test['Class'].sum())} frauds, {fraud_eur:,.0f} EUR stolen), never "
    f"seen in training. Each one is scored by its total cost: missed fraud, investigation "
    f"time and customers blocked by mistake. **Move the sliders** to change the business "
    f"hypotheses; open the **Alerts** tab to see why the model flags a transaction."
)
repo_url = os.environ.get("REPO_URL")
st.caption(
    f"Model {scorer.version} · XGBoost + {scorer.calibration_method} calibration · SHAP reason "
    f"codes" + (f" · [code and method on GitHub]({repo_url})" if repo_url else "")
)

with st.sidebar:
    st.header("Business hypotheses")
    c_inv = st.slider("Investigation cost c_inv (EUR)", 0.0, 50.0, float(base.c_inv), 1.0)
    c_fp = st.slider("Customer friction c_fp (EUR)", 0.0, 100.0, float(base.c_fp), 5.0)
    r = st.slider("Recovery rate r", 0.0, 1.0, float(base.recovery_rate), 0.05)
    k = st.slider("Capacity K (alerts / hour)", 1, 60, int(base.capacity_per_hour or 10), 1)
    st.caption("The threshold is re-optimised on the validation period, then applied to test.")

params = CostParams(c_inv, c_fp, r, k)
t_star, curve = choose_threshold(c_inv, c_fp, r, k)
y, p, amount, hour = test["Class"], test["p_cal"], test["Amount"], test["hour"]
policies = {
    "Model's default threshold (0.5)": threshold_policy(test["p_raw"], 0.5, hour, k),
    f"Threshold tuned on validation (t* = {t_star:.3f})": threshold_policy(p, t_star, hour, k),
    "Top-K by expected loss (profitable only)": topk_policy(
        p, amount, hour, k, params, only_profitable=True
    ),
}
summary = pd.DataFrame(
    {name: cost_summary(y, alert, amount, params) for name, alert in policies.items()}
).T

decision_tab, alerts_tab, model_tab = st.tabs(["Decision", "Alerts", "Model"])

with decision_tab:
    reference = summary.iloc[0]
    cols = st.columns(len(summary))
    for col, (name, row) in zip(cols, summary.iterrows(), strict=True):
        delta = None if name == summary.index[0] else f"{row['cost'] - reference['cost']:,.0f} EUR"
        col.metric(name, f"{row['cost']:,.0f} EUR", delta, delta_color="inverse")
        col.caption(
            f"{int(row['n_alerts'])} alerts · {int(row['tp'])} frauds caught · "
            f"{int(row['fp'])} false positives · {100 * row['amount_recall']:.0f}% of fraud EUR caught"
        )
    st.caption("Total cost on the held-out period; the arrows compare with the default threshold.")
    st.subheader("Cost of the threshold policy on validation")
    st.line_chart(curve.set_index("threshold")[["cost"]], x_label="threshold", y_label="EUR")
    st.dataframe(
        summary[
            ["cost", "n_alerts", "tp", "fp", "fn", "precision", "recall", "amount_recall"]
        ].style.format(
            {
                "cost": "{:,.0f}",
                "precision": "{:.1%}",
                "recall": "{:.1%}",
                "amount_recall": "{:.1%}",
            }
        ),
        width="stretch",
    )

with alerts_tab:
    policy = st.radio("Policy", list(policies), index=1, horizontal=True)
    alerts = test[policies[policy]].assign(expected_loss=lambda d: d["p_cal"] * d["Amount"])
    alerts = alerts.sort_values("expected_loss", ascending=False)
    st.write(f"{len(alerts)} alerts, {int(alerts['Class'].sum())} true frauds.")
    st.dataframe(
        alerts[["Time", "Amount", "p_cal", "expected_loss", "Class"]].rename(
            columns={"p_cal": "probability", "Class": "is_fraud"}
        ),
        width="stretch",
        height=260,
    )
    if len(alerts):
        pick = st.selectbox(
            "Explain an alert",
            alerts.index,
            format_func=lambda i: (
                f"t = {test.loc[i, 'Time'] / 3600:.2f} h · {test.loc[i, 'Amount']:.2f} EUR · p = {test.loc[i, 'p_cal']:.3f}"
            ),
        )
        codes = scorer.explain(test.loc[[pick], RAW_FEATURES])[0]
        st.write("**Reason codes** (SHAP, log-odds pushing towards fraud):")
        bars = (
            alt.Chart(pd.DataFrame(codes))
            .mark_bar(color="#1f4e79")
            .encode(
                x=alt.X("contribution:Q", title="SHAP contribution (log-odds)"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", "label", "value", "contribution"],
            )
            .properties(height=140)
        )
        st.altair_chart(bars, width="stretch")
        st.table(pd.DataFrame(codes)[["feature", "label", "value", "contribution"]])

with model_tab:
    models = pd.DataFrame(results["models"]).set_index("label")
    st.subheader("Ranking quality on test (PR-AUC with 95% bootstrap interval)")
    st.dataframe(
        models[["pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "brier"]].style.format("{:.4f}"),
        width="stretch",
    )
    figures = REPORTS / "figures"
    for name, caption in [
        ("policies_test.png", "Policies at constant capacity (training run hypotheses)"),
        ("calibration_test.png", "Calibration"),
        ("shap_beeswarm.png", "SHAP global view"),
        ("sensitivity.png", "Sensitivity to the cost hypotheses"),
    ]:
        if (figures / name).exists():
            st.image(str(figures / name), caption=caption)
    st.caption(f"Score PSI train -> test: {results['monitoring']['score_psi']:.3f}")
