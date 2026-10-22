"""Request / response schemas. Invalid input is rejected with HTTP 422 before scoring."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# A fraud from the test period of the public dataset (Kaggle mlg-ulb/creditcardfraud).
EXAMPLE_TRANSACTION = {
    "Time": 150138.0,
    "V1": -2.150855, "V2": 2.187917, "V3": -3.430516, "V4": 0.119476, "V5": -0.17321,
    "V6": 0.2907, "V7": -2.808988, "V8": -2.679351, "V9": -0.556685, "V10": -4.485483,
    "V11": 1.903999, "V12": -2.644219, "V13": -0.982273, "V14": -4.691151, "V15": -0.69308,
    "V16": -2.553251, "V17": -3.483436, "V18": -0.064852, "V19": 1.490329, "V20": 0.532145,
    "V21": -0.073205, "V22": 0.561496, "V23": -0.075034, "V24": -0.437619, "V25": 0.353841,
    "V26": -0.521339, "V27": 0.144465, "V28": 0.026588,
    "Amount": 50.0,
}  # fmt: skip


class Transaction(BaseModel):
    """A raw card transaction, as in the dataset (V1..V28 are PCA components)."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={"examples": [EXAMPLE_TRANSACTION]},
    )

    Time: float = Field(ge=0, description="Seconds since the first transaction of the period")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0, description="Transaction amount (EUR)")


class ReasonCode(BaseModel):
    feature: str
    label: str
    value: float = Field(description="Feature value after feature engineering")
    contribution: float = Field(description="SHAP value (log-odds of the uncalibrated model)")
    direction: Literal["increases_risk", "decreases_risk"]


class Prediction(BaseModel):
    fraud_probability: float = Field(description="Calibrated probability of fraud")
    expected_loss: float = Field(description="fraud_probability x Amount (EUR)")
    alert: bool
    decision: Literal["ALERT", "PASS"]
    threshold: float = Field(description="Cost-optimal threshold t* chosen on validation")
    reason_codes: list[ReasonCode] = Field(description="Top 3 features pushing towards fraud")
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=10_000)
    capacity_per_hour: int | None = Field(
        default=None, ge=0, description="Analysts' capacity K (default: the model's)"
    )


class BatchItem(Prediction):
    prioritised: bool = Field(
        description="Selected for investigation by the top-K expected-loss queue"
    )


class BatchResponse(BaseModel):
    predictions: list[BatchItem]
    n_alerts: int = Field(description="Transactions above the threshold t*")
    n_prioritised: int = Field(description="Transactions in the top-K queue")
    capacity_per_hour: int | None
    model_version: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None


class ModelInfo(BaseModel):
    model_version: str
    calibration_method: str
    threshold: float
    cost_params: dict
    trained_at: str | None
    test_pr_auc: float | None
    feature_names: list[str]
