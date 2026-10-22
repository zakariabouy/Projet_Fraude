"""Fraud scoring API.

    uvicorn src.api.main:app --reload          # docs on http://localhost:8000/docs

The model bundle is loaded once at start-up (MODEL_PATH, default models/fraud_model.joblib).
POST /predict returns the calibrated probability, the decision at the cost-optimal threshold
and 3 SHAP reason codes. POST /predict/batch also ranks a batch into the top-K queue.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from src import __version__
from src.api.schemas import (
    BatchItem,
    BatchRequest,
    BatchResponse,
    Health,
    ModelInfo,
    Prediction,
    Transaction,
)
from src.predict import FraudScorer

log = logging.getLogger("api")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "fraud_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.scorer = None
    if path.exists():
        app.state.scorer = FraudScorer.from_path(path)
        log.info("Model %s loaded from %s", app.state.scorer.version, path)
    else:
        log.error("No model found at %s: run `make train` first", path)
    yield


app = FastAPI(
    title="Fraud detection API",
    version=__version__,
    description="Calibrated fraud probability, cost-optimal decision and SHAP reason codes.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{1000 * (time.perf_counter() - start):.2f}"
    return response


def get_scorer(request: Request) -> FraudScorer:
    scorer = request.app.state.scorer
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return scorer


def _frame(transactions: list[Transaction]) -> pd.DataFrame:
    return pd.DataFrame([t.model_dump() for t in transactions])


def _prediction(row: pd.Series, scorer: FraudScorer) -> dict:
    return {
        "fraud_probability": float(row["fraud_probability"]),
        "expected_loss": float(row["expected_loss"]),
        "alert": bool(row["alert"]),
        "decision": "ALERT" if row["alert"] else "PASS",
        "threshold": scorer.threshold,
        "reason_codes": row["reason_codes"],
        "model_version": scorer.version,
    }


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    scorer = request.app.state.scorer
    return Health(
        status="ok" if scorer else "degraded",
        model_loaded=scorer is not None,
        model_version=scorer.version if scorer else None,
    )


@app.get("/model", response_model=ModelInfo)
def model_info(scorer: FraudScorer = Depends(get_scorer)) -> ModelInfo:
    meta = scorer.bundle.get("metadata", {})
    return ModelInfo(
        model_version=scorer.version,
        calibration_method=scorer.calibration_method,
        threshold=scorer.threshold,
        cost_params=scorer.cost_params.to_dict(),
        trained_at=meta.get("trained_at"),
        test_pr_auc=meta.get("test_pr_auc"),
        feature_names=meta.get("feature_names", []),
    )


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, scorer: FraudScorer = Depends(get_scorer)) -> Prediction:
    row = scorer.score(_frame([transaction])).iloc[0]
    return Prediction(**_prediction(row, scorer))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, scorer: FraudScorer = Depends(get_scorer)) -> BatchResponse:
    X = _frame(batch.transactions)
    scored = scorer.score(X, with_reasons=False)
    prioritised = scorer.prioritise(X, batch.capacity_per_hour)
    # Reason codes only where an analyst will look: alerts and the top-K queue.
    flagged = (scored["alert"] | prioritised).to_numpy()
    reasons = [[] for _ in range(len(X))]
    if flagged.any():
        for i, codes in zip(flagged.nonzero()[0], scorer.explain(X[flagged]), strict=True):
            reasons[i] = codes
    scored["reason_codes"] = reasons
    items = [
        BatchItem(**_prediction(row, scorer), prioritised=bool(p))
        for (_, row), p in zip(scored.iterrows(), prioritised, strict=True)
    ]
    capacity = batch.capacity_per_hour
    return BatchResponse(
        predictions=items,
        n_alerts=int(scored["alert"].sum()),
        n_prioritised=int(prioritised.sum()),
        capacity_per_hour=scorer.cost_params.capacity_per_hour if capacity is None else capacity,
        model_version=scorer.version,
    )
