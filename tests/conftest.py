"""Shared fixtures. Tests never need the real dataset: a tiny model is trained on
synthetic transactions with the same schema."""

from __future__ import annotations

import pytest

from src.calibration import fit_calibrator
from src.cost import CostParams
from src.data import temporal_split, xy
from src.features import FEATURE_NAMES
from src.models import xgb_pipeline
from src.predict import make_bundle, save_bundle
from src.synthetic import make_synthetic_transactions


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_transactions(n=20_000, fraud_rate=0.01, seed=1)


@pytest.fixture(scope="session")
def fitted_pipeline(synthetic_df):
    split = temporal_split(synthetic_df, (0.6, 0.2, 0.2))
    X_train, y_train = xy(split.train)
    params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.2, "scale_pos_weight": 5.0}
    return xgb_pipeline(params, seed=0).fit(X_train, y_train), split


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory, fitted_pipeline):
    model, split = fitted_pipeline
    X_valid, y_valid = xy(split.valid)
    bundle = make_bundle(
        fit_calibrator(model, X_valid, y_valid, "sigmoid"),
        threshold=0.5,
        cost_params=CostParams(),
        calibration_method="sigmoid",
        version="test",
        metadata={"feature_names": FEATURE_NAMES, "test_pr_auc": 0.9},
    )
    return save_bundle(bundle, tmp_path_factory.mktemp("model") / "fraud_model.joblib")


@pytest.fixture()
def client(bundle_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.setenv("MODEL_PATH", str(bundle_path))
    with TestClient(app) as test_client:  # the context manager runs the lifespan (model load)
        yield test_client
