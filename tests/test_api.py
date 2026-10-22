import pytest

from src.api.schemas import EXAMPLE_TRANSACTION


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_model_info(client):
    body = client.get("/model").json()
    assert body["calibration_method"] == "sigmoid"
    assert body["cost_params"]["capacity_per_hour"] == 10
    assert len(body["feature_names"]) == 33


def test_predict_valid_transaction(client):
    response = client.post("/predict", json=EXAMPLE_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["expected_loss"] == pytest.approx(
        body["fraud_probability"] * EXAMPLE_TRANSACTION["Amount"], rel=1e-6
    )
    assert body["decision"] == ("ALERT" if body["alert"] else "PASS")
    assert body["alert"] == (body["fraud_probability"] >= body["threshold"])
    assert len(body["reason_codes"]) == 3
    assert {"feature", "label", "value", "contribution", "direction"} <= body["reason_codes"][
        0
    ].keys()
    assert "X-Process-Time-Ms" in response.headers


@pytest.mark.parametrize(
    "change",
    [
        {"Amount": -5.0},  # negative amount
        {"V14": "not a number"},  # wrong type
        {"unexpected": 1.0},  # unknown field
    ],
)
def test_predict_invalid_input_returns_422(client, change):
    assert client.post("/predict", json={**EXAMPLE_TRANSACTION, **change}).status_code == 422


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in EXAMPLE_TRANSACTION.items() if k != "V14"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_ranks_within_capacity(client, synthetic_df):
    rows = synthetic_df.drop(columns="Class").iloc[:40].copy()
    rows["Time"] = 100.0  # same hour: the queue can only take `capacity` of them
    rows["Amount"] = 1000.0
    payload = {"transactions": rows.to_dict("records"), "capacity_per_hour": 3}
    body = client.post("/predict/batch", json=payload).json()
    assert len(body["predictions"]) == 40
    assert body["n_prioritised"] <= 3
    assert body["capacity_per_hour"] == 3
    for item in body["predictions"]:
        if item["alert"] or item["prioritised"]:
            assert len(item["reason_codes"]) == 3


def test_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_predict_without_model_returns_503(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict", json=EXAMPLE_TRANSACTION).status_code == 503
