import numpy as np
import pytest

from src.data import xy
from src.explain import native_contributions, reason_codes


def test_native_treeshap_is_additive(fitted_pipeline):
    model, split = fitted_pipeline
    X, _ = xy(split.test.iloc[:100])
    contrib, bias, X_t = native_contributions(model, X)
    margin = model.named_steps["model"].predict(X_t, output_margin=True)
    np.testing.assert_allclose(contrib.sum(axis=1) + bias, margin, atol=1e-4)


def test_native_treeshap_matches_shap_library(fitted_pipeline):
    pytest.importorskip("shap")
    from src.explain import shap_explanation

    model, split = fitted_pipeline
    X, _ = xy(split.test.iloc[:50])
    contrib, _, _ = native_contributions(model, X)
    np.testing.assert_allclose(shap_explanation(model, X).values, contrib.to_numpy(), atol=1e-4)


def test_reason_codes_are_the_top_three_risk_drivers(fitted_pipeline):
    model, split = fitted_pipeline
    X, _ = xy(split.test.iloc[:1])
    contrib, _, X_t = native_contributions(model, X)
    codes = reason_codes(contrib.iloc[0], X_t.iloc[0], top=3)
    assert len(codes) == 3
    values = [c["contribution"] for c in codes]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(contrib.iloc[0].max())
