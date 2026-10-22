import numpy as np
import pandas as pd

from src.monitoring import psi, psi_report, psi_status


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=20_000), rng.normal(size=20_000)) < 0.01


def test_shifted_distribution_is_drift():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=20_000), rng.normal(loc=1.0, size=20_000)) > 0.25


def test_binary_feature_is_supported():
    reference = np.r_[np.zeros(800), np.ones(200)]
    current = np.r_[np.zeros(500), np.ones(500)]
    assert psi(reference, current) > 0.1


def test_status_thresholds():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.2) == "watch"
    assert psi_status(0.4) == "drift"


def test_report_is_sorted():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000)})
    cur = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(loc=2, size=5000)})
    report = psi_report(ref, cur)
    assert report["feature"].iloc[0] == "b"
    assert report["status"].iloc[0] == "drift"
