import numpy as np
import pandas as pd
import pytest

from src.data import RAW_FEATURES, xy
from src.features import FEATURE_NAMES, FraudFeatureEngineer, hour_index, hour_of_day


def _row(time: float, amount: float = 10.0) -> pd.DataFrame:
    data = {c: [0.0] for c in RAW_FEATURES}
    data["Time"], data["Amount"] = [time], [amount]
    return pd.DataFrame(data)


def test_output_shape_names_and_no_nan(synthetic_df):
    X, _ = xy(synthetic_df)
    out = FraudFeatureEngineer().fit(X).transform(X)
    assert list(out.columns) == FEATURE_NAMES
    assert out.shape == (len(X), len(FEATURE_NAMES))
    assert not out.isna().any().any()
    assert "Time" not in out.columns


def test_statistics_are_learned_on_fit_data_only(synthetic_df):
    X, _ = xy(synthetic_df)
    fe = FraudFeatureEngineer().fit(X.iloc[:5000])
    before = fe.hour_mean_.copy()
    fe.transform(X.iloc[5000:])
    np.testing.assert_array_equal(before, fe.hour_mean_)


def test_hours_missing_from_training_fall_back_to_global_stats(synthetic_df):
    X, _ = xy(synthetic_df)
    daytime = X[(hour_of_day(X["Time"]) >= 9) & (hour_of_day(X["Time"]) < 12)]
    out = FraudFeatureEngineer().fit(daytime).transform(X)
    assert not out.isna().any().any()


def test_cyclic_hour_makes_23h_and_midnight_close():
    fe = FraudFeatureEngineer().fit(pd.concat([_row(t) for t in range(0, 86400, 600)]))
    late, midnight, noon = (fe.transform(_row(t)) for t in (23.9 * 3600, 0.0, 12 * 3600))
    cols = ["hour_sin", "hour_cos"]
    near = np.linalg.norm(late[cols].to_numpy() - midnight[cols].to_numpy())
    far = np.linalg.norm(noon[cols].to_numpy() - midnight[cols].to_numpy())
    assert near < 0.1 < far


def test_is_night_and_log_amount():
    fe = FraudFeatureEngineer(night_start=0, night_end=6).fit(_row(0.0))
    assert fe.transform(_row(3 * 3600))["is_night"].iloc[0] == 1
    assert fe.transform(_row(12 * 3600))["is_night"].iloc[0] == 0
    assert fe.transform(_row(0.0, amount=np.e - 1))["log_amount"].iloc[0] == pytest.approx(1.0)


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="Missing columns"):
        FraudFeatureEngineer().fit(_row(0.0).drop(columns="V14"))


def test_hour_index_counts_absolute_hours():
    np.testing.assert_array_equal(hour_index([0, 3599, 3600, 90000]), [0, 0, 1, 25])
