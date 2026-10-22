import numpy as np
import pytest

from src.data import TARGET, split_in_time, temporal_split, validate


def test_split_is_chronological_and_complete(synthetic_df):
    split = temporal_split(synthetic_df.sample(frac=1, random_state=0), (0.6, 0.2, 0.2))
    assert split.train["Time"].max() <= split.valid["Time"].min()
    assert split.valid["Time"].max() <= split.test["Time"].min()
    assert len(split.train) + len(split.valid) + len(split.test) == len(synthetic_df)
    assert split.train["Time"].is_monotonic_increasing


def test_identical_timestamps_stay_in_the_same_part(synthetic_df):
    df = synthetic_df.copy()
    df["Time"] = (df["Time"] // 600) * 600  # many ties
    split = temporal_split(df, (0.6, 0.2, 0.2))
    assert split.train["Time"].max() < split.valid["Time"].min()


def test_fallback_when_too_few_frauds(synthetic_df):
    split = temporal_split(
        synthetic_df, (0.8, 0.1, 0.1), min_frauds=10**6, fallback_fractions=(0.5, 0.25, 0.25)
    )
    assert split.fractions == (0.5, 0.25, 0.25)


def test_split_in_time(synthetic_df):
    first, second = split_in_time(synthetic_df.sort_values("Time"), 0.5)
    assert first["Time"].max() <= second["Time"].min()
    assert abs(len(first) - len(second)) < 0.01 * len(synthetic_df)


def test_validate_rejects_bad_data(synthetic_df):
    with pytest.raises(ValueError, match="Missing columns"):
        validate(synthetic_df.drop(columns="V3"))
    broken = synthetic_df.copy()
    broken.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate(broken)
    with pytest.raises(ValueError, match="Expected"):
        validate(synthetic_df, expected_frauds=int(synthetic_df[TARGET].sum()) + 1)
