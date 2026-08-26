"""Tests for RollingStatisticsFeatures — 100 % branch coverage."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from gators.feature_generation import RollingStatisticsFeatures


@pytest.fixture
def simple_df():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


class TestRollingStatisticsFeatures:
    def test_fit_returns_self(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["mean"])
        assert t.fit(simple_df) is t

    def test_new_column_names_attribute(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x", "y"], window_size=2, func=["mean", "sum"])
        t.fit(simple_df)
        assert t.new_column_names_ == [
            "x__rolling_mean_2",
            "x__rolling_sum_2",
            "y__rolling_mean_2",
            "y__rolling_sum_2",
        ]

    def test_rolling_mean(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["mean"])
        result = t.fit(simple_df).transform(simple_df)
        assert "x__rolling_mean_3" in result.columns
        expected = [None, None, 2.0, 3.0, 4.0]
        assert result["x__rolling_mean_3"].to_list() == expected

    def test_rolling_std(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["std"])
        result = t.fit(simple_df).transform(simple_df)
        assert "x__rolling_std_3" in result.columns
        vals = result["x__rolling_std_3"].to_list()
        assert vals[0] is None and vals[1] is None
        assert vals[2] == pytest.approx(1.0)

    def test_rolling_min(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["min"])
        result = t.fit(simple_df).transform(simple_df)
        assert result["x__rolling_min_3"].to_list() == [None, None, 1.0, 2.0, 3.0]

    def test_rolling_max(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["max"])
        result = t.fit(simple_df).transform(simple_df)
        assert result["x__rolling_max_3"].to_list() == [None, None, 3.0, 4.0, 5.0]

    def test_rolling_sum(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["sum"])
        result = t.fit(simple_df).transform(simple_df)
        assert result["x__rolling_sum_3"].to_list() == [None, None, 6.0, 9.0, 12.0]

    def test_multiple_funcs(self, simple_df):
        t = RollingStatisticsFeatures(
            subset=["x"], window_size=2, func=["mean", "min", "max", "sum", "std"]
        )
        result = t.fit(simple_df).transform(simple_df)
        for fn in ["mean", "min", "max", "sum", "std"]:
            assert f"x__rolling_{fn}_2" in result.columns

    def test_multiple_subset_columns(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x", "y"], window_size=2, func=["mean"])
        result = t.fit(simple_df).transform(simple_df)
        assert "x__rolling_mean_2" in result.columns
        assert "y__rolling_mean_2" in result.columns

    def test_drop_columns_true(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=2, func=["mean"], drop_columns=True)
        result = t.fit(simple_df).transform(simple_df)
        assert "x" not in result.columns
        assert "x__rolling_mean_2" in result.columns

    def test_drop_columns_false(self, simple_df):
        t = RollingStatisticsFeatures(
            subset=["x"], window_size=2, func=["mean"], drop_columns=False
        )
        result = t.fit(simple_df).transform(simple_df)
        assert "x" in result.columns

    def test_min_periods_one(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["mean"], min_periods=1)
        result = t.fit(simple_df).transform(simple_df)
        # min_periods=1: first row should be valid (just itself)
        assert result["x__rolling_mean_3"][0] == pytest.approx(1.0)

    def test_window_size_one(self, simple_df):
        t = RollingStatisticsFeatures(subset=["x"], window_size=1, func=["mean"])
        result = t.fit(simple_df).transform(simple_df)
        assert result["x__rolling_mean_1"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_by_parameter_groups(self):
        X = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "val": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            }
        )
        t = RollingStatisticsFeatures(subset=["val"], window_size=2, func=["mean"], by=["group"])
        result = t.fit(X).transform(X)
        assert "val__rolling_mean_2" in result.columns
        # Within group B the first value should be null (window not full yet)
        group_b = result.filter(pl.col("group") == "B")
        assert group_b["val__rolling_mean_2"][0] is None
        assert group_b["val__rolling_mean_2"][1] == pytest.approx(15.0)

    def test_invalid_func_raises(self):
        with pytest.raises(ValueError, match="Invalid function"):
            RollingStatisticsFeatures(subset=["x"], window_size=2, func=["invalid"])

    def test_get_params(self):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["mean"])
        params = t.get_params()
        assert params["window_size"] == 3

    def test_set_params(self):
        t = RollingStatisticsFeatures(subset=["x"], window_size=3, func=["mean"])
        t.set_params(window_size=5)
        assert t.window_size == 5

    def test_fit_transform(self, simple_df):
        result = RollingStatisticsFeatures(subset=["x"], window_size=2, func=["sum"]).fit_transform(
            simple_df
        )
        assert isinstance(result, pl.DataFrame)
