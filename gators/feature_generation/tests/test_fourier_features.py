"""Tests for FourierFeatures — 100 % branch coverage."""

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation import FourierFeatures


@pytest.fixture
def cyclic_df():
    return pl.DataFrame({"day": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


class TestFourierFeatures:
    def test_fit_returns_self(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods=[7.0])
        assert t.fit(cyclic_df) is t

    def test_column_periods_attribute_list(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods=[7.0])
        t.fit(cyclic_df)
        assert t.column_periods_ == {"day": [7.0]}

    def test_column_periods_attribute_dict(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods={"day": [7.0, 14.0]})
        t.fit(cyclic_df)
        assert t.column_periods_ == {"day": [7.0, 14.0]}

    def test_output_columns_single_harmonic(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods=[7.0], n_harmonics=1)
        result = t.fit(cyclic_df).transform(cyclic_df)
        assert "day__fourier_sin_p7_k1" in result.columns
        assert "day__fourier_cos_p7_k1" in result.columns

    def test_output_columns_multiple_harmonics(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods=[7.0], n_harmonics=3)
        result = t.fit(cyclic_df).transform(cyclic_df)
        for k in range(1, 4):
            assert f"day__fourier_sin_p7_k{k}" in result.columns
            assert f"day__fourier_cos_p7_k{k}" in result.columns

    def test_known_values_sin_cos(self):
        """sin(2π * 0 / 4) = 0, cos(2π * 0 / 4) = 1."""
        X = pl.DataFrame({"x": [0.0, 1.0]})
        t = FourierFeatures(subset=["x"], periods=[4.0])
        result = t.fit(X).transform(X)
        assert result["x__fourier_sin_p4_k1"][0] == pytest.approx(0.0, abs=1e-9)
        assert result["x__fourier_cos_p4_k1"][0] == pytest.approx(1.0)
        # sin(2π * 1 / 4) = 1
        assert result["x__fourier_sin_p4_k1"][1] == pytest.approx(1.0)

    def test_full_cycle_returns_to_origin(self):
        """After a full period the sin/cos values should repeat."""
        X = pl.DataFrame({"x": [0.0, 7.0]})
        t = FourierFeatures(subset=["x"], periods=[7.0])
        result = t.fit(X).transform(X)
        assert result["x__fourier_sin_p7_k1"][0] == pytest.approx(
            result["x__fourier_sin_p7_k1"][1], abs=1e-9
        )
        assert result["x__fourier_cos_p7_k1"][0] == pytest.approx(
            result["x__fourier_cos_p7_k1"][1], abs=1e-9
        )

    def test_multiple_periods(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods=[7.0, 30.0])
        result = t.fit(cyclic_df).transform(cyclic_df)
        assert "day__fourier_sin_p7_k1" in result.columns
        assert "day__fourier_sin_p30_k1" in result.columns

    def test_dict_periods_per_column(self):
        X = pl.DataFrame({"day": [0.0, 3.5], "month": [0.0, 6.0]})
        t = FourierFeatures(
            subset=["day", "month"],
            periods={"day": [7.0], "month": [12.0]},
        )
        result = t.fit(X).transform(X)
        assert "day__fourier_sin_p7_k1" in result.columns
        assert "month__fourier_sin_p12_k1" in result.columns
        # day columns should not have period 12 and vice-versa
        assert "day__fourier_sin_p12_k1" not in result.columns
        assert "month__fourier_sin_p7_k1" not in result.columns

    def test_drop_columns_true(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods=[7.0], drop_columns=True)
        result = t.fit(cyclic_df).transform(cyclic_df)
        assert "day" not in result.columns
        assert "day__fourier_sin_p7_k1" in result.columns

    def test_drop_columns_false(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods=[7.0], drop_columns=False)
        result = t.fit(cyclic_df).transform(cyclic_df)
        assert "day" in result.columns

    def test_period_string_format_removes_trailing_zero(self):
        """Period 7.0 should appear as 'p7' not 'p7.0' in column name."""
        X = pl.DataFrame({"x": [0.0]})
        t = FourierFeatures(subset=["x"], periods=[7.0])
        result = t.fit(X).transform(X)
        assert "x__fourier_sin_p7_k1" in result.columns

    def test_float_period_with_decimal(self):
        X = pl.DataFrame({"x": [0.0]})
        t = FourierFeatures(subset=["x"], periods=[7.5])
        result = t.fit(X).transform(X)
        assert "x__fourier_sin_p7.5_k1" in result.columns

    def test_non_positive_period_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FourierFeatures(subset=["x"], periods=[0.0])

    def test_empty_periods_raises(self):
        with pytest.raises(ValueError, match="empty"):
            FourierFeatures(subset=["x"], periods=[])

    def test_empty_dict_periods_raises(self):
        with pytest.raises(ValueError, match="empty"):
            FourierFeatures(subset=["x"], periods={"x": []})

    def test_negative_period_in_dict_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FourierFeatures(subset=["x"], periods={"x": [-1.0]})

    def test_multiple_subset_columns(self):
        X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        t = FourierFeatures(subset=["a", "b"], periods=[4.0])
        result = t.fit(X).transform(X)
        for col in ["a__fourier_sin_p4_k1", "b__fourier_sin_p4_k1"]:
            assert col in result.columns

    def test_original_columns_unchanged(self, cyclic_df):
        t = FourierFeatures(subset=["day"], periods=[7.0])
        result = t.fit(cyclic_df).transform(cyclic_df)
        assert result["day"].to_list() == cyclic_df["day"].to_list()

    def test_get_params(self):
        t = FourierFeatures(subset=["x"], periods=[7.0], n_harmonics=2)
        params = t.get_params()
        assert params["n_harmonics"] == 2
        assert params["periods"] == [7.0]

    def test_set_params(self):
        t = FourierFeatures(subset=["x"], periods=[7.0])
        t.set_params(n_harmonics=3)
        assert t.n_harmonics == 3

    def test_fit_transform(self, cyclic_df):
        result = FourierFeatures(subset=["day"], periods=[7.0]).fit_transform(cyclic_df)
        assert "day__fourier_sin_p7_k1" in result.columns
