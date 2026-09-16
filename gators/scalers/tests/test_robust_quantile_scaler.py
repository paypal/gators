"""Tests for RobustScaler."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.scalers import RobustScaler


@pytest.fixture
def X():
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": ["x", "y", "x", "y", "x"],
        }
    )


class TestRobustScaler:
    def test_default_drop_columns(self, X):
        scaler = RobustScaler(inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        # originals dropped, scaled columns present
        assert "a" not in result.columns
        assert "b" not in result.columns
        assert "a__robust_quantile_scale" in result.columns
        assert "b__robust_quantile_scale" in result.columns
        # non-numeric preserved
        assert "cat" in result.columns

    def test_keep_original_columns(self, X):
        scaler = RobustScaler(drop_columns=False, inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        assert "a" in result.columns
        assert "a__robust_quantile_scale" in result.columns

    def test_median_is_zero_after_scaling(self, X):
        scaler = RobustScaler(subset=["a", "b"], inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        # The median of the scaled column should be 0.0
        assert result["a__robust_quantile_scale"].median() == pytest.approx(0.0, abs=1e-10)
        assert result["b__robust_quantile_scale"].median() == pytest.approx(0.0, abs=1e-10)

    def test_scaled_iqr_is_one(self, X):
        scaler = RobustScaler(subset=["a", "b"], quantile_range=(0.25, 0.75), inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        for col in ["a__robust_quantile_scale", "b__robust_quantile_scale"]:
            q25 = result[col].quantile(0.25)
            q75 = result[col].quantile(0.75)
            assert (q75 - q25) == pytest.approx(1.0, abs=1e-10)

    def test_subset_only_scales_selected_columns(self, X):
        scaler = RobustScaler(subset=["a"], drop_columns=False, inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        assert "a__robust_quantile_scale" in result.columns
        assert "b__robust_quantile_scale" not in result.columns

    def test_custom_quantile_range(self, X):
        scaler = RobustScaler(subset=["a"], quantile_range=(0.1, 0.9), inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        assert "a__robust_quantile_scale" in result.columns
        q10 = result["a__robust_quantile_scale"].quantile(0.1)
        q90 = result["a__robust_quantile_scale"].quantile(0.9)
        assert (q90 - q10) == pytest.approx(1.0, abs=1e-10)

    def test_outlier_robustness(self):
        # With a big outlier, median/IQR-based scaler should not shift the
        # bulk of the data the way mean/std would.
        X = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        scaler = RobustScaler(subset=["v"], inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        # median of original is 3.5; scaled median ~ 0
        assert result["v__robust_quantile_scale"].median() == pytest.approx(0.0, abs=0.5)

    def test_constant_column_scale_zero(self):
        # Column with zero IQR → scale is 0 → all outputs are 0.0
        X = pl.DataFrame({"c": [5.0, 5.0, 5.0, 5.0]})
        scaler = RobustScaler(subset=["c"], inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        assert (result["c__robust_quantile_scale"] == 0.0).all()

    def test_auto_detect_numeric_subset(self):
        X = pl.DataFrame({"num": [1.0, 2.0, 3.0], "text": ["a", "b", "c"]})
        scaler = RobustScaler(inplace=False)
        scaler.fit(X)
        assert scaler.subset == ["num"]

    def test_fit_returns_self(self, X):
        scaler = RobustScaler(inplace=False)
        assert scaler.fit(X) is scaler

    def test_get_params(self):
        scaler = RobustScaler(quantile_range=(0.1, 0.9), drop_columns=False, inplace=False)
        params = scaler.get_params()
        assert params["quantile_range"] == (0.1, 0.9)
        assert params["drop_columns"] is False

    def test_set_params(self):
        scaler = RobustScaler(inplace=False)
        scaler.set_params(quantile_range=(0.1, 0.9))
        assert scaler.quantile_range == (0.1, 0.9)

    def test_invalid_quantile_range_reversed(self):
        with pytest.raises(Exception):
            RobustScaler(quantile_range=(0.75, 0.25))

    def test_invalid_quantile_range_equal(self):
        with pytest.raises(Exception):
            RobustScaler(quantile_range=(0.5, 0.5))

    def test_invalid_quantile_range_out_of_bounds(self):
        with pytest.raises(Exception):
            RobustScaler(quantile_range=(-0.1, 0.9))

    def test_values_correctness(self):
        # Manual verification: values [1,2,3,4,5], median=3, Q1=2, Q3=4, IQR=2
        X = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaler = RobustScaler(subset=["x"], quantile_range=(0.25, 0.75), inplace=False)
        scaler.fit(X)
        result = scaler.transform(X)
        expected = pl.DataFrame({"x__robust_quantile_scale": [-1.0, -0.5, 0.0, 0.5, 1.0]})
        assert_frame_equal(result, expected)

    def test_inplace_true_default(self, X):
        scaler = RobustScaler().fit(X)
        result = scaler.transform(X)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "a__robust_quantile_scale" not in result.columns
        assert result["a"].median() == pytest.approx(0.0, abs=1e-10)

    def test_inplace_true_subset(self, X):
        scaler = RobustScaler(subset=["a"]).fit(X)
        result = scaler.transform(X)
        assert result.columns == ["a", "b", "cat"]
        assert "a__robust_quantile_scale" not in result.columns

    def test_inplace_true_inverse_transform(self, X):
        scaler = RobustScaler(subset=["a", "b"]).fit(X)
        X_t = scaler.transform(X.select(["a", "b"]))
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_get_params_includes_inplace(self):
        scaler = RobustScaler()
        params = scaler.get_params()
        assert "inplace" in params
        assert params["inplace"] is True
