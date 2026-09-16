"""Tests for inverse_transform on all scalers."""

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from gators.scalers import (
    ArcSinSquareRootScaler,
    ArcSinhScaler,
    BoxCox,
    Log1pScaler,
    MinmaxScaler,
    PowerScaler,
    RobustScaler,
    StandardScaler,
    YeoJohnson,
)


def _roundtrip(scaler, X: pl.DataFrame, drop_columns: bool = True, tol: float = 1e-5) -> None:
    """Fit, transform, inverse_transform and assert recovery within tol."""
    scaler.fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    # Compare only the columns that were in X
    for col in X.columns:
        assert col in X_r.columns, f"Column {col} missing after inverse_transform"
        max_diff = (X_r[col].cast(pl.Float64) - X[col].cast(pl.Float64)).abs().max()
        assert max_diff < tol, f"Column {col}: max diff {max_diff} exceeds {tol}"


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------


class TestStandardScalerInverse:
    def test_roundtrip_drop_columns_true(self):
        X = pl.DataFrame({"a": [10.0, 20.0, 30.0, 40.0], "b": [1.0, 2.0, 3.0, 4.0]})
        _roundtrip(StandardScaler(drop_columns=True, inplace=False), X)

    def test_roundtrip_drop_columns_false(self):
        X = pl.DataFrame({"a": [10.0, 20.0, 30.0, 40.0]})
        scaler = StandardScaler(drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_roundtrip_inplace_true(self):
        X = pl.DataFrame({"a": [10.0, 20.0, 30.0, 40.0]})
        _roundtrip(StandardScaler(), X)

    def test_inverse_transform_constant_column(self):
        """scale==0 branch: inverse_transform must return the original constant."""
        X = pl.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
        scaler = StandardScaler()
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert X_r["a"].to_list() == pytest.approx([5.0, 5.0, 5.0])


# ---------------------------------------------------------------------------
# MinmaxScaler
# ---------------------------------------------------------------------------


class TestMinmaxScalerInverse:
    def test_roundtrip(self):
        X = pl.DataFrame({"a": [0.0, 25.0, 50.0, 75.0, 100.0]})
        _roundtrip(MinmaxScaler(inplace=False), X)

    def test_roundtrip_drop_columns_false(self):
        X = pl.DataFrame({"a": [0.0, 25.0, 50.0, 75.0, 100.0]})
        scaler = MinmaxScaler(drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_roundtrip_inplace_true(self):
        X = pl.DataFrame({"a": [0.0, 25.0, 50.0, 75.0, 100.0]})
        _roundtrip(MinmaxScaler(), X)

    def test_inverse_transform_constant_column(self):
        """scale==0 branch: inverse_transform must return the original constant."""
        X = pl.DataFrame({"a": [7.0, 7.0, 7.0], "b": [1.0, 2.0, 3.0]})
        scaler = MinmaxScaler()
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert X_r["a"].to_list() == pytest.approx([7.0, 7.0, 7.0])


# ---------------------------------------------------------------------------
# RobustScaler
# ---------------------------------------------------------------------------


class TestRobustScalerInverse:
    def test_roundtrip(self):
        X = pl.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 200.0]})
        _roundtrip(RobustScaler(inplace=False), X)

    def test_roundtrip_drop_columns_false(self):
        X = pl.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 200.0]})
        scaler = RobustScaler(drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-4

    def test_roundtrip_inplace_true(self):
        X = pl.DataFrame({"a": [10.0, 20.0, 30.0, 40.0, 200.0]})
        _roundtrip(RobustScaler(), X)

    def test_inverse_transform_constant_column(self):
        """scale==0 branch: inverse_transform must return the original constant."""
        X = pl.DataFrame({"a": [3.0, 3.0, 3.0], "b": [1.0, 2.0, 3.0]})
        scaler = RobustScaler()
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert X_r["a"].to_list() == pytest.approx([3.0, 3.0, 3.0])


# ---------------------------------------------------------------------------
# Log1pScaler
# ---------------------------------------------------------------------------


class TestLog1pScalerInverse:
    def test_roundtrip_base_e(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 10.0, 100.0]})
        _roundtrip(Log1pScaler(base="e", inplace=False), X)

    def test_roundtrip_base_10(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 10.0, 100.0]})
        _roundtrip(Log1pScaler(base="10", inplace=False), X)

    def test_roundtrip_base_2(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 10.0, 100.0]})
        _roundtrip(Log1pScaler(base="2", inplace=False), X)

    def test_roundtrip_drop_columns_false_base_e(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = Log1pScaler(base="e", drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_roundtrip_drop_columns_false_base_10(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = Log1pScaler(base="10", drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_roundtrip_drop_columns_false_base_2(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = Log1pScaler(base="2", drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_roundtrip_inplace_true_all_bases(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 10.0, 100.0]})
        for base in ["e", "10", "2"]:
            _roundtrip(Log1pScaler(base=base), X)


# ---------------------------------------------------------------------------
# PowerScaler
# ---------------------------------------------------------------------------


class TestPowerScalerInverse:
    def test_roundtrip_sqrt(self):
        X = pl.DataFrame({"a": [1.0, 4.0, 9.0, 16.0]})
        _roundtrip(PowerScaler(power=0.5, inplace=False), X)

    def test_roundtrip_square(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        _roundtrip(PowerScaler(power=2.0, inplace=False), X)

    def test_power_zero_raises(self):
        X = pl.DataFrame({"a": [1.0, 2.0]})
        scaler = PowerScaler(power=0.0)
        scaler.fit(X)
        X_t = pl.DataFrame({"a__power_0_0": [1.0, 2.0]})
        with pytest.raises(ValueError, match="power=0"):
            scaler.inverse_transform(X_t)

    def test_roundtrip_drop_columns_false(self):
        X = pl.DataFrame({"a": [1.0, 4.0, 9.0]})
        scaler = PowerScaler(power=0.5, drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_roundtrip_inplace_true(self):
        X = pl.DataFrame({"a": [1.0, 4.0, 9.0, 16.0]})
        _roundtrip(PowerScaler(power=0.5), X)


# ---------------------------------------------------------------------------
# ArcSinhScaler
# ---------------------------------------------------------------------------


class TestArcSinhScalerInverse:
    def test_roundtrip(self):
        X = pl.DataFrame({"a": [-100.0, -10.0, 0.0, 10.0, 100.0]})
        _roundtrip(ArcSinhScaler(inplace=False), X)

    def test_roundtrip_drop_columns_false(self):
        X = pl.DataFrame({"a": [-5.0, 0.0, 5.0]})
        scaler = ArcSinhScaler(drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_roundtrip_inplace_true(self):
        X = pl.DataFrame({"a": [-100.0, 0.0, 100.0]})
        _roundtrip(ArcSinhScaler(), X)


# ---------------------------------------------------------------------------
# ArcSinSquareRootScaler
# ---------------------------------------------------------------------------


class TestArcSinSquareRootScalerInverse:
    def test_roundtrip(self):
        X = pl.DataFrame({"a": [0.0, 0.25, 0.5, 0.75, 1.0]})
        _roundtrip(ArcSinSquareRootScaler(inplace=False), X)

    def test_roundtrip_drop_columns_false(self):
        X = pl.DataFrame({"a": [0.1, 0.5, 0.9]})
        scaler = ArcSinSquareRootScaler(drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_roundtrip_inplace_true(self):
        X = pl.DataFrame({"a": [0.0, 0.25, 0.5, 0.75, 1.0]})
        _roundtrip(ArcSinSquareRootScaler(), X)


# ---------------------------------------------------------------------------
# BoxCox
# ---------------------------------------------------------------------------


class TestBoxCoxInverse:
    def test_roundtrip_lambda_nonzero(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 4.0, 8.0]})
        _roundtrip(BoxCox(lambdas={"a": 0.5}, inplace=False), X)

    def test_roundtrip_lambda_zero(self):
        X = pl.DataFrame({"a": [1.0, math.e, math.e**2, math.e**3]})
        _roundtrip(BoxCox(lambdas={"a": 0}, inplace=False), X)

    def test_roundtrip_drop_columns_false(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = BoxCox(lambdas={"a": 0.5}, drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_missing_column_skipped(self):
        """continue branch: column not in X is silently skipped."""
        X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        scaler = BoxCox(lambdas={"a": 0.5, "b": 0.5}, inplace=False)
        scaler.fit(X)
        # Only pass one scaled column — the other should be skipped without error
        X_partial = pl.DataFrame({"a__boxcox": [0.414, 0.828]})
        result = scaler.inverse_transform(X_partial)
        assert "a" in result.columns
        assert "b" not in result.columns

    def test_roundtrip_inplace_true(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 4.0, 8.0]})
        _roundtrip(BoxCox(lambdas={"a": 0.5}), X)

    def test_roundtrip_inplace_true_lambda_zero(self):
        X = pl.DataFrame({"a": [1.0, math.e, math.e**2]})
        _roundtrip(BoxCox(lambdas={"a": 0}), X)


# ---------------------------------------------------------------------------
# YeoJohnson
# ---------------------------------------------------------------------------


class TestYeoJohnsonInverse:
    def test_roundtrip_general_lambda_positive_values(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 2.0, 5.0, 10.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 0.5}, inplace=False), X)

    def test_roundtrip_general_lambda_negative_values(self):
        X = pl.DataFrame({"a": [-10.0, -5.0, -1.0, 0.0, 1.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 0.5}, inplace=False), X)

    def test_roundtrip_lambda_zero_positive_values(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 0}, inplace=False), X)

    def test_roundtrip_lambda_zero_negative_values(self):
        X = pl.DataFrame({"a": [-3.0, -2.0, -1.0, 0.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 0}, inplace=False), X)

    def test_roundtrip_lambda_two_positive_values(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 2}, inplace=False), X)

    def test_roundtrip_lambda_two_negative_values(self):
        X = pl.DataFrame({"a": [-3.0, -2.0, -1.0, 0.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 2}, inplace=False), X)

    def test_roundtrip_drop_columns_false(self):
        X = pl.DataFrame({"a": [-2.0, 0.0, 2.0]})
        scaler = YeoJohnson(lambdas={"a": 0.5}, drop_columns=False, inplace=False)
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_r = scaler.inverse_transform(X_t)
        assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5

    def test_missing_column_skipped(self):
        """continue branch: column not in X is silently skipped."""
        X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        scaler = YeoJohnson(lambdas={"a": 0.5, "b": 0.5}, inplace=False)
        scaler.fit(X)
        X_partial = pl.DataFrame({"a__yeojonhson": [0.5, 1.0]})
        result = scaler.inverse_transform(X_partial)
        assert "a" in result.columns
        assert "b" not in result.columns

    def test_roundtrip_inplace_true_positive(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 2.0, 5.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 0.5}), X)

    def test_roundtrip_inplace_true_negative(self):
        X = pl.DataFrame({"a": [-3.0, -1.0, 0.0, 1.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 0.5}), X)

    def test_roundtrip_inplace_true_lambda_zero(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 0}), X)

    def test_roundtrip_inplace_true_lambda_two(self):
        X = pl.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        _roundtrip(YeoJohnson(lambdas={"a": 2}), X)
