import polars as pl
import pytest
from polars.testing import assert_frame_equal
import math

from gators.scalers import ArcSinhScaler


def test_arcsinh_scaler_default():
    """Test ArcSinhScaler with default parameters (all columns)."""
    X = pl.DataFrame(
        {
            "col1": [-10.0, -5.0, 0.0, 5.0, 10.0],
            "col2": [-100.0, -50.0, 0.0, 50.0, 100.0],
        }
    )

    scaler = ArcSinhScaler(inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            pl.col("col1").arcsinh().alias("col1__arcsinh"),
            pl.col("col2").arcsinh().alias("col2__arcsinh"),
        ]
    ).drop(["col1", "col2"])

    assert_frame_equal(result, expected)


def test_arcsinh_scaler_subset_columns():
    """Test ArcSinhScaler with subset of columns."""
    X = pl.DataFrame(
        {
            "col1": [-10.0, -5.0, 0.0, 5.0],
            "col2": [-20.0, -10.0, 0.0, 10.0],
            "col3": [5.0, 10.0, 15.0, 20.0],
        }
    )

    scaler = ArcSinhScaler(subset=["col1", "col2"], drop_columns=False, inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            pl.col("col1").arcsinh().alias("col1__arcsinh"),
            pl.col("col2").arcsinh().alias("col2__arcsinh"),
        ]
    )

    assert_frame_equal(result, expected)


def test_arcsinh_scaler_symmetry():
    """Test that ArcSinhScaler preserves symmetry: asinh(-x) = -asinh(x)."""
    X = pl.DataFrame(
        {
            "col1": [-10.0, -5.0, -1.0, 1.0, 5.0, 10.0],
        }
    )

    scaler = ArcSinhScaler(inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    values = result["col1__arcsinh"].to_list()

    # Check symmetry
    assert values[0] == pytest.approx(-values[5], rel=1e-10)  # -10 vs 10
    assert values[1] == pytest.approx(-values[4], rel=1e-10)  # -5 vs 5
    assert values[2] == pytest.approx(-values[3], rel=1e-10)  # -1 vs 1


def test_arcsinh_scaler_zero():
    """Test that ArcSinhScaler maps 0 to 0."""
    X = pl.DataFrame(
        {
            "col1": [-1.0, 0.0, 1.0],
        }
    )

    scaler = ArcSinhScaler(inplace=False)
    result = scaler.fit_transform(X)

    # asinh(0) = 0
    assert result["col1__arcsinh"].to_list()[1] == pytest.approx(0.0, abs=1e-10)


def test_arcsinh_scaler_large_values():
    """Test ArcSinhScaler behaves like log for large values."""
    X = pl.DataFrame(
        {
            "col1": [100.0, 1000.0, 10000.0],
        }
    )

    scaler = ArcSinhScaler(inplace=False)
    result = scaler.fit_transform(X)

    # For large x, asinh(x) ≈ log(2x)
    # Verify it's approximately logarithmic
    values = result["col1__arcsinh"].to_list()
    expected_approx = [
        math.log(2 * 100),
        math.log(2 * 1000),
        math.log(2 * 10000),
    ]

    assert values == pytest.approx(expected_approx, rel=0.01)  # Within 1%


def test_arcsinh_scaler_negative_values():
    """Test ArcSinhScaler handles negative values correctly."""
    X = pl.DataFrame(
        {
            "returns": [-100.0, -10.0, 0.0, 10.0, 100.0],
        }
    )

    scaler = ArcSinhScaler(inplace=False)
    result = scaler.fit_transform(X)

    # Verify all values are finite (no NaN or inf)
    assert result["returns__arcsinh"].null_count() == 0
    assert all(math.isfinite(v) for v in result["returns__arcsinh"].to_list())


def test_arcsinh_scaler_fit_transform():
    """Test ArcSinhScaler fit_transform method."""
    X = pl.DataFrame(
        {
            "col1": [-5.0, 0.0, 5.0],
        }
    )

    scaler = ArcSinhScaler(inplace=False)
    result = scaler.fit_transform(X)

    expected = X.with_columns(pl.col("col1").arcsinh().alias("col1__arcsinh")).drop("col1")

    assert_frame_equal(result, expected)


def test_arcsinh_scaler_drop_columns_false():
    """Test ArcSinhScaler with drop_columns=False."""
    X = pl.DataFrame(
        {
            "col1": [-10.0, 0.0, 10.0],
        }
    )

    scaler = ArcSinhScaler(drop_columns=False, inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    # Should have both original and transformed columns
    assert "col1" in result.columns
    assert "col1__arcsinh" in result.columns
    assert len(result.columns) == 2


def test_arcsinh_scaler_near_zero():
    """Test ArcSinhScaler behaves linearly near zero."""
    X = pl.DataFrame(
        {
            "col1": [-0.01, -0.001, 0.0, 0.001, 0.01],
        }
    )

    scaler = ArcSinhScaler(inplace=False)
    result = scaler.fit_transform(X)

    # For small x, asinh(x) ≈ x
    values = result["col1__arcsinh"].to_list()
    original = X["col1"].to_list()

    for v, o in zip(values, original):
        # Should be very close to original value for small values
        assert abs(v - o) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_arcsinh_scaler_inplace_true_default():
    X = pl.DataFrame({"col1": [-10.0, 0.0, 10.0], "col2": [-5.0, 0.0, 5.0]})
    scaler = ArcSinhScaler().fit(X)
    result = scaler.transform(X)
    assert result.columns == ["col1", "col2"]
    assert "col1__arcsinh" not in result.columns
    assert result["col1"].to_list()[1] == pytest.approx(0.0, abs=1e-10)


def test_arcsinh_scaler_inplace_true_subset():
    X = pl.DataFrame(
        {"col1": [-5.0, 0.0, 5.0], "col2": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]}
    )
    scaler = ArcSinhScaler(subset=["col1"]).fit(X)
    result = scaler.transform(X)
    assert result.columns == ["col1", "col2", "cat"]
    assert result["col2"].to_list() == [1.0, 2.0, 3.0]


def test_arcsinh_scaler_inverse_transform_inplace_true():
    X = pl.DataFrame({"a": [-100.0, 0.0, 100.0]})
    scaler = ArcSinhScaler().fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5


def test_arcsinh_scaler_get_params_includes_inplace():
    scaler = ArcSinhScaler()
    params = scaler.get_params()
    assert "inplace" in params
    assert params["inplace"] is True
