import polars as pl
import pytest
from polars.testing import assert_frame_equal
import math

from gators.scalers import ArcSinSquareRootScaler


def test_arcsin_scaler_default():
    """Test ArcSinSquareRootScaler with default parameters (all columns)."""
    X =  pl.DataFrame(
        {
            "col1": [0.0, 0.25, 0.5, 0.75, 1.0],
            "col2": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    scaler = ArcSinSquareRootScaler()
    scaler.fit(X)
    result = scaler.transform(X)
    
    expected = X.with_columns(
        [
            pl.col("col1").sqrt().arcsin().alias("col1__arcsin"),
            pl.col("col2").sqrt().arcsin().alias("col2__arcsin"),
        ]
    ).drop(["col1", "col2"])
    
    assert_frame_equal(result, expected)


def test_arcsin_scaler_subset_columns():
    """Test ArcSinSquareRootScaler with subset of columns."""
    X =  pl.DataFrame(
        {
            "col1": [0.0, 0.25, 0.5, 0.75],
            "col2": [0.1, 0.3, 0.5, 0.7],
            "col3": [5.0, 10.0, 15.0, 20.0],
        }
    )

    scaler = ArcSinSquareRootScaler(subset=["col1", "col2"], drop_columns=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            pl.col("col1").sqrt().arcsin().alias("col1__arcsin"),
            pl.col("col2").sqrt().arcsin().alias("col2__arcsin"),
        ]
    )

    assert_frame_equal(result, expected)


def test_arcsin_scaler_proportions():
    """Test ArcSinSquareRootScaler with typical proportion data."""
    X =  pl.DataFrame(
        {
            "success_rate": [0.1, 0.25, 0.5, 0.75, 0.9],
        }
    )

    scaler = ArcSinSquareRootScaler()
    scaler.fit(X)
    result = scaler.transform(X)

    # Verify transformation is correct
    expected_values = [
        math.asin(math.sqrt(0.1)),
        math.asin(math.sqrt(0.25)),
        math.asin(math.sqrt(0.5)),
        math.asin(math.sqrt(0.75)),
        math.asin(math.sqrt(0.9)),
    ]

    assert result["success_rate__arcsin"].to_list() == pytest.approx(expected_values, rel=1e-10)


def test_arcsin_scaler_boundary_values():
    """Test ArcSinSquareRootScaler with boundary values 0 and 1."""
    X =  pl.DataFrame(
        {
            "col1": [0.0, 1.0],
        }
    )

    scaler = ArcSinSquareRootScaler()
    scaler.fit(X)
    result = scaler.transform(X)

    # arcsin(sqrt(0)) = 0, arcsin(sqrt(1)) = π/2
    expected_values = [0.0, math.pi / 2]
    assert result["col1__arcsin"].to_list() == pytest.approx(expected_values, rel=1e-10)


def test_arcsin_scaler_fit_transform():
    """Test ArcSinSquareRootScaler fit_transform method."""
    X =  pl.DataFrame(
        {
            "col1": [0.25, 0.5, 0.75],
        }
    )

    scaler = ArcSinSquareRootScaler()
    result = scaler.fit_transform(X)

    expected = X.with_columns(
        pl.col("col1").sqrt().arcsin().alias("col1__arcsin")
    ).drop("col1")

    assert_frame_equal(result, expected)


def test_arcsin_scaler_drop_columns_false():
    """Test ArcSinSquareRootScaler with drop_columns=False."""
    X =  pl.DataFrame(
        {
            "col1": [0.1, 0.5, 0.9],
        }
    )

    scaler = ArcSinSquareRootScaler(drop_columns=False)
    scaler.fit(X)
    result = scaler.transform(X)

    # Should have both original and transformed columns
    assert "col1" in result.columns
    assert "col1__arcsin" in result.columns
    assert len(result.columns) == 2


def test_arcsin_scaler_range():
    """Test that ArcSinSquareRootScaler output is in expected range [0, π/2]."""
    X =  pl.DataFrame(
        {
            "col1": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    )

    scaler = ArcSinSquareRootScaler()
    result = scaler.fit_transform(X)

    # All values should be between 0 and π/2
    values = result["col1__arcsin"].to_list()
    assert all(0 <= v <= math.pi / 2 for v in values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
