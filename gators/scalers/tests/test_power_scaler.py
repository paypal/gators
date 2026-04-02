import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.scalers import PowerScaler


def test_power_scaler_default():
    """Test PowerScaler with default parameters (power=0.5, all columns)."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 4.0, 9.0, 16.0, 25.0],
            "col2": [100.0, 400.0, 900.0, 1600.0, 2500.0],
        }
    )

    scaler = PowerScaler()
    scaler.fit(X)
    result = scaler.transform(X)
    
    expected = X.with_columns(
        [
            (pl.col("col1") ** 0.5).alias("col1__power_0_5"),
            (pl.col("col2") ** 0.5).alias("col2__power_0_5"),
        ]
    ).drop(["col1", "col2"])
    
    assert_frame_equal(result, expected)


def test_power_scaler_subset_columns():
    """Test PowerScaler with subset of columns."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 4.0, 9.0, 16.0],
            "col2": [2.0, 8.0, 18.0, 32.0],
            "col3": [5.0, 10.0, 15.0, 20.0],
        }
    )

    scaler = PowerScaler(subset=["col1", "col2"], power=0.5, drop_columns=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            (pl.col("col1") ** 0.5).alias("col1__power_0_5"),
            (pl.col("col2") ** 0.5).alias("col2__power_0_5"),
        ]
    )

    assert_frame_equal(result, expected)


def test_power_scaler_square():
    """Test PowerScaler with power=2 (squaring)."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0],
            "col2": [10.0, 20.0, 30.0, 40.0],
        }
    )

    scaler = PowerScaler(subset=["col1"], power=2.0)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = pl.DataFrame(
        {
            "col2": [10.0, 20.0, 30.0, 40.0],
            "col1__power_2_0": [1.0, 4.0, 9.0, 16.0],
        }
    )

    assert_frame_equal(result, expected)


def test_power_scaler_cube_root():
    """Test PowerScaler with power=1/3 (cube root)."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 8.0, 27.0, 64.0, 125.0],
        }
    )

    scaler = PowerScaler(power=1/3)
    scaler.fit(X)
    result = scaler.transform(X)

    expected_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["col1__power_0_3333333333333333"].to_list() == pytest.approx(expected_values, rel=1e-5)


def test_power_scaler_fit_transform():
    """Test PowerScaler fit_transform method."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 4.0, 9.0, 16.0],
        }
    )

    scaler = PowerScaler(power=0.5)
    result = scaler.fit_transform(X)

    expected = pl.DataFrame(
        {
            "col1__power_0_5": [1.0, 2.0, 3.0, 4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_power_scaler_drop_columns_false():
    """Test PowerScaler with drop_columns=False."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 4.0, 9.0],
        }
    )

    scaler = PowerScaler(power=2.0, drop_columns=False)
    scaler.fit(X)
    result = scaler.transform(X)

    # Should have both original and transformed columns
    assert "col1" in result.columns
    assert "col1__power_2_0" in result.columns
    assert len(result.columns) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
