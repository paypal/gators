import polars as pl
import pytest
from polars.testing import assert_frame_equal
import math

from gators.scalers import LogScaler


def test_log_scaler_default_natural():
    """Test LogScaler with default parameters (natural log, all columns)."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 2.718281828, 7.389056099, 20.085536923],
            "col2": [1.0, 10.0, 100.0, 1000.0],
        }
    )

    scaler = LogScaler()
    scaler.fit(X)
    result = scaler.transform(X)
    
    expected = X.with_columns(
        [
            pl.col("col1").log().alias("col1__log_ln"),
            pl.col("col2").log().alias("col2__log_ln"),
        ]
    ).drop(["col1", "col2"])
    
    assert_frame_equal(result, expected)


def test_log_scaler_log10():
    """Test LogScaler with base 10."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 10.0, 100.0, 1000.0],
            "col2": [0.1, 1.0, 10.0, 100.0],
        }
    )

    scaler = LogScaler(base="10")
    scaler.fit(X)
    result = scaler.transform(X)
    
    expected = X.with_columns(
        [
            pl.col("col1").log10().alias("col1__log_10"),
            pl.col("col2").log10().alias("col2__log_10"),
        ]
    ).drop(["col1", "col2"])
    
    assert_frame_equal(result, expected)


def test_log_scaler_log2():
    """Test LogScaler with base 2."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 2.0, 4.0, 8.0, 16.0],
            "col2": [1.0, 2.0, 4.0, 8.0, 16.0],
        }
    )

    scaler = LogScaler(base="2")
    scaler.fit(X)
    result = scaler.transform(X)
    
    expected = X.with_columns(
        [
            pl.col("col1").log(base=2).alias("col1__log_2"),
            pl.col("col2").log(base=2).alias("col2__log_2"),
        ]
    ).drop(["col1", "col2"])
    
    assert_frame_equal(result, expected)


def test_log_scaler_subset_columns():
    """Test LogScaler with subset of columns."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 10.0, 100.0],
            "col2": [2.0, 20.0, 200.0],
            "col3": [5.0, 10.0, 15.0],
        }
    )

    scaler = LogScaler(subset=["col1", "col2"], base="e", drop_columns=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            pl.col("col1").log().alias("col1__log_ln"),
            pl.col("col2").log().alias("col2__log_ln"),
        ]
    )

    assert_frame_equal(result, expected)


def test_log_scaler_natural_values():
    """Test LogScaler with natural log values."""
    X =  pl.DataFrame(
        {
            "col1": [math.e ** 0, math.e ** 1, math.e ** 2, math.e ** 3],
        }
    )

    scaler = LogScaler(base="e")
    result = scaler.fit_transform(X)

    # ln(e^x) = x
    expected_values = [0.0, 1.0, 2.0, 3.0]
    assert result["col1__log_ln"].to_list() == pytest.approx(expected_values, rel=1e-10)


def test_log_scaler_log10_powers():
    """Test LogScaler with log10 on powers of 10."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 10.0, 100.0, 1000.0, 10000.0],
        }
    )

    scaler = LogScaler(base="10")
    result = scaler.fit_transform(X)

    # log10(10^x) = x
    expected_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result["col1__log_10"].to_list() == pytest.approx(expected_values, rel=1e-10)


def test_log_scaler_log2_powers():
    """Test LogScaler with log2 on powers of 2."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        }
    )

    scaler = LogScaler(base="2")
    result = scaler.fit_transform(X)

    # log2(2^x) = x
    expected_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["col1__log_2"].to_list() == pytest.approx(expected_values, rel=1e-10)


def test_log_scaler_fit_transform():
    """Test LogScaler fit_transform method."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 10.0, 100.0],
        }
    )

    scaler = LogScaler(base="10")
    result = scaler.fit_transform(X)

    expected = pl.DataFrame(
        {
            "col1__log_10": [0.0, 1.0, 2.0],
        }
    )

    assert_frame_equal(result, expected)


def test_log_scaler_drop_columns_false():
    """Test LogScaler with drop_columns=False."""
    X =  pl.DataFrame(
        {
            "col1": [1.0, 10.0, 100.0],
        }
    )

    scaler = LogScaler(base="e", drop_columns=False)
    scaler.fit(X)
    result = scaler.transform(X)

    # Should have both original and transformed columns
    assert "col1" in result.columns
    assert "col1__log_ln" in result.columns
    assert len(result.columns) == 2


def test_log_scaler_all_bases():
    """Test that different bases produce different results."""
    X =  pl.DataFrame(
        {
            "col1": [10.0, 100.0],
        }
    )

    scaler_ln = LogScaler(base="e")
    scaler_10 = LogScaler(base="10")
    scaler_2 = LogScaler(base="2")

    result_ln = scaler_ln.fit_transform(X)
    result_10 = scaler_10.fit_transform(X)
    result_2 = scaler_2.fit_transform(X)

    # All should be different
    values_ln = result_ln["col1__log_ln"].to_list()
    values_10 = result_10["col1__log_10"].to_list()
    values_2 = result_2["col1__log_2"].to_list()

    assert values_ln != values_10
    assert values_ln != values_2
    assert values_10 != values_2


def test_log_scaler_column_naming():
    """Test that column naming is correct for each base."""
    X =  pl.DataFrame({"col1": [1.0, 10.0]})

    scaler_ln = LogScaler(base="e")
    scaler_ln.fit(X)
    assert scaler_ln._column_mapping == {"col1": "col1__log_ln"}

    scaler_10 = LogScaler(base="10")
    scaler_10.fit(X)
    assert scaler_10._column_mapping == {"col1": "col1__log_10"}

    scaler_2 = LogScaler(base="2")
    scaler_2.fit(X)
    assert scaler_2._column_mapping == {"col1": "col1__log_2"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
