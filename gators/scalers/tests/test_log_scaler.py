import polars as pl
import pytest
from polars.testing import assert_frame_equal
import math

from gators.scalers import Log1pScaler


def test_log_scaler_default_natural():
    """Test Log1pScaler with default parameters (natural log, all columns)."""
    X = pl.DataFrame(
        {
            "col1": [1.0, 2.718281828, 7.389056099, 20.085536923],
            "col2": [1.0, 10.0, 100.0, 1000.0],
        }
    )

    scaler = Log1pScaler(inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            (pl.col("col1") + 1).log().alias("col1__log1p"),
            (pl.col("col2") + 1).log().alias("col2__log1p"),
        ]
    ).drop(["col1", "col2"])

    assert_frame_equal(result, expected)


def test_log_scaler_log10():
    """Test Log1pScaler with base 10."""
    X = pl.DataFrame(
        {
            "col1": [1.0, 10.0, 100.0, 1000.0],
            "col2": [0.1, 1.0, 10.0, 100.0],
        }
    )

    scaler = Log1pScaler(base="10", inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            (pl.col("col1") + 1).log(base=10).alias("col1__log1p_10"),
            (pl.col("col2") + 1).log(base=10).alias("col2__log1p_10"),
        ]
    ).drop(["col1", "col2"])

    assert_frame_equal(result, expected)


def test_log_scaler_log2():
    """Test Log1pScaler with base 2."""
    X = pl.DataFrame(
        {
            "col1": [1.0, 2.0, 4.0, 8.0, 16.0],
            "col2": [1.0, 2.0, 4.0, 8.0, 16.0],
        }
    )

    scaler = Log1pScaler(base="2", inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            (pl.col("col1") + 1).log(base=2).alias("col1__log1p_2"),
            (pl.col("col2") + 1).log(base=2).alias("col2__log1p_2"),
        ]
    ).drop(["col1", "col2"])

    assert_frame_equal(result, expected)


def test_log_scaler_subset_columns():
    """Test Log1pScaler with subset of columns."""
    X = pl.DataFrame(
        {
            "col1": [1.0, 10.0, 100.0],
            "col2": [2.0, 20.0, 200.0],
            "col3": [5.0, 10.0, 15.0],
        }
    )

    scaler = Log1pScaler(subset=["col1", "col2"], base="e", drop_columns=False, inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            (pl.col("col1") + 1).log().alias("col1__log1p"),
            (pl.col("col2") + 1).log().alias("col2__log1p"),
        ]
    )

    assert_frame_equal(result, expected)


def test_log_scaler_natural_values():
    """Test Log1pScaler with natural log values."""
    X = pl.DataFrame(
        {
            "col1": [0.0, math.e - 1, math.e**2 - 1, math.e**3 - 1],
        }
    )

    scaler = Log1pScaler(base="e", inplace=False)
    result = scaler.fit_transform(X)

    # ln(1 + e^x - 1) = x
    expected_values = [0.0, 1.0, 2.0, 3.0]
    assert result["col1__log1p"].to_list() == pytest.approx(expected_values, rel=1e-10)


def test_log_scaler_log10_powers():
    """Test Log1pScaler with log10 on powers of 10."""
    X = pl.DataFrame(
        {
            "col1": [0.0, 9.0, 99.0, 999.0, 9999.0],
        }
    )

    scaler = Log1pScaler(base="10", inplace=False)
    result = scaler.fit_transform(X)

    # log10(1 + 10^x - 1) = x
    expected_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result["col1__log1p_10"].to_list() == pytest.approx(expected_values, rel=1e-10)


def test_log_scaler_log2_powers():
    """Test Log1pScaler with log2 on powers of 2."""
    X = pl.DataFrame(
        {
            "col1": [0.0, 1.0, 3.0, 7.0, 15.0, 31.0],
        }
    )

    scaler = Log1pScaler(base="2", inplace=False)
    result = scaler.fit_transform(X)

    # log2(1 + 2^x - 1) = x
    expected_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["col1__log1p_2"].to_list() == pytest.approx(expected_values, rel=1e-10)


def test_log_scaler_fit_transform():
    """Test Log1pScaler fit_transform method."""
    X = pl.DataFrame(
        {
            "col1": [0.0, 9.0, 99.0],
        }
    )

    scaler = Log1pScaler(base="10", inplace=False)
    result = scaler.fit_transform(X)

    expected = pl.DataFrame(
        {
            "col1__log1p_10": [0.0, 1.0, 2.0],
        }
    )

    assert_frame_equal(result, expected)


def test_log_scaler_drop_columns_false():
    """Test Log1pScaler with drop_columns=False."""
    X = pl.DataFrame(
        {
            "col1": [1.0, 10.0, 100.0],
        }
    )

    scaler = Log1pScaler(base="e", drop_columns=False, inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    # Should have both original and transformed columns
    assert "col1" in result.columns
    assert "col1__log1p" in result.columns
    assert len(result.columns) == 2


def test_log_scaler_all_bases():
    """Test that different bases produce different results."""
    X = pl.DataFrame(
        {
            "col1": [10.0, 100.0],
        }
    )

    scaler_ln = Log1pScaler(base="e", inplace=False)
    scaler_10 = Log1pScaler(base="10", inplace=False)
    scaler_2 = Log1pScaler(base="2", inplace=False)

    result_ln = scaler_ln.fit_transform(X)
    result_10 = scaler_10.fit_transform(X)
    result_2 = scaler_2.fit_transform(X)

    # All should be different
    values_ln = result_ln["col1__log1p"].to_list()
    values_10 = result_10["col1__log1p_10"].to_list()
    values_2 = result_2["col1__log1p_2"].to_list()

    assert values_ln != values_10
    assert values_ln != values_2
    assert values_10 != values_2


def test_log_scaler_column_naming():
    """Test that column naming is correct for each base."""
    X = pl.DataFrame({"col1": [1.0, 10.0]})

    scaler_ln = Log1pScaler(base="e", inplace=False)
    scaler_ln.fit(X)
    assert scaler_ln._column_mapping == {"col1": "col1__log1p"}

    scaler_10 = Log1pScaler(base="10", inplace=False)
    scaler_10.fit(X)
    assert scaler_10._column_mapping == {"col1": "col1__log1p_10"}

    scaler_2 = Log1pScaler(base="2", inplace=False)
    scaler_2.fit(X)
    assert scaler_2._column_mapping == {"col1": "col1__log1p_2"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_log_scaler_inplace_true_base_e():
    X = pl.DataFrame({"col1": [0.0, 1.0, 9.0], "col2": [0.0, 9.0, 99.0]})
    scaler = Log1pScaler(base="e").fit(X)
    result = scaler.transform(X)
    assert result.columns == ["col1", "col2"]
    assert "col1__log1p" not in result.columns
    import math
    assert result["col1"].to_list()[0] == pytest.approx(math.log(1), abs=1e-10)


def test_log_scaler_inplace_true_base_10():
    X = pl.DataFrame({"col1": [0.0, 9.0, 99.0]})
    scaler = Log1pScaler(base="10").fit(X)
    result = scaler.transform(X)
    assert result.columns == ["col1"]
    assert result["col1"].to_list() == pytest.approx([0.0, 1.0, 2.0], rel=1e-5)


def test_log_scaler_inplace_true_base_2():
    X = pl.DataFrame({"col1": [0.0, 1.0, 3.0]})
    scaler = Log1pScaler(base="2").fit(X)
    result = scaler.transform(X)
    assert result.columns == ["col1"]
    assert result["col1"].to_list() == pytest.approx([0.0, 1.0, 2.0], rel=1e-5)


def test_log_scaler_inverse_transform_inplace_true_base_e():
    X = pl.DataFrame({"a": [0.0, 1.0, 10.0]})
    scaler = Log1pScaler(base="e").fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5


def test_log_scaler_inverse_transform_inplace_true_base_10():
    X = pl.DataFrame({"a": [0.0, 1.0, 10.0]})
    scaler = Log1pScaler(base="10").fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5


def test_log_scaler_inverse_transform_inplace_true_base_2():
    X = pl.DataFrame({"a": [0.0, 1.0, 10.0]})
    scaler = Log1pScaler(base="2").fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5


def test_log_scaler_get_params_includes_inplace():
    scaler = Log1pScaler()
    params = scaler.get_params()
    assert "inplace" in params
    assert params["inplace"] is True
