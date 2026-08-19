import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.imputers.numeric_imputer import NumericImputer
from gators.exceptions import NotFittedError


@pytest.fixture
def sample_dataframe():
    return pl.DataFrame(
        {
            "A": [1, 2, None, 4, 5],
            "B": [None, 2, 2, None, 5],
            "C": ["x", "y", "z", "x", "y"],
            "D": [1.1, 2.2, None, 4.4, None],
        }
    )


def test_imputer_constant(sample_dataframe):
    imputer = NumericImputer(strategy="constant", value=0, inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(0).alias("A__impute_constant"),
            pl.col("B").fill_null(0).alias("B__impute_constant"),
            pl.col("D").fill_null(0).alias("D__impute_constant"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


# Test case when drop_columns is False
def test_imputer_constant_no_drop_columns(sample_dataframe):
    imputer = NumericImputer(strategy="constant", value=0, drop_columns=False, inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(0).alias("A__impute_constant"),
            pl.col("B").fill_null(0).alias("B__impute_constant"),
            pl.col("D").fill_null(0).alias("D__impute_constant"),
        ]
    )

    assert_frame_equal(transformed, expected)


def test_imputer_specific_columns(sample_dataframe):
    imputer = NumericImputer(strategy="mean", subset=["A", "B"], inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="mean").alias("A__impute_mean"),
            pl.col("B").fill_null(strategy="mean").alias("B__impute_mean"),
        ]
    ).drop(["A", "B"])

    assert_frame_equal(transformed, expected)


def test_imputer_mean(sample_dataframe):
    imputer = NumericImputer(strategy="mean", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="mean").alias("A__impute_mean"),
            pl.col("B").fill_null(strategy="mean").alias("B__impute_mean"),
            pl.col("D").fill_null(strategy="mean").alias("D__impute_mean"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_median(sample_dataframe):
    imputer = NumericImputer(strategy="median", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(sample_dataframe["A"].median()).alias("A__impute_median"),
            pl.col("B").fill_null(sample_dataframe["B"].median()).alias("B__impute_median"),
            pl.col("D").fill_null(sample_dataframe["D"].median()).alias("D__impute_median"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_mode(sample_dataframe):
    imputer = NumericImputer(strategy="most_frequent", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A")
            .fill_null(sample_dataframe["A"].drop_nulls().drop_nans().mode().sort()[0])
            .alias("A__impute_most_frequent"),
            pl.col("B")
            .fill_null(sample_dataframe["B"].drop_nulls().drop_nans().mode().sort()[0])
            .alias("B__impute_most_frequent"),
            pl.col("D")
            .fill_null(sample_dataframe["D"].drop_nulls().drop_nans().mode().sort()[0])
            .alias("D__impute_most_frequent"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_min(sample_dataframe):
    imputer = NumericImputer(strategy="min", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="min").alias("A__impute_min"),
            pl.col("B").fill_null(strategy="min").alias("B__impute_min"),
            pl.col("D").fill_null(strategy="min").alias("D__impute_min"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_max(sample_dataframe):
    imputer = NumericImputer(strategy="max", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="max").alias("A__impute_max"),
            pl.col("B").fill_null(strategy="max").alias("B__impute_max"),
            pl.col("D").fill_null(strategy="max").alias("D__impute_max"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_forward(sample_dataframe):
    imputer = NumericImputer(strategy="forward", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="forward").alias("A__impute_forward"),
            pl.col("B").fill_null(strategy="forward").alias("B__impute_forward"),
            pl.col("D").fill_null(strategy="forward").alias("D__impute_forward"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_backward(sample_dataframe):
    imputer = NumericImputer(strategy="backward", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="backward").alias("A__impute_backward"),
            pl.col("B").fill_null(strategy="backward").alias("B__impute_backward"),
            pl.col("D").fill_null(strategy="backward").alias("D__impute_backward"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_zero(sample_dataframe):
    imputer = NumericImputer(strategy="zero", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="zero").alias("A__impute_zero"),
            pl.col("B").fill_null(strategy="zero").alias("B__impute_zero"),
            pl.col("D").fill_null(strategy="zero").alias("D__impute_zero"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_one(sample_dataframe):
    imputer = NumericImputer(strategy="one", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="one").alias("A__impute_one"),
            pl.col("B").fill_null(strategy="one").alias("B__impute_one"),
            pl.col("D").fill_null(strategy="one").alias("D__impute_one"),
        ]
    ).drop(["A", "B", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_forward_no_drop_columns(sample_dataframe):
    imputer = NumericImputer(strategy="forward", drop_columns=False, inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="forward").alias("A__impute_forward"),
            pl.col("B").fill_null(strategy="forward").alias("B__impute_forward"),
            pl.col("D").fill_null(strategy="forward").alias("D__impute_forward"),
        ]
    )

    assert_frame_equal(transformed, expected)


def test_imputer_zero_specific_columns(sample_dataframe):
    imputer = NumericImputer(strategy="zero", subset=["A", "D"], inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="zero").alias("A__impute_zero"),
            pl.col("D").fill_null(strategy="zero").alias("D__impute_zero"),
        ]
    ).drop(["A", "D"])

    assert_frame_equal(transformed, expected)


def test_imputer_mean_inplace_true(sample_dataframe):
    """Test imputer with inplace=True and a built-in strategy."""
    imputer = NumericImputer(strategy="mean", subset=["A", "B"], inplace=True)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    # Original columns should be imputed in place
    expected = sample_dataframe.with_columns(
        [
            pl.col("A").fill_null(strategy="mean"),
            pl.col("B").fill_null(strategy="mean"),
        ]
    )

    assert_frame_equal(transformed, expected)


def test_imputer_constant_default_value(sample_dataframe):
    """constant strategy uses the default value=0.0 when no value is specified."""
    imputer = NumericImputer(strategy="constant")  # value defaults to 0.0
    imputer.fit(sample_dataframe)
    assert imputer.value == 0
    assert imputer._statistics["A"] == 0


def test_transform_without_fit_returns_x_unchanged():
    """transform() before fit() raises NotFittedError."""


def test_transform_without_fit_returns_x_unchanged():
    X = pl.DataFrame({"A": [1.0, None], "B": [2.0, 3.0]})
    imputer = NumericImputer(strategy="mean")
    with pytest.raises(NotFittedError):
        imputer.transform(X)


def test_imputer_constant_inplace_true(sample_dataframe):
    """constant strategy with inplace=True fills nulls in-place (else/inplace branch)."""
    imputer = NumericImputer(strategy="constant", value=0, inplace=True)
    imputer.fit(sample_dataframe)
    result = imputer.transform(sample_dataframe)
    assert result["A"].null_count() == 0
    assert result["B"].null_count() == 0
    assert result["D"].null_count() == 0


@pytest.mark.parametrize("strategy", ["mean", "median", "min", "max"])
def test_all_null_column_statistics_defaults_to_zero(strategy):
    # _statistics fallback to 0 is used by the ONNX exporter; transform uses Polars native for mean/min/max
    X = pl.DataFrame({"A": pl.Series([None, None, None], dtype=pl.Float64)})
    imputer = NumericImputer(strategy=strategy)
    imputer.fit(X)
    assert imputer._statistics["A"] == 0


def test_all_null_column_median_fills_zero():
    # median uses _statistics in transform, so all-null → 0
    X = pl.DataFrame({"A": pl.Series([None, None, None], dtype=pl.Float64)})
    imputer = NumericImputer(strategy="median")
    imputer.fit(X)
    result = imputer.transform(X)
    assert result["A"].to_list() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("strategy", ["mean", "median", "min", "max"])
def test_all_nan_column_defaults_to_zero(strategy):
    X = pl.DataFrame({"A": pl.Series([float("nan"), float("nan")], dtype=pl.Float64)})
    imputer = NumericImputer(strategy=strategy)
    imputer.fit(X)
    assert imputer._statistics["A"] == 0


def test_most_frequent_all_null_defaults_to_zero():
    X = pl.DataFrame({"A": pl.Series([None, None, None], dtype=pl.Float64)})
    imputer = NumericImputer(strategy="most_frequent")
    imputer.fit(X)
    assert imputer._statistics["A"] == 0
    result = imputer.transform(X)
    assert result["A"].to_list() == [0.0, 0.0, 0.0]
