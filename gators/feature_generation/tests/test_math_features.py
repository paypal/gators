import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import MathFeatures


def test_transform_default():

    X = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    expected_X = X.with_columns(
        [
            (pl.col("col1") + pl.col("col2")).alias("col1_col2_sum"),
            (pl.col("col1") + pl.col("col2") + pl.col("col3")).alias(
                "col1_col2_col3_sum"
            ),
            ((pl.col("col1") + pl.col("col2")) / 2.0).alias("col1_col2_mean"),
            pl.min_horizontal(pl.col("col1"), pl.col("col2")).alias("col1_col2_min"),
            ((pl.col("col1") + pl.col("col2") + pl.col("col3")) / 3.0).alias(
                "col1_col2_col3_mean"
            ),
            pl.min_horizontal(pl.col("col1"), pl.col("col2"), pl.col("col3")).alias(
                "col1_col2_col3_min"
            ),
        ]
    )

    transformer = MathFeatures(
        groups=[["col1", "col2"], ["col1", "col2", "col3"]],
        operations=["sum", "mean", "min"],
    )
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


def test_transform_with_new_column_names():

    X = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    expected_X = X.with_columns(
        [
            (pl.col("col1") + pl.col("col2")).alias("12_sum"),
            (pl.col("col1") + pl.col("col2") + pl.col("col3")).alias("123_sum"),
            ((pl.col("col1") + pl.col("col2")) / 2.0).alias("12_mean"),
            pl.min_horizontal(pl.col("col1"), pl.col("col2")).alias("12_min"),
            ((pl.col("col1") + pl.col("col2") + pl.col("col3")) / 3.0).alias(
                "123_mean"
            ),
            pl.min_horizontal(pl.col("col1"), pl.col("col2"), pl.col("col3")).alias(
                "123_min"
            ),
        ]
    )

    transformer = MathFeatures(
        groups=[["col1", "col2"], ["col1", "col2", "col3"]],
        operations=["sum", "mean", "min"],
        new_column_names=["12", "123"],
    )
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


def test_transform_with_drop_columns():

    X = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})

    expected_X = X.with_columns(
        [
            (pl.col("col1") + pl.col("col2")).alias("col1_col2_sum"),
            (pl.col("col1") + pl.col("col2") + pl.col("col3")).alias(
                "col1_col2_col3_sum"
            ),
            ((pl.col("col1") + pl.col("col2")) / 2.0).alias("col1_col2_mean"),
            pl.min_horizontal(pl.col("col1"), pl.col("col2")).alias("col1_col2_min"),
            ((pl.col("col1") + pl.col("col2") + pl.col("col3")) / 3.0).alias(
                "col1_col2_col3_mean"
            ),
            pl.min_horizontal(pl.col("col1"), pl.col("col2"), pl.col("col3")).alias(
                "col1_col2_col3_min"
            ),
        ]
    ).drop(["col1", "col2", "col3"])
    transformer = MathFeatures(
        groups=[["col1", "col2"], ["col1", "col2", "col3"]],
        operations=["sum", "mean", "min"],
        drop_columns=True,
    )
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


def test_check_operators_valid():
    # Test with valid operations
    valid_operations = ["sum", "mean", "mul", "min", "max"]
    transformer = MathFeatures(groups=[["A", "B"]], operations=valid_operations)
    assert transformer.operations == valid_operations


def test_check_operators_invalid():
    # Test with an invalid operation
    invalid_operations = ["invalid_op"]
    with pytest.raises(
        ValidationError,
        match="invalid_op is not in the predefined list of datetime functions.",
    ):
        MathFeatures(groups=[["A", "B"]], operations=invalid_operations)


def test_check_operators_mixed():
    # Test with a mix of valid and invalid operations
    mixed_operations = ["sum", "invalid_op"]
    with pytest.raises(
        ValidationError,
        match="invalid_op is not in the predefined list of datetime functions.",
    ):
        MathFeatures(groups=[["A", "B"]], operations=mixed_operations)


if __name__ == "__main__":
    pytest.main()
