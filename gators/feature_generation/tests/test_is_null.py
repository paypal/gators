import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation.is_null import IsNull


def test_transform_no_nulls():
    """Test transformation when no null values are present."""
    X = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    expected_X = X.with_columns(
        [
            pl.col("col1").is_null().alias("col1__is_null"),
            pl.col("col2").is_null().alias("col2__is_null"),
        ]
    )

    transformer = IsNull(subset=["col1", "col2"])
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


def test_transform_with_nulls():
    """Test transformation when null values are present."""
    X = pl.DataFrame({"col1": [1, None, 3, None], "col2": [4, 5, None, 7], "col3": [7, 8, 9, 10]})
    expected_X = X.with_columns(
        [
            pl.col("col1").is_null().alias("col1__is_null"),
            pl.col("col2").is_null().alias("col2__is_null"),
        ]
    )

    transformer = IsNull(subset=["col1", "col2"])
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


def test_transform_single_column():
    """Test transformation with a single column."""
    X = pl.DataFrame({"col1": [1, None, 3], "col2": [4, 5, 6]})
    expected_X = X.with_columns([pl.col("col1").is_null().alias("col1__is_null")])

    transformer = IsNull(subset=["col1"])
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


def test_transform_all_nulls():
    """Test transformation when all values are null."""
    X = pl.DataFrame({"col1": [None, None, None], "col2": [4, 5, 6]})
    expected_X = X.with_columns([pl.col("col1").is_null().alias("col1__is_null")])

    transformer = IsNull(subset=["col1"])
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


def test_column_mapping():
    """Test that _column_mapping is correctly populated."""
    X = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    transformer = IsNull(subset=["col1", "col2"])
    _ = transformer.fit(X)

    expected_mapping = {"col1": "col1__is_null", "col2": "col2__is_null"}
    assert transformer._column_mapping == expected_mapping


def test_transform_preserves_original_columns():
    """Test that original columns are preserved after transformation."""
    X = pl.DataFrame({"col1": [1, None, 3], "col2": [4, 5, None]})

    transformer = IsNull(subset=["col1", "col2"])
    _ = transformer.fit(X)
    result_X = transformer.transform(X)

    # Check that original columns still exist
    assert "col1" in result_X.columns
    assert "col2" in result_X.columns
    # Check that new columns were added
    assert "col1__is_null" in result_X.columns
    assert "col2__is_null" in result_X.columns


def test_transform_multiple_columns():
    """Test transformation with multiple columns."""
    X = pl.DataFrame(
        {
            "A": [1, None, 3, 4],
            "B": [None, 2, 3, 4],
            "C": [1, 2, None, 4],
            "D": [1, 2, 3, 4],
        }
    )
    expected_X = X.with_columns(
        [
            pl.col("A").is_null().alias("A__is_null"),
            pl.col("B").is_null().alias("B__is_null"),
            pl.col("C").is_null().alias("C__is_null"),
        ]
    )

    transformer = IsNull(subset=["A", "B", "C"])
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


def test_transform_with_none_subset():
    """Test transformation when subset=None (should use all columns)."""
    X = pl.DataFrame({"col1": [1, None, 3], "col2": [4, 5, None], "col3": [7, 8, 9]})
    expected_X = X.with_columns(
        [
            pl.col("col1").is_null().alias("col1__is_null"),
            pl.col("col2").is_null().alias("col2__is_null"),
            pl.col("col3").is_null().alias("col3__is_null"),
        ]
    )

    transformer = IsNull()
    _ = transformer.fit(X)
    result_X = transformer.transform(X)
    assert_frame_equal(result_X, expected_X, check_column_order=False)


if __name__ == "__main__":
    pytest.main()
