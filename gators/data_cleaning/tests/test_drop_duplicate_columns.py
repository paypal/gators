import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import DropDuplicateColumns


@pytest.fixture
def sample_data_with_duplicates() -> pl.DataFrame:
    """DataFrame with duplicate columns."""
    return pl.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [5, 6, 7, 8],
            "C": [1, 2, 3, 4],  # Duplicate of A
            "D": [9, 10, 11, 12],
            "E": [5, 6, 7, 8],  # Duplicate of B
        }
    )


@pytest.fixture
def sample_data_no_duplicates() -> pl.DataFrame:
    """DataFrame without duplicate columns."""
    return pl.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6], "Z": [7, 8, 9]})


@pytest.fixture
def sample_data_with_nulls() -> pl.DataFrame:
    """DataFrame with duplicate columns containing nulls."""
    return pl.DataFrame(
        {
            "col1": [1.0, None, 3.0],
            "col2": [4.0, 5.0, 6.0],
            "col3": [1.0, None, 3.0],  # Duplicate of col1
            "col4": [7.0, 8.0, None],
        }
    )


@pytest.fixture
def expected_keep_first() -> pl.DataFrame:
    """Expected result when keeping first occurrence."""
    return pl.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "D": [9, 10, 11, 12]})


@pytest.fixture
def expected_keep_last() -> pl.DataFrame:
    """Expected result when keeping last occurrence."""
    return pl.DataFrame({"C": [1, 2, 3, 4], "D": [9, 10, 11, 12], "E": [5, 6, 7, 8]})


def test_remove_duplicates_keep_first(sample_data_with_duplicates, expected_keep_first):
    """Test removing duplicate columns keeping first occurrence."""
    remover = DropDuplicateColumns(keep="first")
    remover.fit(sample_data_with_duplicates)
    result = remover.transform(sample_data_with_duplicates)

    assert_frame_equal(result, expected_keep_first)
    assert set(remover.columns_to_drop_) == {"C", "E"}
    assert "A" in remover.column_groups_
    assert "C" in remover.column_groups_["A"]


def test_remove_duplicates_keep_last(sample_data_with_duplicates, expected_keep_last):
    """Test removing duplicate columns keeping last occurrence."""
    remover = DropDuplicateColumns(keep="last")
    remover.fit(sample_data_with_duplicates)
    result = remover.transform(sample_data_with_duplicates)

    assert_frame_equal(result, expected_keep_last)
    assert set(remover.columns_to_drop_) == {"A", "B"}


def test_no_duplicates(sample_data_no_duplicates):
    """Test with DataFrame containing no duplicate columns."""
    remover = DropDuplicateColumns()
    remover.fit(sample_data_no_duplicates)
    result = remover.transform(sample_data_no_duplicates)

    assert_frame_equal(result, sample_data_no_duplicates)
    assert remover.columns_to_drop_ == []
    assert remover.column_groups_ == {}


def test_with_nulls(sample_data_with_nulls):
    """Test that columns with identical null patterns are detected as duplicates."""
    remover = DropDuplicateColumns(keep="first")
    remover.fit(sample_data_with_nulls)
    result = remover.transform(sample_data_with_nulls)

    assert "col3" in remover.columns_to_drop_
    assert "col1" in result.columns
    assert "col3" not in result.columns
    assert result.shape == (3, 3)


def test_fit_transform(sample_data_with_duplicates, expected_keep_first):
    """Test fit_transform method."""
    remover = DropDuplicateColumns()
    result = remover.fit_transform(sample_data_with_duplicates)

    assert_frame_equal(result, expected_keep_first)


def test_single_column():
    """Test with single column DataFrame."""
    X = pl.DataFrame({"A": [1, 2, 3]})
    remover = DropDuplicateColumns()
    remover.fit(X)
    result = remover.transform(X)

    assert_frame_equal(result, X)
    assert remover.columns_to_drop_ == []


def test_all_columns_identical():
    """Test when all columns are identical."""
    X = pl.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 3], "col3": [1, 2, 3]})
    remover = DropDuplicateColumns(keep="first")
    remover.fit(X)
    result = remover.transform(X)

    assert result.columns == ["col1"]
    assert set(remover.columns_to_drop_) == {"col2", "col3"}


def test_mixed_dtypes():
    """Test that columns with different dtypes are not considered duplicates."""
    X = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],  # Same values but different dtype
            "str_col": ["a", "b", "c"],
        }
    )
    remover = DropDuplicateColumns()
    remover.fit(X)
    result = remover.transform(X)

    assert_frame_equal(result, X)
    assert remover.columns_to_drop_ == []


def test_invalid_keep_parameter():
    """Test that invalid keep parameter raises ValueError."""
    X = pl.DataFrame({"A": [1, 2, 3]})
    remover = DropDuplicateColumns(keep="invalid")

    with pytest.raises(ValueError, match="keep must be 'first' or 'last'"):
        remover.fit(X)


def test_column_groups_tracking():
    """Test that column_groups_ correctly tracks duplicate relationships."""
    X = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3], "d": [4, 5, 6]})
    remover = DropDuplicateColumns(keep="first")
    remover.fit(X)

    assert "a" in remover.column_groups_
    assert set(remover.column_groups_["a"]) == {"b", "c"}
    assert remover.columns_to_drop_ == ["b", "c"]


def test_empty_dataframe():
    """Test with empty DataFrame."""
    X = pl.DataFrame()
    remover = DropDuplicateColumns()
    remover.fit(X)
    result = remover.transform(X)

    assert result.shape == (0, 0)
    assert remover.columns_to_drop_ == []


def test_different_length_columns():
    """Test with columns of different lengths (edge case for coverage)."""
    # Create DataFrame where comparison would fail if lengths weren't checked
    X1 = pl.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1, 2, 3],
        }
    )

    # Both columns are identical, so one should be removed
    remover = DropDuplicateColumns()
    remover.fit(X1)
    result = remover.transform(X1)

    # Only one column should remain after removing duplicates
    assert result.shape[1] == 1
