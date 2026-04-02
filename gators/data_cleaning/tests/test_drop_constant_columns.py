"""Tests for DropConstantColumns transformer."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import DropConstantColumns


class TestDropConstantColumns:
    """Test suite for DropConstantColumns."""

    def test_remove_constant_numeric_column(self):
        """Test removing a constant numeric column."""
        X = pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "constant_num": [42, 42, 42, 42, 42],
                "varying": [10, 20, 30, 40, 50],
            }
        )

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        expected = pl.DataFrame(
            {"id": [1, 2, 3, 4, 5], "varying": [10, 20, 30, 40, 50]}
        )

        assert_frame_equal(result, expected)
        assert remover._to_drop == ["constant_num"]

    def test_remove_constant_categorical_column(self):
        """Test removing a constant categorical column."""
        X = pl.DataFrame(
            {
                "country": ["USA", "USA", "USA", "USA"],
                "city": ["NYC", "LA", "Chicago", "Boston"],
                "status": ["active", "active", "active", "active"],
            }
        )

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"city": ["NYC", "LA", "Chicago", "Boston"]})

        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"country", "status"}

    def test_remove_multiple_constant_columns(self):
        """Test removing multiple constant columns at once."""
        X = pl.DataFrame(
            {
                "const1": [1, 1, 1, 1],
                "const2": ["A", "A", "A", "A"],
                "const3": [True, True, True, True],
                "varying": [10, 20, 30, 40],
            }
        )

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": [10, 20, 30, 40]})

        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"const1", "const2", "const3"}

    def test_no_constant_columns(self):
        """Test behavior when no constant columns exist."""
        X = pl.DataFrame(
            {"col1": [1, 2, 3], "col2": ["A", "B", "C"], "col3": [True, False, True]}
        )

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_include_na_true(self):
        """Test with include_na=True (default) - NaN counts as unique value."""
        X = pl.DataFrame(
            {
                "all_null": [None, None, None],
                "mixed": [1, None, 1],
                "varying": [1, 2, 3],
            }
        )

        remover = DropConstantColumns(include_na=True)
        result = remover.fit_transform(X)

        # all_null has 1 unique value (None), so it's constant
        expected = pl.DataFrame({"mixed": [1, None, 1], "varying": [1, 2, 3]})

        assert_frame_equal(result, expected)
        assert remover._to_drop == ["all_null"]

    def test_include_na_false(self):
        """Test with include_na=False - NaN is ignored when counting unique values."""
        X = pl.DataFrame(
            {
                "all_null": [None, None, None],
                "mixed": [1, None, 1],
                "varying": [1, 2, 3],
            }
        )

        remover = DropConstantColumns(include_na=False)
        result = remover.fit_transform(X)

        # all_null has 0 unique values (excluding None), constant
        # mixed has 1 unique value (1, excluding None), constant
        expected = pl.DataFrame({"varying": [1, 2, 3]})

        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"all_null", "mixed"}

    def test_subset_of_columns(self):
        """Test checking only a subset of columns."""
        X = pl.DataFrame({"col1": [1, 1, 1], "col2": [5, 5, 5], "col3": [10, 20, 30]})

        remover = DropConstantColumns(subset=["col1", "col2"])
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"col3": [10, 20, 30]})

        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"col1", "col2"}

    def test_subset_columns_mixed(self):
        """Test subset with mix of constant and varying columns."""
        X = pl.DataFrame(
            {
                "const1": [1, 1, 1],
                "varying1": [1, 2, 3],
                "const2": ["A", "A", "A"],
                "varying2": ["X", "Y", "Z"],
            }
        )

        remover = DropConstantColumns(subset=["const1", "varying1", "const2"])
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying1": [1, 2, 3], "varying2": ["X", "Y", "Z"]})

        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"const1", "const2"}

    def test_all_columns_constant(self):
        """Test when all columns are constant."""
        X = pl.DataFrame({"col1": [1, 1, 1], "col2": ["A", "A", "A"]})

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        # Result should be empty DataFrame (Polars drops rows when all columns are dropped)
        assert result.shape[1] == 0
        assert set(remover._to_drop) == {"col1", "col2"}

    def test_single_row_dataframe(self):
        """Test with single row - all columns are constant."""
        X = pl.DataFrame({"col1": [1], "col2": ["A"], "col3": [True]})

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        # All columns have only 1 value, so all are constant
        assert result.shape[1] == 0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({"col1": [], "col2": []}).cast(
            {"col1": pl.Int64, "col2": pl.String}
        )

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        # Empty columns have 0 unique values, so they're constant
        assert result.shape[1] == 0

    def test_fit_and_transform_separately(self):
        """Test sklearn-style separate fit and transform."""
        train_X = pl.DataFrame({"const": [1, 1, 1], "varying": [10, 20, 30]})

        test_X = pl.DataFrame({"const": [1, 1], "varying": [40, 50]})

        remover = DropConstantColumns()
        remover.fit(train_X)
        result = remover.transform(test_X)

        expected = pl.DataFrame({"varying": [40, 50]})

        assert_frame_equal(result, expected)

    def test_transform_consistency(self):
        """Test that transform on new data uses fitted columns."""
        train_X = pl.DataFrame({"const": [1, 1, 1], "varying": [10, 20, 30]})

        # Test data where 'const' is NOT constant anymore
        test_X = pl.DataFrame(
            {"const": [1, 2, 3], "varying": [40, 50, 60]}  # Now varying!
        )

        remover = DropConstantColumns()
        remover.fit(train_X)
        result = remover.transform(test_X)

        # Should still drop 'const' because it was constant in training
        expected = pl.DataFrame({"varying": [40, 50, 60]})

        assert_frame_equal(result, expected)

    def test_boolean_constant_column(self):
        """Test removing constant boolean column."""
        X = pl.DataFrame(
            {
                "all_true": [True, True, True],
                "mixed": [True, False, True],
                "all_false": [False, False, False],
            }
        )

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"mixed": [True, False, True]})

        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"all_true", "all_false"}

    def test_float_constant_column(self):
        """Test removing constant float column."""
        X = pl.DataFrame(
            {"const_float": [3.14, 3.14, 3.14], "varying_float": [1.1, 2.2, 3.3]}
        )

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying_float": [1.1, 2.2, 3.3]})

        assert_frame_equal(result, expected)
        assert remover._to_drop == ["const_float"]

    def test_mixed_dtypes(self):
        """Test with mixed data types."""
        X = pl.DataFrame(
            {
                "int_const": [1, 1, 1],
                "str_const": ["A", "A", "A"],
                "float_const": [2.5, 2.5, 2.5],
                "bool_const": [True, True, True],
                "varying": [1, 2, 3],
            }
        )

        remover = DropConstantColumns()
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": [1, 2, 3]})

        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {
            "int_const",
            "str_const",
            "float_const",
            "bool_const",
        }

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"const": [1, 1, 1], "varying": [1, 2, 3]})

        remover = DropConstantColumns()

        # Test fit returns self
        assert remover.fit(X) is remover

        # Test fit_transform
        result = remover.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform produce same result
        remover2 = DropConstantColumns()
        remover2.fit(X)
        result2 = remover2.transform(X)
        assert_frame_equal(result, result2)

    def test_none_columns_parameter(self):
        """Test that subset=None checks all columns."""
        X = pl.DataFrame({"col1": [1, 1, 1], "col2": [2, 2, 2], "col3": [3, 4, 5]})

        remover = DropConstantColumns(subset=None)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"col3": [3, 4, 5]})

        assert_frame_equal(result, expected)

    def test_empty_columns_list(self):
        """Test with empty columns list."""
        X = pl.DataFrame({"const": [1, 1, 1], "varying": [1, 2, 3]})

        remover = DropConstantColumns(subset=[])
        result = remover.fit_transform(X)

        # No columns to check, so nothing should be removed
        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_column_with_only_nulls(self):
        """Test column with only NULL values."""
        X = pl.DataFrame({"only_null": [None, None, None], "varying": [1, 2, 3]})

        # With include_na=True, NULL is a unique value, so it's constant
        remover = DropConstantColumns(include_na=True)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": [1, 2, 3]})

        assert_frame_equal(result, expected)
        assert remover._to_drop == ["only_null"]
