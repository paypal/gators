"""Tests for DropDuplicateRows transformer."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import DropDuplicateRows


class TestDropDuplicateRows:
    """Test suite for DropDuplicateRows."""

    def test_remove_full_duplicates_keep_first(self):
        """Test removing full duplicate rows keeping first occurrence."""
        X = pl.DataFrame(
            {
                "id": [1, 2, 2, 3, 4, 4],
                "name": ["Alice", "Bob", "Bob", "Charlie", "David", "David"],
                "age": [25, 30, 30, 35, 40, 40],
            }
        )

        remover = DropDuplicateRows(keep="first")
        result = remover.fit_transform(X)

        expected = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
            }
        )

        assert_frame_equal(result, expected)

    def test_remove_full_duplicates_keep_last(self):
        """Test removing full duplicate rows keeping last occurrence."""
        X = pl.DataFrame({"id": [1, 2, 2, 3, 3, 3], "value": [10, 20, 20, 30, 30, 30]})

        remover = DropDuplicateRows(keep="last")
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        assert_frame_equal(result, expected)

    def test_remove_duplicates_keep_none(self):
        """Test removing all duplicate rows (keep none)."""
        X = pl.DataFrame(
            {
                "user_id": [1, 2, 2, 3, 4, 4, 5],
                "action": ["login", "view", "view", "click", "buy", "buy", "logout"],
            }
        )

        remover = DropDuplicateRows(keep="none")
        result = remover.fit_transform(X)

        expected = pl.DataFrame(
            {"user_id": [1, 3, 5], "action": ["login", "click", "logout"]}
        )

        assert_frame_equal(result, expected)

    def test_subset_columns_keep_first(self):
        """Test removing duplicates based on subset of columns keeping first."""
        X = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Bob", "Alice", "Bob"],
                "score": [85, 90, 88, 92],
            }
        )

        remover = DropDuplicateRows(subset=["name"], keep="first")
        result = remover.fit_transform(X)

        expected = pl.DataFrame(
            {"id": [1, 2], "name": ["Alice", "Bob"], "score": [85, 90]}
        )

        assert_frame_equal(result, expected)

    def test_subset_columns_keep_last(self):
        """Test removing duplicates based on subset of columns keeping last."""
        X = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Bob", "Alice", "Bob"],
                "score": [85, 90, 88, 92],
            }
        )

        remover = DropDuplicateRows(subset=["name"], keep="last")
        result = remover.fit_transform(X)

        expected = pl.DataFrame(
            {"id": [3, 4], "name": ["Alice", "Bob"], "score": [88, 92]}
        )

        assert_frame_equal(result, expected)

    def test_subset_columns_keep_none(self):
        """Test removing all duplicates based on subset (keep none)."""
        X = pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "C", "B"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        remover = DropDuplicateRows(subset=["category"], keep="none")
        result = remover.fit_transform(X)

        # A and B appear twice, so only C (id=4) remains
        expected = pl.DataFrame({"id": [4], "category": ["C"], "value": [40]})

        assert_frame_equal(result, expected)

    def test_no_duplicates(self):
        """Test behavior when DataFrame has no duplicates."""
        X = pl.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        )

        remover = DropDuplicateRows()
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)

    def test_all_duplicates(self):
        """Test when all rows are duplicates."""
        X = pl.DataFrame({"id": [1, 1, 1, 1], "value": [10, 10, 10, 10]})

        remover = DropDuplicateRows(keep="first")
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"id": [1], "value": [10]})

        assert_frame_equal(result, expected)

    def test_all_duplicates_keep_none(self):
        """Test when all rows are duplicates and keep='none'."""
        X = pl.DataFrame({"id": [1, 1, 1], "value": [10, 10, 10]})

        remover = DropDuplicateRows(keep="none")
        result = remover.fit_transform(X)

        # All rows are duplicates, so result is empty
        assert result.shape[0] == 0
        assert result.shape[1] == 2  # Columns are preserved

    def test_multiple_subset_columns(self):
        """Test with multiple columns in subset."""
        X = pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", "A", "B", "A", "B"],
                "subcategory": ["X", "Y", "X", "X", "Y"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        remover = DropDuplicateRows(subset=["category", "subcategory"], keep="first")
        result = remover.fit_transform(X)

        # Duplicates: (A,X) at rows 0,3 - keep 0
        #             (B,Y) at row 4 (no dup)
        expected = pl.DataFrame(
            {
                "id": [1, 2, 3, 5],
                "category": ["A", "A", "B", "B"],
                "subcategory": ["X", "Y", "X", "Y"],
                "value": [10, 20, 30, 50],
            }
        )

        assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({"col1": [], "col2": []}).cast(
            {"col1": pl.Int64, "col2": pl.String}
        )

        remover = DropDuplicateRows()
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)

    def test_single_row(self):
        """Test with single row - no duplicates possible."""
        X = pl.DataFrame({"id": [1], "name": ["Alice"]})

        remover = DropDuplicateRows()
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)

    def test_maintain_order(self):
        """Test that row order is maintained."""
        X = pl.DataFrame(
            {"id": [5, 1, 1, 3, 2, 2], "value": ["E", "A", "A", "C", "B", "B"]}
        )

        remover = DropDuplicateRows(keep="first")
        result = remover.fit_transform(X)

        # Order should be preserved
        expected = pl.DataFrame({"id": [5, 1, 3, 2], "value": ["E", "A", "C", "B"]})

        assert_frame_equal(result, expected)

    def test_mixed_dtypes(self):
        """Test with mixed data types."""
        X = pl.DataFrame(
            {
                "int_col": [1, 2, 1, 3],
                "str_col": ["A", "B", "A", "C"],
                "float_col": [1.1, 2.2, 1.1, 3.3],
                "bool_col": [True, False, True, False],
            }
        )

        remover = DropDuplicateRows(keep="first")
        result = remover.fit_transform(X)

        expected = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "str_col": ["A", "B", "C"],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, False],
            }
        )

        assert_frame_equal(result, expected)

    def test_null_values_in_data(self):
        """Test handling of NULL values."""
        X = pl.DataFrame(
            {"id": [1, 2, None, None, 3], "value": ["A", None, "B", "B", "C"]}
        )

        remover = DropDuplicateRows(keep="first")
        result = remover.fit_transform(X)

        # (None, 'B') appears twice, keep first
        expected = pl.DataFrame({"id": [1, 2, None, 3], "value": ["A", None, "B", "C"]})

        assert_frame_equal(result, expected)

    def test_validation_missing_subset_columns(self):
        """Test validation error when subset columns don't exist."""
        X = pl.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})

        remover = DropDuplicateRows(subset=["col1", "col3"])

        with pytest.raises(ValueError, match="not found in DataFrame"):
            remover.fit(X)

    def test_fit_and_transform_separately(self):
        """Test sklearn-style separate fit and transform."""
        train_X = pl.DataFrame({"id": [1, 2, 2, 3], "value": [10, 20, 20, 30]})

        test_X = pl.DataFrame({"id": [4, 5, 5], "value": [40, 50, 50]})

        remover = DropDuplicateRows(keep="first")
        remover.fit(train_X)
        result = remover.transform(test_X)

        expected = pl.DataFrame({"id": [4, 5], "value": [40, 50]})

        assert_frame_equal(result, expected)

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"id": [1, 1, 2], "value": [10, 10, 20]})

        remover = DropDuplicateRows()

        # Test fit returns self
        assert remover.fit(X) is remover

        # Test fit_transform
        result = remover.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform produce same result
        remover2 = DropDuplicateRows()
        remover2.fit(X)
        result2 = remover2.transform(X)
        assert_frame_equal(result, result2)

    def test_default_parameters(self):
        """Test default parameters (subset=None, keep='first')."""
        X = pl.DataFrame({"col1": [1, 1, 2], "col2": ["A", "A", "B"]})

        remover = DropDuplicateRows()
        result = remover.fit_transform(X)

        # Default is keep='first', subset=None
        expected = pl.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})

        assert_frame_equal(result, expected)
        assert remover.subset is None
        assert remover.keep == "first"

    def test_subset_single_column(self):
        """Test with single column in subset."""
        X = pl.DataFrame(
            {"id": [1, 2, 3], "category": ["A", "A", "B"], "value": [10, 20, 30]}
        )

        remover = DropDuplicateRows(subset=["category"], keep="first")
        result = remover.fit_transform(X)

        expected = pl.DataFrame(
            {"id": [1, 3], "category": ["A", "B"], "value": [10, 30]}
        )

        assert_frame_equal(result, expected)

    def test_consecutive_duplicates(self):
        """Test with consecutive duplicate rows."""
        X = pl.DataFrame(
            {"id": [1, 1, 1, 2, 2, 3], "value": ["A", "A", "A", "B", "B", "C"]}
        )

        remover = DropDuplicateRows(keep="first")
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"id": [1, 2, 3], "value": ["A", "B", "C"]})

        assert_frame_equal(result, expected)

    def test_non_consecutive_duplicates(self):
        """Test with non-consecutive duplicate rows."""
        X = pl.DataFrame(
            {"id": [1, 2, 1, 3, 2, 1], "value": ["A", "B", "A", "C", "B", "A"]}
        )

        remover = DropDuplicateRows(keep="first")
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"id": [1, 2, 3], "value": ["A", "B", "C"]})

        assert_frame_equal(result, expected)

    def test_partial_duplicates_with_subset(self):
        """Test rows that are duplicates only on subset columns."""
        X = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "category": ["A", "A", "B"],
                "value": [10, 20, 30],  # Different values
            }
        )

        remover = DropDuplicateRows(subset=["category"], keep="last")
        result = remover.fit_transform(X)

        # Category A appears in rows 0 and 1, keep last (row 1)
        expected = pl.DataFrame(
            {"id": [2, 3], "category": ["A", "B"], "value": [20, 30]}
        )

        assert_frame_equal(result, expected)
