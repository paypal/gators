"""Tests for HighCardinalityFilter transformer."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import HighCardinalityFilter


class TestHighCardinalityFilter:
    """Test suite for HighCardinalityFilter."""

    def test_max_unique_absolute_threshold(self):
        """Test removing columns by absolute unique count."""
        X = pl.DataFrame(
            {
                "id": range(1000),  # 1000 unique
                "category": ["A", "B", "C"] * 333 + ["A"],  # 3 unique
                "subcategory": ["X", "Y"] * 500,  # 2 unique
            }
        )

        filter = HighCardinalityFilter(max_unique=100)
        result = filter.fit_transform(X)

        # id should be removed (1000 > 100)
        assert "id" not in result.columns
        assert "category" in result.columns
        assert "subcategory" in result.columns
        assert result.shape[0] == 1000

    def test_max_ratio_threshold(self):
        """Test removing columns by ratio of unique values."""
        X = pl.DataFrame(
            {
                "col1": range(100),  # 100 unique, ratio=1.0
                "col2": list(range(50)) * 2,  # 50 unique, ratio=0.5
                "col3": ["A", "B"] * 50,  # 2 unique, ratio=0.02
            }
        )

        filter = HighCardinalityFilter(max_ratio=0.8)
        result = filter.fit_transform(X)

        # col1 should be removed (ratio 1.0 > 0.8)
        assert "col1" not in result.columns
        assert "col2" in result.columns
        assert "col3" in result.columns

    def test_combined_thresholds_or_logic(self):
        """Test that either threshold triggers removal (OR logic)."""
        X = pl.DataFrame(
            {
                "col1": range(50),  # 50 unique, ratio=1.0
                "col2": list(range(25)) * 2,  # 25 unique, ratio=0.5
                "col3": ["A", "B"] * 25,  # 2 unique, ratio=0.04
            }
        )

        # Remove if >30 unique OR >0.8 ratio
        filter = HighCardinalityFilter(max_unique=30, max_ratio=0.8)
        result = filter.fit_transform(X)

        # col1 removed by both thresholds (50 > 30 AND 1.0 > 0.8)
        # col2 stays (25 <= 30 AND 0.5 <= 0.8)
        # col3 stays
        assert "col1" not in result.columns
        assert "col2" in result.columns
        assert "col3" in result.columns

    def test_ignore_na_true(self):
        """Test ignoring NaN when counting unique values (default)."""
        X = pl.DataFrame(
            {
                "col1": [1, 2, 3, None, None] * 20,  # 3 unique + NaN
                "col2": list(range(90)) + [None] * 10,  # 90 unique + NaN
            }
        )

        filter = HighCardinalityFilter(max_unique=50, ignore_na=True)
        result = filter.fit_transform(X)

        # col1: 3 unique (ignoring NaN) <= 50 → kept
        # col2: 90 unique (ignoring NaN) > 50 → removed
        assert "col1" in result.columns
        assert "col2" not in result.columns

    def test_ignore_na_false(self):
        """Test counting NaN as a unique value."""
        X = pl.DataFrame(
            {
                "col1": [1, 2, 3, None, None] * 10,  # 4 unique including NaN
                "col2": list(range(45)) + [None] * 5,  # 46 unique including NaN
            }
        )

        filter = HighCardinalityFilter(max_unique=45, ignore_na=False)
        result = filter.fit_transform(X)

        # col1: 4 unique (with NaN) <= 45 → kept
        # col2: 46 unique (with NaN) > 45 → removed
        assert "col1" in result.columns
        assert "col2" not in result.columns

    def test_subset_columns(self):
        """Test filtering only specified columns."""
        X = pl.DataFrame(
            {"id1": range(100), "id2": range(100), "feature": ["A", "B"] * 50}
        )

        # Only check id1 and id2
        filter = HighCardinalityFilter(subset=["id1", "id2"], max_unique=50)
        result = filter.fit_transform(X)

        # id1 and id2 should be removed, feature kept
        assert "id1" not in result.columns
        assert "id2" not in result.columns
        assert "feature" in result.columns

    def test_no_columns_removed(self):
        """Test when no columns exceed thresholds."""
        X = pl.DataFrame(
            {
                "col1": [1, 2, 3] * 10,
                "col2": ["A", "B", "C", "D"] * 7 + ["A", "A"],
                "col3": ["X", "Y"] * 15,
            }
        )

        filter = HighCardinalityFilter(max_unique=100)
        result = filter.fit_transform(X)

        # All columns should be kept
        assert_frame_equal(result, X)

    def test_all_columns_removed(self):
        """Test when all columns exceed thresholds."""
        X = pl.DataFrame(
            {"col1": range(100), "col2": range(100, 200), "col3": range(200, 300)}
        )

        filter = HighCardinalityFilter(max_unique=50)
        result = filter.fit_transform(X)

        # All columns removed, Polars creates empty DataFrame with 0 rows
        assert result.shape[1] == 0
        assert result.shape[0] == 0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({"col1": [], "col2": []}).cast(
            {"col1": pl.Int64, "col2": pl.String}
        )

        filter = HighCardinalityFilter(max_unique=10)
        result = filter.fit_transform(X)

        # Empty DataFrame should be handled gracefully
        assert result.shape == (0, 2)

    def test_single_row_dataframe(self):
        """Test with single row."""
        X = pl.DataFrame({"col1": [1], "col2": ["A"], "col3": [True]})

        filter = HighCardinalityFilter(max_ratio=0.5)
        result = filter.fit_transform(X)

        # All columns have ratio=1.0 (1 unique / 1 row) > 0.5
        assert result.shape[1] == 0

    def test_all_same_value(self):
        """Test columns with all same value (1 unique)."""
        X = pl.DataFrame(
            {"col1": [1] * 100, "col2": ["A"] * 100, "col3": [True] * 100}
        )

        filter = HighCardinalityFilter(max_unique=10)
        result = filter.fit_transform(X)

        # All columns should be kept (1 unique <= 10)
        assert_frame_equal(result, X)

    def test_exact_threshold_values(self):
        """Test behavior at exact threshold boundaries."""
        X = pl.DataFrame(
            {
                "col1": list(range(50)) + [0] * 51,  # Exactly 50 unique in 101 rows
                "col2": list(range(51)) + [0] * 50,  # 51 unique in 101 rows
                "col3": list(range(49)) + [0] * 52,  # 49 unique in 101 rows
            }
        )

        filter = HighCardinalityFilter(max_unique=50)
        result = filter.fit_transform(X)

        # col1: 50 <= 50 → kept
        # col2: 51 > 50 → removed
        # col3: 49 <= 50 → kept
        assert "col1" in result.columns
        assert "col2" not in result.columns
        assert "col3" in result.columns

    def test_ratio_exact_boundary(self):
        """Test ratio at exact boundary."""
        X = pl.DataFrame(
            {
                "col1": list(range(80))
                + [0] * 20,  # 80 unique in 100 rows, ratio = 0.8
                "col2": list(range(81))
                + [0] * 19,  # 81 unique in 100 rows, ratio = 0.81
                "col3": list(range(79))
                + [0] * 21,  # 79 unique in 100 rows, ratio = 0.79
            }
        )

        filter = HighCardinalityFilter(max_ratio=0.8)
        result = filter.fit_transform(X)

        # col1: 0.8 <= 0.8 → kept
        # col2: 0.81 > 0.8 → removed
        # col3: 0.79 <= 0.8 → kept
        assert "col1" in result.columns
        assert "col2" not in result.columns
        assert "col3" in result.columns

    def test_mixed_dtypes(self):
        """Test with mixed data types."""
        X = pl.DataFrame(
            {
                "int_col": range(100),
                "str_col": [f"str_{i}" for i in range(100)],
                "float_col": [i * 0.1 for i in range(100)],
                "bool_col": [True, False] * 50,
                "cat_col": ["A", "B", "C"] * 33 + ["A"],
            }
        )

        filter = HighCardinalityFilter(max_unique=10)
        result = filter.fit_transform(X)

        # Only bool_col (2 unique) and cat_col (3 unique) should remain
        assert "int_col" not in result.columns
        assert "str_col" not in result.columns
        assert "float_col" not in result.columns
        assert "bool_col" in result.columns
        assert "cat_col" in result.columns

    def test_validation_no_thresholds(self):
        """Test validation error when no thresholds provided."""
        with pytest.raises(ValueError, match="At least one of max_unique or max_ratio"):
            HighCardinalityFilter()

    def test_validation_max_unique_negative(self):
        """Test validation error for invalid max_unique."""
        with pytest.raises(ValueError, match="max_unique must be >= 1"):
            HighCardinalityFilter(max_unique=0)

        with pytest.raises(ValueError, match="max_unique must be >= 1"):
            HighCardinalityFilter(max_unique=-5)

    def test_validation_max_ratio_invalid(self):
        """Test validation error for invalid max_ratio."""
        with pytest.raises(ValueError, match="max_ratio must be between 0 and 1"):
            HighCardinalityFilter(max_ratio=-0.1)

        with pytest.raises(ValueError, match="max_ratio must be between 0 and 1"):
            HighCardinalityFilter(max_ratio=1.5)

    def test_fit_and_transform_separately(self):
        """Test sklearn-style separate fit and transform."""
        train_X = pl.DataFrame({"col1": range(100), "col2": ["A", "B"] * 50})

        test_X = pl.DataFrame({"col1": range(100, 150), "col2": ["A", "B"] * 25})

        filter = HighCardinalityFilter(max_unique=50)
        filter.fit(train_X)
        result = filter.transform(test_X)

        # col1 was identified as high cardinality in fit
        assert "col1" not in result.columns
        assert "col2" in result.columns

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"col1": range(100), "col2": ["A", "B"] * 50})

        filter = HighCardinalityFilter(max_unique=50)

        # Test fit returns self
        assert filter.fit(X) is filter

        # Test fit_transform
        result = filter.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform produce same result
        filter2 = HighCardinalityFilter(max_unique=50)
        filter2.fit(X)
        result2 = filter2.transform(X)
        assert_frame_equal(result, result2)

    def test_to_drop_attribute(self):
        """Test that _to_drop attribute is correctly populated."""
        X = pl.DataFrame(
            {"col1": range(100), "col2": range(100, 200), "col3": ["A", "B"] * 50}
        )

        filter = HighCardinalityFilter(max_unique=50)
        filter.fit(X)

        assert "col1" in filter._to_drop
        assert "col2" in filter._to_drop
        assert "col3" not in filter._to_drop

    def test_preserve_column_order(self):
        """Test that remaining columns preserve their order."""
        X = pl.DataFrame(
            {
                "col_a": ["X"] * 100,
                "col_b": range(100),  # To be removed
                "col_c": ["Y", "Z"] * 50,
                "col_d": range(100, 200),  # To be removed
                "col_e": ([1, 2, 3] * 33) + [1],
            }
        )

        filter = HighCardinalityFilter(max_unique=50)
        result = filter.fit_transform(X)

        assert result.columns == ["col_a", "col_c", "col_e"]

    def test_only_max_unique(self):
        """Test using only max_unique threshold."""
        X = pl.DataFrame({"col1": range(100), "col2": list(range(30)) + [0] * 70})

        filter = HighCardinalityFilter(max_unique=50)
        result = filter.fit_transform(X)

        assert "col1" not in result.columns
        assert "col2" in result.columns

    def test_only_max_ratio(self):
        """Test using only max_ratio threshold."""
        X = pl.DataFrame(
            {
                "col1": list(range(90))
                + [0] * 10,  # 90 unique in 100 rows, ratio = 0.9
                "col2": list(range(60))
                + [0] * 40,  # 60 unique in 100 rows, ratio = 0.6
            }
        )

        filter = HighCardinalityFilter(max_ratio=0.7)
        result = filter.fit_transform(X)

        assert "col1" not in result.columns
        assert "col2" in result.columns

    def test_multiple_na_handling(self):
        """Test with multiple NaN patterns."""
        X = pl.DataFrame(
            {
                "col1": [1, 2, None, None, None] * 10,  # 2 unique + NaN
                "col2": [None] * 50,  # Only NaN
                "col3": [1, 2, 3, 4, 5] + [None] * 45,  # 5 unique + NaN
            }
        )

        filter = HighCardinalityFilter(max_unique=3, ignore_na=True)
        result = filter.fit_transform(X)

        # col1: 2 unique <= 3 → kept
        # col2: 0 unique (only NaN) <= 3 → kept
        # col3: 5 unique > 3 → removed
        assert "col1" in result.columns
        assert "col2" in result.columns
        assert "col3" not in result.columns

    def test_default_ignore_na(self):
        """Test that ignore_na defaults to True."""
        X = pl.DataFrame({"col1": [1, 2, 3, None, None] * 10})

        filter = HighCardinalityFilter(max_unique=5)
        filter.fit(X)

        # With ignore_na=True (default), only 3 unique values
        assert "col1" not in filter._to_drop

    def test_large_cardinality(self):
        """Test with very high cardinality."""
        X = pl.DataFrame(
            {"id": range(10000), "category": ["A", "B", "C", "D", "E"] * 2000}
        )

        filter = HighCardinalityFilter(max_unique=1000)
        result = filter.fit_transform(X)

        assert "id" not in result.columns
        assert "category" in result.columns
        assert result.shape[0] == 10000

    def test_ratio_with_small_dataset(self):
        """Test ratio calculation with small dataset."""
        X = pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],  # ratio = 5/5 = 1.0
                "col2": [1, 1, 2, 2, 3],  # ratio = 3/5 = 0.6
            }
        )

        filter = HighCardinalityFilter(max_ratio=0.8)
        result = filter.fit_transform(X)

        assert "col1" not in result.columns
        assert "col2" in result.columns

    def test_subset_with_nonexistent_columns(self):
        """Test behavior when subset includes non-existent columns."""
        X = pl.DataFrame({"col1": range(100), "col2": ["A", "B"] * 50})

        # Polars will raise error when trying to access non-existent column
        filter = HighCardinalityFilter(subset=["col1", "col3"], max_unique=50)

        with pytest.raises(Exception):  # Could be KeyError or ColumnNotFoundError
            filter.fit(X)
