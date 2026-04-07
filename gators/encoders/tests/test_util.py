"""Tests for encoders util module."""

import polars as pl
import pytest

from gators.encoders.util import determine_encoding_strategy


class TestDetermineEncodingStrategy:
    """Test suite for determine_encoding_strategy function."""

    def test_basic_separation(self):
        """Test basic separation of low and high cardinality columns."""
        X = pl.DataFrame(
            {
                "low_card": ["A", "B", "A", "B", "A"],  # 2 unique
                "medium_card": ["X", "Y", "Z", "X", "Y"],  # 3 unique
                "high_card": [f"val_{i}" for i in range(5)],  # 5 unique, all different
                "numeric": [1, 2, 3, 4, 5],  # Not a string column
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=3)

        # Low and medium cardinality should use WOE (≤ 3 unique values)
        assert set(woe_cols) == {"low_card", "medium_card"}
        # High cardinality should use one-hot (> 3 unique values)
        assert set(onehot_cols) == {"high_card"}

    def test_all_low_cardinality(self):
        """Test when all categorical columns have low cardinality."""
        X = pl.DataFrame(
            {
                "col1": ["A", "B", "A"],
                "col2": ["X", "Y", "X"],
                "col3": ["P", "Q", "P"],
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=10)

        assert set(woe_cols) == {"col1", "col2", "col3"}
        assert onehot_cols == []

    def test_all_high_cardinality(self):
        """Test when all categorical columns have high cardinality."""
        X = pl.DataFrame(
            {
                "col1": [f"A{i}" for i in range(50)],
                "col2": [f"B{i}" for i in range(50)],
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=10)

        assert woe_cols == []
        assert set(onehot_cols) == {"col1", "col2"}

    def test_no_string_columns(self):
        """Test with no string columns."""
        X = pl.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X)

        assert woe_cols == []
        assert onehot_cols == []

    def test_mixed_types(self):
        """Test with mixed column types."""
        X = pl.DataFrame(
            {
                "string_low": ["A", "B", "A", "B"],
                "string_high": [f"val_{i}" for i in range(4)],
                "int_col": [1, 2, 3, 4],
                "bool_col": [True, False, True, False],
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=2)

        # Only string columns should be considered
        assert "string_low" in woe_cols
        assert "string_high" in onehot_cols
        assert "int_col" not in woe_cols and "int_col" not in onehot_cols
        assert "bool_col" not in woe_cols and "bool_col" not in onehot_cols

    def test_threshold_boundary(self):
        """Test exact threshold boundary."""
        X = pl.DataFrame(
            {
                "exactly_threshold": ["A", "B", "C", "D", "E"],  # 5 unique
                "below_threshold": ["A", "B", "C", "D", "A"],  # 4 unique
                "above_threshold": ["val_0", "val_1", "val_2", "val_3", "val_4", "val_5"][
                    :5
                ],  # 5 unique (first 5)
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=5)

        # Exactly at threshold should use WOE (≤)
        assert "exactly_threshold" in woe_cols
        assert "below_threshold" in woe_cols
        # above_threshold also has 5 unique values, so it's also at threshold
        assert "above_threshold" in woe_cols

    def test_default_threshold(self):
        """Test with default max_count_woe threshold."""
        X = pl.DataFrame(
            {
                "low_card": ["A"] * 50 + ["B"] * 50,  # 2 unique
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X)  # default max_count_woe=100

        assert "low_card" in woe_cols

    def test_single_unique_value(self):
        """Test column with single unique value."""
        X = pl.DataFrame(
            {
                "constant": ["A"] * 10,
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=5)

        assert "constant" in woe_cols

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame()

        woe_cols, onehot_cols = determine_encoding_strategy(X)

        assert woe_cols == []
        assert onehot_cols == []

    def test_with_nulls(self):
        """Test handling of null values."""
        X = pl.DataFrame(
            {
                "with_nulls": ["A", "B", None, "A", None, "B"],
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=5)

        # Should count unique values including nulls
        assert "with_nulls" in woe_cols or "with_nulls" in onehot_cols

    def test_large_threshold(self):
        """Test with very large threshold."""
        X = pl.DataFrame(
            {
                "col": [f"val_{i}" for i in range(50)],
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=1000)

        # With high threshold, all should use WOE
        assert "col" in woe_cols
        assert onehot_cols == []

    def test_zero_threshold(self):
        """Test with threshold of 0."""
        X = pl.DataFrame(
            {
                "col": ["A", "B"],
            }
        )

        woe_cols, onehot_cols = determine_encoding_strategy(X, max_count_woe=0)

        # With threshold 0, all should use one-hot
        assert woe_cols == []
        assert "col" in onehot_cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
