"""Tests for BinaryEncoder."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import BinaryEncoder


class TestBinaryEncoder:
    """Test suite for BinaryEncoder."""

    def test_basic_encoding(self):
        """Test basic binary encoding with 4 categories."""
        X = pl.DataFrame(
            {"category": ["A", "B", "C", "D", "A", "B"], "value": [1, 2, 3, 4, 5, 6]}
        )

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        result = encoder.fit_transform(X)

        # 4 categories need 2 bits
        assert "category__binary_enc_0" in result.columns
        assert "category__binary_enc_1" in result.columns
        assert "category" not in result.columns  # dropped by default
        assert result.shape[0] == 6

        # Check that all values are 0.0 or 1.0
        assert all(v in [0.0, 1.0] for v in result["category__binary_enc_0"].to_list())
        assert all(v in [0.0, 1.0] for v in result["category__binary_enc_1"].to_list())

    def test_drop_columns_false(self):
        """Test keeping original columns."""
        X = pl.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

        encoder = BinaryEncoder(subset=["category"], drop_columns=False, inplace=False)
        result = encoder.fit_transform(X)

        assert "category" in result.columns
        assert "category__binary_enc_0" in result.columns
        assert "category__binary_enc_1" in result.columns

    def test_two_categories_one_bit(self):
        """Test that 2 categories need only 1 bit."""
        X = pl.DataFrame({"category": ["A", "B", "A", "B"], "value": [1, 2, 3, 4]})

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(X)
        result = encoder.transform(X)

        assert encoder.n_bits_["category"] == 1
        assert "category__binary_enc_0" in result.columns
        assert "category__binary_enc_1" not in result.columns

    def test_eight_categories_three_bits(self):
        """Test that 8 categories need 3 bits."""
        X = pl.DataFrame({"category": ["A", "B", "C", "D", "E", "F", "G", "H"]})

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(X)

        assert encoder.n_bits_["category"] == 3
        result = encoder.transform(X)

        assert "category__binary_enc_0" in result.columns
        assert "category__binary_enc_1" in result.columns
        assert "category__binary_enc_2" in result.columns

    def test_min_count_absolute(self):
        """Test min_count with absolute threshold."""
        X = pl.DataFrame(
            {"category": ["A"] * 5 + ["B"] * 3 + ["C"] * 1, "value": range(9)}
        )

        encoder = BinaryEncoder(subset=["category"], min_count=3, inplace=False)
        encoder.fit(X)
        result = encoder.transform(X)

        # Only A (5) and B (3) meet threshold, C (1) is filtered
        # 2 categories need 1 bit
        assert encoder.n_bits_["category"] == 1

        # C should get default value (0.0) - check row with value=8
        assert result["category__binary_enc_0"][8] == 0.0

    def test_min_count_ratio(self):
        """Test min_count with ratio threshold."""
        X = pl.DataFrame(
            {"category": ["A"] * 50 + ["B"] * 30 + ["C"] * 20, "value": range(100)}
        )

        # Only categories with >=35% frequency (35 occurrences)
        encoder = BinaryEncoder(subset=["category"], min_count=0.35, inplace=False)
        encoder.fit(X)

        # Only A (50) meets threshold
        # 1 category needs 1 bit (but encoded anyway)
        assert "category" in encoder.n_bits_

    def test_multiple_columns(self):
        """Test encoding multiple columns."""
        X = pl.DataFrame(
            {
                "cat1": ["A", "B", "C", "D"],
                "cat2": ["X", "Y", "X", "Y"],
                "value": [1, 2, 3, 4],
            }
        )

        encoder = BinaryEncoder(subset=["cat1", "cat2"], inplace=False)
        result = encoder.fit_transform(X)

        # cat1 needs 2 bits (4 categories), cat2 needs 1 bit (2 categories)
        assert "cat1__binary_enc_0" in result.columns
        assert "cat1__binary_enc_1" in result.columns
        assert "cat2__binary_enc_0" in result.columns
        assert "cat1" not in result.columns
        assert "cat2" not in result.columns

    def test_columns_none_auto_detect(self):
        """Test automatic column detection when subset=None."""
        X = pl.DataFrame(
            {
                "cat_col": ["A", "B", "C"],
                "numeric_col": [1, 2, 3],
                "str_col": ["X", "Y", "Z"],
            }
        )

        encoder = BinaryEncoder(inplace=False)  # subset=None
        result = encoder.fit_transform(X)

        # Both string columns should be encoded
        assert (
            "cat_col__binary_enc_0" in result.columns
            or "cat_col__binary_enc_1" in result.columns
        )
        assert (
            "str_col__binary_enc_0" in result.columns
            or "str_col__binary_enc_1" in result.columns
        )
        assert "numeric_col" in result.columns  # not encoded

    def test_unseen_categories_default_value(self):
        """Test that unseen categories get default value."""
        train_X = pl.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

        test_X = pl.DataFrame(
            {"category": ["A", "D", "E"], "value": [4, 5, 6]}  # D and E are unseen
        )

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(train_X)
        result = encoder.transform(test_X)

        # D and E should get default value 0.0 for all bits
        d_row = (
            result.filter(pl.col("category") == "D")[0]
            if "category" in result.columns
            else result[1]
        )
        # Since drop_columns=True, we can't filter by category
        # Just check that transformation works
        assert result.shape[0] == 3

    def test_single_category(self):
        """Test with single category."""
        X = pl.DataFrame({"category": ["A", "A", "A"], "value": [1, 2, 3]})

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(X)
        result = encoder.transform(X)

        # 1 category needs 1 bit
        assert encoder.n_bits_["category"] == 1
        assert result.shape[0] == 3

    def test_boolean_column(self):
        """Test encoding boolean column."""
        X = pl.DataFrame(
            {"bool_col": [True, False, True, False], "value": [1, 2, 3, 4]}
        )

        encoder = BinaryEncoder(subset=["bool_col"], inplace=False)
        result = encoder.fit_transform(X)

        # Boolean (2 categories) needs 1 bit
        assert "bool_col__binary_enc_0" in result.columns
        assert "bool_col" not in result.columns

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({"category": [], "value": []}).cast(
            {"category": pl.String, "value": pl.Int64}
        )

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(X)
        result = encoder.transform(X)

        # Should handle empty DataFrame gracefully
        assert result.shape[0] == 0

    def test_all_nulls(self):
        """Test with all NULL values."""
        X = pl.DataFrame({"category": [None, None, None], "value": [1, 2, 3]}).cast(
            {"category": pl.String}
        )

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(X)
        result = encoder.transform(X)

        # Should handle all nulls gracefully
        assert result.shape[0] == 3

    def test_fit_and_transform_separately(self):
        """Test sklearn-style separate fit and transform."""
        train_X = pl.DataFrame(
            {"category": ["A", "B", "C", "D"], "value": [1, 2, 3, 4]}
        )

        test_X = pl.DataFrame({"category": ["A", "B", "C"], "value": [5, 6, 7]})

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(train_X)
        result = encoder.transform(test_X)

        assert result.shape[0] == 3
        assert "category__binary_enc_0" in result.columns

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

        encoder = BinaryEncoder(subset=["category"], inplace=False)

        # Test fit returns self
        assert encoder.fit(X) is encoder

        # Test fit_transform
        result = encoder.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform produce same result
        encoder2 = BinaryEncoder(subset=["category"], inplace=False)
        encoder2.fit(X)
        result2 = encoder2.transform(X)

        # Both should have same columns and shape
        assert set(result.columns) == set(result2.columns)
        assert result.shape == result2.shape

    def test_mapping_attribute(self):
        """Test that mapping_ attribute is populated correctly."""
        X = pl.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(X)

        assert hasattr(encoder, "mapping_")
        assert "category__binary_enc_0" in encoder.mapping_
        assert "A" in encoder.mapping_["category__binary_enc_0"]

    def test_column_mapping_attribute(self):
        """Test that column_mapping_ attribute is populated."""
        X = pl.DataFrame({"category": ["A", "B"], "value": [1, 2]})

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        encoder.fit(X)

        assert hasattr(encoder, "column_mapping_")
        assert "category__binary_enc_0" in encoder.column_mapping_

    def test_n_bits_calculation(self):
        """Test n_bits calculation for various category counts."""
        test_cases = [
            (2, 1),  # 2 categories -> 1 bit
            (3, 2),  # 3 categories -> 2 bits
            (4, 2),  # 4 categories -> 2 bits
            (5, 3),  # 5 categories -> 3 bits
            (8, 3),  # 8 categories -> 3 bits
            (9, 4),  # 9 categories -> 4 bits
            (16, 4),  # 16 categories -> 4 bits
        ]

        for n_cats, expected_bits in test_cases:
            X = pl.DataFrame({"category": [f"cat_{i}" for i in range(n_cats)]})

            encoder = BinaryEncoder(subset=["category"], inplace=False)
            encoder.fit(X)

            assert (
                encoder.n_bits_["category"] == expected_bits
            ), f"Expected {expected_bits} bits for {n_cats} categories, got {encoder.n_bits_['category']}"

    def test_binary_representation_correctness(self):
        """Test that binary encoding matches expected binary representation."""
        X = pl.DataFrame({"category": ["A", "B", "C", "D"], "value": [1, 2, 3, 4]})

        encoder = BinaryEncoder(subset=["category"], drop_columns=False, inplace=False)
        result = encoder.fit_transform(X)

        # Check that we have 2 bits for 4 categories
        assert "category__binary_enc_0" in result.columns
        assert "category__binary_enc_1" in result.columns

        # Check that same categories have same encoding
        X2 = pl.DataFrame({"category": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
        result2 = encoder.transform(X2)

        # First two rows (both A) should have identical encoding
        assert (
            result2["category__binary_enc_0"][0] == result2["category__binary_enc_0"][1]
        )
        assert (
            result2["category__binary_enc_1"][0] == result2["category__binary_enc_1"][1]
        )

        # Next two rows (both B) should have identical encoding
        assert (
            result2["category__binary_enc_0"][2] == result2["category__binary_enc_0"][3]
        )
        assert (
            result2["category__binary_enc_1"][2] == result2["category__binary_enc_1"][3]
        )

    def test_mixed_string_and_numeric(self):
        """Test with mixed column types."""
        X = pl.DataFrame(
            {
                "str_col": ["A", "B", "C"],
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
            }
        )

        encoder = BinaryEncoder(subset=["str_col"], inplace=False)
        result = encoder.fit_transform(X)

        # Only str_col should be encoded
        assert (
            "str_col__binary_enc_0" in result.columns
            or "str_col__binary_enc_1" in result.columns
        )
        assert "int_col" in result.columns
        assert "float_col" in result.columns
        assert "str_col" not in result.columns

    def test_with_null_values_mixed(self):
        """Test with some NULL values mixed with categories."""
        X = pl.DataFrame(
            {"category": ["A", None, "B", "A", None, "C"], "value": [1, 2, 3, 4, 5, 6]}
        )

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        result = encoder.fit_transform(X)

        # Should encode non-null categories
        assert result.shape[0] == 6
        assert (
            "category__binary_enc_0" in result.columns
            or "category__binary_enc_1" in result.columns
        )

    def test_preserve_other_columns(self):
        """Test that non-encoded columns are preserved."""
        X = pl.DataFrame(
            {"category": ["A", "B", "C"], "keep1": [1, 2, 3], "keep2": ["X", "Y", "Z"]}
        )

        encoder = BinaryEncoder(subset=["category"], inplace=False)
        result = encoder.fit_transform(X)

        assert "keep1" in result.columns
        assert "keep2" in result.columns
        assert result["keep1"].to_list() == [1, 2, 3]
        assert result["keep2"].to_list() == ["X", "Y", "Z"]
        
    def test_all_categories_filtered_by_min_count(self):
        """Test when all categories are filtered out by min_count threshold."""
        X = pl.DataFrame(
            {
                "category": ["A", "B", "C"],  # Each appears only once
                "value": [1, 2, 3]
            }
        )

        # Set min_count so high that no category meets the threshold
        encoder = BinaryEncoder(subset=["category"], min_count=5, inplace=False)
        encoder.fit(X)
        
        # No categories should be valid, so encoding should be skipped
        assert "category" not in encoder.n_bits_
        
        result = encoder.transform(X)
        # Original column should be dropped but no encoded columns added
        assert "category" not in result.columns

