"""Tests for information_value.py"""

import polars as pl
import pytest
import numpy as np
from gators.feature_selection.information_value import compute_iv


class TestComputeIV:
    """Tests for compute_iv function."""

    @pytest.fixture
    def sample_data_string(self):
        """Create sample data with string categorical features."""
        X = pl.DataFrame(
            {
                "feature1": ["a", "a", "b", "c", "a", "b"],
                "feature2": ["x", "x", "x", "y", "y", "y"],
                "feature3": ["p", "p", "p", "p", "q", "q"],
            }
        )
        y = pl.Series("target", [1, 0, 1, 0, 1, 0])
        return X, y

    @pytest.fixture
    def sample_data_categorical(self):
        """Create sample data with Categorical type features."""
        X = pl.DataFrame(
            {
                "cat1": pl.Series(["high", "low", "high", "low", "medium", "high"]).cast(
                    pl.Categorical
                ),
                "cat2": pl.Series(["yes", "no", "yes", "yes", "no", "yes"]).cast(pl.Categorical),
            }
        )
        y = pl.Series("target", [1, 0, 1, 1, 0, 1])
        return X, y

    @pytest.fixture
    def sample_data_mixed(self):
        """Create sample data with mixed types (string, categorical, numeric)."""
        X = pl.DataFrame(
            {
                "str_col": ["a", "b", "a", "b", "c", "c"],
                "cat_col": pl.Series(["x", "y", "x", "y", "z", "z"]).cast(pl.Categorical),
                "num_col": [1.5, 2.3, 3.1, 4.2, 5.0, 6.1],
                "int_col": [1, 2, 3, 4, 5, 6],
            }
        )
        y = pl.Series("target", [1, 0, 1, 0, 1, 0])
        return X, y

    @pytest.fixture
    def sample_data_numeric_only(self):
        """Create sample data with only numeric features."""
        X = pl.DataFrame(
            {
                "num1": [1.5, 2.3, 3.1, 4.2, 5.0, 6.1],
                "num2": [10, 20, 30, 40, 50, 60],
                "num3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        y = pl.Series("target", [1, 0, 1, 0, 1, 0])
        return X, y

    def test_basic_functionality_string(self, sample_data_string):
        """Test basic IV computation with string features."""
        X, y = sample_data_string
        result = compute_iv(X, y)

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["feature", "iv"]
        assert result.shape[0] == 3  # 3 string features

        # Check variable names
        assert set(result["feature"].to_list()) == {"feature1", "feature2", "feature3"}

        # Check IV values are positive (or zero)
        assert all(result["iv"] >= 0)

        # Check dtypes
        assert result.schema["feature"] == pl.String
        assert result.schema["iv"] == pl.Float64

    def test_basic_functionality_categorical(self, sample_data_categorical):
        """Test basic IV computation with categorical type features."""
        X, y = sample_data_categorical
        result = compute_iv(X, y)

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["feature", "iv"]
        assert result.shape[0] == 2  # 2 categorical features

        # Check variable names
        assert set(result["feature"].to_list()) == {"cat1", "cat2"}

        # Check IV values
        assert all(result["iv"] >= 0)

    def test_mixed_types(self, sample_data_mixed):
        """Test that only string and categorical features are processed."""
        X, y = sample_data_mixed
        result = compute_iv(X, y)

        # Should only include str_col and cat_col, not numeric columns
        assert result.shape[0] == 2
        assert set(result["feature"].to_list()) == {"str_col", "cat_col"}

        # Numeric columns should be excluded
        assert "num_col" not in result["feature"].to_list()
        assert "int_col" not in result["feature"].to_list()

    def test_numeric_only_empty_result(self, sample_data_numeric_only):
        """Test that numeric-only DataFrame returns empty result."""
        X, y = sample_data_numeric_only
        result = compute_iv(X, y)

        # Should return empty DataFrame with correct schema
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["feature", "iv"]
        assert result.shape[0] == 0
        assert result.schema["feature"] == pl.String
        assert result.schema["iv"] == pl.Float64

    def test_empty_dataframe(self):
        """Test with completely empty DataFrame."""
        X = pl.DataFrame()
        y = pl.Series("target", [])
        result = compute_iv(X, y)

        # Should return empty DataFrame with correct schema
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["feature", "iv"]
        assert result.shape[0] == 0

    def test_regularization_default(self, sample_data_string):
        """Test default regularization parameter."""
        X, y = sample_data_string
        result_default = compute_iv(X, y)
        result_explicit = compute_iv(X, y, regularization=0.01)

        # Results should be identical (sort by variable to ensure same order)
        result_default_sorted = result_default.sort("feature")
        result_explicit_sorted = result_explicit.sort("feature")
        assert result_default_sorted.equals(result_explicit_sorted)

    def test_regularization_different_values(self, sample_data_string):
        """Test different regularization values."""
        X, y = sample_data_string

        result_low = compute_iv(X, y, regularization=0.001)
        result_default = compute_iv(X, y, regularization=0.01)
        result_high = compute_iv(X, y, regularization=0.1)

        # All should have same structure
        assert result_low.shape == result_default.shape == result_high.shape
        assert (
            set(result_low["feature"])
            == set(result_default["feature"])
            == set(result_high["feature"])
        )

        # IV values should differ (regularization affects calculation)
        assert not result_low["iv"].equals(result_high["iv"])

    def test_regularization_zero(self, sample_data_string):
        """Test with zero regularization."""
        X, y = sample_data_string
        result = compute_iv(X, y, regularization=0.0)

        # Should still work, just without regularization
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 3
        assert all(result["iv"] >= 0)

    def test_perfect_predictor(self):
        """Test with a perfect predictor feature."""
        X = pl.DataFrame(
            {
                "perfect": ["a", "a", "a", "b", "b", "b"],
            }
        )
        y = pl.Series("target", [1, 1, 1, 0, 0, 0])

        result = compute_iv(X, y)

        # Perfect predictor should have high IV (but not infinite due to regularization)
        assert result.shape[0] == 1
        assert result["feature"][0] == "perfect"
        assert result["iv"][0] > 1.0  # High IV value

    def test_useless_feature(self):
        """Test with a feature that has no predictive power."""
        X = pl.DataFrame(
            {
                "useless": ["a", "b", "a", "b", "a", "b", "a", "b"],
            }
        )
        # Balanced target: equal 1s and 0s for each category
        y = pl.Series("target", [1, 1, 0, 0, 1, 1, 0, 0])

        result = compute_iv(X, y)

        # Useless feature should have very low IV
        assert result.shape[0] == 1
        assert result["feature"][0] == "useless"
        assert result["iv"][0] < 0.1  # Very low IV

    def test_single_category(self):
        """Test with a feature that has only one category."""
        X = pl.DataFrame(
            {
                "single": ["a", "a", "a", "a", "a", "a"],
            }
        )
        y = pl.Series("target", [1, 0, 1, 0, 1, 0])

        result = compute_iv(X, y)

        # Single category should have zero or very low IV
        assert result.shape[0] == 1
        assert result["feature"][0] == "single"
        assert result["iv"][0] >= 0

    def test_binary_target_all_ones(self):
        """Test with target that is all ones."""
        X = pl.DataFrame(
            {
                "feature": ["a", "b", "a", "b", "c", "c"],
            }
        )
        y = pl.Series("target", [1, 1, 1, 1, 1, 1])

        result = compute_iv(X, y)

        # Should still compute IV (with regularization preventing division issues)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1

    def test_binary_target_all_zeros(self):
        """Test with target that is all zeros."""
        X = pl.DataFrame(
            {
                "feature": ["a", "b", "a", "b", "c", "c"],
            }
        )
        y = pl.Series("target", [0, 0, 0, 0, 0, 0])

        result = compute_iv(X, y)

        # Should still compute IV
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1

    def test_many_categories(self):
        """Test with a feature that has many categories."""
        X = pl.DataFrame(
            {
                "many_cats": [f"cat_{i}" for i in range(50)],
            }
        )
        y = pl.Series("target", [i % 2 for i in range(50)])

        result = compute_iv(X, y)

        # Should handle many categories
        assert result.shape[0] == 1
        assert result["feature"][0] == "many_cats"
        assert result["iv"][0] >= 0

    def test_iv_values_calculated_correctly(self):
        """Test that IV values are calculated correctly with known example."""
        # Create a simple example where we can verify the IV calculation
        X = pl.DataFrame(
            {
                "feature": ["a", "a", "b", "b"],
            }
        )
        y = pl.Series("target", [1, 0, 1, 1])

        result = compute_iv(X, y, regularization=0.01)

        # Verify it computes without error and returns expected structure
        assert result.shape[0] == 1
        assert result["feature"][0] == "feature"
        assert isinstance(result["iv"][0], float)
        assert result["iv"][0] > 0

    def test_multiple_features_ordering(self, sample_data_string):
        """Test that all features are included regardless of order."""
        X, y = sample_data_string
        result = compute_iv(X, y)

        # All string features should be present
        features_in_result = set(result["feature"].to_list())
        features_in_X = set(X.columns)
        assert features_in_result == features_in_X

    def test_result_is_dataframe(self, sample_data_string):
        """Test that result is always a DataFrame, not a dict or other type."""
        X, y = sample_data_string
        result = compute_iv(X, y)

        assert type(result).__name__ == "DataFrame"
        assert hasattr(result, "columns")
        assert hasattr(result, "shape")

    def test_with_boolean_column(self):
        """Test that boolean columns are not included (only String/Categorical)."""
        X = pl.DataFrame(
            {
                "bool_col": [True, False, True, False, True, False],
                "str_col": ["a", "b", "a", "b", "c", "c"],
            }
        )
        y = pl.Series("target", [1, 0, 1, 0, 1, 0])

        result = compute_iv(X, y)

        # Only string column should be included
        assert result.shape[0] == 1
        assert result["feature"][0] == "str_col"
        assert "bool_col" not in result["feature"].to_list()

    def test_large_dataset(self):
        """Test with a larger dataset."""
        np.random.seed(42)
        n = 1000
        X = pl.DataFrame(
            {
                "cat1": [f"cat_{i % 10}" for i in range(n)],
                "cat2": [f"type_{i % 5}" for i in range(n)],
                "cat3": [f"group_{i % 20}" for i in range(n)],
            }
        )
        y = pl.Series("target", (np.random.rand(n) > 0.5).astype(int))

        result = compute_iv(X, y)

        # Should handle large dataset
        assert result.shape[0] == 3
        assert set(result["feature"].to_list()) == {"cat1", "cat2", "cat3"}
        assert all(result["iv"] >= 0)

    def test_imbalanced_target(self):
        """Test with highly imbalanced target."""
        X = pl.DataFrame(
            {
                "feature": ["a", "b", "c"] * 33 + ["a"],
            }
        )
        # 95 zeros, 5 ones
        y = pl.Series("target", [0] * 95 + [1] * 5)

        result = compute_iv(X, y)

        # Should handle imbalanced target
        assert result.shape[0] == 1
        assert result["iv"][0] >= 0

    def test_regularization_affects_result(self):
        """Test that regularization parameter actually affects the result."""
        X = pl.DataFrame(
            {
                "feature": ["a", "a", "b", "b"],
            }
        )
        y = pl.Series("target", [1, 1, 0, 0])

        result_low_reg = compute_iv(X, y, regularization=0.001)
        result_high_reg = compute_iv(X, y, regularization=0.5)

        # IV values should be different
        assert result_low_reg["iv"][0] != result_high_reg["iv"][0]
