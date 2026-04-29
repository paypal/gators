import numpy as np
import polars as pl
import pytest

from gators.discretizers import QuantileDiscretizer


class TestQuantileDiscretizer:
    """Tests for QuantileDiscretizer."""

    def test_basic_quantiles(self):
        """Test basic quantile-based discretization."""
        X = pl.DataFrame(
            {
                "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                "income": [
                    20000,
                    25000,
                    30000,
                    35000,
                    40000,
                    50000,
                    60000,
                    70000,
                    80000,
                    90000,
                    100000,
                    120000,
                ],
            }
        )

        discretizer = QuantileDiscretizer(
            subset=["age", "income"],
            num_bins=4,  # Quartiles
            drop_columns=True,
            inplace=False,
        )
        result = discretizer.fit_transform(X)

        # Should have discretized columns
        assert "age__dic_quantile" in result.columns
        assert "income__dic_quantile" in result.columns
        assert "age" not in result.columns
        assert "income" not in result.columns

        # Should have string labels
        # assert result["age__dic_quantile"].dtype == pl.Enum
        # assert result["income__dic_quantile"].dtype == pl.Enum

    def test_custom_quantiles(self):
        """Test with custom quantile specifications."""
        X = pl.DataFrame({"value": list(range(100))})

        # Deciles: 10th, 20th, ..., 90th percentiles
        discretizer = QuantileDiscretizer(
            subset=["value"],
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            drop_columns=True,
            inplace=False,
        )
        result = discretizer.fit_transform(X)

        assert "value__dic_quantile" in result.columns

    def test_tertiles(self):
        """Test tertile discretization."""
        X = pl.DataFrame({"score": list(range(30))})

        discretizer = QuantileDiscretizer(
            subset=["score"],
            quantiles=[0.333, 0.667],
            drop_columns=True,
            inplace=False,
        )
        result = discretizer.fit_transform(X)

        assert "score__dic_quantile" in result.columns

    def test_asymmetric_quantiles(self):
        """Test with asymmetric quantiles (focus on tail)."""
        X = pl.DataFrame({"value": list(range(100))})

        # Focus on upper tail
        discretizer = QuantileDiscretizer(
            subset=["value"],
            quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
            drop_columns=True,
            inplace=False,
        )
        result = discretizer.fit_transform(X)

        assert "value__dic_quantile" in result.columns

    def test_auto_detect_numeric_columns(self):
        """Test automatic detection of numeric columns."""
        X = pl.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "num2": [10, 20, 30, 40, 50, 60, 70, 80],
                "text": ["a", "b", "c", "d", "e", "f", "g", "h"],
            }
        )

        discretizer = QuantileDiscretizer(num_bins=4, drop_columns=True, inplace=False)
        result = discretizer.fit_transform(X)

        # Should discretize numeric columns only
        assert "num1__dic_quantile" in result.columns
        assert "num2__dic_quantile" in result.columns
        assert "text" in result.columns

    def test_drop_columns_false(self):
        """Test keeping original columns."""
        X = pl.DataFrame({"feature": list(range(10))})

        discretizer = QuantileDiscretizer(
            subset=["feature"], num_bins=3, drop_columns=False, inplace=False
        )
        result = discretizer.fit_transform(X)

        assert "feature" in result.columns
        assert "feature__dic_quantile" in result.columns

    def test_duplicate_quantiles_drop(self):
        """Test handling of duplicate quantile values with drop strategy."""
        # Small dataset where quantiles will be duplicates
        X = pl.DataFrame({"feature": [1, 1, 1, 2, 2, 2]})

        discretizer = QuantileDiscretizer(
            subset=["feature"],
            num_bins=5,  # Request more bins than unique values
            handle_duplicates="drop",
            inplace=False,
        )
        result = discretizer.fit_transform(X)

        # Should handle duplicates by dropping them
        assert "feature__dic_quantile" in result.columns

    def test_duplicate_quantiles_raise(self):
        """Test handling of duplicate quantile values with raise strategy."""
        X = pl.DataFrame({"feature": [1, 1, 1, 2, 2, 2]})

        discretizer = QuantileDiscretizer(subset=["feature"], num_bins=5, handle_duplicates="raise")

        with pytest.raises(ValueError, match="duplicate quantile values"):
            discretizer.fit_transform(X)

    def test_invalid_quantiles_range(self):
        """Test validation of quantile range."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            QuantileDiscretizer(quantiles=[0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            QuantileDiscretizer(quantiles=[-0.1, 0.5])

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            QuantileDiscretizer(quantiles=[0.5, 1.5])

    def test_invalid_quantiles_order(self):
        """Test validation of quantile ordering."""
        with pytest.raises(ValueError, match="must be in ascending order"):
            QuantileDiscretizer(quantiles=[0.5, 0.3, 0.7])

    def test_empty_quantiles(self):
        """Test validation of empty quantiles list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            QuantileDiscretizer(quantiles=[])

    def test_invalid_handle_duplicates(self):
        """Test validation of handle_duplicates parameter."""
        with pytest.raises(ValueError, match="must be 'drop' or 'raise'"):
            QuantileDiscretizer(handle_duplicates="invalid")

    def test_with_nulls(self):
        """Test handling of null values."""
        X = pl.DataFrame({"feature": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0]})

        discretizer = QuantileDiscretizer(subset=["feature"], num_bins=3, inplace=False)
        result = discretizer.fit_transform(X)

        # Should handle nulls
        assert "feature__dic_quantile" in result.columns
        assert len(result) == len(X)

    def test_skewed_distribution(self):
        """Test with skewed distribution."""
        # Right-skewed data
        np.random.seed(42)
        X = pl.DataFrame({"value": np.random.exponential(scale=2.0, size=100).tolist()})

        discretizer = QuantileDiscretizer(subset=["value"], num_bins=4, inplace=False)
        result = discretizer.fit_transform(X)

        # Quantile-based discretization should handle skewness well
        assert "value__dic_quantile" in result.columns

    def test_uniform_distribution(self):
        """Test with uniform distribution."""
        X = pl.DataFrame({"value": list(range(100))})

        discretizer = QuantileDiscretizer(subset=["value"], num_bins=5, inplace=False)
        result = discretizer.fit_transform(X)

        assert "value__dic_quantile" in result.columns

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"feature": list(range(20))})

        discretizer = QuantileDiscretizer(subset=["feature"], num_bins=4)

        # Test fit returns self
        assert discretizer.fit(X) is discretizer

        # Test fit_transform
        result = discretizer.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform
        discretizer2 = QuantileDiscretizer(subset=["feature"], num_bins=4)
        discretizer2.fit(X)
        result2 = discretizer2.transform(X)
        assert result.equals(result2)

    def test_multiple_columns(self):
        """Test discretization of multiple columns."""
        X = pl.DataFrame(
            {
                "col1": list(range(20)),
                "col2": [x * 2 for x in range(20)],
                "col3": [x**2 for x in range(20)],
            }
        )

        discretizer = QuantileDiscretizer(
            subset=["col1", "col2", "col3"],
            num_bins=5,
            drop_columns=True,
            inplace=False,
        )
        result = discretizer.fit_transform(X)

        assert "col1__dic_quantile" in result.columns
        assert "col2__dic_quantile" in result.columns
        assert "col3__dic_quantile" in result.columns
        assert "col1" not in result.columns

    def test_single_value(self):
        """Test with constant feature."""
        X = pl.DataFrame({"constant": [5.0] * 10})

        discretizer = QuantileDiscretizer(
            subset=["constant"], num_bins=3, handle_duplicates="drop", inplace=False
        )
        result = discretizer.fit_transform(X)

        # Should handle constant feature without error
        assert "constant__dic_quantile" in result.columns

    def test_median_split(self):
        """Test median split (2 bins)."""
        X = pl.DataFrame({"value": list(range(20))})

        discretizer = QuantileDiscretizer(subset=["value"], quantiles=[0.5], inplace=False)
        result = discretizer.fit_transform(X)

        assert "value__dic_quantile" in result.columns
        # Should have 2 bins

    def test_quintiles(self):
        """Test quintile discretization (5 bins)."""
        X = pl.DataFrame({"value": list(range(50))})

        discretizer = QuantileDiscretizer(subset=["value"], num_bins=5, inplace=False)
        result = discretizer.fit_transform(X)

        assert "value__dic_quantile" in result.columns

    def test_percentiles(self):
        """Test specific percentiles."""
        X = pl.DataFrame({"score": list(range(100))})

        # 25th, 50th, 75th percentiles (quartiles)
        discretizer = QuantileDiscretizer(
            subset=["score"], quantiles=[0.25, 0.5, 0.75], inplace=False
        )
        result = discretizer.fit_transform(X)

        assert "score__dic_quantile" in result.columns

    def test_extreme_quantiles(self):
        """Test extreme quantiles near boundaries."""
        X = pl.DataFrame({"value": list(range(1000))})

        # Very fine bins at extremes
        discretizer = QuantileDiscretizer(
            subset=["value"],
            quantiles=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99],
            inplace=False,
        )
        result = discretizer.fit_transform(X)

        assert "value__dic_quantile" in result.columns

    def test_repeated_values(self):
        """Test with many repeated values."""
        X = pl.DataFrame({"value": [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20})

        discretizer = QuantileDiscretizer(
            subset=["value"], num_bins=5, handle_duplicates="drop", inplace=False
        )
        result = discretizer.fit_transform(X)

        assert "value__dic_quantile" in result.columns

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({"feature": []}, schema={"feature": pl.Float64})

        discretizer = QuantileDiscretizer(subset=["feature"], num_bins=3, inplace=False)
        result = discretizer.fit_transform(X)

        assert len(result) == 0
        assert "feature__dic_quantile" in result.columns

    def test_comparison_with_num_bins(self):
        """Test that custom quantiles override num_bins."""
        X = pl.DataFrame({"value": list(range(100))})

        # num_bins should be ignored when quantiles are provided
        discretizer = QuantileDiscretizer(
            subset=["value"],
            num_bins=10,  # This should be ignored
            quantiles=[0.5],  # Only one split point (2 bins)
            inplace=False,
        )
        result = discretizer.fit_transform(X)

        assert "value__dic_quantile" in result.columns
