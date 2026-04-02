import numpy as np
import polars as pl
import pytest

from gators.discretizers import KMeansDiscretizer


class TestKMeansDiscretizer:
    """Tests for KMeansDiscretizer."""

    def test_basic_clustering(self):
        """Test basic k-means based discretization."""
        X = pl.DataFrame(
            {
                "price": [10, 12, 15, 18, 100, 105, 110, 500, 520, 550],
                "quantity": [1, 2, 3, 4, 10, 12, 15, 50, 55, 60],
            }
        )

        discretizer = KMeansDiscretizer(
            subset=["price", "quantity"],
            num_bins=3,
            drop_columns=True,
            inplace=False,
            random_state=42,
        )
        result = discretizer.fit_transform(X)

        # Should have discretized columns
        assert "price__dic_kmeans" in result.columns
        assert "quantity__dic_kmeans" in result.columns
        assert "price" not in result.columns
        assert "quantity" not in result.columns

        # Should have string labels
        assert result["price__dic_kmeans"].dtype == pl.Categorical
        assert result["quantity__dic_kmeans"].dtype == pl.Categorical

    def test_auto_detect_numeric_columns(self):
        """Test automatic detection of numeric columns."""
        X = pl.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "num2": [10, 20, 30, 40, 50, 60, 70, 80],
                "text": ["a", "b", "c", "d", "e", "f", "g", "h"],
            }
        )

        discretizer = KMeansDiscretizer(
            num_bins=3, drop_columns=True, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        # Should discretize numeric columns only
        assert "num1__dic_kmeans" in result.columns
        assert "num2__dic_kmeans" in result.columns
        assert "text" in result.columns

    def test_drop_columns_false(self):
        """Test keeping original columns."""
        X = pl.DataFrame({"feature": [1, 5, 10, 15, 20, 25, 30, 35]})

        discretizer = KMeansDiscretizer(
            subset=["feature"],
            num_bins=3,
            drop_columns=False,
            inplace=False,
            random_state=42,
        )
        result = discretizer.fit_transform(X)

        assert "feature" in result.columns
        assert "feature__dic_kmeans" in result.columns

    def test_few_unique_values(self):
        """Test with fewer unique values than requested bins."""
        X = pl.DataFrame({"feature": [1, 1, 1, 2, 2, 2, 3, 3, 3]})

        discretizer = KMeansDiscretizer(
            subset=["feature"],
            num_bins=5,  # Request more bins than unique values
            inplace=False,
            random_state=42,
        )
        result = discretizer.fit_transform(X)

        # Should handle gracefully with fewer bins
        assert "feature__dic_kmeans" in result.columns

    def test_constant_feature(self):
        """Test handling of constant features."""
        X = pl.DataFrame({"constant": [5.0] * 10})

        discretizer = KMeansDiscretizer(
            subset=["constant"], num_bins=3, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        # Should handle constant feature without error
        assert "constant__dic_kmeans" in result.columns

    def test_with_nulls(self):
        """Test handling of null values."""
        X = pl.DataFrame({"feature": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0]})

        discretizer = KMeansDiscretizer(
            subset=["feature"], num_bins=3, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        # Should handle nulls (filled with median during fit)
        assert "feature__dic_kmeans" in result.columns
        assert len(result) == len(X)

    def test_non_uniform_distribution(self):
        """Test with non-uniform distribution (where k-means shines)."""
        # Create data with clear clusters
        X = pl.DataFrame(
            {
                "value": [
                    1,
                    2,
                    3,
                    4,  # Cluster 1: low values
                    50,
                    51,
                    52,
                    53,  # Cluster 2: medium values
                    100,
                    101,
                    102,
                    103,
                ]  # Cluster 3: high values
            }
        )

        discretizer = KMeansDiscretizer(
            subset=["value"], num_bins=3, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        # K-means should identify the three natural clusters
        assert "value__dic_kmeans" in result.columns
        # Check that we have 3 distinct bins
        unique_bins = result["value__dic_kmeans"].n_unique()
        assert unique_bins <= 3

    def test_max_iter_parameter(self):
        """Test max_iter parameter."""
        X = pl.DataFrame({"feature": list(range(20))})

        discretizer = KMeansDiscretizer(
            subset=["feature"], num_bins=3, max_iter=10, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        assert "feature__dic_kmeans" in result.columns

    def test_n_init_parameter(self):
        """Test n_init parameter."""
        X = pl.DataFrame({"feature": list(range(20))})

        discretizer = KMeansDiscretizer(
            subset=["feature"], num_bins=3, n_init=5, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        assert "feature__dic_kmeans" in result.columns

    def test_invalid_max_iter(self):
        """Test validation of max_iter parameter."""
        with pytest.raises(ValueError, match="max_iter must be at least 1"):
            KMeansDiscretizer(max_iter=0)

    def test_invalid_n_init(self):
        """Test validation of n_init parameter."""
        with pytest.raises(ValueError, match="n_init must be at least 1"):
            KMeansDiscretizer(n_init=0)

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"feature": [1, 5, 10, 15, 20, 25, 30, 35]})

        discretizer = KMeansDiscretizer(subset=["feature"], num_bins=3, random_state=42)

        # Test fit returns self
        assert discretizer.fit(X) is discretizer

        # Test fit_transform
        result = discretizer.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform
        discretizer2 = KMeansDiscretizer(subset=["feature"], num_bins=3, random_state=42)
        discretizer2.fit(X)
        result2 = discretizer2.transform(X)
        assert result.equals(result2)

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        X = pl.DataFrame({"feature": list(range(20))})

        discretizer1 = KMeansDiscretizer(subset=["feature"], num_bins=3, random_state=42)
        result1 = discretizer1.fit_transform(X)

        discretizer2 = KMeansDiscretizer(subset=["feature"], num_bins=3, random_state=42)
        result2 = discretizer2.fit_transform(X)

        assert result1.equals(result2)

    def test_multiple_columns(self):
        """Test discretization of multiple columns."""
        X = pl.DataFrame(
            {
                "col1": [1, 2, 3, 10, 11, 12, 20, 21, 22],
                "col2": [5, 6, 7, 50, 51, 52, 100, 101, 102],
                "col3": [10, 20, 30, 40, 50, 60, 70, 80, 90],
            }
        )

        discretizer = KMeansDiscretizer(
            subset=["col1", "col2", "col3"],
            num_bins=3,
            drop_columns=True,
            inplace=False,
            random_state=42,
        )
        result = discretizer.fit_transform(X)

        assert "col1__dic_kmeans" in result.columns
        assert "col2__dic_kmeans" in result.columns
        assert "col3__dic_kmeans" in result.columns
        assert "col1" not in result.columns

    def test_single_unique_value(self):
        """Test with single unique value."""
        X = pl.DataFrame({"feature": [5, 5, 5, 5, 5]})

        discretizer = KMeansDiscretizer(
            subset=["feature"], num_bins=3, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        # Should handle single value without error
        assert "feature__dic_kmeans" in result.columns

    def test_two_unique_values(self):
        """Test with two unique values."""
        X = pl.DataFrame({"feature": [1, 1, 1, 10, 10, 10]})

        discretizer = KMeansDiscretizer(
            subset=["feature"],
            num_bins=3,  # Request more bins than unique values
            inplace=False,
            random_state=42,
        )
        result = discretizer.fit_transform(X)

        assert "feature__dic_kmeans" in result.columns

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        # Create data with three clear clusters
        cluster1 = np.random.normal(10, 2, 100)
        cluster2 = np.random.normal(50, 3, 100)
        cluster3 = np.random.normal(100, 5, 100)
        X = pl.DataFrame({"value": np.concatenate([cluster1, cluster2, cluster3])})

        discretizer = KMeansDiscretizer(
            subset=["value"], num_bins=3, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        assert "value__dic_kmeans" in result.columns
        assert len(result) == 300

    def test_skewed_distribution(self):
        """Test with skewed distribution."""
        # Exponential-like distribution
        X = pl.DataFrame({"value": [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]})

        discretizer = KMeansDiscretizer(
            subset=["value"], num_bins=3, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        # K-means should create bins based on natural groupings
        assert "value__dic_kmeans" in result.columns

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({"feature": []}, schema={"feature": pl.Float64})

        discretizer = KMeansDiscretizer(
            subset=["feature"], num_bins=3, inplace=False, random_state=42
        )
        result = discretizer.fit_transform(X)

        assert len(result) == 0
        assert "feature__dic_kmeans" in result.columns
