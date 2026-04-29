import numpy as np
import polars as pl
import pytest

from gators.discretizers import TreeBasedDiscretizer


class TestTreeBasedDiscretizer:
    """Tests for TreeBasedDiscretizer."""

    def test_classification_basic(self):
        """Test basic classification-based discretization."""
        X = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
                "income": [
                    30000,
                    35000,
                    40000,
                    50000,
                    55000,
                    60000,
                    70000,
                    75000,
                    80000,
                    90000,
                ],
            }
        )
        y = pl.Series("target", [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        discretizer = TreeBasedDiscretizer(
            subset=["age", "income"],
            num_bins=3,
            task="classification",
            drop_columns=True,
            min_samples_leaf=2,
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        # Should have discretized columns
        assert "age__dic_tree" in result.columns
        assert "income__dic_tree" in result.columns
        assert "age" not in result.columns
        assert "income" not in result.columns

        # Should have string labels
        # assert result["age__dic_tree"].dtype == pl.Categorical
        # assert result["income__dic_tree"].dtype == pl.Categorical

    def test_regression_basic(self):
        """Test basic regression-based discretization."""
        X = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )
        y = pl.Series("target", [10, 15, 18, 25, 30, 35, 42, 50, 60, 75])

        discretizer = TreeBasedDiscretizer(
            subset=["feature1"],
            num_bins=4,
            task="regression",
            drop_columns=True,
            min_samples_leaf=2,
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        assert "feature1__dic_tree" in result.columns
        assert result["feature1__dic_tree"].dtype == pl.Categorical

    def test_no_target_raises_error(self):
        """Test that missing target raises error."""
        X = pl.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5],
            }
        )

        y = pl.Series("target", [0, 0, 0, 1, 1])
        discretizer = TreeBasedDiscretizer(subset=["feature"])

        with pytest.raises(ValueError, match="requires.*target variable"):
            discretizer.fit(X)

    def test_invalid_target_raises_error(self):
        """Test that invalid target column name raises error."""
        X = pl.DataFrame({"feature": [1, 2, 3, 4, 5], "target": [0, 0, 1, 1, 1]})

    def test_auto_detect_numeric_columns(self):
        """Test automatic detection of numeric columns."""
        X = pl.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "num2": [10, 20, 30, 40, 50],
                "text": ["a", "b", "c", "d", "e"],
            }
        )
        y = pl.Series("target", [0, 0, 1, 1, 1])
        discretizer = TreeBasedDiscretizer(
            num_bins=2,
            task="classification",
            drop_columns=True,
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        # Should discretize numeric columns only
        assert "num1__dic_tree" in result.columns
        assert "num2__dic_tree" in result.columns
        assert "text" in result.columns

    def test_drop_columns_false(self):
        """Test keeping original columns."""
        X = pl.DataFrame({"feature": [1, 2, 3, 4, 5, 6, 7, 8], "target": [0, 0, 0, 0, 1, 1, 1, 1]})
        y = pl.Series("target", [0, 0, 0, 0, 1, 1, 1, 1])

        discretizer = TreeBasedDiscretizer(
            subset=["feature"],
            num_bins=2,
            task="classification",
            drop_columns=False,
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        assert "feature" in result.columns
        assert "feature__dic_tree" in result.columns

    def test_min_samples_leaf(self):
        """Test min_samples_leaf parameter."""
        X = pl.DataFrame(
            {
                "feature": list(range(20)),
            }
        )
        y = pl.Series("target", [0] * 10 + [1] * 10)
        discretizer = TreeBasedDiscretizer(
            subset=["feature"],
            num_bins=5,
            task="classification",
            min_samples_leaf=5,
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        # With min_samples_leaf=5, each bin should have at least 5 samples
        assert "feature__dic_tree" in result.columns

    def test_invalid_task(self):
        """Test validation of task parameter."""
        with pytest.raises(ValueError, match="task must be"):
            TreeBasedDiscretizer(task="invalid")

    def test_invalid_min_samples_leaf(self):
        """Test validation of min_samples_leaf parameter."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than 0"):
            TreeBasedDiscretizer(min_samples_leaf=0)

        # Also test with negative value
        with pytest.raises(ValidationError, match="greater than 0"):
            TreeBasedDiscretizer(min_samples_leaf=-1)

    def test_constant_feature(self):
        """Test handling of constant features."""
        X = pl.DataFrame(
            {
                "constant": [5.0] * 10,
            }
        )
        y = pl.Series("target", [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        discretizer = TreeBasedDiscretizer(
            subset=["constant"],
            num_bins=3,
            task="classification",
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        # Should handle constant feature without error
        assert "constant__dic_tree" in result.columns

    def test_with_nulls(self):
        """Test handling of null values."""
        X = pl.DataFrame(
            {
                "feature": [1.0, 2.0, None, 4.0, 5.0, 6.0, None, 8.0],
            }
        )
        y = pl.Series("target", [0, 0, 0, 0, 1, 1, 1, 1])

        discretizer = TreeBasedDiscretizer(
            subset=["feature"],
            num_bins=2,
            task="classification",
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        # Should handle nulls (filled with median during fit)
        assert "feature__dic_tree" in result.columns
        assert len(result) == len(X)

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        y = pl.Series("target", [0, 0, 0, 0, 1, 1, 1, 1])

        discretizer = TreeBasedDiscretizer(
            subset=["feature"], num_bins=2, task="classification", random_state=42
        )

        # Test fit returns self
        assert discretizer.fit(X, y) is discretizer

        # Test fit_transform
        result = discretizer.fit_transform(X, y)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform
        discretizer2 = TreeBasedDiscretizer(
            subset=["feature"], num_bins=2, task="classification", random_state=42
        )
        discretizer2.fit(X, y)
        result2 = discretizer2.transform(X)
        assert result.equals(result2)

    def test_multiclass_classification(self):
        """Test with multiclass target."""
        X = pl.DataFrame(
            {
                "feature": list(range(30)),
            }
        )
        y = pl.Series("target", [0] * 10 + [1] * 10 + [2] * 10)
        discretizer = TreeBasedDiscretizer(
            subset=["feature"],
            num_bins=4,
            task="classification",
            min_samples_leaf=3,
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        assert "feature__dic_tree" in result.columns

    def test_few_unique_values(self):
        """Test with feature having few unique values."""
        X = pl.DataFrame(
            {
                "feature": [1, 1, 2, 2, 3, 3, 4, 4],
            }
        )
        y = pl.Series("target", [0, 0, 0, 1, 1, 1, 1, 1])
        discretizer = TreeBasedDiscretizer(
            subset=["feature"],
            num_bins=3,
            task="classification",
            random_state=42,
            inplace=False,
        )
        result = discretizer.fit_transform(X, y)

        assert "feature__dic_tree" in result.columns

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        X = pl.DataFrame(
            {
                "feature": list(range(20)),
            }
        )
        y = pl.Series("target", [0] * 10 + [1] * 10)
        discretizer1 = TreeBasedDiscretizer(
            subset=["feature"], num_bins=3, task="classification", random_state=42
        )
        result1 = discretizer1.fit_transform(X, y)

        discretizer2 = TreeBasedDiscretizer(
            subset=["feature"], num_bins=3, task="classification", random_state=42
        )
        result2 = discretizer2.fit_transform(X, y)

        assert result1.equals(result2)

    def test_y_parameter_backward_compatibility(self):
        """Test that y parameter still works for backward compatibility."""
        X = pl.DataFrame(
            {
                "feature": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        y = pl.Series("target", [0, 0, 0, 0, 1, 1, 1, 1])
        discretizer = TreeBasedDiscretizer(
            subset=["feature"],
            num_bins=2,
            task="classification",
            random_state=42,
            inplace=False,
        )

        # Using y parameter instead of target
        result = discretizer.fit_transform(X, y)
        assert "feature__dic_tree" in result.columns
