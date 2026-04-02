import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import OutlierFilter


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 200],
            "income": [
                30000,
                35000,
                40000,
                45000,
                50000,
                55000,
                60000,
                65000,
                70000,
                75000,
            ],
            "score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )


@pytest.fixture
def imbalanced_X():
    """DataFrame with imbalanced classes for testing class_aware mode."""
    return pl.DataFrame(
        {
            "transaction_amount": [100, 120, 110, 105, 115, 5000, 4800, 4900],
            "transaction_count": [1, 2, 1, 2, 1, 10, 12, 11],
            "is_fraud": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )


def test_iqr_remove(sample_X):
    """Test IQR method with row removal."""
    transformer = OutlierFilter(
        subset=["age"], method="iqr", threshold=1.5, action="remove"
    )
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    # Age 200 should be removed as outlier
    assert len(transformed_X) == 9
    assert 200 not in transformed_X["age"].to_list()


def test_iqr_cap(sample_X):
    """Test IQR method with value capping."""
    transformer = OutlierFilter(
        subset=["age"], method="iqr", threshold=1.5, action="cap"
    )
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    # All rows preserved but age 200 should be capped
    assert len(transformed_X) == 10
    assert transformed_X["age"].max() < 200


def test_zscore_remove(sample_X):
    """Test Z-score method with row removal."""
    transformer = OutlierFilter(
        subset=["age"], method="zscore", threshold=2.0, action="remove"
    )
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    # Age 200 should be removed as outlier (z-score > 2)
    assert len(transformed_X) == 9
    assert 200 not in transformed_X["age"].to_list()


def test_zscore_cap(sample_X):
    """Test Z-score method with value capping."""
    transformer = OutlierFilter(
        subset=["age"], method="zscore", threshold=2.0, action="cap"
    )
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    # All rows preserved but age 200 should be capped
    assert len(transformed_X) == 10
    assert transformed_X["age"].max() < 200


def test_percentile_remove(sample_X):
    """Test percentile method with row removal."""
    transformer = OutlierFilter(
        subset=["age"],
        method="percentile",
        lower_percentile=0.05,
        upper_percentile=0.90,
        action="remove",
    )
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    # Values above 90th percentile should be removed
    assert len(transformed_X) < 10


def test_percentile_cap(sample_X):
    """Test percentile method with value capping."""
    transformer = OutlierFilter(
        subset=["age"],
        method="percentile",
        lower_percentile=0.10,
        upper_percentile=0.90,
        action="cap",
    )
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    # All rows preserved
    assert len(transformed_X) == 10


def test_multiple_columns(sample_X):
    """Test outlier detection on multiple columns."""
    transformer = OutlierFilter(
        subset=["age", "income"], method="iqr", threshold=1.5, action="remove"
    )
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    # Only age has outlier, should remove 1 row
    assert len(transformed_X) == 9


def test_no_columns_specified(sample_X):
    """Test with no columns specified - should check all numeric columns."""
    transformer = OutlierFilter(method="iqr", threshold=1.5, action="remove")
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    # Should detect outlier in age column
    assert len(transformed_X) == 9


def test_class_aware_remove(imbalanced_X):
    """Test class-aware mode prevents removing minority class."""
    # Without class_aware: fraud cases may be removed as outliers
    transformer_basic = OutlierFilter(
        subset=["transaction_amount"],
        method="iqr",
        threshold=1.5,
        action="remove",
        class_aware=False,
    )
    transformer_basic.fit(imbalanced_X)
    result_basic = transformer_basic.transform(imbalanced_X)

    # With class_aware: fraud cases should be preserved
    transformer_aware = OutlierFilter(
        subset=["transaction_amount"],
        method="iqr",
        threshold=1.5,
        action="remove",
        class_aware=True,
    )
    transformer_aware.fit(imbalanced_X, y="is_fraud")
    result_aware = transformer_aware.transform(imbalanced_X)

    # Class-aware should preserve more rows (especially minority class)
    assert len(result_aware) >= len(result_basic)
    # Verify fraud cases are preserved
    fraud_count_aware = result_aware.filter(pl.col("is_fraud") == 1).shape[0]
    assert fraud_count_aware == 3  # All 3 fraud cases preserved


def test_class_aware_cap(imbalanced_X):
    """Test class-aware mode with capping action."""
    transformer = OutlierFilter(
        subset=["transaction_amount"],
        method="iqr",
        threshold=1.5,
        action="cap",
        class_aware=True,
    )
    transformer.fit(imbalanced_X, y="is_fraud")
    transformed_X = transformer.transform(imbalanced_X)

    # All rows preserved
    assert len(transformed_X) == 8
    # Verify fraud class has different bounds than normal class
    assert transformed_X["transaction_amount"].max() > 1000


def test_class_aware_multiple_columns(imbalanced_X):
    """Test class-aware mode on multiple columns."""
    transformer = OutlierFilter(
        subset=["transaction_amount", "transaction_count"],
        method="iqr",
        threshold=1.5,
        action="remove",
        class_aware=True,
    )
    transformer.fit(imbalanced_X, y="is_fraud")
    transformed_X = transformer.transform(imbalanced_X)

    # Should preserve minority class across both columns
    fraud_count = transformed_X.filter(pl.col("is_fraud") == 1).shape[0]
    assert fraud_count == 3


def test_class_aware_zscore(imbalanced_X):
    """Test class-aware mode with z-score method."""
    transformer = OutlierFilter(
        subset=["transaction_amount"],
        method="zscore",
        threshold=2.0,
        action="remove",
        class_aware=True,
    )
    transformer.fit(imbalanced_X, y="is_fraud")
    transformed_X = transformer.transform(imbalanced_X)

    # Should preserve all data since outliers computed per class
    assert len(transformed_X) == 8


def test_class_aware_percentile(imbalanced_X):
    """Test class-aware mode with percentile method."""
    transformer = OutlierFilter(
        subset=["transaction_amount"],
        method="percentile",
        lower_percentile=0.05,
        upper_percentile=0.95,
        action="remove",
        class_aware=True,
    )
    transformer.fit(imbalanced_X, y="is_fraud")
    transformed_X = transformer.transform(imbalanced_X)

    # Should handle small minority class gracefully
    fraud_count = transformed_X.filter(pl.col("is_fraud") == 1).shape[0]
    assert fraud_count > 0


def test_class_aware_requires_target():
    """Test that class_aware=True requires target column."""
    X =  pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    transformer = OutlierFilter(class_aware=True)

    with pytest.raises(ValueError, match="Target column name 'y' must be provided"):
        transformer.fit(X)


def test_class_aware_target_not_in_columns():
    """Test error when target column not found."""
    X =  pl.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    transformer = OutlierFilter(class_aware=True)

    with pytest.raises(ValueError, match="Target column 'z' not found"):
        transformer.fit(X, y="z")


def test_empty_columns_list():
    """Test with empty columns after filtering."""
    X =  pl.DataFrame({"cat": ["a", "b", "c"]})
    transformer = OutlierFilter(method="iqr", action="remove")
    transformer.fit(X)
    transformed_X = transformer.transform(X)

    # No numeric columns, should return original
    assert_frame_equal(transformed_X, X)


def test_fit_transform(sample_X):
    """Test fit_transform method."""
    transformer = OutlierFilter(
        subset=["age"], method="iqr", threshold=1.5, action="remove"
    )
    transformed_X = transformer.fit_transform(sample_X)

    assert len(transformed_X) == 9
    assert 200 not in transformed_X["age"].to_list()


def test_class_aware_with_numeric_target():
    """Test that numeric target column is excluded from outlier detection."""
    X =  pl.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 100],  # Has outlier
            "target": [0, 0, 0, 1, 1, 1],  # Numeric target
            "feature2": [10, 20, 30, 40, 50, 60],
        }
    )
    # Don't specify columns - let it auto-detect numeric columns
    transformer = OutlierFilter(method="iqr", threshold=1.5, action="remove", class_aware=True)
    transformer.fit(X, y="target")
    
    # Target should not be in columns being checked for outliers
    assert "target" not in transformer.subset
    
    transformed_X = transformer.transform(X)
    # Should still have data
    assert len(transformed_X) > 0


def test_zscore_with_null_values_class_aware():
    """Test z-score method with null values in class_aware mode."""
    X =  pl.DataFrame(
        {
            "feature": [None, None, None, 1.0, 2.0, 3.0],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    transformer = OutlierFilter(
        subset=["feature"],
        method="zscore",
        threshold=2.0,
        action="cap",
        class_aware=True,
    )
    transformer.fit(X, y="target")
    # Should handle None values gracefully
    transformed_X = transformer.transform(X)
    assert len(transformed_X) == 6


def test_zscore_with_null_values_global():
    """Test z-score method with null values in global mode to hit else branch."""
    X =  pl.DataFrame(
        {
            "feature": [None, None, None],
            "other": [1, 2, 3],
        }
    )
    transformer = OutlierFilter(
        subset=["feature"],
        method="zscore",
        threshold=2.0,
        action="cap",
        class_aware=False,
    )
    transformer.fit(X)
    # Bounds should be None since all values are None
    assert transformer._bounds["feature"]["lower"] is None
    assert transformer._bounds["feature"]["upper"] is None
    
    transformed_X = transformer.transform(X)
    assert len(transformed_X) == 3


def test_percentile_global_mode(sample_X):
    """Test percentile method in global mode."""
    transformer = OutlierFilter(
        subset=["age", "income"],
        method="percentile",
        lower_percentile=0.1,
        upper_percentile=0.9,
        action="cap",
    )
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)
    
    # All rows should be preserved with capping
    assert len(transformed_X) == 10
    # Extreme values should be capped
    assert transformed_X["age"].max() < 200


def test_cap_with_none_bounds_global():
    """Test cap action when bounds are None in global mode."""
    X =  pl.DataFrame(
        {
            "feature1": [None, None, None],
            "feature2": [1, 2, 3],
        }
    )
    transformer = OutlierFilter(
        subset=["feature1"],
        method="zscore",
        threshold=2.0,
        action="cap",
        class_aware=False,
    )
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    
    # Should keep all columns including the one with None bounds
    assert "feature1" in transformed_X.columns
    assert "feature2" in transformed_X.columns
    assert len(transformed_X) == 3


def test_cap_with_none_bounds_class_aware():
    """Test cap action when bounds are None in class_aware mode."""
    X =  pl.DataFrame(
        {
            "feature": [None, None, 1.0, 2.0],
            "target": [0, 0, 1, 1],
            "other": [10, 20, 30, 40],
        }
    )
    transformer = OutlierFilter(
        subset=["feature"],
        method="zscore",
        threshold=2.0,
        action="cap",
        class_aware=True,
    )
    transformer.fit(X, y="target")
    transformed_X = transformer.transform(X)
    
    # Should keep all columns
    assert "feature" in transformed_X.columns
    assert "other" in transformed_X.columns
    assert len(transformed_X) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
