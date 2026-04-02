import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import GroupStatisticsFeatures


def test_transform_basic_single_group_single_agg():
    """Test basic transformation with single groupby column and single aggregation."""
    X = pl.DataFrame(
        {
            "amount": [100, 200, 150, 300, 250],
            "cat1": ["A", "A", "B", "B", "A"],
        }
    )

    transformer = GroupStatisticsFeatures(
        subset=["amount"],
        by=["cat1"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    # Check column exists
    assert "mean_amount__per_cat1" in result.columns

    # Group A: [100, 200, 250] -> mean = 183.333...
    # Group B: [150, 300] -> mean = 225.0
    assert result["mean_amount__per_cat1"][0] == pytest.approx(183.333333, abs=1e-5)
    assert result["mean_amount__per_cat1"][2] == pytest.approx(225.0)


def test_transform_multiple_aggregations():
    """Test with multiple aggregation functions."""
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "min", "max", "count"],
    )
    result = transformer.fit_transform(X)

    # Check all expected columns are present
    assert "mean_value__per_group" in result.columns
    assert "min_value__per_group" in result.columns
    assert "max_value__per_group" in result.columns
    assert "count_value__per_group" in result.columns

    # Group A: [10, 20] -> mean=15, min=10, max=20, count=2
    assert result["mean_value__per_group"][0] == pytest.approx(15.0)
    assert result["min_value__per_group"][0] == 10
    assert result["max_value__per_group"][0] == 20
    assert result["count_value__per_group"][0] == 2


def test_transform_multiple_numerical_columns():
    """Test with multiple numerical columns."""
    X = pl.DataFrame(
        {
            "col1": [100, 200, 150, 300],
            "col2": [50, 100, 75, 150],
            "group": ["A", "A", "B", "B"],
        }
    )

    transformer = GroupStatisticsFeatures(
        subset=["col1", "col2"],
        by=["group"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    assert "mean_col1__per_group" in result.columns
    assert "mean_col2__per_group" in result.columns

    # Group A: col1=[100, 200], mean=150
    assert result["mean_col1__per_group"][0] == pytest.approx(150.0)
    assert result["mean_col2__per_group"][0] == pytest.approx(75.0)


def test_transform_multi_column_groupby():
    """Test with multiple separate groupby columns."""
    X = pl.DataFrame(
        {
            "value": [100, 200, 150, 300, 250, 175],
            "amount": [50, 100, 75, 150, 125, 88],
            "cat1": ["A", "A", "B", "B", "A", "A"],
            "cat2": ["X", "Y", "X", "X", "X", "Y"],
        }
    )

    transformer = GroupStatisticsFeatures(
        subset=["value", "amount"],
        by=["cat1", "cat2"],
        func=["mean", "std"],
    )
    result = transformer.fit_transform(X)

    # Should create 2 numerical × 2 groupby × 2 func = 8 features
    assert "mean_value__per_cat1" in result.columns
    assert "mean_value__per_cat2" in result.columns
    assert "mean_amount__per_cat1" in result.columns
    assert "mean_amount__per_cat2" in result.columns
    assert "std_value__per_cat1" in result.columns
    assert "std_value__per_cat2" in result.columns
    assert "std_amount__per_cat1" in result.columns
    assert "std_amount__per_cat2" in result.columns

    # Group A for cat1: [100, 200, 250, 175] -> mean = 181.25
    assert result["mean_value__per_cat1"][0] == pytest.approx(181.25)


def test_transform_all_aggregations():
    """Test all supported aggregation functions."""
    X = pl.DataFrame({"value": [10, 20, 30, 40, 50, 60], "group": ["A", "A", "A", "B", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "std", "median", "min", "max", "sum", "count"],
    )
    result = transformer.fit_transform(X)

    # Check all aggregation columns are present
    assert "mean_value__per_group" in result.columns
    assert "std_value__per_group" in result.columns
    assert "median_value__per_group" in result.columns
    assert "min_value__per_group" in result.columns
    assert "max_value__per_group" in result.columns
    assert "sum_value__per_group" in result.columns
    assert "count_value__per_group" in result.columns

    # Group A: [10, 20, 30] -> sum=60, min=10, max=30, count=3
    assert result["sum_value__per_group"][0] == pytest.approx(60)
    assert result["min_value__per_group"][0] == 10
    assert result["max_value__per_group"][0] == 30
    assert result["count_value__per_group"][0] == 3
    assert result["median_value__per_group"][0] == pytest.approx(20)


def test_transform_with_drop_columns():
    """Test dropping original numerical columns."""
    X = pl.DataFrame({"value": [10, 20, 30], "other": [1, 2, 3], "group": ["A", "A", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    # Original 'value' column should be dropped
    assert "value" not in result.columns
    assert "mean_value__per_group" in result.columns
    # Other columns should remain
    assert "other" in result.columns
    assert "group" in result.columns


def test_transform_with_custom_column_names():
    """Test using custom column names."""
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "max"],
        new_column_names=["custom_mean", "custom_max"],
    )
    result = transformer.fit_transform(X)

    assert "custom_mean" in result.columns
    assert "custom_max" in result.columns
    assert "mean_value__per_group" not in result.columns
    assert "max_value__per_group" not in result.columns


def test_validation_invalid_aggregation():
    """Test validation error with invalid aggregation function."""
    with pytest.raises(
        ValidationError,
        match="invalid_agg is not in the predefined list of aggregation functions",
    ):
        GroupStatisticsFeatures(
            subset=["value"],
            by=["group"],
            func=["mean", "invalid_agg"],
        )


def test_validation_mismatched_new_column_names_length():
    """Test validation error when new_column_names length doesn't match."""
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})

    # Should create 1 num_col × 2 func = 2 features
    # But only providing 1 name
    with pytest.raises(
        ValueError,
        match="Length of new_column_names .* must match the total number of features created",
    ):
        GroupStatisticsFeatures(
            subset=["value"],
            by=["group"],
            func=["mean", "max"],
            new_column_names=["name1"],  # Should have 2 names
        )


def test_fit_return_self():
    """Test that fit returns self."""
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
    )

    result = transformer.fit(X)
    assert result is transformer


def test_column_mapping_generation():
    """Test that column mapping is correctly generated during fit."""
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "max"],
    )
    transformer.fit(X)

    # Check that column mapping is created
    assert len(transformer._column_mapping) == 2
    assert "mean_value__per_group" in transformer._column_mapping
    assert "max_value__per_group" in transformer._column_mapping


def test_std_aggregation():
    """Test standard deviation aggregation."""
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["std"],
    )
    result = transformer.fit_transform(X)

    # Group A: [10, 20, 30] -> std ≈ 10
    assert result["std_value__per_group"][0] == pytest.approx(10.0, abs=0.1)


def test_median_aggregation():
    """Test median aggregation specifically."""
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["median"],
    )
    result = transformer.fit_transform(X)

    # Group A: [10, 20, 30] -> median = 20
    # Group B: [15, 25, 35] -> median = 25
    assert result["median_value__per_group"][0] == pytest.approx(20.0)
    assert result["median_value__per_group"][3] == pytest.approx(25.0)


def test_empty_dataframe():
    """Test behavior with empty dataframe."""
    X = pl.DataFrame({"value": [], "group": []}, schema={"value": pl.Int64, "group": pl.Utf8})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    assert result.shape[0] == 0
    assert "mean_value__per_group" in result.columns


def test_single_row_dataframe():
    """Test behavior with single row dataframe."""
    X = pl.DataFrame({"value": [100], "group": ["A"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "min", "max"],
    )
    result = transformer.fit_transform(X)

    # With one value, mean = min = max = value
    assert result["mean_value__per_group"][0] == pytest.approx(100.0)
    assert result["min_value__per_group"][0] == 100
    assert result["max_value__per_group"][0] == 100


def test_with_null_values():
    """Test handling of null values in data."""
    X = pl.DataFrame({"value": [10, None, 30, 40, None], "group": ["A", "A", "A", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "count"],
    )
    result = transformer.fit_transform(X)

    # Group A mean: (10 + 30) / 2 = 20 (nulls excluded)
    # Group A count: 2 (nulls excluded from count)
    assert result["mean_value__per_group"][0] == pytest.approx(20.0)
    assert result["count_value__per_group"][0] == 2


def test_negative_values():
    """Test with negative values."""
    X = pl.DataFrame({"value": [-10, -20, 30, 40], "group": ["A", "A", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "sum"],
    )
    result = transformer.fit_transform(X)

    # Group A: [-10, -20] -> mean = -15, sum = -30
    assert result["mean_value__per_group"][0] == pytest.approx(-15.0)
    assert result["sum_value__per_group"][0] == pytest.approx(-30.0)


def test_float_values():
    """Test with float values."""
    X = pl.DataFrame({"value": [10.5, 20.7, 30.2, 40.8], "group": ["A", "A", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    # Group A: [10.5, 20.7] -> mean = 15.6
    assert result["mean_value__per_group"][0] == pytest.approx(15.6)


def test_count_aggregation():
    """Test count aggregation returns correct counts."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30, 40, 50],
            "group": ["A", "A", "A", "B", "B"],
        }
    )

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["count"],
    )
    result = transformer.fit_transform(X)

    # Group A has 3 values, Group B has 2 values
    assert result["count_value__per_group"][0] == 3
    assert result["count_value__per_group"][3] == 2


def test_sum_aggregation():
    """Test sum aggregation."""
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["sum"],
    )
    result = transformer.fit_transform(X)

    # Group A: 10 + 20 = 30
    # Group B: 30 + 40 = 70
    assert result["sum_value__per_group"][0] == pytest.approx(30.0)
    assert result["sum_value__per_group"][2] == pytest.approx(70.0)


def test_min_max_aggregation():
    """Test min and max func."""
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})

    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["min", "max"],
    )
    result = transformer.fit_transform(X)

    # Group A: min=10, max=30
    assert result["min_value__per_group"][0] == 10
    assert result["max_value__per_group"][0] == 30
    # Group B: min=15, max=35
    assert result["min_value__per_group"][3] == 15
    assert result["max_value__per_group"][3] == 35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
