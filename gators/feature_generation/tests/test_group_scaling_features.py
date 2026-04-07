import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation.group_scaling_features import GroupScalingFeatures


def test_transform_basic_single_group_single_agg():
    """Test basic transformation with single groupby column and single aggregation."""
    X = pl.DataFrame(
        {
            "amount": [100, 200, 150, 300, 250],
            "cat1": ["A", "A", "B", "B", "A"],
        }
    )

    transformer = GroupScalingFeatures(
        subset=["amount"],
        by=["cat1"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    # Group A: [100, 200, 250] -> mean = 183.33...
    # Group B: [150, 300] -> mean = 225.0
    expected = X.with_columns(
        [
            pl.when(pl.col("cat1") == "A")
            .then(pl.col("amount") / 183.333333)
            .otherwise(pl.col("amount") / 225.0)
            .alias("amount__mean_cat1")
        ]
    )

    assert_frame_equal(result, expected, abs_tol=1e-5)
    assert "amount__mean_cat1" in result.columns


def test_transform_multiple_scaling_functions():
    """Test with multiple scaling functions."""
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "median", "zscore"],
    )
    result = transformer.fit_transform(X)

    # Check all expected columns are present
    assert "value__mean_group" in result.columns
    assert "value__median_group" in result.columns
    assert "value__zscore_group" in result.columns

    # Group A: [10, 20] -> mean=15, median=15, std=7.07
    # Group B: [30, 40] -> mean=35, median=35, std=7.07
    assert result["value__mean_group"][0] == pytest.approx(10 / 15)
    assert result["value__median_group"][0] == pytest.approx(10 / 15)
    # zscore for 10 in group A: (10-15)/7.07 ≈ -0.707
    assert result["value__zscore_group"][0] == pytest.approx(-0.707, abs=0.01)


def test_transform_multiple_numerical_columns():
    """Test with multiple numerical columns."""
    X = pl.DataFrame(
        {
            "col1": [100, 200, 150, 300],
            "col2": [50, 100, 75, 150],
            "group": ["A", "A", "B", "B"],
        }
    )

    transformer = GroupScalingFeatures(
        subset=["col1", "col2"],
        by=["group"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    assert "col1__mean_group" in result.columns
    assert "col2__mean_group" in result.columns

    # Group A: col1=[100, 200], mean=150; col2=[50, 100], mean=75
    assert result["col1__mean_group"][0] == pytest.approx(100 / 150)
    assert result["col2__mean_group"][0] == pytest.approx(50 / 75)


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

    transformer = GroupScalingFeatures(
        subset=["value", "amount"],
        by=["cat1", "cat2"],
        func=["mean", "median"],
    )
    result = transformer.fit_transform(X)

    # Should create 2 numerical × 2 groupby × 2 func = 8 features
    assert "value__mean_cat1" in result.columns
    assert "value__mean_cat2" in result.columns
    assert "amount__mean_cat1" in result.columns
    assert "amount__mean_cat2" in result.columns
    assert "value__median_cat1" in result.columns
    assert "value__median_cat2" in result.columns
    assert "amount__median_cat1" in result.columns
    assert "amount__median_cat2" in result.columns

    # Group A for cat1: [100, 200, 250, 175] -> mean = 181.25
    assert result["value__mean_cat1"][0] == pytest.approx(100 / 181.25)


def test_transform_zscore():
    """Test z-score scaling."""
    X = pl.DataFrame({"value": [10, 20, 30, 40, 50, 60], "group": ["A", "A", "A", "B", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["zscore"],
    )
    result = transformer.fit_transform(X)

    # Check zscore column is present
    assert "value__zscore_group" in result.columns

    # Group A: [10, 20, 30] -> mean=20, std=10
    # zscore for 10: (10-20)/10 = -1.0
    # zscore for 20: (20-20)/10 = 0.0
    # zscore for 30: (30-20)/10 = 1.0
    assert result["value__zscore_group"][0] == pytest.approx(-1.0)
    assert result["value__zscore_group"][1] == pytest.approx(0.0)
    assert result["value__zscore_group"][2] == pytest.approx(1.0)

    # Group B: [40, 50, 60] -> mean=50, std=10
    assert result["value__zscore_group"][3] == pytest.approx(-1.0)
    assert result["value__zscore_group"][4] == pytest.approx(0.0)
    assert result["value__zscore_group"][5] == pytest.approx(1.0)


def test_transform_minmax():
    """Test min-max scaling."""
    X = pl.DataFrame({"value": [10, 20, 30, 40, 50, 60], "group": ["A", "A", "A", "B", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["minmax"],
    )
    result = transformer.fit_transform(X)

    # Check minmax column is present
    assert "value__minmax_group" in result.columns

    # Group A: [10, 20, 30] -> min=10, max=30, range=20
    # minmax for 10: (10-10)/20 = 0.0
    # minmax for 20: (20-10)/20 = 0.5
    # minmax for 30: (30-10)/20 = 1.0
    assert result["value__minmax_group"][0] == pytest.approx(0.0)
    assert result["value__minmax_group"][1] == pytest.approx(0.5)
    assert result["value__minmax_group"][2] == pytest.approx(1.0)

    # Group B: [40, 50, 60] -> min=40, max=60, range=20
    assert result["value__minmax_group"][3] == pytest.approx(0.0)
    assert result["value__minmax_group"][4] == pytest.approx(0.5)
    assert result["value__minmax_group"][5] == pytest.approx(1.0)


def test_transform_minmax_with_negative_values():
    """Test min-max scaling with negative values."""
    X = pl.DataFrame({"value": [-20, -10, 0, 10, 20], "group": ["A", "A", "A", "A", "A"]})
    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["minmax"],
    )
    result = transformer.fit_transform(X)

    # min=-20, max=20, range=40
    # minmax for -20: (-20-(-20))/40 = 0.0
    # minmax for 0: (0-(-20))/40 = 0.5
    # minmax for 20: (20-(-20))/40 = 1.0
    assert result["value__minmax_group"][0] == pytest.approx(0.0)
    assert result["value__minmax_group"][2] == pytest.approx(0.5)
    assert result["value__minmax_group"][4] == pytest.approx(1.0)


def test_transform_all_scaling_functions():
    """Test all supported scaling functions."""
    X = pl.DataFrame({"value": [10, 20, 30, 40, 50, 60], "group": ["A", "A", "A", "B", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "median", "zscore", "minmax"],
    )
    result = transformer.fit_transform(X)

    # Check all scaling function columns are present
    assert "value__mean_group" in result.columns
    assert "value__median_group" in result.columns
    assert "value__zscore_group" in result.columns
    assert "value__minmax_group" in result.columns

    # Group A: [10, 20, 30]
    # mean=20, median=20, std=10, min=10, max=30
    # For value=10:
    assert result["value__mean_group"][0] == pytest.approx(10 / 20)  # 0.5
    assert result["value__median_group"][0] == pytest.approx(10 / 20)  # 0.5
    assert result["value__zscore_group"][0] == pytest.approx(-1.0)  # (10-20)/10
    assert result["value__minmax_group"][0] == pytest.approx(0.0)  # (10-10)/20


def test_transform_with_zero_denominator_mean():
    """Test handling of division by zero with fill_value for mean."""
    X = pl.DataFrame({"value": [0, 0, 20, 20], "group": ["A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
        fill_value=999.0,
    )
    result = transformer.fit_transform(X)

    # Group A mean: 0, so division by zero -> fill_value
    assert result["value__mean_group"][0] == 999.0
    assert result["value__mean_group"][1] == 999.0
    # Group B mean: 20, normal division
    assert result["value__mean_group"][2] == pytest.approx(1.0)


def test_transform_with_zero_std_zscore():
    """Test z-score when std is zero (all values identical)."""
    X = pl.DataFrame({"value": [10, 10, 10, 20, 20, 20], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["zscore"],
        fill_value=0.0,
    )
    result = transformer.fit_transform(X)

    # Group A: all values are 10, std=0 -> fill_value
    assert result["value__zscore_group"][0] == 0.0
    assert result["value__zscore_group"][1] == 0.0
    assert result["value__zscore_group"][2] == 0.0


def test_transform_with_zero_range_minmax():
    """Test min-max when all values are identical (range=0)."""
    X = pl.DataFrame({"value": [10, 10, 10, 20, 20, 20], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["minmax"],
        fill_value=0.5,
    )
    result = transformer.fit_transform(X)
    # Group A: all values are 10, range=0 -> fill_value
    assert result["value__minmax_group"][0] == 0.5
    assert result["value__minmax_group"][1] == 0.5
    assert result["value__minmax_group"][2] == 0.5


def test_transform_with_explicit_zero_in_median():
    """Test division by zero when median results in zero."""
    # Create a case where median could be zero
    X = pl.DataFrame({"value": [-10, 0, 10, 20], "group": ["A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["median"],
        fill_value=-999.0,
    )
    result = transformer.fit_transform(X)

    # Group A median: (-10 + 0) / 2 = -5, normal division works
    # Group B median: (10 + 20) / 2 = 15, normal division works
    assert result["value__median_group"][0] == pytest.approx(-10 / -5)
    assert result["value__median_group"][2] == pytest.approx(10 / 15)


def test_transform_with_null_values():
    """Test handling of null values in data."""
    X = pl.DataFrame({"value": [10, None, 30, 40, None], "group": ["A", "A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
        fill_value=-1.0,
    )
    result = transformer.fit_transform(X)

    # When numerator is null, result should be null (not fill_value)
    # Group A mean: (10 + 30) / 2 = 20 (nulls excluded from mean)
    assert result["value__mean_group"][0] == pytest.approx(10 / 20)
    assert (
        result["value__mean_group"][1] is None
        or pl.Series([result["value__mean_group"][1]]).is_null()[0]
    )


def test_transform_with_null_values_zscore():
    """Test z-score with null values."""
    X = pl.DataFrame({"value": [10, None, 30, 40, None], "group": ["A", "A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["zscore"],
        fill_value=0.0,
    )
    result = transformer.fit_transform(X)

    # Zscore should handle nulls
    assert "value__zscore_group" in result.columns


def test_transform_with_drop_columns():
    """Test dropping original numerical columns."""
    X = pl.DataFrame({"value": [10, 20, 30], "other": [1, 2, 3], "group": ["A", "A", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    # Original 'value' column should be dropped
    assert "value" not in result.columns
    assert "value__mean_group" in result.columns
    # Other columns should remain
    assert "other" in result.columns
    assert "group" in result.columns


def test_transform_with_custom_column_names():
    """Test using custom column names."""
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "median"],
        new_column_names=["custom_mean", "custom_median"],
    )
    result = transformer.fit_transform(X)

    assert "custom_mean" in result.columns
    assert "custom_median" in result.columns
    assert "value__mean_group" not in result.columns
    assert "value__median_group" not in result.columns


def test_validation_invalid_scaling_function():
    """Test validation error with invalid scaling function."""
    with pytest.raises(
        ValidationError,
        match="invalid_func is not in the predefined list of scaling functions",
    ):
        GroupScalingFeatures(
            subset=["value"],
            by=["group"],
            func=["mean", "invalid_func"],
        )


def test_validation_mismatched_new_column_names_length():
    """Test validation error when new_column_names length doesn't match."""
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})

    # Should create 1 num_col * 2 func = 2 features
    # But only providing 1 name
    with pytest.raises(
        ValueError,
        match="Length of new_column_names .* must match the total number of features created",
    ):
        GroupScalingFeatures(
            subset=["value"],
            by=["group"],
            func=["mean", "median"],
            new_column_names=["name1"],  # Should have 2 names
        )


def test_fit_return_self():
    """Test that fit returns self."""
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
    )

    result = transformer.fit(X)
    assert result is transformer


def test_column_mapping_generation():
    """Test that column mapping is correctly generated during fit."""
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "median"],
    )
    transformer.fit(X)

    # Check that column mapping is created
    assert len(transformer._column_mapping) == 2
    assert "value__mean_group" in transformer._column_mapping
    assert "value__median_group" in transformer._column_mapping


def test_median_scaling():
    """Test median scaling specifically."""
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["median"],
    )
    result = transformer.fit_transform(X)

    # Group A: [10, 20, 30] -> median = 20
    # Group B: [15, 25, 35] -> median = 25
    assert result["value__median_group"][0] == pytest.approx(10 / 20)
    assert result["value__median_group"][1] == pytest.approx(20 / 20)
    assert result["value__median_group"][3] == pytest.approx(15 / 25)


def test_empty_dataframe():
    """Test behavior with empty dataframe."""
    X = pl.DataFrame({"value": [], "group": []}, schema={"value": pl.Int64, "group": pl.Utf8})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    assert result.shape[0] == 0
    assert "value__mean_group" in result.columns


def test_single_row_dataframe():
    """Test behavior with single row dataframe."""
    X = pl.DataFrame({"value": [100], "group": ["A"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "median"],
    )
    result = transformer.fit_transform(X)

    # With one value, mean = median = value, so ratios should be 1.0
    assert result["value__mean_group"][0] == pytest.approx(1.0)
    assert result["value__median_group"][0] == pytest.approx(1.0)


def test_single_value_per_group_zscore():
    """Test z-score when groups have single values (std = 0)."""
    X = pl.DataFrame({"value": [10, 20, 30], "group": ["A", "B", "C"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["zscore"],
        fill_value=0.0,
    )
    result = transformer.fit_transform(X)

    # Std of single value is 0, which should trigger fill_value
    assert result["value__zscore_group"][0] == 0.0
    assert result["value__zscore_group"][1] == 0.0
    assert result["value__zscore_group"][2] == 0.0


def test_single_value_per_group_minmax():
    """Test min-max when groups have single values (range = 0)."""
    X = pl.DataFrame({"value": [10, 20, 30], "group": ["A", "B", "C"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["minmax"],
        fill_value=0.5,
    )
    result = transformer.fit_transform(X)

    # Range is 0 for single value groups, should use fill_value
    assert result["value__minmax_group"][0] == 0.5
    assert result["value__minmax_group"][1] == 0.5
    assert result["value__minmax_group"][2] == 0.5


def test_negative_values():
    """Test with negative values."""
    X = pl.DataFrame({"value": [-10, -20, 30, 40], "group": ["A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    # Group A: [-10, -20] -> mean = -15
    assert result["value__mean_group"][0] == pytest.approx(-10 / -15)


def test_negative_values_zscore():
    """Test z-score with negative values."""
    X = pl.DataFrame({"value": [-10, -20, 30, 40], "group": ["A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["zscore"],
    )
    result = transformer.fit_transform(X)

    # Group A: [-10, -20] -> mean=-15, std= 7.07
    # zscore for -10: (-10 - (-15)) / 7.07 ≈ 0.707
    assert result["value__zscore_group"][0] == pytest.approx(0.707, abs=0.01)


def test_float_values():
    """Test with float values."""
    X = pl.DataFrame({"value": [10.5, 20.7, 30.2, 40.8], "group": ["A", "A", "B", "B"]})

    transformer = GroupScalingFeatures(
        subset=["value"],
        by=["group"],
        func=["mean"],
    )
    result = transformer.fit_transform(X)

    # Group A: [10.5, 20.7] -> mean = 15.6
    assert result["value__mean_group"][0] == pytest.approx(10.5 / 15.6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
