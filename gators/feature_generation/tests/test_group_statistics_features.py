import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import GroupStatisticsFeatures

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def X_basic():
    return pl.DataFrame(
        {
            "amount": [100, 200, 150, 300, 250],
            "cat1": ["A", "A", "B", "B", "A"],
            "cat2": ["X", "Y", "X", "X", "X"],
        }
    )


@pytest.fixture
def X_two_groups():
    return pl.DataFrame(
        {"value": [10, 20, 30, 40, 50, 60], "group": ["A", "A", "A", "B", "B", "B"]}
    )


# ---------------------------------------------------------------------------
# Absolute statistics
# ---------------------------------------------------------------------------


def test_transform_basic_single_group_single_agg(X_basic):
    """Basic transformation: single groupby column, single aggregation."""
    transformer = GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["mean"])
    result = transformer.fit_transform(X_basic)

    assert "mean_amount__per_cat1" in result.columns
    # Group A: [100, 200, 250] -> mean = 183.333...
    assert result["mean_amount__per_cat1"][0] == pytest.approx(183.333333, abs=1e-5)
    # Group B: [150, 300] -> mean = 225.0
    assert result["mean_amount__per_cat1"][2] == pytest.approx(225.0)


def test_transform_multiple_aggregations():
    """Multiple aggregation functions in one call."""
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["mean", "min", "max", "count"]
    )
    result = transformer.fit_transform(X)

    assert "mean_value__per_group" in result.columns
    assert "min_value__per_group" in result.columns
    assert "max_value__per_group" in result.columns
    assert "count_value__per_group" in result.columns

    # Group A: [10, 20]
    assert result["mean_value__per_group"][0] == pytest.approx(15.0)
    assert result["min_value__per_group"][0] == 10
    assert result["max_value__per_group"][0] == 20
    assert result["count_value__per_group"][0] == 2


def test_transform_multiple_numerical_columns():
    """Multiple numerical columns."""
    X = pl.DataFrame(
        {"col1": [100, 200, 150, 300], "col2": [50, 100, 75, 150], "group": ["A", "A", "B", "B"]}
    )
    transformer = GroupStatisticsFeatures(subset=["col1", "col2"], by=["group"], func=["mean"])
    result = transformer.fit_transform(X)

    assert "mean_col1__per_group" in result.columns
    assert "mean_col2__per_group" in result.columns
    assert result["mean_col1__per_group"][0] == pytest.approx(150.0)
    assert result["mean_col2__per_group"][0] == pytest.approx(75.0)


def test_transform_multi_column_groupby():
    """Multiple separate groupby columns."""
    X = pl.DataFrame(
        {
            "value": [100, 200, 150, 300, 250, 175],
            "amount": [50, 100, 75, 150, 125, 88],
            "cat1": ["A", "A", "B", "B", "A", "A"],
            "cat2": ["X", "Y", "X", "X", "X", "Y"],
        }
    )
    transformer = GroupStatisticsFeatures(
        subset=["value", "amount"], by=["cat1", "cat2"], func=["mean", "std"]
    )
    result = transformer.fit_transform(X)

    # 2 numerical × 2 groupby × 2 func = 8 features
    for col in [
        "mean_value__per_cat1",
        "mean_value__per_cat2",
        "mean_amount__per_cat1",
        "mean_amount__per_cat2",
        "std_value__per_cat1",
        "std_value__per_cat2",
        "std_amount__per_cat1",
        "std_amount__per_cat2",
    ]:
        assert col in result.columns

    # Group A for cat1: [100, 200, 250, 175] -> mean = 181.25
    assert result["mean_value__per_cat1"][0] == pytest.approx(181.25)


def test_transform_all_absolute_aggregations(X_two_groups):
    """All absolute aggregation functions."""
    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean", "std", "median", "min", "max", "sum", "count"],
    )
    result = transformer.fit_transform(X_two_groups)

    # Group A: [10, 20, 30]
    assert result["sum_value__per_group"][0] == pytest.approx(60)
    assert result["min_value__per_group"][0] == 10
    assert result["max_value__per_group"][0] == 30
    assert result["count_value__per_group"][0] == 3
    assert result["median_value__per_group"][0] == pytest.approx(20)


def test_transform_std_aggregation():
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["std"])
    result = transformer.fit_transform(X)
    assert result["std_value__per_group"][0] == pytest.approx(10.0, abs=0.1)


def test_transform_median_aggregation():
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["median"])
    result = transformer.fit_transform(X)
    assert result["median_value__per_group"][0] == pytest.approx(20.0)
    assert result["median_value__per_group"][3] == pytest.approx(25.0)


def test_transform_sum_aggregation():
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["sum"])
    result = transformer.fit_transform(X)
    assert result["sum_value__per_group"][0] == pytest.approx(30.0)
    assert result["sum_value__per_group"][2] == pytest.approx(70.0)


def test_transform_count_aggregation():
    X = pl.DataFrame({"value": [10, 20, 30, 40, 50], "group": ["A", "A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["count"])
    result = transformer.fit_transform(X)
    assert result["count_value__per_group"][0] == 3
    assert result["count_value__per_group"][3] == 2


def test_transform_min_max_aggregation():
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["min", "max"])
    result = transformer.fit_transform(X)
    assert result["min_value__per_group"][0] == 10
    assert result["max_value__per_group"][0] == 30
    assert result["min_value__per_group"][3] == 15
    assert result["max_value__per_group"][3] == 35


def test_transform_range_aggregation():
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["range"])
    result = transformer.fit_transform(X)
    assert result["range_value__per_group"][0] == pytest.approx(20.0)
    assert result["range_value__per_group"][3] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Relative statistics
# ---------------------------------------------------------------------------


def test_transform_mean_ratio():
    """mean_ratio = value / group_mean."""
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["mean_ratio"])
    result = transformer.fit_transform(X)

    assert "mean_ratio_value__per_group" in result.columns
    # Group A mean = 15
    assert result["mean_ratio_value__per_group"][0] == pytest.approx(10 / 15)
    assert result["mean_ratio_value__per_group"][1] == pytest.approx(20 / 15)


def test_transform_median_ratio():
    """median_ratio = value / group_median."""
    X = pl.DataFrame({"value": [10, 20, 30, 15, 25, 35], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["median_ratio"])
    result = transformer.fit_transform(X)

    # Group A median = 20, Group B median = 25
    assert result["median_ratio_value__per_group"][0] == pytest.approx(10 / 20)
    assert result["median_ratio_value__per_group"][1] == pytest.approx(20 / 20)
    assert result["median_ratio_value__per_group"][3] == pytest.approx(15 / 25)


def test_transform_zscore(X_two_groups):
    """zscore = (value - group_mean) / group_std."""
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["zscore"])
    result = transformer.fit_transform(X_two_groups)

    assert "zscore_value__per_group" in result.columns
    # Group A: [10, 20, 30] -> mean=20, std=10
    assert result["zscore_value__per_group"][0] == pytest.approx(-1.0)
    assert result["zscore_value__per_group"][1] == pytest.approx(0.0)
    assert result["zscore_value__per_group"][2] == pytest.approx(1.0)
    # Group B: [40, 50, 60] -> mean=50, std=10
    assert result["zscore_value__per_group"][3] == pytest.approx(-1.0)
    assert result["zscore_value__per_group"][4] == pytest.approx(0.0)
    assert result["zscore_value__per_group"][5] == pytest.approx(1.0)


def test_transform_minmax(X_two_groups):
    """minmax = (value - group_min) / (group_max - group_min)."""
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["minmax"])
    result = transformer.fit_transform(X_two_groups)

    assert "minmax_value__per_group" in result.columns
    # Group A: [10, 20, 30] -> min=10, max=30, range=20
    assert result["minmax_value__per_group"][0] == pytest.approx(0.0)
    assert result["minmax_value__per_group"][1] == pytest.approx(0.5)
    assert result["minmax_value__per_group"][2] == pytest.approx(1.0)
    # Group B: [40, 50, 60]
    assert result["minmax_value__per_group"][3] == pytest.approx(0.0)
    assert result["minmax_value__per_group"][4] == pytest.approx(0.5)
    assert result["minmax_value__per_group"][5] == pytest.approx(1.0)


def test_transform_minmax_with_negative_values():
    X = pl.DataFrame({"value": [-20, -10, 0, 10, 20], "group": ["A"] * 5})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["minmax"])
    result = transformer.fit_transform(X)
    # min=-20, max=20, range=40
    assert result["minmax_value__per_group"][0] == pytest.approx(0.0)
    assert result["minmax_value__per_group"][2] == pytest.approx(0.5)
    assert result["minmax_value__per_group"][4] == pytest.approx(1.0)


def test_transform_all_relative_functions(X_two_groups):
    """All relative functions together."""
    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean_ratio", "median_ratio", "zscore", "minmax"],
    )
    result = transformer.fit_transform(X_two_groups)

    for col in [
        "mean_ratio_value__per_group",
        "median_ratio_value__per_group",
        "zscore_value__per_group",
        "minmax_value__per_group",
    ]:
        assert col in result.columns

    # Group A: [10, 20, 30] -> mean=median=20, std=10, min=10, max=30
    assert result["mean_ratio_value__per_group"][0] == pytest.approx(10 / 20)
    assert result["median_ratio_value__per_group"][0] == pytest.approx(10 / 20)
    assert result["zscore_value__per_group"][0] == pytest.approx(-1.0)
    assert result["minmax_value__per_group"][0] == pytest.approx(0.0)


def test_transform_mix_absolute_and_relative():
    """Absolute and relative functions can be combined in one call."""
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["mean", "zscore", "minmax"]
    )
    result = transformer.fit_transform(X)

    assert "mean_value__per_group" in result.columns
    assert "zscore_value__per_group" in result.columns
    assert "minmax_value__per_group" in result.columns
    # mean is absolute (group mean)
    assert result["mean_value__per_group"][0] == pytest.approx(15.0)
    # zscore: Group A = [10, 20] -> mean=15, std=7.07; zscore(10) = -0.707
    assert result["zscore_value__per_group"][0] == pytest.approx(-0.707, abs=0.01)


# ---------------------------------------------------------------------------
# fill_value (zero-denominator handling for relative functions)
# ---------------------------------------------------------------------------


def test_fill_value_mean_ratio_zero_denominator():
    """fill_value used when group mean is zero."""
    X = pl.DataFrame({"value": [0, 0, 20, 20], "group": ["A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["mean_ratio"], fill_value=999.0
    )
    result = transformer.fit_transform(X)
    assert result["mean_ratio_value__per_group"][0] == 999.0
    assert result["mean_ratio_value__per_group"][1] == 999.0
    assert result["mean_ratio_value__per_group"][2] == pytest.approx(1.0)


def test_fill_value_median_ratio_zero_denominator():
    """fill_value used when group median is zero."""
    X = pl.DataFrame({"value": [-5, 0, 5, 10, 20], "group": ["A", "A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["median_ratio"], fill_value=-999.0
    )
    result = transformer.fit_transform(X)
    # Group A median = 0 -> fill_value
    assert result["median_ratio_value__per_group"][0] == -999.0


def test_fill_value_zscore_zero_std():
    """fill_value used when group std is zero (all identical values)."""
    X = pl.DataFrame({"value": [10, 10, 10, 20, 20, 20], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["zscore"], fill_value=0.0
    )
    result = transformer.fit_transform(X)
    assert result["zscore_value__per_group"][0] == 0.0
    assert result["zscore_value__per_group"][1] == 0.0
    assert result["zscore_value__per_group"][2] == 0.0


def test_fill_value_minmax_zero_range():
    """fill_value used when group range is zero (all identical values)."""
    X = pl.DataFrame({"value": [10, 10, 10, 20, 20, 20], "group": ["A", "A", "A", "B", "B", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["minmax"], fill_value=0.5
    )
    result = transformer.fit_transform(X)
    assert result["minmax_value__per_group"][0] == 0.5
    assert result["minmax_value__per_group"][1] == 0.5
    assert result["minmax_value__per_group"][2] == 0.5


# ---------------------------------------------------------------------------
# Shared behaviours (drop_columns, custom names, nulls, edge cases)
# ---------------------------------------------------------------------------


def test_transform_with_drop_columns():
    X = pl.DataFrame({"value": [10, 20, 30], "other": [1, 2, 3], "group": ["A", "A", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["mean"], drop_columns=True
    )
    result = transformer.fit_transform(X)
    assert "value" not in result.columns
    assert "mean_value__per_group" in result.columns
    assert "other" in result.columns
    assert "group" in result.columns


def test_transform_with_drop_columns_relative():
    X = pl.DataFrame({"value": [10, 20, 30], "other": [1, 2, 3], "group": ["A", "A", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["zscore"], drop_columns=True
    )
    result = transformer.fit_transform(X)
    assert "value" not in result.columns
    assert "zscore_value__per_group" in result.columns
    assert "other" in result.columns


def test_transform_with_custom_column_names():
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


def test_transform_with_custom_column_names_relative():
    X = pl.DataFrame({"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"],
        by=["group"],
        func=["mean_ratio", "zscore"],
        new_column_names=["ratio_col", "zscore_col"],
    )
    result = transformer.fit_transform(X)
    assert "ratio_col" in result.columns
    assert "zscore_col" in result.columns
    assert "mean_ratio_value__per_group" not in result.columns


def test_with_null_values():
    X = pl.DataFrame({"value": [10, None, 30, 40, None], "group": ["A", "A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["mean", "count"])
    result = transformer.fit_transform(X)
    # Nulls excluded from mean and count
    assert result["mean_value__per_group"][0] == pytest.approx(20.0)
    assert result["count_value__per_group"][0] == 2


def test_with_null_values_zscore():
    X = pl.DataFrame({"value": [10, None, 30, 40, None], "group": ["A", "A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["zscore"], fill_value=0.0
    )
    result = transformer.fit_transform(X)
    assert "zscore_value__per_group" in result.columns


def test_negative_values():
    X = pl.DataFrame({"value": [-10, -20, 30, 40], "group": ["A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["mean", "sum"])
    result = transformer.fit_transform(X)
    assert result["mean_value__per_group"][0] == pytest.approx(-15.0)
    assert result["sum_value__per_group"][0] == pytest.approx(-30.0)


def test_float_values():
    X = pl.DataFrame({"value": [10.5, 20.7, 30.2, 40.8], "group": ["A", "A", "B", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["mean"])
    result = transformer.fit_transform(X)
    assert result["mean_value__per_group"][0] == pytest.approx(15.6)


def test_empty_dataframe():
    X = pl.DataFrame({"value": [], "group": []}, schema={"value": pl.Int64, "group": pl.Utf8})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["mean"])
    result = transformer.fit_transform(X)
    assert result.shape[0] == 0
    assert "mean_value__per_group" in result.columns


def test_single_row_dataframe():
    X = pl.DataFrame({"value": [100], "group": ["A"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["mean", "min", "max"]
    )
    result = transformer.fit_transform(X)
    assert result["mean_value__per_group"][0] == pytest.approx(100.0)
    assert result["min_value__per_group"][0] == 100
    assert result["max_value__per_group"][0] == 100


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_invalid_aggregation():
    with pytest.raises(
        ValidationError,
        match="invalid_agg is not in the predefined list of aggregation functions",
    ):
        GroupStatisticsFeatures(subset=["value"], by=["group"], func=["mean", "invalid_agg"])


def test_validation_mismatched_new_column_names_length():
    with pytest.raises(
        ValueError,
        match="Length of new_column_names .* must match the total number of features created",
    ):
        GroupStatisticsFeatures(
            subset=["value"],
            by=["group"],
            func=["mean", "max"],
            new_column_names=["name1"],
        )


# ---------------------------------------------------------------------------
# fit / column mapping
# ---------------------------------------------------------------------------


def test_fit_return_self():
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["mean"])
    assert transformer.fit(X) is transformer


def test_column_mapping_generation_absolute():
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})
    transformer = GroupStatisticsFeatures(subset=["value"], by=["group"], func=["mean", "max"])
    transformer.fit(X)
    assert len(transformer._column_mapping) == 2
    assert "mean_value__per_group" in transformer._column_mapping
    assert "max_value__per_group" in transformer._column_mapping


def test_column_mapping_generation_relative():
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})
    transformer = GroupStatisticsFeatures(
        subset=["value"], by=["group"], func=["mean_ratio", "zscore"]
    )
    transformer.fit(X)
    assert len(transformer._column_mapping) == 2
    assert "mean_ratio_value__per_group" in transformer._column_mapping
    assert "zscore_value__per_group" in transformer._column_mapping


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
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
