import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import RowStatisticsFeatures


def test_transform_basic_single_group_single_func():
    """Test basic transformation with single column group and single aggregation."""
    X = pl.DataFrame({
        "A": [9, 9, 7],
        "B": [3, 4, 5],
        "C": [6, 7, 8]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"cluster_1": ["A", "B"]},
        func=["mean"]
    )
    result = transformer.fit_transform(X)

    # Row 0: mean([9, 3]) = 6.0
    # Row 1: mean([9, 4]) = 6.5
    # Row 2: mean([7, 5]) = 6.0
    assert "cluster_1__mean" in result.columns
    assert result["cluster_1__mean"][0] == pytest.approx(6.0)
    assert result["cluster_1__mean"][1] == pytest.approx(6.5)
    assert result["cluster_1__mean"][2] == pytest.approx(6.0)


def test_transform_single_group_multiple_func():
    """Test single column group with multiple aggregation functions."""
    X = pl.DataFrame({
        "A": [9, 9, 7],
        "B": [3, 4, 5],
        "C": [6, 7, 8]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"cluster_1": ["A", "B"]},
        func=["mean", "std"]
    )
    result = transformer.fit_transform(X)

    # Row 0: mean([9, 3]) = 6.0, std([9, 3]) ≈ 4.243
    assert "cluster_1__mean" in result.columns
    assert "cluster_1__std" in result.columns
    assert result["cluster_1__mean"][0] == pytest.approx(6.0)
    assert result["cluster_1__std"][0] == pytest.approx(4.242641, abs=1e-4)


def test_transform_multiple_groups():
    """Test with multiple column groups."""
    X = pl.DataFrame({
        "A": [9, 9, 7],
        "B": [3, 4, 5],
        "C": [6, 7, 8],
        "D": [1, 2, 3]
    })

    transformer = RowStatisticsFeatures(
        column_groups={
            "cluster_1": ["A", "B"],
            "cluster_2": ["C", "D"]
        },
        func=["min", "max"]
    )
    result = transformer.fit_transform(X)

    assert "cluster_1__min" in result.columns
    assert "cluster_1__max" in result.columns
    assert "cluster_2__min" in result.columns
    assert "cluster_2__max" in result.columns

    # Row 0: cluster_1 min([9, 3]) = 3, max([9, 3]) = 9
    assert result["cluster_1__min"][0] == 3
    assert result["cluster_1__max"][0] == 9
    # Row 0: cluster_2 min([6, 1]) = 1, max([6, 1]) = 6
    assert result["cluster_2__min"][0] == 1
    assert result["cluster_2__max"][0] == 6


def test_transform_all_aggregation_functions():
    """Test all supported aggregation functions."""
    X = pl.DataFrame({
        "A": [10, 20, 30],
        "B": [5, 10, 15],
        "C": [2, 4, 6]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B", "C"]},
        func=["min", "max", "mean", "median", "std", "range", "sum"]
    )
    result = transformer.fit_transform(X)

    # Check all columns are present
    assert "group__min" in result.columns
    assert "group__max" in result.columns
    assert "group__mean" in result.columns
    assert "group__median" in result.columns
    assert "group__std" in result.columns
    assert "group__range" in result.columns
    assert "group__sum" in result.columns

    # Row 0: [10, 5, 2]
    assert result["group__min"][0] == 2
    assert result["group__max"][0] == 10
    assert result["group__mean"][0] == pytest.approx(5.666667, abs=1e-5)
    assert result["group__median"][0] == 5.0
    assert result["group__range"][0] == 8  # 10 - 2
    assert result["group__sum"][0] == 17  # 10 + 5 + 2


def test_transform_range_aggregation():
    """Test range aggregation specifically (max - min)."""
    X = pl.DataFrame({
        "col1": [100, 50, 80],
        "col2": [90, 70, 60],
        "col3": [110, 55, 90]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"values": ["col1", "col2", "col3"]},
        func=["range"]
    )
    result = transformer.fit_transform(X)

    # Row 0: range([100, 90, 110]) = 110 - 90 = 20
    # Row 1: range([50, 70, 55]) = 70 - 50 = 20
    # Row 2: range([80, 60, 90]) = 90 - 60 = 30
    assert result["values__range"][0] == 20
    assert result["values__range"][1] == 20
    assert result["values__range"][2] == 30


def test_transform_with_three_columns():
    """Test with three columns in a group."""
    X = pl.DataFrame({
        "A": [100, 200, 150],
        "B": [50, 100, 75],
        "C": [25, 50, 30]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"amounts": ["A", "B", "C"]},
        func=["mean", "std"]
    )
    result = transformer.fit_transform(X)

    # Row 0: mean([100, 50, 25]) = 58.333...
    assert result["amounts__mean"][0] == pytest.approx(58.333333, abs=1e-5)
    assert result["amounts__std"][0] == pytest.approx(38.188, abs=1e-2)


def test_transform_with_custom_column_names():
    """Test using custom column names."""
    X = pl.DataFrame({
        "A": [10, 20, 30],
        "B": [5, 10, 15],
        "C": [2, 4, 6]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]},
        func=["mean", "std"],
        new_column_names=["avg_value", "std_value"]
    )
    result = transformer.fit_transform(X)

    assert "avg_value" in result.columns
    assert "std_value" in result.columns
    assert "group__mean" not in result.columns
    assert "group__std" not in result.columns

    # Row 0: mean([10, 5]) = 7.5
    assert result["avg_value"][0] == pytest.approx(7.5)


def test_transform_with_drop_columns_false():
    """Test that original columns are kept when drop_columns=False."""
    X = pl.DataFrame({
        "A": [10, 20],
        "B": [5, 10],
        "C": [100, 200]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]},
        func=["mean"],
        drop_columns=False
    )
    result = transformer.fit_transform(X)

    # Original columns should still be present
    assert "A" in result.columns
    assert "B" in result.columns
    assert "C" in result.columns
    assert "group__mean" in result.columns


def test_transform_with_drop_columns_true():
    """Test dropping original columns when drop_columns=True."""
    X = pl.DataFrame({
        "A": [10, 20],
        "B": [5, 10],
        "C": [100, 200]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]},
        func=["mean"],
        drop_columns=True
    )
    result = transformer.fit_transform(X)

    # Columns in the group should be dropped
    assert "A" not in result.columns
    assert "B" not in result.columns
    # Column not in the group should remain
    assert "C" in result.columns
    assert "group__mean" in result.columns


def test_transform_drop_columns_multiple_groups():
    """Test drop_columns with multiple groups - validation should catch invalid group."""
    X = pl.DataFrame({
        "A": [10, 20],
        "B": [5, 10],
        "C": [100, 200],
        "D": [50, 100]
    })

    # This should raise validation error because group2 has only 1 column
    with pytest.raises(ValidationError, match="must contain at least 2 columns"):
        RowStatisticsFeatures(
            column_groups={
                "group1": ["A", "B"],
                "group2": ["C"]  # This fails validation - need at least 2 cols
            },
            func=["mean"],
            drop_columns=True
        )


def test_transform_with_nulls():
    """Test handling of null values."""
    X = pl.DataFrame({
        "A": [10, None, 30],
        "B": [5, 10, None],
        "C": [2, 4, 6]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B", "C"]},
        func=["mean", "min", "max"]
    )
    result = transformer.fit_transform(X)

    # Polars list aggregations handle nulls by excluding them
    # Row 0: mean([10, 5, 2]) = 5.666...
    assert result["group__mean"][0] == pytest.approx(5.666667, abs=1e-5)
    # Row 1: mean([None, 10, 4]) = mean([10, 4]) = 7.0
    assert result["group__mean"][1] == pytest.approx(7.0)
    # Row 2: mean([30, None, 6]) = mean([30, 6]) = 18.0
    assert result["group__mean"][2] == pytest.approx(18.0)


def test_transform_preserves_column_order():
    """Test that new columns are added at the end."""
    X = pl.DataFrame({
        "A": [10, 20, 30],
        "B": [5, 10, 15],
        "C": [2, 4, 6],
        "D": [1, 2, 3]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]},
        func=["mean"]
    )
    result = transformer.fit_transform(X)

    # Original columns should come first
    assert list(result.columns) == ["A", "B", "C", "D", "group__mean"]


def test_validation_empty_column_groups():
    """Test validation error with empty column_groups."""
    with pytest.raises(ValueError, match="column_groups cannot be empty"):
        RowStatisticsFeatures(
            column_groups={},
            func=["mean"]
        )


def test_validation_single_column_in_group():
    """Test validation error when group has only 1 column."""
    with pytest.raises(
        ValueError,
        match="must contain at least 2 columns for row-level aggregation"
    ):
        RowStatisticsFeatures(
            column_groups={"group": ["A"]},
            func=["mean"]
        )


def test_validation_invalid_aggregation_function():
    """Test validation error with invalid aggregation function."""
    with pytest.raises(
        ValidationError,
        match="invalid_func is not in the predefined list of aggregation functions"
    ):
        RowStatisticsFeatures(
            column_groups={"group": ["A", "B"]},
            func=["mean", "invalid_func"]
        )


def test_validation_mismatched_new_column_names_length():
    """Test validation error when new_column_names length doesn't match."""
    with pytest.raises(
        ValueError,
        match="Length of new_column_names .* must match the total number of features created"
    ):
        RowStatisticsFeatures(
            column_groups={"group1": ["A", "B"], "group2": ["C", "D"]},
            func=["mean", "std"],  # 2 groups × 2 func = 4 features
            new_column_names=["name1", "name2"]  # Only 2 names provided
        )


def test_validation_column_groups_not_list():
    """Test validation error when column_groups value is not a list."""
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        RowStatisticsFeatures(
            column_groups={"group": "A"},  # String instead of list
            func=["mean"]
        )


def test_fit_transform_equivalence():
    """Test that fit().transform() equals fit_transform()."""
    X = pl.DataFrame({
        "A": [10, 20, 30],
        "B": [5, 10, 15],
        "C": [2, 4, 6]
    })

    transformer1 = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]},
        func=["mean", "std"]
    )
    result1 = transformer1.fit_transform(X)

    transformer2 = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]},
        func=["mean", "std"]
    )
    transformer2.fit(X)
    result2 = transformer2.transform(X)

    assert_frame_equal(result1, result2)


def test_multiple_groups_different_columns():
    """Test multiple groups with completely different columns."""
    X = pl.DataFrame({
        "A": [10, 20, 30],
        "B": [5, 10, 15],
        "C": [100, 200, 300],
        "D": [50, 100, 150],
        "E": [1, 2, 3]
    })

    transformer = RowStatisticsFeatures(
        column_groups={
            "group1": ["A", "B"],
            "group2": ["C", "D"],
        },
        func=["mean", "range"]
    )
    result = transformer.fit_transform(X)

    # Should have 2 groups × 2 func = 4 new columns
    assert "group1__mean" in result.columns
    assert "group1__range" in result.columns
    assert "group2__mean" in result.columns
    assert "group2__range" in result.columns

    # Row 0: group1 mean([10, 5]) = 7.5, range = 5
    assert result["group1__mean"][0] == pytest.approx(7.5)
    assert result["group1__range"][0] == 5
    # Row 0: group2 mean([100, 50]) = 75, range = 50
    assert result["group2__mean"][0] == pytest.approx(75.0)
    assert result["group2__range"][0] == 50


def test_median_aggregation():
    """Test median aggregation specifically."""
    X = pl.DataFrame({
        "A": [1, 10, 100],
        "B": [2, 20, 200],
        "C": [3, 30, 300]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B", "C"]},
        func=["median"]
    )
    result = transformer.fit_transform(X)

    # Row 0: median([1, 2, 3]) = 2
    # Row 1: median([10, 20, 30]) = 20
    # Row 2: median([100, 200, 300]) = 200
    assert result["group__median"][0] == 2
    assert result["group__median"][1] == 20
    assert result["group__median"][2] == 200


def test_sum_aggregation():
    """Test sum aggregation specifically."""
    X = pl.DataFrame({
        "A": [10, 20, 30],
        "B": [5, 10, 15],
        "C": [2, 4, 6]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B", "C"]},
        func=["sum"]
    )
    result = transformer.fit_transform(X)

    # Row 0: sum([10, 5, 2]) = 17
    # Row 1: sum([20, 10, 4]) = 34
    # Row 2: sum([30, 15, 6]) = 51
    assert result["group__sum"][0] == 17
    assert result["group__sum"][1] == 34
    assert result["group__sum"][2] == 51


def test_fraud_detection_use_case():
    """Test realistic fraud detection scenario with verification fields."""
    X = pl.DataFrame({
        "card_cvv_match": [1, 0, 1, 1],
        "card_addr_match": [1, 1, 0, 1],
        "card_zip_match": [1, 1, 1, 0],
        "transaction_id": [101, 102, 103, 104]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"verification": ["card_cvv_match", "card_addr_match", "card_zip_match"]},
        func=["mean", "std", "min"],
        drop_columns=False
    )
    result = transformer.fit_transform(X)

    # Legitimate transaction (all matches = 1): mean=1, std=0, min=1
    assert result["verification__mean"][0] == pytest.approx(1.0)
    assert result["verification__std"][0] == pytest.approx(0.0)
    assert result["verification__min"][0] == 1

    # Suspicious transactions have lower mean, higher std
    assert result["verification__mean"][1] < 1.0
    assert result["verification__std"][1] > 0.0


def test_works_with_float_columns():
    """Test that it works with float columns."""
    X = pl.DataFrame({
        "A": [10.5, 20.3, 30.7],
        "B": [5.2, 10.8, 15.1],
        "C": [2.1, 4.9, 6.3]
    })

    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]},
        func=["mean", "std"]
    )
    result = transformer.fit_transform(X)

    assert "group__mean" in result.columns
    assert "group__std" in result.columns
    # Row 0: mean([10.5, 5.2]) ≈ 7.85
    assert result["group__mean"][0] == pytest.approx(7.85)
