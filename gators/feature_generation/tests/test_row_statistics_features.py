import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import RowStatisticsFeatures

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def X_basic():
    return pl.DataFrame({"A": [9, 9, 7], "B": [3, 4, 5], "C": [6, 7, 8]})


@pytest.fixture
def X_four_cols():
    return pl.DataFrame({"A": [9, 9, 7], "B": [3, 4, 5], "C": [6, 7, 8], "D": [1, 2, 3]})


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_transform_single_group_single_func(X_basic):
    transformer = RowStatisticsFeatures(column_groups={"cluster_1": ["A", "B"]}, func=["mean"])
    result = transformer.fit_transform(X_basic)

    assert "cluster_1__mean" in result.columns
    assert result["cluster_1__mean"][0] == pytest.approx(6.0)
    assert result["cluster_1__mean"][1] == pytest.approx(6.5)
    assert result["cluster_1__mean"][2] == pytest.approx(6.0)


def test_transform_single_group_multiple_func(X_basic):
    transformer = RowStatisticsFeatures(
        column_groups={"cluster_1": ["A", "B"]}, func=["mean", "std"]
    )
    result = transformer.fit_transform(X_basic)

    assert "cluster_1__mean" in result.columns
    assert "cluster_1__std" in result.columns
    assert result["cluster_1__mean"][0] == pytest.approx(6.0)
    assert result["cluster_1__std"][0] == pytest.approx(4.242641, abs=1e-4)


def test_transform_multiple_groups(X_four_cols):
    transformer = RowStatisticsFeatures(
        column_groups={"cluster_1": ["A", "B"], "cluster_2": ["C", "D"]},
        func=["min", "max"],
    )
    result = transformer.fit_transform(X_four_cols)

    assert "cluster_1__min" in result.columns
    assert "cluster_1__max" in result.columns
    assert "cluster_2__min" in result.columns
    assert "cluster_2__max" in result.columns

    # Row 0: cluster_1 = [9, 3] → min=3, max=9
    assert result["cluster_1__min"][0] == 3
    assert result["cluster_1__max"][0] == 9
    # Row 0: cluster_2 = [6, 1] → min=1, max=6
    assert result["cluster_2__min"][0] == 1
    assert result["cluster_2__max"][0] == 6


def test_transform_three_columns_in_group():
    X = pl.DataFrame({"A": [100, 200, 150], "B": [50, 100, 75], "C": [25, 50, 30]})
    transformer = RowStatisticsFeatures(
        column_groups={"amounts": ["A", "B", "C"]}, func=["mean", "std"]
    )
    result = transformer.fit_transform(X)

    assert result["amounts__mean"][0] == pytest.approx(58.333333, abs=1e-5)
    assert result["amounts__std"][0] == pytest.approx(38.188, abs=1e-2)


def test_transform_all_aggregation_functions():
    X = pl.DataFrame({"A": [10, 20, 30], "B": [5, 10, 15], "C": [2, 4, 6]})
    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B", "C"]},
        func=["min", "max", "mean", "median", "std", "range", "sum", "count"],
    )
    result = transformer.fit_transform(X)

    for f in ["min", "max", "mean", "median", "std", "range", "sum", "count"]:
        assert f"group__{f}" in result.columns

    # Row 0: [10, 5, 2]
    assert result["group__min"][0] == 2
    assert result["group__max"][0] == 10
    assert result["group__mean"][0] == pytest.approx(5.666667, abs=1e-5)
    assert result["group__median"][0] == 5.0
    assert result["group__range"][0] == 8
    assert result["group__sum"][0] == 17
    assert result["group__count"][0] == 3


def test_transform_range_aggregation():
    X = pl.DataFrame({"col1": [100, 50, 80], "col2": [90, 70, 60], "col3": [110, 55, 90]})
    transformer = RowStatisticsFeatures(
        column_groups={"values": ["col1", "col2", "col3"]}, func=["range"]
    )
    result = transformer.fit_transform(X)

    assert result["values__range"][0] == 20  # 110 - 90
    assert result["values__range"][1] == 20  # 70 - 50
    assert result["values__range"][2] == 30  # 90 - 60


def test_transform_median_aggregation():
    X = pl.DataFrame({"A": [1, 10, 100], "B": [2, 20, 200], "C": [3, 30, 300]})
    transformer = RowStatisticsFeatures(column_groups={"group": ["A", "B", "C"]}, func=["median"])
    result = transformer.fit_transform(X)

    assert result["group__median"][0] == 2
    assert result["group__median"][1] == 20
    assert result["group__median"][2] == 200


def test_transform_sum_aggregation():
    X = pl.DataFrame({"A": [10, 20, 30], "B": [5, 10, 15], "C": [2, 4, 6]})
    transformer = RowStatisticsFeatures(column_groups={"group": ["A", "B", "C"]}, func=["sum"])
    result = transformer.fit_transform(X)

    assert result["group__sum"][0] == 17
    assert result["group__sum"][1] == 34
    assert result["group__sum"][2] == 51


def test_transform_count_aggregation():
    X = pl.DataFrame({"A": [10, None, 30], "B": [5, 10, None], "C": [2, None, 6]})
    transformer = RowStatisticsFeatures(column_groups={"group": ["A", "B", "C"]}, func=["count"])
    result = transformer.fit_transform(X)

    assert result["group__count"][0] == 3  # no nulls
    assert result["group__count"][1] == 1  # only B non-null
    assert result["group__count"][2] == 2  # A and C non-null


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------


def test_auto_naming_column_order():
    X = pl.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})
    transformer = RowStatisticsFeatures(column_groups={"grp": ["A", "B"]}, func=["mean"])
    result = transformer.fit_transform(X)
    assert list(result.columns) == ["A", "B", "C", "D", "grp__mean"]


def test_auto_naming_multiple_groups():
    X = pl.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})
    transformer = RowStatisticsFeatures(
        column_groups={"g1": ["A", "B"], "g2": ["C", "D"]}, func=["mean", "std"]
    )
    result = transformer.fit_transform(X)
    assert "g1__mean" in result.columns
    assert "g1__std" in result.columns
    assert "g2__mean" in result.columns
    assert "g2__std" in result.columns


def test_custom_column_names():
    X = pl.DataFrame({"A": [10, 20, 30], "B": [5, 10, 15], "C": [2, 4, 6]})
    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]},
        func=["mean", "std"],
        new_column_names=["avg_value", "std_value"],
    )
    result = transformer.fit_transform(X)

    assert "avg_value" in result.columns
    assert "std_value" in result.columns
    assert "group__mean" not in result.columns
    assert result["avg_value"][0] == pytest.approx(7.5)


def test_custom_column_names_multiple_groups():
    X = pl.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})
    transformer = RowStatisticsFeatures(
        column_groups={"g1": ["A", "B"], "g2": ["C", "D"]},
        func=["mean"],
        new_column_names=["ab_mean", "cd_mean"],
    )
    result = transformer.fit_transform(X)
    assert "ab_mean" in result.columns
    assert "cd_mean" in result.columns


# ---------------------------------------------------------------------------
# drop_columns
# ---------------------------------------------------------------------------


def test_drop_columns_false(X_basic):
    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]}, func=["mean"], drop_columns=False
    )
    result = transformer.fit_transform(X_basic)

    assert "A" in result.columns
    assert "B" in result.columns
    assert "C" in result.columns
    assert "group__mean" in result.columns


def test_drop_columns_true(X_basic):
    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B"]}, func=["mean"], drop_columns=True
    )
    result = transformer.fit_transform(X_basic)

    assert "A" not in result.columns
    assert "B" not in result.columns
    assert "C" in result.columns  # not in any group → preserved
    assert "group__mean" in result.columns


def test_drop_columns_multiple_groups(X_four_cols):
    transformer = RowStatisticsFeatures(
        column_groups={"g1": ["A", "B"], "g2": ["C", "D"]},
        func=["mean"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X_four_cols)

    for col in ["A", "B", "C", "D"]:
        assert col not in result.columns
    assert "g1__mean" in result.columns
    assert "g2__mean" in result.columns


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


def test_nulls_excluded_from_aggregation():
    X = pl.DataFrame({"A": [10, None, 30], "B": [5, 10, None], "C": [2, 4, 6]})
    transformer = RowStatisticsFeatures(
        column_groups={"group": ["A", "B", "C"]}, func=["mean", "min", "max"]
    )
    result = transformer.fit_transform(X)

    assert result["group__mean"][0] == pytest.approx(5.666667, abs=1e-5)
    # Row 1: [None, 10, 4] → mean of [10, 4] = 7.0
    assert result["group__mean"][1] == pytest.approx(7.0)
    # Row 2: [30, None, 6] → mean of [30, 6] = 18.0
    assert result["group__mean"][2] == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# fit / transform equivalence
# ---------------------------------------------------------------------------


def test_fit_transform_equivalence(X_basic):
    t1 = RowStatisticsFeatures(column_groups={"g": ["A", "B"]}, func=["mean", "std"])
    r1 = t1.fit_transform(X_basic)

    t2 = RowStatisticsFeatures(column_groups={"g": ["A", "B"]}, func=["mean", "std"])
    t2.fit(X_basic)
    r2 = t2.transform(X_basic)

    assert_frame_equal(r1, r2)


def test_fit_returns_self(X_basic):
    transformer = RowStatisticsFeatures(column_groups={"g": ["A", "B"]}, func=["mean"])
    assert transformer.fit(X_basic) is transformer


# ---------------------------------------------------------------------------
# Fraud detection use-case
# ---------------------------------------------------------------------------


def test_fraud_detection_verification_fields():
    X = pl.DataFrame(
        {
            "card_cvv_match": [1, 0, 1, 1],
            "card_addr_match": [1, 1, 0, 1],
            "card_zip_match": [1, 1, 1, 0],
            "transaction_id": [101, 102, 103, 104],
        }
    )
    transformer = RowStatisticsFeatures(
        column_groups={"verification": ["card_cvv_match", "card_addr_match", "card_zip_match"]},
        func=["mean", "std", "min"],
        new_column_names=["verif__mean", "verif__std", "verif__min"],
    )
    result = transformer.fit_transform(X)

    # Legitimate (all 1s): mean=1, std=0, min=1
    assert result["verif__mean"][0] == pytest.approx(1.0)
    assert result["verif__std"][0] == pytest.approx(0.0)
    assert result["verif__min"][0] == 1
    # Suspicious: lower mean, higher std
    assert result["verif__mean"][1] < 1.0
    assert result["verif__std"][1] > 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_empty_column_groups():
    with pytest.raises(ValueError, match="column_groups cannot be empty"):
        RowStatisticsFeatures(column_groups={}, func=["mean"])


def test_validation_single_column_in_group():
    with pytest.raises(ValueError, match="must contain at least 2 columns"):
        RowStatisticsFeatures(column_groups={"group": ["A"]}, func=["mean"])


def test_validation_invalid_aggregation_function():
    with pytest.raises(
        ValidationError, match="invalid_func is not in the predefined list of aggregation functions"
    ):
        RowStatisticsFeatures(column_groups={"group": ["A", "B"]}, func=["mean", "invalid_func"])


def test_validation_mismatched_new_column_names_length():
    with pytest.raises(
        ValueError,
        match="Length of new_column_names .* must match the total number of features created",
    ):
        RowStatisticsFeatures(
            column_groups={"g1": ["A", "B"], "g2": ["C", "D"]},
            func=["mean", "std"],  # 2 groups × 2 func = 4 features
            new_column_names=["n1", "n2"],  # only 2 provided
        )


def test_validation_column_groups_value_not_list():
    with pytest.raises((TypeError, ValidationError)):
        RowStatisticsFeatures(
            column_groups={"group": "A"},  # type: ignore[arg-type]
            func=["mean"],
        )


def test_float_columns():
    X = pl.DataFrame({"A": [10.5, 20.3], "B": [5.2, 10.8], "C": [2.1, 4.9]})
    transformer = RowStatisticsFeatures(column_groups={"group": ["A", "B"]}, func=["mean", "std"])
    result = transformer.fit_transform(X)

    assert "group__mean" in result.columns
    assert result["group__mean"][0] == pytest.approx(7.85)


def test_check_column_groups_non_list_raises_type_error():
    """check_column_groups validator raises TypeError when a value is not a list."""
    with pytest.raises(TypeError, match="must be a list"):
        RowStatisticsFeatures.check_column_groups({"key": "not_a_list"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
