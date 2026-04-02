import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.imputers.groupby_imputer import GroupByImputer


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with groups and missing values."""
    return pl.DataFrame(
        {
            "district": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "value1": [1.0, 2.0, None, 4.0, None, 6.0, None, 8.0],
            "value2": [10.0, None, 30.0, None, 50.0, 60.0, 70.0, None],
            "category": ["x", "y", "x", "y", "x", "y", "x", "y"],
        }
    )


def test_groupby_imputer_median(sample_dataframe):
    """Test GroupByImputer with median strategy."""
    imputer = GroupByImputer(group_by_column="district", strategy="median", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    # District A: value1 median = 1.5, value2 median = 20.0
    # District B: value1 median = 5.0, value2 median = 55.0
    # District C: value1 median = 8.0, value2 median = 70.0

    expected = pl.DataFrame(
        {
            "district": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "category": ["x", "y", "x", "y", "x", "y", "x", "y"],
            "value1__impute_groupby_median": [1.0, 2.0, 1.5, 4.0, 5.0, 6.0, 8.0, 8.0],
            "value2__impute_groupby_median": [
                10.0,
                20.0,
                30.0,
                55.0,
                50.0,
                60.0,
                70.0,
                70.0,
            ],
        }
    )

    assert_frame_equal(transformed, expected)


def test_groupby_imputer_mean(sample_dataframe):
    """Test GroupByImputer with mean strategy."""
    imputer = GroupByImputer(group_by_column="district", strategy="mean", inplace=False)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    # District A: value1 mean = 1.5, value2 mean = 20.0
    # District B: value1 mean = 5.0, value2 mean = 55.0
    # District C: value1 mean = 8.0, value2 mean = 70.0

    expected = pl.DataFrame(
        {
            "district": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "category": ["x", "y", "x", "y", "x", "y", "x", "y"],
            "value1__impute_groupby_mean": [1.0, 2.0, 1.5, 4.0, 5.0, 6.0, 8.0, 8.0],
            "value2__impute_groupby_mean": [
                10.0,
                20.0,
                30.0,
                55.0,
                50.0,
                60.0,
                70.0,
                70.0,
            ],
        }
    )

    assert_frame_equal(transformed, expected)


def test_groupby_imputer_inplace(sample_dataframe):
    """Test GroupByImputer with inplace=True."""
    imputer = GroupByImputer(group_by_column="district", strategy="median", inplace=True)
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    expected = pl.DataFrame(
        {
            "district": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "value1": [1.0, 2.0, 1.5, 4.0, 5.0, 6.0, 8.0, 8.0],
            "value2": [10.0, 20.0, 30.0, 55.0, 50.0, 60.0, 70.0, 70.0],
            "category": ["x", "y", "x", "y", "x", "y", "x", "y"],
        }
    )

    assert_frame_equal(transformed, expected)


def test_groupby_imputer_no_drop_columns(sample_dataframe):
    """Test GroupByImputer with drop_columns=False."""
    imputer = GroupByImputer(
        group_by_column="district", strategy="mean", drop_columns=False, inplace=False
    )
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    # Should keep original columns plus new imputed columns
    assert "value1" in transformed.columns
    assert "value2" in transformed.columns
    assert "value1__impute_groupby_mean" in transformed.columns
    assert "value2__impute_groupby_mean" in transformed.columns
    assert len(transformed.columns) == 6  # district, value1, value2, category, + 2 new


def test_groupby_imputer_specific_columns(sample_dataframe):
    """Test GroupByImputer with specific columns."""
    imputer = GroupByImputer(
        group_by_column="district", strategy="median", subset=["value1"], inplace=False
    )
    imputer.fit(sample_dataframe)
    transformed = imputer.transform(sample_dataframe)

    # Should only impute value1, not value2
    assert "value1__impute_groupby_median" in transformed.columns
    assert "value2" in transformed.columns
    assert "value2__impute_groupby_median" not in transformed.columns
    assert "value1" not in transformed.columns  # Original dropped


def test_groupby_imputer_auto_detect_numeric():
    """Test that GroupByImputer auto-detects numeric columns."""
    X = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "num1": [1.0, None, 3.0, 4.0],
            "num2": [10, 20, None, 40],
            "text": ["x", "y", "z", "w"],
        }
    )

    imputer = GroupByImputer(group_by_column="group", strategy="mean")
    imputer.fit(X)

    # Should auto-detect num1 and num2
    assert set(imputer.subset) == {"num1", "num2"}


def test_groupby_imputer_single_value_per_group():
    """Test GroupByImputer when there's only one value per group."""
    X = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1.0, 1.0, None, None],
        }
    )

    imputer = GroupByImputer(group_by_column="group", strategy="median", inplace=False)
    imputer.fit(X)
    transformed = imputer.transform(X)

    # Group A: median = 1.0, Group B: median = None (no values)
    expected = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value__impute_groupby_median": [1.0, 1.0, None, None],
        }
    )

    assert_frame_equal(transformed, expected)


def test_groupby_imputer_fit_transform():
    """Test fit_transform method."""
    X = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1.0, None, 3.0, None],
        }
    )

    imputer = GroupByImputer(group_by_column="group", strategy="mean", inplace=False)
    result = imputer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value__impute_groupby_mean": [1.0, 1.0, 3.0, 3.0],
        }
    )

    assert_frame_equal(result, expected)


def test_groupby_imputer_new_groups_in_transform():
    """Test behavior when transform sees new groups not in fit data."""
    train_X = pl.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1.0, 2.0, 4.0, 6.0],
        }
    )

    test_X = pl.DataFrame(
        {
            "group": ["A", "B", "C", "C"],  # C is new
            "value": [None, None, None, 8.0],
        }
    )

    imputer = GroupByImputer(group_by_column="group", strategy="mean", inplace=False)
    imputer.fit(train_X)
    transformed = imputer.transform(test_X)

    # A: mean=1.5, B: mean=5.0, C: not in training so null
    expected = pl.DataFrame(
        {
            "group": ["A", "B", "C", "C"],
            "value__impute_groupby_mean": [1.5, 5.0, None, 8.0],
        }
    )

    assert_frame_equal(transformed, expected)
