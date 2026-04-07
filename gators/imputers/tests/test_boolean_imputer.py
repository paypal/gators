import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.imputers.boolean_imputer import BooleanImputer


@pytest.fixture
def sample_X():
    return pl.DataFrame({"A": [True, False, None, True], "B": [None, True, False, False]})


def test_fit_constant_strategy(sample_X):
    imputer = BooleanImputer(strategy="constant", value=False)
    imputer.fit(sample_X)

    assert imputer._statistics["A"] == False
    assert imputer._statistics["B"] == False


def test_fit_most_frequent_strategy(sample_X):
    imputer = BooleanImputer(strategy="most_frequent")
    imputer.fit(sample_X)

    assert imputer._statistics["A"] == True
    assert imputer._statistics["B"] == False


def test_transform_constant_strategy(sample_X):
    imputer = BooleanImputer(strategy="constant", value=False, inplace=False)
    imputer.fit(sample_X)

    transformed_X = imputer.transform(sample_X)

    expected_X = pl.DataFrame(
        {
            "A__impute_constant": [True, False, False, True],
            "B__impute_constant": [False, True, False, False],
        }
    )

    assert_frame_equal(transformed_X, expected_X)


def test_transform_most_frequent_strategy(sample_X):
    imputer = BooleanImputer(strategy="most_frequent", inplace=False)
    imputer.fit(sample_X)

    transformed_X = imputer.transform(sample_X)

    expected_X = pl.DataFrame(
        {
            "A__impute_most_frequent": [True, False, True, True],
            "B__impute_most_frequent": [False, True, False, False],
        }
    )

    assert_frame_equal(transformed_X, expected_X)


def test_subset_columns(sample_X):
    imputer = BooleanImputer(strategy="most_frequent", subset=["B"], inplace=False)
    imputer.fit(sample_X)

    transformed_X = imputer.transform(sample_X)

    expected_X = pl.DataFrame(
        {
            "A": [True, False, None, True],
            "B__impute_most_frequent": [False, True, False, False],
        }
    )

    assert_frame_equal(transformed_X, expected_X)


def test_transform_no_drop(sample_X):
    imputer = BooleanImputer(strategy="most_frequent", drop_columns=False, inplace=False)
    imputer.fit(sample_X)

    transformed_X = imputer.transform(sample_X)

    expected_X = pl.DataFrame(
        {
            "A": [True, False, None, True],
            "B": [None, True, False, False],
            "A__impute_most_frequent": [True, False, True, True],
            "B__impute_most_frequent": [False, True, False, False],
        }
    )

    assert_frame_equal(transformed_X, expected_X)


def test_transform_inplace_true(sample_X):
    """Test with inplace=True to modify columns in place."""
    imputer = BooleanImputer(strategy="most_frequent", inplace=True)
    imputer.fit(sample_X)

    transformed_X = imputer.transform(sample_X)

    # Original columns should be modified in place
    expected_X = pl.DataFrame(
        {
            "A": [True, False, True, True],  # None imputed with True (most frequent)
            "B": [False, True, False, False],  # None imputed with False (most frequent)
        }
    )

    assert_frame_equal(transformed_X, expected_X)
