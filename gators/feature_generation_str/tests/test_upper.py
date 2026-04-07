from typing import List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import Upper


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X = {
        "column1": ["Hello", None, "WorLD"],
        "column2": [True, False, None],
        "column3": [1, 2, 3],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_X() -> pl.DataFrame:
    X = {
        "column1__upper": ["HELLO", None, "WORLD"],
        "column2__upper": ["TRUE", "FALSE", None],
        "column3": [1, 2, 3],
    }
    return pl.DataFrame(X)


def test_upper_transform(sample_data):
    upper = Upper(inplace=False)
    upper.fit(sample_data)
    transformed_X = upper.transform(sample_data)
    expected_X = pl.DataFrame(
        {
            "column1__upper": ["HELLO", None, "WORLD"],
            "column2__upper": ["TRUE", "FALSE", None],
            "column3": [1, 2, 3],
        }
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_upper_transform_drop_columns(sample_data):
    upper = Upper(subset=["column1"], drop_columns=False, inplace=False)
    upper.fit(sample_data)
    transformed_X = upper.transform(sample_data)
    expected_X = pl.DataFrame(
        {
            "column1__upper": ["HELLO", None, "WORLD"],
            "column1": ["Hello", None, "WorLD"],
            "column2": [True, False, None],
            "column3": [1, 2, 3],
        }
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_upper_transform_inplace_true(sample_data):
    """Test with inplace=True to modify columns in place."""
    upper = Upper(subset=["column1"], inplace=True)
    upper.fit(sample_data)
    transformed_X = upper.transform(sample_data)

    # Original column should be modified in place
    assert "column1" in transformed_X.columns
    assert "column1__upper" not in transformed_X.columns
    assert transformed_X["column1"][0] == "HELLO"
    assert transformed_X["column1"][2] == "WORLD"


def test_upper_transform_inplace_false_with_drop(sample_data):
    """Test with inplace=False and drop_columns=True."""
    upper = Upper(subset=["column1", "column2"], drop_columns=True, inplace=False)
    upper.fit(sample_data)
    transformed_X = upper.transform(sample_data)

    # Original columns should be dropped
    assert "column1" not in transformed_X.columns
    assert "column2" not in transformed_X.columns
    # New columns should exist
    assert "column1__upper" in transformed_X.columns
    assert "column2__upper" in transformed_X.columns
