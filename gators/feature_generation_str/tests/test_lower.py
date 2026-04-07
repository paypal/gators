from typing import List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import Lower


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X = {
        "column1": ["Hello", None, "WORLD"],
        "column2": [True, False, None],
        "column3": [1, 2, 3],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_X() -> pl.DataFrame:
    X = {
        "column1__lower": ["hello", None, "world"],
        "column2__lower": ["true", "false", None],
        "column3": [1, 2, 3],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_data_no_drop() -> pl.DataFrame:
    X = {
        "column1": ["Hello", None, "WORLD"],
        "column2": [True, False, None],
        "column3": [1, 2, 3],
        "column1__lower": ["hello", None, "world"],
    }
    return pl.DataFrame(X)


def test_lower_transform(sample_data, expected_X):
    lower = Lower(drop_columns=True, inplace=False)
    lower.fit(sample_data)
    transformed_X = lower.transform(sample_data)
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_lower_transform_no_columns_no_drop_columns(sample_data, expected_data_no_drop):
    lower = Lower(subset=["column1"], drop_columns=False, inplace=False)
    lower.fit(sample_data)
    transformed_X = lower.transform(sample_data)
    assert_frame_equal(transformed_X, expected_data_no_drop, check_column_order=False)


def test_lower_transform_inplace_true(sample_data):
    """Test with inplace=True to modify columns in place."""
    lower = Lower(subset=["column1"], inplace=True)
    lower.fit(sample_data)
    transformed_X = lower.transform(sample_data)

    # Original column should be modified in place
    assert "column1" in transformed_X.columns
    assert "column1__lower" not in transformed_X.columns
    assert transformed_X["column1"][0] == "hello"
    assert transformed_X["column1"][2] == "world"


def test_lower_transform_inplace_false_with_drop(sample_data):
    """Test with inplace=False and drop_columns=True."""
    lower = Lower(subset=["column1", "column2"], drop_columns=True, inplace=False)
    lower.fit(sample_data)
    transformed_X = lower.transform(sample_data)

    # Original columns should be dropped
    assert "column1" not in transformed_X.columns
    assert "column2" not in transformed_X.columns
    # New columns should exist
    assert "column1__lower" in transformed_X.columns
    assert "column2__lower" in transformed_X.columns
