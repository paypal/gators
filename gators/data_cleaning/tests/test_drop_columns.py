import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import DropColumns


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X = {
        "column1": [1, 2, 3],
        "column2": ["A", "B", "C"],
        "column3": [True, False, True],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_X() -> pl.DataFrame:
    X = {
        "column3": [True, False, True],
    }
    return pl.DataFrame(X)


def test_drop_columns_transform(
    sample_data,
    expected_X,
):
    drop_columns = DropColumns(subset=["column1", "column2"])
    drop_columns.fit(sample_data)
    transformed_X = drop_columns.transform(sample_data)
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_drop_columns_transform_no_columns(
    sample_data,
):
    drop_columns = DropColumns(subset=[])
    drop_columns.fit(sample_data)
    transformed_X = drop_columns.transform(sample_data)
    assert_frame_equal(transformed_X, sample_data, check_column_order=False)
