import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import RenameColumns


@pytest.fixture
def sample_dataframe():
    return pl.DataFrame(
        {
            "col1": ["a", "a", "b", "c"],
            "col2": ["x", "x", "x", "y"],
            "col3": [1, 2, 3, 4],
        }
    )


def test_transform_all_columns(sample_dataframe):
    expected_X = pl.DataFrame(
        {
            "column1": ["a", "a", "b", "c"],
            "column2": ["x", "x", "x", "y"],
            "column3": [1, 2, 3, 4],
        }
    )
    rename_columns = RenameColumns(
        column_mapping={"col1": "column1", "col2": "column2", "col3": "column3"}
    )
    rename_columns.fit(sample_dataframe)
    result = rename_columns.transform(sample_dataframe)
    assert_frame_equal(result, expected_X)


def test_transform_subset_columns(sample_dataframe):
    rename_columns = RenameColumns(column_mapping={"col1": "column1", "col3": "column3"})
    expected_X = pl.DataFrame(
        {
            "column1": ["a", "a", "b", "c"],
            "col2": ["x", "x", "x", "y"],
            "column3": [1, 2, 3, 4],
        }
    )
    rename_columns.fit(sample_dataframe)
    result = rename_columns.transform(sample_dataframe)
    assert_frame_equal(result, expected_X)
