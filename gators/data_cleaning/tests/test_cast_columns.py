from typing import List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import CastColumns


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X = {
        "column1": ["10", "20", "30"],
        "column2": ["1.1", "2.2", "3.3"],
        "column3": [True, False, True],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_data_int() -> pl.DataFrame:
    X = {
        "column1__cast_int64": [10, 20, 30],
        "column2": ["1.1", "2.2", "3.3"],
        "column3": [True, False, True],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_data_float() -> pl.DataFrame:
    X = {
        "column1__cast_float64": [10.0, 20.0, 30.0],
        "column2__cast_float64": [1.1, 2.2, 3.3],
        "column3": [True, False, True],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_data_float_drop_columns_false() -> pl.DataFrame:
    X = {
        "column1": ["10", "20", "30"],
        "column2": ["1.1", "2.2", "3.3"],
        "column1__cast_float64": [10.0, 20.0, 30.0],
        "column2__cast_float64": [1.1, 2.2, 3.3],
        "column3": [True, False, True],
    }
    return pl.DataFrame(X)


def test_cast_columns_transform_int(
    sample_data,
    expected_data_int,
):
    cast_columns = CastColumns(subset=["column1"], dtype=pl.Int64, drop_columns=True, inplace=False)
    cast_columns.fit(sample_data)
    transformed_X = cast_columns.transform(sample_data)
    assert_frame_equal(transformed_X, expected_data_int, check_column_order=False)


def test_cast_columns_transform_float(
    sample_data,
    expected_data_float,
):
    cast_columns = CastColumns(
        subset=["column1", "column2"],
        dtype=pl.Float64,
        drop_columns=True,
        inplace=False,
    )
    cast_columns.fit(sample_data)
    transformed_X = cast_columns.transform(sample_data)
    assert_frame_equal(transformed_X, expected_data_float, check_column_order=False)


def test_cast_columns_transform_float_drop_false(
    sample_data,
    expected_data_float_drop_columns_false,
):
    cast_columns = CastColumns(
        subset=["column1", "column2"],
        dtype=pl.Float64,
        drop_columns=False,
        inplace=False,
    )
    cast_columns.fit(sample_data)
    transformed_X = cast_columns.transform(sample_data)
    assert_frame_equal(
        transformed_X, expected_data_float_drop_columns_false, check_column_order=False
    )


def test_cast_columns_transform_all_columns():
    """Test casting all columns when subset=None."""
    X = pl.DataFrame(
        {
            "col1": ["10", "20", "30"],
            "col2": ["1", "2", "3"],
        }
    )
    cast_columns = CastColumns(subset=None, dtype=pl.Int64, inplace=True)
    cast_columns.fit(X)
    transformed_X = cast_columns.transform(X)

    assert transformed_X["col1"].dtype == pl.Int64
    assert transformed_X["col2"].dtype == pl.Int64


def test_cast_columns_datetime_conversion_inplace_false():
    """Test datetime conversion with inplace=False."""
    X = pl.DataFrame(
        {
            "date_col": ["2020-01-01 10:00:00", "2020-01-02 11:00:00", "2020-01-03 12:00:00"],
            "other_col": [1, 2, 3],
        }
    )
    cast_columns = CastColumns(
        subset=["date_col"], dtype=pl.Datetime, inplace=False, drop_columns=True
    )
    cast_columns.fit(X)
    transformed_X = cast_columns.transform(X)

    assert "date_col__cast_datetime" in transformed_X.columns
    assert "date_col" not in transformed_X.columns
    assert transformed_X["date_col__cast_datetime"].dtype == pl.Datetime


def test_cast_columns_date_conversion_inplace_false():
    """Test date conversion with inplace=False."""
    X = pl.DataFrame(
        {
            "date_col": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "other_col": [1, 2, 3],
        }
    )
    cast_columns = CastColumns(subset=["date_col"], dtype=pl.Date, inplace=False, drop_columns=True)
    cast_columns.fit(X)
    transformed_X = cast_columns.transform(X)

    assert "date_col__cast_date" in transformed_X.columns
    assert "date_col" not in transformed_X.columns
    assert transformed_X["date_col__cast_date"].dtype == pl.Date


def test_cast_columns_generic_cast_inplace_false():
    """Test generic cast operation with inplace=False."""
    X = pl.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        }
    )
    cast_columns = CastColumns(subset=["col1"], dtype=pl.Float64, inplace=False, drop_columns=True)
    cast_columns.fit(X)
    transformed_X = cast_columns.transform(X)

    assert "col1__cast_float64" in transformed_X.columns
    assert "col1" not in transformed_X.columns
    assert transformed_X["col1__cast_float64"].dtype == pl.Float64


def test_cast_columns_datetime_conversion_inplace_true():
    """Test datetime conversion with inplace=True."""
    X = pl.DataFrame(
        {
            "date_col": ["2020-01-01 10:00:00", "2020-01-02 11:00:00", "2020-01-03 12:00:00"],
            "other_col": [1, 2, 3],
        }
    )
    cast_columns = CastColumns(subset=["date_col"], dtype=pl.Datetime, inplace=True)
    cast_columns.fit(X)
    transformed_X = cast_columns.transform(X)

    assert "date_col" in transformed_X.columns
    assert transformed_X["date_col"].dtype == pl.Datetime


def test_cast_columns_date_conversion_inplace_true():
    """Test date conversion with inplace=True."""
    X = pl.DataFrame(
        {
            "date_col": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "other_col": [1, 2, 3],
        }
    )
    cast_columns = CastColumns(subset=["date_col"], dtype=pl.Date, inplace=True)
    cast_columns.fit(X)
    transformed_X = cast_columns.transform(X)

    assert "date_col" in transformed_X.columns
    assert transformed_X["date_col"].dtype == pl.Date
