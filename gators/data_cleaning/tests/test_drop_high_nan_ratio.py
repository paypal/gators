import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import DropHighNaNRatio


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X ={
        "column1": [None, 2, 3],
        "column2": ["A", None, None],
        "column3": [1, 2, 3],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_data1() -> pl.DataFrame:
    X ={
        "column1": [None, 2, 3],
        "column3": [1, 2, 3],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_data2() -> pl.DataFrame:
    X ={
        "column3": [1, 2, 3],
    }
    return pl.DataFrame(X)


@pytest.fixture
def drop_nans() -> DropHighNaNRatio:
    return DropHighNaNRatio(max_ratio=0.5)


def test_drop_high_nan_ratio_transform1(
    sample_data: pl.DataFrame,
    expected_data1: pl.DataFrame,
):
    drop_nans = DropHighNaNRatio(max_ratio=0.5)
    drop_nans.fit(sample_data)
    transformed_X = drop_nans.transform(sample_data)
    assert_frame_equal(transformed_X, expected_data1)


@pytest.fixture
def drop_high_nan_ratio_instance2() -> DropHighNaNRatio:
    return DropHighNaNRatio(max_ratio=0.3)


def test_drop_high_nan_ratio_transform2(
    drop_high_nan_ratio_instance2: DropHighNaNRatio,
    sample_data: pl.DataFrame,
    expected_data2: pl.DataFrame,
):
    """Test that the transform method correctly drops columns with high NaN ratio."""
    drop_high_nan_ratio_instance2.fit(sample_data)
    transformed_X = drop_high_nan_ratio_instance2.transform(sample_data)
    assert_frame_equal(transformed_X, expected_data2)
