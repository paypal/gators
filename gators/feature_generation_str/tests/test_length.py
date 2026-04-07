from typing import List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import Length


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X = {
        "column1": ["short", None, "much longer string"],
        "column2": ["tiny", "bit lengthy", "even longer string"],
        "column3": [1, 2, 3],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_X() -> pl.DataFrame:
    X = {
        "column1": ["short", None, "much longer string"],
        "column2": ["tiny", "bit lengthy", "even longer string"],
        "column3": [1, 2, 3],
        "column1__length": [5, None, 18],
        "column2__length": [4, 11, 18],
    }
    return pl.DataFrame(X)


def test_length_transform_with_columns(sample_data):
    length = Length(subset=["column1"])
    length.fit(sample_data)
    transformed_X = length.transform(sample_data)
    expected_X = pl.DataFrame(
        {
            "column1": ["short", None, "much longer string"],
            "column2": ["tiny", "bit lengthy", "even longer string"],
            "column3": [1, 2, 3],
            "column1__length": [5, None, 18],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_length_transform(sample_data):
    length = Length()
    length.fit(sample_data)
    transformed_X = length.transform(sample_data)
    expected_X = pl.DataFrame(
        {
            "column1": ["short", None, "much longer string"],
            "column2": ["tiny", "bit lengthy", "even longer string"],
            "column3": [1, 2, 3],
            "column1__length": [5, None, 18],
            "column2__length": [4, 11, 18],
        }
    )
    assert_frame_equal(transformed_X, expected_X)
