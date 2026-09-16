from typing import List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import Contains


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X = {
        "column1": ["substring1 is here", None, "substring2 is also here"],
        "column2": [None, "contains substring3", "another no match"],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_X() -> pl.DataFrame:
    X = {
        "column1": ["substring1 is here", None, "substring2 is also here"],
        "column2": [None, "contains substring3", "another no match"],
        "column1__contains_substring1": [1.0, None, 0.0],
        "column1__contains_substring2": [0.0, None, 1.0],
        "column2__contains_substring3": [None, 1.0, 0.0],
    }
    return pl.DataFrame(X)


def test_contains_transform(sample_data, expected_X):
    contains_dict: dict[str, list[str]] = {
        "column1": ["substring1", "substring2"],
        "column2": ["substring3"],
    }
    contains = Contains(contains_dict=contains_dict)
    contains.fit(sample_data)
    transformed_X = contains.transform(sample_data)
    assert_frame_equal(transformed_X, expected_X)
