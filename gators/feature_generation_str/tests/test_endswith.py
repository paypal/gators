from typing import Dict, List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import Endswith


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X ={
        "column1": ["ends_with1", None, "also_ends_with2"],
        "column2": [None, "does_end_with3", "another_no_match"],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_X() -> pl.DataFrame:
    X ={
        "column1": ["ends_with1", None, "also_ends_with2"],
        "column2": [None, "does_end_with3", "another_no_match"],
        "column1__endswith_with1": [True, None, False],
        "column1__endswith_with2": [False, None, True],
        "column2__endswith_with3": [None, True, False],
    }
    return pl.DataFrame(X)


@pytest.fixture
def endswith_instance() -> Endswith:
    endswith_dict: Dict[str, List[str]] = {
        "column1": ["with1", "with2"],
        "column2": ["with3"],
    }
    return Endswith(endswith_dict=endswith_dict)


def test_endswith_transform(endswith_instance: Endswith, sample_data, expected_X):
    """Test that the transform method correctly adds boolean columns."""
    endswith_instance.fit(sample_data)
    transformed_X = endswith_instance.transform(sample_data)
    assert_frame_equal(transformed_X, expected_X)
