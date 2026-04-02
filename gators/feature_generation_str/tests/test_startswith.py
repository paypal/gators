from typing import Dict, List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import Startswith


@pytest.fixture
def sample_data() -> pl.DataFrame:
    X = {
        "column1": ["prefix1_sample", None, "prefix2_sample"],
        "column2": [None, "prefix3_sample", "no_match"],
    }
    return pl.DataFrame(X)


@pytest.fixture
def expected_X() -> pl.DataFrame:
    X = {
        "column1": ["prefix1_sample", None, "prefix2_sample"],
        "column2": [None, "prefix3_sample", "no_match"],
        "column1__startswith_prefix1": [True, None, False],
        "column1__startswith_prefix2": [False, None, True],
        "column2__startswith_prefix3": [None, True, False],
    }
    return pl.DataFrame(X)


@pytest.fixture
def startswith_instance() -> Startswith:
    startswith_dict: Dict[str, List[str]] = {
        "column1": ["prefix1", "prefix2"],
        "column2": ["prefix3"],
    }
    return Startswith(startswith_dict=startswith_dict)


def test_startswith_transform(
    startswith_instance: Startswith,
    sample_data: pl.DataFrame,
    expected_X: pl.DataFrame,
):
    """Test that the transform method correctly adds boolean columns."""
    startswith_instance.fit(sample_data)
    transformed_X = startswith_instance.transform(sample_data)
    assert_frame_equal(transformed_X, expected_X)
