import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import WOEEncoder


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "A": ["cat", "dog", "cat", "dog", "cat"],
            "B": ["x", "x", "y", "y", "x"],
        }
    )


@pytest.fixture
def sample_target():
    return pl.Series("target", [1, 0, 1, 1, 0])


def test_woe_encoder_transform_defaults(sample_data, sample_target):
    expected_X = pl.DataFrame(
        {
            "A__encode_woe": [0.286025, -0.402159, 0.286025, -0.402159, 0.286025],
            "B__encode_woe": [-1.090344, -1.090344, 4.901146, 4.901146, -1.090344],
        }
    )
    woe_encoder = WOEEncoder(inplace=False)
    woe_encoder.fit(sample_data, y=sample_target)
    result = woe_encoder.transform(sample_data)
    assert_frame_equal(result, expected_X, check_column_order=False)


def test_woe_encoder_transform_no_drop_columns(sample_data, sample_target):
    expected_X = pl.DataFrame(
        {
            "A": ["cat", "dog", "cat", "dog", "cat"],
            "B": ["x", "x", "y", "y", "x"],
            "B__encode_woe": [-1.090344, -1.090344, 4.901146, 4.901146, -1.090344],
        }
    )
    woe_encoder = WOEEncoder(subset=["B"], drop_columns=False, inplace=False)
    woe_encoder.fit(sample_data, y=sample_target)

    result = woe_encoder.transform(sample_data)

    assert_frame_equal(result, expected_X, check_column_order=False)


def test_woe_unseen_categories(sample_data, sample_target):

    woe_encoder = WOEEncoder(inplace=False)
    sample_data_new = pl.DataFrame(
        {
            "A": [None, "beta", "cat", "dog", "cat"],
            "B": ["alpha", None, "y", "y", "x"],
        }
    )
    woe_encoder.fit(sample_data, y=sample_target)
    expected_X = pl.DataFrame(
        {
            "A__encode_woe": [0.0, 0.0, 0.286025, -0.402159, 0.286025],
            "B__encode_woe": [0.0, 0.0, 4.901146, 4.901146, -1.090344],
        }
    )
    result = woe_encoder.transform(sample_data_new)
    assert_frame_equal(result, expected_X, check_column_order=False)
