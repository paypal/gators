import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import RareCategoryEncoder


@pytest.fixture
def sample_data():
    X = {
        "A": ["cat", "dog", "cat", "mouse", "dog", "cat", "rabbit"],
        "B": ["red", "blue", "red", "green", "blue", "red", "blue"],
        "C": [1, 2, 3, 4, 5, 6, 7],
    }
    return pl.DataFrame(X)


def test_defaults(sample_data):
    encoder = RareCategoryEncoder(inplace=False)
    encoder.fit(sample_data)
    transformed_X = encoder.transform(sample_data)
    expected_X = {
        "C": [1, 2, 3, 4, 5, 6, 7],
        "A__encode_rare": ["cat", "dog", "cat", "RARE", "dog", "cat", "RARE"],
        "B__encode_rare": ["red", "blue", "red", "RARE", "blue", "red", "blue"],
    }
    expected_X = pl.DataFrame(expected_X)

    assert_frame_equal(transformed_X, expected_X)


def test_columns_subset(sample_data):
    encoder = RareCategoryEncoder(subset=["A"], min_count=3, drop_columns=False, inplace=False)
    encoder.fit(sample_data)
    transformed_X = encoder.transform(sample_data)

    expected_X = {
        "A": ["cat", "dog", "cat", "mouse", "dog", "cat", "rabbit"],
        "B": ["red", "blue", "red", "green", "blue", "red", "blue"],
        "C": [1, 2, 3, 4, 5, 6, 7],
        "A__encode_rare": ["cat", "RARE", "cat", "RARE", "RARE", "cat", "RARE"],
    }
    expected_X = pl.DataFrame(expected_X)
    assert_frame_equal(transformed_X, expected_X)


def test_inplace_true(sample_data):
    """Test with inplace=True to modify columns in place."""
    encoder = RareCategoryEncoder(subset=["A"], min_count=3, inplace=True)
    encoder.fit(sample_data)
    transformed_X = encoder.transform(sample_data)

    # With inplace=True, original columns should be modified
    expected_X = {
        "A": ["cat", "RARE", "cat", "RARE", "RARE", "cat", "RARE"],
        "B": ["red", "blue", "red", "green", "blue", "red", "blue"],
        "C": [1, 2, 3, 4, 5, 6, 7],
    }
    expected_X = pl.DataFrame(expected_X)
    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()
