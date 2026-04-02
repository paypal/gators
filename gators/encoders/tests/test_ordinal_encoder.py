import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import OrdinalEncoder


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "baz"],
            "B": [True, False, True, True, False],
        }
    )


@pytest.fixture
def encoder():
    return OrdinalEncoder(inplace=False)


def test_transform_defaults(encoder, sample_X):
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__ordinal_enc": [3.0, 2.0, 3.0, 2.0, 1.0],
            "B__ordinal_enc": [2.0, 1.0, 2.0, 2.0, 1.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_with_columns_and_no_drop(sample_X):
    encoder = OrdinalEncoder(subset=["A"], drop_columns=False, inplace=False)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "baz"],
            "B": [True, False, True, True, False],
            "A__ordinal_enc": [3.0, 2.0, 3.0, 2.0, 1.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_with_min_count(sample_X):
    encoder = OrdinalEncoder(min_count=2, inplace=False)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__ordinal_enc": [2.0, 1.0, 2.0, 1.0, 0.0],
            "B__ordinal_enc": [2.0, 1.0, 2.0, 2.0, 1.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_with_min_count_ratio(sample_X):
    encoder = OrdinalEncoder(min_count=0.3, inplace=False)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__ordinal_enc": [2.0, 1.0, 2.0, 1.0, 0.0],
            "B__ordinal_enc": [2.0, 1.0, 2.0, 2.0, 1.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_unseen_categories(sample_X):
    encoder = OrdinalEncoder(inplace=False)
    encoder.fit(sample_X)
    sample_X_new = pl.DataFrame(
        {
            "A": ["foo", None, "foo", "bar", "alpha"],
            "B": [True, False, True, True, None],
        }
    )
    transformed_X = encoder.transform(sample_X_new)
    expected_X = pl.DataFrame(
        {
            "A__ordinal_enc": [3.0, 0.0, 3.0, 2.0, 0.0],
            "B__ordinal_enc": [2.0, 1.0, 2.0, 2.0, 0.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)
