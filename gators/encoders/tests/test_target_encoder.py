import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import TargetEncoder


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "baz"],
            "B": [True, False, True, True, False],
        }
    )


@pytest.fixture
def sample_target():
    return pl.Series("target", [1, 0, 1, 1, 0])


def test_transform_defaults(sample_X, sample_target):
    encoder = TargetEncoder(inplace=False)
    encoder.fit(sample_X, sample_target)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__target_enc": [1.0, 0.5, 1.0, 0.5, 0.0],
            "B__target_enc": [1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_with_columns_and_no_drop(sample_X, sample_target):
    encoder = TargetEncoder(subset=["A"], drop_columns=False, inplace=False)
    encoder.fit(sample_X, sample_target)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "baz"],
            "B": [True, False, True, True, False],
            "A__target_enc": [1.0, 0.5, 1.0, 0.5, 0.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_with_min_count(sample_X, sample_target):
    encoder = TargetEncoder(min_count=2, inplace=False)
    encoder.fit(sample_X, sample_target)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__target_enc": [1.0, 0.5, 1.0, 0.5, 0.0],
            "B__target_enc": [1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_with_min_count_ratio(sample_X, sample_target):
    encoder = TargetEncoder(min_count=0.3, inplace=False)
    encoder.fit(sample_X, sample_target)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__target_enc": [1.0, 0.5, 1.0, 0.5, 0.0],
            "B__target_enc": [1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_unseen(sample_X, sample_target):
    encoder = TargetEncoder(inplace=False)
    encoder.fit(sample_X, sample_target)
    sample_X_new = pl.DataFrame(
        {
            "A": [None, "bar", "foo", "bar", "zeta"],
            "B": [None, False, True, True, False],
        }
    )
    transformed_X = encoder.transform(sample_X_new)
    expected_X = pl.DataFrame(
        {
            "A__target_enc": [0.0, 0.5, 1.0, 0.5, 0.0],
            "B__target_enc": [0.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)
