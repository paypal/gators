import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import CountEncoder


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "category": ["A", "B", "A", "C", "C", "A", "B"],
            "value": [1, 2, 3, 4, 5, 6, 7],
            "other": ["foo", "bar", "baz", "qux", "quux", "corge", "grault"],
        }
    )


def test_default_parameters(sample_X):
    encoder = CountEncoder(min_count=1, inplace=False)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6, 7],
            "category__count_enc": [3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 2.0],
            "other__count_enc": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_min_count(sample_X):
    encoder = CountEncoder(min_count=3, inplace=False)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)

    expected_X = pl.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6, 7],
            "category__count_enc": [3.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0],
            "other__count_enc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_columns_subset_drop_columns_false(sample_X):
    encoder = CountEncoder(
        subset=["category"], min_count=1, drop_columns=False, inplace=False
    )
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)

    expected_X = sample_X.with_columns(
        [
            pl.col("category")
            .replace({"A": 3, "B": 2, "C": 2})
            .cast(pl.Float64)
            .alias("category__count_enc")
        ]
    )
    assert_frame_equal(transformed_X, expected_X)


def test_unseen_categories(sample_X):
    encoder = CountEncoder(min_count=1, inplace=False)
    encoder.fit(sample_X)
    sample_X_new = pl.DataFrame(
        {
            "category": ["A", "B", "x", "y", "z", "A", None],
            "value": [1, 2, 3, 4, 5, 6, 7],
            "other": ["foo", "bar", None, "qux", "quux", "corge", "grault"],
        }
    )
    transformed_X = encoder.transform(sample_X_new)
    expected_X = pl.DataFrame(
        {
            "value": [1, 2, 3, 4, 5, 6, 7],
            "category__count_enc": [3.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            "other__count_enc": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()
