import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import VarianceFilter


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [0.5, 0.5, 0.5, 0.5],
            "feature3": [5, 6, 7, 8],
            "label": [0, 1, 0, 1],
        }
    )


def test_default_parameters(sample_X):
    transformer = VarianceFilter(min_var=0.1)
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    expected_X = sample_X.drop("feature2")
    assert_frame_equal(transformed_X, expected_X)


def test_columns_subset(sample_X):
    transformer = VarianceFilter(subset=["feature1", "feature2"], min_var=0.1)
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    expected_X = sample_X.drop("feature2")
    assert_frame_equal(transformed_X, expected_X)


def test_no_columns_dropped(sample_X):
    transformer = VarianceFilter(subset=["feature1", "feature3"], min_var=0.1)
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    expected_X = sample_X
    assert_frame_equal(transformed_X, expected_X)


def test_drop_zero_std_dev():
    sample_X = pl.DataFrame(
        {
            "feature1": [1, 1, 1, 2],
            "feature2": [0.5, 0.5, 0.5, 0.5],
        }
    )
    transformer = VarianceFilter(min_var=0.0)
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)
    expected_X = sample_X.drop("feature2")
    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()
