import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import DropLowCardinality


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "first_name": ["Alice", "Alice", "Alice", "Alice"],
            "last_name": ["Smith", "Johnson", "Williams", "Brown"],
            "age": [25, 30, 35, 40],
            "city": ["New York", "Los Angeles", "Chicago", "Houston"],
        }
    )


def test_default_parameters(sample_X):
    transformer = DropLowCardinality(min_count=2)
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "last_name": ["Smith", "Johnson", "Williams", "Brown"],
            "age": [25, 30, 35, 40],
            "city": ["New York", "Los Angeles", "Chicago", "Houston"],
        }
    )

    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()
