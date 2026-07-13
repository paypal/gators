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


def test_fit_no_categorical_columns_skips_filtering():
    """When auto-detected subset is empty (no String/Boolean/Enum cols), _to_drop is set to []."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    dropper = DropLowCardinality(min_count=2)
    dropper.fit(X)
    assert dropper._to_drop == []
    assert_frame_equal(dropper.transform(X), X)


if __name__ == "__main__":
    pytest.main()
