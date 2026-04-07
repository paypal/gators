import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import Replace


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "first_name": ["Alice", "Bob", "Charlie", "David"],
            "last_name": ["Smith", "Johnson", "Williams", "Brown"],
            "age": [25, 30, 35, 40],
            "city": ["NYC", "LA", "Chicago", "Houston"],
        }
    )


def test_default_parameters(sample_X):
    to_replace = {
        "first_name": {"Alice": "Alicia", "Bob": "Robert"},
        "city": {"NYC": "New York", "LA": "Los Angeles"},
    }
    expected_X = sample_X.with_columns(
        [
            pl.col("first_name")
            .replace({"Alice": "Alicia", "Bob": "Robert"})
            .alias("first_name__replace"),
            pl.col("city").replace({"NYC": "New York", "LA": "Los Angeles"}).alias("city__replace"),
        ]
    ).drop("first_name", "city")
    transformer = Replace(to_replace=to_replace, inplace=False)
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)
    assert_frame_equal(transformed_X, expected_X)


def test_replace_transformer_drop_columns_false(sample_X):
    to_replace = {
        "first_name": {"Alice": "Alicia", "Bob": "Robert"},
        "city": {"NYC": "New York", "LA": "Los Angeles"},
    }
    expected_X = sample_X.with_columns(
        [
            pl.col("first_name")
            .replace({"Alice": "Alicia", "Bob": "Robert"})
            .alias("first_name__replace"),
            pl.col("city").replace({"NYC": "New York", "LA": "Los Angeles"}).alias("city__replace"),
        ]
    )
    transformer = Replace(to_replace=to_replace, drop_columns=False, inplace=False)
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    assert_frame_equal(transformed_X, expected_X)


def test_replace_transformer_inplace_true(sample_X):
    """Test replace with inplace=True to modify columns in place."""
    to_replace = {
        "first_name": {"Alice": "Alicia", "Bob": "Robert"},
        "city": {"NYC": "New York", "LA": "Los Angeles"},
    }
    expected_X = sample_X.with_columns(
        [
            pl.col("first_name").replace({"Alice": "Alicia", "Bob": "Robert"}),
            pl.col("city").replace({"NYC": "New York", "LA": "Los Angeles"}),
        ]
    )
    transformer = Replace(to_replace=to_replace, inplace=True)
    transformer.fit(sample_X)
    transformed_X = transformer.transform(sample_X)

    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()
