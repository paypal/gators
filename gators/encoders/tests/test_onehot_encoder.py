import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import OneHotEncoder


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "baz"],
            "B": ["one", "one", "two", "two", "one"],
        }
    )


@pytest.fixture
def encoder():
    return OneHotEncoder()


def test_transform_defaults(sample_X):
    encoder = OneHotEncoder()
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__foo": [1, 0, 1, 0, 0],
            "A__bar": [0, 1, 0, 1, 0],
            "A__baz": [0, 0, 0, 0, 1],
            "B__one": [1, 1, 0, 0, 1],
            "B__two": [0, 0, 1, 1, 0],
        },
        schema={
            "A__foo": pl.Float64,
            "A__bar": pl.Float64,
            "A__baz": pl.Float64,
            "B__one": pl.Float64,
            "B__two": pl.Float64,
        },
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_with_columns_and_no_drop(sample_X):
    encoder = OneHotEncoder(subset=["A"], drop_columns=False)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "baz"],
            "B": ["one", "one", "two", "two", "one"],
            "A__foo": [1, 0, 1, 0, 0],
            "A__bar": [0, 1, 0, 1, 0],
            "A__baz": [0, 0, 0, 0, 1],
        },
        schema={
            "A": pl.String,
            "B": pl.String,
            "A__foo": pl.Float64,
            "A__bar": pl.Float64,
            "A__baz": pl.Float64,
        },
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_with_min_count(sample_X):
    encoder = OneHotEncoder(min_count=2)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__foo": [1, 0, 1, 0, 0],
            "A__bar": [0, 1, 0, 1, 0],
            "B__one": [1, 1, 0, 0, 1],
            "B__two": [0, 0, 1, 1, 0],
        },
        schema={
            "A__foo": pl.Float64,
            "A__bar": pl.Float64,
            "B__one": pl.Float64,
            "B__two": pl.Float64,
        },
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_with_min_count_ratio(sample_X):
    encoder = OneHotEncoder(min_count=0.3)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__foo": [1, 0, 1, 0, 0],
            "A__bar": [0, 1, 0, 1, 0],
            "B__one": [1, 1, 0, 0, 1],
            "B__two": [0, 0, 1, 1, 0],
        },
        schema={
            "A__foo": pl.Float64,
            "A__bar": pl.Float64,
            "B__one": pl.Float64,
            "B__two": pl.Float64,
        },
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_with_categories(sample_X):
    categories = {"A": ["foo", "baz"], "B": ["one"]}
    encoder = OneHotEncoder(categories=categories)
    encoder.fit(sample_X)
    transformed_X = encoder.transform(sample_X)
    expected_X = pl.DataFrame(
        {
            "A__foo": [1, 0, 1, 0, 0],
            "A__baz": [0, 0, 0, 0, 1],
            "B__one": [1, 1, 0, 0, 1],
        },
        schema={"A__foo": pl.Float64, "A__baz": pl.Float64, "B__one": pl.Float64},
    )
    assert_frame_equal(transformed_X, expected_X, check_column_order=False)


def test_transform_with_missing_categories_at_test_time():
    """Test handling of categories present in fit but missing in transform."""
    train_X = pl.DataFrame({
        "A": ["foo", "bar", "baz"],
        "B": ["one", "two", "three"],
    })
    
    test_X = pl.DataFrame({
        "A": ["foo", "bar"],  # Missing "baz"
        "B": ["one", "two"],  # Missing "three"
    })
    
    encoder = OneHotEncoder()
    encoder.fit(train_X)
    transformed_X = encoder.transform(test_X)
    
    # Should have columns for all categories from fit, missing ones filled with 0
    assert "A__foo" in transformed_X.columns
    assert "A__bar" in transformed_X.columns
    assert "A__baz" in transformed_X.columns  # Should exist with zeros
    assert "B__three" in transformed_X.columns  # Should exist with zeros
    
    # Check that missing categories are filled with zeros
    assert all(transformed_X["A__baz"] == 0)
    assert all(transformed_X["B__three"] == 0)
