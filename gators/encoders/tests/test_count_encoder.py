import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

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
    encoder = CountEncoder(subset=["category"], min_count=1, drop_columns=False, inplace=False)
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


def test_invalid_parameter_names():
    """Test that CountEncoder raises ValidationError for invalid parameter names.

    This test verifies that _BaseTransformer (via _BaseEncoder) properly validates
    parameter names and rejects incorrect ones like 'columns' instead of 'subset'.
    """
    # Test with incorrect parameter name 'columns' instead of 'subset'
    with pytest.raises(ValidationError) as exc_info:
        CountEncoder(columns=["category"], min_count=1, inplace=True)

    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "columns" in str(exc_info.value)

    # Test with another incorrect parameter name
    with pytest.raises(ValidationError) as exc_info:
        CountEncoder(feature_list=["category"], min_count=1, inplace=True)

    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "feature_list" in str(exc_info.value)

    # Test with multiple incorrect parameter names
    with pytest.raises(ValidationError) as exc_info:
        CountEncoder(columns=["category"], invalid_param=True, min_count=1)

    assert "Extra inputs are not permitted" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main()
