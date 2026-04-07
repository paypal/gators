from typing import Dict, List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.discretizers.custom_discretizer import CustomDiscretizer


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature1": [0.5, 1.5, 2.5, 3.5, 4.5],
            "feature2": [1, 3, 5, 7, 9],
        }
    )


def test_transform_with_defaults(sample_dataframe: pl.DataFrame):
    custom_discretizer = CustomDiscretizer(
        subset=["feature1", "feature2"],
        bins={"feature1": [2, 3], "feature2": [3, 5]},
        num_bins=5,
        rounding=3,
        drop_columns=True,
        inplace=False,
    )
    _ = custom_discretizer.fit(sample_dataframe)
    transformed_X = custom_discretizer.transform(sample_dataframe).cast(pl.String)
    expected_X = pl.DataFrame(
        {
            "feature1__discretize_custom": [
                "(-inf,2.0]",
                "(-inf,2.0]",
                "(2.0,3.0]",
                "(3.0,inf)",
                "(3.0,inf)",
            ],
            "feature2__discretize_custom": [
                "(-inf,3.0]",
                "(-inf,3.0]",
                "(3.0,5.0]",
                "(5.0,inf)",
                "(5.0,inf)",
            ],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_without_dropping_columns(sample_dataframe: pl.DataFrame):
    custom_discretizer = CustomDiscretizer(
        subset=["feature1"],
        bins={"feature1": [2, 3], "feature2": [3, 5]},
        num_bins=5,
        rounding=3,
        drop_columns=False,
        inplace=False,
    )
    _ = custom_discretizer.fit(sample_dataframe)
    transformed_X = custom_discretizer.transform(sample_dataframe).with_columns(
        pl.col("feature1__discretize_custom").cast(pl.String)
    )
    expected_X = pl.DataFrame(
        {
            "feature1": [0.5, 1.5, 2.5, 3.5, 4.5],
            "feature2": [1, 3, 5, 7, 9],
            "feature1__discretize_custom": [
                "(-inf,2.0]",
                "(-inf,2.0]",
                "(2.0,3.0]",
                "(3.0,inf)",
                "(3.0,inf)",
            ],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_columns_none(sample_dataframe: pl.DataFrame):
    """Test with subset=None, should use bins dict keys."""
    custom_discretizer = CustomDiscretizer(
        subset=None,
        bins={"feature1": [2, 3]},
        num_bins=5,
        rounding=3,
        drop_columns=True,
        inplace=False,
    )
    _ = custom_discretizer.fit(sample_dataframe)
    # Columns should be derived from bins
    assert custom_discretizer.subset == ["feature1"]

    transformed_X = custom_discretizer.transform(sample_dataframe)
    expected_X = pl.DataFrame(
        {
            "feature2": [1, 3, 5, 7, 9],
            "feature1__discretize_custom": [
                "(-inf,2.0]",
                "(-inf,2.0]",
                "(2.0,3.0]",
                "(3.0,inf)",
                "(3.0,inf)",
            ],
        }
    ).with_columns(pl.col("feature1__discretize_custom").cast(pl.Categorical))
    assert_frame_equal(transformed_X, expected_X)


def test_transform_as_numerics(sample_dataframe: pl.DataFrame):
    """Test with as_numerics=True, should generate numeric labels."""
    custom_discretizer = CustomDiscretizer(
        subset=["feature1"],
        bins={"feature1": [2, 3]},
        num_bins=5,
        rounding=3,
        drop_columns=True,
        inplace=False,
        as_numerics=True,
    )
    _ = custom_discretizer.fit(sample_dataframe)
    transformed_X = custom_discretizer.transform(sample_dataframe)
    expected_X = pl.DataFrame(
        {
            "feature2": [1, 3, 5, 7, 9],
            "feature1__discretize_custom": [0, 0, 1, 2, 2],
        }
    ).with_columns(pl.col("feature1__discretize_custom").cast(pl.Int32))
    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()
