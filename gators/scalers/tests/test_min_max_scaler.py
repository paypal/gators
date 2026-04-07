import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.scalers import MinmaxScaler


def test_transform_default():
    X = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    scaler = MinmaxScaler().fit(X)
    transformed_X = scaler.transform(X)

    expected_X = pl.DataFrame(
        {
            "C": ["a", "b", "c", "d", "e"],
            "A__minmax_scale": [0.0, 0.25, 0.5, 0.75, 1.0],
            "B__minmax_scale": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    )

    assert_frame_equal(transformed_X, expected_X)


def test_transform_subset_columns():
    X = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    scaler = MinmaxScaler(subset=["A"]).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = pl.DataFrame(
        {
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
            "A__minmax_scale": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    )

    assert_frame_equal(transformed_X, expected_X)


def test_transform_drop_columns_false():
    X = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    scaler = MinmaxScaler(drop_columns=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            ((pl.col("A") - 1.0) * (1.0 / 4.0)).alias("A__minmax_scale"),
            ((pl.col("B") - 10.0) * (1.0 / 40.0)).alias("B__minmax_scale"),
        ]
    )

    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()
