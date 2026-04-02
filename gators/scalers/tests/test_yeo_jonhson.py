import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.scalers import YeoJonhson


def test_transform_default():
    X =  pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0, "B": 2}
    scaler = YeoJonhson(lambdas=lambdas).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            pl.when(pl.col("A") < 0)
            .then(pl.col("A").log1p())
            .otherwise(-((-pl.col("A") + 1) ** 2 - 1) / 2)
            .alias("A__yeojonhson"),
            pl.when(pl.col("B") < 0)
            .then(((pl.col("B") + 1) ** 2 - 1) / 2)
            .otherwise(-pl.col("B").log1p())
            .alias("B__yeojonhson"),
        ]
    ).drop(["A", "B"])

    assert_frame_equal(transformed_X, expected_X)


def test_transform_subset_columns():
    X =  pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0}
    scaler = YeoJonhson(lambdas=lambdas).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            pl.when(pl.col("A") < 0)
            .then(pl.col("A").log1p())
            .otherwise(-((-pl.col("A") + 1) ** 2 - 1) / 2)
            .alias("A__yeojonhson")
        ]
    ).drop(["A"])

    assert_frame_equal(transformed_X, expected_X)


def test_transform_drop_columns_false():
    X =  pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0, "B": 2}
    scaler = YeoJonhson(lambdas=lambdas, drop_columns=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            pl.when(pl.col("A") < 0)
            .then(pl.col("A").log1p())
            .otherwise(-((-pl.col("A") + 1) ** 2 - 1) / 2)
            .alias("A__yeojonhson"),
            pl.when(pl.col("B") < 0)
            .then(((pl.col("B") + 1) ** 2 - 1) / 2)
            .otherwise(-pl.col("B").log1p())
            .alias("B__yeojonhson"),
        ]
    )

    assert_frame_equal(transformed_X, expected_X)


@pytest.fixture
def sample_data():
    return pl.DataFrame({"A": [-1, 0, 1, 2], "B": [-2, -1, 0, 1]})


def test_transform_lambda_zero(sample_data):
    transformer = YeoJonhson(lambdas={"A": 0}, drop_columns=False)
    transformer.fit(sample_data)
    transformed_X =transformer.transform(sample_data)
    expected_X =sample_data.with_columns(
        [
            pl.when(pl.col("A") < 0)
            .then(pl.col("A").log1p())
            .otherwise(-((-pl.col("A") + 1) ** 2 - 1) / 2)
            .alias("A__yeojonhson")
        ]
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_lambda_two(sample_data):
    transformer = YeoJonhson(lambdas={"A": 2}, drop_columns=False)
    transformer.fit(sample_data)
    transformed_X =transformer.transform(sample_data)
    expected_X =sample_data.with_columns(
        [
            pl.when(pl.col("A") < 0)
            .then(((pl.col("A") + 1) ** 2 - 1) / 2)
            .otherwise(-pl.col("A").log1p())
            .alias("A__yeojonhson")
        ]
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_lambda_non_zero_non_two(sample_data):
    transformer = YeoJonhson(lambdas={"A": 1.5}, drop_columns=False)
    transformer.fit(sample_data)
    transformed_X =transformer.transform(sample_data)
    expected_X =sample_data.with_columns(
        [
            pl.when(pl.col("A") < 0)
            .then(((pl.col("A") + 1) ** 1.5 - 1) / 1.5)
            .otherwise(-((-pl.col("A") + 1) ** 0.5 - 1) / 0.5)
            .alias("A__yeojonhson")
        ]
    )
    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()
