import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.scalers import YeoJohnson


def test_transform_default():
    X = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0, "B": 2}
    scaler = YeoJohnson(lambdas=lambdas, inplace=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            pl.when(pl.col("A") >= 0)
            .then(pl.col("A").log1p())
            .otherwise(-((-pl.col("A") + 1) ** 2 - 1) / 2)
            .alias("A__yeojonhson"),
            pl.when(pl.col("B") >= 0)
            .then(((pl.col("B") + 1) ** 2 - 1) / 2)
            .otherwise(-(-pl.col("B")).log1p())
            .alias("B__yeojonhson"),
        ]
    ).drop(["A", "B"])

    assert_frame_equal(transformed_X, expected_X)


def test_transform_subset_columns():
    X = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0}
    scaler = YeoJohnson(lambdas=lambdas, inplace=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            pl.when(pl.col("A") >= 0)
            .then(pl.col("A").log1p())
            .otherwise(-((-pl.col("A") + 1) ** 2 - 1) / 2)
            .alias("A__yeojonhson")
        ]
    ).drop(["A"])

    assert_frame_equal(transformed_X, expected_X)


def test_transform_drop_columns_false():
    X = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0, "B": 2}
    scaler = YeoJohnson(lambdas=lambdas, drop_columns=False, inplace=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            pl.when(pl.col("A") >= 0)
            .then(pl.col("A").log1p())
            .otherwise(-((-pl.col("A") + 1) ** 2 - 1) / 2)
            .alias("A__yeojonhson"),
            pl.when(pl.col("B") >= 0)
            .then(((pl.col("B") + 1) ** 2 - 1) / 2)
            .otherwise(-(-pl.col("B")).log1p())
            .alias("B__yeojonhson"),
        ]
    )

    assert_frame_equal(transformed_X, expected_X)


@pytest.fixture
def sample_data():
    return pl.DataFrame({"A": [-1, 0, 1, 2], "B": [-2, -1, 0, 1]})


def test_transform_lambda_zero(sample_data):
    transformer = YeoJohnson(lambdas={"A": 0}, drop_columns=False, inplace=False)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)
    expected_X = sample_data.with_columns(
        [
            pl.when(pl.col("A") >= 0)
            .then(pl.col("A").log1p())
            .otherwise(-((-pl.col("A") + 1) ** 2 - 1) / 2)
            .alias("A__yeojonhson")
        ]
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_lambda_two(sample_data):
    transformer = YeoJohnson(lambdas={"A": 2}, drop_columns=False, inplace=False)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)
    expected_X = sample_data.with_columns(
        [
            pl.when(pl.col("A") >= 0)
            .then(((pl.col("A") + 1) ** 2 - 1) / 2)
            .otherwise(-(-pl.col("A")).log1p())
            .alias("A__yeojonhson")
        ]
    )
    assert_frame_equal(transformed_X, expected_X)


def test_transform_lambda_non_zero_non_two(sample_data):
    transformer = YeoJohnson(lambdas={"A": 1.5}, drop_columns=False, inplace=False)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)
    expected_X = sample_data.with_columns(
        [
            pl.when(pl.col("A") >= 0)
            .then(((pl.col("A") + 1) ** 1.5 - 1) / 1.5)
            .otherwise(-((-pl.col("A") + 1) ** 0.5 - 1) / 0.5)
            .alias("A__yeojonhson")
        ]
    )
    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    pytest.main()


def test_yeojohnson_inplace_true_default():
    X = pl.DataFrame({"A": [0, 1, 2], "B": [-1, 0, 1], "C": ["a", "b", "c"]})
    scaler = YeoJohnson(lambdas={"A": 0.5, "B": 1.5}).fit(X)
    result = scaler.transform(X)
    assert "A" in result.columns
    assert "B" in result.columns
    assert "A__yeojonhson" not in result.columns


def test_yeojohnson_inplace_true_lambda_zero():
    X = pl.DataFrame({"A": [-1, 0, 1, 2]})
    scaler = YeoJohnson(lambdas={"A": 0}).fit(X)
    result = scaler.transform(X)
    assert result.columns == ["A"]
    assert "A__yeojonhson" not in result.columns


def test_yeojohnson_inplace_true_lambda_two():
    X = pl.DataFrame({"A": [-2, -1, 0, 1, 2]})
    scaler = YeoJohnson(lambdas={"A": 2}).fit(X)
    result = scaler.transform(X)
    assert result.columns == ["A"]


def test_yeojohnson_inplace_true_inverse_transform():
    X = pl.DataFrame({"A": [0.0, 1.0, 2.0, 5.0]})
    scaler = YeoJohnson(lambdas={"A": 0.5}).fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["A"].cast(pl.Float64) - X["A"].cast(pl.Float64)).abs().max() < 1e-5


def test_yeojohnson_inplace_true_inverse_transform_negative():
    X = pl.DataFrame({"A": [-3.0, -1.0, 0.0, 1.0]})
    scaler = YeoJohnson(lambdas={"A": 0.5}).fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["A"].cast(pl.Float64) - X["A"].cast(pl.Float64)).abs().max() < 1e-5


def test_yeojohnson_get_params_includes_inplace():
    scaler = YeoJohnson(lambdas={"A": 0.5})
    params = scaler.get_params()
    assert "inplace" in params
    assert params["inplace"] is True


def test_yeojohnson_inplace_true_inverse_transform_missing_column_skipped():
    """Test that missing columns are silently skipped in inplace=True inverse_transform."""
    import polars as pl
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    scaler = YeoJohnson(lambdas={"A": 0.5, "B": 0.5}).fit(X)
    X_partial = pl.DataFrame({"A": [0.5, 1.0]})  # only A, missing B
    result = scaler.inverse_transform(X_partial)
    assert "A" in result.columns
    assert "B" not in result.columns
