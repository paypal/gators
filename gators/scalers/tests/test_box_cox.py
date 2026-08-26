import polars as pl
import pytest
from polars.testing import assert_frame_equal
import math

from gators.scalers import BoxCox


def test_transform_default():
    """Test BoxCox with default drop_columns=True."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0, "B": 2}
    scaler = BoxCox(lambdas=lambdas, inplace=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            pl.col("A").log().alias("A__boxcox"),
            ((pl.col("B") ** 2 - 1) / 2).alias("B__boxcox"),
        ]
    ).drop(["A", "B"])

    assert_frame_equal(transformed_X, expected_X)


def test_transform_subset_columns():
    """Test BoxCox with subset of columns."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0}
    scaler = BoxCox(lambdas=lambdas, inplace=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns([pl.col("A").log().alias("A__boxcox")]).drop(["A"])

    assert_frame_equal(transformed_X, expected_X)


def test_transform_drop_columns_false():
    """Test BoxCox with drop_columns=False."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": ["a", "b", "c", "d", "e"],
        }
    )

    lambdas = {"A": 0, "B": 2}
    scaler = BoxCox(lambdas=lambdas, drop_columns=False, inplace=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            pl.col("A").log().alias("A__boxcox"),
            ((pl.col("B") ** 2 - 1) / 2).alias("B__boxcox"),
        ]
    )

    assert_frame_equal(transformed_X, expected_X)


@pytest.fixture
def sample_data():
    """Positive values only for Box-Cox."""
    return pl.DataFrame({"A": [0.5, 1.0, 2.0, 5.0], "B": [1.0, 2.0, 3.0, 4.0]})


def test_transform_lambda_zero(sample_data):
    """Test Box-Cox with lambda=0 (log transformation)."""
    transformer = BoxCox(lambdas={"A": 0}, drop_columns=False, inplace=False)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)
    expected_X = sample_data.with_columns([pl.col("A").log().alias("A__boxcox")])
    assert_frame_equal(transformed_X, expected_X)


def test_transform_lambda_one(sample_data):
    """Test Box-Cox with lambda=1 (linear transformation: x - 1)."""
    transformer = BoxCox(lambdas={"A": 1}, drop_columns=False, inplace=False)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)
    expected_X = sample_data.with_columns([((pl.col("A") ** 1 - 1) / 1).alias("A__boxcox")])
    assert_frame_equal(transformed_X, expected_X)


def test_transform_lambda_half():
    """Test Box-Cox with lambda=0.5 (square root transformation)."""
    X = pl.DataFrame({"A": [1.0, 4.0, 9.0, 16.0]})
    transformer = BoxCox(lambdas={"A": 0.5}, drop_columns=False, inplace=False)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    expected_X = X.with_columns([((pl.col("A") ** 0.5 - 1) / 0.5).alias("A__boxcox")])
    assert_frame_equal(transformed_X, expected_X)


def test_transform_lambda_negative():
    """Test Box-Cox with negative lambda."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    transformer = BoxCox(lambdas={"A": -1}, drop_columns=False, inplace=False)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    expected_X = X.with_columns([((pl.col("A") ** -1 - 1) / -1).alias("A__boxcox")])
    assert_frame_equal(transformed_X, expected_X)


def test_transform_multiple_columns():
    """Test Box-Cox with multiple columns and different lambdas."""
    X = pl.DataFrame(
        {
            "sales": [10.0, 20.0, 30.0, 40.0],
            "price": [5.0, 15.0, 25.0, 35.0],
            "quantity": [1.0, 2.0, 3.0, 4.0],
        }
    )

    lambdas = {"sales": 0.5, "price": 0, "quantity": 2}
    scaler = BoxCox(lambdas=lambdas, inplace=False).fit(X)
    transformed_X = scaler.transform(X)

    expected_X = X.with_columns(
        [
            ((pl.col("sales") ** 0.5 - 1) / 0.5).alias("sales__boxcox"),
            pl.col("price").log().alias("price__boxcox"),
            ((pl.col("quantity") ** 2 - 1) / 2).alias("quantity__boxcox"),
        ]
    ).drop(["sales", "price", "quantity"])

    assert_frame_equal(transformed_X, expected_X)


def test_transform_fit_transform():
    """Test BoxCox fit_transform method."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    scaler = BoxCox(lambdas={"A": 0}, inplace=False)
    transformed_X = scaler.fit_transform(X)

    expected_X = pl.DataFrame(
        {"A__boxcox": [math.log(1.0), math.log(2.0), math.log(3.0), math.log(4.0)]}
    )

    assert_frame_equal(transformed_X, expected_X)


def test_boxcox_values():
    """Test that Box-Cox transformation produces expected numerical values."""
    X = pl.DataFrame({"A": [1.0, 2.0, 4.0, 8.0]})

    # Lambda = 0 (log transformation)
    scaler_log = BoxCox(lambdas={"A": 0}, inplace=False)
    result_log = scaler_log.fit_transform(X)
    expected_log = [math.log(1.0), math.log(2.0), math.log(4.0), math.log(8.0)]
    assert result_log["A__boxcox"].to_list() == pytest.approx(expected_log)

    # Lambda = 0.5
    scaler_sqrt = BoxCox(lambdas={"A": 0.5}, inplace=False)
    result_sqrt = scaler_sqrt.fit_transform(X)
    expected_sqrt = [
        (1.0**0.5 - 1) / 0.5,
        (2.0**0.5 - 1) / 0.5,
        (4.0**0.5 - 1) / 0.5,
        (8.0**0.5 - 1) / 0.5,
    ]
    assert result_sqrt["A__boxcox"].to_list() == pytest.approx(expected_sqrt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_boxcox_inplace_true_default():
    X = pl.DataFrame({"A": [1.0, 2.0, 4.0, 8.0], "B": [1.0, 2.0, 3.0, 4.0]})
    scaler = BoxCox(lambdas={"A": 0.5, "B": 0}).fit(X)
    result = scaler.transform(X)
    assert result.columns == ["A", "B"]
    assert "A__boxcox" not in result.columns
    expected_A = [(1.0**0.5 - 1) / 0.5, (2.0**0.5 - 1) / 0.5, (4.0**0.5 - 1) / 0.5, (8.0**0.5 - 1) / 0.5]
    assert result["A"].to_list() == pytest.approx(expected_A)


def test_boxcox_inplace_true_lambda_zero():
    X = pl.DataFrame({"A": [1.0, 2.0, 4.0]})
    scaler = BoxCox(lambdas={"A": 0}).fit(X)
    result = scaler.transform(X)
    assert result.columns == ["A"]
    assert result["A"].to_list() == pytest.approx([math.log(1), math.log(2), math.log(4)])


def test_boxcox_inplace_true_inverse_transform():
    X = pl.DataFrame({"A": [1.0, 2.0, 4.0, 8.0]})
    scaler = BoxCox(lambdas={"A": 0.5}).fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["A"].cast(pl.Float64) - X["A"].cast(pl.Float64)).abs().max() < 1e-5


def test_boxcox_inplace_true_inverse_transform_lambda_zero():
    X = pl.DataFrame({"A": [1.0, math.e, math.e**2]})
    scaler = BoxCox(lambdas={"A": 0}).fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["A"].cast(pl.Float64) - X["A"].cast(pl.Float64)).abs().max() < 1e-5


def test_boxcox_get_params_includes_inplace():
    scaler = BoxCox(lambdas={"A": 0.5})
    params = scaler.get_params()
    assert "inplace" in params
    assert params["inplace"] is True


def test_boxcox_inplace_true_inverse_transform_missing_column_skipped():
    """Test that missing columns are silently skipped in inplace=True inverse_transform."""
    import polars as pl
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    scaler = BoxCox(lambdas={"A": 0.5, "B": 0.5}).fit(X)
    X_partial = pl.DataFrame({"A": [0.414, 0.828]})  # only A, missing B
    result = scaler.inverse_transform(X_partial)
    assert "A" in result.columns
    assert "B" not in result.columns
