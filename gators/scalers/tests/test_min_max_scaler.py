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

    scaler = MinmaxScaler(inplace=False).fit(X)
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

    scaler = MinmaxScaler(subset=["A"], inplace=False).fit(X)
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

    scaler = MinmaxScaler(drop_columns=False, inplace=False).fit(X)
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


def test_transform_inplace_true_default():
    X = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )
    scaler = MinmaxScaler().fit(X)
    result = scaler.transform(X)
    expected = pl.DataFrame(
        {
            "A": [0.0, 0.25, 0.5, 0.75, 1.0],
            "B": [0.0, 0.25, 0.5, 0.75, 1.0],
            "C": ["a", "b", "c", "d", "e"],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_inplace_true_subset():
    X = pl.DataFrame(
        {
            "A": [0, 25, 50, 75, 100],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
    )
    scaler = MinmaxScaler(subset=["A"]).fit(X)
    result = scaler.transform(X)
    assert result["A"].to_list() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert "B" in result.columns
    assert "C" in result.columns
    assert "A__minmax_scale" not in result.columns


def test_inverse_transform_inplace_true():
    X = pl.DataFrame({"A": [0.0, 25.0, 50.0, 75.0, 100.0]})
    scaler = MinmaxScaler().fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["A"].cast(pl.Float64) - X["A"].cast(pl.Float64)).abs().max() < 1e-5


def test_get_params_includes_inplace():
    scaler = MinmaxScaler()
    params = scaler.get_params()
    assert "inplace" in params
    assert params["inplace"] is True


def test_set_params_inplace():
    scaler = MinmaxScaler()
    scaler.set_params(inplace=False)
    assert scaler.inplace is False
