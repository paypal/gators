import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.scalers import StandardScaler


def test_standard_scaler_default():
    X = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
            "col3": [100, 200, 300, 400, 500],
        }
    )

    scaler = StandardScaler(inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)
    expected = X.with_columns(
        [
            ((pl.col("col1") - X["col1"].mean()) / X["col1"].std()).alias("col1__standard_scale"),
            ((pl.col("col2") - X["col2"].mean()) / X["col2"].std()).alias("col2__standard_scale"),
            ((pl.col("col3") - X["col3"].mean()) / X["col3"].std()).alias("col3__standard_scale"),
        ]
    ).drop(["col1", "col2", "col3"])
    assert_frame_equal(result, expected)


def test_standard_scaler_subset_columns():
    X = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
            "col3": [100, 200, 300, 400, 500],
        }
    )

    scaler = StandardScaler(subset=["col1", "col2"], drop_columns=False, inplace=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            ((pl.col("col1") - X["col1"].mean()) / X["col1"].std()).alias("col1__standard_scale"),
            ((pl.col("col2") - X["col2"].mean()) / X["col2"].std()).alias("col2__standard_scale"),
        ]
    )

    assert_frame_equal(result, expected)


if __name__ == "__main__":
    pytest.main()


def test_standard_scaler_inplace_true_default():
    X = pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    scaler = StandardScaler().fit(X)
    result = scaler.transform(X)
    assert result.columns == ["col1", "col2"]
    assert "col1__standard_scale" not in result.columns
    assert result["col1"].mean() == pytest.approx(0.0, abs=1e-10)


def test_standard_scaler_inplace_true_subset():
    X = pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0],
            "col2": [10.0, 20.0, 30.0],
            "cat": ["a", "b", "c"],
        }
    )
    scaler = StandardScaler(subset=["col1"]).fit(X)
    result = scaler.transform(X)
    assert result.columns == ["col1", "col2", "cat"]
    assert result["col1"].mean() == pytest.approx(0.0, abs=1e-10)
    assert result["col2"].to_list() == [10.0, 20.0, 30.0]


def test_standard_scaler_inverse_transform_inplace_true():
    X = pl.DataFrame({"a": [10.0, 20.0, 30.0, 40.0]})
    scaler = StandardScaler().fit(X)
    X_t = scaler.transform(X)
    X_r = scaler.inverse_transform(X_t)
    assert (X_r["a"].cast(pl.Float64) - X["a"].cast(pl.Float64)).abs().max() < 1e-5


def test_standard_scaler_get_params_includes_inplace():
    scaler = StandardScaler()
    params = scaler.get_params()
    assert "inplace" in params
    assert params["inplace"] is True
