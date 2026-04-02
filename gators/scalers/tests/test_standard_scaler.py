import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.scalers import StandardScaler


def test_standard_scaler_default():
    X =  pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
            "col3": [100, 200, 300, 400, 500],
        }
    )

    scaler = StandardScaler()
    scaler.fit(X)
    result = scaler.transform(X)
    expected = X.with_columns(
        [
            ((pl.col("col1") - X["col1"].mean()) / X["col1"].std()).alias(
                "col1__standard_scale"
            ),
            ((pl.col("col2") - X["col2"].mean()) / X["col2"].std()).alias(
                "col2__standard_scale"
            ),
            ((pl.col("col3") - X["col3"].mean()) / X["col3"].std()).alias(
                "col3__standard_scale"
            ),
        ]
    ).drop(["col1", "col2", "col3"])
    assert_frame_equal(result, expected)


def test_standard_scaler_subset_columns():
    X =  pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
            "col3": [100, 200, 300, 400, 500],
        }
    )

    scaler = StandardScaler(subset=["col1", "col2"], drop_columns=False)
    scaler.fit(X)
    result = scaler.transform(X)

    expected = X.with_columns(
        [
            ((pl.col("col1") - X["col1"].mean()) / X["col1"].std()).alias(
                "col1__standard_scale"
            ),
            ((pl.col("col2") - X["col2"].mean()) / X["col2"].std()).alias(
                "col2__standard_scale"
            ),
        ]
    )

    assert_frame_equal(result, expected)


if __name__ == "__main__":
    pytest.main()
