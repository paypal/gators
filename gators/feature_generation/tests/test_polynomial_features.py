import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation import PolynomialFeatures


@pytest.fixture
def sample_data():
    return pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


def test_default_parameters(sample_data):
    transformer = PolynomialFeatures(include_bias=True)
    transformer.fit(sample_data)

    transformed_X = transformer.transform(sample_data)
    expected_X = sample_data.with_columns(
        [
            pl.lit(1).alias("bias"),
            (sample_data["A"] * sample_data["A"]).alias("A__A"),
            (sample_data["A"] * sample_data["B"]).alias("A__B"),
            (sample_data["A"] * sample_data["C"]).alias("A__C"),
            (sample_data["B"] * sample_data["B"]).alias("B__B"),
            (sample_data["B"] * sample_data["C"]).alias("B__C"),
            (sample_data["C"] * sample_data["C"]).alias("C__C"),
        ]
    )
    assert_frame_equal(transformed_X, expected_X)


def test_columns_as_subset(sample_data):
    transformer = PolynomialFeatures(subset=["A", "B"], include_bias=True)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    expected_X = sample_data.with_columns(
        [
            pl.lit(1).alias("bias"),
            (sample_data["A"] * sample_data["A"]).alias("A__A"),
            (sample_data["A"] * sample_data["B"]).alias("A__B"),
            (sample_data["B"] * sample_data["B"]).alias("B__B"),
        ]
    )

    assert_frame_equal(transformed_X, expected_X)


def test_interaction_only(sample_data):
    transformer = PolynomialFeatures(interaction_only=True, include_bias=False)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    expected_X = sample_data.with_columns(
        [
            (sample_data["A"] * sample_data["B"]).alias("A__B"),
            (sample_data["A"] * sample_data["C"]).alias("A__C"),
            (sample_data["B"] * sample_data["C"]).alias("B__C"),
        ]
    )

    assert_frame_equal(transformed_X, expected_X)
