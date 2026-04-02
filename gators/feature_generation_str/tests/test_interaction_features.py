import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import InteractionFeatures


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "A": ["foo", "bar", "baz"],
            "B": ["qux", "quux", "quuz"],
            "C": ["quuz", "corge", "grault"],
        }
    )


@pytest.fixture
def transformer():
    return InteractionFeatures()


def test_default_parameters(sample_data, transformer):
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    expected_X = sample_data.with_columns(
        [
            (sample_data["A"] + "_" + sample_data["B"]).alias("A__B"),
            (sample_data["A"] + "_" + sample_data["C"]).alias("A__C"),
            (sample_data["B"] + "_" + sample_data["C"]).alias("B__C"),
        ]
    )

    assert_frame_equal(transformed_X, expected_X)


def test_columns_as_subset(sample_data):
    transformer = InteractionFeatures(subset=["A", "B"])
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    expected_X = sample_data.with_columns(
        [(sample_data["A"] + "_" + sample_data["B"]).alias("A__B")]
    )

    assert_frame_equal(transformed_X, expected_X)


def test_degree_three(sample_data):
    transformer = InteractionFeatures(degree=3)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    expected_X = sample_data.with_columns(
        [
            (sample_data["A"] + "_" + sample_data["B"]).alias("A__B"),
            (sample_data["A"] + "_" + sample_data["C"]).alias("A__C"),
            (sample_data["B"] + "_" + sample_data["C"]).alias("B__C"),
            (sample_data["A"] + "_" + sample_data["B"] + "_" + sample_data["C"]).alias("A__B__C"),
        ]
    )

    assert_frame_equal(transformed_X, expected_X)
