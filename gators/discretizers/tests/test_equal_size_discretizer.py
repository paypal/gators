import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.discretizers import EqualSizeDiscretizer


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "A": [0.92, 0.8, 0.62, 0.53, 0.21, 0.69, 0.63, 0.21, 0.6, 0.91],
            "B": [2, 35, 11, 42, 70, 97, 70, 99, 51, 62],
            "C": [0.99, 0.28, 0.85, 0.36, 0.87, 0.36, 0.7, 0.46, 0.8, 0.29],
            "D": [73, 17, 77, 29, 28, 51, 39, 62, 61, 89],
        }
    )


def expected_data_default_parameters():
    return pl.DataFrame(
        {
            "A__discretize_size": [
                "(0.69,inf)",
                "(0.69,inf)",
                "(0.6,0.69]",
                "(-inf,0.6]",
                "(-inf,0.6]",
                "(0.6,0.69]",
                "(0.6,0.69]",
                "(-inf,0.6]",
                "(-inf,0.6]",
                "(0.69,inf)",
            ],
            "B__discretize_size": [
                "(-inf,42.0]",
                "(-inf,42.0]",
                "(-inf,42.0]",
                "(-inf,42.0]",
                "(42.0,70.0]",
                "(70.0,inf)",
                "(42.0,70.0]",
                "(70.0,inf)",
                "(42.0,70.0]",
                "(42.0,70.0]",
            ],
            "C__discretize_size": [
                "(0.8,inf)",
                "(-inf,0.36]",
                "(0.8,inf)",
                "(-inf,0.36]",
                "(0.8,inf)",
                "(-inf,0.36]",
                "(0.36,0.8]",
                "(0.36,0.8]",
                "(0.36,0.8]",
                "(-inf,0.36]",
            ],
            "D__discretize_size": [
                "(62.0,inf)",
                "(-inf,39.0]",
                "(62.0,inf)",
                "(-inf,39.0]",
                "(-inf,39.0]",
                "(39.0,62.0]",
                "(-inf,39.0]",
                "(39.0,62.0]",
                "(39.0,62.0]",
                "(62.0,inf)",
            ],
        }
    )


def expected_data_subset_columns():
    return pl.DataFrame(
        {
            "C": [0.99, 0.28, 0.85, 0.36, 0.87, 0.36, 0.7, 0.46, 0.8, 0.29],
            "D": [73, 17, 77, 29, 28, 51, 39, 62, 61, 89],
            "A__discretize_size": [
                "(0.69,inf)",
                "(0.69,inf)",
                "(0.6,0.69]",
                "(-inf,0.6]",
                "(-inf,0.6]",
                "(0.6,0.69]",
                "(0.6,0.69]",
                "(-inf,0.6]",
                "(-inf,0.6]",
                "(0.69,inf)",
            ],
            "B__discretize_size": [
                "(-inf,42.0]",
                "(-inf,42.0]",
                "(-inf,42.0]",
                "(-inf,42.0]",
                "(42.0,70.0]",
                "(70.0,inf)",
                "(42.0,70.0]",
                "(70.0,inf)",
                "(42.0,70.0]",
                "(42.0,70.0]",
            ],
        }
    )


def test_default_parameters(sample_data):
    discretizer = EqualSizeDiscretizer(num_bins=3, inplace=False)
    discretizer.fit(sample_data)
    assert set(discretizer.subset) == {"A", "B", "C", "D"}
    assert len(discretizer._bins) == 4
    assert len(discretizer._labels) == 4
    transformed_X = discretizer.transform(sample_data)
    transformed_X = transformed_X.with_columns(
        [pl.col(col).cast(pl.String) for col in discretizer._column_mapping.values()]
    )
    assert_frame_equal(transformed_X, expected_data_default_parameters())


def test_subset_columns(sample_data):
    discretizer = EqualSizeDiscretizer(num_bins=3, subset=["A", "B"], inplace=False)
    discretizer.fit(sample_data)
    assert set(discretizer.subset) == {"A", "B"}
    assert len(discretizer._bins) == 2
    assert len(discretizer._labels) == 2
    transformed_X = discretizer.transform(sample_data)
    transformed_X = transformed_X.with_columns(
        [pl.col(col).cast(pl.String) for col in discretizer._column_mapping.values()]
    )
    assert_frame_equal(transformed_X, expected_data_subset_columns())


def test_as_numerics(sample_data):
    """Test with as_numerics=True, should generate numeric labels."""
    discretizer = EqualSizeDiscretizer(num_bins=3, subset=["A"], inplace=False, as_numerics=True)
    discretizer.fit(sample_data)
    transformed_X = discretizer.transform(sample_data)

    # Labels should be numeric integers 0, 1, 2
    unique_labels = transformed_X["A__discretize_size"].unique().sort().to_list()
    # Should have numeric labels
    assert all(label in [0, 1, 2] for label in unique_labels)
