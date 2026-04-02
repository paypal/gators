import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.discretizers._base_discretizer import _BaseDiscretizer


class ExampleDiscretizer(_BaseDiscretizer):
    def fit(self, X):
        self._column_mapping = {"A": "A__dicretize", "B": "B__dicretize"}
        self._bins = {"A": [0.5], "B": [5]}
        self._labels = {"A": ["a1", "a2"], "B": ["b1", "b2"]}
        return self


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "A": [0.92, 0.8, 0.62, 0.53, 0.21, 0.69, 0.63, 0.21, 0.6, 0.91],
            "B": [2, 35, 11, 42, 70, 97, 70, 99, 51, 62],
        }
    )


def expected_data_default_parameters():
    return pl.DataFrame(
        {
            "A__dicretize": [
                "a2",
                "a2",
                "a2",
                "a2",
                "a1",
                "a2",
                "a2",
                "a1",
                "a2",
                "a2",
            ],
            "B__dicretize": [
                "b1",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
            ],
        }
    )


def expected_data_drop_columns_false():
    return pl.DataFrame(
        {
            "A": [0.92, 0.8, 0.62, 0.53, 0.21, 0.69, 0.63, 0.21, 0.6, 0.91],
            "B": [2, 35, 11, 42, 70, 97, 70, 99, 51, 62],
            "A__dicretize": [
                "a2",
                "a2",
                "a2",
                "a2",
                "a1",
                "a2",
                "a2",
                "a1",
                "a2",
                "a2",
            ],
            "B__dicretize": [
                "b1",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
                "b2",
            ],
        }
    )


def test_default_parameters(sample_data):
    discretizer = ExampleDiscretizer(subset=["A", "B"], inplace=False)
    discretizer.fit(sample_data)
    assert set(discretizer.subset) == {"A", "B"}
    assert len(discretizer._bins) == 2
    assert len(discretizer._labels) == 2
    transformed_X =discretizer.transform(sample_data)
    transformed_X =transformed_X.with_columns(
        [pl.col(col).cast(pl.String) for col in discretizer._column_mapping.values()]
    )
    assert_frame_equal(transformed_X, expected_data_default_parameters())


def test_subset_columns(sample_data):
    discretizer = ExampleDiscretizer(
        subset=["A", "B"], drop_columns=False, inplace=False
    )
    discretizer.fit(sample_data)
    assert set(discretizer.subset) == {"A", "B"}
    assert len(discretizer._bins) == 2
    assert len(discretizer._labels) == 2
    transformed_X =discretizer.transform(sample_data)
    transformed_X =transformed_X.with_columns(
        [pl.col(col).cast(pl.String) for col in discretizer._column_mapping.values()]
    )
    assert_frame_equal(transformed_X, expected_data_drop_columns_false())


def test_inplace_true():
    """Test inplace discretization (replaces original columns)."""
    X =pl.DataFrame({"A": [0.1, 0.2, 0.8, 0.9], "B": [5, 15, 25, 35]})

    discretizer = ExampleDiscretizer(subset=["A", "B"], inplace=True)
    discretizer._bins = {"A": [0.5], "B": [20]}
    discretizer._labels = {"A": ["low", "high"], "B": ["small", "large"]}

    result = discretizer.transform(X)

    # When inplace=True, columns should be replaced
    assert "A" in result.columns
    assert "B" in result.columns
    # Polars cut returns Categorical dtype
    assert result["A"].dtype == pl.Categorical
    assert result["B"].dtype == pl.Categorical
    assert result["A"].to_list() == ["low", "low", "high", "high"]
    assert result["B"].to_list() == ["small", "small", "large", "large"]


def test_as_numerics_true():
    """Test discretization with numeric output."""
    X =pl.DataFrame({"A": [0.1, 0.2, 0.8, 0.9], "B": [5, 15, 25, 35]})

    discretizer = ExampleDiscretizer(
        subset=["A", "B"], as_numerics=True, drop_columns=True, inplace=False
    )
    discretizer._column_mapping = {"A": "A__disc", "B": "B__disc"}
    discretizer._bins = {"A": [0.5], "B": [20]}
    discretizer._labels = {"A": ["low", "high"], "B": ["small", "large"]}

    result = discretizer.transform(X)

    # Should return numeric bins (indices of the categorical values)
    assert "A__disc" in result.columns
    assert "B__disc" in result.columns
    assert result["A__disc"].dtype == pl.Int32
    assert result["B__disc"].dtype == pl.Int32
    # Just verify the values are consistent
    assert len(set(result["A__disc"].to_list())) == 2  # Two unique values
    assert len(set(result["B__disc"].to_list())) == 2


def test_inplace_with_as_numerics():
    """Test inplace discretization with numeric output."""
    X =pl.DataFrame({"A": [0.1, 0.2, 0.8, 0.9], "value": [10, 20, 30, 40]})

    discretizer = ExampleDiscretizer(subset=["A"], inplace=True, as_numerics=True)
    discretizer._bins = {"A": [0.5]}
    discretizer._labels = {"A": ["low", "high"]}

    result = discretizer.transform(X)

    # Column A should be replaced with numeric bins
    assert "A" in result.columns
    assert result["A"].dtype == pl.Int32
    # Just verify the values are consistent
    assert len(set(result["A"].to_list())) == 2  # Two unique values
    assert "value" in result.columns


def test_generate_labels_function():
    """Test the generate_labels helper function."""
    from gators.discretizers._base_discretizer import generate_labels

    bins = {"A": [0.5, 1.0], "B": [10, 20, 30]}

    labels = generate_labels(bins, rounding=1)

    assert "A" in labels
    assert "B" in labels
    assert labels["A"] == ["(-inf,0.5]", "(0.5,1.0]", "(1.0,inf)"]
    # B values are integers so rounding shows them as integers
    assert len(labels["B"]) == 4
    assert labels["B"][0].startswith("(-inf,")
    assert labels["B"][-1].endswith(",inf)")


def test_generate_labels_empty_bins():
    """Test generate_labels with empty bins (constant feature)."""
    from gators.discretizers._base_discretizer import generate_labels

    bins = {"constant_col": []}
    labels = generate_labels(bins)

    assert labels["constant_col"] == ["constant"]


if __name__ == "__main__":
    pytest.main()
