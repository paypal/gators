import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.discretizers import GeometricDiscretizer


@pytest.fixture
def sample_data():
    """Sample data with values spanning multiple orders of magnitude."""
    return pl.DataFrame(
        {
            "A": [1, 10, 100, 1000, 10000],
            "B": [0.1, 1, 10, 100, 1000],
            "C": [2, 4, 8, 16, 32],
            "D": [5, 10, 15, 20, 25],
        }
    )


@pytest.fixture
def sample_data_with_negatives():
    """Sample data with zero and negative values."""
    return pl.DataFrame(
        {
            "A": [-10, 0, 10, 100, 1000],
            "B": [0, 1, 10, 100, 1000],
        }
    )


def test_default_parameters(sample_data):
    """Test with default parameters - all numeric columns."""
    discretizer = GeometricDiscretizer(num_bins=3, inplace=False)
    discretizer.fit(sample_data)

    assert set(discretizer.subset) == {"A", "B", "C", "D"}
    assert len(discretizer._bins) == 4
    assert len(discretizer._labels) == 4

    transformed_X = discretizer.transform(sample_data)

    # Check that new columns were created
    assert "A__discretize_geom" in transformed_X.columns
    assert "B__discretize_geom" in transformed_X.columns
    assert "C__discretize_geom" in transformed_X.columns
    assert "D__discretize_geom" in transformed_X.columns

    # Check that original columns are dropped (drop_columns=True by default)
    assert "A" not in transformed_X.columns


def test_subset_columns(sample_data):
    """Test with subset of columns."""
    discretizer = GeometricDiscretizer(num_bins=3, subset=["A", "B"], inplace=False)
    discretizer.fit(sample_data)

    assert set(discretizer.subset) == {"A", "B"}
    assert len(discretizer._bins) == 2
    assert len(discretizer._labels) == 2

    transformed_X = discretizer.transform(sample_data)

    # Check that only specified columns are discretized
    assert "A__discretize_geom" in transformed_X.columns
    assert "B__discretize_geom" in transformed_X.columns
    # Original non-subset columns should still be there
    assert "C" in transformed_X.columns
    assert "D" in transformed_X.columns


def test_inplace_true(sample_data):
    """Test with inplace=True."""
    discretizer = GeometricDiscretizer(num_bins=3, subset=["A", "B"], inplace=True)
    discretizer.fit(sample_data)
    transformed_X = discretizer.transform(sample_data)

    # Original columns should be replaced
    assert "A" in transformed_X.columns
    assert "B" in transformed_X.columns
    assert "A__discretize_geom" not in transformed_X.columns
    assert "B__discretize_geom" not in transformed_X.columns

    # Check that values are categorical strings
    # assert transformed_X.schema["A"] == pl.String or transformed_X.schema["A"] == pl.Enum


def test_drop_columns_false(sample_data):
    """Test with drop_columns=False and inplace=False."""
    discretizer = GeometricDiscretizer(num_bins=3, subset=["A"], inplace=False, drop_columns=False)
    discretizer.fit(sample_data)
    transformed_X = discretizer.transform(sample_data)

    # Both original and discretized columns should exist
    assert "A" in transformed_X.columns
    assert "A__discretize_geom" in transformed_X.columns


def test_as_numerics(sample_data):
    """Test with as_numerics=True, should generate numeric labels."""
    discretizer = GeometricDiscretizer(num_bins=3, subset=["A"], inplace=False, as_numerics=True)
    discretizer.fit(sample_data)
    transformed_X = discretizer.transform(sample_data)

    # Should have numeric dtype
    assert transformed_X["A__discretize_geom"].dtype in [pl.Int32, pl.Int64]
    # Labels should be numeric integers 0, 1, 2, 3
    unique_labels = transformed_X["A__discretize_geom"].unique().sort().to_list()
    # Should have numeric labels
    assert all(isinstance(label, int) and label in [0, 1, 2, 3] for label in unique_labels)
    assert transformed_X.schema["A__discretize_geom"] == pl.Int32


def test_geometric_progression_bins(sample_data):
    """Test that bins follow geometric progression."""
    discretizer = GeometricDiscretizer(num_bins=4, subset=["A"], inplace=False)
    discretizer.fit(sample_data)

    bins = discretizer._bins["A"]

    # For column A: [1, 10, 100, 1000, 10000]
    # Geometric ratio should be (10000/1)^(1/4) = 10
    # Bins should be approximately: [10, 100, 1000]
    assert len(bins) == 3

    # Check that ratios between consecutive bins are approximately equal
    if len(bins) >= 2:
        ratios = [bins[i + 1] / bins[i] for i in range(len(bins) - 1)]
        # All ratios should be approximately the same (within tolerance)
        for i in range(len(ratios) - 1):
            assert abs(ratios[i] - ratios[i + 1]) / ratios[i] < 0.1  # 10% tolerance


def test_handle_negative_values(sample_data_with_negatives):
    """Test handling of zero and negative values."""
    discretizer = GeometricDiscretizer(num_bins=3, subset=["A", "B"], inplace=False)
    discretizer.fit(sample_data_with_negatives)
    transformed_X = discretizer.transform(sample_data_with_negatives)

    # Should successfully discretize without errors
    assert "A__discretize_geom" in transformed_X.columns
    assert "B__discretize_geom" in transformed_X.columns

    # Check that all rows are assigned to bins
    assert transformed_X["A__discretize_geom"].null_count() == 0
    assert transformed_X["B__discretize_geom"].null_count() == 0


def test_constant_column():
    """Test with constant value column."""
    X = pl.DataFrame({"A": [5, 5, 5, 5, 5], "B": [1, 2, 3, 4, 5]})

    discretizer = GeometricDiscretizer(num_bins=3, inplace=False)
    discretizer.fit(X)
    transformed_X = discretizer.transform(X)

    # Constant column should have empty bins and a single "constant" label
    # or all values in one bin
    assert "A__discretize_geom" in transformed_X.columns


def test_fit_transform(sample_data):
    """Test fit_transform method."""
    discretizer = GeometricDiscretizer(num_bins=3, subset=["A", "B"], inplace=False)
    transformed_X = discretizer.fit_transform(sample_data)

    assert "A__discretize_geom" in transformed_X.columns
    assert "B__discretize_geom" in transformed_X.columns


def test_bins_computed_correctly():
    """Test specific bin computation for known data."""
    # Create data where we know the expected geometric bins
    X = pl.DataFrame({"A": [1, 2, 4, 8, 16, 32, 64, 128]})

    discretizer = GeometricDiscretizer(num_bins=3, subset=["A"], inplace=False)
    discretizer.fit(X)

    bins = discretizer._bins["A"]

    # Min=1, Max=128
    # r = (128/1)^(1/3) ≈ 5.04
    # Expected bins approximately: [1 * 5.04, 1 * 5.04^2] = [5.04, 25.4]
    # Or more precisely using cube root of 128

    assert len(bins) == 2  # num_bins - 1 for 3 bins
    # First bin edge should be around 5-6
    assert 4 < bins[0] < 9
    # Second bin edge should be around 25-32
    assert 20 < bins[1] < 40


def test_rounding_parameter():
    """Test that rounding parameter affects label precision."""
    X = pl.DataFrame({"A": [1.111, 2.222, 3.333, 4.444, 5.555]})

    discretizer_r1 = GeometricDiscretizer(num_bins=2, rounding=1, inplace=False)
    discretizer_r1.fit(X)

    discretizer_r5 = GeometricDiscretizer(num_bins=2, rounding=5, inplace=False)
    discretizer_r5.fit(X)

    # Labels should have different precision
    labels_r1 = discretizer_r1._labels["A"]
    labels_r5 = discretizer_r5._labels["A"]

    # Check that rounding is applied (labels should differ in precision)
    assert labels_r1 != labels_r5
