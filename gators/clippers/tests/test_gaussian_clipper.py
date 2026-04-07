import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.clippers.gaussian_clipper import GaussianClipper


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with outliers for testing.
    
    Note: Uses data with many values to establish distribution,
    then adds outliers that will be clipped.
    """
    return pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 100.0],  # 100.0 is an outlier
            "B": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],  # -100.0 is an outlier
            "C": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],  # No outliers
            "D": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z"],  # String column (should be ignored)
        }
    )


@pytest.fixture
def sample_dataframe_no_outliers():
    """Sample DataFrame without outliers."""
    return pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [5.0, 6.0, 7.0, 8.0, 9.0],
            "C": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def test_clipper_default_inplace(sample_dataframe):
    """Test clipping with default parameters (inplace=True, n_sigmas=3)."""
    clipper = GaussianClipper(inplace=True)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Verify the shape
    assert transformed.shape == (12, 4)
    assert "D" in transformed.columns  # String column should remain

    # Verify outliers are clipped
    assert transformed["A"][11] < 100.0  # Outlier should be clipped
    assert transformed["B"][0] > -100.0  # Outlier should be clipped


def test_clipper_not_inplace(sample_dataframe):
    """Test clipping with inplace=False."""
    clipper = GaussianClipper(n_sigmas=3, inplace=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have new columns instead of original numeric columns
    assert "A__clip_gaussian" in transformed.columns
    assert "B__clip_gaussian" in transformed.columns
    assert "C__clip_gaussian" in transformed.columns
    assert "A" not in transformed.columns  # Original should be dropped
    assert "D" in transformed.columns  # String column should remain


def test_clipper_not_inplace_no_drop(sample_dataframe):
    """Test clipping with inplace=False and drop_columns=False."""
    clipper = GaussianClipper(n_sigmas=3, inplace=False, drop_columns=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have both original and clipped columns
    assert "A" in transformed.columns
    assert "A__clip_gaussian" in transformed.columns
    assert "B" in transformed.columns
    assert "B__clip_gaussian" in transformed.columns
    assert "D" in transformed.columns


def test_clipper_subset(sample_dataframe):
    """Test clipping only a subset of columns."""
    clipper = GaussianClipper(n_sigmas=3, subset=["A"], inplace=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have clipped version of only column A
    assert "A__clip_gaussian" in transformed.columns
    assert "B__clip_gaussian" not in transformed.columns
    assert "B" in transformed.columns  # Original B should remain
    assert "C" in transformed.columns  # Original C should remain
    assert "D" in transformed.columns


def test_clipper_n_sigmas_2(sample_dataframe):
    """Test clipping with n_sigmas=2 (more aggressive)."""
    clipper_2 = GaussianClipper(n_sigmas=2, inplace=False)
    clipper_2.fit(sample_dataframe)
    transformed_2 = clipper_2.transform(sample_dataframe)

    clipper_3 = GaussianClipper(n_sigmas=3, inplace=False)
    clipper_3.fit(sample_dataframe)
    transformed_3 = clipper_3.transform(sample_dataframe)

    # With n_sigmas=2, outliers should be clipped more aggressively
    # The clipped value for column A should be smaller with n_sigmas=2
    assert transformed_2["A__clip_gaussian"][11] <= transformed_3["A__clip_gaussian"][11]
    # At least one of them should actually clip the outlier
    assert (transformed_2["A__clip_gaussian"][11] < 100.0 or 
            transformed_3["A__clip_gaussian"][11] < 100.0)


def test_clipper_no_outliers(sample_dataframe_no_outliers):
    """Test clipping on data without outliers (should remain unchanged)."""
    clipper = GaussianClipper(n_sigmas=3, inplace=False)
    clipper.fit(sample_dataframe_no_outliers)
    transformed = clipper.transform(sample_dataframe_no_outliers)

    # With no significant outliers, values should remain mostly the same
    # (or change only slightly at the edges)
    original_sum = sample_dataframe_no_outliers.select(pl.sum("A", "B", "C")).row(0)
    transformed_sum = transformed.select(
        pl.sum("A__clip_gaussian", "B__clip_gaussian", "C__clip_gaussian")
    ).row(0)

    # Sums should be very close (within small tolerance)
    assert abs(original_sum[0] - transformed_sum[0]) < 1.0
    assert abs(original_sum[1] - transformed_sum[1]) < 1.0
    assert abs(original_sum[2] - transformed_sum[2]) < 1.0


def test_clipper_compute_bounds():
    """Test that clipping bounds are computed correctly."""
    X = pl.DataFrame(
        {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],  # mean=2.0, std≈1.58
        }
    )
    clipper = GaussianClipper(n_sigmas=2, inplace=False)
    clipper.fit(X)

    # Bounds should be approximately: mean ± 2*std = 2.0 ± 2*1.58
    lower, upper = clipper._clip_bounds["A"]
    assert lower < 0.0  # Should be negative
    assert upper > 4.0  # Should be greater than 4


def test_clipper_fit_transform():
    """Test fit_transform method."""
    X = pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 100.0],
            "B": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
        }
    )
    clipper = GaussianClipper(n_sigmas=3, inplace=False)
    transformed = clipper.fit_transform(X)

    assert "A__clip_gaussian" in transformed.columns
    assert "B__clip_gaussian" in transformed.columns
    assert transformed["A__clip_gaussian"][12] < 100.0
    assert transformed["B__clip_gaussian"][0] > -100.0


def test_clipper_n_sigmas_validation():
    """Test that n_sigmas must be positive."""
    with pytest.raises(Exception):  # Pydantic will raise a validation error
        GaussianClipper(n_sigmas=0)

    with pytest.raises(Exception):  # Pydantic will raise a validation error
        GaussianClipper(n_sigmas=-1)


def test_clipper_empty_subset():
    """Test behavior when subset is explicitly set to empty list.
    
    Note: Empty list is falsy, so it triggers auto-selection of numeric columns.
    """
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": ["x", "y", "z"],
        }
    )
    clipper = GaussianClipper(subset=[], inplace=False)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Empty list triggers auto-selection, so A should be clipped
    assert "A__clip_gaussian" in transformed.columns
    assert "B" in transformed.columns


def test_clipper_single_column():
    """Test clipping with a single column."""
    X = pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 100.0],
        }
    )
    clipper = GaussianClipper(n_sigmas=3, inplace=True)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Outlier should be clipped
    assert transformed["A"][12] < 100.0
    assert len(transformed.columns) == 1
