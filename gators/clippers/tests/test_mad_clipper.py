import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.clippers.mad_clipper import MADClipper


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with outliers for testing."""
    return pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 100.0],
            "B": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
            "C": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
            "D": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],
        }
    )


def test_mad_clipper_default_inplace(sample_dataframe):
    """Test clipping with default parameters (inplace=True, n_mads=3)."""
    clipper = MADClipper(inplace=True)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Verify the shape
    assert transformed.shape == (13, 4)
    assert "D" in transformed.columns

    # Verify outliers are clipped
    assert transformed["A"][12] < 100.0
    assert transformed["B"][0] > -100.0


def test_mad_clipper_not_inplace(sample_dataframe):
    """Test clipping with inplace=False."""
    clipper = MADClipper(n_mads=3.0, inplace=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have new columns instead of original numeric columns
    assert "A__clip_mad" in transformed.columns
    assert "B__clip_mad" in transformed.columns
    assert "C__clip_mad" in transformed.columns
    assert "A" not in transformed.columns
    assert "D" in transformed.columns


def test_mad_clipper_not_inplace_no_drop(sample_dataframe):
    """Test clipping with inplace=False and drop_columns=False."""
    clipper = MADClipper(inplace=False, drop_columns=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have both original and clipped columns
    assert "A" in transformed.columns
    assert "A__clip_mad" in transformed.columns
    assert "B" in transformed.columns
    assert "B__clip_mad" in transformed.columns


def test_mad_clipper_subset(sample_dataframe):
    """Test clipping only a subset of columns."""
    clipper = MADClipper(subset=["A"], inplace=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have clipped version of only column A
    assert "A__clip_mad" in transformed.columns
    assert "B__clip_mad" not in transformed.columns
    assert "B" in transformed.columns
    assert "C" in transformed.columns


def test_mad_clipper_different_n_mads(sample_dataframe):
    """Test clipping with different n_mads values."""
    clipper_2 = MADClipper(n_mads=2.0, inplace=False)
    clipper_2.fit(sample_dataframe)
    transformed_2 = clipper_2.transform(sample_dataframe)

    clipper_3 = MADClipper(n_mads=3.0, inplace=False)
    clipper_3.fit(sample_dataframe)
    transformed_3 = clipper_3.transform(sample_dataframe)

    # More aggressive clipping (n_mads=2) should clip more
    assert transformed_2["A__clip_mad"][12] <= transformed_3["A__clip_mad"][12]


def test_mad_clipper_fit_transform():
    """Test fit_transform method."""
    X = pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 100.0],
            "B": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
        }
    )
    clipper = MADClipper(n_mads=3.0, inplace=False)
    transformed = clipper.fit_transform(X)

    assert "A__clip_mad" in transformed.columns
    assert "B__clip_mad" in transformed.columns
    assert transformed["A__clip_mad"][12] < 100.0
    assert transformed["B__clip_mad"][0] > -100.0


def test_mad_clipper_validation():
    """Test parameter validation."""
    # n_mads must be positive
    with pytest.raises(Exception):  # Pydantic validation error
        MADClipper(n_mads=0.0)

    with pytest.raises(Exception):  # Pydantic validation error
        MADClipper(n_mads=-1.0)


def test_mad_clipper_empty_subset():
    """Test behavior when subset is explicitly set to empty list."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": ["x", "y", "z"],
        }
    )
    clipper = MADClipper(subset=[], inplace=False)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Empty list triggers auto-selection, so A should be clipped
    assert "A__clip_mad" in transformed.columns
    assert "B" in transformed.columns


def test_mad_clipper_single_column():
    """Test clipping with a single column."""
    X = pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 100.0],
        }
    )
    clipper = MADClipper(n_mads=3.0, inplace=True)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Outlier should be clipped
    assert transformed["A"][12] < 100.0
    assert len(transformed.columns) == 1


def test_mad_clipper_compute_bounds():
    """Test that MAD bounds are computed correctly."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],  # median=3, MAD=median(|[2,1,0,1,2]|)=1
        }
    )
    clipper = MADClipper(n_mads=2.0, inplace=False)
    clipper.fit(X)

    # Bounds should be approximately: median ± 2*MAD = 3 ± 2*1 = [1, 5]
    lower, upper = clipper._clip_bounds["A"]
    assert abs(lower - 1.0) < 0.5
    assert abs(upper - 5.0) < 0.5


def test_mad_clipper_no_outliers():
    """Test clipping on data without extreme outliers."""
    X = pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "B": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        }
    )
    clipper = MADClipper(n_mads=3.0, inplace=False)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # With uniformly distributed data, minimal clipping should occur
    assert abs(transformed["A__clip_mad"].sum() - X["A"].sum()) < 5.0
