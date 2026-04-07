import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.clippers.iqr_clipper import IQRClipper


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


def test_iqr_clipper_default_inplace(sample_dataframe):
    """Test clipping with default parameters (inplace=True, n_iqrs=1.5)."""
    clipper = IQRClipper(inplace=True)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Verify the shape
    assert transformed.shape == (13, 4)
    assert "D" in transformed.columns

    # Verify outliers are clipped
    assert transformed["A"][12] < 100.0
    assert transformed["B"][0] > -100.0


def test_iqr_clipper_not_inplace(sample_dataframe):
    """Test clipping with inplace=False."""
    clipper = IQRClipper(n_iqrs=1.5, inplace=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have new columns instead of original numeric columns
    assert "A__clip_iqr" in transformed.columns
    assert "B__clip_iqr" in transformed.columns
    assert "C__clip_iqr" in transformed.columns
    assert "A" not in transformed.columns
    assert "D" in transformed.columns


def test_iqr_clipper_not_inplace_no_drop(sample_dataframe):
    """Test clipping with inplace=False and drop_columns=False."""
    clipper = IQRClipper(inplace=False, drop_columns=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have both original and clipped columns
    assert "A" in transformed.columns
    assert "A__clip_iqr" in transformed.columns
    assert "B" in transformed.columns
    assert "B__clip_iqr" in transformed.columns


def test_iqr_clipper_subset(sample_dataframe):
    """Test clipping only a subset of columns."""
    clipper = IQRClipper(subset=["A"], inplace=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have clipped version of only column A
    assert "A__clip_iqr" in transformed.columns
    assert "B__clip_iqr" not in transformed.columns
    assert "B" in transformed.columns
    assert "C" in transformed.columns


def test_iqr_clipper_different_n_iqrs(sample_dataframe):
    """Test clipping with different n_iqrs values."""
    clipper_1_5 = IQRClipper(n_iqrs=1.5, inplace=False)
    clipper_1_5.fit(sample_dataframe)
    transformed_1_5 = clipper_1_5.transform(sample_dataframe)

    clipper_3_0 = IQRClipper(n_iqrs=3.0, inplace=False)
    clipper_3_0.fit(sample_dataframe)
    transformed_3_0 = clipper_3_0.transform(sample_dataframe)

    # More aggressive clipping (n_iqrs=1.5) should clip more than n_iqrs=3.0
    assert transformed_1_5["A__clip_iqr"][12] <= transformed_3_0["A__clip_iqr"][12]


def test_iqr_clipper_fit_transform():
    """Test fit_transform method."""
    X = pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 100.0],
            "B": [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
        }
    )
    clipper = IQRClipper(n_iqrs=1.5, inplace=False)
    transformed = clipper.fit_transform(X)

    assert "A__clip_iqr" in transformed.columns
    assert "B__clip_iqr" in transformed.columns
    assert transformed["A__clip_iqr"][12] < 100.0
    assert transformed["B__clip_iqr"][0] > -100.0


def test_iqr_clipper_validation():
    """Test parameter validation."""
    # n_iqrs must be positive
    with pytest.raises(Exception):  # Pydantic validation error
        IQRClipper(n_iqrs=0.0)

    with pytest.raises(Exception):  # Pydantic validation error
        IQRClipper(n_iqrs=-1.0)


def test_iqr_clipper_empty_subset():
    """Test behavior when subset is explicitly set to empty list."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": ["x", "y", "z"],
        }
    )
    clipper = IQRClipper(subset=[], inplace=False)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Empty list triggers auto-selection, so A should be clipped
    assert "A__clip_iqr" in transformed.columns
    assert "B" in transformed.columns


def test_iqr_clipper_single_column():
    """Test clipping with a single column."""
    X = pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 100.0],
        }
    )
    clipper = IQRClipper(n_iqrs=1.5, inplace=True)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Outlier should be clipped
    assert transformed["A"][12] < 100.0
    assert len(transformed.columns) == 1


def test_iqr_clipper_compute_bounds():
    """Test that IQR bounds are computed correctly."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    clipper = IQRClipper(n_iqrs=1.5, inplace=False)
    clipper.fit(X)

    # Q1=3.25, Q3=7.75, IQR=4.5
    # Bounds: [3.25 - 1.5*4.5, 7.75 + 1.5*4.5] = [-3.5, 14.5]
    lower, upper = clipper._clip_bounds["A"]
    assert lower < 1.0
    assert upper > 10.0


def test_iqr_clipper_no_outliers():
    """Test clipping on data without extreme outliers."""
    X = pl.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "B": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        }
    )
    clipper = IQRClipper(n_iqrs=1.5, inplace=False)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # With uniformly distributed data, minimal clipping should occur
    assert abs(transformed["A__clip_iqr"].sum() - X["A"].sum()) < 5.0


def test_iqr_clipper_standard_boxplot_threshold():
    """Test that n_iqrs=1.5 matches standard box plot outlier detection."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 50.0],  # 50 is outlier
        }
    )
    clipper = IQRClipper(n_iqrs=1.5, inplace=False)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Standard box plot would clip 50
    assert transformed["A__clip_iqr"][10] < 50.0
