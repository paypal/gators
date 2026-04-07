import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.clippers.quantile_clipper import QuantileClipper


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with outliers for testing."""
    return pl.DataFrame(
        {
            "A": [float(i) for i in range(1, 101)] + [1000.0],  # 1-100 + 1000 outlier
            "B": [-1000.0] + [float(i) for i in range(1, 101)],  # -1000 outlier + 1-100
            "C": [float(i * 10) for i in range(1, 102)],  # No outliers
            "D": ["x", "y", "z"] * 33 + ["x", "y"],  # 101 categorical values
        }
    )


def test_quantile_clipper_default_inplace(sample_dataframe):
    """Test clipping with default parameters (inplace=True)."""
    clipper = QuantileClipper(inplace=True)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Verify the shape
    assert transformed.shape == (101, 4)
    assert "D" in transformed.columns

    # Verify outliers are clipped
    assert transformed["A"][100] < 1000.0
    assert transformed["B"][0] > -1000.0


def test_quantile_clipper_not_inplace(sample_dataframe):
    """Test clipping with inplace=False."""
    clipper = QuantileClipper(lower_quantile=0.1, upper_quantile=0.9, inplace=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have new columns instead of original numeric columns
    assert "A__clip_quantile" in transformed.columns
    assert "B__clip_quantile" in transformed.columns
    assert "C__clip_quantile" in transformed.columns
    assert "A" not in transformed.columns
    assert "D" in transformed.columns


def test_quantile_clipper_not_inplace_no_drop(sample_dataframe):
    """Test clipping with inplace=False and drop_columns=False."""
    clipper = QuantileClipper(inplace=False, drop_columns=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have both original and clipped columns
    assert "A" in transformed.columns
    assert "A__clip_quantile" in transformed.columns
    assert "B" in transformed.columns
    assert "B__clip_quantile" in transformed.columns


def test_quantile_clipper_subset(sample_dataframe):
    """Test clipping only a subset of columns."""
    clipper = QuantileClipper(subset=["A"], inplace=False)
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have clipped version of only column A
    assert "A__clip_quantile" in transformed.columns
    assert "B__clip_quantile" not in transformed.columns
    assert "B" in transformed.columns
    assert "C" in transformed.columns


def test_quantile_clipper_different_thresholds(sample_dataframe):
    """Test clipping with different quantile thresholds."""
    clipper_1_99 = QuantileClipper(lower_quantile=0.01, upper_quantile=0.99, inplace=False)
    clipper_1_99.fit(sample_dataframe)
    transformed_1_99 = clipper_1_99.transform(sample_dataframe)

    clipper_05_95 = QuantileClipper(lower_quantile=0.05, upper_quantile=0.95, inplace=False)
    clipper_05_95.fit(sample_dataframe)
    transformed_05_95 = clipper_05_95.transform(sample_dataframe)

    # More aggressive clipping (0.05, 0.95) should clip more
    assert transformed_05_95["A__clip_quantile"][10] <= transformed_1_99["A__clip_quantile"][10]


def test_quantile_clipper_fit_transform():
    """Test fit_transform method."""
    X = pl.DataFrame(
        {
            "A": [float(i) for i in range(1, 101)] + [1000.0],
            "B": [-1000.0] + [float(i) for i in range(1, 101)],
        }
    )
    clipper = QuantileClipper(inplace=False)
    transformed = clipper.fit_transform(X)

    assert "A__clip_quantile" in transformed.columns
    assert "B__clip_quantile" in transformed.columns
    assert transformed["A__clip_quantile"][100] < 1000.0


def test_quantile_clipper_validation():
    """Test parameter validation."""
    # lower_quantile must be less than upper_quantile
    clipper = QuantileClipper(lower_quantile=0.9, upper_quantile=0.1)
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="lower_quantile.*must be less than.*upper_quantile"):
        clipper.fit(X)

    # Quantiles must be between 0 and 1
    with pytest.raises(Exception):  # Pydantic validation error
        QuantileClipper(lower_quantile=-0.1)

    with pytest.raises(Exception):  # Pydantic validation error
        QuantileClipper(upper_quantile=1.1)


def test_quantile_clipper_equal_quantiles():
    """Test error when lower_quantile equals upper_quantile."""
    clipper = QuantileClipper(lower_quantile=0.5, upper_quantile=0.5)
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        clipper.fit(X)


def test_quantile_clipper_empty_subset():
    """Test behavior when subset is explicitly set to empty list."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": ["x", "y", "z"],
        }
    )
    clipper = QuantileClipper(subset=[], inplace=False)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Empty list triggers auto-selection, so A should be clipped
    assert "A__clip_quantile" in transformed.columns
    assert "B" in transformed.columns


def test_quantile_clipper_single_column():
    """Test clipping with a single column."""
    X = pl.DataFrame(
        {
            "A": [float(i) for i in range(1, 101)] + [1000.0],
        }
    )
    clipper = QuantileClipper(inplace=True)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Outlier should be clipped
    assert transformed["A"][100] < 1000.0
    assert len(transformed.columns) == 1


def test_quantile_clipper_no_outliers():
    """Test clipping on data without extreme outliers."""
    X = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        }
    )
    clipper = QuantileClipper(inplace=False)
    clipper.fit(X)
    transformed = clipper.transform(X)

    # With uniformly distributed data, clipping should be minimal
    # at least the values should be very close
    assert abs(transformed["A__clip_quantile"].sum() - X["A"].sum()) < 1.0
