import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import CorrelationFilter


def test_correlation_filter_default():

    X = {"A": [1, 2, 3], "B": [1, 2, 3], "C": [4.5, 6.5, 5.5], "D": [7.8, 8.9, 9.1]}
    X = pl.DataFrame(X)

    filter = CorrelationFilter(max_corr=1.0)
    filter.fit(X)

    transformed_X = filter.transform(X)

    expected_X = {"A": [1, 2, 3], "C": [4.5, 6.5, 5.5], "D": [7.8, 8.9, 9.1]}
    expected_X = pl.DataFrame(expected_X)

    assert_frame_equal(transformed_X, expected_X)


def test_correlation_filter_subset_columns():

    X = {"A": [1, 2, 3], "B": [1, 2, 3], "C": [4.5, 6.5, 5.5], "D": [7.8, 8.9, 9.1]}
    X = pl.DataFrame(X)

    filter = CorrelationFilter(subset=["A", "B"], max_corr=1.0)
    filter.fit(X)

    transformed_X = filter.transform(X)

    expected_X = {"A": [1, 2, 3], "C": [4.5, 6.5, 5.5], "D": [7.8, 8.9, 9.1]}
    expected_X = pl.DataFrame(expected_X)

    assert_frame_equal(transformed_X, expected_X)


def test_correlation_filter_no_correlations():
    """Test when no columns exceed correlation threshold."""
    # Use data with low correlation - avoid perfectly correlated columns
    X = {"A": [1, 5, 3, 7, 2], "B": [2, 3, 5, 1, 4], "C": [7, 2, 4, 3, 6]}
    X = pl.DataFrame(X)

    # Set threshold high - only perfect correlation should be filtered
    filter = CorrelationFilter(max_corr=0.99)
    filter.fit(X)

    transformed_X = filter.transform(X)

    # All columns should remain  since correlations are low
    assert transformed_X.shape[1] == 3
    assert set(transformed_X.columns) == {"A", "B", "C"}


def test_fit_all_constant_columns_skips_filtering():
    """When all subset columns are constant (std=0), _to_drop is set to [] and fit returns early."""
    X = pl.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0, 2.0],
        }
    )
    f = CorrelationFilter(max_corr=0.9)
    f.fit(X)
    # No columns should be dropped since there are no correlations to compute
    assert f._to_drop == []
    assert_frame_equal(f.transform(X), X)


if __name__ == "__main__":
    pytest.main()
