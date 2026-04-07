from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_dt.ordinal_features import (
    COMPONENT_FUNCTIONS,
    OrdinalFeatures,
)


@pytest.fixture
def sample_ordinal_data():
    """Create sample data for ordinal feature testing."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 15, 10, 30, 45),  # Jan, Q1, Monday
                datetime(2023, 6, 20, 14, 15, 30),  # June, Q2, Tuesday
                datetime(2024, 12, 31, 23, 59, 59),  # Dec, Q4, Tuesday
            ],
            "value": [100, 200, 300],
        }
    )


def test_basic_year_month(sample_ordinal_data):
    """Test basic year and month extraction."""
    transformer = OrdinalFeatures(subset=["timestamp"], components=["year", "month"])
    result = transformer.fit_transform(sample_ordinal_data)

    assert "timestamp__year" in result.columns
    assert "timestamp__month" in result.columns
    assert result["timestamp__year"].to_list() == [2023, 2023, 2024]
    assert result["timestamp__month"].to_list() == [1, 6, 12]


def test_quarter_component(sample_ordinal_data):
    """Test quarter extraction."""
    transformer = OrdinalFeatures(subset=["timestamp"], components=["quarter"])
    result = transformer.fit_transform(sample_ordinal_data)

    assert "timestamp__quarter" in result.columns
    assert result["timestamp__quarter"].to_list() == [1, 2, 4]


def test_semester_component(sample_ordinal_data):
    """Test semester extraction (H1/H2)."""
    transformer = OrdinalFeatures(subset=["timestamp"], components=["semester"])
    result = transformer.fit_transform(sample_ordinal_data)

    assert "timestamp__semester" in result.columns
    assert result["timestamp__semester"].to_list() == [1, 1, 2]


def test_week_component(sample_ordinal_data):
    """Test week of year extraction."""
    transformer = OrdinalFeatures(subset=["timestamp"], components=["week"])
    result = transformer.fit_transform(sample_ordinal_data)

    assert "timestamp__week" in result.columns


def test_day_components():
    """Test day-related components."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),  # Monday, day 1 of month, day 1 of year
                datetime(2024, 1, 15),  # Monday, day 15 of month
                datetime(2024, 12, 31),  # Tuesday, day 31 of month, day 366 of year
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = OrdinalFeatures(
        subset=["timestamp"], components=["day_of_week", "day_of_month", "day_of_year"]
    )
    result = transformer.fit_transform(X)

    assert "timestamp__day_of_week" in result.columns
    assert "timestamp__day_of_month" in result.columns
    assert "timestamp__day_of_year" in result.columns
    assert result["timestamp__day_of_month"].to_list() == [1, 15, 31]


def test_time_components(sample_ordinal_data):
    """Test time components (hour, minute, second)."""
    transformer = OrdinalFeatures(subset=["timestamp"], components=["hour", "minute", "second"])
    result = transformer.fit_transform(sample_ordinal_data)

    assert "timestamp__hour" in result.columns
    assert "timestamp__minute" in result.columns
    assert "timestamp__second" in result.columns
    assert result["timestamp__hour"].to_list() == [10, 14, 23]
    assert result["timestamp__minute"].to_list() == [30, 15, 59]
    assert result["timestamp__second"].to_list() == [45, 30, 59]


def test_weekend_component():
    """Test weekend indicator component."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),  # Monday (1)
                datetime(2024, 1, 6),  # Saturday (6)
                datetime(2024, 1, 7),  # Sunday (7)
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = OrdinalFeatures(subset=["timestamp"], components=["weekend"])
    result = transformer.fit_transform(X)

    assert "timestamp__weekend" in result.columns
    assert result["timestamp__weekend"].to_list() == [False, True, True]


def test_leap_year_component():
    """Test leap year indicator component."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1),  # Not a leap year
                datetime(2024, 1, 1),  # Leap year
            ],
            "value": [1, 2],
        }
    )

    transformer = OrdinalFeatures(subset=["timestamp"], components=["leap_year"])
    result = transformer.fit_transform(X)

    assert "timestamp__leap_year" in result.columns
    assert result["timestamp__leap_year"].to_list() == [False, True]


def test_century_component():
    """Test century extraction."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(1999, 12, 31),
                datetime(2000, 1, 1),
                datetime(2024, 1, 1),
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = OrdinalFeatures(subset=["timestamp"], components=["century"])
    result = transformer.fit_transform(X)

    assert "timestamp__century" in result.columns


def test_all_components():
    """Test all valid ordinal components."""
    X = pl.DataFrame({"timestamp": [datetime(2024, 6, 15, 12, 30, 45)], "value": [1]})

    components = list(COMPONENT_FUNCTIONS.keys())

    transformer = OrdinalFeatures(subset=["timestamp"], components=components)
    result = transformer.fit_transform(X)

    for comp in components:
        assert f"timestamp__{comp}" in result.columns


def test_drop_columns_true(sample_ordinal_data):
    """Test dropping original columns when drop_columns=True."""
    transformer = OrdinalFeatures(
        subset=["timestamp"], components=["year", "month"], drop_columns=True
    )
    result = transformer.fit_transform(sample_ordinal_data)

    assert "timestamp" not in result.columns
    assert "timestamp__year" in result.columns


def test_drop_columns_false(sample_ordinal_data):
    """Test keeping original columns when drop_columns=False."""
    transformer = OrdinalFeatures(
        subset=["timestamp"], components=["year", "month"], drop_columns=False
    )
    result = transformer.fit_transform(sample_ordinal_data)

    assert "timestamp" in result.columns
    assert "timestamp__year" in result.columns


def test_auto_detect_datetime_columns():
    """Test automatic detection of datetime columns."""
    X = pl.DataFrame({"dt1": [datetime(2024, 1, 1)], "dt2": [datetime(2024, 6, 1)], "value": [100]})

    transformer = OrdinalFeatures(components=["month"])
    result = transformer.fit_transform(X)

    assert "dt1__month" in result.columns
    assert "dt2__month" in result.columns
    assert result["dt1__month"].to_list() == [1]
    assert result["dt2__month"].to_list() == [6]


def test_validation_invalid_component():
    """Test validation for invalid component."""
    with pytest.raises(ValueError, match="is not a valid component"):
        OrdinalFeatures(components=["invalid_component"])


def test_sklearn_compatibility(sample_ordinal_data):
    """Test sklearn pipeline compatibility."""
    transformer = OrdinalFeatures(subset=["timestamp"], components=["year", "month"])

    # Test fit returns self
    result = transformer.fit(sample_ordinal_data)
    assert result is transformer

    # Test fit_transform
    result = transformer.fit_transform(sample_ordinal_data)
    assert isinstance(result, pl.DataFrame)


def test_multiple_datetime_columns():
    """Test with multiple datetime columns."""
    X = pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1), datetime(2024, 6, 1)],
            "updated_at": [datetime(2024, 3, 1), datetime(2024, 9, 1)],
            "value": [1, 2],
        }
    )

    transformer = OrdinalFeatures(
        subset=["created_at", "updated_at"], components=["month", "quarter"]
    )
    result = transformer.fit_transform(X)

    assert "created_at__month" in result.columns
    assert "updated_at__month" in result.columns
    assert "created_at__quarter" in result.columns
    assert "updated_at__quarter" in result.columns


def test_day_of_week_values():
    """Test day of week values (0=Monday to 6=Sunday)."""
    X = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, i) for i in range(1, 8)],  # Jan 1-7, 2024
            "value": list(range(7)),
        }
    )

    transformer = OrdinalFeatures(subset=["timestamp"], components=["day_of_week"])
    result = transformer.fit_transform(X)

    # Jan 1, 2024 is Monday (0), so should be [0, 1, 2, 3, 4, 5, 6]
    assert "timestamp__day_of_week" in result.columns


def test_single_row():
    """Test with single row dataframe."""
    X = pl.DataFrame({"timestamp": [datetime(2024, 6, 15, 12, 30, 45)], "value": [1]})

    transformer = OrdinalFeatures(
        subset=["timestamp"], components=["year", "month", "day_of_month", "hour"]
    )
    result = transformer.fit_transform(X)

    assert result["timestamp__year"].to_list() == [2024]
    assert result["timestamp__month"].to_list() == [6]
    assert result["timestamp__day_of_month"].to_list() == [15]
    assert result["timestamp__hour"].to_list() == [12]


def test_year_boundaries():
    """Test year boundary dates."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 12, 31, 23, 59, 59),
                datetime(2024, 1, 1, 0, 0, 0),
            ],
            "value": [1, 2],
        }
    )

    transformer = OrdinalFeatures(subset=["timestamp"], components=["year", "month", "day_of_year"])
    result = transformer.fit_transform(X)

    assert result["timestamp__year"].to_list() == [2023, 2024]
    assert result["timestamp__month"].to_list() == [12, 1]


def test_february_leap_year():
    """Test February in leap year vs non-leap year."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 2, 28),  # Not leap year
                datetime(2024, 2, 29),  # Leap year
            ],
            "value": [1, 2],
        }
    )

    transformer = OrdinalFeatures(
        subset=["timestamp"], components=["leap_year", "month", "day_of_month"]
    )
    result = transformer.fit_transform(X)

    assert result["timestamp__leap_year"].to_list() == [False, True]
    assert result["timestamp__month"].to_list() == [2, 2]
    assert result["timestamp__day_of_month"].to_list() == [28, 29]


def test_date_type_conversion():
    """Test that string date columns are converted to Datetime before processing."""
    X = pl.DataFrame(
        {
            "date_col": ["2024-01-01", "2024-06-15"],
            "value": [1, 2],
        }
    )

    transformer = OrdinalFeatures(subset=["date_col"], components=["year", "month"])
    result = transformer.fit_transform(X)

    assert "date_col__year" in result.columns
    assert "date_col__month" in result.columns
    assert result["date_col__year"].to_list() == [2024, 2024]
    assert result["date_col__month"].to_list() == [1, 6]
