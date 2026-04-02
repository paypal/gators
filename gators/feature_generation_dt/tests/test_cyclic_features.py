from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_dt.cyclic_features import CyclicFeatures


@pytest.fixture
def sample_datetime_data():
    """Create sample data for cyclic feature testing."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, 0),  # January, Q1
                datetime(2024, 4, 1, 6, 30, 15),  # April, Q2
                datetime(2024, 7, 1, 12, 45, 30),  # July, Q3
                datetime(2024, 10, 1, 18, 15, 45),  # October, Q4
            ],
            "value": [100, 200, 300, 400],
        }
    )


def test_basic_month_cyclic(sample_datetime_data):
    """Test basic month cyclic feature with single angle."""
    transformer = CyclicFeatures(
        subset=["timestamp"], components=["month"], angles=[0], drop_columns=True
    )
    result = transformer.fit_transform(sample_datetime_data)

    assert "timestamp__month__sin0" in result.columns
    assert "timestamp" not in result.columns  # Default drop_columns=True


def test_multiple_angles(sample_datetime_data):
    """Test cyclic features with multiple phase angles."""
    transformer = CyclicFeatures(
        subset=["timestamp"], components=["month"], angles=[0, 90, 180]
    )
    result = transformer.fit_transform(sample_datetime_data)

    assert "timestamp__month__sin0" in result.columns
    assert "timestamp__month__sin90" in result.columns
    assert "timestamp__month__sin180" in result.columns


def test_multiple_components(sample_datetime_data):
    """Test with multiple cyclic components."""
    transformer = CyclicFeatures(
        subset=["timestamp"], components=["month", "quarter", "hour"], angles=[0]
    )
    result = transformer.fit_transform(sample_datetime_data)

    assert "timestamp__month__sin0" in result.columns
    assert "timestamp__quarter__sin0" in result.columns
    assert "timestamp__hour__sin0" in result.columns


def test_day_of_month_special_handling(sample_datetime_data):
    """Test day_of_month which requires days_in_month calculation."""
    transformer = CyclicFeatures(
        subset=["timestamp"], components=["day_of_month"], angles=[0]
    )
    result = transformer.fit_transform(sample_datetime_data)

    assert "timestamp__day_of_month__sin0" in result.columns
    # Should not have temporary days_in_month column
    assert "timestamp__days_in_month" not in result.columns


def test_all_cyclic_components():
    """Test all valid cyclic components."""
    X =pl.DataFrame(
        {"timestamp": [datetime(2024, 6, 15, 12, 30, 45)], "value": [1]}
    )

    components = [
        "month",
        "quarter",
        "semester",
        "week",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "hour",
        "minute",
        "second",
    ]

    transformer = CyclicFeatures(
        subset=["timestamp"], components=components, angles=[0]
    )
    result = transformer.fit_transform(X)

    for comp in components:
        assert f"timestamp__{comp}__sin0" in result.columns


def test_drop_columns_false(sample_datetime_data):
    """Test keeping original columns when drop_columns=False."""
    transformer = CyclicFeatures(
        subset=["timestamp"], components=["month"], angles=[0], drop_columns=False
    )
    result = transformer.fit_transform(sample_datetime_data)

    assert "timestamp" in result.columns
    assert "timestamp__month__sin0" in result.columns


def test_auto_detect_datetime_columns():
    """Test automatic detection of datetime columns."""
    X =pl.DataFrame(
        {"dt1": [datetime(2024, 1, 1)], "dt2": [datetime(2024, 6, 1)], "value": [100]}
    )

    transformer = CyclicFeatures(components=["month"], angles=[0])
    result = transformer.fit_transform(X)

    assert "dt1__month__sin0" in result.columns
    assert "dt2__month__sin0" in result.columns


def test_fractional_angle():
    """Test with fractional angle values."""
    X =pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [1]})

    transformer = CyclicFeatures(
        subset=["timestamp"], components=["month"], angles=[45.5, 90.25]
    )
    result = transformer.fit_transform(X)

    assert "timestamp__month__sin45.5" in result.columns
    assert "timestamp__month__sin90.25" in result.columns


def test_validation_invalid_component():
    """Test validation for invalid component."""
    with pytest.raises(ValueError, match="is not a valid cyclic component"):
        CyclicFeatures(components=["invalid_component"], angles=[0])


def test_sklearn_compatibility(sample_datetime_data):
    """Test sklearn pipeline compatibility."""
    transformer = CyclicFeatures(
        subset=["timestamp"], components=["month"], angles=[0]
    )

    # Test fit returns self
    result = transformer.fit(sample_datetime_data)
    assert result is transformer

    # Test fit_transform
    result = transformer.fit_transform(sample_datetime_data)
    assert isinstance(result, pl.DataFrame)


def test_semester_cyclic():
    """Test semester component (H1/H2)."""
    X =pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),  # Q1 -> Semester 1
                datetime(2024, 4, 1),  # Q2 -> Semester 1
                datetime(2024, 7, 1),  # Q3 -> Semester 2
                datetime(2024, 10, 1),  # Q4 -> Semester 2
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = CyclicFeatures(
        subset=["timestamp"], components=["semester"], angles=[0]
    )
    result = transformer.fit_transform(X)

    assert "timestamp__semester__sin0" in result.columns


def test_week_of_year_cyclic(sample_datetime_data):
    """Test week of year cyclic features."""
    transformer = CyclicFeatures(
        subset=["timestamp"], components=["week"], angles=[0]
    )
    result = transformer.fit_transform(sample_datetime_data)

    assert "timestamp__week__sin0" in result.columns


def test_time_components_cyclic():
    """Test time-based cyclic features (hour, minute, second)."""
    X =pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 6, 15, 30),
                datetime(2024, 1, 1, 12, 30, 45),
                datetime(2024, 1, 1, 18, 45, 59),
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = CyclicFeatures(
        subset=["timestamp"], components=["hour", "minute", "second"], angles=[0]
    )
    result = transformer.fit_transform(X)

    assert "timestamp__hour__sin0" in result.columns
    assert "timestamp__minute__sin0" in result.columns
    assert "timestamp__second__sin0" in result.columns


def test_multiple_datetime_columns():
    """Test with multiple datetime columns."""
    X =pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1), datetime(2024, 6, 1)],
            "updated_at": [datetime(2024, 3, 1), datetime(2024, 9, 1)],
            "value": [1, 2],
        }
    )

    transformer = CyclicFeatures(
        subset=["created_at", "updated_at"], components=["month"], angles=[0]
    )
    result = transformer.fit_transform(X)

    assert "created_at__month__sin0" in result.columns
    assert "updated_at__month__sin0" in result.columns


def test_cyclic_values_are_bounded():
    """Test that cyclic sine values are between -1 and 1."""
    X =pl.DataFrame(
        {
            "timestamp": [datetime(2024, i, 1) for i in range(1, 13)],
            "value": list(range(12)),
        }
    )

    transformer = CyclicFeatures(
        subset=["timestamp"], components=["month"], angles=[0]
    )
    result = transformer.fit_transform(X)

    sin_values = result["timestamp__month__sin0"].to_list()
    assert all(-1 <= val <= 1 for val in sin_values)


def test_day_of_week_cyclic():
    """Test day of week cyclic features."""
    X =pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, i) for i in range(1, 8)],  # Mon-Sun
            "value": list(range(7)),
        }
    )

    transformer = CyclicFeatures(
        subset=["timestamp"], components=["day_of_week"], angles=[0]
    )
    result = transformer.fit_transform(X)

    assert "timestamp__day_of_week__sin0" in result.columns


def test_day_of_year_cyclic():
    """Test day of year cyclic features."""
    X =pl.DataFrame(
        {"timestamp": [datetime(2024, 1, 1), datetime(2024, 12, 31)], "value": [1, 2]}
    )

    transformer = CyclicFeatures(
        subset=["timestamp"], components=["day_of_year"], angles=[0]
    )
    result = transformer.fit_transform(X)

    assert "timestamp__day_of_year__sin0" in result.columns


def test_quarter_cyclic(sample_datetime_data):
    """Test quarter cyclic features."""
    transformer = CyclicFeatures(
        subset=["timestamp"], components=["quarter"], angles=[0, 90]
    )
    result = transformer.fit_transform(sample_datetime_data)

    assert "timestamp__quarter__sin0" in result.columns
    assert "timestamp__quarter__sin90" in result.columns


def test_string_datetime_conversion():
    """Test that string datetime columns are converted to Datetime during transform."""
    X =pl.DataFrame(
        {
            "timestamp_str": [
                "2024-01-15 10:30:00",
                "2024-06-20 14:45:00",
                "2024-12-25 20:15:00",
            ],
            "value": [1, 2, 3],
        }
    )
    
    transformer = CyclicFeatures(
        subset=["timestamp_str"], 
        components=["month", "hour"], 
        angles=[0]
    )
    result = transformer.fit_transform(X)
    
    # Check that cyclic features were created from string datetime
    assert "timestamp_str__month__sin0" in result.columns
    assert "timestamp_str__hour__sin0" in result.columns
    # Verify the result has correct number of rows
    assert result.height == 3
