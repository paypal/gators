from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_dt.business_time_features import BusinessTimeFeatures


@pytest.fixture
def sample_business_data():
    """Create sample data with various business time scenarios."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15, 8, 0),  # Monday (1), before business hours
                datetime(2024, 1, 15, 10, 30),  # Monday (1), during business hours
                datetime(2024, 1, 15, 18, 0),  # Monday (1), after business hours
                datetime(2024, 1, 20, 10, 0),  # Saturday (6), weekend
                datetime(2024, 1, 22, 14, 0),  # Monday (1), during hours
            ],
            "value": [100, 200, 300, 400, 500],
        }
    )


def test_basic_is_business_hour(sample_business_data):
    """Test basic business hour detection."""
    transformer = BusinessTimeFeatures(
        subset=["timestamp"], features=["is_business_hour"]
    )
    result = transformer.fit_transform(sample_business_data)

    assert "timestamp__is_business_hour" in result.columns
    assert result["timestamp__is_business_hour"].to_list() == [
        False,
        True,
        False,
        True,
        True,
    ]


def test_basic_is_business_day(sample_business_data):
    """Test basic business day detection."""
    transformer = BusinessTimeFeatures(
        subset=["timestamp"], features=["is_business_day"]
    )
    result = transformer.fit_transform(sample_business_data)

    assert "timestamp__is_business_day" in result.columns
    assert result["timestamp__is_business_day"].to_list() == [
        True,
        True,
        True,
        False,
        True,
    ]


def test_time_of_business_day(sample_business_data):
    """Test time of business day categorization."""
    transformer = BusinessTimeFeatures(
        subset=["timestamp"], features=["time_of_business_day"]
    )
    result = transformer.fit_transform(sample_business_data)

    assert "timestamp__time_of_business_day" in result.columns
    expected = [
        "before_hours",
        "during_hours",
        "after_hours",
        "weekend",
        "during_hours",
    ]
    assert result["timestamp__time_of_business_day"].to_list() == expected


def test_hour_of_business_day(sample_business_data):
    """Test hour of business day calculation."""
    transformer = BusinessTimeFeatures(
        subset=["timestamp"], features=["hour_of_business_day"]
    )
    result = transformer.fit_transform(sample_business_data)

    assert "timestamp__hour_of_business_day" in result.columns
    # 8am: None (before hours), 10:30am: 1 (10-9), 6pm: None (after), Saturday: 1 (10-9), Monday 2pm: 5 (14-9)
    expected = [None, 1, None, 1, 5]
    assert result["timestamp__hour_of_business_day"].to_list() == expected


def test_all_features_together(sample_business_data):
    """Test all features generated together."""
    transformer = BusinessTimeFeatures(
        subset=["timestamp"],
        features=[
            "is_business_hour",
            "is_business_day",
            "time_of_business_day",
            "hour_of_business_day",
        ],
    )
    result = transformer.fit_transform(sample_business_data)

    assert "timestamp__is_business_hour" in result.columns
    assert "timestamp__is_business_day" in result.columns
    assert "timestamp__time_of_business_day" in result.columns
    assert "timestamp__hour_of_business_day" in result.columns


def test_custom_business_hours():
    """Test with custom business hours."""
    X =pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15, 6, 0),  # Before custom hours
                datetime(2024, 1, 15, 8, 0),  # During custom hours
                datetime(2024, 1, 15, 16, 0),  # After custom hours
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = BusinessTimeFeatures(
        subset=["timestamp"],
        business_hours_start=7,
        business_hours_end=15,
        features=["is_business_hour"],
    )
    result = transformer.fit_transform(X)

    assert result["timestamp__is_business_hour"].to_list() == [False, True, False]


def test_custom_weekend_days():
    """Test with custom weekend days (e.g., Friday-Saturday)."""
    X =pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15, 10, 0),  # Monday (1)
                datetime(2024, 1, 19, 10, 0),  # Friday (5) (custom weekend)
                datetime(2024, 1, 20, 10, 0),  # Saturday (6) (custom weekend)
                datetime(2024, 1, 21, 10, 0),  # Sunday (7) (regular day)
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = BusinessTimeFeatures(
        subset=["timestamp"],
        weekend_days=[5, 6],  # Friday and Saturday (polars weekday: 1=Mon, 7=Sun)
        features=["is_business_day"],
    )
    result = transformer.fit_transform(X)

    assert result["timestamp__is_business_day"].to_list() == [True, False, False, True]


def test_drop_columns_true(sample_business_data):
    """Test that original columns are dropped when drop_columns=True."""
    transformer = BusinessTimeFeatures(
        subset=["timestamp"], features=["is_business_hour"], drop_columns=True
    )
    result = transformer.fit_transform(sample_business_data)

    assert "timestamp" not in result.columns
    assert "timestamp__is_business_hour" in result.columns


def test_drop_columns_false(sample_business_data):
    """Test that original columns are kept when drop_columns=False."""
    transformer = BusinessTimeFeatures(
        subset=["timestamp"], features=["is_business_hour"], drop_columns=False
    )
    result = transformer.fit_transform(sample_business_data)

    assert "timestamp" in result.columns
    assert "timestamp__is_business_hour" in result.columns


def test_auto_detect_datetime_columns():
    """Test automatic detection of datetime columns."""
    X =pl.DataFrame(
        {
            "timestamp1": [datetime(2024, 1, 15, 10, 0)],
            "timestamp2": [datetime(2024, 1, 15, 18, 0)],
            "value": [100],
        }
    )

    transformer = BusinessTimeFeatures(features=["is_business_hour"])
    result = transformer.fit_transform(X)

    assert "timestamp1__is_business_hour" in result.columns
    assert "timestamp2__is_business_hour" in result.columns


def test_validation_invalid_business_hours_start():
    """Test validation for invalid business_hours_start."""
    with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
        BusinessTimeFeatures(business_hours_start=25)


def test_validation_invalid_business_hours_end():
    """Test validation for invalid business_hours_end."""
    with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
        BusinessTimeFeatures(business_hours_end=-1)


def test_validation_invalid_weekend_day():
    """Test validation for invalid weekend day."""
    with pytest.raises(ValueError, match="Weekend day must be between 0 and 6"):
        BusinessTimeFeatures(weekend_days=[0, 7])


def test_validation_invalid_feature():
    """Test validation for invalid feature name."""
    with pytest.raises(ValueError, match="is not supported"):
        BusinessTimeFeatures(features=["invalid_feature"])


def test_sklearn_compatibility(sample_business_data):
    """Test sklearn pipeline compatibility."""
    transformer = BusinessTimeFeatures(
        subset=["timestamp"], features=["is_business_hour"]
    )

    # Test fit returns self
    result = transformer.fit(sample_business_data)
    assert result is transformer

    # Test fit_transform
    result = transformer.fit_transform(sample_business_data)
    assert isinstance(result, pl.DataFrame)


def test_multiple_datetime_columns():
    """Test with multiple datetime columns."""
    X =pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 15, 10, 0), datetime(2024, 1, 15, 18, 0)],
            "updated_at": [datetime(2024, 1, 15, 11, 0), datetime(2024, 1, 15, 19, 0)],
            "value": [1, 2],
        }
    )

    transformer = BusinessTimeFeatures(
        subset=["created_at", "updated_at"], features=["is_business_hour"]
    )
    result = transformer.fit_transform(X)

    assert "created_at__is_business_hour" in result.columns
    assert "updated_at__is_business_hour" in result.columns
    assert result["created_at__is_business_hour"].to_list() == [True, False]
    assert result["updated_at__is_business_hour"].to_list() == [True, False]


def test_edge_case_exact_business_hour_boundaries():
    """Test exact boundary conditions for business hours."""
    X =pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15, 9, 0),  # Exactly start time
                datetime(2024, 1, 15, 17, 0),  # Exactly end time
                datetime(2024, 1, 15, 16, 59),  # Just before end
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = BusinessTimeFeatures(
        subset=["timestamp"], features=["is_business_hour"]
    )
    result = transformer.fit_transform(X)

    # 9:00 is in, 17:00 is out (< not <=), 16:59 is in
    assert result["timestamp__is_business_hour"].to_list() == [True, False, True]
