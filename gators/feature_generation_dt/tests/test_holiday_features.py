from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_dt.holiday_features import HolidayFeatures


@pytest.fixture
def sample_holiday_data():
    """Create sample data with various dates including holidays."""
    return pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),  # New Year's Day
                datetime(2024, 7, 4),  # Independence Day
                datetime(2024, 12, 25),  # Christmas
                datetime(2024, 6, 15),  # Regular day
                datetime(2024, 3, 10),  # Regular day
            ],
            "value": [100, 200, 300, 400, 500],
        }
    )


def test_basic_is_holiday(sample_holiday_data):
    """Test basic holiday detection."""
    transformer = HolidayFeatures(subset=["date"], features=["is_holiday"])
    result = transformer.fit_transform(sample_holiday_data)

    assert "date__is_holiday" in result.columns
    # Jan 1, July 4, and Dec 25 should be holidays
    holiday_flags = result["date__is_holiday"].to_list()
    assert holiday_flags[0] == True  # New Year
    assert holiday_flags[1] == True  # Independence Day
    assert holiday_flags[2] == True  # Christmas


def test_us_federal_holidays():
    """Test US federal holidays detection."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),  # New Year's Day
                datetime(2024, 7, 4),  # Independence Day
                datetime(2024, 11, 11),  # Veterans Day
                datetime(2024, 12, 25),  # Christmas
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], country="US", features=["is_holiday"]
    )
    result = transformer.fit_transform(X)

    assert all(result["date__is_holiday"].to_list())


def test_uk_holidays():
    """Test UK holidays detection."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),  # New Year's Day
                datetime(2024, 12, 25),  # Christmas Day
                datetime(2024, 12, 26),  # Boxing Day
                datetime(2024, 6, 1),  # Regular day
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], country="UK", features=["is_holiday"]
    )
    result = transformer.fit_transform(X)

    holiday_flags = result["date__is_holiday"].to_list()
    assert holiday_flags[0] == True  # New Year's Day
    assert holiday_flags[1] == True  # Christmas Day
    assert holiday_flags[2] == True  # Boxing Day
    assert holiday_flags[3] == False  # Regular day


def test_canadian_holidays():
    """Test Canadian holidays detection."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),  # New Year's Day
                datetime(2024, 7, 1),  # Canada Day
                datetime(2024, 12, 25),  # Christmas Day
                datetime(2024, 6, 15),  # Regular day
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], country="CA", features=["is_holiday"]
    )
    result = transformer.fit_transform(X)

    holiday_flags = result["date__is_holiday"].to_list()
    assert holiday_flags[0] == True  # New Year's Day
    assert holiday_flags[1] == True  # Canada Day
    assert holiday_flags[2] == True  # Christmas Day
    assert holiday_flags[3] == False  # Regular day


def test_non_holiday_dates():
    """Test that regular dates are not flagged as holidays."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 3, 15),
                datetime(2024, 6, 10),
                datetime(2024, 8, 20),
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = HolidayFeatures(subset=["date"], features=["is_holiday"])
    result = transformer.fit_transform(X)

    assert not any(result["date__is_holiday"].to_list())


def test_drop_columns_false(sample_holiday_data):
    """Test keeping original columns when drop_columns=False."""
    transformer = HolidayFeatures(
        subset=["date"], features=["is_holiday"], drop_columns=False
    )
    result = transformer.fit_transform(sample_holiday_data)

    assert "date" in result.columns
    assert "date__is_holiday" in result.columns


def test_drop_columns_true(sample_holiday_data):
    """Test dropping original columns when drop_columns=True."""
    transformer = HolidayFeatures(
        subset=["date"], features=["is_holiday"], drop_columns=True
    )
    result = transformer.fit_transform(sample_holiday_data)

    assert "date" not in result.columns
    assert "date__is_holiday" in result.columns


def test_auto_detect_datetime_columns():
    """Test automatic detection of datetime columns."""
    X =pl.DataFrame(
        {"dt1": [datetime(2024, 1, 1)], "dt2": [datetime(2024, 7, 4)], "value": [100]}
    )

    transformer = HolidayFeatures(features=["is_holiday"])
    result = transformer.fit_transform(X)

    assert "dt1__is_holiday" in result.columns
    assert "dt2__is_holiday" in result.columns


def test_validation_invalid_feature():
    """Test validation for invalid feature name."""
    with pytest.raises(ValueError, match="is not supported"):
        HolidayFeatures(features=["invalid_feature"])


def test_sklearn_compatibility(sample_holiday_data):
    """Test sklearn pipeline compatibility."""
    transformer = HolidayFeatures(subset=["date"], features=["is_holiday"])

    # Test fit returns self
    result = transformer.fit(sample_holiday_data)
    assert result is transformer

    # Test fit_transform
    result = transformer.fit_transform(sample_holiday_data)
    assert isinstance(result, pl.DataFrame)


def test_multiple_datetime_columns():
    """Test with multiple datetime columns."""
    X =pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1), datetime(2024, 6, 15)],
            "updated_at": [datetime(2024, 7, 4), datetime(2024, 3, 10)],
            "value": [1, 2],
        }
    )

    transformer = HolidayFeatures(
        subset=["created_at", "updated_at"], features=["is_holiday"]
    )
    result = transformer.fit_transform(X)

    assert "created_at__is_holiday" in result.columns
    assert "updated_at__is_holiday" in result.columns


def test_mlk_day():
    """Test MLK Day detection (3rd Monday of January - Jan 15, 2024)."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 15),  # MLK Day 2024
                datetime(2024, 1, 16),  # Not a holiday
                datetime(2024, 1, 20),  # Not a holiday
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], country="US", features=["is_holiday"]
    )
    result = transformer.fit_transform(X)

    holiday_flags = result["date__is_holiday"].to_list()
    assert holiday_flags[0] == True  # MLK Day
    assert holiday_flags[1] == False  # Not a holiday
    assert holiday_flags[2] == False  # Not a holiday


def test_thanksgiving():
    """Test Thanksgiving detection (4th Thursday of November - Nov 28, 2024)."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 11, 28),  # Thanksgiving 2024
                datetime(2024, 11, 27),  # Not a holiday
                datetime(2024, 11, 29),  # Not a holiday
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], country="US", features=["is_holiday"]
    )
    result = transformer.fit_transform(X)

    holiday_flags = result["date__is_holiday"].to_list()
    assert holiday_flags[0] == True  # Thanksgiving
    assert holiday_flags[1] == False  # Not a holiday
    assert holiday_flags[2] == False  # Not a holiday


def test_nearest_holiday_distance():
    """Test nearest holiday distance feature."""
    X =pl.DataFrame(
        {"date": [datetime(2024, 1, 1), datetime(2024, 6, 15)], "value": [1, 2]}
    )

    transformer = HolidayFeatures(
        subset=["date"], features=["nearest_holiday_distance"]
    )
    result = transformer.fit_transform(X)

    assert "date__nearest_holiday_distance" in result.columns


def test_multiple_years():
    """Test holiday detection across multiple years."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2023, 1, 1),  # New Year 2023
                datetime(2024, 1, 1),  # New Year 2024
                datetime(2025, 1, 1),  # New Year 2025
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], country="US", features=["is_holiday"]
    )
    result = transformer.fit_transform(X)

    # All should be New Year's Day
    assert all(result["date__is_holiday"].to_list())


def test_invalid_country_code():
    """Test error handling for invalid country code."""
    X =pl.DataFrame({"date": [datetime(2024, 1, 1)], "value": [1]})

    transformer = HolidayFeatures(
        subset=["date"], country="INVALID", features=["is_holiday"]
    )

    with pytest.raises(ValueError, match="is not supported by the holidays library"):
        transformer.fit_transform(X)


def test_custom_years_parameter():
    """Test explicit years parameter."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),  # New Year
                datetime(2024, 7, 4),  # Independence Day
            ],
            "value": [1, 2],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], country="US", years=[2024], features=["is_holiday"]
    )
    result = transformer.fit_transform(X)

    assert all(result["date__is_holiday"].to_list())


def test_single_row():
    """Test with single row dataframe."""
    X =pl.DataFrame({"date": [datetime(2024, 12, 25)], "value": [1]})

    transformer = HolidayFeatures(subset=["date"], features=["is_holiday"])
    result = transformer.fit_transform(X)

    assert result["date__is_holiday"].to_list() == [True]


def test_days_to_holiday():
    """Test days_to_holiday feature calculation."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 6, 30),  # 4 days before July 4th
                datetime(2024, 7, 3),  # 1 day before July 4th
                datetime(2024, 7, 4),  # On July 4th
                datetime(2024, 7, 5),  # After July 4th
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = HolidayFeatures(subset=["date"], features=["days_to_holiday"])
    result = transformer.fit_transform(X)

    assert "date__days_to_holiday" in result.columns
    days_to = result["date__days_to_holiday"].to_list()
    
    # June 30 should be 4 days to July 4
    assert days_to[0] == 4
    # July 3 should be 1 day to July 4
    assert days_to[1] == 1
    # July 4 should be 0 (on the holiday)
    assert days_to[2] == 0


def test_days_from_holiday():
    """Test days_from_holiday feature calculation."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 7, 3),  # Before July 4th
                datetime(2024, 7, 4),  # On July 4th
                datetime(2024, 7, 5),  # 1 day after July 4th
                datetime(2024, 7, 10),  # 6 days after July 4th
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = HolidayFeatures(subset=["date"], features=["days_from_holiday"])
    result = transformer.fit_transform(X)

    assert "date__days_from_holiday" in result.columns
    days_from = result["date__days_from_holiday"].to_list()
    
    # July 4 should be 0 (on the holiday)
    assert days_from[1] == 0
    # July 5 should be 1 day from July 4
    assert days_from[2] == 1
    # July 10 should be 6 days from July 4
    assert days_from[3] == 6


def test_all_distance_features_combined():
    """Test all distance features together."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 7, 4),  # July 4th (holiday)
                datetime(2024, 7, 10),  # Between holidays
            ],
            "value": [1, 2],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"],
        features=["days_to_holiday", "days_from_holiday", "nearest_holiday_distance"],
    )
    result = transformer.fit_transform(X)

    assert "date__days_to_holiday" in result.columns
    assert "date__days_from_holiday" in result.columns
    assert "date__nearest_holiday_distance" in result.columns


def test_all_features_combined():
    """Test all available features together."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 1),  # New Year's Day
                datetime(2024, 6, 15),  # Regular day
            ],
            "value": [1, 2],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"],
        features=[
            "is_holiday",
            "days_to_holiday",
            "days_from_holiday",
            "nearest_holiday_distance",
        ],
    )
    result = transformer.fit_transform(X)

    assert "date__is_holiday" in result.columns
    assert "date__days_to_holiday" in result.columns
    assert "date__days_from_holiday" in result.columns
    assert "date__nearest_holiday_distance" in result.columns


def test_days_to_holiday_before_all_holidays():
    """Test days_to_holiday when date is before first holiday of year."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 12, 20),  # Before Christmas
                datetime(2024, 12, 24),  # Day before Christmas
                datetime(2024, 12, 25),  # Christmas
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], features=["days_to_holiday"], years=[2024]
    )
    result = transformer.fit_transform(X)

    days_to = result["date__days_to_holiday"].to_list()
    assert days_to[0] == 5  # 5 days to Christmas
    assert days_to[1] == 1  # 1 day to Christmas
    assert days_to[2] == 0  # On Christmas


def test_days_from_holiday_after_all_holidays():
    """Test days_from_holiday when date is after last holiday of year."""
    X =pl.DataFrame(
        {
            "date": [
                datetime(2024, 12, 25),  # Christmas
                datetime(2024, 12, 26),  # Day after Christmas
                datetime(2024, 12, 30),  # 5 days after Christmas
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = HolidayFeatures(
        subset=["date"], features=["days_from_holiday"], years=[2024]
    )
    result = transformer.fit_transform(X)

    days_from = result["date__days_from_holiday"].to_list()
    assert days_from[0] == 0  # On Christmas
    assert days_from[1] == 1  # 1 day after Christmas
    assert days_from[2] == 5  # 5 days after Christmas
