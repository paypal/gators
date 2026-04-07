from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_dt.time_bin_features import TimeBinFeatures


@pytest.fixture
def sample_timebin_data():
    """Create sample data for time bin testing."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 5, 3, 0),  # Night, winter, beginning of month
                datetime(2024, 1, 15, 8, 0),  # Morning, winter, middle of month
                datetime(2024, 4, 25, 14, 0),  # Afternoon, spring, end of month
                datetime(2024, 7, 10, 18, 30),  # Evening, summer, beginning of month
                datetime(2024, 10, 20, 21, 0),  # Evening, fall, end of month
            ],
            "value": [100, 200, 300, 400, 500],
        }
    )


def test_part_of_day_night(sample_timebin_data):
    """Test part of day binning - night."""
    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["part_of_day"])
    result = transformer.fit_transform(sample_timebin_data)

    assert "timestamp__part_of_day" in result.columns
    assert result["timestamp__part_of_day"].to_list()[0] == "night"


def test_part_of_day_morning(sample_timebin_data):
    """Test part of day binning - morning."""
    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["part_of_day"])
    result = transformer.fit_transform(sample_timebin_data)

    assert result["timestamp__part_of_day"].to_list()[1] == "morning"


def test_part_of_day_afternoon(sample_timebin_data):
    """Test part of day binning - afternoon."""
    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["part_of_day"])
    result = transformer.fit_transform(sample_timebin_data)

    assert result["timestamp__part_of_day"].to_list()[2] == "afternoon"


def test_part_of_day_evening(sample_timebin_data):
    """Test part of day binning - evening."""
    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["part_of_day"])
    result = transformer.fit_transform(sample_timebin_data)

    assert result["timestamp__part_of_day"].to_list()[3] == "evening"


def test_part_of_day_all_categories():
    """Test all part of day categories."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 3, 0),  # Night (< 6)
                datetime(2024, 1, 1, 9, 0),  # Morning (6-12)
                datetime(2024, 1, 1, 15, 0),  # Afternoon (12-18)
                datetime(2024, 1, 1, 20, 0),  # Evening (>= 18)
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["part_of_day"])
    result = transformer.fit_transform(X)

    expected = ["night", "morning", "afternoon", "evening"]
    assert result["timestamp__part_of_day"].to_list() == expected


def test_season_northern_hemisphere():
    """Test season binning for northern hemisphere."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15),  # Winter
                datetime(2024, 4, 15),  # Spring
                datetime(2024, 7, 15),  # Summer
                datetime(2024, 10, 15),  # Fall
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["season"], hemisphere="northern")
    result = transformer.fit_transform(X)

    expected = ["winter", "spring", "summer", "fall"]
    assert result["timestamp__season"].to_list() == expected


def test_season_southern_hemisphere():
    """Test season binning for southern hemisphere."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15),  # Summer (southern)
                datetime(2024, 4, 15),  # Fall (southern)
                datetime(2024, 7, 15),  # Winter (southern)
                datetime(2024, 10, 15),  # Spring (southern)
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["season"], hemisphere="southern")
    result = transformer.fit_transform(X)

    expected = ["summer", "fall", "winter", "spring"]
    assert result["timestamp__season"].to_list() == expected


def test_time_of_month():
    """Test time of month binning."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 5),  # Beginning (<= 10)
                datetime(2024, 1, 15),  # Middle (11-20)
                datetime(2024, 1, 25),  # End (> 20)
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["time_of_month"])
    result = transformer.fit_transform(X)

    expected = ["beginning", "middle", "end"]
    assert result["timestamp__time_of_month"].to_list() == expected


def test_time_of_year():
    """Test time of year binning."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 2, 15),  # Early (<= 4)
                datetime(2024, 6, 15),  # Mid (5-8)
                datetime(2024, 10, 15),  # Late (> 8)
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["time_of_year"])
    result = transformer.fit_transform(X)

    expected = ["early", "mid", "late"]
    assert result["timestamp__time_of_year"].to_list() == expected


def test_rush_hour():
    """Test rush hour binning."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 7, 30),  # Morning rush (7-9)
                datetime(2024, 1, 1, 17, 30),  # Evening rush (17-19)
                datetime(2024, 1, 1, 12, 0),  # Off peak
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["rush_hour"])
    result = transformer.fit_transform(X)

    expected = ["morning_rush", "evening_rush", "off_peak"]
    assert result["timestamp__rush_hour"].to_list() == expected


def test_multiple_bin_types(sample_timebin_data):
    """Test with multiple bin types."""
    transformer = TimeBinFeatures(
        subset=["timestamp"], bin_types=["part_of_day", "season", "time_of_month"]
    )
    result = transformer.fit_transform(sample_timebin_data)

    assert "timestamp__part_of_day" in result.columns
    assert "timestamp__season" in result.columns
    assert "timestamp__time_of_month" in result.columns


def test_all_bin_types():
    """Test all bin types together."""
    X = pl.DataFrame({"timestamp": [datetime(2024, 6, 15, 14, 0)], "value": [1]})

    transformer = TimeBinFeatures(
        subset=["timestamp"],
        bin_types=[
            "part_of_day",
            "season",
            "time_of_month",
            "time_of_year",
            "rush_hour",
        ],
    )
    result = transformer.fit_transform(X)

    assert "timestamp__part_of_day" in result.columns
    assert "timestamp__season" in result.columns
    assert "timestamp__time_of_month" in result.columns
    assert "timestamp__time_of_year" in result.columns
    assert "timestamp__rush_hour" in result.columns


def test_drop_columns_false(sample_timebin_data):
    """Test keeping original columns when drop_columns=False."""
    transformer = TimeBinFeatures(
        subset=["timestamp"], bin_types=["part_of_day"], drop_columns=False
    )
    result = transformer.fit_transform(sample_timebin_data)

    assert "timestamp" in result.columns
    assert "timestamp__part_of_day" in result.columns


def test_drop_columns_true(sample_timebin_data):
    """Test dropping original columns when drop_columns=True."""
    transformer = TimeBinFeatures(
        subset=["timestamp"], bin_types=["part_of_day"], drop_columns=True
    )
    result = transformer.fit_transform(sample_timebin_data)

    assert "timestamp" not in result.columns
    assert "timestamp__part_of_day" in result.columns


def test_auto_detect_datetime_columns():
    """Test automatic detection of datetime columns."""
    X = pl.DataFrame(
        {
            "dt1": [datetime(2024, 1, 1, 10, 0)],
            "dt2": [datetime(2024, 6, 1, 15, 0)],
            "value": [100],
        }
    )

    transformer = TimeBinFeatures(bin_types=["part_of_day"])
    result = transformer.fit_transform(X)

    assert "dt1__part_of_day" in result.columns
    assert "dt2__part_of_day" in result.columns


def test_validation_invalid_bin_type():
    """Test validation for invalid bin type."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        TimeBinFeatures(bin_types=["invalid_bin_type"])


def test_sklearn_compatibility(sample_timebin_data):
    """Test sklearn pipeline compatibility."""
    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["part_of_day"])

    # Test fit returns self
    result = transformer.fit(sample_timebin_data)
    assert result is transformer

    # Test fit_transform
    result = transformer.fit_transform(sample_timebin_data)
    assert isinstance(result, pl.DataFrame)


def test_multiple_datetime_columns():
    """Test with multiple datetime columns."""
    X = pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 15, 10, 0), datetime(2024, 7, 15, 15, 0)],
            "updated_at": [datetime(2024, 3, 10, 8, 0), datetime(2024, 9, 20, 20, 0)],
            "value": [1, 2],
        }
    )

    transformer = TimeBinFeatures(
        subset=["created_at", "updated_at"], bin_types=["part_of_day", "season"]
    )
    result = transformer.fit_transform(X)

    assert "created_at__part_of_day" in result.columns
    assert "updated_at__part_of_day" in result.columns
    assert "created_at__season" in result.columns
    assert "updated_at__season" in result.columns


def test_edge_case_hour_boundaries():
    """Test exact hour boundaries for part of day."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 6, 0),  # Exactly 6am (morning)
                datetime(2024, 1, 1, 12, 0),  # Exactly 12pm (afternoon)
                datetime(2024, 1, 1, 18, 0),  # Exactly 6pm (evening)
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["part_of_day"])
    result = transformer.fit_transform(X)

    expected = ["morning", "afternoon", "evening"]
    assert result["timestamp__part_of_day"].to_list() == expected


def test_edge_case_day_boundaries():
    """Test exact day boundaries for time of month."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 10),  # Exactly day 10 (beginning)
                datetime(2024, 1, 20),  # Exactly day 20 (middle)
                datetime(2024, 1, 21),  # Day 21 (end)
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["time_of_month"])
    result = transformer.fit_transform(X)

    expected = ["beginning", "middle", "end"]
    assert result["timestamp__time_of_month"].to_list() == expected


def test_edge_case_month_boundaries():
    """Test exact month boundaries for seasons."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 2, 28),  # End of winter month
                datetime(2024, 3, 1),  # Start of spring month
                datetime(2024, 5, 31),  # End of spring month
                datetime(2024, 6, 1),  # Start of summer month
            ],
            "value": [1, 2, 3, 4],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["season"], hemisphere="northern")
    result = transformer.fit_transform(X)

    expected = ["winter", "spring", "spring", "summer"]
    assert result["timestamp__season"].to_list() == expected


def test_rush_hour_boundaries():
    """Test rush hour exact boundaries."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 6, 59),  # Just before morning rush
                datetime(2024, 1, 1, 7, 0),  # Start of morning rush
                datetime(2024, 1, 1, 9, 0),  # End of morning rush (off peak)
                datetime(2024, 1, 1, 17, 0),  # Start of evening rush
                datetime(2024, 1, 1, 19, 0),  # End of evening rush (off peak)
            ],
            "value": [1, 2, 3, 4, 5],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["rush_hour"])
    result = transformer.fit_transform(X)

    expected = ["off_peak", "morning_rush", "off_peak", "evening_rush", "off_peak"]
    assert result["timestamp__rush_hour"].to_list() == expected


def test_winter_months_all():
    """Test all winter months in northern hemisphere."""
    X = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 12, 15),
                datetime(2024, 1, 15),
                datetime(2024, 2, 15),
            ],
            "value": [1, 2, 3],
        }
    )

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["season"], hemisphere="northern")
    result = transformer.fit_transform(X)

    assert all(s == "winter" for s in result["timestamp__season"].to_list())


def test_single_row():
    """Test with single row dataframe."""
    X = pl.DataFrame({"timestamp": [datetime(2024, 6, 15, 14, 30)], "value": [1]})

    transformer = TimeBinFeatures(subset=["timestamp"], bin_types=["part_of_day", "season"])
    result = transformer.fit_transform(X)

    assert result["timestamp__part_of_day"].to_list() == ["afternoon"]
    assert result["timestamp__season"].to_list() == ["summer"]
