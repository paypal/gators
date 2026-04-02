from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_dt.duration_to_datetime import DurationToDatetime


@pytest.fixture
def sample_offset_data():
    """Create sample data with numeric time offsets."""
    return pl.DataFrame(
        {
            "TransactionDT": [86400, 172800, 259200],  # 1, 2, 3 days in seconds
            "offset_days": [1, 2, 3],
            "value": [100, 200, 300],
        }
    )


def test_basic_seconds_to_datetime(sample_offset_data):
    """Test basic conversion of seconds to datetime with fixed reference date."""
    ref_date = datetime(2017, 11, 30)
    transformer = DurationToDatetime(
        subset=["TransactionDT"], reference_date=ref_date, unit="s", drop_columns=False
    )
    result = transformer.fit_transform(sample_offset_data)

    assert "TransactionDT__datetime" in result.columns
    assert "TransactionDT" in result.columns  # Original column preserved
    
    expected_dates = [
        datetime(2017, 12, 1, 0, 0, 0),
        datetime(2017, 12, 2, 0, 0, 0),
        datetime(2017, 12, 3, 0, 0, 0),
    ]
    assert result["TransactionDT__datetime"].to_list() == expected_dates


def test_days_to_datetime(sample_offset_data):
    """Test conversion of days to datetime."""
    ref_date = datetime(2024, 1, 1)
    transformer = DurationToDatetime(
        subset=["offset_days"], reference_date=ref_date, unit="d", drop_columns=False
    )
    result = transformer.fit_transform(sample_offset_data)

    assert "offset_days__datetime" in result.columns
    
    expected_dates = [
        datetime(2024, 1, 2, 0, 0, 0),
        datetime(2024, 1, 3, 0, 0, 0),
        datetime(2024, 1, 4, 0, 0, 0),
    ]
    assert result["offset_days__datetime"].to_list() == expected_dates


def test_hours_to_datetime():
    """Test conversion of hours to datetime."""
    X = pl.DataFrame(
        {
            "offset_hours": [24, 48, 72],
            "value": [1, 2, 3],
        }
    )
    
    ref_date = datetime(2024, 1, 1, 0, 0, 0)
    transformer = DurationToDatetime(
        subset=["offset_hours"], reference_date=ref_date, unit="h", drop_columns=False
    )
    result = transformer.fit_transform(X)

    expected_dates = [
        datetime(2024, 1, 2, 0, 0, 0),
        datetime(2024, 1, 3, 0, 0, 0),
        datetime(2024, 1, 4, 0, 0, 0),
    ]
    assert result["offset_hours__datetime"].to_list() == expected_dates


def test_minutes_to_datetime():
    """Test conversion of minutes to datetime."""
    X = pl.DataFrame(
        {
            "offset_minutes": [60, 120, 180],
            "value": [1, 2, 3],
        }
    )
    
    ref_date = datetime(2024, 1, 1, 10, 0, 0)
    transformer = DurationToDatetime(
        subset=["offset_minutes"], reference_date=ref_date, unit="m", drop_columns=False
    )
    result = transformer.fit_transform(X)

    expected_dates = [
        datetime(2024, 1, 1, 11, 0, 0),
        datetime(2024, 1, 1, 12, 0, 0),
        datetime(2024, 1, 1, 13, 0, 0),
    ]
    assert result["offset_minutes__datetime"].to_list() == expected_dates


def test_milliseconds_to_datetime():
    """Test conversion of milliseconds to datetime."""
    X = pl.DataFrame(
        {
            "offset_ms": [1000, 2000, 3000],  # 1, 2, 3 seconds
            "value": [1, 2, 3],
        }
    )
    
    ref_date = datetime(2024, 1, 1, 0, 0, 0)
    transformer = DurationToDatetime(
        subset=["offset_ms"], reference_date=ref_date, unit="ms", drop_columns=False
    )
    result = transformer.fit_transform(X)

    expected_dates = [
        datetime(2024, 1, 1, 0, 0, 1),
        datetime(2024, 1, 1, 0, 0, 2),
        datetime(2024, 1, 1, 0, 0, 3),
    ]
    assert result["offset_ms__datetime"].to_list() == expected_dates


def test_microseconds_to_datetime():
    """Test conversion of microseconds to datetime."""
    X = pl.DataFrame(
        {
            "offset_us": [1_000_000, 2_000_000, 3_000_000],  # 1, 2, 3 seconds
            "value": [1, 2, 3],
        }
    )
    
    ref_date = datetime(2024, 1, 1, 0, 0, 0)
    transformer = DurationToDatetime(
        subset=["offset_us"], reference_date=ref_date, unit="us", drop_columns=False
    )
    result = transformer.fit_transform(X)

    expected_dates = [
        datetime(2024, 1, 1, 0, 0, 1),
        datetime(2024, 1, 1, 0, 0, 2),
        datetime(2024, 1, 1, 0, 0, 3),
    ]
    assert result["offset_us__datetime"].to_list() == expected_dates


def test_reference_date_as_column():
    """Test using a column as reference date (different per row)."""
    X = pl.DataFrame(
        {
            "BaseDate": [
                datetime(2024, 1, 1),
                datetime(2024, 2, 1),
                datetime(2024, 3, 1),
            ],
            "offset_days": [7, 14, 21],
            "value": [100, 200, 300],
        }
    )
    
    transformer = DurationToDatetime(
        subset=["offset_days"], reference_date="BaseDate", unit="d", drop_columns=False
    )
    result = transformer.fit_transform(X)

    assert "offset_days__datetime" in result.columns
    assert "BaseDate" in result.columns
    
    expected_dates = [
        datetime(2024, 1, 8, 0, 0, 0),
        datetime(2024, 2, 15, 0, 0, 0),
        datetime(2024, 3, 22, 0, 0, 0),
    ]
    assert result["offset_days__datetime"].to_list() == expected_dates


def test_reference_date_as_iso_string():
    """Test using ISO format string as reference date."""
    X = pl.DataFrame(
        {
            "offset_days": [1, 2, 3],
            "value": [100, 200, 300],
        }
    )
    
    transformer = DurationToDatetime(
        subset=["offset_days"], 
        reference_date="2024-01-01", 
        unit="d", 
        drop_columns=False
    )
    result = transformer.fit_transform(X)

    expected_dates = [
        datetime(2024, 1, 2, 0, 0, 0),
        datetime(2024, 1, 3, 0, 0, 0),
        datetime(2024, 1, 4, 0, 0, 0),
    ]
    assert result["offset_days__datetime"].to_list() == expected_dates


def test_multiple_columns():
    """Test transforming multiple columns at once."""
    X = pl.DataFrame(
        {
            "offset1": [1, 2, 3],
            "offset2": [10, 20, 30],
            "value": [100, 200, 300],
        }
    )
    
    ref_date = datetime(2024, 1, 1)
    transformer = DurationToDatetime(
        subset=["offset1", "offset2"], 
        reference_date=ref_date, 
        unit="d", 
        drop_columns=False
    )
    result = transformer.fit_transform(X)

    assert "offset1__datetime" in result.columns
    assert "offset2__datetime" in result.columns
    assert "offset1" in result.columns
    assert "offset2" in result.columns


def test_drop_columns_true(sample_offset_data):
    """Test dropping original columns when drop_columns=True."""
    ref_date = datetime(2017, 11, 30)
    transformer = DurationToDatetime(
        subset=["TransactionDT"], reference_date=ref_date, unit="s", drop_columns=True
    )
    result = transformer.fit_transform(sample_offset_data)

    assert "TransactionDT__datetime" in result.columns
    assert "TransactionDT" not in result.columns
    assert "value" in result.columns


def test_drop_columns_false(sample_offset_data):
    """Test keeping original columns when drop_columns=False."""
    ref_date = datetime(2017, 11, 30)
    transformer = DurationToDatetime(
        subset=["TransactionDT"], reference_date=ref_date, unit="s", drop_columns=False
    )
    result = transformer.fit_transform(sample_offset_data)

    assert "TransactionDT__datetime" in result.columns
    assert "TransactionDT" in result.columns
    assert result.shape[1] == sample_offset_data.shape[1] + 1


def test_invalid_unit():
    """Test that invalid units raise ValidationError from Pydantic."""
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError, match="Input should be"):
        DurationToDatetime(
            subset=["col"], reference_date=datetime(2024, 1, 1), unit="invalid"
        )


def test_invalid_reference_date_type():
    """Test that invalid reference_date types raise ValidationError from Pydantic."""
    from pydantic import ValidationError
    
    X = pl.DataFrame({"offset": [1, 2, 3]})
    
    with pytest.raises(ValidationError):
        DurationToDatetime(
            subset=["offset"], reference_date=[1, 2, 3], unit="d"  # Invalid type (list)
        )


def test_invalid_column_reference():
    """Test that non-existent column reference raises ValueError."""
    X = pl.DataFrame({"offset": [1, 2, 3]})
    
    transformer = DurationToDatetime(
        subset=["offset"], 
        reference_date="NonExistentColumn",  # Column doesn't exist
        unit="d"
    )
    
    with pytest.raises(ValueError, match="neither a column .* nor a valid ISO format"):
        transformer.fit(X)


def test_negative_offsets():
    """Test handling negative time offsets (dates before reference)."""
    X = pl.DataFrame(
        {
            "offset_days": [-5, -10, -15],
            "value": [1, 2, 3],
        }
    )
    
    ref_date = datetime(2024, 1, 15)
    transformer = DurationToDatetime(
        subset=["offset_days"], reference_date=ref_date, unit="d", drop_columns=False
    )
    result = transformer.fit_transform(X)

    expected_dates = [
        datetime(2024, 1, 10, 0, 0, 0),
        datetime(2024, 1, 5, 0, 0, 0),
        datetime(2023, 12, 31, 0, 0, 0),
    ]
    assert result["offset_days__datetime"].to_list() == expected_dates


def test_fit_transform_consistency():
    """Test that fit().transform() gives same result as fit_transform()."""
    X = pl.DataFrame(
        {
            "offset_days": [1, 2, 3],
            "value": [100, 200, 300],
        }
    )
    
    ref_date = datetime(2024, 1, 1)
    transformer1 = DurationToDatetime(
        subset=["offset_days"], reference_date=ref_date, unit="d"
    )
    transformer2 = DurationToDatetime(
        subset=["offset_days"], reference_date=ref_date, unit="d"
    )
    
    result1 = transformer1.fit_transform(X)
    result2 = transformer2.fit(X).transform(X)
    
    assert_frame_equal(result1, result2)