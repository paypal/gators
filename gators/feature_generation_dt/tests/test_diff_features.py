from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_dt.diff_features import DiffFeatures


@pytest.fixture
def sample_diff_data():
    """Create sample data for time difference testing."""
    return pl.DataFrame(
        {
            "created_at": [
                datetime(2023, 1, 1, 0, 0, 0),
                datetime(2023, 6, 15, 12, 0, 0),
                datetime(2024, 1, 1, 0, 0, 0),
            ],
            "updated_at": [
                datetime(2023, 1, 10, 0, 0, 0),
                datetime(2023, 7, 1, 12, 0, 0),
                datetime(2024, 2, 1, 0, 0, 0),
            ],
            "value": [100, 200, 300],
        }
    )


def test_basic_pairwise_diff_days(sample_diff_data):
    """Test basic pairwise difference in days."""
    transformer = DiffFeatures(column_pairs=[("updated_at", "created_at")], units=["d"])
    result = transformer.fit_transform(sample_diff_data)

    assert "updated_at_minus_created_at__days" in result.columns
    assert result["updated_at_minus_created_at__days"].to_list() == [9, 16, 31]


def test_pairwise_diff_multiple_units(sample_diff_data):
    """Test pairwise difference with multiple units."""
    transformer = DiffFeatures(column_pairs=[("updated_at", "created_at")], units=["d", "h"])
    result = transformer.fit_transform(sample_diff_data)

    assert "updated_at_minus_created_at__days" in result.columns
    assert "updated_at_minus_created_at__hours" in result.columns
    assert result["updated_at_minus_created_at__days"].to_list() == [9, 16, 31]
    assert result["updated_at_minus_created_at__hours"].to_list() == [216, 384, 744]


def test_pairwise_diff_minutes_and_seconds():
    """Test pairwise difference in minutes and seconds."""
    X = pl.DataFrame(
        {
            "start_time": [datetime(2024, 1, 1, 10, 0, 0)],
            "end_time": [datetime(2024, 1, 1, 10, 5, 30)],
            "value": [1],
        }
    )

    transformer = DiffFeatures(column_pairs=[("end_time", "start_time")], units=["m", "s"])
    result = transformer.fit_transform(X)

    assert result["end_time_minus_start_time__minutes"].to_list() == [5]
    assert result["end_time_minus_start_time__seconds"].to_list() == [330]


def test_reference_date_string(sample_diff_data):
    """Test difference from reference date (as string)."""
    transformer = DiffFeatures(reference_dates={"created_at": "2023-01-01"}, units=["d"])
    result = transformer.fit_transform(sample_diff_data)

    assert "created_at_since_ref__days" in result.columns
    assert result["created_at_since_ref__days"].to_list() == [0, 165, 365]


def test_reference_date_datetime_object(sample_diff_data):
    """Test difference from reference date (as datetime object)."""
    ref_date = datetime(2023, 1, 1)

    transformer = DiffFeatures(reference_dates={"created_at": ref_date}, units=["d"])
    result = transformer.fit_transform(sample_diff_data)

    assert "created_at_since_ref__days" in result.columns
    assert result["created_at_since_ref__days"].to_list() == [0, 165, 365]


def test_multiple_reference_dates():
    """Test with multiple reference dates."""
    X = pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1), datetime(2024, 6, 1)],
            "updated_at": [datetime(2024, 2, 1), datetime(2024, 7, 1)],
            "value": [1, 2],
        }
    )

    transformer = DiffFeatures(
        reference_dates={"created_at": "2024-01-01", "updated_at": "2024-01-01"},
        units=["d"],
    )
    result = transformer.fit_transform(X)

    assert "created_at_since_ref__days" in result.columns
    assert "updated_at_since_ref__days" in result.columns


def test_multiple_column_pairs():
    """Test with multiple column pairs."""
    X = pl.DataFrame(
        {
            "start1": [datetime(2024, 1, 1)],
            "end1": [datetime(2024, 1, 10)],
            "start2": [datetime(2024, 2, 1)],
            "end2": [datetime(2024, 2, 5)],
            "value": [1],
        }
    )

    transformer = DiffFeatures(column_pairs=[("end1", "start1"), ("end2", "start2")], units=["d"])
    result = transformer.fit_transform(X)

    assert "end1_minus_start1__days" in result.columns
    assert "end2_minus_start2__days" in result.columns
    assert result["end1_minus_start1__days"].to_list() == [9]
    assert result["end2_minus_start2__days"].to_list() == [4]


def test_drop_columns_false(sample_diff_data):
    """Test keeping original columns when drop_columns=False."""
    transformer = DiffFeatures(
        column_pairs=[("updated_at", "created_at")], units=["d"], drop_columns=False
    )
    result = transformer.fit_transform(sample_diff_data)

    assert "created_at" in result.columns
    assert "updated_at" in result.columns
    assert "updated_at_minus_created_at__days" in result.columns


def test_drop_columns_true(sample_diff_data):
    """Test dropping original columns when drop_columns=True."""
    transformer = DiffFeatures(
        column_pairs=[("updated_at", "created_at")], units=["d"], drop_columns=True
    )
    result = transformer.fit_transform(sample_diff_data)

    assert "created_at" not in result.columns
    assert "updated_at" not in result.columns
    assert "updated_at_minus_created_at__days" in result.columns


def test_drop_columns_with_reference_dates(sample_diff_data):
    """Test dropping columns when using reference dates."""
    transformer = DiffFeatures(
        reference_dates={"created_at": "2023-01-01"}, units=["d"], drop_columns=True
    )
    result = transformer.fit_transform(sample_diff_data)

    assert "created_at" not in result.columns
    assert "created_at_since_ref__days" in result.columns


def test_validation_invalid_unit():
    """Test validation for invalid unit."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        DiffFeatures(column_pairs=[("end", "start")], units=["milliseconds"])  # Invalid unit


def test_sklearn_compatibility(sample_diff_data):
    """Test sklearn pipeline compatibility."""
    transformer = DiffFeatures(column_pairs=[("updated_at", "created_at")], units=["d"])

    # Test fit returns self
    result = transformer.fit(sample_diff_data)
    assert result is transformer

    # Test fit_transform
    result = transformer.fit_transform(sample_diff_data)
    assert isinstance(result, pl.DataFrame)


def test_all_units_together():
    """Test all units calculated together."""
    X = pl.DataFrame(
        {
            "start": [datetime(2024, 1, 1, 0, 0, 0)],
            "end": [datetime(2024, 1, 2, 2, 30, 45)],
            "value": [1],
        }
    )

    transformer = DiffFeatures(column_pairs=[("end", "start")], units=["d", "h", "m", "s"])
    result = transformer.fit_transform(X)

    assert result["end_minus_start__days"].to_list() == [1]
    assert result["end_minus_start__hours"].to_list() == [26]
    assert result["end_minus_start__minutes"].to_list() == [1590]
    assert result["end_minus_start__seconds"].to_list() == [95445]


def test_negative_time_difference():
    """Test negative time differences (end before start)."""
    X = pl.DataFrame(
        {"start": [datetime(2024, 1, 10)], "end": [datetime(2024, 1, 5)], "value": [1]}
    )

    transformer = DiffFeatures(column_pairs=[("end", "start")], units=["d"])
    result = transformer.fit_transform(X)

    assert result["end_minus_start__days"].to_list() == [-5]


def test_combined_pairwise_and_reference():
    """Test using both pairwise differences and reference dates."""
    X = pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1)],
            "updated_at": [datetime(2024, 1, 10)],
            "value": [1],
        }
    )

    transformer = DiffFeatures(
        column_pairs=[("updated_at", "created_at")],
        reference_dates={"created_at": "2023-01-01"},
        units=["d"],
    )
    result = transformer.fit_transform(X)

    assert "updated_at_minus_created_at__days" in result.columns
    assert "created_at_since_ref__days" in result.columns


def test_zero_time_difference():
    """Test zero time difference (same datetime)."""
    X = pl.DataFrame(
        {
            "start": [datetime(2024, 1, 1, 10, 0, 0)],
            "end": [datetime(2024, 1, 1, 10, 0, 0)],
            "value": [1],
        }
    )

    transformer = DiffFeatures(column_pairs=[("end", "start")], units=["d", "h", "m", "s"])
    result = transformer.fit_transform(X)

    assert result["end_minus_start__days"].to_list() == [0]
    assert result["end_minus_start__hours"].to_list() == [0]
    assert result["end_minus_start__minutes"].to_list() == [0]
    assert result["end_minus_start__seconds"].to_list() == [0]


def test_reference_date_future():
    """Test difference when reference date is in the future."""
    X = pl.DataFrame({"event_date": [datetime(2023, 1, 1)], "value": [1]})

    transformer = DiffFeatures(reference_dates={"event_date": "2024-01-01"}, units=["d"])
    result = transformer.fit_transform(X)

    # Should be negative since event is before reference
    assert result["event_date_since_ref__days"].to_list() == [-365]


def test_hours_conversion():
    """Test hours conversion accuracy."""
    X = pl.DataFrame(
        {
            "start": [datetime(2024, 1, 1, 0, 0, 0)],
            "end": [datetime(2024, 1, 1, 3, 30, 0)],
            "value": [1],
        }
    )

    transformer = DiffFeatures(column_pairs=[("end", "start")], units=["h"])
    result = transformer.fit_transform(X)

    assert result["end_minus_start__hours"].to_list() == [3]
