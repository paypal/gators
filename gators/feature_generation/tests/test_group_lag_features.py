import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import GroupLagFeatures


def test_transform_basic_single_lag():
    """Test basic transformation with single lag."""
    X = pl.DataFrame(
        {
            "amount": [100, 200, 250, 150, 300, 180],
            "cat1": ["A", "A", "A", "B", "B", "B"],
            "time": [1, 2, 3, 1, 2, 3],
        }
    ).sort(["cat1", "time"])

    transformer = GroupLagFeatures(
        subset=["amount"], by=["cat1"], lags=[1]
    )
    result = transformer.fit_transform(X)

    assert "amount_lag1_cat1" in result.columns

    # First row in group A should be null
    assert result["amount_lag1_cat1"][0] is None or result.select(
        pl.col("amount_lag1_cat1").is_null()
    )[0, 0]
    # Second row in group A should be 100
    assert result["amount_lag1_cat1"][1] == 100
    # Third row in group A should be 200
    assert result["amount_lag1_cat1"][2] == 200


def test_transform_multiple_lags():
    """Test with multiple lag periods."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30, 40, 50, 60],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1, 2]
    )
    result = transformer.fit_transform(X)

    assert "value_lag1_group" in result.columns
    assert "value_lag2_group" in result.columns

    # Group A
    assert result["value_lag1_group"][0] is None or result.select(
        pl.col("value_lag1_group").is_null()
    )[0, 0]
    assert result["value_lag1_group"][1] == 10
    assert result["value_lag2_group"][2] == 10


def test_transform_with_leads():
    """Test with lead features."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30, 40, 50, 60],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1], leads=[1]
    )
    result = transformer.fit_transform(X)

    assert "value_lag1_group" in result.columns
    assert "value_lead1_group" in result.columns

    # Group A: lag looks backward, lead looks forward
    assert result["value_lag1_group"][1] == 10
    assert result["value_lead1_group"][0] == 20


def test_transform_multiple_leads():
    """Test with multiple lead periods."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30, 40, 50, 60],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[], leads=[1, 2]
    )
    result = transformer.fit_transform(X)

    assert "value_lead1_group" in result.columns
    assert "value_lead2_group" in result.columns

    # Last row should have null leads
    assert result["value_lead1_group"][2] is None or result.select(
        pl.col("value_lead1_group").is_null()
    )[2, 0]


def test_transform_multiple_subset():
    """Test with multiple numerical columns."""
    X = pl.DataFrame(
        {
            "col1": [100, 200, 150, 300],
            "col2": [10, 20, 15, 30],
            "group": ["A", "A", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["col1", "col2"], by=["group"], lags=[1]
    )
    result = transformer.fit_transform(X)

    assert "col1_lag1_group" in result.columns
    assert "col2_lag1_group" in result.columns

    assert result["col1_lag1_group"][1] == 100
    assert result["col2_lag1_group"][1] == 10


def test_transform_multiple_by():
    """Test grouping by multiple columns simultaneously."""
    X = pl.DataFrame(
        {
            "value": [100, 200, 150, 300, 250, 175],
            "cat1": ["A", "A", "B", "B", "A", "A"],
            "cat2": ["X", "Y", "X", "X", "X", "Y"],
        }
    ).sort(["cat1", "cat2"])

    transformer = GroupLagFeatures(
        subset=["value"], by=["cat1", "cat2"], lags=[1]
    )
    result = transformer.fit_transform(X)

    assert "value_lag1_cat1_cat2" in result.columns


def test_transform_with_fill_value():
    """Test using fill_value for missing lags."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30, 40],
            "group": ["A", "A", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["value"],
        by=["group"],
        lags=[1],
        fill_value=0.0,
    )
    result = transformer.fit_transform(X)

    # First row should have fill_value instead of null
    assert result["value_lag1_group"][0] == 0.0


def test_transform_with_fill_value_leads():
    """Test fill_value works for lead features too."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30, 40],
            "group": ["A", "A", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["value"],
        by=["group"],
        lags=[],
        leads=[1],
        fill_value=-999.0,
    )
    result = transformer.fit_transform(X)

    # Last row in each group should have fill_value
    assert result["value_lead1_group"][1] == -999.0


def test_transform_with_drop_columns():
    """Test dropping original numerical columns."""
    X = pl.DataFrame(
        {"value": [10, 20, 30], "other": [1, 2, 3], "group": ["A", "A", "B"]}
    )

    transformer = GroupLagFeatures(
        subset=["value"],
        by=["group"],
        lags=[1],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    assert "value" not in result.columns
    assert "value_lag1_group" in result.columns
    assert "other" in result.columns
    assert "group" in result.columns


def test_transform_with_custom_column_names():
    """Test using custom column names."""
    X = pl.DataFrame(
        {"value": [10, 20, 30, 40], "group": ["A", "A", "B", "B"]}
    )

    transformer = GroupLagFeatures(
        subset=["value"],
        by=["group"],
        lags=[1],
        leads=[1],
        new_column_names=["prev_value", "next_value"],
    )
    result = transformer.fit_transform(X)

    assert "prev_value" in result.columns
    assert "next_value" in result.columns
    assert "value_lag1_group" not in result.columns
    assert "value_lead1_group" not in result.columns


def test_complex_scenario():
    """Test complex scenario with multiple nums, lags, and leads."""
    X = pl.DataFrame(
        {
            "amt1": [100, 200, 150, 300],
            "amt2": [10, 20, 15, 30],
            "group": ["A", "A", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["amt1", "amt2"],
        by=["group"],
        lags=[1, 2],
        leads=[1],
    )
    result = transformer.fit_transform(X)

    # Should create 2 numerical * (2 lags + 1 lead) = 6 new features
    expected_cols = [
        "amt1_lag1_group",
        "amt1_lag2_group",
        "amt1_lead1_group",
        "amt2_lag1_group",
        "amt2_lag2_group",
        "amt2_lead1_group",
    ]

    for col in expected_cols:
        assert col in result.columns


def test_validation_invalid_lag():
    """Test validation error with invalid lag value."""
    with pytest.raises(ValidationError, match="All lag values must be positive"):
        GroupLagFeatures(
            subset=["value"],
            by=["group"],
            lags=[0],  # Invalid: must be positive
        )


def test_validation_negative_lag():
    """Test validation error with negative lag value."""
    with pytest.raises(ValidationError, match="All lag values must be positive"):
        GroupLagFeatures(
            subset=["value"],
            by=["group"],
            lags=[-1],  # Invalid: negative
        )


def test_validation_empty_lags():
    """Test validation error with empty lags and leads lists."""
    with pytest.raises(
        ValidationError, match="At least one of 'lags' or 'leads' must be non-empty"
    ):
        GroupLagFeatures(
            subset=["value"], by=["group"], lags=[]
        )


def test_validation_invalid_lead():
    """Test validation error with invalid lead value."""
    with pytest.raises(ValidationError, match="All lead values must be positive"):
        GroupLagFeatures(
            subset=["value"],
            by=["group"],
            lags=[1],
            leads=[0],  # Invalid
        )


def test_validation_mismatched_new_column_names_length():
    """Test validation error when new_column_names length doesn't match."""
    with pytest.raises(
        ValueError,
        match="Length of new_column_names .* must match the total number of features created",
    ):
        GroupLagFeatures(
            subset=["value"],
            by=["group"],
            lags=[1, 2],
            leads=[1],
            new_column_names=["name1", "name2"],  # Should have 3 names (2 lags + 1 lead)
        )


def test_fit_return_self():
    """Test that fit returns self."""
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1]
    )

    result = transformer.fit(X)
    assert result is transformer


def test_column_mapping_generation():
    """Test that column mapping is correctly generated during fit."""
    X = pl.DataFrame({"value": [10, 20], "group": ["A", "B"]})

    transformer = GroupLagFeatures(
        subset=["value"],
        by=["group"],
        lags=[1],
        leads=[1],
    )
    transformer.fit(X)

    assert len(transformer._column_mapping) == 2
    assert "value_lag1_group" in transformer._column_mapping
    assert "value_lead1_group" in transformer._column_mapping


def test_empty_dataframe():
    """Test behavior with empty dataframe."""
    X = pl.DataFrame(
        {"value": [], "group": []}, schema={"value": pl.Int64, "group": pl.Utf8}
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1]
    )
    result = transformer.fit_transform(X)

    assert result.shape[0] == 0
    assert "value_lag1_group" in result.columns


def test_single_row_dataframe():
    """Test behavior with single row dataframe."""
    X = pl.DataFrame({"value": [100], "group": ["A"]})

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1]
    )
    result = transformer.fit_transform(X)

    # Only one row, so lag should be null
    assert result["value_lag1_group"][0] is None or result.select(
        pl.col("value_lag1_group").is_null()
    )[0, 0]


def test_with_null_values():
    """Test handling of null values in X."""
    X = pl.DataFrame(
        {"value": [10, None, 30, 40, None], "group": ["A", "A", "A", "B", "B"]}
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1]
    )
    result = transformer.fit_transform(X)

    # Lag of null is still null
    assert result["value_lag1_group"][1] == 10  # Lag of second row is first row


def test_negative_values():
    """Test with negative values."""
    X = pl.DataFrame(
        {"value": [-10, -20, 30, 40], "group": ["A", "A", "B", "B"]}
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1]
    )
    result = transformer.fit_transform(X)

    assert result["value_lag1_group"][1] == -10


def test_float_values():
    """Test with float values."""
    X = pl.DataFrame(
        {"value": [10.5, 20.7, 30.2, 40.8], "group": ["A", "A", "B", "B"]}
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1]
    )
    result = transformer.fit_transform(X)

    assert result["value_lag1_group"][1] == pytest.approx(10.5)


def test_large_lag_value():
    """Test with lag value larger than group size."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30, 40],
            "group": ["A", "A", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[5]
    )
    result = transformer.fit_transform(X)

    # All rows should have null since lag is larger than group size
    assert all(result.select(pl.col("value_lag5_group").is_null()).to_series())


def test_lag_and_lead_symmetry():
    """Test that lag and lead are symmetric."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30],
            "group": ["A", "A", "A"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["value"],
        by=["group"],
        lags=[1],
        leads=[1],
    )
    result = transformer.fit_transform(X)

    # Middle row: lag should be 10, lead should be 30
    assert result["value_lag1_group"][1] == 10
    assert result["value_lead1_group"][1] == 30


def test_time_series_pattern():
    """Test realistic time series pattern."""
    X = pl.DataFrame(
        {
            "amount": [100, 150, 200, 80, 120, 90],
            "user": ["U1", "U1", "U1", "U2", "U2", "U2"],
            "timestamp": [1, 2, 3, 1, 2, 3],
        }
    ).sort(["user", "timestamp"])

    transformer = GroupLagFeatures(
        subset=["amount"],
        by=["user"],
        lags=[1, 2],
    )
    result = transformer.fit_transform(X)

    # User U1, timestamp 3: lag1=150, lag2=100
    assert result.filter(pl.col("user") == "U1")["amount_lag1_user"][2] == 150
    assert result.filter(pl.col("user") == "U1")["amount_lag2_user"][2] == 100


def test_multiple_groups_independence():
    """Test that groups are independent."""
    X = pl.DataFrame(
        {
            "value": [10, 20, 30, 100, 200, 300],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    transformer = GroupLagFeatures(
        subset=["value"], by=["group"], lags=[1]
    )
    result = transformer.fit_transform(X)

    # First row of group B should not have lag from group A
    assert result["value_lag1_group"][3] is None or result.select(
        pl.col("value_lag1_group").is_null()
    )[3, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
