import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import RatioFeatures


def test_transform_basic():
    # Test basic ratio calculation
    X = pl.DataFrame({"A": [10, 20, 30, 40], "B": [2, 4, 5, 8]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    transformer.fit(X)
    result = transformer.transform(X)

    expected = pl.DataFrame(
        {"A": [10, 20, 30, 40], "B": [2, 4, 5, 8], "A__div__B": [5.0, 5.0, 6.0, 5.0]}
    )

    assert_frame_equal(result, expected)


def test_transform_multiple_ratios():
    # Test multiple ratio features
    X = pl.DataFrame(
        {
            "revenue": [100, 200, 300, 400],
            "cost": [80, 100, 150, 200],
            "clicks": [1000, 2000, 3000, 4000],
            "impressions": [10000, 20000, 30000, 40000],
        }
    )

    transformer = RatioFeatures(
        numerator_columns=["revenue", "clicks"],
        denominator_columns=["cost", "impressions"],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "revenue": [100, 200, 300, 400],
            "cost": [80, 100, 150, 200],
            "clicks": [1000, 2000, 3000, 4000],
            "impressions": [10000, 20000, 30000, 40000],
            "revenue__div__cost": [1.25, 2.0, 2.0, 2.0],
            "clicks__div__impressions": [0.1, 0.1, 0.1, 0.1],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_division_by_zero():
    # Test that division by zero produces null
    X = pl.DataFrame({"A": [10, 20, 30, 40], "B": [2, 0, 5, 0]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {"A": [10, 20, 30, 40], "B": [2, 0, 5, 0], "A__div__B": [5.0, None, 6.0, None]}
    )

    assert_frame_equal(result, expected)


def test_transform_with_nulls():
    # Test handling of null values in numerator and denominator
    X = pl.DataFrame({"A": [10, None, 30, 40, 50], "B": [2, 5, None, 0, 10]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, None, 30, 40, 50],
            "B": [2, 5, None, 0, 10],
            "A__div__B": [5.0, None, None, None, 5.0],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_custom_column_names():
    # Test custom column names
    X = pl.DataFrame(
        {"revenue": [100, 200, 300], "cost": [80, 100, 150], "clicks": [1000, 2000, 3000]}
    )

    transformer = RatioFeatures(
        numerator_columns=["revenue"],
        denominator_columns=["cost"],
        new_column_names=["profit_margin"],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "revenue": [100, 200, 300],
            "cost": [80, 100, 150],
            "clicks": [1000, 2000, 3000],
            "profit_margin": [1.25, 2.0, 2.0],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_with_drop_columns():
    # Test dropping original columns
    X = pl.DataFrame({"A": [10, 20, 30], "B": [2, 4, 5], "C": [100, 200, 300]})

    transformer = RatioFeatures(
        numerator_columns=["A"], denominator_columns=["B"], drop_columns=True
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame({"C": [100, 200, 300], "A__div__B": [5.0, 5.0, 6.0]})

    assert_frame_equal(result, expected)


def test_transform_drop_columns_multiple_ratios():
    # Test dropping columns when creating multiple ratios
    X = pl.DataFrame(
        {
            "A": [10, 20, 30],
            "B": [2, 4, 5],
            "C": [100, 200, 300],
            "D": [10, 20, 30],
            "E": [1, 2, 3],
        }
    )

    transformer = RatioFeatures(
        numerator_columns=["A", "C"],
        denominator_columns=["B", "D"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {"E": [1, 2, 3], "A__div__B": [5.0, 5.0, 6.0], "C__div__D": [10.0, 10.0, 10.0]}
    )

    assert_frame_equal(result, expected)


def test_transform_drop_columns_with_overlap():
    # Test dropping columns when same column is used as both numerator and denominator
    X = pl.DataFrame({"A": [10, 20, 30], "B": [2, 4, 5], "C": [100, 200, 300]})

    transformer = RatioFeatures(
        numerator_columns=["A", "B"],
        denominator_columns=["B", "A"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "C": [100, 200, 300],
            "A__div__B": [5.0, 5.0, 6.0],
            "B__div__A": [0.2, 0.2, 0.16666666666666666],
        }
    )

    assert_frame_equal(result, expected)


def test_fit_transform():
    # Test fit_transform method
    X = pl.DataFrame({"A": [10, 20, 30], "B": [2, 4, 5]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame({"A": [10, 20, 30], "B": [2, 4, 5], "A__div__B": [5.0, 5.0, 6.0]})

    assert_frame_equal(result, expected)


def test_length_mismatch_columns():
    # Test that mismatched lengths raise an error
    with pytest.raises(ValidationError):
        RatioFeatures(numerator_columns=["A", "B"], denominator_columns=["C"])


def test_length_mismatch_new_column_names():
    # Test that new_column_names length must match
    with pytest.raises(ValidationError):
        RatioFeatures(
            numerator_columns=["A", "B"],
            denominator_columns=["C", "D"],
            new_column_names=["ratio1"],
        )


def test_negative_values():
    # Test handling of negative values
    X = pl.DataFrame({"A": [-10, 20, -30, 40], "B": [2, -4, 5, -8]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {"A": [-10, 20, -30, 40], "B": [2, -4, 5, -8], "A__div__B": [-5.0, -5.0, -6.0, -5.0]}
    )

    assert_frame_equal(result, expected)


def test_float_values():
    # Test handling of float values
    X = pl.DataFrame({"A": [10.5, 20.2, 30.8], "B": [2.1, 4.04, 5.0]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10.5, 20.2, 30.8],
            "B": [2.1, 4.04, 5.0],
            "A__div__B": [5.0, 5.0, 6.16],
        }
    )

    assert_frame_equal(result, expected)


def test_zero_numerator():
    # Test when numerator is zero (should return 0.0)
    X = pl.DataFrame({"A": [0, 0, 10], "B": [2, 5, 10]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame({"A": [0, 0, 10], "B": [2, 5, 10], "A__div__B": [0.0, 0.0, 1.0]})

    assert_frame_equal(result, expected)
