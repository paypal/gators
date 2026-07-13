import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import RatioFeatures


def test_transform_basic():
    # B=[1,3,9,19] so denom+1=[2,4,10,20] → [5.0, 5.0, 3.0, 2.0]
    X = pl.DataFrame({"A": [10, 20, 30, 40], "B": [1, 3, 9, 19]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    transformer.fit(X)
    result = transformer.transform(X)

    expected = pl.DataFrame(
        {"A": [10, 20, 30, 40], "B": [1, 3, 9, 19], "A__div__B": [5.0, 5.0, 3.0, 2.0]}
    )

    assert_frame_equal(result, expected)


def test_transform_multiple_ratios():
    # cost=[99,99,149,199] → denom+1=[100,100,150,200] → revenue ratios [1.0,2.0,2.0,2.0]
    # impressions=[9999,...] → denom+1=[10000,...] → click ratios [0.1,...]
    X = pl.DataFrame(
        {
            "revenue": [100, 200, 300, 400],
            "cost": [99, 99, 149, 199],
            "clicks": [1000, 2000, 3000, 4000],
            "impressions": [9999, 19999, 29999, 39999],
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
            "cost": [99, 99, 149, 199],
            "clicks": [1000, 2000, 3000, 4000],
            "impressions": [9999, 19999, 29999, 39999],
            "revenue__div__cost": [1.0, 2.0, 2.0, 2.0],
            "clicks__div__impressions": [0.1, 0.1, 0.1, 0.1],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_zero_denominator_laplace():
    # Laplace smoothing: denom+1 ensures zero denominators never produce null
    X = pl.DataFrame({"A": [10, 20, 30, 40], "B": [1, 0, 4, 0]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    # B=[1,0,4,0] → denom+1=[2,1,5,1] → [5.0, 20.0, 6.0, 40.0]  (no nulls)
    expected = pl.DataFrame(
        {"A": [10, 20, 30, 40], "B": [1, 0, 4, 0], "A__div__B": [5.0, 20.0, 6.0, 40.0]}
    )

    assert_frame_equal(result, expected)


def test_transform_with_nulls():
    # Input nulls propagate; zero denominator (B=0) is smoothed to 1, not null
    X = pl.DataFrame({"A": [10, None, 30, 40, 50], "B": [1, 4, None, 0, 9]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    # B=[1,4,None,0,9] → denom+1=[2,5,null,1,10]
    # A=[10,null,30,40,50] → [5.0, null, null, 40.0, 5.0]
    expected = pl.DataFrame(
        {
            "A": [10, None, 30, 40, 50],
            "B": [1, 4, None, 0, 9],
            "A__div__B": [5.0, None, None, 40.0, 5.0],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_custom_column_names():
    # cost=[99,99,149] → denom+1=[100,100,150] → [1.0, 2.0, 2.0]
    X = pl.DataFrame(
        {"revenue": [100, 200, 300], "cost": [99, 99, 149], "clicks": [1000, 2000, 3000]}
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
            "cost": [99, 99, 149],
            "clicks": [1000, 2000, 3000],
            "profit_margin": [1.0, 2.0, 2.0],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_with_drop_columns():
    # B=[1,3,9] → denom+1=[2,4,10] → [5.0, 5.0, 3.0]
    X = pl.DataFrame({"A": [10, 20, 30], "B": [1, 3, 9], "C": [100, 200, 300]})

    transformer = RatioFeatures(
        numerator_columns=["A"], denominator_columns=["B"], drop_columns=True
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame({"C": [100, 200, 300], "A__div__B": [5.0, 5.0, 3.0]})

    assert_frame_equal(result, expected)


def test_transform_drop_columns_multiple_ratios():
    # A/B: B=[1,3,9] → [5.0, 5.0, 3.0]; C/D: D=[9,19,29] → [10.0, 10.0, 10.0]
    X = pl.DataFrame(
        {
            "A": [10, 20, 30],
            "B": [1, 3, 9],
            "C": [100, 200, 300],
            "D": [9, 19, 29],
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
        {"E": [1, 2, 3], "A__div__B": [5.0, 5.0, 3.0], "C__div__D": [10.0, 10.0, 10.0]}
    )

    assert_frame_equal(result, expected)


def test_transform_drop_columns_with_overlap():
    # A/B and B/A with Laplace: A=[10,20,30], B=[4,4,9]
    # A/(B+1): [10/5=2.0, 20/5=4.0, 30/10=3.0]
    # B/(A+1): [4/11, 4/21, 9/31]
    X = pl.DataFrame({"A": [10, 20, 30], "B": [4, 4, 9], "C": [100, 200, 300]})

    transformer = RatioFeatures(
        numerator_columns=["A", "B"],
        denominator_columns=["B", "A"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "C": [100, 200, 300],
            "A__div__B": [2.0, 4.0, 3.0],
            "B__div__A": [4 / 11, 4 / 21, 9 / 31],
        }
    )

    assert_frame_equal(result, expected)


def test_fit_transform():
    # B=[1,3,9] → denom+1=[2,4,10] → [5.0, 5.0, 3.0]
    X = pl.DataFrame({"A": [10, 20, 30], "B": [1, 3, 9]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame({"A": [10, 20, 30], "B": [1, 3, 9], "A__div__B": [5.0, 5.0, 3.0]})

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
    # A/(B+1) with negative denominators: B+1 may be negative or zero
    X = pl.DataFrame({"A": [-10, 20, -30, 40], "B": [2, -4, 5, -8]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    # B+1 = [3, -3, 6, -7]
    expected = pl.DataFrame(
        {
            "A": [-10, 20, -30, 40],
            "B": [2, -4, 5, -8],
            "A__div__B": [-10 / 3, 20 / -3, -30 / 6, 40 / -7],
        }
    )

    assert_frame_equal(result, expected)


def test_float_values():
    # A/(B+1) with float denominators
    X = pl.DataFrame({"A": [10.5, 20.2, 30.8], "B": [2.1, 4.04, 5.0]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10.5, 20.2, 30.8],
            "B": [2.1, 4.04, 5.0],
            "A__div__B": [10.5 / 3.1, 20.2 / 5.04, 30.8 / 6.0],
        }
    )

    assert_frame_equal(result, expected)


def test_zero_numerator():
    # Zero numerator → 0/(denom+1) = 0.0; B=[1,4,9] → denom+1=[2,5,10]
    X = pl.DataFrame({"A": [0, 0, 10], "B": [1, 4, 9]})

    transformer = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame({"A": [0, 0, 10], "B": [1, 4, 9], "A__div__B": [0.0, 0.0, 1.0]})

    assert_frame_equal(result, expected)
