import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import EntropyFeatures

EPS = 1e-8


def _entropy(*vals: float) -> float:
    """Reference Shannon entropy implementation matching the transformer's epsilon."""
    total = sum(vals) + EPS
    shares = [v / total for v in vals]
    return -sum(s * math.log(s) for s in shares if s > 0)


# ---------------------------------------------------------------------------
# Basic transform
# ---------------------------------------------------------------------------


def test_transform_single_group():
    X = pl.DataFrame({"A": [10.0, 20.0, 0.0], "B": [30.0, 10.0, 0.0], "C": [60.0, 70.0, 0.0]})
    transformer = EntropyFeatures(column_groups=[["A", "B", "C"]])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10.0, 20.0, 0.0],
            "B": [30.0, 10.0, 0.0],
            "C": [60.0, 70.0, 0.0],
            "A__B__C__entropy": [_entropy(10, 30, 60), _entropy(20, 10, 70), _entropy(0, 0, 0)],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_equal_shares_is_max_entropy():
    # All equal shares -> entropy = ln(n)
    X = pl.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0]})
    transformer = EntropyFeatures(column_groups=[["A", "B", "C", "D"]])
    result = transformer.fit_transform(X)
    assert result["A__B__C__D__entropy"][0] == pytest.approx(math.log(4), abs=1e-6)


def test_transform_monopoly_is_zero_entropy():
    # One column holds all the value -> entropy ~= 0
    X = pl.DataFrame({"A": [100.0], "B": [0.0]})
    transformer = EntropyFeatures(column_groups=[["A", "B"]])
    result = transformer.fit_transform(X)
    assert result["A__B__entropy"][0] == pytest.approx(0.0, abs=1e-6)


def test_transform_multiple_groups():
    X = pl.DataFrame(
        {
            "a1": [10.0, 50.0],
            "a2": [90.0, 50.0],
            "b1": [25.0, 25.0],
            "b2": [25.0, 75.0],
        }
    )
    transformer = EntropyFeatures(column_groups=[["a1", "a2"], ["b1", "b2"]])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "a1": [10.0, 50.0],
            "a2": [90.0, 50.0],
            "b1": [25.0, 25.0],
            "b2": [25.0, 75.0],
            "a1__a2__entropy": [_entropy(10, 90), _entropy(50, 50)],
            "b1__b2__entropy": [_entropy(25, 25), _entropy(25, 75)],
        }
    )
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Custom names / drop_columns
# ---------------------------------------------------------------------------


def test_transform_custom_column_names():
    X = pl.DataFrame({"brand_a": [10.0], "brand_b": [30.0], "brand_c": [60.0]})
    transformer = EntropyFeatures(
        column_groups=[["brand_a", "brand_b", "brand_c"]],
        new_column_names=["market_entropy"],
    )
    result = transformer.fit_transform(X)
    assert "market_entropy" in result.columns
    assert result["market_entropy"][0] == pytest.approx(_entropy(10, 30, 60))


def test_transform_drop_columns():
    X = pl.DataFrame({"brand_a": [10.0], "brand_b": [30.0], "brand_c": [60.0]})
    transformer = EntropyFeatures(column_groups=[["brand_a", "brand_b"]], drop_columns=True)
    result = transformer.fit_transform(X)
    assert "brand_a" not in result.columns
    assert "brand_b" not in result.columns
    assert "brand_c" in result.columns


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_group_with_single_column_raises():
    with pytest.raises(ValidationError):
        EntropyFeatures(column_groups=[["A"]])


def test_new_column_names_length_mismatch_raises():
    with pytest.raises(ValidationError):
        EntropyFeatures(
            column_groups=[["A", "B"], ["C", "D"]],
            new_column_names=["only_one"],
        )


def test_output_dtypes_are_float64():
    X = pl.DataFrame({"A": [10.0], "B": [30.0]})
    transformer = EntropyFeatures(column_groups=[["A", "B"]])
    transformer.fit(X)
    assert transformer._output_dtypes == {"A__B__entropy": pl.Float64}
