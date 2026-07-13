import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import HHIFeatures

EPS = 1e-8


def _hhi(*vals: float) -> float:
    """Reference HHI implementation matching the transformer's epsilon."""
    total = sum(vals) + EPS
    return sum((v / total) ** 2 for v in vals)


# ---------------------------------------------------------------------------
# Basic transform
# ---------------------------------------------------------------------------


def test_transform_single_group():
    # Row 0: [10, 30, 60] → HHI = 0.01 + 0.09 + 0.36 = 0.46
    # Row 1: [20, 10, 70] → HHI = 0.04 + 0.01 + 0.49 = 0.54
    # Row 2: [0, 0, 0]   → all shares ≈ 0 → HHI ≈ 0
    X = pl.DataFrame({"A": [10.0, 20.0, 0.0], "B": [30.0, 10.0, 0.0], "C": [60.0, 70.0, 0.0]})
    transformer = HHIFeatures(column_groups=[["A", "B", "C"]])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10.0, 20.0, 0.0],
            "B": [30.0, 10.0, 0.0],
            "C": [60.0, 70.0, 0.0],
            "A__B__C__hhi": [_hhi(10, 30, 60), _hhi(20, 10, 70), _hhi(0, 0, 0)],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_equal_shares():
    # All equal → HHI = 1/n  (here n=4, so HHI ≈ 0.25)
    X = pl.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0], "D": [10.0]})
    transformer = HHIFeatures(column_groups=[["A", "B", "C", "D"]])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10.0],
            "B": [10.0],
            "C": [10.0],
            "D": [10.0],
            "A__B__C__D__hhi": [_hhi(10, 10, 10, 10)],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_monopoly():
    # One column holds all the value → HHI ≈ 1.0
    X = pl.DataFrame({"A": [100.0], "B": [0.0]})
    transformer = HHIFeatures(column_groups=[["A", "B"]])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [100.0],
            "B": [0.0],
            "A__B__hhi": [_hhi(100, 0)],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_two_column_group():
    X = pl.DataFrame({"X": [1.0, 3.0], "Y": [3.0, 1.0]})
    transformer = HHIFeatures(column_groups=[["X", "Y"]])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "X": [1.0, 3.0],
            "Y": [3.0, 1.0],
            "X__Y__hhi": [_hhi(1, 3), _hhi(3, 1)],
        }
    )
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Multiple groups
# ---------------------------------------------------------------------------


def test_transform_multiple_groups():
    X = pl.DataFrame(
        {
            "a1": [10.0, 50.0],
            "a2": [90.0, 50.0],
            "b1": [25.0, 25.0],
            "b2": [25.0, 75.0],
        }
    )
    transformer = HHIFeatures(column_groups=[["a1", "a2"], ["b1", "b2"]])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "a1": [10.0, 50.0],
            "a2": [90.0, 50.0],
            "b1": [25.0, 25.0],
            "b2": [25.0, 75.0],
            "a1__a2__hhi": [_hhi(10, 90), _hhi(50, 50)],
            "b1__b2__hhi": [_hhi(25, 25), _hhi(25, 75)],
        }
    )
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Custom column names
# ---------------------------------------------------------------------------


def test_transform_custom_column_names():
    X = pl.DataFrame({"A": [10.0], "B": [30.0], "C": [60.0]})
    transformer = HHIFeatures(
        column_groups=[["A", "B", "C"]],
        new_column_names=["market_hhi"],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10.0],
            "B": [30.0],
            "C": [60.0],
            "market_hhi": [_hhi(10, 30, 60)],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_custom_column_names_multiple_groups():
    X = pl.DataFrame({"A": [10.0], "B": [30.0], "C": [5.0], "D": [5.0]})
    transformer = HHIFeatures(
        column_groups=[["A", "B"], ["C", "D"]],
        new_column_names=["hhi_ab", "hhi_cd"],
    )
    result = transformer.fit_transform(X)

    assert "hhi_ab" in result.columns
    assert "hhi_cd" in result.columns


# ---------------------------------------------------------------------------
# drop_columns
# ---------------------------------------------------------------------------


def test_transform_drop_columns_removes_sources():
    X = pl.DataFrame({"A": [10.0], "B": [30.0], "C": [60.0], "other": [1.0]})
    transformer = HHIFeatures(
        column_groups=[["A", "B", "C"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    assert "A" not in result.columns
    assert "B" not in result.columns
    assert "C" not in result.columns
    assert "other" in result.columns
    assert "A__B__C__hhi" in result.columns


def test_transform_drop_columns_shared_across_groups():
    # Column 'B' appears in both groups; it should be dropped once
    X = pl.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
    transformer = HHIFeatures(
        column_groups=[["A", "B"], ["B", "C"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    assert "A" not in result.columns
    assert "B" not in result.columns
    assert "C" not in result.columns
    assert "A__B__hhi" in result.columns
    assert "B__C__hhi" in result.columns


# ---------------------------------------------------------------------------
# Null propagation
# ---------------------------------------------------------------------------


def test_transform_null_in_one_column():
    # Null is treated as 0 by pl.sum_horizontal in the total, but propagates
    # through (null / total)**2 so that term is dropped by pl.sum_horizontal.
    # The resulting HHI reflects only the non-null shares.
    X = pl.DataFrame({"A": [10.0, None], "B": [30.0, 30.0], "C": [60.0, 60.0]})
    transformer = HHIFeatures(column_groups=[["A", "B", "C"]])
    result = transformer.fit_transform(X)

    # Row 0: all present
    hhi_r0 = _hhi(10, 30, 60)
    # Row 1: A=null → total = 0+30+60+eps = 90+eps; (null/total)**2 is null and
    #         dropped by sum_horizontal, so only B and C shares contribute.
    total_r1 = 0.0 + 30.0 + 60.0 + EPS
    hhi_r1 = (30.0 / total_r1) ** 2 + (60.0 / total_r1) ** 2

    expected = pl.DataFrame(
        {
            "A": [10.0, None],
            "B": [30.0, 30.0],
            "C": [60.0, 60.0],
            "A__B__C__hhi": [hhi_r0, hhi_r1],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_all_null_row():
    # All columns null → total = sum_horizontal([null, null]) + eps = 0 + eps = eps.
    # Each share (null / eps)**2 = null.
    # pl.sum_horizontal over all-null terms returns 0.0 (Polars treats nulls as 0).
    X = pl.DataFrame({"A": [None], "B": [None]})
    transformer = HHIFeatures(column_groups=[["A", "B"]])
    result = transformer.fit_transform(X)

    assert result["A__B__hhi"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_group_too_small():
    with pytest.raises(ValidationError, match="at least two"):
        HHIFeatures(column_groups=[["only_one"]])


def test_validation_empty_group():
    with pytest.raises(ValidationError, match="at least two"):
        HHIFeatures(column_groups=[[]])


def test_validation_new_column_names_wrong_length():
    with pytest.raises(ValidationError, match="Length of new_column_names"):
        HHIFeatures(
            column_groups=[["A", "B"], ["C", "D"]],
            new_column_names=["only_one_name"],
        )


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------


def test_get_params():
    transformer = HHIFeatures(
        column_groups=[["A", "B"]],
        epsilon=1e-6,
        drop_columns=True,
    )
    params = transformer.get_params()
    assert params["column_groups"] == [["A", "B"]]
    assert params["epsilon"] == 1e-6
    assert params["drop_columns"] is True


def test_set_params():
    transformer = HHIFeatures(column_groups=[["A", "B"]])
    transformer.set_params(drop_columns=True)
    assert transformer.drop_columns is True


def test_fit_returns_self():
    X = pl.DataFrame({"A": [1.0], "B": [2.0]})
    transformer = HHIFeatures(column_groups=[["A", "B"]])
    assert transformer.fit(X) is transformer
