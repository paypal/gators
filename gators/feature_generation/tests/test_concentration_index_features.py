import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import ConcentrationIndexFeatures

# ---------------------------------------------------------------------------
# Basic transform
# ---------------------------------------------------------------------------


def test_transform_basic_smoothing():
    # denom_sum = B+C = [10,20,0], +1 => [11,21,1]
    # result = A / denom_sum = [10/11, 20/21, 0/1]
    X = pl.DataFrame({"A": [10, 20, 0], "B": [5, 10, 0], "C": [5, 10, 0]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 0],
            "B": [5, 10, 0],
            "C": [5, 10, 0],
            "A__conc__B__C": [10 / 11, 20 / 21, 0.0],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_no_smoothing():
    # denom_sum = B+C = [10,0]; no smoothing → [10/10, 0/0] = [1.0, NaN]
    X = pl.DataFrame({"A": [10, 0], "B": [5, 0], "C": [5, 0]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
        smoothing=False,
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 0],
            "B": [5, 0],
            "C": [5, 0],
            "A__conc__B__C": [1.0, float("nan")],
        }
    )
    assert_frame_equal(result, expected, check_exact=False)


def test_transform_zero_denominators_smoothed():
    # All denominators are zero; smoothing prevents division-by-zero
    X = pl.DataFrame({"A": [10, 20], "B": [0, 0], "C": [0, 0]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
        smoothing=True,
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20],
            "B": [0, 0],
            "C": [0, 0],
            "A__conc__B__C": [10.0, 20.0],
        }
    )
    assert_frame_equal(result, expected)


def test_transform_single_denominator_column():
    # Single denominator: semantically same as RatioFeatures but via ConcentrationIndex
    # denom_sum = B = [1,3,9], +1 => [2,4,10]
    X = pl.DataFrame({"A": [10, 20, 30], "B": [1, 3, 9]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B"]],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 30],
            "B": [1, 3, 9],
            "A__conc__B": [5.0, 5.0, 3.0],
        }
    )
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Multiple concentration indices
# ---------------------------------------------------------------------------


def test_transform_multiple_indices():
    # conc_a = num_a / (d_a1 + d_a2 + 1)
    # conc_b = num_b / (d_b1 + d_b2 + 1)
    X = pl.DataFrame(
        {
            "num_a": [10, 20],
            "num_b": [5, 15],
            "d_a1": [3, 4],
            "d_a2": [7, 16],
            "d_b1": [4, 9],
            "d_b2": [6, 6],
        }
    )

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["num_a", "num_b"],
        denominator_columns=[["d_a1", "d_a2"], ["d_b1", "d_b2"]],
    )
    result = transformer.fit_transform(X)

    # conc_a: [10/11, 20/21]
    # conc_b: [5/11, 15/16]
    expected = pl.DataFrame(
        {
            "num_a": [10, 20],
            "num_b": [5, 15],
            "d_a1": [3, 4],
            "d_a2": [7, 16],
            "d_b1": [4, 9],
            "d_b2": [6, 6],
            "num_a__conc__d_a1__d_a2": [10 / 11, 20 / 21],
            "num_b__conc__d_b1__d_b2": [5 / 11, 15 / 16],
        }
    )
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Null propagation
# ---------------------------------------------------------------------------


def test_transform_null_propagation():
    # Null numerator propagates. Null denominators are treated as 0 by
    # sum_horizontal, so row 3 (B=null, C=null) has denom_sum=0 → +1=1 → 30/1=30.0
    X = pl.DataFrame({"A": [10, None, 30], "B": [5, 5, None], "C": [5, 5, None]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
    )
    result = transformer.fit_transform(X)

    # Row 1: 10 / (5+5+1) = 10/11
    # Row 2: null numerator → null
    # Row 3: B=null, C=null → sum_horizontal=0, +1=1 → 30/1=30.0
    expected = pl.DataFrame(
        {
            "A": [10, None, 30],
            "B": [5, 5, None],
            "C": [5, 5, None],
            "A__conc__B__C": [10 / 11, None, 30.0],
        }
    )
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Custom column names
# ---------------------------------------------------------------------------


def test_transform_custom_column_names():
    X = pl.DataFrame({"A": [10, 20], "B": [5, 10], "C": [5, 10]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
        new_column_names=["my_share"],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20],
            "B": [5, 10],
            "C": [5, 10],
            "my_share": [10 / 11, 20 / 21],
        }
    )
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# drop_columns
# ---------------------------------------------------------------------------


def test_transform_drop_columns():
    X = pl.DataFrame({"A": [10, 20], "B": [5, 10], "C": [5, 10], "other": [1, 2]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    # A, B, C dropped; 'other' and the new column remain
    assert "A" not in result.columns
    assert "B" not in result.columns
    assert "C" not in result.columns
    assert "other" in result.columns
    assert "A__conc__B__C" in result.columns
    assert_frame_equal(
        result.select("A__conc__B__C"),
        pl.DataFrame({"A__conc__B__C": [10 / 11, 20 / 21]}),
    )


def test_transform_drop_columns_overlapping_denoms():
    # When a column appears as a denominator for multiple numerators it must not
    # be double-dropped
    X = pl.DataFrame({"A": [10], "B": [5], "C": [5], "D": [3]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C", "D"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    assert result.columns == ["A__conc__B__C__D"]


# ---------------------------------------------------------------------------
# Column name auto-generation
# ---------------------------------------------------------------------------


def test_default_column_name_multiple_denoms():
    X = pl.DataFrame({"N": [1], "X": [2], "Y": [3], "Z": [4]})

    transformer = ConcentrationIndexFeatures(
        numerator_columns=["N"],
        denominator_columns=[["X", "Y", "Z"]],
    )
    result = transformer.fit_transform(X)

    assert "N__conc__X__Y__Z" in result.columns


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------


def test_get_params():
    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
        smoothing=False,
    )
    params = transformer.get_params()

    assert params == {
        "numerator_columns": ["A"],
        "denominator_columns": [["B", "C"]],
        "smoothing": False,
        "new_column_names": None,
        "drop_columns": False,
    }


def test_set_params():
    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B"]],
    )
    transformer.set_params(smoothing=False, drop_columns=True)

    assert transformer.smoothing is False
    assert transformer.drop_columns is True


def test_fit_returns_self():
    X = pl.DataFrame({"A": [1], "B": [2]})
    transformer = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B"]],
    )
    assert transformer.fit(X) is transformer


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_numerator_denominator_length_mismatch():
    with pytest.raises(ValidationError, match="Length of numerator_columns"):
        ConcentrationIndexFeatures(
            numerator_columns=["A", "B"],
            denominator_columns=[["C"]],
        )


def test_validation_empty_denominator_group():
    with pytest.raises(ValidationError, match="at least one column name"):
        ConcentrationIndexFeatures(
            numerator_columns=["A"],
            denominator_columns=[[]],
        )


def test_validation_new_column_names_length_mismatch():
    with pytest.raises(ValidationError, match="Length of new_column_names"):
        ConcentrationIndexFeatures(
            numerator_columns=["A"],
            denominator_columns=[["B"]],
            new_column_names=["x", "y"],
        )


def test_validation_positional_args_rejected():
    with pytest.raises(TypeError):
        ConcentrationIndexFeatures(["A"], [["B"]])
