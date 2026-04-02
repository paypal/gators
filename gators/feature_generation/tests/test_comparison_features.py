import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import ComparisonFeatures


def test_transform_greater_than():
    # Test > operator
    X =pl.DataFrame({"A": [10, 20, 30, 40], "B": [15, 10, 30, 35]})

    transformer = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=[">"])
    transformer.fit(X)
    result = transformer.transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 30, 40],
            "B": [15, 10, 30, 35],
            "A_gt_B": [False, True, False, True],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_less_than():
    # Test < operator
    X =pl.DataFrame({"A": [10, 20, 30, 40], "B": [15, 10, 30, 35]})

    transformer = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=["<"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 30, 40],
            "B": [15, 10, 30, 35],
            "A_lt_B": [True, False, False, False],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_equals():
    # Test == operator
    X =pl.DataFrame({"A": [10, 20, 30, 40], "B": [10, 10, 30, 35]})

    transformer = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=["=="])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 30, 40],
            "B": [10, 10, 30, 35],
            "A_eq_B": [True, False, True, False],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_multiple_comparisons():
    # Test multiple comparison features
    X =pl.DataFrame(
        {"A": [10, 20, 30, 40], "B": [15, 10, 30, 35], "C": [5, 25, 20, 50]}
    )

    transformer = ComparisonFeatures(
        subset_a=["A", "B", "A"], subset_b=["B", "C", "C"], operators=[">", "<", ">="]
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 30, 40],
            "B": [15, 10, 30, 35],
            "C": [5, 25, 20, 50],
            "A_gt_B": [False, True, False, True],
            "B_lt_C": [False, True, False, True],
            "A_gte_C": [True, False, True, False],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_with_drop_columns():
    # Test dropping original columns
    X =pl.DataFrame(
        {"A": [10, 20, 30, 40], "B": [15, 10, 30, 35], "C": [100, 200, 300, 400]}
    )

    transformer = ComparisonFeatures(
        subset_a=["A"], subset_b=["B"], operators=[">"], drop_columns=True
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {"C": [100, 200, 300, 400], "A_gt_B": [False, True, False, True]}
    )

    assert_frame_equal(result, expected)


def test_transform_greater_than_or_equal():
    # Test >= operator
    X =pl.DataFrame({"A": [10, 20, 30, 40], "B": [10, 10, 30, 35]})

    transformer = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=[">="])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 30, 40],
            "B": [10, 10, 30, 35],
            "A_gte_B": [True, True, True, True],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_less_than_or_equal():
    # Test <= operator
    X =pl.DataFrame({"A": [10, 20, 30, 40], "B": [10, 10, 30, 35]})

    transformer = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=["<="])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 30, 40],
            "B": [10, 10, 30, 35],
            "A_lte_B": [True, False, True, False],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_not_equals():
    # Test != operator
    X =pl.DataFrame({"A": [10, 20, 30, 40], "B": [10, 10, 30, 35]})

    transformer = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=["!="])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 20, 30, 40],
            "B": [10, 10, 30, 35],
            "A_ne_B": [False, True, False, True],
        }
    )

    assert_frame_equal(result, expected)


def test_validation_length_mismatch_columns():
    # Test validation error when column lengths don't match
    with pytest.raises(ValidationError, match="must match"):
        ComparisonFeatures(subset_a=["A", "B"], subset_b=["C"], operators=[">"])


def test_validation_length_mismatch_operators():
    # Test validation error when operator length doesn't match
    with pytest.raises(ValidationError, match="must match"):
        ComparisonFeatures(subset_a=["A", "B"], subset_b=["C", "D"], operators=[">"])


def test_validation_invalid_operator():
    # Test validation error for invalid operator
    with pytest.raises(ValidationError, match="Input should be"):
        ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=["invalid"])


def test_transform_is_null():
    # Test is_null unary operator
    X =pl.DataFrame(
        {"A": [10, None, 30, None], "B": [15, 10, None, 35], "C": [100, 200, 300, 400]}
    )

    transformer = ComparisonFeatures(
        subset_a=["A", "B"],
        subset_b=["", ""],  # Ignored for unary operators
        operators=["is_null", "is_null"],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, None, 30, None],
            "B": [15, 10, None, 35],
            "C": [100, 200, 300, 400],
            "A__is_null": [False, True, False, True],
            "B__is_null": [False, False, True, False],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_is_not_null():
    # Test is_not_null unary operator
    X =pl.DataFrame({"A": [10, None, 30, None], "B": [15, 10, None, 35]})

    transformer = ComparisonFeatures(
        subset_a=["A", "B"],
        subset_b=["", ""],  # Ignored for unary operators
        operators=["is_not_null", "is_not_null"],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, None, 30, None],
            "B": [15, 10, None, 35],
            "A__is_not_null": [True, False, True, False],
            "B__is_not_null": [True, True, False, True],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_mixed_operators():
    # Test mixing binary and unary operators
    X =pl.DataFrame(
        {"A": [10, None, 30, 40], "B": [15, 10, 30, None], "C": [5, 25, 20, 50]}
    )

    transformer = ComparisonFeatures(
        subset_a=["A", "B", "A"],
        subset_b=["B", "", "C"],  # Second is ignored for unary operator
        operators=[">", "is_null", "<="],
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, None, 30, 40],
            "B": [15, 10, 30, None],
            "C": [5, 25, 20, 50],
            "A_gt_B": [False, None, False, None],
            "B__is_null": [False, False, False, True],
            "A_lte_C": [False, None, False, True],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_is_null_with_drop_columns():
    # Test is_null with drop_columns=True
    X =pl.DataFrame(
        {"A": [10, None, 30, None], "B": [15, 10, 30, 35], "C": [100, 200, 300, 400]}
    )

    transformer = ComparisonFeatures(
        subset_a=["A"],
        subset_b=[""],  # Ignored for unary operator
        operators=["is_null"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "B": [15, 10, 30, 35],
            "C": [100, 200, 300, 400],
            "A__is_null": [False, True, False, True],
        }
    )

    assert_frame_equal(result, expected)


def test_invalid_operator_raises_error():
    """Test that invalid operator raises ValidationError."""
    from pydantic import ValidationError
    X =pl.DataFrame({"A": [10, 20, 30], "B": [15, 10, 30]})
    
    with pytest.raises(ValidationError, match="literal_error"):
        ComparisonFeatures(
            subset_a=["A"],
            subset_b=["B"],
            operators=["invalid_op"]
        )


if __name__ == "__main__":
    pytest.main()
