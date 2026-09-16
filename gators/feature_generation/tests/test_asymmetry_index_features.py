import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import AsymmetryIndexFeatures
from gators.exceptions import NotFittedError


def test_transform_basic_with_smoothing():
    # x=y=10: (11²)/(11*22) = 121/242 = 0.5
    X = pl.DataFrame({"A": [10, 0, 50], "B": [10, 20, 5]})

    transformer = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 0, 50],
            "B": [10, 20, 5],
            "A__asym__B": [
                (11**2) / (11 * 22),  # 0.5
                (1**2) / (21 * 22),  # 0.002165
                (51**2) / (6 * 57),  # 7.605263
            ],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_equal_values_gives_half():
    # x == y always → 0.5 (with smoothing)
    X = pl.DataFrame({"A": [1, 5, 100], "B": [1, 5, 100]})

    transformer = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"])
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [1, 5, 100],
            "B": [1, 5, 100],
            "A__asym__B": [0.5, 0.5, 0.5],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_no_smoothing():
    # smoothing=False: x=10,y=10 → 10²/(10*20) = 100/200 = 0.5
    # x=50,y=5 → 50²/(5*55) = 2500/275 = 9.0909...
    X = pl.DataFrame({"A": [10, 50], "B": [10, 5]})

    transformer = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"], smoothing=False)
    result = transformer.fit_transform(X)

    expected = pl.DataFrame(
        {
            "A": [10, 50],
            "B": [10, 5],
            "A__asym__B": [
                (10**2) / (10 * 20),  # 0.5
                (50**2) / (5 * 55),  # 9.090909
            ],
        }
    )

    assert_frame_equal(result, expected)


def test_transform_no_smoothing_zero_values_produces_nan():
    # x=0, y=0, no smoothing: 0/(0*0) = NaN
    X = pl.DataFrame({"A": [0, 10], "B": [0, 5]})

    transformer = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"], smoothing=False)
    result = transformer.fit_transform(X)

    # Row 0: 0/(0*0) = NaN; Row 1: 100/(5*15) = 1.333...
    assert result["A__asym__B"][0] != result["A__asym__B"][0]  # NaN check
    assert abs(result["A__asym__B"][1] - (100 / 75)) < 1e-9


def test_transform_multiple_pairs():
    X = pl.DataFrame(
        {
            "views_a": [100.0, 200.0],
            "views_b": [50.0, 200.0],
            "sales_a": [10.0, 30.0],
            "sales_b": [40.0, 30.0],
        }
    )

    transformer = AsymmetryIndexFeatures(
        x_columns=["views_a", "sales_a"],
        y_columns=["views_b", "sales_b"],
    )
    result = transformer.fit_transform(X)

    assert "views_a__asym__views_b" in result.columns
    assert "sales_a__asym__sales_b" in result.columns
    assert result.shape == (2, 6)


def test_transform_custom_column_names():
    X = pl.DataFrame({"A": [10, 20], "B": [10, 5]})

    transformer = AsymmetryIndexFeatures(
        x_columns=["A"], y_columns=["B"], new_column_names=["click_asym"]
    )
    result = transformer.fit_transform(X)

    assert "click_asym" in result.columns
    assert "A__asym__B" not in result.columns


def test_transform_drop_columns():
    X = pl.DataFrame({"A": [10, 20], "B": [10, 5], "C": [1, 2]})

    transformer = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"], drop_columns=True)
    result = transformer.fit_transform(X)

    assert "A" not in result.columns
    assert "B" not in result.columns
    assert "C" in result.columns
    assert "A__asym__B" in result.columns


def test_transform_drop_columns_with_overlap():
    # When x and y share a column name, it should still be dropped only once
    X = pl.DataFrame({"A": [10, 20], "B": [5, 10], "C": [1, 2]})

    transformer = AsymmetryIndexFeatures(
        x_columns=["A", "B"],
        y_columns=["B", "A"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    assert "A" not in result.columns
    assert "B" not in result.columns
    assert "C" in result.columns


def test_transform_without_fit_returns_x_unchanged():
    """transform() before fit() raises NotFittedError."""


def test_transform_without_fit_returns_x_unchanged():
    X = pl.DataFrame({"A": [10, 20], "B": [10, 5]})
    transformer = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"])
    with pytest.raises(NotFittedError):
        transformer.transform(X)


def test_fit_with_custom_names_does_not_overwrite():
    # When new_column_names is provided at construction, fit() must not overwrite it
    X = pl.DataFrame({"A": [10, 20], "B": [10, 5]})

    transformer = AsymmetryIndexFeatures(
        x_columns=["A"], y_columns=["B"], new_column_names=["my_asym"]
    )
    transformer.fit(X)

    assert transformer.new_column_names == ["my_asym"]
    result = transformer.transform(X)
    assert "my_asym" in result.columns


def test_fit_transform_pipeline():
    X = pl.DataFrame({"A": [10, 20, 30], "B": [10, 5, 30]})

    transformer = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"])
    result = transformer.fit_transform(X)

    assert result.shape == (3, 3)
    assert "A__asym__B" in result.columns


def test_length_mismatch_x_y_columns_raises():
    with pytest.raises(ValidationError):
        AsymmetryIndexFeatures(x_columns=["A", "B"], y_columns=["C"])


def test_length_mismatch_new_column_names_raises():
    with pytest.raises(ValidationError):
        AsymmetryIndexFeatures(
            x_columns=["A", "B"],
            y_columns=["C", "D"],
            new_column_names=["only_one"],
        )
