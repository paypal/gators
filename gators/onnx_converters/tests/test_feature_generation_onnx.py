from __future__ import annotations

import math
from itertools import combinations_with_replacement

import numpy as np
import polars as pl
import pytest

from gators.feature_generation import (
    AsymmetryIndexFeatures,
    ComparisonFeatures,
    ConcentrationIndexFeatures,
    ConditionFeatures,
    DistanceFeatures,
    FourierFeatures,
    GeneralizedRatioFeatures,
    IsNull,
    MathFeatures,
    PlanRotationFeatures,
    PolynomialFeatures,
    RatioFeatures,
    RowStatisticsFeatures,
    ScalarMathFeatures,
    WeightedSumFeatures,
)
from gators.onnx_converters import to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({"A": [1.0, 2.0, 4.0], "B": [3.0, 4.0, 0.0]})


# ── ScalarMathFeatures ────────────────────────────────────────────────────────

@pytest.mark.parametrize("op,scalar", [("+", 10.0), ("-", 5.0), ("*", 2.5), ("/", 4.0), ("**", 2.0)])
def test_scalar_math_basic(df, op, scalar):
    t = ScalarMathFeatures(operations=[{"column": "A", "op": op, "scalar": scalar}])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_scalar_math_floor_div(df):
    t = ScalarMathFeatures(operations=[{"column": "A", "op": "//", "scalar": 3.0}])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_scalar_math_modulo(df):
    t = ScalarMathFeatures(operations=[{"column": "A", "op": "%", "scalar": 3.0}])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_scalar_math_custom_names(df):
    t = ScalarMathFeatures(
        operations=[{"column": "A", "op": "/", "scalar": 2.0}],
        new_column_names=["A_half"],
    )
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert "A_half" in onnx_out


def test_scalar_math_passthrough(df):
    """Column B (not in operations) must pass through unchanged."""
    t = ScalarMathFeatures(operations=[{"column": "A", "op": "*", "scalar": 2.0}])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    np.testing.assert_allclose(onnx_out["B"].astype(float), df["B"].to_numpy(allow_copy=True), atol=1e-5)


def test_scalar_math_output_onnx_type(df):
    """get_output_onnx_type inherits the SOURCE column's onnx type — exercised via pipeline_to_onnx."""
    from gators.onnx_converters import get_output_onnx_type, pipeline_to_onnx
    from gators.pipeline import Pipeline

    t = ScalarMathFeatures(operations=[{"column": "A", "op": "*", "scalar": 2.0}], new_column_names=["A_double"])
    pipe = Pipeline(steps=[("smf", t)])
    pipe.fit(df)
    pipeline_to_onnx(pipe)
    assert get_output_onnx_type(t, "A_double") == get_output_onnx_type(t, "A")
    assert get_output_onnx_type(t, "B") == get_output_onnx_type(t, "B")


# ── IsNull ────────────────────────────────────────────────────────────────────

def test_isnull_values():
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [None, 2.0, 3.0]})
    t = IsNull(subset=["A", "B"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-7)


def test_isnull_no_nulls(df):
    t = IsNull(subset=["A"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    np.testing.assert_allclose(onnx_out["A__is_null"], np.zeros(len(df), dtype=np.float32), atol=1e-7)


# ── RatioFeatures ─────────────────────────────────────────────────────────────

def test_ratio_values(df):
    t = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_ratio_drop_columns(df):
    t = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"], drop_columns=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert "A" not in onnx_out
    assert "B" not in onnx_out
    assert "A__div__B" in onnx_out


def test_ratio_output_onnx_type(df):
    """get_output_onnx_type inherits the DENOMINATOR column's onnx type — via pipeline_to_onnx."""
    from gators.onnx_converters import get_output_onnx_type, pipeline_to_onnx
    from gators.pipeline import Pipeline

    t = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    pipe = Pipeline(steps=[("ratio", t)])
    pipe.fit(df)
    pipeline_to_onnx(pipe)
    assert get_output_onnx_type(t, "A__div__B") == get_output_onnx_type(t, "B")
    assert get_output_onnx_type(t, "A") == get_output_onnx_type(t, "A")


def test_ratio_mixed_dtype_cast():
    """Numerator (Int64) and denominator (Float64) differ in onnx type -> numerator is Cast
    to match the denominator's type before the Div node."""
    X = pl.DataFrame({
        "A": pl.Series([2, 4, 8], dtype=pl.Int64),
        "B": pl.Series([1.0, 2.0, 4.0], dtype=pl.Float64),
    })
    t = RatioFeatures(numerator_columns=["A"], denominator_columns=["B"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    np.testing.assert_allclose(onnx_out["A__div__B"], expected["A__div__B"].to_numpy(allow_copy=True), atol=1e-5)


# ── WeightedSumFeatures ───────────────────────────────────────────────────────

def test_weighted_sum_values(df):
    t = WeightedSumFeatures(column_groups=[["A", "B"]], coefficients=[[2.0, 3.0]], biases=[1.0])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_weighted_sum_multiple_groups(df):
    t = WeightedSumFeatures(
        column_groups=[["A", "B"], ["A"]],
        coefficients=[[1.5, 0.5], [2.0]],
        biases=[0.0, -1.0],
    )
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


# ── AsymmetryIndexFeatures ────────────────────────────────────────────────────

def test_asymmetry_smoothing(df):
    t = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"], smoothing=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_asymmetry_no_smoothing():
    X = pl.DataFrame({"A": [2.0, 3.0, 4.0], "B": [1.0, 2.0, 3.0]})
    t = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"], smoothing=False)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


# ── FourierFeatures ───────────────────────────────────────────────────────────

def test_fourier_sin_cos():
    X = pl.DataFrame({"day": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    t = FourierFeatures(subset=["day"], periods=[7.0], n_harmonics=1)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_fourier_multiple_harmonics():
    X = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0]})
    t = FourierFeatures(subset=["t"], periods=[4.0], n_harmonics=2)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_fourier_drop_columns():
    X = pl.DataFrame({"day": [0.0, 1.0, 2.0], "val": [10.0, 20.0, 30.0]})
    t = FourierFeatures(subset=["day"], periods=[7.0], drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "day" not in onnx_out
    assert "val" in onnx_out


# ── PolynomialFeatures ────────────────────────────────────────────────────────

def test_polynomial_degree2(df):
    t = PolynomialFeatures(subset=["A", "B"], degree=2)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_polynomial_interaction_only(df):
    t = PolynomialFeatures(subset=["A", "B"], degree=2, interaction_only=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)
    assert "A__A" not in onnx_out
    assert "B__B" not in onnx_out
    assert "A__B" in onnx_out


def test_polynomial_include_bias(df):
    t = PolynomialFeatures(subset=["A"], degree=2, include_bias=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    np.testing.assert_allclose(onnx_out["bias"], np.ones(len(df), dtype=np.float32), atol=1e-5)


# ── ComparisonFeatures ────────────────────────────────────────────────────────

@pytest.mark.parametrize("op,onnx_key", [
    (">", "A_gt_B"), ("<", "A_lt_B"), (">=", "A_gte_B"),
    ("<=", "A_lte_B"), ("==", "A_eq_B"), ("!=", "A_ne_B"),
])
def test_comparison_binary(df, op, onnx_key):
    t = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=[op])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-7)


def test_comparison_is_null():
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [1.0, 2.0, 3.0]})
    t = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=["is_null"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-7)


def test_comparison_is_not_null():
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [1.0, 2.0, 3.0]})
    t = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=["is_not_null"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-7)


# ── drop_columns=True branches ────────────────────────────────────────────────

def test_weighted_sum_drop_columns(df):
    t = WeightedSumFeatures(column_groups=[["A", "B"]], drop_columns=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert "A" not in onnx_out
    assert "B" not in onnx_out
    assert "A__B__wsum" in onnx_out


def test_asymmetry_drop_columns():
    X = pl.DataFrame({"A": [2.0, 3.0, 4.0], "B": [1.0, 2.0, 3.0], "C": [5.0, 6.0, 7.0]})
    t = AsymmetryIndexFeatures(x_columns=["A"], y_columns=["B"], drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "A" not in onnx_out
    assert "B" not in onnx_out
    assert "C" in onnx_out


def test_comparison_drop_columns(df):
    t = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=[">"], drop_columns=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert "A" not in onnx_out
    assert "B" not in onnx_out
    assert "A_gt_B" in onnx_out


def test_comparison_output_onnx_type(df):
    """get_output_onnx_type always FLOAT for the comparison result — via pipeline_to_onnx."""
    from onnx import TensorProto
    from gators.onnx_converters import get_output_onnx_type, pipeline_to_onnx
    from gators.pipeline import Pipeline

    t = ComparisonFeatures(subset_a=["A"], subset_b=["B"], operators=[">"])
    pipe = Pipeline(steps=[("cmp", t)])
    pipe.fit(df)
    pipeline_to_onnx(pipe)
    assert get_output_onnx_type(t, "A_gt_B") == TensorProto.FLOAT
    assert get_output_onnx_type(t, "A") == get_output_onnx_type(t, "A")


# ── ConditionFeatures ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("op", [">", "<", ">=", "<=", "==", "!="])
def test_condition_scalar(df, op):
    t = ConditionFeatures(conditions=[{"column": "A", "op": op, "value": 2.0}])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    out_col = t._generated_column_names[0]
    assert_onnx_close(onnx_out, expected.select(["A", "B", out_col]), atol=1e-5)


def test_condition_column_comparison(df):
    t = ConditionFeatures(conditions=[{"column": "A", "op": ">", "other_column": "B"}])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_condition_is_null():
    df2 = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, None]})
    t = ConditionFeatures(conditions=[{"column": "A", "op": "is_null"}])
    t.fit(df2)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df2)
    expected = t.transform(df2)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_condition_is_not_null():
    df2 = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, None]})
    t = ConditionFeatures(conditions=[{"column": "A", "op": "is_not_null"}])
    t.fit(df2)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df2)
    expected = t.transform(df2)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_condition_multiple(df):
    t = ConditionFeatures(
        conditions=[
            {"column": "A", "op": ">=", "value": 2.0},
            {"column": "B", "op": "<", "value": 4.0},
            {"column": "A", "op": ">", "other_column": "B"},
        ],
        new_column_names=["a_ge_2", "b_lt_4", "a_gt_b"],
    )
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_condition_output_onnx_type(df):
    """get_output_onnx_type inherits the SOURCE column's onnx type — via pipeline_to_onnx."""
    from gators.onnx_converters import get_output_onnx_type, pipeline_to_onnx
    from gators.pipeline import Pipeline

    t = ConditionFeatures(conditions=[{"column": "A", "op": ">", "value": 2.0}], new_column_names=["a_gt_2"])
    pipe = Pipeline(steps=[("cond", t)])
    pipe.fit(df)
    pipeline_to_onnx(pipe)
    assert get_output_onnx_type(t, "a_gt_2") == get_output_onnx_type(t, "A")
    assert get_output_onnx_type(t, "B") == get_output_onnx_type(t, "B")


# ── MathFeatures ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "range"])
def test_math_basic_ops(op):
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]})
    t = MathFeatures(groups=[["A", "B", "C"]], func=[op])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_math_std_two_cols():
    X = pl.DataFrame({"A": [1.0, 2.0, 10.0], "B": [3.0, 6.0, 4.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["std"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_math_multiple_groups_and_ops():
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    t = MathFeatures(groups=[["A", "B"], ["B", "C"]], func=["sum", "mean"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_math_drop_columns():
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["sum"], drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "A" not in onnx_out
    assert "B" not in onnx_out
    assert "A_B_sum" in onnx_out


def test_math_custom_names():
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["max"], new_column_names=["ab_group"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "ab_group_max" in onnx_out


def test_math_median_raises():
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["median"])
    t.fit(X)
    with pytest.raises(Exception):
        to_onnx_graph(t, errors="raise")


# ── RowStatisticsFeatures ─────────────────────────────────────────────────────

@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "range", "std", "count"])
def test_rowstats_basic_ops(op):
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]})
    t = RowStatisticsFeatures(column_groups={"grp": ["A", "B", "C"]}, func=[op])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_rowstats_multiple_groups():
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0], "D": [7.0, 8.0]})
    t = RowStatisticsFeatures(
        column_groups={"g1": ["A", "B"], "g2": ["C", "D"]},
        func=["min", "max"],
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_rowstats_drop_columns():
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    t = RowStatisticsFeatures(column_groups={"g": ["A", "B"]}, func=["sum"], drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "A" not in onnx_out
    assert "B" not in onnx_out
    assert "g__sum" in onnx_out


def test_rowstats_custom_names():
    X = pl.DataFrame({"A": [1.0, 9.0], "B": [5.0, 3.0]})
    t = RowStatisticsFeatures(
        column_groups={"g": ["A", "B"]},
        func=["mean", "std"],
        new_column_names=["avg", "dev"],
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


# ── ConcentrationIndexFeatures ────────────────────────────────────────────────

def test_conc_smoothing():
    X = pl.DataFrame({"A": [10.0, 20.0, 0.0], "B": [30.0, 10.0, 0.0], "C": [60.0, 70.0, 0.0]})
    t = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
        smoothing=True,
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_conc_no_smoothing():
    X = pl.DataFrame({"A": [10.0, 20.0], "B": [30.0, 10.0], "C": [60.0, 70.0]})
    t = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
        smoothing=False,
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_conc_multiple_pairs():
    X = pl.DataFrame({"A": [5.0, 10.0], "B": [20.0, 30.0], "C": [15.0, 25.0], "D": [40.0, 50.0]})
    t = ConcentrationIndexFeatures(
        numerator_columns=["A", "C"],
        denominator_columns=[["B", "C"], ["D"]],
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_conc_drop_columns():
    X = pl.DataFrame({"A": [5.0, 10.0], "B": [20.0, 30.0], "C": [15.0, 25.0]})
    t = ConcentrationIndexFeatures(
        numerator_columns=["A"],
        denominator_columns=[["B", "C"]],
        drop_columns=True,
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "A" not in onnx_out


# ── GeneralizedRatioFeatures ──────────────────────────────────────────────────

def test_gratio_equal_weights():
    X = pl.DataFrame({"f1": [5.0, 20.0], "f2": [500.0, 200.0], "f3": [60.0, 100.0], "f4": [6000.0, 4000.0]})
    t = GeneralizedRatioFeatures(
        numerator_columns=[["f1", "f2"]],
        denominator_columns=[["f3", "f4"]],
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_gratio_custom_coefficients():
    X = pl.DataFrame({"f1": [5.0, 10.0], "f2": [100.0, 200.0], "f3": [50.0, 100.0]})
    t = GeneralizedRatioFeatures(
        numerator_columns=[["f1", "f2"]],
        denominator_columns=[["f3"]],
        numerator_coefficients=[[1.0, 2.0]],
        denominator_coefficients=[[0.5]],
        numerator_biases=[1.0],
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_gratio_constant_numerator():
    X = pl.DataFrame({"f3": [60.0, 100.0], "f4": [6000.0, 4000.0]})
    t = GeneralizedRatioFeatures(
        numerator_columns=[[]],
        denominator_columns=[["f3", "f4"]],
        numerator_biases=[1.0],
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_gratio_drop_columns():
    X = pl.DataFrame({"f1": [5.0, 10.0], "f3": [60.0, 100.0]})
    t = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
        drop_columns=True,
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "f1" not in onnx_out
    assert "f3" not in onnx_out


# ── PlanRotationFeatures ──────────────────────────────────────────────────────

def test_planrot_basic():
    X = pl.DataFrame({"X": [200.0, 210.0], "Y": [140.0, 160.0]})
    t = PlanRotationFeatures(columns=[["X", "Y"]], angles=[45.0])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_planrot_multiple_angles():
    X = pl.DataFrame({"X": [200.0, 210.0], "Y": [140.0, 160.0]})
    t = PlanRotationFeatures(columns=[["X", "Y"]], angles=[30.0, 45.0, 90.0])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_planrot_multiple_pairs():
    X = pl.DataFrame({"X": [200.0, 210.0], "Y": [140.0, 160.0], "Z": [100.0, 125.0]})
    t = PlanRotationFeatures(columns=[["X", "Y"], ["X", "Z"]], angles=[45.0, 60.0])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_planrot_passthrough():
    """Original columns must survive in the output."""
    X = pl.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
    t = PlanRotationFeatures(columns=[["X", "Y"]], angles=[45.0])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out["X"].astype(float), X["X"].to_numpy(allow_copy=True), atol=1e-5)
    np.testing.assert_allclose(onnx_out["Y"].astype(float), X["Y"].to_numpy(allow_copy=True), atol=1e-5)


# ── DistanceFeatures ──────────────────────────────────────────────────────────

def test_dist_euclidean():
    X = pl.DataFrame({"lat1": [0.0, 1.0], "lat2": [3.0, 4.0], "lon1": [0.0, 1.0], "lon2": [4.0, 5.0]})
    t = DistanceFeatures(lats=["lat1", "lat2"], longs=["lon1", "lon2"], method="euclidean", drop_columns=False)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_dist_manhattan():
    X = pl.DataFrame({"lat1": [0.0, 1.0], "lat2": [3.0, 4.0], "lon1": [0.0, 1.0], "lon2": [4.0, 5.0]})
    t = DistanceFeatures(lats=["lat1", "lat2"], longs=["lon1", "lon2"], method="manhattan", drop_columns=False)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_dist_haversine():
    X = pl.DataFrame({
        "billing_lat":  [40.7128, 34.0522],
        "shipping_lat": [40.7580, 34.0522],
        "billing_lon":  [-74.0060, -118.2437],
        "shipping_lon": [-73.9855, -118.2437],
    })
    t = DistanceFeatures(
        lats=["billing_lat", "shipping_lat"],
        longs=["billing_lon", "shipping_lon"],
        method="haversine",
        unit="km",
        drop_columns=False,
    )
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-2)


def test_dist_haversine_units():
    X = pl.DataFrame({
        "lat1": [40.7128, 34.0522],
        "lat2": [40.7580, 34.1000],
        "lon1": [-74.0060, -118.2437],
        "lon2": [-73.9855, -118.3000],
    })
    for unit in ("km", "miles", "meters"):
        t = DistanceFeatures(lats=["lat1", "lat2"], longs=["lon1", "lon2"],
                             method="haversine", unit=unit, drop_columns=False)
        t.fit(X)
        model = to_onnx_graph(t)
        onnx_out = run_onnx(model, X)
        expected = t.transform(X)
        assert_onnx_close(onnx_out, expected, atol=1e-2)


def test_dist_drop_columns():
    X = pl.DataFrame({"lat1": [0.0, 1.0], "lat2": [3.0, 4.0], "lon1": [0.0, 1.0], "lon2": [4.0, 5.0]})
    t = DistanceFeatures(lats=["lat1", "lat2"], longs=["lon1", "lon2"], method="euclidean", drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "lat1" not in onnx_out
    assert "lon1" not in onnx_out


# ── RuleFeatures ──────────────────────────────────────────────────────────────

from gators.feature_generation import RuleFeatures


@pytest.fixture
def rule_df():
    return pl.DataFrame({
        "amount": [100.0, 500.0, 1200.0, 50.0, 2000.0],
        "velocity": [1.0, 3.0, 5.0, 0.0, 10.0],
        "score": [0.1, 0.5, 0.9, 0.2, 0.8],
    })


def test_rule_single_condition_scalar(rule_df):
    t = RuleFeatures(rules=[[{"column": "amount", "op": ">", "value": 1000}]], new_column_names=["high"], drop_conditions=True)
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    expected = t.transform(rule_df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_rule_and_multiple_conditions(rule_df):
    t = RuleFeatures(
        rules=[[
            {"column": "amount", "op": ">", "value": 500},
            {"column": "velocity", "op": ">=", "value": 5},
        ]],
        rule_logic="and",
        new_column_names=["high_risk"],
        drop_conditions=True,
    )
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    expected = t.transform(rule_df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_rule_or_logic(rule_df):
    t = RuleFeatures(
        rules=[[
            {"column": "amount", "op": ">", "value": 1000},
            {"column": "velocity", "op": ">=", "value": 5},
        ]],
        rule_logic="or",
        new_column_names=["either_high"],
        drop_conditions=True,
    )
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    expected = t.transform(rule_df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_rule_multiple_rules(rule_df):
    t = RuleFeatures(
        rules=[
            [{"column": "amount", "op": ">", "value": 1000}],
            [{"column": "velocity", "op": ">=", "value": 5}],
        ],
        new_column_names=["high_amount", "high_velocity"],
        drop_conditions=True,
    )
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    expected = t.transform(rule_df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_rule_column_comparison(rule_df):
    t = RuleFeatures(
        rules=[[{"column": "amount", "op": ">", "other_column": "velocity"}]],
        new_column_names=["amount_gt_velocity"],
        drop_conditions=True,
    )
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    expected = t.transform(rule_df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


@pytest.mark.parametrize("op", [">", "<", ">=", "<=", "==", "!="])
def test_rule_all_scalar_ops(rule_df, op):
    t = RuleFeatures(
        rules=[[{"column": "amount", "op": op, "value": 500.0}]],
        new_column_names=["result"],
        drop_conditions=True,
    )
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    expected = t.transform(rule_df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_rule_three_conditions_and(rule_df):
    t = RuleFeatures(
        rules=[[
            {"column": "amount", "op": ">", "value": 100},
            {"column": "velocity", "op": ">", "value": 2},
            {"column": "score", "op": ">", "value": 0.4},
        ]],
        rule_logic="and",
        new_column_names=["triple_and"],
        drop_conditions=True,
    )
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    expected = t.transform(rule_df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_rule_passthrough(rule_df):
    """Original columns must survive unchanged."""
    t = RuleFeatures(rules=[[{"column": "amount", "op": ">", "value": 1000}]], new_column_names=["flag"])
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    import numpy as np
    np.testing.assert_allclose(onnx_out["amount"].astype(float), rule_df["amount"].to_numpy(allow_copy=True), atol=1e-5)


# ── coverage gap tests ────────────────────────────────────────────────────────

from gators.feature_generation import HHIFeatures


# FourierFeatures: float32 input hits the simple Cos path (no cast bridge)
def test_fourier_float32():
    X = pl.DataFrame({"A": pl.Series([1.0, 2.0, 4.0], dtype=pl.Float32)})
    t = FourierFeatures(subset=["A"], periods=[4.0], n_harmonics=1)
    t.fit(X)
    model = to_onnx_graph(t, float_datatype="float32")
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


# ConditionFeatures: column-to-column != comparison
def test_condition_column_ne(df):
    t = ConditionFeatures(
        conditions=[{"column": "A", "op": "!=", "other_column": "B"}],
        new_column_names=["a_ne_b"],
    )
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


# HHIFeatures
def test_hhi_basic():
    X = pl.DataFrame({"A": [10.0, 20.0, 30.0], "B": [30.0, 10.0, 20.0], "C": [1.0, 2.0, 3.0]})
    t = HHIFeatures(column_groups=[["A", "B"]], new_column_names=["hhi_ab"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_hhi_drop_columns():
    X = pl.DataFrame({"A": [10.0, 20.0], "B": [30.0, 40.0]})
    t = HHIFeatures(column_groups=[["A", "B"]], new_column_names=["hhi"], drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "A" not in onnx_out
    assert "B" not in onnx_out
    assert "hhi" in onnx_out


# EntropyFeatures
from gators.feature_generation import EntropyFeatures


def test_entropy_basic():
    X = pl.DataFrame({"A": [10.0, 20.0, 0.0], "B": [30.0, 10.0, 0.0], "C": [60.0, 70.0, 0.0]})
    t = EntropyFeatures(column_groups=[["A", "B", "C"]], new_column_names=["entropy_abc"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_entropy_drop_columns():
    X = pl.DataFrame({"A": [10.0, 20.0], "B": [30.0, 40.0]})
    t = EntropyFeatures(column_groups=[["A", "B"]], new_column_names=["entropy"], drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "A" not in onnx_out
    assert "B" not in onnx_out
    assert "entropy" in onnx_out


def test_entropy_multiple_groups():
    X = pl.DataFrame({"a1": [10.0, 50.0], "a2": [90.0, 50.0], "b1": [25.0, 25.0], "b2": [25.0, 75.0]})
    t = EntropyFeatures(column_groups=[["a1", "a2"], ["b1", "b2"]])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_entropy_passthrough_ungrouped_column():
    """A column not part of any group must survive unchanged (Identity passthrough)."""
    X = pl.DataFrame({"A": [10.0, 20.0, 0.0], "B": [30.0, 10.0, 0.0], "C": [1.0, 2.0, 3.0]})
    t = EntropyFeatures(column_groups=[["A", "B"]], new_column_names=["entropy_ab"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


# MathFeatures: reduce ops (minus, mul, div, abs_diff)
@pytest.mark.parametrize("op", ["minus", "mul", "div", "abs_diff"])
def test_math_reduce_ops(op):
    X = pl.DataFrame({"A": [4.0, 8.0, 12.0], "B": [2.0, 2.0, 3.0]})
    t = MathFeatures(groups=[["A", "B"]], func=[op])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


# MathFeatures: var op (2 cols → Identity of var_out, not Sqrt)
def test_math_var():
    X = pl.DataFrame({"A": [1.0, 2.0, 10.0], "B": [3.0, 6.0, 4.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["var"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


# MathFeatures: counting ops (Polars returns list columns; compare against known values)
def test_math_count_null():
    X = pl.DataFrame({"A": [0.0, 1.0, 2.0], "B": [0.0, 0.0, 3.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["count_null"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    # X has no nulls → all rows are 0
    np.testing.assert_allclose(onnx_out["A_B_count_null"].astype(float), np.zeros(3), atol=1e-5)


def test_math_count_zero():
    X = pl.DataFrame({"A": [0.0, 1.0, 2.0], "B": [0.0, 0.0, 3.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["count_zero"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out["A_B_count_zero"].astype(float), [2.0, 1.0, 0.0], atol=1e-5)


def test_math_count_nonzero():
    X = pl.DataFrame({"A": [0.0, 1.0, 2.0], "B": [0.0, 0.0, 3.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["count_nonzero"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out["A_B_count_nonzero"].astype(float), [0.0, 1.0, 2.0], atol=1e-5)


# MathFeatures: n==1 identity paths for mean/min/max/range/minus/mul/div/abs_diff
@pytest.mark.parametrize("op", ["mean", "min", "max", "range", "minus", "mul", "div", "abs_diff"])
def test_math_single_col_passthrough(op):
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    t = MathFeatures(groups=[["A"]], func=[op])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


# MathFeatures: std/var n==1 → ONNX produces 0 (col * 0); just check it runs
@pytest.mark.parametrize("op", ["std", "var"])
def test_math_std_var_single_col(op):
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    t = MathFeatures(groups=[["A"]], func=[op])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out[f"A_{op}"], np.zeros(3, dtype=np.float32), atol=1e-7)


# DistanceFeatures: haversine with float32 input hits the simple Cos/Asin path
def test_dist_haversine_float32():
    X = pl.DataFrame({
        "lat1": pl.Series([40.7128, 34.0522], dtype=pl.Float32),
        "lat2": pl.Series([40.7580, 34.0522], dtype=pl.Float32),
        "lon1": pl.Series([-74.0060, -118.2437], dtype=pl.Float32),
        "lon2": pl.Series([-73.9855, -118.2437], dtype=pl.Float32),
    })
    t = DistanceFeatures(lats=["lat1", "lat2"], longs=["lon1", "lon2"],
                         method="haversine", unit="km", drop_columns=False)
    t.fit(X)
    model = to_onnx_graph(t, float_datatype="float32")
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-1)


# RuleFeatures: column-to-column != comparison
def test_rule_column_ne(rule_df):
    t = RuleFeatures(
        rules=[[{"column": "amount", "op": "!=", "other_column": "velocity"}]],
        new_column_names=["ne_flag"],
        drop_conditions=True,
    )
    t.fit(rule_df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, rule_df)
    expected = t.transform(rule_df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


# ── IsNull: string column path (lines 169, 176-178, 198) ─────────────────────

def test_isnull_string_column():
    """IsNull on a string column uses LabelEncoder (lines 198+); input type → STRING (line 169)."""
    X = pl.DataFrame({"A": ["hello", None, "world"], "B": [1.0, None, 3.0]})
    t = IsNull(subset=["A", "B"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    # Check null-indicator columns (float) only; original String column is a passthrough
    np.testing.assert_allclose(onnx_out["A__is_null"], expected["A__is_null"].to_numpy(allow_copy=True), atol=1e-7)
    np.testing.assert_allclose(onnx_out["B__is_null"], expected["B__is_null"].to_numpy(allow_copy=True), atol=1e-7)


def test_isnull_output_onnx_type():
    """get_output_onnx_type for IsNull: indicators → FLOAT, pass-throughs → input type (lines 176-178)."""
    from onnx import TensorProto
    from gators.onnx_converters._converters import get_output_onnx_type

    X = pl.DataFrame({"A": ["a", None, "b"], "B": [1.0, None, 3.0]})
    t = IsNull(subset=["A", "B"])
    t.fit(X)
    assert get_output_onnx_type(t, "A__is_null") == TensorProto.FLOAT   # indicator → FLOAT (line 177)
    assert get_output_onnx_type(t, "B__is_null") == TensorProto.FLOAT
    assert get_output_onnx_type(t, "A") == TensorProto.STRING            # pass-through String (lines 178, 169)
    assert get_output_onnx_type(t, "B") == TensorProto.DOUBLE            # pass-through Float64 → DOUBLE


def test_isnull_integer_column():
    """IsNull on an Int64 column: no null repr once exported, so IsNaN always evaluates False
    via a Cast(FLOAT)->IsNaN fallback (integer columns carry no native ONNX null)."""
    X = pl.DataFrame({"A": pl.Series([1, 2, 3], dtype=pl.Int64)})
    t = IsNull(subset=["A"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["A__is_null"], np.zeros(3, dtype=np.float32))


# ── MathFeatures: Float64 output type and mixed-type cast (lines 963-973, 1002-1004) ─

def test_math_output_type_double():
    """get_output_onnx_type returns DOUBLE for Float64 group; FLOAT for pass-through (lines 963-973)."""
    from onnx import TensorProto
    from gators.onnx_converters._converters import get_output_onnx_type

    X = pl.DataFrame({
        "A": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
        "B": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float64),
    })
    t = MathFeatures(groups=[["A", "B"]], func=["sum"])
    t.fit(X)
    assert get_output_onnx_type(t, "A_B_sum") == TensorProto.DOUBLE   # DOUBLE group → DOUBLE (line 968)
    assert get_output_onnx_type(t, "A") == TensorProto.DOUBLE          # pass-through (line 972)


def test_math_mixed_float32_float64_cast():
    """Float32 column in a Float64 group triggers a Cast node (lines 1002-1004).

    to_onnx_graph/pipeline_to_onnx normalize all float columns to the same
    float_datatype before building nodes, so the mismatched-dtype Cast bridge
    is only reachable by calling to_onnx_nodes directly with the transformer's
    naturally-captured (un-normalized) per-column _input_dtypes.
    """
    from gators.onnx_converters._converters import to_onnx_nodes

    X = pl.DataFrame({
        "A": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
        "B": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float64),
    })
    t = MathFeatures(groups=[["A", "B"]], func=["sum"])
    t.fit(X)
    input_names = {"A": "A__in", "B": "B__in"}
    output_names = {"A_B_sum": "A_B_sum"}
    nodes, _ = to_onnx_nodes(t, input_names, output_names)
    cast_nodes = [n for n in nodes if n.op_type == "Cast"]
    assert len(cast_nodes) == 1


# ── Identity pass-through: non-group columns must be unchanged ────────────────

def test_feature_generation_passthrough_column():
    """Column not in any MathFeatures group passes through the ONNX graph unchanged."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [10.0, 20.0, 30.0]})
    t = MathFeatures(groups=[["A", "B"]], func=["sum"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["C"], X["C"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_feature_gen_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 3
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = MathFeatures(groups=[['A', 'B']], func=['sum'])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))
