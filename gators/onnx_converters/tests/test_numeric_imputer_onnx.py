from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from gators.imputers import NumericImputer
from gators.onnx_converters import OnnxNotSupportedError, to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({"A": [1.0, None, 3.0, 4.0], "B": [None, 2.0, 3.0, 4.0]})


# ── strategy: median ──────────────────────────────────────────────────────────

def test_numeric_imputer_median_values(df):
    imputer = NumericImputer(strategy="median")
    imputer.fit(df)
    model = to_onnx_graph(imputer)

    onnx_out = run_onnx(model, df)
    expected = imputer.transform(df)

    assert_onnx_close(onnx_out, expected)


def test_numeric_imputer_median_no_nulls():
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    imputer = NumericImputer(strategy="median")
    imputer.fit(X)
    model = to_onnx_graph(imputer)

    onnx_out = run_onnx(model, X)
    expected = imputer.transform(X)

    assert_onnx_close(onnx_out, expected)


# ── strategy: constant ────────────────────────────────────────────────────────

def test_numeric_imputer_constant(df):
    imputer = NumericImputer(strategy="constant", value=-99.0)
    imputer.fit(df)
    model = to_onnx_graph(imputer)

    onnx_out = run_onnx(model, df)
    expected = imputer.transform(df)

    assert_onnx_close(onnx_out, expected)


def test_numeric_imputer_constant_zero(df):
    imputer = NumericImputer(strategy="zero")
    imputer.fit(df)
    model = to_onnx_graph(imputer)

    onnx_out = run_onnx(model, df)
    expected = imputer.transform(df)

    assert_onnx_close(onnx_out, expected)


def test_numeric_imputer_constant_one(df):
    imputer = NumericImputer(strategy="one")
    imputer.fit(df)
    model = to_onnx_graph(imputer)

    onnx_out = run_onnx(model, df)
    expected = imputer.transform(df)

    assert_onnx_close(onnx_out, expected)


# ── strategy: most_frequent ───────────────────────────────────────────────────

def test_numeric_imputer_most_frequent():
    X = pl.DataFrame({"A": [1.0, 1.0, 2.0, None], "B": [3.0, None, 3.0, 3.0]})
    imputer = NumericImputer(strategy="most_frequent")
    imputer.fit(X)
    model = to_onnx_graph(imputer)

    onnx_out = run_onnx(model, X)
    expected = imputer.transform(X)

    assert_onnx_close(onnx_out, expected)


# ── subset: only some columns are targeted ────────────────────────────────────

def test_numeric_imputer_subset_passthrough(df):
    """Columns not in subset must pass through unchanged (Identity semantics)."""
    imputer = NumericImputer(strategy="median", subset=["A"])
    imputer.fit(df)
    model = to_onnx_graph(imputer)

    onnx_out = run_onnx(model, df)

    # Column B should be unchanged (including NaN for the null position)
    assert math.isnan(float(onnx_out["B"][0]))
    np.testing.assert_allclose(onnx_out["B"][1:], np.array([2.0, 3.0, 4.0], dtype=np.float32))


# ── unsupported strategies ────────────────────────────────────────────────────

# ── mean / min / max: now export correctly (statistics stored during fit) ───────

@pytest.mark.parametrize("strategy", ["mean", "min", "max"])
def test_numeric_imputer_mean_min_max(df, strategy):
    """mean/min/max are now stored during fit() and export to a static ONNX graph."""
    imputer = NumericImputer(strategy=strategy)
    imputer.fit(df)
    model = to_onnx_graph(imputer)
    onnx_out = run_onnx(model, df)
    # null at position 1 of column A must be filled (no NaN in output)
    assert not math.isnan(float(onnx_out["A"][1]))


# ── unsupported strategies (forward / backward only) ──────────────────────

@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_numeric_imputer_unfixed_strategy_coerce(df, strategy):
    """errors='coerce' emits Identity nodes — nulls remain NaN in ONNX output."""
    imputer = NumericImputer(strategy=strategy)
    imputer.fit(df)
    model = to_onnx_graph(imputer, errors="coerce")  # should not raise

    onnx_out = run_onnx(model, df)
    # Column A: position 1 was null → NaN in the ONNX identity pass-through
    assert math.isnan(float(onnx_out["A"][1]))


@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_numeric_imputer_unfixed_strategy_raise(df, strategy):
    """errors='raise' raises OnnxNotSupportedError for strategies with no stored stats."""
    imputer = NumericImputer(strategy=strategy)
    imputer.fit(df)
    with pytest.raises(OnnxNotSupportedError, match=strategy):
        to_onnx_graph(imputer, errors="raise")


# ── unregistered transformer ──────────────────────────────────────────────────

def test_unregistered_transformer_raise():
    from gators.imputers import KNNImputer
    from gators.onnx_converters._converters import to_onnx_nodes

    imputer = KNNImputer()
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, None]})
    imputer.fit(X)

    with pytest.raises(OnnxNotSupportedError):
        to_onnx_nodes(imputer, {"A": "A__in", "B": "B__in"}, {"A": "A", "B": "B"}, errors="raise")


def test_unregistered_transformer_coerce():
    from gators.imputers import KNNImputer
    from gators.onnx_converters._converters import to_onnx_nodes

    imputer = KNNImputer()
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, None]})
    imputer.fit(X)

    nodes, inits = to_onnx_nodes(
        imputer, {"A": "A__in", "B": "B__in"}, {"A": "A", "B": "B"}, errors="coerce"
    )
    assert all(n.op_type == "Identity" for n in nodes)
    assert inits == []


def test_unregistered_transformer_coerce():
    from gators.imputers import KNNImputer
    from gators.onnx_converters._converters import to_onnx_nodes

    imputer = KNNImputer()
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, None]})
    imputer.fit(X)

    nodes, inits = to_onnx_nodes(
        imputer, {"A": "A__in", "B": "B__in"}, {"A": "A", "B": "B"}, errors="coerce"
    )
    assert all(n.op_type == "Identity" for n in nodes)
    assert inits == []


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_numeric_imputer_passthrough_column():
    """Column not in subset passes through the ONNX graph unchanged."""
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [10.0, 20.0, 30.0]})
    t = NumericImputer(strategy="mean", subset=["A"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["B"], X["B"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ────────────────────────────────────

def test_numeric_imputer_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    X = pl.DataFrame({
        "A":            [1.0, None, 3.0],
        "float32_pass": pl.Series([10.0, 20.0, 30.0], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True],  dtype=pl.Boolean),
    })
    t = NumericImputer(strategy="mean", subset=["A"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))
