from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.clippers import CustomClipper, GaussianClipper, IQRClipper, MADClipper, QuantileClipper
from gators.onnx_converters import to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 100.0],
        "B": [5.0, 6.0, 7.0, 8.0, -100.0],
    })


@pytest.mark.parametrize("cls", [GaussianClipper, IQRClipper, MADClipper, QuantileClipper])
def test_clipper_values(df, cls):
    t = cls()
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_custom_clipper():
    X = pl.DataFrame({"A": [-10.0, 0.0, 5.0, 20.0], "B": [3.0, 6.0, 9.0, 12.0]})
    t = CustomClipper(lower_bounds={"A": 0.0, "B": 4.0}, upper_bounds={"A": 10.0, "B": 10.0})
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_clipper_clamps_extremes(df):
    t = IQRClipper()
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    # The outlier row (index 4) must be clipped
    assert float(onnx_out["A"][4]) < 100.0
    assert float(onnx_out["B"][4]) > -100.0


def test_clipper_partial_subset():
    """Column not in subset passes through unchanged (line 46: Identity for non-targeted col)."""
    X = pl.DataFrame({"A": [-5.0, 0.0, 5.0, 15.0], "B": [1.0, 2.0, 3.0, 4.0]})
    t = CustomClipper(lower_bounds={"A": 0.0}, upper_bounds={"A": 10.0})
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    # A clipped
    assert float(onnx_out["A"][0]) == pytest.approx(0.0)
    assert float(onnx_out["A"][3]) == pytest.approx(10.0)
    # B passes through
    np.testing.assert_allclose(onnx_out["B"].astype(float), X["B"].to_numpy(allow_copy=True), atol=1e-5)


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_clipper_passthrough_column():
    """Column not in subset passes through the ONNX graph unchanged."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 100.0], "B": [10.0, 20.0, 30.0, 40.0]})
    t = IQRClipper(subset=["A"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["B"], X["B"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_clipper_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 4
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 100.0],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = IQRClipper(subset=['A'])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))
