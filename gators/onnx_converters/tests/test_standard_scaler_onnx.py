from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.scalers import StandardScaler
from gators.onnx_converters import to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({
        "age": [20.0, 30.0, 40.0, 50.0],
        "income": [20_000.0, 40_000.0, 60_000.0, 80_000.0],
    })


def test_standard_scaler_values(df):
    scaler = StandardScaler()
    scaler.fit(df)
    model = to_onnx_graph(scaler)

    onnx_out = run_onnx(model, df)
    expected = scaler.transform(df)

    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_standard_scaler_subset_passthrough(df):
    """Columns not in subset must pass through with their original values."""
    scaler = StandardScaler(subset=["age"])
    scaler.fit(df)
    model = to_onnx_graph(scaler)

    onnx_out = run_onnx(model, df)

    import numpy as np
    np.testing.assert_allclose(
        onnx_out["income"].astype(float),
        df["income"].to_numpy(allow_copy=True),
        atol=1e-3,
    )


def test_standard_scaler_single_column():
    X = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    scaler = StandardScaler()
    scaler.fit(X)
    model = to_onnx_graph(scaler)

    onnx_out = run_onnx(model, X)
    expected = scaler.transform(X)

    assert_onnx_close(onnx_out, expected, atol=1e-5)


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_standard_scaler_passthrough_column():
    """Column not in subset passes through the ONNX graph unchanged."""
    import numpy as np
    X = pl.DataFrame({"age": [20.0, 30.0, 40.0], "extra": [1.0, 2.0, 3.0]})
    t = StandardScaler(subset=["age"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["extra"], X["extra"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_standard_scaler_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 3
    X = pl.DataFrame({"age": [20.0, 30.0, 40.0],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = StandardScaler(subset=['age'])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))
