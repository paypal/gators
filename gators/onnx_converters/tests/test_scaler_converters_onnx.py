from __future__ import annotations

import math
import numpy as np
import polars as pl
import pytest

from gators.scalers import (
    ArcSinSquareRootScaler, ArcSinhScaler, BoxCox, Log1pScaler,
    MinmaxScaler, PowerScaler, RobustScaler, YeoJohnson,
)
from gators.onnx_converters import to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]})


def _compare(t, df, atol=1e-4, float_datatype="float64"):
    """Fit, export, run ONNX and compare against Polars transform output."""
    t.fit(df)
    model = to_onnx_graph(t, float_datatype=float_datatype)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=atol)


def test_minmax_scaler(df):
    _compare(MinmaxScaler(), df)


def test_robust_scaler(df):
    _compare(RobustScaler(), df)


@pytest.mark.parametrize("base", ["e", "2", "10"])
def test_log1p_scaler(df, base):
    _compare(Log1pScaler(base=base), df)


def test_arcsinh_scaler(df):
    _compare(ArcSinhScaler(), df)


def test_arcsinh_scaler_float32():
    X = pl.DataFrame({"A": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32)})
    _compare(ArcSinhScaler(), X, float_datatype="float32")


def test_arcsin_sqrt_scaler():
    # ArcSin requires values in [0, 1]
    X = pl.DataFrame({"A": [0.1, 0.3, 0.5, 0.9], "B": [0.2, 0.4, 0.6, 0.8]})
    _compare(ArcSinSquareRootScaler(), X)


def test_arcsin_sqrt_scaler_float32():
    X = pl.DataFrame({"A": pl.Series([0.1, 0.3, 0.5, 0.9], dtype=pl.Float32)})
    _compare(ArcSinSquareRootScaler(), X, float_datatype="float32")


def test_power_scaler(df):
    _compare(PowerScaler(power=2.0), df)


def test_boxcox_lambda_nonzero(df):
    _compare(BoxCox(lambdas={"A": 0.5, "B": 2.0}), df)


def test_boxcox_lambda_zero(df):
    _compare(BoxCox(lambdas={"A": 0.0, "B": 0.0}), df)


def test_yeo_johnson_lambda_nonzero(df):
    _compare(YeoJohnson(lambdas={"A": 0.5, "B": 1.5}), df)


def test_yeo_johnson_lambda_zero(df):
    _compare(YeoJohnson(lambdas={"A": 0.0, "B": 0.0}), df)


def test_yeo_johnson_lambda_two():
    # lambda=2 triggers -log(-X+1) for negative branch
    X = pl.DataFrame({"A": [-1.0, -0.5, 0.5, 1.0]})
    _compare(YeoJohnson(lambdas={"A": 2.0}), X)


def test_scaler_passthrough(df):
    """Columns not in subset must pass through unchanged (bit-accurate)."""
    t = MinmaxScaler(subset=["A"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    np.testing.assert_array_equal(onnx_out["B"], df["B"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_scaler_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 3
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = MinmaxScaler(subset=['A'])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))
