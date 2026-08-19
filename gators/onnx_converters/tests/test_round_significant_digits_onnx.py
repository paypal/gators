from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.data_cleaning import RoundSignificantDigits
from gators.onnx_converters import to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({
        "a": [0.001234, 1234.0, -9876.5, 0.0],
        "b": [3.14159, 0.0, 9.9999, 100.0],
    })


def test_rsd_inplace(df):
    t = RoundSignificantDigits(n_digits=3)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_rsd_inplace_subset(df):
    t = RoundSignificantDigits(n_digits=2, subset=["a"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_rsd_not_inplace_drop(df):
    t = RoundSignificantDigits(n_digits=2, subset=["a"], inplace=False, drop_columns=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_rsd_not_inplace_keep(df):
    t = RoundSignificantDigits(n_digits=2, subset=["a"], inplace=False, drop_columns=False)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    # Only check the new column; original float64 vs float32 tolerance for pass-through
    np.testing.assert_allclose(
        onnx_out["a__round_2sig"].astype(float),
        expected["a__round_2sig"].to_numpy(allow_copy=True),
        atol=1e-4,
    )


def test_rsd_zero_passthrough(df):
    """Rows with x=0 must produce 0.0, not NaN."""
    t = RoundSignificantDigits(n_digits=3)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    # Row index 3: a=0.0, b=0.0
    assert float(onnx_out["a"][3]) == pytest.approx(0.0)
    assert float(onnx_out["b"][1]) == pytest.approx(0.0)
