from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.encoders import OrdinalEncoder, OneHotEncoder
from gators.imputers import NumericImputer
from gators.pipeline import Pipeline
from gators.onnx_converters import to_onnx_graph, pipeline_to_onnx
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({
        "cat": ["foo", "bar", "foo", "baz", "bar"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


# ── Basic OHE ─────────────────────────────────────────────────────────────────

def test_ohe_binary_values(df):
    enc = OneHotEncoder(subset=["cat"])
    enc.fit(df)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df)
    expected = enc.transform(df)

    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_ohe_output_columns_match(df):
    """Output tensor names must match the Polars OHE column names."""
    enc = OneHotEncoder(subset=["cat"])
    enc.fit(df)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df)
    expected = enc.transform(df)

    assert set(onnx_out.keys()) == set(expected.columns)


def test_ohe_unknown_category(df):
    """Unknown categories produce 0.0 in all binary columns."""
    enc = OneHotEncoder(subset=["cat"])
    enc.fit(df)
    model = to_onnx_graph(enc)

    X_new = pl.DataFrame({"cat": ["foo", "UNKNOWN"], "num": [1.0, 2.0]})
    onnx_out = run_onnx(model, X_new)

    # UNKNOWN row: all OHE columns should be 0
    for key, arr in onnx_out.items():
        if key.startswith("cat__"):
            assert float(arr[1]) == pytest.approx(0.0), f"{key}[1] should be 0 for unknown"


def test_ohe_passthrough_float(df):
    """Float columns not in subset pass through unchanged."""
    enc = OneHotEncoder(subset=["cat"])
    enc.fit(df)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df)

    np.testing.assert_allclose(
        onnx_out["num"].astype(float),
        df["num"].to_numpy(allow_copy=True),
        atol=1e-5,
    )


def test_ohe_multiple_columns():
    X = pl.DataFrame({
        "A": ["x", "y", "x"],
        "B": ["p", "q", "p"],
    })
    enc = OneHotEncoder()
    enc.fit(X)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, X)
    expected = enc.transform(X)

    assert_onnx_close(onnx_out, expected, atol=1e-5)


# ── Pipeline: OHE then Scaler ─────────────────────────────────────────────────

def test_pipeline_ohe_then_scaler(df):
    """OHE changes the column set; subsequent StandardScaler must receive the OHE outputs."""
    from gators.scalers import StandardScaler

    pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(subset=["cat"])),
        ("scaler", StandardScaler()),
    ])
    pipe.fit(df)
    model = pipeline_to_onnx(pipe)

    onnx_out = run_onnx(model, df)
    expected = pipe.transform(df)

    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_pipeline_imputer_then_ohe():
    """NumericImputer on float col, OHE on string col — non-overlapping subsets."""
    X = pl.DataFrame({
        "cat": ["foo", "bar", "foo"],
        "val": [1.0, None, 3.0],
    })
    pipe = Pipeline(steps=[
        ("imputer", NumericImputer(strategy="median", subset=["val"])),
        ("ohe", OneHotEncoder(subset=["cat"])),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)

    onnx_out = run_onnx(model, X)

    X_imp = pipe.steps[0][1].transform(X)
    X_enc = pipe.steps[1][1].transform(X_imp)
    assert_onnx_close(onnx_out, X_enc, atol=1e-5)


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_ohe_passthrough_column():
    """Float column not in OHE subset passes through the ONNX graph unchanged."""
    X = pl.DataFrame({"cat": ["a", "b", "a"], "val": [1.0, 2.0, 3.0]})
    t = OneHotEncoder(subset=["cat"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["val"], X["val"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_ohe_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 3
    X = pl.DataFrame({"cat": ["a", "b", "a"],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = OneHotEncoder(subset=['cat'])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))
