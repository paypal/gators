"""Shared helpers for gators.onnx_converters tests.

Tests are skipped automatically when 'onnx' or 'onnxruntime' are not installed.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# Skip the entire test module if either optional package is absent
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def run_onnx(model, df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Run an exported ONNX model on a Polars DataFrame.

    Feeds named inputs ``{col}__in`` as float32 or string depending on the
    session's declared input type, and returns a dict of output arrays.
    """
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_meta = {inp.name: inp.type for inp in sess.get_inputs()}
    feeds: dict[str, np.ndarray] = {}
    for col in df.columns:
        key = f"{col}__in"
        if key not in input_meta:
            continue
        if input_meta[key] == "tensor(string)":
            # Polars null → "None" via .astype(str); explicit fill_null("") converts to the "" sentinel instead.
            feeds[key] = df[col].fill_null("").to_numpy(allow_copy=True).astype(str)
        elif input_meta[key] == "tensor(bool)":
            feeds[key] = df[col].fill_null(False).to_numpy(allow_copy=True)
        elif input_meta[key] == "tensor(int64)":
            if df[col].dtype == pl.Boolean:
                # nullable bool→string encoding: null=-1, False=0, True=1
                feeds[key] = df[col].cast(pl.Int64).fill_null(-1).to_numpy(allow_copy=True)
            elif hasattr(df[col].dtype, 'time_unit') or df[col].dtype == pl.Date:
                # Datetime/Date: convert to physical int64 (μs/ms/ns since epoch, or days)
                feeds[key] = df[col].to_physical().cast(pl.Int64).to_numpy(allow_copy=True)
            else:
                feeds[key] = df[col].cast(pl.Int64).to_numpy(allow_copy=True)
        elif input_meta[key] == "tensor(double)":
            feeds[key] = df[col].cast(pl.Float64).to_numpy(allow_copy=True)
        else:
            feeds[key] = df[col].cast(pl.Float32).to_numpy(allow_copy=True)
    outputs = sess.run(None, feeds)
    return {out.name: arr for out, arr in zip(sess.get_outputs(), outputs)}


def assert_onnx_close(onnx_out: dict[str, np.ndarray], expected: pl.DataFrame, *, atol: float = 1e-5):
    """Compare ONNX float32 outputs against a Polars float64 expected DataFrame."""
    for col in expected.columns:
        np.testing.assert_allclose(
            onnx_out[col].astype(np.float64),
            expected[col].to_numpy(allow_copy=True),
            atol=atol,
            err_msg=f"Mismatch for column '{col}'",
        )
