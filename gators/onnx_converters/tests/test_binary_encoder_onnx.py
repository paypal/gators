from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.encoders import BinaryEncoder
from gators.imputers import NumericImputer
from gators.pipeline import Pipeline
from gators.onnx_converters import to_onnx_graph, pipeline_to_onnx
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({
        "cat": ["A", "B", "C", "D", "A", "B"],
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


def test_binary_encoder_drop_columns(df):
    enc = BinaryEncoder(inplace=False, drop_columns=True)
    enc.fit(df)
    model = to_onnx_graph(enc)
    onnx_out = run_onnx(model, df)
    expected = enc.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_binary_encoder_drop_columns_false_output_type():
    """pipeline_to_onnx calls get_output_onnx_type for a passed-through subset column (STRING)."""
    X = pl.DataFrame({"cat": ["A", "B", "C", "D", "A", "B"]})
    pipe = Pipeline(steps=[
        ("enc", BinaryEncoder(subset=["cat"], inplace=False, drop_columns=False)),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["cat"], X["cat"].to_numpy(allow_copy=True))


def test_binary_encoder_keep_columns(df):
    enc = BinaryEncoder(inplace=False, drop_columns=False)
    enc.fit(df)
    model = to_onnx_graph(enc)
    onnx_out = run_onnx(model, df)
    expected = enc.transform(df)
    # Compare only numeric output columns (bit columns + numeric pass-through)
    numeric_cols = [c for c in expected.columns if expected[c].dtype != pl.String]
    assert_onnx_close({c: onnx_out[c] for c in numeric_cols}, expected.select(numeric_cols), atol=1e-5)


def test_binary_encoder_subset(df):
    enc = BinaryEncoder(subset=["cat"], inplace=False, drop_columns=True)
    enc.fit(df)
    model = to_onnx_graph(enc)
    onnx_out = run_onnx(model, df)
    expected = enc.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_binary_encoder_unknown_category(df):
    enc = BinaryEncoder(subset=["cat"], inplace=False, drop_columns=True)
    enc.fit(df)
    model = to_onnx_graph(enc)
    X_new = pl.DataFrame({"cat": ["A", "UNKNOWN"], "val": [1.0, 2.0]})
    onnx_out = run_onnx(model, X_new)
    # Unknown category → default 0.0 for every bit column
    assert float(onnx_out["cat__binary_enc_0"][1]) == pytest.approx(0.0)
    assert float(onnx_out["cat__binary_enc_1"][1]) == pytest.approx(0.0)


def test_binary_encoder_passthrough_numeric(df):
    enc = BinaryEncoder(subset=["cat"], inplace=False, drop_columns=True)
    enc.fit(df)
    model = to_onnx_graph(enc)
    onnx_out = run_onnx(model, df)
    np.testing.assert_allclose(
        onnx_out["val"].astype(float),
        df["val"].to_numpy(allow_copy=True),
        atol=1e-5,
    )


def test_binary_encoder_in_pipeline(df):
    """Exercises get_output_onnx_type via pipeline_to_onnx."""
    pipe = Pipeline(steps=[
        ("num_imp", NumericImputer(strategy="mean", subset=["val"])),
        ("enc", BinaryEncoder(subset=["cat"], inplace=False, drop_columns=True)),
    ])
    pipe.fit(df)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, df)
    expected = pipe.transform(df)
    numeric_cols = [c for c in expected.columns if expected[c].dtype != pl.String]
    assert_onnx_close({c: onnx_out[c] for c in numeric_cols}, expected.select(numeric_cols), atol=1e-5)


def test_binary_encoder_output_columns_column_without_valid_categories():
    """A subset column whose categories never meet min_count has no entry in n_bits_."""
    from gators.onnx_converters import get_output_columns

    X = pl.DataFrame({"rare": ["a", "b", "c", "d"], "cat": ["x", "y", "x", "y"]})
    enc = BinaryEncoder(subset=["rare", "cat"], min_count=2, inplace=False, drop_columns=True)
    enc.fit(X)
    assert "rare" not in enc.n_bits_
    result = get_output_columns(enc, list(X.columns))
    assert result == ["cat__binary_enc_0"]
