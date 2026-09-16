"""ONNX tests for InteractionFeatures."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_str import InteractionFeatures
from gators.onnx_converters import pipeline_to_onnx, to_onnx_graph
from gators.imputers import StringImputer
from gators.pipeline import Pipeline
from .conftest import run_onnx


@pytest.fixture
def X():
    return pl.DataFrame({
        "A": ["cat", "dog", "cat"],
        "B": ["x",   "x",   "y"],
        "C": ["red", "blue", "green"],
        "num": [1.0, 2.0, 3.0],
    })


def _str_eq(out, exp, col):
    return out[col].tolist() == exp[col].to_list()


def test_degree2_all_pairs(X):
    t = InteractionFeatures(degree=2)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    for col in ["A__B", "A__C", "B__C"]:
        assert _str_eq(out, exp, col), col


def test_degree3_includes_triples(X):
    t = InteractionFeatures(degree=3)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    for col in ["A__B", "A__C", "B__C", "A__B__C"]:
        assert _str_eq(out, exp, col), col


def test_subset_restricts_columns(X):
    t = InteractionFeatures(subset=["A", "B"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _str_eq(out, exp, "A__B")
    assert "A__C" not in out and "B__C" not in out


def test_original_columns_kept(X):
    t = InteractionFeatures(degree=2)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert "A" in out and "B" in out and "C" in out


def test_numeric_passthrough(X):
    t = InteractionFeatures(degree=2)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    np.testing.assert_allclose(out["num"].astype(float), X["num"].to_numpy(allow_copy=True), atol=1e-5)


def test_in_pipeline(X):
    pipe = Pipeline(steps=[
        ("str_imp", StringImputer(strategy="most_frequent", subset=["A", "B"])),
        ("if_", InteractionFeatures(subset=["A", "B"])),
    ])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    exp = pipe.transform(X)
    assert _str_eq(out, exp, "A__B")


def test_non_string_subset_column_raises(X):
    """No ONNX operator reproduces Polars' numeric-to-string formatting exactly, so a
    non-string subset column must raise rather than silently produce mismatched strings."""
    from gators.onnx_converters._exceptions import OnnxNotSupportedError

    t = InteractionFeatures(subset=["A", "num"])
    t.fit(X)
    with pytest.raises(OnnxNotSupportedError, match="non-string subset columns"):
        to_onnx_graph(t)

