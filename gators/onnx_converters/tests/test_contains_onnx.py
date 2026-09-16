"""ONNX tests for Contains."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_str import Contains
from gators.onnx_converters import pipeline_to_onnx, to_onnx_graph
from gators.pipeline import Pipeline
from .conftest import run_onnx


@pytest.fixture
def X():
    return pl.DataFrame({
        "text": ["sub1 here", "no match", "sub1sub1", "nothing", "xyz"],
        "num":  [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_contains_basic(X):
    t = Contains(contains_dict={"text": ["sub1"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)["text__contains_sub1"].fill_null(False).to_list()
    assert out["text__contains_sub1"].tolist() == exp
    assert list(out["text__contains_sub1"]) == [True, False, True, False, False]


def test_contains_multiple_substrings(X):
    t = Contains(contains_dict={"text": ["sub1", "xyz"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert out["text__contains_sub1"].tolist() == exp["text__contains_sub1"].fill_null(False).to_list()
    assert out["text__contains_xyz"].tolist() == exp["text__contains_xyz"].fill_null(False).to_list()


def test_contains_multiple_columns():
    X = pl.DataFrame({"a": ["abc", "xyz"], "b": ["abc", "no"], "num": [1.0, 2.0]})
    t = Contains(contains_dict={"a": ["abc"], "b": ["abc"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert out["a__contains_abc"].tolist() == exp["a__contains_abc"].fill_null(False).to_list()
    assert out["b__contains_abc"].tolist() == exp["b__contains_abc"].fill_null(False).to_list()


def test_contains_source_passthrough(X):
    t = Contains(contains_dict={"text": ["sub1"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert "text" in out
    np.testing.assert_allclose(out["num"].astype(float), X["num"].to_numpy(allow_copy=True), atol=1e-5)


def test_contains_in_pipeline(X):
    """pipeline_to_onnx exercises get_output_onnx_type and get_input_onnx_type."""
    pipe = Pipeline(steps=[("co", Contains(contains_dict={"text": ["sub1"]}))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    exp = pipe.transform(X)
    assert out["text__contains_sub1"].tolist() == exp["text__contains_sub1"].fill_null(False).to_list()


def test_contains_pipeline_string_passthrough(X):
    """String passthrough column exercises the dtype-STRING branch in _co_input_type."""
    X_extra = X.with_columns(pl.lit("extra").alias("cat"))
    pipe = Pipeline(steps=[("co", Contains(contains_dict={"text": ["sub1"]}))])
    pipe.fit(X_extra)
    out = run_onnx(pipeline_to_onnx(pipe), X_extra)
    exp = pipe.transform(X_extra)
    assert out["text__contains_sub1"].tolist() == exp["text__contains_sub1"].fill_null(False).to_list()
