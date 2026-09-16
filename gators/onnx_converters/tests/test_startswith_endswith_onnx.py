"""ONNX tests for Startswith and Endswith.

Algorithm:
  Startswith: StringSplit(delimiter=prefix) → Gather first part → Equal('') → And(not_empty)
  Endswith:   StringSplit(delimiter=suffix) → GatherND last part → Equal('') → And(not_empty)

Note: null source strings arrive as '' in ORT and produce False (vs Polars null).
Tests use non-null data to avoid this discrepancy.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_str import Endswith, Startswith
from gators.imputers import StringImputer
from gators.onnx_converters import pipeline_to_onnx, to_onnx_graph
from gators.pipeline import Pipeline
from .conftest import run_onnx


@pytest.fixture
def X():
    return pl.DataFrame({
        "text": ["abcdef", "xyzabc", "abc", "xyz", "abcxyzabc"],
        "num":  [1.0, 2.0, 3.0, 4.0, 5.0],
    })


# ── Startswith ────────────────────────────────────────────────────────────────

def test_startswith_basic(X):
    t = Startswith(startswith_dict={"text": ["abc"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)["text__startswith_abc"].fill_null(False).to_list()
    assert out["text__startswith_abc"].tolist() == exp
    assert list(out["text__startswith_abc"]) == [True, False, True, False, True]


def test_startswith_multiple_prefixes(X):
    t = Startswith(startswith_dict={"text": ["abc", "xyz"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert out["text__startswith_abc"].tolist() == exp["text__startswith_abc"].fill_null(False).to_list()
    assert out["text__startswith_xyz"].tolist() == exp["text__startswith_xyz"].fill_null(False).to_list()


def test_startswith_multiple_columns():
    X = pl.DataFrame({"a": ["pre_a", "no"], "b": ["pre_b", "pre_b"], "num": [1.0, 2.0]})
    t = Startswith(startswith_dict={"a": ["pre_"], "b": ["pre_"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert out["a__startswith_pre_"].tolist() == exp["a__startswith_pre_"].fill_null(False).to_list()
    assert out["b__startswith_pre_"].tolist() == exp["b__startswith_pre_"].fill_null(False).to_list()


def test_startswith_source_passthrough(X):
    t = Startswith(startswith_dict={"text": ["abc"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert "text" in out
    np.testing.assert_allclose(out["num"].astype(float), X["num"].to_numpy(allow_copy=True), atol=1e-5)


def test_startswith_in_pipeline(X):
    pipe = Pipeline(steps=[
        ("imp", StringImputer(strategy="most_frequent", subset=["text"])),
        ("sw", Startswith(startswith_dict={"text": ["abc"]})),
    ])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    exp = pipe.transform(X)
    assert out["text__startswith_abc"].tolist() == exp["text__startswith_abc"].fill_null(False).to_list()


def test_startswith_pipeline_string_passthrough(X):
    """String passthrough column exercises the dtype-STRING branch in _sw_ew_input_type."""
    X_extra = X.with_columns(pl.lit("extra").alias("cat"))
    pipe = Pipeline(steps=[
        ("sw", Startswith(startswith_dict={"text": ["abc"]})),
    ])
    pipe.fit(X_extra)
    out = run_onnx(pipeline_to_onnx(pipe), X_extra)
    exp = pipe.transform(X_extra)
    assert out["text__startswith_abc"].tolist() == exp["text__startswith_abc"].fill_null(False).to_list()


# ── Endswith ──────────────────────────────────────────────────────────────────

def test_endswith_basic(X):
    t = Endswith(endswith_dict={"text": ["abc"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)["text__endswith_abc"].fill_null(False).to_list()
    assert out["text__endswith_abc"].tolist() == exp
    assert list(out["text__endswith_abc"]) == [False, True, True, False, True]


def test_endswith_multiple_suffixes(X):
    t = Endswith(endswith_dict={"text": ["abc", "xyz"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert out["text__endswith_abc"].tolist() == exp["text__endswith_abc"].fill_null(False).to_list()
    assert out["text__endswith_xyz"].tolist() == exp["text__endswith_xyz"].fill_null(False).to_list()


def test_endswith_multiple_columns():
    X = pl.DataFrame({"a": ["x_suf", "no"], "b": ["y_suf", "y_suf"], "num": [1.0, 2.0]})
    t = Endswith(endswith_dict={"a": ["_suf"], "b": ["_suf"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert out["a__endswith__suf"].tolist() == exp["a__endswith__suf"].fill_null(False).to_list()
    assert out["b__endswith__suf"].tolist() == exp["b__endswith__suf"].fill_null(False).to_list()


def test_endswith_source_passthrough(X):
    t = Endswith(endswith_dict={"text": ["abc"]})
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert "text" in out
    np.testing.assert_allclose(out["num"].astype(float), X["num"].to_numpy(allow_copy=True), atol=1e-5)


def test_endswith_in_pipeline(X):
    pipe = Pipeline(steps=[
        ("imp", StringImputer(strategy="most_frequent", subset=["text"])),
        ("ew", Endswith(endswith_dict={"text": ["abc"]})),
    ])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    exp = pipe.transform(X)
    assert out["text__endswith_abc"].tolist() == exp["text__endswith_abc"].fill_null(False).to_list()


def test_endswith_pipeline_string_passthrough(X):
    """String passthrough column exercises the dtype-STRING branch in _sw_ew_input_type."""
    X_extra = X.with_columns(pl.lit("extra").alias("cat"))
    pipe = Pipeline(steps=[
        ("ew", Endswith(endswith_dict={"text": ["abc"]})),
    ])
    pipe.fit(X_extra)
    out = run_onnx(pipeline_to_onnx(pipe), X_extra)
    exp = pipe.transform(X_extra)
    assert out["text__endswith_abc"].tolist() == exp["text__endswith_abc"].fill_null(False).to_list()
