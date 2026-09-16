"""ONNX tests for TfidfFeatures.

Pipeline: StringNormalizer → StringSplit → TfIdfVectorizer(TF) → length-normalise → IDF-weight → Gather.

Note: null documents arrive as '' in ONNX (ORT convention), which produces
0.0 TF-IDF for every token — matching the Polars transformer's behaviour.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_str import TfidfFeatures
from gators.onnx_converters import pipeline_to_onnx, to_onnx_graph
from gators.imputers import StringImputer
from gators.pipeline import Pipeline
from .conftest import run_onnx


@pytest.fixture
def X():
    return pl.DataFrame({
        "text": [
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog",
        ],
        "val": [1.0, 2.0, 3.0],
    })


def _check(onnx_out, expected, atol=1e-5):
    for col in [c for c in expected.columns if "__tfidf_" in c]:
        np.testing.assert_allclose(
            onnx_out[col].astype(float),
            expected[col].cast(pl.Float64).to_numpy(allow_copy=True),
            atol=atol,
            err_msg=f"Mismatch for {col}",
        )


def test_basic_tfidf(X):
    t = TfidfFeatures(subset=["text"], max_features=5)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _check(out, t.transform(X))


def test_drop_columns(X):
    t = TfidfFeatures(subset=["text"], max_features=3, drop_columns=True)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    _check(out, exp)
    assert "text" not in out


def test_no_lowercase(X):
    t = TfidfFeatures(subset=["text"], max_features=3, lowercase=False)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _check(out, t.transform(X))


def test_custom_separator():
    X = pl.DataFrame({"text": ["a|b|c", "a|b", "c|d"], "num": [1.0, 2.0, 3.0]})
    t = TfidfFeatures(subset=["text"], separator="|", max_features=4)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _check(out, t.transform(X))


def test_numeric_passthrough(X):
    t = TfidfFeatures(subset=["text"], max_features=3)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    np.testing.assert_allclose(out["val"].astype(float), X["val"].to_numpy(allow_copy=True), atol=1e-5)


def test_multiple_columns():
    X = pl.DataFrame({
        "title": ["cat dog", "dog fish", "cat fish"],
        "body":  ["the quick fox", "the lazy dog", "the quick dog"],
    })
    t = TfidfFeatures(subset=["title", "body"], max_features=3)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _check(out, t.transform(X))


def test_in_pipeline(X):
    """pipeline_to_onnx exercises get_output_onnx_type."""
    pipe = Pipeline(steps=[
        ("imp", StringImputer(strategy="most_frequent", subset=["text"])),
        ("tfidf", TfidfFeatures(subset=["text"], max_features=3)),
    ])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    exp = pipe.transform(X)
    _check(out, exp)


def test_pipeline_string_passthrough(X):
    """String passthrough column exercises the dtype-STRING branch in get_input_onnx_type."""
    X_extra = X.with_columns(pl.lit("extra").alias("cat"))
    pipe = Pipeline(steps=[
        ("tfidf", TfidfFeatures(subset=["text"], max_features=3)),
    ])
    pipe.fit(X_extra)
    out = run_onnx(pipeline_to_onnx(pipe), X_extra)
    exp = pipe.transform(X_extra)
    _check(out, exp)
