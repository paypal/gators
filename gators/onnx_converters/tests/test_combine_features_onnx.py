"""ONNX tests for CombineFeatures.

Uses StringConcat (opset 20) to concatenate string columns with a fixed separator.

Note: null values arrive as '' in ONNX (ORT convention) rather than 'null' as in
Polars. Tests use non-null data to avoid this discrepancy.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_str import CombineFeatures
from gators.imputers import StringImputer
from gators.onnx_converters import pipeline_to_onnx, to_onnx_graph
from gators.pipeline import Pipeline
from .conftest import run_onnx


@pytest.fixture
def X():
    return pl.DataFrame({
        "cat1":   ["A", "B", "A"],
        "cat2":   ["X", "Y", "X"],
        "addr1":  ["US", "UK", "CA"],
        "amount": [100.0, 200.0, 300.0],
    })


def _str_eq(onnx_out: dict, expected: pl.DataFrame, col: str) -> bool:
    return onnx_out[col].tolist() == expected[col].to_list()


# ── Basic two-column group ────────────────────────────────────────────────────

def test_two_col_group_keep_columns(X):
    t = CombineFeatures(column_groups=[["cat1", "cat2"]], separator="_", drop_columns=False)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _str_eq(out, exp, "cat1__cat2")
    assert list(out["cat1__cat2"]) == ["A_X", "B_Y", "A_X"]


def test_two_col_group_drop_columns(X):
    t = CombineFeatures(column_groups=[["cat1", "cat2"]], separator="_", drop_columns=True)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _str_eq(out, exp, "cat1__cat2")
    assert "cat1" not in out and "cat2" not in out
    assert "addr1" in out  # non-group columns are kept


# ── Three-column group ────────────────────────────────────────────────────────

def test_three_col_group_custom_separator(X):
    t = CombineFeatures(
        column_groups=[["cat1", "cat2", "addr1"]],
        separator="|",
        new_column_names=["uid"],
        drop_columns=True,
    )
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _str_eq(out, exp, "uid")
    assert list(out["uid"]) == ["A|X|US", "B|Y|UK", "A|X|CA"]


# ── Multiple groups ───────────────────────────────────────────────────────────

def test_multiple_groups(X):
    t = CombineFeatures(
        column_groups=[["cat1", "cat2"], ["cat1", "addr1"]],
        separator="_",
        drop_columns=False,
    )
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _str_eq(out, exp, "cat1__cat2")
    assert _str_eq(out, exp, "cat1__addr1")


def test_multiple_groups_drop_shared_column(X):
    """A column appearing in multiple groups is dropped once."""
    t = CombineFeatures(
        column_groups=[["cat1", "cat2"], ["cat1", "addr1"]],
        separator="-",
        drop_columns=True,
    )
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _str_eq(out, exp, "cat1__cat2")
    assert _str_eq(out, exp, "cat1__addr1")
    assert "cat1" not in out and "cat2" not in out and "addr1" not in out


# ── Single-column group (edge case) ──────────────────────────────────────────

def test_single_col_group(X):
    t = CombineFeatures(
        column_groups=[["cat1"]],
        new_column_names=["cat1_copy"],
        drop_columns=False,
    )
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _str_eq(out, exp, "cat1_copy")


# ── Numeric passthrough ───────────────────────────────────────────────────────

def test_numeric_passthrough(X):
    t = CombineFeatures(column_groups=[["cat1", "cat2"]], drop_columns=False)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    np.testing.assert_allclose(out["amount"].astype(float), X["amount"].to_numpy(allow_copy=True), atol=1e-5)


# ── pipeline_to_onnx exercises get_output_onnx_type ─────────────────────────

def test_in_pipeline(X):
    pipe = Pipeline(steps=[
        ("str_imp", StringImputer(strategy="most_frequent", subset=["cat1", "cat2"])),
        ("cf", CombineFeatures(column_groups=[["cat1", "cat2"]], separator="_", drop_columns=False)),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    out = run_onnx(model, X)
    exp = pipe.transform(X)
    assert _str_eq(out, exp, "cat1__cat2")
