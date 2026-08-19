"""ONNX tests for GroupStatisticsFeatures.

The ONNX converter uses training-time group statistics (stored in _group_stats
during fit) rather than batch-time .over() aggregations.  On the training data
these produce identical results; on new data the group stats are frozen.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.feature_generation import GroupStatisticsFeatures
from gators.onnx_converters import to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def X():
    return pl.DataFrame({
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
        "cat1":   ["A",   "A",   "B",   "B",   "A"],
        "cat2":   ["X",   "Y",   "X",   "X",   "X"],
    })


def _compare(t: GroupStatisticsFeatures, df: pl.DataFrame, *, atol: float = 1e-4) -> dict:
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    # Compare generated feature columns and subset columns if retained
    feature_cols = list(t._column_mapping.values())
    if not t.drop_columns:
        feature_cols = [c for c in t.subset] + feature_cols
    assert_onnx_close({c: onnx_out[c] for c in feature_cols}, expected.select(feature_cols), atol=atol)
    # String passthrough columns (by + any other string cols) are unchanged
    for by_col in t.by:
        np.testing.assert_array_equal(onnx_out[by_col], df[by_col].to_numpy(allow_copy=True).astype(str))
    return onnx_out


# ── absolute statistics ───────────────────────────────────────────────────────

def test_mean(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["mean"]), X)

def test_std(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["std"]), X)

def test_median(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["median"]), X)

def test_min(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["min"]), X)

def test_max(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["max"]), X)

def test_sum(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["sum"]), X)

def test_count(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["count"]), X)

def test_range(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["range"]), X)


# ── relative statistics ───────────────────────────────────────────────────────

def test_mean_ratio(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["mean_ratio"]), X)

def test_median_ratio(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["median_ratio"]), X)

def test_zscore(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["zscore"]), X)

def test_minmax(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["minmax"]), X)


# ── fill_value for zero denominator ──────────────────────────────────────────

def test_fill_value_zero_mean():
    """Group with all-zero values → mean is 0 → fill_value applied."""
    X = pl.DataFrame({"v": [0.0, 0.0, 5.0, 10.0], "g": ["A", "A", "B", "B"]})
    t = GroupStatisticsFeatures(subset=["v"], by=["g"], func=["mean_ratio"], fill_value=-1.0)
    onnx_out = _compare(t, X)
    assert float(onnx_out["mean_ratio_v__per_g"][0]) == pytest.approx(-1.0)


def test_fill_value_zero_range():
    """Group with constant values → range is 0 → fill_value applied."""
    X = pl.DataFrame({"v": [5.0, 5.0, 2.0, 8.0], "g": ["A", "A", "B", "B"]})
    t = GroupStatisticsFeatures(subset=["v"], by=["g"], func=["minmax"], fill_value=-1.0)
    onnx_out = _compare(t, X)
    assert float(onnx_out["minmax_v__per_g"][0]) == pytest.approx(-1.0)


# ── multiple groupby columns ──────────────────────────────────────────────────

def test_multiple_by_columns(X):
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1", "cat2"], func=["mean"]), X)


# ── multiple subset columns ───────────────────────────────────────────────────

def test_multiple_subset_columns():
    X = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0], "g": ["X", "X", "Y", "Y"]})
    _compare(GroupStatisticsFeatures(subset=["a", "b"], by=["g"], func=["mean", "zscore"]), X)


# ── all 12 functions at once ──────────────────────────────────────────────────

def test_all_functions(X):
    all_funcs = ["mean", "std", "median", "min", "max", "sum", "count", "range",
                 "mean_ratio", "median_ratio", "zscore", "minmax"]
    _compare(GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=all_funcs), X)


# ── drop_columns ──────────────────────────────────────────────────────────────

def test_drop_columns(X):
    t = GroupStatisticsFeatures(subset=["amount"], by=["cat1"], func=["mean"], drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "amount" not in onnx_out
    assert "mean_amount__per_cat1" in onnx_out


# ── unseen group at inference time ────────────────────────────────────────────

def test_unseen_group_absolute():
    """Unseen groups fall back to 0.0 for absolute stats."""
    X_train = pl.DataFrame({"v": [1.0, 2.0], "g": ["A", "A"]})
    X_test  = pl.DataFrame({"v": [5.0], "g": ["Z"]})
    t = GroupStatisticsFeatures(subset=["v"], by=["g"], func=["mean"])
    t.fit(X_train)
    model = to_onnx_graph(t)
    out = run_onnx(model, X_test)
    assert float(out["mean_v__per_g"][0]) == pytest.approx(0.0)


# ── Pipeline + Float32 (exercise get_output_onnx_type and no-cast path) ─────────────────────

def test_gsf_in_pipeline():
    """pipeline_to_onnx calls get_output_onnx_type for GroupStatisticsFeatures."""
    from gators.pipeline import Pipeline
    from gators.onnx_converters import pipeline_to_onnx
    X = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "g": ["A", "A", "B", "B"]})
    pipe = Pipeline(steps=[("gsf", GroupStatisticsFeatures(subset=["v"], by=["g"], func=["mean"]))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    np.testing.assert_array_equal(out["g"], X["g"].to_numpy(allow_copy=True).astype(str))


def test_gsf_float32_input():
    """Float32 input: no Cast node needed (covers the else: fill_name=fill_f32_name branch)."""
    X = pl.DataFrame({
        "v": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
        "g": ["A", "A", "B", "B"],
    })
    t = GroupStatisticsFeatures(subset=["v"], by=["g"], func=["mean"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    np.testing.assert_allclose(out["mean_v__per_g"].astype(float), exp["mean_v__per_g"].cast(pl.Float64).to_numpy(), atol=1e-4)
