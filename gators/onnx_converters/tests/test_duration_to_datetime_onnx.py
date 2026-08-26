"""ONNX tests for DurationToDatetime.

Output columns are int64 (physical microseconds/milliseconds/nanoseconds since
Unix epoch) matching the convention of all other datetime ONNX converters.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_dt import DurationToDatetime
from gators.onnx_converters import pipeline_to_onnx, to_onnx_graph
from gators.onnx_converters._exceptions import OnnxNotSupportedError
from gators.pipeline import Pipeline
from .conftest import run_onnx


@pytest.fixture
def X_seconds():
    return pl.DataFrame({
        "TransactionDT": [86400, 172800, 259200],
        "val": [1.0, 2.0, 3.0],
    })


@pytest.fixture
def X_col_ref():
    return pl.DataFrame({
        "BaseDate": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
        "offset_days": [7, 14, 21],
        "val": [100.0, 200.0, 300.0],
    })


# ── Fixed reference date ──────────────────────────────────────────────────────

def test_fixed_ref_seconds_keep_columns(X_seconds):
    t = DurationToDatetime(
        subset=["TransactionDT"],
        reference_date=datetime(2017, 11, 30),
        unit="s",
        drop_columns=False,
    )
    t.fit(X_seconds)
    model = to_onnx_graph(t)
    out = run_onnx(model, X_seconds)
    expected = t.transform(X_seconds)["TransactionDT__datetime"].to_physical().to_list()
    assert list(out["TransactionDT__datetime"]) == expected


def test_fixed_ref_seconds_drop_columns(X_seconds):
    t = DurationToDatetime(
        subset=["TransactionDT"],
        reference_date=datetime(2017, 11, 30),
        unit="s",
        drop_columns=True,
    )
    t.fit(X_seconds)
    model = to_onnx_graph(t)
    out = run_onnx(model, X_seconds)
    assert "TransactionDT" not in out
    expected = t.transform(X_seconds)["TransactionDT__datetime"].to_physical().to_list()
    assert list(out["TransactionDT__datetime"]) == expected


def test_fixed_ref_days(X_seconds):
    t = DurationToDatetime(
        subset=["TransactionDT"],
        reference_date=datetime(2020, 1, 1),
        unit="d",
        drop_columns=True,
    )
    t.fit(X_seconds)
    model = to_onnx_graph(t)
    out = run_onnx(model, X_seconds)
    expected = t.transform(X_seconds)["TransactionDT__datetime"].to_physical().to_list()
    assert list(out["TransactionDT__datetime"]) == expected


def test_fixed_ref_hours(X_seconds):
    t = DurationToDatetime(
        subset=["TransactionDT"],
        reference_date=datetime(2024, 1, 1),
        unit="h",
        drop_columns=True,
    )
    t.fit(X_seconds)
    model = to_onnx_graph(t)
    out = run_onnx(model, X_seconds)
    expected = t.transform(X_seconds)["TransactionDT__datetime"].to_physical().to_list()
    assert list(out["TransactionDT__datetime"]) == expected


def test_fixed_ref_minutes():
    X = pl.DataFrame({"offset_m": [60, 120, 180], "val": [1.0, 2.0, 3.0]})
    t = DurationToDatetime(subset=["offset_m"], reference_date=datetime(2024, 1, 1), unit="m", drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    out = run_onnx(model, X)
    expected = t.transform(X)["offset_m__datetime"].to_physical().to_list()
    assert list(out["offset_m__datetime"]) == expected


def test_fixed_ref_milliseconds():
    X = pl.DataFrame({"offset_ms": [1000, 2000, 3000], "val": [1.0, 2.0, 3.0]})
    t = DurationToDatetime(subset=["offset_ms"], reference_date=datetime(2024, 1, 1), unit="ms", drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    out = run_onnx(model, X)
    expected = t.transform(X)["offset_ms__datetime"].to_physical().to_list()
    assert list(out["offset_ms__datetime"]) == expected


def test_fixed_ref_microseconds_factor_one():
    """unit='us' with a μs-based fixed reference: factor==1, no Mul node emitted."""
    X = pl.DataFrame({"offset_us": [1_000_000, 2_000_000, 3_000_000], "val": [1.0, 2.0, 3.0]})
    t = DurationToDatetime(subset=["offset_us"], reference_date=datetime(2024, 1, 1), unit="us", drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    out = run_onnx(model, X)
    expected = t.transform(X)["offset_us__datetime"].to_physical().to_list()
    assert list(out["offset_us__datetime"]) == expected


def test_fixed_ref_iso_string():
    """reference_date as ISO datetime string (not a column name)."""
    X = pl.DataFrame({"offset_s": [86400, 172800], "val": [1.0, 2.0]})
    t = DurationToDatetime(subset=["offset_s"], reference_date="2017-11-30T00:00:00", unit="s", drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    out = run_onnx(model, X)
    expected = t.transform(X)["offset_s__datetime"].to_physical().to_list()
    assert list(out["offset_s__datetime"]) == expected


def test_fixed_ref_multiple_subset():
    """Multiple offset columns with the same fixed reference."""
    X = pl.DataFrame({"a": [86400, 172800], "b": [3600, 7200], "val": [1.0, 2.0]})
    t = DurationToDatetime(subset=["a", "b"], reference_date=datetime(2024, 1, 1), unit="s", drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    out = run_onnx(model, X)
    expected = t.transform(X)
    assert list(out["a__datetime"]) == expected["a__datetime"].to_physical().to_list()
    assert list(out["b__datetime"]) == expected["b__datetime"].to_physical().to_list()


def test_passthrough_float_col(X_seconds):
    t = DurationToDatetime(subset=["TransactionDT"], reference_date=datetime(2017, 11, 30), unit="s", drop_columns=True)
    t.fit(X_seconds)
    model = to_onnx_graph(t)
    out = run_onnx(model, X_seconds)
    np.testing.assert_allclose(out["val"].astype(float), X_seconds["val"].to_numpy(allow_copy=True), atol=1e-5)


# ── Column-based reference date ───────────────────────────────────────────────

def test_col_ref_days_keep_columns(X_col_ref):
    t = DurationToDatetime(subset=["offset_days"], reference_date="BaseDate", unit="d", drop_columns=False)
    t.fit(X_col_ref)
    model = to_onnx_graph(t)
    out = run_onnx(model, X_col_ref)
    expected = t.transform(X_col_ref)["offset_days__datetime"].to_physical().to_list()
    assert list(out["offset_days__datetime"]) == expected


def test_col_ref_days_drop_columns(X_col_ref):
    t = DurationToDatetime(subset=["offset_days"], reference_date="BaseDate", unit="d", drop_columns=True)
    t.fit(X_col_ref)
    model = to_onnx_graph(t)
    out = run_onnx(model, X_col_ref)
    assert "offset_days" not in out
    expected = t.transform(X_col_ref)["offset_days__datetime"].to_physical().to_list()
    assert list(out["offset_days__datetime"]) == expected


def test_col_ref_seconds(X_col_ref):
    X = pl.DataFrame({
        "BaseDate": [datetime(2024, 1, 1), datetime(2024, 2, 1)],
        "offset_s": [3600, 7200],
    })
    t = DurationToDatetime(subset=["offset_s"], reference_date="BaseDate", unit="s", drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    out = run_onnx(model, X)
    expected = t.transform(X)["offset_s__datetime"].to_physical().to_list()
    assert list(out["offset_s__datetime"]) == expected


def test_col_ref_get_output_onnx_type_via_pipeline(X_col_ref):
    """get_output_onnx_type is exercised through pipeline_to_onnx."""
    from gators.imputers import NumericImputer
    pipe = Pipeline(steps=[
        ("imp", NumericImputer(strategy="mean", subset=["val"])),
        ("dtd", DurationToDatetime(subset=["offset_days"], reference_date="BaseDate", unit="d", drop_columns=True)),
    ])
    pipe.fit(X_col_ref)
    model = pipeline_to_onnx(pipe)
    out = run_onnx(model, X_col_ref)
    expected = pipe.transform(X_col_ref)["offset_days__datetime"].to_physical().to_list()
    assert list(out["offset_days__datetime"]) == expected


# ── Unsupported: fractional unit conversion ───────────────────────────────────

def test_us_offset_with_ms_reference_raises():
    """unit='us' with a datetime[ms] reference column → fractional: OnnxNotSupportedError."""
    X = pl.DataFrame({
        "ref_ms": pl.Series([datetime(2024, 1, 1)]).dt.cast_time_unit("ms"),
        "offset_us": [1000],
    })
    t = DurationToDatetime(subset=["offset_us"], reference_date="ref_ms", unit="us", drop_columns=True)
    t.fit(X)
    with pytest.raises(OnnxNotSupportedError, match="fractional"):
        to_onnx_graph(t, errors="raise")


def test_us_offset_with_ms_reference_coerce():
    """errors='coerce' falls back to Identity pass-through."""
    X = pl.DataFrame({
        "ref_ms": pl.Series([datetime(2024, 1, 1)]).dt.cast_time_unit("ms"),
        "offset_us": [1000],
    })
    t = DurationToDatetime(subset=["offset_us"], reference_date="ref_ms", unit="us", drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t, errors="coerce")
    assert model is not None


def test_pipeline_string_and_drop_columns_false(X_seconds):
    """Exercises string/offset passthrough branches in get_input/output_onnx_type via pipeline_to_onnx."""
    from gators.imputers import StringImputer
    X = pl.DataFrame({
        "TransactionDT": [86400, 172800],
        "cat": ["a", "b"],
        "val": [1.0, 2.0],
    })
    pipe = Pipeline(steps=[
        ("str_imp", StringImputer(strategy="most_frequent", subset=["cat"])),
        ("dtd", DurationToDatetime(subset=["TransactionDT"], reference_date=datetime(2017, 11, 30), unit="s", drop_columns=False)),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    out = run_onnx(model, X)
    expected = pipe.transform(X)["TransactionDT__datetime"].to_physical().to_list()
    assert list(out["TransactionDT__datetime"]) == expected
