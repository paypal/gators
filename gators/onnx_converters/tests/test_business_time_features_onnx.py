"""ONNX tests for BusinessTimeFeatures.

Datetime columns are fed as tensor(int64) containing microseconds since the
Unix epoch (the Polars physical representation).  All component extractions
use integer arithmetic; results match Polars for timestamps after 1970-01-01.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_dt import BusinessTimeFeatures
from gators.onnx_converters import to_onnx_graph
from .conftest import run_onnx


@pytest.fixture
def X():
    return pl.DataFrame({"ts": [
        "2024-01-15 08:00:00",   # Monday    before hours
        "2024-01-15 10:30:00",   # Monday    during hours  (1h30 after 9am)
        "2024-01-15 18:00:00",   # Monday    after hours
        "2024-01-20 10:00:00",   # Saturday  weekend
    ]}).with_columns(pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))


def _compare_bool(onnx_val: np.ndarray, polars_val: pl.Series) -> None:
    np.testing.assert_array_equal(
        onnx_val.astype(bool),
        polars_val.to_numpy(allow_copy=True),
    )


def _compare_float(onnx_val: np.ndarray, polars_val: pl.Series, *, atol: float = 1e-5) -> None:
    np.testing.assert_allclose(
        onnx_val.astype(float),
        polars_val.cast(pl.Float64).to_numpy(allow_copy=True),
        atol=atol,
        equal_nan=True,
    )


def _compare_str(onnx_val: np.ndarray, polars_val: pl.Series) -> None:
    np.testing.assert_array_equal(onnx_val, polars_val.to_numpy(allow_copy=True).astype(str))


# ── individual features ───────────────────────────────────────────────────────

def test_is_business_hour(X):
    t = BusinessTimeFeatures(subset=["ts"], features=["is_business_hour"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _compare_bool(out["ts__is_business_hour"], t.transform(X)["ts__is_business_hour"])


def test_is_business_day(X):
    t = BusinessTimeFeatures(subset=["ts"], features=["is_business_day"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _compare_bool(out["ts__is_business_day"], t.transform(X)["ts__is_business_day"])


def test_time_of_business_day(X):
    t = BusinessTimeFeatures(subset=["ts"], features=["time_of_business_day"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _compare_str(out["ts__time_of_business_day"], t.transform(X)["ts__time_of_business_day"])
    assert list(out["ts__time_of_business_day"]) == ["before_hours", "during_hours", "after_hours", "weekend"]


def test_hour_of_business_day(X):
    t = BusinessTimeFeatures(subset=["ts"], features=["hour_of_business_day"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _compare_float(out["ts__hour_of_business_day"], t.transform(X)["ts__hour_of_business_day"].cast(pl.Float64))
    assert np.isnan(float(out["ts__hour_of_business_day"][0]))  # before hours → NaN
    assert float(out["ts__hour_of_business_day"][1]) == pytest.approx(1.0)  # 10:30 → hour=10, 10-9=1


def test_all_features(X):
    t = BusinessTimeFeatures(
        subset=["ts"],
        features=["is_business_hour", "is_business_day", "time_of_business_day", "hour_of_business_day"],
    )
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    _compare_bool(out["ts__is_business_hour"],    exp["ts__is_business_hour"])
    _compare_bool(out["ts__is_business_day"],      exp["ts__is_business_day"])
    _compare_str( out["ts__time_of_business_day"], exp["ts__time_of_business_day"])
    _compare_float(out["ts__hour_of_business_day"], exp["ts__hour_of_business_day"].cast(pl.Float64))


# ── custom hours ──────────────────────────────────────────────────────────────

def test_custom_business_hours():
    X = pl.DataFrame({"ts": [
        "2024-01-15 07:00:00",   # before 8am start
        "2024-01-15 09:00:00",   # during
        "2024-01-15 19:00:00",   # after 18h end
    ]}).with_columns(pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    t = BusinessTimeFeatures(subset=["ts"], features=["is_business_hour"], business_hours_start=8, business_hours_end=18)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _compare_bool(out["ts__is_business_hour"], t.transform(X)["ts__is_business_hour"])


def test_custom_weekend():
    # weekend_days validator allows 0–6; Polars weekday 6 = Saturday.
    X = pl.DataFrame({"ts": [
        "2024-01-20 10:00:00",   # Saturday (Polars weekday 6)
        "2024-01-15 10:00:00",   # Monday   (Polars weekday 1)
    ]}).with_columns(pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    t = BusinessTimeFeatures(subset=["ts"], features=["is_business_day"], weekend_days=[6])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    _compare_bool(out["ts__is_business_day"], t.transform(X)["ts__is_business_day"])
    assert out["ts__is_business_day"][0] == 0.0  # Saturday → weekend
    assert out["ts__is_business_day"][1] == 1.0  # Monday → business day


# ── drop_columns ──────────────────────────────────────────────────────────────

def test_drop_columns(X):
    t = BusinessTimeFeatures(subset=["ts"], features=["is_business_day"], drop_columns=True)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert "ts" not in out
    assert "ts__is_business_day" in out


# ── multiple datetime columns ─────────────────────────────────────────────────

def test_multiple_datetime_columns():
    X = pl.DataFrame({
        "ts1": ["2024-01-15 09:30:00", "2024-01-20 09:30:00"],
        "ts2": ["2024-01-15 19:00:00", "2024-01-15 12:00:00"],
    }).with_columns([
        pl.col("ts1").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.col("ts2").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
    ])
    t = BusinessTimeFeatures(features=["is_business_hour", "is_business_day"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    for col in ["ts1__is_business_hour", "ts1__is_business_day",
                "ts2__is_business_hour", "ts2__is_business_day"]:
        _compare_bool(out[col], exp[col])


# ── millisecond datetime ──────────────────────────────────────────────────────

def test_datetime_ms():
    X = pl.DataFrame({"ts": [
        "2024-01-15 10:00:00",
        "2024-01-15 07:00:00",
    ]}).with_columns(pl.col("ts").str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S"))
    t = BusinessTimeFeatures(subset=["ts"], features=["is_business_hour", "is_business_day"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    _compare_bool(out["ts__is_business_hour"], exp["ts__is_business_hour"])
    _compare_bool(out["ts__is_business_day"],  exp["ts__is_business_day"])


# ── pl.Date columns ───────────────────────────────────────────────────────────

def test_business_time_features_date_column():
    """pl.Date columns: weekday works; is_business_hour always False (no time component)."""
    X = pl.DataFrame({"dt": ["2024-01-15", "2024-01-20"]}).with_columns(
        pl.col("dt").str.strptime(pl.Date, "%Y-%m-%d")
    )
    t = BusinessTimeFeatures(subset=["dt"], features=["is_business_day"])
    t.fit(X)
    assert t._dt_units["dt"] == "date"
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    _compare_bool(out["dt__is_business_day"], exp["dt__is_business_day"])
    assert out["dt__is_business_day"][0] == 1.0   # Monday
    assert out["dt__is_business_day"][1] == 0.0   # Saturday


def test_business_time_features_string_passthrough():
    """String + float columns not in subset pass through (covers Identity + STRING type paths)."""
    X = pl.DataFrame({
        "ts":  ["2024-01-15 10:30:00"],
        "cat": ["foo"],
        "val": [1.0],
    }).with_columns(pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    t = BusinessTimeFeatures(subset=["ts"], features=["is_business_hour"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    np.testing.assert_array_equal(out["cat"], X["cat"].to_numpy(allow_copy=True).astype(str))
    np.testing.assert_allclose(out["val"].astype(float), X["val"].to_numpy(allow_copy=True))
