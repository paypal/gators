"""ONNX tests for HolidayFeatures.

Holiday dates are pre-computed at fit() time; the ONNX graph uses
broadcast arithmetic on a static [1 × M] constant array of holiday
days-since-epoch, so no calendar logic is needed at inference time.

Generated columns are INT64 (days since epoch for numeric features)
or BOOL (is_holiday), matching the other datetime ONNX converters.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_dt import HolidayFeatures
from gators.imputers import NumericImputer
from gators.onnx_converters import pipeline_to_onnx, to_onnx_graph
from gators.pipeline import Pipeline
from .conftest import run_onnx

# Fixed year set so tests are deterministic regardless of current date
_YEARS = [2024]


@pytest.fixture
def X():
    return pl.DataFrame({"date": [
        datetime(2024, 1, 1),   # New Year's Day (US holiday)
        datetime(2024, 1, 15),  # regular day
        datetime(2024, 7, 3),   # day before Independence Day
        datetime(2024, 7, 4),   # Independence Day (US holiday)
        datetime(2024, 7, 5),   # day after
        datetime(2024, 12, 31), # last day of year (no holiday after in fitted range)
    ], "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def _hf(features, drop_columns=False):
    return HolidayFeatures(subset=["date"], features=features, years=_YEARS, lookahead_years=0, drop_columns=drop_columns)


def _check(onnx_out, expected, col):
    polars_vals = expected[col].to_list()
    onnx_vals = onnx_out[col].tolist()
    assert str(polars_vals) == str(onnx_vals), f"{col}: polars={polars_vals} onnx={onnx_vals}"


# ── is_holiday ────────────────────────────────────────────────────────────────

def test_is_holiday_basic(X):
    t = _hf(["is_holiday"])
    t.fit(X)
    onnx_out = run_onnx(to_onnx_graph(t), X)
    expected = t.transform(X)
    _check(onnx_out, expected, "date__is_holiday")
    assert onnx_out["date__is_holiday"].dtype == np.float64


def test_is_holiday_drop_columns(X):
    t = _hf(["is_holiday"], drop_columns=True)
    t.fit(X)
    onnx_out = run_onnx(to_onnx_graph(t), X)
    expected = t.transform(X)
    assert "date" not in onnx_out
    _check(onnx_out, expected, "date__is_holiday")


def test_is_holiday_passthrough_float(X):
    t = _hf(["is_holiday"])
    t.fit(X)
    onnx_out = run_onnx(to_onnx_graph(t), X)
    np.testing.assert_allclose(onnx_out["val"].astype(float), X["val"].to_numpy(allow_copy=True), atol=1e-5)


# ── nearest_holiday_distance ──────────────────────────────────────────────────

def test_nearest_holiday_distance(X):
    t = _hf(["nearest_holiday_distance"])
    t.fit(X)
    onnx_out = run_onnx(to_onnx_graph(t), X)
    expected = t.transform(X)
    _check(onnx_out, expected, "date__nearest_holiday_distance")


# ── days_to_holiday ───────────────────────────────────────────────────────────

def test_days_to_holiday(X):
    t = _hf(["days_to_holiday"])
    t.fit(X)
    onnx_out = run_onnx(to_onnx_graph(t), X)
    expected = t.transform(X)
    _check(onnx_out, expected, "date__days_to_holiday")
    # Last day of year → no future holiday in fitted range → sentinel -1
    assert int(onnx_out["date__days_to_holiday"][-1]) == -1


# ── days_from_holiday ─────────────────────────────────────────────────────────

def test_days_from_holiday(X):
    t = _hf(["days_from_holiday"])
    t.fit(X)
    onnx_out = run_onnx(to_onnx_graph(t), X)
    expected = t.transform(X)
    _check(onnx_out, expected, "date__days_from_holiday")


# ── All four features together ────────────────────────────────────────────────

def test_all_features(X):
    feats = ["is_holiday", "nearest_holiday_distance", "days_to_holiday", "days_from_holiday"]
    t = _hf(feats)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    for feat in feats:
        _check(onnx_out, expected, f"date__{feat}")


# ── UK holidays ───────────────────────────────────────────────────────────────

def test_uk_holidays():
    X = pl.DataFrame({"date": [datetime(2024, 1, 1), datetime(2024, 12, 25), datetime(2024, 7, 4)]})
    t = HolidayFeatures(subset=["date"], features=["is_holiday"], country="GB", years=_YEARS, lookahead_years=0)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    _check(onnx_out, expected, "date__is_holiday")


# ── Datetime[ms] column ───────────────────────────────────────────────────────

def test_datetime_ms_column():
    X = pl.DataFrame({"date_ms": pl.Series([datetime(2024, 1, 1), datetime(2024, 7, 4)]).dt.cast_time_unit("ms"),
                      "val": [1.0, 2.0]})
    t = HolidayFeatures(subset=["date_ms"], features=["is_holiday"], years=_YEARS, lookahead_years=0)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    _check(onnx_out, expected, "date_ms__is_holiday")
    assert onnx_out["date_ms__is_holiday"].tolist() == [True, True]


# ── Date column (physical = days, no division needed) ─────────────────────────

def test_date_column():
    X = pl.DataFrame({"d": pl.Series([datetime(2024, 1, 1), datetime(2024, 7, 3), datetime(2024, 7, 4)]).dt.date(),
                      "val": [1.0, 2.0, 3.0]})
    t = HolidayFeatures(subset=["d"], features=["is_holiday", "nearest_holiday_distance"], years=_YEARS, lookahead_years=0)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    _check(onnx_out, expected, "d__is_holiday")
    _check(onnx_out, expected, "d__nearest_holiday_distance")


# ── pipeline_to_onnx (exercises get_output_onnx_type dispatch) ───────────────

def test_pipeline_to_onnx(X):
    pipe = Pipeline(steps=[
        ("num_imp", NumericImputer(strategy="mean", subset=["val"])),
        ("hf", _hf(["is_holiday", "days_from_holiday"])),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    expected = pipe.transform(X)
    _check(onnx_out, expected, "date__is_holiday")
    _check(onnx_out, expected, "date__days_from_holiday")


# ── Multiple subset columns ───────────────────────────────────────────────────

def test_multiple_subset_columns():
    X = pl.DataFrame({
        "ts1": [datetime(2024, 1, 1), datetime(2024, 7, 4)],
        "ts2": [datetime(2024, 7, 3), datetime(2024, 12, 25)],
    })
    t = HolidayFeatures(subset=["ts1", "ts2"], features=["is_holiday"], years=_YEARS, lookahead_years=0)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    _check(onnx_out, expected, "ts1__is_holiday")
    _check(onnx_out, expected, "ts2__is_holiday")


def test_string_passthrough_via_pipeline(X):
    """Exercises STRING branch in get_input_onnx_type and FLOAT fallback in get_output_onnx_type."""
    X_str = X.with_columns(pl.lit("cat").alias("cat_col"))
    pipe = Pipeline(steps=[
        ("hf", _hf(["is_holiday"])),
    ])
    pipe.fit(X_str)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X_str)
    expected = pipe.transform(X_str)
    _check(onnx_out, expected, "date__is_holiday")


def test_empty_holidays_is_holiday_all_false():
    """When _holidays is empty (country has no holidays for the years), is_holiday is all False."""
    X = pl.DataFrame({"date": [datetime(2024, 1, 1), datetime(2024, 7, 4)]})
    t = HolidayFeatures(subset=["date"], features=["is_holiday"], years=_YEARS, lookahead_years=0)
    t.fit(X)
    # Force-empty the holiday dict to exercise the empty branch
    t._holidays = {}
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert onnx_out["date__is_holiday"].tolist() == [False, False]
