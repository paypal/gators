"""ONNX tests for OrdinalFeatures, CyclicFeatures, DiffFeatures, TimeBinFeatures.

All datetime inputs are fed as tensor(int64) (Polars physical representation).
Only components/bins that require pure integer arithmetic are supported.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_dt import CyclicFeatures, DiffFeatures, OrdinalFeatures, TimeBinFeatures
from gators.onnx_converters import to_onnx_graph, pipeline_to_onnx
from gators.onnx_converters._exceptions import OnnxNotSupportedError
from gators.pipeline import Pipeline
from .conftest import run_onnx


@pytest.fixture
def X():
    return pl.DataFrame({
        "ts":  ["2024-01-15 10:30:45", "2024-01-20 07:15:30"],
        "val": [1.0, 2.0],       # float passthrough column
        "cat": ["a", "b"],       # string passthrough column
    }).with_columns(pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))


def _bool_ok(onnx_val: np.ndarray, polars_val: pl.Series) -> bool:
    return np.array_equal(onnx_val.astype(bool), polars_val.to_numpy(allow_copy=True))

def _int_ok(onnx_val: np.ndarray, polars_val: pl.Series) -> bool:
    return np.array_equal(onnx_val, polars_val.to_numpy(allow_copy=True))

def _float_ok(onnx_val: np.ndarray, polars_val: pl.Series, atol: float = 1e-4) -> bool:
    return np.allclose(onnx_val.astype(float), polars_val.cast(pl.Float64).to_numpy(allow_copy=True), atol=atol)


# ── OrdinalFeatures ───────────────────────────────────────────────────────────

def test_ordinal_hour(X):
    t = OrdinalFeatures(subset=["ts"], components=["hour"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert _int_ok(out["ts__hour"], t.transform(X)["ts__hour"])
    assert list(out["ts__hour"]) == [10, 7]


def test_ordinal_minute(X):
    t = OrdinalFeatures(subset=["ts"], components=["minute"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert _int_ok(out["ts__minute"], t.transform(X)["ts__minute"])
    assert list(out["ts__minute"]) == [30, 15]


def test_ordinal_second(X):
    t = OrdinalFeatures(subset=["ts"], components=["second"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert _int_ok(out["ts__second"], t.transform(X)["ts__second"])
    assert list(out["ts__second"]) == [45, 30]


def test_ordinal_day_of_week(X):
    t = OrdinalFeatures(subset=["ts"], components=["day_of_week"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert _int_ok(out["ts__day_of_week"], t.transform(X)["ts__day_of_week"])
    # Monday=1, Saturday=6
    assert list(out["ts__day_of_week"]) == [1, 6]


def test_ordinal_weekend(X):
    t = OrdinalFeatures(subset=["ts"], components=["weekend"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert _float_ok(out["ts__weekend"], t.transform(X)["ts__weekend"].cast(pl.Float64))
    assert out["ts__weekend"][0] == 0.0  # Monday not weekend
    assert out["ts__weekend"][1] == 1.0  # Saturday is weekend


def test_ordinal_all_easy(X):
    t = OrdinalFeatures(subset=["ts"], components=["hour", "minute", "second", "day_of_week", "weekend"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    for c in ["ts__hour", "ts__minute", "ts__second", "ts__day_of_week"]:
        assert _int_ok(out[c], exp[c])


def test_ordinal_month(X):
    t = OrdinalFeatures(subset=["ts"], components=["month"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert _int_ok(out["ts__month"], t.transform(X)["ts__month"])


def test_ordinal_calendar_components(X):
    """All nine calendar components from the Richards algorithm."""
    comps = ["year", "quarter", "semester", "century", "month",
             "day_of_month", "day_of_year", "week", "leap_year"]
    t = OrdinalFeatures(subset=["ts"], components=comps)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    for c in comps:
        col = f"ts__{c}"
        if c == "leap_year":
            assert out[col].tolist() == exp[col].to_list(), f"{c} mismatch"
        else:
            assert _int_ok(out[col], exp[col]), f"{c} mismatch"


def test_ordinal_drop_columns(X):
    t = OrdinalFeatures(subset=["ts"], components=["hour"], drop_columns=True)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert "ts" not in out and "ts__hour" in out


# ── CyclicFeatures ────────────────────────────────────────────────────────────

def test_cyclic_hour_single_angle(X):
    t = CyclicFeatures(subset=["ts"], components=["hour"], angles=[0])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _float_ok(out["ts__hour__sin0"], exp["ts__hour__sin0"])


def test_cyclic_hour_multiple_angles(X):
    t = CyclicFeatures(subset=["ts"], components=["hour"], angles=[0, 45, 90])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    for col in ["ts__hour__sin0", "ts__hour__sin45", "ts__hour__sin90"]:
        assert _float_ok(out[col], exp[col])


def test_cyclic_minute_second(X):
    t = CyclicFeatures(subset=["ts"], components=["minute", "second"], angles=[0])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _float_ok(out["ts__minute__sin0"], exp["ts__minute__sin0"])
    assert _float_ok(out["ts__second__sin0"], exp["ts__second__sin0"])


def test_cyclic_day_of_week(X):
    t = CyclicFeatures(subset=["ts"], components=["day_of_week"], angles=[0])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _float_ok(out["ts__day_of_week__sin0"], exp["ts__day_of_week__sin0"])


def test_cyclic_month(X):
    t = CyclicFeatures(subset=["ts"], components=["month"], angles=[0])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert _float_ok(out["ts__month__sin0"], t.transform(X)["ts__month__sin0"])


def test_cyclic_calendar_components(X):
    """All six new cyclic calendar components."""
    comps = ["month", "quarter", "semester", "week", "day_of_month", "day_of_year"]
    t = CyclicFeatures(subset=["ts"], components=comps, angles=[0, 90])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    for c in comps:
        for a in [0, 90]:
            col = f"ts__{c}__sin{a}"
            assert _float_ok(out[col], exp[col]), f"{col} mismatch"


# ── DiffFeatures ──────────────────────────────────────────────────────────────

@pytest.fixture
def X_diff():
    return pl.DataFrame({
        "a": ["2024-01-15 10:00:00", "2024-06-01 00:00:00"],
        "b": ["2024-01-10 10:00:00", "2024-05-31 12:00:00"],
    }).with_columns([
        pl.col("a").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.col("b").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
    ])


def test_diff_column_pairs_days(X_diff):
    t = DiffFeatures(column_pairs=[("a", "b")], units=["d"])
    t.fit(X_diff)
    out = run_onnx(to_onnx_graph(t), X_diff)
    exp = t.transform(X_diff)
    assert _int_ok(out["a_minus_b__days"], exp["a_minus_b__days"])


def test_diff_column_pairs_all_units(X_diff):
    t = DiffFeatures(column_pairs=[("a", "b")], units=["d", "h", "m", "s"])
    t.fit(X_diff)
    out = run_onnx(to_onnx_graph(t), X_diff)
    exp = t.transform(X_diff)
    for col in ["a_minus_b__days", "a_minus_b__hours", "a_minus_b__minutes", "a_minus_b__seconds"]:
        assert _int_ok(out[col], exp[col])


def test_diff_reference_date():
    X = pl.DataFrame({"ts": ["2024-01-15 10:00:00", "2024-01-20 10:00:00"]}).with_columns(
        pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    t = DiffFeatures(reference_dates={"ts": "2024-01-01 00:00:00"}, units=["d", "h"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    assert _int_ok(out["ts_since_ref__days"],  exp["ts_since_ref__days"])
    assert _int_ok(out["ts_since_ref__hours"], exp["ts_since_ref__hours"])


def test_diff_drop_columns(X_diff):
    t = DiffFeatures(column_pairs=[("a", "b")], units=["d"], drop_columns=True)
    t.fit(X_diff)
    out = run_onnx(to_onnx_graph(t), X_diff)
    assert "a" not in out and "b" not in out
    assert "a_minus_b__days" in out


# ── TimeBinFeatures ───────────────────────────────────────────────────────────

def test_timebin_part_of_day(X):
    t = TimeBinFeatures(subset=["ts"], bin_types=["part_of_day"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    np.testing.assert_array_equal(out["ts__part_of_day"], exp["ts__part_of_day"].to_numpy(allow_copy=True).astype(str))


def test_timebin_day(X):
    """'day' bin_type: ISO weekday name, computed via ((date_days + 3) % 7) + 1 -> LabelEncoder."""
    t = TimeBinFeatures(subset=["ts"], bin_types=["day"])
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    exp = t.transform(X)
    np.testing.assert_array_equal(out["ts__day"], exp["ts__day"].to_numpy(allow_copy=True).astype(str))


def test_timebin_rush_hour(X):
    X_rush = pl.DataFrame({"ts": [
        "2024-01-15 08:00:00",   # morning rush
        "2024-01-15 18:00:00",   # evening rush
        "2024-01-15 12:00:00",   # off peak
    ]}).with_columns(pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    t = TimeBinFeatures(subset=["ts"], bin_types=["rush_hour"])
    t.fit(X_rush)
    out = run_onnx(to_onnx_graph(t), X_rush)
    exp = t.transform(X_rush)
    np.testing.assert_array_equal(out["ts__rush_hour"], exp["ts__rush_hour"].to_numpy(allow_copy=True).astype(str))
    assert out["ts__rush_hour"][0] == "morning_rush"
    assert out["ts__rush_hour"][1] == "evening_rush"
    assert out["ts__rush_hour"][2] == "off_peak"


def test_timebin_season_northern(X):
    X_s = pl.DataFrame({"ts": [
        datetime(2024,  1, 15, 12, 0),  # winter
        datetime(2024,  4,  1, 12, 0),  # spring
        datetime(2024,  7,  4, 12, 0),  # summer
        datetime(2024, 10, 31, 12, 0),  # fall
        datetime(2024, 12,  1, 12, 0),  # winter (December)
    ]})
    t = TimeBinFeatures(subset=["ts"], bin_types=["season"], hemisphere="northern")
    t.fit(X_s)
    out = run_onnx(to_onnx_graph(t), X_s)
    exp = t.transform(X_s)
    np.testing.assert_array_equal(out["ts__season"], exp["ts__season"].to_numpy(allow_copy=True).astype(str))
    assert list(out["ts__season"]) == ["winter", "spring", "summer", "fall", "winter"]


def test_timebin_season_southern(X):
    X_s = pl.DataFrame({"ts": [
        datetime(2024,  1, 15, 12, 0),  # summer (southern)
        datetime(2024,  7,  4, 12, 0),  # winter (southern)
    ]})
    t = TimeBinFeatures(subset=["ts"], bin_types=["season"], hemisphere="southern")
    t.fit(X_s)
    out = run_onnx(to_onnx_graph(t), X_s)
    exp = t.transform(X_s)
    np.testing.assert_array_equal(out["ts__season"], exp["ts__season"].to_numpy(allow_copy=True).astype(str))
    assert list(out["ts__season"]) == ["summer", "winter"]


def test_timebin_time_of_month():
    X_m = pl.DataFrame({"ts": [
        datetime(2024,  3,  5, 12, 0),  # beginning
        datetime(2024,  3, 15, 12, 0),  # middle
        datetime(2024,  3, 25, 12, 0),  # end
        datetime(2024,  2, 29, 12, 0),  # end (leap day)
    ]})
    t = TimeBinFeatures(subset=["ts"], bin_types=["time_of_month"])
    t.fit(X_m)
    out = run_onnx(to_onnx_graph(t), X_m)
    exp = t.transform(X_m)
    np.testing.assert_array_equal(out["ts__time_of_month"], exp["ts__time_of_month"].to_numpy(allow_copy=True).astype(str))
    assert list(out["ts__time_of_month"]) == ["beginning", "middle", "end", "end"]


def test_timebin_time_of_year():
    X_y = pl.DataFrame({"ts": [
        datetime(2024,  2, 15, 12, 0),  # early
        datetime(2024,  6, 15, 12, 0),  # mid
        datetime(2024, 11, 15, 12, 0),  # late
    ]})
    t = TimeBinFeatures(subset=["ts"], bin_types=["time_of_year"])
    t.fit(X_y)
    out = run_onnx(to_onnx_graph(t), X_y)
    exp = t.transform(X_y)
    np.testing.assert_array_equal(out["ts__time_of_year"], exp["ts__time_of_year"].to_numpy(allow_copy=True).astype(str))
    assert list(out["ts__time_of_year"]) == ["early", "mid", "late"]


def test_timebin_all_five_bins_leap_day():
    """All bin types including Richards edge case (2024-02-29 is a leap day)."""
    X_l = pl.DataFrame({"ts": [datetime(2024, 2, 29, 8, 30)]})
    t = TimeBinFeatures(subset=["ts"], bin_types=["part_of_day", "season", "time_of_month", "time_of_year", "rush_hour"])
    t.fit(X_l)
    out = run_onnx(to_onnx_graph(t), X_l)
    exp = t.transform(X_l)
    for b in t.bin_types:
        col = f"ts__{b}"
        np.testing.assert_array_equal(out[col], exp[col].to_numpy(allow_copy=True).astype(str))


def test_timebin_date_column_calendar_bins():
    """pl.Date column (physical=days): exercises unit=='date' branch in _get_date_days."""
    X_d = pl.DataFrame({"d": pl.Series([datetime(2024, 1, 15), datetime(2024, 7, 4)]).dt.date()})
    t = TimeBinFeatures(subset=["d"], bin_types=["season", "time_of_month", "time_of_year"])
    t.fit(X_d)
    out = run_onnx(to_onnx_graph(t), X_d)
    exp = t.transform(X_d)
    for b in t.bin_types:
        col = f"d__{b}"
        np.testing.assert_array_equal(out[col], exp[col].to_numpy(allow_copy=True).astype(str))


def test_timebin_drop_columns(X):
    t = TimeBinFeatures(subset=["ts"], bin_types=["part_of_day"], drop_columns=True)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert "ts" not in out and "ts__part_of_day" in out


# ── Pipeline tests (exercise get_input/output_onnx_type hooks) ────────────────

def test_ordinal_in_pipeline(X):
    """pipeline_to_onnx calls get_input/output_onnx_type for OrdinalFeatures."""
    pipe = Pipeline(steps=[("of", OrdinalFeatures(subset=["ts"], components=["hour", "minute", "weekend", "leap_year", "month"]))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "ts__hour" in out and "ts__minute" in out and "ts__weekend" in out
    assert "ts__leap_year" in out and "ts__month" in out
    np.testing.assert_allclose(out["val"].astype(float), X["val"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(out["cat"], X["cat"].to_numpy(allow_copy=True).astype(str))


def test_btf_in_pipeline(X):
    """pipeline_to_onnx calls get_output_onnx_type for BusinessTimeFeatures (covers STRING type path)."""
    from gators.feature_generation_dt import BusinessTimeFeatures
    pipe = Pipeline(steps=[("btf", BusinessTimeFeatures(subset=["ts"], features=["is_business_day"]))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "ts__is_business_day" in out
    np.testing.assert_array_equal(out["cat"], X["cat"].to_numpy(allow_copy=True).astype(str))


def test_cyclic_in_pipeline(X):
    """pipeline_to_onnx calls get_input/output_onnx_type for CyclicFeatures."""
    pipe = Pipeline(steps=[("cf", CyclicFeatures(subset=["ts"], components=["hour"], angles=[0], drop_columns=False))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "ts__hour__sin0" in out and "ts" in out


def test_cyclic_drop_columns_in_pipeline(X):
    """CyclicFeatures with drop_columns=True in pipeline covers _cf_output_columns drop path."""
    pipe = Pipeline(steps=[("cf", CyclicFeatures(subset=["ts"], components=["hour"], angles=[0], drop_columns=True))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "ts__hour__sin0" in out and "ts" not in out


def test_diff_in_pipeline(X_diff):
    """pipeline_to_onnx calls get_input/output_onnx_type for DiffFeatures."""
    X = X_diff.with_columns([pl.lit("x").alias("cat"), pl.lit(1.0).alias("score")])
    pipe = Pipeline(steps=[("df", DiffFeatures(column_pairs=[("a", "b")], units=["d"]))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "a_minus_b__days" in out
    np.testing.assert_array_equal(out["cat"], X["cat"].to_numpy(allow_copy=True).astype(str))


def test_diff_output_type_string_passthrough():
    """_df_output_type returns STRING for non-datetime, non-generated String columns."""
    from gators.onnx_converters._feature_generation_dt_converters import _df_output_type
    from onnx import TensorProto as TP
    X = pl.DataFrame({"a": ["2024-01-15"], "b": ["2024-01-10"], "cat": ["x"]}).with_columns(
        [pl.col(c).str.strptime(pl.Datetime, "%Y-%m-%d") for c in ["a", "b"]]
    )
    t = DiffFeatures(column_pairs=[("a", "b")], units=["d"])
    t.fit(X)
    assert _df_output_type(t, "cat") == TP.STRING


def test_timebin_in_pipeline(X):
    """pipeline_to_onnx calls get_input/output_onnx_type for TimeBinFeatures."""
    pipe = Pipeline(steps=[("tbf", TimeBinFeatures(subset=["ts"], bin_types=["part_of_day", "rush_hour"]))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "ts__part_of_day" in out and "ts__rush_hour" in out
