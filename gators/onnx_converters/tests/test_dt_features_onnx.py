"""ONNX tests for OrdinalFeatures, CyclicFeatures, DiffFeatures, TimeBinFeatures.

All datetime inputs are fed as tensor(int64) (Polars physical representation).
Only components/bins that require pure integer arithmetic are supported.
"""
from __future__ import annotations

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


def test_ordinal_hard_component_raises(X):
    t = OrdinalFeatures(subset=["ts"], components=["month"])
    t.fit(X)
    with pytest.raises(OnnxNotSupportedError, match="calendar arithmetic"):
        to_onnx_graph(t, errors="raise")


def test_ordinal_hard_component_coerce(X):
    """errors='coerce' skips hard components (covers the continue branch)."""
    t = OrdinalFeatures(subset=["ts"], components=["hour", "month"])
    t.fit(X)
    model = to_onnx_graph(t, errors="coerce")
    out = run_onnx(model, X)
    assert "ts__hour" in out and "ts__month" not in out


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


def test_cyclic_hard_component_raises(X):
    t = CyclicFeatures(subset=["ts"], components=["month"], angles=[0])
    t.fit(X)
    with pytest.raises(OnnxNotSupportedError, match="calendar arithmetic"):
        to_onnx_graph(t, errors="raise")


def test_cyclic_hard_component_coerce(X):
    """errors='coerce' skips hard components (covers the continue branch)."""
    t = CyclicFeatures(subset=["ts"], components=["hour", "month"], angles=[0])
    t.fit(X)
    model = to_onnx_graph(t, errors="coerce")
    out = run_onnx(model, X)
    assert "ts__hour__sin0" in out and "ts__month__sin0" not in out


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


def test_timebin_hard_bin_raises(X):
    t = TimeBinFeatures(subset=["ts"], bin_types=["season"])
    t.fit(X)
    with pytest.raises(OnnxNotSupportedError, match="calendar arithmetic"):
        to_onnx_graph(t, errors="raise")


def test_timebin_drop_columns(X):
    t = TimeBinFeatures(subset=["ts"], bin_types=["part_of_day"], drop_columns=True)
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert "ts" not in out and "ts__part_of_day" in out


# ── Pipeline tests (exercise get_input/output_onnx_type hooks) ────────────────

def test_ordinal_in_pipeline(X):
    """pipeline_to_onnx calls get_input/output_onnx_type for OrdinalFeatures."""
    pipe = Pipeline(steps=[("of", OrdinalFeatures(subset=["ts"], components=["hour", "minute"]))])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "ts__hour" in out and "ts__minute" in out
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
