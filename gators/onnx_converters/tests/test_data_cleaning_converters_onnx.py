from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.data_cleaning import (
    CastColumns, CorrelationFilter, DropColumns, DropConstantColumns, DropDuplicateColumns,
    DropHighNaNRatio, DropNearConstantColumns, RenameColumns, RoundDigits, SelectColumns, VarianceFilter,
)
from gators.onnx_converters import get_output_columns, to_onnx_graph, pipeline_to_onnx
from gators.onnx_converters._exceptions import OnnxNotSupportedError
from gators.pipeline import Pipeline
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({
        "A": [1.1, 2.2, 3.3, 4.4],
        "B": [1.0, 1.0, 1.0, 1.0],  # constant
        "C": [4.4, 5.5, 6.6, 7.7],
    })


# ── DropColumns ───────────────────────────────────────────────────────────────

def test_drop_columns(df):
    t = DropColumns(subset=["B"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert set(onnx_out.keys()) == {"A", "C"}


def test_drop_columns_get_output_columns(df):
    t = DropColumns(subset=["B"])
    t.fit(df)
    assert get_output_columns(t, list(df.columns)) == ["A", "C"]


# ── SelectColumns ─────────────────────────────────────────────────────────────

def test_select_columns(df):
    t = SelectColumns(subset=["A", "C"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert set(onnx_out.keys()) == {"A", "C"}
    np.testing.assert_allclose(onnx_out["A"].astype(float), df["A"].to_numpy(allow_copy=True))
    np.testing.assert_allclose(onnx_out["C"].astype(float), df["C"].to_numpy(allow_copy=True))


def test_select_columns_get_output_columns(df):
    t = SelectColumns(subset=["A", "C"])
    t.fit(df)
    assert get_output_columns(t, list(df.columns)) == ["A", "C"]


def test_select_columns_preserves_order(df):
    """Output column order follows subset, not original DataFrame order."""
    t = SelectColumns(subset=["C", "A"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert list(onnx_out.keys()) == ["C", "A"]


# ── DropConstantColumns ───────────────────────────────────────────────────────

def test_drop_constant_columns(df):
    t = DropConstantColumns()
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert "B" not in onnx_out
    assert "A" in onnx_out and "C" in onnx_out


def test_drop_constant_get_output_columns(df):
    t = DropConstantColumns()
    t.fit(df)
    out = get_output_columns(t, list(df.columns))
    assert "B" not in out


# ── DropHighNaNRatio ──────────────────────────────────────────────────────────

def test_drop_high_nan_ratio():
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [None, None, 1.0], "C": [1.0, 2.0, 3.0]})
    t = DropHighNaNRatio(max_ratio=0.5)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert "B" not in onnx_out
    assert get_output_columns(t, list(X.columns)) == ["A", "C"]


# ── DropNearConstantColumns ───────────────────────────────────────────────────

def test_drop_near_constant_columns():
    X = pl.DataFrame({"A": [1.0, 1.0, 1.0, 2.0], "B": [1.0, 2.0, 3.0, 4.0]})
    t = DropNearConstantColumns()
    t.fit(X)
    out = get_output_columns(t, list(X.columns))
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert set(onnx_out.keys()) == set(out)


# ── DropDuplicateColumns ──────────────────────────────────────────────────────

def test_drop_duplicate_columns():
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 3.0], "C": [4.0, 5.0, 6.0]})
    t = DropDuplicateColumns()
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert len(onnx_out) < 3
    assert get_output_columns(t, list(X.columns)) is not None


# ── VarianceFilter ────────────────────────────────────────────────────────────

def test_variance_filter(df):
    t = VarianceFilter(min_var=0.5)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert "B" not in onnx_out
    assert get_output_columns(t, list(df.columns)) == ["A", "C"]


# ── CorrelationFilter ─────────────────────────────────────────────────────────

def test_correlation_filter():
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [1.0, 2.0, 3.0, 4.0], "C": [4.0, 3.0, 2.0, 1.0]})
    t = CorrelationFilter(max_corr=0.9)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    out_cols = get_output_columns(t, list(X.columns))
    assert set(onnx_out.keys()) == set(out_cols)


# ── RenameColumns ─────────────────────────────────────────────────────────────

def test_rename_columns(df):
    t = RenameColumns(column_mapping={"A": "AA", "C": "CC"})
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert "AA" in onnx_out and "CC" in onnx_out and "B" in onnx_out
    assert get_output_columns(t, list(df.columns)) == ["AA", "B", "CC"]


def test_rename_values_preserved(df):
    t = RenameColumns(column_mapping={"A": "AA"})
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    np.testing.assert_allclose(
        onnx_out["AA"].astype(float), df["A"].to_numpy(allow_copy=True), atol=1e-5
    )


def test_rename_string_column():
    df_str = pl.DataFrame({"name": ["alice", "bob"], "score": [1.0, 2.0]})
    t = RenameColumns(column_mapping={"name": "full_name"})
    t.fit(df_str)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_str)
    assert "full_name" in onnx_out
    np.testing.assert_array_equal(
        onnx_out["full_name"], df_str["name"].to_numpy(allow_copy=True).astype(str)
    )


def test_rename_partial_mapping(df):
    """Columns not in column_mapping pass through unchanged."""
    t = RenameColumns(column_mapping={"A": "X"})
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    assert "X" in onnx_out and "B" in onnx_out and "C" in onnx_out


# ── RoundDigits ───────────────────────────────────────────────────────────────

def test_round_digits(df):
    t = RoundDigits(n_digits=1, subset=["A", "C"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_round_digits_passthrough(df):
    t = RoundDigits(n_digits=0, subset=["A"])
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    np.testing.assert_allclose(onnx_out["B"].astype(float), df["B"].to_numpy(allow_copy=True), atol=1e-5)


# ── CastColumns ───────────────────────────────────────────────────────────────

@pytest.fixture
def df_cast():
    return pl.DataFrame({"text": ["1.5", "2.5", "3.5"], "val": [1.0, 2.0, 3.0]})


def test_cast_columns_str_to_float64_inplace(df_cast):
    t = CastColumns(subset=["text"], dtype=pl.Float64, inplace=True)
    t.fit(df_cast)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_cast)
    np.testing.assert_allclose(onnx_out["text"].astype(float), [1.5, 2.5, 3.5], atol=1e-6)


def test_cast_columns_float_to_float32_inplace(df):
    t = CastColumns(subset=["A", "C"], dtype=pl.Float32, inplace=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_cast_columns_inplace_false_drop(df_cast):
    t = CastColumns(subset=["text"], dtype=pl.Float64, inplace=False, drop_columns=True)
    t.fit(df_cast)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_cast)
    assert "text" not in onnx_out
    np.testing.assert_allclose(onnx_out["text__cast_float64"].astype(float), [1.5, 2.5, 3.5], atol=1e-6)


def test_cast_columns_inplace_false_keep(df_cast):
    t = CastColumns(subset=["text"], dtype=pl.Float64, inplace=False, drop_columns=False)
    t.fit(df_cast)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_cast)
    # original string column still present
    np.testing.assert_array_equal(onnx_out["text"], df_cast["text"].to_numpy(allow_copy=True).astype(str))
    np.testing.assert_allclose(onnx_out["text__cast_float64"].astype(float), [1.5, 2.5, 3.5], atol=1e-6)


def test_cast_columns_unsupported_dtype_raises(df):
    t = CastColumns(subset=["A"], dtype=pl.String, inplace=True)
    t.fit(df)
    with pytest.raises(OnnxNotSupportedError):
        to_onnx_graph(t)


def test_cast_columns_unsupported_dtype_coerce(df):
    t = CastColumns(subset=["A"], dtype=pl.String, inplace=True)
    t.fit(df)
    model = to_onnx_graph(t, errors="coerce")
    onnx_out = run_onnx(model, df)
    # coerce → Identity passthrough for all columns
    assert set(onnx_out.keys()) == set(df.columns)


def test_cast_columns_datetime_dtype_passthrough(df):
    """CastColumns(dtype=pl.Date/Datetime) is a no-op in ONNX — int64 IS the physical form."""
    t = CastColumns(subset=["A"], dtype=pl.Date, inplace=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    # Column passes through unchanged (its int64 values are the datetime physical representation)
    assert set(onnx_out.keys()) == set(df.columns)


def test_cast_columns_bool_to_string():
    X = pl.DataFrame({"flag": [True, False, True], "val": [1.0, 2.0, 3.0]})
    t = CastColumns(subset=["flag"], dtype=pl.String, inplace=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)["flag"].to_numpy(allow_copy=True)
    np.testing.assert_array_equal(onnx_out["flag"], expected)


def test_cast_columns_bool_to_string_with_null():
    X = pl.DataFrame({"flag": pl.Series([True, None, False], dtype=pl.Boolean), "val": [1.0, 2.0, 3.0]})
    t = CastColumns(subset=["flag"], dtype=pl.String, inplace=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    # null maps to "" (the empty-string null sentinel consumed by StringImputer downstream)
    np.testing.assert_array_equal(onnx_out["flag"], np.array(["true", "", "false"]))


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_data_cleaning_passthrough_column():
    """Column not in CastColumns subset passes through the ONNX graph unchanged."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]})
    t = CastColumns(subset=["A"], dtype=pl.Float32)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["B"], X["B"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_data_cleaning_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 3
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = CastColumns(subset=['A'], dtype=pl.Float32)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))


def test_cast_columns_passthrough_string_dtype():
    """String column not in CastColumns subset passes through unchanged (STRING in ONNX)."""
    X = pl.DataFrame({
        "target":   [1.0, 2.0, 3.0],
        "str_pass": ["alpha", "beta", "gamma"],
    })
    t = CastColumns(subset=["target"], dtype=pl.Float32)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["str_pass"].astype(str), X["str_pass"].to_numpy(allow_copy=True))


# ── Pipeline tests: exercise get_output_onnx_type hooks ──────────────────────

def test_rename_columns_in_pipeline():
    """pipeline_to_onnx calls get_output_onnx_type(RenameColumns, renamed_col)."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    pipe = Pipeline(steps=[("rename", RenameColumns(column_mapping={"A": "A_new"}))])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out["A_new"].astype(float), X["A"].to_numpy(allow_copy=True), atol=1e-5)


def test_cast_columns_in_pipeline():
    """pipeline_to_onnx calls get_output_onnx_type(CastColumns, col)."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    pipe = Pipeline(steps=[("cast", CastColumns(subset=["A"], dtype=pl.Float32, inplace=True))])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out["A"].astype(float), X["A"].to_numpy(allow_copy=True), atol=1e-5)


def test_cast_columns_categorical_dtype_coerce(df):
    """Non-temporal, non-numeric unsupported dtype with errors='coerce' → Identity passthrough."""
    t = CastColumns(subset=["A"], dtype=pl.Categorical, inplace=True)
    t.fit(df)
    model = to_onnx_graph(t, errors="coerce")
    onnx_out = run_onnx(model, df)
    assert set(onnx_out.keys()) == set(df.columns)


def test_cast_columns_categorical_dtype_raises(df):
    """Non-temporal, non-numeric unsupported dtype with errors='raise' → OnnxNotSupportedError."""
    t = CastColumns(subset=["A"], dtype=pl.Categorical, inplace=True)
    t.fit(df)
    with pytest.raises(OnnxNotSupportedError):
        to_onnx_graph(t, errors="raise")


def test_cast_columns_datetime_in_pipeline():
    """CastColumns(dtype=pl.Datetime) → pipeline_to_onnx calls get_output_onnx_type (INT64 path)."""
    from gators.feature_generation_dt import BusinessTimeFeatures
    X = pl.DataFrame({"ts_i64": [1705305600_000_000, 1705330800_000_000]})
    pipe = Pipeline(steps=[
        ("cast", CastColumns(subset=["ts_i64"], dtype=pl.Datetime)),
        ("btf",  BusinessTimeFeatures(subset=["ts_i64"], features=["is_business_hour"])),
    ])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "ts_i64__is_business_hour" in out
