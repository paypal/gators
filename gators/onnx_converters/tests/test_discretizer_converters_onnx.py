from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.discretizers import (
    CustomDiscretizer, EqualLengthDiscretizer, EqualSizeDiscretizer,
    GeometricDiscretizer, QuantileDiscretizer,
)
from gators.onnx_converters import to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "B": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})


def _compare_disc(t, df, atol=1e-5):
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    polars_out = t.transform(df)
    for col in polars_out.columns:
        np.testing.assert_allclose(
            onnx_out[col].astype(np.float64),
            polars_out[col].cast(pl.Float64).to_numpy(allow_copy=True),
            atol=atol,
            err_msg=f"{type(t).__name__}.{col}",
        )


def test_equal_length_discretizer(df):
    t = EqualLengthDiscretizer(num_bins=3, subset=["A", "B"], as_numerics=True)
    t.fit(df)
    _compare_disc(t, df)


def test_equal_size_discretizer(df):
    t = EqualSizeDiscretizer(num_bins=3, subset=["A", "B"], as_numerics=True)
    t.fit(df)
    _compare_disc(t, df)


def test_quantile_discretizer(df):
    t = QuantileDiscretizer(num_bins=3, subset=["A", "B"], as_numerics=True)
    t.fit(df)
    _compare_disc(t, df)


def test_geometric_discretizer():
    X = pl.DataFrame({"A": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
    t = GeometricDiscretizer(num_bins=3, subset=["A"], as_numerics=True)
    t.fit(X)
    _compare_disc(t, X)


def test_custom_discretizer(df):
    t = CustomDiscretizer(bins={"A": [2.0, 4.0], "B": [2.0, 4.0]}, as_numerics=True)
    t.fit(df)
    _compare_disc(t, df)


def _compare_disc_strings(t, df):
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    polars_out = t.transform(df)
    for col in polars_out.columns:
        np.testing.assert_array_equal(
            onnx_out[col].astype(str),
            polars_out[col].to_numpy(allow_copy=True).astype(str),
            err_msg=f"{type(t).__name__}.{col}",
        )


def test_as_numerics_false_equal_length(df):
    t = EqualLengthDiscretizer(num_bins=3, subset=["A", "B"], as_numerics=False)
    t.fit(df)
    _compare_disc_strings(t, df)


def test_as_numerics_false_geometric():
    X = pl.DataFrame({"A": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
    t = GeometricDiscretizer(num_bins=3, subset=["A"], as_numerics=False)
    t.fit(X)
    _compare_disc_strings(t, X)


def test_as_numerics_false_quantile(df):
    t = QuantileDiscretizer(num_bins=3, subset=["A"], as_numerics=False)
    t.fit(df)
    _compare_disc_strings(t, df)


def test_as_numerics_false_not_inplace(df):
    t = EqualLengthDiscretizer(num_bins=3, subset=["A"], as_numerics=False,
                               inplace=False, drop_columns=False)
    t.fit(df)
    _compare_disc_strings(t, df)


def test_as_numerics_false_constant_column():
    """Empty bins (constant column) must map bin-index 0 → the single label."""
    X = pl.DataFrame({"A": [5.0, 5.0, 5.0]})
    t = CustomDiscretizer(bins={"A": []}, as_numerics=False)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    polars_out = t.transform(X)
    np.testing.assert_array_equal(
        onnx_out["A"].astype(str), polars_out["A"].to_numpy(allow_copy=True).astype(str)
    )


def test_as_numerics_false_single_edge():
    """2-bin (1 edge) as_numerics=False — covers the single-term Identity path."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    t = EqualLengthDiscretizer(num_bins=2, subset=["A"], as_numerics=False)
    t.fit(X)
    assert len(t._bins["A"]) == 1
    _compare_disc_strings(t, X)


def test_as_numerics_false_get_output_onnx_type(df):
    from gators.onnx_converters import get_output_onnx_type
    from onnx import TensorProto
    t = EqualLengthDiscretizer(num_bins=3, subset=["A"], as_numerics=False)
    t.fit(df)
    assert get_output_onnx_type(t, "A") == TensorProto.STRING


def test_as_numerics_true_get_output_onnx_type(df):
    from gators.onnx_converters import get_output_onnx_type
    from onnx import TensorProto
    t = EqualLengthDiscretizer(num_bins=3, subset=["A"], as_numerics=True)
    t.fit(df)
    assert get_output_onnx_type(t, "A") != TensorProto.STRING


def test_discretizer_passthrough(df):
    """Columns not in subset must pass through unchanged."""
    t = EqualLengthDiscretizer(num_bins=3, subset=["A"], as_numerics=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    np.testing.assert_allclose(onnx_out["B"].astype(float), df["B"].to_numpy(allow_copy=True), atol=1e-5)


def test_get_output_columns_inplace(df):
    from gators.onnx_converters import get_output_columns
    t = EqualLengthDiscretizer(num_bins=3, subset=["A"], as_numerics=True)
    t.fit(df)
    out = get_output_columns(t, list(df.columns))
    assert out == list(df.columns)  # inplace=True: schema unchanged


def test_get_output_columns_not_inplace(df):
    from gators.onnx_converters import get_output_columns
    t = EqualLengthDiscretizer(num_bins=3, subset=["A"], as_numerics=True, inplace=False)
    t.fit(df)
    out = get_output_columns(t, list(df.columns))
    assert "B" in out
    assert any("disc" in c or "A" in c for c in out)


def test_discretizer_single_edge_two_bins():
    """2-bin discretizer has 1 edge → single Cast+Identity path (line 140)."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    t = EqualLengthDiscretizer(num_bins=2, subset=["A"], as_numerics=True)
    t.fit(X)
    assert len(t._bins["A"]) == 1  # exactly 1 edge
    _compare_disc(t, X)


def test_discretizer_not_inplace_drop_columns(df):
    """inplace=False, drop_columns=True: output schema uses new col names."""
    from gators.onnx_converters import get_output_columns
    t = EqualLengthDiscretizer(num_bins=3, subset=["A"], as_numerics=True,
                               inplace=False, drop_columns=True)
    t.fit(df)
    out = get_output_columns(t, list(df.columns))
    # Original 'A' dropped; new disc column + 'B' present
    assert "A" not in out
    assert "B" in out


def test_discretizer_not_inplace_no_drop(df):
    """inplace=False, drop_columns=False: original + new columns coexist."""
    from gators.onnx_converters import get_output_columns
    t = EqualLengthDiscretizer(num_bins=3, subset=["A"], as_numerics=True,
                               inplace=False, drop_columns=False)
    t.fit(df)
    out = get_output_columns(t, list(df.columns))
    assert "A" in out
    assert "B" in out


def test_discretizer_empty_bins_constant_column():
    """Empty bins (no edges) → constant bin 0 path (lines 108-116)."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    t = CustomDiscretizer(bins={"A": []}, as_numerics=True)
    t.fit(X)
    assert t._bins["A"] == []
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out["A"], np.zeros(len(X), dtype=np.float32), atol=1e-5)


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_discretizer_passthrough_column():
    """Column not in bins must pass through the ONNX graph unchanged."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 20.0, 30.0, 40.0]})
    t = EqualLengthDiscretizer(num_bins=2, subset=["A"], as_numerics=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["B"], X["B"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_discretizer_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 4
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = EqualLengthDiscretizer(num_bins=2, subset=['A'], as_numerics=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))


def test_discretizer_passthrough_string_column():
    """String column not in bins must pass through as STRING, not fall back to FLOAT."""
    from gators.onnx_converters import get_output_onnx_type
    from onnx import TensorProto
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "c", "d"]})
    t = EqualLengthDiscretizer(num_bins=2, subset=["A"], as_numerics=True)
    t.fit(X)
    assert get_output_onnx_type(t, "cat") == TensorProto.STRING
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert list(onnx_out["cat"]) == X["cat"].to_list()
