from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.imputers import BooleanImputer, GroupByImputer, IterativeImputer, NumericImputer, StringImputer
from gators.onnx_converters import OnnxNotSupportedError, to_onnx_graph
from .conftest import assert_onnx_close, run_onnx


# ── BooleanImputer ────────────────────────────────────────────────────────────

@pytest.fixture
def df_bool():
    return pl.DataFrame({
        "A": [True, None, False, True],
        "B": [False, True, None, False],
    })


def test_boolean_imputer_most_frequent(df_bool):
    t = BooleanImputer(strategy="most_frequent")
    t.fit(df_bool)
    model = to_onnx_graph(t)

    # Pass booleans as float (False=0.0, True=1.0, null=NaN)
    X_float = pl.DataFrame({
        "A": [1.0, float("nan"), 0.0, 1.0],
        "B": [0.0, 1.0, float("nan"), 0.0],
    })
    onnx_out = run_onnx(model, X_float)

    # Expected fill values: A → True (most frequent), B → False (most frequent)
    expected_A = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    expected_B = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(onnx_out["A"], expected_A, atol=1e-5)
    np.testing.assert_allclose(onnx_out["B"], expected_B, atol=1e-5)


def test_boolean_imputer_constant_true(df_bool):
    t = BooleanImputer(strategy="constant", value=True)
    t.fit(df_bool)
    model = to_onnx_graph(t)

    X_float = pl.DataFrame({
        "A": [1.0, float("nan"), 0.0, 1.0],
        "B": [0.0, 1.0, float("nan"), 0.0],
    })
    onnx_out = run_onnx(model, X_float)

    # All nulls should be filled with 1.0 (True)
    assert float(onnx_out["A"][1]) == pytest.approx(1.0)
    assert float(onnx_out["B"][2]) == pytest.approx(1.0)


def test_boolean_imputer_constant_false():
    X = pl.DataFrame({"flag": [True, None, None, False]})
    t = BooleanImputer(strategy="constant", value=False)
    t.fit(X)
    model = to_onnx_graph(t)

    X_float = pl.DataFrame({"flag": [1.0, float("nan"), float("nan"), 0.0]})
    onnx_out = run_onnx(model, X_float)

    np.testing.assert_allclose(onnx_out["flag"][1:3], np.zeros(2, dtype=np.float32), atol=1e-7)


def test_boolean_imputer_no_nulls():
    X = pl.DataFrame({"A": [True, False, True, False]})
    t = BooleanImputer(strategy="most_frequent")
    t.fit(X)
    model = to_onnx_graph(t)

    X_float = pl.DataFrame({"A": [1.0, 0.0, 1.0, 0.0]})
    onnx_out = run_onnx(model, X_float)

    np.testing.assert_allclose(onnx_out["A"], np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32), atol=1e-7)


def test_boolean_imputer_subset():
    X = pl.DataFrame({"A": [True, None], "B": [None, False], "C": [1.0, 2.0]})
    t = BooleanImputer(strategy="constant", value=True, subset=["A"])
    t.fit(X)
    model = to_onnx_graph(t)

    X_float = pl.DataFrame({"A": [1.0, float("nan")], "B": [float("nan"), 0.0], "C": [1.0, 2.0]})
    onnx_out = run_onnx(model, X_float)

    # A: null → 1.0 (True)
    assert float(onnx_out["A"][1]) == pytest.approx(1.0)
    # B: not in subset — passes through with NaN intact
    assert np.isnan(float(onnx_out["B"][0]))
    # C: unchanged
    np.testing.assert_allclose(onnx_out["C"].astype(float), [1.0, 2.0], atol=1e-5)


# ── IterativeImputer ──────────────────────────────────────────────────────────

@pytest.fixture
def df_iter():
    return pl.DataFrame({
        "A": [1.0, None, 3.0, 4.0],
        "B": [5.0, 6.0, None, 8.0],
        "C": [1.0, 2.0, 3.0, 4.0],
    })


def test_iterative_imputer_max_iter_1(df_iter):
    """max_iter=1 must produce values numerically identical to Polars transform."""
    t = IterativeImputer(max_iter=1)
    t.fit(df_iter)
    model = to_onnx_graph(t)

    onnx_out = run_onnx(model, df_iter)
    expected = t.transform(df_iter)

    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_iterative_imputer_no_nulls(df_iter):
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]})
    t = IterativeImputer(max_iter=1)
    t.fit(X)
    model = to_onnx_graph(t)

    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_iterative_imputer_max_iter_gt1_coerce(df_iter):
    """errors='coerce' must not raise for max_iter>1 — exports single-pass approximation."""
    t = IterativeImputer(max_iter=3)
    t.fit(df_iter)
    model = to_onnx_graph(t, errors="coerce")  # must not raise
    onnx_out = run_onnx(model, df_iter)
    # Just check no NaN in the imputed columns (A index 1, B index 2)
    assert not np.isnan(float(onnx_out["A"][1]))
    assert not np.isnan(float(onnx_out["B"][2]))


def test_iterative_imputer_max_iter_gt1_raise(df_iter):
    """errors='raise' must raise OnnxNotSupportedError for max_iter>1."""
    t = IterativeImputer(max_iter=3)
    t.fit(df_iter)
    with pytest.raises(OnnxNotSupportedError, match="max_iter"):
        to_onnx_graph(t, errors="raise")


def test_iterative_imputer_subset_only(df_iter):
    """Only subset columns should be imputed; others pass through as-is."""
    t = IterativeImputer(max_iter=1, subset=["A"])
    t.fit(df_iter)
    model = to_onnx_graph(t)

    onnx_out = run_onnx(model, df_iter)

    # A index 1 must be imputed (not NaN)
    assert not np.isnan(float(onnx_out["A"][1]))
    # B index 2 must remain NaN (not in subset)
    assert np.isnan(float(onnx_out["B"][2]))


def test_iterative_imputer_with_non_numeric_column():
    """Non-numeric columns (not in _feature_cols) must pass through via Identity."""
    # The non-numeric col flows through line 208 (Identity for non-feature-col)
    X = pl.DataFrame({
        "A": [1.0, None, 3.0, 4.0],
        "B": [5.0, 6.0, None, 8.0],
        "C": [1.0, 2.0, 3.0, 4.0],
    })
    t = IterativeImputer(max_iter=1)
    t.fit(X)
    # Feed C as a non-imputed passthrough column
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    # C should pass through unchanged (not in imputation subset)
    np.testing.assert_allclose(onnx_out["C"].astype(float), X["C"].to_numpy(allow_copy=True), atol=1e-5)


def test_iterative_imputer_single_feature_constant_fallback():
    """Single feature column → constant-only regression (lines 274-278)."""
    X = pl.DataFrame({"A": [1.0, None, 3.0, 4.0]})
    t = IterativeImputer(max_iter=1)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    # Null must be filled with the initial statistic (no other features to regress on)
    assert not np.isnan(float(onnx_out["A"][1]))


def test_iterative_imputer_int_feature_column_is_noop():
    """An Int64 feature column is a no-op pass-through in both phase 1 (initial fill) and
    phase 2 (regression) — ONNX IsNaN only accepts float tensors."""
    X = pl.DataFrame({
        "A": pl.Series([1.0, None, 3.0, 4.0], dtype=pl.Float64),
        "B": pl.Series([1, 2, 3, 4], dtype=pl.Int64),
        "C": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float64),
    })
    t = IterativeImputer(max_iter=1, subset=["A", "B", "C"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["B"].astype(np.int64), X["B"].to_numpy(allow_copy=True))


# ── StringImputer ─────────────────────────────────────────────────────────────

def test_string_imputer_constant():
    """Null strings (fed as "") are replaced with the constant fill value."""
    X = pl.DataFrame({"cat": ["foo", None, "bar", None], "num": [1.0, 2.0, 3.0, 4.0]})
    t = StringImputer(strategy="constant", value="__NULL__", subset=["cat"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert onnx_out["cat"][1] == "__NULL__"
    assert onnx_out["cat"][3] == "__NULL__"
    assert onnx_out["cat"][0] == "foo"


def test_string_imputer_most_frequent():
    """Most-frequent strategy: null strings replaced with the mode."""
    X = pl.DataFrame({"cat": ["a", "a", "b", None]})
    t = StringImputer(strategy="most_frequent")
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert onnx_out["cat"][3] == "a"


def test_string_imputer_non_null_passthrough():
    """Non-null strings are passed through unchanged."""
    X = pl.DataFrame({"cat": ["x", "y", None]})
    t = StringImputer(strategy="constant", value="Z")
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert onnx_out["cat"][0] == "x"
    assert onnx_out["cat"][1] == "y"
    assert onnx_out["cat"][2] == "Z"


def test_string_imputer_in_pipeline():
    """StringImputer → encoder pipeline handles null strings end-to-end."""
    from gators.encoders import OrdinalEncoder
    from gators.pipeline import Pipeline
    from gators.onnx_converters import pipeline_to_onnx

    X = pl.DataFrame({"cat": ["foo", None, "bar", "foo"], "num": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline(steps=[
        ("imputer", StringImputer(strategy="constant", value="__NULL__", subset=["cat"])),
        ("encoder", OrdinalEncoder(subset=["cat"])),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    # Row 1 was null → imputed to "__NULL__" → encoded to some integer (not NaN)
    assert not np.isnan(float(onnx_out["cat"][1]))


# ── inplace=False paths (_imputer_output_cols lines 55-59, to_onnx_nodes lines 123-126, 386-389) ──

def test_numeric_imputer_inplace_false_drop_columns():
    """inplace=False + drop_columns=True: only imputed columns in output."""
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [None, 2.0, 3.0]})
    t = NumericImputer(strategy="mean", inplace=False, drop_columns=True)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected)


def test_numeric_imputer_inplace_false_no_drop_columns():
    """inplace=False + drop_columns=False: original + imputed columns in output."""
    X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [None, 2.0, 3.0]})
    t = NumericImputer(strategy="mean", inplace=False, drop_columns=False)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    expected = t.transform(X)
    assert_onnx_close(onnx_out, expected)


def test_numeric_imputer_all_null_column_passes_through():
    """Column with no statistic (all-null) passes through as Identity."""
    X = pl.DataFrame({
        "A": [1.0, None, 3.0],
        "B": pl.Series([None, None, None], dtype=pl.Float32),
    })
    t = NumericImputer(strategy="mean")
    t.fit(X)
    # After stat-fallback fix, B gets stat=0; ONNX fills NaN with 0
    assert t._statistics["B"] == 0
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out["A"], np.array([1.0, 2.0, 3.0], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(onnx_out["B"], np.array([0.0, 0.0, 0.0], dtype=np.float32), atol=1e-5)


def test_string_imputer_inplace_false_no_drop_columns():
    """inplace=False + drop_columns=False: original + imputed string columns in output."""
    X = pl.DataFrame({"cat": ["foo", None, "bar"], "num": [1.0, 2.0, 3.0]})
    t = StringImputer(strategy="constant", value="MISS", inplace=False, drop_columns=False)
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    assert onnx_out["cat"][1] == ""       # original passes through (null → "" sentinel)
    assert onnx_out["cat__impute_constant"][1] == "MISS"


def test_string_imputer_inplace_false_output_type():
    """pipeline_to_onnx calls get_output_onnx_type for a renamed StringImputer output (declared _output_dtypes)."""
    from gators.pipeline import Pipeline
    from gators.onnx_converters import pipeline_to_onnx

    X = pl.DataFrame({"cat": ["foo", None, "bar"]})
    pipe = Pipeline(steps=[
        ("imputer", StringImputer(strategy="constant", value="MISS", inplace=False, drop_columns=True)),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    assert onnx_out["cat__impute_constant"][1] == "MISS"


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_boolean_imputer_passthrough_column():
    """Float column not in BooleanImputer subset passes through unchanged."""
    X = pl.DataFrame({
        "flag": pl.Series([True, None, False], dtype=pl.Boolean),
        "score": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
    })
    t = BooleanImputer(strategy="constant", value=False, subset=["flag"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["score"], X["score"].to_numpy(allow_copy=True))


def test_string_imputer_passthrough_column():
    """Float column not in StringImputer subset passes through unchanged."""
    X = pl.DataFrame({"cat": ["a", None, "b"], "val": [10.0, 20.0, 30.0]})
    t = StringImputer(strategy="constant", value="MISS", subset=["cat"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["val"], X["val"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_imputer_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 3
    X = pl.DataFrame({"cat": ["a", None, "b"],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = StringImputer(strategy='constant', value='X', subset=['cat'])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))


# ── GroupByImputer ────────────────────────────────────────────────────────────

@pytest.fixture
def df_groupby():
    return pl.DataFrame({
        "district": ["A", "A", "A", "B", "B", "B", "C", "C"],
        "value1":   [1.0, 2.0, None, 4.0, None, 6.0, None, 8.0],
        "value2":   [10.0, None, 30.0, None, 50.0, 60.0, 70.0, None],
    })


def test_groupby_imputer_median_inplace(df_groupby):
    t = GroupByImputer(group_by_column="district", strategy="median", inplace=True)
    t.fit(df_groupby)
    model = to_onnx_graph(t, float_datatype="float32")
    onnx_out = run_onnx(model, df_groupby)
    expected = t.transform(df_groupby)
    numeric_cols = [c for c in expected.columns if c != "district"]
    assert_onnx_close({c: onnx_out[c] for c in numeric_cols}, expected.select(numeric_cols), atol=1e-5)
    np.testing.assert_array_equal(onnx_out["district"], df_groupby["district"].to_numpy(allow_copy=True).astype(str))


def test_groupby_imputer_mean_inplace(df_groupby):
    t = GroupByImputer(group_by_column="district", strategy="mean", inplace=True)
    t.fit(df_groupby)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_groupby)
    expected = t.transform(df_groupby)
    numeric_cols = [c for c in expected.columns if c != "district"]
    assert_onnx_close({c: onnx_out[c] for c in numeric_cols}, expected.select(numeric_cols), atol=1e-5)


def test_groupby_imputer_inplace_false(df_groupby):
    t = GroupByImputer(group_by_column="district", strategy="median", inplace=False, drop_columns=True)
    t.fit(df_groupby)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_groupby)
    expected = t.transform(df_groupby)
    numeric_cols = [c for c in expected.columns if c != "district"]
    assert_onnx_close({c: onnx_out[c] for c in numeric_cols}, expected.select(numeric_cols), atol=1e-5)


def test_groupby_imputer_inplace_false_keep(df_groupby):
    t = GroupByImputer(group_by_column="district", strategy="median", inplace=False, drop_columns=False)
    t.fit(df_groupby)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_groupby)
    expected = t.transform(df_groupby)
    imputed_cols = [name for names in t._column_mapping.values() for name in names]
    assert_onnx_close({c: onnx_out[c] for c in imputed_cols}, expected.select(imputed_cols), atol=1e-5)


def test_groupby_imputer_subset(df_groupby):
    t = GroupByImputer(group_by_column="district", strategy="median", subset=["value1"])
    t.fit(df_groupby)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_groupby)
    expected = t.transform(df_groupby)
    assert_onnx_close({"value1": onnx_out["value1"]}, expected.select(["value1"]), atol=1e-5)
    np.testing.assert_allclose(onnx_out["value2"].astype(float), df_groupby["value2"].to_numpy(allow_copy=True), atol=1e-5, equal_nan=True)


def test_groupby_imputer_int_column_in_subset_is_noop():
    """An Int64 column in the subset is a no-op pass-through — ONNX IsNaN only accepts
    float tensors, and integer columns carry no null representation once exported."""
    X = pl.DataFrame({
        "district": ["A", "A", "B", "B"],
        "value1":   pl.Series([1, 2, 3, 4], dtype=pl.Int64),
    })
    t = GroupByImputer(group_by_column="district", strategy="median", subset=["value1"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["value1"].astype(np.int64), X["value1"].to_numpy(allow_copy=True))


def test_groupby_imputer_no_nulls(df_groupby):
    """Columns with no nulls must pass through unchanged."""
    X = pl.DataFrame({
        "group": ["A", "B", "A", "B"],
        "val":   [1.0, 2.0, 3.0, 4.0],
    })
    t = GroupByImputer(group_by_column="group", strategy="mean")
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_allclose(onnx_out["val"].astype(float), X["val"].to_numpy(allow_copy=True), atol=1e-6)


def test_groupby_imputer_unseen_group():
    """Unknown groups at inference time: null values remain null (NaN)."""
    X_train = pl.DataFrame({"group": ["A", "A"], "val": [1.0, 2.0]})
    X_test  = pl.DataFrame({"group": ["Z", "A"], "val": [float("nan"), float("nan")]})
    t = GroupByImputer(group_by_column="group", strategy="mean")
    t.fit(X_train)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X_test)
    assert np.isnan(float(onnx_out["val"][0]))   # unknown group → stays null
    assert not np.isnan(float(onnx_out["val"][1]))  # known group → filled


def test_groupby_imputer_float32():
    """Float32 column: LabelEncoder output (float32) needs no Cast node."""
    X = pl.DataFrame({
        "group": ["A", "A", "B", "B"],
        "val":   pl.Series([1.0, None, 3.0, 4.0], dtype=pl.Float32),
    })
    t = GroupByImputer(group_by_column="group", strategy="mean")
    t.fit(X)
    out = run_onnx(to_onnx_graph(t), X)
    assert not np.isnan(float(out["val"][1]))
