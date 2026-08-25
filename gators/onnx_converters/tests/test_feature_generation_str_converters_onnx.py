from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.feature_generation_str import Split, SplitExtract, Lower, Upper, Length
from gators.onnx_converters import to_onnx_graph, pipeline_to_onnx
from gators.onnx_converters._exceptions import OnnxNotSupportedError
from gators.pipeline import Pipeline
from .conftest import run_onnx

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def assert_str_equal(onnx_out: dict[str, np.ndarray], expected: pl.DataFrame, cols: list[str]):
    for col in cols:
        np.testing.assert_array_equal(
            onnx_out[col],
            expected[col].fill_null("").to_numpy(allow_copy=True).astype(str),
            err_msg=f"Mismatch for column '{col}'",
        )


@pytest.fixture
def df():
    return pl.DataFrame({"full_name": ["John Doe", "Jane Smith Williams", "Alice Johnson"]})


@pytest.fixture
def df_multi():
    return pl.DataFrame({
        "first": ["John Doe", "Jane Smith Williams", "Alice Johnson"],
        "second": ["foo bar baz", "a b", "x y z"],
        "score": [1.0, 2.0, 3.0],
    })


# ── Split ──────────────────────────────────────────────────────────────────────

def test_split_drop_true(df):
    t = Split(subset=["full_name"], by=" ", max_splits=3, drop_columns=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_str_equal(onnx_out, expected, expected.columns)


def test_split_drop_false(df):
    t = Split(subset=["full_name"], by=" ", max_splits=2, drop_columns=False)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    split_cols = [c for c in expected.columns if "__split_" in c]
    assert_str_equal(onnx_out, expected, split_cols)


def test_split_multi_column(df_multi):
    t = Split(subset=["first", "second"], by=" ", max_splits=2, drop_columns=True)
    t.fit(df_multi)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_multi)
    expected = t.transform(df_multi)
    split_cols = [c for c in expected.columns if "__split_" in c]
    assert_str_equal(onnx_out, expected, split_cols)


def test_split_passthrough_float_col(df_multi):
    t = Split(subset=["first"], by=" ", max_splits=2, drop_columns=True)
    t.fit(df_multi)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_multi)
    expected = t.transform(df_multi)
    # score column passes through as float
    np.testing.assert_allclose(
        onnx_out["score"].astype(np.float64),
        expected["score"].to_numpy(allow_copy=True),
        atol=1e-6,
    )


def test_split_non_space_delimiter(df):
    df2 = pl.DataFrame({"col": ["a-b-c", "x-y", "p-q-r-s"]})
    t = Split(subset=["col"], by="-", max_splits=3, drop_columns=True)
    t.fit(df2)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df2)
    expected = t.transform(df2)
    assert_str_equal(onnx_out, expected, expected.columns)


# ── SplitExtract ──────────────────────────────────────────────────────────────

def test_split_extract_n0_drop_true(df):
    t = SplitExtract(subset=["full_name"], by=" ", n=0, drop_columns=True)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    assert_str_equal(onnx_out, expected, expected.columns)


def test_split_extract_n1_drop_false(df):
    t = SplitExtract(subset=["full_name"], by=" ", n=1, drop_columns=False)
    t.fit(df)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df)
    expected = t.transform(df)
    split_cols = [c for c in expected.columns if "__split_" in c]
    assert_str_equal(onnx_out, expected, split_cols)


def test_split_extract_multi_column(df_multi):
    t = SplitExtract(subset=["first", "second"], by=" ", n=1, drop_columns=True)
    t.fit(df_multi)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_multi)
    expected = t.transform(df_multi)
    split_cols = [c for c in expected.columns if "__split_" in c]
    assert_str_equal(onnx_out, expected, split_cols)


def test_split_extract_non_space_delimiter():
    df2 = pl.DataFrame({"col": ["a-b-c", "x-y-z", "p-q-r"]})
    t = SplitExtract(subset=["col"], by="-", n=2, drop_columns=True)
    t.fit(df2)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df2)
    expected = t.transform(df2)
    assert_str_equal(onnx_out, expected, expected.columns)


# ── Lower ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def df_case():
    return pl.DataFrame({"name": ["Hello World", "FOO BAR", "baz"], "val": [1.0, 2.0, 3.0]})


def test_lower_inplace(df_case):
    t = Lower(inplace=True, subset=["name"])
    t.fit(df_case)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_case)
    expected = t.transform(df_case)
    assert_str_equal(onnx_out, expected, ["name"])
    np.testing.assert_allclose(onnx_out["val"].astype(float), df_case["val"].to_numpy(allow_copy=True))


def test_lower_inplace_false_drop(df_case):
    t = Lower(inplace=False, drop_columns=True, subset=["name"])
    t.fit(df_case)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_case)
    expected = t.transform(df_case)
    assert_str_equal(onnx_out, expected, ["name__lower"])


def test_lower_inplace_false_keep(df_case):
    t = Lower(inplace=False, drop_columns=False, subset=["name"])
    t.fit(df_case)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_case)
    expected = t.transform(df_case)
    assert_str_equal(onnx_out, expected, ["name__lower"])
    # original "name" column passes through unchanged
    np.testing.assert_array_equal(onnx_out["name"], df_case["name"].to_numpy(allow_copy=True).astype(str))


# ── Upper ─────────────────────────────────────────────────────────────────────────

def test_upper_inplace(df_case):
    t = Upper(inplace=True, subset=["name"])
    t.fit(df_case)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_case)
    expected = t.transform(df_case)
    assert_str_equal(onnx_out, expected, ["name"])


def test_pipeline_upper_inplace(df_case):
    """_upper_output_type (line 161) called via pipeline_to_onnx."""
    pipe = Pipeline(steps=[("upper", Upper(inplace=True, subset=["name"]))])
    pipe.fit(df_case)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, df_case)
    assert onnx_out["name"][0] == "HELLO WORLD"


# ── Pipeline output-type hooks (Lower / Split / SplitExtract) ────────────────

def test_pipeline_lower_inplace(df_case):
    """_str_normalizer_output_type inplace=True branch (lines 71-72)."""
    pipe = Pipeline(steps=[("lower", Lower(inplace=True, subset=["name"]))])
    pipe.fit(df_case)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, df_case)
    assert onnx_out["name"][0] == "hello world"


def test_pipeline_lower_inplace_false_keep(df_case):
    """_str_normalizer_output_type inplace=False branch (lines 73-74)."""
    pipe = Pipeline(steps=[("lower", Lower(inplace=False, drop_columns=False, subset=["name"]))])
    pipe.fit(df_case)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, df_case)
    assert onnx_out["name__lower"][0] == "hello world"


def test_pipeline_split_int_passthrough():
    """_split_output_type (208-216) and _to_onnx_type FLOAT branch (180)."""
    X = pl.DataFrame({
        "text": ["hello world", "foo bar baz"],
        "count": pl.Series([10, 20], dtype=pl.Int32),
    })
    pipe = Pipeline(steps=[("split", Split(subset=["text"], by=" ", max_splits=2, drop_columns=True))])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    by_clean = "_"  # " ".replace(" ", "_")
    assert onnx_out[f"text__split_{by_clean}_0"][0] == "hello"


def test_pipeline_split_extract_output_type():
    """_split_extract_output_type (297-301)."""
    X = pl.DataFrame({"text": ["hello world", "foo bar"], "val": [1.0, 2.0]})
    pipe = Pipeline(steps=[("se", SplitExtract(subset=["text"], by=" ", n=1, drop_columns=True))])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    by_clean = "_"  # " ".replace(" ", "_")
    assert onnx_out[f"text__split_{by_clean}_1"][0] == "world"


# ── Length (not supported) ────────────────────────────────────────────────

def test_length_raises_onnx_not_supported():
    """_length_input_type (362) is hit during input-type scan; to_onnx_nodes raises (372)."""
    X = pl.DataFrame({"text": ["hello", "world"], "val": [1.0, 2.0]})
    t = Length(subset=["text"])
    t.fit(X)
    pipe = Pipeline(steps=[("length", t)])
    pipe.fit(X)
    with pytest.raises(OnnxNotSupportedError):
        pipeline_to_onnx(pipe)


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_str_transformer_passthrough_column():
    """Float column not in Lower subset passes through the ONNX graph unchanged."""
    X = pl.DataFrame({"cat": ["Hello", "World"], "val": [1.0, 2.0]})
    t = Lower(subset=["cat"])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    import numpy as np
    np.testing.assert_array_equal(onnx_out["val"], X["val"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_str_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 3
    X = pl.DataFrame({"cat": ["Hello", "World", "Test"],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = Lower(subset=['cat'])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))


# ── ExtractSubstring: no ONNX string-slice op, always coerces/raises ─────────

def test_extract_substring_raises():
    from gators.feature_generation_str import ExtractSubstring
    X = pl.DataFrame({"text": ["hello", "world"]})
    t = ExtractSubstring(subset=["text"], start=0, end=3)
    t.fit(X)
    with pytest.raises(OnnxNotSupportedError):
        to_onnx_graph(t)


def test_extract_substring_coerce_identity_passthrough():
    from gators.feature_generation_str import ExtractSubstring
    from gators.onnx_converters import get_output_columns, get_input_onnx_type, get_output_onnx_type
    from onnx import TensorProto

    X = pl.DataFrame({"text": ["hello", "world"], "other": ["a", "b"]})
    t = ExtractSubstring(subset=["text"], start=0, end=3)
    t.fit(X)

    assert get_output_columns(t, list(X.columns)) == ["text", "other", "text__start0_end3"]
    assert get_input_onnx_type(t, "text") == TensorProto.STRING
    assert get_output_onnx_type(t, "text__start0_end3") == TensorProto.STRING
    assert get_output_onnx_type(t, "other") == TensorProto.STRING  # passthrough, not declared

    model = to_onnx_graph(t, errors="coerce")
    onnx_out = run_onnx(model, X)
    # Coerced to Identity: the "extracted" column is just the original string, unsliced.
    np.testing.assert_array_equal(onnx_out["text__start0_end3"], X["text"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["other"], X["other"].to_numpy(allow_copy=True))
