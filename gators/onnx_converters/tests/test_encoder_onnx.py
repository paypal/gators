from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.encoders import OrdinalEncoder, TargetEncoder, RareCategoryEncoder, OneHotEncoder, WOEEncoder, CountEncoder, HashEncoder
from gators.imputers import NumericImputer
from gators.pipeline import Pipeline
from gators.onnx_converters import to_onnx_graph, pipeline_to_onnx
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df_str():
    return pl.DataFrame({
        "cat": ["foo", "bar", "foo", "baz", "bar"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def target():
    return pl.Series("y", [1.0, 0.0, 1.0, 0.0, 1.0])


# ── OrdinalEncoder (no y needed) ──────────────────────────────────────────────

def test_ordinal_encoder_string_column(df_str):
    enc = OrdinalEncoder(subset=["cat"])
    enc.fit(df_str)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_str)
    expected = enc.transform(df_str)

    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_ordinal_encoder_unknown_category(df_str):
    """Unknown category must produce 0.0 (matching replace_strict default)."""
    enc = OrdinalEncoder(subset=["cat"])
    enc.fit(df_str)
    model = to_onnx_graph(enc)

    X_new = pl.DataFrame({"cat": ["foo", "UNKNOWN"], "num": [1.0, 2.0]})
    onnx_out = run_onnx(model, X_new)

    assert float(onnx_out["cat"][1]) == pytest.approx(0.0)


def test_ordinal_encoder_passthrough_float(df_str):
    """Float columns not in subset must pass through unchanged."""
    enc = OrdinalEncoder(subset=["cat"])
    enc.fit(df_str)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_str)

    np.testing.assert_allclose(
        onnx_out["num"].astype(float),
        df_str["num"].to_numpy(allow_copy=True),
        atol=1e-5,
    )


# ── TargetEncoder (requires y) ────────────────────────────────────────────────

def test_target_encoder_values(df_str, target):
    enc = TargetEncoder(subset=["cat"])
    enc.fit(df_str, y=target)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_str)
    expected = enc.transform(df_str)

    assert_onnx_close(onnx_out, expected, atol=1e-5)


# ── Boolean column handling ───────────────────────────────────────────────────

def test_ordinal_encoder_boolean_column():
    """Boolean columns arrive as float (0.0/1.0) and use keys_floats LabelEncoder."""
    X = pl.DataFrame({"flag": [True, False, True, False], "val": [1.0, 2.0, 3.0, 4.0]})
    enc = OrdinalEncoder(subset=["flag"])
    enc.fit(X)
    model = to_onnx_graph(enc)

    # Pass boolean column as float
    X_float = pl.DataFrame({"flag": [1.0, 0.0, 1.0, 0.0], "val": X["val"]})
    onnx_out = run_onnx(model, X_float)

    # Verify the two distinct values map to the two encoded values
    true_val = onnx_out["flag"][0]
    false_val = onnx_out["flag"][1]
    assert true_val != false_val
    assert true_val != pytest.approx(0.0)
    assert false_val != pytest.approx(0.0)


# ── RareCategoryEncoder ───────────────────────────────────────────────────────

@pytest.fixture
def df_rare():
    return pl.DataFrame({
        "cat": ["foo", "foo", "foo", "bar", "baz"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def _assert_string_equal(onnx_out: dict, expected: pl.DataFrame) -> None:
    for col in expected.columns:
        np.testing.assert_array_equal(
            onnx_out[col],
            expected[col].to_numpy(allow_copy=True).astype(str),
            err_msg=f"Mismatch for column '{col}'",
        )


def test_rare_encoder_inplace(df_rare):
    enc = RareCategoryEncoder(min_count=2, inplace=True)
    enc.fit(df_rare)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_rare)
    expected = enc.transform(df_rare)

    _assert_string_equal({"cat": onnx_out["cat"]}, expected.select("cat"))
    np.testing.assert_allclose(onnx_out["num"].astype(float), df_rare["num"].to_numpy(allow_copy=True))


def test_rare_encoder_non_inplace_drop(df_rare):
    enc = RareCategoryEncoder(min_count=2, inplace=False, drop_columns=True)
    enc.fit(df_rare)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_rare)
    expected = enc.transform(df_rare)

    _assert_string_equal({"cat__encode_rare": onnx_out["cat__encode_rare"]}, expected.select("cat__encode_rare"))


def test_rare_encoder_non_inplace_keep(df_rare):
    enc = RareCategoryEncoder(min_count=2, inplace=False, drop_columns=False)
    enc.fit(df_rare)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_rare)
    expected = enc.transform(df_rare)

    _assert_string_equal(
        {"cat__encode_rare": onnx_out["cat__encode_rare"]},
        expected.select("cat__encode_rare"),
    )
    np.testing.assert_array_equal(onnx_out["cat"], df_rare["cat"].to_numpy(allow_copy=True).astype(str))


def test_rare_encoder_non_rare_passthrough(df_rare):
    """Non-rare categories must be returned unchanged."""
    enc = RareCategoryEncoder(min_count=2, inplace=True)
    enc.fit(df_rare)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_rare)
    # "foo" is frequent → must remain "foo"
    assert onnx_out["cat"][0] == "foo"


def test_rare_encoder_rare_replaced(df_rare):
    """Rare categories must be replaced with the default string."""
    enc = RareCategoryEncoder(min_count=2, inplace=True)
    enc.fit(df_rare)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_rare)
    # "bar" and "baz" each appear once → rare → replaced with default
    assert onnx_out["cat"][3] == enc.default
    assert onnx_out["cat"][4] == enc.default


def test_rare_encoder_no_rare_categories():
    """When no categories are rare, all values should pass through."""
    X = pl.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
    enc = RareCategoryEncoder(min_count=1, inplace=True)
    enc.fit(X)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["cat"], X["cat"].to_numpy(allow_copy=True).astype(str))


# ── OneHotEncoder ─────────────────────────────────────────────────────────────

def test_ohe_basic(df_str):
    enc = OneHotEncoder(subset=["cat"])
    enc.fit(df_str)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_str)
    expected = enc.transform(df_str)

    assert_onnx_close(onnx_out, expected, atol=1e-5)


# ── OrdinalEncoder inplace=False ──────────────────────────────────────────────

def test_ordinal_encoder_inplace_false_drop(df_str):
    enc = OrdinalEncoder(subset=["cat"], inplace=False, drop_columns=True)
    enc.fit(df_str)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_str)
    expected = enc.transform(df_str)

    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_ordinal_encoder_inplace_false_keep(df_str):
    enc = OrdinalEncoder(subset=["cat"], inplace=False, drop_columns=False)
    enc.fit(df_str)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, df_str)
    expected = enc.transform(df_str)

    assert_onnx_close(
        {k: v for k, v in onnx_out.items() if k != "cat"},
        expected.drop("cat"),
        atol=1e-5,
    )
    np.testing.assert_array_equal(onnx_out["cat"], df_str["cat"].to_numpy(allow_copy=True).astype(str))


# ── RareCategoryEncoder via pipeline (exercises get_output_onnx_type) ─────────

def test_pipeline_rare_encoder_inplace(df_rare):
    """STRING output type hook is called via pipeline_to_onnx for inplace=True."""
    pipe = Pipeline(steps=[("rare_enc", RareCategoryEncoder(min_count=2, inplace=True))])
    pipe.fit(df_rare)
    model = pipeline_to_onnx(pipe)

    onnx_out = run_onnx(model, df_rare)
    expected = pipe.transform(df_rare)

    _assert_string_equal({"cat": onnx_out["cat"]}, expected.select("cat"))
    np.testing.assert_allclose(onnx_out["num"].astype(float), df_rare["num"].to_numpy(allow_copy=True))


def test_pipeline_rare_encoder_non_inplace_keep(df_rare):
    """STRING output type for renamed column (inplace=False) is covered via pipeline_to_onnx."""
    pipe = Pipeline(steps=[("rare_enc", RareCategoryEncoder(min_count=2, inplace=False, drop_columns=False))])
    pipe.fit(df_rare)
    model = pipeline_to_onnx(pipe)

    onnx_out = run_onnx(model, df_rare)
    expected_rare = pipe.transform(df_rare)["cat__encode_rare"].to_numpy(allow_copy=True).astype(str)

    np.testing.assert_array_equal(onnx_out["cat__encode_rare"], expected_rare)


# ── WOEEncoder on Boolean column (is_bool_str keys) ──────────────────────────

def test_woe_encoder_boolean_column():
    """WOEEncoder on a Boolean column uses 'true'/'false' string keys mapped to FLOAT ONNX input."""
    X = pl.DataFrame({"flag": [True, False, True, False, True], "val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pl.Series([1, 0, 1, 0, 1])
    enc = WOEEncoder(subset=["flag"])
    enc.fit(X, y=y)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, X)
    expected = enc.transform(X)

    assert_onnx_close(onnx_out, expected.select(["flag", "val"]), atol=1e-4)


# ── CountEncoder on numeric column (float mapping keys) ──────────────────────

def test_count_encoder_numeric_column():
    """CountEncoder on a Float32 column produces numeric (float) mapping keys."""
    X = pl.DataFrame({
        "a": pl.Series([1.0, 2.0, 1.0, 3.0, 2.0], dtype=pl.Float32),
        "b": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float32),
    })
    enc = CountEncoder(subset=["a"])
    enc.fit(X)
    model = to_onnx_graph(enc)

    onnx_out = run_onnx(model, X)
    expected = enc.transform(X)

    assert_onnx_close(onnx_out, expected, atol=1e-5)

# ── Pipeline: imputer (numeric) + encoder (categorical) ───────────────────────

def test_pipeline_imputer_and_encoder(df_str, target):
    """Mixed pipeline: NumericImputer on 'num', TargetEncoder on 'cat'."""
    X = df_str.with_columns(pl.Series("num", [1.0, None, 3.0, 4.0, None]))

    pipe = Pipeline(steps=[
        ("imputer", NumericImputer(strategy="median", subset=["num"])),
        ("encoder", TargetEncoder(subset=["cat"])),
    ])
    pipe.fit(X, y=target)
    model = pipeline_to_onnx(pipe)

    onnx_out = run_onnx(model, X)

    # Manually reproduce expected
    X_imp = pipe.steps[0][1].transform(X)
    X_enc = pipe.steps[1][1].transform(X_imp)
    assert_onnx_close(onnx_out, X_enc, atol=1e-4)


# ── Identity pass-through: non-subset columns must be unchanged ───────────────

def test_encoder_passthrough_column(df_str):
    """Float column not in encoder subset passes through the ONNX graph unchanged."""
    t = OrdinalEncoder(subset=["cat"])
    t.fit(df_str)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, df_str)
    np.testing.assert_array_equal(onnx_out["num"], df_str["num"].to_numpy(allow_copy=True))


# ── Float / Int / Bool pass-through dtypes ───────────────────────────────

def test_encoder_passthrough_float_int_bool():
    """Float32, Float64, Int, and Bool columns not in subset pass through unchanged."""
    n = 3
    X = pl.DataFrame({"cat": ["a", "b", "a"],
        "float32_pass": pl.Series([10.0, 20.0, 30.0, 40.0][:n], dtype=pl.Float32),
        "float64_pass": pl.Series([1.5,  2.5,  3.5,  4.5][:n],  dtype=pl.Float64),
        "int_pass":     pl.Series([100,  200,  300,  400][:n],   dtype=pl.Int64),
        "bool_pass":    pl.Series([True, False, True, False][:n], dtype=pl.Boolean),
    })
    t = OrdinalEncoder(subset=['cat'])
    t.fit(X)
    model = to_onnx_graph(t)
    onnx_out = run_onnx(model, X)
    np.testing.assert_array_equal(onnx_out["float32_pass"], X["float32_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["float64_pass"], X["float64_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["int_pass"].astype(np.int64),  X["int_pass"].to_numpy(allow_copy=True))
    np.testing.assert_array_equal(onnx_out["bool_pass"].astype(bool),     X["bool_pass"].to_numpy(allow_copy=True))


# ── HashEncoder ───────────────────────────────────────────────────────────────

def test_hash_encoder_inplace(df_str):
    enc = HashEncoder(n_features=8, inplace=True)
    enc.fit(df_str)
    model = to_onnx_graph(enc)
    onnx_out = run_onnx(model, df_str)
    expected = enc.transform(df_str)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_hash_encoder_inplace_false_drop(df_str):
    enc = HashEncoder(n_features=8, inplace=False, drop_columns=True)
    enc.fit(df_str)
    model = to_onnx_graph(enc)
    onnx_out = run_onnx(model, df_str)
    expected = enc.transform(df_str)
    assert_onnx_close(onnx_out, expected, atol=1e-5)


def test_hash_encoder_inplace_false_keep(df_str):
    enc = HashEncoder(n_features=8, inplace=False, drop_columns=False)
    enc.fit(df_str)
    model = to_onnx_graph(enc)
    onnx_out = run_onnx(model, df_str)
    expected = enc.transform(df_str)
    # Numeric passthrough is identical; compare encoded columns only
    assert_onnx_close(
        {"cat__hash": onnx_out["cat__hash"]},
        expected.select(["cat__hash"]),
        atol=1e-5,
    )
    np.testing.assert_array_equal(onnx_out["cat"], df_str["cat"].to_numpy(allow_copy=True).astype(str))


def test_hash_encoder_unknown_value(df_str):
    """Values unseen during fit fall back to bucket 0."""
    enc = HashEncoder(n_features=8, inplace=True)
    enc.fit(df_str)
    model = to_onnx_graph(enc)
    X_new = pl.DataFrame({"cat": ["UNSEEN_VALUE"], "num": [1.0]})
    onnx_out = run_onnx(model, X_new)
    assert float(onnx_out["cat"][0]) == pytest.approx(0.0)


def test_hash_encoder_passthrough(df_str):
    """Float column not in subset passes through unchanged."""
    enc = HashEncoder(n_features=8, subset=["cat"])
    enc.fit(df_str)
    model = to_onnx_graph(enc)
    onnx_out = run_onnx(model, df_str)
    np.testing.assert_allclose(onnx_out["num"].astype(float), df_str["num"].to_numpy(allow_copy=True), atol=1e-6)


def test_hash_encoder_in_pipeline(df_str):
    """pipeline_to_onnx calls get_output_onnx_type for HashEncoder."""
    pipe = Pipeline(steps=[("hash", HashEncoder(n_features=8, inplace=True))])
    pipe.fit(df_str)
    out = run_onnx(pipeline_to_onnx(pipe), df_str)
    exp = pipe.transform(df_str)
    assert_onnx_close({"cat": out["cat"]}, exp.select(["cat"]), atol=1e-5)
