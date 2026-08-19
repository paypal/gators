from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gators.imputers import NumericImputer, BooleanImputer
from gators.scalers import StandardScaler
from gators.pipeline import Pipeline
from gators.onnx_converters import pipeline_to_onnx, get_output_columns, check_pipeline_onnx_compatibility
from .conftest import assert_onnx_close, run_onnx


@pytest.fixture
def df():
    return pl.DataFrame({
        "A": [1.0, None, 3.0, 4.0],
        "B": [None, 2.0, 3.0, 4.0],
    })


def test_pipeline_imputer_then_scaler(df):
    pipe = Pipeline(steps=[
        ("imputer", NumericImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipe.fit(df)
    model = pipeline_to_onnx(pipe)

    onnx_out = run_onnx(model, df)
    expected = pipe.transform(df)

    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_pipeline_single_step(df):
    """A one-step pipeline behaves identically to to_onnx_graph."""
    from gators.onnx_converters import to_onnx_graph

    imputer = NumericImputer(strategy="median")
    pipe = Pipeline(steps=[("imputer", imputer)])
    pipe.fit(df)

    model_pipe = pipeline_to_onnx(pipe)
    model_single = to_onnx_graph(pipe.steps[0][1])

    out_pipe = run_onnx(model_pipe, df)
    out_single = run_onnx(model_single, df)

    for col in df.columns:
        np.testing.assert_allclose(out_pipe[col], out_single[col], atol=1e-7)


def test_pipeline_no_nulls(df):
    """Pipeline on null-free data must produce the same result as the Polars pipeline."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]})
    pipe = Pipeline(steps=[
        ("imputer", NumericImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)

    onnx_out = run_onnx(model, X)
    expected = pipe.transform(X)

    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_pipeline_inplace_false_drop_columns():
    """inplace=False + drop_columns=True is fully supported: renamed columns flow to next step."""
    X = pl.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
    from gators.encoders import OrdinalEncoder
    pipe = Pipeline(steps=[
        ("enc", OrdinalEncoder(subset=["cat"], inplace=False, drop_columns=True)),
        ("imputer", NumericImputer(strategy="mean")),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    expected = pipe.transform(X)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_pipeline_inplace_false_drop_columns_coerce():
    """inplace=False + drop_columns=True with errors='coerce' must not raise."""
    X = pl.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
    from gators.encoders import OrdinalEncoder
    pipe = Pipeline(steps=[
        ("enc", OrdinalEncoder(subset=["cat"], inplace=False, drop_columns=True)),
        ("imputer", NumericImputer(strategy="mean")),
    ])
    pipe.fit(X)
    pipeline_to_onnx(pipe, errors="coerce")


def test_pipeline_ohe_woe_imputer_hhi():
    """Complex pipeline: NumericImputer → WOEEncoder → OneHotEncoder → HHIFeatures."""
    from gators.encoders import OneHotEncoder, WOEEncoder
    from gators.feature_generation import HHIFeatures

    X = pl.DataFrame({
        "cat1": ["A", "B", "A", "C", "B"],   # WOE encoded → inplace float
        "cat2": ["X", "Y", "X", "Y", "X"],   # OHE → binary columns
        "num1": [1.0, None, 3.0, 4.0, 5.0], # imputed
        "num2": [2.0, 3.0, None, 5.0, 6.0], # imputed
    })
    y = pl.Series("y", [1, 0, 1, 1, 0])

    pipe = Pipeline(steps=[
        ("imputer", NumericImputer(strategy="median", subset=["num1", "num2"])),
        ("woe",     WOEEncoder(subset=["cat1"])),
        ("ohe",     OneHotEncoder(subset=["cat2"])),
        ("hhi",     HHIFeatures(column_groups=[["num1", "num2"]])),
    ])
    pipe.fit(X, y=y)
    model = pipeline_to_onnx(pipe)

    onnx_out = run_onnx(model, X)
    expected = pipe.transform(X)

    assert set(onnx_out.keys()) == set(expected.columns)
    assert_onnx_close(onnx_out, expected, atol=1e-4)


def test_pipeline_to_scoring_onnx():
    """pipeline_to_scoring_onnx chains a Gators pipeline with a LightGBM ONNX model."""
    lgb = pytest.importorskip("lightgbm")
    onnxmltools = pytest.importorskip("onnxmltools")
    from onnxmltools.convert.common.data_types import FloatTensorType

    import numpy as np
    from gators.onnx_converters import pipeline_to_scoring_onnx

    X_train = pl.DataFrame({"A": [1.0, None, 3.0, 4.0, 5.0], "B": [5.0, 6.0, None, 8.0, 9.0]})
    y = np.array([1, 0, 1, 0, 1])

    pipe = Pipeline(steps=[
        ("imputer", NumericImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipe.fit(X_train)
    feature_columns = ["A", "B"]

    # Fit LightGBM on preprocessed features
    X_feat = pipe.transform(X_train).select(feature_columns).to_numpy().astype(np.float32)
    lgb_model = lgb.LGBMClassifier(n_estimators=3, max_depth=2)
    lgb_model.fit(X_feat, y)

    # Export to ONNX and combine with preprocessing
    model_onnx = onnxmltools.convert_lightgbm(
        lgb_model.booster_,
        initial_types=[("input", FloatTensorType([None, len(feature_columns)]))],
    )
    scoring_model = pipeline_to_scoring_onnx(pipe, model_onnx, feature_columns)

    # Run inference
    import onnxruntime as ort
    X_test = pl.DataFrame({"A": [2.0, 4.0], "B": [7.0, 8.0]})
    sess = ort.InferenceSession(scoring_model.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {
        "A__in": X_test["A"].to_numpy(allow_copy=True),
        "B__in": X_test["B"].to_numpy(allow_copy=True),
    }
    label, proba = sess.run(None, feeds)
    assert label.shape == (2,)
    assert len(proba) == 2


# ── check_pipeline_onnx_compatibility ────────────────────────────────────────

def test_check_pipeline_onnx_compatibility(df):
    pipe = Pipeline(steps=[
        ("imputer", NumericImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipe.fit(df)
    result = check_pipeline_onnx_compatibility(pipe)
    assert result == {"imputer": True, "scaler": True}


# ── get_output_columns with drop_columns=False ────────────────────────────────

def test_get_output_columns_drop_columns_false():
    """_base_transformer_output_columns returns input + new cols when drop_columns=False."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    t = StandardScaler(inplace=False, drop_columns=False)
    t.fit(X)
    result = get_output_columns(t, list(X.columns))
    assert result == ["A", "A__standard_scale"]


# ── Pipeline with renaming transformer (get_output_onnx_type reverse lookup) ──

def test_pipeline_scaler_inplace_false_drop_false():
    """pipeline_to_onnx triggers get_output_onnx_type reverse-_column_mapping lookup."""
    X = pl.DataFrame({"A": [1.0, None, 3.0, 4.0], "B": [5.0, 6.0, None, 8.0]})
    pipe = Pipeline(steps=[
        ("imputer", NumericImputer(strategy="median", inplace=False, drop_columns=True)),
    ])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe, errors="coerce")
    onnx_out = run_onnx(model, X)
    assert "A__impute_median" in onnx_out


# ── Pipeline with STRING column (graph_input_types STRING promotion) ──────────

def test_pipeline_string_column_type_promotion():
    """pipeline_to_onnx promotes a column to STRING when a step declares STRING input."""
    from gators.imputers import StringImputer
    X = pl.DataFrame({"cat": ["a", None, "b", "a"], "val": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline(steps=[("imputer", StringImputer(strategy="constant", value="MISS"))])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    assert onnx_out["cat"][1] == "MISS"


# ── Pipeline with BOOL→INT64 promotion ───────────────────────────────────────

def test_pipeline_bool_to_string_cast():
    """CastColumns(dtype=String) on a Boolean column promotes graph input to INT64."""
    from gators.data_cleaning import CastColumns
    X = pl.DataFrame({"flag": pl.Series([True, None, False], dtype=pl.Boolean), "val": [1.0, 2.0, 3.0]})
    pipe = Pipeline(steps=[("cast", CastColumns(subset=["flag"], dtype=pl.String))])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    assert onnx_out["flag"][0] == "true"
    assert onnx_out["flag"][1] == ""
    assert onnx_out["flag"][2] == "false"


# ── Pipeline with BooleanImputer (get_output_onnx_type hook) ─────────────────

def test_pipeline_boolean_imputer():
    """pipeline_to_onnx calls get_output_onnx_type(BooleanImputer, col)."""
    X = pl.DataFrame({"flag": [True, None, False, True], "val": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline(steps=[("imputer", BooleanImputer(strategy="most_frequent"))])
    pipe.fit(X)
    model = pipeline_to_onnx(pipe)
    onnx_out = run_onnx(model, X)
    assert not np.isnan(float(onnx_out["flag"][1]))


# ── pipeline_to_scoring_onnx with minimal ONNX model ─────────────────────────

def test_pipeline_to_scoring_onnx_minimal(df):
    """pipeline_to_scoring_onnx with a hand-crafted model covers the extra-opset loop."""
    import onnx
    import onnx.helper as oh
    from onnx import TensorProto
    from gators.onnx_converters import pipeline_to_scoring_onnx

    pipe = Pipeline(steps=[("imputer", NumericImputer(strategy="median"))])
    pipe.fit(df)

    # Minimal identity model with an extra opset domain not present in the preprocessing graph
    ml_graph = oh.make_graph(
        [oh.make_node("Identity", inputs=["X"], outputs=["output"])],
        "MinimalModel",
        [oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])],
        [oh.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
    )
    ml_model = oh.make_model(ml_graph, opset_imports=[
        oh.make_opsetid("", 17),
        oh.make_opsetid("com.test.extra", 1),
    ])
    onnx.checker.check_model(ml_model)

    scoring_model = pipeline_to_scoring_onnx(pipe, ml_model, feature_columns=["A", "B"])
    assert scoring_model is not None
    domains = {o.domain for o in scoring_model.opset_import}
    assert "com.test.extra" in domains


def test_pipeline_int64_type_promotion():
    """FLOAT/DOUBLE → INT64 promotion when a step declares INT64 (export.py line 167)."""
    from gators.data_cleaning import CastColumns
    from gators.feature_generation_dt import BusinessTimeFeatures
    X = pl.DataFrame({"ts_i64": [1705305600_000_000, 1705330800_000_000]})
    pipe = Pipeline(steps=[
        ("cast", CastColumns(subset=["ts_i64"], dtype=pl.Datetime)),
        ("btf",  BusinessTimeFeatures(subset=["ts_i64"], features=["is_business_hour"])),
    ])
    pipe.fit(X)
    out = run_onnx(pipeline_to_onnx(pipe), X)
    assert "ts_i64__is_business_hour" in out
