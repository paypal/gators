"""
Tests for the Gators Pipeline class.
"""

import polars as pl
import pytest

import numpy as np
from gators.imputers import NumericImputer, StringImputer
from gators.pipeline import Pipeline
from gators.scalers import StandardScaler

def test_pipeline_creation():
    """Test that a Pipeline can be created."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
        (
            "string_imputer",
            StringImputer(strategy="constant", value="__NULL__", inplace=True),
        ),
    ]
    pipe = Pipeline(steps=steps)
    assert len(pipe) == 2
    assert "numeric_imputer" in pipe.named_steps
    assert "string_imputer" in pipe.named_steps


def test_pipeline_fit_transform():
    """Test fit_transform on a simple dataset."""
    # Create test data
    X = pl.DataFrame({"num_col": [1.0, 2.0, None, 4.0, 5.0], "str_col": ["a", "b", None, "d", "e"]})

    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
        (
            "string_imputer",
            StringImputer(strategy="constant", value="__NULL__", inplace=True),
        ),
    ]

    pipe = Pipeline(steps=steps)
    result = pipe.fit_transform(X)

    # Check no nulls remain
    assert result.null_count().sum_horizontal()[0] == 0
    assert isinstance(result, pl.DataFrame)


def test_pipeline_fit_transform_lazyframe():
    """Test fit_transform collects a LazyFrame input before fitting."""
    X = pl.LazyFrame({"num_col": [1.0, 2.0, None, 4.0, 5.0], "str_col": ["a", "b", None, "d", "e"]})

    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
        (
            "string_imputer",
            StringImputer(strategy="constant", value="__NULL__", inplace=True),
        ),
    ]

    pipe = Pipeline(steps=steps)
    result = pipe.fit_transform(X)

    assert isinstance(result, pl.DataFrame)
    assert result.null_count().sum_horizontal()[0] == 0


def test_pipeline_fit_then_transform():
    """Test separate fit and transform calls."""
    X_train = pl.DataFrame(
        {"num_col": [1.0, 2.0, None, 4.0, 5.0], "str_col": ["a", "b", None, "d", "e"]}
    )

    X_test = pl.DataFrame({"num_col": [None, 7.0, 8.0], "str_col": ["x", None, "z"]})

    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
        (
            "string_imputer",
            StringImputer(strategy="constant", value="__NULL__", inplace=True),
        ),
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(X_train)
    result = pipe.transform(X_test)

    # Check no nulls remain
    assert result.null_count().sum_horizontal()[0] == 0
    assert isinstance(result, pl.DataFrame)


def test_pipeline_get_set_params():
    """Test get_params and set_params."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps)
    params = pipe.get_params(deep=True)

    assert "steps" in params
    assert "verbose" in params
    assert "numeric_imputer__strategy" in params

    # Test set_params
    pipe.set_params(verbose=True)
    assert pipe.verbose is True


def test_pipeline_indexing():
    """Test accessing pipeline steps by index."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
        (
            "string_imputer",
            StringImputer(strategy="constant", value="__NULL__", inplace=True),
        ),
    ]

    pipe = Pipeline(steps=steps)

    # Test integer indexing
    first_step = pipe[0]
    assert isinstance(first_step, NumericImputer)

    # Test slice
    sub_pipe = pipe[0:1]
    assert isinstance(sub_pipe, Pipeline)
    assert len(sub_pipe) == 1


def test_pipeline_validation_missing_fit():
    """Test that validation catches missing fit method."""

    class BadTransformer:
        def transform(self, X):
            return X

    steps = [("bad", BadTransformer())]

    with pytest.raises(TypeError, match="All steps must have a 'fit' method"):
        Pipeline(steps=steps)


def test_pipeline_validation_missing_transform():
    """Test that validation catches missing transform method."""

    class BadTransformer:
        def fit(self, X, y=None):
            return self

    steps = [("bad", BadTransformer())]

    with pytest.raises(TypeError, match="All steps must have a 'transform' method"):
        Pipeline(steps=steps)


def test_pipeline_verbose_fit(capsys):
    """Verbose fit emits row count, col count, null count and elapsed time."""
    X = pl.DataFrame({"num_col": [1.0, None, 3.0]})

    pipe = Pipeline(
        steps=[("numeric_imputer", NumericImputer(strategy="median", inplace=True))],
        verbose=True,
    )
    pipe.fit(X)

    out = capsys.readouterr().out
    assert "[Pipeline] fit" in out
    assert "1/1" in out
    assert "numeric_imputer" in out
    assert "rows=3" in out
    assert "cols=1" in out
    assert "nulls=1" in out
    # timing suffix like (0.001s)
    assert "s)" in out


def test_pipeline_verbose_transform(capsys):
    """Verbose transform emits before/after stats and elapsed time."""
    X = pl.DataFrame({"num_col": [1.0, None, 3.0]})

    pipe = Pipeline(
        steps=[("numeric_imputer", NumericImputer(strategy="median", inplace=True))],
        verbose=True,
    )
    pipe.fit(X)
    capsys.readouterr()  # discard fit output

    result = pipe.transform(X)
    assert isinstance(result, pl.DataFrame)

    out = capsys.readouterr().out
    assert "[Pipeline] transform" in out
    assert "1/1" in out
    assert "numeric_imputer" in out
    assert "in:" in out
    assert "out:" in out
    assert "→" in out
    assert "rows=3" in out
    assert "s)" in out


def test_pipeline_verbose_fit_transform(capsys):
    """Verbose fit_transform emits before/after stats and elapsed time."""
    X = pl.DataFrame({"num_col": [1.0, None, 3.0]})

    pipe = Pipeline(
        steps=[("numeric_imputer", NumericImputer(strategy="median", inplace=True))],
        verbose=True,
    )
    result = pipe.fit_transform(X)
    assert isinstance(result, pl.DataFrame)

    out = capsys.readouterr().out
    assert "[Pipeline] fit+transform" in out
    assert "1/1" in out
    assert "numeric_imputer" in out
    assert "in:" in out
    assert "out:" in out
    assert "→" in out
    assert "rows=3" in out
    assert "s)" in out


def test_pipeline_verbose_false_no_output(capsys):
    """verbose=False must produce zero stdout output."""
    X = pl.DataFrame({"num_col": [1.0, None, 3.0]})
    pipe = Pipeline(
        steps=[("numeric_imputer", NumericImputer(strategy="median", inplace=True))],
        verbose=False,
    )
    pipe.fit_transform(X)
    assert capsys.readouterr().out == ""


def test_pipeline_verbose_multi_step(capsys):
    """Each step emits its own line; null counts reflect intermediate state."""
    X = pl.DataFrame({"num_col": [1.0, None, 3.0], "str_col": ["a", None, "c"]})
    pipe = Pipeline(
        steps=[
            ("num", NumericImputer(strategy="median", inplace=True)),
            ("str", StringImputer(strategy="constant", value="__NULL__", inplace=True)),
        ],
        verbose=True,
    )
    pipe.fit_transform(X)
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 2
    assert "num" in lines[0]
    assert "str" in lines[1]


def test_pipeline_get_params_shallow():
    """Test get_params with deep=False."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps)
    params = pipe.get_params(deep=False)

    assert "steps" in params
    assert "verbose" in params
    # Should not have nested parameters
    assert "numeric_imputer__strategy" not in params


def test_pipeline_set_params_invalid():
    """Test set_params with invalid parameter."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps)

    with pytest.raises(ValueError, match="Invalid parameter"):
        pipe.set_params(invalid_param="value")


def test_pipeline_set_params_pipeline_level():
    """Test set_params with pipeline-level parameters."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps, verbose=False)
    pipe.set_params(verbose=True)

    assert pipe.verbose == True


def test_pipeline_repr():
    """Test string representation of pipeline."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
        ("string_imputer", StringImputer(strategy="constant", value="__NULL__", inplace=True)),
    ]

    pipe = Pipeline(steps=steps)
    repr_str = repr(pipe)

    assert "Pipeline" in repr_str
    assert "numeric_imputer" in repr_str
    assert "string_imputer" in repr_str
    assert "NumericImputer" in repr_str
    assert "StringImputer" in repr_str


def test_pipeline_empty_set_params():
    """Test set_params with no parameters."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps)
    result = pipe.set_params()

    assert result is pipe


def test_pipeline_get_params_no_deep_support():
    """Test get_params with transformer that doesn't support deep parameter."""

    class TransformerNoDeep:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def get_params(self):
            # Doesn't accept deep parameter
            return {"some_param": "value"}

    steps = [("no_deep", TransformerNoDeep())]
    pipe = Pipeline(steps=steps)
    params = pipe.get_params(deep=True)

    assert "steps" in params
    assert "no_deep__some_param" in params


def test_pipeline_get_params_exception():
    """Test get_params with transformer whose get_params raises exception."""

    class TransformerBadGetParams:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def get_params(self):
            raise RuntimeError("get_params failed")

    steps = [("bad_get", TransformerBadGetParams())]
    pipe = Pipeline(steps=steps)
    params = pipe.get_params(deep=True)

    # Should still return pipeline params without crashing
    assert "steps" in params
    assert "verbose" in params


def test_pipeline_get_params_pydantic_fallback():
    """Test get_params falls back to Pydantic for gators transformers."""
    # Gators transformers are Pydantic models
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps)
    params = pipe.get_params(deep=True)

    # Should have Pydantic fields
    assert "numeric_imputer__strategy" in params
    assert "numeric_imputer__inplace" in params


def test_pipeline_set_params_no_set_params_method():
    """Test set_params with transformer lacking set_params method."""

    class TransformerNoSetParams:
        def __init__(self):
            self.param1 = "initial"

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def get_params(self, deep=True):
            return {"param1": self.param1}

    steps = [("no_set", TransformerNoSetParams())]
    pipe = Pipeline(steps=steps)

    pipe.set_params(no_set__param1="modified")

    assert pipe.named_steps["no_set"].param1 == "modified"


def test_pipeline_set_params_set_params_raises_exception():
    """Test set_params with transformer whose set_params raises exception."""

    class TransformerBadSetParams:
        def __init__(self):
            self.param1 = "initial"

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def get_params(self, deep=True):
            return {"param1": self.param1}

        def set_params(self, **params):
            raise TypeError("set_params not supported")

    steps = [("bad_set", TransformerBadSetParams())]
    pipe = Pipeline(steps=steps)

    # Should fall back to direct attribute setting
    pipe.set_params(bad_set__param1="modified")

    assert pipe.named_steps["bad_set"].param1 == "modified"


def test_pipeline_set_params_attribute_error():
    """Test set_params with transformer that raises AttributeError."""

    class TransformerAttrError:
        def __init__(self):
            self.param1 = "initial"

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def get_params(self, deep=True):
            return {"param1": self.param1}

        def set_params(self, **params):
            raise AttributeError("Attribute not found")

    steps = [("attr_err", TransformerAttrError())]
    pipe = Pipeline(steps=steps)

    # Should fall back to direct attribute setting
    pipe.set_params(attr_err__param1="modified")

    assert pipe.named_steps["attr_err"].param1 == "modified"


def test_clone_returns_new_pipeline():
    """clone() must return a distinct Pipeline object."""
    pipe = Pipeline(
        steps=[
            ("impute", NumericImputer(strategy="median", inplace=True)),
            ("impute_str", StringImputer(strategy="constant", value="__NULL__")),
        ]
    )
    cloned = pipe.clone()
    assert cloned is not pipe


def test_clone_steps_are_new_instances():
    """Each transformer in the clone must be a new object, not the same reference."""
    pipe = Pipeline(
        steps=[
            ("impute", NumericImputer(strategy="median", inplace=True)),
        ]
    )
    cloned = pipe.clone()
    assert cloned.named_steps["impute"] is not pipe.named_steps["impute"]


def test_clone_preserves_hyperparameters():
    """Clone must carry over all constructor parameters."""
    pipe = Pipeline(
        steps=[
            ("impute", NumericImputer(strategy="mean", inplace=False)),
            ("impute_str", StringImputer(strategy="constant", value="N/A")),
        ],
        verbose=True,
    )
    cloned = pipe.clone()
    assert cloned.named_steps["impute"].strategy == "mean"
    assert cloned.named_steps["impute"].inplace is False
    assert cloned.named_steps["impute_str"].value == "N/A"
    assert cloned.verbose is True


def test_clone_is_unfitted():
    """After cloning a fitted pipeline, the clone must be unfitted."""
    X = pl.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, 3.0]})
    pipe = Pipeline(steps=[("impute", NumericImputer(strategy="median", inplace=True))])
    pipe.fit(X)

    cloned = pipe.clone()
    # The original has computed _statistics; the clone must not
    imputer_clone = cloned.named_steps["impute"]
    # _statistics is a PrivateAttr initialised to {}; after clone it stays empty
    assert imputer_clone._statistics == {}


def test_clone_can_be_fitted_independently():
    """The cloned pipeline can be fitted independently from the original."""
    X = pl.DataFrame({"a": [1.0, None, 3.0, 4.0, 5.0]})
    pipe = Pipeline(steps=[("impute", NumericImputer(strategy="median", inplace=True))])
    pipe.fit(X)

    cloned = pipe.clone()
    cloned.fit(X)  # must not raise

    result = cloned.transform(X)
    assert result.null_count().sum_horizontal()[0] == 0


# ── get_initial_features ──────────────────────────────────────────────────────

def test_get_initial_features_select():
    """Only the selected columns are traced back to the initial input."""
    from gators.data_cleaning import SelectColumns
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    pipe = Pipeline(steps=[
        ("imp", NumericImputer(strategy="median")),
        ("sel", SelectColumns(subset=["A", "B"])),
    ])
    pipe.fit(X)
    assert pipe.get_initial_features() == ["A", "B"]


def test_get_initial_features_drop():
    """Dropped columns are excluded from the result."""
    from gators.data_cleaning import DropColumns
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    pipe = Pipeline(steps=[("drop", DropColumns(subset=["C"]))])
    pipe.fit(X)
    assert pipe.get_initial_features() == ["A", "B"]


def test_get_initial_features_feature_gen_source_traced():
    """Source column used for feature generation is included even after SelectColumns drops it."""
    from gators.data_cleaning import SelectColumns
    from gators.feature_generation import ScalarMathFeatures
    X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    pipe = Pipeline(steps=[
        ("gen", ScalarMathFeatures(operations=[{"column": "A", "op": "+", "scalar": 1.0}])),
        ("sel", SelectColumns(subset=["A_plus_1"])),
    ])
    pipe.fit(X)
    assert pipe.get_initial_features() == ["A"]


def test_get_initial_features_passthrough_all():
    """When nothing is dropped or generated the full initial set is returned."""
    X = pl.DataFrame({"A": [1.0, None], "B": [2.0, 3.0]})
    pipe = Pipeline(steps=[("imp", NumericImputer(strategy="median"))])
    pipe.fit(X)
    assert pipe.get_initial_features() == ["A", "B"]


def test_get_initial_features_not_fitted():
    from gators.exceptions import NotFittedError
    pipe = Pipeline(steps=[("imp", NumericImputer(strategy="median"))])
    with pytest.raises(NotFittedError):
        pipe.get_initial_features()


def test_get_initial_features_diff_features_column_pairs():
    """column_pairs sources are included in the lineage (covers the column_pairs branch)."""
    from gators.data_cleaning import SelectColumns
    from gators.feature_generation_dt import DiffFeatures
    X = pl.DataFrame({
        "a": ["2024-01-15"], "b": ["2024-01-10"], "extra": [42.0],
    }).with_columns([
        pl.col("a").str.strptime(pl.Datetime, "%Y-%m-%d"),
        pl.col("b").str.strptime(pl.Datetime, "%Y-%m-%d"),
    ])
    pipe = Pipeline(steps=[
        ("diff", DiffFeatures(column_pairs=[("a", "b")], units=["d"])),
        ("sel",  SelectColumns(subset=["a_minus_b__days"])),
    ])
    pipe.fit(X)
    assert pipe.get_initial_features() == ["a", "b"]


def test_get_initial_features_group_statistics_by():
    """by-columns are included in the lineage (covers the by branch)."""
    from gators.data_cleaning import SelectColumns
    from gators.feature_generation import GroupStatisticsFeatures
    X = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "g": ["A", "A", "B", "B"], "extra": [0.0]*4})
    pipe = Pipeline(steps=[
        ("gsf", GroupStatisticsFeatures(subset=["v"], by=["g"], func=["mean"])),
        ("sel", SelectColumns(subset=["mean_v__per_g"])),
    ])
    pipe.fit(X)
    assert "v" in pipe.get_initial_features()
    assert "g" in pipe.get_initial_features()


def test_get_initial_features_groupby_imputer():
    """group_by_column is included in the lineage (covers the group_by_column branch)."""
    from gators.data_cleaning import SelectColumns
    from gators.imputers import GroupByImputer
    X = pl.DataFrame({"grp": ["A", "A", "B", "B"], "val": [1.0, None, 3.0, None], "extra": [0.0]*4})
    pipe = Pipeline(steps=[
        ("imp", GroupByImputer(group_by_column="grp", strategy="mean")),
        ("sel", SelectColumns(subset=["grp", "val"])),
    ])
    pipe.fit(X)
    assert "grp" in pipe.get_initial_features()
    assert "val" in pipe.get_initial_features()


def test_get_initial_features_renamed_column():
    """Renamed column (inplace=False) has its source traced (covers the reverse-map branch)."""
    from gators.data_cleaning import SelectColumns
    from gators.scalers import StandardScaler
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    pipe = Pipeline(steps=[
        ("scale", StandardScaler(inplace=False, drop_columns=True)),
        ("sel",   SelectColumns(subset=["A__standard_scale"])),
    ])
    pipe.fit(X)
    assert pipe.get_initial_features() == ["A"]


# ---------------------------------------------------------------------------
# trim_to() tests
# ---------------------------------------------------------------------------

@pytest.fixture
def four_step_pipe():
    """imp → isnull → poly → scale pipeline, pre-fitted."""
    from gators.feature_generation import PolynomialFeatures, IsNull

    X = pl.DataFrame({"A": [1.0, 2.0, None], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]})
    pipe = Pipeline(steps=[
        ("imp",    NumericImputer(strategy="mean")),
        ("isnull", IsNull()),
        ("poly",   PolynomialFeatures(subset=["A", "B", "C"], degree=2)),
        ("scale",  StandardScaler()),
    ])
    pipe.fit(X)
    return pipe, X


def test_trim_to_drop_isnull_poly(four_step_pipe):
    """Requesting only raw scaled columns drops isnull and poly."""
    pipe, X = four_step_pipe
    full = pipe.transform(X)
    trimmed = pipe.trim_to(["A", "B", "C"])
    assert [n for n, _ in trimmed.steps] == ["imp", "scale"]
    out = trimmed.transform(X)
    for col in ["A", "B", "C"]:
        np.testing.assert_allclose(full[col].to_numpy(), out[col].to_numpy(), atol=1e-6)


def test_trim_to_keep_poly(four_step_pipe):
    """Requesting poly output columns keeps imp, poly, and scale."""
    pipe, X = four_step_pipe
    full = pipe.transform(X)
    trimmed = pipe.trim_to(["A__A", "B__B", "A", "B"])
    assert [n for n, _ in trimmed.steps] == ["imp", "poly", "scale"]
    out = trimmed.transform(X)
    for col in ["A__A", "B__B", "A", "B"]:
        np.testing.assert_allclose(full[col].to_numpy(), out[col].to_numpy(), atol=1e-6)


def test_trim_to_isnull_only(four_step_pipe):
    """Requesting only isnull features drops poly.

    IsNull now outputs Float64 (not Boolean), so StandardScaler's auto-detected
    subset picks up A__is_null too — scale is legitimately needed here.
    """
    pipe, X = four_step_pipe
    full = pipe.transform(X)
    trimmed = pipe.trim_to(["A__is_null"])
    assert [n for n, _ in trimmed.steps] == ["imp", "isnull", "scale"]
    out = trimmed.transform(X)
    np.testing.assert_array_equal(
        full["A__is_null"].to_numpy(), out["A__is_null"].to_numpy()
    )


def test_trim_to_unknown_column_raises(four_step_pipe):
    """trim_to raises ValueError for columns not produced by the pipeline."""
    pipe, _ = four_step_pipe
    with pytest.raises(ValueError, match="not produced"):
        pipe.trim_to(["NOPE"])


def test_trim_to_not_fitted_raises():
    """trim_to raises NotFittedError when called before fit."""
    pipe = Pipeline(steps=[("scale", StandardScaler())])
    with pytest.raises(Exception):  # NotFittedError
        pipe.trim_to(["A"])


def test_trim_to_returns_fitted_pipeline(four_step_pipe):
    """The trimmed pipeline reports is_fitted and has correct input columns."""
    pipe, X = four_step_pipe
    trimmed = pipe.trim_to(["A", "B", "C"])
    assert trimmed._is_fitted
    assert trimmed._input_columns == pipe._input_columns


def test_trim_to_row_statistics_column_groups_dict():
    """RowStatisticsFeatures' dict-shaped column_groups source columns are traced
    (covers the column_groups-as-dict branch), keeping the upstream WOE step."""
    from gators.encoders import WOEEncoder
    from gators.feature_generation import RowStatisticsFeatures

    X = pl.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
    y = pl.Series([0, 1, 0, 1])
    pipe = Pipeline(steps=[
        ("woe", WOEEncoder(subset=["cat"], drop_columns=True, inplace=False)),
        ("rs",  RowStatisticsFeatures(column_groups={"g1": ["cat__encode_woe", "num"]}, func=["sum"])),
    ])
    pipe.fit(X, y=y)
    assert "cat" in pipe.get_initial_features()
    assert "num" in pipe.get_initial_features()

    trimmed = pipe.trim_to(["g1__sum"])
    assert [n for n, _ in trimmed.steps] == ["woe", "rs"]
    np.testing.assert_array_equal(
        trimmed.transform(X)["g1__sum"].to_numpy(), pipe.transform(X)["g1__sum"].to_numpy()
    )


def test_trim_to_hhi_column_groups_list():
    """HHIFeatures' list-shaped column_groups source columns are traced
    (covers the column_groups-as-list branch), keeping the upstream WOE step."""
    from gators.encoders import WOEEncoder
    from gators.feature_generation import HHIFeatures

    X = pl.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
    y = pl.Series([0, 1, 0, 1])
    pipe = Pipeline(steps=[
        ("woe", WOEEncoder(subset=["cat"], drop_columns=True, inplace=False)),
        ("hhi", HHIFeatures(column_groups=[["cat__encode_woe", "num"]])),
    ])
    pipe.fit(X, y=y)
    assert "cat" in pipe.get_initial_features()
    assert "num" in pipe.get_initial_features()

    trimmed = pipe.trim_to(["cat__encode_woe__num__hhi"])
    assert [n for n, _ in trimmed.steps] == ["woe", "hhi"]
    np.testing.assert_array_equal(
        trimmed.transform(X)["cat__encode_woe__num__hhi"].to_numpy(),
        pipe.transform(X)["cat__encode_woe__num__hhi"].to_numpy(),
    )


def test_trim_to_generalized_ratio_numerator_denominator():
    """GeneralizedRatioFeatures' numerator/denominator source columns are traced
    (covers the numerator_columns/denominator_columns branch), keeping the upstream
    WOE step."""
    from gators.encoders import WOEEncoder
    from gators.feature_generation import GeneralizedRatioFeatures

    X = pl.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
    y = pl.Series([0, 1, 0, 1])
    pipe = Pipeline(steps=[
        ("woe", WOEEncoder(subset=["cat"], drop_columns=True, inplace=False)),
        ("ratio", GeneralizedRatioFeatures(
            numerator_columns=[["cat__encode_woe"]],
            denominator_columns=[["num"]],
            new_column_names=["ratio_feat"],
        )),
    ])
    pipe.fit(X, y=y)
    assert "cat" in pipe.get_initial_features()
    assert "num" in pipe.get_initial_features()

    trimmed = pipe.trim_to(["ratio_feat"])
    assert [n for n, _ in trimmed.steps] == ["woe", "ratio"]
    np.testing.assert_allclose(
        trimmed.transform(X)["ratio_feat"].to_numpy(), pipe.transform(X)["ratio_feat"].to_numpy()
    )
