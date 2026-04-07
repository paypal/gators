"""
Tests for the Gators Pipeline class.
"""

import polars as pl
import pytest

from gators.imputers import NumericImputer, StringImputer
from gators.pipeline import Pipeline


def test_pipeline_creation():
    """Test that a Pipeline can be created."""
    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
        (
            "string_imputer",
            StringImputer(strategy="constant", value="MISSING", inplace=True),
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
            StringImputer(strategy="constant", value="MISSING", inplace=True),
        ),
    ]

    pipe = Pipeline(steps=steps)
    result = pipe.fit_transform(X)

    # Check no nulls remain
    assert result.null_count().sum_horizontal()[0] == 0
    assert isinstance(result, pl.DataFrame)


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
            StringImputer(strategy="constant", value="MISSING", inplace=True),
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
            StringImputer(strategy="constant", value="MISSING", inplace=True),
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


def test_pipeline_verbose_fit():
    """Test verbose mode during fit."""
    X = pl.DataFrame({"num_col": [1.0, 2.0, 3.0]})

    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps, verbose=True)
    pipe.fit(X)
    assert True


def test_pipeline_verbose_transform():
    """Test verbose mode during transform."""
    X = pl.DataFrame({"num_col": [1.0, 2.0, 3.0]})

    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps, verbose=True)
    pipe.fit(X)
    result = pipe.transform(X)
    assert isinstance(result, pl.DataFrame)


def test_pipeline_verbose_fit_transform():
    """Test verbose mode during fit_transform."""
    X = pl.DataFrame({"num_col": [1.0, 2.0, 3.0]})

    steps = [
        ("numeric_imputer", NumericImputer(strategy="median", inplace=True)),
    ]

    pipe = Pipeline(steps=steps, verbose=True)
    result = pipe.fit_transform(X)
    assert isinstance(result, pl.DataFrame)


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
        ("string_imputer", StringImputer(strategy="constant", value="MISSING", inplace=True)),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
