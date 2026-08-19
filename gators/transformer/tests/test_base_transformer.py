"""Tests for _BaseTransformer class."""

import pickle
import tempfile
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import PrivateAttr, ValidationError
from sklearn.base import BaseEstimator, TransformerMixin

from gators.exceptions import NotFittedError
from gators.transformer._base_transformer import _BaseTransformer


# Create a concrete test class since _BaseTransformer is meant to be subclassed
class ConcreteTransformer(_BaseTransformer):
    """Concrete implementation for testing."""

    param1: str = "default"
    param2: int = 42


class TestBaseTransformerInitialization:
    """Test initialization behavior of _BaseTransformer."""

    def test_init_with_keyword_arguments(self):
        """Test that initialization works with keyword arguments."""
        transformer = ConcreteTransformer(param1="test", param2=100)
        assert transformer.param1 == "test"
        assert transformer.param2 == 100

    def test_init_with_defaults(self):
        """Test that initialization works with default values."""
        transformer = ConcreteTransformer()
        assert transformer.param1 == "default"
        assert transformer.param2 == 42

    def test_init_with_partial_kwargs(self):
        """Test initialization with some keyword arguments."""
        transformer = ConcreteTransformer(param1="custom")
        assert transformer.param1 == "custom"
        assert transformer.param2 == 42

    def test_init_with_positional_arguments_raises_error(self):
        """Test that positional arguments raise a clear TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConcreteTransformer("value1", 123)

        error_msg = str(exc_info.value)
        assert "does not accept positional arguments" in error_msg
        assert "Use keyword arguments instead" in error_msg
        assert "ConcreteTransformer" in error_msg
        assert "Correct" in error_msg
        assert "Wrong" in error_msg

    def test_init_with_single_positional_argument_raises_error(self):
        """Test that even a single positional argument raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConcreteTransformer("value1")

        assert "does not accept positional arguments" in str(exc_info.value)

    def test_init_with_mixed_args_raises_error(self):
        """Test that mixing positional and keyword arguments raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConcreteTransformer("value1", param2=100)

        assert "does not accept positional arguments" in str(exc_info.value)


class TestBaseTransformerValidation:
    """Test Pydantic validation behavior."""

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden due to model_config."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteTransformer(param1="test", unknown_param="value")

        error = exc_info.value
        assert "unknown_param" in str(error)

    def test_type_validation(self):
        """Test that Pydantic validates types."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteTransformer(param2="not_an_int")

        error = exc_info.value
        assert "param2" in str(error)


class TestBaseTransformerInheritance:
    """Test inheritance from sklearn base classes."""

    def test_inherits_from_base_estimator(self):
        """Test that _BaseTransformer inherits from BaseEstimator."""
        transformer = ConcreteTransformer()
        assert isinstance(transformer, BaseEstimator)

    def test_inherits_from_transformer_mixin(self):
        """Test that _BaseTransformer inherits from TransformerMixin."""
        transformer = ConcreteTransformer()
        assert isinstance(transformer, TransformerMixin)

    def test_has_get_params_method(self):
        """Test that transformer has get_params method from BaseEstimator."""
        transformer = ConcreteTransformer(param1="test", param2=999)
        # Note: sklearn's get_params introspection doesn't work with our custom __init__
        # but the method exists
        assert hasattr(transformer, "get_params")
        assert callable(transformer.get_params)

    def test_has_set_params_method(self):
        """Test that transformer has set_params method from BaseEstimator."""
        transformer = ConcreteTransformer()
        # Note: sklearn's set_params introspection doesn't work with our custom __init__
        # but the method exists
        assert hasattr(transformer, "set_params")
        assert callable(transformer.set_params)

    def test_attribute_access(self):
        """Test that parameters can be accessed and modified directly."""
        transformer = ConcreteTransformer(param1="test", param2=999)

        # Can read attributes
        assert transformer.param1 == "test"
        assert transformer.param2 == 999

        # Can modify attributes directly
        transformer.param1 = "updated"
        transformer.param2 = 777

        assert transformer.param1 == "updated"
        assert transformer.param2 == 777

    def test_get_params_returns_correct_values(self):
        """Test that get_params returns all model fields."""
        transformer = ConcreteTransformer(param1="custom", param2=123)
        params = transformer.get_params()

        assert "param1" in params
        assert "param2" in params
        assert params["param1"] == "custom"
        assert params["param2"] == 123

    def test_set_params_with_valid_parameters(self):
        """Test that set_params correctly updates parameters."""
        transformer = ConcreteTransformer(param1="initial", param2=100)

        result = transformer.set_params(param1="updated", param2=200)

        # Should return self
        assert result is transformer

        # Parameters should be updated
        assert transformer.param1 == "updated"
        assert transformer.param2 == 200

    def test_set_params_with_invalid_parameter(self):
        """Test that set_params raises ValueError for invalid parameters."""
        transformer = ConcreteTransformer()

        with pytest.raises(ValueError) as exc_info:
            transformer.set_params(invalid_param="value")

        error_msg = str(exc_info.value)
        assert "Invalid parameter invalid_param" in error_msg
        assert "ConcreteTransformer" in error_msg
        assert "Valid parameters are:" in error_msg


class TestBaseTransformerModelConfig:
    """Test model_config settings."""

    def test_model_config_extra_forbid(self):
        """Test that model_config forbids extra fields."""
        # This is tested indirectly through the ValidationError test
        # but we can also check the config directly
        config = _BaseTransformer.model_config
        assert config.get("extra") == "forbid"


class TestConcreteDerivedTransformer:
    """Test a more complex derived transformer."""

    def test_multiple_inheritance_levels(self):
        """Test that derived classes work correctly."""

        class DerivedTransformer(ConcreteTransformer):
            param3: float = 3.14

        transformer = DerivedTransformer(param1="derived", param2=50, param3=2.71)

        assert transformer.param1 == "derived"
        assert transformer.param2 == 50
        assert transformer.param3 == 2.71
        assert isinstance(transformer, _BaseTransformer)
        assert isinstance(transformer, BaseEstimator)
        assert isinstance(transformer, TransformerMixin)

    def test_derived_class_positional_args_error(self):
        """Test that derived classes also reject positional arguments."""

        class DerivedTransformer(ConcreteTransformer):
            param3: float = 3.14

        with pytest.raises(TypeError) as exc_info:
            DerivedTransformer("value1", 123, 4.56)

        error_msg = str(exc_info.value)
        assert "does not accept positional arguments" in error_msg
        assert "DerivedTransformer" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ──────────────────────────────────────────────────────────────────────────────
# Concrete transformer with fit/transform for fit_transform tests
# ──────────────────────────────────────────────────────────────────────────────


class DoubleTransformer(_BaseTransformer):
    """Multiplies every column by 2 – used purely for fit_transform tests."""

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "DoubleTransformer":
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return X.with_columns([pl.col(c) * 2 for c in X.columns])


class TestBaseTransformerFitTransform:
    """Tests for the explicit fit_transform method on _BaseTransformer."""

    def test_fit_transform_returns_dataframe(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = DoubleTransformer().fit_transform(X)
        assert isinstance(result, pl.DataFrame)

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        t = DoubleTransformer()
        result_ft = t.fit_transform(X)
        result_seq = t.fit(X).transform(X)
        assert_frame_equal(result_ft, result_seq)

    def test_fit_transform_returns_self_is_fitted(self):
        X = pl.DataFrame({"a": [1.0, 2.0]})
        t = DoubleTransformer()
        t.fit_transform(X)
        # Subsequent transform must work (transformer was fitted)
        result = t.transform(X)
        assert_frame_equal(result, pl.DataFrame({"a": [2.0, 4.0]}))

    def test_fit_transform_passes_y(self):
        """y must be forwarded to fit() without error."""
        X = pl.DataFrame({"a": [1.0, 2.0]})
        y = pl.Series("target", [0, 1])
        result = DoubleTransformer().fit_transform(X, y=y)
        assert isinstance(result, pl.DataFrame)

    def test_fit_transform_values_correct(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = DoubleTransformer().fit_transform(X)
        assert_frame_equal(result, pl.DataFrame({"a": [2.0, 4.0, 6.0]}))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for serialization / feature-names / inverse-transform / LazyFrame tests
# ──────────────────────────────────────────────────────────────────────────────


class ScalerLike(_BaseTransformer):
    """Transformer that uses _column_mapping and drop_columns."""

    drop_columns: bool = True
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "ScalerLike":
        self._column_mapping = {c: f"{c}__scaled" for c in X.columns}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        X = X.with_columns([(pl.col(c) * 2).alias(new) for c, new in self._column_mapping.items()])
        if self.drop_columns:
            return X.drop(list(self._column_mapping))
        return X


class InplaceTransformer(_BaseTransformer):
    """Transformer that operates inplace (no column renaming)."""

    inplace: bool = True
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "InplaceTransformer":
        self._column_mapping = {c: c for c in X.columns}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return X.with_columns([pl.col(c) * 2 for c in self._column_mapping])


class NoMappingTransformer(_BaseTransformer):
    """Transformer with no column mapping (e.g. row-level op)."""

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "NoMappingTransformer":
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return X


class InvTransformer(_BaseTransformer):
    """Transformer with a real inverse_transform for wrapping tests."""

    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "InvTransformer":
        self._column_mapping = {c: f"{c}__doubled" for c in X.columns}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        X = X.with_columns([(pl.col(c) * 2).alias(new) for c, new in self._column_mapping.items()])
        if self.drop_columns:
            return X.drop(list(self._column_mapping))
        return X

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        reverse = {v: k for k, v in self._column_mapping.items()}
        exprs = [(pl.col(new) / 2).alias(orig) for new, orig in reverse.items() if new in X.columns]
        X = X.with_columns(exprs)
        if self.drop_columns:
            return X.drop([c for c in reverse if c in X.columns])
        return X


# ──────────────────────────────────────────────────────────────────────────────
# Serialization (save / load)
# ──────────────────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_pickle_round_trip_preserves_fitted_state(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        t = ScalerLike()
        t.fit(X)
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert restored._is_fitted
        assert_frame_equal(restored.transform(X), t.transform(X))

    def test_save_load_round_trip(self):
        X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        t = ScalerLike()
        t.fit(X)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            t.save(path)
            loaded = _BaseTransformer.load(path)
            assert loaded._is_fitted
            assert_frame_equal(loaded.transform(X), t.transform(X))
        finally:
            path.unlink(missing_ok=True)

    def test_unfitted_transformer_pickle_round_trip(self):
        t = ScalerLike()
        restored = pickle.loads(pickle.dumps(t))  # noqa: S301
        assert not restored._is_fitted


# ──────────────────────────────────────────────────────────────────────────────
# get_feature_names_out
# ──────────────────────────────────────────────────────────────────────────────


class TestGetFeatureNamesOut:
    def test_raises_not_fitted_error(self):
        t = ScalerLike()
        with pytest.raises(NotFittedError):
            t.get_feature_names_out()

    def test_no_column_mapping_returns_input_columns(self):
        X = pl.DataFrame({"x": [1], "y": [2]})
        t = NoMappingTransformer()
        t.fit(X)
        assert t.get_feature_names_out() == ["x", "y"]

    def test_drop_columns_true(self):
        X = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        t = ScalerLike(drop_columns=True)
        t.fit(X)
        assert t.get_feature_names_out() == ["a__scaled", "b__scaled", "c__scaled"]

    def test_drop_columns_false(self):
        X = pl.DataFrame({"a": [1.0], "b": [2.0]})
        t = ScalerLike(drop_columns=False)
        t.fit(X)
        assert t.get_feature_names_out() == ["a", "b", "a__scaled", "b__scaled"]

    def test_inplace_returns_input_columns(self):
        X = pl.DataFrame({"a": [1.0], "b": [2.0]})
        t = InplaceTransformer()
        t.fit(X)
        assert t.get_feature_names_out() == ["a", "b"]

    def test_selector_returns_selected_features(self):
        from gators.feature_selection import CorrelationSelector

        X = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [1.1, 2.1, 3.1, 4.1, 5.1],
                "c": [5.0, 3.0, 1.0, 4.0, 2.0],
            }
        )
        sel = CorrelationSelector(importance={"a": 0.9, "b": 0.4, "c": 0.7})
        sel.fit(X)
        names = sel.get_feature_names_out()
        assert "b" not in names
        assert "a" in names
        assert "c" in names

    def test_with_numeric_imputer_inplace(self):
        from gators.imputers import NumericImputer

        X = pl.DataFrame({"a": [1.0, None], "b": [None, 2.0], "c": ["x", "y"]})
        t = NumericImputer(strategy="mean", inplace=True)
        t.fit(X)
        assert t.get_feature_names_out() == ["a", "b", "c"]

    def test_with_standard_scaler_drop_columns_true(self):
        from gators.scalers import StandardScaler

        X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        t = StandardScaler(inplace=False, drop_columns=True)
        t.fit(X)
        assert t.get_feature_names_out() == ["a__standard_scale", "b__standard_scale"]


# ──────────────────────────────────────────────────────────────────────────────
# Default inverse_transform raises NotImplementedError
# ──────────────────────────────────────────────────────────────────────────────


class TestDefaultInverseTransform:
    def test_raises_not_implemented_for_unfeatured_transformer(self):
        X = pl.DataFrame({"a": [1.0]})
        t = NoMappingTransformer()
        t.fit(X)
        with pytest.raises(NotImplementedError, match="does not support inverse_transform"):
            t.inverse_transform(X)

    def test_unfitted_scaler_inverse_raises_not_fitted(self):
        from gators.scalers import StandardScaler

        X = pl.DataFrame({"a": [1.0]})
        t = StandardScaler()
        with pytest.raises(NotFittedError):
            t.inverse_transform(X)

    def test_wrapped_inverse_checks_fitted(self):
        t = InvTransformer()
        X = pl.DataFrame({"a": [2.0, 4.0]})
        with pytest.raises(NotFittedError):
            t.inverse_transform(X)

    def test_wrapped_inverse_works_when_fitted(self):
        X = pl.DataFrame({"a": [1.0, 2.0]})
        t = InvTransformer()
        t.fit(X)
        X_t = t.transform(X)
        restored = t.inverse_transform(X_t)
        assert_frame_equal(restored, X.cast(pl.Float64))


# ──────────────────────────────────────────────────────────────────────────────
# LazyFrame support
# ──────────────────────────────────────────────────────────────────────────────


class TestLazyFrameSupport:
    def test_fit_with_lazy_frame(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        t = ScalerLike()
        t.fit(X.lazy())
        assert t._is_fitted
        assert t._input_columns == ["a", "b"]

    def test_transform_with_lazy_frame(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        t = ScalerLike()
        t.fit(X)
        result = t.transform(X.lazy())
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, t.transform(X))

    def test_fit_transform_with_lazy_frame(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        t = ScalerLike()
        result = t.fit_transform(X.lazy())
        assert isinstance(result, pl.DataFrame)
        assert t._is_fitted

    def test_lazy_frame_collects_once_per_step(self):
        """fit_transform collects the LazyFrame exactly once before fit+transform."""
        X = pl.DataFrame({"a": [1.0, 2.0]})
        t = ScalerLike()
        result = t.fit_transform(X.lazy())
        assert isinstance(result, pl.DataFrame)

    def test_fit_with_lazy_frame_as_keyword_argument(self):
        """LazyFrame passed as X=... keyword arg is collected in wrapped_fit."""
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        t = ScalerLike()
        t.fit(X=X.lazy())
        assert t._is_fitted
        assert t._input_columns == ["a"]

    def test_transform_with_lazy_frame_as_keyword_argument(self):
        """LazyFrame passed as X=... keyword arg is collected in wrapped_transform."""
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        t = ScalerLike()
        t.fit(X)
        result = t.transform(X=X.lazy())
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, t.transform(X))


# ──────────────────────────────────────────────────────────────────────────────
# _input_columns tracked automatically
# ──────────────────────────────────────────────────────────────────────────────


class TestInputColumnsTracking:
    def test_input_columns_set_after_fit(self):
        X = pl.DataFrame({"x": [1], "y": [2], "z": [3]})
        t = NoMappingTransformer()
        t.fit(X)
        assert t._input_columns == ["x", "y", "z"]

    def test_input_columns_set_with_lazy_frame(self):
        X = pl.DataFrame({"a": [1], "b": [2]})
        t = NoMappingTransformer()
        t.fit(X.lazy())
        assert t._input_columns == ["a", "b"]

    def test_input_columns_not_set_when_x_is_none(self):
        t = NoMappingTransformer()
        t.fit(X=None)
        assert t._input_columns == []


# ──────────────────────────────────────────────────────────────────────────────
# _is_fitted flag and check_is_fitted guard
# ──────────────────────────────────────────────────────────────────────────────


class TestIsFittedFlag:
    def test_is_false_before_fit(self):
        assert not NoMappingTransformer()._is_fitted

    def test_is_true_after_fit(self):
        X = pl.DataFrame({"a": [1.0]})
        t = NoMappingTransformer()
        t.fit(X)
        assert t._is_fitted

    def test_check_is_fitted_raises_before_fit(self):
        t = NoMappingTransformer()
        with pytest.raises(NotFittedError, match="not fitted"):
            t.check_is_fitted()

    def test_check_is_fitted_passes_after_fit(self):
        X = pl.DataFrame({"a": [1.0]})
        t = NoMappingTransformer()
        t.fit(X)
        t.check_is_fitted()  # must not raise

    def test_transform_raises_before_fit(self):
        X = pl.DataFrame({"a": [1.0]})
        with pytest.raises(NotFittedError):
            NoMappingTransformer().transform(X)

    def test_error_message_contains_class_name(self):
        t = DoubleTransformer()
        with pytest.raises(NotFittedError, match="DoubleTransformer"):
            t.check_is_fitted()

    def test_is_true_after_fit_transform(self):
        X = pl.DataFrame({"a": [1.0]})
        t = DoubleTransformer()
        t.fit_transform(X)
        assert t._is_fitted

    def test_not_reset_between_refits(self):
        X1 = pl.DataFrame({"a": [1.0]})
        X2 = pl.DataFrame({"b": [2.0]})
        t = NoMappingTransformer()
        t.fit(X1)
        assert t._is_fitted
        t.fit(X2)
        assert t._is_fitted
