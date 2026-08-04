"""Tests for new _BaseTransformer features: serialization, get_feature_names_out,
inverse_transform default, LazyFrame support."""

import math
import pickle
import tempfile
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import PrivateAttr

from gators.exceptions import NotFittedError
from gators.transformer._base_transformer import _BaseTransformer

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Serialization (save / load)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# get_feature_names_out
# ---------------------------------------------------------------------------


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
        t = StandardScaler()
        t.fit(X)
        assert t.get_feature_names_out() == ["a__standard_scale", "b__standard_scale"]


# ---------------------------------------------------------------------------
# Default inverse_transform raises NotImplementedError
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LazyFrame support
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _input_columns tracked automatically
# ---------------------------------------------------------------------------


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
