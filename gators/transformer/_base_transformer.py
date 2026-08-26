import functools
import pickle
from pathlib import Path
from typing import Any, cast

import polars as pl
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin

from ..exceptions import NotFittedError


class _BaseTransformer(BaseModel, BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    Base class for all transformers in the gators library.

    This class provides common functionality for all transformers.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _is_fitted: bool = PrivateAttr(default=False)
    _input_columns: list[str] = PrivateAttr(default_factory=list)
    _input_dtypes: dict[str, Any] = PrivateAttr(default_factory=dict)
    # Declared by fit() for columns the transformer creates or changes; columns absent
    # from this dict keep their _input_dtypes (pass-through). Consumed by ONNX export
    # to determine output tensor types without needing a real transform() call.
    _output_dtypes: dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if "fit" in cls.__dict__:
            original_fit = cls.__dict__["fit"]

            @functools.wraps(original_fit)
            def wrapped_fit(self: Any, *args: Any, **kw: Any) -> Any:
                X = args[0] if args else kw.get("X")
                if isinstance(X, pl.LazyFrame):
                    X = X.collect()
                    args = (X, *args[1:]) if args else args
                    if not args:
                        kw = {**kw, "X": X}
                result = original_fit(self, *args, **kw)
                self._is_fitted = True
                if X is not None and hasattr(X, "columns"):
                    self._input_columns = list(X.columns)
                    self._input_dtypes = dict(zip(X.columns, X.dtypes, strict=False))
                return result

            cls.fit = wrapped_fit

        if "transform" in cls.__dict__:
            original_transform = cls.__dict__["transform"]

            @functools.wraps(original_transform)
            def wrapped_transform(self: Any, *args: Any, **kw: Any) -> Any:
                self.check_is_fitted()
                X = args[0] if args else kw.get("X")
                if isinstance(X, pl.LazyFrame):
                    X = X.collect()
                    if args:
                        args = (X, *args[1:])
                    else:
                        kw = {**kw, "X": X}
                return original_transform(self, *args, **kw)

            cls.transform = wrapped_transform

        if "inverse_transform" in cls.__dict__:
            original_inv = cls.__dict__["inverse_transform"]

            @functools.wraps(original_inv)
            def wrapped_inverse(self: Any, *args: Any, **kw: Any) -> Any:
                self.check_is_fitted()
                return original_inv(self, *args, **kw)

            cls.inverse_transform = wrapped_inverse  # type: ignore[method-assign]

    def check_is_fitted(self) -> None:
        """Raise NotFittedError if the transformer has not been fitted yet.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called on this instance.
        """
        if not self._is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before 'transform'."
            )

    def get_feature_names_out(self) -> list[str]:
        """Return output column names after transformation.

        Uses ``_column_mapping`` to compute the output column list. Each source
        column maps to a *list* of generated column names (supports both 1:1 and
        1:many transformers). Feature selectors return ``selected_features_`` directly.

        Returns
        -------
        list[str]
            Column names of the transformed DataFrame.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.
        """
        self.check_is_fitted()

        # Feature selectors expose _selected_features
        selected = getattr(self, "_selected_features", None)
        if selected is not None:
            return list(selected)

        col_map: dict[str, list[str]] = dict(getattr(self, "_column_mapping", None) or {})

        if not col_map:
            return list(self._input_columns)

        inplace: bool = getattr(self, "inplace", False)
        drop_columns: bool = getattr(self, "drop_columns", True)
        new_cols = [name for names in col_map.values() for name in names]

        if inplace:
            return list(self._input_columns)

        if drop_columns:
            dropped = set(col_map)
            return [c for c in self._input_columns if c not in dropped] + new_cols

        return list(self._input_columns) + new_cols

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Reverse the transformation.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support inverse_transform.")

    def save(self, path: str | Path) -> None:
        """Save the fitted transformer to a file using pickle.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "_BaseTransformer":
        """Load a fitted transformer from a pickle file.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        _BaseTransformer
            The loaded transformer instance.
        """
        with open(path, "rb") as f:
            return cast("_BaseTransformer", pickle.load(f))  # noqa: S301

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize transformer with clear error message for positional arguments.

        Raises
        ------
        TypeError
            If positional arguments are provided instead of keyword arguments.
        """
        if args:
            raise TypeError(
                f"{self.__class__.__name__}() does not accept positional arguments. "
                "Use keyword arguments instead:\n"
                f"  {self.__class__.__name__}(param1=value1, param2=value2)  # Correct\n"
                f"  {self.__class__.__name__}(value1, value2)                # Wrong - raises this error"
            )
        super().__init__(**kwargs)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        This overrides sklearn's BaseEstimator.get_params() to work with
        Pydantic models instead of relying on __init__ signature introspection.

        Parameters
        ----------
        deep : bool, default=True
            Not used for Pydantic models (included for sklearn compatibility).

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        # Return all Pydantic model fields
        return {key: getattr(self, key) for key in self.__class__.model_fields.keys()}

    def set_params(self, **params: Any) -> "_BaseTransformer":
        """Set parameters for this estimator.

        This overrides sklearn's BaseEstimator.set_params() to work with
        Pydantic models.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
            The estimator instance.

        Raises
        ------
        ValueError
            If an invalid parameter name is provided.
        """
        valid_params = set(self.__class__.model_fields.keys())

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params)}"
                )
            setattr(self, key, value)

        return self

    def fit_transform(
        self, X: pl.DataFrame | pl.LazyFrame, y: pl.Series | None = None
    ) -> pl.DataFrame:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : pl.DataFrame or pl.LazyFrame
            Input DataFrame. A LazyFrame is collected once before fitting.
        y : pl.Series or None, default=None
            Target series for supervised transformers.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
        return cast(pl.DataFrame, self.fit(X, y).transform(X))
