"""
Pipeline for chaining Gators transformers.

This Pipeline class is designed specifically for Gators transformers that work
with Polars DataFrames. Unlike sklearn's Pipeline, it doesn't perform type
conversion or validation that can cause issues with Polars DataFrames.
"""

import time
from typing import Any, List, Optional, Tuple

import polars as pl
from pydantic import ConfigDict

from ..transformer._base_transformer import _BaseTransformer


def _step_stats(X: pl.DataFrame) -> str:
    """Return a compact stats string for a DataFrame (used by verbose mode)."""
    n_nulls = sum(X.null_count().row(0))
    return f"rows={len(X)}  cols={len(X.columns)}  nulls={n_nulls}"


class Pipeline(_BaseTransformer):
    """
    Pipeline of transformers for Polars DataFrames.

    Sequentially applies a list of transforms. This is a lightweight alternative
    to sklearn.pipeline.Pipeline specifically designed for Gators transformers
    that work with Polars DataFrames.

    Parameters
    ----------
    steps : list[tuple[ str, Any]]
        List of (name, transform) tuples that are chained in the order they
        are specified. Each transform must implement fit and transform methods.
    verbose : bool, default=False
        If True, emits a one-line summary per step to stdout showing the step
        name, row count, column count, total null count, and wall-clock time.
        When ``False`` there is zero measurement overhead.

    Examples
    --------
    >>> from gators.pipeline import Pipeline
    >>> from gators.imputers import NumericImputer, StringImputer
    >>> from gators.encoders import WOEEncoder
    >>>
    >>> steps = [
    ...     ('numeric_imputer', NumericImputer(strategy='median')),
    ...     ('string_imputer', StringImputer(strategy='constant', value='MISSING')),
    ...     ('woe_encoder', WOEEncoder(subset=['cat_col']))
    ... ]
    >>> pipe = Pipeline(steps=steps)
    >>> pipe.fit(X_train, y=y_train)
    >>> X_transformed = pipe.transform(X_train)
    """

    steps: list[tuple[str, Any]]
    verbose: bool = False

    def model_post_init(self, __context: Any) -> None:
        """Called after Pydantic model initialization to validate steps."""
        super().model_post_init(__context)
        self._validate_steps()

    def _validate_steps(self):
        """Validate that all steps have fit and transform methods.

        Raises
        ------
        TypeError
            If any transformer is missing fit or transform methods.
        """
        for name, transformer in self.steps:
            if not hasattr(transformer, "fit"):
                raise TypeError(
                    f"All steps must have a 'fit' method. "
                    f"'{name}' (type {type(transformer)}) doesn't."
                )
            if not hasattr(transformer, "transform"):
                raise TypeError(
                    f"All steps must have a 'transform' method. "
                    f"'{name}' (type {type(transformer)}) doesn't."
                )

    @property
    def named_steps(self):
        """Access steps by name.

        Returns
        -------
        dict
            Dictionary mapping step names to transformer instances.
        """
        return dict(self.steps)

    def __len__(self):
        """Return the length of the Pipeline.

        Returns
        -------
        int
            Number of steps in the pipeline.
        """
        return len(self.steps)

    def __getitem__(self, ind):
        """Return a step by index or slice.

        Parameters
        ----------
        ind : int or slice
            Index or slice to access steps.

        Returns
        -------
        TransformerMixin or Pipeline
            If index: returns the transformer at that position.
            If slice: returns a new Pipeline with the selected steps.
        """
        if isinstance(ind, slice):
            return self.__class__(steps=self.steps[ind], verbose=self.verbose)
        name, transformer = self.steps[ind]
        return transformer

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "Pipeline":
        """Fit all transformers in the pipeline.

        Fits each transformer sequentially, transforming the data before
        fitting the next transformer. This ensures each transformer sees
        the output of the previous transformer.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : pl.Series, default=None
            Target series for supervised transformers (e.g., WOEEncoder).

        Returns
        -------
        Pipeline
            The fitted pipeline instance.
        """
        X_transformed = X

        for step_idx, (name, transformer) in enumerate(self.steps):
            if self.verbose:
                t0 = time.perf_counter()
                in_stats = _step_stats(X_transformed)

            # Fit the transformer
            transformer.fit(X_transformed, y=y)

            # Transform for the next step (except for the last step in fit)
            if step_idx < len(self.steps) - 1:
                X_transformed = transformer.transform(X_transformed)

            if self.verbose:
                elapsed = time.perf_counter() - t0
                print(
                    f"[Pipeline] fit   {step_idx + 1}/{len(self.steps)} · {name}"
                    f"  |  {in_stats}  ({elapsed:.3f}s)"
                )

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform data by applying all transformers in sequence.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        X_transformed = X

        for step_idx, (name, transformer) in enumerate(self.steps):
            if self.verbose:
                t0 = time.perf_counter()
                in_stats = _step_stats(X_transformed)

            X_transformed = transformer.transform(X_transformed)

            if self.verbose:
                elapsed = time.perf_counter() - t0
                out_stats = _step_stats(X_transformed)
                print(
                    f"[Pipeline] transform   {step_idx + 1}/{len(self.steps)} · {name}"
                    f"  |  in: {in_stats}  →  out: {out_stats}  ({elapsed:.3f}s)"
                )

        return X_transformed

    def fit_transform(self, X: pl.DataFrame, y: pl.Series | None = None) -> pl.DataFrame:
        """Fit all transformers and transform the data.

        Fits and transforms each transformer sequentially. This is more
        efficient than calling fit() followed by transform() separately.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit and transform.
        y : pl.Series, default=None
            Target series for supervised transformers.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        X_transformed = X

        for step_idx, (name, transformer) in enumerate(self.steps):
            if self.verbose:
                t0 = time.perf_counter()
                in_stats = _step_stats(X_transformed)

            # Fit and transform the data, honouring any custom fit_transform
            X_transformed = transformer.fit_transform(X_transformed, y=y)

            if self.verbose:
                elapsed = time.perf_counter() - t0
                out_stats = _step_stats(X_transformed)
                print(
                    f"[Pipeline] fit+transform   {step_idx + 1}/{len(self.steps)} · {name}"
                    f"  |  in: {in_stats}  →  out: {out_stats}  ({elapsed:.3f}s)"
                )

        return X_transformed

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns parameters of all sub-estimators.
            If False, only returns pipeline-level parameters.

        Returns
        -------
        dict
            Parameter names mapped to their values. Nested parameters
            use double underscore notation (e.g., 'step_name__param').
        """
        if not deep:
            return {"steps": self.steps, "verbose": self.verbose}

        out = {"steps": self.steps, "verbose": self.verbose}

        for name, transformer in self.steps:
            # Get transformer parameters
            transformer_params = {}

            if hasattr(transformer, "get_params"):
                try:
                    # Try with deep parameter (gators/sklearn convention)
                    transformer_params = transformer.get_params(deep=False)
                except TypeError:
                    # Transformer doesn't accept deep parameter
                    try:
                        transformer_params = transformer.get_params()
                    except Exception:
                        # get_params raises exception - skip this transformer
                        pass

            # Add transformer params with nested naming
            for key, value in transformer_params.items():
                out[f"{name}__{key}"] = value

        return out

    def clone(self) -> "Pipeline":
        """Return a new unfitted pipeline with the same hyperparameters.

        Each transformer is re-instantiated using only its public constructor
        parameters (obtained via ``get_params()``).  Private attributes that
        hold fitted state (e.g. ``_statistics``, ``mapping_``) are not copied,
        so the returned pipeline is guaranteed to be unfitted.

        This is the recommended alternative to ``copy.deepcopy`` for
        cross-validation workflows where you need multiple independent copies
        of the same pipeline configuration.

        Returns
        -------
        Pipeline
            A new, unfitted ``Pipeline`` instance with identical hyperparameters.

        Examples
        --------
        >>> from gators.pipeline import Pipeline
        >>> from gators.imputers import NumericImputer
        >>> from gators.scalers import StandardScaler

        >>> pipe = Pipeline(steps=[
        ...     ('impute', NumericImputer(strategy='median')),
        ...     ('scale', StandardScaler()),
        ... ])
        >>> pipe_clone = pipe.clone()
        >>> pipe_clone is pipe
        False
        >>> pipe_clone.named_steps['impute'] is pipe.named_steps['impute']
        False
        """
        cloned_steps = [
            (name, type(transformer)(**transformer.get_params()))
            for name, transformer in self.steps
        ]
        return Pipeline(steps=cloned_steps, verbose=self.verbose)

    def set_params(self, **params):
        """Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters. Use double underscore notation for
            nested parameters (e.g., step_name__param_name=value).

        Returns
        -------
        Pipeline
            The pipeline instance.

        Raises
        ------
        ValueError
            If an invalid parameter name is provided.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        nested_params = {}

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self}. "
                    f"Valid parameters are: {list(valid_params.keys())}"
                )

            if "__" in key:
                # Handle nested parameters
                step_name, param_name = key.split("__", 1)
                if step_name not in nested_params:
                    nested_params[step_name] = {}
                nested_params[step_name][param_name] = value
            else:
                # Handle pipeline-level parameters
                setattr(self, key, value)

        # Set nested parameters
        for step_name, step_params in nested_params.items():
            transformer = self.named_steps[step_name]

            # Try using set_params method
            if hasattr(transformer, "set_params"):
                try:
                    transformer.set_params(**step_params)
                except (TypeError, AttributeError):
                    # Fall back to direct attribute setting
                    for param_name, param_value in step_params.items():
                        setattr(transformer, param_name, param_value)
            else:
                # Direct attribute setting for transformers without set_params
                for param_name, param_value in step_params.items():
                    setattr(transformer, param_name, param_value)

        return self

    def __repr__(self):
        """String representation of the pipeline.

        Returns
        -------
        str
            Human-readable string representation showing all steps.
        """
        steps_str = "\n".join(
            f"    {name}: {transformer.__class__.__name__}" for name, transformer in self.steps
        )
        return f"Pipeline(\n{steps_str}\n)"
