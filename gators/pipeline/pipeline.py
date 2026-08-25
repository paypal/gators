"""
Pipeline for chaining Gators transformers.

This Pipeline class is designed specifically for Gators transformers that work
with Polars DataFrames. Unlike sklearn's Pipeline, it doesn't perform type
conversion or validation that can cause issues with Polars DataFrames.
"""

import time
from typing import Any

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


def _step_stats(X: pl.DataFrame) -> str:
    """Return a compact stats string for a DataFrame (used by verbose mode)."""
    n_nulls = sum(X.null_count().row(0))
    return f"rows={len(X)}  cols={len(X.columns)}  nulls={n_nulls}"


def _transformer_source_columns(transformer: Any) -> list[str]:
    """Return every column name a transformer reads from.

    Most transformers expose their inputs via ``subset``, but feature generators
    such as ``GeneralizedRatioFeatures``, ``HHIFeatures`` and
    ``RowStatisticsFeatures`` key their source columns under different attribute
    names (``column_groups``, ``numerator_columns``/``denominator_columns``)
    instead. Both lineage tracing (``get_initial_features``) and step pruning
    (``trim_to``) need this full set to avoid dropping/ignoring columns those
    generators still depend on.
    """
    cols: list[str] = list(getattr(transformer, "subset", None) or [])
    for op in getattr(transformer, "operations", None) or []:
        if isinstance(op, dict) and "column" in op:
            cols.append(op["column"])
    for pair in getattr(transformer, "column_pairs", None) or []:
        cols.extend(pair)
    for col in getattr(transformer, "by", None) or []:
        cols.append(col)
    group_by = getattr(transformer, "group_by_column", None)
    if group_by:
        cols.append(group_by)

    column_groups = getattr(transformer, "column_groups", None)
    if isinstance(column_groups, dict):
        for group in column_groups.values():
            cols.extend(group)
    elif column_groups:
        for group in column_groups:
            cols.extend(group)

    for attr in ("numerator_columns", "denominator_columns"):
        for group in getattr(transformer, attr, None) or []:
            cols.extend(group)

    return cols


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

    def fit_transform(self, X: pl.DataFrame | pl.LazyFrame, y: pl.Series | None = None) -> pl.DataFrame:
        """Fit all transformers and transform the data.

        Fits and transforms each transformer sequentially. This is more
        efficient than calling fit() followed by transform() separately.

        Parameters
        ----------
        X : pl.DataFrame or pl.LazyFrame
            Input DataFrame to fit and transform. A LazyFrame is collected once
            before fitting.
        y : pl.Series, default=None
            Target series for supervised transformers.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        if isinstance(X, pl.LazyFrame):
            X = X.collect()
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

        self._is_fitted = True
        if hasattr(X, "columns"):
            self._input_columns = list(X.columns)
            self._input_dtypes = dict(zip(X.columns, X.dtypes, strict=False))
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

    def get_initial_features(self) -> list[str]:
        """Return the initial input columns that contribute to the pipeline output.

        Traces forward through each step's column evolution and then resolves
        which of the original input columns are actually needed, accounting for:

        - **Passthrough** columns that survive unchanged to the final output.
        - **Source** columns whose values were used to build new features (even
          if the source column itself was later dropped).
        - **Dropped** columns that neither survive nor produce any final output
          column — these are *excluded* from the result.

        The returned list preserves the original column order from the DataFrame
        passed to ``fit()``.

        Returns
        -------
        list[str]
            Initial input column names required to produce the current output.

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted yet.

        Examples
        --------
        >>> import polars as pl
        >>> from gators.pipeline import Pipeline
        >>> from gators.imputers import NumericImputer
        >>> from gators.data_cleaning import SelectColumns
        >>> from gators.scalers import StandardScaler

        >>> X = pl.DataFrame({"A": [1.0, None], "B": [2.0, 3.0], "C": [4.0, 5.0]})
        >>> pipe = Pipeline(steps=[
        ...     ("impute", NumericImputer(strategy="median")),
        ...     ("select", SelectColumns(subset=["A", "B"])),
        ...     ("scale",  StandardScaler()),
        ... ])
        >>> pipe.fit(X)
        >>> pipe.get_initial_features()
        ['A', 'B']
        """
        self.check_is_fitted()

        try:
            from ..onnx_converters import get_output_columns as _get_output_columns
        except ImportError:  # pragma: no cover
            # Fallback when onnx is not installed: return all initial columns
            return list(self._input_columns)

        # ── Phase 1: forward simulation ──────────────────────────────────────
        # Compute the column list at each step boundary [0=initial, 1=after step 0, …]
        col_at: list[list[str]] = [list(self._input_columns)]
        current = list(self._input_columns)
        for _, transformer in self.steps:
            current = _get_output_columns(transformer, current)
            col_at.append(current)

        # ── Phase 2: build column lineage (forward) ───────────────────────────
        # lineage[col] = set of initial input columns that contributed to col
        lineage: dict[str, set[str]] = {c: {c} for c in self._input_columns}

        for step_idx, (_, transformer) in enumerate(self.steps):
            col_map: dict[str, list[str]] = dict(getattr(transformer, "_column_mapping", {}) or {})
            # Reverse map: generated_name → source_name (for renamed/encoded cols)
            reverse: dict[str, str] = {v: k for k, values in col_map.items() for v in values}
            # Collect all column names this transformer uses as source inputs
            subset: list[str] = _transformer_source_columns(transformer)

            new_lineage: dict[str, set[str]] = {}
            for out_col in col_at[step_idx + 1]:
                if out_col in lineage:
                    # Passthrough: column existed before this step
                    new_lineage[out_col] = lineage[out_col]
                elif out_col in reverse and reverse[out_col] in lineage:
                    # Renamed / encoded: inherits lineage of its source column
                    new_lineage[out_col] = lineage[reverse[out_col]]
                else:
                    # New column from a feature generator: collect lineage of
                    # all source columns in subset (conservative union)
                    src: set[str] = set()
                    for s in subset:
                        if s in lineage:
                            src.update(lineage[s])
                    new_lineage[out_col] = src

            lineage = new_lineage

        # ── Phase 3: collect required initial columns ─────────────────────────
        needed: set[str] = set()
        for sources in lineage.values():
            needed.update(sources)

        return [c for c in self._input_columns if c in needed]

    def trim_to(self, output_cols: list[str]) -> "Pipeline":
        """Return a minimal fitted Pipeline that produces exactly ``output_cols``.

        Walks backward from the desired output, keeping only steps that either
        create or operate inplace on at least one requested column.  Steps that
        exclusively produce columns absent from ``output_cols`` are dropped.

        The returned pipeline is already fitted (the individual transformers
        carry their learned state); it can be called with ``transform()``
        immediately.  Its input schema is identical to the original pipeline's
        input schema — the caller is responsible for providing all original
        input columns.

        Parameters
        ----------
        output_cols : list[str]
            Column names to retain in the output.  Every element must appear in
            the full pipeline's output.

        Returns
        -------
        Pipeline
            A new, fitted ``Pipeline`` with redundant steps removed.

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted yet.
        ValueError
            If any element of ``output_cols`` is not produced by the pipeline.

        Examples
        --------
        >>> import polars as pl
        >>> from gators.pipeline import Pipeline
        >>> from gators.imputers import NumericImputer
        >>> from gators.scalers import StandardScaler
        >>> from gators.feature_generation import PolynomialFeatures

        >>> X = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
        >>> pipe = Pipeline(steps=[
        ...     ("imp",   NumericImputer(strategy="mean")),
        ...     ("poly",  PolynomialFeatures(subset=["A", "B", "C"], degree=2)),
        ...     ("scale", StandardScaler()),
        ... ])
        >>> pipe.fit(X)
        >>> trimmed = pipe.trim_to(["A", "B", "C"])
        >>> [name for name, _ in trimmed.steps]
        ['imp', 'scale']
        """
        self.check_is_fitted()

        try:
            from ..onnx_converters import get_output_columns as _goc
        except ImportError:  # pragma: no cover
            return Pipeline(steps=list(self.steps), verbose=self.verbose)

        # Forward simulation: column set at each boundary
        col_at: list[list[str]] = [list(self._input_columns)]
        cur = list(self._input_columns)
        for _, t in self.steps:
            cur = _goc(t, cur)
            col_at.append(cur)

        unknown = set(output_cols) - set(col_at[-1])
        if unknown:
            raise ValueError(f"Columns not produced by this pipeline: {sorted(unknown)}")

        needed: set[str] = set(output_cols)
        keep_indices: list[int] = []

        for i in range(len(self.steps) - 1, -1, -1):
            _, transformer = self.steps[i]
            before = set(col_at[i])
            after  = set(col_at[i + 1])
            new_cols    = after - before
            subset_cols = set(_transformer_source_columns(transformer))

            if new_cols & needed:
                # Step creates columns we need → keep; its source cols become needed.
                # Only propagate passthrough cols that originate from the pipeline
                # input (not from other column-adding steps we may later drop).
                passthrough = (before - new_cols) & (set(self._input_columns) | subset_cols)
                keep_indices.append(i)
                needed = (needed - new_cols) | subset_cols | passthrough
            elif not new_cols and subset_cols & needed & before:
                # Step modifies values inplace on needed columns — no new dependencies
                keep_indices.append(i)

        # Forward-simulate through the TRIMMED steps to get the narrowed column
        # set at each step boundary (different from the original col_at because
        # dropped steps no longer add intermediate columns).
        sorted_keep = sorted(keep_indices)
        trimmed_cols: list[list[str]] = [list(self._input_columns)]
        cur = list(self._input_columns)
        for i in sorted_keep:
            cur = _goc(self.steps[i][1], cur)
            trimmed_cols.append(cur)

        # Deepcopy kept transformers and restrict their subset/mappings to
        # columns actually available in the trimmed context.
        import copy
        narrowed: list[tuple[str, Any]] = []
        for j, i in enumerate(sorted_keep):
            name, transformer = self.steps[i]
            available = set(trimmed_cols[j])  # cols before step i in trimmed pipeline
            orig_subset = set(getattr(transformer, "subset", None) or [])
            t = copy.deepcopy(transformer)

            if hasattr(t, "subset") and t.subset:
                t.subset = [c for c in t.subset if c in available]
            new_subset = set(getattr(t, "subset", None) or [])

            # Restrict column-keyed fitted dicts to available columns — but only when
            # a dict is actually keyed by subset column names (e.g. WOEEncoder's
            # _column_mapping). Feature generators (RowStatisticsFeatures, HHIFeatures,
            # GeneralizedRatioFeatures, …) key these dicts by synthetic group names
            # instead, so pruning by column availability would wipe them out entirely.
            for attr in ("_offset", "_scale", "_statistics", "_median",
                         "_clip_bounds", "_column_mapping", "_bins", "_labels"):
                val = getattr(t, attr, None)
                if isinstance(val, dict) and set(val) <= orig_subset:
                    setattr(t, attr, {k: v for k, v in val.items() if k in new_subset})

            narrowed.append((name, t))

        new_pipe = Pipeline(steps=narrowed, verbose=self.verbose)
        new_pipe._is_fitted = True
        new_pipe._input_columns = list(self._input_columns)
        new_pipe._input_dtypes  = dict(self._input_dtypes)
        return new_pipe

