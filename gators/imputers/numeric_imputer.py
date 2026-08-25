from __future__ import annotations

import math
from typing import Literal

import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class NumericImputer(_BaseTransformer):
    """
    Impute missing values in numeric columns using various strategies.

    Parameters
    ----------
    strategy : Literal['constant', 'most_frequent', 'median', 'mean', 'min', 'max', 'forward', 'backward', 'zero', 'one']
        Strategy to use for imputing missing values.

        - 'constant': Fill missing values with `value`
        - 'most_frequent': Fill with the most frequent value in each column
        - 'median': Fill with the median of each column
        - 'mean': Fill with the mean of each column
        - 'min': Fill with the minimum value in each column
        - 'max': Fill with the maximum value in each column
        - 'forward': Fill with the previous non-null value (forward fill)
        - 'backward': Fill with the next non-null value (backward fill)
        - 'zero': Fill missing values with 0
        - 'one': Fill missing values with 1
    subset : list[str], default=None
        List of numeric columns to impute. If None, all numeric columns are selected.
    value : int | float, default=0
        Value to use when strategy is 'constant'.
    inplace : bool, default=True
        If True, impute values in the original columns.
        If False, create new columns with suffix '__impute_{strategy}'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after imputation.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.imputers import NumericImputer

    >>> # Sample DataFrame
    >>> X = pl.DataFrame({
    ...     'A': [1.0, 2.0, None, 4.0],
    ...     'B': [5.0, None, 7.0, 8.0],
    ...     'C': [None, 2.0, 3.0, 4.0]
    ... })

    >>> # Impute using the mean strategy
    >>> imputer = NumericImputer(strategy='mean', inplace=False)
    >>> imputer.fit(X)
    NumericImputer(strategy='mean', subset=['A', 'B', 'C'], value=0.0, drop_columns=True, inplace=False)
    >>> transformed_X = imputer.transform(X)
    >>> print(transformed_X)
    shape: (4, 3)
    ┌────────────────┬────────────────┬────────────────┐
    │ A__impute_mean ┆ B__impute_mean ┆ C__impute_mean │
    │ ---            ┆ ---            ┆ ---            │
    │ f64            ┆ f64            ┆ f64            │
    ╞════════════════╪════════════════╪════════════════╡
    │ 1.0            ┆ 5.0            ┆ 3.0            │
    │ 2.0            ┆ 6.666667       ┆ 2.0            │
    │ 2.333333       ┆ 7.0            ┆ 3.0            │
    │ 4.0            ┆ 8.0            ┆ 4.0            │
    └────────────────┴────────────────┴────────────────┘

    >>> # Impute using a constant value
    >>> imputer_constant = NumericImputer(strategy='constant', value=0, inplace=False)
    >>> imputer_constant.fit(X)
    NumericImputer(strategy='constant', subset=['A', 'B', 'C'], value=0, drop_columns=True, inplace=False)
    >>> transformed_X_constant = imputer_constant.transform(X)
    >>> print(transformed_X_constant)
    shape: (4, 3)
    ┌────────────────────┬────────────────────┬────────────────────┐
    │ A__impute_constant ┆ B__impute_constant ┆ C__impute_constant │
    │ ---                ┆ ---                ┆ ---                │
    │ f64                ┆ f64                ┆ f64                │
    ╞════════════════════╪════════════════════╪════════════════════╡
    │ 1.0                ┆ 5.0                ┆ 0.0                │
    │ 2.0                ┆ 0.0                ┆ 2.0                │
    │ 0.0                ┆ 7.0                ┆ 3.0                │
    │ 4.0                ┆ 8.0                ┆ 4.0                │
    └────────────────────┴────────────────────┴────────────────────┘

    >>> # Impute with drop_columns=False
    >>> imputer_no_drop = NumericImputer(strategy='mean', drop_columns=False, inplace=False)
    >>> imputer_no_drop.fit(X)
    NumericImputer(strategy='mean', subset=['A', 'B', 'C'], value=0.0, drop_columns=False, inplace=False)
    >>> transformed_X_no_drop = imputer_no_drop.transform(X)
    >>> print(transformed_X_no_drop)
    shape: (4, 6)
    ┌──────┬──────┬──────┬────────────────┬────────────────┬────────────────┐
    │ A    ┆ B    ┆ C    ┆ A__impute_mean ┆ B__impute_mean ┆ C__impute_mean │
    │ ---  ┆ ---  ┆ ---  ┆ ---            ┆ ---            ┆ ---            │
    │ f64  ┆ f64  ┆ f64  ┆ f64            ┆ f64            ┆ f64            │
    ╞══════╪══════╪══════╪════════════════╪════════════════╪════════════════╡
    │ 1.0  ┆ 5.0  ┆ null ┆ 1.0            ┆ 5.0            ┆ 3.0            │
    │ 2.0  ┆ null ┆ 2.0  ┆ 2.0            ┆ 6.666667       ┆ 2.0            │
    │ null ┆ 7.0  ┆ 3.0  ┆ 2.333333       ┆ 7.0            ┆ 3.0            │
    │ 4.0  ┆ 8.0  ┆ 4.0  ┆ 4.0            ┆ 8.0            ┆ 4.0            │
    └──────┴──────┴──────┴────────────────┴────────────────┴────────────────┘

    >>> # Impute with a subset of columns
    >>> imputer_subset = NumericImputer(strategy='mean', subset=['A'], inplace=False)
    >>> imputer_subset.fit(X)
    NumericImputer(strategy='mean', subset=['A'], value=0.0, drop_columns=True, inplace=False)
    >>> transformed_X_subset = imputer_subset.transform(X)
    >>> print(transformed_X_subset)
    shape: (4, 3)
    ┌──────┬──────┬────────────────┐
    │ B    ┆ C    ┆ A__impute_mean │
    │ ---  ┆ ---  ┆ ---            │
    │ f64  ┆ f64  ┆ f64            │
    ╞══════╪══════╪════════════════╡
    │ 5.0  ┆ null ┆ 1.0            │
    │ null ┆ 2.0  ┆ 2.0            │
    │ 7.0  ┆ 3.0  ┆ 2.333333       │
    │ 8.0  ┆ 4.0  ┆ 4.0            │
    └──────┴──────┴────────────────┘
    """

    strategy: Literal[
        "constant",
        "most_frequent",
        "median",
        "mean",
        "min",
        "max",
        "forward",
        "backward",
        "zero",
        "one",
    ]
    subset: list[str] | None = None
    value: int | float = 0.0
    drop_columns: bool = True
    inplace: bool = True
    _statistics: dict[str, int | float] = PrivateAttr(default_factory=dict)
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> NumericImputer:
        """Fit the transformer by computing imputation statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : pl.Series, default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        NumericImputer
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes, strict=False)
                if dtype not in [pl.String, pl.Boolean]
            ]
        if not self.inplace:
            self._column_mapping = {col: [f"{col}__impute_{self.strategy}"] for col in self.subset}
            self._output_dtypes = {new: X.schema[old] for old, news in self._column_mapping.items() for new in news}

        # Only compute statistics for strategies that need them
        if self.strategy == "constant":
            self._statistics = {col: self.value for col in self.subset}
        elif self.strategy == "median":
            # Compute all medians in single pass
            median_results = X.select([pl.col(c).median() for c in self.subset]).row(0)
            self._statistics: dict[str, int | float] = {}
            for i, col in enumerate(self.subset):
                val = median_results[i]
                self._statistics[col] = 0 if (val is None or (isinstance(val, float) and math.isnan(val))) else val  # type: ignore[assignment]
        elif self.strategy in ("mean", "min", "max"):
            results = X.select([getattr(pl.col(c), self.strategy)() for c in self.subset]).row(0)
            self._statistics = {
                col: (0 if (val is None or (isinstance(val, float) and math.isnan(val))) else val)
                for col, val in zip(self.subset, results, strict=False)
            }
        elif self.strategy == "most_frequent":
            self._statistics = {
                col: (
                    X[col].drop_nulls().drop_nans().mode().sort()[0]
                    if X[col].drop_nulls().drop_nans().len() > 0
                    else 0
                )
                for col in self.subset
            }
        # No statistics needed for forward, backward, zero, one

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by imputing missing values in numeric columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns containing null values.

        Returns
        -------
        pl.DataFrame
            DataFrame with imputed numeric columns.
        """
        if self.subset is None:
            return X  # pragma: no cover
        _float_dtypes = {pl.Float32, pl.Float64}
        # Build all transformations at once based on strategy
        if self.strategy in ["forward", "backward", "zero", "one"]:
            if self.inplace:
                transformations = [
                    (pl.col(col).fill_nan(None) if X.schema[col] in _float_dtypes else pl.col(col)).fill_null(strategy=self.strategy)  # type: ignore[arg-type]
                    for col in self.subset
                ]
            else:
                transformations = [
                    (pl.col(col).fill_nan(None) if X.schema[col] in _float_dtypes else pl.col(col)).fill_null(strategy=self.strategy).alias(new)  # type: ignore[arg-type]
                    for col, [new] in self._column_mapping.items()
                ]
        else:
            # Use pre-computed statistics (constant, median, most_frequent, mean, min, max).
            # For integer columns: round the stat to an int so the original dtype is preserved
            # (median/mean always compute a Float64 result even for integer input).
            _int_dtypes = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
            _round_for_int = {"mean", "min", "max", "constant", "most_frequent", "median"}

            def _stat_expr(col: str) -> pl.Expr:
                stat = self._statistics[col]
                dtype = X.schema[col]
                if dtype in _int_dtypes and self.strategy in _round_for_int:
                    stat = int(round(stat))
                expr = pl.col(col)
                if dtype in _float_dtypes:
                    expr = expr.fill_nan(stat)
                return expr.fill_null(stat)

            if self.inplace:
                transformations = [_stat_expr(col) for col in self.subset]
            else:
                transformations = [_stat_expr(col).alias(new) for col, [new] in self._column_mapping.items()]

        X = X.with_columns(transformations)

        if not self.inplace and self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X
