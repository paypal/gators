from typing import Dict, List, Literal, Optional, Union

import polars as pl
from pydantic import BaseModel, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin


class NumericImputer(BaseModel, BaseEstimator, TransformerMixin):
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
    subset : Optional[List[str]], default=None
        List of numeric columns to impute. If None, all numeric columns are selected.
    value : Optional[Union[int, float]], default=None
        Value to use when strategy is 'constant'. Required when strategy='constant', ignored otherwise.
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
    NumericImputer(strategy='mean', subset=['A', 'B', 'C'], value=None, drop_columns=True, inplace=False)
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
    NumericImputer(strategy='mean', subset=['A', 'B', 'C'], value=None, drop_columns=False, inplace=False)
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
    NumericImputer(strategy='mean', subset=['A'], value=None, drop_columns=True, inplace=False)
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
    subset: Optional[List[str]] = None
    value: Optional[Union[int, float]] = None
    drop_columns: bool = True
    inplace: bool = True
    _statistics: Dict[str, Union[int, float]] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "NumericImputer":
        """Fit the transformer by computing imputation statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        NumericImputer
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype not in [pl.String, pl.Boolean]
            ]
        if not self.inplace:
            self._column_mapping = {col: f"{col}__impute_{self.strategy}" for col in self.subset}

        # Only compute statistics for strategies that need them
        if self.strategy == "constant":
            self._statistics = {col: self.value for col in self.subset}
        elif self.strategy == "median":
            # Compute all medians in single pass
            self._statistics = dict(
                zip(
                    self.subset,
                    X.select([pl.col(c).median() for c in self.subset]).row(0),
                )
            )
        elif self.strategy == "most_frequent":
            # Compute all modes in single pass, handle ties by taking smallest value
            self._statistics = {
                col: X[col].drop_nulls().drop_nans().mode().sort()[0] for col in self.subset
            }
        # No statistics needed for mean, min, max, forward, backward, zero, one

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
        # Build all transformations at once based on strategy
        if self.strategy in [
            "mean",
            "min",
            "max",
            "forward",
            "backward",
            "zero",
            "one",
        ]:
            # Use Polars built-in strategies
            if self.inplace:
                transformations = [
                    pl.col(col).fill_null(strategy=self.strategy) for col in self.subset
                ]
            else:
                transformations = [
                    pl.col(col).fill_null(strategy=self.strategy).alias(new)
                    for col, new in self._column_mapping.items()
                ]
        else:
            # Use pre-computed statistics (constant, median, most_frequent)
            if self.inplace:
                transformations = [
                    pl.col(col).fill_null(self._statistics[col]) for col in self.subset
                ]
            else:
                transformations = [
                    pl.col(col).fill_null(self._statistics[col]).alias(new)
                    for col, new in self._column_mapping.items()
                ]

        X = X.with_columns(transformations)

        if not self.inplace and self.drop_columns:
            return X.drop(self.subset)
        return X
