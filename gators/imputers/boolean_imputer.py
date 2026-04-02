from typing import Dict, List, Optional, cast

import polars as pl
from pydantic import BaseModel, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Literal


class BooleanImputer(BaseModel, BaseEstimator, TransformerMixin):
    """
    Imputes missing values in boolean columns using a specified strategy.

    Parameters
    ----------
    strategy : Literal["constant", "most_frequent"]
        Strategy to use for imputing missing values.
    
        - "constant": Fill with a constant value specified by `value`
        - "most_frequent": Fill with the mode (most frequent value)
    subset : Optional[List[str]], default=None
        List of boolean columns to impute. If None, all boolean columns are selected.
    value : Optional[bool], default=None
        Value to use when strategy is 'constant'. Must be True or False.
        Required when strategy='constant', ignored otherwise.
    inplace : bool, default=True
        If True, impute values in the original columns.
        If False, create new columns with suffix '__impute_{strategy}'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after imputation.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.imputers import BooleanImputer

    >>> # Sample data
    >>> X =pl.DataFrame({
    ...     'A': [True, False, None, True, None],
    ...     'B': [False, None, True, None, False]
    ... })

    >>> # Impute with 'most_frequent' strategy
    >>> imputer = BooleanImputer(strategy="most_frequent", inplace=False)
    >>> _ = imputer.fit(X)
    >>> transformed_X =imputer.transform(X)
    >>> print(transformed_X)
    shape: (5, 2)
    ┌─────────────────────────┬─────────────────────────┐
    │ A__impute_most_frequent ┆ B__impute_most_frequent │
    │ ---                     ┆ ---                     │
    │ bool                    ┆ bool                    │
    ╞═════════════════════════╪═════════════════════════╡
    │ true                    ┆ false                   │
    │ false                   ┆ false                   │
    │ true                    ┆ true                    │
    │ true                    ┆ false                   │
    │ true                    ┆ false                   │
    └─────────────────────────┴─────────────────────────┘

    >>> # Impute with 'constant' strategy
    >>> from gators.imputers import BooleanImputer
    >>> imputer = BooleanImputer(strategy="constant", value=False, drop_columns=False, inplace=False)
    >>> _ = imputer.fit(X)
    >>> transformed_X =imputer.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌───────┬───────┬────────────────────┬────────────────────┐
    │ A     ┆ B     ┆ A__impute_constant ┆ B__impute_constant │
    │ ---   ┆ ---   ┆ ---                ┆ ---                │
    │ bool  ┆ bool  ┆ bool               ┆ bool               │
    ╞═══════╪═══════╪════════════════════╪════════════════════╡
    │ true  ┆ false ┆ true               ┆ false              │
    │ false ┆ null  ┆ false              ┆ false              │
    │ null  ┆ true  ┆ false              ┆ true               │
    │ true  ┆ null  ┆ true               ┆ false              │
    │ null  ┆ false ┆ false              ┆ false              │
    └───────┴───────┴────────────────────┴────────────────────┘

    >>> # Impute with columns specified
    >>> from gators.imputers import BooleanImputer
    >>> imputer = BooleanImputer(strategy="constant", value=True, subset=['B'], drop_columns=False, inplace=False)
    >>> _ = imputer.fit(X)
    >>> transformed_X =imputer.transform(X)
    >>> print(transformed_X)
    shape: (5, 3)
    ┌───────┬───────┬────────────────────┐
    │ A     ┆ B     ┆ B__impute_constant │
    │ ---   ┆ ---   ┆ ---                │
    │ bool  ┆ bool  ┆ bool               │
    ╞═══════╪═══════╪════════════════════╡
    │ true  ┆ false ┆ false              │
    │ false ┆ null  ┆ true               │
    │ null  ┆ true  ┆ true               │
    │ true  ┆ null  ┆ true               │
    │ null  ┆ false ┆ false              │
    └───────┴───────┴────────────────────┘
    """

    strategy: Literal["constant", "most_frequent"]
    subset: Optional[List[str]] = None
    value: Optional[bool] = None
    drop_columns: bool = True
    inplace: bool = True
    _statistics: Dict[str, bool] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "BooleanImputer":
        """Fit the transformer by computing imputation statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with boolean columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        BooleanImputer
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in zip(X.columns, X.dtypes) if dtype in [pl.Boolean]
            ]
        if not self.inplace:
            self._column_mapping = {
                col: f"{col}__impute_{self.strategy}" for col in self.subset
            }
        strategies = {
            "most_frequent": lambda col: bool(X[col].drop_nulls().mode()[0]),
            "constant": lambda col: bool(self.value),
        }
        self._statistics = {col: strategies[self.strategy](col) for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by imputing missing values in boolean columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with boolean columns containing null values.

        Returns
        -------
        pl.DataFrame
            DataFrame with imputed boolean columns.
        """
        # Ensure columns is set (should be set during fit)
        columns = cast(List[str], self.subset)
        
        if self.inplace:
            transformations = [
                pl.col(col).fill_null(self._statistics[col]) for col in columns
            ]
            return X.with_columns(transformations)

        transformations = [
            pl.col(col).fill_null(self._statistics[col]).alias(new)
            for col, new in self._column_mapping.items()
        ]
        X = X.with_columns(transformations)
        if self.drop_columns:
            return X.drop(columns)
        return X
