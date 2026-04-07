from typing import Dict, List, Optional, cast

import polars as pl
from pydantic import BaseModel, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Literal


class StringImputer(BaseModel, BaseEstimator, TransformerMixin):
    """
    Impute missing values in string columns of a Polars DataFrame.

    Parameters
    ----------
    strategy : Literal['constant', 'most_frequent']
        Strategy to use for imputing missing values.

        - "constant": Fill with a constant value specified by `value`
        - "most_frequent": Fill with the mode (most frequent value)
    subset : Optional[List[str]], default=None
        List of string columns to impute. If None, all string columns are selected.
    value : Optional[str], default=None
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
    >>> from gators.imputers import StringImputer
    >>> X = pl.DataFrame({
    ...     'col1': ['a', None, 'b', None],
    ...     'col2': ['cat', 'dog', 'mouse', None],
    ...     'col3': ['Paris', 'London', 'Berlin', '']
    ... })
    >>> imputer = StringImputer(strategy='most_frequent', inplace=False)
    >>> imputer.fit(X)
    StringImputer(strategy='most_frequent', subset=['col1', 'col2', 'col3'], value=None, drop_columns=True, inplace=False)
    >>> X_imputed = imputer.transform(X)
    >>> print(X_imputed)
    shape: (4, 3)
    ┌────────────────────────────┬────────────────────────────┬────────────────────────────┐
    │ col1__impute_most_frequent ┆ col2__impute_most_frequent ┆ col3__impute_most_frequent │
    │ ---                        ┆ ---                        ┆ ---                        │
    │ str                        ┆ str                        ┆ str                        │
    ╞════════════════════════════╪════════════════════════════╪════════════════════════════╡
    │ a                          ┆ cat                        ┆ Paris                      │
    │ a                          ┆ dog                        ┆ London                     │
    │ b                          ┆ mouse                      ┆ Berlin                     │
    │ a                          ┆ cat                        ┆                            │
    └────────────────────────────┴────────────────────────────┴────────────────────────────┘

    >>> imputer = StringImputer(strategy='most_frequent', drop_columns=False, inplace=False)
    >>> X_imputed = imputer.fit_transform(X)
    >>> print(X_imputed)
    shape: (4, 6)
    ┌──────┬───────┬────────┬──────────────────┬─────────────────┬─────────────────┐
    │ col1 ┆ col2  ┆ col3   ┆ col1__impute_mos ┆ col2__impute_mo ┆ col3__impute_mo │
    │ ---  ┆ ---   ┆ ---    ┆ t_frequent       ┆ st_frequent     ┆ st_frequent     │
    │ str  ┆ str   ┆ str    ┆ ---              ┆ ---             ┆ ---             │
    │      ┆       ┆        ┆ str              ┆ str             ┆ str             │
    ╞══════╪═══════╪════════╪══════════════════╪═════════════════╪═════════════════╡
    │ a    ┆ cat   ┆ Paris  ┆ a                ┆ cat             ┆ Paris           │
    │ null ┆ dog   ┆ London ┆ a                ┆ dog             ┆ London          │
    │ b    ┆ mouse ┆ Berlin ┆ b                ┆ mouse           ┆ Berlin          │
    │ null ┆ null  ┆        ┆ a                ┆ cat             ┆                 │
    └──────┴───────┴────────┴──────────────────┴─────────────────┴─────────────────┘
    """

    strategy: Literal["constant", "most_frequent"]
    subset: Optional[List[str]] = None
    value: Optional[str] = None
    drop_columns: bool = True
    inplace: bool = True
    _statistics: Dict[str, Optional[str]] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "StringImputer":
        """Fit the transformer by computing imputation statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with string columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        StringImputer
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [col for col, dtype in zip(X.columns, X.dtypes) if dtype == pl.String]
        if not self.inplace:
            self._column_mapping = {col: f"{col}__impute_{self.strategy}" for col in self.subset}

        if self.strategy == "constant":
            self._statistics = {col: self.value for col in self.subset}
        else:  # most_frequent
            # Compute all modes in single pass, handle ties by taking smallest value (alphabetically)
            self._statistics = {col: X[col].drop_nulls().mode().sort()[0] for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by imputing missing values in string columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with string columns containing null values.

        Returns
        -------
        pl.DataFrame
            DataFrame with imputed string columns.
        """
        # Ensure columns is set (should be set during fit)
        columns = cast(List[str], self.subset)

        if self.inplace:
            transformations = [pl.col(col).fill_null(self._statistics[col]) for col in columns]
            return X.with_columns(transformations)

        transformations = [
            pl.col(col).fill_null(self._statistics[col]).alias(new)
            for col, new in self._column_mapping.items()
        ]
        X = X.with_columns(transformations)
        if self.drop_columns:
            return X.drop(columns)
        return X
