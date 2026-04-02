from typing import Dict, List, Optional, cast

import polars as pl
from pydantic import BaseModel, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin


class CastColumns(BaseModel, BaseEstimator, TransformerMixin):
    """
    Casts specified columns to a given data type.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of column names to cast. If None, all columns will be cast.
    dtype : type
        Target Polars data type (e.g., pl.Float64, pl.String, pl.Int64, pl.Datetime, pl.Date).
    inplace : bool, default=True
        If True, cast values in the original columns.
        If False, create new columns with suffix '__cast_{dtype}'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after casting.
        Ignored when inplace=True.

    Examples
    --------
    **Example 1: Cast columns with inplace=False and keep originals**

    >>> import polars as pl
    >>> from gators.data_cleaning import CastColumns
    >>> X = pl.DataFrame({
    ...     "col1": ["10", "20", "30"],
    ...     "col2": ["1.1", "2.2", "3.3"],
    ...     "col3": [True, False, True]
    ... })
    >>> cast_columns = CastColumns(
    ...     subset=["col1", "col2"],
    ...     dtype=pl.Float64,
    ...     inplace=False,
    ...     drop_columns=False
    ... )
    >>> cast_columns.fit(X)
    >>> transformed_X = cast_columns.transform(X)
    >>> print(transformed_X)
    shape: (3, 5)
    ┌──────┬──────┬────────────────────┬────────────────────┬───────┐
    │ col1 │ col2 │ col1__cast_float64 │ col2__cast_float64 │ col3  │
    ├──────┼──────┼────────────────────┼────────────────────┼───────┤
    │ 10   │ 1.1  │ 10.0               │ 1.1                │ True  │
    ├──────┼──────┼────────────────────┼────────────────────┼───────┤
    │ 20   │ 2.2  │ 20.0               │ 2.2                │ False │
    ├──────┼──────┼────────────────────┼────────────────────┼───────┤
    │ 30   │ 3.3  │ 30.0               │ 3.3                │ True  │
    └──────┴──────┴────────────────────┴────────────────────┴───────┘

    **Example 2: Cast columns with inplace=False and drop originals**

    >>> cast_columns = CastColumns(
    ...     subset=["col1", "col2"],
    ...     dtype=pl.Float64,
    ...     inplace=False,
    ...     drop_columns=True
    ... )
    >>> cast_columns.fit(X)
    >>> transformed_X = cast_columns.transform(X)
    >>> print(transformed_X)
    shape: (3, 3)
    ┌────────────────────┬────────────────────┬───────┐
    │ col1__cast_float64 │ col2__cast_float64 │ col3  │
    ├────────────────────┼────────────────────┼───────┤
    │ 10.0               │ 1.1                │ True  │
    ├────────────────────┼────────────────────┼───────┤
    │ 20.0               │ 2.2                │ False │
    ├────────────────────┼────────────────────┼───────┤
    │ 30.0               │ 3.3                │ True  │
    └────────────────────┴────────────────────┴───────┘

    **Example 3: Cast columns in place**

    >>> cast_columns = CastColumns(
    ...     subset=["col1", "col2"],
    ...     dtype=pl.Float64,
    ...     inplace=True
    ... )
    >>> cast_columns.fit(X)
    >>> transformed_X = cast_columns.transform(X)
    >>> print(transformed_X)
    shape: (3, 3)
    ┌──────┬──────┬───────┐
    │ col1 │ col2 │ col3  │
    ├──────┼──────┼───────┤
    │ 10.0 │ 1.1  │ True  │
    ├──────┼──────┼───────┤
    │ 20.0 │ 2.2  │ False │
    ├──────┼──────┼───────┤
    │ 30.0 │ 3.3  │ True  │
    └──────┴──────┴───────┘

    Notes
    -----

    - When casting to Datetime or Date from String, the transformer handles format parsing automatically
    - If subset=None, all columns in the DataFrame will be cast to the specified dtype
    - When inplace=True, the drop_columns parameter is ignored as original columns are replaced
    """

    subset: Optional[List[str]] = None
    dtype: type
    inplace: bool = True
    drop_columns: bool = True
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CastColumns":
        """Fit the transformer by identifying columns to cast.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        CastColumns
            The fitted transformer instance.
        """
        if self.subset is None:
            self.subset = X.columns
        else:
            available_columns = X.columns
            self.subset = [col for col in self.subset if col in available_columns]
        if not self.inplace:
            self._column_mapping = {
                col: f"{col}__cast_{str(self.dtype).lower()}" for col in self.subset
            }
        return self

    def transform(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> pl.DataFrame:
        """Transform the DataFrame by casting columns to the target type.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        pl.DataFrame
            DataFrame with cast columns.
        """
        # Ensure columns is set (should be set during fit)
        columns = cast(List[str], self.subset)

        if self.inplace:
            transformations = []
            for col in columns:
                if self.dtype == pl.Datetime and X[col].dtype == pl.String:
                    # Handle string to datetime conversion
                    transformations.append(
                        pl.col(col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias(col)
                    )
                elif self.dtype == pl.Date and X[col].dtype == pl.String:
                    # Handle string to date conversion
                    transformations.append(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d").alias(col))
                else:
                    transformations.append(pl.col(col).cast(self.dtype, strict=False))
            return X.with_columns(transformations)

        transformations = []
        for col, new in self._column_mapping.items():
            if self.dtype == pl.Datetime and X[col].dtype == pl.String:
                # Handle string to datetime conversion
                transformations.append(
                    pl.col(col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias(new)
                )
            elif self.dtype == pl.Date and X[col].dtype == pl.String:
                # Handle string to date conversion
                transformations.append(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d").alias(new))
            else:
                transformations.append(X[col].cast(self.dtype).alias(new))

        X = X.with_columns(transformations)
        if self.drop_columns:
            return X.drop(columns)
        return X
