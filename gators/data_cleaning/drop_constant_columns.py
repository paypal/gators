from typing import Dict, List, Optional

import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class DropConstantColumns(_BaseTransformer):
    """
    Removes columns that have only a single unique value (constant columns).

    Identifies and removes columns with zero information content. More specific
    than VarianceFilter (which only works on numerics) and faster than variance
    calculation. Handles both numeric and categorical constant columns.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of columns to check for constant values. If None, all columns
        are checked.
    include_na : bool, default=True
        Whether to count NaN/null as a unique value. If True, a column with
        all NaN is considered constant. If False, NaN values are ignored
        when counting unique values.

    Examples
    --------
    **Example 1: Remove constant numeric column**

    >>> from gators.data_cleaning import DropConstantColumns
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'id': [1, 2, 3, 4, 5],
    ...     'constant_num': [42, 42, 42, 42, 42],
    ...     'varying': [10, 20, 30, 40, 50]
    ... })
    >>> remover = DropConstantColumns()
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (5, 2)
    ┌─────┬─────────┐
    │ id  ┆ varying │
    │ --- ┆ ---     │
    │ i64 ┆ i64     │
    ├─────┼─────────┤
    │ 1   ┆ 10      │
    │ 2   ┆ 20      │
    │ 3   ┆ 30      │
    │ 4   ┆ 40      │
    │ 5   ┆ 50      │
    └─────┴─────────┘

    **Example 2: Remove constant categorical column**

    >>> X = pl.DataFrame({
    ...     'country': ['USA', 'USA', 'USA', 'USA'],
    ...     'city': ['NYC', 'LA', 'Chicago', 'Boston'],
    ...     'status': ['active', 'active', 'active', 'active']
    ... })
    >>> remover = DropConstantColumns()
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (4, 1)
    ┌─────────┐
    │ city    │
    │ ---     │
    │ str     │
    ├─────────┤
    │ NYC     │
    │ LA      │
    │ Chicago │
    │ Boston  │
    └─────────┘

    **Example 3: Handle NaN values (with include_na=True)**

    >>> X = pl.DataFrame({
    ...     'all_null': [None, None, None],
    ...     'mixed': [1, None, 1],
    ...     'varying': [1, 2, 3]
    ... })
    >>> remover = DropConstantColumns(include_na=True)
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (3, 2)
    ┌───────┬─────────┐
    │ mixed ┆ varying │
    │ ---   ┆ ---     │
    │ i64   ┆ i64     │
    ├───────┼─────────┤
    │ 1     ┆ 1       │
    │ null  ┆ 2       │
    │ 1     ┆ 3       │
    └───────┴─────────┘

    **Example 4: Handle NaN values (with include_na=False)**

    >>> remover = DropConstantColumns(include_na=False)
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (3, 1)
    ┌─────────┐
    │ varying │
    │ ---     │
    │ i64     │
    ├─────────┤
    │ 1       │
    │ 2       │
    │ 3       │
    └─────────┘

    **Example 5: Subset of columns**

    >>> X = pl.DataFrame({
    ...     'col1': [1, 1, 1],
    ...     'col2': [5, 5, 5],
    ...     'col3': [10, 20, 30]
    ... })
    >>> remover = DropConstantColumns(subset=['col1', 'col2'])
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (3, 1)
    ┌──────┐
    │ col3 │
    │ ---  │
    │ i64  │
    ├──────┤
    │ 10   │
    │ 20   │
    │ 30   │
    └──────┘
    """

    subset: Optional[List[str]] = None
    include_na: bool = True
    _to_drop: List[str] = PrivateAttr(default_factory=list)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "DropConstantColumns":
        """Fit the transformer by identifying constant columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        DropConstantColumns
            Fitted transformer instance.
        """
        # Use all columns if not specified
        columns_to_check = self.subset if self.subset is not None else X.columns

        self._to_drop = []

        for col in columns_to_check:
            if self.include_na:
                # Count all unique values including NaN
                n_unique = X[col].n_unique()
            else:
                # Count unique values excluding NaN
                n_unique = X[col].drop_nulls().n_unique()

            # If only 1 unique value (or 0 when excluding NaN), it's constant
            if n_unique <= 1:
                self._to_drop.append(col)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by removing constant columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with constant columns removed.
        """
        if self._to_drop:
            return X.drop(self._to_drop)
        return X
