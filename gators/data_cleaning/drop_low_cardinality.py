from typing import Annotated

import polars as pl
from pydantic import Field, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class DropLowCardinality(_BaseTransformer):
    """
    Drops columns with low cardinality.

    Parameters
    ----------
    min_count : int
        Minimum number of unique values for a column to be retained. Must be >= 1.
        Columns with unique count < min_count will be dropped.
    subset : list[str], default=None
        List of columns to check for low cardinality. If None, all string, boolean, and categorical columns are checked.

    Examples
    --------
    Initializing and using `DropLowCardinality` transformer.

    Example when `drop_columns` is True and `columns` is None:

    >>> X = pl.DataFrame({
    ...     "col1": ["a", "a", "b", "c"],
    ...     "col2": ["x", "x", "x", "y"],
    ...     "col3": [1, 2, 3, 4]
    ... })
    >>> transformer = DropLowCardinality(min_count=2, columns=None, drop_columns=True)
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 1)
    ┌─────┐
    │ col3│
    │ i64 │
    ├─────┤
    │  1  │
    │  2  │
    │  3  │
    │  4  │
    └─────┘

    Example when `drop_columns` is True and `columns` is a subset:

    >>> transformer = DropLowCardinality(min_count=2, subset=['col1'], drop_columns=True)
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌─────┬─────┐
    │ col2│ col3│
    │ str │ i64 │
    ├─────┼─────┤
    │  x  │  1  │
    │  x  │  2  │
    │  x  │  3  │
    │  y  │  4  │
    └─────┴─────┘

    Example when `drop_columns` is False and `columns` is None:

    >>> transformer = DropLowCardinality(min_count=2, columns=None, drop_columns=False)
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 3)
    ┌─────┬─────┬─────┐
    │ col1│ col2│ col3│
    │ str │ str │ i64 │
    ├─────┼─────┼─────┤
    │  a  │  x  │  1  │
    │  a  │  x  │  2  │
    │  b  │  x  │  3  │
    │  c  │  y  │  4  │
    └─────┴─────┴─────┘

    Example when `drop_columns` is False and `columns` is a subset:

    >>> transformer = DropLowCardinality(min_count=2, subset=['col1'], drop_columns=False)
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 3)
    ┌─────┬─────┬─────┐
    │ col1│ col2│ col3│
    │ str │ str │ i64 │
    ├─────┼─────┼─────┤
    │  a  │  x  │  1  │
    │  a  │  x  │  2  │
    │  b  │  x  │  3  │
    │  c  │  y  │  4  │
    └─────┴─────┴─────┘
    """

    min_count: Annotated[int, Field(ge=1)]
    subset: list[str] | None = None
    _to_drop: list[str] = PrivateAttr(default_factory=list)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "DropLowCardinality":
        """Fit the transformer by identifying columns with low cardinality.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        DropLowCardinality
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype in [pl.String, pl.Boolean, pl.Enum]
            ]

        if not self.subset:
            self._to_drop = []
            return self

        counts = X[self.subset].with_columns(pl.all().n_unique()).row(0, named=True)
        self._to_drop = [col for col, val in counts.items() if val < self.min_count]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by dropping low cardinality columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with low-cardinality columns removed.
        """
        return X.drop(self._to_drop)
