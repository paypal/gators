from typing import Annotated, Dict, List, Optional

import polars as pl
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin


class DropHighNaNRatio(BaseModel, BaseEstimator, TransformerMixin):
    """
    Drops columns with a high ratio of NaN values.

    Parameters
    ----------
    max_ratio : float
        Maximum allowed ratio of NaN values (0.0-1.0). Columns with NaN ratio >= max_ratio will be dropped.
    subset : Optional[List[str]], default=None
        List of columns to check for high NaN ratio. If None, all columns are checked.

    Examples
    --------
    Initializing and using `DropHighNaNRatio` transformer.

    Example when `drop_columns` is True and `columns` is None:

    >>> import polars as pl
    >>> from gators.data_cleaning import DropHighNaNRatio
    >>> X = pl.DataFrame({
    ...     "col1": ["a", None, "b", "c"],
    ...     "col2": ["x", "x", "x", None],
    ...     "col3": [1, 2, None, None]
    ... })
    >>> transformer = DropHighNaNRatio(max_ratio=0.5)
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 1)
    ┌─────┐
    │ col1│
    │ str │
    ├─────┤
    │  a  │
    │ None│
    │  b  │
    │  c  │
    └─────┘

    Example when `drop_columns` is True and `columns` is a subset:

    >>> transformer = DropHighNaNRatio(max_ratio=0.5, subset=['col2', 'col3'])
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌─────┬─────┐
    │ col1│ col2│
    │ str │ str │
    ├─────┼─────┤
    │  a  │  x  │
    │ None│  x  │
    │  b  │  x  │
    │  c  │ None│
    └─────┴─────┘

    Example when `drop_columns` is False and `columns` is None:

    >>> transformer = DropHighNaNRatio(max_ratio=0.5)
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 3)
    ┌─────┬─────┬──────┐
    │ col1│ col2│ col3 │
    │ str │ str │ i64  │
    ├─────┼─────┼──────┤
    │  a  │  x  │   1  │
    │ None│  x  │   2  │
    │  b  │  x  │ None │
    │  c  │ None│ None │
    └─────┴─────┴──────┘

    Example when `drop_columns` is False and `columns` is a subset:

    >>> transformer = DropHighNaNRatio(max_ratio=0.5, subset=['col2', 'col3'])
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 3)
    ┌─────┬─────┬──────┐
    │ col1│ col2│ col3 │
    │ str │ str │ i64  │
    ├─────┼─────┼──────┤
    │  a  │  x  │   1  │
    │ None│  x  │   2  │
    │  b  │  x  │ None │
    │  c  │ None│ None │
    └─────┴─────┴──────┘
    """

    max_ratio: Annotated[float, Field(ge=0.0, le=1.0)]
    subset: Optional[List[str]] = None
    _to_drop: List[str]
    _column_mapping = Dict[str, str]

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "DropHighNaNRatio":
        """Fit the transformer by identifying columns with high NaN ratios.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        DropHighNaNRatio
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = list(X.columns)
        ratios = X[self.subset].with_columns(pl.all().is_null()).mean().row(0, named=True)
        self._to_drop = [col for col, ratio in ratios.items() if ratio >= self.max_ratio]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by dropping columns with high NaN ratios.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with high-NaN columns removed.
        """
        return X.drop(self._to_drop)
