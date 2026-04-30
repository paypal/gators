from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class IsNull(_BaseTransformer):
    """
    Creates boolean features indicating whether values are null for specified columns.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of column names to check for null values.
        If None, all columns in the DataFrame are used.

    Examples
    --------
    >>> from is_null import IsNull
    >>> import polars as pl

    >>> X ={'A': [1, None, 3, 4],
    ...         'B': [4, 3, None, 1],
    ...         'C': [1, 2, 1, 2]}
    >>> X = pl.DataFrame(X)

    >>> transformer = IsNull(subset=['A', 'B'])
    >>> transformer.fit(X)
    IsNull(subset=['A', 'B'])
    >>> result = transformer.transform(X)
    >>> result
    shape: (4, 5)
    ┌──────┬──────┬─────┬──────────────┬──────────────┐
    │  A   │  B   │  C  │ A__is_null   │ B__is_null   │
    │ i64  │ i64  │ i64 │ bool         │ bool         │
    ├──────┼──────┼─────┼──────────────┼──────────────┤
    │  1   │  4   │  1  │ false        │ false        │
    │ null │  3   │  2  │ true         │ false        │
    │  3   │ null │  1  │ false        │ true         │
    │  4   │  1   │  2  │ false        │ false        │
    └──────┴──────┴─────┴──────────────┴──────────────┘
    """

    subset: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = {}

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "IsNull":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        IsNull
            Fitted transformer instance.
        """
        if self.subset is None:
            self.subset = X.columns

        self._column_mapping = {col: f"{col}__is_null" for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by adding is_null indicator columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with additional is_null columns.
        """
        if self.subset is None:
            return X
        new_columns = [
            pl.col(col).is_null().alias(self._column_mapping[col]) for col in self.subset
        ]
        return X.with_columns(new_columns)
