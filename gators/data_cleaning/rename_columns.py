from typing import Dict, Optional

import polars as pl
from pydantic import BaseModel, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin


class RenameColumns(BaseModel, BaseEstimator, TransformerMixin):
    """
    Renames columns based on a provided mapping.

    Parameters
    ----------
    column_mapping : Dict[str, str]
        Dictionary mapping original column names to new column names.

    Examples
    --------
    Example when renaming all columns:

    >>> import polars as pl
    >>> from gators.data_cleaning import RenameColumns
    >>> X = pl.DataFrame({
    ...     "col1": ["a", "a", "b", "c"],
    ...     "col2": ["x", "x", "x", "y"],
    ...     "col3": [1, 2, 3, 4]
    ... })
    >>> transformer = RenameColumns(column_mapping={"col1": "column1", "col2": "column2", "col3": "column3"})
    >>> transformer.fit(X)
    ...
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 3)
    ┌────────┬────────┬────────┐
    │ column1│ column2│ column3│
    │ str    │ str    │ i64    │
    ├────────┼────────┼────────┤
    │ a      │  x     │  1     │
    │ a      │  x     │  2     │
    │ b      │  x     │  3     │
    │ c      │  y     │  4     │
    └────────┴────────┴────────┘
    """

    column_mapping: Dict[str, str]
    _column_mapping: Dict[str, str] = PrivateAttr()

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "RenameColumns":
        """Fit the transformer by storing the column mapping.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        RenameColumns
            The fitted transformer instance.
        """
        self._column_mapping = {col: new_col for col, new_col in self.column_mapping.items()}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by extracting specified components.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        return X.rename(self.column_mapping)
