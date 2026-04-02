from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseModel, BaseEstimator, TransformerMixin):
    """
    Drops specified columns from a DataFrame.

    Parameters
    ----------
    subset : List[str]
        List of column names to drop.

    Examples
    --------
    Create an instance of the DropColumns class:

    >>> import polars as pl
    >>> from gators.data_cleaning import DropColumns
    >>> drop_columns = DropColumns(subset=["col1", "col2"])

    Fit the transformer:

    >>> drop_columns.fit(X)

    Transform the DataFrame:

    >>> X = pl.DataFrame({"col1": [1, 2, 3],
    ...                    "col2": ["A", "B", "C"],
    ...                    "col3": [True, False, True]})
    >>> transformed_X = drop_columns.transform(X)
    >>> print(transformed_X)
    shape: (3, 1)
    ┌───────┐
    │ col3  │
    ├───────┤
    │ True  │
    ├───────┤
    │ False │
    ├───────┤
    │ True  │
    └───────┘

    """

    subset: List[str]
    _column_mapping = Dict[str, str]

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "DropColumns":
        """Fit the transformer (no-op for DropColumns).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        DropColumns
            The fitted transformer instance.
        """
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
        return X.drop(self.subset)
