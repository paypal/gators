from typing import List, Optional

import polars as pl
from pydantic import BaseModel, PositiveInt
from sklearn.base import BaseEstimator, TransformerMixin


class Split(BaseModel, BaseEstimator, TransformerMixin):
    """
    Generates features by splitting columns into multiple columns based on a delimiter.

    The number of split columns is specified by the max_splits parameter to ensure
    consistent column structure between training and new data.

    Parameters
    ----------
    subset : List[str]
        List of column names to split.
    by : str
        Delimiter to split the columns by.
    max_splits : PositiveInt
        Maximum number of split columns to create. If a value has more splits,
        extra splits are truncated. If fewer, remaining columns are filled with empty strings.
    drop_columns : bool, optional
        Whether to drop the original columns after splitting, by default True.

    Examples
    --------
    >>> from gators.feature_generation_str import Split
    >>> import polars as pl

    >>> X ={'full_name': ['John Doe', 'Jane Smith Williams', 'Alice Johnson']}
    >>> X = pl.DataFrame(X)

    **Example 1: Split with max_splits=3 and drop_columns=True (default)**

    >>> transformer = Split(subset=['full_name'], by=' ', max_splits=3, drop_columns=True)
    >>> transformer.fit(X)
    Split(subset=['full_name'], by=' ', max_splits=3, drop_columns=True)
    >>> result = transformer.transform(X)
    >>> result
    shape: (3, 3)
    ┌─────────────────────┬─────────────────────┬─────────────────────┐
    │ full_name__split_0  │ full_name__split_1  │ full_name__split_2  │
    │         str         │         str         │         str         │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │        John         │         Doe         │                     │
    │        Jane         │        Smith        │      Williams       │
    │       Alice         │       Johnson       │                     │
    └─────────────────────┴─────────────────────┴─────────────────────┘

    **Example 2: Split with max_splits=2 and drop_columns=False**

    >>> transformer = Split(subset=['full_name'], by=' ', max_splits=2, drop_columns=False)
    >>> transformer.fit(X)
    Split(subset=['full_name'], by=' ', max_splits=2, drop_columns=False)
    >>> result = transformer.transform(X)
    >>> result
    shape: (3, 3)
    ┌──────────────────────┬─────────────────────┬─────────────────────┐
    │      full_name       │ full_name__split_0  │ full_name__split_1  │
    │         str          │         str         │         str         │
    ├──────────────────────┼─────────────────────┼─────────────────────┤
    │      John Doe        │        John         │         Doe         │
    │  Jane Smith Williams │        Jane         │        Smith        │
    │   Alice Johnson      │       Alice         │       Johnson       │
    └──────────────────────┴─────────────────────┴─────────────────────┘
    """

    subset: List[str]
    by: str
    max_splits: PositiveInt
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "Split":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        Split
            Fitted transformer instance.
        """
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by splitting columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        by_clean = self.by.replace(" ", "_")

        # Collect all column expressions
        new_columns = []
        for col in self.subset:
            for i in range(self.max_splits):
                new_col_name = f"{col}__split_{by_clean}_{i}"
                new_columns.append(
                    pl.col(col)
                    .str.split(by=self.by)
                    .list.get(i, null_on_oob=True)
                    .fill_null("")
                    .alias(new_col_name)
                )

        # Apply all columns at once
        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
