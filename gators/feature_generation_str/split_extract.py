from typing import List, Optional

import polars as pl
from pydantic import NonNegativeInt

from ..transformer._base_transformer import _BaseTransformer


class SplitExtract(_BaseTransformer):
    """
    Generates features by splitting columns and extracting a specific part by index.

    Parameters
    ----------
    subset : List[str]
        List of column names to split and extract from.
    by : str
        Delimiter to split the columns by.
    n : NonNegativeInt
        Index of the part to extract from the split (0-indexed).
    drop_columns : bool, optional
        Whether to drop the original columns after splitting, by default True.

    Examples
    --------
    >>> from gators.feature_generation_str import SplitExtract
    >>> import polars as pl

    >>> X ={'full_name': ['John Doe', 'Jane Smith', 'Alice Johnson']}
    >>> X = pl.DataFrame(X)

    **Example 1: Extract first part (n=0)**

    >>> transformer = SplitExtract(subset=['full_name'], by=' ', n=0, drop_columns=True)
    >>> transformer.fit(X)
    SplitExtract(subset=['full_name'], by=' ', n=0, drop_columns=True)
    >>> result = transformer.transform(X)
    >>> result
    shape: (3, 1)
    ┌────────────────────┐
    │ full_name__split_0 │
    │         str        │
    ├────────────────────┤
    │        John        │
    │        Jane        │
    │       Alice        │
    └────────────────────┘

    **Example 2: Extract second part (n=1)**

    >>> transformer = SplitExtract(subset=['full_name'], by=' ', n=1, drop_columns=False)
    >>> transformer.fit(X)
    SplitExtract(subset=['full_name'], by=' ', n=1, drop_columns=False)
    >>> result = transformer.transform(X)
    >>> result
    shape: (3, 2)
    ┌──────────────────┬────────────────────┐
    │    full_name     │ full_name__split_1 │
    │       str        │         str        │
    ├──────────────────┼────────────────────┤
    │     John Doe     │        Doe         │
    │    Jane Smith    │       Smith        │
    │   Alice Johnson  │      Johnson       │
    └──────────────────┴────────────────────┘
    """

    subset: List[str]
    by: str
    n: NonNegativeInt
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "SplitExtract":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        SplitExtract
            Fitted transformer instance.
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
        # Split and extract in a single expression using a subquery approach
        new_columns = []
        by = self.by.replace(" ", "_")
        for col in self.subset:
            split_expr = pl.col(col).str.split(by=self.by)
            new_columns.append(
                pl.when(split_expr.list.len() > self.n)
                .then(split_expr.list.get(self.n))
                .otherwise(None)
                .alias(f"{col}__split_{by}_{self.n}")
            )

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
