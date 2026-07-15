from typing import Literal

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class DropDuplicateRows(_BaseTransformer):
    """
    Removes duplicate rows from the DataFrame.

    Identifies and removes duplicate rows based on all columns or a subset of
    columns. Critical for preventing data leakage and ensuring data quality.

    Parameters
    ----------
    subset : list[str], default=None
        List of columns to consider for identifying duplicates. If None, all
        columns are used.
    keep : str, default='first'
        Strategy for keeping duplicates:

        - 'first': Keep first occurrence, drop subsequent duplicates
        - 'last': Keep last occurrence, drop previous duplicates
        - 'none': Drop all duplicates (keep no occurrences)

    Examples
    --------
    **Example 1: Remove full duplicate rows (keep first)**

    >>> from gators.data_cleaning import DropDuplicateRows
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'id': [1, 2, 2, 3, 4, 4],
    ...     'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
    ...     'age': [25, 30, 30, 35, 40, 40]
    ... })
    >>> remover = DropDuplicateRows(keep='first')
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (4, 3)
    ┌─────┬─────────┬─────┐
    │ id  ┆ name    ┆ age │
    │ --- ┆ ---     ┆ --- │
    │ i64 ┆ str     ┆ i64 │
    ├─────┼─────────┼─────┤
    │ 1   ┆ Alice   ┆ 25  │
    │ 2   ┆ Bob     ┆ 30  │
    │ 3   ┆ Charlie ┆ 35  │
    │ 4   ┆ David   ┆ 40  │
    └─────┴─────────┴─────┘

    **Example 2: Remove duplicates based on subset (keep last)**

    >>> X = pl.DataFrame({
    ...     'id': [1, 2, 3, 4],
    ...     'name': ['Alice', 'Bob', 'Alice', 'Bob'],
    ...     'score': [85, 90, 88, 92]
    ... })
    >>> remover = DropDuplicateRows(subset=['name'], keep='last')
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (2, 3)
    ┌─────┬───────┬───────┐
    │ id  ┆ name  ┆ score │
    │ --- ┆ ---   ┆ ---   │
    │ i64 ┆ str   ┆ i64   │
    ├─────┼───────┼───────┤
    │ 3   ┆ Alice ┆ 88    │
    │ 4   ┆ Bob   ┆ 92    │
    └─────┴───────┴───────┘

    **Example 3: Drop all duplicate occurrences (keep none)**

    >>> X = pl.DataFrame({
    ...     'user_id': [1, 2, 2, 3, 4, 4, 5],
    ...     'action': ['login', 'view', 'view', 'click', 'buy', 'buy', 'logout']
    ... })
    >>> remover = DropDuplicateRows(subset=['user_id'], keep='none')
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (3, 2)
    ┌─────────┬────────┐
    │ user_id ┆ action │
    │ ---     ┆ ---    │
    │ i64     ┆ str    │
    ├─────────┼────────┤
    │ 1       ┆ login  │
    │ 3       ┆ click  │
    │ 5       ┆ logout │
    └─────────┴────────┘

    **Example 4: Check for duplicates without subset**

    >>> X = pl.DataFrame({
    ...     'a': [1, 1, 2],
    ...     'b': [10, 10, 20],
    ...     'c': [100, 100, 200]
    ... })
    >>> remover = DropDuplicateRows()
    >>> result = remover.fit_transform(X)
    >>> print(result)
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ├─────┼─────┼─────┤
    │ 1   ┆ 10  ┆ 100 │
    │ 2   ┆ 20  ┆ 200 │
    └─────┴─────┴─────┘
    """

    subset: list[str] | None = None
    keep: Literal["first", "last", "none"] = "first"

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "DropDuplicateRows":
        """Fit the transformer by validating parameters.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        DropDuplicateRows
            Fitted transformer instance.

        Raises
        ------
        ValueError
            If subset columns are specified but not found in DataFrame.
        """
        # Validate subset columns exist
        if self.subset is not None:
            missing_cols = set(self.subset) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Subset columns {missing_cols} not found in DataFrame")

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by removing duplicate rows.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with duplicates removed.
        """
        if self.keep == "none":
            # Drop all duplicates (keep no occurrences)
            if self.subset is None:
                # Use all columns
                return X.filter(~X.is_duplicated())
            else:
                # Use subset columns
                duplicate_mask = X.select(self.subset).is_duplicated()
                return X.filter(~duplicate_mask)
        else:
            # Use Polars unique method with keep parameter
            keep_strategy = self.keep  # 'first' or 'last'
            if self.subset is None:
                return X.unique(maintain_order=True, keep=keep_strategy)
            else:
                return X.unique(subset=self.subset, maintain_order=True, keep=keep_strategy)
