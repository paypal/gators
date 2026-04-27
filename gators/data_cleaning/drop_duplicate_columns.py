from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class DropDuplicateColumns(_BaseTransformer):
    """
    Removes duplicate columns from the DataFrame.

    Identifies and removes columns that have identical values across all rows.
    This is useful for reducing dimensionality and removing redundant features
    that don't add predictive value.

    Parameters
    ----------
    keep : str, default='first'
        Strategy for keeping duplicate columns:

        - 'first': Keep first occurrence of duplicate columns
        - 'last': Keep last occurrence of duplicate columns

    Examples
    --------
    **Example 1: Remove duplicate columns (keep first)**

    >>> from gators.data_cleaning import DropDuplicateColumns
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [5, 6, 7, 8],
    ...     'C': [1, 2, 3, 4],  # Duplicate of A
    ...     'D': [9, 10, 11, 12],
    ...     'E': [5, 6, 7, 8]   # Duplicate of B
    ... })
    >>> remover = DropDuplicateColumns(keep='first')
    >>> remover.fit(X)
    >>> result = remover.transform(X)
    >>> print(result)
    shape: (4, 3)
    ┌─────┬─────┬──────┐
    │ A   ┆ B   ┆ D    │
    │ --- ┆ --- ┆ ---  │
    │ i64 ┆ i64 ┆ i64  │
    ├─────┼─────┼──────┤
    │ 1   ┆ 5   ┆ 9    │
    │ 2   ┆ 6   ┆ 10   │
    │ 3   ┆ 7   ┆ 11   │
    │ 4   ┆ 8   ┆ 12   │
    └─────┴─────┴──────┘

    **Example 2: Remove duplicate columns (keep last)**

    >>> X = pl.DataFrame({
    ...     'feature_1': [1.0, 2.0, 3.0],
    ...     'feature_2': [4.0, 5.0, 6.0],
    ...     'feature_3': [1.0, 2.0, 3.0],  # Duplicate of feature_1
    ...     'target': [0, 1, 0]
    ... })
    >>> remover = DropDuplicateColumns(keep='last')
    >>> remover.fit(X)
    >>> print(f"Columns to drop: {remover.columns_to_drop_}")
    Columns to drop: ['feature_1']
    >>> print(f"Column groups: {remover.column_groups_}")
    Column groups: {'feature_3': ['feature_1']}
    >>> result = remover.transform(X)
    >>> print(result)
    shape: (3, 3)
    ┌───────────┬───────────┬────────┐
    │ feature_2 | feature_3 ┆ target │
    │ ---       | ---       ┆ ---    │
    │ f64       | f64       ┆ i64    │
    ├───────────┼───────────┼────────┤
    │ 4.0       | 1.0       ┆ 0      │
    │ 5.0       | 2.0       ┆ 1      │
    │ 6.0       | 3.0       ┆ 0      │
    └───────────┴───────────┴────────┘

    **Example 3: Check duplicate groups**

    >>> X = pl.DataFrame({
    ...     'a': [1, 2, 3],
    ...     'b': [1, 2, 3],
    ...     'c': [1, 2, 3],
    ...     'd': [4, 5, 6]
    ... })
    >>> remover = DropDuplicateColumns()
    >>> remover.fit(X)
    >>> print(f"Kept column groups: {remover.column_groups_}")
    Kept column groups: {'a': ['b', 'c']}
    >>> result = remover.transform(X)
    >>> print(result.columns)
    ['a', 'd']

    **Example 4: No duplicates**

    >>> X = pl.DataFrame({
    ...     'x': [1, 2, 3],
    ...     'y': [4, 5, 6],
    ...     'z': [7, 8, 9]
    ... })
    >>> remover = DropDuplicateColumns()
    >>> remover.fit(X)
    >>> print(f"Columns to drop: {remover.columns_to_drop_}")
    Columns to drop: []
    >>> result = remover.transform(X)
    >>> print(result.shape)
    (3, 3)
    """

    keep: str = "first"
    columns_to_drop_: List[str] = []
    column_groups_: Dict[str, List[str]] = {}

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "DropDuplicateColumns":
        """Fit the transformer by identifying duplicate columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        DropDuplicateColumns
            Fitted transformer instance.

        Raises
        ------
        ValueError
            If keep parameter is not 'first' or 'last'.
        """
        if self.keep not in ["first", "last"]:
            raise ValueError(f"keep must be 'first' or 'last', got '{self.keep}'")

        available_columns = X.columns
        n_cols = len(available_columns)

        if n_cols <= 1:
            self.columns_to_drop_ = []
            self.column_groups_ = {}
            return self

        # Find duplicate columns by comparing their values
        duplicate_groups: Dict[str, List[str]] = {}
        processed = set()

        for i in range(n_cols):
            col_i = available_columns[i]

            if col_i in processed:
                continue

            duplicates = []

            for j in range(i + 1, n_cols):
                col_j = available_columns[j]

                if col_j in processed:
                    continue

                # Compare columns - check if they are identical
                if self._columns_equal(X[col_i], X[col_j]):
                    duplicates.append(col_j)
                    processed.add(col_j)

            if duplicates:
                duplicate_groups[col_i] = duplicates

        # Determine which columns to drop based on keep strategy
        self.column_groups_ = {}
        columns_to_drop = []

        for kept_col, dup_cols in duplicate_groups.items():
            if self.keep == "first":
                # Keep the first column (kept_col), drop the rest
                self.column_groups_[kept_col] = dup_cols
                columns_to_drop.extend(dup_cols)
            else:  # keep == "last"
                # Keep the last duplicate, drop the kept_col and earlier duplicates
                last_dup = dup_cols[-1]
                columns_to_drop.append(kept_col)
                columns_to_drop.extend(dup_cols[:-1])
                self.column_groups_[last_dup] = [kept_col] + dup_cols[:-1]

        self.columns_to_drop_ = columns_to_drop

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by removing duplicate columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with duplicate columns removed.
        """
        if not self.columns_to_drop_:
            return X

        return X.drop(self.columns_to_drop_)

    @staticmethod
    def _columns_equal(col1: pl.Series, col2: pl.Series) -> bool:
        """Check if two columns are equal (including null handling).

        Parameters
        ----------
        col1 : pl.Series
            First column to compare.
        col2 : pl.Series
            Second column to compare.

        Returns
        -------
        bool
            True if columns are identical, False otherwise.
        """
        if len(col1) != len(col2):
            return False

        # Handle different dtypes - try to compare
        if col1.dtype != col2.dtype:
            # Different types are not duplicates
            return False

        # Compare values including nulls
        # Two nulls in the same position should be considered equal
        return col1.equals(col2)
