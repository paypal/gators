# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

import polars as pl
from pydantic import Field, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class DropNearConstantColumns(_BaseTransformer):
    """
    Removes columns where a single value dominates almost all rows.

    A column is *near-constant* when its most frequent value's share is at
    least ``max_ratio``. ``max_ratio=1.0`` drops exactly the same columns as
    :class:`DropConstantColumns` (for both ``include_na`` settings); lower
    values progressively catch columns where one value merely dominates (e.g.
    ``max_ratio=0.99`` drops any column whose top value covers 99%+ of rows).

    Parameters
    ----------
    max_ratio : float, default=0.99
        Maximum allowed share of rows the most frequent value may occupy
        before the column is dropped. Must be in the range ``[0.0, 1.0]``.
    subset : list[str], default=None
        Columns to evaluate. When ``None`` all columns are evaluated.
    include_na : bool, default=True
        Whether null values count as their own category when determining the
        most frequent value. When ``True`` the share is computed over all rows
        (an all-null column counts as 100% one value: null). When ``False``
        nulls are excluded from BOTH the numerator and the denominator: the
        share is the dominant value's count among non-null rows only. A
        column with no non-null values at all has no value to be dominant, so
        it is treated as constant (dropped) regardless of ``max_ratio``,
        matching :class:`DropConstantColumns`.

    Examples
    --------
    **Example 1: Drop near-constant numeric column**

    >>> from gators.data_cleaning import DropNearConstantColumns
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'id': list(range(100)),
    ...     'near_const': [42] * 99 + [0],
    ...     'varying': list(range(100)),
    ... })
    >>> remover = DropNearConstantColumns(max_ratio=0.98)
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['id', 'varying']

    **Example 2: Drop near-constant categorical column**

    >>> X = pl.DataFrame({
    ...     'country': ['USA'] * 9 + ['UK'],
    ...     'city': ['NYC', 'LA', 'Chicago', 'Boston', 'Seattle',
    ...              'Denver', 'Miami', 'Austin', 'Portland', 'Dallas'],
    ... })
    >>> remover = DropNearConstantColumns(max_ratio=0.8)
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['city']

    **Example 3: Handle NaN values (with include_na=True)**

    >>> X = pl.DataFrame({
    ...     'mostly_null': [None] * 9 + [1],
    ...     'varying': list(range(10)),
    ... })
    >>> remover = DropNearConstantColumns(max_ratio=0.8, include_na=True)
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['varying']

    **Example 4: Handle NaN values (with include_na=False)**

    >>> X = pl.DataFrame({
    ...     'same_non_null': [1] * 9 + [None],
    ...     'varying': list(range(10)),
    ... })
    >>> remover = DropNearConstantColumns(max_ratio=0.8, include_na=False)
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['varying']

    **Example 5: Subset of columns**

    >>> X = pl.DataFrame({
    ...     'col1': [1] * 9 + [2],
    ...     'col2': [5] * 9 + [6],
    ...     'col3': list(range(10)),
    ... })
    >>> remover = DropNearConstantColumns(max_ratio=0.8, subset=['col1', 'col2'])
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['col3']
    """

    max_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.99
    subset: list[str] | None = None
    include_na: bool = True
    _to_drop: list[str] = PrivateAttr(default_factory=list)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "DropNearConstantColumns":
        """Fit the transformer by identifying near-constant columns.

        A column is near-constant when its most frequent value's share of
        rows is at least ``max_ratio``.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for API compatibility.

        Returns
        -------
        DropNearConstantColumns
            Fitted transformer instance.
        """
        columns_to_check = self.subset if self.subset is not None else X.columns
        n_rows = len(X)
        if n_rows == 0:
            self._to_drop = []
            return self

        if self.include_na:
            mode_counts = X.select(
                [pl.col(c).value_counts().struct.field("count").max().alias(c) for c in columns_to_check]
            ).row(0)
            self._to_drop = [
                col
                for col, mode_count in zip(columns_to_check, mode_counts, strict=False)
                if (mode_count / n_rows) >= self.max_ratio
            ]
        else:
            # Denominator is the NON-null count, so nulls are fully excluded from the
            # population being judged (matching DropConstantColumns' include_na=False
            # behaviour) instead of diluting the share by rows that carry no signal.
            mode_counts = X.select(
                [
                    pl.col(c).drop_nulls().value_counts().struct.field("count").max().fill_null(0).alias(c)
                    for c in columns_to_check
                ]
            ).row(0)
            non_null_counts = X.select([pl.col(c).count().alias(c) for c in columns_to_check]).row(0)
            self._to_drop = [
                col
                for col, mode_count, non_null_count in zip(
                    columns_to_check, mode_counts, non_null_counts, strict=False
                )
                # No non-null values at all -> no signal, treated as constant (dropped).
                if non_null_count == 0 or (mode_count / non_null_count) >= self.max_ratio
            ]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by removing near-constant columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with near-constant columns removed.
        """
        if self._to_drop:
            return X.drop(self._to_drop)
        return X
