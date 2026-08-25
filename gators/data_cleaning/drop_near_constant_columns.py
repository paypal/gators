import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class DropNearConstantColumns(_BaseTransformer):
    """
    Removes columns where fewer than a given fraction of rows have a distinct value.

    A column is considered *near-constant* when the number of unique values is
    not greater than ``threshold * n_rows``.  This generalises
    :class:`DropConstantColumns` (threshold=0) to low-variance categorical and
    numeric columns alike without requiring variance computation.

    Parameters
    ----------
    threshold : float, default=0.01
        Minimum fraction of rows that must be distinct for a column to be kept.
        A column is dropped when ``n_unique <= threshold * n_rows``.
        Must be in the range ``[0.0, 1.0)``.
    subset : list[str], default=None
        Columns to evaluate. When ``None`` all columns are evaluated.
    include_na : bool, default=True
        Whether null values count as a distinct value. When ``True`` a column
        whose only values are ``null`` contributes one unique value.  When
        ``False`` nulls are excluded before counting unique values.

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
    >>> remover = DropNearConstantColumns(threshold=0.02)
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['id', 'varying']

    **Example 2: Drop near-constant categorical column**

    >>> X = pl.DataFrame({
    ...     'country': ['USA'] * 9 + ['UK'],
    ...     'city': ['NYC', 'LA', 'Chicago', 'Boston', 'Seattle',
    ...              'Denver', 'Miami', 'Austin', 'Portland', 'Dallas'],
    ... })
    >>> remover = DropNearConstantColumns(threshold=0.15)
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['city']

    **Example 3: Handle NaN values (with include_na=True)**

    >>> X = pl.DataFrame({
    ...     'mostly_null': [None] * 9 + [1],
    ...     'varying': list(range(10)),
    ... })
    >>> remover = DropNearConstantColumns(threshold=0.15, include_na=True)
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['varying']

    **Example 4: Handle NaN values (with include_na=False)**

    >>> X = pl.DataFrame({
    ...     'same_non_null': [1] * 9 + [None],
    ...     'varying': list(range(10)),
    ... })
    >>> remover = DropNearConstantColumns(threshold=0.15, include_na=False)
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['varying']

    **Example 5: Subset of columns**

    >>> X = pl.DataFrame({
    ...     'col1': [1] * 9 + [2],
    ...     'col2': [5] * 9 + [6],
    ...     'col3': list(range(10)),
    ... })
    >>> remover = DropNearConstantColumns(threshold=0.15, subset=['col1', 'col2'])
    >>> result = remover.fit_transform(X)
    >>> result.columns
    ['col3']
    """

    threshold: float = 0.01
    subset: list[str] | None = None
    include_na: bool = True
    _to_drop: list[str] = PrivateAttr(default_factory=list)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "DropNearConstantColumns":
        """Fit the transformer by identifying near-constant columns.

        A column is near-constant when its number of unique values does not
        exceed ``threshold * len(X)``.

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
        max_unique = self.threshold * len(X)

        if self.include_na:
            n_unique_values = X.select(
                [pl.col(c).n_unique().alias(c) for c in columns_to_check]
            ).row(0)
        else:
            n_unique_values = X.select(
                [pl.col(c).drop_nulls().n_unique().alias(c) for c in columns_to_check]
            ).row(0)

        self._to_drop = [
            col
            for col, n_unique in zip(columns_to_check, n_unique_values, strict=False)
            if n_unique <= max_unique
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
