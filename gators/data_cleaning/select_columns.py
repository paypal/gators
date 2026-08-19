import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class SelectColumns(_BaseTransformer):
    """
    Retains only the specified columns from a DataFrame, dropping all others.

    Parameters
    ----------
    subset : list[str]
        List of column names to keep.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.data_cleaning import SelectColumns

    >>> X = pl.DataFrame({
    ...     "col1": [1, 2, 3],
    ...     "col2": ["A", "B", "C"],
    ...     "col3": [True, False, True],
    ... })
    >>> transformer = SelectColumns(subset=["col1", "col3"])
    >>> transformer.fit(X)
    SelectColumns(subset=['col1', 'col3'])
    >>> print(transformer.transform(X))
    shape: (3, 2)
    ┌──────┬───────┐
    │ col1 ┆ col3  │
    │ ---  ┆ ---   │
    │ i64  ┆ bool  │
    ╞══════╪═══════╡
    │ 1    ┆ true  │
    │ 2    ┆ false │
    │ 3    ┆ true  │
    └──────┴───────┘
    """

    subset: list[str]

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "SelectColumns":
        """Fit the transformer (no-op for SelectColumns).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        SelectColumns
            The fitted transformer instance.
        """
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by retaining only the specified columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame containing only the columns in ``subset``.
        """
        return X.select(self.subset)
