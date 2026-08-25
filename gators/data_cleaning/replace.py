import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class Replace(_BaseTransformer):
    """
    Replaces values in specified columns.

    Parameters
    ----------
    to_replace : dict[str, dict[str, any]]
        Nested dictionary specifying replacement mappings. Outer keys are column names,
        inner dictionaries map old values to new values.
    inplace : bool, default=True
        If True, replace values in the original columns.
        If False, create new columns with suffix '__replace'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after replacement.
        Ignored when inplace=True.

    Examples
    --------
    Initializing and using `Replace` transformer.

    Example with `drop_columns=True` and `columns=None`:

    >>> X = pl.DataFrame({
    ...     "col1": ["a", "a", "b", "c"],
    ...     "col2": ["x", "x", "x", "y"],
    ...     "col3": [1, 2, 3, 4]
    ... })
    >>> replace_map = {
    ...     "col1": {"a": "alpha", "b": "bravo"},
    ...     "col2": {"x": "x-ray", "y": "yankee"}
    ... }
    >>> transformer = Replace(to_replace=replace_map, drop_columns=True)
    >>> transformer.fit(X)
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌───────────────┬───────────────┐
    │ col1__replace │ col2__replace │
    │ str           │   str         │
    ├───────────────┬───────────────┤
    │ alpha         │  x-ray        │
    │ alpha         │ x-ray         │
    │ bravo         │ x-ray         │
    │ charlie       │ yankee        │
    └───────────────┴───────────────┘

    Example with `drop_columns=True` and `columns` as a subset:

    >>> X = pl.DataFrame({
    ...     "col1": ["a", "a", "b", "c"],
    ...     "col2": ["x", "x", "x", "y"],
    ...     "col3": [1, 2, 3, 4]
    ... })
    >>> replace_map = {
    ...     "col1": {"a": "alpha", "b": "bravo"}
    ... }
    >>> transformer = Replace(to_replace=replace_map, drop_columns=True)
    >>> transformer.fit(X)
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 3)
    ┌───────────────────┬─────────────────────┬────────────────────┐
    │           col1    │           col2      │          col3      │
    │ str               │          str        │           i64      │
    ├───────────────────┬─────────────────────┬────────────────────┤
    │ alpha             │         x           │           1        │
    │ alpha             │         x           │           2        │
    │ bravo             │         x           │           3        │
    │ charlie           │         y           │           4        │
    └───────────────────┴─────────────────────┴────────────────────┘

    Example with `drop_columns=False` and `columns=None`:

    >>> X = pl.DataFrame({
    ...     "col1": ["a", "a", "b", "c"],
    ...     "col2": ["x", "x", "x", "y"],
    ...     "col3": [1, 2, 3, 4]
    ... })
    >>> replace_map = {
    ...     "col1": {"a": "alpha", "b": "bravo"},
    ...     "col2": {"x": "x-ray", "y": "yankee"}
    ... }
    >>> transformer = Replace(to_replace=replace_map, drop_columns=False)
    >>> transformer.fit(X)
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 4)
    ┌─────────────────┬──────────────────────┬─────────────────────┐─────────────────────────┐
    │         col1    │           col2       │           col3      │           col1__replace │
    │        str      │          str         │           i64       │           str           │
    │─────────────────┬──────────────────────┬─────────────────────┬─────────────────────────┤
    │            alpha│                x-ray │           1         │          alpha          │
    │            alpha│                x-ray │           2         │          alpha          │
    │            bravo│                x-ray │           3         │          bravo          │
    │        charlie  │               yankee │           4         │          charlie        │
    └─────────────────┴──────────────────────┴─────────────────────┴─────────────────────────┘
    """

    to_replace: dict[str, dict[str, str]]
    inplace: bool = True
    drop_columns: bool = True
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _columns: list[str] = PrivateAttr(default_factory=list)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "Replace":
        """Fit the transformer.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Name of the target column (if needed).

        Returns
        -------
        Self
            The fitted transformer instance.
        """
        available_columns = X.columns
        self._columns = [col for col in list(self.to_replace.keys()) if col in available_columns]
        if not self.inplace:
            self._column_mapping = {col: [f"{col}__replace"] for col in self._columns}
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
        if self.inplace:
            transformations = []
            for col in self._columns:
                expr = pl.col(col)
                for old_val, new_val in self.to_replace[col].items():
                    expr = expr.str.replace_all(old_val, new_val)
                transformations.append(expr)
            return X.with_columns(transformations)

        transformations = []
        for col in self._columns:
            expr = pl.col(col)
            for old_val, new_val in self.to_replace[col].items():
                expr = expr.str.replace_all(old_val, new_val)
            transformations.append(expr.alias(self._column_mapping[col][0]))
        X = X.with_columns(transformations)
        if self.drop_columns:
            return X.drop(self._columns)
        return X
