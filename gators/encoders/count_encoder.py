from typing import Optional

import polars as pl

from ._base_encoder import _BaseEncoder


class CountEncoder(_BaseEncoder):
    """
    Encodes categorical values with their occurrence counts.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of categorical columns to encode. If None, all string, boolean, and categorical columns are selected.
    min_count : Union[int, float], default=1
        Minimum count threshold for encoding categories. If >= 1, treated as absolute count; if < 1, treated as frequency.
    inplace : bool, default=True
        If True, replace original columns with encoded values.
        If False, create new columns with suffix '__count_enc'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after encoding.
        Ignored when inplace=True.

    Examples
    --------
    Initialize and use `CountEncoder`.

    Example with `drop_columns=True` and `columns=None`:

    >>> import polars as pl
    >>> from gators.encoders import CountEncoder
    >>> X = pl.DataFrame({
    ...     "category": ["A", "B", "A", "C", "C", "A", "B"],
    ...     "value": [1, 2, 3, 4, 5, 6, 7],
    ...     "other": ["foo", "bar", "baz", "qux", "quux", "corge", "grault"]
    ... })
    >>> encoder = CountEncoder(min_count=1, inplace=False)
    >>> _ = encoder.fit(X)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (7, 3)
    ┌───────┬─────────────────────┬──────────────────┐
    │ value ┆ category__count_enc ┆ other__count_enc │
    │ ---   ┆ ---                 ┆ ---              │
    │ i64   ┆ f64                 ┆ f64              │
    ╞═══════╪═════════════════════╪══════════════════╡
    │ 1     ┆ 3.0                 ┆ 1.0              │
    │ 2     ┆ 2.0                 ┆ 1.0              │
    │ 3     ┆ 3.0                 ┆ 1.0              │
    │ 4     ┆ 2.0                 ┆ 1.0              │
    │ 5     ┆ 2.0                 ┆ 1.0              │
    │ 6     ┆ 3.0                 ┆ 1.0              │
    │ 7     ┆ 2.0                 ┆ 1.0              │
    └───────┴─────────────────────┴──────────────────┘

    Example with `drop_columns=True` and `columns` as a subset:

    >>> X = pl.DataFrame({
    ...     "category": ["A", "B", "A", "C", "C", "A", "B"],
    ...     "value": [1, 2, 3, 4, 5, 6, 7],
    ...     "other": ["foo", "bar", "baz", "qux", "quux", "corge", "grault"]
    ... })
    >>> encoder = CountEncoder(subset=["category"], min_count=1, drop_columns=True, inplace=False)
    >>> _ = encoder.fit(X)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (7, 3)
    ┌───────┬────────┬────────────────────────┐
    │ value ┆ other  ┆ category__encode_count │
    │ ---   ┆ ---    ┆ ---                    │
    │ i64   ┆ str    ┆ f64                    │
    ╞═══════╪════════╪════════════════════════╡
    │ 1     ┆ foo    ┆ 3.0                    │
    │ 2     ┆ bar    ┆ 2.0                    │
    │ 3     ┆ baz    ┆ 3.0                    │
    │ 4     ┆ qux    ┆ 2.0                    │
    │ 5     ┆ quux   ┆ 2.0                    │
    │ 6     ┆ corge  ┆ 3.0                    │
    │ 7     ┆ grault ┆ 2.0                    │
    └───────┴────────┴────────────────────────┘

    Example with `drop_columns=False` and `columns=None`:

    >>> import polars as pl
    >>> from gators.encoders import CountEncoder
    >>> X = pl.DataFrame({
    ...     "category": ["A", "B", "A", "C", "C", "A", "B"],
    ...     "value": [1, 2, 3, 4, 5, 6, 7],
    ...     "other": ["foo", "bar", "baz", "qux", "quux", "corge", "grault"]
    ... })
    >>> encoder = CountEncoder(min_count=1, drop_columns=False, inplace=False)
    >>> _ = encoder.fit(X)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (7, 5)
    ┌──────────┬───────┬────────┬────────────────────────┬─────────────────────┐
    │ category ┆ value ┆ other  ┆ category__encode_count ┆ other__encode_count │
    │ ---      ┆ ---   ┆ ---    ┆ ---                    ┆ ---                 │
    │ str      ┆ i64   ┆ str    ┆ f64                    ┆ f64                 │
    ╞══════════╪═══════╪════════╪════════════════════════╪═════════════════════╡
    │ A        ┆ 1     ┆ foo    ┆ 3.0                    ┆ 1.0                 │
    │ B        ┆ 2     ┆ bar    ┆ 2.0                    ┆ 1.0                 │
    │ A        ┆ 3     ┆ baz    ┆ 3.0                    ┆ 1.0                 │
    │ C        ┆ 4     ┆ qux    ┆ 2.0                    ┆ 1.0                 │
    │ C        ┆ 5     ┆ quux   ┆ 2.0                    ┆ 1.0                 │
    │ A        ┆ 6     ┆ corge  ┆ 3.0                    ┆ 1.0                 │
    │ B        ┆ 7     ┆ grault ┆ 2.0                    ┆ 1.0                 │
    └──────────┴───────┴────────┴────────────────────────┴─────────────────────┘
    """

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CountEncoder":
        """Fit the transformer by computing count statistics for each category.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        CountEncoder
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.String, pl.Boolean, pl.Categorical]
            ]
        self.mapping_ = {
            col: dict(zip(d[col].to_list(), d["count"].to_list()))
            for col in self.subset
            if not (d := X[col].value_counts()).is_empty()
        }
        min_threshold_count = (
            self.min_count if self.min_count >= 1 else self.min_count * len(X)
        )
        self.mapping_ = {
            col: {k: v for k, v in counts.items() if v >= min_threshold_count}
            for col, counts in self.mapping_.items()
        }
        self.column_mapping_ = {col: f"{col}__count_enc" for col in self.subset}

        return self
