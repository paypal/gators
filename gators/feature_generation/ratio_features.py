import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class RatioFeatures(_BaseTransformer):
    """
    Generates ratio features by dividing numerator columns by denominator columns
    with Laplace smoothing applied to the denominator.

    Each feature is computed as:

        ``numerator / (denominator + 1)``

    Adding ``1`` to the denominator (Laplace smoothing) prevents division-by-zero
    and avoids extreme values when the denominator is small or zero. This is
    particularly useful for count-based features such as event/trial ratios in
    fraud detection or click-through rates.

    Parameters
    ----------
    numerator_columns : list[str]
        List of column names to use as numerators.
    denominator_columns : list[str]
        List of column names to use as denominators. Must have the same length as
        ``numerator_columns``.
    new_column_names : list[str], optional
        List of custom names for the ratio features. If ``None``, names will be
        automatically generated as ``'{numerator}__div__{denominator}'``,
        by default ``None``.
    drop_columns : bool, optional
        Whether to drop the original numerator and denominator columns after
        creating ratios, by default ``False``.

    Examples
    --------
    >>> from gators.feature_generation import RatioFeatures
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'events': [10, 20, 0, 5],
    ...     'trials': [1, 3, 0, 9],
    ... })

    **Example 1: Basic ratio with Laplace smoothing**

    Zero trials are smoothed to 1, producing ``events / 1`` instead of null.

    >>> transformer = RatioFeatures(
    ...     numerator_columns=['events'],
    ...     denominator_columns=['trials']
    ... )
    >>> transformer.fit(X)
    RatioFeatures(numerator_columns=['events'], denominator_columns=['trials'])
    >>> result = transformer.transform(X)
    >>> result
    shape: (4, 3)
    ┌────────┬────────┬─────────────────────┐
    │ events │ trials │ events__div__trials │
    │ i64    │ i64    │ f64                 │
    ├────────┼────────┼─────────────────────┤
    │ 10     │ 1      │ 5.0                 │
    │ 20     │ 3      │ 5.0                 │
    │ 0      │ 0      │ 0.0                 │
    │ 5      │ 9      │ 0.5                 │
    └────────┴────────┴─────────────────────┘

    **Example 2: Multiple ratio features**

    >>> X2 = pl.DataFrame({
    ...     'hits_a': [9, 19, 0],
    ...     'hits_b': [4, 9, 9],
    ...     'views_a': [2, 4, 0],
    ...     'views_b': [1, 4, 9],
    ... })
    >>> transformer = RatioFeatures(
    ...     numerator_columns=['hits_a', 'hits_b'],
    ...     denominator_columns=['views_a', 'views_b']
    ... )
    >>> result = transformer.fit_transform(X2)
    >>> result
    shape: (3, 6)
    ┌────────┬────────┬─────────┬─────────┬──────────────────────┬──────────────────────┐
    │ hits_a │ hits_b │ views_a │ views_b │ hits_a__div__views_a │ hits_b__div__views_b │
    │ i64    │ i64    │ i64     │ i64     │ f64                  │ f64                  │
    ├────────┼────────┼─────────┼─────────┼──────────────────────┼──────────────────────┤
    │ 9      │ 4      │ 2       │ 1       │ 3.0                  │ 2.0                  │
    │ 19     │ 9      │ 4       │ 4       │ 3.8                  │ 1.8                  │
    │ 0      │ 9      │ 0       │ 9       │ 0.0                  │ 0.9                  │
    └────────┴────────┴─────────┴─────────┴──────────────────────┴──────────────────────┘

    **Example 3: Custom column names**

    >>> transformer = RatioFeatures(
    ...     numerator_columns=['events'],
    ...     denominator_columns=['trials'],
    ...     new_column_names=['smoothed_rate']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 3)
    ┌────────┬────────┬───────────────┐
    │ events │ trials │ smoothed_rate │
    │ i64    │ i64    │ f64           │
    ├────────┼────────┼───────────────┤
    │ 10     │ 1      │ 5.0           │
    │ 20     │ 3      │ 5.0           │
    │ 0      │ 0      │ 0.0           │
    │ 5      │ 9      │ 0.5           │
    └────────┴────────┴───────────────┘

    **Example 4: With drop_columns=True**

    >>> transformer = RatioFeatures(
    ...     numerator_columns=['events'],
    ...     denominator_columns=['trials'],
    ...     drop_columns=True
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 1)
    ┌─────────────────────┐
    │ events__div__trials │
    │ f64                 │
    ├─────────────────────┤
    │ 5.0                 │
    │ 5.0                 │
    │ 0.0                 │
    │ 0.5                 │
    └─────────────────────┘

    **Example 5: Null propagation**

    Input nulls still propagate through the ratio; only zero denominators are
    smoothed, not missing ones.

    >>> X_nulls = pl.DataFrame({
    ...     'A': [10, None, 30],
    ...     'B': [1, 3, None]
    ... })
    >>> transformer = RatioFeatures(
    ...     numerator_columns=['A'],
    ...     denominator_columns=['B']
    ... )
    >>> result = transformer.fit_transform(X_nulls)
    >>> result
    shape: (3, 3)
    ┌──────┬──────┬────────────┐
    │ A    │ B    │ A__div__B  │
    │ i64  │ i64  │ f64        │
    ├──────┼──────┼────────────┤
    │ 10   │ 1    │ 5.0        │
    │ null │ 3    │ null       │
    │ 30   │ null │ null       │
    └──────┴──────┴────────────┘
    """

    numerator_columns: list[str]
    denominator_columns: list[str]
    new_column_names: list[str] | None = None
    drop_columns: bool = False
    _column_mapping: dict[str, str] = {}

    @field_validator("denominator_columns", mode="after")
    def check_lengths_match(cls, denominator_columns, info):
        numerator_columns = info.data.get("numerator_columns", [])

        if len(numerator_columns) != len(denominator_columns):
            raise ValueError(
                f"Length of numerator_columns ({len(numerator_columns)}) "
                f"must match length of denominator_columns ({len(denominator_columns)})"
            )

        return denominator_columns

    @field_validator("new_column_names", mode="after")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            numerator_columns = info.data.get("numerator_columns", [])
            if len(new_column_names) != len(numerator_columns):
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match length of numerator_columns ({len(numerator_columns)})"
                )

        return new_column_names

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "RatioFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        RatioFeatures
            Fitted transformer instance.
        """
        default_names = [
            f"{num}__div__{denom}"
            for num, denom in zip(self.numerator_columns, self.denominator_columns)
        ]

        if self.new_column_names is None:
            self.new_column_names = default_names

        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating ratio features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with ratio features.
        """
        new_columns = []

        for num_col, denom_col in zip(self.numerator_columns, self.denominator_columns):
            default_name = f"{num_col}__div__{denom_col}"
            new_col_name = self._column_mapping[default_name]

            ratio_expr = (pl.col(num_col) / (pl.col(denom_col) + 1)).alias(new_col_name)

            new_columns.append(ratio_expr)

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = set(self.numerator_columns + self.denominator_columns)
            X = X.drop(list(columns_to_drop))

        return X
