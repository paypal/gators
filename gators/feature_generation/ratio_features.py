from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel, field_validator
from sklearn.base import BaseEstimator, TransformerMixin


class RatioFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Generates ratio features by dividing numerator columns by denominator columns.

    This transformer creates ratio features in a 1-to-1 pairing between numerator and denominator
    columns. Division by zero is handled by replacing the result with null values.

    Parameters
    ----------
    numerator_columns : List[str]
        List of column names to use as numerators.
    denominator_columns : List[str]
        List of column names to use as denominators. Must have the same length as numerator_columns.
    new_column_names : Optional[List[str]], optional
        List of custom names for the ratio features. If None, names will be automatically
        generated as '{numerator}__div__{denominator}', by default None.
    drop_columns : bool, optional
        Whether to drop the original numerator and denominator columns after creating ratios,
        by default False.

    Examples
    --------
    >>> from gators.feature_generation import RatioFeatures
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'revenue': [100, 200, 300, 400],
    ...     'cost': [80, 100, 150, 0],
    ...     'clicks': [1000, 2000, 3000, 4000],
    ...     'impressions': [10000, 20000, 30000, 40000]
    ... })

    **Example 1: Basic ratio features**

    >>> transformer = RatioFeatures(
    ...     numerator_columns=['revenue', 'clicks'],
    ...     denominator_columns=['cost', 'impressions']
    ... )
    >>> transformer.fit(X)
    RatioFeatures(numerator_columns=['revenue', 'clicks'], denominator_columns=['cost', 'impressions'])
    >>> result = transformer.transform(X)
    >>> result
    shape: (4, 6)
    ┌─────────┬──────┬────────┬─────────────┬────────────────────┬─────────────────────────┐
    │ revenue │ cost │ clicks │ impressions │ revenue__div__cost │ clicks__div__impressions│
    │ i64     │ i64  │ i64    │ i64         │ f64                │ f64                     │
    ├─────────┼──────┼────────┼─────────────┼────────────────────┼─────────────────────────┤
    │ 100     │ 80   │ 1000   │ 10000       │ 1.25               │ 0.1                     │
    │ 200     │ 100  │ 2000   │ 20000       │ 2.0                │ 0.1                     │
    │ 300     │ 150  │ 3000   │ 30000       │ 2.0                │ 0.1                     │
    │ 400     │ 0    │ 4000   │ 40000       │ null               │ 0.1                     │
    └─────────┴──────┴────────┴─────────────┴────────────────────┴─────────────────────────┘

    **Example 2: Custom column names**

    >>> transformer = RatioFeatures(
    ...     numerator_columns=['revenue'],
    ...     denominator_columns=['cost'],
    ...     new_column_names=['profit_margin']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 5)
    ┌─────────┬──────┬────────┬─────────────┬───────────────┐
    │ revenue │ cost │ clicks │ impressions │ profit_margin │
    │ i64     │ i64  │ i64    │ i64         │ f64           │
    ├─────────┼──────┼────────┼─────────────┼───────────────┤
    │ 100     │ 80   │ 1000   │ 10000       │ 1.25          │
    │ 200     │ 100  │ 2000   │ 20000       │ 2.0           │
    │ 300     │ 150  │ 3000   │ 30000       │ 2.0           │
    │ 400     │ 0    │ 4000   │ 40000       │ null          │
    └─────────┴──────┴────────┴─────────────┴───────────────┘

    **Example 3: With drop_columns=True**

    >>> transformer = RatioFeatures(
    ...     numerator_columns=['revenue'],
    ...     denominator_columns=['cost'],
    ...     drop_columns=True
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 3)
    ┌────────┬─────────────┬────────────────────┐
    │ clicks │ impressions │ revenue__div__cost │
    │ i64    │ i64         │ f64                │
    ├────────┼─────────────┼────────────────────┤
    │ 1000   │ 10000       │ 1.25               │
    │ 2000   │ 20000       │ 2.0                │
    │ 3000   │ 30000       │ 2.0                │
    │ 4000   │ 40000       │ null               │
    └────────┴─────────────┴────────────────────┘

    **Example 4: Handling null values**

    >>> X_with_nulls = pl.DataFrame({
    ...     'A': [10, None, 30, 40],
    ...     'B': [2, 5, None, 0]
    ... })
    >>> transformer = RatioFeatures(
    ...     numerator_columns=['A'],
    ...     denominator_columns=['B']
    ... )
    >>> result = transformer.fit_transform(X_with_nulls)
    >>> result
    shape: (4, 3)
    ┌──────┬──────┬──────────────┐
    │ A    │ B    │ A__div__B    │
    │ i64  │ i64  │ f64          │
    ├──────┼──────┼──────────────┤
    │ 10   │ 2    │ 5.0          │
    │ null │ 5    │ null         │
    │ 30   │ null │ null         │
    │ 40   │ 0    │ null         │
    └──────┴──────┴──────────────┘
    """

    numerator_columns: List[str]
    denominator_columns: List[str]
    new_column_names: Optional[List[str]] = None
    drop_columns: bool = False
    _column_mapping: Dict[str, str] = {}

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

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "RatioFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
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

            # Create ratio with division by zero handling
            ratio_expr = (
                pl.when(pl.col(denom_col) == 0)
                .then(None)
                .otherwise(pl.col(num_col) / pl.col(denom_col))
                .alias(new_col_name)
            )

            new_columns.append(ratio_expr)

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = set(self.numerator_columns + self.denominator_columns)
            X = X.drop(list(columns_to_drop))

        return X
