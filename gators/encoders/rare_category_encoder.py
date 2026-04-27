from typing import Dict, List, Optional, Union

import polars as pl
from pydantic import Field, PositiveFloat, PositiveInt

from ..transformer._base_transformer import _BaseTransformer


class RareCategoryEncoder(_BaseTransformer):
    """
    Encodes rare categories.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of categorical columns to encode. If None, all string, boolean, and categorical columns are selected.
    default : str, default="RARE"
        Value to replace rare categories with.
    min_count : Union[PositiveInt, PositiveFloat], default=2
        Minimum count threshold for categories. Categories below this threshold are replaced with `default`. If >= 1, treated as absolute count; if < 1, treated as frequency.
    inplace : bool, default=True
        If True, replace original columns with encoded values.
        If False, create new columns with suffix '__encode_rare'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after encoding.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.encoders import RareCategoryEncoder

    >>> # Sample data
    >>> X =pl.DataFrame({
    ...     'A': ['cat', 'dog', 'cat', 'dog', 'cat'],
    ...     'B': ['x', 'x', 'y', 'y', 'x'],
    ...     'target': [1, 0, 1, 1, 0]
    ... })

    >>> encoder = RareCategoryEncoder(inplace=False)
    >>> encoder.fit(X)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 2)
    ┌───────────────────┬───────────────────┐
    │ A__encode_rare    │ B__encode_rare    │
    │ ---               │ ---               │
    │ str               │ str               │
    ├───────────────────┼───────────────────┤
    │ cat               │ x                 │
    │ dog               │ x                 │
    │ cat               │ RARE              │
    │ dog               │ RARE              │
    │ cat               │ x                 │
    └───────────────────┴───────────────────┘

    >>> encoder = RareCategoryEncoder(drop_columns=False, inplace=False)
    >>> encoder.fit(X)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 5)
    ┌─────┬─────┬────────┬───────────────────┬───────────────────┐
    │ A   │ B   │ target │ A__encode_rare    │ B__encode_rare    │
    │ --- │ --- │ ---    │ ---               │ ---               │
    │ str │ str │ i64    │ str               │ str               │
    ├─────┼─────┼────────┼───────────────────┼───────────────────┤
    │ cat │ x   │ 1      │ cat               │ x                 │
    │ dog │ x   │ 0      │ dog               │ x                 │
    │ cat │ y   │ 1      │ cat               │ RARE              │
    │ dog │ y   │ 1      │ dog               │ RARE              │
    │ cat │ x   │ 0      │ cat               │ x                 │
    └─────┴─────┴────────┴───────────────────┴───────────────────┘

    >>> encoder = RareCategoryEncoder(subset=['A'], inplace=False)
    >>> encoder.fit(X)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌─────┬─────┬────────┬───────────────────┐
    │ A   │ B   │ target │ A__encode_rare    │
    │ --- │ --- │ ---    │ ---               │
    │ str │ str │ i64    │ str               │
    ├─────┼─────┼────────┼───────────────────┤
    │ cat │ x   │ 1      │ cat               │
    │ dog │ x   │ 0      │ dog               │
    │ cat │ y   │ 1      │ cat               │
    │ dog │ y   │ 1      │ dog               │
    │ cat │ x   │ 0      │ cat               │
    └─────┴─────┴────────┴───────────────────┘
    """

    subset: Optional[List[str]] = None
    mapping_: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    default: str = "RARE"
    column_mapping_: Dict[str, str] = Field(default_factory=dict)
    min_count: Union[PositiveInt, PositiveFloat] = 2
    drop_columns: bool = True
    inplace: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None):
        """Fit the transformer by identifying rare categories.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        RareCategoryEncoder
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.String, pl.Categorical]
            ]

        self.mapping_ = {
            col: dict(zip(d[col].to_list(), d["count"].to_list()))
            for col in self.subset
            if not (d := X[col].value_counts()).is_empty()
        }

        min_threshold_count = self.min_count if self.min_count >= 1 else self.min_count * len(X)

        self.mapping_ = {
            col: {k: self.default for k, v in counts.items() if int(v) < min_threshold_count}
            for col, counts in self.mapping_.items()
        }

        if not self.inplace:
            self.column_mapping_ = {col: f"{col}__encode_rare" for col in self.subset}

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by replacing rare categories with the default value.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with rare categories replaced.
        """
        if self.inplace:
            transformations = [
                pl.col(col).replace(mapping) for col, mapping in self.mapping_.items()
            ]
            return X.with_columns(transformations)

        transformations = [
            pl.col(col).replace(mapping).alias(self.column_mapping_[col])
            for col, mapping in self.mapping_.items()
        ]
        X = X.with_columns(transformations)
        if self.drop_columns and self.subset:
            X = X.drop(self.subset)
        return X
