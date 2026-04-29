from typing import Dict, List, Optional, Union

import polars as pl
from pydantic import PositiveFloat, PositiveInt

from ..transformer._base_transformer import _BaseTransformer


class OneHotEncoder(_BaseTransformer):
    """
    One-hot encodes categorical values.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of string columns to encode. If None, all string columns are selected.
    categories : Optional[Dict[str, List[str]]], default=None
        Pre-defined categories for each column. If None, categories are inferred from data during fit.
    min_count : Union[PositiveInt, PositiveFloat], default=1
        Minimum count threshold for encoding categories. If >= 1, treated as absolute count; if < 1, treated as frequency.
    drop_columns : bool, default=True
        Whether to drop the original columns after encoding.

    Examples
    --------
    Basic usage:

    >>> from gators.encoders import OneHotEncoder
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     "A": ["foo", "bar", "foo", "bar", "baz"],
    ...     "B": ["one", "one", "two", "two", "one"],
    ... })
    >>> encoder = OneHotEncoder()
    >>> encoder.fit(X)
    OneHotEncoder(...)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 5)
    ┌───────┬───────┬───────┬───────┬───────┐
    │ A|foo │ A|bar │ A|baz │ B|one │ B|two │
    │ f64   │ f64   │ f64   │ f64   │ f64   │
    ╞═══════╪═══════╪═══════╪═══════╪═══════╡
    │ 1.0   │ 0.0   │ 0.0   │ 1.0   │ 0.0   │
    │ 0.0   │ 1.0   │ 0.0   │ 1.0   │ 0.0   │
    │ 1.0   │ 0.0   │ 0.0   │ 0.0   │ 1.0   │
    │ 0.0   │ 1.0   │ 0.0   │ 0.0   │ 1.0   │
    │ 0.0   │ 0.0   │ 1.0   │ 1.0   │ 0.0   │
    └───────┴───────┴───────┴───────┴───────┘

    Drop columns:

    >>> encoder = OneHotEncoder(drop_columns=True)
    >>> encoder.fit(X)
    OneHotEncoder(...)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 5)
    ┌────────┬────────┬────────┬────────┬────────┐
    │ A__foo │ A__bar │ A__baz │ B__one │ B__two │
    │ f64    │ f64    │ f64    │ f64    │ f64    │
    ╞════════╪════════╪════════╪════════╪════════╡
    │ 1.0    │ 0.0    │ 0.0    │ 1.0    │ 0.0    │
    │ 0.0    │ 1.0    │ 0.0    │ 1.0    │ 0.0    │
    │ 1.0    │ 0.0    │ 0.0    │ 0.0    │ 1.0    │
    │ 0.0    │ 1.0    │ 0.0    │ 0.0    │ 1.0    │
    │ 0.0    │ 0.0    │ 1.0    │ 1.0    │ 0.0    │
    └────────┴────────┴────────┴────────┴────────┘

    Subset of columns:

    >>> encoder = OneHotEncoder(subset=["A"])
    >>> encoder.fit(X)
    OneHotEncoder(...)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 3)
    ┌────────┬────────┬────────┐
    │ A__foo │ A__bar │ A__baz │
    │ f64    │ f64    │ f64    │
    ╞════════╪════════╪════════╡
    │ 1.0    │ 0.0    │ 0.0    │
    │ 0.0    │ 1.0    │ 0.0    │
    │ 1.0    │ 0.0    │ 0.0    │
    │ 0.0    │ 1.0    │ 0.0    │
    │ 0.0    │ 0.0    │ 1.0    │
    └────────┴────────┴────────┘

    """

    subset: Optional[List[str]] = None
    categories: Optional[Dict[str, List[str]]] = None
    min_count: Union[PositiveInt, PositiveFloat] = 1
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "OneHotEncoder":
        """Fit the transformer by identifying categories for one-hot encoding.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with string columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        OneHotEncoder
            The fitted transformer instance.
        """
        if self.categories:
            self.subset = list(set(self.categories.keys()))
            return self

        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype in [pl.String]
            ]
        
        X_filled = X.with_columns([pl.col(col).fill_null("MISSING_") for col in self.subset])
        
        self.categories = {}
        n = len(X)
        threshold = self.min_count if self.min_count >= 1 else self.min_count * n
        
        for col in self.subset:
            counts = X_filled[col].value_counts(sort=True)
            valid_categories = counts.filter(pl.col("count") >= threshold)
            self.categories[col] = valid_categories[col].to_list()
        
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying one-hot encoding to categorical columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with string columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with one-hot encoded columns (one binary column per category).
        """
        if self.categories is None:
            return X
        
        # Use native Polars to_dummies - single efficient call
        cols_to_encode = list(self.categories.keys())
        dummies = X.select(cols_to_encode).to_dummies(separator="__")
        
        # Build expected columns list (pre-computed for efficiency)
        expected_cols = [
            f"{col}__{cat}" for col, cat_list in self.categories.items() for cat in cat_list
        ]
        expected_cols_set = set(expected_cols)
        
        # Identify existing and missing columns efficiently
        existing_cols = [c for c in expected_cols if c in dummies.columns]
        missing_cols = expected_cols_set - set(dummies.columns)
        
        # Select existing columns and add missing columns in single operation
        if missing_cols:
            # Batch: select existing + create missing columns together
            dummies = dummies.select(existing_cols).with_columns(
                [pl.lit(0.0).alias(col_name) for col_name in sorted(missing_cols)]
            )
        else:
            # Just select existing columns
            dummies = dummies.select(existing_cols)
        
        # Cast all to Float64 in single operation
        dummies = dummies.select(pl.all().cast(pl.Float64))
        
        # Concatenate with original dataframe
        X = pl.concat([X, dummies], how="horizontal")

        # Drop original columns if requested
        if self.drop_columns and self.subset:
            X = X.drop(self.subset)
        
        return X
