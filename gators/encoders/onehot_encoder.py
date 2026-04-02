from typing import Dict, List, Optional, Union

import polars as pl
from pydantic import BaseModel, PositiveFloat, PositiveInt
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoder(BaseModel, BaseEstimator, TransformerMixin):
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
                col
                for col, dtype in dict(zip(X.columns, X.dtypes)).items()
                if dtype in [pl.String]
            ]
        X = X.with_columns([pl.col(col).fill_null("MISSING_") for col in self.subset])
        self.categories = {}
        n = len(X)
        for col in self.subset:
            counts = X[col].value_counts().to_pandas()
            if self.min_count < 1:
                counts["count"] /= n
            counts = counts[counts["count"] >= self.min_count]

            self.categories[col] = counts[col].to_list()
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
        # Use native Polars to_dummies - single call for all columns
        cols_to_encode = list(self.categories.keys())
        dummies = X.select(cols_to_encode).to_dummies(separator="__")
        dummies = dummies.select(pl.all().cast(pl.Float64))

        # Build list of expected columns from fit
        expected_cols = [
            f"{col}__{cat}"
            for col, cat_list in self.categories.items()
            for cat in cat_list
        ]

        # Keep only columns that exist and were learned during fit
        existing_cols = [c for c in expected_cols if c in dummies.columns]
        dummies = dummies.select(existing_cols)

        # Add missing categories as zero columns
        missing_cols = set(expected_cols) - set(existing_cols)
        if missing_cols:
            dummies = dummies.with_columns(
                [pl.lit(0.0).alias(col_name) for col_name in sorted(missing_cols)]
            )

        # Concatenate with original dataframe
        X = pl.concat([X, dummies], how="horizontal")

        if self.drop_columns and self.subset:
            X = X.drop(self.subset)
        return X
