# SPDX-License-Identifier: Apache-2.0
import re

import polars as pl
from pydantic import PositiveFloat, PositiveInt, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer

_norm_col = re.compile(r'_{3,}')


class OneHotEncoder(_BaseTransformer):
    _CAT_DTYPES = {pl.String, pl.Categorical, pl.Enum}

    """
    One-hot encodes categorical values.

    Parameters
    ----------
    subset : list[str], default=None
        List of string columns to encode. If None, all string columns are selected.
    categories : dict[str, list[str]], default=None
        Pre-defined categories for each column. If None, categories are inferred from data during fit.
    min_count : PositiveInt | PositiveFloat, default=1
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

    subset: list[str] | None = None
    column_categories: dict[str, list[str]] | None = None
    min_count: PositiveInt | PositiveFloat = 1
    drop_columns: bool = True
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)


    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "OneHotEncoder":
        """Fit the transformer by identifying categories for one-hot encoding.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with string columns.
        y : pl.Series, default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        OneHotEncoder
            The fitted transformer instance.
        """
        if self.column_categories:
            # dict preserves insertion order; a set here would make column order (and
            # therefore output column order) nondeterministic across runs.
            self.subset = list(self.column_categories.keys())
            self._column_mapping = {
                col: [_norm_col.sub("__", f"{col}__{cat}") for cat in cats]
                for col, cats in self.column_categories.items()
            }
            self._set_output_dtypes()
            return self

        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype.base_type() in self._CAT_DTYPES
            ]

        X_filled = X.with_columns([pl.col(col).fill_null("MISSING__") for col in self.subset])

        self.column_categories = {}
        n = len(X)
        threshold = self.min_count if self.min_count >= 1 else self.min_count * n

        for col in self.subset:
            counts = X_filled[col].value_counts(sort=True)
            valid_categories = counts.filter(pl.col("count") >= threshold)
            self.column_categories[col] = valid_categories[col].to_list()

        self._column_mapping = {
            col: [_norm_col.sub("__", f"{col}__{cat}") for cat in cats]
            for col, cats in self.column_categories.items()
        }
        self._set_output_dtypes()
        return self

    def _set_output_dtypes(self) -> None:
        """Declare Float64 for every generated one-hot indicator column."""
        # Reuses the names already normalized in _column_mapping instead of
        # recomputing the same regex substitution a second time.
        self._output_dtypes = {
            name: pl.Float64 for names in (self._column_mapping or {}).values() for name in names
        }

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
        if self.column_categories is None:
            return X  # pragma: no cover

        # Use native Polars to_dummies - single efficient call
        cols_to_encode = list(self.column_categories.keys())
        # Only Categorical/Enum columns need an explicit cast to String before to_dummies;
        # casting an already-String column is a no-op that still costs a full pass, so it's
        # skipped for the common case (subset columns are already strings).
        needs_string_cast = {
            col for col in cols_to_encode if X[col].dtype.base_type() in (pl.Categorical, pl.Enum)
        }
        # Nulls must map to the "MISSING__" category learned in fit(), not Polars' own null
        # handling. Cast (where needed) and fill_null are combined into one with_columns pass
        # instead of two, to avoid a second full-frame materialization.
        X_encode = X.select(cols_to_encode).with_columns(
            [
                (pl.col(c).cast(pl.String) if c in needs_string_cast else pl.col(c)).fill_null(
                    "MISSING__"
                )
                for c in cols_to_encode
            ]
        )
        dummies = X_encode.to_dummies(separator="__")
        # Normalize 3+ consecutive underscores to __ (e.g. col____MISSING__ → col____MISSING__ → col__MISSING__)
        dummies = dummies.rename({c: _norm_col.sub('__', c) for c in dummies.columns})

        # Reuse the names already normalized in fit()'s _column_mapping instead of
        # recomputing the same regex substitution again here.
        expected_cols = [name for names in self._column_mapping.values() for name in names]

        # Materialize dummies.columns into a set ONCE. Checking `c in dummies.columns`
        # inside a loop (the previous approach) re-materializes the list *and* does an
        # O(n) scan on every iteration, making category lookups O(cardinality^2); at
        # cardinality=1000 that dominates the whole transform (see benchmarks/README.md,
        # "Extended benchmark matrix" -- this was the root cause of the reported
        # OneHotEncoder regression at high cardinality).
        dummies_cols = set(dummies.columns)
        missing_cols = [c for c in expected_cols if c not in dummies_cols]
        if missing_cols:
            dummies = dummies.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])

        # Single pass: reorder to column_categories order (plain names -- cheap, no
        # per-column expression objects) and drop any unseen-category dummy columns
        # (anything not in expected_cols) at once; cast is a single bulk DataFrame.cast
        # call rather than one pl.col(c).cast(...) expression per category, which
        # profiling showed was a meaningful share of Python-side overhead at high
        # cardinality (building/parsing 1,000+ expression objects vs. one cast call).
        dummies = dummies.select(expected_cols).cast(pl.Float64)

        # Concatenate with original dataframe
        X = pl.concat([X, dummies], how="horizontal_extend")

        # Drop original columns if requested
        if self.drop_columns and self.subset:
            X = X.drop(self.subset)

        return X
