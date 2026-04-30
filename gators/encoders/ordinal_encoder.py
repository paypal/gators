from typing import Optional

import numpy as np
import polars as pl

from ._base_encoder import _BaseEncoder


class OrdinalEncoder(_BaseEncoder):
    """
    Encodes categorical values as ordinal.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of categorical columns to encode. If None, all string, boolean, and categorical columns are selected.
    min_count : Union[int, float], default=1
        Minimum count threshold for encoding categories. If >= 1, treated as absolute count; if < 1, treated as frequency.
    inplace : bool, default=True
        If True, replace original columns with encoded values.
        If False, create new columns with suffix '__ordinal_enc'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after encoding.
        Ignored when inplace=True.

    Examples
    --------
    Basic usage:

    >>> from gators.encoders import OrdinalEncoder
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     "A": ["foo", "bar", "foo", "bar", "baz"],
    ...     "B": [True, False, True, True, False],
    ... })
    >>> encoder = OrdinalEncoder(inplace=False)
    >>> _ = encoder.fit(X)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 2)
    ┌───────────────┬───────────────┐
    │ A__ordinal_enc│ B__ordinal_enc│
    │ f64           │ f64           │
    ╞═══════════════╪═══════════════╡
    │ 3.0           │ 2.0           │
    │ 2.0           │ 1.0           │
    │ 3.0           │ 2.0           │
    │ 2.0           │ 2.0           │
    │ 1.0           │ 1.0           │
    └───────────────┴───────────────┘

    Drop columns:

    >>> encoder = OrdinalEncoder(drop_columns=False, inplace=False)
    >>> _ = encoder.fit(X)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    │ A            │        B     │A__ordinal_enc│B__ordinal_enc│
    │ str          │        bool  │f64           │ f64          │
    ╞══════════════╪══════════════╪══════════════╪══════════════╡
    │ foo          │        true  │3.0           │ 2.0          │
    │ bar          │        false │2.0           │ 1.0          │
    │ foo          │        true  │3.0           │ 2.0          │
    │ bar          │        true  │2.0           │ 2.0          │
    │ baz          │        false │1.0           │ 1.0          │
    └──────────────┴──────────────┴──────────────┴──────────────┘

    Subset of columns:

    >>> encoder = OrdinalEncoder(subset=["A"], inplace=False)
    >>> _ = encoder.fit(X)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 1)
    ┌───────────────┐
    │ A__ordinal_enc│
    │ f64           │
    ╞═══════════════╡
    │ 3.0           │
    │ 2.0           │
    │ 3.0           │
    │ 2.0           │
    │ 1.0           │
    └───────────────┘

    """

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "OrdinalEncoder":
        """Fit the transformer by computing ordinal mappings based on category frequency.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        OrdinalEncoder
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype in [pl.String, pl.Boolean, pl.Enum]
            ]
        self.mapping_ = {}
        n = len(X)
        for col in self.subset:
            counts = X[col].value_counts().sort(["count", col])
            if self.min_count >= 1:
                counts = counts.filter(pl.col("count") >= self.min_count)
            else:
                counts = counts.filter(pl.col("count") / n >= self.min_count)

            values = np.arange(1, len(counts) + 1, dtype=float)
            self.mapping_[col] = dict(zip(counts[col], values))
        self.column_mapping_ = {col: f"{col}__ordinal_enc" for col in self.subset}

        return self
