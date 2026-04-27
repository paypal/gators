from typing import Dict, Optional

import numpy as np
import polars as pl

from ._base_encoder import _BaseEncoder


class BinaryEncoder(_BaseEncoder):
    """
    Encodes categorical values using binary representation.

    Each category is first encoded as an integer, then converted to binary,
    with each binary digit becoming a separate column. This is more compact
    than one-hot encoding for high cardinality features.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of categorical columns to encode. If None, all string, boolean, and categorical columns are selected.
    min_count : Union[int, float], default=1
        Minimum count threshold for encoding categories. If >= 1, treated as absolute count; if < 1, treated as frequency.
    inplace : bool, default=True
        If True, replace original columns with encoded values.
        If False, create new columns with suffix '__binary_enc_{bit_index}'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after encoding.
        Ignored when inplace=True.

    Examples
    --------
    Initialize and use `BinaryEncoder`.

    Example with `drop_columns=True` and `columns=None`:

    >>> import polars as pl
    >>> from gators.encoders import BinaryEncoder
    >>> X = pl.DataFrame({
    ...     "category": ["A", "B", "C", "D", "A", "B"],
    ...     "value": [1, 2, 3, 4, 5, 6]
    ... })
    >>> encoder = BinaryEncoder(min_count=1, inplace=False, drop_columns=True)
    >>> _ = encoder.fit(X)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (6, 3)
    ┌───────┬────────────────────────┬────────────────────────┐
    │ value ┆ category__binary_enc_0 ┆ category__binary_enc_1 │
    │ ---   ┆ ---                    ┆ ---                    │
    │ i64   ┆ f64                    ┆ f64                    │
    ╞═══════╪════════════════════════╪════════════════════════╡
    │ 1     ┆ 1.0                    ┆ 1.0                    │
    │ 2     ┆ 1.0                    ┆ 0.0                    │
    │ 3     ┆ 0.0                    ┆ 0.0                    │
    │ 4     ┆ 0.0                    ┆ 1.0                    │
    │ 5     ┆ 1.0                    ┆ 1.0                    │
    │ 6     ┆ 1.0                    ┆ 0.0                    │
    └───────┴────────────────────────┴────────────────────────┘

    Example with `drop_columns=False`:

    >>> X = pl.DataFrame({
    ...     "category": ["A", "B", "C", "D", "A", "B"],
    ...     "value": [1, 2, 3, 4, 5, 6]
    ... })
    >>> encoder = BinaryEncoder(subset=["category"], inplace=False, drop_columns=False)
    >>> _ = encoder.fit(X)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (6, 4)
    ┌──────────┬───────┬────────────────────────┬────────────────────────┐
    │ category ┆ value ┆ category__binary_enc_0 ┆ category__binary_enc_1 │
    │ ---      ┆ ---   ┆ ---                    ┆ ---                    │
    │ str      ┆ i64   ┆ f64                    ┆ f64                    │
    ╞══════════╪═══════╪════════════════════════╪════════════════════════╡
    │ A        ┆ 1     ┆ 0.0                    ┆ 0.0                    │
    │ B        ┆ 2     ┆ 1.0                    ┆ 0.0                    │
    │ C        ┆ 3     ┆ 0.0                    ┆ 1.0                    │
    │ D        ┆ 4     ┆ 1.0                    ┆ 1.0                    │
    │ A        ┆ 5     ┆ 0.0                    ┆ 0.0                    │
    │ B        ┆ 6     ┆ 1.0                    ┆ 0.0                    │
    └──────────┴───────┴────────────────────────┴────────────────────────┘
    """

    n_bits_: Dict[str, int] = {}

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "BinaryEncoder":
        """Fit the transformer by computing binary encoding mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        BinaryEncoder
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.String, pl.Boolean, pl.Categorical]
            ]

        min_threshold_count = self.min_count if self.min_count >= 1 else self.min_count * len(X)

        self.n_bits_ = {}
        self.mapping_ = {}

        for col in self.subset:
            # Get unique categories that meet min_count threshold
            value_counts = X[col].value_counts()
            if value_counts.is_empty():
                continue

            valid_categories = [
                cat
                for cat, count in zip(value_counts[col].to_list(), value_counts["count"].to_list())
                if count >= min_threshold_count
            ]

            if not valid_categories:
                continue

            # Calculate number of bits needed
            n_categories = len(valid_categories)
            n_bits = int(np.ceil(np.log2(n_categories))) if n_categories > 1 else 1
            self.n_bits_[col] = n_bits

            # Create binary encoding for each category
            for idx, category in enumerate(valid_categories):
                # Convert index to binary representation
                binary_str = format(idx, f"0{n_bits}b")

                # Store each bit as a separate mapping
                for bit_idx, bit in enumerate(reversed(binary_str)):
                    bit_col = f"{col}__binary_enc_{bit_idx}"
                    if bit_col not in self.mapping_:
                        self.mapping_[bit_col] = {}
                    self.mapping_[bit_col][category] = float(bit)

        # Column mapping for drop_columns functionality
        self.column_mapping_ = {col: col for col in self.mapping_.keys()}

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying binary encoding to categorical columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with binary encoded columns (each bit as a separate column).
        """
        default_value = 0.0
        expressions = []
        if self.subset is None:
            return X
        for col in self.subset:
            if col not in self.n_bits_:
                continue

            for bit_idx in range(self.n_bits_[col]):
                bit_col = f"{col}__binary_enc_{bit_idx}"
                if bit_col in self.mapping_:
                    expr = (
                        pl.col(col)
                        .replace_strict(
                            self.mapping_[bit_col],
                            default=default_value,
                            return_dtype=pl.Float64,
                        )
                        .alias(bit_col)
                    )
                    expressions.append(expr)

        X = X.with_columns(expressions)

        if self.drop_columns and self.subset:
            X = X.drop(self.subset)

        return X
