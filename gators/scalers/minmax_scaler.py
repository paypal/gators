from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class MinmaxScaler(_BaseTransformer):
    """
    Scales numeric features to a [0, 1] range using min-max normalization.

    Transforms features by scaling each feature to the range [0, 1] based on
    the minimum and maximum values observed during fitting. The transformation
    is given by: X_scaled = (X - X_min) / (X_max - X_min).

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to scale. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    drop_columns : bool, default=True
        If True, drop the original columns after scaling.
        If False, keep both original and scaled columns.

    Examples
    --------
    Create an instance of the MinmaxScaler class:

    >>> import polars as pl
    >>> from gators.scalers import MinmaxScaler
    >>> scaler = MinmaxScaler(subset=["age", "income"])

    Fit the transformer:

    >>> X = pl.DataFrame({"age": [20, 30, 40, 50],
    ...                    "income": [20000, 40000, 60000, 80000]})
    >>> scaler.fit(X)

    Transform the DataFrame:

    >>> transformed_X = scaler.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌───────────────────┬─────────────────────┐
    │ age__minmax_scale ┆ income__minmax_scale│
    │ ---               ┆ ---                 │
    │ f64               ┆ f64                 │
    ├───────────────────┼─────────────────────┤
    │ 0.0               ┆ 0.0                 │
    │ 0.333             ┆ 0.333               │
    │ 0.667             ┆ 0.667               │
    │ 1.0               ┆ 1.0                 │
    └───────────────────┴─────────────────────┘

    """

    subset: Optional[List[str]] = None
    _offset: Dict[str, float]
    _scale: Dict[str, float]
    _column_mapping: Dict[str, str]
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "MinmaxScaler":
        """Fit the transformer by computing min and max values.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        MinmaxScaler
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
            ]
        self._column_mapping = {col: f"{col}__minmax_scale" for col in self.subset}

        # Single-pass min/max computation - build all expressions at once
        min_max_exprs = []
        for col in self.subset:
            min_max_exprs.append(pl.col(col).min().alias(f"{col}__min"))
            min_max_exprs.append(pl.col(col).max().alias(f"{col}__max"))

        stats = X.select(min_max_exprs).row(0)

        self._offset = {}
        self._scale = {}
        for i, col in enumerate(self.subset):
            min_val = stats[i * 2]
            max_val = stats[i * 2 + 1]
            self._offset[col] = min_val if min_val is not None else 0.0
            range_val = (max_val - min_val) if (max_val is not None and min_val is not None) else 0.0
            self._scale[col] = 1.0 / range_val if range_val else 0.0

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying min-max scaling.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with scaled columns.
        """
        transformations = [
            (self._scale[col] * (pl.col(col) - self._offset[col])).alias(new)
            for col, new in self._column_mapping.items()
        ]

        X = X.with_columns(transformations)
        if self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X
