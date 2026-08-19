import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class MinmaxScaler(_BaseTransformer):
    """
    Scales numeric features to a [0, 1] range using min-max normalization.

    Transforms features by scaling each feature to the range [0, 1] based on
    the minimum and maximum values observed during fitting. The transformation
    is given by: X_scaled = (X - X_min) / (X_max - X_min).

    Parameters
    ----------
    subset : list[str], default=None
        List of numeric column names to scale. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    inplace : bool, default=True
        If True, scale values in the original columns (keep original column names).
        If False, create new columns with suffix ``__minmax_scale``.
    drop_columns : bool, default=True
        If ``inplace=False``, whether to drop the original columns after scaling.
        Ignored when ``inplace=True``.

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

    subset: list[str] | None = None
    inplace: bool = True
    drop_columns: bool = True
    _offset: dict[str, float] = PrivateAttr(default_factory=dict)
    _scale: dict[str, float] = PrivateAttr(default_factory=dict)
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "MinmaxScaler":
        """Fit the transformer by computing min and max values.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : pl.Series, default=None
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
                if dtype.is_numeric()
            ]
        if not self.inplace:
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
            range_val = (
                (max_val - min_val) if (max_val is not None and min_val is not None) else 0.0
            )
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
        if self.inplace:
            transformations = [
                (self._scale[col] * (pl.col(col) - self._offset[col])).alias(col)
                for col in self.subset
            ]
            return X.with_columns(transformations)

        transformations = [
            (self._scale[col] * (pl.col(col) - self._offset[col])).alias(new)
            for col, new in self._column_mapping.items()
        ]
        X = X.with_columns(transformations)
        if self.drop_columns:
            return X.drop(self.subset)
        return X

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Reverse the min-max scaling.

        Parameters
        ----------
        X : pl.DataFrame
            DataFrame with scaled columns (output of ``transform``).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns restored to their original scale.
        """
        def _inv_expr(scaled_col: str, orig_col: str) -> pl.Expr:
            scale = self._scale[orig_col]
            offset = self._offset[orig_col]
            if scale == 0.0:
                return pl.lit(offset).alias(orig_col)
            return (pl.col(scaled_col) / scale + offset).alias(orig_col)

        if self.inplace:
            exprs = [_inv_expr(col, col) for col in self.subset]
            return X.with_columns(exprs)
        reverse_map = {v: k for k, v in self._column_mapping.items()}
        exprs = [_inv_expr(new, orig) for new, orig in reverse_map.items() if new in X.columns]
        X = X.with_columns(exprs)
        if self.drop_columns:
            return X.drop([c for c in reverse_map if c in X.columns])
        return X
