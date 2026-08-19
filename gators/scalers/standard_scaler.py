import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class StandardScaler(_BaseTransformer):
    """
    Standardizes numeric features by removing the mean and scaling to unit variance.

    Transforms features by centering them around zero and scaling by the standard
    deviation. The transformation is given by: X_scaled = (X - mean) / std.
    This is also known as z-score normalization.

    Parameters
    ----------
    subset : list[str], default=None
        List of numeric column names to standardize. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    inplace : bool, default=True
        If True, standardize values in the original columns (keep original column names).
        If False, create new columns with suffix ``__standard_scale``.
    drop_columns : bool, default=True
        If ``inplace=False``, whether to drop the original columns after standardizing.
        Ignored when ``inplace=True``.

    Examples
    --------
    Create an instance of the StandardScaler class:

    >>> import polars as pl
    >>> from gators.scalers import StandardScaler
    >>> scaler = StandardScaler(subset=["age", "income"])

    Fit the transformer:

    >>> X = pl.DataFrame({"age": [20, 30, 40, 50],
    ...                    "income": [20000, 40000, 60000, 80000]})
    >>> scaler.fit(X)

    Transform the DataFrame:

    >>> transformed_X = scaler.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌─────────────────────┬──────────────────────┐
    │ age__standard_scale  ┆ income__standard_scale │
    │ ---                 ┆ ---                   │
    │ f64                 ┆ f64                   │
    ├────────────────────┼──────────────────────┤
    │ -1.161              ┆ -1.161                │
    │ -0.387              ┆ -0.387                │
    │ 0.387               ┆ 0.387                 │
    │ 1.161               ┆ 1.161                 │
    └────────────────────┴──────────────────────┘

    """

    subset: list[str] | None = None
    inplace: bool = True
    drop_columns: bool = True
    _offset: dict[str, float] = PrivateAttr(default_factory=dict)
    _scale: dict[str, float] = PrivateAttr(default_factory=dict)
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "StandardScaler":
        """Fit the transformer by computing mean and standard deviation.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : pl.Series, default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        StandardScaler
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype.is_numeric()
            ]
        if not self.inplace:
            self._column_mapping = {col: f"{col}__standard_scale" for col in self.subset}

        mean_std_exprs = []
        for col in self.subset:
            mean_std_exprs.append(pl.col(col).mean().alias(f"{col}__mean"))
            mean_std_exprs.append(pl.col(col).std().alias(f"{col}__std"))

        stats = X.select(mean_std_exprs).row(0)

        self._offset = {}
        self._scale = {}
        for i, col in enumerate(self.subset):
            mean_val = stats[i * 2]
            std_val = stats[i * 2 + 1]
            self._offset[col] = mean_val if mean_val is not None else 0.0
            self._scale[col] = 1.0 / std_val if std_val else 0.0

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying standard scaling.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with standardized columns.
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
        """Reverse the standard scaling.

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
