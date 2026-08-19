import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer


class RobustScaler(_BaseTransformer):
    """Scale numeric features using median and a configurable quantile range.

    Applies the transformation::

        X_scaled = (X - median) / (Q(q_high) - Q(q_low))

    Because centering and scaling are driven by robust statistics (median and
    inter-quantile range) rather than mean and standard deviation, the scaler
    is largely unaffected by outliers.

    Parameters
    ----------
    quantile_range : tuple[float, float], default=(0.25, 0.75)
        ``(q_low, q_high)`` — the two quantile levels used to compute the
        scale.  Both values must be in ``[0.0, 1.0]`` and ``q_low < q_high``.
        The default ``(0.25, 0.75)`` is equivalent to the classic IQR-based
        robust scaler.
    subset : list[str] or None, default=None
        Numeric columns to scale.  When ``None`` all Float64, Float32, Int64,
        and Int32 columns are selected automatically.
    inplace : bool, default=True
        If True, scale values in the original columns (keep original column names).
        If False, create new columns with suffix ``__robust_quantile_scale``.
    drop_columns : bool, default=True
        If ``inplace=False``, whether to drop the original columns after scaling.
        Ignored when ``inplace=True``.

    Attributes
    ----------
    _median : dict[str, float]
        Fitted median per column.
    _scale : dict[str, float]
        Fitted IQR-based scale (``1 / (Q_high - Q_low)``) per column.
        Columns where the quantile range is zero get a scale of ``0.0``
        (i.e. scaled output will be all zeros).
    _column_mapping : dict[str, str]
        Mapping from original column name to scaled column name.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.scalers import RobustScaler

    >>> X = pl.DataFrame({
    ...     "age":    [20.0, 30.0, 40.0, 50.0, 200.0],
    ...     "income": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
    ... })
    >>> scaler = RobustScaler(quantile_range=(0.25, 0.75))
    >>> scaler.fit(X)
    >>> X_scaled = scaler.transform(X)
    """

    quantile_range: tuple[float, float] = (0.25, 0.75)
    subset: list[str] | None = None
    inplace: bool = True
    drop_columns: bool = True

    _median: dict[str, float] = PrivateAttr(default_factory=dict)
    _scale: dict[str, float] = PrivateAttr(default_factory=dict)
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("quantile_range")
    @classmethod
    def _validate_quantile_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        q_low, q_high = v
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError(f"quantile_range must satisfy 0.0 <= q_low < q_high <= 1.0, got {v}")
        return v

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "RobustScaler":
        """Compute median and quantile-range scale for each column.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : pl.Series, default=None
            Not used; present for sklearn compatibility.

        Returns
        -------
        RobustScaler
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype.is_numeric()
            ]

        if not self.inplace:
            self._column_mapping = {col: f"{col}__robust_quantile_scale" for col in self.subset}

        q_low, q_high = self.quantile_range
        stat_exprs = []
        for col in self.subset:
            stat_exprs.append(pl.col(col).median().alias(f"{col}__median"))
            stat_exprs.append(pl.col(col).quantile(q_low).alias(f"{col}__q_low"))
            stat_exprs.append(pl.col(col).quantile(q_high).alias(f"{col}__q_high"))

        stats = X.select(stat_exprs).row(0)

        self._median = {}
        self._scale = {}
        for i, col in enumerate(self.subset):
            median_val = stats[i * 3]
            q_low_val = stats[i * 3 + 1]
            q_high_val = stats[i * 3 + 2]
            self._median[col] = median_val if median_val is not None else 0.0
            iqr = (q_high_val or 0.0) - (q_low_val or 0.0)
            self._scale[col] = 1.0 / iqr if iqr else 0.0

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Scale the numeric columns using the fitted median and quantile range.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with robust-scaled columns.
        """
        if self.inplace:
            transformations = [
                (self._scale[col] * (pl.col(col) - self._median[col])).alias(col)
                for col in self.subset
            ]
            return X.with_columns(transformations)

        transformations = [
            (self._scale[col] * (pl.col(col) - self._median[col])).alias(new_col)
            for col, new_col in self._column_mapping.items()
        ]
        X = X.with_columns(transformations)
        if self.drop_columns:
            return X.drop(self.subset)
        return X

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Reverse the robust scaling.

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
            median = self._median[orig_col]
            if scale == 0.0:
                return pl.lit(median).alias(orig_col)
            return (pl.col(scaled_col) / scale + median).alias(orig_col)

        if self.inplace:
            exprs = [_inv_expr(col, col) for col in self.subset]
            return X.with_columns(exprs)
        reverse_map = {v: k for k, v in self._column_mapping.items()}
        exprs = [_inv_expr(new, orig) for new, orig in reverse_map.items() if new in X.columns]
        X = X.with_columns(exprs)
        if self.drop_columns:
            return X.drop([c for c in reverse_map if c in X.columns])
        return X
