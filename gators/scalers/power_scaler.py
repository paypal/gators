from typing import Dict, List, Optional

import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class PowerScaler(_BaseTransformer):
    """
    Scales numeric features using power transformation X^power.

    Applies a power transformation to selected features. Useful for reducing
    skewness or adjusting the scale of features. Common power values:

    - power < 1: compress large values (e.g., 0.5 for square root)
    - power > 1: expand large values (e.g., 2 for squaring)

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to transform. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    power : float, default=0.5
        The power exponent to apply. Default is 0.5 (square root).
    drop_columns : bool, default=True
        If True, drop the original columns after transformation.
        If False, keep both original and transformed columns.

    Examples
    --------
    Create an instance of the PowerScaler class:

    >>> import polars as pl
    >>> from gators.scalers import PowerScaler
    >>> scaler = PowerScaler(subset=["sales", "revenue"], power=0.5)

    Fit the transformer:

    >>> X = pl.DataFrame({"sales": [100, 400, 900, 1600],
    ...                    "revenue": [1000, 4000, 9000, 16000]})
    >>> scaler.fit(X)

    Transform the DataFrame:

    >>> transformed_X = scaler.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌──────────────────┬─────────────────────┐
    │ sales__power_0.5 ┆ revenue__power_0.5  │
    │ ---              ┆ ---                 │
    │ f64              ┆ f64                 │
    ├──────────────────┼─────────────────────┤
    │ 10.0             ┆ 31.62               │
    │ 20.0             ┆ 63.25               │
    │ 30.0             ┆ 94.87               │
    │ 40.0             ┆ 126.49              │
    └──────────────────┴─────────────────────┘

    >>> # Square transformation
    >>> scaler2 = PowerScaler(subset=["count"], power=2.0)
    >>> X2 = pl.DataFrame({"count": [1, 2, 3, 4]})
    >>> scaler2.fit(X2)
    >>> scaler2.transform(X2)
    shape: (4, 1)
    ┌────────────────┐
    │ count__power_2 │
    │ ---            │
    │ f64            │
    ├────────────────┤
    │ 1.0            │
    │ 4.0            │
    │ 9.0            │
    │ 16.0           │
    └────────────────┘
    """

    subset: Optional[List[str]] = None
    power: float = 0.5
    _column_mapping: Dict[str, str] = PrivateAttr()
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "PowerScaler":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        PowerScaler
            The fitted transformer instance.
        """
        if not self.subset:
            # Use set for O(1) dtype lookup instead of list O(n) lookup
            numeric_dtypes = {pl.Float64, pl.Int64, pl.Float32, pl.Int32}
            self.subset = [
                col for col, dtype in zip(X.columns, X.dtypes) if dtype in numeric_dtypes
            ]

        # Format power value for column naming
        power_str = str(self.power).replace(".", "_")
        self._column_mapping = {col: f"{col}__power_{power_str}" for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying power transformation.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with power-transformed columns.
        """
        transformations = [
            (pl.col(col) ** self.power).alias(new) for col, new in self._column_mapping.items()
        ]

        X = X.with_columns(transformations)

        if self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X
