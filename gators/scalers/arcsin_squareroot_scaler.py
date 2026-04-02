from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin


class ArcSinSquareRootScaler(BaseModel, BaseEstimator, TransformerMixin):
    """
    Applies arcsine square root transformation for proportion data.

    The transformation arcsin(sqrt(X)) is a variance-stabilizing transformation
    commonly used for proportion or percentage data. It's particularly useful
    when data are bounded between 0 and 1, making the variance more homogeneous
    across the range.

    This transformation is often used in:

    - Analysis of proportions, percentages, or rates
    - Binomial proportion data
    - Data from beta distributions

    Note: Input values should be in the range [0, 1]. Values outside this range
    will produce NaN or complex numbers.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to transform. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    drop_columns : bool, default=True
        If True, drop the original columns after transformation.
        If False, keep both original and transformed columns.

    Examples
    --------
    Create an instance of the ArcSinSquareRootScaler class:

    >>> import polars as pl
    >>> from gators.scalers import ArcSinSquareRootScaler
    >>> scaler = ArcSinSquareRootScaler(subset=["success_rate", "conversion_rate"])

    Fit the transformer:

    >>> X = pl.DataFrame({
    ...     "success_rate": [0.1, 0.25, 0.5, 0.75, 0.9],
    ...     "conversion_rate": [0.05, 0.15, 0.30, 0.60, 0.85]
    ... })
    >>> scaler.fit(X)

    Transform the DataFrame:

    >>> transformed_X = scaler.transform(X)
    >>> print(transformed_X)
    shape: (5, 2)
    ┌─────────────────────────┬────────────────────────────┐
    │ success_rate__arcsin    ┆ conversion_rate__arcsin    │
    │ ---                     ┆ ---                        │
    │ f64                     ┆ f64                        │
    ├─────────────────────────┼────────────────────────────┤
    │ 0.322                   ┆ 0.227                      │
    │ 0.524                   ┆ 0.395                      │
    │ 0.785                   ┆ 0.588                      │
    │ 1.047                   ┆ 0.908                      │
    │ 1.249                   ┆ 1.150                      │
    └─────────────────────────┴────────────────────────────┘

    Notes
    -----
    The transformation maps [0, 1] to [0, π/2] (0 to ~1.571).
    """

    subset: Optional[List[str]] = None
    _column_mapping: Dict[str, str]
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "ArcSinSquareRootScaler":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        ArcSinSquareRootScaler
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
            ]

        self._column_mapping = {col: f"{col}__arcsin" for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying arcsin(sqrt(X)).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform. Values should be in [0, 1].

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with arcsin-transformed columns.
        """
        transformations = [
            pl.col(col).sqrt().arcsin().alias(new) for col, new in self._column_mapping.items()
        ]

        X = X.with_columns(transformations)

        if self.drop_columns:
            return X.drop(self.subset)
        return X
