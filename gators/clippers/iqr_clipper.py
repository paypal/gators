from typing import Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseModel, BaseEstimator, TransformerMixin):
    """
    Clip numeric values based on Interquartile Range (IQR).

    This transformer caps values that fall outside the range 
    [Q1 - n_iqrs*IQR, Q3 + n_iqrs*IQR], where Q1 and Q3 are the first and 
    third quartiles, and IQR = Q3 - Q1. This is a robust method commonly 
    used for outlier detection (n_iqrs=1.5 is the standard for box plots).

    Parameters
    ----------
    n_iqrs : float, default=1.5
        Number of IQRs beyond Q1/Q3 to use for clipping bounds.
        Must be a positive number. Common values:
        - 1.5: Standard box plot outlier threshold
        - 3.0: Extreme outlier threshold
    subset : Optional[List[str]], default=None
        List of numeric columns to clip. If None, all numeric columns are selected.
    inplace : bool, default=True
        If True, clip values in the original columns.
        If False, create new columns with suffix '__clip_iqr'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after clipping.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.clipping import IQRClipper

    >>> # Sample DataFrame with outliers
    >>> X = pl.DataFrame({
    ...     'A': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 100.0],
    ...     'B': [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    ... })

    >>> # Clip using 1.5 IQRs (default, standard box plot threshold)
    >>> clipper = IQRClipper(inplace=False)
    >>> clipper.fit(X)
    IQRClipper(n_iqrs=1.5, subset=['A', 'B'], drop_columns=True, inplace=False)
    >>> transformed_X = clipper.transform(X)
    >>> print(transformed_X)
    shape: (12, 2)
    ┌──────────────┬──────────────┐
    │ A__clip_iqr  ┆ B__clip_iqr  │
    │ ---          ┆ ---          │
    │ f64          ┆ f64          │
    ╞══════════════╪══════════════╡
    │ 10.0         ┆ 1.25         │
    │ 11.0         ┆ 10.0         │
    │ 12.0         ┆ 11.0         │
    │ 13.0         ┆ 12.0         │
    │ 14.0         ┆ 13.0         │
    │ 15.0         ┆ 14.0         │
    │ 16.0         ┆ 15.0         │
    │ 17.0         ┆ 16.0         │
    │ 18.0         ┆ 17.0         │
    │ 19.0         ┆ 18.0         │
    │ 20.0         ┆ 19.0         │
    │ 28.75        ┆ 20.0         │
    └──────────────┴──────────────┘

    >>> # More conservative clipping with 3 IQRs
    >>> clipper_3iqr = IQRClipper(n_iqrs=3.0, inplace=False)
    >>> clipper_3iqr.fit(X)
    IQRClipper(n_iqrs=3.0, subset=['A', 'B'], drop_columns=True, inplace=False)
    >>> transformed_X_3iqr = clipper_3iqr.transform(X)
    >>> print(transformed_X_3iqr)
    shape: (12, 2)
    ┌──────────────┬──────────────┐
    │ A__clip_iqr  ┆ B__clip_iqr  │
    │ ---          ┆ ---          │
    │ f64          ┆ f64          │
    ╞══════════════╪══════════════╡
    │ 10.0         ┆ -15.0        │
    │ 11.0         ┆ 10.0         │
    │ 12.0         ┆ 11.0         │
    │ 13.0         ┆ 12.0         │
    │ 14.0         ┆ 13.0         │
    │ 15.0         ┆ 14.0         │
    │ 16.0         ┆ 15.0         │
    │ 17.0         ┆ 16.0         │
    │ 18.0         ┆ 17.0         │
    │ 19.0         ┆ 18.0         │
    │ 20.0         ┆ 19.0         │
    │ 43.0         ┆ 20.0         │
    └──────────────┴──────────────┘
    """

    n_iqrs: float = Field(default=1.5, gt=0.0)
    subset: Optional[List[str]] = None
    drop_columns: bool = True
    inplace: bool = True
    _clip_bounds: Dict[str, Tuple[float, float]] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "IQRClipper":
        """Fit the transformer by computing IQR-based clipping bounds.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        IQRClipper
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype not in [pl.String, pl.Boolean]
            ]

        if not self.inplace:
            self._column_mapping = {
                col: f"{col}__clip_iqr" for col in self.subset
            }

        # Compute Q1, Q3, and IQR for each column
        for col in self.subset:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            
            # Clipping bounds: [Q1 - n_iqrs*IQR, Q3 + n_iqrs*IQR]
            lower_bound = q1 - self.n_iqrs * iqr
            upper_bound = q3 + self.n_iqrs * iqr
            self._clip_bounds[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by clipping values based on IQR.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with clipped numeric columns.
        """
        # Build all transformations at once
        if self.inplace:
            transformations = [
                pl.col(col).clip(
                    lower_bound=self._clip_bounds[col][0],
                    upper_bound=self._clip_bounds[col][1]
                )
                for col in self.subset
            ]
        else:
            transformations = [
                pl.col(col).clip(
                    lower_bound=self._clip_bounds[col][0],
                    upper_bound=self._clip_bounds[col][1]
                ).alias(new)
                for col, new in self._column_mapping.items()
            ]

        X = X.with_columns(transformations)

        if not self.inplace and self.drop_columns:
            return X.drop(self.subset)
        return X
