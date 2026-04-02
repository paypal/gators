from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin


class StandardScaler(BaseModel, BaseEstimator, TransformerMixin):
    """
    Standardizes numeric features by removing the mean and scaling to unit variance.

    Transforms features by centering them around zero and scaling by the standard
    deviation. The transformation is given by: X_scaled = (X - mean) / std.
    This is also known as z-score normalization.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to standardize. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    drop_columns : bool, default=True
        If True, drop the original columns after scaling.
        If False, keep both original and scaled columns.

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
    ┌────────────────────┬──────────────────────┐
    │ age__standard_scale ┆ income__standard_scale│
    │ ---                 ┆ ---                   │
    │ f64                 ┆ f64                   │
    ├────────────────────┼──────────────────────┤
    │ -1.161              ┆ -1.161                │
    │ -0.387              ┆ -0.387                │
    │ 0.387               ┆ 0.387                 │
    │ 1.161               ┆ 1.161                 │
    └────────────────────┴──────────────────────┘

    """

    subset: Optional[List[str]] = None
    _offset: Dict[str, float]
    _scale: Dict[str, float]
    _column_mapping: Dict[str, str]
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "StandardScaler":
        """Fit the transformer by computing mean and standard deviation.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : Optional[pl.Series], default=None
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
                if dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
            ]
        self._column_mapping = {col: f"{col}__standard_scale" for col in self.subset}
        means = X[self.subset].mean().to_dict(as_series=False)
        self._offset = {col: val[0] for col, val in means.items()}
        stds = X[self.subset].std().to_dict(as_series=False)
        self._scale = {col: 1.0 / val[0] for col, val in stds.items()}
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
        transformations = [
            (self._scale[col] * (pl.col(col) - self._offset[col])).alias(new)
            for col, new in self._column_mapping.items()
        ]

        X = X.with_columns(transformations)

        return X.drop(self.subset) if self.drop_columns else X
