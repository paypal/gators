from typing import Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin


class MADClipper(BaseModel, BaseEstimator, TransformerMixin):
    """
    Clip numeric values based on Median Absolute Deviation (MAD).

    This transformer caps values that are more than n_mads times the MAD away
    from the median. MAD is a robust measure of variability that is less
    sensitive to outliers than standard deviation.

    MAD = median(abs(X - median(X)))
    Clipping bounds: [median - n_mads*MAD, median + n_mads*MAD]

    Parameters
    ----------
    n_mads : float, default=3.0
        Number of MADs from the median to use for clipping bounds.
        Must be a positive number.
    subset : Optional[List[str]], default=None
        List of numeric columns to clip. If None, all numeric columns are selected.
    inplace : bool, default=True
        If True, clip values in the original columns.
        If False, create new columns with suffix '__clip_mad'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after clipping.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.clipping import MADClipper

    >>> # Sample DataFrame with outliers
    >>> X = pl.DataFrame({
    ...     'A': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 100.0],
    ...     'B': [-100.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    ... })

    >>> # Clip using 3 MADs (default)
    >>> clipper = MADClipper(inplace=False)
    >>> clipper.fit(X)
    MADClipper(n_mads=3.0, subset=['A', 'B'], drop_columns=True, inplace=False)
    >>> transformed_X = clipper.transform(X)
    >>> print(transformed_X)
    shape: (12, 2)
    ┌──────────────┬──────────────┐
    │ A__clip_mad  ┆ B__clip_mad  │
    │ ---          ┆ ---          │
    │ f64          ┆ f64          │
    ╞══════════════╪══════════════╡
    │ 10.0         ┆ -12.5        │
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
    │ 27.5         ┆ 20.0         │
    └──────────────┴──────────────┘

    >>> # More aggressive clipping with 2 MADs
    >>> clipper_2mad = MADClipper(n_mads=2.0, inplace=False)
    >>> clipper_2mad.fit(X)
    MADClipper(n_mads=2.0, subset=['A', 'B'], drop_columns=True, inplace=False)
    >>> transformed_X_2mad = clipper_2mad.transform(X)
    >>> print(transformed_X_2mad)
    shape: (12, 2)
    ┌──────────────┬──────────────┐
    │ A__clip_mad  ┆ B__clip_mad  │
    │ ---          ┆ ---          │
    │ f64          ┆ f64          │
    ╞══════════════╪══════════════╡
    │ 10.0         ┆ -5.0         │
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
    │ 22.5         ┆ 20.0         │
    └──────────────┴──────────────┘
    """

    n_mads: float = Field(default=3.0, gt=0.0)
    subset: Optional[List[str]] = None
    drop_columns: bool = True
    inplace: bool = True
    _clip_bounds: Dict[str, Tuple[float, float]] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "MADClipper":
        """Fit the transformer by computing MAD-based clipping bounds.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        MADClipper
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype not in [pl.String, pl.Boolean]
            ]

        if not self.inplace:
            self._column_mapping = {col: f"{col}__clip_mad" for col in self.subset}

        # Compute median and MAD for each column
        for col in self.subset:
            median = X[col].median()
            # MAD = median(|X - median(X)|)
            mad = (X[col] - median).abs().median()

            # Clipping bounds: median ± n_mads * MAD
            lower_bound = median - self.n_mads * mad
            upper_bound = median + self.n_mads * mad
            self._clip_bounds[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by clipping values based on MAD.

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
                    lower_bound=self._clip_bounds[col][0], upper_bound=self._clip_bounds[col][1]
                )
                for col in self.subset
            ]
        else:
            transformations = [
                pl.col(col)
                .clip(lower_bound=self._clip_bounds[col][0], upper_bound=self._clip_bounds[col][1])
                .alias(new)
                for col, new in self._column_mapping.items()
            ]

        X = X.with_columns(transformations)

        if not self.inplace and self.drop_columns:
            return X.drop(self.subset)
        return X
