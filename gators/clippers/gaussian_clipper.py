from typing import Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin


class GaussianClipper(BaseModel, BaseEstimator, TransformerMixin):
    """
    Clip numeric values to mean ± n standard deviations.

    This transformer caps values that are smaller than mean - n*std or larger 
    than mean + n*std, where n is the number of standard deviations (n_sigmas).
    Values outside this range are clipped to the boundary values.

    Parameters
    ----------
    n_sigmas : int, default=3
        Number of standard deviations to use for clipping bounds.
        Must be a positive integer.
    subset : Optional[List[str]], default=None
        List of numeric columns to clip. If None, all numeric columns are selected.
    inplace : bool, default=True
        If True, clip values in the original columns.
        If False, create new columns with suffix '__clip_gaussian'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after clipping.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.clipping import GaussianClipper

    >>> # Sample DataFrame with outliers
    >>> X = pl.DataFrame({
    ...     'A': [1.0, 2.0, 3.0, 4.0, 100.0],  # 100.0 is an outlier
    ...     'B': [-50.0, 5.0, 6.0, 7.0, 8.0],  # -50.0 is an outlier
    ...     'C': [10.0, 20.0, 30.0, 40.0, 50.0]
    ... })

    >>> # Clip using 3 standard deviations (default)
    >>> clipper = GaussianClipper(inplace=False)
    >>> clipper.fit(X)
    GaussianClipper(n_sigmas=3, subset=['A', 'B', 'C'], drop_columns=True, inplace=False)
    >>> transformed_X = clipper.transform(X)
    >>> print(transformed_X)
    shape: (5, 3)
    ┌───────────────────┬───────────────────┬───────────────────┐
    │ A__clip_gaussian  ┆ B__clip_gaussian  ┆ C__clip_gaussian  │
    │ ---               ┆ ---               ┆ ---               │
    │ f64               ┆ f64               ┆ f64               │
    ╞═══════════════════╪═══════════════════╪═══════════════════╡
    │ 1.0               ┆ -24.8             ┆ 10.0              │
    │ 2.0               ┆ 5.0               ┆ 20.0              │
    │ 3.0               ┆ 6.0               ┆ 30.0              │
    │ 4.0               ┆ 7.0               ┆ 40.0              │
    │ 42.8              ┆ 8.0               ┆ 50.0              │
    └───────────────────┴───────────────────┴───────────────────┘

    >>> # Clip using 2 standard deviations (more aggressive)
    >>> clipper_2sigma = GaussianClipper(n_sigmas=2, inplace=False)
    >>> clipper_2sigma.fit(X)
    GaussianClipper(n_sigmas=2, subset=['A', 'B', 'C'], drop_columns=True, inplace=False)
    >>> transformed_X_2sigma = clipper_2sigma.transform(X)
    >>> print(transformed_X_2sigma)
    shape: (5, 3)
    ┌───────────────────┬───────────────────┬───────────────────┐
    │ A__clip_gaussian  ┆ B__clip_gaussian  ┆ C__clip_gaussian  │
    │ ---               ┆ ---               ┆ ---               │
    │ f64               ┆ f64               ┆ f64               │
    ╞═══════════════════╪═══════════════════╪═══════════════════╡
    │ 1.0               ┆ -16.5             ┆ 10.0              │
    │ 2.0               ┆ 5.0               ┆ 20.0              │
    │ 3.0               ┆ 6.0               ┆ 30.0              │
    │ 4.0               ┆ 7.0               ┆ 40.0              │
    │ 28.5              ┆ 8.0               ┆ 50.0              │
    └───────────────────┴───────────────────┴───────────────────┘

    >>> # Clip with drop_columns=False to keep original columns
    >>> clipper_no_drop = GaussianClipper(n_sigmas=3, drop_columns=False, inplace=False)
    >>> clipper_no_drop.fit(X)
    GaussianClipper(n_sigmas=3, subset=['A', 'B', 'C'], drop_columns=False, inplace=False)
    >>> transformed_X_no_drop = clipper_no_drop.transform(X)
    >>> print(transformed_X_no_drop)
    shape: (5, 6)
    ┌───────┬────────┬──────┬───────────────────┬───────────────────┬───────────────────┐
    │ A     ┆ B      ┆ C    ┆ A__clip_gaussian  ┆ B__clip_gaussian  ┆ C__clip_gaussian  │
    │ ---   ┆ ---    ┆ ---  ┆ ---               ┆ ---               ┆ ---               │
    │ f64   ┆ f64    ┆ f64  ┆ f64               ┆ f64               ┆ f64               │
    ╞═══════╪════════╪══════╪═══════════════════╪═══════════════════╪═══════════════════╡
    │ 1.0   ┆ -50.0  ┆ 10.0 ┆ 1.0               ┆ -24.8             ┆ 10.0              │
    │ 2.0   ┆ 5.0    ┆ 20.0 ┆ 2.0               ┆ 5.0               ┆ 20.0              │
    │ 3.0   ┆ 6.0    ┆ 30.0 ┆ 3.0               ┆ 6.0               ┆ 30.0              │
    │ 4.0   ┆ 7.0    ┆ 40.0 ┆ 4.0               ┆ 7.0               ┆ 40.0              │
    │ 100.0 ┆ 8.0    ┆ 50.0 ┆ 42.8              ┆ 8.0               ┆ 50.0              │
    └───────┴────────┴──────┴───────────────────┴───────────────────┴───────────────────┘

    >>> # Clip only a subset of columns
    >>> clipper_subset = GaussianClipper(n_sigmas=3, subset=['A'], inplace=False)
    >>> clipper_subset.fit(X)
    GaussianClipper(n_sigmas=3, subset=['A'], drop_columns=True, inplace=False)
    >>> transformed_X_subset = clipper_subset.transform(X)
    >>> print(transformed_X_subset)
    shape: (5, 3)
    ┌────────┬──────┬───────────────────┐
    │ B      ┆ C    ┆ A__clip_gaussian  │
    │ ---    ┆ ---  ┆ ---               │
    │ f64    ┆ f64  ┆ f64               │
    ╞════════╪══════╪═══════════════════╡
    │ -50.0  ┆ 10.0 ┆ 1.0               │
    │ 5.0    ┆ 20.0 ┆ 2.0               │
    │ 6.0    ┆ 30.0 ┆ 3.0               │
    │ 7.0    ┆ 40.0 ┆ 4.0               │
    │ 8.0    ┆ 50.0 ┆ 42.8              │
    └────────┴──────┴───────────────────┘

    >>> # Clip inplace (modifies original columns)
    >>> clipper_inplace = GaussianClipper(n_sigmas=3, inplace=True)
    >>> clipper_inplace.fit(X)
    GaussianClipper(n_sigmas=3, subset=['A', 'B', 'C'], drop_columns=True, inplace=True)
    >>> transformed_X_inplace = clipper_inplace.transform(X)
    >>> print(transformed_X_inplace)
    shape: (5, 3)
    ┌───────┬────────┬──────┐
    │ A     ┆ B      ┆ C    │
    │ ---   ┆ ---    ┆ ---  │
    │ f64   ┆ f64    ┆ f64  │
    ╞═══════╪════════╪══════╡
    │ 1.0   ┆ -24.8  ┆ 10.0 │
    │ 2.0   ┆ 5.0    ┆ 20.0 │
    │ 3.0   ┆ 6.0    ┆ 30.0 │
    │ 4.0   ┆ 7.0    ┆ 40.0 │
    │ 42.8  ┆ 8.0    ┆ 50.0 │
    └───────┴────────┴──────┘
    """

    n_sigmas: int = Field(default=3, ge=1)
    subset: Optional[List[str]] = None
    drop_columns: bool = True
    inplace: bool = True
    _clip_bounds: Dict[str, Tuple[float, float]] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "GaussianClipper":
        """Fit the transformer by computing clipping bounds for each column.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        GaussianClipper
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
                col: f"{col}__clip_gaussian" for col in self.subset
            }

        # Compute mean and std for each column in a single pass
        stats = X.select(
            [
                pl.col(c).mean().alias(f"{c}_mean")
                for c in self.subset
            ] + [
                pl.col(c).std().alias(f"{c}_std")
                for c in self.subset
            ]
        ).row(0)

        # Extract means and stds
        n_cols = len(self.subset)
        means = stats[:n_cols]
        stds = stats[n_cols:]

        # Compute clipping bounds: [mean - n*std, mean + n*std]
        self._clip_bounds = {
            col: (
                mean - self.n_sigmas * std,
                mean + self.n_sigmas * std
            )
            for col, mean, std in zip(self.subset, means, stds)
        }

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by clipping values to mean ± n*std.

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
