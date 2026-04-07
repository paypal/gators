"""Custom clipper for capping values with user-defined bounds."""

from typing import Dict, Optional

import polars as pl
import polars.selectors as cs
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from sklearn.base import BaseEstimator, TransformerMixin


class CustomClipper(BaseModel, BaseEstimator, TransformerMixin):
    """Clip column values using custom lower and upper bounds.

    This transformer allows you to specify custom clipping bounds for each column
    independently. You can specify only lower bounds, only upper bounds, or both
    for different columns. Columns not specified in either dictionary are left unchanged.

    Parameters
    ----------
    lower_bounds : dict of str to float, optional
        Dictionary mapping column names to their lower bounds.
        Values below the lower bound will be clipped to the lower bound.
        Default is None (no lower bounds).
    upper_bounds : dict of str to float, optional
        Dictionary mapping column names to their upper bounds.
        Values above the upper bound will be clipped to the upper bound.
        Default is None (no upper bounds).
    inplace : bool, default=True
        If True, clip values in the original columns.
        If False, create new columns with the suffix '__clip_custom'.
    drop_columns : bool, default=True
        If True and inplace=False, drop the original columns after clipping.
        If False and inplace=False, keep both original and clipped columns.
        Ignored if inplace=True.

    Attributes
    ----------
    _columns : list of str
        List of columns that will be clipped (union of lower_bounds and upper_bounds keys).
    _bounds_map : dict of str to tuple
        Mapping of column names to (lower_bound, upper_bound) tuples.
        None values indicate no bound on that side.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.clippers import CustomClipper

    Clip with both lower and upper bounds:

    >>> X = pl.DataFrame({
    ...     "age": [-5, 25, 150],
    ...     "salary": [-1000, 50000, 2000000]
    ... })
    >>> clipper = CustomClipper(
    ...     lower_bounds={"age": 0, "salary": 0},
    ...     upper_bounds={"age": 120, "salary": 1000000}
    ... )
    >>> clipper.fit_transform(X)
    shape: (3, 2)
    ┌─────┬─────────┐
    │ age ┆ salary  │
    │ --- ┆ ---     │
    │ f64 ┆ f64     │
    ╞═════╪═════════╡
    │ 0.0 ┆ 0.0     │
    │ 25.0┆ 50000.0 │
    │ 120.0┆1000000.0│
    └─────┴─────────┘

    Clip with only lower bounds:

    >>> clipper = CustomClipper(lower_bounds={"age": 0})
    >>> clipper.fit_transform(X)
    shape: (3, 2)
    ┌─────┬─────────┐
    │ age ┆ salary  │
    │ --- ┆ ---     │
    │ f64 ┆ f64     │
    ╞═════╪═════════╡
    │ 0.0 ┆ -1000.0 │
    │ 25.0┆ 50000.0 │
    │ 150.0┆2000000.0│
    └─────┴─────────┘

    Create new columns instead of modifying in place:

    >>> clipper = CustomClipper(
    ...     lower_bounds={"age": 0},
    ...     upper_bounds={"age": 120},
    ...     inplace=False
    ... )
    >>> clipper.fit_transform(X)
    shape: (3, 2)
    ┌──────────────────┬─────────┐
    │ age__clip_custom ┆ salary  │
    │ ---              ┆ ---     │
    │ f64              ┆ f64     │
    ╞══════════════════╪═════════╡
    │ 0.0              ┆ -1000.0 │
    │ 25.0             ┆ 50000.0 │
    │ 120.0            ┆2000000.0│
    └──────────────────┴─────────┘

    Notes
    -----
    - Non-numeric columns are automatically ignored.
    - Columns not specified in either bounds dictionary are left unchanged.
    - You can specify bounds for only some columns while leaving others untouched.
    - If a column appears in both dictionaries, both bounds are applied.

    See Also
    --------
    GaussianClipper : Clip values based on mean and standard deviation.
    QuantileClipper : Clip values based on quantiles.
    MADClipper : Clip values based on median absolute deviation.
    IQRClipper : Clip values based on interquartile range.
    """

    lower_bounds: Optional[Dict[str, float]] = Field(default=None)
    upper_bounds: Optional[Dict[str, float]] = Field(default=None)
    inplace: bool = Field(default=True)
    drop_columns: bool = Field(default=True)
    _columns: list = PrivateAttr(default_factory=list)
    _bounds_map: Dict[str, tuple] = PrivateAttr(default_factory=dict)

    @field_validator("lower_bounds", "upper_bounds")
    @classmethod
    def validate_bounds(cls, v):
        """Validate that bounds are dictionaries with string keys and numeric values."""
        if v is not None:
            if not isinstance(v, dict):
                raise TypeError("Bounds must be a dictionary")
            if not all(isinstance(k, str) for k in v.keys()):
                raise TypeError("All keys in bounds must be strings (column names)")
            if not all(isinstance(val, (int, float)) for val in v.values()):
                raise TypeError("All values in bounds must be numeric")
        return v

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CustomClipper":
        """Fit the clipper by identifying columns to clip.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, optional
            Target values (ignored, present for sklearn compatibility).

        Returns
        -------
        self : CustomClipper
            Fitted clipper.

        Raises
        ------
        ValueError
            If no bounds are specified or if specified columns don't exist in X.
        """
        if self.lower_bounds is None and self.upper_bounds is None:
            raise ValueError("At least one of lower_bounds or upper_bounds must be specified")

        # Get all columns that have bounds specified
        lower_cols = set(self.lower_bounds.keys()) if self.lower_bounds else set()
        upper_cols = set(self.upper_bounds.keys()) if self.upper_bounds else set()
        all_bound_cols = lower_cols | upper_cols

        # Check that all specified columns exist in X
        missing_cols = all_bound_cols - set(X.columns)
        if missing_cols:
            raise ValueError(f"Columns specified in bounds not found in DataFrame: {missing_cols}")

        # Filter to only numeric columns
        numeric_cols = set(X.select(cs.numeric()).columns)
        self._columns = sorted(all_bound_cols & numeric_cols)

        # Create bounds map: column -> (lower_bound, upper_bound)
        self._bounds_map = {}
        for col in self._columns:
            lower = self.lower_bounds.get(col) if self.lower_bounds else None
            upper = self.upper_bounds.get(col) if self.upper_bounds else None
            self._bounds_map[col] = (lower, upper)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Clip values using the custom bounds.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to clip.

        Returns
        -------
        pl.DataFrame
            DataFrame with clipped values.
        """
        X_transformed = X.clone()

        for col in self._columns:
            lower, upper = self._bounds_map[col]
            clipped_col = X_transformed[col].clip(lower, upper)

            if self.inplace:
                X_transformed = X_transformed.with_columns(clipped_col.alias(col))
            else:
                new_col_name = f"{col}__clip_custom"
                X_transformed = X_transformed.with_columns(clipped_col.alias(new_col_name))

        # Drop original columns if not inplace and drop_columns is True
        if not self.inplace and self.drop_columns:
            X_transformed = X_transformed.drop(self._columns)

        return X_transformed

    def fit_transform(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> pl.DataFrame:
        """Fit the clipper and transform the data.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, optional
            Target values (ignored, present for sklearn compatibility).

        Returns
        -------
        pl.DataFrame
            DataFrame with clipped values.
        """
        return self.fit(X, y).transform(X)
