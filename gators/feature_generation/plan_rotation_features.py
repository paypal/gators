# License: Apache-2.0
from math import cos
from math import pi as PI
from math import sin
from typing import List, Optional

import polars as pl
from pydantic import BaseModel, field_validator, model_validator
from sklearn.base import BaseEstimator, TransformerMixin


class PlanRotationFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """Create new columns based on the plan rotation mapping.

    The data should be composed of numerical columns only.
    Use `gators.encoders` to replace the categorical columns by
    numerical ones before using `PlanRotationFeatures`.

    Parameters
    ----------
    subset : List[List[str]]
        List of pair-wise columns.
    angles : List[float]
        List of rotation angles.

    Examples
    --------
    **Basic usage with plan rotation**

    Imports and initialization:

    >>> from gators.feature_generation import PlanRotationFeatures
    >>> obj = PlanRotationFeatures(
    ... subset=[['X', 'Y'], ['X', 'Z']] , angles=[45.0, 60.0])

    The `fit`, `transform`, and `fit_transform` methods accept `polars` dataframes:

    >>> import polars as pl
    >>> X = pl.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [100.0, 125.0]})

    The result is a transformed polars dataframe.

    >>> obj.fit_transform(X)
    shape: (2, 9)
    ┌───────┬───────┬───────┬────────────┬───┬────────────┬────────────┬────────────┐
    │ X     ┆ Y     ┆ Z     ┆ XY_x_45.0… ┆ … ┆ XZ_y_45.0… ┆ XZ_x_60.0… ┆ XZ_y_60.0… │
    │ ---   ┆ ---   ┆ ---   ┆ ---        ┆   ┆ ---        ┆ ---        ┆ ---        │
    │ f64   ┆ f64   ┆ f64   ┆ f64        ┆   ┆ f64        ┆ f64        ┆ f64        │
    ╞═══════╪═══════╪═══════╪════════════╪═══╪════════════╪════════════╪════════════╡
    │ 200.0 ┆ 140.0 ┆ 100.0 ┆ 42.426407  ┆ … ┆ 212.132034 ┆ 13.397460  ┆ 223.205081 │
    │ 210.0 ┆ 160.0 ┆ 125.0 ┆ 35.355339  ┆ … ┆ 236.880772 ┆ -3.253175  ┆ 244.365335 │
    └───────┴───────┴───────┴────────────┴───┴────────────┴────────────┴────────────┘

    """

    columns: List[List[str]]
    angles: List[float]
    column_names: List[str] = []
    flatten_columns: List[str] = []

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def compute_column_names(self):
        """Compute column names after initialization."""
        column_names = [
            [
                f"{x}{y}_x{int(t) if t == int(t) else t}",
                f"{x}{y}_y{int(t) if t == int(t) else t}",
            ]
            for (x, y) in self.columns
            for t in self.angles
        ]
        self.column_names = [c for cols in column_names for c in cols]
        return self

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "PlanRotationFeatures":
        """Fit the transformer by identifying columns to flatten.

        Parameters
        ----------
        X : pl.DataFrame
            Input dataframe.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        PlanRotationFeatures
            Fitted transformer instance.
        """
        self.flatten_columns = [c for cols in self.columns for c in cols]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : pl.DataFrame.
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Transformed dataframe.
        """
        new_columns = []

        for x, y in zip(self.flatten_columns[::2], self.flatten_columns[1::2]):
            for theta in self.angles:
                cos_theta = cos(theta * PI / 180)
                sin_theta = sin(theta * PI / 180)
                angle_int = int(theta) if theta == int(theta) else round(theta, 2)

                x_rotated = (pl.col(x) * cos_theta - pl.col(y) * sin_theta).alias(
                    f"{x}{y}_x{angle_int}"
                )
                y_rotated = (pl.col(x) * sin_theta + pl.col(y) * cos_theta).alias(
                    f"{x}{y}_y{angle_int}"
                )
                new_columns.extend([x_rotated, y_rotated])

        return X.with_columns(new_columns)
