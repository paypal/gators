from typing import Dict, List, Optional, Union

import polars as pl
from pydantic import BaseModel, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin


class YeoJonhson(BaseModel, BaseEstimator, TransformerMixin):
    """
    Applies the Yeo-Johnson power transformation to numeric features.

    The Yeo-Johnson transformation is a family of power transformations that
    can be applied to both positive and negative values (unlike Box-Cox which
    requires positive values). It can help normalize skewed data and stabilize
    variance.

    For each feature x with parameter lambda:

    - If x >= 0 and lambda != 0: ((x + 1)^lambda - 1) / lambda
    - If x >= 0 and lambda == 0: log(x + 1)
    - If x < 0 and lambda != 2: -((-x + 1)^(2-lambda) - 1) / (2 - lambda)
    - If x < 0 and lambda == 2: -log(-x + 1)

    Parameters
    ----------
    lambdas : Dict[str, Union[int, float]]
        Dictionary mapping column names to their lambda (power) parameters.
        Lambda values typically range from -2 to 2.
    drop_columns : bool, default=True
        If True, drop the original columns after transformation.
        If False, keep both original and transformed columns.

    Examples
    --------
    Create an instance of the YeoJonhson class:

    >>> import polars as pl
    >>> from gators.scalers import YeoJonhson
    >>> transformer = YeoJonhson(lambdas={"sales": 0.5, "profit": 0.0})

    Fit the transformer:

    >>> X = pl.DataFrame({"sales": [10, 20, 30, 40],
    ...                    "profit": [-5, 5, 15, 25]})
    >>> transformer.fit(X)

    Transform the DataFrame:

    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌───────────────────┬────────────────────┐
    │ sales__yeojonhson ┆ profit__yeojonhson │
    │ ---               ┆ ---                │
    │ f64               ┆ f64                │
    ├───────────────────┼────────────────────┤
    │ ...               ┆ ...                │
    └───────────────────┴────────────────────┘

    """

    lambdas: Dict[str, Union[int, float]]
    _scale: Dict[str, float] = PrivateAttr()
    drop_columns: bool = True
    _columns: List[str] = PrivateAttr()
    _column_mapping: Dict[str, str] = PrivateAttr()

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "YeoJonhson":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        YeoJonhson
            The fitted transformer instance.
        """
        self._columns = list(self.lambdas.keys())
        self._column_mapping = {col: f"{col}__yeojonhson" for col in self._columns}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying Yeo-Johnson transformation.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with power-transformed columns.
        """
        for col, lmbda in self.lambdas.items():
            new = self._column_mapping[col]
            if lmbda == 0:
                X = X.with_columns(
                    pl.when(pl.col(col) < 0)
                    .then(pl.col(col).log1p())
                    .otherwise(-((-pl.col(col) + 1) ** (2 - lmbda) - 1) / (2 - lmbda))
                    .alias(new)
                )
            elif lmbda == 2:
                X = X.with_columns(
                    pl.when(pl.col(col) < 0)
                    .then(((pl.col(col) + 1) ** lmbda - 1) / lmbda)
                    .otherwise(-pl.col(col).log1p())
                    .alias(new)
                )
            else:
                X = X.with_columns(
                    pl.when(pl.col(col) < 0)
                    .then(((pl.col(col) + 1) ** lmbda - 1) / lmbda)
                    .otherwise(-((-pl.col(col) + 1) ** (2 - lmbda) - 1) / (2 - lmbda))
                    .alias(new)
                )

        if self.drop_columns:
            return X.drop(self._columns)
        return X
