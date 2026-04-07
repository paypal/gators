from typing import Dict, List, Optional, Union

import polars as pl
from pydantic import BaseModel, PrivateAttr, field_validator
from sklearn.base import BaseEstimator, TransformerMixin


class BoxCox(BaseModel, BaseEstimator, TransformerMixin):
    """
    Applies the Box-Cox power transformation to numeric features.

    The Box-Cox transformation is a family of power transformations that
    can help normalize skewed data and stabilize variance. Unlike Yeo-Johnson,
    Box-Cox requires all values to be strictly positive (x > 0).

    For each feature x with parameter lambda:

    - If lambda != 0: (x^lambda - 1) / lambda
    - If lambda == 0: log(x)

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
    Create an instance of the BoxCox class:

    >>> import polars as pl
    >>> from gators.scalers import BoxCox
    >>> transformer = BoxCox(lambdas={"sales": 0.5, "price": 0.0})

    Fit the transformer:

    >>> X = pl.DataFrame({"sales": [10, 20, 30, 40],
    ...                    "price": [5, 15, 25, 35]})
    >>> transformer.fit(X)

    Transform the DataFrame:

    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌─────────────────┬─────────────────┐
    │ sales__boxcox   ┆ price__boxcox   │
    │ ---             ┆ ---             │
    │ f64             ┆ f64             │
    ├─────────────────┼─────────────────┤
    │ ...             ┆ ...             │
    └─────────────────┴─────────────────┘

    Notes
    -----
    All input values must be strictly positive (> 0). Negative or zero values
    will produce invalid results. Use Yeo-Johnson transformation if you need
    to handle zero or negative values.
    """

    lambdas: Dict[str, Union[int, float]]
    drop_columns: bool = True
    _columns: List[str] = PrivateAttr()
    _column_mapping: Dict[str, str] = PrivateAttr()

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "BoxCox":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit. All values in specified columns must be positive.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        BoxCox
            The fitted transformer instance.
        """
        self._columns = list(self.lambdas.keys())
        self._column_mapping = {col: f"{col}__boxcox" for col in self._columns}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying Box-Cox transformation.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform. All values in specified columns
            must be strictly positive (> 0).

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with power-transformed columns.
        """
        for col, lmbda in self.lambdas.items():
            new = self._column_mapping[col]
            if lmbda == 0:
                # log(x)
                X = X.with_columns(pl.col(col).log().alias(new))
            else:
                # (x^lambda - 1) / lambda
                X = X.with_columns(((pl.col(col) ** lmbda - 1) / lmbda).alias(new))

        if self.drop_columns:
            return X.drop(self._columns)
        return X
