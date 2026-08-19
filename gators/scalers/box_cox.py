import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class BoxCox(_BaseTransformer):
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
    lambdas : dict[str, int | float]
        Dictionary mapping column names to their lambda (power) parameters.
        Lambda values typically range from -2 to 2.
    inplace : bool, default=True
        If True, transform values in the original columns (keep original column names).
        If False, create new columns with suffix ``__boxcox``.
    drop_columns : bool, default=True
        If ``inplace=False``, whether to drop the original columns after transformation.
        Ignored when ``inplace=True``.

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

    lambdas: dict[str, int | float]
    inplace: bool = True
    drop_columns: bool = True
    _columns: list[str] = PrivateAttr(default_factory=list)
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "BoxCox":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit. All values in specified columns must be positive.
        y : pl.Series, default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        BoxCox
            The fitted transformer instance.
        """
        self._columns = list(self.lambdas.keys())
        if not self.inplace:
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
        if self.inplace:
            exprs = [
                (
                    pl.col(col).log().alias(col)
                    if lmbda == 0
                    else ((pl.col(col) ** lmbda - 1) / lmbda).alias(col)
                )
                for col, lmbda in self.lambdas.items()
            ]
            return X.with_columns(exprs)

        # Build all transformation expressions
        exprs = [
            (
                pl.col(col).log().alias(self._column_mapping[col])
                if lmbda == 0
                else ((pl.col(col) ** lmbda - 1) / lmbda).alias(self._column_mapping[col])
            )
            for col, lmbda in self.lambdas.items()
        ]
        X = X.with_columns(exprs)
        if self.drop_columns:
            return X.drop(self._columns)
        return X

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Reverse the Box-Cox transformation.

        Parameters
        ----------
        X : pl.DataFrame
            DataFrame with Box-Cox-transformed columns (output of ``transform``).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns restored to their original scale.
        """
        if self.inplace:
            exprs = []
            for col, lmbda in self.lambdas.items():
                if col not in X.columns:
                    continue
                if lmbda == 0:
                    expr = pl.col(col).exp().alias(col)
                else:
                    expr = ((pl.col(col) * lmbda + 1) ** (1.0 / lmbda)).alias(col)
                exprs.append(expr)
            return X.with_columns(exprs)
        reverse_map = {v: k for k, v in self._column_mapping.items()}
        exprs = []
        for new_col, orig_col in reverse_map.items():
            if new_col not in X.columns:
                continue
            lmbda = self.lambdas[orig_col]
            if lmbda == 0:
                expr = pl.col(new_col).exp().alias(orig_col)
            else:
                expr = ((pl.col(new_col) * lmbda + 1) ** (1.0 / lmbda)).alias(orig_col)
            exprs.append(expr)
        X = X.with_columns(exprs)
        if self.drop_columns:
            return X.drop([c for c in reverse_map if c in X.columns])
        return X
