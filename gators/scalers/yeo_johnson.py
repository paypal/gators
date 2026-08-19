import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class YeoJohnson(_BaseTransformer):
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
    lambdas : dict[str, int | float]
        Dictionary mapping column names to their lambda (power) parameters.
        Lambda values typically range from -2 to 2.
    inplace : bool, default=True
        If True, transform values in the original columns (keep original column names).
        If False, create new columns with suffix ``__yeojonhson``.
    drop_columns : bool, default=True
        If ``inplace=False``, whether to drop the original columns after transformation.
        Ignored when ``inplace=True``.

    Examples
    --------
    Create an instance of the YeoJohnson class:

    >>> import polars as pl
    >>> from gators.scalers import YeoJohnson
    >>> transformer = YeoJohnson(lambdas={"sales": 0.5, "profit": 0.0})

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

    lambdas: dict[str, int | float]
    _scale: dict[str, float] = PrivateAttr(default_factory=dict)
    inplace: bool = True
    drop_columns: bool = True
    _columns: list[str] = PrivateAttr(default_factory=list)
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "YeoJohnson":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : pl.Series, default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        YeoJohnson
            The fitted transformer instance.
        """
        self._columns = list(self.lambdas.keys())
        if not self.inplace:
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
        exprs = []
        for col, lmbda in self.lambdas.items():
            new = self._column_mapping.get(col, col)  # inplace uses original name
            if lmbda == 0:
                expr = (
                    pl.when(pl.col(col) >= 0)
                    .then(pl.col(col).log1p())
                    .otherwise(-((-pl.col(col) + 1) ** (2 - lmbda) - 1) / (2 - lmbda))
                    .alias(new)
                )
            elif lmbda == 2:
                expr = (
                    pl.when(pl.col(col) >= 0)
                    .then(((pl.col(col) + 1) ** lmbda - 1) / lmbda)
                    .otherwise(-(-pl.col(col)).log1p())
                    .alias(new)
                )
            else:
                expr = (
                    pl.when(pl.col(col) >= 0)
                    .then(((pl.col(col) + 1) ** lmbda - 1) / lmbda)
                    .otherwise(-((-pl.col(col) + 1) ** (2 - lmbda) - 1) / (2 - lmbda))
                    .alias(new)
                )
            exprs.append(expr)

        X = X.with_columns(exprs)

        if not self.inplace and self.drop_columns:
            return X.drop(self._columns)
        return X

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Reverse the Yeo-Johnson transformation.

        Parameters
        ----------
        X : pl.DataFrame
            DataFrame with Yeo-Johnson-transformed columns (output of ``transform``).

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
                y = pl.col(col)
                if lmbda == 0:
                    pos_branch = y.exp() - 1
                    neg_branch = 1 - (1 - 2 * y).sqrt()
                elif lmbda == 2:
                    pos_branch = (2 * y + 1).sqrt() - 1
                    neg_branch = 1 - (-y).exp()
                else:
                    pos_branch = (y * lmbda + 1) ** (1.0 / lmbda) - 1
                    neg_branch = 1 - (1 - y * (2 - lmbda)) ** (1.0 / (2 - lmbda))
                exprs.append(pl.when(y >= 0).then(pos_branch).otherwise(neg_branch).alias(col))
            return X.with_columns(exprs)
        reverse_map = {v: k for k, v in self._column_mapping.items()}
        exprs = []
        for new_col, orig_col in reverse_map.items():
            if new_col not in X.columns:
                continue
            lmbda = self.lambdas[orig_col]
            y = pl.col(new_col)
            if lmbda == 0:
                pos_branch = y.exp() - 1
                neg_branch = 1 - (1 - 2 * y).sqrt()
            elif lmbda == 2:
                pos_branch = (2 * y + 1).sqrt() - 1
                neg_branch = 1 - (-y).exp()
            else:
                pos_branch = (y * lmbda + 1) ** (1.0 / lmbda) - 1
                neg_branch = 1 - (1 - y * (2 - lmbda)) ** (1.0 / (2 - lmbda))
            exprs.append(pl.when(y >= 0).then(pos_branch).otherwise(neg_branch).alias(orig_col))
        X = X.with_columns(exprs)
        if self.drop_columns:
            return X.drop([c for c in reverse_map if c in X.columns])
        return X
