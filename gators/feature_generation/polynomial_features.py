from itertools import combinations_with_replacement
from typing import List, Optional

import polars as pl
from pydantic import Field

from ..transformer._base_transformer import _BaseTransformer


class PolynomialFeatures(_BaseTransformer):
    """
    Generates polynomial and interaction features.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        Subset of columns to transform. If None, all columns
        except strings and booleans.
    degree : int, default=2
        The degree of the polynomial features.
    interaction_only : bool, default=False
        If True, only interaction features are produced.
    include_bias : bool, default=True
        If True, include a bias column (column of ones).

    Examples
    --------
    **Example 1: Degree 2 polynomial with bias term**

    >>> from gators.discretizers import PolynomialFeatures
    >>> import polars as pl
    >>> X = pl.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> transformer = PolynomialFeatures(degree=2, include_bias=True)
    >>> transformer.fit(X)
    >>> transformer.transform(X)
    shape: (2, 5)
    ┌─────┬─────┬─────┬─────┬─────┐─────┐
    │ A   │ B   │ A__A│ A__B│ B__B│ bias|
    ├─────┼─────┼─────┼─────┼─────┤─────┤
    │ 1   │ 3   │ 1   │ 3   │ 9   │ 1   │
    │ 2   │ 4   │ 4   │ 8   │ 16  │ 1   │
    └─────┴─────┴─────┴─────┴─────┴─────┘

    **Example 2: Polynomial on subset of columns**

    >>> transformer = PolynomialFeatures(subset=['A'], degree=2)
    >>> transformer.fit(X)
    >>> transformer.transform(X)
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ A   │ B   │ A__A│
    ├─────┼─────┼─────┤
    │ 1   │ 3   │ 1   │
    │ 2   │ 4   │ 4   │
    └─────┴─────┴─────┘

    **Example 3: Interaction features only**

    >>> transformer = PolynomialFeatures(degree=2, interaction_only=True)
    >>> transformer.fit(X)
    >>> transformer.transform(X)
    shape: (2, 4)
    ┌─────┬─────┬─────┐
    │ A   │ B   │ A__B│
    ├─────┼─────┼─────┼
    │ 1   │ 3   │ 3   │
    │ 2   │ 4   │ 8   │
    └─────┴─────┴─────┴
    """

    subset: Optional[List[str]] = None
    degree: int = Field(default=2, gt=1)
    interaction_only: bool = False
    include_bias: bool = False

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "PolynomialFeatures":
        """Fit the transformer by identifying columns to transform.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        PolynomialFeatures
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype not in [pl.String, pl.Boolean]
            ]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by extracting specified components.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        transformations = []

        if self.include_bias:
            transformations.append(pl.lit(1).alias("bias"))

        if self.subset is None:
            return X

        for i in range(2, self.degree + 1):
            for combination in combinations_with_replacement(self.subset, i):
                if self.interaction_only and len(set(combination)) != i:
                    continue
                new_col_name = "__".join(combination)
                # Multiply columns directly to preserve dtype
                new_col_expr = pl.col(combination[0])
                for col in combination[1:]:
                    new_col_expr = new_col_expr * pl.col(col)
                transformations.append(new_col_expr.alias(new_col_name))

        return X.with_columns(transformations)
