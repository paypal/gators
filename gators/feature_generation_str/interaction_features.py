from itertools import combinations
from typing import List, Optional

import polars as pl
from pydantic import Field

from ..transformer._base_transformer import _BaseTransformer


class InteractionFeatures(_BaseTransformer):
    """
    Generates interaction features for categorical variables.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of columns to consider for interaction.
    degree : conint(gt=1), default=2
        Degree of interaction terms.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_generation_str import InteractionFeatures

    >>> # Sample data
    >>> X =pl.DataFrame({
    ...     'A': ['cat', 'dog', 'cat', 'dog', 'cat'],
    ...     'B': ['x', 'x', 'y', 'y', 'x'],
    ...     'C': ['red', 'blue', 'green', 'blue', 'red']
    ... })

    >>> # Interaction with default parameters (degree=2)
    >>> interaction_features = InteractionFeatures()
    >>> interaction_features.fit(X)
    >>> transformed_X =interaction_features.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌─────┬─────┬──────┬─────────────┐
    │ A   │ B   │ C    │ A__B        │
    │ --- │ --- │ ---  │ ---         │
    │ str │ str │ str  │ str         │
    ├─────┼─────┼──────┼─────────────┤
    │ cat │ x   │ red  │ cat_x       │
    │ dog │ x   │ blue │ dog_x       │
    │ cat │ y   │ green│ cat_y       │
    │ dog │ y   │ blue │ dog_y       │
    │ cat │ x   │ red  │ cat_x       │
    └─────┴─────┴──────┴─────────────┘

    >>> # Interaction with degree=3
    >>> interaction_features = InteractionFeatures(degree=3)
    >>> interaction_features.fit(X)
    >>> transformed_X =interaction_features.transform(X)
    >>> print(transformed_X)
    shape: (5, 8)
    ┌─────┬─────┬──────┬─────────────┬──────────────┬───────────────┬───────────────┐
    │ A   │ B   │ C    │ A__B        │ A__C         │ B__C          │ A__B__C       │
    │ --- │ --- │ ---  │ ---         │ ---          │ ---           │ ---           │
    │ str │ str │ str  │ str         │ str          │ str           │ str           │
    ├─────┼─────┼──────┼─────────────┼──────────────┼───────────────┼───────────────┤
    │ cat │ x   │ red  │ cat_x       │ cat_red      │ x_red         │ cat_x_red     │
    │ dog │ x   │ blue │ dog_x       │ dog_blue     │ x_blue        │ dog_x_blue    │
    │ cat │ y   │ green│ cat_y       │ cat_green    │ y_green       │ cat_y_green   │
    │ dog │ y   │ blue │ dog_y       │ dog_blue     │ y_blue        │ dog_y_blue    │
    │ cat │ x   │ red  │ cat_x       │ cat_red      │ x_red         │ cat_x_red     │
    └─────┴─────┴──────┴─────────────┴──────────────┴───────────────┴───────────────┘

    >>> # Interaction with columns=None
    >>> interaction_features = InteractionFeatures(subset=['A', 'B'])
    >>> interaction_features.fit(X)
    >>> transformed_X =interaction_features.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌─────┬─────┬──────┬─────────────┐
    │ A   │ B   │ C    │ A__B        │
    │ --- │ --- │ ---  │ ---         │
    │ str │ str │ str  │ str         │
    ├─────┼─────┼──────┼─────────────┤
    │ cat │ x   │ red  │ cat_x       │
    │ dog │ x   │ blue │ dog_x       │
    │ cat │ y   │ green│ cat_y       │
    │ dog │ y   │ blue │ dog_y       │
    │ cat │ x   │ red  │ cat_x       │
    └─────┴─────┴──────┴─────────────┘

    """

    subset: Optional[List[str]] = None
    degree: int = Field(default=2, gt=1)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "InteractionFeatures":
        """Fit the transformer by identifying categorical columns if not specified.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        InteractionFeatures
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.String, pl.Boolean, pl.Categorical]
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
        if self.subset is None:
            return X

        transformations = []
        for i in range(2, self.degree + 1):
            for combination in combinations(self.subset, i):
                new_col_name = "__".join(combination)
                new_col_expr = X[combination[0]]
                for col in combination[1:]:
                    new_col_expr = new_col_expr + "_" + X[col]
                transformations.append(new_col_expr.alias(new_col_name))

        return X.with_columns(transformations)
