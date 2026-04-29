from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class Length(_BaseTransformer):
    """
    Generates features based on the length of the variables.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of columns to calculate length for.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.discretizers import Length

    >>> # Sample data
    >>> X =pl.DataFrame({
    ...     'A': ['cat', 'dog', 'cat', 'dog', 'cat'],
    ...     'B': ['yes', 'no', 'yes', 'no', 'yes'],
    ...     'C': ['quick', 'brown', 'fox', 'jumps', 'over']
    ... })

    >>> # Calculate lengths with default parameters
    >>> encoder = Length()
    >>> encoder.fit(X)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 6)
    ┌─────┬──────┬───────┬──────────┬──────────┬───────────┐
    │ A   │ B    │ C     │ A__length│ B__length│ C__length │
    │ --- │ ---  │ ---   │ ---      │ ---      │ ---       │
    │ str │ str  │ str   │ i64      │ i64      │ i64       │
    ├─────┼──────┼───────┼──────────┼──────────┼───────────┤
    │ cat │ yes  │ quick │ 3        │ 3        │ 5         │
    │ dog │ no   │ brown │ 3        │ 2        │ 5         │
    │ cat │ yes  │ fox   │ 3        │ 3        │ 3         │
    │ dog │ no   │ jumps │ 3        │ 2        │ 5         │
    │ cat │ yes  │ over  │ 3        │ 3        │ 4         │
    └─────┴──────┴───────┴──────────┴──────────┴───────────┘

    >>> # Calculate lengths with columns as a subset
    >>> encoder = Length(subset=['B'])
    >>> encoder.fit(X)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌─────┬──────┬───────┬──────────┐
    │ A   │ B    │ C     │ B__length│
    │ --- │ ---  │ ---   │ ---      │
    │ str │ str  │ str   │ i64      │
    ├─────┼──────┼───────┼──────────┤
    │ cat │ yes  │ quick │ 3        │
    │ dog │ no   │ brown │ 2        │
    │ cat │ yes  │ fox   │ 3        │
    │ dog │ no   │ jumps │ 2        │
    │ cat │ yes  │ over  │ 3        │
    └─────┴──────┴───────┴──────────┘
    """

    subset: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = {}

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "Length":
        """Fit the transformer by identifying categorical columns and generating column mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        Length
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype in [pl.String, pl.Boolean, pl.Enum]
            ]
        self._column_mapping = {col: f"{col}__length" for col in self.subset}
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

        transformations = [
            pl.col(col).str.len_chars().cast(pl.Int64).alias(self._column_mapping[col])
            for col in self.subset
        ]
        return X.with_columns(transformations)
