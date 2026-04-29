from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class Upper(_BaseTransformer):
    """
    Converts string and Boolean columns to uppercase.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of columns to convert to uppercase.
    drop_columns : bool, default=True
        Whether to drop original columns after transformation.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.discretizers import Upper

    >>> # Sample data
    >>> X =pl.DataFrame({
    ...     'A': ['cat', 'dog', 'cat', 'dog', 'cat'],
    ...     'B': ['yes', 'no', 'yes', 'no', 'yes'],
    ...     'C': ['quick', 'brown', 'fox', 'jumps', 'over']
    ... })

    >>> # Transform to uppercase with default parameters (drop_columns=True)
    >>> encoder = Upper()
    >>> encoder.fit(X)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 3)
    ┌──────────┬──────────┬──────────┐
    │ A__upper │ B__upper │ C__upper │
    │ ---      │ ---      │ ---      │
    │ str      │ str      │ str      │
    ├──────────┼──────────┼──────────┤
    │ CAT      │ YES      │ QUICK    │
    │ DOG      │ NO       │ BROWN    │
    │ CAT      │ YES      │ FOX      │
    │ DOG      │ NO       │ JUMPS    │
    │ CAT      │ YES      │ OVER     │
    └──────────┴──────────┴──────────┘

    >>> # Transform to uppercase with drop_columns=False
    >>> encoder = Upper(drop_columns=False)
    >>> encoder.fit(X)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 6)
    ┌─────┬──────┬───────┬──────────┬──────────┬──────────┐
    │ A   │ B    │ C     │ A__upper │ B__upper │ C__upper │
    │ --- │ ---  │ ---   │ ---      │ ---      │ ---      │
    │ str │ str  │ str   │ str      │ str      │ str      │
    ├─────┼──────┼───────┼──────────┼──────────┼──────────┤
    │ cat │ yes  │ quick │ CAT      │ YES      │ QUICK    │
    │ dog │ no   │ brown │ DOG      │ NO       │ BROWN    │
    │ cat │ yes  │ fox   │ CAT      │ YES      │ FOX      │
    │ dog │ no   │ jumps │ DOG      │ NO       │ JUMPS    │
    │ cat │ yes  │ over  │ CAT      │ YES      │ OVER     │
    └─────┴──────┴───────┴──────────┴──────────┴──────────┘

    >>> # Transform to uppercase with columns as a subset
    >>> encoder = Upper(subset=['B'])
    >>> encoder.fit(X)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌─────┬──────┬───────┬──────────┐
    │ A   │ B    │ C     │ B__upper │
    │ --- │ ---  │ ---   │ ---      │
    │ str │ str  │ str   │ str      │
    ├─────┼──────┼───────┼──────────┤
    │ cat │ yes  │ quick │ YES      │
    │ dog │ no   │ brown │ NO       │
    │ cat │ yes  │ fox   │ YES      │
    │ dog │ no   │ jumps │ NO       │
    │ cat │ yes  │ over  │ YES      │
    └─────┴──────┴───────┴──────────┘
    """

    subset: Optional[List[str]] = None
    drop_columns: bool = True
    inplace: bool = True
    _column_mapping: Dict[str, str] = {}

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "Upper":
        """Fit the transformer by identifying categorical columns and generating column mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        Upper
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in X.schema.items()
                if dtype in [pl.String, pl.Boolean, pl.Enum]
            ]
        if not self.inplace:
            self._column_mapping = {col: f"{col}__upper" for col in self.subset}
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

        if self.inplace:
            transformations = [
                pl.col(col).cast(pl.String).str.to_uppercase() for col in self.subset
            ]
            return X.with_columns(transformations)

        transformations = [
            pl.col(col).cast(pl.String).str.to_uppercase().alias(new_col)
            for col, new_col in self._column_mapping.items()
        ]
        X = X.with_columns(transformations)
        if self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X
