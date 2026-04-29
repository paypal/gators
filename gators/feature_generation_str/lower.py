from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class Lower(_BaseTransformer):
    """
    Converts string and Boolean columns to lowercase.

    Examples
    --------
    Create an instance of the Lower class:

    >>> from gators.feature_feneration_str import Lower
    >>> lower = Lower(subset=["col1"], drop_columns=False)

    Fit the transformer:

    >>> lower.fit(X)

    Transform the dataframe:

    >>> X = pl.DataFrame({"col1": ["Hello", "WORLD"], "col2": [True, False]})
    >>> transformed_X = lower.transform(X)
    >>> print(transformed_X)
    shape: (2, 3)
    ┌───────┬───────┬─────────────┐
    │ col1  │ col2  │ col1__lower │
    ├───────┼───────┼─────────────┤
    │ HeLLo │ True  │ hello       │
    │ WORLD │ False │ world       │
    └───────┴───────┴─────────────┘

    If drop_columns is True, the original columns are dropped:

    >>> lower.drop_columns = True
    >>> transformed_X = lower.transform(X)
    >>> print(transformed_X)
    shape: (2, 1)
    ┌─────────────┐
    │ col1__lower │
    ├─────────────┤
    │ hello       │
    │ world       │
    └─────────────┘

    """

    subset: Optional[List[str]] = None
    drop_columns: bool = True
    inplace: bool = True
    _column_mapping: Dict[str, str] = {}

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "Lower":
        """Fit the transformer by identifying categorical columns and generating column mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        Lower
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in X.schema.items()
                if dtype in [pl.String, pl.Boolean, pl.Categorical]
            ]
        if not self.inplace:
            self._column_mapping = {col: f"{col}__lower" for col in self.subset}
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
                pl.col(col).cast(pl.String).str.to_lowercase() for col in self.subset
            ]
            return X.with_columns(transformations)

        transformations = [
            pl.col(col).cast(pl.String).str.to_lowercase().alias(new_col)
            for col, new_col in self._column_mapping.items()
        ]
        X = X.with_columns(transformations)
        if self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X
