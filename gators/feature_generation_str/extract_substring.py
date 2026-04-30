from typing import List, Optional

import polars as pl
from pydantic import Field

from ..transformer._base_transformer import _BaseTransformer


class ExtractSubstring(_BaseTransformer):
    subset: List[str]
    start: int = Field(ge=0)
    end: Optional[int] = Field(default=None, ge=1)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "ExtractSubstring":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        ExtractSubstring
            Fitted transformer instance.
        """
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
        for col in self.subset:
            # Calculate length from start and end
            if self.end is None:
                length = None
                col_name = f"{col}__start{self.start}_endNone"
            else:
                length = self.end - self.start
                col_name = f"{col}__start{self.start}_end{self.end}"

            extract_col = pl.col(col).str.slice(self.start, length).alias(col_name)
            transformations.append(extract_col)
        X = X.with_columns(transformations)

        return X
