import polars as pl
from pydantic import Field, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class ExtractSubstring(_BaseTransformer):
    subset: list[str]
    start: int = Field(ge=0)
    end: int | None = Field(default=None, ge=1)
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "ExtractSubstring":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        ExtractSubstring
            Fitted transformer instance.
        """
        suffix = f"start{self.start}_endNone" if self.end is None else f"start{self.start}_end{self.end}"
        self._column_mapping = {col: [f"{col}__{suffix}"] for col in self.subset}
        self._output_dtypes = {
            new: pl.String for names in self._column_mapping.values() for new in names
        }
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
