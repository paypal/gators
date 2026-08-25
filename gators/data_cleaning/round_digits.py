from typing import Annotated

import polars as pl
from pydantic import Field, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class RoundDigits(_BaseTransformer):
    """Round selected numeric columns to a fixed number of decimal places.

    Parameters
    ----------
    n_digits : int
        Number of decimal places to keep. Must be >= 0.
    subset : list[str], default=None
        Columns to round. When ``None``, all numeric columns in the
        DataFrame are rounded automatically.
    inplace : bool, default=True
        If ``True``, the original columns are replaced in-place.
        If ``False``, new columns named ``{col}__round_{n_digits}digits`` are
        added alongside the originals.
    drop_columns : bool, default=True
        Relevant only when ``inplace=False``. If ``True``, the original
        columns are dropped after the new rounded columns are added.
        Ignored when ``inplace=True``.
    """

    n_digits: Annotated[int, Field(ge=0)]
    subset: list[str] | None = None
    inplace: bool = True
    drop_columns: bool = True
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "RoundDigits":
        """Fit the transformer by recording which columns to round."""
        if self.subset is None:
            self.subset = [col for col in X.columns if X[col].dtype.is_numeric()]
        if not self.inplace:
            self._column_mapping = {
                col: [f"{col}__round_{self.n_digits}digits"] for col in self.subset
            }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Round the selected columns to the configured number of decimal places."""
        columns: list[str] = self.subset or []

        def _round_expr(col: str, alias: str) -> pl.Expr:
            return pl.col(col).round(self.n_digits).alias(alias)

        if self.inplace:
            return X.with_columns([_round_expr(col, col) for col in columns])

        X = X.with_columns([_round_expr(col, new) for col, [new] in self._column_mapping.items()])
        if self.drop_columns:
            return X.drop(columns)
        return X
