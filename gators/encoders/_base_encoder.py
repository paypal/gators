from abc import ABCMeta

import polars as pl
from pydantic import Field, PositiveFloat, PositiveInt, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class _BaseEncoder(_BaseTransformer, metaclass=ABCMeta):
    """
    Base encoder class for encoding categorical columns.

    Parameters
    ----------
    subset : list of str, default=None
        List of columns to encode. If None, all applicable columns are encoded.
    min_count : PositiveInt, PositiveFloat, default=1
        Minimum count or frequency for encoding categories.
    drop_columns : bool, default=True
        If True, the original columns are dropped after encoding.
    inplace : bool, default=True
        If True, replaces column values in-place. If False, creates new columns with suffix.

    Note
    ----
    _BaseEncoder is a base class and should not be used directly.
    Use one of the concrete encoder implementations instead.

    Boolean columns are not treated as categorical - cast them to String first
    (e.g. ``CastColumns(dtype=pl.String)``) if you want them encoded.

    """

    subset: list[str] | None = None
    mapping_: dict[str, dict[str, float]] = Field(default_factory=dict)
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    min_count: PositiveInt | PositiveFloat = 1
    drop_columns: bool = True
    inplace: bool = True

    _CAT_DTYPES: set = {pl.String, pl.Categorical, pl.Enum}

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
        default_value = 0.0

        expressions = []

        if self.inplace:
            for col in self.mapping_:
                expr = pl.col(col).replace_strict(
                    self.mapping_[col],
                    default=default_value,
                    return_dtype=pl.Float64,
                )
                expressions.append(expr)
            return X.with_columns(expressions)

        for col in self.mapping_:
            new_col_name = self._column_mapping[col][0]

            expr = (
                pl.col(col)
                .replace_strict(self.mapping_[col], default=default_value, return_dtype=pl.Float64)
                .alias(new_col_name)
            )
            expressions.append(expr)

        X = X.with_columns(expressions)

        if self.drop_columns and self.subset:
            X = X.drop(self.subset)

        return X
