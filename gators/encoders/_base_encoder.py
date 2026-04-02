from abc import ABCMeta
from typing import Dict, List, Optional, Union

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["_BaseEncoder"]


class _BaseEncoder(BaseModel, BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """
    Base encoder class for encoding categorical columns.

    Parameters
    ----------
    subset : list of str, default=None
        List of columns to encode. If None, all applicable columns are encoded.
    min_count : Union[int, float], default=1
        Minimum count or frequency for encoding categories.
    drop_columns : bool, default=True
        If True, the original columns are dropped after encoding.
    inplace : bool, default=True
        If True, replaces column values in-place. If False, creates new columns with suffix.

    Note
    ----
    _BaseEncoder is a base class and should not be used directly.
    Use one of the concrete encoder implementations instead.

    """

    subset: Optional[List[str]] = None
    mapping_: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    column_mapping_: Dict[str, str] = Field(default_factory=dict)
    min_count: Union[PositiveInt, PositiveFloat] = 1
    drop_columns: bool = True
    inplace: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        if self.inplace:
            expressions = []
            for col in self.mapping_:
                # Cast boolean columns to string for replacement
                if X[col].dtype == pl.Boolean:
                    # Convert boolean keys to lowercase string format used by Polars
                    string_mapping = {str(k).lower(): v for k, v in self.mapping_[col].items()}
                    expr = (
                        pl.col(col)
                        .cast(pl.String)
                        .replace_strict(
                            string_mapping,
                            default=default_value,
                            return_dtype=pl.Float64,
                        )
                    )
                else:
                    expr = pl.col(col).replace_strict(
                        self.mapping_[col],
                        default=default_value,
                        return_dtype=pl.Float64,
                    )
                expressions.append(expr)
            return X.with_columns(expressions)

        expressions = []
        for col, mapping in self.mapping_.items():
            # Cast boolean columns to string for replacement, then to float
            if X[col].dtype == pl.Boolean:
                # Convert boolean keys to lowercase string format used by Polars
                string_mapping = {str(k).lower(): v for k, v in mapping.items()}
                expr = (
                    pl.col(col)
                    .cast(pl.String)
                    .replace_strict(string_mapping, default=default_value, return_dtype=pl.Float64)
                    .alias(self.column_mapping_[col])
                )
            else:
                expr = (
                    pl.col(col)
                    .replace_strict(mapping, default=default_value, return_dtype=pl.Float64)
                    .alias(self.column_mapping_[col])
                )
            expressions.append(expr)
        X = X.with_columns(expressions)
        if self.drop_columns and self.subset:
            X = X.drop(self.subset)
        return X
