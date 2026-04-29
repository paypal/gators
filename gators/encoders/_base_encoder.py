from abc import ABCMeta
from typing import Dict, List, Optional, Union

import polars as pl
from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt

from ..transformer._base_transformer import _BaseTransformer


class _BaseEncoder(_BaseTransformer, metaclass=ABCMeta):
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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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
        
        dtypes = dict(zip(X.columns, X.dtypes))
        boolean_cols = {col for col in self.mapping_ if dtypes.get(col) == pl.Boolean}
        
        boolean_string_mappings = {
            col: {str(k).lower(): v for k, v in self.mapping_[col].items()}
            for col in boolean_cols
        }
        
        expressions = []
        
        if self.inplace:
            for col in self.mapping_:
                if col in boolean_cols:
                    # Boolean column: cast to string then replace
                    expr = (
                        pl.col(col)
                        .cast(pl.String)
                        .replace_strict(
                            boolean_string_mappings[col],
                            default=default_value,
                            return_dtype=pl.Float64,
                        )
                    )
                else:
                    # Non-boolean: direct replacement
                    expr = pl.col(col).replace_strict(
                        self.mapping_[col],
                        default=default_value,
                        return_dtype=pl.Float64,
                    )
                expressions.append(expr)
            return X.with_columns(expressions)
        
        for col in self.mapping_:
            new_col_name = self.column_mapping_[col]
            
            if col in boolean_cols:
                expr = (
                    pl.col(col)
                    .cast(pl.String)
                    .replace_strict(
                        boolean_string_mappings[col],
                        default=default_value,
                        return_dtype=pl.Float64
                    )
                    .alias(new_col_name)
                )
            else:
                expr = (
                    pl.col(col)
                    .replace_strict(
                        self.mapping_[col],
                        default=default_value,
                        return_dtype=pl.Float64
                    )
                    .alias(new_col_name)
                )
            expressions.append(expr)
        
        X = X.with_columns(expressions)
        
        if self.drop_columns and self.subset:
            X = X.drop(self.subset)
        
        return X
