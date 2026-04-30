from typing import Dict, Optional, Tuple

import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class _BaseClipper(_BaseTransformer):
    """Base class for clippers that cap values based on specified bounds.

    This abstract class provides common functionality for clippers that apply
    value capping based on various criteria (e.g., quantiles, MAD, Gaussian).
    Subclasses must implement the specific logic to compute clipping bounds and
    apply the clipping transformation.

    Parameters
    ----------
    inplace : bool, default=True
        If True, clip values in the original columns.
        If False, create new columns with appropriate suffixes.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after clipping.
        Ignored when inplace=True.

    """

    drop_columns: bool = True
    inplace: bool = True
    _clip_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = PrivateAttr(
        default_factory=dict
    )
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by clipping values to quantile thresholds.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with clipped numeric columns.
        """
        if self.inplace:
            transformations = [
                pl.col(col).clip(
                    lower_bound=self._clip_bounds[col][0], upper_bound=self._clip_bounds[col][1]
                )
                for col in self.subset
            ]
        else:
            transformations = [
                pl.col(col)
                .clip(lower_bound=self._clip_bounds[col][0], upper_bound=self._clip_bounds[col][1])
                .alias(new)
                for col, new in self._column_mapping.items()
            ]

        X = X.with_columns(transformations)

        if not self.inplace and self.drop_columns:
            return X.drop(self.subset)
        return X
