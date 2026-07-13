import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class _BaseSelector(_BaseTransformer):
    """Base class for feature selectors.

    Provides the shared private attributes, public properties, and the
    default ``transform`` implementation used by all concrete selector
    subclasses.  Subclasses must implement ``fit``.

    Attributes
    ----------
    selected_features_ : list[str]
        Column names that survive the selection filter (set after ``fit``).
    columns_to_drop_ : list[str]
        Column names removed by the selection filter (set after ``fit``).
    """

    _selected_features: list[str] = PrivateAttr(default_factory=list)
    _columns_to_drop: list[str] = PrivateAttr(default_factory=list)

    @property
    def selected_features_(self) -> list[str]:
        return self._selected_features

    @property
    def columns_to_drop_(self) -> list[str]:
        return self._columns_to_drop

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Drop selected columns from the DataFrame.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``columns_to_drop_`` removed.
        """
        if not self._columns_to_drop:
            return X
        return X.drop(self._columns_to_drop)
