from abc import ABCMeta
from typing import Dict, List, Optional

import polars as pl
from pydantic import PositiveInt, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer

__all__ = ["_BaseDiscretizer", "generate_labels"]


def generate_labels(bins: Dict[str, List[float]], rounding=3) -> Dict[str, List[str]]:
    """
    Generate labels for equal-length discretizer bins.

    Parameters
    ----------
    bins : Dict[str, List[float]]
        Dictionary where keys are column names and values are lists of bin edges.
    rounding : int, optional
        Number of decimal places to round the bin edges for the labels. Default is 3.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary where keys are column names and values are lists of string labels
        for the bins.

    Examples
    --------
    >>> bins = {
    ...     'A': [-inf, 0.5, 1.0, inf],
    ...     'B': [-inf, 10, 20, inf]
    ... }
    >>> generate_labels(bins, rounding=1)
    {'A': ['(-inf,0.5]', '(0.5,1.0)', '(1.0,inf)'],
     'B': ['(-inf,10.0]', '(10.0,20.0)', '(20.0,inf)']}
    """
    labels = {}
    for col, arr in bins.items():
        # Handle empty bins (constant features)
        if len(arr) == 0:
            labels[col] = ["constant"]
            continue

        if arr[0] != float("-inf"):
            arr = [float("-inf")] + arr
        if arr[-1] != float("inf"):
            arr = arr + [float("inf")]

        labels[col] = [
            f"({round(a, rounding)},{round(b, rounding)}]" for a, b in zip(arr[:-1], arr[1:])
        ]
        if labels[col][-1].endswith("inf]"):
            labels[col][-1] = labels[col][-1].replace("]", ")")
    return labels


class _BaseDiscretizer(_BaseTransformer, metaclass=ABCMeta):
    """
    Base class for discretizers.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of column names to discretize. If None, all numeric columns are used.
    num_bins : PositiveInt, default=5
        Number of bins to divide each numeric column into.
    rounding : PositiveInt, default=3
        Decimal places to round bin edges for labels.
    drop_columns : bool, default=True
        If True, drops original columns after discretizing.

    Examples
    --------
    >>> from gators.discretizers import _BaseDiscretizer
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'A': [0.1, 0.2, 0.2, 0.4],
    ...     'B': [10, 20, 30, 40]
    ... })
    >>> discretizer = _BaseDiscretizer(num_bins=3, drop_columns=True)
    >>> discretizer.subset=['A', 'B']
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌───────────────┬───────────────┐
    │ A__dic_length │ B__dic_length │
    │ ---           │ ---           │
    │ str           │ str           │
    ├───────────────┼───────────────┤
    │ (0.1,0.2]     │ (10,20]       │
    │ (0.1,0.2]     │ (20,30]       │
    │ (0.2,0.3]     │ (20,30]       │
    │ (0.3,0.4]     │ (30,40]       │
    └───────────────┴───────────────┘

    >>> discretizer.drop_columns = False
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 4)
    ┌─────┬─────┬───────────────┬───────────────┐
    │ A   │ B   │ A__dic_length │ B__dic_length │
    │ --- │ --- │ ---           │ ---           │
    │ f64 │ i64 │ str           │ str           │
    ├─────┼─────┼───────────────┼───────────────┤
    │ 0.1 │ 10  │ (0.1,0.2]     │ (10,20]       │
    │ 0.2 │ 20  │ (0.1,0.2]     │ (20,30]       │
    │ 0.2 │ 30  │ (0.2,0.3]     │ (20,30]       │
    │ 0.4 │ 40  │ (0.3,0.4]     │ (30,40]       │
    └─────┴─────┴───────────────┴───────────────┘

    >>> discretizer.columns = None
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌───────────────┬───────────────┐
    │ A__dic_length │ B__dic_length │
    │ ---           │ ---           │
    │ str           │ str           │
    ├───────────────┼───────────────┤
    │ (0.1,0.2]     │ (10,20]       │
    │ (0.1,0.2]     │ (20,30]       │
    │ (0.2,0.3]     │ (20,30]       │
    │ (0.3,0.4]     │ (30,40]       │
    └───────────────┴───────────────┘

    >>> discretizer.subset=['A']
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 3)
    ┌─────┬─────┬───────────────┐
    │ A   │ B   │ A__dic_length │
    │ --- │ --- │ ---           │
    │ f64 │ i64 │ str           │
    ├─────┼─────┼───────────────┤
    │ 0.1 │ 10  │ (0.1,0.2]     │
    │ 0.2 │ 20  │ (0.1,0.2]     │
    │ 0.2 │ 30  │ (0.2,0.3]     │
    │ 0.4 │ 40  │ (0.3,0.4]     │
    └─────┴─────┴───────────────┘
    """

    subset: Optional[List[str]] = None
    num_bins: PositiveInt = 5
    rounding: PositiveInt = 3
    as_numerics: bool = False
    drop_columns: bool = True
    inplace: bool = True
    _bins: Dict[str, List[float]] = PrivateAttr(default_factory=dict)
    _labels: Dict[str, List[str]] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

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
            # subset is guaranteed to be set during fit
            transformations = [
                pl.col(col).cut(breaks=self._bins[col], labels=self._labels[col])
                for col in self.subset
            ]
            if self.as_numerics:
                transformations = [t.cast(pl.Int32) for t in transformations]
            return X.with_columns(transformations)

        transformations = [
            pl.col(col).cut(breaks=self._bins[col], labels=self._labels[col]).alias(new)
            for col, new in self._column_mapping.items()
        ]
        if self.as_numerics:
            transformations = [t.cast(pl.Int32) for t in transformations]
        X = X.with_columns(transformations)
        if self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X
