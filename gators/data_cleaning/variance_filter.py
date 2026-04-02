from typing import Annotated, Dict, List, Optional

import polars as pl
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin


class VarianceFilter(BaseModel, BaseEstimator, TransformerMixin):
    """
    Removes numerical columns with a low variance.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric columns to check for variance. If None, all numeric columns are checked.
    min_var : float
        Minimum variance threshold. Columns with variance <= min_var will be dropped. Must be >= 0.0.

    Examples
    --------
    Initialize and use ``VarianceFilter``.

    Example with all numeric columns:

    >>> import polars as pl
    >>> from gators.data_cleaning import VarianceFilter
    >>> X = pl.DataFrame({
    ...     "feature1": [1, 2, 3, 4],
    ...     "feature2": [0.5, 0.5, 0.5, 0.5],  # Low variance
    ...     "feature3": [5, 6, 7, 8],
    ...     "label": [0, 1, 0, 1]
    ... })
    >>> transformer = VarianceFilter(min_var=0.1)
    >>> transformer.fit(X)
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 3)
    ┌──────────┬─────────┬───────┐
    │ feature1 │feature3 │ label │
    │     i64  │    i64  │  i64  │
    ├──────────┼────────–┼──────⁠–┤
    │       1  │      5  │    0  │
    │       2  │      6  │    1  │
    │       3  │      7  │    0  │
    │       4  │      8  │    1  │
    └──────────┴────────━┴─────–─┘

    Example with specific columns:

    >>> X = pl.DataFrame({
    ...     "feature1": [1, 2, 3, 4],
    ...     "feature2": [0.5, 0.5, 0.5, 0.5],
    ...     "feature3": [5, 6, 7, 8],
    ...     "label": [0, 1, 0, 1]
    ... })
    >>> transformer = VarianceFilter(subset=['feature1’, ‘feature3'], min_var=0.1)
    >>> transformer.fit(X)
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (4, 4)
    ┌──────────┬─────────┬──────────┬───────┐
    │ feature1 │feature3 │ feature2 │ label │
    │     i64  │    i64  │    i64   │  i64  │
    ├──────────┼────────–┼─────────⁠–┼──────⁠–┤
    │       1  │      5  │      0.5 │   0   │
    │       2  │      6  │      0.5 │   1   │
    │       3  │      7  │      0.5 │   0   │
    │       4  │      8  │      0.5 │   1   │
    └──────────┴─────────┴──────────┴──────–┘

    """

    subset: Optional[List[str]] = None
    min_var: Annotated[float, Field(ge=0.0)]
    _to_drop: List[str]
    _column_mapping = Dict[str, str]
    _std_devs: Dict[str, float]

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "VarianceFilter":
        """Fit the transformer by identifying low-variance columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        VarianceFilter
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in dict(zip(X.columns, X.dtypes)).items()
                if dtype not in [pl.String, pl.Boolean]
            ]

        self._std_devs = X.select(self.subset).std().row(0, named=True)
        self._to_drop = [col for col, ratio in self._std_devs.items() if ratio <= self.min_var]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by dropping low-variance columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with low-variance columns removed.
        """
        if not self._to_drop:
            return X
        return X.drop(self._to_drop)
