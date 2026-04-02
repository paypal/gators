from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel, field_validator
from sklearn.base import BaseEstimator, TransformerMixin


class HighCardinalityFilter(BaseModel, BaseEstimator, TransformerMixin):
    """
    Removes columns with too many unique values (high cardinality).

    Identifies and removes columns with excessive cardinality, which can cause
    issues for tree-based models (memory, overfitting) and create sparse encodings.
    Common use case: remove ID-like columns, timestamps, or free-text fields.

    Opposite of DropLowCardinality. Can filter by absolute count threshold
    or by ratio of unique values to total rows.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of columns to check for high cardinality. If None, all columns
        are checked.
    max_unique : Optional[int], default=None
        Maximum number of unique values allowed. Columns with more unique
        values will be removed. If None, no absolute threshold is applied.
    max_ratio : Optional[float], default=None
        Maximum ratio of unique values to total rows. Must be between 0 and 1.
        For example, 0.9 means columns where >90% of rows are unique will be
        removed. If None, no ratio threshold is applied.
    ignore_na : bool, default=True
        Whether to ignore NaN/null values when counting unique values.
        If True, NaN is not counted as a unique value.

    Examples
    --------
    **Example 1: Remove by absolute count**

    >>> from gators.data_cleaning import HighCardinalityFilter
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'user_id': range(1000),
    ...     'country': ['USA'] * 500 + ['UK'] * 500,
    ...     'transaction_id': [f'tx_{i}' for i in range(1000)]
    ... })
    >>> filter = HighCardinalityFilter(max_unique=100)
    >>> result = filter.fit_transform(X)
    >>> print(result)
    shape: (1000, 1)
    ┌─────────┐
    │ country │
    │ ---     │
    │ str     │
    ├─────────┤
    │ USA     │
    │ USA     │
    │ ...     │
    │ UK      │
    │ UK      │
    └─────────┘

    **Example 2: Remove by ratio**

    >>> X = pl.DataFrame({
    ...     'id': range(100),
    ...     'category': ['A', 'B', 'C'] * 33 + ['A'],
    ...     'subcategory': ['X', 'Y'] * 50
    ... })
    >>> filter = HighCardinalityFilter(max_ratio=0.95)
    >>> result = filter.fit_transform(X)
    >>> print(result.columns)
    ['category', 'subcategory']

    **Example 3: Combined thresholds**

    >>> X = pl.DataFrame({
    ...     'col1': range(50),  # 50 unique, ratio=1.0
    ...     'col2': list(range(25)) * 2,  # 25 unique, ratio=0.5
    ...     'col3': ['A', 'B'] * 25  # 2 unique, ratio=0.04
    ... })
    >>> filter = HighCardinalityFilter(max_unique=30, max_ratio=0.8)
    >>> result = filter.fit_transform(X)
    >>> print(result.columns)
    ['col2', 'col3']

    **Example 4: Handling NaN**

    >>> X = pl.DataFrame({
    ...     'col1': [1, 2, 3, None, None] * 20,  # 3 unique + NaN
    ...     'col2': list(range(90)) + [None] * 10  # 90 unique + NaN
    ... })
    >>> filter = HighCardinalityFilter(max_unique=50, ignore_na=True)
    >>> result = filter.fit_transform(X)
    >>> print(result.columns)
    ['col1']

    **Example 5: Subset of columns**

    >>> X = pl.DataFrame({
    ...     'id1': range(100),
    ...     'id2': range(100),
    ...     'feature': ['A', 'B'] * 50
    ... })
    >>> filter = HighCardinalityFilter(subset=['id1', 'id2'], max_unique=50)
    >>> result = filter.fit_transform(X)
    >>> print(result.columns)
    ['feature']
    """

    subset: Optional[List[str]] = None
    max_unique: Optional[int] = None
    max_ratio: Optional[float] = None
    ignore_na: bool = True
    _to_drop: List[str] = []

    @field_validator("max_unique")
    @classmethod
    def validate_max_unique(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("max_unique must be >= 1")
        return v

    @field_validator("max_ratio")
    @classmethod
    def validate_max_ratio(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 1):
            raise ValueError("max_ratio must be between 0 and 1")
        return v

    def model_post_init(self, __context):
        """Validate that at least one threshold is provided."""
        if self.max_unique is None and self.max_ratio is None:
            raise ValueError("At least one of max_unique or max_ratio must be provided")

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "HighCardinalityFilter":
        """Fit the transformer by identifying high-cardinality columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        HighCardinalityFilter
            Fitted transformer instance.
        """
        # Use all columns if not specified
        columns_to_check = self.subset if self.subset is not None else X.columns

        self._to_drop = []
        n_rows = len(X)

        for col in columns_to_check:
            # Count unique values
            if self.ignore_na:
                n_unique = X[col].drop_nulls().n_unique()
            else:
                n_unique = X[col].n_unique()

            # Check absolute threshold
            exceeds_count = self.max_unique is not None and n_unique > self.max_unique

            # Check ratio threshold
            ratio = n_unique / n_rows if n_rows > 0 else 0
            exceeds_ratio = self.max_ratio is not None and ratio > self.max_ratio

            # Remove if either threshold is exceeded
            if exceeds_count or exceeds_ratio:
                self._to_drop.append(col)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by removing high-cardinality columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with high-cardinality columns removed.
        """
        if self._to_drop:
            return X.drop(self._to_drop)
        return X
