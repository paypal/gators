from typing import Dict, List, Literal, Optional, Union

import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class GroupByImputer(_BaseTransformer):
    """
    Impute missing values in numeric columns by grouping with a categorical column
    and filling with the median or mean of each group.

    Parameters
    ----------
    group_by_column : str
        The categorical column to group by for computing group-wise statistics.
    strategy : Literal['median', 'mean']
        Strategy to use for imputing missing values within each group.

        - 'median': Fill with the median of each group
        - 'mean': Fill with the mean of each group
    subset : Optional[List[str]], default=None
        List of numeric columns to impute. If None, all numeric columns (except group_by_column) are selected.
    inplace : bool, default=True
        If True, impute values in the original columns.
        If False, create new columns with suffix '__impute_groupby_{strategy}'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after imputation.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.imputers import GroupByImputer

    >>> # Sample DataFrame
    >>> X = pl.DataFrame({
    ...     'district': ['A', 'A', 'B', 'B', 'C'],
    ...     'value1': [1.0, None, 3.0, 4.0, None],
    ...     'value2': [10.0, 20.0, None, 40.0, 50.0]
    ... })

    >>> # Impute using group median
    >>> imputer = GroupByImputer(group_by_column='district', strategy='median', inplace=False)
    >>> imputer.fit(X)
    GroupByImputer(group_by_column='district', strategy='median', subset=['value1', 'value2'], drop_columns=True, inplace=False)
    >>> transformed_X = imputer.transform(X)
    >>> print(transformed_X)
    shape: (5, 3)
    ┌──────────┬───────────────────────────────┬───────────────────────────────┐
    │ district ┆ value1__impute_groupby_median ┆ value2__impute_groupby_median │
    │ ---      ┆ ---                           ┆ ---                           │
    │ str      ┆ f64                           ┆ f64                           │
    ╞══════════╪═══════════════════════════════╪═══════════════════════════════╡
    │ A        ┆ 1.0                           ┆ 10.0                          │
    │ A        ┆ 1.0                           ┆ 20.0                          │
    │ B        ┆ 3.0                           ┆ 40.0                          │
    │ B        ┆ 4.0                           ┆ 40.0                          │
    │ C        ┆ null                          ┆ 50.0                          │
    └──────────┴───────────────────────────────┴───────────────────────────────┘

    >>> # Impute using group mean with inplace=True
    >>> imputer_inplace = GroupByImputer(
    ...     group_by_column='district',
    ...     strategy='mean',
    ...     inplace=True
    ... )
    >>> imputer_inplace.fit(X)
    GroupByImputer(group_by_column='district', strategy='mean', subset=['value1', 'value2'], drop_columns=True, inplace=True)
    >>> transformed_X_inplace = imputer_inplace.transform(X)
    >>> print(transformed_X_inplace)
    shape: (5, 3)
    ┌──────────┬────────┬────────┐
    │ district ┆ value1 ┆ value2 │
    │ ---      ┆ ---    ┆ ---    │
    │ str      ┆ f64    ┆ f64    │
    ╞══════════╪════════╪════════╡
    │ A        ┆ 1.0    ┆ 10.0   │
    │ A        ┆ 1.0    ┆ 20.0   │
    │ B        ┆ 3.0    ┆ 40.0   │
    │ B        ┆ 4.0    ┆ 40.0   │
    │ C        ┆ null   ┆ 50.0   │
    └──────────┴────────┴────────┘
    """

    group_by_column: str
    strategy: Literal["median", "mean"]
    subset: Optional[List[str]] = None
    drop_columns: bool = True
    inplace: bool = True
    _statistics: Dict[str, Dict[str, Union[int, float]]] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "GroupByImputer":
        """Fit the transformer by computing group-wise imputation statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns and a grouping column.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        GroupByImputer
            The fitted transformer instance.
        """
        if not self.subset:
            # Auto-detect numeric columns
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype not in [pl.String, pl.Boolean] and col != self.group_by_column
            ]

        if not self.inplace:
            self._column_mapping = {
                col: f"{col}__impute_groupby_{self.strategy}" for col in self.subset
            }

        # Compute group statistics for each column
        for col in self.subset:
            if self.strategy == "median":
                group_stats = X.group_by(self.group_by_column).agg(
                    pl.col(col).median().alias("stat")
                )
            else:  # mean
                group_stats = X.group_by(self.group_by_column).agg(pl.col(col).mean().alias("stat"))

            # Convert to dictionary for fast lookup
            self._statistics[col] = dict(
                zip(
                    group_stats[self.group_by_column].to_list(),
                    group_stats["stat"].to_list(),
                )
            )

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by imputing missing values using group-wise statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns containing null values and a grouping column.

        Returns
        -------
        pl.DataFrame
            DataFrame with imputed numeric columns based on group statistics.
        """
        # Join all group statistics to the dataframe first
        temp_columns = []
        if self.subset is None:
            return X

        for col in self.subset:
            # Create temporary column name for group statistics
            temp_col = f"__{col}_group_stat"
            temp_columns.append(temp_col)

            # Join with group statistics
            group_stats_X = pl.DataFrame(
                {
                    self.group_by_column: list(self._statistics[col].keys()),
                    temp_col: list(self._statistics[col].values()),
                }
            )

            X = X.join(group_stats_X, on=self.group_by_column, how="left")

        # Build transformations for each column
        transformations = []
        
        for col in self.subset:
            temp_col = f"__{col}_group_stat"

            # Fill nulls in the original column with group statistics
            if self.inplace:
                transformations.append(pl.col(col).fill_null(pl.col(temp_col)).alias(col))
            else:
                new_col = self._column_mapping[col]
                transformations.append(pl.col(col).fill_null(pl.col(temp_col)).alias(new_col))

        # Apply all transformations at once
        X = X.with_columns(transformations)

        # Drop temporary columns
        X = X.drop(temp_columns)

        if not self.inplace and self.drop_columns and self.subset is not None:
            return X.drop(self.subset)

        return X
