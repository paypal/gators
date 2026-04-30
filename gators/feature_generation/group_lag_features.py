from typing import Dict, List, Optional

import polars as pl
from pydantic import field_validator, model_validator

from ..transformer._base_transformer import _BaseTransformer


class GroupLagFeatures(_BaseTransformer):
    """
    Generates lag (previous values) and lead (next values) features within groups.

    This transformer creates features like:

    - Previous transaction amount for this card
    - Next transaction amount for this card
    - Value N periods ago within group

    Useful for time-series analysis and detecting changes in behavior patterns.

    Parameters
    ----------
    subset : List[str]
        List of numerical column names to create lag/lead features for.
    by : List[str]
        List of columns to group by. Lags/leads are computed within each group.
    lags : List[int]
        List of lag periods. Positive integers create lag features (previous values).
        Example: [1, 2, 3] creates lag_1, lag_2, lag_3
    leads : List[int], default=[]
        List of lead periods. Positive integers create lead features (next values).
        Example: [1, 2] creates lead_1, lead_2
    fill_value : Optional[float], default=None
        Value to use for missing lag/lead values. If None, uses null.
    drop_columns : bool, default=False
        Whether to drop the original numerical columns after creating lag features.
    new_column_names : Optional[List[str]], default=None
        List of custom names for the lag/lead columns. If None, uses default naming pattern
        '{num_col}_lag{n}_{groupby_cols}' or '{num_col}_lead{n}_{groupby_cols}'.
        Must have same length as the total number of features created.

    Examples
    --------
    >>> from gators.feature_generation import GroupLagFeatures
    >>> import polars as pl

    >>> X ={
    ...     'amount': [100, 200, 150, 300, 250, 180],
    ...     'cat1': ['A', 'A', 'B', 'B', 'A', 'B'],
    ...     'time': [1, 2, 1, 2, 3, 3]
    ... }
    >>> X = pl.DataFrame(X).sort(['cat1', 'time'])

    **Example 1: Basic lag features**

    >>> transformer = GroupLagFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     lags=[1, 2]
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (6, 5)
    ┌────────┬───────┬──────┬─────────────────────┬─────────────────────┐
    │ amount ┆ cat1  ┆ time ┆ amount_lag1_cat1    ┆ amount_lag2_cat1    │
    │ ---    ┆ ---   ┆ ---  ┆ ---                 ┆ ---                 │
    │ i64    ┆ str   ┆ i64  ┆ i64                 ┆ i64                 │
    ╞════════╪═══════╪══════╪═════════════════════╪═════════════════════╡
    │ 100    ┆ A     ┆ 1    ┆ null                ┆ null                │
    │ 200    ┆ A     ┆ 2    ┆ 100                 ┆ null                │
    │ 250    ┆ A     ┆ 3    ┆ 200                 ┆ 100                 │
    │ 150    ┆ B     ┆ 1    ┆ null                ┆ null                │
    │ 300    ┆ B     ┆ 2    ┆ 150                 ┆ null                │
    │ 180    ┆ B     ┆ 3    ┆ 300                 ┆ 150                 │
    └────────┴───────┴──────┴─────────────────────┴─────────────────────┘

    **Example 2: Lag and lead features**

    >>> transformer = GroupLagFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     lags=[1],
    ...     leads=[1]
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (6, 5)
    ┌────────┬───────┬──────┬───────────────────┬────────────────────┐
    │ amount ┆ cat1  ┆ time ┆ amount_lag1_cat1  ┆ amount_lead1_cat1  │
    │ ---    ┆ ---   ┆ ---  ┆ ---               ┆ ---                │
    │ i64    ┆ str   ┆ i64  ┆ i64               ┆ i64                │
    ╞════════╪═══════╪══════╪═══════════════════╪════════════════════╡
    │ 100    ┆ A     ┆ 1    ┆ null              ┆ 200                │
    │ 200    ┆ A     ┆ 2    ┆ 100               ┆ 250                │
    │ 250    ┆ A     ┆ 3    ┆ 200               ┆ null               │
    │ 150    ┆ B     ┆ 1    ┆ null              ┆ 300                │
    │ 300    ┆ B     ┆ 2    ┆ 150               ┆ 180                │
    │ 180    ┆ B     ┆ 3    ┆ 300               ┆ null               │
    └────────┴───────┴──────┴───────────────────┴────────────────────┘

    **Example 3: With fill_value**

    >>> transformer = GroupLagFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     lags=[1],
    ...     fill_value=0.0
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result['amount_lag1_cat1'][0]  # First row, no previous value
    0.0

    Notes
    -----
    - Data should be sorted by by and time before transformation
    - Lag features look backwards: lag_1 is the previous row within the group
    - Lead features look forwards: lead_1 is the next row within the group
    - First rows in each group will have null (or fill_value) for lag features
    - Last rows in each group will have null (or fill_value) for lead features
    """

    subset: List[str]
    by: List[str]
    lags: List[int]
    leads: List[int] = []
    fill_value: Optional[float] = None
    drop_columns: bool = False
    new_column_names: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = {}

    @field_validator("lags")
    def check_lags(cls, lags):
        for lag in lags:
            if lag <= 0:
                raise ValueError(f"All lag values must be positive, got {lag}")
        return lags

    @field_validator("leads")
    def check_leads(cls, leads):
        for lead in leads:
            if lead <= 0:
                raise ValueError(f"All lead values must be positive, got {lead}")
        return leads

    @model_validator(mode="after")
    def check_lags_or_leads(self):
        if not self.lags and not self.leads:
            raise ValueError("At least one of 'lags' or 'leads' must be non-empty")
        return self

    @field_validator("new_column_names")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            subset = info.data.get("subset", [])
            lags = info.data.get("lags", [])
            leads = info.data.get("leads", [])
            expected_length = len(subset) * (len(lags) + len(leads))
            if len(new_column_names) != expected_length:
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match the total number of features created ({expected_length})"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "GroupLagFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        GroupLagFeatures
            Fitted transformer instance.
        """
        default_names = []
        group_name = "_".join(self.by)

        for num_col in self.subset:
            for lag in self.lags:
                default_names.append(f"{num_col}_lag{lag}_{group_name}")
            for lead in self.leads:
                default_names.append(f"{num_col}_lead{lead}_{group_name}")

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating lag/lead features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with lag/lead features.
        """
        new_columns = []
        group_name = "_".join(self.by)

        for num_col in self.subset:
            # Create lag features
            for lag in self.lags:
                default_name = f"{num_col}_lag{lag}_{group_name}"
                new_col_name = self._column_mapping[default_name]

                lag_expr = pl.col(num_col).shift(lag).over(self.by)

                if self.fill_value is not None:
                    lag_expr = lag_expr.fill_null(self.fill_value)

                new_columns.append(lag_expr.alias(new_col_name))

            # Create lead features
            for lead in self.leads:
                default_name = f"{num_col}_lead{lead}_{group_name}"
                new_col_name = self._column_mapping[default_name]

                lead_expr = pl.col(num_col).shift(-lead).over(self.by)

                if self.fill_value is not None:
                    lead_expr = lead_expr.fill_null(self.fill_value)

                new_columns.append(lead_expr.alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
