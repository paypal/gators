from datetime import datetime
from typing import List, Literal, Optional, Union

import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class DiffFeatures(_BaseTransformer):
    """
    Generates time difference features between datetime columns or against reference dates.

    Calculates differences in various units (days, hours, minutes, seconds) which are
    particularly useful for tree-based models to capture recency, age, and time elapsed.

    Parameters
    ----------
    column_pairs : Optional[List[tuple[str, str]]], default=None
        List of column pairs (col_a, col_b) to compute differences (col_a - col_b).
        If None, no pairwise differences are computed.
    reference_dates : Optional[dict[str, Union[str, datetime]]], default=None
        Dictionary mapping column names to reference dates. Computes (column - reference_date).
        Reference dates can be ISO format strings or datetime objects.
    units : List[Literal["d", "h", "m", "s"]], default=["d"]
        Units for computing time differences.
    drop_columns : bool, default=False
        Whether to drop the original datetime columns after creating differences.

    Examples
    --------
    >>> from gators.feature_generation_dt import DiffFeatures
    >>> import polars as pl
    >>> from datetime import datetime

    >>> X =pl.DataFrame({
    ...     'created_at': ['2023-01-01', '2023-06-15', '2024-01-01'],
    ...     'updated_at': ['2023-01-10', '2023-07-01', '2024-02-01'],
    ...     'value': [100, 200, 300]
    ... }).with_columns([
    ...     pl.col('created_at').str.strptime(pl.Datetime, '%Y-%m-%d'),
    ...     pl.col('updated_at').str.strptime(pl.Datetime, '%Y-%m-%d')
    ... ])

    **Example 1: Pairwise difference**

    >>> transformer = DiffFeatures(
    ...     column_pairs=[('updated_at', 'created_at')],
    ...     units=['days']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (3, 4)
    ┌─────────────────────┬─────────────────────┬───────┬──────────────────────────┐
    │ created_at          ┆ updated_at          ┆ value ┆ updated_at_minus_created │
    │ ---                 ┆ ---                 ┆ ---   ┆ _at__days                │
    │ datetime[μs]        ┆ datetime[μs]        ┆ i64   ┆ i64                      │
    ├─────────────────────┼─────────────────────┼───────┼──────────────────────────┤
    │ 2023-01-01 00:00:00 ┆ 2023-01-10 00:00:00 ┆ 100   ┆ 9                        │
    │ 2023-06-15 00:00:00 ┆ 2023-07-01 00:00:00 ┆ 200   ┆ 16                       │
    │ 2024-01-01 00:00:00 ┆ 2024-02-01 00:00:00 ┆ 300   ┆ 31                       │
    └─────────────────────┴─────────────────────┴───────┴──────────────────────────┘

    **Example 2: Reference date (time since reference)**

    >>> transformer = DiffFeatures(
    ...     reference_dates={'created_at': '2024-01-01'},
    ...     units=['days', 'hours']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (3, 5)
    ┌─────────────────────┬─────────────────────┬───────┬─────────────┬──────────────┐
    │ created_at          ┆ updated_at          ┆ value ┆ created_at_ ┆ created_at_  │
    │ ---                 ┆ ---                 ┆ ---   ┆ since_ref   ┆ since_ref    │
    │ datetime[μs]        ┆ datetime[μs]        ┆ i64   ┆ __days      ┆ __hours      │
    ├─────────────────────┼─────────────────────┼───────┼─────────────┼──────────────┤
    │ 2023-01-01 00:00:00 ┆ 2023-01-10 00:00:00 ┆ 100   ┆ -365        ┆ -8760        │
    │ 2023-06-15 00:00:00 ┆ 2023-07-01 00:00:00 ┆ 200   ┆ -200        ┆ -4800        │
    │ 2024-01-01 00:00:00 ┆ 2024-02-01 00:00:00 ┆ 300   ┆ 0           ┆ 0            │
    └─────────────────────┴─────────────────────┴───────┴─────────────┴──────────────┘

    **Example 3: Multiple units**

    >>> transformer = DiffFeatures(
    ...     column_pairs=[('updated_at', 'created_at')],
    ...     units=['days', 'hours', 'minutes']
    ... )
    >>> result = transformer.fit_transform(X)
    """

    column_pairs: Optional[List[tuple[str, str]]] = None
    reference_dates: Optional[dict[str, Union[str, datetime]]] = None
    units: List[Literal["d", "h", "m", "s"]] = ["d"]
    drop_columns: bool = False
    _parsed_reference_dates: dict = {}

    @field_validator("units")
    def check_units(cls, units):
        valid_units = ["d", "h", "m", "s"]
        for unit in units:
            if unit not in valid_units:
                raise ValueError(
                    f"Unit '{unit}' is not supported. " f"Supported units: {valid_units}"
                )
        return units

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "DiffFeatures":
        """Fit the transformer by parsing reference dates.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        DiffFeatures
            Fitted transformer instance.
        """
        # Parse reference dates once during fit
        if self.reference_dates:
            for col, ref_date in self.reference_dates.items():
                if isinstance(ref_date, str):
                    # Parse string to datetime
                    self._parsed_reference_dates[col] = pl.lit(
                        datetime.fromisoformat(ref_date)
                    ).cast(pl.Datetime)
                elif isinstance(ref_date, datetime):
                    self._parsed_reference_dates[col] = pl.lit(ref_date).cast(pl.Datetime)
                else:
                    raise ValueError(
                        f"Reference date for '{col}' must be string or datetime, "
                        f"got {type(ref_date)}"
                    )
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating time difference features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with time difference features.
        """
        new_columns = []
        columns_to_drop = set()

        # Unit conversion factors (to the base unit)
        unit_conversions = {
            "d": lambda diff: diff.dt.total_days(),
            "h": lambda diff: diff.dt.total_hours(),
            "m": lambda diff: diff.dt.total_minutes(),
            "s": lambda diff: diff.dt.total_seconds(),
        }

        # Unit names for column suffixes
        unit_names = {
            "d": "days",
            "h": "hours",
            "m": "minutes",
            "s": "seconds",
        }

        # Pairwise column differences
        if self.column_pairs:
            for col_a, col_b in self.column_pairs:
                diff = pl.col(col_a) - pl.col(col_b)

                for unit in self.units:
                    col_name = f"{col_a}_minus_{col_b}__{unit_names[unit]}"
                    new_columns.append(unit_conversions[unit](diff).cast(pl.Int64).alias(col_name))

                if self.drop_columns:
                    columns_to_drop.add(col_a)
                    columns_to_drop.add(col_b)

        # Reference date differences
        if self.reference_dates:
            for col, ref_expr in self._parsed_reference_dates.items():
                diff = pl.col(col) - ref_expr

                for unit in self.units:
                    col_name = f"{col}_since_ref__{unit_names[unit]}"
                    new_columns.append(unit_conversions[unit](diff).cast(pl.Int64).alias(col_name))

                if self.drop_columns:
                    columns_to_drop.add(col)

        X = X.with_columns(new_columns)

        if self.drop_columns and columns_to_drop:
            X = X.drop(list(columns_to_drop))

        return X
