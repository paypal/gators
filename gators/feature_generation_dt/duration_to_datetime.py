from datetime import datetime
from typing import List, Literal, Optional, Union

import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class DurationToDatetime(_BaseTransformer):
    """
    Converts numeric time offset columns to datetime by adding durations to a reference date.

    This transformer is useful when you have numeric time offsets (e.g., seconds, days) that
    need to be converted to actual datetime values by adding them to a reference date. The
    reference date can be a fixed datetime literal or a column containing dates.

    Parameters
    ----------
    subset : List[str]
        List of column names containing numeric time offsets to convert.
    reference_date : Union[datetime, str]
        Reference date to add offsets to. Can be:

        - A datetime object: Same reference date for all rows
        - A string (column name): Different reference date per row from that column
    unit : Literal["s", "m", "h", "d", "ms", "us"], default="s"
        Time unit of the numeric offset columns:

        - "s": seconds
        - "m": minutes
        - "h": hours
        - "d": days
        - "ms": milliseconds
        - "us": microseconds
    drop_columns : bool, default=False
        Whether to drop the original numeric offset columns after conversion.

    Examples
    --------
    >>> from gators.feature_generation_dt import DurationToDatetime
    >>> import polars as pl
    >>> from datetime import datetime

    >>> X = pl.DataFrame({
    ...     'TransactionDT': [86400, 172800, 259200],  # seconds
    ...     'value': [100, 200, 300]
    ... })

    **Example 1: Convert seconds to datetime with fixed reference date**

    >>> transformer = DurationToDatetime(
    ...     subset=['TransactionDT'],
    ...     reference_date=datetime(2017, 11, 30),
    ...     unit='s',
    ...     drop_columns=False
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (3, 3)
    ┌───────────────┬───────┬──────────────────────────┐
    │ TransactionDT ┆ value ┆ TransactionDT__datetime  │
    │ ---           ┆ ---   ┆ ---                      │
    │ i64           ┆ i64   ┆ datetime[μs]             │
    ├───────────────┼───────┼──────────────────────────┤
    │ 86400         ┆ 100   ┆ 2017-12-01 00:00:00      │
    │ 172800        ┆ 200   ┆ 2017-12-02 00:00:00      │
    │ 259200        ┆ 300   ┆ 2017-12-03 00:00:00      │
    └───────────────┴───────┴──────────────────────────┘

    **Example 2: Convert with column-based reference dates**

    >>> X = pl.DataFrame({
    ...     'BaseDate': [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
    ...     'offset_days': [7, 14, 21],
    ...     'value': [100, 200, 300]
    ... })
    >>> transformer = DurationToDatetime(
    ...     subset=['offset_days'],
    ...     reference_date='BaseDate',  # column name
    ...     unit='d',
    ...     drop_columns=False
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (3, 4)
    ┌─────────────────────┬──────────────┬───────┬─────────────────────────┐
    │ BaseDate            ┆ offset_days  ┆ value ┆ offset_days__datetime   │
    │ ---                 ┆ ---          ┆ ---   ┆ ---                     │
    │ datetime[μs]        ┆ i64          ┆ i64   ┆ datetime[μs]            │
    ├─────────────────────┼──────────────┼───────┼─────────────────────────┤
    │ 2024-01-01 00:00:00 ┆ 7            ┆ 100   ┆ 2024-01-08 00:00:00     │
    │ 2024-02-01 00:00:00 ┆ 14           ┆ 200   ┆ 2024-02-15 00:00:00     │
    │ 2024-03-01 00:00:00 ┆ 21           ┆ 300   ┆ 2024-03-22 00:00:00     │
    └─────────────────────┴──────────────┴───────┴─────────────────────────┘

    **Example 3: Multiple columns with different units**

    >>> X = pl.DataFrame({
    ...     'offset_hours': [24, 48, 72],
    ...     'offset_minutes': [60, 120, 180],
    ...     'value': [1, 2, 3]
    ... })
    >>> transformer1 = DurationToDatetime(
    ...     subset=['offset_hours'],
    ...     reference_date=datetime(2024, 1, 1),
    ...     unit='h'
    ... )
    >>> transformer2 = DurationToDatetime(
    ...     subset=['offset_minutes'],
    ...     reference_date=datetime(2024, 1, 1),
    ...     unit='m'
    ... )
    >>> result = transformer1.fit_transform(X)
    >>> result = transformer2.fit_transform(result)
    """

    subset: List[str]
    reference_date: Union[datetime, str]
    unit: Literal["s", "m", "h", "d", "ms", "us"] = "s"
    drop_columns: bool = False
    _reference_expr: Optional[pl.Expr] = None
    _is_column_reference: bool = False

    @field_validator("unit")
    def check_unit(cls, unit):
        valid_units = ["s", "m", "h", "d", "ms", "us"]
        if unit not in valid_units:
            raise ValueError(f"Unit '{unit}' is not supported. " f"Supported units: {valid_units}")
        return unit

    @field_validator("reference_date")
    def check_reference_date(cls, reference_date):
        if not isinstance(reference_date, (datetime, str)):
            raise ValueError(
                f"reference_date must be a datetime object or string, "
                f"got {type(reference_date).__name__}"
            )
        return reference_date

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "DurationToDatetime":
        """Fit the transformer by preparing the reference date expression.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        DurationToDatetime
            Fitted transformer instance.
        """
        # Determine if reference_date is a column or a literal
        if isinstance(self.reference_date, str):
            # Check if it's a column name
            if self.reference_date in X.columns:
                self._is_column_reference = True
                self._reference_expr = pl.col(self.reference_date)
            else:
                # Try parsing as ISO format datetime string
                try:
                    parsed_date = datetime.fromisoformat(self.reference_date)
                    self._reference_expr = pl.lit(parsed_date).cast(pl.Datetime)
                    self._is_column_reference = False
                except ValueError:
                    raise ValueError(
                        f"reference_date '{self.reference_date}' is neither a column "
                        f"in the DataFrame nor a valid ISO format datetime string"
                    )
        elif isinstance(self.reference_date, datetime):
            self._reference_expr = pl.lit(self.reference_date).cast(pl.Datetime)
            self._is_column_reference = False
        else:
            raise ValueError(
                f"reference_date must be a datetime object or string, "
                f"got {type(self.reference_date)}"
            )

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform numeric offset columns to datetime columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with datetime columns.
        """
        new_columns = []

        # Duration function mapping
        duration_functions = {
            "s": pl.duration,
            "m": pl.duration,
            "h": pl.duration,
            "d": pl.duration,
            "ms": pl.duration,
            "us": pl.duration,
        }

        # Parameter name mapping for pl.duration
        param_names = {
            "s": "seconds",
            "m": "minutes",
            "h": "hours",
            "d": "days",
            "ms": "milliseconds",
            "us": "microseconds",
        }

        for col in self.subset:
            # Create duration and add to reference date
            param_name = param_names[self.unit]
            kwargs = {param_name: pl.col(col)}
            duration_expr = duration_functions[self.unit](**kwargs)  # type: ignore[arg-type]
            datetime_expr = self._reference_expr + duration_expr

            new_col_name = f"{col}__datetime"
            new_columns.append(datetime_expr.alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
