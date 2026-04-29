from typing import List, Literal, Optional

import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class TimeBinFeatures(_BaseTransformer):
    """
    Generates time bin features by categorizing datetime values into periods.

    Bins datetime components into meaningful categories like part of day, season,
    time of month, etc. These categorical features are particularly useful for
    tree-based models to capture non-linear temporal patterns.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of datetime columns to extract features from. If None, all datetime columns
        will be used.
    bin_types : List[Literal["part_of_day", "season", "time_of_month", "time_of_year", "rush_hour"]], default=["part_of_day", "season", "time_of_month", "time_of_year", "rush_hour"]
        Types of time bins to generate. Options:

        - "part_of_day": night, morning, afternoon, evening
        - "season": spring, summer, fall, winter
        - "time_of_month": beginning, middle, end
        - "time_of_year": early, mid, late
        - "rush_hour": morning_rush, evening_rush, off_peak
    hemisphere : Literal["northern", "southern"], default="northern"
        Hemisphere for season calculation.
    drop_columns : bool, default=False
        Whether to drop the original datetime columns after feature extraction.

    Examples
    --------
    >>> from gators.feature_generation_dt import TimeBinFeatures
    >>> import polars as pl

    >>> X =pl.DataFrame({
    ...     'timestamp': [
    ...         '2024-01-15 06:00:00',
    ...         '2024-01-15 10:00:00',
    ...         '2024-01-15 14:00:00',
    ...         '2024-01-15 20:00:00',
    ...         '2024-07-15 14:00:00',
    ...     ]
    ... }).with_columns(
    ...     pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S')
    ... )

    **Example 1: Part of day**

    >>> transformer = TimeBinFeatures(
    ...     subset=['timestamp'],
    ...     bin_types=['part_of_day']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (5, 2)
    ┌─────────────────────┬─────────────────────────┐
    │ timestamp           ┆ timestamp__part_of_day  │
    │ ---                 ┆ ---                     │
    │ datetime[μs]        ┆ str                     │
    ├─────────────────────┼─────────────────────────┤
    │ 2024-01-15 06:00:00 ┆ morning                 │
    │ 2024-01-15 10:00:00 ┆ morning                 │
    │ 2024-01-15 14:00:00 ┆ afternoon               │
    │ 2024-01-15 20:00:00 ┆ evening                 │
    │ 2024-07-15 14:00:00 ┆ afternoon               │
    └─────────────────────┴─────────────────────────┘

    **Example 2: Season (northern hemisphere)**

    >>> transformer = TimeBinFeatures(
    ...     subset=['timestamp'],
    ...     bin_types=['season'],
    ...     hemisphere='northern'
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (5, 2)
    ┌─────────────────────┬───────────────────┐
    │ timestamp           ┆ timestamp__season │
    │ ---                 ┆ ---               │
    │ datetime[μs]        ┆ str               │
    ├─────────────────────┼───────────────────┤
    │ 2024-01-15 06:00:00 ┆ winter            │
    │ 2024-01-15 10:00:00 ┆ winter            │
    │ 2024-01-15 14:00:00 ┆ winter            │
    │ 2024-01-15 20:00:00 ┆ winter            │
    │ 2024-07-15 14:00:00 ┆ summer            │
    └─────────────────────┴───────────────────┘

    **Example 3: Multiple bin types**

    >>> transformer = TimeBinFeatures(
    ...     subset=['timestamp'],
    ...     bin_types=['part_of_day', 'time_of_month', 'rush_hour']
    ... )
    >>> result = transformer.fit_transform(X)
    """

    subset: Optional[List[str]] = None
    bin_types: List[
        Literal["part_of_day", "season", "time_of_month", "time_of_year", "rush_hour"]
    ] = ["part_of_day", "season", "time_of_month", "time_of_year", "rush_hour"]
    hemisphere: Literal["northern", "southern"] = "northern"
    drop_columns: bool = False

    @field_validator("bin_types")
    def check_bin_types(cls, bin_types):
        valid_types = [
            "part_of_day",
            "season",
            "time_of_month",
            "time_of_year",
            "rush_hour",
        ]
        for bin_type in bin_types:
            if bin_type not in valid_types:
                raise ValueError(
                    f"Bin type '{bin_type}' is not supported. " f"Supported types: {valid_types}"
                )
        return bin_types

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "TimeBinFeatures":
        """Fit the transformer by identifying datetime columns if not specified.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        TimeBinFeatures
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in X.schema.items()
                if dtype == pl.Datetime or dtype == pl.Date
            ]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating time bin features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with time bin features.
        """
        if self.subset is None:
            return X
            
        new_columns = []

        for col in self.subset:
            hour = pl.col(col).dt.hour()
            month = pl.col(col).dt.month()
            day = pl.col(col).dt.day()

            if "part_of_day" in self.bin_types:
                part_of_day = (
                    pl.when(hour < 6)
                    .then(pl.lit("night"))
                    .when(hour < 12)
                    .then(pl.lit("morning"))
                    .when(hour < 18)
                    .then(pl.lit("afternoon"))
                    .otherwise(pl.lit("evening"))
                ).alias(f"{col}__part_of_day")
                new_columns.append(part_of_day)

            if "season" in self.bin_types:
                if self.hemisphere == "northern":
                    season = (
                        pl.when(month.is_in([12, 1, 2]))
                        .then(pl.lit("winter"))
                        .when(month.is_in([3, 4, 5]))
                        .then(pl.lit("spring"))
                        .when(month.is_in([6, 7, 8]))
                        .then(pl.lit("summer"))
                        .otherwise(pl.lit("fall"))
                    ).alias(f"{col}__season")
                else:  # southern hemisphere
                    season = (
                        pl.when(month.is_in([6, 7, 8]))
                        .then(pl.lit("winter"))
                        .when(month.is_in([9, 10, 11]))
                        .then(pl.lit("spring"))
                        .when(month.is_in([12, 1, 2]))
                        .then(pl.lit("summer"))
                        .otherwise(pl.lit("fall"))
                    ).alias(f"{col}__season")
                new_columns.append(season)

            if "time_of_month" in self.bin_types:
                time_of_month = (
                    pl.when(day <= 10)
                    .then(pl.lit("beginning"))
                    .when(day <= 20)
                    .then(pl.lit("middle"))
                    .otherwise(pl.lit("end"))
                ).alias(f"{col}__time_of_month")
                new_columns.append(time_of_month)

            if "time_of_year" in self.bin_types:
                time_of_year = (
                    pl.when(month <= 4)
                    .then(pl.lit("early"))
                    .when(month <= 8)
                    .then(pl.lit("mid"))
                    .otherwise(pl.lit("late"))
                ).alias(f"{col}__time_of_year")
                new_columns.append(time_of_year)

            if "rush_hour" in self.bin_types:
                rush_hour = (
                    pl.when((hour >= 7) & (hour < 9))
                    .then(pl.lit("morning_rush"))
                    .when((hour >= 17) & (hour < 19))
                    .then(pl.lit("evening_rush"))
                    .otherwise(pl.lit("off_peak"))
                ).alias(f"{col}__rush_hour")
                new_columns.append(rush_hour)

        X = X.with_columns(new_columns)

        if self.drop_columns and self.subset is not None:
            X = X.drop(self.subset)

        return X
