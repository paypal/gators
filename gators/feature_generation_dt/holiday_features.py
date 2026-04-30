from datetime import datetime
from typing import List, Optional

import holidays
import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class HolidayFeatures(_BaseTransformer):
    """
    Generates holiday-related features from datetime columns.

    Detects holidays and calculates distance to nearest holidays, which is useful
    for retail, finance, and other domains where holidays affect patterns.
    Uses the `holidays` library for accurate, year-specific holiday dates.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of datetime columns to extract features from. If None, all datetime columns
        will be used.
    country : str, default="US"
        Country code for holidays (e.g., "US", "UK", "CA", "DE", "FR", "JP").
        Supports any country code from the holidays library.
        See https://pypi.org/project/holidays/ for full list of supported countries.
    years : Optional[List[int]], default=None
        Years to include holidays for. If None, will automatically detect years from data.
    features : List[str], default=["is_holiday", "days_to_holiday", "days_from_holiday"]
        Features to generate. Options:

        - "is_holiday": Boolean for whether date is a holiday
        - "days_to_holiday": Days until next holiday (negative if past)
        - "days_from_holiday": Days since last holiday (negative if future)
        - "nearest_holiday_distance": Absolute days to nearest holiday
    drop_columns : bool, default=False
        Whether to drop the original datetime columns after feature extraction.

    Examples
    --------
    >>> from gators.feature_generation_dt import HolidayFeatures
    >>> import polars as pl

    >>> X =pl.DataFrame({
    ...     'date': [
    ...         '2024-01-01',  # New Year's Day
    ...         '2024-01-15',  # Around MLK Day
    ...         '2024-07-03',  # Day before Independence Day
    ...         '2024-07-04',  # Independence Day
    ...         '2024-07-05',  # Day after Independence Day
    ...     ]
    ... }).with_columns(
    ...     pl.col('date').str.strptime(pl.Datetime, '%Y-%m-%d')
    ... )

    **Example 1: Is holiday detection**

    >>> transformer = HolidayFeatures(
    ...     subset=['date'],
    ...     features=['is_holiday']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (5, 2)
    ┌─────────────────────┬──────────────────┐
    │ date                ┆ date__is_holiday │
    │ ---                 ┆ ---              │
    │ datetime[μs]        ┆ bool             │
    ├─────────────────────┼──────────────────┤
    │ 2024-01-01 00:00:00 ┆ true             │
    │ 2024-01-15 00:00:00 ┆ true             │
    │ 2024-07-03 00:00:00 ┆ false            │
    │ 2024-07-04 00:00:00 ┆ true             │
    │ 2024-07-05 00:00:00 ┆ false            │
    └─────────────────────┴──────────────────┘

    **Example 2: Distance to holidays**

    >>> transformer = HolidayFeatures(
    ...     subset=['date'],
    ...     features=['nearest_holiday_distance']
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 3: UK holidays**

    >>> transformer = HolidayFeatures(
    ...     subset=['date'],
    ...     country='UK',
    ...     features=['is_holiday']
    ... )
    >>> result = transformer.fit_transform(X)
    """

    subset: Optional[List[str]] = None
    country: str = "US"
    years: Optional[List[int]] = None
    features: List[str] = ["is_holiday", "days_to_holiday", "days_from_holiday"]
    drop_columns: bool = False
    _holidays: dict = {}
    _years: List[int] = []

    @field_validator("features")
    def check_features(cls, features):
        valid_features = [
            "is_holiday",
            "days_to_holiday",
            "days_from_holiday",
            "nearest_holiday_distance",
        ]
        for feature in features:
            if feature not in valid_features:
                raise ValueError(
                    f"Feature '{feature}' is not supported. "
                    f"Supported features: {valid_features}"
                )
        return features

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "HolidayFeatures":
        """Fit the transformer by identifying datetime columns and building holiday list.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        HolidayFeatures
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype == pl.Datetime or dtype == pl.Date
            ]

        # Determine years to use
        if self.years:
            self._years = self.years
        else:
            # Extract years from datetime columns
            years_set = set()
            for col in self.subset:
                col_years = X.select(pl.col(col).dt.year().unique()).to_series().to_list()
                years_set.update(col_years)
            self._years = sorted(list(years_set))

        # Build holiday dictionary using holidays library
        try:
            country_holidays = holidays.country_holidays(self.country, years=self._years)
            self._holidays = {date: name for date, name in country_holidays.items()}
        except (AttributeError, KeyError, NotImplementedError, Exception) as e:
            raise ValueError(
                f"Country code '{self.country}' is not supported by the holidays library. "
                f"Please check https://pypi.org/project/holidays/ for supported countries."
            ) from e

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating holiday features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with holiday features.
        """
        if self.subset is None:
            return X

        new_columns = []

        for col in self.subset:
            # Convert datetime column to date for holiday lookup
            date_col = pl.col(col).dt.date()

            if "is_holiday" in self.features:
                # Check if date is in holidays dictionary using native is_in
                holiday_dates_list = list(self._holidays.keys())

                # Use is_in for efficient lookup
                is_holiday_expr = date_col.is_in(holiday_dates_list)
                new_columns.append(is_holiday_expr.alias(f"{col}__is_holiday"))

            # Calculate distance features (optimized for large datasets)
            if (
                "nearest_holiday_distance" in self.features
                or "days_to_holiday" in self.features
                or "days_from_holiday" in self.features
            ):
                if self._holidays:
                    holiday_dates_sorted = sorted(self._holidays.keys())

                    # For each holiday, calculate the difference in days
                    # This creates multiple columns, then we'll aggregate them
                    if "nearest_holiday_distance" in self.features:
                        # Calculate distance to each holiday and find minimum
                        distance_exprs = [
                            (date_col - pl.lit(hdate)).dt.total_days().abs().cast(pl.Int32)
                            for hdate in holiday_dates_sorted
                        ]
                        if distance_exprs:
                            nearest_expr = pl.min_horizontal(*distance_exprs)
                            new_columns.append(
                                nearest_expr.alias(f"{col}__nearest_holiday_distance")
                            )

                    if "days_to_holiday" in self.features:
                        # Days to next holiday (positive values only for future dates)
                        future_distance_exprs = [
                            pl.when((date_col - pl.lit(hdate)).dt.total_days() <= 0)
                            .then((pl.lit(hdate) - date_col).dt.total_days().cast(pl.Int32))
                            .otherwise(None)
                            for hdate in holiday_dates_sorted
                        ]
                        if future_distance_exprs:
                            days_to_expr = pl.min_horizontal(*future_distance_exprs).fill_null(-1)
                            new_columns.append(days_to_expr.alias(f"{col}__days_to_holiday"))

                    if "days_from_holiday" in self.features:
                        # Days from last holiday (positive values only for past dates)
                        past_distance_exprs = [
                            pl.when((date_col - pl.lit(hdate)).dt.total_days() >= 0)
                            .then((date_col - pl.lit(hdate)).dt.total_days().cast(pl.Int32))
                            .otherwise(None)
                            for hdate in holiday_dates_sorted
                        ]
                        if past_distance_exprs:
                            days_from_expr = pl.min_horizontal(*past_distance_exprs).fill_null(-1)
                            new_columns.append(days_from_expr.alias(f"{col}__days_from_holiday"))

        X = X.with_columns(new_columns)

        if self.drop_columns and self.subset is not None:
            X = X.drop(self.subset)

        return X
