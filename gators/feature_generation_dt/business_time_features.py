from typing import List, Optional

import polars as pl
from pydantic import BaseModel, field_validator
from sklearn.base import BaseEstimator, TransformerMixin


class BusinessTimeFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Generates business time features from datetime columns.

    Creates binary indicators and classifications for business-relevant time periods
    such as business hours, business days, and time of business day.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of datetime columns to extract features from. If None, all datetime columns
        will be used.
    business_hours_start : int, default=9
        Start hour for business hours (24-hour format, 0-23).
    business_hours_end : int, default=17
        End hour for business hours (24-hour format, 0-23).
    weekend_days : List[int], default=[5, 6]
        Days of week considered weekend (0=Monday, 6=Sunday). Default is Saturday and Sunday.
    features : List[str], default=["is_business_hour", "is_business_day", "time_of_business_day"]
        List of features to generate. Options:

        - "is_business_hour": Boolean for whether time is during business hours
        - "is_business_day": Boolean for whether day is a business day (not weekend)
        - "time_of_business_day": Category (before_hours, during_hours, after_hours)
        - "hour_of_business_day": Hour within business day (0-based from start)
    drop_columns : bool, default=False
        Whether to drop the original datetime columns after feature extraction.

    Examples
    --------
    >>> from gators.feature_generation_dt import BusinessTimeFeatures
    >>> import polars as pl

    >>> X =pl.DataFrame({
    ...     'timestamp': [
    ...         '2024-01-15 08:00:00',  # Monday, before hours
    ...         '2024-01-15 10:30:00',  # Monday, during hours
    ...         '2024-01-15 18:00:00',  # Monday, after hours
    ...         '2024-01-20 10:00:00',  # Saturday, weekend
    ...     ]
    ... }).with_columns(
    ...     pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S')
    ... )

    **Example 1: Default business time features**

    >>> transformer = BusinessTimeFeatures(
    ...     subset=['timestamp'],
    ...     features=['is_business_hour', 'is_business_day']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (4, 3)
    ┌─────────────────────┬───────────────────┬─────────────────┐
    │ timestamp           ┆ timestamp__is_bus ┆ timestamp__is_b │
    │ ---                 ┆ iness_hour        ┆ usiness_day     │
    │ datetime[μs]        ┆ ---               ┆ ---             │
    │                     ┆ bool              ┆ bool            │
    ├─────────────────────┼───────────────────┼─────────────────┤
    │ 2024-01-15 08:00:00 ┆ false             ┆ true            │
    │ 2024-01-15 10:30:00 ┆ true              ┆ true            │
    │ 2024-01-15 18:00:00 ┆ false             ┆ true            │
    │ 2024-01-20 10:00:00 ┆ true              ┆ false           │
    └─────────────────────┴───────────────────┴─────────────────┘

    **Example 2: Time of business day classification**

    >>> transformer = BusinessTimeFeatures(
    ...     subset=['timestamp'],
    ...     features=['time_of_business_day'],
    ...     business_hours_start=9,
    ...     business_hours_end=17
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (4, 2)
    ┌─────────────────────┬─────────────────────────────┐
    │ timestamp           ┆ timestamp__time_of_business │
    │ ---                 ┆ _day                        │
    │ datetime[μs]        ┆ ---                         │
    │                     ┆ str                         │
    ├─────────────────────┼─────────────────────────────┤
    │ 2024-01-15 08:00:00 ┆ before_hours                │
    │ 2024-01-15 10:30:00 ┆ during_hours                │
    │ 2024-01-15 18:00:00 ┆ after_hours                 │
    │ 2024-01-20 10:00:00 ┆ weekend                     │
    └─────────────────────┴─────────────────────────────┘

    **Example 3: All features with custom hours**

    >>> transformer = BusinessTimeFeatures(
    ...     subset=['timestamp'],
    ...     features=['is_business_hour', 'is_business_day', 'time_of_business_day', 'hour_of_business_day'],
    ...     business_hours_start=8,
    ...     business_hours_end=18
    ... )
    >>> result = transformer.fit_transform(X)
    """

    subset: Optional[List[str]] = None
    business_hours_start: int = 9
    business_hours_end: int = 17
    weekend_days: List[int] = [5, 6]
    features: List[str] = [
        "is_business_hour",
        "is_business_day",
        "time_of_business_day",
    ]
    drop_columns: bool = False

    @field_validator("business_hours_start", "business_hours_end")
    def check_hours(cls, hour):
        if not 0 <= hour <= 23:
            raise ValueError(f"Hour must be between 0 and 23, got {hour}")
        return hour

    @field_validator("weekend_days")
    def check_weekend_days(cls, days):
        for day in days:
            if not 0 <= day <= 6:
                raise ValueError(f"Weekend day must be between 0 and 6, got {day}")
        return days

    @field_validator("features")
    def check_features(cls, features):
        valid_features = [
            "is_business_hour",
            "is_business_day",
            "time_of_business_day",
            "hour_of_business_day",
        ]
        for feature in features:
            if feature not in valid_features:
                raise ValueError(
                    f"Feature '{feature}' is not supported. "
                    f"Supported features: {valid_features}"
                )
        return features

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "BusinessTimeFeatures":
        """Fit the transformer by identifying datetime columns if not specified.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        BusinessTimeFeatures
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in dict(zip(X.columns, X.dtypes)).items()
                if dtype == pl.Datetime or dtype == pl.Date
            ]

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating business time features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with business time features.
        """
        new_columns = []

        for col in self.subset:
            hour = pl.col(col).dt.hour()
            weekday = pl.col(col).dt.weekday()

            if "is_business_hour" in self.features:
                is_business_hour = (
                    (hour >= self.business_hours_start) & (hour < self.business_hours_end)
                ).alias(f"{col}__is_business_hour")
                new_columns.append(is_business_hour)

            if "is_business_day" in self.features:
                is_business_day = (~weekday.is_in(self.weekend_days)).alias(
                    f"{col}__is_business_day"
                )
                new_columns.append(is_business_day)

            if "time_of_business_day" in self.features:
                time_of_day = (
                    pl.when(weekday.is_in(self.weekend_days))
                    .then(pl.lit("weekend"))
                    .when(hour < self.business_hours_start)
                    .then(pl.lit("before_hours"))
                    .when(hour < self.business_hours_end)
                    .then(pl.lit("during_hours"))
                    .otherwise(pl.lit("after_hours"))
                ).alias(f"{col}__time_of_business_day")
                new_columns.append(time_of_day)

            if "hour_of_business_day" in self.features:
                hour_of_biz_day = (
                    pl.when((hour >= self.business_hours_start) & (hour < self.business_hours_end))
                    .then(hour - self.business_hours_start)
                    .otherwise(None)
                ).alias(f"{col}__hour_of_business_day")
                new_columns.append(hour_of_biz_day)

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
