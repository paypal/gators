from typing import Callable, Dict, List, Optional

import polars as pl
from pydantic import BaseModel, ValidationInfo, field_validator
from sklearn.base import BaseEstimator, TransformerMixin

COMPONENT_FUNCTIONS: Dict[str, Callable] = {
    "century": lambda x: x.dt.century(),
    "year": lambda x: x.dt.year(),
    "semester": lambda x: pl.when(x.dt.quarter() <= 2).then(1).otherwise(2),
    "quarter": lambda x: x.dt.quarter(),
    "month": lambda x: x.dt.month(),
    "week": lambda x: x.dt.week(),
    "day_of_week": lambda x: x.dt.weekday(),
    "day_of_month": lambda x: x.dt.day(),
    "day_of_year": lambda x: x.dt.ordinal_day(),
    "weekend": lambda x: x.dt.weekday().is_in([6, 7]),  # Saturday=6, Sunday=7
    "leap_year": lambda x: x.dt.is_leap_year(),
    "hour": lambda x: x.dt.hour(),
    "minute": lambda x: x.dt.minute(),
    "second": lambda x: x.dt.second(),
}


class OrdinalFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Generates ordinal features from datetime columns.

    Ordinal features extract standard temporal components (year, month, hour, etc.)
    as integer values from datetime columns.

    Parameters
    ----------
    subset : Optional[List[str]], optional
        List of datetime columns to extract features from. If None, all datetime columns
        in the dataframe will be used, by default None.
    components : List[str]
        List of date and time components to extract.
        Valid values: 'century', 'year', 'semester', 'quarter', 'month', 'week',
        'day_of_week', 'day_of_month', 'day_of_year', 'weekend', 'leap_year',
        'hour', 'minute', 'second'.
    drop_columns : bool, optional
        Whether to drop the original datetime columns after feature extraction, by default False.

    Examples
    --------
    >>> from datetime_ordinal_features import OrdinalFeatures
    >>> import polars as pl

    >>> X ={'date': ['2023-01-01', '2023-02-01', '2023-03-01'],
    ...         'datetime': ['2023-01-01T00:00:00', '2023-02-01T12:00:00', '2023-03-01T23:59:59']}
    >>> X = pl.DataFrame(X).with_columns([
    ...     pl.col('date').str.strptime(pl.Date, '%Y-%m-%d'),
    ...     pl.col('datetime').str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M:%S')
    ... ])

    **Example 1: Extract year and month from all datetime columns**

    >>> transformer = DatetimeOrdinalFeatures(components=['year', 'month'], drop_columns=True)
    >>> transformer.fit(X)
    DatetimeOrdinalFeatures(components=['year', 'month'], drop_columns=True)
    >>> result = transformer.transform(X)
    >>> result
    shape: (3, 4)
    ┌────────────┬──────────────┬────────────────┬─────────────────┐
    │ date__year │ date__month  │ datetime__year │ datetime__month │
    │    i64     │     i64      │      i64       │      i64        │
    ├────────────┼──────────────┼────────────────┼─────────────────┤
    │   2023     │      1       │      2023      │       1         │
    │   2023     │      2       │      2023      │       2         │
    │   2023     │      3       │      2023      │       3         │
    └────────────┴──────────────┴────────────────┴─────────────────┘

    **Example 2: Extract from specific column, keep original**

    >>> transformer = DatetimeOrdinalFeatures(subset=['date'], components=['month', 'weekend'], drop_columns=False)
    >>> transformer.fit(X)
    DatetimeOrdinalFeatures(subset=['date'], components=['month', 'weekend'], drop_columns=False)
    >>> result = transformer.transform(X)
    >>> result
    shape: (3, 5)
    ┌────────────┬─────────────────────┬──────────────┬───────────────┐
    │    date    │       datetime      │ date__month  │ date__weekend │
    │    date    │      datetime       │     i64      │      bool     │
    ├────────────┼─────────────────────┼──────────────┼───────────────┤
    │ 2023-01-01 │ 2023-01-01T00:00:00 │      1       │     true      │
    │ 2023-02-01 │ 2023-02-01T12:00:00 │      2       │     false     │
    │ 2023-03-01 │ 2023-03-01T23:59:59 │      3       │     false     │
    └────────────┴─────────────────────┴──────────────┴───────────────┘
    """

    subset: Optional[List[str]] = None
    components: List[str]
    drop_columns: bool = False

    @field_validator("components")
    def check_components(cls, components, info: ValidationInfo):
        for component in components:
            if component not in COMPONENT_FUNCTIONS:
                raise ValueError(
                    f"{component} is not a valid component. "
                    f"Valid components are: {list(COMPONENT_FUNCTIONS.keys())}"
                )
        return components

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "OrdinalFeatures":
        """Fit the transformer.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        OrdinalFeatures
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
        """Transform the input DataFrame by extracting ordinal features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with ordinal features.
        """
        # Parse datetime columns only if needed
        datetime_conversions = []
        for col in self.subset:
            if X.schema[col] != pl.Datetime:
                datetime_conversions.append(pl.col(col).str.to_datetime().alias(col))

        if datetime_conversions:
            X = X.with_columns(datetime_conversions)

        # Build all features in one list to minimize with_columns calls
        all_features = [
            COMPONENT_FUNCTIONS[comp](pl.col(col)).alias(f"{col}__{comp}")
            for col in self.subset
            for comp in self.components
        ]

        X = X.with_columns(all_features)

        return X.drop(self.subset) if self.drop_columns else X
