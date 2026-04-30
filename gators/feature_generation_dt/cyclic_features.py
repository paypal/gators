from math import pi
from typing import Any, Callable, Dict, List, Optional

import polars as pl
from pydantic import ValidationInfo, field_validator

from ..transformer._base_transformer import _BaseTransformer

COMPONENT_FUNCTIONS: Dict[str, Callable[[Any], Any]] = {
    "semester": lambda x: pl.when(x.dt.quarter() <= 2).then(1).otherwise(2),
    "quarter": lambda x: x.dt.quarter(),
    "month": lambda x: x.dt.month(),
    "week": lambda x: x.dt.week(),
    "day_of_week": lambda x: x.dt.weekday(),
    "day_of_month": lambda x: x.dt.day(),
    "day_of_year": lambda x: x.dt.ordinal_day(),
    "hour": lambda x: x.dt.hour(),
    "minute": lambda x: x.dt.minute(),
    "second": lambda x: x.dt.second(),
}

TWO_PI = 2 * pi

CYCLIC_FACTORS: Dict[str, float] = {
    "month": TWO_PI / 12.0,
    "quarter": TWO_PI / 4.0,
    "semester": TWO_PI / 2.0,
    "week": TWO_PI / 52.0,
    "day_of_week": TWO_PI / 7.0,
    "day_of_year": TWO_PI / 366.0,
    "hour": TWO_PI / 24.0,
    "minute": TWO_PI / 60.0,
    "second": TWO_PI / 60.0,
}


class CyclicFeatures(_BaseTransformer):
    """
    Generates cyclic features from datetime columns using sine transformations with multiple phase angles.

    Cyclic features are useful for representing periodic temporal patterns (e.g., month, hour)
    where the start and end of the cycle should be close in feature space.

    Parameters
    ----------
    subset : Optional[List[str]], optional
        List of datetime columns to extract features from. If None, all datetime columns
        in the dataframe will be used, by default None.
    components : List[str]
        List of date and time components to extract cyclic features from.
        Valid values: 'semester', 'quarter', 'month', 'week', 'day_of_week',
        'day_of_month', 'day_of_year', 'hour', 'minute', 'second'.
    angles : List[float]
        List of phase shift angles in degrees. For each component, a sine feature will be
        generated for each angle. For example, [0, 45, 90, 135, 180] will create five features
        with 0°, 45°, 90°, 135°, and 180° phase shifts.
    drop_columns : bool, optional
        Whether to drop the original datetime columns after feature extraction, by default False.

    Examples
    --------
    >>> from datetime_cyclic_features import CyclicFeatures
    >>> import polars as pl

    >>> X ={'date': ['2023-01-01', '2023-02-01', '2023-03-01'],
    ...         'datetime': ['2023-01-01T00:00:00', '2023-02-01T12:00:00', '2023-03-01T23:59:59']}
    >>> X = pl.DataFrame(X).with_columns([
    ...     pl.col('date').str.strptime(pl.Date, '%Y-%m-%d'),
    ...     pl.col('datetime').str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M:%S')
    ... ])

    # Example: Generate cyclic features for month with multiple angles
    >>> transformer = DatetimeCyclicFeatures(subset=['date'], components=['month'], angles=[0, 45, 90, 135, 180], drop_columns=False)
    >>> transformer.fit(X)
    DatetimeCyclicFeatures(subset=['date'], components=['month'], angles=[0, 45, 90, 135, 180], drop_columns=False)
    >>> result = transformer.transform(X)
    >>> result
    shape: (3, 7)
    ┌────────────┬──────────────┬────────────┬────────────┬────────────┬─────────────┬─────────────┐
    │    date    │   datetime   │ date__mon… │ date__mon… │ date__mon… │ date__mon…  │ date__mon…  │
    │            │              │ th__sin0   │ th__sin45  │ th__sin90  │ th__sin135  │ th__sin180  │
    │    date    │   datetime   │    f64     │    f64     │    f64     │    f64      │    f64      │
    ├────────────┼──────────────┼────────────┼────────────┼────────────┼─────────────┼─────────────┤
    │ 2023-01-01 │ 2023-01-01…  │  0.500000  │  0.965926  │  0.866025  │  0.258819   │ -0.500000   │
    │ 2023-02-01 │ 2023-02-01…  │  0.866025  │  0.965926  │  0.500000  │ -0.258819   │ -0.866025   │
    │ 2023-03-01 │ 2023-03-01…  │  1.000000  │  0.707107  │  0.000000  │ -0.707107   │ -1.000000   │
    └────────────┴──────────────┴────────────┴────────────┴────────────┴─────────────┴─────────────┘
    """

    subset: Optional[List[str]] = None
    components: List[str]
    angles: List[float]
    drop_columns: bool = False

    @field_validator("components")
    def check_components(cls, components, info: ValidationInfo):
        valid_components = list(CYCLIC_FACTORS.keys()) + ["day_of_month"]
        for component in components:
            if component not in valid_components:
                raise ValueError(
                    f"{component} is not a valid cyclic component. "
                    f"Valid components are: {valid_components}"
                )
        return components

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CyclicFeatures":
        """Fit the transformer.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        CyclicFeatures
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype == pl.Datetime or dtype == pl.Date
            ]

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by extracting cyclic features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with cyclic features (sine and cosine).
        """
        if self.subset is None:
            return X

        # Parse datetime columns only if needed
        datetime_conversions = []
        for col in self.subset:
            if X.schema[col] != pl.Datetime:
                datetime_conversions.append(pl.col(col).str.to_datetime().alias(col))

        if datetime_conversions:
            X = X.with_columns(datetime_conversions)

        # Extract base components (ordinal values)
        all_features = [
            COMPONENT_FUNCTIONS[comp](pl.col(col)).alias(f"{col}__{comp}")
            for col in self.subset
            for comp in self.components
        ]

        # Add days_in_month for day_of_month cyclic features
        if "day_of_month" in self.components:
            all_features.extend(
                [
                    pl.col(col).dt.month_end().dt.day().alias(f"{col}__days_in_month")
                    for col in self.subset
                ]
            )

        X = X.with_columns(all_features)

        # Pre-filter cyclic components to avoid repeated checks
        cyclic_comps = [comp for comp in self.components if comp in CYCLIC_FACTORS]

        # Convert angles to radians
        angles_rad = [angle * pi / 180 for angle in self.angles]

        # Build complete cyclic features list in one go
        cyclic_features = []

        # Sin features for standard cyclic components with phase shifts
        for col in self.subset:
            for comp in cyclic_comps:
                for angle_deg, angle_rad in zip(self.angles, angles_rad):
                    angle_int = (
                        int(angle_deg) if angle_deg == int(angle_deg) else round(angle_deg, 2)
                    )
                    cyclic_features.append(
                        (CYCLIC_FACTORS[comp] * pl.col(f"{col}__{comp}") + angle_rad)
                        .sin()
                        .alias(f"{col}__{comp}__sin{angle_int}")
                    )

        # Special handling for day_of_month
        if "day_of_month" in self.components:
            comp = "day_of_month"
            factor = 2 * pi

            for col in self.subset:
                for angle_deg, angle_rad in zip(self.angles, angles_rad):
                    angle_int = (
                        int(angle_deg) if angle_deg == int(angle_deg) else round(angle_deg, 2)
                    )
                    cyclic_features.append(
                        (
                            factor * pl.col(f"{col}__{comp}") / pl.col(f"{col}__days_in_month")
                            + angle_rad
                        )
                        .sin()
                        .alias(f"{col}__{comp}__sin{angle_int}")
                    )

        X = X.with_columns(cyclic_features)

        # Drop base ordinal columns
        columns_to_drop = [f"{col}__{comp}" for col in self.subset for comp in self.components]

        # Drop temporary days_in_month columns
        if "day_of_month" in self.components:
            columns_to_drop.extend([f"{col}__days_in_month" for col in self.subset])

        X = X.drop(columns_to_drop)

        return X.drop(self.subset) if (self.drop_columns and self.subset is not None) else X
