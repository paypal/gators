from typing import Dict, List, Literal, Optional

import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer

EARTH_RADIUS = {
    "km": 6371.0,
    "miles": 3958.8,
    "meters": 6371000.0,
    "feet": 20902231.0,
}


class DistanceFeatures(_BaseTransformer):
    """
    Calculates distances between geographic coordinate pairs.

    This transformer computes distances between consecutive pairs of latitude/longitude coordinates
    using different distance metrics (euclidean, manhattan, haversine) and units (km, miles, meters, feet).

    For fraud detection, distance features are valuable for:

    - Detecting location anomalies (billing vs shipping address distance)
    - Identifying suspicious IP geolocation patterns
    - Flagging transactions far from customer's typical location
    - Calculating travel feasibility (transaction velocity checks)

    Parameters
    ----------
    lats : List[str]
        List of latitude column names. Must have at least 2 elements.
        Coordinates are paired sequentially: (lats[0], longs[0]) to (lats[1], longs[1]), etc.
    longs : List[str]
        List of longitude column names. Must have same length as lats.
    unit : Literal["km", "miles", "meters", "feet"], default="km"
        Unit for distance output.
    method : Literal["euclidean", "manhattan", "haversine"], default="haversine"
        Distance calculation method:
        - 'haversine': Great-circle distance on a sphere (recommended for lat/long)
        - 'euclidean': Straight-line distance
        - 'manhattan': Sum of absolute differences (taxicab distance)
    drop_columns : bool, default=True
        Whether to drop the original coordinate columns.
    new_column_names : Optional[List[str]], default=None
        Custom names for distance columns. If None, uses pattern:
        'distance__{lat1}_to_{lat2}__{method}_{unit}'

    Examples
    --------
    >>> from gators.feature_generation import DistanceFeatures
    >>> import polars as pl

    **Example 1: Haversine distance between two locations**

    >>> X = pl.DataFrame({
    ...     'billing_lat': [40.7128, 34.0522, 41.8781],
    ...     'billing_long': [-74.0060, -118.2437, -87.6298],
    ...     'shipping_lat': [40.7580, 34.0522, 42.3601],
    ...     'shipping_long': [-73.9855, -118.2437, -71.0589]
    ... })
    >>> transformer = DistanceFeatures(
    ...     lats=['billing_lat', 'shipping_lat'],
    ...     longs=['billing_long', 'shipping_long'],
    ...     method='haversine',
    ...     unit='km'
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['distance__billing_lat_to_shipping_lat__haversine_km']
    >>> result['distance__billing_lat_to_shipping_lat__haversine_km'][0]
    5.376...

    **Example 2: Multiple distance pairs**

    >>> X = pl.DataFrame({
    ...     'home_lat': [40.7128, 34.0522],
    ...     'home_long': [-74.0060, -118.2437],
    ...     'work_lat': [40.7580, 34.0700],
    ...     'work_long': [-73.9855, -118.3000],
    ...     'shop_lat': [40.7489, 34.0800],
    ...     'shop_long': [-73.9680, -118.3500]
    ... })
    >>> transformer = DistanceFeatures(
    ...     lats=['home_lat', 'work_lat', 'shop_lat'],
    ...     longs=['home_long', 'work_long', 'shop_long'],
    ...     method='haversine',
    ...     unit='miles',
    ...     drop_columns=False
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['home_lat', 'home_long', 'work_lat', 'work_long', 'shop_lat', 'shop_long',
     'distance__home_lat_to_work_lat__haversine_miles',
     'distance__work_lat_to_shop_lat__haversine_miles']

    **Example 3: Euclidean distance**

    >>> X = pl.DataFrame({
    ...     'x1': [0.0, 1.0, 2.0],
    ...     'y1': [0.0, 1.0, 2.0],
    ...     'x2': [3.0, 4.0, 5.0],
    ...     'y2': [4.0, 5.0, 6.0]
    ... })
    >>> transformer = DistanceFeatures(
    ...     lats=['x1', 'x2'],
    ...     longs=['y1', 'y2'],
    ...     method='euclidean',
    ...     unit='meters'
    ... )
    >>> result = transformer.fit_transform(X)
    """

    lats: List[str]
    longs: List[str]
    unit: Literal["km", "miles", "meters", "feet"] = "km"
    method: Literal["euclidean", "manhattan", "haversine"] = "haversine"
    drop_columns: bool = True
    new_column_names: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("lats")
    def check_lats_length(cls, lats):
        if len(lats) < 2:
            raise ValueError("lats must contain at least 2 latitude column names")
        return lats

    @field_validator("longs")
    def check_longs_length(cls, longs, info):
        lats = info.data.get("lats", [])
        if len(longs) != len(lats):
            raise ValueError(
                f"longs must have same length as lats. Got {len(longs)} longs and {len(lats)} lats"
            )
        return longs

    @field_validator("new_column_names")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            lats = info.data.get("lats", [])
            expected_length = len(lats) - 1  # n points create n-1 distances
            if len(new_column_names) != expected_length:
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match number of distance pairs ({expected_length})"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "DistanceFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        DistanceFeatures
            Fitted transformer instance.
        """
        # Generate default column names
        default_names = []
        for i in range(len(self.lats) - 1):
            lat1, lat2 = self.lats[i], self.lats[i + 1]
            default_names.append(f"distance__{lat1}_to_{lat2}__{self.method}_{self.unit}")

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by calculating distance features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with distance features.
        """
        new_columns = []

        for i in range(len(self.lats) - 1):
            lat1_col, lat2_col = self.lats[i], self.lats[i + 1]
            long1_col, long2_col = self.longs[i], self.longs[i + 1]

            default_name = f"distance__{lat1_col}_to_{lat2_col}__{self.method}_{self.unit}"
            new_col_name = self._column_mapping[default_name]

            if self.method == "haversine":
                # Validate lat/long ranges for haversine
                lat1 = pl.col(lat1_col)
                lat2 = pl.col(lat2_col)
                long1 = pl.col(long1_col)
                long2 = pl.col(long2_col)

                # Check if any coordinate is null or out of range
                valid_coords = (
                    lat1.is_not_null()
                    & lat2.is_not_null()
                    & long1.is_not_null()
                    & long2.is_not_null()
                    & (lat1 >= -90)
                    & (lat1 <= 90)
                    & (lat2 >= -90)
                    & (lat2 <= 90)
                    & (long1 >= -180)
                    & (long1 <= 180)
                    & (long2 >= -180)
                    & (long2 <= 180)
                )

                # Haversine formula
                # Convert to radians
                deg_to_rad = 3.141592653589793 / 180.0
                lat1_rad = lat1 * deg_to_rad
                lat2_rad = lat2 * deg_to_rad
                long1_rad = long1 * deg_to_rad
                long2_rad = long2 * deg_to_rad

                dlat = lat2_rad - lat1_rad
                dlong = long2_rad - long1_rad

                # Haversine formula: a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlong/2)
                a = (dlat / 2.0).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (
                    dlong / 2.0
                ).sin().pow(2)

                # c = 2 * asin(√a)
                c = 2.0 * a.sqrt().arcsin()

                # distance = R * c
                distance = c * EARTH_RADIUS[self.unit]

                # Return null for invalid coordinates
                distance_expr = pl.when(valid_coords).then(distance).otherwise(None)

            elif self.method == "euclidean":
                lat1 = pl.col(lat1_col)
                lat2 = pl.col(lat2_col)
                long1 = pl.col(long1_col)
                long2 = pl.col(long2_col)

                # Euclidean distance: √((lat2-lat1)² + (long2-long1)²)
                distance_expr = ((lat2 - lat1).pow(2) + (long2 - long1).pow(2)).sqrt()

            elif self.method == "manhattan":
                lat1 = pl.col(lat1_col)
                lat2 = pl.col(lat2_col)
                long1 = pl.col(long1_col)
                long2 = pl.col(long2_col)

                # Manhattan distance: |lat2-lat1| + |long2-long1|
                distance_expr = (lat2 - lat1).abs() + (long2 - long1).abs()

            new_columns.append(distance_expr.alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = list(set(self.lats + self.longs))
            X = X.drop(columns_to_drop)

        return X
