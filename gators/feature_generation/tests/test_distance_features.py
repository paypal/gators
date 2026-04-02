import numpy as np
import polars as pl
import pytest

from gators.feature_generation.distance_features import DistanceFeatures


@pytest.fixture
def simple_coordinates():
    """Simple coordinate pairs for testing."""
    return pl.DataFrame(
        {
            "lat1": [40.7128, 34.0522, 51.5074],
            "long1": [-74.0060, -118.2437, -0.1278],
            "lat2": [40.7580, 34.0522, 48.8566],
            "long2": [-73.9855, -118.2437, 2.3522],
        }
    )


@pytest.fixture
def multiple_pairs():
    """Multiple coordinate pairs."""
    return pl.DataFrame(
        {
            "home_lat": [40.7128, 34.0522],
            "home_long": [-74.0060, -118.2437],
            "work_lat": [40.7580, 34.0700],
            "work_long": [-73.9855, -118.3000],
            "shop_lat": [40.7489, 34.0800],
            "shop_long": [-73.9680, -118.3500],
        }
    )


@pytest.fixture
def null_coordinates():
    """Coordinates with null values."""
    return pl.DataFrame(
        {
            "lat1": [40.7128, None, 51.5074, 34.0522],
            "long1": [-74.0060, -118.2437, None, -118.2437],
            "lat2": [40.7580, 34.0522, 48.8566, None],
            "long2": [-73.9855, -118.2437, 2.3522, -118.2437],
        }
    )


@pytest.fixture
def invalid_coordinates():
    """Coordinates with invalid lat/long values."""
    return pl.DataFrame(
        {
            "lat1": [40.7128, 95.0, -95.0, 40.7128],
            "long1": [-74.0060, -118.2437, 0.0, 200.0],
            "lat2": [40.7580, 34.0522, 48.8566, 40.7580],
            "long2": [-73.9855, -118.2437, 2.3522, -73.9855],
        }
    )


# Validator tests
def test_lats_too_short():
    """Test that lats must have at least 2 elements."""
    with pytest.raises(ValueError, match="lats must contain at least 2"):
        DistanceFeatures(lats=["lat1"], longs=["long1"])


def test_longs_length_mismatch():
    """Test that longs must have same length as lats."""
    with pytest.raises(
        ValueError, match="longs must have same length as lats"
    ):
        DistanceFeatures(lats=["lat1", "lat2"], longs=["long1"])


def test_new_column_names_wrong_length():
    """Test that new_column_names must match number of distance pairs."""
    with pytest.raises(
        ValueError, match="Length of new_column_names .* must match"
    ):
        DistanceFeatures(
            lats=["lat1", "lat2", "lat3"],
            longs=["long1", "long2", "long3"],
            new_column_names=["dist1"],  # Should be 2 names
        )


# Haversine distance tests
def test_haversine_km(simple_coordinates):
    """Test haversine distance in kilometers."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="km",
        drop_columns=False,
    )
    result = transformer.fit_transform(simple_coordinates)

    # Check column exists
    assert "distance__lat1_to_lat2__haversine_km" in result.columns

    # NYC to slightly north Manhattan: ~5.4 km
    assert 5.0 < result["distance__lat1_to_lat2__haversine_km"][0] < 6.0

    # Same location: 0 km
    assert abs(result["distance__lat1_to_lat2__haversine_km"][1]) < 0.001

    # London to Paris: ~340 km
    assert 330 < result["distance__lat1_to_lat2__haversine_km"][2] < 350


def test_haversine_miles(simple_coordinates):
    """Test haversine distance in miles."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="miles",
        drop_columns=False,
    )
    result = transformer.fit_transform(simple_coordinates)

    # NYC to slightly north Manhattan: ~3.3 miles
    assert 3.0 < result["distance__lat1_to_lat2__haversine_miles"][0] < 4.0

    # London to Paris: ~210 miles
    assert 200 < result["distance__lat1_to_lat2__haversine_miles"][2] < 220


def test_haversine_meters(simple_coordinates):
    """Test haversine distance in meters."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="meters",
        drop_columns=False,
    )
    result = transformer.fit_transform(simple_coordinates)

    # NYC to slightly north Manhattan: ~5400 meters
    assert 5000 < result["distance__lat1_to_lat2__haversine_meters"][0] < 6000

    # London to Paris: ~340000 meters
    assert 330000 < result["distance__lat1_to_lat2__haversine_meters"][2] < 350000


def test_haversine_feet(simple_coordinates):
    """Test haversine distance in feet."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="feet",
        drop_columns=False,
    )
    result = transformer.fit_transform(simple_coordinates)

    # NYC to slightly north Manhattan: ~17700 feet
    assert 16000 < result["distance__lat1_to_lat2__haversine_feet"][0] < 20000

    # London to Paris: ~1115000 feet
    assert 1000000 < result["distance__lat1_to_lat2__haversine_feet"][2] < 1200000


# Euclidean distance tests
def test_euclidean():
    """Test euclidean distance calculation."""
    X = pl.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0],
            "y1": [0.0, 1.0, 2.0],
            "x2": [3.0, 4.0, 5.0],
            "y2": [4.0, 5.0, 6.0],
        }
    )
    transformer = DistanceFeatures(
        lats=["x1", "x2"],
        longs=["y1", "y2"],
        method="euclidean",
        unit="meters",
        drop_columns=False,
    )
    result = transformer.fit_transform(X)

    # Euclidean: sqrt((3-0)^2 + (4-0)^2) = sqrt(25) = 5
    assert abs(result["distance__x1_to_x2__euclidean_meters"][0] - 5.0) < 0.001

    # Euclidean: sqrt((4-1)^2 + (5-1)^2) = sqrt(25) = 5
    assert abs(result["distance__x1_to_x2__euclidean_meters"][1] - 5.0) < 0.001


# Manhattan distance tests
def test_manhattan():
    """Test manhattan distance calculation."""
    X = pl.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0],
            "y1": [0.0, 1.0, 2.0],
            "x2": [3.0, 4.0, 5.0],
            "y2": [4.0, 5.0, 6.0],
        }
    )
    transformer = DistanceFeatures(
        lats=["x1", "x2"],
        longs=["y1", "y2"],
        method="manhattan",
        unit="km",
        drop_columns=False,
    )
    result = transformer.fit_transform(X)

    # Manhattan: |3-0| + |4-0| = 7
    assert abs(result["distance__x1_to_x2__manhattan_km"][0] - 7.0) < 0.001

    # Manhattan: |4-1| + |5-1| = 7
    assert abs(result["distance__x1_to_x2__manhattan_km"][1] - 7.0) < 0.001


# Multiple pairs tests
def test_multiple_pairs(multiple_pairs):
    """Test creating multiple distance features from 3+ coordinate pairs."""
    transformer = DistanceFeatures(
        lats=["home_lat", "work_lat", "shop_lat"],
        longs=["home_long", "work_long", "shop_long"],
        method="haversine",
        unit="miles",
        drop_columns=False,
    )
    result = transformer.fit_transform(multiple_pairs)

    # Should create 2 distance columns (3 points → 2 distances)
    assert "distance__home_lat_to_work_lat__haversine_miles" in result.columns
    assert "distance__work_lat_to_shop_lat__haversine_miles" in result.columns

    # Original columns should still exist
    assert "home_lat" in result.columns
    assert "shop_long" in result.columns


# Custom column names tests
def test_custom_column_names(simple_coordinates):
    """Test using custom column names."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="km",
        new_column_names=["distance_km"],
        drop_columns=False,
    )
    result = transformer.fit_transform(simple_coordinates)

    assert "distance_km" in result.columns
    assert "distance__lat1_to_lat2__haversine_km" not in result.columns


def test_custom_column_names_multiple(multiple_pairs):
    """Test custom column names with multiple pairs."""
    transformer = DistanceFeatures(
        lats=["home_lat", "work_lat", "shop_lat"],
        longs=["home_long", "work_long", "shop_long"],
        method="euclidean",
        unit="meters",
        new_column_names=["home_to_work", "work_to_shop"],
        drop_columns=False,
    )
    result = transformer.fit_transform(multiple_pairs)

    assert "home_to_work" in result.columns
    assert "work_to_shop" in result.columns


# Drop columns tests
def test_drop_columns_true(simple_coordinates):
    """Test dropping original coordinate columns."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="km",
        drop_columns=True,
    )
    result = transformer.fit_transform(simple_coordinates)

    # Original columns should be dropped
    assert "lat1" not in result.columns
    assert "long1" not in result.columns
    assert "lat2" not in result.columns
    assert "long2" not in result.columns

    # Distance column should exist
    assert "distance__lat1_to_lat2__haversine_km" in result.columns


def test_drop_columns_false(simple_coordinates):
    """Test keeping original coordinate columns."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="km",
        drop_columns=False,
    )
    result = transformer.fit_transform(simple_coordinates)

    # Original columns should still exist
    assert "lat1" in result.columns
    assert "long1" in result.columns
    assert "lat2" in result.columns
    assert "long2" in result.columns


# Null handling tests
def test_haversine_null_coordinates(null_coordinates):
    """Test that null coordinates return null distances for haversine."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="km",
        drop_columns=False,
    )
    result = transformer.fit_transform(null_coordinates)

    # Row 0: valid coordinates, should have distance
    assert result["distance__lat1_to_lat2__haversine_km"][0] is not None

    # Row 1: lat1 is null, should return null
    assert result["distance__lat1_to_lat2__haversine_km"][1] is None

    # Row 2: long1 is null, should return null
    assert result["distance__lat1_to_lat2__haversine_km"][2] is None

    # Row 3: lat2 is null, should return null
    assert result["distance__lat1_to_lat2__haversine_km"][3] is None


def test_euclidean_null_coordinates(null_coordinates):
    """Test that euclidean distance handles nulls (propagates null)."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="euclidean",
        unit="km",
        drop_columns=False,
    )
    result = transformer.fit_transform(null_coordinates)

    # Row 0: valid coordinates, should have distance
    assert result["distance__lat1_to_lat2__euclidean_km"][0] is not None

    # Other rows: nulls propagate in arithmetic operations
    assert result["distance__lat1_to_lat2__euclidean_km"][1] is None
    assert result["distance__lat1_to_lat2__euclidean_km"][2] is None
    assert result["distance__lat1_to_lat2__euclidean_km"][3] is None


# Invalid coordinate tests (for haversine validation)
def test_haversine_invalid_coordinates(invalid_coordinates):
    """Test that invalid coordinates return null for haversine."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="km",
        drop_columns=False,
    )
    result = transformer.fit_transform(invalid_coordinates)

    # Row 0: valid coordinates
    assert result["distance__lat1_to_lat2__haversine_km"][0] is not None

    # Row 1: lat1 = 95 (> 90), should return null
    assert result["distance__lat1_to_lat2__haversine_km"][1] is None

    # Row 2: lat1 = -95 (< -90), should return null
    assert result["distance__lat1_to_lat2__haversine_km"][2] is None

    # Row 3: long1 = 200 (> 180), should return null
    assert result["distance__lat1_to_lat2__haversine_km"][3] is None


# Fit method tests
def test_fit_returns_self(simple_coordinates):
    """Test that fit returns the transformer instance."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
    )
    result = transformer.fit(simple_coordinates)
    assert result is transformer


def test_fit_with_y_parameter(simple_coordinates):
    """Test that fit works with y parameter (for compatibility)."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
    )
    y = pl.Series([0, 1, 0])
    result = transformer.fit(simple_coordinates, y)
    assert result is transformer


# Integration tests
def test_fit_transform_chain(simple_coordinates):
    """Test fit followed by transform."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="km",
    )
    transformer.fit(simple_coordinates)
    result = transformer.transform(simple_coordinates)

    assert "distance__lat1_to_lat2__haversine_km" in result.columns


def test_multiple_transformations(simple_coordinates):
    """Test transformer can be used multiple times."""
    transformer = DistanceFeatures(
        lats=["lat1", "lat2"],
        longs=["long1", "long2"],
        method="haversine",
        unit="km",
        drop_columns=False,
    )
    transformer.fit(simple_coordinates)

    result1 = transformer.transform(simple_coordinates)
    result2 = transformer.transform(simple_coordinates)

    # Results should be identical
    assert result1.equals(result2)
