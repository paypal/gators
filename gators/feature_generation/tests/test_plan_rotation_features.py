"""Tests for PlanRotationFeatures transformer."""

import polars as pl
import pytest

from gators.feature_generation.plan_rotation_features import PlanRotationFeatures


def test_basic_rotation():
    """Test basic plan rotation with single pair and angle."""
    X =pl.DataFrame({"X": [200.0, 210.0], "Y": [140.0, 160.0], "Z": [100.0, 125.0]})

    transformer = PlanRotationFeatures(columns=[["X", "Y"]], angles=[45.0])
    result = transformer.fit_transform(X)

    # Check new columns are created
    assert "XY_x45" in result.columns
    assert "XY_y45" in result.columns
    # Original columns should still be there
    assert "X" in result.columns
    assert "Y" in result.columns


def test_multiple_angles():
    """Test rotation with multiple angles."""
    X =pl.DataFrame({"X": [10.0, 20.0], "Y": [5.0, 10.0]})

    transformer = PlanRotationFeatures(columns=[["X", "Y"]], angles=[30.0, 45.0, 60.0])
    result = transformer.fit_transform(X)

    # Check all angle variations are created
    assert "XY_x30" in result.columns
    assert "XY_y30" in result.columns
    assert "XY_x45" in result.columns
    assert "XY_y45" in result.columns
    assert "XY_x60" in result.columns
    assert "XY_y60" in result.columns


def test_multiple_column_pairs():
    """Test rotation with multiple column pairs."""
    X =pl.DataFrame({"X": [10.0], "Y": [5.0], "Z": [3.0]})

    transformer = PlanRotationFeatures(columns=[["X", "Y"], ["X", "Z"]], angles=[45.0])
    result = transformer.fit_transform(X)

    # Check both pairs are rotated
    assert "XY_x45" in result.columns
    assert "XY_y45" in result.columns
    assert "XZ_x45" in result.columns
    assert "XZ_y45" in result.columns


def test_fractional_angle():
    """Test rotation with fractional angle."""
    X =pl.DataFrame({"X": [10.0], "Y": [5.0]})

    transformer = PlanRotationFeatures(columns=[["X", "Y"]], angles=[22.5])
    result = transformer.fit_transform(X)

    # Check fractional angle column name
    assert "XY_x22.5" in result.columns
    assert "XY_y22.5" in result.columns


def test_sklearn_compatibility():
    """Test sklearn pipeline compatibility."""
    X =pl.DataFrame({"X": [10.0], "Y": [5.0]})

    transformer = PlanRotationFeatures(columns=[["X", "Y"]], angles=[45.0])

    # Test fit returns self
    result = transformer.fit(X)
    assert result is transformer

    # Test fit_transform
    result = transformer.fit_transform(X)
    assert isinstance(result, pl.DataFrame)


def test_column_names_computed():
    """Test that column names are computed correctly after initialization."""
    transformer = PlanRotationFeatures(
        columns=[["X", "Y"], ["A", "B"]], angles=[30.0, 60.0]
    )

    # Check column_names are computed
    assert len(transformer.column_names) == 8  # 2 pairs * 2 angles * 2 (cos/sin)
    assert "XY_x30" in transformer.column_names
    assert "XY_y30" in transformer.column_names
    assert "AB_x60" in transformer.column_names
    assert "AB_y60" in transformer.column_names


def test_flatten_columns_computed():
    """Test that flatten_columns is computed during fit."""
    X =pl.DataFrame({"X": [10.0], "Y": [5.0], "Z": [3.0]})

    transformer = PlanRotationFeatures(columns=[["X", "Y"], ["Y", "Z"]], angles=[45.0])
    transformer.fit(X)

    # Check flatten_columns contains all columns
    assert transformer.flatten_columns == ["X", "Y", "Y", "Z"]


def test_zero_angle():
    """Test rotation with zero angle (identity transformation)."""
    X =pl.DataFrame({"X": [10.0], "Y": [5.0]})

    transformer = PlanRotationFeatures(columns=[["X", "Y"]], angles=[0.0])
    result = transformer.fit_transform(X)

    # At 0 degrees: cos(0)=1, sin(0)=0
    # x_rotated = x*1 - y*0 = x
    # y_rotated = x*0 + y*1 = y
    assert "XY_x0" in result.columns
    assert "XY_y0" in result.columns
    assert result["XY_x0"][0] == pytest.approx(10.0, rel=1e-10)
    assert result["XY_y0"][0] == pytest.approx(5.0, rel=1e-10)


if __name__ == "__main__":
    pytest.main()
