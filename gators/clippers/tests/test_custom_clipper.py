import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.clippers.custom_clipper import CustomClipper


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with various values for testing."""
    return pl.DataFrame(
        {
            "age": [-5.0, 25.0, 150.0, 80.0],
            "salary": [-1000.0, 50000.0, 2000000.0, 75000.0],
            "temperature": [-100.0, 20.0, 150.0, 25.0],
            "category": ["A", "B", "C", "D"],
        }
    )


def test_custom_clipper_both_bounds_inplace(sample_dataframe):
    """Test clipping with both lower and upper bounds (inplace=True)."""
    clipper = CustomClipper(
        lower_bounds={"age": 0.0, "salary": 0.0},
        upper_bounds={"age": 120.0, "salary": 1000000.0},
        inplace=True,
    )
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Verify the shape
    assert transformed.shape == (4, 4)
    assert "category" in transformed.columns

    # Verify clipping for age
    assert transformed["age"][0] == 0.0  # -5 clipped to 0
    assert transformed["age"][1] == 25.0  # unchanged
    assert transformed["age"][2] == 120.0  # 150 clipped to 120
    assert transformed["age"][3] == 80.0  # unchanged

    # Verify clipping for salary
    assert transformed["salary"][0] == 0.0  # -1000 clipped to 0
    assert transformed["salary"][1] == 50000.0  # unchanged
    assert transformed["salary"][2] == 1000000.0  # 2000000 clipped to 1000000
    assert transformed["salary"][3] == 75000.0  # unchanged

    # Verify temperature is unchanged (no bounds specified)
    assert transformed["temperature"][0] == -100.0
    assert transformed["temperature"][2] == 150.0


def test_custom_clipper_not_inplace(sample_dataframe):
    """Test clipping with inplace=False."""
    clipper = CustomClipper(
        lower_bounds={"age": 0.0},
        upper_bounds={"age": 120.0},
        inplace=False,
    )
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have new column instead of original age
    assert "age__clip_custom" in transformed.columns
    assert "age" not in transformed.columns
    assert "salary" in transformed.columns  # unchanged column
    assert "category" in transformed.columns

    # Verify clipping
    assert transformed["age__clip_custom"][0] == 0.0
    assert transformed["age__clip_custom"][2] == 120.0


def test_custom_clipper_not_inplace_no_drop(sample_dataframe):
    """Test clipping with inplace=False and drop_columns=False."""
    clipper = CustomClipper(
        lower_bounds={"age": 0.0},
        upper_bounds={"age": 120.0},
        inplace=False,
        drop_columns=False,
    )
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Should have both original and clipped columns
    assert "age" in transformed.columns
    assert "age__clip_custom" in transformed.columns
    assert transformed.shape == (4, 5)

    # Original should be unchanged
    assert transformed["age"][0] == -5.0
    # Clipped should be bounded
    assert transformed["age__clip_custom"][0] == 0.0


def test_custom_clipper_only_lower_bounds(sample_dataframe):
    """Test clipping with only lower bounds."""
    clipper = CustomClipper(lower_bounds={"age": 0.0, "salary": 10000.0})
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Verify lower bounds applied
    assert transformed["age"][0] == 0.0  # -5 clipped to 0
    assert transformed["age"][2] == 150.0  # no upper bound, unchanged
    assert transformed["salary"][0] == 10000.0  # -1000 clipped to 10000
    assert transformed["salary"][2] == 2000000.0  # no upper bound, unchanged


def test_custom_clipper_only_upper_bounds(sample_dataframe):
    """Test clipping with only upper bounds."""
    clipper = CustomClipper(upper_bounds={"age": 100.0, "temperature": 50.0})
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Verify upper bounds applied
    assert transformed["age"][0] == -5.0  # no lower bound, unchanged
    assert transformed["age"][2] == 100.0  # 150 clipped to 100
    assert transformed["temperature"][2] == 50.0  # 150 clipped to 50
    assert transformed["temperature"][0] == -100.0  # no lower bound, unchanged


def test_custom_clipper_different_bounds_per_column(sample_dataframe):
    """Test different combinations of bounds for different columns."""
    clipper = CustomClipper(
        lower_bounds={"age": 0.0},  # Only lower for age
        upper_bounds={"salary": 100000.0, "temperature": 30.0},  # Only upper for these
    )
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Age: only lower bound
    assert transformed["age"][0] == 0.0
    assert transformed["age"][2] == 150.0

    # Salary: only upper bound
    assert transformed["salary"][0] == -1000.0
    assert transformed["salary"][2] == 100000.0

    # Temperature: only upper bound
    assert transformed["temperature"][0] == -100.0
    assert transformed["temperature"][2] == 30.0


def test_custom_clipper_fit_transform():
    """Test fit_transform method."""
    X = pl.DataFrame(
        {
            "A": [-10.0, 5.0, 20.0, 50.0],
            "B": [0.0, 100.0, 200.0, 300.0],
        }
    )
    clipper = CustomClipper(
        lower_bounds={"A": 0.0}, upper_bounds={"A": 30.0, "B": 250.0}, inplace=False
    )
    transformed = clipper.fit_transform(X)

    assert "A__clip_custom" in transformed.columns
    assert "B__clip_custom" in transformed.columns
    assert transformed["A__clip_custom"][0] == 0.0  # -10 clipped to 0
    assert transformed["A__clip_custom"][3] == 30.0  # 50 clipped to 30
    assert transformed["B__clip_custom"][3] == 250.0  # 300 clipped to 250


def test_custom_clipper_no_bounds_error():
    """Test that error is raised when no bounds are specified."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    clipper = CustomClipper()

    with pytest.raises(ValueError, match="At least one of lower_bounds or upper_bounds"):
        clipper.fit(X)


def test_custom_clipper_missing_column_error(sample_dataframe):
    """Test that error is raised when specified column doesn't exist."""
    clipper = CustomClipper(lower_bounds={"nonexistent_column": 0.0})

    with pytest.raises(ValueError, match="not found in DataFrame"):
        clipper.fit(sample_dataframe)


def test_custom_clipper_invalid_bounds_type():
    """Test parameter validation for invalid bounds types."""
    # Pydantic will raise validation errors for invalid types
    with pytest.raises(Exception):
        CustomClipper(lower_bounds="invalid")

    with pytest.raises(Exception):
        CustomClipper(lower_bounds={1: 0.0})

    with pytest.raises(Exception):
        CustomClipper(lower_bounds={"A": "invalid"})


def test_custom_clipper_single_column():
    """Test clipping with a single column."""
    X = pl.DataFrame({"A": [-5.0, 10.0, 100.0]})
    clipper = CustomClipper(lower_bounds={"A": 0.0}, upper_bounds={"A": 50.0}, inplace=True)
    clipper.fit(X)
    transformed = clipper.transform(X)

    assert transformed["A"][0] == 0.0
    assert transformed["A"][1] == 10.0
    assert transformed["A"][2] == 50.0
    assert len(transformed.columns) == 1


def test_custom_clipper_ignores_non_numeric():
    """Test that non-numeric columns are ignored."""
    X = pl.DataFrame(
        {"A": [-5.0, 10.0, 100.0], "B": ["x", "y", "z"], "C": [1, 2, 3]}  # Will be int64
    )
    clipper = CustomClipper(
        lower_bounds={"A": 0.0, "B": 0.0, "C": 0.0},  # B is non-numeric
        upper_bounds={"A": 50.0},
    )
    clipper.fit(X)

    # Should only clip numeric columns A and C
    assert set(clipper._columns) == {"A", "C"}
    assert "B" not in clipper._columns

    transformed = clipper.transform(X)
    assert transformed["A"][0] == 0.0
    assert transformed["A"][2] == 50.0
    assert transformed["B"][0] == "x"  # unchanged


def test_custom_clipper_no_clipping_needed():
    """Test on data where no values need clipping."""
    X = pl.DataFrame({"A": [10.0, 20.0, 30.0], "B": [50.0, 60.0, 70.0]})
    clipper = CustomClipper(lower_bounds={"A": 0.0}, upper_bounds={"A": 100.0})
    clipper.fit(X)
    transformed = clipper.transform(X)

    # All values should remain unchanged
    assert_frame_equal(transformed, X)


def test_custom_clipper_empty_bounds_dict():
    """Test with empty bounds dictionaries."""
    X = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    clipper = CustomClipper(lower_bounds={}, upper_bounds={"A": 2.5})
    clipper.fit(X)
    transformed = clipper.transform(X)

    # Only upper bound should be applied
    assert transformed["A"][2] == 2.5


def test_custom_clipper_preserves_other_columns(sample_dataframe):
    """Test that columns without bounds are completely unchanged."""
    clipper = CustomClipper(lower_bounds={"age": 0.0})
    clipper.fit(sample_dataframe)
    transformed = clipper.transform(sample_dataframe)

    # Temperature should be exactly the same
    assert_frame_equal(transformed.select("temperature"), sample_dataframe.select("temperature"))
    # Salary should be exactly the same
    assert_frame_equal(transformed.select("salary"), sample_dataframe.select("salary"))
