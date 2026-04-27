import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation import ScalarMathFeatures
from gators.pipeline import Pipeline


@pytest.fixture
def sample_X():
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "Age": [25, 30, 45, 12, 65],
            "Price": [100.0, 150.0, 200.0, 75.0, 300.0],
            "Temperature": [20.0, 25.0, 15.0, 30.0, 22.0],
            "Count": [10, 23, 37, 15, 48],
        }
    )


def test_initialization_valid():
    """Test that ScalarMathFeatures initializes correctly with valid inputs."""
    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "/", "scalar": 365},
            {"column": "Price", "op": "*", "scalar": 1.1},
        ]
    )
    assert len(transformer.operations) == 2
    assert transformer.new_column_names is None


def test_initialization_with_custom_names():
    """Test initialization with custom column names."""
    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "/", "scalar": 365},
            {"column": "Price", "op": "*", "scalar": 1.1},
        ],
        new_column_names=["Age_years", "Price_with_tax"],
    )
    assert transformer.new_column_names == ["Age_years", "Price_with_tax"]


def test_validation_empty_operations():
    """Test that empty operations list raises error."""
    with pytest.raises(ValueError, match="At least one operation must be specified"):
        ScalarMathFeatures(operations=[])


def test_validation_missing_column():
    """Test that missing 'column' key raises error."""
    with pytest.raises(ValueError, match="'column' key is required"):
        ScalarMathFeatures(operations=[{"op": "+", "scalar": 10}])


def test_validation_missing_op():
    """Test that missing 'op' key raises error."""
    with pytest.raises(ValueError, match="'op' key is required"):
        ScalarMathFeatures(operations=[{"column": "Age", "scalar": 10}])


def test_validation_missing_scalar():
    """Test that missing 'scalar' key raises error."""
    with pytest.raises(ValueError, match="'scalar' key is required"):
        ScalarMathFeatures(operations=[{"column": "Age", "op": "+"}])


def test_validation_invalid_operator():
    """Test that invalid operator raises error."""
    with pytest.raises(ValueError, match="not supported"):
        ScalarMathFeatures(operations=[{"column": "Age", "op": "&", "scalar": 10}])


def test_validation_non_numeric_scalar():
    """Test that non-numeric scalar raises error."""
    with pytest.raises(ValueError, match="must be numeric"):
        ScalarMathFeatures(operations=[{"column": "Age", "op": "+", "scalar": "10"}])


def test_validation_column_names_length_mismatch():
    """Test that mismatched column names length raises error."""
    with pytest.raises(ValueError, match="must match number of operations"):
        ScalarMathFeatures(
            operations=[
                {"column": "Age", "op": "+", "scalar": 10},
                {"column": "Price", "op": "*", "scalar": 2},
            ],
            new_column_names=["only_one_name"],
        )


def test_validation_duplicate_column_names():
    """Test that duplicate column names raise error."""
    with pytest.raises(ValueError, match="must be unique"):
        ScalarMathFeatures(
            operations=[
                {"column": "Age", "op": "+", "scalar": 10},
                {"column": "Price", "op": "*", "scalar": 2},
            ],
            new_column_names=["same_name", "same_name"],
        )


def test_validation_empty_column_names():
    """Test that empty column names list raises error."""
    with pytest.raises(ValueError, match="cannot be empty"):
        ScalarMathFeatures(
            operations=[{"column": "Age", "op": "+", "scalar": 10}],
            new_column_names=[],
        )


def test_fit_auto_naming(sample_X):
    """Test that fit generates column names automatically."""
    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "/", "scalar": 365},
            {"column": "Price", "op": "*", "scalar": 1.1},
        ]
    )
    transformer.fit(sample_X)

    assert transformer._generated_column_names == ["Age_div_365", "Price_mul_1.1"]


def test_fit_auto_naming_integer_float():
    """Test that auto-naming formats floats as integers when appropriate."""
    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "/", "scalar": 365.0},  # Should become 365
            {"column": "Price", "op": "*", "scalar": 1.5},  # Should stay 1.5
        ]
    )
    transformer.fit(pl.DataFrame({"Age": [25], "Price": [100.0]}))

    assert transformer._generated_column_names == ["Age_div_365", "Price_mul_1.5"]


def test_fit_custom_names(sample_X):
    """Test that fit uses custom column names when provided."""
    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "/", "scalar": 365},
            {"column": "Price", "op": "*", "scalar": 1.1},
        ],
        new_column_names=["Age_years", "Price_with_tax"],
    )
    transformer.fit(sample_X)

    assert transformer._generated_column_names == ["Age_years", "Price_with_tax"]


def test_transform_addition(sample_X):
    """Test addition operation."""
    transformer = ScalarMathFeatures(
        operations=[{"column": "Age", "op": "+", "scalar": 10}],
        new_column_names=["Age_plus_10"],
    )
    result = transformer.fit_transform(sample_X)

    expected = sample_X.with_columns((pl.col("Age") + 10).alias("Age_plus_10"))
    assert_frame_equal(result, expected)


def test_transform_subtraction(sample_X):
    """Test subtraction operation."""
    transformer = ScalarMathFeatures(
        operations=[{"column": "Temperature", "op": "-", "scalar": 5}],
        new_column_names=["Temp_minus_5"],
    )
    result = transformer.fit_transform(sample_X)

    expected = sample_X.with_columns((pl.col("Temperature") - 5).alias("Temp_minus_5"))
    assert_frame_equal(result, expected)


def test_transform_multiplication(sample_X):
    """Test multiplication operation."""
    transformer = ScalarMathFeatures(
        operations=[{"column": "Price", "op": "*", "scalar": 2}],
        new_column_names=["Price_doubled"],
    )
    result = transformer.fit_transform(sample_X)

    expected = sample_X.with_columns((pl.col("Price") * 2).alias("Price_doubled"))
    assert_frame_equal(result, expected)


def test_transform_division(sample_X):
    """Test division operation."""
    transformer = ScalarMathFeatures(
        operations=[{"column": "Price", "op": "/", "scalar": 100}],
        new_column_names=["Price_pct"],
    )
    result = transformer.fit_transform(sample_X)

    expected = sample_X.with_columns((pl.col("Price") / 100).alias("Price_pct"))
    assert_frame_equal(result, expected)


def test_transform_power(sample_X):
    """Test power operation."""
    transformer = ScalarMathFeatures(
        operations=[{"column": "Age", "op": "**", "scalar": 2}],
        new_column_names=["Age_squared"],
    )
    result = transformer.fit_transform(sample_X)

    expected = sample_X.with_columns((pl.col("Age") ** 2).alias("Age_squared"))
    assert_frame_equal(result, expected)


def test_transform_floor_division(sample_X):
    """Test floor division operation."""
    transformer = ScalarMathFeatures(
        operations=[{"column": "Age", "op": "//", "scalar": 10}],
        new_column_names=["Age_decade"],
    )
    result = transformer.fit_transform(sample_X)

    expected = sample_X.with_columns((pl.col("Age") // 10).alias("Age_decade"))
    assert_frame_equal(result, expected)


def test_transform_modulo(sample_X):
    """Test modulo operation."""
    transformer = ScalarMathFeatures(
        operations=[{"column": "Count", "op": "%", "scalar": 10}],
        new_column_names=["Count_mod_10"],
    )
    result = transformer.fit_transform(sample_X)

    expected = sample_X.with_columns((pl.col("Count") % 10).alias("Count_mod_10"))
    assert_frame_equal(result, expected)


def test_transform_multiple_operations(sample_X):
    """Test multiple operations at once."""
    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "/", "scalar": 365},
            {"column": "Price", "op": "*", "scalar": 1.1},
            {"column": "Temperature", "op": "+", "scalar": 273.15},
        ],
        new_column_names=["Age_years", "Price_with_tax", "Temp_kelvin"],
    )
    result = transformer.fit_transform(sample_X)

    expected = sample_X.with_columns(
        [
            (pl.col("Age") / 365).alias("Age_years"),
            (pl.col("Price") * 1.1).alias("Price_with_tax"),
            (pl.col("Temperature") + 273.15).alias("Temp_kelvin"),
        ]
    )
    assert_frame_equal(result, expected)


def test_transform_auto_naming(sample_X):
    """Test transform with auto-generated names."""
    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "+", "scalar": 10},
            {"column": "Price", "op": "*", "scalar": 2},
        ]
    )
    result = transformer.fit_transform(sample_X)

    # Check that auto-named columns exist
    assert "Age_plus_10" in result.columns
    assert "Price_mul_2" in result.columns

    # Check values
    expected = sample_X.with_columns(
        [
            (pl.col("Age") + 10).alias("Age_plus_10"),
            (pl.col("Price") * 2).alias("Price_mul_2"),
        ]
    )
    assert_frame_equal(result, expected)


def test_transform_all_operators(sample_X):
    """Test all supported operators."""
    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "+", "scalar": 5},
            {"column": "Age", "op": "-", "scalar": 5},
            {"column": "Age", "op": "*", "scalar": 2},
            {"column": "Age", "op": "/", "scalar": 5},
            {"column": "Age", "op": "**", "scalar": 2},
            {"column": "Age", "op": "//", "scalar": 10},
            {"column": "Age", "op": "%", "scalar": 10},
        ]
    )
    result = transformer.fit_transform(sample_X)

    # Check all columns were created
    assert "Age_plus_5" in result.columns
    assert "Age_minus_5" in result.columns
    assert "Age_mul_2" in result.columns
    assert "Age_div_5" in result.columns
    assert "Age_pow_2" in result.columns
    assert "Age_floordiv_10" in result.columns
    assert "Age_mod_10" in result.columns


def test_transform_preserves_original_columns(sample_X):
    """Test that original columns are preserved."""
    transformer = ScalarMathFeatures(operations=[{"column": "Age", "op": "+", "scalar": 10}])
    result = transformer.fit_transform(sample_X)

    # All original columns should still be present
    for col in sample_X.columns:
        assert col in result.columns


def test_pipeline_compatibility(sample_X):
    """Test compatibility with sklearn pipeline."""
    # from sklearn.pipeline import Pipeline

    transformer = ScalarMathFeatures(
        operations=[
            {"column": "Age", "op": "/", "scalar": 365},
            {"column": "Price", "op": "*", "scalar": 1.1},
        ]
    )

    # Should work in a pipeline
    pipe = Pipeline(steps=[("scalar_math", transformer)])
    result = pipe.fit_transform(sample_X)

    assert isinstance(result, pl.DataFrame)
    assert "Age_div_365" in result.columns
    assert "Price_mul_1.1" in result.columns


def test_operator_name_mapping():
    """Test that operator names are mapped correctly."""
    from gators.feature_generation.scalar_math_features import OPERATOR_NAMES

    assert OPERATOR_NAMES["+"] == "plus"
    assert OPERATOR_NAMES["-"] == "minus"
    assert OPERATOR_NAMES["*"] == "mul"
    assert OPERATOR_NAMES["/"] == "div"
    assert OPERATOR_NAMES["**"] == "pow"
    assert OPERATOR_NAMES["//"] == "floordiv"
    assert OPERATOR_NAMES["%"] == "mod"


def test_empty_new_column_names_raises_error():
    """Test that empty new_column_names list raises ValueError."""
    X = pl.DataFrame({"A": [10, 20, 30]})

    with pytest.raises(ValueError, match="Column names list cannot be empty"):
        ScalarMathFeatures(
            subset=["A"],
            scalars=[5],
            operators=["+"],
            new_column_names=[],  # Empty list should raise error
        )
