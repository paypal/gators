import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders._base_encoder import _BaseEncoder


class ExampleEncoder(_BaseEncoder):
    pass


@pytest.fixture
def sample_X():
    return pl.DataFrame(
        {
            "category": ["A", "B", "A", "C"],
            "value": [1, 2, 3, 4],
            "other": ["foo", "bar", "baz", "qux"],
        }
    )


def test_default_parameters(sample_X):
    encoder = ExampleEncoder(
        subset=["category"],
        drop_columns=True,
        inplace=False,
        mapping_={"category": {"A": 1.0, "B": 2.0, "C": 3.0}},
        column_mapping_={"category": "category_encoded"},
    )
    transformed_X = encoder.transform(sample_X)

    expected_X = pl.DataFrame(
        {
            "value": [1, 2, 3, 4],
            "other": ["foo", "bar", "baz", "qux"],
            "category_encoded": [1.0, 2.0, 1.0, 3.0],
        }
    )
    assert_frame_equal(transformed_X, expected_X)


def test_columns_subset_drop_columns_false(sample_X):
    encoder = ExampleEncoder(
        subset=["category"],
        drop_columns=False,
        inplace=False,
        mapping_={"category": {"A": 1.0, "B": 2.0, "C": 3.0}},
        column_mapping_={"category": "category_encoded"},
    )
    transformed_X = encoder.transform(sample_X)

    expected_X = sample_X.with_columns(
        pl.col("category")
        .replace({"A": 1.0, "B": 2.0, "C": 3.0})
        .cast(pl.Float64)
        .alias("category_encoded")
    )
    assert_frame_equal(transformed_X, expected_X)


def test_inplace_true():
    """Test inplace encoding (replaces original columns)."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B", "A", "C"],
            "cat2": ["X", "Y", "X", "Z"],
            "value": [1, 2, 3, 4],
        }
    )

    encoder = ExampleEncoder(
        subset=["cat1", "cat2"],
        inplace=True,
        mapping_={
            "cat1": {"A": 1.0, "B": 2.0, "C": 3.0},
            "cat2": {"X": 10.0, "Y": 20.0, "Z": 30.0},
        },
    )
    result = encoder.transform(X)

    # When inplace=True, columns should be replaced in place
    assert "cat1" in result.columns
    assert "cat2" in result.columns
    assert result["cat1"].dtype == pl.Float64
    assert result["cat2"].dtype == pl.Float64
    assert result["cat1"].to_list() == [1.0, 2.0, 1.0, 3.0]
    assert result["cat2"].to_list() == [10.0, 20.0, 10.0, 30.0]


def test_boolean_column_encoding():
    """Test encoding boolean columns."""
    X = pl.DataFrame({"bool_col": [True, False, True, False], "value": [1, 2, 3, 4]})

    encoder = ExampleEncoder(
        subset=["bool_col"],
        drop_columns=False,
        inplace=False,
        mapping_={"bool_col": {"true": 1.0, "false": 0.0}},  # Boolean keys as lowercase strings
        column_mapping_={"bool_col": "bool_col_encoded"},
    )
    result = encoder.transform(X)

    assert "bool_col_encoded" in result.columns
    assert result["bool_col_encoded"].dtype == pl.Float64
    assert result["bool_col_encoded"].to_list() == [1.0, 0.0, 1.0, 0.0]


def test_boolean_column_inplace():
    """Test encoding boolean columns with inplace=True."""
    X = pl.DataFrame({"bool_col": [True, False, True, False], "value": [1, 2, 3, 4]})

    encoder = ExampleEncoder(
        subset=["bool_col"],
        inplace=True,
        mapping_={"bool_col": {"true": 1.0, "false": 0.0}},  # Boolean keys as lowercase strings
    )
    result = encoder.transform(X)

    assert "bool_col" in result.columns
    assert result["bool_col"].dtype == pl.Float64
    assert result["bool_col"].to_list() == [1.0, 0.0, 1.0, 0.0]


def test_missing_category_default_value():
    """Test that missing categories get default value 0.0."""
    X = pl.DataFrame(
        {"category": ["A", "B", "D", "C"], "value": [1, 2, 3, 4]}  # D is not in mapping
    )

    encoder = ExampleEncoder(
        subset=["category"],
        drop_columns=False,
        inplace=False,
        mapping_={"category": {"A": 1.0, "B": 2.0, "C": 3.0}},  # No "D"
        column_mapping_={"category": "category_encoded"},
    )
    result = encoder.transform(X)

    assert result["category_encoded"].to_list() == [1.0, 2.0, 0.0, 3.0]  # D -> 0.0


if __name__ == "__main__":
    pytest.main()
