import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import WOEEncoder


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "A": ["cat", "dog", "cat", "dog", "cat"],
            "B": ["x", "x", "y", "y", "x"],
        }
    )


@pytest.fixture
def sample_target():
    return pl.Series("target", [1, 0, 1, 1, 0])


def test_woe_encoder_transform_defaults(sample_data, sample_target):
    expected_X = pl.DataFrame(
        {
            "A__encode_woe": [0.286025, -0.402159, 0.286025, -0.402159, 0.286025],
            "B__encode_woe": [-1.090344, -1.090344, 4.901146, 4.901146, -1.090344],
        }
    )
    woe_encoder = WOEEncoder(inplace=False)
    woe_encoder.fit(sample_data, y=sample_target)
    result = woe_encoder.transform(sample_data)
    assert_frame_equal(result, expected_X, check_column_order=False)


def test_woe_encoder_transform_no_drop_columns(sample_data, sample_target):
    expected_X = pl.DataFrame(
        {
            "A": ["cat", "dog", "cat", "dog", "cat"],
            "B": ["x", "x", "y", "y", "x"],
            "B__encode_woe": [-1.090344, -1.090344, 4.901146, 4.901146, -1.090344],
        }
    )
    woe_encoder = WOEEncoder(subset=["B"], drop_columns=False, inplace=False)
    woe_encoder.fit(sample_data, y=sample_target)

    result = woe_encoder.transform(sample_data)

    assert_frame_equal(result, expected_X, check_column_order=False)


def test_woe_unseen_categories(sample_data, sample_target):

    woe_encoder = WOEEncoder(inplace=False)
    sample_data_new = pl.DataFrame(
        {
            "A": [None, "beta", "cat", "dog", "cat"],
            "B": ["alpha", None, "y", "y", "x"],
        }
    )
    woe_encoder.fit(sample_data, y=sample_target)
    expected_X = pl.DataFrame(
        {
            "A__encode_woe": [0.0, 0.0, 0.286025, -0.402159, 0.286025],
            "B__encode_woe": [0.0, 0.0, 4.901146, 4.901146, -1.090344],
        }
    )
    result = woe_encoder.transform(sample_data_new)
    assert_frame_equal(result, expected_X, check_column_order=False)


def test_fit_without_y_raises_value_error():
    """fit() raises ValueError when y is None."""
    X = pl.DataFrame({"A": ["a", "b", "c"]})
    encoder = WOEEncoder(subset=["A"])
    with pytest.raises(ValueError, match="requires a target variable"):
        encoder.fit(X, y=None)


def test_woe_encoder_with_enum_columns():
    """Enum columns are cast to String before unpivot so no SchemaError is raised."""
    X = pl.DataFrame(
        {
            "A": pl.Series(["[0,1)", "[1,2)", "[0,1)", "[1,2)", "[0,1)"]).cast(
                pl.Enum(["[0,1)", "[1,2)"])
            ),
            "B": pl.Series(["[0,5)", "[5,10)", "[5,10)", "[0,5)", "[0,5)"]).cast(
                pl.Enum(["[0,5)", "[5,10)"])
            ),
        }
    )
    y = pl.Series("target", [1, 0, 1, 1, 0])
    encoder = WOEEncoder(inplace=False)
    encoder.fit(X, y=y)
    result = encoder.transform(X)
    assert result.shape == (5, 2)
    assert "A__encode_woe" in result.columns
    assert "B__encode_woe" in result.columns
    assert result.dtypes == [pl.Float64, pl.Float64]


def test_woe_encoder_with_categorical_columns():
    """Categorical columns are cast to String before unpivot so no SchemaError is raised."""
    X = pl.DataFrame(
        {
            "A": pl.Series(["cat", "dog", "cat", "dog", "cat"]).cast(pl.Categorical),
            "B": pl.Series(["x", "x", "y", "y", "x"]).cast(pl.Categorical),
        }
    )
    y = pl.Series("target", [1, 0, 1, 1, 0])
    encoder = WOEEncoder(inplace=False)
    encoder.fit(X, y=y)
    result = encoder.transform(X)
    assert result.shape == (5, 2)
    assert result.dtypes == [pl.Float64, pl.Float64]
