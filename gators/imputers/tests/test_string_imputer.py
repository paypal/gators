import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.imputers.string_imputer import StringImputer


@pytest.fixture
def sample_X():
    return pl.DataFrame({"A": ["a", "b", None], "B": ["x", None, "y"], "C": [1, 2, 3]})


@pytest.fixture
def mode_X():
    return pl.DataFrame(
        {
            "A": ["a", "b", None, "a"],
            "B": ["x", "x", "y", None],
        }
    )


def test_fit_constant_strategy(sample_X):
    imputer = StringImputer(strategy="constant", value="missing")
    imputer.fit(sample_X)

    assert imputer._statistics["A"] == "missing"
    assert imputer._statistics["B"] == "missing"


def test_fit_mode_strategy(mode_X):
    imputer = StringImputer(strategy="most_frequent")
    imputer.fit(mode_X)

    assert imputer._statistics["A"] == "a"
    assert imputer._statistics["B"] == "x"


def test_transform_constant_strategy(sample_X):
    imputer = StringImputer(strategy="constant", value="missing", inplace=False)
    imputer.fit(sample_X)

    transformed = imputer.transform(sample_X)

    expected = pl.DataFrame(
        {
            "C": [1, 2, 3],
            "A__impute_constant": ["a", "b", "missing"],
            "B__impute_constant": ["x", "missing", "y"],
        }
    )

    assert_frame_equal(transformed, expected)


def test_transform_mode_strategy(mode_X):
    imputer = StringImputer(strategy="most_frequent", inplace=False)
    imputer.fit(mode_X)

    transformed = imputer.transform(mode_X)

    expected = pl.DataFrame(
        {
            "A__impute_most_frequent": ["a", "b", "a", "a"],
            "B__impute_most_frequent": ["x", "x", "y", "x"],
        }
    )

    assert_frame_equal(transformed, expected)


@pytest.fixture
def sample_X_drop():
    return pl.DataFrame(
        {
            "A": ["a", "b", None],
            "B": ["x", None, "y"],
        }
    )


def test_drop_columns(sample_X_drop):
    imputer = StringImputer(strategy="constant", value="missing", drop_columns=False, inplace=False)
    imputer.fit(sample_X_drop)
    transformed = imputer.transform(sample_X_drop)
    expected = pl.DataFrame(
        {
            "A": ["a", "b", None],
            "B": ["x", None, "y"],
            "A__impute_constant": ["a", "b", "missing"],
            "B__impute_constant": ["x", "missing", "y"],
        }
    )

    assert_frame_equal(transformed, expected)
