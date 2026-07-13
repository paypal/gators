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


def test_transform_without_fit_returns_x_unchanged():
    """transform() before fit() returns X unchanged when subset is None."""
    X = pl.DataFrame({"A": ["a", None, "b"]})
    imputer = StringImputer(strategy="constant", value="missing")
    result = imputer.transform(X)
    assert_frame_equal(result, X)


def test_transform_inplace_true():
    """inplace=True fills nulls in the original columns (lines 124-126)."""
    X = pl.DataFrame({"A": ["a", None, "b"], "B": ["x", "y", None]})
    imputer = StringImputer(strategy="constant", value="missing", inplace=True)
    imputer.fit(X)
    result = imputer.transform(X)
    assert result["A"].null_count() == 0
    assert result["B"].null_count() == 0
    assert result["A"][1] == "missing"
    assert result["B"][2] == "missing"
