import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.imputers.string_imputer import StringImputer
from gators.exceptions import NotFittedError


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
    imputer = StringImputer(strategy="constant", value="__NULL__")
    imputer.fit(sample_X)

    assert imputer._statistics["A"] == "__NULL__"
    assert imputer._statistics["B"] == "__NULL__"


def test_fit_mode_strategy(mode_X):
    imputer = StringImputer(strategy="most_frequent")
    imputer.fit(mode_X)

    assert imputer._statistics["A"] == "a"
    assert imputer._statistics["B"] == "x"


def test_transform_constant_strategy(sample_X):
    imputer = StringImputer(strategy="constant", value="__NULL__", inplace=False)
    imputer.fit(sample_X)

    transformed = imputer.transform(sample_X)

    expected = pl.DataFrame(
        {
            "C": [1, 2, 3],
            "A__impute_constant": ["a", "b", "__NULL__"],
            "B__impute_constant": ["x", "__NULL__", "y"],
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
    imputer = StringImputer(strategy="constant", value="__NULL__", drop_columns=False, inplace=False)
    imputer.fit(sample_X_drop)
    transformed = imputer.transform(sample_X_drop)
    expected = pl.DataFrame(
        {
            "A": ["a", "b", None],
            "B": ["x", None, "y"],
            "A__impute_constant": ["a", "b", "__NULL__"],
            "B__impute_constant": ["x", "__NULL__", "y"],
        }
    )

    assert_frame_equal(transformed, expected)


def test_transform_without_fit_returns_x_unchanged():
    """transform() before fit() raises NotFittedError."""


def test_transform_without_fit_returns_x_unchanged():
    X = pl.DataFrame({"A": ["a", None, "b"]})
    imputer = StringImputer(strategy="constant", value="__NULL__")
    with pytest.raises(NotFittedError):
        imputer.transform(X)


def test_most_frequent_all_null_defaults_to_null_sentinel():
    X = pl.DataFrame({"A": pl.Series([None, None, None], dtype=pl.String)})
    imputer = StringImputer(strategy="most_frequent")
    imputer.fit(X)
    assert imputer._statistics["A"] == "__NULL__"
    result = imputer.transform(X)
    assert result["A"].to_list() == ["__NULL__", "__NULL__", "__NULL__"]


def test_transform_inplace_true():
    """inplace=True fills nulls in the original columns (lines 124-126)."""
    X = pl.DataFrame({"A": ["a", None, "b"], "B": ["x", "y", None]})
    imputer = StringImputer(strategy="constant", value="__NULL__", inplace=True)
    imputer.fit(X)
    result = imputer.transform(X)
    assert result["A"].null_count() == 0
    assert result["B"].null_count() == 0
    assert result["A"][1] == "__NULL__"
    assert result["B"][2] == "__NULL__"
