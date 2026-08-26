import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import SelectColumns


@pytest.fixture
def X() -> pl.DataFrame:
    return pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["A", "B", "C"],
        "col3": [True, False, True],
    })


def test_select_columns(X):
    t = SelectColumns(subset=["col1", "col3"])
    t.fit(X)
    result = t.transform(X)
    expected = pl.DataFrame({"col1": [1, 2, 3], "col3": [True, False, True]})
    assert_frame_equal(result, expected)


def test_select_single_column(X):
    t = SelectColumns(subset=["col2"])
    t.fit(X)
    result = t.transform(X)
    assert result.columns == ["col2"]


def test_select_all_columns(X):
    t = SelectColumns(subset=["col1", "col2", "col3"])
    t.fit(X)
    result = t.transform(X)
    assert_frame_equal(result, X)


def test_select_preserves_order(X):
    """Column order follows subset, not original DataFrame order."""
    t = SelectColumns(subset=["col3", "col1"])
    t.fit(X)
    result = t.transform(X)
    assert result.columns == ["col3", "col1"]


def test_fit_returns_self(X):
    t = SelectColumns(subset=["col1"])
    assert t.fit(X) is t


def test_get_params():
    t = SelectColumns(subset=["a", "b"])
    assert t.get_params() == {"subset": ["a", "b"]}
