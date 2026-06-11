import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import RoundSignificantDigits


@pytest.fixture
def sample_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0.001234, 1234.0, -9876.5, None],
            "col2": [3.14159, 0.0, 9.9999, 1.0],
            "label": ["a", "b", "c", "d"],
        }
    )


# ---------------------------------------------------------------------------
# inplace=True (default)
# ---------------------------------------------------------------------------


def test_round_inplace_3sig(sample_data):
    t = RoundSignificantDigits(n_digits=3)
    t.fit(sample_data)
    result = t.transform(sample_data)

    expected = pl.DataFrame(
        {
            "col1": [0.00123, 1230.0, -9880.0, None],
            "col2": [3.14, 0.0, 10.0, 1.0],
            "label": ["a", "b", "c", "d"],
        }
    )
    assert_frame_equal(result, expected)


def test_round_inplace_1sig(sample_data):
    t = RoundSignificantDigits(n_digits=1, subset=["col2"])
    t.fit(sample_data)
    result = t.transform(sample_data)

    expected_col2 = pl.Series("col2", [3.0, 0.0, 10.0, 1.0])
    assert_frame_equal(result.select("col2"), expected_col2.to_frame())


def test_round_inplace_subset(sample_data):
    """Only the listed column should be rounded; others left unchanged."""
    t = RoundSignificantDigits(n_digits=2, subset=["col1"])
    t.fit(sample_data)
    result = t.transform(sample_data)

    # col2 must be untouched
    assert_frame_equal(result.select("col2"), sample_data.select("col2"))
    # col1 rounded to 2 sig figs
    expected_col1 = pl.Series("col1", [0.0012, 1200.0, -9900.0, None])
    assert_frame_equal(result.select("col1"), expected_col1.to_frame())


def test_round_inplace_zero_and_null():
    """Zero values must stay 0.0; nulls must propagate."""
    X = pl.DataFrame({"v": [0.0, None, 1.5e-10]})
    t = RoundSignificantDigits(n_digits=2)
    t.fit(X)
    result = t.transform(X)

    expected = pl.DataFrame({"v": [0.0, None, 1.5e-10]})
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# inplace=False, drop_columns=True (default when inplace=False)
# ---------------------------------------------------------------------------


def test_round_not_inplace_drop_true(sample_data):
    t = RoundSignificantDigits(n_digits=2, subset=["col1"], inplace=False, drop_columns=True)
    t.fit(sample_data)
    result = t.transform(sample_data)

    assert "col1" not in result.columns
    assert "col1__round_2sig" in result.columns
    assert "col2" in result.columns
    assert "label" in result.columns

    expected_rounded = pl.Series("col1__round_2sig", [0.0012, 1200.0, -9900.0, None])
    assert_frame_equal(result.select("col1__round_2sig"), expected_rounded.to_frame())


def test_round_not_inplace_drop_false(sample_data):
    t = RoundSignificantDigits(n_digits=2, subset=["col1"], inplace=False, drop_columns=False)
    t.fit(sample_data)
    result = t.transform(sample_data)

    assert "col1" in result.columns
    assert "col1__round_2sig" in result.columns


# ---------------------------------------------------------------------------
# Auto-detection of numeric columns (subset=None)
# ---------------------------------------------------------------------------


def test_round_auto_detects_numeric_only():
    """String columns must not be included when subset=None."""
    X = pl.DataFrame(
        {
            "num": [1.23456, 7.89],
            "text": ["hello", "world"],
        }
    )
    t = RoundSignificantDigits(n_digits=3)
    t.fit(X)

    assert "num" in t.subset
    assert "text" not in t.subset

    result = t.transform(X)
    assert result["text"].to_list() == ["hello", "world"]
    assert_frame_equal(
        result.select("num"),
        pl.DataFrame({"num": [1.23, 7.89]}),
    )


def test_round_integer_columns():
    """Integer columns should be cast to Float64 and rounded correctly."""
    X = pl.DataFrame({"v": [1234, 9876, 100]})
    t = RoundSignificantDigits(n_digits=2)
    t.fit(X)
    result = t.transform(X)

    expected = pl.DataFrame({"v": [1200.0, 9900.0, 100.0]})
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Column-name suffix
# ---------------------------------------------------------------------------


def test_column_name_suffix():
    X = pl.DataFrame({"val": [3.14159]})
    t = RoundSignificantDigits(n_digits=4, inplace=False, drop_columns=True)
    t.fit(X)
    result = t.transform(X)

    assert "val__round_4sig" in result.columns


# ---------------------------------------------------------------------------
# Sklearn compatibility
# ---------------------------------------------------------------------------


def test_get_params():
    t = RoundSignificantDigits(n_digits=3, subset=["a"], inplace=False, drop_columns=False)
    params = t.get_params()
    assert params == {"n_digits": 3, "subset": ["a"], "inplace": False, "drop_columns": False}


def test_set_params():
    t = RoundSignificantDigits(n_digits=3)
    t.set_params(n_digits=5)
    assert t.n_digits == 5


def test_fit_returns_self(sample_data):
    t = RoundSignificantDigits(n_digits=2)
    result = t.fit(sample_data)
    assert result is t
