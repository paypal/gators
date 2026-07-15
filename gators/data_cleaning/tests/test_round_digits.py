import polars as pl
from polars.testing import assert_frame_equal

from gators.data_cleaning import RoundDigits


def test_round_inplace_rounds_decimal_places():
    X = pl.DataFrame(
        {
            "a": [1.23456, -2.34567, None],
            "b": [3.14159, 0.0, 9.99999],
            "label": ["x", "y", "z"],
        }
    )

    transformer = RoundDigits(n_digits=2)
    transformer.fit(X)
    result = transformer.transform(X)

    expected = pl.DataFrame(
        {
            "a": [1.23, -2.35, None],
            "b": [3.14, 0.0, 10.0],
            "label": ["x", "y", "z"],
        }
    )
    assert_frame_equal(result, expected)


def test_round_not_inplace_adds_new_columns():
    X = pl.DataFrame({"a": [1.23456, 2.34567], "b": [5.0, 6.0]})

    transformer = RoundDigits(n_digits=1, subset=["a"], inplace=False, drop_columns=True)
    transformer.fit(X)
    result = transformer.transform(X)

    assert "a" not in result.columns
    assert "a__round_1digits" in result.columns
    assert "b" in result.columns

    expected = pl.DataFrame({"b": [5.0, 6.0], "a__round_1digits": [1.2, 2.3]})
    assert_frame_equal(result, expected)


def test_round_not_inplace_keeps_original_columns_when_drop_columns_false():
    X = pl.DataFrame({"a": [1.23456, 2.34567], "b": [5.0, 6.0]})

    transformer = RoundDigits(n_digits=1, subset=["a"], inplace=False, drop_columns=False)
    transformer.fit(X)
    result = transformer.transform(X)

    assert "a" in result.columns
    assert "a__round_1digits" in result.columns
    assert "b" in result.columns

    expected = pl.DataFrame(
        {
            "a": [1.23456, 2.34567],
            "b": [5.0, 6.0],
            "a__round_1digits": [1.2, 2.3],
        }
    )
    assert_frame_equal(result, expected)


def test_get_params_and_set_params():
    transformer = RoundDigits(n_digits=3, subset=["a"], inplace=False, drop_columns=False)
    params = transformer.get_params()
    assert params == {"n_digits": 3, "subset": ["a"], "inplace": False, "drop_columns": False}

    transformer.set_params(n_digits=4)
    assert transformer.n_digits == 4
