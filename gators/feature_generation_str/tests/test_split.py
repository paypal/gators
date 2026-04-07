import polars as pl
from polars.testing import assert_frame_equal

from gators.feature_generation_str import Split


def test_split_basic():
    """Test basic splitting."""
    X = {"Column1": ["a|b|c", "d|e|f", "g|h|i"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_|_0": ["a", "d", "g"],
        "Column1__split_|_1": ["b", "e", "h"],
        "Column1__split_|_2": ["c", "f", "i"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_uneven_splits():
    """Test splitting with uneven number of parts (should fill with empty string)."""
    X = {"Column1": ["a|b|c", "d|e", "g"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_|_0": ["a", "d", "g"],
        "Column1__split_|_1": ["b", "e", ""],
        "Column1__split_|_2": ["c", "", ""],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_space_delimiter():
    """Test splitting with space delimiter."""
    X = {"full_name": ["John Doe", "Jane Smith Williams", "Alice Johnson"]}
    X = pl.DataFrame(X)

    expected_X = {
        "full_name__split___0": ["John", "Jane", "Alice"],
        "full_name__split___1": ["Doe", "Smith", "Johnson"],
        "full_name__split___2": ["", "Williams", ""],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["full_name"], by=" ", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_keep_original():
    """Test splitting with drop_columns=False."""
    X = {"Column1": ["a|b", "c|d"], "OtherColumn": ["x", "y"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1": ["a|b", "c|d"],
        "OtherColumn": ["x", "y"],
        "Column1__split_|_0": ["a", "c"],
        "Column1__split_|_1": ["b", "d"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=2, drop_columns=False)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_multiple_columns():
    """Test splitting multiple columns."""
    X = {"Column1": ["a|b|c", "d|e|f"], "Column2": ["x|y", "z|w"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_|_0": ["a", "d"],
        "Column1__split_|_1": ["b", "e"],
        "Column1__split_|_2": ["c", "f"],
        "Column2__split_|_0": ["x", "z"],
        "Column2__split_|_1": ["y", "w"],
        "Column2__split_|_2": ["", ""],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1", "Column2"], by="|", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_new_data_consistency():
    """Test that new data with more splits is truncated to max_splits."""
    X_train = pl.DataFrame({"Column1": ["a|b", "c|d"]})
    X_test = pl.DataFrame({"Column1": ["x|y|z|w", "m|n"]})

    # max_splits=2 ensures only 2 columns are created
    expected_X = {
        "Column1__split_|_0": ["x", "m"],
        "Column1__split_|_1": ["y", "n"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=2)
    transformer.fit(X_train)
    transformed_X = transformer.transform(X_test)
    assert_frame_equal(transformed_X, expected_X)


def test_split_single_value():
    """Test splitting when there's only one part (no delimiter found)."""
    X = {"Column1": ["abc", "def", "ghi"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_|_0": ["abc", "def", "ghi"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=1)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_fit_returns_self():
    """Test that fit returns self for method chaining."""
    X = {"Column1": ["a|b", "c|d"]}
    X = pl.DataFrame(X)

    transformer = Split(subset=["Column1"], by="|", max_splits=2)
    result = transformer.fit(X)
    assert result is transformer


def test_split_fit_with_y_parameter():
    """Test that fit ignores the y parameter."""
    X = {"Column1": ["a|b", "c|d"]}
    X = pl.DataFrame(X)
    y = pl.Series("target", [0, 1])

    expected_X = {
        "Column1__split_|_0": ["a", "c"],
        "Column1__split_|_1": ["b", "d"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=2)
    transformer.fit(X, y)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_comma_delimiter():
    """Test splitting with comma delimiter."""
    X = {"Column1": ["a,b,c", "d,e,f"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_,_0": ["a", "d"],
        "Column1__split_,_1": ["b", "e"],
        "Column1__split_,_2": ["c", "f"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by=",", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_hyphen_delimiter():
    """Test splitting with hyphen delimiter."""
    X = {"Column1": ["a-b-c", "d-e-f"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_-_0": ["a", "d"],
        "Column1__split_-_1": ["b", "e"],
        "Column1__split_-_2": ["c", "f"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="-", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_with_empty_strings():
    """Test splitting with empty strings in the data."""
    X = {"Column1": ["a|b", "", "c|d|e"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_|_0": ["a", "", "c"],
        "Column1__split_|_1": ["b", "", "d"],
        "Column1__split_|_2": ["", "", "e"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_large_max_splits():
    """Test splitting with max_splits larger than any actual splits."""
    X = {"Column1": ["a|b", "c|d", "e"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_|_0": ["a", "c", "e"],
        "Column1__split_|_1": ["b", "d", ""],
        "Column1__split_|_2": ["", "", ""],
        "Column1__split_|_3": ["", "", ""],
        "Column1__split_|_4": ["", "", ""],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=5)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_max_splits_one():
    """Test splitting with max_splits=1 (only first part)."""
    X = {"Column1": ["a|b|c", "d|e|f", "g|h"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_|_0": ["a", "d", "g"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=1)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_column_order_with_multiple_columns():
    """Test column order when splitting multiple columns with drop_columns=False."""
    X = {"A": ["a|b", "c|d"], "B": ["x|y", "z|w"], "C": [1, 2]}
    X = pl.DataFrame(X)

    expected_X = {
        "A": ["a|b", "c|d"],
        "B": ["x|y", "z|w"],
        "C": [1, 2],
        "A__split_|_0": ["a", "c"],
        "A__split_|_1": ["b", "d"],
        "B__split_|_0": ["x", "z"],
        "B__split_|_1": ["y", "w"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["A", "B"], by="|", max_splits=2, drop_columns=False)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_underscore_delimiter():
    """Test splitting with underscore delimiter."""
    X = {"Column1": ["a_b_c", "d_e_f"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split___0": ["a", "d"],
        "Column1__split___1": ["b", "e"],
        "Column1__split___2": ["c", "f"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="_", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_dot_delimiter():
    """Test splitting with dot delimiter."""
    X = {"Column1": ["a.b.c", "d.e.f"]}
    X = pl.DataFrame(X)

    expected_X = {
        "Column1__split_._0": ["a", "d"],
        "Column1__split_._1": ["b", "e"],
        "Column1__split_._2": ["c", "f"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by=".", max_splits=3)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_preserves_other_columns():
    """Test that columns not in subset are preserved."""
    X = {
        "Column1": ["a|b", "c|d"],
        "Column2": ["x|y", "z|w"],
        "Column3": [1, 2],
        "Column4": ["keep", "this"],
    }
    X = pl.DataFrame(X)

    expected_X = {
        "Column2": ["x|y", "z|w"],
        "Column3": [1, 2],
        "Column4": ["keep", "this"],
        "Column1__split_|_0": ["a", "c"],
        "Column1__split_|_1": ["b", "d"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = Split(subset=["Column1"], by="|", max_splits=2, drop_columns=True)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
