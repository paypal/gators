import polars as pl
from polars.testing import assert_frame_equal

from gators.feature_generation_str import SplitExtract


def test_split_extract_first_part():
    """Test extracting the first part (n=0)."""
    X ={"Column1": ["a|b|c", "d|e|f", "g|h|i"]}
    X =  pl.DataFrame(X)

    expected_X ={
        "Column1__split_|_0": ["a", "d", "g"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = SplitExtract(subset=["Column1"], by="|", n=0)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_extract_second_part():
    """Test extracting the second part (n=1)."""
    X ={"Column1": ["a|b|c", "d|e|f", "g|h|i"]}
    X =  pl.DataFrame(X)

    expected_X ={
        "Column1__split_|_1": ["b", "e", "h"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = SplitExtract(subset=["Column1"], by="|", n=1)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_extract_third_part():
    """Test extracting the third part (n=2)."""
    X ={"Column1": ["a|b|c", "d|e|f", "g|h|i"]}
    X =  pl.DataFrame(X)

    expected_X ={
        "Column1__split_|_2": ["c", "f", "i"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = SplitExtract(subset=["Column1"], by="|", n=2)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_extract_keep_original():
    """Test extracting with drop_columns=False."""
    X ={"Column1": ["a|b|c", "d|e|f", "g|h|i"], "OtherColumn": ["x", "y", "z"]}
    X =  pl.DataFrame(X)

    expected_X ={
        "Column1": ["a|b|c", "d|e|f", "g|h|i"],
        "OtherColumn": ["x", "y", "z"],
        "Column1__split_|_0": ["a", "d", "g"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = SplitExtract(subset=["Column1"], by="|", n=0, drop_columns=False)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


def test_split_extract_multiple_columns():
    """Test splitting multiple columns."""
    X ={"Column1": ["a|b", "c|d"], "Column2": ["x|y", "z|w"]}
    X =  pl.DataFrame(X)

    expected_X ={
        "Column1__split_|_0": ["a", "c"],
        "Column2__split_|_0": ["x", "z"],
    }
    expected_X = pl.DataFrame(expected_X)

    transformer = SplitExtract(subset=["Column1", "Column2"], by="|", n=0)
    transformer.fit(X)
    transformed_X = transformer.transform(X)
    assert_frame_equal(transformed_X, expected_X)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
