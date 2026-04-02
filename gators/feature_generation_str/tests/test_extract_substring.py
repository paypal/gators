import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import ExtractSubstring


def test_transform_default():
    X ={
        "A": ["example1", "example2", "example3", "example4"],
        "B": ["text1", "text2", "text3", "text4"],
        "C": ["short1", "short2", "short3", "short4"],
    }
    X =  pl.DataFrame(X)
    transformer = ExtractSubstring(subset=["A"], start=2, end=6)
    transformer.fit(X)
    result_X = transformer.transform(X)
    expected_X = X.with_columns(pl.col("A").str.slice(2, 4).alias("A__start2_end6"))
    assert_frame_equal(result_X, expected_X)


def test_transform_subset_columns():
    X ={
        "A": ["example1", "example2", "example3", "example4"],
        "B": ["text1", "text2", "text3", "text4"],
        "C": ["short1", "short2", "short3", "short4"],
    }
    X =  pl.DataFrame(X)
    transformer = ExtractSubstring(subset=["A", "B"], start=1, end=4)
    transformer.fit(X)
    result_X = transformer.transform(X)
    expected_X = X.with_columns(
        [
            pl.col("A").str.slice(1, 3).alias("A__start1_end4"),
            pl.col("B").str.slice(1, 3).alias("B__start1_end4"),
        ]
    )
    assert_frame_equal(result_X, expected_X)


def test_transform_drop_columns():
    X ={
        "A": ["example1", "example2", "example3", "example4"],
        "B": ["text1", "text2", "text3", "text4"],
        "C": ["short1", "short2", "short3", "short4"],
    }
    X =  pl.DataFrame(X)
    transformer = ExtractSubstring(subset=["A"], start=3, end=5)
    transformer.fit(X)
    result_X = transformer.transform(X)
    expected_X = X.with_columns(pl.col("A").str.slice(3, 2).alias("A__start3_end5"))
    assert_frame_equal(result_X, expected_X)


def test_transform_keep_columns():
    X ={
        "A": ["example1", "example2", "example3", "example4"],
        "B": ["text1", "text2", "text3", "text4"],
        "C": ["short1", "short2", "short3", "short4"],
    }
    X =  pl.DataFrame(X)
    transformer = ExtractSubstring(subset=["A"], start=0, end=3)
    transformer.fit(X)
    result_X = transformer.transform(X)
    expected_X = X.with_columns(pl.col("A").str.slice(0, 3).alias("A__start0_end3"))
    assert_frame_equal(result_X, expected_X)


def test_transform_different_datatypes():
    X ={
        "A": ["example1", "example2", "example3", "example4"],
        "B": ["text1", "text2", "text3", "text4"],
        "C": ["short1", "short2", "short3", "short4"],
        "D": [True, False, True, False],
    }
    X =  pl.DataFrame(X)
    transformer = ExtractSubstring(subset=["A", "C"], start=2, end=8)
    transformer.fit(X)
    result_X = transformer.transform(X)
    expected_X = X.with_columns(
        [
            pl.col("A").str.slice(2, 6).alias("A__start2_end8"),
            pl.col("C").str.slice(2, 6).alias("C__start2_end8"),
        ]
    )
    assert_frame_equal(result_X, expected_X)


def test_transform_end_none():
    """Test extracting from start to end of string (end=None)."""
    X ={
        "A": ["example1", "example2", "example3", "example4"],
        "B": ["text1", "text2", "text3", "text4"],
    }
    X =  pl.DataFrame(X)
    transformer = ExtractSubstring(subset=["A"], start=3, end=None)
    transformer.fit(X)
    result_X = transformer.transform(X)
    expected_X = X.with_columns(
        pl.col("A").str.slice(3, None).alias("A__start3_endNone")
    )
    assert_frame_equal(result_X, expected_X)
