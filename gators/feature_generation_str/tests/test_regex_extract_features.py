"""Tests for RegexExtractFeatures — 100 % branch coverage."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation_str import RegexExtractFeatures


class TestRegexExtractFeatures:
    def test_fit_returns_self(self):
        X = pl.DataFrame({"phone": ["123-456"]})
        t = RegexExtractFeatures(subset=["phone"], pattern=r"(?P<area>\d+)-(?P<num>\d+)")
        assert t.fit(X) is t

    def test_group_names_attribute(self):
        t = RegexExtractFeatures(subset=["col"], pattern=r"(?P<first>\w+)-(?P<second>\w+)")
        t.fit(pl.DataFrame({"col": ["a-b"]}))
        assert t.group_names_ == ["first", "second"]

    def test_single_group_extraction(self):
        X = pl.DataFrame({"text": ["hello world", "foo bar", None]})
        t = RegexExtractFeatures(subset=["text"], pattern=r"(?P<word>\w+)")
        result = t.fit(X).transform(X)
        assert "text__word" in result.columns
        assert result["text__word"][0] == "hello"
        assert result["text__word"][2] is None

    def test_multiple_groups(self):
        X = pl.DataFrame({"code": ["AB-001", "CD-999", None]})
        t = RegexExtractFeatures(subset=["code"], pattern=r"(?P<prefix>[A-Z]+)-(?P<suffix>\d+)")
        result = t.fit(X).transform(X)
        assert "code__prefix" in result.columns
        assert "code__suffix" in result.columns
        assert result["code__prefix"].to_list() == ["AB", "CD", None]
        assert result["code__suffix"].to_list() == ["001", "999", None]

    def test_no_match_gives_null(self):
        X = pl.DataFrame({"text": ["abc", "123", "xyz"]})
        t = RegexExtractFeatures(subset=["text"], pattern=r"(?P<digits>\d+)")
        result = t.fit(X).transform(X)
        assert result["text__digits"][0] is None
        assert result["text__digits"][1] == "123"

    def test_multiple_subset_columns(self):
        X = pl.DataFrame({"col1": ["a-1", "b-2"], "col2": ["c-3", "d-4"]})
        t = RegexExtractFeatures(
            subset=["col1", "col2"],
            pattern=r"(?P<letter>[a-z])-(?P<number>\d)",
        )
        result = t.fit(X).transform(X)
        for col in ["col1__letter", "col1__number", "col2__letter", "col2__number"]:
            assert col in result.columns

    def test_drop_columns_true(self):
        X = pl.DataFrame({"text": ["hello-world"]})
        t = RegexExtractFeatures(
            subset=["text"],
            pattern=r"(?P<a>\w+)-(?P<b>\w+)",
            drop_columns=True,
        )
        result = t.fit(X).transform(X)
        assert "text" not in result.columns
        assert "text__a" in result.columns

    def test_drop_columns_false(self):
        X = pl.DataFrame({"text": ["hello-world"]})
        t = RegexExtractFeatures(
            subset=["text"],
            pattern=r"(?P<a>\w+)-(?P<b>\w+)",
            drop_columns=False,
        )
        result = t.fit(X).transform(X)
        assert "text" in result.columns

    def test_original_columns_preserved(self):
        X = pl.DataFrame({"extra": [1, 2], "text": ["a-1", "b-2"]})
        t = RegexExtractFeatures(subset=["text"], pattern=r"(?P<letter>[a-z])-(?P<num>\d)")
        result = t.fit(X).transform(X)
        assert "extra" in result.columns

    def test_pattern_without_named_groups_raises(self):
        with pytest.raises(ValueError, match="named capture group"):
            RegexExtractFeatures(subset=["col"], pattern=r"(\w+)")

    def test_get_params(self):
        t = RegexExtractFeatures(subset=["col"], pattern=r"(?P<x>\w+)")
        params = t.get_params()
        assert params["pattern"] == r"(?P<x>\w+)"
        assert params["drop_columns"] is False

    def test_set_params(self):
        t = RegexExtractFeatures(subset=["col"], pattern=r"(?P<x>\w+)")
        t.set_params(drop_columns=True)
        assert t.drop_columns is True

    def test_fit_transform(self):
        X = pl.DataFrame({"val": ["key=123", "key=456"]})
        t = RegexExtractFeatures(subset=["val"], pattern=r"key=(?P<id>\d+)")
        result = t.fit_transform(X)
        assert "val__id" in result.columns
        assert result["val__id"].to_list() == ["123", "456"]
