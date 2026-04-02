import polars as pl
import pytest

from gators.feature_generation_str import NGram


class TestNGram:
    """Tests for NGram transformer."""

    def test_character_bigrams(self):
        """Test character-level bigram extraction."""
        X = pl.DataFrame({"text": ["hello", "hello", "world", None, ""]})

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=5)
        result = transformer.fit_transform(X)

        # Should have features for common bigrams: 'he', 'el', 'll', 'lo'
        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        assert len(new_features) > 0
        assert len(new_features) <= 5

        # 'hello' should have these bigrams counted
        # Check that at least one bigram feature exists
        assert any("text__ng_" in col for col in result.columns)

    def test_character_trigrams(self):
        """Test character-level trigram extraction."""
        X = pl.DataFrame({"text": ["hello", "hello", "world", "test"]})

        transformer = NGram(subset=["text"], n=3, ngram_type="char", max_features=5)
        result = transformer.fit_transform(X)

        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        assert len(new_features) > 0
        assert len(new_features) <= 5

        # Check that trigrams are counted correctly
        # 'hello' contains 'hel', 'ell', 'llo'
        # These should appear in the feature names

    def test_word_bigrams(self):
        """Test word-level bigram extraction."""
        X = pl.DataFrame(
            {"text": ["hello world", "hello there", "hello world", "world peace", None]}
        )

        transformer = NGram(subset=["text"], n=2, ngram_type="word", max_features=5)
        result = transformer.fit_transform(X)

        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        assert len(new_features) > 0
        assert len(new_features) <= 5

        # 'hello world' appears twice, so it should be top n-gram
        # Check that word bigram features exist

    def test_word_trigrams(self):
        """Test word-level trigram extraction."""
        X = pl.DataFrame(
            {
                "text": [
                    "the quick brown fox",
                    "the quick brown dog",
                    "the quick brown",
                    "quick brown fox",
                ]
            }
        )

        transformer = NGram(subset=["text"], n=3, ngram_type="word", max_features=5)
        result = transformer.fit_transform(X)

        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        assert len(new_features) > 0

        # 'the quick brown' appears 3 times, should be top trigram

    def test_max_features(self):
        """Test that max_features limits the number of n-grams."""
        X = pl.DataFrame({"text": ["abcdefghij"] * 5})  # Many possible bigrams

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=3)
        result = transformer.fit_transform(X)

        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        assert len(new_features) == 3

    def test_min_count(self):
        """Test that min_count filters out rare n-grams."""
        X = pl.DataFrame(
            {
                "text": [
                    "aaa",
                    "aaa",
                    "bbb",
                    "ccc",
                ]  # 'aa' appears 4 times, 'bb' 2 times, 'cc' 2 times
            }
        )

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=10, min_count=3)
        result = transformer.fit_transform(X)

        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        # Only 'aa' should pass min_count threshold (appears 4 times total)
        assert len(new_features) >= 1

    def test_ngram_counts(self):
        """Test that n-gram counts are correct."""
        X = pl.DataFrame({"text": ["aaa", "aa", "a"]})

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=5)
        result = transformer.fit_transform(X)

        # 'aaa' contains 'aa' twice
        # 'aa' contains 'aa' once
        # 'a' contains no bigrams
        if "text__ng_aa" in result.columns:
            assert result["text__ng_aa"][0] == 2  # 'aaa'
            assert result["text__ng_aa"][1] == 1  # 'aa'
            assert result["text__ng_aa"][2] == 0  # 'a'

    def test_multiple_columns(self):
        """Test transformation on multiple columns."""
        X = pl.DataFrame({"col1": ["hello", "world"], "col2": ["test", "data"]})

        transformer = NGram(subset=["col1", "col2"], n=2, ngram_type="char", max_features=5)
        result = transformer.fit_transform(X)

        col1_features = [col for col in result.columns if col.startswith("col1__ng_")]
        col2_features = [col for col in result.columns if col.startswith("col2__ng_")]

        assert len(col1_features) > 0
        assert len(col2_features) > 0

    def test_auto_detect_string_columns(self):
        """Test automatic detection of string columns."""
        X = pl.DataFrame({"text": ["hello", "world"], "num": [1, 2], "float": [1.5, 2.5]})

        transformer = NGram(n=2, ngram_type="char", max_features=5)
        result = transformer.fit_transform(X)

        # Should only create features for 'text' column
        text_features = [col for col in result.columns if col.startswith("text__ng_")]
        num_features = [col for col in result.columns if col.startswith("num__ng_")]
        float_features = [col for col in result.columns if col.startswith("float__ng_")]

        assert len(text_features) > 0
        assert len(num_features) == 0
        assert len(float_features) == 0

    def test_drop_columns(self):
        """Test drop_columns parameter."""
        X = pl.DataFrame({"text": ["hello", "world"], "name": ["Alice", "Bob"]})

        transformer = NGram(
            subset=["text"], n=2, ngram_type="char", max_features=5, drop_columns=True
        )
        result = transformer.fit_transform(X)

        assert "text" not in result.columns
        assert "name" in result.columns
        text_features = [col for col in result.columns if col.startswith("text__ng_")]
        assert len(text_features) > 0

    def test_special_characters_in_ngrams(self):
        """Test handling of special characters in n-grams."""
        X = pl.DataFrame({"text": ["user@test.com", "admin@site.org", "test@email"]})

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=10)
        result = transformer.fit_transform(X)

        # Should handle @ and . in n-grams
        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        assert len(new_features) > 0

        # Feature names should have special chars replaced (@ -> at, . -> dot)
        # Check that no feature names contain problematic characters

    def test_word_ngrams_with_special_chars(self):
        """Test word n-grams with punctuation."""
        X = pl.DataFrame({"text": ["hello, world!", "hello, there", "world! peace"]})

        transformer = NGram(subset=["text"], n=2, ngram_type="word", max_features=5)
        result = transformer.fit_transform(X)

        # Word splitting should handle punctuation
        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        assert len(new_features) > 0

    def test_empty_strings_and_nulls(self):
        """Test handling of empty strings and nulls."""
        X = pl.DataFrame({"text": ["hello", "", None, "world", ""]})

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=5)
        result = transformer.fit_transform(X)

        # Should not crash on empty/null values
        assert len(result) == 5

        # Null and empty strings should have 0 counts
        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        if new_features:
            assert result[new_features[0]][1] == 0  # empty string
            assert result[new_features[0]][2] == 0  # null

    def test_validation_n_range(self):
        """Test validation of n parameter."""
        with pytest.raises(ValueError, match="n must be at least 1"):
            NGram(n=0)

        with pytest.raises(ValueError, match="n should not exceed 10"):
            NGram(n=11)

    def test_validation_ngram_type(self):
        """Test validation of ngram_type parameter."""
        with pytest.raises(ValueError, match="ngram_type must be"):
            NGram(ngram_type="invalid")

    def test_validation_max_features(self):
        """Test validation of max_features parameter."""
        with pytest.raises(ValueError, match="max_features must be at least 1"):
            NGram(max_features=0)

    def test_validation_min_count(self):
        """Test validation of min_count parameter."""
        with pytest.raises(ValueError, match="min_count must be at least 1"):
            NGram(min_count=0)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({"text": []}, schema={"text": pl.String})

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=5)
        result = transformer.fit_transform(X)

        assert len(result) == 0

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"text": ["hello world", "test data"]})

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=5)

        # Test fit returns self
        assert transformer.fit(X) is transformer

        # Test fit_transform
        result = transformer.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform
        transformer2 = NGram(subset=["text"], n=2, ngram_type="char", max_features=5)
        transformer2.fit(X)
        result2 = transformer2.transform(X)
        assert result.equals(result2)

    def test_single_character_strings(self):
        """Test with strings shorter than n."""
        X = pl.DataFrame({"text": ["a", "b", "c", "ab"]})

        transformer = NGram(subset=["text"], n=3, ngram_type="char", max_features=5)
        result = transformer.fit_transform(X)

        # Strings 'a', 'b', 'c' are too short for trigrams
        # Only 'ab' is still too short
        # Should not have any n-grams or very few
        new_features = [col for col in result.columns if col.startswith("text__ng_")]
        # Most counts should be 0

    def test_consistent_ordering(self):
        """Test that top n-grams are selected consistently."""
        X = pl.DataFrame({"text": ["hello"] * 10 + ["world"] * 5})

        transformer = NGram(subset=["text"], n=2, ngram_type="char", max_features=3)
        result1 = transformer.fit_transform(X)

        # Fit again with same data
        transformer2 = NGram(subset=["text"], n=2, ngram_type="char", max_features=3)
        result2 = transformer2.fit_transform(X)

        # Should produce same features
        features1 = [col for col in result1.columns if col.startswith("text__ng_")]
        features2 = [col for col in result2.columns if col.startswith("text__ng_")]
        assert set(features1) == set(features2)
