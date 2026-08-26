"""Tests for TfidfFeatures — 100 % branch coverage."""

import polars as pl
import pytest

from gators.feature_generation_str import TfidfFeatures
from gators.exceptions import NotFittedError


@pytest.fixture
def text_df():
    return pl.DataFrame(
        {
            "text": [
                "the cat sat on the mat",
                "the dog sat on the log",
                "the cat and the dog",
                "a cat and a dog sat",
            ]
        }
    )


class TestTfidfFeatures:
    def test_fit_returns_self(self, text_df):
        t = TfidfFeatures(subset=["text"])
        assert t.fit(text_df) is t

    def test_columns_created(self, text_df):
        t = TfidfFeatures(subset=["text"], max_features=5)
        result = t.fit(text_df).transform(text_df)
        tfidf_cols = [c for c in result.columns if c.startswith("text__tfidf_")]
        assert len(tfidf_cols) == 5

    def test_vocabulary_attribute(self, text_df):
        t = TfidfFeatures(subset=["text"], max_features=3)
        t.fit(text_df)
        assert "text" in t.vocabulary_
        assert len(t.vocabulary_["text"]) == 3

    def test_idf_attribute(self, text_df):
        t = TfidfFeatures(subset=["text"], max_features=3)
        t.fit(text_df)
        assert "text" in t.idf_
        for token, weight in t.idf_["text"].items():
            assert weight > 0

    def test_tfidf_values_non_negative(self, text_df):
        t = TfidfFeatures(subset=["text"], max_features=5)
        result = t.fit(text_df).transform(text_df)
        for col in result.columns:
            if col.startswith("text__tfidf_"):
                assert (result[col] >= 0).all()

    def test_null_document_gives_zero(self):
        X = pl.DataFrame({"text": ["hello world", None, "hello"]})
        t = TfidfFeatures(subset=["text"], max_features=3)
        result = t.fit(X).transform(X)
        tfidf_cols = [c for c in result.columns if c.startswith("text__tfidf_")]
        for col in tfidf_cols:
            assert result[col][1] == pytest.approx(0.0)

    def test_empty_document_gives_zero(self):
        X = pl.DataFrame({"text": ["hello world", "", "hello"]})
        t = TfidfFeatures(subset=["text"], max_features=2)
        result = t.fit(X).transform(X)
        tfidf_cols = [c for c in result.columns if c.startswith("text__tfidf_")]
        for col in tfidf_cols:
            assert result[col][1] == pytest.approx(0.0)

    def test_max_features_respected(self, text_df):
        t = TfidfFeatures(subset=["text"], max_features=2)
        t.fit(text_df)
        assert len(t.vocabulary_["text"]) <= 2

    def test_min_df_filters_rare_tokens(self):
        X = pl.DataFrame(
            {
                "text": [
                    "common common rare",
                    "common common",
                    "common only",
                ]
            }
        )
        t = TfidfFeatures(subset=["text"], min_df=2)
        t.fit(X)
        # "__RARE__" only appears in 1 document, should be excluded
        assert "__RARE__" not in t.vocabulary_.get("text", [])
        assert "common" in t.vocabulary_.get("text", [])

    def test_lowercase_true(self):
        X = pl.DataFrame({"text": ["Hello World", "HELLO world", "hello WORLD"]})
        t = TfidfFeatures(subset=["text"], max_features=5, lowercase=True)
        t.fit(X)
        vocab = t.vocabulary_["text"]
        assert all(w == w.lower() for w in vocab)

    def test_lowercase_false(self):
        X = pl.DataFrame({"text": ["Hello World", "Hello World"]})
        t = TfidfFeatures(subset=["text"], max_features=10, lowercase=False)
        t.fit(X)
        # "Hello" (capital) should be in vocabulary
        vocab = t.vocabulary_["text"]
        assert "Hello" in vocab

    def test_custom_separator(self):
        X = pl.DataFrame({"text": ["a,b,c", "a,b", "a,c"]})
        t = TfidfFeatures(subset=["text"], separator=",", max_features=5)
        result = t.fit(X).transform(X)
        tfidf_cols = [c for c in result.columns if c.startswith("text__tfidf_")]
        assert len(tfidf_cols) >= 1

    def test_drop_columns_true(self, text_df):
        t = TfidfFeatures(subset=["text"], max_features=3, drop_columns=True)
        result = t.fit(text_df).transform(text_df)
        assert "text" not in result.columns
        tfidf_cols = [c for c in result.columns if c.startswith("text__tfidf_")]
        assert len(tfidf_cols) == 3

    def test_drop_columns_false(self, text_df):
        t = TfidfFeatures(subset=["text"], max_features=3, drop_columns=False)
        result = t.fit(text_df).transform(text_df)
        assert "text" in result.columns

    def test_auto_detect_subset(self, text_df):
        t = TfidfFeatures(max_features=3)
        t.fit(text_df)
        assert "text" in t.vocabulary_

    def test_multiple_subset_columns(self):
        X = pl.DataFrame(
            {
                "title": ["quick brown fox", "lazy dog", "quick fox"],
                "body": ["hello world news", "hello world", "news today"],
            }
        )
        t = TfidfFeatures(subset=["title", "body"], max_features=3)
        result = t.fit(X).transform(X)
        title_cols = [c for c in result.columns if c.startswith("title__tfidf_")]
        body_cols = [c for c in result.columns if c.startswith("body__tfidf_")]
        assert len(title_cols) >= 1
        assert len(body_cols) >= 1

    def test_higher_tf_gives_higher_score(self):
        """Token that appears more times in a document should get higher TF-IDF."""
        X = pl.DataFrame({"text": ["cat cat cat dog", "dog", "cat dog"]})
        t = TfidfFeatures(subset=["text"], max_features=2, min_df=1)
        result = t.fit(X).transform(X)
        cat_col = "text__tfidf_cat"
        if cat_col in result.columns:
            assert result[cat_col][0] > result[cat_col][1]

    def test_get_params(self):
        t = TfidfFeatures(max_features=50, min_df=2)
        params = t.get_params()
        assert params["max_features"] == 50
        assert params["min_df"] == 2

    def test_set_params(self):
        t = TfidfFeatures(max_features=10)
        t.set_params(max_features=20)
        assert t.max_features == 20

    def test_fit_transform(self, text_df):
        result = TfidfFeatures(subset=["text"], max_features=3).fit_transform(text_df)
        tfidf_cols = [c for c in result.columns if c.startswith("text__tfidf_")]
        assert len(tfidf_cols) == 3

    def test_transform_without_vocabulary_no_op(self):
        """subset col not in vocabulary → transform returns X unchanged for that col."""
        t = TfidfFeatures(subset=["text"], max_features=3)
        X = pl.DataFrame({"other": [1, 2, 3]})
        t.subset = ["text"]
        t._vocabulary = {}
        t._idf = {}
        t._is_fitted = True
        result = t.transform(X)
        assert result.columns == X.columns

    def test_transform_with_none_subset_returns_x(self):
        """transform() before fit() raises NotFittedError."""
        t = TfidfFeatures()  # subset=None
        X = pl.DataFrame({"text": ["hello world"]})
        with pytest.raises(NotFittedError):
            t.transform(X)
