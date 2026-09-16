import polars as pl
import pytest

from gators.feature_generation_str import WordStatistics


@pytest.fixture
def X():
    return pl.DataFrame(
        {
            "address": ["123 Main St", "One Infinite Loop", "", None],
        }
    )


def test_transform_n_words(X):
    transformer = WordStatistics(subset=["address"], features=["n_words"])
    result = transformer.fit_transform(X)
    assert result["address__n_words"].to_list() == [3.0, 3.0, 0.0, 0.0]


def test_transform_avg_word_length(X):
    transformer = WordStatistics(subset=["address"], features=["avg_word_length"])
    result = transformer.fit_transform(X)
    # "123 Main St" -> word lengths [3, 4, 2] -> mean = 3.0
    assert result["address__avg_word_length"][0] == pytest.approx(3.0)
    assert result["address__avg_word_length"][2] == 0.0
    assert result["address__avg_word_length"][3] == 0.0


def test_transform_n_unique_words():
    X = pl.DataFrame({"text": ["the cat and the dog", "unique words only here"]})
    transformer = WordStatistics(subset=["text"], features=["n_unique_words"])
    result = transformer.fit_transform(X)
    assert result["text__n_unique_words"][0] == 4.0
    assert result["text__n_unique_words"][1] == 4.0


def test_transform_max_min_word_length(X):
    transformer = WordStatistics(subset=["address"], features=["max_word_length", "min_word_length"])
    result = transformer.fit_transform(X)
    # "One Infinite Loop" -> [3, 8, 4]
    assert result["address__max_word_length"][1] == 8.0
    assert result["address__min_word_length"][1] == 3.0
    assert result["address__max_word_length"][2] == 0.0
    assert result["address__min_word_length"][2] == 0.0


def test_transform_default_subset_autodetect():
    X = pl.DataFrame({"text": ["hello world"], "num": [1]})
    transformer = WordStatistics()
    result = transformer.fit_transform(X)
    assert "text__n_words" in result.columns
    assert "num__n_words" not in result.columns


def test_transform_drop_columns(X):
    transformer = WordStatistics(subset=["address"], features=["n_words"], drop_columns=True)
    result = transformer.fit_transform(X)
    assert "address" not in result.columns
    assert "address__n_words" in result.columns


def test_invalid_feature_raises():
    with pytest.raises(ValueError):
        WordStatistics(subset=["address"], features=["not_a_feature"])


def test_output_dtypes_are_float64(X):
    transformer = WordStatistics(subset=["address"], features=["n_words"])
    transformer.fit(X)
    assert transformer._output_dtypes == {"address__n_words": pl.Float64}
