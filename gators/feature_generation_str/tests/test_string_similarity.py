import polars as pl
import pytest
from pydantic import ValidationError
from rapidfuzz.distance import JaroWinkler, Levenshtein

from gators.feature_generation_str import StringSimilarity


@pytest.fixture
def X_names():
    return pl.DataFrame(
        {
            "billing_name": ["John Smith", "Jane Doe", "Robert Paulson"],
            "shipping_name": ["John Smith", "Jon Doe", "Bob Paulson"],
        }
    )


def test_transform_levenshtein(X_names):
    transformer = StringSimilarity(
        subset_a=["billing_name"], subset_b=["shipping_name"], method="levenshtein"
    )
    result = transformer.fit_transform(X_names)
    col = "billing_name__shipping_name__levenshtein_similarity"
    assert col in result.columns
    assert result[col][0] == pytest.approx(1.0)
    assert result[col][1] == pytest.approx(Levenshtein.normalized_similarity("jane doe", "jon doe"))


def test_transform_jaro_winkler(X_names):
    transformer = StringSimilarity(
        subset_a=["billing_name"], subset_b=["shipping_name"], method="jaro_winkler"
    )
    result = transformer.fit_transform(X_names)
    col = "billing_name__shipping_name__jaro_winkler_similarity"
    assert result[col][2] == pytest.approx(
        JaroWinkler.normalized_similarity("robert paulson", "bob paulson")
    )


def test_transform_case_sensitive():
    X = pl.DataFrame({"a": ["ABC"], "b": ["abc"]})
    case_insensitive = StringSimilarity(subset_a=["a"], subset_b=["b"], case_sensitive=False)
    result = case_insensitive.fit_transform(X)
    assert result["a__b__levenshtein_similarity"][0] == pytest.approx(1.0)

    case_sensitive = StringSimilarity(subset_a=["a"], subset_b=["b"], case_sensitive=True)
    result = case_sensitive.fit_transform(X)
    assert result["a__b__levenshtein_similarity"][0] == pytest.approx(
        Levenshtein.normalized_similarity("ABC", "abc")
    )


def test_transform_null_values():
    X = pl.DataFrame({"a": ["John", None, "Jane"], "b": [None, "Smith", "Jane"]})
    transformer = StringSimilarity(subset_a=["a"], subset_b=["b"])
    result = transformer.fit_transform(X)
    col = "a__b__levenshtein_similarity"
    assert result[col][0] is None
    assert result[col][1] is None
    assert result[col][2] == pytest.approx(1.0)


def test_transform_custom_column_names_and_drop(X_names):
    transformer = StringSimilarity(
        subset_a=["billing_name"],
        subset_b=["shipping_name"],
        new_column_names=["name_similarity"],
        drop_columns=True,
    )
    result = transformer.fit_transform(X_names)
    assert result.columns == ["name_similarity"]


def test_multiple_pairs():
    X = pl.DataFrame(
        {
            "name_a": ["John", "Jane"],
            "name_b": ["Jon", "Jane"],
            "addr_a": ["1 Main St", "2 Oak Ave"],
            "addr_b": ["1 Main St", "99 Elm St"],
        }
    )
    transformer = StringSimilarity(
        subset_a=["name_a", "addr_a"], subset_b=["name_b", "addr_b"]
    )
    result = transformer.fit_transform(X)
    assert "name_a__name_b__levenshtein_similarity" in result.columns
    assert "addr_a__addr_b__levenshtein_similarity" in result.columns
    assert result["addr_a__addr_b__levenshtein_similarity"][0] == pytest.approx(1.0)


def test_mismatched_lengths_raises():
    with pytest.raises(ValidationError):
        StringSimilarity(subset_a=["a", "b"], subset_b=["c"])


def test_mismatched_new_column_names_length_raises():
    with pytest.raises(ValidationError):
        StringSimilarity(
            subset_a=["a", "b"], subset_b=["c", "d"], new_column_names=["only_one"]
        )


def test_output_dtypes_are_float64(X_names):
    transformer = StringSimilarity(subset_a=["billing_name"], subset_b=["shipping_name"])
    transformer.fit(X_names)
    assert transformer._output_dtypes == {
        "billing_name__shipping_name__levenshtein_similarity": pl.Float64
    }
