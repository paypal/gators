import polars as pl
import pytest

from gators.feature_generation_str import CharacterStatistics


class TestCharacterStatistics:
    """Tests for CharacterStatistics transformer."""

    def test_basic_character_counts(self):
        """Test basic character counting features."""
        X =pl.DataFrame(
            {
                "text": ["Hello123", "WORLD!!!", "Test 99", "", None],
                "name": ["John Doe", "MARY", "alice", "X", ""],
            }
        )

        transformer = CharacterStatistics(
            subset=["text"],
            features=[
                "n_digits",
                "n_letters",
                "n_uppercase",
                "n_lowercase",
                "n_spaces",
                "n_special",
            ],
        )
        result = transformer.fit_transform(X)

        # Check Hello123
        assert result["text__n_digits"][0] == 3
        assert result["text__n_letters"][0] == 5
        assert result["text__n_uppercase"][0] == 1
        assert result["text__n_lowercase"][0] == 4
        assert result["text__n_spaces"][0] == 0
        assert result["text__n_special"][0] == 0

        # Check WORLD!!!
        assert result["text__n_digits"][1] == 0
        assert result["text__n_letters"][1] == 5
        assert result["text__n_uppercase"][1] == 5
        assert result["text__n_lowercase"][1] == 0
        assert result["text__n_spaces"][1] == 0
        assert result["text__n_special"][1] == 3

        # Check Test 99
        assert result["text__n_digits"][2] == 2
        assert result["text__n_letters"][2] == 4
        assert result["text__n_uppercase"][2] == 1
        assert result["text__n_lowercase"][2] == 3
        assert result["text__n_spaces"][2] == 1
        assert result["text__n_special"][2] == 0

        # Check empty string
        assert result["text__n_digits"][3] == 0
        assert result["text__n_letters"][3] == 0

        # Check null (treated as empty)
        assert result["text__n_digits"][4] == 0
        assert result["text__n_letters"][4] == 0

    def test_ratio_features(self):
        """Test ratio-based features."""
        X =pl.DataFrame({"text": ["ABC123", "abcdef", "!!!###", "A", ""]})

        transformer = CharacterStatistics(
            subset=["text"],
            features=["ratio_uppercase", "ratio_digits", "ratio_special"],
        )
        result = transformer.fit_transform(X)

        # ABC123: 3 uppercase / 6 letters = 0.5, 3 digits / 6 total = 0.5, 0 special / 6 = 0
        assert result["text__ratio_uppercase"][0] == pytest.approx(0.5)
        assert result["text__ratio_digits"][0] == pytest.approx(0.5)
        assert result["text__ratio_special"][0] == 0.0

        # abcdef: 0/6 = 0, 0/6 = 0, 0/6 = 0
        assert result["text__ratio_uppercase"][1] == 0.0
        assert result["text__ratio_digits"][1] == 0.0
        assert result["text__ratio_special"][1] == 0.0

        # !!!###: no letters so 0, 0/6 = 0, 6/6 = 1.0
        assert result["text__ratio_uppercase"][2] == 0.0
        assert result["text__ratio_digits"][2] == 0.0
        assert result["text__ratio_special"][2] == 1.0

        # A: 1/1 = 1.0, 0/1 = 0, 0/1 = 0
        assert result["text__ratio_uppercase"][3] == 1.0
        assert result["text__ratio_digits"][3] == 0.0
        assert result["text__ratio_special"][3] == 0.0

        # Empty: 0/0 = 0 (handled by when clause)
        assert result["text__ratio_uppercase"][4] == 0.0
        assert result["text__ratio_digits"][4] == 0.0
        assert result["text__ratio_special"][4] == 0.0

    def test_multiple_columns(self):
        """Test transformation on multiple columns."""
        X =pl.DataFrame({"col1": ["A1", "B2"], "col2": ["!!", "XY"]})

        transformer = CharacterStatistics(
            subset=["col1", "col2"], features=["n_digits", "n_letters"]
        )
        result = transformer.fit_transform(X)

        assert "col1__n_digits" in result.columns
        assert "col1__n_letters" in result.columns
        assert "col2__n_digits" in result.columns
        assert "col2__n_letters" in result.columns

        assert result["col1__n_digits"][0] == 1
        assert result["col1__n_letters"][0] == 1
        assert result["col2__n_digits"][0] == 0
        assert result["col2__n_letters"][0] == 0

    def test_auto_detect_string_columns(self):
        """Test automatic detection of string columns."""
        X =pl.DataFrame(
            {"text": ["Hello", "World"], "num": [1, 2], "float": [1.5, 2.5]}
        )

        transformer = CharacterStatistics(features=["n_digits", "n_letters"])
        result = transformer.fit_transform(X)

        # Should only create features for 'text' column
        assert "text__n_digits" in result.columns
        assert "text__n_letters" in result.columns
        assert "num__n_digits" not in result.columns
        assert "float__n_digits" not in result.columns

    def test_drop_columns(self):
        """Test drop_columns parameter."""
        X =pl.DataFrame({"text": ["Hello", "World"], "name": ["Alice", "Bob"]})

        transformer = CharacterStatistics(
            subset=["text"], features=["n_digits"], drop_columns=True
        )
        result = transformer.fit_transform(X)

        assert "text" not in result.columns
        assert "name" in result.columns
        assert "text__n_digits" in result.columns

    def test_all_features(self):
        """Test using all available features."""
        X =pl.DataFrame({"text": ["Hello123!", "Test"]})

        transformer = CharacterStatistics(
            subset=["text"],
            features=[
                "n_digits",
                "n_letters",
                "n_uppercase",
                "n_lowercase",
                "n_spaces",
                "n_special",
                "n_unique_chars",
                "ratio_uppercase",
                "ratio_digits",
                "ratio_special",
            ],
        )
        result = transformer.fit_transform(X)

        # Should have 10 new features
        new_features = [col for col in result.columns if col.startswith("text__")]
        assert len(new_features) == 10

    def test_invalid_feature(self):
        """Test validation of feature names."""
        with pytest.raises(ValueError, match="Feature .* is not supported"):
            CharacterStatistics(subset=["text"], features=["invalid_feature"])

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X =pl.DataFrame({"text": []}, schema={"text": pl.String})

        transformer = CharacterStatistics(
            subset=["text"], features=["n_digits", "n_letters"]
        )
        result = transformer.fit_transform(X)

        assert len(result) == 0
        assert "text__n_digits" in result.columns
        assert "text__n_letters" in result.columns

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X =pl.DataFrame({"text": ["Hello", "World"]})

        transformer = CharacterStatistics(subset=["text"], features=["n_digits"])

        # Test fit returns self
        assert transformer.fit(X) is transformer

        # Test fit_transform
        result = transformer.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform
        transformer2 = CharacterStatistics(subset=["text"], features=["n_digits"])
        transformer2.fit(X)
        result2 = transformer2.transform(X)
        assert result.equals(result2)
