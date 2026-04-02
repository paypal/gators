import polars as pl
import pytest

from gators.feature_generation_str import PatternDetector


class TestPatternDetector:
    """Tests for PatternDetector transformer."""

    def test_email_detection(self):
        """Test email pattern detection."""
        X = pl.DataFrame(
            {
                "contact": [
                    "user@test.com",
                    "ADMIN@SITE.ORG",
                    "not-an-email",
                    "test@",
                    "@test.com",
                    None,
                    "",
                ]
            }
        )

        transformer = PatternDetector(subset=["contact"], patterns=["is_email"])
        result = transformer.fit_transform(X)

        assert result["contact__is_email"][0] is True
        assert result["contact__is_email"][1] is True
        assert result["contact__is_email"][2] is False
        assert result["contact__is_email"][3] is False
        assert result["contact__is_email"][4] is False
        assert result["contact__is_email"][5] is False  # null
        assert result["contact__is_email"][6] is False  # empty

    def test_url_detection(self):
        """Test URL pattern detection."""
        X = pl.DataFrame(
            {
                "link": [
                    "https://site.com",
                    "http://example.org",
                    "ftp://file.com",
                    "site.com",
                    None,
                    "",
                ]
            }
        )

        transformer = PatternDetector(subset=["link"], patterns=["is_url"])
        result = transformer.fit_transform(X)

        assert result["link__is_url"][0] is True
        assert result["link__is_url"][1] is True
        assert result["link__is_url"][2] is False  # ftp not matched
        assert result["link__is_url"][3] is False  # no protocol
        assert result["link__is_url"][4] is False  # null
        assert result["link__is_url"][5] is False  # empty

    def test_phone_detection(self):
        """Test phone number pattern detection."""
        X = pl.DataFrame(
            {
                "phone": [
                    "555-1234",
                    "(555) 123-4567",
                    "5551234567",
                    "555.123.4567",
                    "not-a-phone!",
                    None,
                ]
            }
        )

        transformer = PatternDetector(subset=["phone"], patterns=["is_phone"])
        result = transformer.fit_transform(X)

        assert result["phone__is_phone"][0] is True
        assert result["phone__is_phone"][1] is True
        assert result["phone__is_phone"][2] is True
        assert result["phone__is_phone"][3] is True
        assert result["phone__is_phone"][4] is False  # contains !
        assert result["phone__is_phone"][5] is False  # null

    def test_numeric_detection(self):
        """Test numeric pattern detection."""
        X = pl.DataFrame({"value": ["123", "45.67", "-89", "1.2.3", "abc", "12a", "", None]})

        transformer = PatternDetector(subset=["value"], patterns=["is_numeric"])
        result = transformer.fit_transform(X)

        assert result["value__is_numeric"][0] is True
        assert result["value__is_numeric"][1] is True
        assert result["value__is_numeric"][2] is True
        assert result["value__is_numeric"][3] is False  # multiple dots
        assert result["value__is_numeric"][4] is False
        assert result["value__is_numeric"][5] is False
        assert result["value__is_numeric"][6] is False  # empty
        assert result["value__is_numeric"][7] is False  # null

    def test_alphanumeric_and_alpha_detection(self):
        """Test alphanumeric and alpha-only detection."""
        X = pl.DataFrame({"code": ["ABC123", "XYZ", "123", "ABC-123", "", None]})

        transformer = PatternDetector(subset=["code"], patterns=["is_alphanumeric", "is_alpha"])
        result = transformer.fit_transform(X)

        # ABC123
        assert result["code__is_alphanumeric"][0] is True
        assert result["code__is_alpha"][0] is False

        # XYZ
        assert result["code__is_alphanumeric"][1] is True
        assert result["code__is_alpha"][1] is True

        # 123
        assert result["code__is_alphanumeric"][2] is True
        assert result["code__is_alpha"][2] is False

        # ABC-123 (contains -)
        assert result["code__is_alphanumeric"][3] is False
        assert result["code__is_alpha"][3] is False

        # Empty
        assert result["code__is_alphanumeric"][4] is False
        assert result["code__is_alpha"][4] is False

        # Null
        assert result["code__is_alphanumeric"][5] is False
        assert result["code__is_alpha"][5] is False

    def test_url_component_detection(self):
        """Test URL component detection (has_http, has_www, has_at)."""
        X = pl.DataFrame(
            {
                "text": [
                    "https://www.example.com",
                    "http://example.com",
                    "www.example.com",
                    "email@test.com",
                    "plain text",
                    None,
                ]
            }
        )

        transformer = PatternDetector(subset=["text"], patterns=["has_http", "has_www", "has_at"])
        result = transformer.fit_transform(X)

        # https://www.example.com
        assert result["text__has_http"][0] is True
        assert result["text__has_www"][0] is True
        assert result["text__has_at"][0] is False

        # http://example.com
        assert result["text__has_http"][1] is True
        assert result["text__has_www"][1] is False
        assert result["text__has_at"][1] is False

        # www.example.com
        assert result["text__has_http"][2] is False
        assert result["text__has_www"][2] is True
        assert result["text__has_at"][2] is False

        # email@test.com
        assert result["text__has_http"][3] is False
        assert result["text__has_www"][3] is False
        assert result["text__has_at"][3] is True

        # plain text
        assert result["text__has_http"][4] is False
        assert result["text__has_www"][4] is False
        assert result["text__has_at"][4] is False

        # null
        assert result["text__has_http"][5] is False
        assert result["text__has_www"][5] is False
        assert result["text__has_at"][5] is False

    def test_multiple_columns(self):
        """Test transformation on multiple columns."""
        X = pl.DataFrame(
            {
                "col1": ["user@test.com", "not-email"],
                "col2": ["https://site.com", "no-url"],
            }
        )

        transformer = PatternDetector(subset=["col1", "col2"], patterns=["is_email", "is_url"])
        result = transformer.fit_transform(X)

        assert "col1__is_email" in result.columns
        assert "col1__is_url" in result.columns
        assert "col2__is_email" in result.columns
        assert "col2__is_url" in result.columns

    def test_auto_detect_string_columns(self):
        """Test automatic detection of string columns."""
        X = pl.DataFrame({"text": ["user@test.com", "test"], "num": [1, 2], "float": [1.5, 2.5]})

        transformer = PatternDetector(patterns=["is_email"])
        result = transformer.fit_transform(X)

        # Should only create features for 'text' column
        assert "text__is_email" in result.columns
        assert "num__is_email" not in result.columns
        assert "float__is_email" not in result.columns

    def test_drop_columns(self):
        """Test drop_columns parameter."""
        X = pl.DataFrame({"email": ["user@test.com", "admin@site.org"], "name": ["Alice", "Bob"]})

        transformer = PatternDetector(subset=["email"], patterns=["is_email"], drop_columns=True)
        result = transformer.fit_transform(X)

        assert "email" not in result.columns
        assert "name" in result.columns
        assert "email__is_email" in result.columns

    def test_all_patterns(self):
        """Test using all available patterns."""
        X = pl.DataFrame({"text": ["test@email.com", "https://site.com"]})

        transformer = PatternDetector(
            subset=["text"],
            patterns=[
                "is_numeric",
                "is_email",
                "is_url",
                "is_phone",
                "is_alphanumeric",
                "is_alpha",
                "has_http",
                "has_www",
                "has_at",
            ],
        )
        result = transformer.fit_transform(X)

        # Should have 9 new features
        new_features = [col for col in result.columns if col.startswith("text__")]
        assert len(new_features) == 9

    def test_invalid_pattern(self):
        """Test validation of pattern names."""
        with pytest.raises(ValueError, match="Pattern .* is not supported"):
            PatternDetector(subset=["text"], patterns=["invalid_pattern"])

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({"text": []}, schema={"text": pl.String})

        transformer = PatternDetector(subset=["text"], patterns=["is_email"])
        result = transformer.fit_transform(X)

        assert len(result) == 0
        assert "text__is_email" in result.columns

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"email": ["user@test.com", "test"]})

        transformer = PatternDetector(subset=["email"], patterns=["is_email"])

        # Test fit returns self
        assert transformer.fit(X) is transformer

        # Test fit_transform
        result = transformer.fit_transform(X)
        assert isinstance(result, pl.DataFrame)

        # Test separate fit and transform
        transformer2 = PatternDetector(subset=["email"], patterns=["is_email"])
        transformer2.fit(X)
        result2 = transformer2.transform(X)
        assert result.equals(result2)
