"""Tests for _BaseTransformer class."""
import pytest
from pydantic import ValidationError
from sklearn.base import BaseEstimator, TransformerMixin

from gators.transformer._base_transformer import _BaseTransformer


# Create a concrete test class since _BaseTransformer is meant to be subclassed
class ConcreteTransformer(_BaseTransformer):
    """Concrete implementation for testing."""
    param1: str = "default"
    param2: int = 42


class TestBaseTransformerInitialization:
    """Test initialization behavior of _BaseTransformer."""

    def test_init_with_keyword_arguments(self):
        """Test that initialization works with keyword arguments."""
        transformer = ConcreteTransformer(param1="test", param2=100)
        assert transformer.param1 == "test"
        assert transformer.param2 == 100

    def test_init_with_defaults(self):
        """Test that initialization works with default values."""
        transformer = ConcreteTransformer()
        assert transformer.param1 == "default"
        assert transformer.param2 == 42

    def test_init_with_partial_kwargs(self):
        """Test initialization with some keyword arguments."""
        transformer = ConcreteTransformer(param1="custom")
        assert transformer.param1 == "custom"
        assert transformer.param2 == 42

    def test_init_with_positional_arguments_raises_error(self):
        """Test that positional arguments raise a clear TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConcreteTransformer("value1", 123)
        
        error_msg = str(exc_info.value)
        assert "does not accept positional arguments" in error_msg
        assert "Use keyword arguments instead" in error_msg
        assert "ConcreteTransformer" in error_msg
        assert "Correct" in error_msg
        assert "Wrong" in error_msg

    def test_init_with_single_positional_argument_raises_error(self):
        """Test that even a single positional argument raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConcreteTransformer("value1")
        
        assert "does not accept positional arguments" in str(exc_info.value)

    def test_init_with_mixed_args_raises_error(self):
        """Test that mixing positional and keyword arguments raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ConcreteTransformer("value1", param2=100)
        
        assert "does not accept positional arguments" in str(exc_info.value)


class TestBaseTransformerValidation:
    """Test Pydantic validation behavior."""

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden due to model_config."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteTransformer(param1="test", unknown_param="value")
        
        error = exc_info.value
        assert "unknown_param" in str(error)

    def test_type_validation(self):
        """Test that Pydantic validates types."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteTransformer(param2="not_an_int")
        
        error = exc_info.value
        assert "param2" in str(error)


class TestBaseTransformerInheritance:
    """Test inheritance from sklearn base classes."""

    def test_inherits_from_base_estimator(self):
        """Test that _BaseTransformer inherits from BaseEstimator."""
        transformer = ConcreteTransformer()
        assert isinstance(transformer, BaseEstimator)

    def test_inherits_from_transformer_mixin(self):
        """Test that _BaseTransformer inherits from TransformerMixin."""
        transformer = ConcreteTransformer()
        assert isinstance(transformer, TransformerMixin)

    def test_has_get_params_method(self):
        """Test that transformer has get_params method from BaseEstimator."""
        transformer = ConcreteTransformer(param1="test", param2=999)
        # Note: sklearn's get_params introspection doesn't work with our custom __init__
        # but the method exists
        assert hasattr(transformer, 'get_params')
        assert callable(transformer.get_params)

    def test_has_set_params_method(self):
        """Test that transformer has set_params method from BaseEstimator."""
        transformer = ConcreteTransformer()
        # Note: sklearn's set_params introspection doesn't work with our custom __init__
        # but the method exists
        assert hasattr(transformer, 'set_params')
        assert callable(transformer.set_params)

    def test_attribute_access(self):
        """Test that parameters can be accessed and modified directly."""
        transformer = ConcreteTransformer(param1="test", param2=999)
        
        # Can read attributes
        assert transformer.param1 == "test"
        assert transformer.param2 == 999
        
        # Can modify attributes directly
        transformer.param1 = "updated"
        transformer.param2 = 777
        
        assert transformer.param1 == "updated"
        assert transformer.param2 == 777

    def test_get_params_returns_correct_values(self):
        """Test that get_params returns all model fields."""
        transformer = ConcreteTransformer(param1="custom", param2=123)
        params = transformer.get_params()
        
        assert "param1" in params
        assert "param2" in params
        assert params["param1"] == "custom"
        assert params["param2"] == 123

    def test_set_params_with_valid_parameters(self):
        """Test that set_params correctly updates parameters."""
        transformer = ConcreteTransformer(param1="initial", param2=100)
        
        result = transformer.set_params(param1="updated", param2=200)
        
        # Should return self
        assert result is transformer
        
        # Parameters should be updated
        assert transformer.param1 == "updated"
        assert transformer.param2 == 200

    def test_set_params_with_invalid_parameter(self):
        """Test that set_params raises ValueError for invalid parameters."""
        transformer = ConcreteTransformer()
        
        with pytest.raises(ValueError) as exc_info:
            transformer.set_params(invalid_param="value")
        
        error_msg = str(exc_info.value)
        assert "Invalid parameter invalid_param" in error_msg
        assert "ConcreteTransformer" in error_msg
        assert "Valid parameters are:" in error_msg


class TestBaseTransformerModelConfig:
    """Test model_config settings."""

    def test_model_config_extra_forbid(self):
        """Test that model_config forbids extra fields."""
        # This is tested indirectly through the ValidationError test
        # but we can also check the config directly
        config = _BaseTransformer.model_config
        assert config.get('extra') == 'forbid'


class TestConcreteDerivedTransformer:
    """Test a more complex derived transformer."""

    def test_multiple_inheritance_levels(self):
        """Test that derived classes work correctly."""
        class DerivedTransformer(ConcreteTransformer):
            param3: float = 3.14
        
        transformer = DerivedTransformer(param1="derived", param2=50, param3=2.71)
        
        assert transformer.param1 == "derived"
        assert transformer.param2 == 50
        assert transformer.param3 == 2.71
        assert isinstance(transformer, _BaseTransformer)
        assert isinstance(transformer, BaseEstimator)
        assert isinstance(transformer, TransformerMixin)

    def test_derived_class_positional_args_error(self):
        """Test that derived classes also reject positional arguments."""
        class DerivedTransformer(ConcreteTransformer):
            param3: float = 3.14
        
        with pytest.raises(TypeError) as exc_info:
            DerivedTransformer("value1", 123, 4.56)
        
        error_msg = str(exc_info.value)
        assert "does not accept positional arguments" in error_msg
        assert "DerivedTransformer" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
