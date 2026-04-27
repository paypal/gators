from typing import Any, Dict

from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator, TransformerMixin


class _BaseTransformer(BaseModel, BaseEstimator, TransformerMixin):
    """
    Base class for all transformers in the gators library.

    This class provides common functionality for all transformers.

    """

    model_config = ConfigDict(extra="forbid")

    def __init__(self, *args, **kwargs):
        """Initialize transformer with clear error message for positional arguments.
        
        Raises
        ------
        TypeError
            If positional arguments are provided instead of keyword arguments.
        """
        if args:
            raise TypeError(
                f"{self.__class__.__name__}() does not accept positional arguments. "
                "Use keyword arguments instead:\n"
                f"  {self.__class__.__name__}(param1=value1, param2=value2)  # Correct\n"
                f"  {self.__class__.__name__}(value1, value2)                # Wrong - raises this error"
            )
        super().__init__(**kwargs)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        This overrides sklearn's BaseEstimator.get_params() to work with
        Pydantic models instead of relying on __init__ signature introspection.

        Parameters
        ----------
        deep : bool, default=True
            Not used for Pydantic models (included for sklearn compatibility).

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        # Return all Pydantic model fields
        return {key: getattr(self, key) for key in self.__class__.model_fields.keys()}

    def set_params(self, **params) -> "_BaseTransformer":
        """Set parameters for this estimator.

        This overrides sklearn's BaseEstimator.set_params() to work with
        Pydantic models.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
            The estimator instance.

        Raises
        ------
        ValueError
            If an invalid parameter name is provided.
        """
        valid_params = set(self.__class__.model_fields.keys())
        
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params)}"
                )
            setattr(self, key, value)
        
        return self
