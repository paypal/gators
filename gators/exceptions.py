class NotFittedError(ValueError, AttributeError):
    """Raised when transform() is called before fit()."""
