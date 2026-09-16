# SPDX-License-Identifier: Apache-2.0
class NotFittedError(ValueError, AttributeError):
    """Raised when transform() is called before fit()."""
