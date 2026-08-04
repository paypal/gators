"""Test that TreeBasedDiscretizer raises a clear ImportError when lightgbm is absent."""

import sys
from unittest.mock import patch

import polars as pl
import pytest

from gators.discretizers import TreeBasedDiscretizer


def test_tree_based_discretizer_import_error_without_lightgbm():
    """fit() raises ImportError with an install hint when lightgbm is absent."""
    X = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pl.Series([0, 0, 1, 1, 1])
    disc = TreeBasedDiscretizer(subset=["a"], num_bins=2)

    # Setting sys.modules["lightgbm"] = None makes `import lightgbm` raise ImportError
    with patch.dict(sys.modules, {"lightgbm": None}):
        with pytest.raises(ImportError, match="gators\\[tree\\]"):
            disc.fit(X, y)
