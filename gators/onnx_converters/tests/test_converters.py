"""Unit tests for the generic (non-registered-type) fallbacks in _converters.py."""

from types import SimpleNamespace

import numpy as np
import polars as pl
from onnx import TensorProto

from gators.onnx_converters._converters import get_output_onnx_type


def test_generic_output_dtype_string():
    """_output_dtypes declares String -> STRING, via the generic singledispatch base."""
    t = SimpleNamespace(_output_dtypes={"new_col": pl.String}, _input_dtypes={}, _column_mapping={})
    assert get_output_onnx_type(t, "new_col") == TensorProto.STRING


def test_generic_output_dtype_other():
    """_output_dtypes declares a non-string/non-float dtype -> _POLARS_TO_ONNX fallback lookup."""
    t = SimpleNamespace(_output_dtypes={"new_col": pl.Boolean}, _input_dtypes={}, _column_mapping={})
    assert get_output_onnx_type(t, "new_col") == TensorProto.BOOL


def test_generic_passthrough_via_input_dtypes():
    """Column not in _output_dtypes but present in _input_dtypes -> delegates to get_input_onnx_type."""
    t = SimpleNamespace(_output_dtypes={}, _input_dtypes={"col": pl.Float64}, _column_mapping={})
    assert get_output_onnx_type(t, "col") == TensorProto.DOUBLE


def test_generic_reverse_column_mapping_lookup():
    """Column resolved via reversed _column_mapping when absent from _output_dtypes/_input_dtypes."""
    t = SimpleNamespace(
        _output_dtypes={},
        _input_dtypes={"orig": pl.Float32},
        _column_mapping={"orig": ["new_col"]},
    )
    assert get_output_onnx_type(t, "new_col") == TensorProto.FLOAT


def test_generic_unresolved_returns_undefined():
    """Column absent from every lookup source -> UNDEFINED."""
    t = SimpleNamespace(_output_dtypes={}, _input_dtypes={}, _column_mapping={})
    assert get_output_onnx_type(t, "unknown") == TensorProto.UNDEFINED


def test_onnx_type_to_numpy_all_branches():
    """_onnx_type_to_numpy maps DOUBLE/INT64/other (FLOAT fallback) to the right numpy dtype."""
    from gators.onnx_converters._converters import _onnx_type_to_numpy

    assert _onnx_type_to_numpy(TensorProto.DOUBLE) is np.float64
    assert _onnx_type_to_numpy(TensorProto.INT64) is np.int64
    assert _onnx_type_to_numpy(TensorProto.FLOAT) is np.float32
