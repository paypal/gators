# OnnxNotSupportedError has no onnx dependency and must always be importable.
from ._exceptions import OnnxNotSupportedError

try:
    from . import (
        _clipper_converters,  # noqa: F401
        _data_cleaning_converters,  # noqa: F401
        _discretizer_converters,  # noqa: F401
        _encoder_converters,  # noqa: F401
        _feature_generation_converters,  # noqa: F401
        _feature_generation_dt_converters,  # noqa: F401
        _feature_generation_str_converters,  # noqa: F401
        _imputer_converters,  # noqa: F401
        _scaler_converters,  # noqa: F401
    )
    from ._converters import (
        check_pipeline_onnx_compatibility,
        get_input_onnx_type,
        get_output_columns,
        get_output_onnx_type,
        to_onnx_nodes,
    )
    from ._export import (
        create_session,
        pipeline_to_onnx,
        pipeline_to_scoring_onnx,
        run_session,
        to_onnx_graph,
    )
except ImportError as _e:  # pragma: no cover
    # onnx is optional. Keep the module importable so test collection never
    # errors; calling any function below will raise the original ImportError.
    _msg = _e

    def _unavailable(*args, **kwargs):  # type: ignore[misc]
        raise _msg

    to_onnx_graph = pipeline_to_onnx = to_onnx_nodes = _unavailable  # type: ignore[assignment]
    get_input_onnx_type = get_output_columns = get_output_onnx_type = _unavailable  # type: ignore[assignment]
    create_session = run_session = _unavailable  # type: ignore[assignment]

__all__ = [
    "OnnxNotSupportedError",
    "check_pipeline_onnx_compatibility",
    "create_session",
    "get_input_onnx_type",
    "get_output_columns",
    "get_output_onnx_type",
    "pipeline_to_onnx",
    "pipeline_to_scoring_onnx",
    "run_session",
    "to_onnx_graph",
    "to_onnx_nodes",
]

