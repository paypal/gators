"""ONNX converters for gators.clippers transformers.

Importing this module registers all supported converters.

Supported
---------
All _BaseClipper subclasses (CustomClipper, GaussianClipper, IQRClipper,
MADClipper, QuantileClipper): Clip(X, lower, upper) using fitted _clip_bounds.
"""
from __future__ import annotations

from ..clippers._base_clipper import _BaseClipper
from ._converters import _onnx_type_to_numpy, get_input_onnx_type, to_onnx_nodes

try:
    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'onnx' package is required for ONNX export. "
        "Install it with: pip install gators[onnx]"
    ) from exc


@to_onnx_nodes.register(_BaseClipper)
def _base_clipper_to_onnx_nodes(
    transformer: _BaseClipper,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Clip(X, lower, upper) using fitted _clip_bounds.

    _clip_bounds maps column name to (lower, upper) tuple stored during fit().
    All clippers share this interface: CustomClipper, GaussianClipper,
    IQRClipper, MADClipper, and QuantileClipper.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    targeted = set(transformer._clip_bounds)

    for col, in_name in input_names.items():
        if col not in targeted:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, (lower, upper) in transformer._clip_bounds.items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = output_names[col]
        p = f"{in_name}__Clipper"
        lower_init = f"{p}__lower"
        upper_init = f"{p}__upper"
        np_dtype = _onnx_type_to_numpy(get_input_onnx_type(transformer, col))
        initializers.append(
            onnx.numpy_helper.from_array(np.array([lower], dtype=np_dtype), name=lower_init)
        )
        initializers.append(
            onnx.numpy_helper.from_array(np.array([upper], dtype=np_dtype), name=upper_init)
        )
        # Clip(input, min, max) — min and max are inputs in opset 11+
        nodes.append(oh.make_node("Clip", inputs=[in_name, lower_init, upper_init], outputs=[out_name]))

    return nodes, initializers
