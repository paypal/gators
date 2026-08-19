"""ONNX converters for gators.discretizers transformers.

Importing this module registers all supported converters.

Supported (registered on _BaseDiscretizer)
------------------------------------------
CustomDiscretizer, EqualLengthDiscretizer, EqualSizeDiscretizer,
GeometricDiscretizer, QuantileDiscretizer

All share the same fitted attribute: _bins: dict[str, list[float]]
containing the interior break-points per column.

ONNX implementation
-------------------
Bin assignment via counting how many break-points each value exceeds:

    bin_idx = sum( Greater(X, edge_i) for edge_i in _bins[col] )

This yields 0 for values below all edges, increasing to len(edges) for
values above all edges — equivalent to as_numerics=True output.  The ONNX
graph always produces float bin indices regardless of the as_numerics setting
used when fitting the transformer.

Not supported
-------------
KMeansDiscretizer    : centroid distances require a matrix op over training data.
TreeBasedDiscretizer : nested conditionals are complex to express in ONNX.
"""
from __future__ import annotations

from ._converters import _onnx_type_to_numpy, get_input_onnx_type, get_output_columns, to_onnx_nodes
from ._exceptions import OnnxNotSupportedError
from ..discretizers._base_discretizer import _BaseDiscretizer

try:
    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper
    from onnx import TensorProto
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'onnx' package is required for ONNX export. "
        "Install it with: pip install gators[onnx]"
    ) from exc


# ── Schema hook ───────────────────────────────────────────────────────────────

@get_output_columns.register(_BaseDiscretizer)
def _discretizer_output_columns(transformer: _BaseDiscretizer, input_columns: list[str]) -> list[str]:
    targeted = set(transformer._column_mapping)
    if transformer.drop_columns:
        passthrough = [c for c in input_columns if c not in targeted]
    else:
        passthrough = list(input_columns)
    # inplace=True → output cols keep original names (same as input)
    # inplace=False → _column_mapping values give the new names
    if transformer.inplace:
        return list(input_columns)  # columns stay in place; output schema unchanged
    new_cols = list(transformer._column_mapping.values())
    if transformer.drop_columns:
        return passthrough + new_cols
    return passthrough + new_cols


# ── ONNX nodes ────────────────────────────────────────────────────────────────

@to_onnx_nodes.register(_BaseDiscretizer)
def _base_discretizer_to_onnx_nodes(
    transformer: _BaseDiscretizer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Bin assignment via: bin_idx = sum(Greater(X, edge_i) for edge_i in edges).

    Always outputs float bin indices (0.0, 1.0, 2.0, …); as_numerics=False
    (string bin labels) is not supported.
    """
    if not getattr(transformer, "as_numerics", True):
        raise OnnxNotSupportedError(
            f"{type(transformer).__name__} with as_numerics=False is not supported "
            "in ONNX export. Refit with as_numerics=True."
        )
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    targeted = set(transformer._bins)

    # Passthrough columns
    for col, in_name in input_names.items():
        if col not in targeted and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, edges in transformer._bins.items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        # Inplace: output to same column name; not-inplace: output to mapped name
        out_col = col if transformer.inplace else transformer._column_mapping.get(col, col)
        out_name = output_names.get(out_col, out_col)

        if not edges:
            # No edges → constant bin 0
            zero_init = f"{in_name}__Disc__zero"
            zero_mul = f"{in_name}__Disc__zmul"
            zero_add = f"{in_name}__Disc__zadd"
            initializers.append(
                onnx.numpy_helper.from_array(
                    np.array([0.0], dtype=_onnx_type_to_numpy(get_input_onnx_type(transformer, col))),
                    name=zero_init,
                )
            )
            nodes.append(oh.make_node("Mul", inputs=[in_name, zero_init], outputs=[zero_mul]))
            nodes.append(oh.make_node("Add", inputs=[zero_mul, zero_init], outputs=[out_name]))
            continue

        p = f"{in_name}__Disc"
        term_names: list[str] = []
        for i, edge in enumerate(edges):
            edge_init = f"{p}__e{i}"
            gt_out = f"{p}__gt{i}"
            cast_out = f"{p}__c{i}"
            initializers.append(
                onnx.numpy_helper.from_array(
                    np.array([edge], dtype=_onnx_type_to_numpy(get_input_onnx_type(transformer, col))),
                    name=edge_init,
                )
            )
            nodes.append(oh.make_node("Greater", inputs=[in_name, edge_init], outputs=[gt_out]))
            nodes.append(oh.make_node("Cast", inputs=[gt_out], outputs=[cast_out], to=get_input_onnx_type(transformer, col)))
            term_names.append(cast_out)

        # Sum all cast terms to get the bin index
        running = term_names[0]
        for j, term in enumerate(term_names[1:]):
            add_out = out_name if j == len(term_names) - 2 else f"{p}__s{j}"
            nodes.append(oh.make_node("Add", inputs=[running, term], outputs=[add_out]))
            running = add_out

        # If only one edge, the single term IS the output
        if len(term_names) == 1:
            nodes.append(oh.make_node("Identity", inputs=[term_names[0]], outputs=[out_name]))

    return nodes, initializers
