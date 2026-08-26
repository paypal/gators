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
values above all edges.

as_numerics=True  → float bin indices (0.0, 1.0, …)
as_numerics=False → string bin labels via ai.onnx.ml.LabelEncoder
                    (INT64 index → label string from _labels[col])
"""
from __future__ import annotations

import polars as pl

from ..discretizers._base_discretizer import _BaseDiscretizer
from ._converters import (
    _POLARS_TO_ONNX,
    _onnx_type_to_numpy,
    get_input_onnx_type,
    get_output_columns,
    get_output_onnx_type,
    to_onnx_nodes,
)

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
    new_cols = [name for names in transformer._column_mapping.values() for name in names]
    if transformer.drop_columns:
        return passthrough + new_cols
    return passthrough + new_cols


# ── ONNX nodes ────────────────────────────────────────────────────────────────

@get_output_onnx_type.register(_BaseDiscretizer)
def _discretizer_output_onnx_type(transformer: _BaseDiscretizer, col: str) -> int:
    # Not using _output_dtypes for the numeric case: _output_dtypes declares a fixed
    # Float64, but the ONNX node casts the bin index to match the INPUT column's
    # precision (FLOAT/DOUBLE) to keep the graph's float_datatype consistent.
    subset = set(transformer.subset or [])
    col_map = transformer._column_mapping or {}
    output_cols = {name for names in col_map.values() for name in names}
    # inplace: targeted columns are the subset itself (column_mapping may be empty)
    is_discretized = (transformer.inplace and col in subset) or col in output_cols
    if not getattr(transformer, "as_numerics", True) and is_discretized:
        return TensorProto.STRING
    dtypes = getattr(transformer, "_input_dtypes", {})
    dtype = dtypes.get(col)
    # _POLARS_TO_ONNX has no String entry (discretizers never emit String themselves),
    # so pass-through String columns must be special-cased or they'd silently fall
    # through to the FLOAT default.
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@to_onnx_nodes.register(_BaseDiscretizer)
def _base_discretizer_to_onnx_nodes(
    transformer: _BaseDiscretizer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Bin assignment: bin_idx = sum(Greater(X, edge_i) for edge_i in edges).

    as_numerics=True  → float bin indices (0.0, 1.0, 2.0, …)
    as_numerics=False → string bin labels via ai.onnx.ml.LabelEncoder
    """
    as_numerics = getattr(transformer, "as_numerics", True)
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
        out_col = col if transformer.inplace else transformer._column_mapping.get(col, [col])[0]
        out_name = output_names.get(out_col, out_col)
        # inplace=False, drop_columns=False: original column also passes through
        if not transformer.inplace and not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
        p = f"{in_name}__Disc"
        # Route float index to a temp tensor when a label-map step follows
        float_out = out_name if as_numerics else f"{p}__floatidx"

        if not edges:
            zero_init = f"{p}__zero"
            zero_mul = f"{p}__zmul"
            initializers.append(
                onnx.numpy_helper.from_array(
                    np.array([0.0], dtype=_onnx_type_to_numpy(get_input_onnx_type(transformer, col))),
                    name=zero_init,
                )
            )
            nodes.append(oh.make_node("Mul", inputs=[in_name, zero_init], outputs=[zero_mul]))
            nodes.append(oh.make_node("Add", inputs=[zero_mul, zero_init], outputs=[float_out]))
        else:
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

            running = term_names[0]
            for j, term in enumerate(term_names[1:]):
                add_out = float_out if j == len(term_names) - 2 else f"{p}__s{j}"
                nodes.append(oh.make_node("Add", inputs=[running, term], outputs=[add_out]))
                running = add_out

            if len(term_names) == 1:
                nodes.append(oh.make_node("Identity", inputs=[term_names[0]], outputs=[float_out]))

        if not as_numerics:
            labels = transformer._labels.get(col, [])
            int_out = f"{p}__intidx"
            nodes.append(oh.make_node("Cast", inputs=[float_out], outputs=[int_out], to=TensorProto.INT64))
            nodes.append(
                oh.make_node(
                    "LabelEncoder",
                    domain="ai.onnx.ml",
                    inputs=[int_out],
                    outputs=[out_name],
                    keys_int64s=list(range(len(labels))),
                    values_strings=labels,
                    default_string="",
                )
            )

    return nodes, initializers
