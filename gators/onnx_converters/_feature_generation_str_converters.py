"""ONNX converters for gators.feature_generation_str transformers.

Importing this module registers all supported converters.

Supported
---------
Lower        : StringNormalizer(case_change_action="LOWER")
Upper        : StringNormalizer(case_change_action="UPPER")
Split        : StringSplit (opset 20) + Gather per split index
SplitExtract : StringSplit (opset 20) + Gather at index n

Not supported (no ONNX string-length operator)
-----------------------------------------------
Length

Notes
-----
Split / SplitExtract use the ONNX ``StringSplit`` operator (opset 20), which
requires ORT ≥ 1.17.  ``StringSplit`` performs a full split (no maxsplit limit)
so the padded output tensor has a dynamic second dimension equal to the maximum
number of parts in the batch.  Gathering beyond that dimension is undefined;
ensure inference batches contain at least one string with enough parts to cover
every requested index.

For ``SplitExtract``: strings with fewer than ``n + 1`` parts produce an empty
string ``""`` in ONNX (the tensor is padded with ``""``), whereas the Polars
transformer returns ``null``.
"""
from __future__ import annotations

import numpy as np
import onnx.numpy_helper
import polars as pl

from ._converters import get_input_onnx_type, get_output_columns, get_output_onnx_type, to_onnx_nodes
from ._exceptions import OnnxNotSupportedError
from ..transformer._base_transformer import _BaseTransformer
from ..feature_generation_str.length import Length
from ..feature_generation_str.lower import Lower
from ..feature_generation_str.split import Split
from ..feature_generation_str.split_extract import SplitExtract
from ..feature_generation_str.upper import Upper

try:
    import onnx
    import onnx.helper as oh
    from onnx import TensorProto
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'onnx' package is required for ONNX export. "
        "Install it with: pip install gators[onnx]"
    ) from exc


# ── shared helpers ────────────────────────────────────────────────────────────

def _str_normalizer_output_columns(transformer: _BaseTransformer, input_columns: list[str]) -> list[str]:
    if transformer.inplace or not transformer._column_mapping:
        return list(input_columns)
    new_cols = list(transformer._column_mapping.values())
    if transformer.drop_columns:
        dropped = set(transformer._column_mapping)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


def _str_normalizer_input_type(transformer: _BaseTransformer, col: str) -> int:
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


def _str_normalizer_output_type(transformer: _BaseTransformer, col: str) -> int:
    if transformer.inplace:
        output_string_cols = set(transformer.subset or [])
    else:
        output_string_cols = set(transformer._column_mapping.values())
    return TensorProto.STRING if col in output_string_cols else TensorProto.FLOAT


def _str_normalizer_nodes(
    transformer: _BaseTransformer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    case_change_action: str,
) -> tuple[list, list]:
    """Emit StringNormalizer nodes for each targeted column; Identity for the rest."""
    nodes: list[onnx.NodeProto] = []
    targeted = set(transformer.subset or [])
    source_to_out: dict[str, str] = {}

    if transformer.inplace:
        # Output tensor names are the same as the input column names
        for col in targeted:
            source_to_out[col] = output_names.get(col, col)
    else:
        # Output tensor names come from _column_mapping
        for orig, new in transformer._column_mapping.items():
            source_to_out[orig] = output_names.get(new, new)

    for col, in_name in input_names.items():
        if col in source_to_out:
            # Apply StringNormalizer
            nodes.append(
                oh.make_node(
                    "StringNormalizer",
                    inputs=[in_name],
                    outputs=[source_to_out[col]],
                    case_change_action=case_change_action,
                )
            )
            # For inplace=False + drop_columns=False the source col also passes through
            if not transformer.inplace and not transformer.drop_columns and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
        elif col in output_names:
            # Passthrough
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    return nodes, []


# ── Lower ─────────────────────────────────────────────────────────────────────

@get_output_columns.register(Lower)
def _lower_output_cols(transformer: Lower, input_columns: list[str]) -> list[str]:
    return _str_normalizer_output_columns(transformer, input_columns)


@get_input_onnx_type.register(Lower)
def _lower_input_type(transformer: Lower, col: str) -> int:
    return _str_normalizer_input_type(transformer, col)


@get_output_onnx_type.register(Lower)
def _lower_output_type(transformer: Lower, col: str) -> int:
    return _str_normalizer_output_type(transformer, col)


@to_onnx_nodes.register(Lower)
def _lower_to_onnx_nodes(
    transformer: Lower,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringNormalizer(case_change_action="LOWER")."""
    return _str_normalizer_nodes(transformer, input_names, output_names, "LOWER")


# ── Upper ─────────────────────────────────────────────────────────────────────

@get_output_columns.register(Upper)
def _upper_output_cols(transformer: Upper, input_columns: list[str]) -> list[str]:
    return _str_normalizer_output_columns(transformer, input_columns)


@get_input_onnx_type.register(Upper)
def _upper_input_type(transformer: Upper, col: str) -> int:
    return _str_normalizer_input_type(transformer, col)


@get_output_onnx_type.register(Upper)
def _upper_output_type(transformer: Upper, col: str) -> int:
    return _str_normalizer_output_type(transformer, col)


@to_onnx_nodes.register(Upper)
def _upper_to_onnx_nodes(
    transformer: Upper,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringNormalizer(case_change_action="UPPER")."""
    return _str_normalizer_nodes(transformer, input_names, output_names, "UPPER")


# ── shared helper for Split / SplitExtract ────────────────────────────────────

def _to_onnx_type(dtype) -> int:
    """Map a Polars dtype to an ONNX TensorProto scalar type."""
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    if dtype == pl.Float64:
        return TensorProto.DOUBLE
    return TensorProto.FLOAT


# ── Split ─────────────────────────────────────────────────────────────────────

@get_output_columns.register(Split)
def _split_output_cols(transformer: Split, input_columns: list[str]) -> list[str]:
    by_clean = transformer.by.replace(" ", "_")
    new_cols = [
        f"{col}__split_{by_clean}_{i}"
        for col in transformer.subset
        for i in range(transformer.max_splits)
    ]
    if transformer.drop_columns:
        dropped = set(transformer.subset)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


@get_input_onnx_type.register(Split)
def _split_input_type(transformer: Split, col: str) -> int:
    if col in transformer.subset:
        return TensorProto.STRING
    return _to_onnx_type(getattr(transformer, "_input_dtypes", {}).get(col))


@get_output_onnx_type.register(Split)
def _split_output_type(transformer: Split, col: str) -> int:
    by_clean = transformer.by.replace(" ", "_")
    split_cols = {
        f"{src}__split_{by_clean}_{i}"
        for src in transformer.subset
        for i in range(transformer.max_splits)
    }
    if col in split_cols:
        return TensorProto.STRING
    return _split_input_type(transformer, col)


@to_onnx_nodes.register(Split)
def _split_to_onnx_nodes(
    transformer: Split,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringSplit (opset 20) → Gather for each split index.

    Uses a full split (no maxsplit) so elements at every requested index are
    exact.  The padded output tensor has a dynamic second dimension; ensure the
    inference batch contains at least one string with >= max_splits parts.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    by_clean = transformer.by.replace(" ", "_")
    targeted = set(transformer.subset)

    for col, in_name in input_names.items():
        if col not in targeted and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    if not transformer.drop_columns:
        for col in transformer.subset:
            if col in input_names and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))

    for col in transformer.subset:
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        y_name = f"{in_name}__strsplit_Y"
        z_name = f"{in_name}__strsplit_Z"
        nodes.append(oh.make_node(
            "StringSplit",
            inputs=[in_name],
            outputs=[y_name, z_name],
            delimiter=transformer.by,
        ))
        for i in range(transformer.max_splits):
            out_col = f"{col}__split_{by_clean}_{i}"
            if out_col not in output_names:  # pragma: no cover
                continue
            idx_name = f"{in_name}__gather_idx_{i}"
            initializers.append(
                onnx.numpy_helper.from_array(np.array(i, dtype=np.int64), name=idx_name)
            )
            nodes.append(oh.make_node(
                "Gather",
                inputs=[y_name, idx_name],
                outputs=[output_names[out_col]],
                axis=1,
            ))

    return nodes, initializers


# ── SplitExtract ──────────────────────────────────────────────────────────────

@get_output_columns.register(SplitExtract)
def _split_extract_output_cols(transformer: SplitExtract, input_columns: list[str]) -> list[str]:
    by_clean = transformer.by.replace(" ", "_")
    new_cols = [f"{col}__split_{by_clean}_{transformer.n}" for col in transformer.subset]
    if transformer.drop_columns:
        dropped = set(transformer.subset)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


@get_input_onnx_type.register(SplitExtract)
def _split_extract_input_type(transformer: SplitExtract, col: str) -> int:
    if col in transformer.subset:
        return TensorProto.STRING
    return _to_onnx_type(getattr(transformer, "_input_dtypes", {}).get(col))


@get_output_onnx_type.register(SplitExtract)
def _split_extract_output_type(transformer: SplitExtract, col: str) -> int:
    by_clean = transformer.by.replace(" ", "_")
    split_cols = {f"{src}__split_{by_clean}_{transformer.n}" for src in transformer.subset}
    if col in split_cols:
        return TensorProto.STRING
    return _split_extract_input_type(transformer, col)


@to_onnx_nodes.register(SplitExtract)
def _split_extract_to_onnx_nodes(
    transformer: SplitExtract,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringSplit (opset 20) → Gather at index n.

    Strings with fewer than n + 1 parts yield ``""`` (ONNX tensor padding)
    rather than ``null`` as in the Polars transformer.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    by_clean = transformer.by.replace(" ", "_")
    targeted = set(transformer.subset)

    for col, in_name in input_names.items():
        if col not in targeted and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    if not transformer.drop_columns:
        for col in transformer.subset:
            if col in input_names and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))

    idx_name = f"__split_extract_idx_{transformer.n}"
    initializers.append(
        onnx.numpy_helper.from_array(np.array(transformer.n, dtype=np.int64), name=idx_name)
    )

    for col in transformer.subset:
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        y_name = f"{in_name}__strsplit_Y"
        z_name = f"{in_name}__strsplit_Z"
        out_col = f"{col}__split_{by_clean}_{transformer.n}"
        nodes.append(oh.make_node(
            "StringSplit",
            inputs=[in_name],
            outputs=[y_name, z_name],
            delimiter=transformer.by,
        ))
        nodes.append(oh.make_node(
            "Gather",
            inputs=[y_name, idx_name],
            outputs=[output_names[out_col]],
            axis=1,
        ))

    return nodes, initializers


# ── Length (not supported) ────────────────────────────────────────────────────

@get_input_onnx_type.register(Length)
def _length_input_type(transformer: Length, col: str) -> int:
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


@to_onnx_nodes.register(Length)
def _length_to_onnx_nodes(
    transformer: Length,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    raise OnnxNotSupportedError(
        "Length cannot be exported to ONNX: the standard ONNX opset has no string-length operator."
    )
