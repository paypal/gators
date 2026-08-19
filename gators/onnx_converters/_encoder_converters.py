"""ONNX converters for gators.encoders transformers.

Importing this module registers all supported converters.

Supported
---------
_BaseEncoder subclasses (OrdinalEncoder, TargetEncoder, WOEEncoder, CountEncoder, …):
    ai.onnx.ml.LabelEncoder — string→float or float→float (boolean) lookup from mapping_.
OneHotEncoder:
    One ai.onnx.ml.LabelEncoder node per category: cat→1.0, others→0.0.
BinaryEncoder:
    One ai.onnx.ml.LabelEncoder per bit column mapping the string category to its bit value (0.0/1.0).
RareCategoryEncoder:
    ai.onnx.ml.LabelEncoder (string→int64) indicator + Cast to bool + Constant default
    string + Where node to replace rare categories and pass through all others unchanged.
HashEncoder:
    ai.onnx.ml.LabelEncoder using the vocabulary→bucket mapping computed at fit time.
    Unknown values at inference time fall back to bucket 0.
"""
from __future__ import annotations

import polars as pl

from ._converters import get_input_onnx_type, get_output_columns, get_output_onnx_type, to_onnx_nodes
from ..encoders._base_encoder import _BaseEncoder
from ..encoders.binary_encoder import BinaryEncoder
from ..encoders.hash_encoder import HashEncoder
from ..encoders.onehot_encoder import OneHotEncoder, _norm_col
from ..encoders.rare_category_encoder import RareCategoryEncoder

try:
    import onnx
    import onnx.helper as oh
    from onnx import TensorProto
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'onnx' package is required for ONNX export. "
        "Install it with: pip install gators[onnx]"
    ) from exc


# ── _BaseEncoder: input/output type hooks ─────────────────────────────────────

@get_output_columns.register(_BaseEncoder)
def _encoder_output_columns(transformer: _BaseEncoder, input_columns: list[str]) -> list[str]:
    if transformer.inplace or not transformer.column_mapping_:
        return list(input_columns)
    new_cols = list(transformer.column_mapping_.values())
    if transformer.drop_columns:
        dropped = set(transformer.column_mapping_)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols

@get_input_onnx_type.register(_BaseEncoder)
def _encoder_input_onnx_type(transformer: _BaseEncoder, col: str) -> int:
    if col not in transformer.mapping_:
        return TensorProto.FLOAT
    first_key = next(iter(transformer.mapping_[col]), None)
    if isinstance(first_key, str):
        # 'true'/'false' keys from a Boolean Polars column → FLOAT (0.0/1.0 in ONNX).
        # But if _input_dtypes says the column is actually String, keep STRING.
        if first_key.lower() in ('true', 'false'):
            col_dtype = (transformer._input_dtypes or {}).get(col)
            if col_dtype != pl.String:
                return TensorProto.FLOAT
        return TensorProto.STRING
    return TensorProto.FLOAT


@get_output_onnx_type.register(_BaseEncoder)
def _encoder_output_onnx_type(transformer: _BaseEncoder, col: str) -> int:
    # New renamed column (e.g. is_vaulted__encode_woe): always FLOAT
    if col in (transformer.column_mapping_ or {}).values():
        return TensorProto.FLOAT
    # Inplace-encoded column (same name, overwritten as FLOAT)
    if col in transformer.mapping_ and transformer.inplace:
        return TensorProto.FLOAT
    # Original column kept as pass-through (inplace=False, drop_columns=False): preserve input type
    return get_input_onnx_type(transformer, col)


# ── OneHotEncoder: schema hooks ───────────────────────────────────────────────

@get_output_columns.register(OneHotEncoder)
def _ohe_output_columns(transformer: OneHotEncoder, input_columns: list[str]) -> list[str]:
    subset = set(transformer.subset or [])
    result = [col for col in input_columns if col not in subset]
    for col, cats in (transformer.column_categories or {}).items():
        for cat in cats:
            result.append(_norm_col.sub('__', f"{col}__{cat}"))
    return result


@get_input_onnx_type.register(OneHotEncoder)
def _ohe_input_onnx_type(transformer: OneHotEncoder, col: str) -> int:
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


# ── _BaseEncoder: to_onnx_nodes ───────────────────────────────────────────────

@to_onnx_nodes.register(_BaseEncoder)
def _base_encoder_to_onnx_nodes(
    transformer: _BaseEncoder,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """ai.onnx.ml.LabelEncoder lookup from mapping_.

    Boolean-keyed columns use keys_floats (False=0.0, True=1.0).
    Numeric-keyed columns (float/int, non-bool) use keys_floats with actual values.
    String-keyed columns use keys_strings.
    Unknown categories → 0.0 (matching replace_strict default=0.0).
    """
    nodes: list[onnx.NodeProto] = []
    targeted = set(transformer.mapping_)

    for col, in_name in input_names.items():
        if col not in targeted and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, mapping in transformer.mapping_.items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        # For inplace=False the output tensor uses the renamed column; for inplace=True it uses the same name.
        if transformer.inplace:
            actual_out = col
        else:
            actual_out = transformer.column_mapping_.get(col, col)
            # If drop_columns=False the source col also passes through unchanged
            if not transformer.drop_columns and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
        out_name = output_names.get(actual_out, actual_out)
        first_key = next(iter(mapping), None)
        col_dtype = (transformer._input_dtypes or {}).get(col)
        # is_bool_str: keys 'true'/'false' from a Boolean column → FLOAT input; not for String columns
        is_bool_str = isinstance(first_key, str) and first_key.lower() in ('true', 'false') and col_dtype != pl.String
        if isinstance(first_key, bool):
            nodes.append(
                oh.make_node(
                    "LabelEncoder",
                    inputs=[in_name],
                    outputs=[out_name],
                    domain="ai.onnx.ml",
                    keys_floats=[0.0, 1.0],
                    values_floats=[float(mapping.get(False, 0.0)), float(mapping.get(True, 0.0))],
                    default_float=0.0,
                )
            )
        elif is_bool_str:
            # Boolean Polars column encoded as 'false'/'true' strings; ONNX tensor is FLOAT 0.0/1.0
            nodes.append(
                oh.make_node(
                    "LabelEncoder",
                    inputs=[in_name],
                    outputs=[out_name],
                    domain="ai.onnx.ml",
                    keys_floats=[0.0, 1.0],
                    values_floats=[float(mapping.get('false', 0.0)), float(mapping.get('true', 0.0))],
                    default_float=0.0,
                )
            )
        elif isinstance(first_key, str):
            nodes.append(
                oh.make_node(
                    "LabelEncoder",
                    inputs=[in_name],
                    outputs=[out_name],
                    domain="ai.onnx.ml",
                    keys_strings=[str(k) for k in mapping],
                    values_floats=[float(v) for v in mapping.values()],
                    default_float=0.0,
                )
            )
        else:
            # numeric (float/int) keys — e.g. WOE-encoded indicator columns
            nodes.append(
                oh.make_node(
                    "LabelEncoder",
                    inputs=[in_name],
                    outputs=[out_name],
                    domain="ai.onnx.ml",
                    keys_floats=[float(k) for k in mapping],
                    values_floats=[float(v) for v in mapping.values()],
                    default_float=0.0,
                )
            )

    return nodes, []


# ── OneHotEncoder: to_onnx_nodes ──────────────────────────────────────────────

@to_onnx_nodes.register(OneHotEncoder)
def _ohe_to_onnx_nodes(
    transformer: OneHotEncoder,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """One LabelEncoder per category: keys_strings=[cat], values_floats=[1.0], default=0.0."""
    nodes: list[onnx.NodeProto] = []
    subset = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, cats in (transformer.column_categories or {}).items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        for cat in cats:
            ohe_col = _norm_col.sub('__', f"{col}__{cat}")
            out_name = output_names.get(ohe_col, ohe_col)
            nodes.append(
                oh.make_node(
                    "LabelEncoder",
                    inputs=[in_name],
                    outputs=[out_name],
                    domain="ai.onnx.ml",
                    keys_strings=[cat],
                    values_floats=[1.0],
                    default_float=0.0,
                )
            )

    return nodes, []


# ── RareCategoryEncoder: schema hooks ─────────────────────────────────────────

@get_output_columns.register(RareCategoryEncoder)
def _rare_encoder_output_columns(transformer: RareCategoryEncoder, input_columns: list[str]) -> list[str]:
    if transformer.inplace or not transformer.column_mapping_:
        return list(input_columns)
    new_cols = list(transformer.column_mapping_.values())
    if transformer.drop_columns:
        dropped = set(transformer.column_mapping_)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


@get_input_onnx_type.register(RareCategoryEncoder)
def _rare_encoder_input_onnx_type(transformer: RareCategoryEncoder, col: str) -> int:
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


@get_output_onnx_type.register(RareCategoryEncoder)
def _rare_encoder_output_onnx_type(transformer: RareCategoryEncoder, col: str) -> int:
    if transformer.inplace and col in (transformer.subset or []):
        return TensorProto.STRING
    if not transformer.inplace and col in transformer.column_mapping_.values():
        return TensorProto.STRING
    return get_input_onnx_type(transformer, col)


# ── RareCategoryEncoder: to_onnx_nodes ────────────────────────────────────────

@to_onnx_nodes.register(RareCategoryEncoder)
def _rare_encoder_to_onnx_nodes(
    transformer: RareCategoryEncoder,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """String→string replacement: rare categories → default, others pass through unchanged.

    LabelEncoder (string→int64) marks rare categories as 1, all others as 0.  Cast to bool
    then selects via Where between a broadcast default constant and the original input.
    """
    nodes: list[onnx.NodeProto] = []
    targeted = set(transformer.mapping_)

    for col, in_name in input_names.items():
        if col not in targeted and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, mapping in transformer.mapping_.items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]

        if transformer.inplace:
            actual_out = col
        else:
            actual_out = transformer.column_mapping_.get(col, col)
            if not transformer.drop_columns and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
        out_name = output_names.get(actual_out, actual_out)

        if not mapping:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[out_name]))
            continue

        indicator = f"{col}__rare_indicator"
        is_rare = f"{col}__is_rare"
        default_const = f"{col}__rare_default_const"

        # integer indicator: 1 for rare categories, 0 for everything else
        nodes.append(
            oh.make_node(
                "LabelEncoder",
                inputs=[in_name],
                outputs=[indicator],
                domain="ai.onnx.ml",
                keys_strings=[str(k) for k in mapping],
                values_int64s=[1] * len(mapping),
                default_int64=0,
            )
        )
        nodes.append(oh.make_node("Cast", inputs=[indicator], outputs=[is_rare], to=TensorProto.BOOL))
        # Scalar [1] default string; Where broadcasts it against the [N] input
        nodes.append(
            oh.make_node(
                "Constant",
                inputs=[],
                outputs=[default_const],
                value=oh.make_tensor(
                    f"{col}__rare_default_val",
                    TensorProto.STRING,
                    [1],
                    [transformer.default.encode()],
                ),
            )
        )
        # Non-rare categories pass through unchanged; rare ones get the default string
        nodes.append(oh.make_node("Where", inputs=[is_rare, default_const, in_name], outputs=[out_name]))

    return nodes, []


# ── HashEncoder ───────────────────────────────────────────────────────────────

@get_output_columns.register(HashEncoder)
def _hash_encoder_output_columns(transformer: HashEncoder, input_columns: list[str]) -> list[str]:
    if transformer.inplace or not transformer._column_mapping:
        return list(input_columns)
    new_cols = list(transformer._column_mapping.values())
    if transformer.drop_columns:
        dropped = set(transformer._column_mapping)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


@get_input_onnx_type.register(HashEncoder)
def _hash_encoder_input_type(transformer: HashEncoder, col: str) -> int:
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


@get_output_onnx_type.register(HashEncoder)
def _hash_encoder_output_type(transformer: HashEncoder, col: str) -> int:
    if col in (transformer._column_mapping or {}).values() or (
        transformer.inplace and col in (transformer.subset or [])
    ):
        return TensorProto.FLOAT
    return get_input_onnx_type(transformer, col)


@to_onnx_nodes.register(HashEncoder)
def _hash_encoder_to_onnx_nodes(
    transformer: HashEncoder,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """LabelEncoder from vocabulary→bucket mapping stored at fit time.

    Known values map to their exact Polars hash bucket.  Values unseen during
    fit fall back to bucket 0.0.
    """
    nodes: list[onnx.NodeProto] = []
    subset = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        mapping = transformer._hash_mapping_.get(col, {})

        if transformer.inplace:
            out_name = output_names.get(col, col)
        else:
            actual_out = transformer._column_mapping.get(col, col)
            if not transformer.drop_columns and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
            out_name = output_names.get(actual_out, actual_out)

        nodes.append(oh.make_node(
            "LabelEncoder", domain="ai.onnx.ml",
            inputs=[in_name], outputs=[out_name],
            keys_strings=list(mapping.keys()),
            values_floats=[float(v) for v in mapping.values()],
            default_float=0.0,
        ))

    return nodes, []


# ── BinaryEncoder ─────────────────────────────────────────────────────────────

@get_output_columns.register(BinaryEncoder)
def _binary_encoder_output_columns(transformer: BinaryEncoder, input_columns: list[str]) -> list[str]:
    subset = set(transformer.subset or [])
    result = [c for c in input_columns if c not in subset] if transformer.drop_columns else list(input_columns)
    for col in (transformer.subset or []):
        n_bits = transformer.n_bits_.get(col, 0)
        result += [f"{col}__binary_enc_{i}" for i in range(n_bits)]
    return result


@get_input_onnx_type.register(BinaryEncoder)
def _binary_encoder_input_onnx_type(transformer: BinaryEncoder, col: str) -> int:
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


@get_output_onnx_type.register(BinaryEncoder)
def _binary_encoder_output_onnx_type(transformer: BinaryEncoder, col: str) -> int:
    return TensorProto.FLOAT


@to_onnx_nodes.register(BinaryEncoder)
def _binary_encoder_to_onnx_nodes(
    transformer: BinaryEncoder,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """One LabelEncoder per bit column: maps string categories to 0.0/1.0 bit values."""
    nodes: list[onnx.NodeProto] = []
    subset = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]

        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

        n_bits = transformer.n_bits_.get(col, 0)
        for bit_idx in range(n_bits):
            bit_col = f"{col}__binary_enc_{bit_idx}"
            mapping = transformer.mapping_.get(bit_col, {})
            out_name = output_names.get(bit_col, bit_col)
            nodes.append(oh.make_node(
                "LabelEncoder", domain="ai.onnx.ml",
                inputs=[in_name], outputs=[out_name],
                keys_strings=[str(k) for k in mapping],
                values_floats=[float(v) for v in mapping.values()],
                default_float=0.0,
            ))

    return nodes, []
