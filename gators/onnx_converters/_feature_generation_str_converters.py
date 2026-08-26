"""ONNX converters for gators.feature_generation_str transformers.

Importing this module registers all supported converters.

Supported
---------
Lower        : StringNormalizer(case_change_action="LOWER")
Upper        : StringNormalizer(case_change_action="UPPER")
Split        : StringSplit (opset 20) + Gather per split index
SplitExtract : StringSplit (opset 20) + Gather at index n
CombineFeatures : chained StringConcat (opset 20) with a constant separator
InteractionFeatures : chained StringConcat (opset 20) with '_' separator for every combination
TfidfFeatures   : StringNormalizer → StringSplit → TfIdfVectorizer(TF) → length-normalise → IDF-weight → Gather per token
Startswith      : StringSplit(delimiter=prefix) → Gather first part → Equal('') → And(not_empty)
Endswith        : StringSplit(delimiter=suffix) → GatherND last part → Equal('') → And(not_empty)
Contains        : StringSplit(delimiter=substring) → Greater(_sz, 1)

Not supported (no ONNX string-length operator)
-----------------------------------------------
Length

Not supported (no ONNX-representable equivalent)
--------------------------------------------------
CharacterStatistics, NGram, Occurrences, PatternDetector, RegexExtractFeatures
StringSimilarity : fuzzy string distance (Levenshtein/Jaro-Winkler) has no ONNX operator
WordStatistics   : variable-length whitespace tokenization + list aggregates
                   (n_unique_words in particular) have no practical ONNX equivalent

Notes
-----
Split / SplitExtract use the ONNX ``StringSplit`` operator (opset 20), which
requires ORT ≥ 1.17. ``StringSplit`` performs a full split (no maxsplit limit),
so its output width is the max number of parts found in the current batch;
the converter pads this output (with ``""``) to guarantee every requested
index is always in bounds, regardless of batch composition.

For ``SplitExtract``: strings with fewer than ``n + 1`` parts produce an empty
string ``""`` in ONNX (the tensor is padded with ``""``), whereas the Polars
transformer returns ``null``.
"""
from __future__ import annotations

import numpy as np
import onnx.numpy_helper
import polars as pl

from ..feature_generation_str.combine_features import CombineFeatures
from ..feature_generation_str.contains import Contains
from ..feature_generation_str.endswith import Endswith
from ..feature_generation_str.extract_substring import ExtractSubstring
from ..feature_generation_str.interaction_features import InteractionFeatures
from ..feature_generation_str.length import Length
from ..feature_generation_str.lower import Lower
from ..feature_generation_str.split import Split
from ..feature_generation_str.split_extract import SplitExtract
from ..feature_generation_str.startswith import Startswith
from ..feature_generation_str.tfidf_features import TfidfFeatures
from ..feature_generation_str.upper import Upper
from ..transformer._base_transformer import _BaseTransformer
from ._converters import (
    get_input_onnx_type,
    get_output_columns,
    get_output_onnx_type,
    resolve_declared_output_dtype,
    to_onnx_nodes,
)
from ._exceptions import OnnxNotSupportedError

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

def _pad_split_output(y_name: str, min_width: int, prefix: str, nodes: list, initializers: list) -> str:
    """Ensure StringSplit's ``Y`` output (axis=1) has at least ``min_width`` columns.

    ``StringSplit``'s second dimension is the max split count found in the CURRENT
    inference batch, which can be smaller than ``min_width`` if no row in that batch
    happens to have enough parts - Gathering at those indices would otherwise be an
    out-of-bounds access that only fails at runtime, depending on batch composition.
    Always appends a ``min_width``-wide block of empty strings (regardless of Y's
    actual width) so any index < ``min_width`` is guaranteed to be in bounds. ONNX's
    ``Pad`` op has no onnxruntime CPU kernel for string tensors, so padding is done
    via Concat + Expand instead.
    """
    shape_name = f"{prefix}__shape"
    nodes.append(oh.make_node("Shape", inputs=[y_name], outputs=[shape_name]))
    rows_idx_init = f"{prefix}__rows_idx"
    initializers.append(onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name=rows_idx_init))
    rows_name = f"{prefix}__rows"
    nodes.append(oh.make_node("Gather", inputs=[shape_name, rows_idx_init], outputs=[rows_name], axis=0))

    min_width_init = f"{prefix}__min_width"
    initializers.append(onnx.numpy_helper.from_array(np.array([min_width], dtype=np.int64), name=min_width_init))

    pad_shape_name = f"{prefix}__pad_shape"
    nodes.append(oh.make_node("Concat", inputs=[rows_name, min_width_init], outputs=[pad_shape_name], axis=0))

    empty_init = f"{prefix}__empty"
    # onnx's make_tensor stub types vals as int|float only; STRING tensors accept bytes at runtime.
    initializers.append(oh.make_tensor(empty_init, TensorProto.STRING, [1], [b""]))  # type: ignore[list-item]

    pad_block_name = f"{prefix}__pad_block"
    nodes.append(oh.make_node("Expand", inputs=[empty_init, pad_shape_name], outputs=[pad_block_name]))

    padded_name = f"{prefix}__padded"
    nodes.append(oh.make_node("Concat", inputs=[y_name, pad_block_name], outputs=[padded_name], axis=1))
    return padded_name


def _str_normalizer_output_columns(transformer: _BaseTransformer, input_columns: list[str]) -> list[str]:
    if transformer.inplace or not transformer._column_mapping:
        return list(input_columns)
    new_cols = [name for names in transformer._column_mapping.values() for name in names]
    if transformer.drop_columns:
        dropped = set(transformer._column_mapping)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


def _str_normalizer_input_type(transformer: _BaseTransformer, col: str) -> int:
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


def _str_normalizer_output_type(transformer: _BaseTransformer, col: str) -> int:
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    if transformer.inplace:
        output_string_cols = set(transformer.subset or [])
    else:
        output_string_cols = {name for names in (transformer._column_mapping or {}).values() for name in names}
    if col in output_string_cols:
        return TensorProto.STRING
    # Pass-through column: use the input type (STRING for string cols, FLOAT for numeric).
    return _str_normalizer_input_type(transformer, col)


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
        for orig, [new] in transformer._column_mapping.items():
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
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    return _split_input_type(transformer, col)


@to_onnx_nodes.register(Split)
def _split_to_onnx_nodes(
    transformer: Split,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringSplit (opset 20) → pad to max_splits → Gather for each split index.

    Uses a full split (no maxsplit) so elements at every requested index are
    exact. ``StringSplit``'s output width is the max split count found in the
    CURRENT batch, which can be smaller than ``max_splits``; the output is
    padded (with "") to guarantee every requested index is in bounds regardless
    of batch composition.
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
        padded_name = _pad_split_output(y_name, transformer.max_splits, f"{in_name}__pad", nodes, initializers)
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
                inputs=[padded_name, idx_name],
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
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    return _split_extract_input_type(transformer, col)


@to_onnx_nodes.register(SplitExtract)
def _split_extract_to_onnx_nodes(
    transformer: SplitExtract,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringSplit (opset 20) → pad to n+1 → Gather at index n.

    Strings with fewer than n + 1 parts yield ``""`` (ONNX tensor padding)
    rather than ``null`` as in the Polars transformer. ``StringSplit``'s output
    width is the max split count found in the CURRENT batch, which can be
    smaller than ``n + 1``; the output is padded (with "") to guarantee index
    ``n`` is always in bounds regardless of batch composition.
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
        padded_name = _pad_split_output(y_name, transformer.n + 1, f"{in_name}__pad", nodes, initializers)
        nodes.append(oh.make_node(
            "Gather",
            inputs=[padded_name, idx_name],
            outputs=[output_names[out_col]],
            axis=1,
        ))

    return nodes, initializers


# ── CombineFeatures ───────────────────────────────────────────────────────────

@get_output_columns.register(CombineFeatures)
def _cf_output_columns(transformer: CombineFeatures, input_columns: list[str]) -> list[str]:
    all_group_cols = {col for group in transformer.column_groups for col in group}
    base = [c for c in input_columns if c not in all_group_cols] if transformer.drop_columns else list(input_columns)
    return base + list(transformer.new_column_names or [])


@get_input_onnx_type.register(CombineFeatures)
def _cf_input_type(transformer: CombineFeatures, col: str) -> int:
    all_group_cols = {c for group in transformer.column_groups for c in group}
    if col in all_group_cols:
        return TensorProto.STRING
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    from ._converters import _POLARS_TO_ONNX
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(CombineFeatures)
def _cf_output_type(transformer: CombineFeatures, col: str) -> int:
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    return _cf_input_type(transformer, col)


@to_onnx_nodes.register(CombineFeatures)
def _cf_to_onnx_nodes(
    transformer: CombineFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Chained StringConcat (opset 20) nodes with a constant separator tensor.

    For a group ['a', 'b', 'c'] with sep='_': StringConcat(a, '_') → StringConcat(_, b) →
    StringConcat(_, '_') → StringConcat(_, c).

    Note: null values arrive as '' (empty string) from the ONNX runtime, whereas
    the Polars transformer converts nulls to the string "null". Test with non-null
    data to avoid this discrepancy.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    all_group_cols = {col for group in transformer.column_groups for col in group}
    generated_cols = set(transformer.new_column_names or [])

    # Passthrough non-group, non-generated columns
    for col, in_name in input_names.items():
        if col not in all_group_cols and col not in generated_cols and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    # Passthrough group columns when drop_columns=False
    if not transformer.drop_columns:
        seen = set()
        for group in transformer.column_groups:
            for col in group:
                if col not in seen and col in input_names and col in output_names:
                    nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))
                    seen.add(col)

    # Constant separator tensor (shape [1] so StringConcat can broadcast)
    sep_name = f"__cf_sep_{id(transformer)}"
    initializers.append(
        onnx.numpy_helper.from_array(
            np.array([transformer.separator], dtype=object), name=sep_name
        )
    )

    for i, group in enumerate(transformer.column_groups):
        assert transformer.new_column_names is not None
        new_col = transformer.new_column_names[i]
        out_name = output_names.get(new_col, new_col)

        # Start with the first column
        current = input_names[group[0]]

        for j, col in enumerate(group[1:], start=1):
            col_in = input_names[col]
            with_sep = f"__cf_g{i}_s{j}_sep"
            nodes.append(oh.make_node("StringConcat", inputs=[current, sep_name], outputs=[with_sep]))
            is_last = j == len(group) - 1
            result = out_name if is_last else f"__cf_g{i}_s{j}_cat"
            nodes.append(oh.make_node("StringConcat", inputs=[with_sep, col_in], outputs=[result]))
            current = result

        # Single-column group (edge case): just copy
        if len(group) == 1:
            nodes.append(oh.make_node("Identity", inputs=[current], outputs=[out_name]))

    return nodes, initializers


# ── ExtractSubstring ──────────────────────────────────────────────────────────

@get_output_columns.register(ExtractSubstring)
def _extract_substring_output_cols(transformer: ExtractSubstring, input_columns: list[str]) -> list[str]:
    end_str = "None" if transformer.end is None else str(transformer.end)
    new_cols = [
        f"{col}__start{transformer.start}_end{end_str}"
        for col in (transformer.subset or [])
        if col in set(input_columns)
    ]
    return list(input_columns) + new_cols


@get_input_onnx_type.register(ExtractSubstring)
def _extract_substring_input_type(transformer: ExtractSubstring, col: str) -> int:
    end_str = "None" if transformer.end is None else str(transformer.end)
    new_cols = {f"{c}__start{transformer.start}_end{end_str}" for c in (transformer.subset or [])}
    # Source columns and their extracted substrings are both STRING
    if col in set(transformer.subset or []) or col in new_cols:
        return TensorProto.STRING
    return _to_onnx_type(getattr(transformer, "_input_dtypes", {}).get(col))


@get_output_onnx_type.register(ExtractSubstring)
def _extract_substring_output_type(transformer: ExtractSubstring, col: str) -> int:
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    return _extract_substring_input_type(transformer, col)


@to_onnx_nodes.register(ExtractSubstring)
def _extract_substring_to_onnx_nodes(
    transformer: ExtractSubstring,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """No ONNX string-slice op — coerce emits Identity pass-throughs."""
    if errors == "raise":
        raise OnnxNotSupportedError(
            "ExtractSubstring has no ONNX implementation (no string-slice op). "
            "Use errors='coerce'."
        )
    end_str = "None" if transformer.end is None else str(transformer.end)
    nodes: list[onnx.NodeProto] = []
    for col, in_name in input_names.items():
        if col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        new_col = f"{col}__start{transformer.start}_end{end_str}"
        if new_col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[new_col]]))
    return nodes, []


# ── InteractionFeatures ───────────────────────────────────────────────────────

def _if_generated_cols(transformer: InteractionFeatures) -> list[str]:
    from itertools import combinations as _comb
    cols = []
    for i in range(2, transformer.degree + 1):
        for combo in _comb(transformer.subset or [], i):
            cols.append("__".join(combo))
    return cols


@get_output_columns.register(InteractionFeatures)
def _if_output_columns(transformer: InteractionFeatures, input_columns: list[str]) -> list[str]:
    return list(input_columns) + _if_generated_cols(transformer)


@get_input_onnx_type.register(InteractionFeatures)
def _if_input_type(transformer: InteractionFeatures, col: str) -> int:
    from ._converters import _POLARS_TO_ONNX
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    if dtype is not None:
        return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)
    # Unknown dtype (not yet resolved): assume STRING for subset columns (the common
    # case for this transformer), else default to FLOAT.
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


@get_output_onnx_type.register(InteractionFeatures)
def _if_output_type(transformer: InteractionFeatures, col: str) -> int:
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    return _if_input_type(transformer, col)


@to_onnx_nodes.register(InteractionFeatures)
def _if_to_onnx_nodes(
    transformer: InteractionFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Chained StringConcat (opset 20) with a hardcoded '_' separator for each combination.

    Mirrors the Polars logic: col_a + '_' + col_b (+ '_' + col_c …) for every
    combination of ``subset`` columns at degrees 2 … ``degree``.

    Raises OnnxNotSupportedError for non-string subset columns: Polars formats numeric
    values into strings using its own float/int Display logic (e.g. "1.0"), which no ONNX
    Cast-to-string operator reproduces exactly - silently concatenating a mismatched
    format would desync category strings from what OneHotEncoder/CountEncoder learned
    at fit() time, so this is flagged rather than silently producing wrong values.
    """
    from itertools import combinations as _comb

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    generated_set = set(_if_generated_cols(transformer))

    non_string_cols = [
        col for col in (transformer.subset or [])
        if get_input_onnx_type(transformer, col) != TensorProto.STRING
    ]
    if non_string_cols:
        raise OnnxNotSupportedError(
            f"InteractionFeatures cannot be exported to ONNX for non-string subset columns "
            f"{non_string_cols}: ONNX has no operator that reproduces Polars' numeric-to-string "
            "formatting exactly, so string-concatenating a non-string column would silently "
            "diverge from the real Polars output."
        )

    for col, in_name in input_names.items():
        if col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    sep_name = f"{next(iter(input_names.values()), 'if')}__if_sep"
    initializers.append(
        onnx.numpy_helper.from_array(np.array(["_"], dtype=object), name=sep_name)
    )

    for i in range(2, transformer.degree + 1):
        for combo in _comb(transformer.subset or [], i):
            out_col = "__".join(combo)
            out_name = output_names.get(out_col, out_col)
            current = input_names[combo[0]]
            for j, col in enumerate(combo[1:], start=1):
                with_sep = f"__if_{'_'.join(combo)}_s{j}_sep"
                is_last = j == len(combo) - 1
                cat_out = out_name if is_last else f"__if_{'_'.join(combo)}_s{j}_cat"
                nodes.append(oh.make_node("StringConcat", inputs=[current, sep_name], outputs=[with_sep]))
                nodes.append(oh.make_node("StringConcat", inputs=[with_sep, input_names[col]], outputs=[cat_out]))
                current = cat_out

    return nodes, initializers


# ── TfidfFeatures ─────────────────────────────────────────────────────────────

def _tfidf_generated_cols(transformer: TfidfFeatures) -> list[str]:
    import re as _re
    cols = []
    for col in (transformer.subset or []):
        for tok in transformer._vocabulary.get(col, []):
            safe = _re.sub(r"\W+", "_", tok).strip("_")
            cols.append(f"{col}__tfidf_{safe}")
    return cols


@get_output_columns.register(TfidfFeatures)
def _tfidf_output_columns(transformer: TfidfFeatures, input_columns: list[str]) -> list[str]:
    generated = _tfidf_generated_cols(transformer)
    if transformer.drop_columns:
        dropped = set(transformer.subset or [])
        return [c for c in input_columns if c not in dropped] + generated
    return list(input_columns) + generated


@get_input_onnx_type.register(TfidfFeatures)
def _tfidf_input_type(transformer: TfidfFeatures, col: str) -> int:
    if col in (transformer.subset or []):
        return TensorProto.STRING
    from ._converters import _POLARS_TO_ONNX
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(TfidfFeatures)
def _tfidf_output_type(transformer: TfidfFeatures, col: str) -> int:
    # Not using _output_dtypes here: TF-IDF weights are hardcoded to FLOAT regardless
    # of float_datatype, while _output_dtypes declares Float64.
    if col in _tfidf_generated_cols(transformer):
        return TensorProto.FLOAT
    return _tfidf_input_type(transformer, col)


@to_onnx_nodes.register(TfidfFeatures)
def _tfidf_to_onnx_nodes(
    transformer: TfidfFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """TF-IDF via standard-domain TfIdfVectorizer (opset 9).

    Pipeline per source column:
      StringNormalizer → StringSplit → TfIdfVectorizer(mode=TF, weights=1)
      → length-normalise → IDF-multiply → Gather each token → Squeeze

    Shared constants (empty string, axis 1, zero) are emitted once per column
    using unique name prefixes.
    """
    import re as _re

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    subset_set = set(transformer.subset or [])
    generated_set = set(_tfidf_generated_cols(transformer))

    for col, in_name in input_names.items():
        if col not in subset_set and col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    if not transformer.drop_columns:
        for col in (transformer.subset or []):
            if col in input_names and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names or col not in transformer._vocabulary:  # pragma: no cover
            continue
        vocab = transformer._vocabulary[col]
        if not vocab:  # pragma: no cover
            continue

        in_name = input_names[col]
        p = f"__tfidf_{col}"

        # Shared constants per column (unique prefix avoids name collisions)
        empty_c  = f"{p}__empty"
        ax1_c    = f"{p}__ax1"
        zero_i_c = f"{p}__zero_i"
        zero_f_c = f"{p}__zero_f"
        initializers += [
            onnx.numpy_helper.from_array(np.array([""],  dtype=object),     name=empty_c),
            onnx.numpy_helper.from_array(np.array([1],   dtype=np.int64),   name=ax1_c),
            onnx.numpy_helper.from_array(np.array([0],   dtype=np.int64),   name=zero_i_c),
            onnx.numpy_helper.from_array(np.array([0.0], dtype=np.float32), name=zero_f_c),
        ]

        # Step 1: optional lowercase (StringNormalizer, standard domain)
        normed = f"{p}__normed"
        if transformer.lowercase:
            nodes.append(oh.make_node("StringNormalizer", inputs=[in_name], outputs=[normed],
                case_change_action="LOWER"))
        else:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[normed]))

        # Step 2: StringSplit → [N, max_parts] padded with ""
        tokens = f"{p}__tokens"
        nodes.append(oh.make_node("StringSplit", inputs=[normed],
            outputs=[tokens, f"{p}__sz"], delimiter=transformer.separator))

        # Step 3: TfIdfVectorizer(mode=TF) → raw token counts [N, V]
        idf_vals = [transformer._idf[col][tok] for tok in vocab]
        counts_raw = f"{p}__counts_raw"
        nodes.append(oh.make_node("TfIdfVectorizer", inputs=[tokens], outputs=[counts_raw],
            min_gram_length=1, max_gram_length=1, mode="TF", max_skip_count=0,
            ngram_counts=[0], ngram_indexes=list(range(len(vocab))),
            pool_strings=vocab, weights=[1.0] * len(vocab)))

        # Step 4: document length = number of non-empty tokens per row
        len_i = f"{p}__len_i"
        len_f = f"{p}__len_f"
        len_f2d = f"{p}__len_f2d"
        is_empty_doc = f"{p}__is_empty_doc"
        empty_2d = f"{p}__empty_2d"
        nodes += [
            oh.make_node("Equal",     inputs=[tokens, empty_c],   outputs=[f"{p}__is_empty_tok"]),
            oh.make_node("Not",       inputs=[f"{p}__is_empty_tok"], outputs=[f"{p}__is_tok"]),
            oh.make_node("Cast",      inputs=[f"{p}__is_tok"],    outputs=[f"{p}__is_tok_i"], to=TensorProto.INT64),
            oh.make_node("ReduceSum", inputs=[f"{p}__is_tok_i", ax1_c], outputs=[len_i], keepdims=0),
            oh.make_node("Cast",      inputs=[len_i],             outputs=[len_f], to=TensorProto.FLOAT),
            oh.make_node("Equal",     inputs=[len_i, zero_i_c],   outputs=[is_empty_doc]),   # [N]
            oh.make_node("Unsqueeze", inputs=[len_f, ax1_c],      outputs=[len_f2d]),          # [N, 1]
            oh.make_node("Unsqueeze", inputs=[is_empty_doc, ax1_c], outputs=[empty_2d]),      # [N, 1]
        ]

        # Step 5: TF = counts / length (zero for empty docs)
        tf = f"{p}__tf"
        nodes += [
            oh.make_node("Cast",  inputs=[counts_raw], outputs=[f"{p}__counts_f"], to=TensorProto.FLOAT),
            oh.make_node("Div",   inputs=[f"{p}__counts_f", len_f2d], outputs=[f"{p}__tf_raw"]),
            oh.make_node("Where", inputs=[empty_2d, zero_f_c, f"{p}__tf_raw"], outputs=[tf]),
        ]

        # Step 6: multiply by IDF weights → [N, V]
        idf_name = f"{p}__idf"
        tfidf = f"{p}__tfidf"
        initializers.append(onnx.numpy_helper.from_array(
            np.array(idf_vals, dtype=np.float32).reshape(1, -1), name=idf_name))
        nodes.append(oh.make_node("Mul", inputs=[tf, idf_name], outputs=[tfidf]))

        # Step 7: Gather + Squeeze to extract each token's column [N]
        for i, tok in enumerate(vocab):
            safe = _re.sub(r"\W+", "_", tok).strip("_")
            out_col = f"{col}__tfidf_{safe}"
            out_name = output_names.get(out_col, out_col)
            idx_c = f"{p}__idx{i}"
            gathered = f"{p}__g{i}"
            initializers.append(onnx.numpy_helper.from_array(np.array([i], dtype=np.int64), name=idx_c))
            nodes += [
                oh.make_node("Gather", inputs=[tfidf, idx_c], outputs=[gathered], axis=1),
                oh.make_node("Squeeze", inputs=[gathered, ax1_c], outputs=[out_name]),
            ]

    return nodes, initializers


# ── Contains ──────────────────────────────────────────────────────────────────

@get_output_columns.register(Contains)
def _co_output_columns(transformer: Contains, input_columns: list[str]) -> list[str]:
    generated = [f"{col}__contains_{sub}" for col, subs in transformer.contains_dict.items() for sub in subs]
    return list(input_columns) + generated


@get_input_onnx_type.register(Contains)
def _co_input_type(transformer: Contains, col: str) -> int:
    if col in transformer.contains_dict:
        return TensorProto.STRING
    from ._converters import _POLARS_TO_ONNX
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(Contains)
def _co_output_type(transformer: Contains, col: str) -> int:
    # Not using _output_dtypes here: hardcoded to FLOAT (matching Polars' .cast(pl.Float64)
    # call) regardless of float_datatype - same convention as IsNull's null indicators.
    generated = {f"{c}__contains_{sub}" for c, subs in transformer.contains_dict.items() for sub in subs}
    if col in generated:
        return TensorProto.FLOAT
    return _co_input_type(transformer, col)


@to_onnx_nodes.register(Contains)
def _co_to_onnx_nodes(
    transformer: Contains,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringSplit(delimiter=substring) → Greater(_sz, 1).

    If the substring is present, splitting by it produces > 1 part (_sz > 1).
    Empty/null strings (arriving as '' from ORT) correctly produce False.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    generated_set = {f"{col}__contains_{sub}" for col, subs in transformer.contains_dict.items() for sub in subs}

    for col, in_name in input_names.items():
        if col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, substrings in transformer.contains_dict.items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]

        for substring in substrings:
            p = f"__co_{col}_{substring}"
            sz_out  = f"{p}__sz"
            one_c   = f"{p}__one"
            raw_out = f"{p}__raw"
            out_name = output_names.get(f"{col}__contains_{substring}", f"{col}__contains_{substring}")
            initializers.append(
                onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name=one_c)
            )
            nodes += [
                oh.make_node("StringSplit", inputs=[in_name], outputs=[f"{p}__split", sz_out], delimiter=substring),
                oh.make_node("Greater", inputs=[sz_out, one_c], outputs=[raw_out]),
                oh.make_node("Cast", inputs=[raw_out], outputs=[out_name], to=TensorProto.FLOAT),
            ]

    return nodes, initializers


# ── Startswith / Endswith ─────────────────────────────────────────────────────

def _sw_generated_cols(transformer: Startswith) -> list[str]:
    return [f"{col}__startswith_{sub}" for col, subs in transformer.startswith_dict.items() for sub in subs]

def _ew_generated_cols(transformer: Endswith) -> list[str]:
    return [f"{col}__endswith_{sub}" for col, subs in transformer.endswith_dict.items() for sub in subs]


def _sw_ew_output_columns(generated: list[str], input_columns: list[str]) -> list[str]:
    return list(input_columns) + generated


def _sw_ew_input_type(source_cols: set[str], transformer, col: str) -> int:
    if col in source_cols:
        return TensorProto.STRING
    from ._converters import _POLARS_TO_ONNX
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


# ── Startswith ──

@get_output_columns.register(Startswith)
def _sw_output_columns(transformer: Startswith, input_columns: list[str]) -> list[str]:
    return _sw_ew_output_columns(_sw_generated_cols(transformer), input_columns)

@get_input_onnx_type.register(Startswith)
def _sw_input_type(transformer: Startswith, col: str) -> int:
    return _sw_ew_input_type(set(transformer.startswith_dict), transformer, col)

@get_output_onnx_type.register(Startswith)
def _sw_output_type(transformer: Startswith, col: str) -> int:
    # Not using _output_dtypes here: hardcoded to FLOAT (matching Polars' .cast(pl.Float64)
    # call) regardless of float_datatype - same convention as IsNull's null indicators.
    if col in _sw_generated_cols(transformer):
        return TensorProto.FLOAT
    return _sw_input_type(transformer, col)

@to_onnx_nodes.register(Startswith)
def _sw_to_onnx_nodes(
    transformer: Startswith,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringSplit(delimiter=prefix) → Gather first part → Equal('') → And(not_empty).

    Algorithm: split the text by the prefix string. If the text starts with the
    prefix, the first resulting part is '' (empty). Guard against empty/null
    inputs (which arrive as '' in ORT) by requiring the original string ≠ ''.

    Note: null values in the source column arrive as '' from ORT and produce
    False — matching common inference semantics (vs. Polars which propagates null).
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    generated_set = set(_sw_generated_cols(transformer))

    for col, in_name in input_names.items():
        if col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, prefixes in transformer.startswith_dict.items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]

        # Shared per-column constants
        p = f"__sw_{col}"
        empty_c = f"{p}__empty"
        idx0_c  = f"{p}__idx0"
        ax1_c   = f"{p}__ax1"
        initializers += [
            onnx.numpy_helper.from_array(np.array([""],  dtype=object),   name=empty_c),
            onnx.numpy_helper.from_array(np.array([0],   dtype=np.int64), name=idx0_c),
            onnx.numpy_helper.from_array(np.array([1],   dtype=np.int64), name=ax1_c),
        ]
        # is_empty_orig: guard for null/empty strings
        is_empty_orig = f"{p}__is_empty_orig"
        not_empty     = f"{p}__not_empty"
        nodes += [
            oh.make_node("Equal", inputs=[in_name, empty_c], outputs=[is_empty_orig]),
            oh.make_node("Not",   inputs=[is_empty_orig],    outputs=[not_empty]),
        ]

        for prefix in prefixes:
            pp = f"{p}_{prefix}"
            split_out = f"{pp}__split"
            sz_out    = f"{pp}__sz"
            first_2d  = f"{pp}__first2d"
            first     = f"{pp}__first"
            sw_raw    = f"{pp}__sw_raw"
            raw_out   = f"{pp}__raw"
            out_name = output_names.get(f"{col}__startswith_{prefix}", f"{col}__startswith_{prefix}")
            nodes += [
                oh.make_node("StringSplit", inputs=[in_name],   outputs=[split_out, sz_out], delimiter=prefix),
                oh.make_node("Gather",      inputs=[split_out, idx0_c], outputs=[first_2d], axis=1),
                oh.make_node("Squeeze",     inputs=[first_2d, ax1_c],   outputs=[first]),
                oh.make_node("Equal",       inputs=[first, empty_c],    outputs=[sw_raw]),
                oh.make_node("And",         inputs=[sw_raw, not_empty], outputs=[raw_out]),
                oh.make_node("Cast",        inputs=[raw_out], outputs=[out_name], to=TensorProto.FLOAT),
            ]

    return nodes, initializers


# ── Endswith ──

@get_output_columns.register(Endswith)
def _ew_output_columns(transformer: Endswith, input_columns: list[str]) -> list[str]:
    return _sw_ew_output_columns(_ew_generated_cols(transformer), input_columns)

@get_input_onnx_type.register(Endswith)
def _ew_input_type(transformer: Endswith, col: str) -> int:
    return _sw_ew_input_type(set(transformer.endswith_dict), transformer, col)

@get_output_onnx_type.register(Endswith)
def _ew_output_type(transformer: Endswith, col: str) -> int:
    # Not using _output_dtypes here: hardcoded to FLOAT (matching Polars' .cast(pl.Float64)
    # call) regardless of float_datatype - same convention as IsNull's null indicators.
    if col in _ew_generated_cols(transformer):
        return TensorProto.FLOAT
    return _ew_input_type(transformer, col)

@to_onnx_nodes.register(Endswith)
def _ew_to_onnx_nodes(
    transformer: Endswith,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """StringSplit(delimiter=suffix) → GatherND last part → Equal('') → And(not_empty).

    Algorithm: split the text by the suffix string. If the text ends with the
    suffix, the LAST non-padded split part is '' (empty). The last part index
    per row is (_sz - 1), retrieved via GatherND with [N, 2] row+col indices
    built from Range(0, N, 1) and _sz - 1.

    Note: null values arrive as '' from ORT and produce False.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    generated_set = set(_ew_generated_cols(transformer))

    for col, in_name in input_names.items():
        if col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, suffixes in transformer.endswith_dict.items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]

        p = f"__ew_{col}"
        empty_c  = f"{p}__empty"
        zero_c   = f"{p}__zero"
        one_c    = f"{p}__one"
        ax1_c    = f"{p}__ax1"
        initializers += [
            onnx.numpy_helper.from_array(np.array([""],  dtype=object),   name=empty_c),
            onnx.numpy_helper.from_array(np.array([0],   dtype=np.int64), name=zero_c),
            onnx.numpy_helper.from_array(np.array([1],   dtype=np.int64), name=one_c),
            onnx.numpy_helper.from_array(np.array([1],   dtype=np.int64), name=ax1_c),
        ]
        is_empty_orig = f"{p}__is_empty_orig"
        not_empty     = f"{p}__not_empty"
        nodes += [
            oh.make_node("Equal", inputs=[in_name, empty_c], outputs=[is_empty_orig]),
            oh.make_node("Not",   inputs=[is_empty_orig],    outputs=[not_empty]),
        ]

        for suffix in suffixes:
            pp = f"{p}_{suffix}"
            split_out  = f"{pp}__split"
            sz_out     = f"{pp}__sz"
            col_idx    = f"{pp}__col_idx"
            shp        = f"{pp}__shp"
            N_scalar   = f"{pp}__N"
            row_idx    = f"{pp}__row_idx"
            row_2d     = f"{pp}__row_2d"
            col_2d     = f"{pp}__col_2d"
            idx_pairs  = f"{pp}__idx_pairs"
            last_part  = f"{pp}__last_part"
            ew_raw     = f"{pp}__ew_raw"
            raw_out    = f"{pp}__raw"
            out_name = output_names.get(f"{col}__endswith_{suffix}", f"{col}__endswith_{suffix}")
            nodes += [
                oh.make_node("StringSplit", inputs=[in_name],                   outputs=[split_out, sz_out], delimiter=suffix),
                oh.make_node("Sub",         inputs=[sz_out, one_c],             outputs=[col_idx]),
                oh.make_node("Shape",       inputs=[split_out],                 outputs=[shp]),
                oh.make_node("Gather",      inputs=[shp, zero_c],               outputs=[N_scalar], axis=0),
                oh.make_node("Range",       inputs=[zero_c, N_scalar, one_c],   outputs=[row_idx]),
                oh.make_node("Unsqueeze",   inputs=[row_idx, ax1_c],            outputs=[row_2d]),
                oh.make_node("Unsqueeze",   inputs=[col_idx, ax1_c],            outputs=[col_2d]),
                oh.make_node("Concat",      inputs=[row_2d, col_2d],            outputs=[idx_pairs], axis=1),
                oh.make_node("GatherND",    inputs=[split_out, idx_pairs],      outputs=[last_part]),
                oh.make_node("Equal",       inputs=[last_part, empty_c],        outputs=[ew_raw]),
                oh.make_node("And",         inputs=[ew_raw, not_empty],         outputs=[raw_out]),
                oh.make_node("Cast",        inputs=[raw_out], outputs=[out_name], to=TensorProto.FLOAT),
            ]

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
