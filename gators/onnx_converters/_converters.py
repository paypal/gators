from __future__ import annotations

from functools import singledispatch
from typing import Any

import polars as pl

from ..pipeline.pipeline import Pipeline
from ..transformer._base_transformer import _BaseTransformer
from ._exceptions import OnnxNotSupportedError

try:
    import numpy as np
    import onnx
    import onnx.helper as oh
    from onnx import TensorProto
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'onnx' package is required for ONNX export. "
        "Install it with: pip install gators[onnx]"
    ) from exc

# Float64 → DOUBLE; Float32 → FLOAT; Boolean → BOOL; integer/temporal dtypes → INT64
# (their physical/ordinal representation); all other numeric types fall back to DOUBLE.
_INTEGER_PL_DTYPES: tuple = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)
_TEMPORAL_PL_DTYPES: tuple = (pl.Datetime, pl.Date, pl.Duration, pl.Time)
_POLARS_TO_ONNX: dict[Any, int] = {
    pl.Float32: TensorProto.FLOAT,
    pl.Float64: TensorProto.DOUBLE,
    pl.Boolean: TensorProto.BOOL,
    **dict.fromkeys(_INTEGER_PL_DTYPES, TensorProto.INT64),
}

_FLOAT_ONNX_TYPES: frozenset = frozenset({TensorProto.FLOAT, TensorProto.DOUBLE})
_FLOAT_DATATYPE_TO_ONNX: dict[str, int] = {"float32": TensorProto.FLOAT, "float64": TensorProto.DOUBLE}
_FLOAT_DATATYPE_TO_PL: dict[str, Any] = {"float32": pl.Float32, "float64": pl.Float64}


def apply_float_dtype(onnx_type: int, float_datatype: str) -> int:
    """Override FLOAT/DOUBLE with ``float_datatype``; STRING/BOOL/INT64 pass through unchanged."""
    return _FLOAT_DATATYPE_TO_ONNX[float_datatype] if onnx_type in _FLOAT_ONNX_TYPES else onnx_type


def col_out_name(transformer: _BaseTransformer, col: str, output_names: dict[str, str]) -> str:
    """Resolve the ONNX output tensor name for ``col``, honoring _column_mapping renames.

    ``_column_mapping`` maps a source column to a LIST of generated column names;
    this helper is for the 1:1 case and uses the first (only) entry.
    """
    col_map: dict[str, list[str]] = getattr(transformer, "_column_mapping", {})
    renamed = col_map.get(col, [col])[0]
    return output_names.get(renamed, renamed)


def _onnx_type_to_numpy(onnx_type: int) -> type:
    """Return the numpy dtype for building an ONNX scalar initializer of this type."""
    if onnx_type == TensorProto.DOUBLE:
        return np.float64
    if onnx_type == TensorProto.INT64:
        return np.int64
    return np.float32


@singledispatch
def get_input_onnx_type(transformer, col: str) -> int:
    """Return the ONNX TensorProto dtype expected for input column ``col``.

    Reads ``_input_dtypes`` stored by ``_BaseTransformer.fit`` when available;
    defaults to FLOAT otherwise.
    """
    dtypes = getattr(transformer, "_input_dtypes", {})
    dtype = dtypes.get(col)
    if dtype in (pl.String, pl.Utf8):  # string columns must never fall through to DOUBLE
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


def resolve_declared_output_dtype(transformer, col: str) -> int | None:
    """Resolve ``col``'s ONNX type from the transformer's own ``self._output_dtypes``.

    ``_output_dtypes`` (declared in every transformer's ``fit()``) is the ground-truth
    source for what a transformer actually produces — this is the single place that
    interprets it, so every ``get_output_onnx_type`` registration (generic or per-family)
    can defer to it instead of re-deriving the type via its own heuristics.

    Returns ``None`` when ``col`` is not declared, so callers can fall back to their
    own logic. String/Boolean/integer/temporal dtypes resolve to a concrete ONNX type;
    Float32/Float64 resolve to UNDEFINED so the caller still applies ``float_datatype``.
    """
    output_dtypes = getattr(transformer, "_output_dtypes", {})
    if col not in output_dtypes:
        return None
    dtype = output_dtypes[col]
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    if dtype in (pl.Float32, pl.Float64):
        return TensorProto.UNDEFINED
    if hasattr(dtype, "base_type") and dtype.base_type() in _TEMPORAL_PL_DTYPES:
        return TensorProto.INT64
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@singledispatch
def get_output_onnx_type(transformer, col: str) -> int:
    """Return the ONNX TensorProto dtype produced for output column ``col``.

    First choice: ``_output_dtypes`` declared by the transformer's ``fit()`` — the
    authoritative source of truth for what it actually produces (see
    ``resolve_declared_output_dtype``).

    Falls back to the pre-existing heuristics (pass-through via ``_input_dtypes``,
    renamed columns via reversed ``_column_mapping``) for transformers that don't yet
    populate ``_output_dtypes``. Returns UNDEFINED for truly new columns that cannot
    be resolved — callers should treat UNDEFINED as "inherits from input type" and
    apply float_datatype.
    """
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared

    dtypes = getattr(transformer, "_input_dtypes", {})
    if col in dtypes:
        return get_input_onnx_type(transformer, col)
    col_map = getattr(transformer, "_column_mapping", {})
    reverse = {v: k for k, values in col_map.items() for v in values}
    orig = reverse.get(col)
    if orig is not None:
        dtype = dtypes.get(orig)
        if dtype in (pl.String, pl.Utf8):
            return TensorProto.STRING  # pragma: no cover
        return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)
    return TensorProto.UNDEFINED


@singledispatch
def get_output_columns(transformer, input_columns: list[str]) -> list[str]:
    """Return the column names produced after applying ``transformer``.

    Falls back to the _BaseTransformer handler which uses _column_mapping when
    present. Register per transformer type to override (e.g. OneHotEncoder).
    """
    return list(input_columns)  # pragma: no cover


@get_output_columns.register(_BaseTransformer)
def _base_transformer_output_columns(transformer: _BaseTransformer, input_columns: list[str]) -> list[str]:
    """Smart default: derive output columns from _column_mapping when non-empty.

    ``_column_mapping`` maps each source column to a LIST of generated column names
    (supports both 1:1 and 1:many transformers).
    """
    col_map: dict[str, list[str]] = getattr(transformer, "_column_mapping", {})
    if not col_map:
        return list(input_columns)
    new_cols = [name for names in col_map.values() for name in names]
    if getattr(transformer, "drop_columns", True):
        dropped = set(col_map)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


@singledispatch
def to_onnx_nodes(
    transformer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """Return the ONNX nodes and initializers that implement ``transformer``.

    Parameters
    ----------
    transformer :
        A fitted Gators transformer instance.
    input_names :
        Mapping from column name to the current ONNX tensor name carrying that column's data.
    output_names :
        Mapping from column name to the desired ONNX tensor name for that column's output.
    errors : {'coerce', 'raise'}, default 'coerce'
        'raise'  – raise OnnxNotSupportedError for unsupported transformers / strategies.
        'coerce' – emit Identity pass-through nodes so the graph stays valid.
    """
    if errors == "raise":
        raise OnnxNotSupportedError(
            f"No ONNX converter registered for '{type(transformer).__name__}'. "
            f"Use errors='coerce' to pass all columns through unchanged, "
            f"or register a converter with @to_onnx_nodes.register(YourTransformer)."
        )
    # coerce: pass every column through unchanged
    nodes = [
        oh.make_node("Identity", inputs=[v], outputs=[output_names[k]])
        for k, v in input_names.items()
    ]
    return nodes, []


def _has_onnx_converter(transformer: _BaseTransformer) -> bool:
    """Return True when a non-fallback ONNX converter is registered for this transformer type."""
    return to_onnx_nodes.dispatch(type(transformer)) is not to_onnx_nodes.registry[object]


def check_pipeline_onnx_compatibility(pipeline: Pipeline) -> dict[str, bool]:
    """Return the ONNX compatibility status of every step in a fitted Gators Pipeline.

    Parameters
    ----------
    pipeline : gators.Pipeline
        A pipeline whose ``steps`` attribute is a list of ``(name, transformer)`` tuples.

    Returns
    -------
    dict[str, bool]
        ``{step_name: True}`` when a converter is registered for the transformer,
        ``{step_name: False}`` when only the fallback (Identity pass-through) would be used.

    Examples
    --------
    >>> from gators.pipeline import Pipeline
    >>> from gators.imputers import NumericImputer
    >>> from gators.scalers import StandardScaler
    >>> pipe = Pipeline(steps=[
    ...     ("imputer", NumericImputer(strategy="mean")),
    ...     ("scaler",  StandardScaler()),
    ... ])
    >>> from gators.onnx_converters import check_pipeline_onnx_compatibility
    >>> check_pipeline_onnx_compatibility(pipe)
    {'imputer': True, 'scaler': True}
    """
    return {name: _has_onnx_converter(transformer) for name, transformer in pipeline.steps}


