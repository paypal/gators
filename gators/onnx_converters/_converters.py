from __future__ import annotations

from functools import singledispatch

import polars as pl

from ._exceptions import OnnxNotSupportedError
from ..pipeline.pipeline import Pipeline
from ..transformer._base_transformer import _BaseTransformer

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

# Float64 → DOUBLE; Float32 → FLOAT; Boolean → BOOL; all other numeric types fall back to DOUBLE.
_POLARS_TO_ONNX: dict = {pl.Float32: TensorProto.FLOAT, pl.Float64: TensorProto.DOUBLE, pl.Boolean: TensorProto.BOOL}


def col_out_name(transformer: _BaseTransformer, col: str, output_names: dict[str, str]) -> str:
    """Resolve the ONNX output tensor name for ``col``, honoring _column_mapping renames."""
    col_map: dict[str, str] = getattr(transformer, "_column_mapping", {})
    renamed = col_map.get(col, col)
    return output_names.get(renamed, renamed)


def _onnx_type_to_numpy(onnx_type: int) -> type:
    """Return the numpy dtype that corresponds to an ONNX float scalar type."""
    return np.float64 if onnx_type == TensorProto.DOUBLE else np.float32


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
    return _POLARS_TO_ONNX.get(dtype, TensorProto.DOUBLE)


@singledispatch
def get_output_onnx_type(transformer, col: str) -> int:
    """Return the ONNX TensorProto dtype produced for output column ``col``.

    Checks ``_input_dtypes`` by direct name first, then resolves via reversed
    ``_column_mapping`` (covers scalers/generators that rename columns).
    Defaults to FLOAT when neither lookup succeeds.
    """
    dtypes = getattr(transformer, "_input_dtypes", {})
    if col in dtypes:
        dtype = dtypes[col]
        if dtype in (pl.String, pl.Utf8):
            return TensorProto.STRING
        return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)
    col_map = getattr(transformer, "_column_mapping", {})
    reverse = {v: k for k, v in col_map.items()}
    orig = reverse.get(col)
    if orig is not None:
        dtype = dtypes.get(orig)
        if dtype in (pl.String, pl.Utf8):
            return TensorProto.STRING  # pragma: no cover
        return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)
    return TensorProto.FLOAT


@singledispatch
def get_output_columns(transformer, input_columns: list[str]) -> list[str]:
    """Return the column names produced after applying ``transformer``.

    Falls back to the _BaseTransformer handler which uses _column_mapping when
    present. Register per transformer type to override (e.g. OneHotEncoder).
    """
    return list(input_columns)  # pragma: no cover


@get_output_columns.register(_BaseTransformer)
def _base_transformer_output_columns(transformer: _BaseTransformer, input_columns: list[str]) -> list[str]:
    """Smart default: derive output columns from _column_mapping when non-empty."""
    col_map = getattr(transformer, "_column_mapping", {})
    if not col_map:
        return list(input_columns)
    new_cols = list(col_map.values())
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


