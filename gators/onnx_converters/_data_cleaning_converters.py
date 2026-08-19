"""ONNX converters for gators.data_cleaning transformers.

Importing this module registers all supported converters.

Supported
---------
DropColumns            : drops transformer.subset columns (get_output_columns only)
SelectColumns          : keeps only transformer.subset columns (get_output_columns only)
DropConstantColumns    : drops _to_drop columns (get_output_columns only)
DropHighNaNRatio       : drops _to_drop columns (get_output_columns only)
DropNearConstantColumns: drops _to_drop columns (get_output_columns only)
DropDuplicateColumns   : drops columns_to_drop_ (get_output_columns only)
VarianceFilter         : drops _to_drop columns (get_output_columns only)
CorrelationFilter      : drops _to_drop columns (get_output_columns only)
RenameColumns          : renames via Identity nodes (changes output tensor names)
RoundDigits            : Round(Mul(X, 10^n)) / 10^n
CastColumns            : Cast node for numeric/boolean target dtypes.

Not supported
-------------
DropDuplicateRows  : requires row-level deduplication — no ONNX equivalent.
CastColumns (String target, numeric source): ONNX Cast→STRING format differs from
    Polars for numeric types (e.g. "-3" vs "-3.0"); raises OnnxNotSupportedError.
    Boolean→String is supported: both ONNX and Polars produce "true"/"false".
Replace            : string pass-through for unmatched values cannot be expressed cleanly.
HighCardinalityFilter: couldn't identify a stable API across versions.
DropLowCardinality : similar API instability.
"""
from __future__ import annotations

import polars as pl

from ._converters import _onnx_type_to_numpy, _POLARS_TO_ONNX, get_input_onnx_type, get_output_columns, get_output_onnx_type, to_onnx_nodes
from ._exceptions import OnnxNotSupportedError
from ..data_cleaning.cast_columns import CastColumns
from ..data_cleaning.drop_columns import DropColumns
from ..data_cleaning.drop_constant_columns import DropConstantColumns
from ..data_cleaning.drop_duplicate_columns import DropDuplicateColumns
from ..data_cleaning.drop_high_nan_ratio import DropHighNaNRatio
from ..data_cleaning.drop_near_constant_columns import DropNearConstantColumns
from ..data_cleaning.rename_columns import RenameColumns
from ..data_cleaning.round_digits import RoundDigits
from ..data_cleaning.round_significant_digits import RoundSignificantDigits
from ..data_cleaning.select_columns import SelectColumns
from ..data_cleaning.variance_filter import VarianceFilter
from ..data_cleaning.correlation_filter import CorrelationFilter

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


# ── Column-dropping transformers: get_output_columns only ─────────────────────
# These transformers produce no computation nodes; they only change the column
# schema.  get_output_columns removes the dropped columns; to_onnx_nodes emits
# Identity pass-through for every surviving column.

def _drop_output_cols(input_columns: list[str], to_drop: list[str]) -> list[str]:
    dropped = set(to_drop or [])
    return [c for c in input_columns if c not in dropped]


def _identity_passthrough(input_names: dict[str, str], output_names: dict[str, str]) -> tuple[list[onnx.NodeProto], list]:
    return [
        oh.make_node("Identity", inputs=[v], outputs=[output_names[k]])
        for k, v in input_names.items()
        if k in output_names
    ], []


@get_output_columns.register(DropColumns)
def _(transformer: DropColumns, input_columns: list[str]) -> list[str]:
    return _drop_output_cols(input_columns, transformer.subset or [])


@to_onnx_nodes.register(DropColumns)
def _(transformer: DropColumns, input_names, output_names, errors="coerce"):
    return _identity_passthrough(input_names, output_names)


@get_output_columns.register(SelectColumns)
def _(transformer: SelectColumns, input_columns: list[str]) -> list[str]:
    return [c for c in transformer.subset if c in input_columns]


@to_onnx_nodes.register(SelectColumns)
def _(transformer: SelectColumns, input_names, output_names, errors="coerce"):
    return _identity_passthrough(input_names, output_names)


@get_output_columns.register(DropConstantColumns)
def _(transformer: DropConstantColumns, input_columns: list[str]) -> list[str]:
    return _drop_output_cols(input_columns, transformer._to_drop)


@to_onnx_nodes.register(DropConstantColumns)
def _(transformer: DropConstantColumns, input_names, output_names, errors="coerce"):
    return _identity_passthrough(input_names, output_names)


@get_output_columns.register(DropHighNaNRatio)
def _(transformer: DropHighNaNRatio, input_columns: list[str]) -> list[str]:
    return _drop_output_cols(input_columns, transformer._to_drop)


@to_onnx_nodes.register(DropHighNaNRatio)
def _(transformer: DropHighNaNRatio, input_names, output_names, errors="coerce"):
    return _identity_passthrough(input_names, output_names)


@get_output_columns.register(DropNearConstantColumns)
def _(transformer: DropNearConstantColumns, input_columns: list[str]) -> list[str]:
    return _drop_output_cols(input_columns, transformer._to_drop)


@to_onnx_nodes.register(DropNearConstantColumns)
def _(transformer: DropNearConstantColumns, input_names, output_names, errors="coerce"):
    return _identity_passthrough(input_names, output_names)


@get_output_columns.register(DropDuplicateColumns)
def _(transformer: DropDuplicateColumns, input_columns: list[str]) -> list[str]:
    return _drop_output_cols(input_columns, transformer.columns_to_drop_)


@to_onnx_nodes.register(DropDuplicateColumns)
def _(transformer: DropDuplicateColumns, input_names, output_names, errors="coerce"):
    return _identity_passthrough(input_names, output_names)


@get_output_columns.register(VarianceFilter)
def _(transformer: VarianceFilter, input_columns: list[str]) -> list[str]:
    return _drop_output_cols(input_columns, transformer._to_drop)


@to_onnx_nodes.register(VarianceFilter)
def _(transformer: VarianceFilter, input_names, output_names, errors="coerce"):
    return _identity_passthrough(input_names, output_names)


@get_output_columns.register(CorrelationFilter)
def _(transformer: CorrelationFilter, input_columns: list[str]) -> list[str]:
    return _drop_output_cols(input_columns, transformer._to_drop)


@to_onnx_nodes.register(CorrelationFilter)
def _(transformer: CorrelationFilter, input_names, output_names, errors="coerce"):
    return _identity_passthrough(input_names, output_names)


# ── RenameColumns: Identity with changed output tensor names ──────────────────

@get_input_onnx_type.register(RenameColumns)
def _rename_input_onnx_type(transformer: RenameColumns, col: str) -> int:
    # Use original fitted dtypes when pipeline_to_onnx has overridden _input_dtypes.
    orig = transformer.__dict__.get('_onnx_orig_input_dtypes')
    dtype = (orig or getattr(transformer, "_input_dtypes", {})).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    # bool→string: INT64 with -1=null, 0=false, 1=true for null-safe encoding
    if dtype == pl.Boolean and transformer.dtype == pl.String:  # pragma: no cover
        return TensorProto.INT64
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_columns.register(RenameColumns)
def _(transformer: RenameColumns, input_columns: list[str]) -> list[str]:
    return [transformer.column_mapping.get(c, c) for c in input_columns]


@get_output_onnx_type.register(RenameColumns)
def _rename_output_onnx_type(transformer: RenameColumns, col: str) -> int:
    # Reverse the name mapping to find the original column and read its dtype.
    reverse = {v: k for k, v in transformer.column_mapping.items()}
    return get_input_onnx_type(transformer, reverse.get(col, col))


@to_onnx_nodes.register(RenameColumns)
def _rename_columns_to_onnx_nodes(transformer: RenameColumns, input_names, output_names, errors="coerce"):
    """Emit Identity(old_tensor) → new_tensor_name for each renamed column."""
    nodes: list[onnx.NodeProto] = []
    for col, in_name in input_names.items():
        new_col = transformer.column_mapping.get(col, col)
        out_name = output_names.get(new_col, new_col)
        nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[out_name]))
    return nodes, []


# ── RoundDigits ───────────────────────────────────────────────────────────────

@to_onnx_nodes.register(RoundDigits)
def _round_digits_to_onnx_nodes(transformer: RoundDigits, input_names, output_names, errors="coerce"):
    """Round(Mul(X, 10^n)) / 10^n — same column names (inplace)."""
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    factor = float(10 ** transformer.n_digits)
    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = output_names[col]
        p = f"{in_name}__RoundDigits"
        factor_init = f"{p}__factor"
        mul_out = f"{p}__mul"
        round_out = f"{p}__round"
        initializers.append(
            onnx.numpy_helper.from_array(
                np.array([factor], dtype=_onnx_type_to_numpy(get_input_onnx_type(transformer, col))),
                name=factor_init,
            )
        )
        nodes.append(oh.make_node("Mul", inputs=[in_name, factor_init], outputs=[mul_out]))
        nodes.append(oh.make_node("Round", inputs=[mul_out], outputs=[round_out]))
        nodes.append(oh.make_node("Div", inputs=[round_out, factor_init], outputs=[out_name]))

    return nodes, initializers


# ── CastColumns ───────────────────────────────────────────────────────────────

# Polars numeric/boolean dtypes that map cleanly to ONNX primitive types.
_CAST_DTYPE_TO_ONNX: dict = {
    pl.Float32: TensorProto.FLOAT,
    pl.Float64: TensorProto.DOUBLE,
    pl.Int8: TensorProto.INT8,
    pl.Int16: TensorProto.INT16,
    pl.Int32: TensorProto.INT32,
    pl.Int64: TensorProto.INT64,
    pl.UInt8: TensorProto.UINT8,
    pl.UInt16: TensorProto.UINT16,
    pl.UInt32: TensorProto.UINT32,
    pl.UInt64: TensorProto.UINT64,
    pl.Boolean: TensorProto.BOOL,
}


@get_output_columns.register(CastColumns)
def _cast_output_cols(transformer: CastColumns, input_columns: list[str]) -> list[str]:
    if transformer.inplace or not transformer._column_mapping:
        return list(input_columns)
    new_cols = list(transformer._column_mapping.values())
    if transformer.drop_columns:
        dropped = set(transformer._column_mapping)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


@get_input_onnx_type.register(CastColumns)
def _cast_input_onnx_type(transformer: CastColumns, col: str) -> int:
    # Use original fitted dtypes when pipeline_to_onnx has overridden _input_dtypes.
    orig = transformer.__dict__.get('_onnx_orig_input_dtypes')
    dtype = (orig or getattr(transformer, "_input_dtypes", {})).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    # bool→string: INT64 with -1=null, 0=false, 1=true for null-safe encoding
    if dtype == pl.Boolean and transformer.dtype == pl.String:
        return TensorProto.INT64
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(CastColumns)
def _cast_output_onnx_type(transformer: CastColumns, col: str) -> int:
    subset = set(transformer.subset or [])
    renamed_vals = set(transformer._column_mapping.values())
    if (transformer.inplace and col in subset) or col in renamed_vals:
        if transformer.dtype == pl.String:
            return TensorProto.STRING
        # Datetime/Date/Duration: physical representation stays INT64
        if transformer.dtype.base_type() in (pl.Datetime, pl.Date, pl.Duration, pl.Time):
            return TensorProto.INT64
        return _CAST_DTYPE_TO_ONNX.get(transformer.dtype, TensorProto.FLOAT)
    return get_input_onnx_type(transformer, col)


@to_onnx_nodes.register(CastColumns)
def _cast_columns_to_onnx_nodes(
    transformer: CastColumns,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """ONNX Cast node for each subset column; Identity for all others.

    Raises OnnxNotSupportedError (or coerces to Identity) for non-boolean→String
    and Datetime/Date target dtypes, which have no reliable ONNX Cast equivalent.
    Boolean→String is allowed: both ONNX and Polars produce "true"/"false".
    """
    target_onnx_type = _CAST_DTYPE_TO_ONNX.get(transformer.dtype)
    if target_onnx_type is None:
        # Datetime/Date/Duration: int64 IS the physical form — passthrough unchanged.
        if transformer.dtype.base_type() in (pl.Datetime, pl.Date, pl.Duration, pl.Time):
            return _identity_passthrough(input_names, output_names)
        if transformer.dtype == pl.String:
            # bool→string is safe via LabelEncoder; numeric→string has format mismatches
            # Use original fitted dtypes (pipeline override may have changed _input_dtypes).
            check_dtypes = transformer.__dict__.get('_onnx_orig_input_dtypes') or getattr(transformer, "_input_dtypes", {})
            non_bool = [c for c in (transformer.subset or []) if check_dtypes.get(c) != pl.Boolean]
            if non_bool:
                msg = (
                    f"CastColumns with dtype=String cannot be exported to ONNX for "
                    f"non-boolean source columns {non_bool}. "
                    "ONNX Cast→STRING format differs from Polars for numeric types."
                )
                if errors == "raise":
                    raise OnnxNotSupportedError(msg)
                return _identity_passthrough(input_names, output_names)
            # INT64 input: -1=null → "" (empty-string null sentinel), 0=false → "false", 1=true → "true"
            # StringImputer's ONNX convention: "" represents null, which it fills with the target value.
            nodes: list[onnx.NodeProto] = []
            subset = set(transformer.subset or [])
            for col, in_name in input_names.items():
                if col not in subset and col in output_names:
                    nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
            for col in (transformer.subset or []):
                if col not in input_names:  # pragma: no cover
                    continue
                in_name = input_names[col]
                out_name = output_names.get(col, col)
                nodes.append(oh.make_node(
                    "LabelEncoder", inputs=[in_name], outputs=[out_name],
                    domain="ai.onnx.ml",
                    keys_int64s=[-1, 0, 1],
                    values_strings=["", "false", "true"],
                    default_string="",
                ))
            return nodes, []
        msg = (
            f"CastColumns with dtype={transformer.dtype!r} cannot be exported to ONNX. "
            "Only numeric and boolean target dtypes are supported."
        )
        if errors == "raise":
            raise OnnxNotSupportedError(msg)
        return _identity_passthrough(input_names, output_names)

    nodes: list[onnx.NodeProto] = []
    subset = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    if not transformer.inplace and not transformer.drop_columns:
        for col in (transformer.subset or []):
            if col in input_names and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        if transformer.inplace:
            out_name = output_names.get(col, col)
        else:
            new_col = transformer._column_mapping.get(col, col)
            out_name = output_names.get(new_col, new_col)
        nodes.append(oh.make_node("Cast", inputs=[in_name], outputs=[out_name], to=target_onnx_type))

    return nodes, []


# ── RoundSignificantDigits ────────────────────────────────────────────────────

@get_output_columns.register(RoundSignificantDigits)
def _rsd_output_columns(transformer: RoundSignificantDigits, input_columns: list[str]) -> list[str]:
    if transformer.inplace:
        return list(input_columns)
    subset = set(transformer.subset or [])
    new_cols = [transformer._column_mapping[c] for c in (transformer.subset or [])]
    if transformer.drop_columns:
        return [c for c in input_columns if c not in subset] + new_cols
    return list(input_columns) + new_cols


@to_onnx_nodes.register(RoundSignificantDigits)
def _rsd_to_onnx_nodes(
    transformer: RoundSignificantDigits,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """Significant-figure rounding: round(x * 10^(n-1-floor(log10(|x|)))) / 10^(n-1-floor(log10(|x|))).

    Zero values are passed through unchanged via a Where node.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset = set(transformer.subset or [])
    n = transformer.n_digits

    # Scalar constants reused across all columns
    ln10_name = "__rsd_ln10"
    ndig_name = f"__rsd_ndig_{n}"
    zero_name = "__rsd_zero"
    eps_name = "__rsd_eps"
    initializers += [
        onnx.numpy_helper.from_array(np.array([np.log(10.0)], dtype=np.float64), name=ln10_name),
        onnx.numpy_helper.from_array(np.array([float(n - 1)], dtype=np.float64), name=ndig_name),
        onnx.numpy_helper.from_array(np.array([0.0], dtype=np.float64), name=zero_name),
        onnx.numpy_helper.from_array(np.array([1e-300], dtype=np.float64), name=eps_name),
    ]

    for col, in_name in input_names.items():
        if col not in subset and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    if not transformer.inplace and not transformer.drop_columns:
        for col in (transformer.subset or []):
            if col in input_names and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        if transformer.inplace:
            out_name = output_names.get(col, col)
        else:
            new_col = transformer._column_mapping.get(col, col)
            out_name = output_names.get(new_col, new_col)

        p = f"{in_name}__rsd"
        # Cast to float64 for precision
        cast_name = f"{p}__f64"
        abs_name = f"{p}__abs"
        abs_safe_name = f"{p}__abs_safe"
        log_abs_name = f"{p}__log_abs"
        log10_name = f"{p}__log10"
        floor_name = f"{p}__floor"
        exp_name = f"{p}__exp"    # n_digits - 1 - floor(log10(|x|))
        ln_exp_name = f"{p}__ln_exp"
        factor_name = f"{p}__factor"
        scaled_name = f"{p}__scaled"
        rounded_name = f"{p}__rounded"
        result_name = f"{p}__result"
        is_zero_name = f"{p}__is_zero"

        nodes += [
            oh.make_node("Cast", inputs=[in_name], outputs=[cast_name], to=TensorProto.DOUBLE),
            oh.make_node("Abs", inputs=[cast_name], outputs=[abs_name]),
            # guard against log(0): max(|x|, eps)
            oh.make_node("Max", inputs=[abs_name, eps_name], outputs=[abs_safe_name]),
            oh.make_node("Log", inputs=[abs_safe_name], outputs=[log_abs_name]),
            oh.make_node("Div", inputs=[log_abs_name, ln10_name], outputs=[log10_name]),
            oh.make_node("Floor", inputs=[log10_name], outputs=[floor_name]),
            # exponent = (n_digits - 1) - floor(log10(|x|))
            oh.make_node("Sub", inputs=[ndig_name, floor_name], outputs=[exp_name]),
            # factor = 10^exponent = exp(exponent * ln(10))
            oh.make_node("Mul", inputs=[exp_name, ln10_name], outputs=[ln_exp_name]),
            oh.make_node("Exp", inputs=[ln_exp_name], outputs=[factor_name]),
            oh.make_node("Mul", inputs=[cast_name, factor_name], outputs=[scaled_name]),
            oh.make_node("Round", inputs=[scaled_name], outputs=[rounded_name]),
            oh.make_node("Div", inputs=[rounded_name, factor_name], outputs=[result_name]),
            # pass through zero unchanged
            oh.make_node("Equal", inputs=[cast_name, zero_name], outputs=[is_zero_name]),
            oh.make_node("Where", inputs=[is_zero_name, zero_name, result_name], outputs=[out_name]),
        ]

    return nodes, initializers
