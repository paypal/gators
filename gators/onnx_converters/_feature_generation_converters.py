"""ONNX converters for gators.feature_generation transformers.

Importing this module registers all supported converters on the
``to_onnx_nodes`` and ``get_output_columns`` singledispatch functions.

Supported
---------
ScalarMathFeatures       : col OP scalar  (+, -, *, /, **, //, %)
IsNull                   : is_null() → IsNaN (numeric) or Equal(x,"") (string) → Cast(FLOAT)
RatioFeatures            : num / (denom + 1)  (Laplace smoothing)
WeightedSumFeatures      : sum(coeff_i * col_i) + bias
AsymmetryIndexFeatures   : x² / (y * (x + y))  ± smoothing
FourierFeatures          : sin/cos harmonics
PolynomialFeatures       : degree-N products and interactions
ComparisonFeatures       : >, <, >=, <=, ==, !=, is_null, is_not_null
ConditionFeatures        : per-condition scalar/column/unary → bool → Cast(FLOAT)
RuleFeatures             : AND/OR of per-condition booleans → Cast(FLOAT)
MathFeatures             : row-wise sum/mean/min/max/range across column groups
RowStatisticsFeatures    : row-wise sum/mean/min/max/range/std/count (not median)
ConcentrationIndexFeatures : num / (sum(denoms) [+ 1])
GeneralizedRatioFeatures : weighted-linear-num / (weighted-linear-denom + ε)
PlanRotationFeatures     : x·cos(θ) ± y·sin(θ) per angle
DistanceFeatures         : euclidean, manhattan, haversine

HHIFeatures              : sum of squared market-share terms
GroupStatisticsFeatures  : LabelEncoder group-stat lookup + arithmetic for all 12 functions
                           Uses training-time group statistics (not batch-time .over()).

Not supported (require window/partition ops absent from ONNX)
------------------------------------------------------------
GroupLagFeatures, RollingStatisticsFeatures

Partial support
---------------
RowStatisticsFeatures : 'median' raises OnnxNotSupportedError (no ONNX median op)
"""
from __future__ import annotations

import math
from itertools import combinations_with_replacement

import polars as pl

from ._converters import _onnx_type_to_numpy, _POLARS_TO_ONNX, get_input_onnx_type, get_output_columns, get_output_onnx_type, to_onnx_nodes
from ._exceptions import OnnxNotSupportedError
from ..feature_generation.asymmetry_index_features import AsymmetryIndexFeatures
from ..feature_generation.comparison_features import ComparisonFeatures
from ..feature_generation.concentration_index_features import ConcentrationIndexFeatures
from ..feature_generation.condition_features import ConditionFeatures
from ..feature_generation.distance_features import DistanceFeatures, EARTH_RADIUS
from ..feature_generation.fourier_features import FourierFeatures
from ..feature_generation.generalized_ratio_features import GeneralizedRatioFeatures
from ..feature_generation.group_statistics_features import GroupStatisticsFeatures
from ..feature_generation.hhi_features import HHIFeatures
from ..feature_generation.is_null import IsNull
from ..feature_generation.math_features import MathFeatures
from ..feature_generation.plan_rotation_features import PlanRotationFeatures
from ..feature_generation.polynomial_features import PolynomialFeatures
from ..feature_generation.ratio_features import RatioFeatures
from ..feature_generation.row_statistics_features import RowStatisticsFeatures
from ..feature_generation.rule_features import RuleFeatures
from ..feature_generation.scalar_math_features import ScalarMathFeatures
from ..feature_generation.weighted_sum_features import WeightedSumFeatures

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


# ── internal helpers ──────────────────────────────────────────────────────────

def _make_init(name: str, value: float, onnx_type: int) -> onnx.TensorProto:
    """Scalar initializer that broadcasts in element-wise ops."""
    return onnx.numpy_helper.from_array(
        np.array([value], dtype=_onnx_type_to_numpy(onnx_type)), name=name
    )


def _passthrough(input_names: dict, output_names: dict) -> list[onnx.NodeProto]:
    """Identity nodes for all original columns that survive into the output."""
    return [
        oh.make_node("Identity", inputs=[v], outputs=[output_names[k]])
        for k, v in input_names.items()
        if k in output_names
    ]


def _cast_numeric(bool_tensor: str, out: str, onnx_type: int) -> onnx.NodeProto:
    return oh.make_node("Cast", inputs=[bool_tensor], outputs=[out], to=onnx_type)


# ── ScalarMathFeatures ────────────────────────────────────────────────────────

@get_output_columns.register(ScalarMathFeatures)
def _smf_output_cols(transformer: ScalarMathFeatures, input_columns: list[str]) -> list[str]:
    new = transformer.new_column_names or transformer._generated_column_names
    return list(input_columns) + list(new)


@to_onnx_nodes.register(ScalarMathFeatures)
def _smf_to_onnx(
    transformer: ScalarMathFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """col OP scalar  using Add/Sub/Mul/Div/Pow/Floor nodes."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    out_cols = transformer.new_column_names or transformer._generated_column_names

    for op_dict, out_col in zip(transformer.operations, out_cols):
        col = op_dict["column"]
        op = op_dict["op"]
        scalar = float(op_dict["scalar"])
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = output_names.get(out_col, out_col)
        p = f"{in_name}__SMF__{out_col}"
        s_init = f"{p}__s"
        initializers.append(_make_init(s_init, scalar, get_input_onnx_type(transformer, col)))

        if op == "+":
            nodes.append(oh.make_node("Add", inputs=[in_name, s_init], outputs=[out_name]))
        elif op == "-":
            nodes.append(oh.make_node("Sub", inputs=[in_name, s_init], outputs=[out_name]))
        elif op == "*":
            nodes.append(oh.make_node("Mul", inputs=[in_name, s_init], outputs=[out_name]))
        elif op == "/":
            nodes.append(oh.make_node("Div", inputs=[in_name, s_init], outputs=[out_name]))
        elif op == "**":
            nodes.append(oh.make_node("Pow", inputs=[in_name, s_init], outputs=[out_name]))
        elif op == "//":
            # floor(x / scalar)
            div_out = f"{p}__div"
            nodes.append(oh.make_node("Div", inputs=[in_name, s_init], outputs=[div_out]))
            nodes.append(oh.make_node("Floor", inputs=[div_out], outputs=[out_name]))
        elif op == "%":
            # x - scalar * floor(x / scalar)
            div_out = f"{p}__div"
            floor_out = f"{p}__floor"
            mul_out = f"{p}__mul"
            nodes.append(oh.make_node("Div", inputs=[in_name, s_init], outputs=[div_out]))
            nodes.append(oh.make_node("Floor", inputs=[div_out], outputs=[floor_out]))
            nodes.append(oh.make_node("Mul", inputs=[floor_out, s_init], outputs=[mul_out]))
            nodes.append(oh.make_node("Sub", inputs=[in_name, mul_out], outputs=[out_name]))

    return nodes, initializers


# ── IsNull ────────────────────────────────────────────────────────────────────

@get_output_columns.register(IsNull)
def _isnull_output_cols(transformer: IsNull, input_columns: list[str]) -> list[str]:
    return list(input_columns) + list(transformer._column_mapping.values())


@get_input_onnx_type.register(IsNull)
def _isnull_input_onnx_type(transformer: IsNull, col: str) -> int:
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(IsNull)
def _isnull_output_onnx_type(transformer: IsNull, col: str) -> int:
    # Null indicators are always FLOAT (0.0 / 1.0) regardless of source column type.
    if col in transformer._column_mapping.values():
        return TensorProto.FLOAT
    return _isnull_input_onnx_type(transformer, col)


@to_onnx_nodes.register(IsNull)
def _isnull_to_onnx(
    transformer: IsNull,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """is_null(): string columns use LabelEncoder(""→1) → Cast(FLOAT); numeric use IsNaN → Cast(FLOAT)."""
    nodes = _passthrough(input_names, output_names)
    for orig_col, null_col in transformer._column_mapping.items():
        if orig_col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[orig_col]
        out_name = output_names.get(null_col, null_col)
        bool_name = f"{in_name}__IsNull__bool"
        if get_input_onnx_type(transformer, orig_col) == TensorProto.STRING:
            # Null strings arrive as "" after fill_null(""); map "" → 1, others → 0
            nodes.append(
                oh.make_node(
                    "LabelEncoder",
                    inputs=[in_name],
                    outputs=[bool_name],
                    domain="ai.onnx.ml",
                    keys_strings=[""],
                    values_int64s=[1],
                    default_int64=0,
                )
            )
        else:
            nodes.append(oh.make_node("IsNaN", inputs=[in_name], outputs=[bool_name]))
        onnx_type = get_input_onnx_type(transformer, orig_col)
        # null indicator is always FLOAT (0/1); cast_type must never be STRING or DOUBLE
        nodes.append(_cast_numeric(bool_name, out_name, TensorProto.FLOAT))
    return nodes, []


# ── RatioFeatures ─────────────────────────────────────────────────────────────

@get_output_columns.register(RatioFeatures)
def _ratio_output_cols(transformer: RatioFeatures, input_columns: list[str]) -> list[str]:
    new = list(transformer._column_mapping.values())
    if transformer.drop_columns:
        dropped = set(transformer.numerator_columns) | set(transformer.denominator_columns)
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(RatioFeatures)
def _ratio_to_onnx(
    transformer: RatioFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """num / (denom + 1) — Laplace smoothing via Add(denom, 1) + Div."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    for num_col, denom_col, out_col in zip(
        transformer.numerator_columns,
        transformer.denominator_columns,
        transformer._column_mapping.values(),
    ):
        if num_col not in input_names or denom_col not in input_names:  # pragma: no cover
            continue
        num_name = input_names[num_col]
        den_name = input_names[denom_col]
        out_name = output_names.get(out_col, out_col)
        p = f"{num_name}__Ratio__{den_name}"
        one_init = f"{p}__one"
        den_p1 = f"{p}__dep1"
        initializers.append(_make_init(one_init, 1.0, get_input_onnx_type(transformer, denom_col)))
        nodes.append(oh.make_node("Add", inputs=[den_name, one_init], outputs=[den_p1]))
        nodes.append(oh.make_node("Div", inputs=[num_name, den_p1], outputs=[out_name]))

    return nodes, initializers


# ── WeightedSumFeatures ───────────────────────────────────────────────────────

@get_output_columns.register(WeightedSumFeatures)
def _wsum_output_cols(transformer: WeightedSumFeatures, input_columns: list[str]) -> list[str]:
    new = list(transformer._column_mapping.values())
    if transformer.drop_columns:
        dropped = {c for group in transformer.column_groups for c in group}
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(WeightedSumFeatures)
def _wsum_to_onnx(
    transformer: WeightedSumFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """sum(coeff_i * col_i) + bias — chained Mul + Add nodes."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    for group, coeffs, bias, out_col in zip(
        transformer.column_groups,
        transformer.coefficients,
        transformer.biases,
        transformer._column_mapping.values(),
    ):
        out_name = output_names.get(out_col, out_col)
        p = f"WSum__{'_'.join(group)}"

        # Compute each weighted term: col_i * coeff_i
        term_names: list[str] = []
        for col, coeff in zip(group, coeffs):
            if col not in input_names:  # pragma: no cover
                continue
            in_name = input_names[col]
            coeff_init = f"{in_name}__{p}__c"
            term_out = f"{in_name}__{p}__term"
            initializers.append(_make_init(coeff_init, float(coeff), get_input_onnx_type(transformer, col)))
            nodes.append(oh.make_node("Mul", inputs=[in_name, coeff_init], outputs=[term_out]))
            term_names.append(term_out)

        if not term_names:  # pragma: no cover
            continue

        # Pairwise Add to sum the terms
        running = term_names[0]
        for i, term in enumerate(term_names[1:]):
            new_sum = f"{p}__sum{i}"
            nodes.append(oh.make_node("Add", inputs=[running, term], outputs=[new_sum]))
            running = new_sum

        # Add bias
        bias_init = f"{p}__bias"
        initializers.append(_make_init(bias_init, float(bias), get_input_onnx_type(transformer, group[0])))
        nodes.append(oh.make_node("Add", inputs=[running, bias_init], outputs=[out_name]))

    return nodes, initializers


# ── AsymmetryIndexFeatures ────────────────────────────────────────────────────

@get_output_columns.register(AsymmetryIndexFeatures)
def _asym_output_cols(transformer: AsymmetryIndexFeatures, input_columns: list[str]) -> list[str]:
    new = list(transformer._column_mapping.values())
    if transformer.drop_columns:
        dropped = set(transformer.x_columns) | set(transformer.y_columns)
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(AsymmetryIndexFeatures)
def _asym_to_onnx(
    transformer: AsymmetryIndexFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """x² / (y * (x + y))  ±smoothing — using Mul/Add/Div."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    for x_col, y_col, out_col in zip(
        transformer.x_columns, transformer.y_columns, transformer._column_mapping.values()
    ):
        if x_col not in input_names or y_col not in input_names:  # pragma: no cover
            continue
        x_in = input_names[x_col]
        y_in = input_names[y_col]
        out_name = output_names.get(out_col, out_col)
        p = f"Asym__{x_in}__{y_in}"

        if transformer.smoothing:
            one_init = f"{p}__one"
            initializers.append(_make_init(one_init, 1.0, get_input_onnx_type(transformer, x_col)))
            xs = f"{p}__xs"
            ys = f"{p}__ys"
            nodes.append(oh.make_node("Add", inputs=[x_in, one_init], outputs=[xs]))
            nodes.append(oh.make_node("Add", inputs=[y_in, one_init], outputs=[ys]))
        else:
            xs, ys = x_in, y_in

        # x² / (y * (x + y))
        xsq = f"{p}__xsq"
        xpy = f"{p}__xpy"
        yts = f"{p}__yts"
        nodes.append(oh.make_node("Mul", inputs=[xs, xs], outputs=[xsq]))
        nodes.append(oh.make_node("Add", inputs=[xs, ys], outputs=[xpy]))
        nodes.append(oh.make_node("Mul", inputs=[ys, xpy], outputs=[yts]))
        nodes.append(oh.make_node("Div", inputs=[xsq, yts], outputs=[out_name]))

    return nodes, initializers


# ── FourierFeatures ───────────────────────────────────────────────────────────

@get_output_columns.register(FourierFeatures)
def _fourier_output_cols(transformer: FourierFeatures, input_columns: list[str]) -> list[str]:
    new: list[str] = []
    for col in transformer.subset:
        for period in transformer._column_periods[col]:
            period_str = f"{period:g}"
            for k in range(1, transformer.n_harmonics + 1):
                new.append(f"{col}__fourier_sin_p{period_str}_k{k}")
                new.append(f"{col}__fourier_cos_p{period_str}_k{k}")
    if transformer.drop_columns:
        return [c for c in input_columns if c not in set(transformer.subset)] + new
    return list(input_columns) + new


@to_onnx_nodes.register(FourierFeatures)
def _fourier_to_onnx(
    transformer: FourierFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """sin/cos(2π·k·col/T) — Mul(col, factor) + Sin/Cos."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    two_pi = 2.0 * math.pi
    for col in transformer.subset:
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        onnx_type = get_input_onnx_type(transformer, col)
        for period in transformer._column_periods[col]:
            period_str = f"{period:g}"
            for k in range(1, transformer.n_harmonics + 1):
                factor = two_pi * k / period
                p = f"{in_name}__Fourier_p{period_str}_k{k}"
                factor_init = f"{p}__factor"
                scaled = f"{p}__scaled"
                sin_col = f"{col}__fourier_sin_p{period_str}_k{k}"
                cos_col = f"{col}__fourier_cos_p{period_str}_k{k}"
                sin_out = output_names.get(sin_col, sin_col)
                cos_out = output_names.get(cos_col, cos_col)

                initializers.append(_make_init(factor_init, factor, onnx_type))
                nodes.append(oh.make_node("Mul", inputs=[in_name, factor_init], outputs=[scaled]))
                nodes.append(oh.make_node("Sin", inputs=[scaled], outputs=[sin_out]))
                # onnxruntime does not support Cos for float64; use a Cast bridge.
                if onnx_type == TensorProto.DOUBLE:
                    cos_f32_in = f"{p}__cos_f32_in"
                    cos_f32_out = f"{p}__cos_f32_out"
                    nodes.append(oh.make_node("Cast", inputs=[scaled], outputs=[cos_f32_in], to=TensorProto.FLOAT))
                    nodes.append(oh.make_node("Cos", inputs=[cos_f32_in], outputs=[cos_f32_out]))
                    nodes.append(oh.make_node("Cast", inputs=[cos_f32_out], outputs=[cos_out], to=TensorProto.DOUBLE))
                else:
                    nodes.append(oh.make_node("Cos", inputs=[scaled], outputs=[cos_out]))

    return nodes, initializers


# ── PolynomialFeatures ────────────────────────────────────────────────────────

@get_output_columns.register(PolynomialFeatures)
def _poly_output_cols(transformer: PolynomialFeatures, input_columns: list[str]) -> list[str]:
    if transformer.subset is None:  # pragma: no cover
        return list(input_columns)
    new: list[str] = []
    if transformer.include_bias:
        new.append("bias")
    for deg in range(2, transformer.degree + 1):
        for combo in combinations_with_replacement(transformer.subset, deg):
            if transformer.interaction_only and len(set(combo)) != deg:
                continue
            new.append("__".join(combo))
    return list(input_columns) + new


@to_onnx_nodes.register(PolynomialFeatures)
def _poly_to_onnx(
    transformer: PolynomialFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Products of column combinations — chained Mul nodes."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    if transformer.subset is None:  # pragma: no cover
        return nodes, initializers

    if transformer.include_bias:
        bias_out = output_names.get("bias", "bias")
        # Produce a vector of 1.0 with the same length as the first input column:
        # 0 * col + 1.0  avoids needing to know N at graph-build time.
        any_col = next((c for c in transformer.subset if c in input_names), None)
        if any_col is not None:
            in_name = input_names[any_col]
            zero_init = "PolyFeatures__bias_zero"
            one_init = "PolyFeatures__bias_one"
            zero_mul = "PolyFeatures__bias_zero_mul"
            onnx_type = get_input_onnx_type(transformer, any_col)
            initializers.append(_make_init(zero_init, 0.0, onnx_type))
            initializers.append(_make_init(one_init, 1.0, onnx_type))
            nodes.append(oh.make_node("Mul", inputs=[in_name, zero_init], outputs=[zero_mul]))
            nodes.append(oh.make_node("Add", inputs=[zero_mul, one_init], outputs=[bias_out]))

    for deg in range(2, transformer.degree + 1):
        for combo in combinations_with_replacement(transformer.subset, deg):
            if transformer.interaction_only and len(set(combo)) != deg:
                continue
            if not all(c in input_names for c in combo):  # pragma: no cover
                continue
            out_col = "__".join(combo)
            out_name = output_names.get(out_col, out_col)
            p = f"Poly__{'_'.join(combo)}"

            # Chain: combo[0] * combo[1] * ... * combo[-1]
            product = input_names[combo[0]]
            for mul_idx, c in enumerate(combo[1:]):
                is_last = mul_idx == len(combo) - 2
                mul_out = out_name if is_last else f"{p}__m{mul_idx}"
                nodes.append(oh.make_node("Mul", inputs=[product, input_names[c]], outputs=[mul_out]))
                product = mul_out

    return nodes, initializers


# ── ComparisonFeatures ────────────────────────────────────────────────────────

_CMP_OP_NAMES = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte", "==": "eq", "!=": "ne"}
_CMP_ONNX_OPS = {
    ">": "Greater",
    "<": "Less",
    ">=": "GreaterOrEqual",
    "<=": "LessOrEqual",
    "==": "Equal",
}


def _cmp_col_name(col_a: str, col_b: str, op: str) -> str:
    if op in ("is_null", "is_not_null"):
        return f"{col_a}__{op}"
    return f"{col_a}_{_CMP_OP_NAMES[op]}_{col_b}"


@get_output_columns.register(ComparisonFeatures)
def _cmp_output_cols(transformer: ComparisonFeatures, input_columns: list[str]) -> list[str]:
    new = [
        _cmp_col_name(a, b, op)
        for a, b, op in zip(transformer.subset_a, transformer.subset_b, transformer.operators)
    ]
    if transformer.drop_columns:
        dropped = set(transformer.subset_a) | set(transformer.subset_b)
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(ComparisonFeatures)
def _cmp_to_onnx(
    transformer: ComparisonFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Comparison ops → bool → Cast(FLOAT).  is_null uses IsNaN."""
    nodes = _passthrough(input_names, output_names)

    for col_a, col_b, op in zip(
        transformer.subset_a, transformer.subset_b, transformer.operators
    ):
        if col_a not in input_names:  # pragma: no cover
            continue
        out_col = _cmp_col_name(col_a, col_b, op)
        out_name = output_names.get(out_col, out_col)
        in_a = input_names[col_a]
        p = f"{in_a}__Cmp__{op.replace('_', '')}"
        bool_out = f"{p}__bool"

        if op == "is_null":
            nodes.append(oh.make_node("IsNaN", inputs=[in_a], outputs=[bool_out]))
            nodes.append(_cast_numeric(bool_out, out_name, get_input_onnx_type(transformer, col_a)))
        elif op == "is_not_null":
            isnan_out = f"{p}__isnan"
            nodes.append(oh.make_node("IsNaN", inputs=[in_a], outputs=[isnan_out]))
            nodes.append(oh.make_node("Not", inputs=[isnan_out], outputs=[bool_out]))
            nodes.append(_cast_numeric(bool_out, out_name, get_input_onnx_type(transformer, col_a)))
        elif op == "!=":
            if col_b not in input_names:  # pragma: no cover
                continue
            in_b = input_names[col_b]
            eq_out = f"{p}__eq"
            nodes.append(oh.make_node("Equal", inputs=[in_a, in_b], outputs=[eq_out]))
            nodes.append(oh.make_node("Not", inputs=[eq_out], outputs=[bool_out]))
            nodes.append(_cast_numeric(bool_out, out_name, get_input_onnx_type(transformer, col_a)))
        else:
            if col_b not in input_names:  # pragma: no cover
                continue
            in_b = input_names[col_b]
            onnx_op = _CMP_ONNX_OPS[op]
            nodes.append(oh.make_node(onnx_op, inputs=[in_a, in_b], outputs=[bool_out]))
            nodes.append(_cast_numeric(bool_out, out_name, get_input_onnx_type(transformer, col_a)))

    return nodes, []


# ── ConditionFeatures ─────────────────────────────────────────────────────────

@get_output_columns.register(ConditionFeatures)
def _cond_output_cols(transformer: ConditionFeatures, input_columns: list[str]) -> list[str]:
    return list(input_columns) + list(transformer._generated_column_names)


@to_onnx_nodes.register(ConditionFeatures)
def _cond_to_onnx(
    transformer: ConditionFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Per-condition comparison → bool → Cast(FLOAT).

    Scalar comparisons use a Constant initializer; column comparisons use the
    second column tensor directly; unary ops use IsNaN / Not(IsNaN).
    """
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    for cond, out_col in zip(transformer.conditions, transformer._generated_column_names):
        col = cond["column"]
        op = cond["op"]
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = output_names.get(out_col, out_col)
        onnx_type = get_input_onnx_type(transformer, col)
        p = f"{in_name}__Cond__{out_col}"
        bool_out = f"{p}__bool"

        if op == "is_null":
            nodes.append(oh.make_node("IsNaN", inputs=[in_name], outputs=[bool_out]))
            nodes.append(_cast_numeric(bool_out, out_name, onnx_type))
        elif op == "is_not_null":
            isnan_out = f"{p}__isnan"
            nodes.append(oh.make_node("IsNaN", inputs=[in_name], outputs=[isnan_out]))
            nodes.append(oh.make_node("Not", inputs=[isnan_out], outputs=[bool_out]))
            nodes.append(_cast_numeric(bool_out, out_name, onnx_type))
        elif "other_column" in cond:
            other_col = cond["other_column"]
            if other_col not in input_names:  # pragma: no cover
                continue
            in_b = input_names[other_col]
            if op == "!=":
                eq_out = f"{p}__eq"
                nodes.append(oh.make_node("Equal", inputs=[in_name, in_b], outputs=[eq_out]))
                nodes.append(oh.make_node("Not", inputs=[eq_out], outputs=[bool_out]))
            else:
                nodes.append(oh.make_node(_CMP_ONNX_OPS[op], inputs=[in_name, in_b], outputs=[bool_out]))
            nodes.append(_cast_numeric(bool_out, out_name, onnx_type))
        else:
            value = float(cond["value"])
            scalar_init = f"{p}__scalar"
            initializers.append(_make_init(scalar_init, value, onnx_type))
            if op == "!=":
                eq_out = f"{p}__eq"
                nodes.append(oh.make_node("Equal", inputs=[in_name, scalar_init], outputs=[eq_out]))
                nodes.append(oh.make_node("Not", inputs=[eq_out], outputs=[bool_out]))
            else:
                nodes.append(oh.make_node(_CMP_ONNX_OPS[op], inputs=[in_name, scalar_init], outputs=[bool_out]))
            nodes.append(_cast_numeric(bool_out, out_name, onnx_type))

    return nodes, initializers


# ── HHIFeatures ───────────────────────────────────────────────────────────────

@get_output_columns.register(HHIFeatures)
def _hhi_output_cols(transformer: HHIFeatures, input_columns: list[str]) -> list[str]:
    new = list(transformer.new_column_names or [])
    if transformer.drop_columns:
        dropped = {c for group in transformer.column_groups for c in group}
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(HHIFeatures)
def _hhi_to_onnx_nodes(
    transformer: HHIFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Herfindahl-Hirschman Index: sum((col_i / (sum(group) + ε))²).

    For each group, computes:
        total = col1 + col2 + ... + colN + epsilon
        hhi   = (col1/total)² + (col2/total)² + ... + (colN/total)²
    """
    nodes: list = []
    initializers: list = []
    source_cols = {c for group in transformer.column_groups for c in group}

    for col, in_name in input_names.items():
        if col not in source_cols and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
        elif col in source_cols and not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for group, hhi_col in zip(transformer.column_groups, transformer.new_column_names or []):
        if not all(c in input_names for c in group):  # pragma: no cover
            continue
        out_name = output_names.get(hhi_col, hhi_col)
        p = f"HHI__{out_name}"  # unique per output — groups may share identical column sets
        eps_init = f"{p}__eps"
        initializers.append(_make_init(eps_init, float(transformer.epsilon), get_input_onnx_type(transformer, group[0])))

        # Sum all group columns
        running = input_names[group[0]]
        for i, col in enumerate(group[1:]):
            add_out = f"{p}__gsum{i}"
            nodes.append(oh.make_node("Add", inputs=[running, input_names[col]], outputs=[add_out]))
            running = add_out

        # Add epsilon → total
        total_out = f"{p}__total"
        nodes.append(oh.make_node("Add", inputs=[running, eps_init], outputs=[total_out]))

        # Compute (col_i / total)² for each column in group
        sq_terms: list[str] = []
        for i, col in enumerate(group):
            share_out = f"{p}__share{i}"
            sq_out = f"{p}__sq{i}"
            nodes.append(oh.make_node("Div", inputs=[input_names[col], total_out], outputs=[share_out]))
            nodes.append(oh.make_node("Mul", inputs=[share_out, share_out], outputs=[sq_out]))
            sq_terms.append(sq_out)

        # Sum the squared shares → HHI output
        if len(sq_terms) == 1:  # pragma: no cover  (column_groups requires ≥2 cols)
            nodes.append(oh.make_node("Identity", inputs=[sq_terms[0]], outputs=[out_name]))
        else:
            running_hhi = sq_terms[0]
            for i, sq in enumerate(sq_terms[1:]):
                is_last = i == len(sq_terms) - 2
                add_out = out_name if is_last else f"{p}__hhi{i}"
                nodes.append(oh.make_node("Add", inputs=[running_hhi, sq], outputs=[add_out]))
                running_hhi = add_out

    return nodes, initializers


# ── shared helpers for row-wise aggregation ops ───────────────────────────────

def _add_chain(names: list[str], prefix: str, suffix: str, final_out: str, nodes: list) -> None:
    """Chain Add nodes over *names*; last node writes to *final_out*."""
    if len(names) == 1:
        nodes.append(oh.make_node("Identity", inputs=[names[0]], outputs=[final_out]))
        return
    running = names[0]
    for i, nm in enumerate(names[1:]):
        is_last = i == len(names) - 2
        out = final_out if is_last else f"{prefix}__{suffix}{i}"
        nodes.append(oh.make_node("Add", inputs=[running, nm], outputs=[out]))
        running = out


def _mk_zero(prefix: str, onnx_type: int, initializers: list) -> str:
    """Add a scalar 0.0 initializer and return its name."""
    name = f"{prefix}__zero"
    initializers.append(_make_init(name, 0.0, onnx_type))
    return name


def _cos_node(inp: str, out: str, onnx_type: int, prefix: str, nodes: list) -> None:
    """Emit Cos, routing through float32 for double inputs (onnxruntime limitation)."""
    if onnx_type == TensorProto.DOUBLE:
        f32_in = f"{prefix}__cos_f32in"
        f32_out = f"{prefix}__cos_f32out"
        nodes.append(oh.make_node("Cast", inputs=[inp], outputs=[f32_in], to=TensorProto.FLOAT))
        nodes.append(oh.make_node("Cos", inputs=[f32_in], outputs=[f32_out]))
        nodes.append(oh.make_node("Cast", inputs=[f32_out], outputs=[out], to=TensorProto.DOUBLE))
    else:
        nodes.append(oh.make_node("Cos", inputs=[inp], outputs=[out]))


def _asin_node(inp: str, out: str, onnx_type: int, prefix: str, nodes: list) -> None:
    """Emit Asin, routing through float32 for double inputs."""
    if onnx_type == TensorProto.DOUBLE:
        f32_in = f"{prefix}__asin_f32in"
        f32_out = f"{prefix}__asin_f32out"
        nodes.append(oh.make_node("Cast", inputs=[inp], outputs=[f32_in], to=TensorProto.FLOAT))
        nodes.append(oh.make_node("Asin", inputs=[f32_in], outputs=[f32_out]))
        nodes.append(oh.make_node("Cast", inputs=[f32_out], outputs=[out], to=TensorProto.DOUBLE))
    else:
        nodes.append(oh.make_node("Asin", inputs=[inp], outputs=[out]))


def _emit_row_agg(
    op: str,
    group: list[str],
    input_names: dict[str, str],
    out_name: str,
    prefix: str,
    onnx_type: int,
    nodes: list,
    initializers: list,
) -> None:
    """Emit ONNX nodes for one row-wise aggregation (sum/mean/min/max/range/std/…)."""
    n = len(group)
    ins = [input_names[c] for c in group]

    if op == "sum":
        _add_chain(ins, prefix, "s", out_name, nodes)

    elif op == "mean":
        if n == 1:
            nodes.append(oh.make_node("Identity", inputs=[ins[0]], outputs=[out_name]))
        else:
            sum_out = f"{prefix}__msum"
            _add_chain(ins, prefix, "ms", sum_out, nodes)
            n_init = f"{prefix}__mn"
            initializers.append(_make_init(n_init, float(n), onnx_type))
            nodes.append(oh.make_node("Div", inputs=[sum_out, n_init], outputs=[out_name]))

    elif op == "min":
        if n == 1:
            nodes.append(oh.make_node("Identity", inputs=[ins[0]], outputs=[out_name]))
        else:
            nodes.append(oh.make_node("Min", inputs=ins, outputs=[out_name]))

    elif op == "max":
        if n == 1:
            nodes.append(oh.make_node("Identity", inputs=[ins[0]], outputs=[out_name]))
        else:
            nodes.append(oh.make_node("Max", inputs=ins, outputs=[out_name]))

    elif op == "range":
        if n == 1:
            nodes.append(oh.make_node("Mul", inputs=[ins[0], _mk_zero(prefix + "__rng", onnx_type, initializers)], outputs=[out_name]))
        else:
            max_out, min_out = f"{prefix}__rng_max", f"{prefix}__rng_min"
            nodes.append(oh.make_node("Max", inputs=ins, outputs=[max_out]))
            nodes.append(oh.make_node("Min", inputs=ins, outputs=[min_out]))
            nodes.append(oh.make_node("Sub", inputs=[max_out, min_out], outputs=[out_name]))

    elif op == "minus":
        if n == 1:
            nodes.append(oh.make_node("Identity", inputs=[ins[0]], outputs=[out_name]))
        else:
            running = ins[0]
            for i, nm in enumerate(ins[1:]):
                is_last = i == n - 2
                curr = out_name if is_last else f"{prefix}__sub{i}"
                nodes.append(oh.make_node("Sub", inputs=[running, nm], outputs=[curr]))
                running = curr

    elif op == "mul":
        if n == 1:
            nodes.append(oh.make_node("Identity", inputs=[ins[0]], outputs=[out_name]))
        else:
            running = ins[0]
            for i, nm in enumerate(ins[1:]):
                is_last = i == n - 2
                curr = out_name if is_last else f"{prefix}__mul{i}"
                nodes.append(oh.make_node("Mul", inputs=[running, nm], outputs=[curr]))
                running = curr

    elif op == "div":
        if n == 1:
            nodes.append(oh.make_node("Identity", inputs=[ins[0]], outputs=[out_name]))
        else:
            running = ins[0]
            for i, nm in enumerate(ins[1:]):
                is_last = i == n - 2
                curr = out_name if is_last else f"{prefix}__div{i}"
                nodes.append(oh.make_node("Div", inputs=[running, nm], outputs=[curr]))
                running = curr

    elif op == "abs_diff":
        if n == 1:
            nodes.append(oh.make_node("Identity", inputs=[ins[0]], outputs=[out_name]))
        else:
            running = ins[0]
            for i, nm in enumerate(ins[1:]):
                is_last = i == n - 2
                sub_out = f"{prefix}__ad_sub{i}"
                curr = out_name if is_last else f"{prefix}__ad{i}"
                nodes.append(oh.make_node("Sub", inputs=[running, nm], outputs=[sub_out]))
                nodes.append(oh.make_node("Abs", inputs=[sub_out], outputs=[curr]))
                running = curr

    elif op in ("std", "var"):
        if n == 1:
            nodes.append(oh.make_node("Mul", inputs=[ins[0], _mk_zero(prefix + "__sv", onnx_type, initializers)], outputs=[out_name]))
        else:
            mean_sum = f"{prefix}__sv_msum"
            _add_chain(ins, prefix, "sv_ms", mean_sum, nodes)
            n_init = f"{prefix}__sv_n"
            mean_out = f"{prefix}__sv_mean"
            initializers.append(_make_init(n_init, float(n), onnx_type))
            nodes.append(oh.make_node("Div", inputs=[mean_sum, n_init], outputs=[mean_out]))
            sq_outs: list[str] = []
            for i, nm in enumerate(ins):
                diff = f"{prefix}__sv_diff{i}"
                sq = f"{prefix}__sv_sq{i}"
                nodes.append(oh.make_node("Sub", inputs=[nm, mean_out], outputs=[diff]))
                nodes.append(oh.make_node("Mul", inputs=[diff, diff], outputs=[sq]))
                sq_outs.append(sq)
            sq_sum = f"{prefix}__sv_sqsum"
            _add_chain(sq_outs, prefix, "sv_sqadd", sq_sum, nodes)
            n1_init = f"{prefix}__sv_n1"
            var_out = f"{prefix}__sv_var"
            initializers.append(_make_init(n1_init, float(n - 1), onnx_type))
            nodes.append(oh.make_node("Div", inputs=[sq_sum, n1_init], outputs=[var_out]))
            if op == "var":
                nodes.append(oh.make_node("Identity", inputs=[var_out], outputs=[out_name]))
            else:
                nodes.append(oh.make_node("Sqrt", inputs=[var_out], outputs=[out_name]))

    elif op == "count_null":
        null_casts: list[str] = []
        for i, nm in enumerate(ins):
            isnan = f"{prefix}__cnu_isnan{i}"
            cast = f"{prefix}__cnu_cast{i}"
            nodes.append(oh.make_node("IsNaN", inputs=[nm], outputs=[isnan]))
            nodes.append(oh.make_node("Cast", inputs=[isnan], outputs=[cast], to=onnx_type))
            null_casts.append(cast)
        _add_chain(null_casts, prefix, "cnu_add", out_name, nodes)

    elif op == "count_zero":
        zero_init = _mk_zero(prefix + "__czr", onnx_type, initializers)
        zero_casts: list[str] = []
        for i, nm in enumerate(ins):
            eq = f"{prefix}__czr_eq{i}"
            cast = f"{prefix}__czr_cast{i}"
            nodes.append(oh.make_node("Equal", inputs=[nm, zero_init], outputs=[eq]))
            nodes.append(oh.make_node("Cast", inputs=[eq], outputs=[cast], to=onnx_type))
            zero_casts.append(cast)
        _add_chain(zero_casts, prefix, "czr_add", out_name, nodes)

    elif op == "count_nonzero":
        zero_init = _mk_zero(prefix + "__cnz", onnx_type, initializers)
        nz_casts: list[str] = []
        for i, nm in enumerate(ins):
            eq = f"{prefix}__cnz_eq{i}"
            not_ = f"{prefix}__cnz_not{i}"
            cast = f"{prefix}__cnz_cast{i}"
            nodes.append(oh.make_node("Equal", inputs=[nm, zero_init], outputs=[eq]))
            nodes.append(oh.make_node("Not", inputs=[eq], outputs=[not_]))
            nodes.append(oh.make_node("Cast", inputs=[not_], outputs=[cast], to=onnx_type))
            nz_casts.append(cast)
        _add_chain(nz_casts, prefix, "cnz_add", out_name, nodes)

    elif op == "count":
        # count non-null values per row (RowStatisticsFeatures)
        nn_casts: list[str] = []
        for i, nm in enumerate(ins):
            isnan = f"{prefix}__cnt_isnan{i}"
            not_ = f"{prefix}__cnt_not{i}"
            cast = f"{prefix}__cnt_cast{i}"
            nodes.append(oh.make_node("IsNaN", inputs=[nm], outputs=[isnan]))
            nodes.append(oh.make_node("Not", inputs=[isnan], outputs=[not_]))
            nodes.append(oh.make_node("Cast", inputs=[not_], outputs=[cast], to=onnx_type))
            nn_casts.append(cast)
        _add_chain(nn_casts, prefix, "cnt_add", out_name, nodes)

    elif op == "median":
        raise OnnxNotSupportedError("op='median' is not supported in ONNX (no standard median op).")


# ── MathFeatures ──────────────────────────────────────────────────────────────

@get_output_columns.register(MathFeatures)
def _math_output_cols(transformer: MathFeatures, input_columns: list[str]) -> list[str]:
    new = []
    for group in transformer.groups:
        group_key = "_".join(group)
        mapped = transformer._column_mapping.get(group_key, group_key)
        for op in transformer.func:
            new.append(f"{mapped}_{op}")
    if transformer.drop_columns:
        dropped = {c for group in transformer.groups for c in group}
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@get_output_onnx_type.register(MathFeatures)
def _math_output_onnx_type(transformer: MathFeatures, col: str) -> int:
    # Report the widest input type for generated columns so pipeline type-tracking stays accurate.
    for group in transformer.groups:
        group_key = "_".join(group)
        mapped = transformer._column_mapping.get(group_key, group_key)
        for op in transformer.func:
            if f"{mapped}_{op}" == col:
                return (
                    TensorProto.DOUBLE
                    if any(get_input_onnx_type(transformer, c) == TensorProto.DOUBLE for c in group)
                    else TensorProto.FLOAT
                )
    return get_input_onnx_type(transformer, col)


@to_onnx_nodes.register(MathFeatures)
def _math_to_onnx(
    transformer: MathFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Row-wise math operations (sum/mean/min/max/range/…) across static column groups."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    for group in transformer.groups:
        if not all(c in input_names for c in group):  # pragma: no cover
            continue
        group_key = "_".join(group)
        mapped = transformer._column_mapping.get(group_key, group_key)
        # Use widest type across the group; DOUBLE wins over FLOAT.
        onnx_type = (
            TensorProto.DOUBLE
            if any(get_input_onnx_type(transformer, c) == TensorProto.DOUBLE for c in group)
            else TensorProto.FLOAT
        )
        # Cast any mismatched group inputs to onnx_type so binary ops see a uniform type.
        group_input_names = dict(input_names)
        for col in group:
            if get_input_onnx_type(transformer, col) != onnx_type:
                cast_out = f"Math__cast__{col}__{group_key}"
                nodes.append(oh.make_node("Cast", inputs=[input_names[col]], outputs=[cast_out], to=onnx_type))
                group_input_names[col] = cast_out
        for op in transformer.func:
            out_col = f"{mapped}_{op}"
            out_name = output_names.get(out_col, out_col)
            prefix = f"Math__{'_'.join(group)}__{op}"
            _emit_row_agg(op, group, group_input_names, out_name, prefix, onnx_type, nodes, initializers)

    return nodes, initializers


# ── RowStatisticsFeatures ─────────────────────────────────────────────────────

@get_output_columns.register(RowStatisticsFeatures)
def _rowstats_output_cols(transformer: RowStatisticsFeatures, input_columns: list[str]) -> list[str]:
    new = list(transformer.new_column_names or [])
    if transformer.drop_columns:
        dropped = {c for cols in transformer.column_groups.values() for c in cols}
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(RowStatisticsFeatures)
def _rowstats_to_onnx(
    transformer: RowStatisticsFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Row-level statistics (sum/mean/min/max/range/std/count) across column groups."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    col_map = transformer._column_mapping
    for group_name, cols in transformer.column_groups.items():
        if not all(c in input_names for c in cols):  # pragma: no cover
            continue
        onnx_type = get_input_onnx_type(transformer, cols[0])
        for f in transformer.func:
            default_col = f"{group_name}__{f}"
            out_col = col_map.get(default_col, default_col)
            out_name = output_names.get(out_col, out_col)
            prefix = f"RowStats__{group_name}__{f}"
            _emit_row_agg(f, cols, input_names, out_name, prefix, onnx_type, nodes, initializers)

    return nodes, initializers


# ── ConcentrationIndexFeatures ────────────────────────────────────────────────

@get_output_columns.register(ConcentrationIndexFeatures)
def _conc_output_cols(transformer: ConcentrationIndexFeatures, input_columns: list[str]) -> list[str]:
    new = list(transformer.new_column_names or [])
    if transformer.drop_columns:
        dropped = {
            col
            for num, denoms in zip(transformer.numerator_columns, transformer.denominator_columns)
            for col in [num] + denoms
        }
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(ConcentrationIndexFeatures)
def _conc_to_onnx(
    transformer: ConcentrationIndexFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """num / (denom₁ + denom₂ + … [+ 1]) with optional Laplace smoothing."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    for num_col, denom_cols, out_col in zip(
        transformer.numerator_columns,
        transformer.denominator_columns,
        transformer.new_column_names or [],
    ):
        if num_col not in input_names or not all(d in input_names for d in denom_cols):  # pragma: no cover
            continue
        num_in = input_names[num_col]
        out_name = output_names.get(out_col, out_col)
        p = f"Conc__{num_col}__{'_'.join(denom_cols)}"
        onnx_type = get_input_onnx_type(transformer, num_col)

        denom_sum = f"{p}__dsum"
        _add_chain([input_names[d] for d in denom_cols], p, "ds", denom_sum, nodes)

        if transformer.smoothing:
            one_init = f"{p}__one"
            denom_total = f"{p}__dtot"
            initializers.append(_make_init(one_init, 1.0, onnx_type))
            nodes.append(oh.make_node("Add", inputs=[denom_sum, one_init], outputs=[denom_total]))
        else:
            denom_total = denom_sum

        nodes.append(oh.make_node("Div", inputs=[num_in, denom_total], outputs=[out_name]))

    return nodes, initializers


# ── GeneralizedRatioFeatures ──────────────────────────────────────────────────

@get_output_columns.register(GeneralizedRatioFeatures)
def _gratio_output_cols(transformer: GeneralizedRatioFeatures, input_columns: list[str]) -> list[str]:
    new = list(transformer.new_column_names or [])
    if transformer.drop_columns:
        dropped = {
            col
            for groups in (transformer.numerator_columns, transformer.denominator_columns)
            for group in groups
            for col in group
        }
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(GeneralizedRatioFeatures)
def _gratio_to_onnx(
    transformer: GeneralizedRatioFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """(bias + Σ aᵢ·xᵢ) / (Σ bⱼ·yⱼ + ε) — chained Mul + Add + Div nodes."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    for num_cols, denom_cols, num_coeffs, denom_coeffs, bias, out_col in zip(
        transformer.numerator_columns,
        transformer.denominator_columns,
        transformer.numerator_coefficients,
        transformer.denominator_coefficients,
        transformer.numerator_biases,
        transformer.new_column_names or [],
    ):
        if not all(c in input_names for c in denom_cols):  # pragma: no cover
            continue
        out_name = output_names.get(out_col, out_col)
        onnx_type = get_input_onnx_type(transformer, denom_cols[0])
        p = f"GRatio__{out_col}"

        # ── numerator ──
        if num_cols:
            if not all(c in input_names for c in num_cols):  # pragma: no cover
                continue
            num_terms: list[str] = []
            for i, (col, coeff) in enumerate(zip(num_cols, num_coeffs)):
                if coeff == 1.0:
                    num_terms.append(input_names[col])
                else:
                    c_init = f"{p}__nc{i}"
                    term = f"{p}__nt{i}"
                    initializers.append(_make_init(c_init, float(coeff), onnx_type))
                    nodes.append(oh.make_node("Mul", inputs=[input_names[col], c_init], outputs=[term]))
                    num_terms.append(term)
            num_sum = f"{p}__nsum"
            _add_chain(num_terms, p, "ns", num_sum, nodes)
            if bias != 0.0:
                bias_init = f"{p}__bias"
                num_total = f"{p}__ntot"
                initializers.append(_make_init(bias_init, float(bias), onnx_type))
                nodes.append(oh.make_node("Add", inputs=[num_sum, bias_init], outputs=[num_total]))
            else:
                num_total = num_sum
        else:
            # constant numerator: bias tensor shaped like input via col*0 + bias
            zero_init = f"{p}__czero"
            bias_init = f"{p}__cbias"
            zero_mul = f"{p}__czero_mul"
            num_total = f"{p}__ntot"
            initializers.append(_make_init(zero_init, 0.0, onnx_type))
            initializers.append(_make_init(bias_init, float(bias), onnx_type))
            nodes.append(oh.make_node("Mul", inputs=[input_names[denom_cols[0]], zero_init], outputs=[zero_mul]))
            nodes.append(oh.make_node("Add", inputs=[zero_mul, bias_init], outputs=[num_total]))

        # ── denominator ──
        denom_terms: list[str] = []
        for i, (col, coeff) in enumerate(zip(denom_cols, denom_coeffs)):
            if coeff == 1.0:
                denom_terms.append(input_names[col])
            else:
                c_init = f"{p}__dc{i}"
                term = f"{p}__dt{i}"
                initializers.append(_make_init(c_init, float(coeff), onnx_type))
                nodes.append(oh.make_node("Mul", inputs=[input_names[col], c_init], outputs=[term]))
                denom_terms.append(term)
        denom_sum = f"{p}__dsum"
        _add_chain(denom_terms, p, "ds", denom_sum, nodes)
        eps_init = f"{p}__eps"
        denom_total = f"{p}__dtot"
        initializers.append(_make_init(eps_init, float(transformer.epsilon), onnx_type))
        nodes.append(oh.make_node("Add", inputs=[denom_sum, eps_init], outputs=[denom_total]))

        nodes.append(oh.make_node("Div", inputs=[num_total, denom_total], outputs=[out_name]))

    return nodes, initializers


# ── PlanRotationFeatures ──────────────────────────────────────────────────────

@get_output_columns.register(PlanRotationFeatures)
def _planrot_output_cols(transformer: PlanRotationFeatures, input_columns: list[str]) -> list[str]:
    return list(input_columns) + list(transformer.column_names)


@to_onnx_nodes.register(PlanRotationFeatures)
def _planrot_to_onnx(
    transformer: PlanRotationFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """x·cos(θ) − y·sin(θ)  /  x·sin(θ) + y·cos(θ) with baked-in scalar constants."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    col_name_idx = 0
    for x_col, y_col in transformer.columns:
        if x_col not in input_names or y_col not in input_names:  # pragma: no cover
            col_name_idx += 2 * len(transformer.angles)
            continue
        x_in = input_names[x_col]
        y_in = input_names[y_col]
        onnx_type = get_input_onnx_type(transformer, x_col)

        for theta in transformer.angles:
            cos_theta = math.cos(theta * math.pi / 180.0)
            sin_theta = math.sin(theta * math.pi / 180.0)
            x_out_col = transformer.column_names[col_name_idx]
            y_out_col = transformer.column_names[col_name_idx + 1]
            col_name_idx += 2

            x_out = output_names.get(x_out_col, x_out_col)
            y_out = output_names.get(y_out_col, y_out_col)
            p = f"Rot__{x_col}{y_col}_{theta}"

            cos_init = f"{p}__cos"
            sin_init = f"{p}__sin"
            initializers.append(_make_init(cos_init, cos_theta, onnx_type))
            initializers.append(_make_init(sin_init, sin_theta, onnx_type))

            # x_rot = x*cos - y*sin
            xc, ys = f"{p}__xc", f"{p}__ys"
            nodes.append(oh.make_node("Mul", inputs=[x_in, cos_init], outputs=[xc]))
            nodes.append(oh.make_node("Mul", inputs=[y_in, sin_init], outputs=[ys]))
            nodes.append(oh.make_node("Sub", inputs=[xc, ys], outputs=[x_out]))

            # y_rot = x*sin + y*cos
            xs, yc = f"{p}__xs", f"{p}__yc"
            nodes.append(oh.make_node("Mul", inputs=[x_in, sin_init], outputs=[xs]))
            nodes.append(oh.make_node("Mul", inputs=[y_in, cos_init], outputs=[yc]))
            nodes.append(oh.make_node("Add", inputs=[xs, yc], outputs=[y_out]))

    return nodes, initializers


# ── DistanceFeatures ──────────────────────────────────────────────────────────

@get_output_columns.register(DistanceFeatures)
def _dist_output_cols(transformer: DistanceFeatures, input_columns: list[str]) -> list[str]:
    new = list(transformer.new_column_names or [])
    if transformer.drop_columns:
        dropped = set(transformer.lats + transformer.longs)
        return [c for c in input_columns if c not in dropped] + new
    return list(input_columns) + new


@to_onnx_nodes.register(DistanceFeatures)
def _dist_to_onnx(
    transformer: DistanceFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Euclidean / Manhattan / Haversine distance between consecutive lat-long pairs."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    for i in range(len(transformer.lats) - 1):
        lat1_col, lat2_col = transformer.lats[i], transformer.lats[i + 1]
        long1_col, long2_col = transformer.longs[i], transformer.longs[i + 1]

        if not all(c in input_names for c in (lat1_col, lat2_col, long1_col, long2_col)):  # pragma: no cover
            continue

        default_name = f"distance__{lat1_col}_to_{lat2_col}__{transformer.method}_{transformer.unit}"
        out_col = (transformer._column_mapping or {}).get(default_name, default_name)
        out_name = output_names.get(out_col, out_col)
        p = f"Dist__{lat1_col}_{lat2_col}"
        onnx_type = get_input_onnx_type(transformer, lat1_col)

        lat1 = input_names[lat1_col]
        lat2 = input_names[lat2_col]
        long1 = input_names[long1_col]
        long2 = input_names[long2_col]

        if transformer.method == "euclidean":
            dlat, dlong = f"{p}__dlat", f"{p}__dlong"
            dlat_sq, dlong_sq, sum_sq = f"{p}__dlat_sq", f"{p}__dlong_sq", f"{p}__sum_sq"
            nodes.append(oh.make_node("Sub", inputs=[lat2, lat1], outputs=[dlat]))
            nodes.append(oh.make_node("Sub", inputs=[long2, long1], outputs=[dlong]))
            nodes.append(oh.make_node("Mul", inputs=[dlat, dlat], outputs=[dlat_sq]))
            nodes.append(oh.make_node("Mul", inputs=[dlong, dlong], outputs=[dlong_sq]))
            nodes.append(oh.make_node("Add", inputs=[dlat_sq, dlong_sq], outputs=[sum_sq]))
            nodes.append(oh.make_node("Sqrt", inputs=[sum_sq], outputs=[out_name]))

        elif transformer.method == "manhattan":
            dlat, dlong = f"{p}__dlat", f"{p}__dlong"
            abs_dlat, abs_dlong = f"{p}__abs_dlat", f"{p}__abs_dlong"
            nodes.append(oh.make_node("Sub", inputs=[lat2, lat1], outputs=[dlat]))
            nodes.append(oh.make_node("Sub", inputs=[long2, long1], outputs=[dlong]))
            nodes.append(oh.make_node("Abs", inputs=[dlat], outputs=[abs_dlat]))
            nodes.append(oh.make_node("Abs", inputs=[dlong], outputs=[abs_dlong]))
            nodes.append(oh.make_node("Add", inputs=[abs_dlat, abs_dlong], outputs=[out_name]))

        elif transformer.method == "haversine":
            deg2rad = math.pi / 180.0
            d2r, half, two, r_init = f"{p}__d2r", f"{p}__half", f"{p}__two", f"{p}__r"
            initializers.append(_make_init(d2r, deg2rad, onnx_type))
            initializers.append(_make_init(half, 0.5, onnx_type))
            initializers.append(_make_init(two, 2.0, onnx_type))
            initializers.append(_make_init(r_init, float(EARTH_RADIUS[transformer.unit]), onnx_type))

            lat1_rad, lat2_rad = f"{p}__lat1r", f"{p}__lat2r"
            long1_rad, long2_rad = f"{p}__long1r", f"{p}__long2r"
            nodes.append(oh.make_node("Mul", inputs=[lat1, d2r], outputs=[lat1_rad]))
            nodes.append(oh.make_node("Mul", inputs=[lat2, d2r], outputs=[lat2_rad]))
            nodes.append(oh.make_node("Mul", inputs=[long1, d2r], outputs=[long1_rad]))
            nodes.append(oh.make_node("Mul", inputs=[long2, d2r], outputs=[long2_rad]))

            dlat, dlong = f"{p}__dlat", f"{p}__dlong"
            hdlat, hdlong = f"{p}__hdlat", f"{p}__hdlong"
            nodes.append(oh.make_node("Sub", inputs=[lat2_rad, lat1_rad], outputs=[dlat]))
            nodes.append(oh.make_node("Sub", inputs=[long2_rad, long1_rad], outputs=[dlong]))
            nodes.append(oh.make_node("Mul", inputs=[dlat, half], outputs=[hdlat]))
            nodes.append(oh.make_node("Mul", inputs=[dlong, half], outputs=[hdlong]))

            sin_hdlat, sin_hdlong = f"{p}__sin_hdlat", f"{p}__sin_hdlong"
            sq_shdlat, sq_shdlong = f"{p}__sq_shdlat", f"{p}__sq_shdlong"
            nodes.append(oh.make_node("Sin", inputs=[hdlat], outputs=[sin_hdlat]))
            nodes.append(oh.make_node("Sin", inputs=[hdlong], outputs=[sin_hdlong]))
            nodes.append(oh.make_node("Mul", inputs=[sin_hdlat, sin_hdlat], outputs=[sq_shdlat]))
            nodes.append(oh.make_node("Mul", inputs=[sin_hdlong, sin_hdlong], outputs=[sq_shdlong]))

            cos_lat1, cos_lat2 = f"{p}__cos_lat1", f"{p}__cos_lat2"
            _cos_node(lat1_rad, cos_lat1, onnx_type, f"{p}__cl1", nodes)
            _cos_node(lat2_rad, cos_lat2, onnx_type, f"{p}__cl2", nodes)

            cos_prod, a_t2, a = f"{p}__cos_prod", f"{p}__at2", f"{p}__a"
            sqrt_a, asin_out, c = f"{p}__sqrt_a", f"{p}__asin", f"{p}__c"
            nodes.append(oh.make_node("Mul", inputs=[cos_lat1, cos_lat2], outputs=[cos_prod]))
            nodes.append(oh.make_node("Mul", inputs=[cos_prod, sq_shdlong], outputs=[a_t2]))
            nodes.append(oh.make_node("Add", inputs=[sq_shdlat, a_t2], outputs=[a]))
            nodes.append(oh.make_node("Sqrt", inputs=[a], outputs=[sqrt_a]))
            _asin_node(sqrt_a, asin_out, onnx_type, p, nodes)
            nodes.append(oh.make_node("Mul", inputs=[two, asin_out], outputs=[c]))
            nodes.append(oh.make_node("Mul", inputs=[c, r_init], outputs=[out_name]))

    return nodes, initializers


# ── RuleFeatures ──────────────────────────────────────────────────────────────

@get_output_columns.register(RuleFeatures)
def _rule_output_cols(transformer: RuleFeatures, input_columns: list[str]) -> list[str]:
    return list(input_columns) + list(transformer.new_column_names)


@to_onnx_nodes.register(RuleFeatures)
def _rule_to_onnx(
    transformer: RuleFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """AND/OR of per-condition boolean comparisons → Cast(FLOAT) per rule."""
    nodes = _passthrough(input_names, output_names)
    initializers: list[onnx.TensorProto] = []

    logic_op = "And" if transformer.rule_logic == "and" else "Or"

    for rule_idx, (rule, out_col) in enumerate(zip(transformer.rules, transformer.new_column_names)):
        first_col = rule[0]["column"]
        if first_col not in input_names:  # pragma: no cover
            continue
        onnx_type = get_input_onnx_type(transformer, first_col)
        out_name = output_names.get(out_col, out_col)
        p = f"Rule{rule_idx}__{out_col}"

        cond_bools: list[str] = []
        skip_rule = False

        for cond_idx, cond in enumerate(rule):
            col = cond["column"]
            op = cond["op"]
            if col not in input_names:  # pragma: no cover
                skip_rule = True
                break
            in_name = input_names[col]
            cp = f"{p}__c{cond_idx}"
            bool_out = f"{cp}__bool"

            if "other_column" in cond:
                other_col = cond["other_column"]
                if other_col not in input_names:  # pragma: no cover
                    skip_rule = True
                    break
                in_b = input_names[other_col]
                if op == "!=":
                    eq_out = f"{cp}__eq"
                    nodes.append(oh.make_node("Equal", inputs=[in_name, in_b], outputs=[eq_out]))
                    nodes.append(oh.make_node("Not", inputs=[eq_out], outputs=[bool_out]))
                else:
                    nodes.append(oh.make_node(_CMP_ONNX_OPS[op], inputs=[in_name, in_b], outputs=[bool_out]))
            else:
                val = cond["value"]
                scalar_init = f"{cp}__scalar"
                # bool values: map True→1.0, False→0.0 for float comparison
                initializers.append(_make_init(scalar_init, 1.0 if val is True else 0.0 if val is False else float(val), onnx_type))
                if op == "!=":
                    eq_out = f"{cp}__eq"
                    nodes.append(oh.make_node("Equal", inputs=[in_name, scalar_init], outputs=[eq_out]))
                    nodes.append(oh.make_node("Not", inputs=[eq_out], outputs=[bool_out]))
                else:
                    nodes.append(oh.make_node(_CMP_ONNX_OPS[op], inputs=[in_name, scalar_init], outputs=[bool_out]))

            cond_bools.append(bool_out)

        if skip_rule:  # pragma: no cover
            continue

        # Combine booleans with AND or OR (binary op → chain for N > 2)
        if len(cond_bools) == 1:
            combined_bool = cond_bools[0]
        else:
            combined_bool = f"{p}__combined"
            running = cond_bools[0]
            for i, b in enumerate(cond_bools[1:]):
                is_last = i == len(cond_bools) - 2
                curr = combined_bool if is_last else f"{p}__logic{i}"
                nodes.append(oh.make_node(logic_op, inputs=[running, b], outputs=[curr]))
                running = curr

        nodes.append(_cast_numeric(combined_bool, out_name, onnx_type))

    return nodes, initializers


# ── GroupStatisticsFeatures ───────────────────────────────────────────────────

@get_output_columns.register(GroupStatisticsFeatures)
def _gsf_output_columns(transformer: GroupStatisticsFeatures, input_columns: list[str]) -> list[str]:
    new_cols = list(transformer._column_mapping.values())
    if transformer.drop_columns:
        dropped = set(transformer.subset)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


@get_input_onnx_type.register(GroupStatisticsFeatures)
def _gsf_input_onnx_type(transformer: GroupStatisticsFeatures, col: str) -> int:
    if col in transformer.by:
        return TensorProto.STRING
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(GroupStatisticsFeatures)
def _gsf_output_onnx_type(transformer: GroupStatisticsFeatures, col: str) -> int:
    if col in transformer._column_mapping.values():
        return TensorProto.FLOAT
    return _gsf_input_onnx_type(transformer, col)


@to_onnx_nodes.register(GroupStatisticsFeatures)
def _gsf_to_onnx_nodes(
    transformer: GroupStatisticsFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """LabelEncoder group-stat lookup + arithmetic for all 12 aggregation functions.

    Uses training-time group statistics stored in _group_stats.  Unknown groups
    at inference time fall back to 0.0 for absolute stats and to fill_value for
    relative stats (matching the zero-denominator behaviour of Polars transform).
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    generated = set(transformer._column_mapping.values())
    subset_set = set(transformer.subset or [])
    by_set = set(transformer.by)

    # Passthrough: columns not in subset, not in by, and not a generated feature column
    for col, in_name in input_names.items():
        if col not in subset_set and col not in by_set and col in output_names and col not in generated:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    # Passthrough for groupby (by) columns — emit once, outside the subset loop
    for groupby_col in transformer.by:
        if groupby_col in input_names and groupby_col in output_names:
            nodes.append(oh.make_node(
                "Identity", inputs=[input_names[groupby_col]], outputs=[output_names[groupby_col]]
            ))

    for num_col in transformer.subset:
        if num_col not in input_names:  # pragma: no cover
            continue
        num_in = input_names[num_col]
        onnx_type = get_input_onnx_type(transformer, num_col)
        np_dtype = _onnx_type_to_numpy(onnx_type)

        if not transformer.drop_columns and num_col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[num_in], outputs=[output_names[num_col]]))

        for groupby_col in transformer.by:
            if groupby_col not in input_names:  # pragma: no cover
                continue
            grp_in = input_names[groupby_col]
            stats = transformer._group_stats.get((num_col, groupby_col), {})

            for fun in transformer.func:
                default_name = f"{fun}_{num_col}__per_{groupby_col}"
                out_col = transformer._column_mapping[default_name]
                out_name = output_names.get(out_col, out_col)
                p = f"{num_in}__GSF__{groupby_col}__{fun}"

                def _lookup(stat_name: str, suffix: str = "") -> str:
                    stat_map = stats.get(stat_name, {})
                    f32 = f"{p}__{stat_name}{suffix}_f32"
                    nodes.append(oh.make_node(
                        "LabelEncoder", domain="ai.onnx.ml",
                        inputs=[grp_in], outputs=[f32],
                        keys_strings=list(stat_map.keys()),
                        values_floats=[float(v) for v in stat_map.values()],
                        default_float=0.0,
                    ))
                    if onnx_type != TensorProto.FLOAT:
                        casted = f"{p}__{stat_name}{suffix}"
                        nodes.append(oh.make_node("Cast", inputs=[f32], outputs=[casted], to=onnx_type))
                        return casted
                    return f32

                def _zero_init(name: str) -> str:
                    initializers.append(onnx.numpy_helper.from_array(np.array([0.0], dtype=np_dtype), name=name))
                    return name

                def _fill_init(name: str) -> str:
                    initializers.append(onnx.numpy_helper.from_array(np.array([transformer.fill_value], dtype=np_dtype), name=name))
                    return name

                def _guard(denom: str) -> str:
                    zero = _zero_init(f"{p}__zero")
                    eq_out, nan_out, or_out = f"{p}__eq", f"{p}__nan", f"{p}__guard"
                    nodes.append(oh.make_node("Equal", inputs=[denom, zero], outputs=[eq_out]))
                    nodes.append(oh.make_node("IsNaN", inputs=[denom], outputs=[nan_out]))
                    nodes.append(oh.make_node("Or", inputs=[eq_out, nan_out], outputs=[or_out]))
                    return or_out

                if fun in {"mean", "std", "median", "min", "max", "sum", "count"}:
                    nodes.append(oh.make_node("Identity", inputs=[_lookup(fun)], outputs=[out_name]))

                elif fun == "range":
                    mn, mx = _lookup("min", "_mn"), _lookup("max", "_mx")
                    nodes.append(oh.make_node("Sub", inputs=[mx, mn], outputs=[out_name]))

                elif fun in {"mean_ratio", "median_ratio"}:
                    stat = "mean" if fun == "mean_ratio" else "median"
                    denom = _lookup(stat)
                    fill = _fill_init(f"{p}__fill")
                    ratio = f"{p}__ratio"
                    nodes.append(oh.make_node("Div", inputs=[num_in, denom], outputs=[ratio]))
                    nodes.append(oh.make_node("Where", inputs=[_guard(denom), fill, ratio], outputs=[out_name]))

                elif fun == "zscore":
                    mean_v = _lookup("mean", "_m")
                    std_v = _lookup("std", "_s")
                    fill = _fill_init(f"{p}__fill")
                    sub_out, z_out = f"{p}__sub", f"{p}__z"
                    nodes.append(oh.make_node("Sub", inputs=[num_in, mean_v], outputs=[sub_out]))
                    nodes.append(oh.make_node("Div", inputs=[sub_out, std_v], outputs=[z_out]))
                    nodes.append(oh.make_node("Where", inputs=[_guard(std_v), fill, z_out], outputs=[out_name]))

                elif fun == "minmax":
                    mn = _lookup("min", "_mn")
                    mx = _lookup("max", "_mx")
                    fill = _fill_init(f"{p}__fill")
                    range_t, sub_t, mm_t = f"{p}__range", f"{p}__sub", f"{p}__mm"
                    zero = _zero_init(f"{p}__zero_mm")
                    eq_mm, nan_mm, or_mm = f"{p}__eq_mm", f"{p}__nan_mm", f"{p}__or_mm"
                    nodes.append(oh.make_node("Sub", inputs=[mx, mn], outputs=[range_t]))
                    nodes.append(oh.make_node("Equal", inputs=[range_t, zero], outputs=[eq_mm]))
                    nodes.append(oh.make_node("IsNaN", inputs=[range_t], outputs=[nan_mm]))
                    nodes.append(oh.make_node("Or", inputs=[eq_mm, nan_mm], outputs=[or_mm]))
                    nodes.append(oh.make_node("Sub", inputs=[num_in, mn], outputs=[sub_t]))
                    nodes.append(oh.make_node("Div", inputs=[sub_t, range_t], outputs=[mm_t]))
                    nodes.append(oh.make_node("Where", inputs=[or_mm, fill, mm_t], outputs=[out_name]))

    return nodes, initializers
