"""ONNX converters for gators.scalers transformers.

Importing this module registers all supported converters.

Supported
---------
StandardScaler         : (X - mean) / std
MinmaxScaler           : (X - min) * (1 / (max - min))
RobustScaler           : (X - median) * scale
Log1pScaler            : log1p(X)  —  base e, 2, or 10
ArcSinhScaler          : Asinh(X)
ArcSinSquareRootScaler : Asin(Sqrt(X))
PowerScaler            : Pow(X, power)
BoxCox                 : (X^λ - 1) / λ  (or Log(X) when λ=0)
YeoJohnson             : two-branch power transform; lambda fixed at fit time
"""
from __future__ import annotations

import math
from typing import Any

from ..scalers.arcsin_squareroot_scaler import ArcSinSquareRootScaler
from ..scalers.arcsinh_scaler import ArcSinhScaler
from ..scalers.box_cox import BoxCox
from ..scalers.log1p_scaler import Log1pScaler
from ..scalers.minmax_scaler import MinmaxScaler
from ..scalers.power_scaler import PowerScaler
from ..scalers.robust_scaler import RobustScaler
from ..scalers.standard_scaler import StandardScaler
from ..scalers.yeo_johnson import YeoJohnson
from ._converters import _onnx_type_to_numpy, col_out_name, get_input_onnx_type, to_onnx_nodes

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


@to_onnx_nodes.register(StandardScaler)
def _standard_scaler_to_onnx_nodes(
    transformer: StandardScaler,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """(X - mean) / std  as  Mul(Sub(X, mean), 1/std).

    mean and 1/std are stored during fit() in _offset and _scale respectively.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    targeted = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in targeted:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        prefix = f"{in_name}__StandardScaler"
        offset_init = f"{prefix}__offset"
        scale_init = f"{prefix}__scale"
        sub_out = f"{prefix}__sub"
        np_dtype = _onnx_type_to_numpy(get_input_onnx_type(transformer, col))
        initializers.append(
            onnx.numpy_helper.from_array(
                np.array([transformer._offset[col]], dtype=np_dtype), name=offset_init
            )
        )
        initializers.append(
            onnx.numpy_helper.from_array(
                np.array([transformer._scale[col]], dtype=np_dtype), name=scale_init
            )
        )
        nodes.append(oh.make_node("Sub", inputs=[in_name, offset_init], outputs=[sub_out]))
        nodes.append(oh.make_node("Mul", inputs=[sub_out, scale_init], outputs=[out_name]))

    return nodes, initializers


def _make_init(name: str, value: float, onnx_type: int) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(
        np.array([value], dtype=_onnx_type_to_numpy(onnx_type)), name=name
    )


def _col_passthrough_and_targeted(
    transformer: Any,
    input_names: dict[str, str],
    output_names: dict[str, str],
    targeted: set[str] | None = None,
) -> tuple[list[onnx.NodeProto], set[str]]:
    """Emit Identity nodes for non-targeted columns; return set of targeted column names."""
    if targeted is None:
        targeted = set(transformer.subset or [])
    nodes = [
        oh.make_node("Identity", inputs=[v], outputs=[output_names[k]])
        for k, v in input_names.items()
        if k not in targeted
    ]
    return nodes, targeted


# ── MinmaxScaler ──────────────────────────────────────────────────────────────

@to_onnx_nodes.register(MinmaxScaler)
def _minmax_scaler_to_onnx_nodes(
    transformer: MinmaxScaler,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """(X - min) * (1/(max-min))  as  Mul(Sub(X, offset), scale)."""
    nodes, targeted = _col_passthrough_and_targeted(transformer, input_names, output_names)
    initializers: list = []
    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        p = f"{in_name}__MinmaxScaler"
        offset_init, scale_init, sub_out = f"{p}__offset", f"{p}__scale", f"{p}__sub"
        onnx_type = get_input_onnx_type(transformer, col)
        initializers.append(_make_init(offset_init, transformer._offset[col], onnx_type))
        initializers.append(_make_init(scale_init, transformer._scale[col], onnx_type))
        nodes.append(oh.make_node("Sub", inputs=[in_name, offset_init], outputs=[sub_out]))
        nodes.append(oh.make_node("Mul", inputs=[sub_out, scale_init], outputs=[out_name]))
    return nodes, initializers


# ── RobustScaler ──────────────────────────────────────────────────────────────

@to_onnx_nodes.register(RobustScaler)
def _robust_scaler_to_onnx_nodes(
    transformer: RobustScaler,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """scale * (X - median)."""
    nodes, targeted = _col_passthrough_and_targeted(transformer, input_names, output_names)
    initializers: list = []
    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        p = f"{in_name}__RobustScaler"
        median_init, scale_init, sub_out = f"{p}__median", f"{p}__scale", f"{p}__sub"
        onnx_type = get_input_onnx_type(transformer, col)
        initializers.append(_make_init(median_init, transformer._median[col], onnx_type))
        initializers.append(_make_init(scale_init, transformer._scale[col], onnx_type))
        nodes.append(oh.make_node("Sub", inputs=[in_name, median_init], outputs=[sub_out]))
        nodes.append(oh.make_node("Mul", inputs=[sub_out, scale_init], outputs=[out_name]))
    return nodes, initializers


# ── Log1pScaler ───────────────────────────────────────────────────────────────

@to_onnx_nodes.register(Log1pScaler)
def _log1p_scaler_to_onnx_nodes(
    transformer: Log1pScaler,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """log(X+1) for base e; divided by log(2) or log(10) for other bases."""
    nodes, targeted = _col_passthrough_and_targeted(transformer, input_names, output_names)
    initializers: list = []
    base = transformer.base
    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        p = f"{in_name}__Log1pScaler"
        one_init, xp1_out, log_out = f"{p}__one", f"{p}__xp1", f"{p}__log"
        onnx_type = get_input_onnx_type(transformer, col)
        initializers.append(_make_init(one_init, 1.0, onnx_type))
        nodes.append(oh.make_node("Add", inputs=[in_name, one_init], outputs=[xp1_out]))
        if base == "e":
            nodes.append(oh.make_node("Log", inputs=[xp1_out], outputs=[out_name]))
        else:
            # log_b(x) = log(x) / log(b)
            divisor_val = math.log(2.0) if base == "2" else math.log(10.0)
            div_init = f"{p}__div"
            initializers.append(_make_init(div_init, divisor_val, onnx_type))
            nodes.append(oh.make_node("Log", inputs=[xp1_out], outputs=[log_out]))
            nodes.append(oh.make_node("Div", inputs=[log_out, div_init], outputs=[out_name]))
    return nodes, initializers


# ── ArcSinhScaler ─────────────────────────────────────────────────────────────

@to_onnx_nodes.register(ArcSinhScaler)
def _arcsinh_scaler_to_onnx_nodes(
    transformer: ArcSinhScaler,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """Asinh(X).  onnxruntime only supports Asinh for float32; Cast-bridges when input is float64."""
    nodes, _ = _col_passthrough_and_targeted(transformer, input_names, output_names)
    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        if get_input_onnx_type(transformer, col) == TensorProto.DOUBLE:
            f32_in = f"{in_name}__ArcSinh__f32"
            f32_out = f"{in_name}__ArcSinh__asinh_f32"
            nodes.append(oh.make_node("Cast", inputs=[in_name], outputs=[f32_in], to=TensorProto.FLOAT))
            nodes.append(oh.make_node("Asinh", inputs=[f32_in], outputs=[f32_out]))
            nodes.append(oh.make_node("Cast", inputs=[f32_out], outputs=[out_name], to=TensorProto.DOUBLE))
        else:
            nodes.append(oh.make_node("Asinh", inputs=[in_name], outputs=[out_name]))
    return nodes, []


# ── ArcSinSquareRootScaler ────────────────────────────────────────────────────

@to_onnx_nodes.register(ArcSinSquareRootScaler)
def _arcsin_sqrt_scaler_to_onnx_nodes(
    transformer: ArcSinSquareRootScaler,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """Asin(Sqrt(X)).  onnxruntime only supports Asin for float32; Cast-bridges when input is float64."""
    nodes, _ = _col_passthrough_and_targeted(transformer, input_names, output_names)
    initializers: list = []
    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        sqrt_out = f"{in_name}__ArcSinSqrt__sqrt"
        nodes.append(oh.make_node("Sqrt", inputs=[in_name], outputs=[sqrt_out]))
        if get_input_onnx_type(transformer, col) == TensorProto.DOUBLE:
            f32_in = f"{in_name}__ArcSinSqrt__f32"
            f32_out = f"{in_name}__ArcSinSqrt__asin_f32"
            nodes.append(oh.make_node("Cast", inputs=[sqrt_out], outputs=[f32_in], to=TensorProto.FLOAT))
            nodes.append(oh.make_node("Asin", inputs=[f32_in], outputs=[f32_out]))
            nodes.append(oh.make_node("Cast", inputs=[f32_out], outputs=[out_name], to=TensorProto.DOUBLE))
        else:
            nodes.append(oh.make_node("Asin", inputs=[sqrt_out], outputs=[out_name]))
    return nodes, initializers


# ── PowerScaler ───────────────────────────────────────────────────────────────

@to_onnx_nodes.register(PowerScaler)
def _power_scaler_to_onnx_nodes(
    transformer: PowerScaler,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """Pow(X, power)."""
    nodes, _ = _col_passthrough_and_targeted(transformer, input_names, output_names)
    initializers: list = []
    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        p = f"{in_name}__PowerScaler"
        pow_init = f"{p}__pow"
        initializers.append(_make_init(pow_init, float(transformer.power), get_input_onnx_type(transformer, col)))
        nodes.append(oh.make_node("Pow", inputs=[in_name, pow_init], outputs=[out_name]))
    return nodes, initializers


# ── BoxCox ────────────────────────────────────────────────────────────────────

@to_onnx_nodes.register(BoxCox)
def _boxcox_to_onnx_nodes(
    transformer: BoxCox,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """(X^λ - 1) / λ  (or Log(X) when λ≈0).  λ is fixed at fit time."""
    nodes, _ = _col_passthrough_and_targeted(transformer, input_names, output_names, targeted=set(transformer.lambdas))
    initializers: list = []
    for col in transformer.lambdas:
        if col not in input_names:  # pragma: no cover
            continue
        lam = transformer.lambdas[col]
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        p = f"{in_name}__BoxCox"
        onnx_type = get_input_onnx_type(transformer, col)
        if abs(lam) < 1e-8:
            # λ ≈ 0 → Log(X)
            nodes.append(oh.make_node("Log", inputs=[in_name], outputs=[out_name]))
        else:
            lam_init = f"{p}__lam"
            one_init = f"{p}__one"
            pow_out = f"{p}__pow"
            m1_out = f"{p}__m1"
            initializers.append(_make_init(lam_init, float(lam), onnx_type))
            initializers.append(_make_init(one_init, 1.0, onnx_type))
            nodes.append(oh.make_node("Pow", inputs=[in_name, lam_init], outputs=[pow_out]))
            nodes.append(oh.make_node("Sub", inputs=[pow_out, one_init], outputs=[m1_out]))
            nodes.append(oh.make_node("Div", inputs=[m1_out, lam_init], outputs=[out_name]))
    return nodes, initializers


# ── YeoJohnson ────────────────────────────────────────────────────────────────

@to_onnx_nodes.register(YeoJohnson)
def _yeo_johnson_to_onnx_nodes(
    transformer: YeoJohnson,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "coerce",
) -> tuple[list, list]:
    """Two-branch power transform; λ is fixed at fit time.

    Positive branch (X ≥ 0):
        λ≠0: ((X+1)^λ - 1) / λ
        λ=0: log(X+1)
    Negative branch (X < 0):
        λ≠2: -((-X+1)^(2-λ) - 1) / (2-λ)
        λ=2: -log(-X+1)
    """
    nodes, _ = _col_passthrough_and_targeted(transformer, input_names, output_names, targeted=set(transformer.lambdas))
    initializers: list = []
    for col in transformer.lambdas:
        if col not in input_names:  # pragma: no cover
            continue
        lam = transformer.lambdas[col]
        in_name = input_names[col]
        out_name = col_out_name(transformer, col, output_names)
        p = f"{in_name}__YeoJohnson"
        onnx_type = get_input_onnx_type(transformer, col)
        # Shared constants
        zero_init = f"{p}__zero"
        one_init = f"{p}__one"
        initializers.append(_make_init(zero_init, 0.0, onnx_type))
        initializers.append(_make_init(one_init, 1.0, onnx_type))
        ge0_out = f"{p}__ge0"   # bool mask: X >= 0
        nodes.append(oh.make_node("GreaterOrEqual", inputs=[in_name, zero_init], outputs=[ge0_out]))

        # ── Positive branch ───────────────────────────────────────────────────
        xp1_out = f"{p}__xp1"
        nodes.append(oh.make_node("Add", inputs=[in_name, one_init], outputs=[xp1_out]))
        if abs(lam) < 1e-8:
            pos_out = f"{p}__pos"
            nodes.append(oh.make_node("Log", inputs=[xp1_out], outputs=[pos_out]))
        else:
            lam_init = f"{p}__lam"
            initializers.append(_make_init(lam_init, float(lam), onnx_type))
            pow_p = f"{p}__powp"
            m1_p = f"{p}__m1p"
            pos_out = f"{p}__pos"
            nodes.append(oh.make_node("Pow", inputs=[xp1_out, lam_init], outputs=[pow_p]))
            nodes.append(oh.make_node("Sub", inputs=[pow_p, one_init], outputs=[m1_p]))
            nodes.append(oh.make_node("Div", inputs=[m1_p, lam_init], outputs=[pos_out]))

        # ── Negative branch ───────────────────────────────────────────────────
        neg_in = f"{p}__negin"    # -X
        neg1_out = f"{p}__neg1"   # -X + 1
        nodes.append(oh.make_node("Neg", inputs=[in_name], outputs=[neg_in]))
        nodes.append(oh.make_node("Add", inputs=[neg_in, one_init], outputs=[neg1_out]))
        two_m_lam = 2.0 - lam
        if abs(two_m_lam) < 1e-8:
            log_neg = f"{p}__logneg"
            neg_log = f"{p}__neglog"
            nodes.append(oh.make_node("Log", inputs=[neg1_out], outputs=[log_neg]))
            nodes.append(oh.make_node("Neg", inputs=[log_neg], outputs=[neg_log]))
            neg_out = neg_log
        else:
            tml_init = f"{p}__tml"
            initializers.append(_make_init(tml_init, float(two_m_lam), onnx_type))
            pow_n = f"{p}__pown"
            m1_n = f"{p}__m1n"
            div_n = f"{p}__divn"
            nodes.append(oh.make_node("Pow", inputs=[neg1_out, tml_init], outputs=[pow_n]))
            nodes.append(oh.make_node("Sub", inputs=[pow_n, one_init], outputs=[m1_n]))
            nodes.append(oh.make_node("Div", inputs=[m1_n, tml_init], outputs=[div_n]))
            neg_out_raw = f"{p}__negneg"
            nodes.append(oh.make_node("Neg", inputs=[div_n], outputs=[neg_out_raw]))
            neg_out = neg_out_raw

        # ── Merge branches ────────────────────────────────────────────────────
        nodes.append(oh.make_node("Where", inputs=[ge0_out, pos_out, neg_out], outputs=[out_name]))

    return nodes, initializers
