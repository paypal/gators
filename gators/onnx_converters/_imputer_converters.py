"""ONNX converters for gators.imputers transformers.

Importing this module registers all supported converters.

Supported
---------
NumericImputer    : Where(IsNaN, fill_stat, input) for stored strategies;
                    Identity pass-through for batch-computed strategies.
BooleanImputer    : Where(IsNaN, float(bool_stat), input)
GroupByImputer    : LabelEncoder(group_col → group_stat) + Where(IsNaN(col), fill, col)
IterativeImputer  : initial-stats fill + sequential linear regression
                    Exact for max_iter=1; approximated (single pass) for max_iter>1.
StringImputer     : Where(Equal(input, ""), fill_const_str, input)
                    Convention: Polars null strings must be fed as "" (empty string).

Not supported
-------------
KNNImputer       : requires nearest-neighbour search over training data.
"""
from __future__ import annotations

from typing import Any

from ..imputers.boolean_imputer import BooleanImputer
from ..imputers.groupby_imputer import GroupByImputer
from ..imputers.iterative_imputer import IterativeImputer
from ..imputers.numeric_imputer import NumericImputer
from ..imputers.string_imputer import StringImputer
from ._converters import (
    _onnx_type_to_numpy,
    get_input_onnx_type,
    get_output_columns,
    get_output_onnx_type,
    resolve_declared_output_dtype,
    to_onnx_nodes,
)
from ._exceptions import OnnxNotSupportedError

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


def _make_init(name: str, value: float, onnx_type: int) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(
        np.array([value], dtype=_onnx_type_to_numpy(onnx_type)), name=name
    )


# Strategies that compute fill values from the batch at transform time — not storable as ONNX initializers.
_IMPUTER_UNFIXED_STRATEGIES = frozenset({"forward", "backward"})


def _imputer_output_cols(transformer: Any, input_columns: list[str]) -> list[str]:
    if transformer.inplace or not transformer._column_mapping:
        return list(input_columns)
    new_cols = [name for names in transformer._column_mapping.values() for name in names]
    if transformer.drop_columns:
        dropped = set(transformer._column_mapping)
        return [c for c in input_columns if c not in dropped] + new_cols
    return list(input_columns) + new_cols


@get_output_columns.register(NumericImputer)
def _numeric_imputer_output_cols(transformer: NumericImputer, input_columns: list[str]) -> list[str]:
    return _imputer_output_cols(transformer, input_columns)


# ── NumericImputer ────────────────────────────────────────────────────────────

@to_onnx_nodes.register(NumericImputer)
def _numeric_imputer_to_onnx_nodes(
    transformer: NumericImputer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Supported strategies: 'constant', 'median', 'most_frequent', 'mean', 'min', 'max', 'zero', 'one'.

    Unsupported strategies ('forward', 'backward') depend on batch ordering and cannot be
    frozen into a static ONNX graph.  errors='coerce' emits Identity nodes; errors='raise'
    raises.  Missing values replaced via: Where(IsNaN(input), fill_constant, input).
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    if transformer.strategy in _IMPUTER_UNFIXED_STRATEGIES:
        if errors == "raise":
            raise OnnxNotSupportedError(
                f"NumericImputer strategy='{transformer.strategy}' computes fill values "
                f"from the input batch at transform time and stores nothing during fit(). "
                f"It cannot be frozen into a static ONNX graph. "
                f"Retrain with a strategy that stores statistics ('median', 'constant', "
                f"'most_frequent') or use errors='coerce' to emit Identity pass-through nodes."
            )
        for col in subset:
            if col in input_names and col in output_names:
                nodes.append(
                    oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]])
                )
        return nodes, initializers

    if transformer.strategy == "zero":
        col_fills = {col: 0.0 for col in subset if col in input_names}
    elif transformer.strategy == "one":
        col_fills = {col: 1.0 for col in subset if col in input_names}
    else:
        # constant / median / most_frequent: use _statistics (authoritative fitted values)
        col_fills = {
            col: float(transformer._statistics[col])
            for col in transformer._statistics
            if col in input_names
        }

    for col, fill_val in col_fills.items():
        in_name = input_names[col]
        if transformer.inplace:
            out_name = output_names.get(col, col)
        else:
            actual_out = transformer._column_mapping.get(col, [col])[0]
            if not transformer.drop_columns and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
            out_name = output_names.get(actual_out, actual_out)

        # ONNX IsNaN only accepts float tensor types; integer columns (e.g. Int64) carry no
        # null representation once exported, so imputation on them is a no-op pass-through.
        if get_input_onnx_type(transformer, col) not in (TensorProto.FLOAT, TensorProto.DOUBLE):
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[out_name]))
            continue

        prefix = f"{in_name}__NumericImputer"
        isnan_out = f"{prefix}__isnan"
        fill_init = f"{prefix}__fill"
        initializers.append(_make_init(fill_init, fill_val, get_input_onnx_type(transformer, col)))
        nodes.append(oh.make_node("IsNaN", inputs=[in_name], outputs=[isnan_out]))
        nodes.append(oh.make_node("Where", inputs=[isnan_out, fill_init, in_name], outputs=[out_name]))

    # Subset columns with no fitted statistic (e.g. all-null column) pass through unchanged
    for col in subset:
        if col not in col_fills and col in input_names and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))  # pragma: no cover

    return nodes, initializers


# ── BooleanImputer ────────────────────────────────────────────────────────────

# Boolean columns use float encoding (False=0.0, True=1.0, null=NaN) so IsNaN works.
@get_input_onnx_type.register(BooleanImputer)
def _boolean_imputer_input_type(transformer: BooleanImputer, col: str) -> int:
    return TensorProto.FLOAT


@get_output_onnx_type.register(BooleanImputer)
def _boolean_imputer_output_type(transformer: BooleanImputer, col: str) -> int:
    # Not using _output_dtypes here: ONNX represents booleans as FLOAT (0.0/1.0/NaN)
    # so IsNaN works, while _output_dtypes declares Boolean (true Polars dtype).
    return TensorProto.FLOAT


@to_onnx_nodes.register(BooleanImputer)
def _boolean_imputer_to_onnx_nodes(
    transformer: BooleanImputer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """ONNX converter for BooleanImputer.

    Boolean columns are expected as float32 tensors (False=0.0, True=1.0, null=NaN).
    Missing values are replaced via:  Where(IsNaN(input), fill_float, input)
    where fill_float = 1.0 for True and 0.0 for False.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col in transformer._statistics:
        if col not in input_names:  # pragma: no cover
            continue

        fill_val = 1.0 if transformer._statistics[col] else 0.0
        in_name = input_names[col]
        out_name = output_names[col]
        prefix = f"{in_name}__BoolImputer"
        isnan_out = f"{prefix}__isnan"
        fill_init = f"{prefix}__fill"

        initializers.append(_make_init(fill_init, fill_val, get_input_onnx_type(transformer, col)))
        nodes.append(oh.make_node("IsNaN", inputs=[in_name], outputs=[isnan_out]))
        nodes.append(
            oh.make_node("Where", inputs=[isnan_out, fill_init, in_name], outputs=[out_name])
        )

    return nodes, initializers


# ── IterativeImputer ──────────────────────────────────────────────────────────

@to_onnx_nodes.register(IterativeImputer)
def _iterative_imputer_to_onnx_nodes(
    transformer: IterativeImputer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """ONNX converter for IterativeImputer.

    Implements one pass of MICE (Multivariate Imputation by Chained Equations):

    Phase 1 — initial fill:
        For every feature column:
            filled_col = Where(IsNaN(col), mean_stat, col)

    Phase 2 — sequential linear regression in _imputation_order:
        For each imputable column col (in fit-determined order):
            pred = intercept + sum(coef_k * current[other_k] for k != j)
            col_out = Where(IsNaN(col_original), pred, col_original)
            current[col] is then updated to col_out so later columns
            benefit from the already-imputed values, matching the in-place
            NumPy mutation in Polars transform().

    Accuracy notes:
        - max_iter=1 : exactly matches the Polars transform.
        - max_iter>1 : the exported graph performs only one regression pass
          (equivalent to max_iter=1), which is an approximation.
          With errors='raise', max_iter>1 raises OnnxNotSupportedError.
    """
    if transformer.max_iter > 1:
        if errors == "raise":
            raise OnnxNotSupportedError(
                f"IterativeImputer(max_iter={transformer.max_iter}): the ONNX graph "
                f"can represent only a single regression pass (max_iter=1). "
                f"Use errors='coerce' to export a single-pass approximation, "
                f"or refit with max_iter=1 for an exact export."
            )

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    feature_cols = transformer._feature_cols
    feat_col_idx = transformer._feat_col_idx
    statistics = transformer._statistics
    subset = set(transformer.subset or [])

    # Non-feature columns pass through unchanged
    for col, in_name in input_names.items():
        if col not in feat_col_idx:  # pragma: no cover
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    # Phase 1: initial fill for all feature columns (mean/median from training)
    # current_name tracks the most recent tensor name for each feature column
    current_name: dict[str, str] = {}
    for col in feature_cols:
        if col not in input_names:  # pragma: no cover
            current_name[col] = col  # fallback; column not in graph
            continue
        in_name = input_names[col]

        # ONNX IsNaN only accepts float tensor types; integer columns (e.g. Int64) carry no
        # null representation once exported, so the initial fill is a no-op pass-through.
        if get_input_onnx_type(transformer, col) not in (TensorProto.FLOAT, TensorProto.DOUBLE):
            current_name[col] = in_name
            continue

        p = f"{in_name}__IterImputer"
        stat_init = f"{p}__stat"
        isnan_out = f"{p}__isnan"
        filled_name = f"{p}__filled"

        initializers.append(_make_init(stat_init, float(statistics.get(col, 0.0)), get_input_onnx_type(transformer, col)))
        nodes.append(oh.make_node("IsNaN", inputs=[in_name], outputs=[isnan_out]))
        nodes.append(oh.make_node("Where", inputs=[isnan_out, stat_init, in_name], outputs=[filled_name]))
        current_name[col] = filled_name

    # Phase 2: sequential linear regression in imputation order
    for col in (transformer._imputation_order or []):
        if col not in input_names or col not in transformer._coefs:  # pragma: no cover
            continue
        j = feat_col_idx[col]
        coefs = transformer._coefs[col]
        # indices of the other feature columns used as predictors
        feat_idx = [k for k in range(len(feature_cols)) if k != j]
        in_name = input_names[col]

        # ONNX IsNaN only accepts float tensor types; integer columns (e.g. Int64) carry no
        # null representation once exported, so regression-based imputation is skipped.
        if get_input_onnx_type(transformer, col) not in (TensorProto.FLOAT, TensorProto.DOUBLE):
            current_name[col] = in_name
            continue

        p = f"{in_name}__IterImputer__reg"
        isnan_reg = f"{p}__isnan"

        # IsNaN on the original input (determines where to apply imputation)
        nodes.append(oh.make_node("IsNaN", inputs=[in_name], outputs=[isnan_reg]))

        if feat_idx and len(coefs) > 1:
            # Build linear prediction: intercept + sum(coef_k * current[other_col])
            intercept_init = f"{p}__intercept"
            initializers.append(_make_init(intercept_init, float(coefs[0]), get_input_onnx_type(transformer, col)))

            term_names: list[str] = []
            for term_pos, fi in enumerate(feat_idx):
                other_col = feature_cols[fi]
                if other_col not in current_name:  # pragma: no cover
                    continue
                other_name = current_name[other_col]
                target_type = get_input_onnx_type(transformer, col)
                # Predictor columns may be integer-typed (e.g. Int64); Mul requires both
                # operands to share a type, so cast to the regressed column's float type.
                if get_input_onnx_type(transformer, other_col) != target_type:
                    cast_name = f"{p}__predcast{term_pos}"
                    nodes.append(oh.make_node("Cast", inputs=[other_name], outputs=[cast_name], to=target_type))
                    other_name = cast_name
                coef_val = float(coefs[term_pos + 1])  # coefs[0] is intercept
                coef_init = f"{p}__coef{term_pos}"
                term_out = f"{p}__term{term_pos}"
                initializers.append(_make_init(coef_init, coef_val, target_type))
                nodes.append(
                    oh.make_node("Mul", inputs=[other_name, coef_init], outputs=[term_out])
                )
                term_names.append(term_out)

            # Sum all terms
            running = term_names[0] if term_names else intercept_init
            for i, term in enumerate(term_names[1:]):
                new_sum = f"{p}__sum{i}"
                nodes.append(oh.make_node("Add", inputs=[running, term], outputs=[new_sum]))
                running = new_sum

            # Add intercept
            pred_name = f"{p}__pred"
            nodes.append(oh.make_node("Add", inputs=[running, intercept_init], outputs=[pred_name]))
        else:
            # Fallback: constant prediction (only intercept stored)
            pred_name = f"{p}__pred"
            intercept_init = f"{p}__intercept"
            initializers.append(_make_init(intercept_init, float(coefs[0]), get_input_onnx_type(transformer, col)))
            # Need pred to be a tensor of shape [N]; broadcast [1] via Where
            nodes.append(oh.make_node("Identity", inputs=[intercept_init], outputs=[pred_name]))

        # Where(original_isnan, prediction, original_input)
        imputed_name = f"{p}__imputed"
        out_name = output_names.get(col, col)
        nodes.append(
            oh.make_node("Where", inputs=[isnan_reg, pred_name, in_name], outputs=[imputed_name])
        )
        # Update current tensor for this column (later imputations in the same
        # pass will use the already-imputed value, matching Polars' in-place update)
        current_name[col] = imputed_name

    # Phase 3: emit final output nodes for all feature columns
    for col in feature_cols:
        if col not in input_names:  # pragma: no cover
            continue
        out_name = output_names.get(col, col)
        if col in subset:
            # Column was imputed (or initial-filled); use the current tensor
            final_tensor = current_name.get(col, input_names[col])
            # Avoid emitting an Identity to itself
            if final_tensor != out_name:
                nodes.append(oh.make_node("Identity", inputs=[final_tensor], outputs=[out_name]))
        else:
            # Column not in subset: write original value through
            nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[out_name]))

    return nodes, initializers


# ── StringImputer ─────────────────────────────────────────────────────────────

@get_output_columns.register(StringImputer)
def _string_imputer_output_cols(transformer: StringImputer, input_columns: list[str]) -> list[str]:
    return _imputer_output_cols(transformer, input_columns)


@get_input_onnx_type.register(StringImputer)
def _string_imputer_input_type(transformer: StringImputer, col: str) -> int:
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


@get_output_onnx_type.register(StringImputer)
def _string_imputer_output_type(transformer: StringImputer, col: str) -> int:
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    return TensorProto.STRING if col in (transformer.subset or []) else TensorProto.FLOAT


@to_onnx_nodes.register(StringImputer)
def _string_imputer_to_onnx_nodes(
    transformer: StringImputer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Replace empty strings (null sentinel) with the fitted fill value.

    Convention: Polars null strings must be passed as \"\" (empty string) at the
    ONNX boundary.  Implementation:
        mask_i64 = LabelEncoder(input, keys=[\"\"], values=[1], default=0)
        mask     = Cast(mask_i64, to=BOOL)
        output   = Where(mask, fill_const, input)
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col, fill_val in transformer._statistics.items():
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        if transformer.inplace:
            out_name = output_names.get(col, col)
        else:
            actual_out = transformer._column_mapping.get(col, [col])[0]
            if not transformer.drop_columns and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
            out_name = output_names.get(actual_out, actual_out)
        prefix = f"{in_name}__StringImputer"
        fill_init = f"{prefix}__fill"
        mask_i64 = f"{prefix}__mask_i64"
        mask_bool = f"{prefix}__mask_bool"

        # onnx's make_tensor stub types vals as int|float only; STRING tensors accept bytes at runtime.
        initializers.append(oh.make_tensor(fill_init, TensorProto.STRING, [1], [fill_val.encode()]))  # type: ignore[list-item]
        # onnxruntime does not support Equal on string tensors; use LabelEncoder to build the mask
        nodes.append(oh.make_node(
            "LabelEncoder", domain="ai.onnx.ml",
            inputs=[in_name], outputs=[mask_i64],
            keys_strings=[""], values_int64s=[1], default_int64=0,
        ))
        nodes.append(oh.make_node("Cast", inputs=[mask_i64], outputs=[mask_bool], to=TensorProto.BOOL))
        nodes.append(oh.make_node("Where", inputs=[mask_bool, fill_init, in_name], outputs=[out_name]))

    return nodes, initializers


# ── GroupByImputer ────────────────────────────────────────────────────────────

@get_output_columns.register(GroupByImputer)
def _groupby_imputer_output_cols(transformer: GroupByImputer, input_columns: list[str]) -> list[str]:
    return _imputer_output_cols(transformer, input_columns)


@get_input_onnx_type.register(GroupByImputer)
def _groupby_imputer_input_type(transformer: GroupByImputer, col: str) -> int:
    if col == transformer.group_by_column:
        return TensorProto.STRING
    from ._converters import _POLARS_TO_ONNX
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@to_onnx_nodes.register(GroupByImputer)
def _groupby_imputer_to_onnx_nodes(
    transformer: GroupByImputer,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Group-stat lookup via LabelEncoder then Where(IsNaN(col), group_fill, col).

    For each numeric column, a LabelEncoder maps the group_by_column (STRING) to
    the pre-computed group statistic (FLOAT).  Unknown groups map to NaN so that
    null values whose group was not seen during fit remain null after imputation,
    matching the Polars left-join behaviour.
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset = set(transformer.subset or [])
    group_col = transformer.group_by_column
    group_in = input_names.get(group_col, group_col)

    for col, in_name in input_names.items():
        if col not in subset and col != group_col and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    # group_by_column always passes through unchanged
    if group_col in input_names and group_col in output_names:
        nodes.append(oh.make_node("Identity", inputs=[group_in], outputs=[output_names[group_col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        stats = transformer._statistics.get(col, {})
        onnx_type = get_input_onnx_type(transformer, col)

        if transformer.inplace:
            out_name = output_names.get(col, col)
        else:
            actual_out = transformer._column_mapping.get(col, [col])[0]
            if not transformer.drop_columns and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))
            out_name = output_names.get(actual_out, actual_out)

        # ONNX IsNaN only accepts float tensor types; integer columns (e.g. Int64) carry no
        # null representation once exported, so imputation on them is a no-op pass-through.
        if onnx_type not in (TensorProto.FLOAT, TensorProto.DOUBLE):
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[out_name]))
            continue

        p = f"{in_name}__GroupByImputer"
        fill_f32_name = f"{p}__fill_f32"
        fill_name = f"{p}__fill"
        isnan_name = f"{p}__isnan"

        # LabelEncoder: group string → fill float (always float32); cast to col type if needed
        nodes.append(oh.make_node(
            "LabelEncoder", domain="ai.onnx.ml",
            inputs=[group_in], outputs=[fill_f32_name],
            keys_strings=[str(k) for k in stats],
            values_floats=[float(v) if v is not None else float("nan") for v in stats.values()],
            default_float=float("nan"),
        ))
        if onnx_type != TensorProto.FLOAT:
            nodes.append(oh.make_node("Cast", inputs=[fill_f32_name], outputs=[fill_name], to=onnx_type))
        else:
            fill_name = fill_f32_name
        nodes.append(oh.make_node("IsNaN", inputs=[in_name], outputs=[isnan_name]))
        nodes.append(oh.make_node("Where", inputs=[isnan_name, fill_name, in_name], outputs=[out_name]))

    return nodes, initializers
