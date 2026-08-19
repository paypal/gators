from __future__ import annotations

import polars as pl

from ._converters import get_input_onnx_type, get_output_columns, get_output_onnx_type, to_onnx_nodes
from ..pipeline.pipeline import Pipeline
from ..transformer._base_transformer import _BaseTransformer

try:
    import onnx
    import onnx.helper as oh
    import onnx.shape_inference
    from onnx import TensorProto
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'onnx' package is required for ONNX export. "
        "Install it with: pip install gators[onnx]"
    ) from exc

_OPSET = 17
_OPSET_V20 = 20  # required by StringSplit (opset 20)
_ML_OPSET = 3  # ai.onnx.ml opset for LabelEncoder


def _opsets(nodes: list[onnx.NodeProto]) -> list[onnx.OperatorSetIdProto]:
    """Return the opset list needed for ``nodes`` (adds ai.onnx.ml when required).

    Bumps to opset 20 automatically when StringSplit nodes are present.
    """
    needs_v20 = any(getattr(n, "op_type", "") == "StringSplit" for n in nodes)
    base = [oh.make_opsetid("", _OPSET_V20 if needs_v20 else _OPSET)]
    if any(getattr(n, "domain", "") == "ai.onnx.ml" for n in nodes):
        base.append(oh.make_opsetid("ai.onnx.ml", _ML_OPSET))
    return base


def to_onnx_graph(transformer: _BaseTransformer, errors: str = "raise") -> onnx.ModelProto:
    """Export a single fitted Gators transformer to a self-contained ONNX ModelProto.

    All columns flow through the graph. Columns not in the transformer's subset
    pass through via Identity nodes. The graph is validated by the ONNX checker
    before being returned.

    Input tensor naming convention : ``{col}__in``
    Output tensor naming convention: ``{col}``  (original column names, inplace semantics)

    Parameters
    ----------
    transformer :
        A fitted Gators transformer instance.
    errors : {'coerce', 'raise'}, default 'coerce'
        How to handle unsupported transformers or strategies.
        'raise'  – raise OnnxNotSupportedError immediately.
        'coerce' – emit Identity pass-through nodes so the graph stays structurally valid.

    Returns
    -------
    onnx.ModelProto
        Validated ONNX model.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from gators.imputers import NumericImputer
    >>> from gators.onnx import to_onnx_graph
    >>>
    >>> X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, None]})
    >>> imputer = NumericImputer(strategy="median")
    >>> imputer.fit(X)
    >>> model = to_onnx_graph(imputer)
    """
    transformer.check_is_fitted()
    columns = transformer._input_columns
    output_columns = get_output_columns(transformer, columns)
    input_names = {col: f"{col}__in" for col in columns}
    output_names = {col: col for col in output_columns}

    nodes, initializers = to_onnx_nodes(transformer, input_names, output_names, errors=errors)

    graph_inputs = [
        oh.make_tensor_value_info(f"{col}__in", get_input_onnx_type(transformer, col), [None])
        for col in columns
    ]
    # Output types inferred by ONNX shape inference; [None] ensures shape field is
    # always present even when inference updates elem_type without setting shape.
    graph_outputs = [
        oh.make_tensor_value_info(col, TensorProto.UNDEFINED, [None])
        for col in output_columns
    ]
    graph = oh.make_graph(
        nodes, "GatorsTransformer", graph_inputs, graph_outputs, initializer=initializers
    )
    model = oh.make_model(graph, opset_imports=_opsets(nodes))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def pipeline_to_onnx(pipeline: Pipeline, errors: str = "raise") -> onnx.ModelProto:
    """Export a fitted Gators Pipeline to a single ONNX ModelProto.

    Each pipeline step contributes its nodes to the same graph, chained via
    intermediate tensor names.  Column additions (IsNull, MathFeatures, …),
    column drops (DropColumns, …), and renames (RenameColumns) are all supported.

    Steps with ``inplace=False`` **and** ``drop_columns=True`` are rejected
    (``errors='raise'``) or skipped with Identity pass-throughs
    (``errors='coerce'``) because they would silently break the column ordering.

    Input tensor naming convention : ``{col}__in``
    Output tensor naming convention: ``{col}``

    Parameters
    ----------
    pipeline : fitted gators.Pipeline
    errors : {'raise', 'coerce'}, default 'raise'
        How to handle unsupported steps or strategies.
        ``'raise'`` – raise :class:`OnnxNotSupportedError` immediately.
        ``'coerce'`` – emit Identity pass-through nodes so the graph stays valid.

    Returns
    -------
    onnx.ModelProto
        Validated ONNX model.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.pipeline import Pipeline
    >>> from gators.imputers import NumericImputer
    >>> from gators.scalers import StandardScaler
    >>> from gators.onnx import pipeline_to_onnx
    >>>
    >>> X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, None]})
    >>> pipe = Pipeline(steps=[
    ...     ("imputer", NumericImputer(strategy="median")),
    ...     ("scaler", StandardScaler()),
    ... ])
    >>> pipe.fit(X)
    >>> model = pipeline_to_onnx(pipe)
    """
    pipeline.check_is_fitted()
    columns = pipeline._input_columns
    n_steps = len(pipeline.steps)

    # Graph input type per column: use stored Polars dtype (FLOAT or DOUBLE); STRING overrides.
    graph_input_types = {col: get_input_onnx_type(pipeline, col) for col in columns}
    for _, transformer in pipeline.steps:
        for col in columns:
            step_type = get_input_onnx_type(transformer, col)
            # Only promote to STRING if not already a known concrete type (BOOL, INT64, etc.);
            # later steps may see STRING because an earlier step cast it, not the raw input.
            if (graph_input_types.get(col) not in (TensorProto.STRING, TensorProto.BOOL, TensorProto.INT64)  # pragma: no cover
                    and step_type == TensorProto.STRING):  # pragma: no cover
                graph_input_types[col] = TensorProto.STRING  # pragma: no cover
            # BOOL→INT64: nullable bool→string uses INT64 with -1=null sentinel
            elif graph_input_types.get(col) == TensorProto.BOOL and step_type == TensorProto.INT64:
                graph_input_types[col] = TensorProto.INT64
            # BOOL→FLOAT: BooleanImputer uses float encoding (False=0.0, True=1.0, null=NaN)
            elif graph_input_types.get(col) == TensorProto.BOOL and step_type == TensorProto.FLOAT:
                graph_input_types[col] = TensorProto.FLOAT
            # FLOAT/DOUBLE→INT64: datetime columns need INT64 (e.g. after CastColumns(Datetime))
            elif step_type == TensorProto.INT64 and graph_input_types.get(col) not in (
                TensorProto.STRING, TensorProto.BOOL, TensorProto.INT64
            ):
                graph_input_types[col] = TensorProto.INT64

    # current_names tracks the ONNX tensor name for each column; current_columns tracks
    # the evolving column set (changes when a step adds/removes columns, e.g. OHE).
    # current_col_types tracks the actual ONNX dtype flowing through the graph; this can
    # differ from _input_dtypes (e.g. encoder outputs FLOAT even when Polars stores Float64).
    current_columns = list(columns)
    current_names = {col: f"{col}__in" for col in columns}
    current_col_types: dict[str, int] = dict(graph_input_types)
    # Maps ONNX type → a Polars dtype that get_input_onnx_type maps back to the same ONNX type.
    _onnx_to_pl = {TensorProto.DOUBLE: pl.Float64, TensorProto.FLOAT: pl.Float32, TensorProto.STRING: pl.String, TensorProto.BOOL: pl.Boolean, TensorProto.INT64: pl.Int64}

    all_nodes: list = []
    all_initializers: list = []

    for step_idx, (_, transformer) in enumerate(pipeline.steps):
        is_last = step_idx == n_steps - 1
        output_cols = get_output_columns(transformer, current_columns)
        if is_last:
            step_output = {col: col for col in output_cols}
        else:
            step_output = {col: f"{col}__step{step_idx}" for col in output_cols}

        # Temporarily override _input_dtypes so converters create initializers that
        # match the *actual ONNX tensor types* flowing through the graph rather than
        # the Polars-stored types (which diverge after encoder steps).
        orig_dtypes = transformer._input_dtypes
        # Store original fitted dtypes so converters (e.g. CastColumns) can check source types.
        transformer.__dict__['_onnx_orig_input_dtypes'] = orig_dtypes
        transformer._input_dtypes = {
            col: _onnx_to_pl.get(current_col_types.get(col, TensorProto.FLOAT), pl.Float32)
            for col in current_columns
        }
        try:
            nodes, initializers = to_onnx_nodes(
                transformer, current_names, step_output, errors=errors
            )
            # Update current_col_types while _input_dtypes reflects the virtual types,
            # so pass-through columns keep their upstream ONNX type correctly.
            new_types: dict[str, int] = {}
            for col in output_cols:
                in_t = get_input_onnx_type(transformer, col)
                out_t = get_output_onnx_type(transformer, col)
                if in_t == TensorProto.STRING or out_t == TensorProto.STRING:
                    new_types[col] = out_t
                elif in_t != TensorProto.FLOAT or out_t != TensorProto.FLOAT:
                    new_types[col] = out_t
                else:
                    new_types[col] = current_col_types.get(col, TensorProto.FLOAT)
            current_col_types = new_types
        finally:
            transformer._input_dtypes = orig_dtypes
            transformer.__dict__.pop('_onnx_orig_input_dtypes', None)

        all_nodes.extend(nodes)
        all_initializers.extend(initializers)
        current_names = step_output
        current_columns = output_cols

    graph_inputs = [
        oh.make_tensor_value_info(f"{col}__in", graph_input_types[col], [None]) for col in columns
    ]
    # Output types inferred by ONNX shape inference; [None] ensures shape field is
    # always present even when inference updates elem_type without setting shape.
    graph_outputs = [
        oh.make_tensor_value_info(col, TensorProto.UNDEFINED, [None])
        for col in current_columns
    ]
    graph = oh.make_graph(
        all_nodes, "GatorsPipeline", graph_inputs, graph_outputs, initializer=all_initializers
    )
    model = oh.make_model(graph, opset_imports=_opsets(all_nodes))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def pipeline_to_scoring_onnx(
    pipeline: Pipeline,
    model_onnx: onnx.ModelProto,
    feature_columns: list[str],
    errors: str = "raise",
) -> onnx.ModelProto:
    """Chain a Gators preprocessing Pipeline with an ONNX ML model (XGBoost/LightGBM).

    The preprocessing graph outputs one 1D ``float[N]`` tensor per column.
    This function inserts a *reshape bridge* that Unsqueezes each selected column
    to ``float[N, 1]`` and Concatenates them into the ``float[N, n_features]``
    matrix the ML model expects, then stitches everything into a single graph.

    Parameters
    ----------
    pipeline : fitted gators.Pipeline
        Preprocessing pipeline (same one used when training the ML model).
    model_onnx : onnx.ModelProto
        The ML model exported to ONNX (e.g. via ``onnxmltools.convert_xgboost``
        or ``onnxmltools.convert_lightgbm``).
    feature_columns : list[str]
        Ordered list of preprocessing output column names to stack into the
        feature matrix.  Must exactly match the column order seen by the model
        at training time.
    errors : {'coerce', 'raise'}, default 'coerce'
        Forwarded to ``pipeline_to_onnx``.

    Returns
    -------
    onnx.ModelProto
        Single validated ONNX model: preprocessing → reshape bridge → scoring.
        Inputs  : same as the preprocessing pipeline (``{col}__in`` tensors).
        Outputs : same as the ML model (labels / probabilities).

    Examples
    --------
    >>> import lightgbm as lgb
    >>> from onnxmltools import convert_lightgbm
    >>> from onnxmltools.utils import float_array_type
    >>> from gators.onnx_converters import pipeline_to_scoring_onnx
    >>>
    >>> lgb_model = lgb.train(...)
    >>> n_features = len(feature_columns)
    >>> model_onnx = convert_lightgbm(lgb_model, initial_types=[
    ...     ("input", float_array_type([None, n_features]))
    ... ])
    >>> scoring_model = pipeline_to_scoring_onnx(pipe, model_onnx, feature_columns)
    """
    import numpy as np

    # 1. Export preprocessing graph
    preprocessing = pipeline_to_onnx(pipeline, errors=errors)

    # 2. Identify the ML model's first input tensor (the feature matrix)
    ml_input_name = model_onnx.graph.input[0].name

    # 3. Build bridge: Unsqueeze each column [N] → [N,1] then Concat → [N, F]
    #    Axes initializer: int64 scalar [1] selects axis=1
    axes_init = oh.make_tensor("gators__bridge_axes", TensorProto.INT64, [1], [1])
    bridge_nodes: list = []
    unsqueezed: list[str] = []
    for col in feature_columns:
        cast_name = f"gators__{col}__cast"
        uq_name = f"gators__{col}__unsq"
        # Cast each column to the ML model's input element type before stacking
        ml_elem_type = model_onnx.graph.input[0].type.tensor_type.elem_type
        bridge_nodes.append(
            oh.make_node("Cast", inputs=[col], outputs=[cast_name], to=ml_elem_type)
        )
        bridge_nodes.append(
            oh.make_node("Unsqueeze", inputs=[cast_name, "gators__bridge_axes"], outputs=[uq_name])
        )
        unsqueezed.append(uq_name)
    # Concat outputs directly to the name the ML model reads as its input
    bridge_nodes.append(
        oh.make_node("Concat", inputs=unsqueezed, outputs=[ml_input_name], axis=1)
    )

    # 4. Merge all nodes and initializers into one graph
    all_nodes = list(preprocessing.graph.node) + bridge_nodes + list(model_onnx.graph.node)
    all_inits = list(preprocessing.graph.initializer) + [axes_init] + list(model_onnx.graph.initializer)

    # Carry over intermediate tensor type annotations from the preprocessing graph so
    # ORT shape inference can resolve types without re-deriving them from STRING inputs.
    # preprocessing.graph.output tensors become internal tensors here, so move them to value_info.
    # Filter out UNDEFINED-typed entries that shape inference didn't resolve.
    pre_value_info = [
        vi for vi in list(preprocessing.graph.value_info) + list(preprocessing.graph.output)
        if vi.type.tensor_type.elem_type != TensorProto.UNDEFINED
    ]

    graph = oh.make_graph(
        all_nodes,
        "GatorsScoringPipeline",
        list(preprocessing.graph.input),   # inputs: same as preprocessing
        list(model_onnx.graph.output),     # outputs: labels / probabilities from ML model
        initializer=all_inits,
        value_info=pre_value_info,
    )

    # 5. Collect opsets from both models (preprocessing + ML model)
    opset_imports = _opsets(all_nodes)
    existing_domains = {o.domain for o in opset_imports}
    for opset in model_onnx.opset_import:
        if opset.domain not in existing_domains:
            opset_imports.append(oh.make_opsetid(opset.domain, opset.version))

    model = oh.make_model(graph, opset_imports=opset_imports)
    onnx.checker.check_model(model)
    return model
