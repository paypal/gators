"""ONNX converters for gators.feature_generation_dt transformers.

Datetime columns are fed as tensor(int64) containing the Polars physical
representation: microseconds/milliseconds/nanoseconds since Unix epoch for
Datetime, days since epoch for Date.  All component extractions use pure
integer arithmetic that ONNX already supports.

Note: ONNX integer Div truncates towards zero; Polars // floors towards
negative infinity.  Results match for timestamps after 1970-01-01 (the
typical range for business features).

Supported
---------
BusinessTimeFeatures : is_business_hour, is_business_day,
                       time_of_business_day, hour_of_business_day
OrdinalFeatures      : hour, minute, second, day_of_week, weekend
CyclicFeatures       : hour, minute, second, day_of_week
DiffFeatures         : all units (d, h, m, s) for column_pairs and reference_dates
TimeBinFeatures      : part_of_day, rush_hour

Not supported (require calendar arithmetic: year, month, day_of_month, etc.)
----------------------------------------------------------------------------
OrdinalFeatures  : century, year, semester, quarter, month, week,
                   day_of_month, day_of_year, leap_year
CyclicFeatures   : month, quarter, semester, week, day_of_month, day_of_year
TimeBinFeatures  : season, time_of_month, time_of_year
DurationToDatetime, HolidayFeatures, TimeWindowFeatures
"""
from __future__ import annotations

from math import pi

import polars as pl

from ._converters import _POLARS_TO_ONNX, get_input_onnx_type, get_output_columns, get_output_onnx_type, to_onnx_nodes
from ._exceptions import OnnxNotSupportedError
from ..feature_generation_dt.business_time_features import BusinessTimeFeatures
from ..feature_generation_dt.cyclic_features import CyclicFeatures, CYCLIC_FACTORS
from ..feature_generation_dt.diff_features import DiffFeatures
from ..feature_generation_dt.ordinal_features import OrdinalFeatures
from ..feature_generation_dt.time_bin_features import TimeBinFeatures

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


# Number of time units per hour and per day for each Polars time unit
_UNIT_PER: dict[str, tuple[int, int]] = {
    "us": (3_600_000_000, 86_400_000_000),
    "ms": (3_600_000,     86_400_000),
    "ns": (3_600_000_000_000, 86_400_000_000_000),
}
# Divisors for minute and second extraction
_UNIT_PER_MIN: dict[str, int] = {"us": 60_000_000, "ms": 60_000, "ns": 60_000_000_000}
_UNIT_PER_SEC: dict[str, int] = {"us": 1_000_000,  "ms": 1_000,  "ns": 1_000_000_000}
# 1970-01-01 was Thursday; in Polars 1-7 notation Thursday=4, so offset=3, then +1
_EPOCH_WEEKDAY_OFFSET = 3
_WEEKDAY_ONE_BASE = 1  # final +1 to match Polars 1=Mon…7=Sun

# Ordinal/cyclic components whose extraction requires only integer arithmetic
_EASY_COMPONENTS = frozenset({"hour", "minute", "second", "day_of_week", "weekend"})
# Cyclic components supported in ONNX (subset of EASY that are in CYCLIC_FACTORS)
_EASY_CYCLIC = frozenset({"hour", "minute", "second", "day_of_week"})


# ── schema hooks ──────────────────────────────────────────────────────────────

@get_output_columns.register(BusinessTimeFeatures)
def _btf_output_columns(transformer: BusinessTimeFeatures, input_columns: list[str]) -> list[str]:
    generated = [
        f"{col}__{feature}"
        for col in (transformer.subset or [])
        for feature in transformer.features
    ]
    if transformer.drop_columns:
        dropped = set(transformer.subset or [])
        return [c for c in input_columns if c not in dropped] + generated
    return list(input_columns) + generated


@get_input_onnx_type.register(BusinessTimeFeatures)
def _btf_input_type(transformer: BusinessTimeFeatures, col: str) -> int:
    if col in (transformer.subset or []):
        return TensorProto.INT64  # datetime → physical int64 (μs/ms/ns since epoch)
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(BusinessTimeFeatures)
def _btf_output_type(transformer: BusinessTimeFeatures, col: str) -> int:
    for dt_col in (transformer.subset or []):
        if col == dt_col:
            return TensorProto.INT64
        for feature in transformer.features:
            if col == f"{dt_col}__{feature}":
                return TensorProto.STRING if feature == "time_of_business_day" else TensorProto.FLOAT
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


# ── ONNX nodes ────────────────────────────────────────────────────────────────

@to_onnx_nodes.register(BusinessTimeFeatures)
def _btf_to_onnx_nodes(
    transformer: BusinessTimeFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Extract hour/weekday from int64 epoch microseconds via integer arithmetic.

    hour    = (ts // US_PER_HOUR) % 24
    weekday = (ts // US_PER_DAY + 3) % 7        # 0=Mon, epoch was Thu=3
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    subset_set = set(transformer.subset or [])
    generated_set = {
        f"{col}__{feature}"
        for col in (transformer.subset or [])
        for feature in transformer.features
    }

    # Passthrough columns that are neither datetime nor generated
    for col, in_name in input_names.items():
        if col not in subset_set and col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        ts_in = input_names[col]
        p = f"{ts_in}__BTF"
        unit = transformer._dt_units.get(col, "us")
        per_hour, per_day = _UNIT_PER.get(unit, _UNIT_PER["us"]) if unit != "date" else (None, 1)

        def i64(name: str, val: int) -> str:
            initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
            return name

        # ── Extract weekday ───────────────────────────────────────────────────
        days_raw = f"{p}__days_raw"
        days_off = f"{p}__days_off"
        wkd0     = f"{p}__wkd0"
        weekday  = f"{p}__weekday"

        if unit == "date":
            epo  = i64(f"{p}__epo", _EPOCH_WEEKDAY_OFFSET)
            one  = i64(f"{p}__one", _WEEKDAY_ONE_BASE)
            w7   = i64(f"{p}__w7",  7)
            nodes.append(oh.make_node("Add", inputs=[ts_in, epo], outputs=[days_off]))
            nodes.append(oh.make_node("Mod", inputs=[days_off, w7], outputs=[wkd0]))
            nodes.append(oh.make_node("Add", inputs=[wkd0, one], outputs=[weekday]))
        else:
            upd  = i64(f"{p}__upd", per_day)
            epo  = i64(f"{p}__epo", _EPOCH_WEEKDAY_OFFSET)
            one  = i64(f"{p}__one", _WEEKDAY_ONE_BASE)
            w7   = i64(f"{p}__w7",  7)
            nodes.append(oh.make_node("Div", inputs=[ts_in, upd], outputs=[days_raw]))
            nodes.append(oh.make_node("Add", inputs=[days_raw, epo], outputs=[days_off]))
            nodes.append(oh.make_node("Mod", inputs=[days_off, w7], outputs=[wkd0]))
            nodes.append(oh.make_node("Add", inputs=[wkd0, one], outputs=[weekday]))

        # ── Extract hour ──────────────────────────────────────────────────────
        hour = f"{p}__hour"
        if unit == "date":
            # Date columns have no time component — hour is always 0
            nodes.append(oh.make_node("Constant", inputs=[], outputs=[hour],
                value=oh.make_tensor("hour_zero", TensorProto.INT64, [1], [0])))
        else:
            uph     = i64(f"{p}__uph", per_hour)
            h24     = i64(f"{p}__h24", 24)
            hour_raw = f"{p}__hour_raw"
            nodes.append(oh.make_node("Div", inputs=[ts_in, uph], outputs=[hour_raw]))
            nodes.append(oh.make_node("Mod", inputs=[hour_raw, h24], outputs=[hour]))

        # ── Shared: is_weekend boolean ────────────────────────────────────────
        eq_outs: list[str] = []
        for wd in transformer.weekend_days:
            wd_c  = i64(f"{p}__wd_{wd}", wd)
            eq_out = f"{p}__eq_wd_{wd}"
            nodes.append(oh.make_node("Equal", inputs=[weekday, wd_c], outputs=[eq_out]))
            eq_outs.append(eq_out)

        is_weekend = f"{p}__is_weekend"
        if len(eq_outs) == 1:
            nodes.append(oh.make_node("Identity", inputs=[eq_outs[0]], outputs=[is_weekend]))
        else:
            running = eq_outs[0]
            for i, eq in enumerate(eq_outs[1:]):
                out = is_weekend if i == len(eq_outs) - 2 else f"{p}__or_wd_{i}"
                nodes.append(oh.make_node("Or", inputs=[running, eq], outputs=[out]))
                running = out

        # ── Shared: business-hour booleans ────────────────────────────────────
        start_c = i64(f"{p}__start", transformer.business_hours_start)
        end_c   = i64(f"{p}__end",   transformer.business_hours_end)

        lt_start  = f"{p}__lt_start"
        lt_end    = f"{p}__lt_end"
        ge_start  = f"{p}__ge_start"
        is_biz    = f"{p}__is_biz"

        nodes.append(oh.make_node("Less",           inputs=[hour, start_c], outputs=[lt_start]))
        nodes.append(oh.make_node("Less",           inputs=[hour, end_c],   outputs=[lt_end]))
        nodes.append(oh.make_node("GreaterOrEqual", inputs=[hour, start_c], outputs=[ge_start]))
        nodes.append(oh.make_node("And",            inputs=[ge_start, lt_end], outputs=[is_biz]))

        # ── Passthrough datetime column ───────────────────────────────────────
        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[ts_in], outputs=[output_names[col]]))

        # ── Feature nodes ─────────────────────────────────────────────────────
        for feature in transformer.features:
            out_name = output_names.get(f"{col}__{feature}", f"{col}__{feature}")

            if feature == "is_business_hour":
                nodes.append(oh.make_node("Cast", inputs=[is_biz], outputs=[out_name], to=TensorProto.FLOAT))

            elif feature == "is_business_day":
                not_wknd = f"{p}__not_wknd"
                nodes.append(oh.make_node("Not",  inputs=[is_weekend], outputs=[not_wknd]))
                nodes.append(oh.make_node("Cast", inputs=[not_wknd],   outputs=[out_name], to=TensorProto.FLOAT))

            elif feature == "time_of_business_day":
                # Integer codes: 0=before_hours, 1=during_hours, 2=after_hours, 3=weekend
                zero_i  = i64(f"{p}__tobd_0", 0)
                one_i   = i64(f"{p}__tobd_1", 1)
                two_i   = i64(f"{p}__tobd_2", 2)
                three_i = i64(f"{p}__tobd_3", 3)
                inner   = f"{p}__tobd_inner"
                mid     = f"{p}__tobd_mid"
                code    = f"{p}__tobd_code"
                # Where(lt_end, 1, 2)  →  during vs after
                nodes.append(oh.make_node("Where", inputs=[lt_end,     one_i,   two_i],  outputs=[inner]))
                # Where(lt_start, 0, inner)  →  before vs (during/after)
                nodes.append(oh.make_node("Where", inputs=[lt_start,   zero_i,  inner],  outputs=[mid]))
                # Where(is_weekend, 3, mid)  →  weekend overrides everything
                nodes.append(oh.make_node("Where", inputs=[is_weekend, three_i, mid],    outputs=[code]))
                nodes.append(oh.make_node(
                    "LabelEncoder", domain="ai.onnx.ml",
                    inputs=[code], outputs=[out_name],
                    keys_int64s=[0, 1, 2, 3],
                    values_strings=["before_hours", "during_hours", "after_hours", "weekend"],
                    default_string="before_hours",
                ))

            elif feature == "hour_of_business_day":
                sub_i64 = f"{p}__hobd_i64"
                sub_f   = f"{p}__hobd_f"
                nan_c   = f"{p}__nan"
                initializers.append(onnx.numpy_helper.from_array(
                    np.array([float("nan")], dtype=np.float32), name=nan_c
                ))
                nodes.append(oh.make_node("Sub",  inputs=[hour, start_c], outputs=[sub_i64]))
                nodes.append(oh.make_node("Cast", inputs=[sub_i64],       outputs=[sub_f], to=TensorProto.FLOAT))
                nodes.append(oh.make_node("Where", inputs=[is_biz, sub_f, nan_c], outputs=[out_name]))

    return nodes, initializers


# ── Shared component extraction helper ───────────────────────────────────────

def _extract_components(ts_in: str, p: str, unit: str,
                        needed: frozenset, nodes: list, initializers: list) -> dict[str, str]:
    """Extract only the requested easy datetime components; return {name: tensor_name}."""
    per_hour, per_day = _UNIT_PER.get(unit, _UNIT_PER["us"])
    per_min = _UNIT_PER_MIN.get(unit, _UNIT_PER_MIN["us"])
    per_sec = _UNIT_PER_SEC.get(unit, _UNIT_PER_SEC["us"])

    def i64(name: str, val: int) -> str:
        initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
        return name

    result: dict[str, str] = {}

    if "hour" in needed:
        uph = i64(f"{p}__uph", per_hour)
        h24 = i64(f"{p}__h24", 24)
        hr_raw = f"{p}__hour_raw"
        hr     = f"{p}__hour"
        nodes.append(oh.make_node("Div", inputs=[ts_in, uph], outputs=[hr_raw]))
        nodes.append(oh.make_node("Mod", inputs=[hr_raw, h24], outputs=[hr]))
        result["hour"] = hr

    if "minute" in needed:
        upm = i64(f"{p}__upm", per_min)
        m60 = i64(f"{p}__m60", 60)
        mn_raw = f"{p}__min_raw"
        mn     = f"{p}__minute"
        nodes.append(oh.make_node("Div", inputs=[ts_in, upm], outputs=[mn_raw]))
        nodes.append(oh.make_node("Mod", inputs=[mn_raw, m60], outputs=[mn]))
        result["minute"] = mn

    if "second" in needed:
        ups = i64(f"{p}__ups", per_sec)
        s60 = i64(f"{p}__s60", 60)
        sc_raw = f"{p}__sec_raw"
        sc     = f"{p}__second"
        nodes.append(oh.make_node("Div", inputs=[ts_in, ups], outputs=[sc_raw]))
        nodes.append(oh.make_node("Mod", inputs=[sc_raw, s60], outputs=[sc]))
        result["second"] = sc

    if "day_of_week" in needed or "weekend" in needed:
        upd  = i64(f"{p}__upd", per_day)
        epo  = i64(f"{p}__epo", _EPOCH_WEEKDAY_OFFSET)
        one  = i64(f"{p}__one", _WEEKDAY_ONE_BASE)
        w7   = i64(f"{p}__w7",  7)
        dr   = f"{p}__days_raw"
        doff = f"{p}__days_off"
        wkd0 = f"{p}__wkd0"
        wd   = f"{p}__weekday"
        nodes.append(oh.make_node("Div", inputs=[ts_in, upd], outputs=[dr]))
        nodes.append(oh.make_node("Add", inputs=[dr, epo], outputs=[doff]))
        nodes.append(oh.make_node("Mod", inputs=[doff, w7], outputs=[wkd0]))
        nodes.append(oh.make_node("Add", inputs=[wkd0, one], outputs=[wd]))
        result["day_of_week"] = wd

    return result


# ── OrdinalFeatures ───────────────────────────────────────────────────────────

@get_output_columns.register(OrdinalFeatures)
def _of_output_columns(transformer: OrdinalFeatures, input_columns: list[str]) -> list[str]:
    generated = [f"{col}__{comp}" for col in (transformer.subset or []) for comp in transformer.components if comp in _EASY_COMPONENTS]
    if transformer.drop_columns:
        dropped = set(transformer.subset or [])
        return [c for c in input_columns if c not in dropped] + generated
    return list(input_columns) + generated


@get_input_onnx_type.register(OrdinalFeatures)
def _of_input_type(transformer: OrdinalFeatures, col: str) -> int:
    if col in (transformer.subset or []):
        return TensorProto.INT64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(OrdinalFeatures)
def _of_output_type(transformer: OrdinalFeatures, col: str) -> int:
    for dt_col in (transformer.subset or []):
        if col == dt_col:
            return TensorProto.INT64
        for comp in transformer.components:
            if col == f"{dt_col}__{comp}":
                return TensorProto.FLOAT if comp == "weekend" else TensorProto.INT64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@to_onnx_nodes.register(OrdinalFeatures)
def _of_to_onnx_nodes(
    transformer: OrdinalFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Integer extraction of easy components; raises for calendar-dependent components."""
    hard = [c for c in transformer.components if c not in _EASY_COMPONENTS]
    if hard:
        msg = (f"OrdinalFeatures components {hard} require calendar arithmetic "
               f"and cannot be exported to ONNX. Use only: {sorted(_EASY_COMPONENTS)}.")
        if errors == "raise":
            raise OnnxNotSupportedError(msg)

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset_set = set(transformer.subset or [])
    generated_set = {f"{col}__{c}" for col in subset_set for c in transformer.components}

    for col, in_name in input_names.items():
        if col not in subset_set and col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        ts_in = input_names[col]
        p = f"{ts_in}__OF"
        unit = transformer._dt_units.get(col, "us")
        easy = frozenset(c for c in transformer.components if c in _EASY_COMPONENTS)
        comps = _extract_components(ts_in, p, unit, easy, nodes, initializers)

        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[ts_in], outputs=[output_names[col]]))

        def i64(name: str, val: int) -> str:
            initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
            return name

        for comp in transformer.components:
            if comp not in _EASY_COMPONENTS:
                continue
            out_name = output_names.get(f"{col}__{comp}", f"{col}__{comp}")

            if comp == "weekend":
                wd = comps["day_of_week"]
                six_c = i64(f"{p}__six", 6)
                seven_c = i64(f"{p}__seven", 7)
                eq6 = f"{p}__eq6"
                eq7 = f"{p}__eq7"
                is_wknd = f"{p}__is_wknd"
                nodes.append(oh.make_node("Equal", inputs=[wd, six_c], outputs=[eq6]))
                nodes.append(oh.make_node("Equal", inputs=[wd, seven_c], outputs=[eq7]))
                nodes.append(oh.make_node("Or", inputs=[eq6, eq7], outputs=[is_wknd]))
                nodes.append(oh.make_node("Cast", inputs=[is_wknd], outputs=[out_name], to=TensorProto.FLOAT))
            else:
                nodes.append(oh.make_node("Identity", inputs=[comps[comp]], outputs=[out_name]))

    return nodes, initializers


# ── CyclicFeatures ────────────────────────────────────────────────────────────

@get_output_columns.register(CyclicFeatures)
def _cf_output_columns(transformer: CyclicFeatures, input_columns: list[str]) -> list[str]:
    generated = [
        f"{col}__{comp}__sin{int(a) if a == int(a) else round(a, 2)}"
        for col in (transformer.subset or [])
        for comp in transformer.components if comp in _EASY_CYCLIC
        for a in transformer.angles
    ]
    if transformer.drop_columns:
        dropped = set(transformer.subset or [])
        return [c for c in input_columns if c not in dropped] + generated
    return list(input_columns) + generated


@get_input_onnx_type.register(CyclicFeatures)
def _cf_input_type(transformer: CyclicFeatures, col: str) -> int:
    if col in (transformer.subset or []):
        return TensorProto.INT64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(CyclicFeatures)
def _cf_output_type(transformer: CyclicFeatures, col: str) -> int:
    for dt_col in (transformer.subset or []):
        if col == dt_col:
            return TensorProto.INT64
        for comp in transformer.components:
            for a in transformer.angles:
                ai = int(a) if a == int(a) else round(a, 2)
                if col == f"{dt_col}__{comp}__sin{ai}":
                    return TensorProto.FLOAT
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@to_onnx_nodes.register(CyclicFeatures)
def _cf_to_onnx_nodes(
    transformer: CyclicFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """sin(factor * component + phase) for hour, minute, second, day_of_week."""
    hard = [c for c in transformer.components if c not in _EASY_CYCLIC]
    if hard:
        msg = (f"CyclicFeatures components {hard} require calendar arithmetic "
               f"and cannot be exported to ONNX. Supported: {sorted(_EASY_CYCLIC)}.")
        if errors == "raise":
            raise OnnxNotSupportedError(msg)

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset_set = set(transformer.subset or [])

    for col, in_name in input_names.items():
        if col not in subset_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    angles_rad = [a * pi / 180.0 for a in transformer.angles]

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        ts_in = input_names[col]
        p = f"{ts_in}__CF"
        unit = transformer._dt_units.get(col, "us")
        easy = frozenset(c for c in transformer.components if c in _EASY_CYCLIC)
        comps = _extract_components(ts_in, p, unit, easy, nodes, initializers)

        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[ts_in], outputs=[output_names[col]]))

        for comp in transformer.components:
            if comp not in _EASY_CYCLIC:
                continue
            factor = CYCLIC_FACTORS[comp]
            comp_i64 = comps[comp]
            # Cast component to float for Sin arithmetic
            comp_f = f"{p}__{comp}_f"
            nodes.append(oh.make_node("Cast", inputs=[comp_i64], outputs=[comp_f], to=TensorProto.FLOAT))

            for a_deg, a_rad in zip(transformer.angles, angles_rad):
                ai = int(a_deg) if a_deg == int(a_deg) else round(a_deg, 2)
                out_name = output_names.get(f"{col}__{comp}__sin{ai}", f"{col}__{comp}__sin{ai}")
                factor_c = f"{p}__{comp}__factor_{ai}"
                product  = f"{p}__{comp}__prod_{ai}"
                shifted  = f"{p}__{comp}__shift_{ai}"
                initializers.append(onnx.numpy_helper.from_array(
                    np.array([factor], dtype=np.float32), name=factor_c
                ))
                nodes.append(oh.make_node("Mul", inputs=[comp_f, factor_c], outputs=[product]))
                if a_rad != 0.0:
                    phase_c = f"{p}__{comp}__phase_{ai}"
                    initializers.append(onnx.numpy_helper.from_array(
                        np.array([a_rad], dtype=np.float32), name=phase_c
                    ))
                    nodes.append(oh.make_node("Add", inputs=[product, phase_c], outputs=[shifted]))
                    nodes.append(oh.make_node("Sin", inputs=[shifted], outputs=[out_name]))
                else:
                    nodes.append(oh.make_node("Sin", inputs=[product], outputs=[out_name]))

    return nodes, initializers


# ── DiffFeatures ──────────────────────────────────────────────────────────────

_UNIT_NAMES = {"d": "days", "h": "hours", "m": "minutes", "s": "seconds"}


def _diff_cols_used(transformer: DiffFeatures) -> set[str]:
    cols: set[str] = set()
    if transformer.column_pairs:
        for a, b in transformer.column_pairs:
            cols.update([a, b])
    if transformer.reference_dates:
        cols.update(transformer.reference_dates.keys())
    return cols


@get_output_columns.register(DiffFeatures)
def _df_output_columns(transformer: DiffFeatures, input_columns: list[str]) -> list[str]:
    generated: list[str] = []
    if transformer.column_pairs:
        for a, b in transformer.column_pairs:
            for u in transformer.units:
                generated.append(f"{a}_minus_{b}__{_UNIT_NAMES[u]}")
    if transformer.reference_dates:
        for col in transformer.reference_dates:
            for u in transformer.units:
                generated.append(f"{col}_since_ref__{_UNIT_NAMES[u]}")
    dt_cols = _diff_cols_used(transformer)
    if transformer.drop_columns:
        return [c for c in input_columns if c not in dt_cols] + generated
    return list(input_columns) + generated


@get_input_onnx_type.register(DiffFeatures)
def _df_input_type(transformer: DiffFeatures, col: str) -> int:
    if col in _diff_cols_used(transformer):
        return TensorProto.INT64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(DiffFeatures)
def _df_output_type(transformer: DiffFeatures, col: str) -> int:
    gen_set = set(_df_output_columns(transformer, []))
    if col in gen_set:
        return TensorProto.INT64
    dt_cols = _diff_cols_used(transformer)
    if col in dt_cols:
        return TensorProto.INT64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@to_onnx_nodes.register(DiffFeatures)
def _df_to_onnx_nodes(
    transformer: DiffFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Difference as int64 epoch subtraction, divided by the unit divisor."""
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    dt_cols = _diff_cols_used(transformer)

    for col, in_name in input_names.items():
        if col not in dt_cols and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    if not transformer.drop_columns:
        for col in dt_cols:
            if col in input_names and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))

    def _unit_divisor(col: str, unit: str) -> int:
        col_unit = transformer._dt_units.get(col, "us")
        per_hour, per_day = _UNIT_PER.get(col_unit, _UNIT_PER["us"])
        per_min = _UNIT_PER_MIN.get(col_unit, _UNIT_PER_MIN["us"])
        per_sec = _UNIT_PER_SEC.get(col_unit, _UNIT_PER_SEC["us"])
        return {"d": per_day, "h": per_hour, "m": per_min, "s": per_sec}[unit]

    def _emit_diff(diff_tensor: str, p: str, a_col: str, out_base: str) -> None:
        for unit in transformer.units:
            divisor = _unit_divisor(a_col, unit)
            div_init = f"{p}__div_{unit}"
            out_name = output_names.get(f"{out_base}__{_UNIT_NAMES[unit]}", f"{out_base}__{_UNIT_NAMES[unit]}")
            initializers.append(onnx.numpy_helper.from_array(
                np.array([divisor], dtype=np.int64), name=div_init
            ))
            nodes.append(oh.make_node("Div", inputs=[diff_tensor, div_init], outputs=[out_name]))

    if transformer.column_pairs:
        for a, b in transformer.column_pairs:
            if a not in input_names or b not in input_names:  # pragma: no cover
                continue
            p = f"{input_names[a]}__DF__{b}"
            diff = f"{p}__diff"
            nodes.append(oh.make_node("Sub", inputs=[input_names[a], input_names[b]], outputs=[diff]))
            _emit_diff(diff, p, a, f"{a}_minus_{b}")

    if transformer.reference_dates:
        for col, ref_phys in transformer._reference_dates_physical.items():
            if col not in input_names:  # pragma: no cover
                continue
            p = f"{input_names[col]}__DF__ref"
            ref_init = f"{p}__ref"
            initializers.append(onnx.numpy_helper.from_array(
                np.array([ref_phys], dtype=np.int64), name=ref_init
            ))
            diff = f"{p}__diff"
            nodes.append(oh.make_node("Sub", inputs=[input_names[col], ref_init], outputs=[diff]))
            _emit_diff(diff, p, col, f"{col}_since_ref")

    return nodes, initializers


# ── TimeBinFeatures ───────────────────────────────────────────────────────────

_EASY_BINS = frozenset({"part_of_day", "rush_hour"})


@get_output_columns.register(TimeBinFeatures)
def _tbf_output_columns(transformer: TimeBinFeatures, input_columns: list[str]) -> list[str]:
    easy = [b for b in transformer.bin_types if b in _EASY_BINS]
    generated = [f"{col}__{b}" for col in (transformer.subset or []) for b in easy]
    if transformer.drop_columns:
        dropped = set(transformer.subset or [])
        return [c for c in input_columns if c not in dropped] + generated
    return list(input_columns) + generated


@get_input_onnx_type.register(TimeBinFeatures)
def _tbf_input_type(transformer: TimeBinFeatures, col: str) -> int:
    if col in (transformer.subset or []):
        return TensorProto.INT64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(TimeBinFeatures)
def _tbf_output_type(transformer: TimeBinFeatures, col: str) -> int:
    for dt_col in (transformer.subset or []):
        if col == dt_col:
            return TensorProto.INT64
        for b in transformer.bin_types:
            if col == f"{dt_col}__{b}":
                return TensorProto.STRING
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@to_onnx_nodes.register(TimeBinFeatures)
def _tbf_to_onnx_nodes(
    transformer: TimeBinFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """part_of_day and rush_hour via hour extraction + cascaded Where + LabelEncoder."""
    hard = [b for b in transformer.bin_types if b not in _EASY_BINS]
    if hard:
        msg = (f"TimeBinFeatures bin_types {hard} require calendar arithmetic "
               f"(month/day) and cannot be exported to ONNX. Supported: {sorted(_EASY_BINS)}.")
        if errors == "raise":
            raise OnnxNotSupportedError(msg)

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset_set = set(transformer.subset or [])
    easy_bins = [b for b in transformer.bin_types if b in _EASY_BINS]
    generated_set = {f"{col}__{b}" for col in subset_set for b in easy_bins}

    for col, in_name in input_names.items():
        if col not in subset_set and col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        ts_in = input_names[col]
        p = f"{ts_in}__TBF"
        unit = transformer._dt_units.get(col, "us") if hasattr(transformer, '_dt_units') else "us"
        comps = _extract_components(ts_in, p, unit, frozenset({"hour"}), nodes, initializers)
        hour = comps["hour"]

        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[ts_in], outputs=[output_names[col]]))

        def i64(name: str, val: int) -> str:
            initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
            return name

        for bin_type in easy_bins:
            out_name = output_names.get(f"{col}__{bin_type}", f"{col}__{bin_type}")

            if bin_type == "part_of_day":
                # 0=night(<6), 1=morning(6-12), 2=afternoon(12-18), 3=evening(>=18)
                lt6  = f"{p}__lt6"
                lt12 = f"{p}__lt12"
                lt18 = f"{p}__lt18"
                c6   = i64(f"{p}__c6",  6)
                c12  = i64(f"{p}__c12", 12)
                c18  = i64(f"{p}__c18", 18)
                nodes.append(oh.make_node("Less", inputs=[hour, c6],  outputs=[lt6]))
                nodes.append(oh.make_node("Less", inputs=[hour, c12], outputs=[lt12]))
                nodes.append(oh.make_node("Less", inputs=[hour, c18], outputs=[lt18]))
                z = i64(f"{p}__pod0", 0)
                one = i64(f"{p}__pod1", 1)
                two = i64(f"{p}__pod2", 2)
                thr = i64(f"{p}__pod3", 3)
                i2 = f"{p}__pod_i2"
                i1 = f"{p}__pod_i1"
                code = f"{p}__pod_code"
                nodes.append(oh.make_node("Where", inputs=[lt18, two, thr], outputs=[i2]))
                nodes.append(oh.make_node("Where", inputs=[lt12, one, i2], outputs=[i1]))
                nodes.append(oh.make_node("Where", inputs=[lt6,  z,   i1], outputs=[code]))
                nodes.append(oh.make_node(
                    "LabelEncoder", domain="ai.onnx.ml", inputs=[code], outputs=[out_name],
                    keys_int64s=[0, 1, 2, 3],
                    values_strings=["night", "morning", "afternoon", "evening"],
                    default_string="night",
                ))

            elif bin_type == "rush_hour":
                # morning_rush(7-9), evening_rush(17-19), off_peak
                ge7  = f"{p}__ge7"
                lt9  = f"{p}__lt9"
                ge17 = f"{p}__ge17"
                lt19 = f"{p}__lt19"
                is_mr = f"{p}__is_mr"
                is_er = f"{p}__is_er"
                c7   = i64(f"{p}__c7",  7)
                c9   = i64(f"{p}__c9",  9)
                c17  = i64(f"{p}__c17", 17)
                c19  = i64(f"{p}__c19", 19)
                nodes.append(oh.make_node("GreaterOrEqual", inputs=[hour, c7],  outputs=[ge7]))
                nodes.append(oh.make_node("Less",           inputs=[hour, c9],  outputs=[lt9]))
                nodes.append(oh.make_node("GreaterOrEqual", inputs=[hour, c17], outputs=[ge17]))
                nodes.append(oh.make_node("Less",           inputs=[hour, c19], outputs=[lt19]))
                nodes.append(oh.make_node("And", inputs=[ge7, lt9],   outputs=[is_mr]))
                nodes.append(oh.make_node("And", inputs=[ge17, lt19], outputs=[is_er]))
                z  = i64(f"{p}__rh0", 0)
                one = i64(f"{p}__rh1", 1)
                two = i64(f"{p}__rh2", 2)
                inner = f"{p}__rh_inner"
                code  = f"{p}__rh_code"
                nodes.append(oh.make_node("Where", inputs=[is_mr, one, z],   outputs=[inner]))
                nodes.append(oh.make_node("Where", inputs=[is_er, two, inner], outputs=[code]))
                nodes.append(oh.make_node(
                    "LabelEncoder", domain="ai.onnx.ml", inputs=[code], outputs=[out_name],
                    keys_int64s=[0, 1, 2],
                    values_strings=["off_peak", "morning_rush", "evening_rush"],
                    default_string="off_peak",
                ))

    return nodes, initializers
