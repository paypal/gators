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
OrdinalFeatures      : hour, minute, second, day_of_week, weekend,
                       month, day_of_month, day_of_year, year, quarter,
                       semester, century, week, leap_year
CyclicFeatures       : hour, minute, second, day_of_week,
                       month, quarter, semester, week, day_of_month, day_of_year
DiffFeatures         : all units (d, h, m, s) for column_pairs and reference_dates
TimeBinFeatures      : part_of_day, rush_hour, season, time_of_month, time_of_year, day

Not supported (require calendar arithmetic beyond 1970 boundary)
-----------------------------------------------------------------
TimeWindowFeatures
"""
from __future__ import annotations

from datetime import date as _date
from datetime import datetime as _datetime
from math import pi

import polars as pl

from ..feature_generation_dt.business_time_features import BusinessTimeFeatures
from ..feature_generation_dt.cyclic_features import CYCLIC_FACTORS, CyclicFeatures
from ..feature_generation_dt.diff_features import DiffFeatures
from ..feature_generation_dt.duration_to_datetime import DurationToDatetime
from ..feature_generation_dt.holiday_features import HolidayFeatures
from ..feature_generation_dt.ordinal_features import OrdinalFeatures
from ..feature_generation_dt.time_bin_features import TimeBinFeatures
from ._converters import (
    _POLARS_TO_ONNX,
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

# All ordinal/cyclic components now supported in ONNX
_EASY_COMPONENTS = frozenset({
    "century", "year", "semester", "quarter", "month", "week",
    "day_of_week", "day_of_month", "day_of_year", "weekend", "leap_year",
    "hour", "minute", "second",
})
# Cyclic components supported in ONNX
_EASY_CYCLIC = frozenset({
    "hour", "minute", "second", "day_of_week",
    "month", "quarter", "semester", "week", "day_of_month", "day_of_year",
})
# Components needing Richards calendar arithmetic
_CAL_COMPONENTS = frozenset({
    "century", "year", "semester", "quarter", "month", "week",
    "day_of_month", "day_of_year", "leap_year",
})


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
    # Not using _output_dtypes here: numeric features are hardcoded to FLOAT regardless
    # of float_datatype, while _output_dtypes declares Float64.
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
            assert per_hour is not None
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
    generated = [f"{col}__{comp}" for col in (transformer.subset or []) for comp in transformer.components]
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
    # Not using _output_dtypes here: numeric features are hardcoded to DOUBLE regardless
    # of float_datatype, while _output_dtypes declares Float64 (which would resolve to
    # UNDEFINED and defer to float_datatype).
    for dt_col in (transformer.subset or []):
        if col == dt_col:
            return TensorProto.INT64
        for comp in transformer.components:
            if col == f"{dt_col}__{comp}":
                return TensorProto.DOUBLE
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
    """Extract all ordinal components via integer arithmetic and the Richards calendar algorithm."""
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset_set = set(transformer.subset or [])
    generated_set = {f"{col}__{c}" for col in subset_set for c in transformer.components}

    for col, in_name in input_names.items():
        if col not in subset_set and col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    _time_comps = frozenset({"hour", "minute", "second", "day_of_week", "weekend"})

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        ts_in = input_names[col]
        p = f"{ts_in}__OF"
        unit = transformer._dt_units.get(col, "us")

        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[ts_in], outputs=[output_names[col]]))

        def i64(name: str, val: int) -> str:
            initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
            return name

        # Extract time-of-day components when requested
        time_needed = frozenset(c for c in transformer.components if c in _time_comps)
        comps = _extract_components(ts_in, p, unit, time_needed, nodes, initializers) if time_needed else {}

        # Extract calendar components (Richards algorithm) when requested
        cal_needed = any(c in _CAL_COMPONENTS for c in transformer.components)
        cal: dict = {}
        is_leap = doy = None
        if cal_needed:
            date_days = _get_date_days(ts_in, p, unit, nodes, initializers)
            cal = _extract_calendar(date_days, p, nodes, initializers)
            month, day, year, jdn = cal["month"], cal["day"], cal["year"], cal["jdn"]

            need_leap = any(c in {"leap_year", "day_of_year"} for c in transformer.components)
            need_doy  = "day_of_year" in transformer.components
            need_week = "week" in transformer.components
            if need_leap or need_doy or need_week:
                is_leap = _compute_leap_year(year, p, nodes, initializers)
            if need_doy or need_week:
                assert is_leap is not None
                doy = _compute_day_of_year(month, day, is_leap, p, nodes, initializers)

        for comp in transformer.components:
            out_name = output_names.get(f"{col}__{comp}", f"{col}__{comp}")

            # ── Time-of-day components (existing) ────────────────────────────
            if comp == "weekend":
                wd = comps["day_of_week"]
                eq6 = f"{p}__eq6"
                eq7 = f"{p}__eq7"
                is_wknd = f"{p}__is_wknd"
                nodes.append(oh.make_node("Equal", inputs=[wd, i64(f"{p}__six", 6)],   outputs=[eq6]))
                nodes.append(oh.make_node("Equal", inputs=[wd, i64(f"{p}__seven", 7)], outputs=[eq7]))
                nodes.append(oh.make_node("Or",    inputs=[eq6, eq7],                   outputs=[is_wknd]))
                nodes.append(oh.make_node("Cast",  inputs=[is_wknd], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp in _time_comps:
                nodes.append(oh.make_node("Cast", inputs=[comps[comp]], outputs=[out_name], to=TensorProto.DOUBLE))

            # ── Calendar components (Richards-derived) ────────────────────────
            elif comp == "month":
                nodes.append(oh.make_node("Cast", inputs=[month], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp == "day_of_month":
                nodes.append(oh.make_node("Cast", inputs=[day], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp == "year":
                nodes.append(oh.make_node("Cast", inputs=[year], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp == "quarter":
                m1 = f"{p}__q_m1"
                m1d3 = f"{p}__q_m1d3"
                q_int = f"{p}__q_int"
                nodes.append(oh.make_node("Sub", inputs=[month, i64(f"{p}__q_c1", 1)], outputs=[m1]))
                nodes.append(oh.make_node("Div", inputs=[m1, i64(f"{p}__q_c3", 3)],    outputs=[m1d3]))
                nodes.append(oh.make_node("Add", inputs=[m1d3, i64(f"{p}__q_c1b", 1)], outputs=[q_int]))
                nodes.append(oh.make_node("Cast", inputs=[q_int], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp == "semester":
                m1 = f"{p}__sem_m1"
                m1d6 = f"{p}__sem_m1d6"
                sem_int = f"{p}__sem_int"
                nodes.append(oh.make_node("Sub", inputs=[month, i64(f"{p}__sem_c1", 1)], outputs=[m1]))
                nodes.append(oh.make_node("Div", inputs=[m1, i64(f"{p}__sem_c6", 6)],    outputs=[m1d6]))
                nodes.append(oh.make_node("Add", inputs=[m1d6, i64(f"{p}__sem_c1b", 1)], outputs=[sem_int]))
                nodes.append(oh.make_node("Cast", inputs=[sem_int], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp == "century":
                # (year - 1) // 100 + 1
                ym1 = f"{p}__cen_ym1"
                ym1d100 = f"{p}__cen_ym1d100"
                cen_int = f"{p}__cen_int"
                nodes.append(oh.make_node("Sub", inputs=[year, i64(f"{p}__cen_c1", 1)],   outputs=[ym1]))
                nodes.append(oh.make_node("Div", inputs=[ym1, i64(f"{p}__cen_c100", 100)], outputs=[ym1d100]))
                nodes.append(oh.make_node("Add", inputs=[ym1d100, i64(f"{p}__cen_c1b", 1)], outputs=[cen_int]))
                nodes.append(oh.make_node("Cast", inputs=[cen_int], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp == "leap_year":
                assert is_leap is not None
                nodes.append(oh.make_node("Cast", inputs=[is_leap], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp == "day_of_year":
                assert doy is not None
                nodes.append(oh.make_node("Cast", inputs=[doy], outputs=[out_name], to=TensorProto.DOUBLE))
            elif comp == "week":
                week = _compute_week(jdn, p, nodes, initializers)
                nodes.append(oh.make_node("Cast", inputs=[week], outputs=[out_name], to=TensorProto.DOUBLE))

    return nodes, initializers


# ── CyclicFeatures ────────────────────────────────────────────────────────────

@get_output_columns.register(CyclicFeatures)
def _cf_output_columns(transformer: CyclicFeatures, input_columns: list[str]) -> list[str]:
    generated = [
        f"{col}__{comp}__sin{int(a) if a == int(a) else round(a, 2)}"
        for col in (transformer.subset or [])
        for comp in transformer.components
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
    # Not using _output_dtypes here: sine features are hardcoded to FLOAT regardless
    # of float_datatype, while _output_dtypes declares Float64.
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
    """sin(factor * component + phase) for all supported cyclic components.

    For day_of_month the factor is 2π / days_in_month (row-level), computed
    from the Richards algorithm + leap year correction.
    """
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

        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[ts_in], outputs=[output_names[col]]))

        # Determine which extractors are needed
        easy_needed = frozenset(c for c in transformer.components if c in {"hour", "minute", "second", "day_of_week"})
        cal_comps  = frozenset(c for c in transformer.components if c in _EASY_CYCLIC - {"hour", "minute", "second", "day_of_week"})

        comps: dict[str, str] = {}
        if easy_needed:
            comps = _extract_components(ts_in, p, unit, easy_needed, nodes, initializers)

        cal: dict = {}
        is_leap = month_t = day_t = doy_t = None
        if cal_comps:
            date_days = _get_date_days(ts_in, p, unit, nodes, initializers)
            cal = _extract_calendar(date_days, p, nodes, initializers)
            month_t, day_t = cal["month"], cal["day"]
            need_leap = any(c in {"day_of_month", "day_of_year"} for c in cal_comps)
            if need_leap:
                is_leap = _compute_leap_year(cal["year"], p, nodes, initializers)
            if "day_of_year" in cal_comps:
                assert is_leap is not None
                doy_t = _compute_day_of_year(month_t, day_t, is_leap, p, nodes, initializers)
            if "week" in cal_comps:
                # week is INT64; CYCLIC_FACTORS["week"] = 2π/52
                pass

        for comp in transformer.components:
            ai_list = [int(a) if a == int(a) else round(a, 2) for a in transformer.angles]

            if comp == "day_of_month":
                # sin(2π * day / days_in_month + phase) — factor is row-level
                assert month_t is not None and day_t is not None and is_leap is not None
                dim = _compute_days_in_month(month_t, is_leap, p, nodes, initializers)
                day_f  = f"{p}__cf_dom_day_f"
                dim_f  = f"{p}__cf_dom_dim_f"
                ratio  = f"{p}__cf_dom_ratio"
                tpi    = f"{p}__cf_dom_2pi"
                scaled = f"{p}__cf_dom_scaled"
                initializers.append(onnx.numpy_helper.from_array(np.array([2.0 * pi], dtype=np.float32), name=tpi))
                nodes.append(oh.make_node("Cast", inputs=[day_t], outputs=[day_f], to=TensorProto.FLOAT))
                nodes.append(oh.make_node("Cast", inputs=[dim],   outputs=[dim_f], to=TensorProto.FLOAT))
                nodes.append(oh.make_node("Div",  inputs=[day_f, dim_f], outputs=[ratio]))
                nodes.append(oh.make_node("Mul",  inputs=[ratio, tpi],   outputs=[scaled]))
                for _a_deg, a_rad, ai in zip(transformer.angles, angles_rad, ai_list, strict=False):
                    out_name = output_names.get(f"{col}__{comp}__sin{ai}", f"{col}__{comp}__sin{ai}")
                    if a_rad != 0.0:
                        phase_c  = f"{p}__cf_dom_ph_{ai}"
                        shifted  = f"{p}__cf_dom_sh_{ai}"
                        initializers.append(onnx.numpy_helper.from_array(np.array([a_rad], dtype=np.float32), name=phase_c))
                        nodes.append(oh.make_node("Add", inputs=[scaled, phase_c], outputs=[shifted]))
                        nodes.append(oh.make_node("Sin", inputs=[shifted], outputs=[out_name]))
                    else:
                        nodes.append(oh.make_node("Sin", inputs=[scaled], outputs=[out_name]))
            else:
                # Standard cyclic: sin(factor * component_int + phase)
                factor = CYCLIC_FACTORS[comp]
                if comp in {"hour", "minute", "second", "day_of_week"}:
                    comp_i64 = comps[comp]
                elif comp == "month":
                    assert month_t is not None
                    comp_i64 = month_t
                elif comp == "day_of_year":
                    assert doy_t is not None
                    comp_i64 = doy_t
                elif comp == "week":
                    comp_i64 = _compute_week(cal["jdn"], p, nodes, initializers)
                elif comp == "quarter":
                    m1 = f"{p}__cf_{comp}_m1"
                    m1d3 = f"{p}__cf_{comp}_m1d3"
                    initializers.append(onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name=f"{p}__cf_{comp}_c1"))
                    initializers.append(onnx.numpy_helper.from_array(np.array([3], dtype=np.int64), name=f"{p}__cf_{comp}_c3"))
                    assert month_t is not None
                    nodes.append(oh.make_node("Sub", inputs=[month_t, f"{p}__cf_{comp}_c1"], outputs=[m1]))
                    nodes.append(oh.make_node("Div", inputs=[m1, f"{p}__cf_{comp}_c3"],      outputs=[m1d3]))
                    qt = f"{p}__cf_qt"
                    initializers.append(onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name=f"{p}__cf_{comp}_c1b"))
                    nodes.append(oh.make_node("Add", inputs=[m1d3, f"{p}__cf_{comp}_c1b"], outputs=[qt]))
                    comp_i64 = qt
                elif comp == "semester":
                    m1 = f"{p}__cf_sem_m1"
                    m1d6 = f"{p}__cf_sem_m1d6"
                    initializers.append(onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name=f"{p}__cf_sem_c1"))
                    initializers.append(onnx.numpy_helper.from_array(np.array([6], dtype=np.int64), name=f"{p}__cf_sem_c6"))
                    assert month_t is not None
                    nodes.append(oh.make_node("Sub", inputs=[month_t, f"{p}__cf_sem_c1"], outputs=[m1]))
                    nodes.append(oh.make_node("Div", inputs=[m1, f"{p}__cf_sem_c6"],      outputs=[m1d6]))
                    sem = f"{p}__cf_sem"
                    initializers.append(onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name=f"{p}__cf_sem_c1b"))
                    nodes.append(oh.make_node("Add", inputs=[m1d6, f"{p}__cf_sem_c1b"], outputs=[sem]))
                    comp_i64 = sem
                else:  # pragma: no cover
                    continue

                comp_f = f"{p}__{comp}_f"
                nodes.append(oh.make_node("Cast", inputs=[comp_i64], outputs=[comp_f], to=TensorProto.FLOAT))
                factor_c = f"{p}__{comp}__factor"
                initializers.append(onnx.numpy_helper.from_array(np.array([factor], dtype=np.float32), name=factor_c))

                for _a_deg, a_rad, ai in zip(transformer.angles, angles_rad, ai_list, strict=False):
                    out_name = output_names.get(f"{col}__{comp}__sin{ai}", f"{col}__{comp}__sin{ai}")
                    product  = f"{p}__{comp}__prod_{ai}"
                    nodes.append(oh.make_node("Mul", inputs=[comp_f, factor_c], outputs=[product]))
                    if a_rad != 0.0:
                        phase_c = f"{p}__{comp}__phase_{ai}"
                        shifted = f"{p}__{comp}__shift_{ai}"
                        initializers.append(onnx.numpy_helper.from_array(np.array([a_rad], dtype=np.float32), name=phase_c))
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
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
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

_ALL_BINS = frozenset({"part_of_day", "rush_hour", "season", "time_of_month", "time_of_year"})


def _get_date_days(ts_in: str, p: str, unit: str, nodes: list, initializers: list) -> str:
    """Return tensor name holding days since Unix epoch (INT64).

    For pl.Date the input is already in days; for Datetime divide by ticks-per-day.
    """
    if unit == "date":
        return ts_in
    per_day = _UNIT_PER.get(unit, _UNIT_PER["us"])[1]
    upd_name = f"{p}__upd_cal"
    days_name = f"{p}__date_days"
    initializers.append(onnx.numpy_helper.from_array(np.array([per_day], dtype=np.int64), name=upd_name))
    nodes.append(oh.make_node("Div", inputs=[ts_in, upd_name], outputs=[days_name]))
    return days_name


def _extract_calendar(date_days: str, p: str, nodes: list, initializers: list) -> dict[str, str]:
    """Richards algorithm extended to return month, day, year, and JDN as INT64 tensors.

    All four are computed from shared intermediates at negligible extra cost.
    Returns dict with keys: 'month', 'day', 'year', 'jdn'.
    """
    def c(name: str, val: int) -> str:
        initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
        return name

    jdn     = f"{p}__jdn"
    a       = f"{p}__a"
    b       = f"{p}__b"
    a4      = f"{p}__a4"
    a4p3    = f"{p}__a4p3"
    b146    = f"{p}__b146"
    b146d4  = f"{p}__b146d4"
    cc      = f"{p}__cc"
    c4      = f"{p}__c4"
    c4p3    = f"{p}__c4p3"
    d       = f"{p}__d"
    d1461   = f"{p}__d1461"
    d1461d4 = f"{p}__d1461d4"
    e       = f"{p}__e"
    e5      = f"{p}__e5"
    e5p2    = f"{p}__e5p2"
    m_cal   = f"{p}__m_cal"
    m153    = f"{p}__m153"
    m153p2  = f"{p}__m153p2"
    m153d5  = f"{p}__m153d5"
    daym1   = f"{p}__daym1"
    day_cal = f"{p}__day_cal"
    m10     = f"{p}__m10"
    m12x10  = f"{p}__m12x10"
    mp3     = f"{p}__mp3"
    month   = f"{p}__month_cal"
    # year = 100*b + d - 4800 + m//10
    b100    = f"{p}__b100"
    bd      = f"{p}__bd"
    bd4800  = f"{p}__bd4800"
    year    = f"{p}__year_cal"

    nodes += [
        oh.make_node("Add", [date_days, c(f"{p}__C2440588", 2440588)], [jdn]),
        oh.make_node("Add", [jdn,       c(f"{p}__C32044",   32044)],   [a]),
        oh.make_node("Mul", [c(f"{p}__C4a",  4),      a],              [a4]),
        oh.make_node("Add", [a4,         c(f"{p}__C3b",  3)],          [a4p3]),
        oh.make_node("Div", [a4p3,       c(f"{p}__C146097", 146097)],  [b]),
        oh.make_node("Mul", [c(f"{p}__C146097c", 146097), b],          [b146]),
        oh.make_node("Div", [b146,       c(f"{p}__C4c",  4)],          [b146d4]),
        oh.make_node("Sub", [a,          b146d4],                       [cc]),
        oh.make_node("Mul", [c(f"{p}__C4d",  4),      cc],             [c4]),
        oh.make_node("Add", [c4,         c(f"{p}__C3d",  3)],          [c4p3]),
        oh.make_node("Div", [c4p3,       c(f"{p}__C1461a", 1461)],     [d]),
        oh.make_node("Mul", [c(f"{p}__C1461b", 1461), d],              [d1461]),
        oh.make_node("Div", [d1461,      c(f"{p}__C4e",  4)],          [d1461d4]),
        oh.make_node("Sub", [cc,         d1461d4],                      [e]),
        oh.make_node("Mul", [c(f"{p}__C5m",  5),      e],              [e5]),
        oh.make_node("Add", [e5,         c(f"{p}__C2m",  2)],          [e5p2]),
        oh.make_node("Div", [e5p2,       c(f"{p}__C153",  153)],       [m_cal]),
        oh.make_node("Mul", [c(f"{p}__C153d", 153), m_cal],            [m153]),
        oh.make_node("Add", [m153,       c(f"{p}__C2d",  2)],          [m153p2]),
        oh.make_node("Div", [m153p2,     c(f"{p}__C5d",  5)],          [m153d5]),
        oh.make_node("Sub", [e,          m153d5],                       [daym1]),
        oh.make_node("Add", [daym1,      c(f"{p}__C1d",  1)],          [day_cal]),
        oh.make_node("Div", [m_cal,      c(f"{p}__C10m", 10)],         [m10]),
        oh.make_node("Mul", [c(f"{p}__C12m", 12), m10],                [m12x10]),
        oh.make_node("Add", [m_cal,      c(f"{p}__C3m",  3)],          [mp3]),
        oh.make_node("Sub", [mp3,        m12x10],                       [month]),
        # year = 100*b + d - 4800 + m//10
        oh.make_node("Mul", [c(f"{p}__C100y", 100), b],                [b100]),
        oh.make_node("Add", [b100,       d],                            [bd]),
        oh.make_node("Sub", [bd,         c(f"{p}__C4800", 4800)],      [bd4800]),
        oh.make_node("Add", [bd4800,     m10],                          [year]),
    ]
    return {"month": month, "day": day_cal, "year": year, "jdn": jdn}


def _compute_leap_year(year: str, p: str, nodes: list, initializers: list) -> str:
    """Emit: (year%4==0) AND (year%100!=0 OR year%400==0). Returns BOOL tensor."""
    def c(name: str, val: int) -> str:
        initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
        return name

    mod4   = f"{p}__ly_mod4"
    mod100 = f"{p}__ly_mod100"
    mod400 = f"{p}__ly_mod400"
    div4   = f"{p}__ly_div4"
    div100 = f"{p}__ly_div100"
    div400 = f"{p}__ly_div400"
    not100 = f"{p}__ly_not100"
    or400  = f"{p}__ly_or400"
    is_leap = f"{p}__is_leap"
    nodes += [
        oh.make_node("Mod",   [year, c(f"{p}__ly_c4",   4)],   [mod4]),
        oh.make_node("Mod",   [year, c(f"{p}__ly_c100", 100)], [mod100]),
        oh.make_node("Mod",   [year, c(f"{p}__ly_c400", 400)], [mod400]),
        oh.make_node("Equal", [mod4,   c(f"{p}__ly_z4",   0)], [div4]),
        oh.make_node("Equal", [mod100, c(f"{p}__ly_z100", 0)], [div100]),
        oh.make_node("Equal", [mod400, c(f"{p}__ly_z400", 0)], [div400]),
        oh.make_node("Not",   [div100],                          [not100]),
        oh.make_node("Or",    [not100, div400],                  [or400]),
        oh.make_node("And",   [div4, or400],                     [is_leap]),
    ]
    return is_leap


def _compute_day_of_year(month: str, day: str, is_leap: str,
                          p: str, nodes: list, initializers: list) -> str:
    """Cumulative day of year (1–366): LabelEncoder for month offset + day + leap adjustment."""
    # Non-leap cumulative days before each month (1-indexed, Jan=0, Feb=31, ...)
    _CUM = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    cum_before = f"{p}__doy_cum"
    month_gt2  = f"{p}__doy_gt2"
    leap_adj   = f"{p}__doy_ladj"
    leap_adj_i = f"{p}__doy_ladji"
    doy        = f"{p}__doy"
    doy_tmp    = f"{p}__doy_tmp"
    initializers.append(onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), name=f"{p}__doy_c2"))
    nodes += [
        oh.make_node("LabelEncoder", domain="ai.onnx.ml", inputs=[month], outputs=[cum_before],
            keys_int64s=list(range(1, 13)), values_int64s=_CUM, default_int64=0),
        oh.make_node("Greater", [month, f"{p}__doy_c2"], [month_gt2]),
        oh.make_node("And",     [is_leap, month_gt2],    [leap_adj]),
        oh.make_node("Cast",    [leap_adj],               [leap_adj_i], to=TensorProto.INT64),
        oh.make_node("Add",     [cum_before, day],        [doy_tmp]),
        oh.make_node("Add",     [doy_tmp, leap_adj_i],    [doy]),
    ]
    return doy


def _compute_week(jdn: str, p: str, nodes: list, initializers: list) -> str:
    """ISO 8601 week number (1–53) via the Tondering algorithm on JDN."""
    def c(name: str, val: int) -> str:
        initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
        return name

    jmod7   = f"{p}__wk_jmod7"
    jplus   = f"{p}__wk_jplus"
    jsub7   = f"{p}__wk_jsub7"
    m6      = f"{p}__wk_m6"
    m36     = f"{p}__wk_m36"
    d4      = f"{p}__wk_d4"
    L       = f"{p}__wk_L"
    d4_L    = f"{p}__wk_d4L"
    d1mod   = f"{p}__wk_d1mod"
    d1      = f"{p}__wk_d1"
    wd7     = f"{p}__wk_d7"
    week    = f"{p}__week"
    nodes += [
        oh.make_node("Mod",  [jdn,    c(f"{p}__wk_c7",    7)],      [jmod7]),
        oh.make_node("Add",  [jdn,    c(f"{p}__wk_c31741", 31741)],  [jplus]),
        oh.make_node("Sub",  [jplus,  jmod7],                         [jsub7]),
        oh.make_node("Mod",  [jsub7,  c(f"{p}__wk_c146097", 146097)],[m6]),
        oh.make_node("Mod",  [m6,     c(f"{p}__wk_c36524",  36524)], [m36]),
        oh.make_node("Mod",  [m36,    c(f"{p}__wk_c1461",   1461)],  [d4]),
        oh.make_node("Div",  [d4,     c(f"{p}__wk_c1460",   1460)],  [L]),
        oh.make_node("Sub",  [d4,     L],                             [d4_L]),
        oh.make_node("Mod",  [d4_L,   c(f"{p}__wk_c365",    365)],   [d1mod]),
        oh.make_node("Add",  [d1mod,  L],                             [d1]),
        oh.make_node("Div",  [d1,     c(f"{p}__wk_c7b",     7)],     [wd7]),
        oh.make_node("Add",  [wd7,    c(f"{p}__wk_c1",      1)],     [week]),
    ]
    return week


def _compute_days_in_month(month: str, is_leap: str,
                            p: str, nodes: list, initializers: list) -> str:
    """Days in month (28–31): LabelEncoder for non-leap months, then fix Feb in leap years."""
    _DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dim_nl   = f"{p}__dim_nl"
    is_feb   = f"{p}__dim_is_feb"
    feb_leap = f"{p}__dim_feb_leap"
    dim      = f"{p}__dim"
    initializers.append(onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), name=f"{p}__dim_c2"))
    initializers.append(onnx.numpy_helper.from_array(np.array([29], dtype=np.int64), name=f"{p}__dim_c29"))
    nodes += [
        oh.make_node("LabelEncoder", domain="ai.onnx.ml", inputs=[month], outputs=[dim_nl],
            keys_int64s=list(range(1, 13)), values_int64s=_DAYS, default_int64=30),
        oh.make_node("Equal", [month, f"{p}__dim_c2"], [is_feb]),
        oh.make_node("And",   [is_feb, is_leap],        [feb_leap]),
        oh.make_node("Where", [feb_leap, f"{p}__dim_c29", dim_nl], [dim]),
    ]
    return dim


@get_output_columns.register(TimeBinFeatures)
def _tbf_output_columns(transformer: TimeBinFeatures, input_columns: list[str]) -> list[str]:
    generated = [f"{col}__{b}" for col in (transformer.subset or []) for b in transformer.bin_types]
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
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    for dt_col in (transformer.subset or []):
        if col == dt_col:
            return TensorProto.INT64
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
    """All five bin types via integer arithmetic.

    part_of_day / rush_hour: hour = ts // ticks_per_hour % 24
    season / time_of_month / time_of_year: month + day via Richards algorithm
      (Unix days → Gregorian calendar using only Add / Mul / Div / Sub).
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    subset_set = set(transformer.subset or [])
    all_bins = list(transformer.bin_types)
    generated_set = {f"{col}__{b}" for col in subset_set for b in all_bins}

    for col, in_name in input_names.items():
        if col not in subset_set and col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    _hour_bins = frozenset({"part_of_day", "rush_hour"})
    _cal_bins  = frozenset({"season", "time_of_month", "time_of_year"})
    need_hour = any(b in _hour_bins for b in all_bins)
    need_cal  = any(b in _cal_bins  for b in all_bins)
    need_day  = "day" in all_bins

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        ts_in = input_names[col]
        p = f"{ts_in}__TBF"
        unit = transformer._dt_units.get(col, "us") if hasattr(transformer, "_dt_units") else "us"

        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[ts_in], outputs=[output_names[col]]))

        def i64(name: str, val: int) -> str:
            initializers.append(onnx.numpy_helper.from_array(np.array([val], dtype=np.int64), name=name))
            return name

        # ── Hour extraction (part_of_day, rush_hour) ──────────────────────────
        hour = None
        if need_hour:
            comps = _extract_components(ts_in, p, unit, frozenset({"hour"}), nodes, initializers)
            hour = comps["hour"]

        # ── Calendar extraction (season, time_of_month, time_of_year) ─────────
        month = day = None
        date_days = None
        if need_cal:
            date_days = _get_date_days(ts_in, p, unit, nodes, initializers)
            _cal = _extract_calendar(date_days, p, nodes, initializers)
            month, day = _cal["month"], _cal["day"]

        # ── Day-of-week extraction (day) ───────────────────────────────────────
        # ISO weekday (1=Monday..7=Sunday): Jan 1 1970 (day 0 since epoch) was a
        # Thursday, so weekday = ((date_days + 3) % 7) + 1.
        weekday = None
        if need_day:
            if date_days is None:
                date_days = _get_date_days(ts_in, p, unit, nodes, initializers)
            wd_mod = f"{p}__wd_mod"
            weekday = f"{p}__weekday"
            nodes.append(oh.make_node("Add", inputs=[date_days, i64(f"{p}__wd_c3", 3)], outputs=[wd_mod]))
            nodes.append(oh.make_node("Mod", inputs=[wd_mod, i64(f"{p}__wd_c7", 7)], outputs=[f"{wd_mod}_r"]))
            nodes.append(oh.make_node("Add", inputs=[f"{wd_mod}_r", i64(f"{p}__wd_c1", 1)], outputs=[weekday]))

        # ── Emit one block per requested bin type ─────────────────────────────
        for bin_type in all_bins:
            out_name = output_names.get(f"{col}__{bin_type}", f"{col}__{bin_type}")

            if bin_type == "part_of_day":
                assert hour is not None
                lt6  = f"{p}__lt6"
                lt12 = f"{p}__lt12"
                lt18 = f"{p}__lt18"
                nodes.append(oh.make_node("Less", inputs=[hour, i64(f"{p}__c6",  6)],  outputs=[lt6]))
                nodes.append(oh.make_node("Less", inputs=[hour, i64(f"{p}__c12", 12)], outputs=[lt12]))
                nodes.append(oh.make_node("Less", inputs=[hour, i64(f"{p}__c18", 18)], outputs=[lt18]))
                i2 = f"{p}__pod_i2"
                i1 = f"{p}__pod_i1"
                code = f"{p}__pod_code"
                nodes.append(oh.make_node("Where", inputs=[lt18, i64(f"{p}__pod2", 2), i64(f"{p}__pod3", 3)], outputs=[i2]))
                nodes.append(oh.make_node("Where", inputs=[lt12, i64(f"{p}__pod1", 1), i2], outputs=[i1]))
                nodes.append(oh.make_node("Where", inputs=[lt6,  i64(f"{p}__pod0", 0), i1], outputs=[code]))
                nodes.append(oh.make_node("LabelEncoder", domain="ai.onnx.ml", inputs=[code], outputs=[out_name],
                    keys_int64s=[0, 1, 2, 3], values_strings=["night", "morning", "afternoon", "evening"], default_string="night"))

            elif bin_type == "rush_hour":
                assert hour is not None
                ge7 = f"{p}__ge7"
                lt9 = f"{p}__lt9"
                ge17 = f"{p}__ge17"
                lt19 = f"{p}__lt19"
                is_mr = f"{p}__is_mr"
                is_er = f"{p}__is_er"
                nodes.append(oh.make_node("GreaterOrEqual", inputs=[hour, i64(f"{p}__c7",  7)],  outputs=[ge7]))
                nodes.append(oh.make_node("Less",           inputs=[hour, i64(f"{p}__c9",  9)],  outputs=[lt9]))
                nodes.append(oh.make_node("GreaterOrEqual", inputs=[hour, i64(f"{p}__c17", 17)], outputs=[ge17]))
                nodes.append(oh.make_node("Less",           inputs=[hour, i64(f"{p}__c19", 19)], outputs=[lt19]))
                nodes.append(oh.make_node("And", inputs=[ge7, lt9],   outputs=[is_mr]))
                nodes.append(oh.make_node("And", inputs=[ge17, lt19], outputs=[is_er]))
                inner = f"{p}__rh_inner"
                code = f"{p}__rh_code"
                nodes.append(oh.make_node("Where", inputs=[is_mr, i64(f"{p}__rh1", 1), i64(f"{p}__rh0", 0)], outputs=[inner]))
                nodes.append(oh.make_node("Where", inputs=[is_er, i64(f"{p}__rh2", 2), inner], outputs=[code]))
                nodes.append(oh.make_node("LabelEncoder", domain="ai.onnx.ml", inputs=[code], outputs=[out_name],
                    keys_int64s=[0, 1, 2], values_strings=["off_peak", "morning_rush", "evening_rush"], default_string="off_peak"))

            elif bin_type == "season":
                # Classify month into winter/spring/summer/fall; supports both hemispheres
                assert month is not None
                hemi = transformer.hemisphere
                lt3  = f"{p}__slt3"
                eq12 = f"{p}__seq12"
                ge3  = f"{p}__sge3"
                lt6  = f"{p}__slt6"
                ge6  = f"{p}__sge6"
                lt9  = f"{p}__slt9"
                ge9  = f"{p}__sge9"
                lt12 = f"{p}__slt12"
                nodes.append(oh.make_node("Less",           inputs=[month, i64(f"{p}__sc3",   3)],  outputs=[lt3]))
                nodes.append(oh.make_node("Equal",          inputs=[month, i64(f"{p}__sc12m", 12)], outputs=[eq12]))
                nodes.append(oh.make_node("GreaterOrEqual", inputs=[month, i64(f"{p}__sc3b",  3)],  outputs=[ge3]))
                nodes.append(oh.make_node("Less",           inputs=[month, i64(f"{p}__sc6",   6)],  outputs=[lt6]))
                nodes.append(oh.make_node("GreaterOrEqual", inputs=[month, i64(f"{p}__sc6b",  6)],  outputs=[ge6]))
                nodes.append(oh.make_node("Less",           inputs=[month, i64(f"{p}__sc9",   9)],  outputs=[lt9]))
                nodes.append(oh.make_node("GreaterOrEqual", inputs=[month, i64(f"{p}__sc9b",  9)],  outputs=[ge9]))
                nodes.append(oh.make_node("Less",           inputs=[month, i64(f"{p}__sc12",  12)], outputs=[lt12]))
                # Boolean groups
                is_jan_feb = lt3
                is_dec     = eq12
                is_win_n = f"{p}__s_win_n"
                is_spr_n = f"{p}__s_spr_n"
                is_sum_n = f"{p}__s_sum_n"
                is_spr_s = f"{p}__s_spr_s"
                nodes.append(oh.make_node("Or",  inputs=[is_jan_feb, is_dec],   outputs=[is_win_n]))  # [12,1,2]
                nodes.append(oh.make_node("And", inputs=[ge3, lt6],             outputs=[is_spr_n]))  # [3,4,5]
                nodes.append(oh.make_node("And", inputs=[ge6, lt9],             outputs=[is_sum_n]))  # [6,7,8]
                nodes.append(oh.make_node("And", inputs=[ge9, lt12],            outputs=[is_spr_s]))  # [9,10,11]
                inner1 = f"{p}__s_i1"
                inner2 = f"{p}__s_i2"
                code = f"{p}__s_code"
                if hemi == "northern":
                    # fall=3 default, summer=2, spring=1, winter=0
                    nodes.append(oh.make_node("Where", inputs=[is_sum_n, i64(f"{p}__s2", 2), i64(f"{p}__s3", 3)], outputs=[inner1]))
                    nodes.append(oh.make_node("Where", inputs=[is_spr_n, i64(f"{p}__s1", 1), inner1], outputs=[inner2]))
                    nodes.append(oh.make_node("Where", inputs=[is_win_n, i64(f"{p}__s0", 0), inner2], outputs=[code]))
                else:  # southern: [6,7,8]=winter, [9,10,11]=spring, [12,1,2]=summer, [3,4,5]=fall
                    nodes.append(oh.make_node("Where", inputs=[is_sum_n, i64(f"{p}__s0s", 0), i64(f"{p}__s3s", 3)], outputs=[inner1]))
                    nodes.append(oh.make_node("Where", inputs=[is_spr_s, i64(f"{p}__s1s", 1), inner1], outputs=[inner2]))
                    nodes.append(oh.make_node("Where", inputs=[is_win_n, i64(f"{p}__s2s", 2), inner2], outputs=[code]))
                nodes.append(oh.make_node("LabelEncoder", domain="ai.onnx.ml", inputs=[code], outputs=[out_name],
                    keys_int64s=[0, 1, 2, 3], values_strings=["winter", "spring", "summer", "fall"], default_string="fall"))

            elif bin_type == "time_of_month":
                # beginning (day ≤ 10) / middle (day ≤ 20) / end (day > 20)
                assert day is not None
                lt11 = f"{p}__tm_lt11"
                lt21 = f"{p}__tm_lt21"
                inner = f"{p}__tm_inner"
                code = f"{p}__tm_code"
                nodes.append(oh.make_node("Less", inputs=[day, i64(f"{p}__tc11", 11)], outputs=[lt11]))
                nodes.append(oh.make_node("Less", inputs=[day, i64(f"{p}__tc21", 21)], outputs=[lt21]))
                nodes.append(oh.make_node("Where", inputs=[lt21, i64(f"{p}__tm1", 1), i64(f"{p}__tm2", 2)], outputs=[inner]))
                nodes.append(oh.make_node("Where", inputs=[lt11, i64(f"{p}__tm0", 0), inner], outputs=[code]))
                nodes.append(oh.make_node("LabelEncoder", domain="ai.onnx.ml", inputs=[code], outputs=[out_name],
                    keys_int64s=[0, 1, 2], values_strings=["beginning", "middle", "end"], default_string="end"))

            elif bin_type == "time_of_year":
                # early (month ≤ 4) / mid (month ≤ 8) / late (month > 8)
                assert month is not None
                lt5 = f"{p}__ty_lt5"
                lt9 = f"{p}__ty_lt9"
                inner = f"{p}__ty_inner"
                code = f"{p}__ty_code"
                nodes.append(oh.make_node("Less", inputs=[month, i64(f"{p}__tyc5", 5)], outputs=[lt5]))
                nodes.append(oh.make_node("Less", inputs=[month, i64(f"{p}__tyc9", 9)], outputs=[lt9]))
                nodes.append(oh.make_node("Where", inputs=[lt9,  i64(f"{p}__ty1", 1), i64(f"{p}__ty2", 2)], outputs=[inner]))
                nodes.append(oh.make_node("Where", inputs=[lt5,  i64(f"{p}__ty0", 0), inner], outputs=[code]))
                nodes.append(oh.make_node("LabelEncoder", domain="ai.onnx.ml", inputs=[code], outputs=[out_name],
                    keys_int64s=[0, 1, 2], values_strings=["early", "mid", "late"], default_string="late"))

            elif bin_type == "day":
                assert weekday is not None
                nodes.append(oh.make_node("LabelEncoder", domain="ai.onnx.ml", inputs=[weekday], outputs=[out_name],
                    keys_int64s=[1, 2, 3, 4, 5, 6, 7],
                    values_strings=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    default_string="Monday"))

    return nodes, initializers


# ── DurationToDatetime ────────────────────────────────────────────────────────

# How many ticks of each Polars physical time unit fit in one offset unit.
# Rows are the offset unit (transformer.unit), columns are the reference/output
# time unit (from _input_dtypes for column-based references, or 'us' for fixed).
_OFFSET_TO_TICKS: dict[str, dict[str, int | None]] = {
    "s":  {"us": 1_000_000,          "ms": 1_000,         "ns": 1_000_000_000},
    "m":  {"us": 60_000_000,         "ms": 60_000,        "ns": 60_000_000_000},
    "h":  {"us": 3_600_000_000,      "ms": 3_600_000,     "ns": 3_600_000_000_000},
    "d":  {"us": 86_400_000_000,     "ms": 86_400_000,    "ns": 86_400_000_000_000},
    "ms": {"us": 1_000,              "ms": 1,              "ns": 1_000_000},
    "us": {"us": 1,                  "ms": None,            "ns": 1_000},
}

# Epoch constant used for timezone-agnostic UTC conversion
_EPOCH = _datetime(1970, 1, 1)


def _ref_dt_to_ticks(ref_dt: _datetime, time_unit: str) -> int:
    """Convert a naive datetime to physical int64 ticks (UTC, no local-timezone offset)."""
    total_seconds = (ref_dt - _EPOCH).total_seconds()
    scale = {"us": 1_000_000, "ms": 1_000, "ns": 1_000_000_000}[time_unit]
    return int(total_seconds * scale)


@get_output_columns.register(DurationToDatetime)
def _dtd_output_columns(transformer: DurationToDatetime, input_columns: list[str]) -> list[str]:
    generated = [f"{col}__datetime" for col in transformer.subset]
    if transformer.drop_columns:
        dropped = set(transformer.subset)
        return [c for c in input_columns if c not in dropped] + generated
    return list(input_columns) + generated


@get_input_onnx_type.register(DurationToDatetime)
def _dtd_input_type(transformer: DurationToDatetime, col: str) -> int:
    # Numeric offset columns are int64
    if col in transformer.subset:
        return TensorProto.INT64
    # Column-based reference date is a datetime → physical int64
    if transformer._is_column_reference and col == transformer.reference_date:
        return TensorProto.INT64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(DurationToDatetime)
def _dtd_output_type(transformer: DurationToDatetime, col: str) -> int:
    declared = resolve_declared_output_dtype(transformer, col)
    if declared is not None:
        return declared
    # Offset columns passed through (drop_columns=False)
    if col in transformer.subset:
        return TensorProto.INT64
    # Column-based reference date passed through
    if transformer._is_column_reference and col == transformer.reference_date:
        return TensorProto.INT64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@to_onnx_nodes.register(DurationToDatetime)
def _dtd_to_onnx_nodes(
    transformer: DurationToDatetime,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Convert numeric offsets to datetime represented as int64 physical ticks.

    result = ref_ticks + offset * ticks_per_offset_unit

    For fixed reference dates the ref constant is computed as UTC ticks since epoch.
    For column-based references the reference column's physical time unit is read
    from _input_dtypes so the factor is scaled accordingly.
    """
    from datetime import datetime as _datetime

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    subset_set = set(transformer.subset)
    generated_set = {f"{col}__datetime" for col in transformer.subset}

    # Pass through non-subset, non-generated columns
    for col, in_name in input_names.items():
        if col not in subset_set and col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    # Pass through offset columns when drop_columns=False
    if not transformer.drop_columns:
        for col in transformer.subset:
            if col in input_names and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))

    # Determine reference: constant int64 or input column, and the physical time unit
    if transformer._is_column_reference:
        ref_col = transformer.reference_date
        assert isinstance(ref_col, str)
        ref_input = input_names.get(ref_col, ref_col)
        ref_dtype = (getattr(transformer, "_input_dtypes", {}) or {}).get(ref_col)
        ref_time_unit = getattr(ref_dtype, "time_unit", "us") or "us"
    else:
        # Fixed datetime → always produces datetime[μs] (pl.lit default)
        ref_time_unit = "us"
        ref_dt = (
            transformer.reference_date
            if isinstance(transformer.reference_date, _datetime)
            else _datetime.fromisoformat(transformer.reference_date)
        )
        ref_ticks = _ref_dt_to_ticks(ref_dt, ref_time_unit)
        ref_const = "__dtd_ref_ticks"
        initializers.append(
            onnx.numpy_helper.from_array(np.array([ref_ticks], dtype=np.int64), name=ref_const)
        )

    # Unit conversion factor: offset_unit → ref physical time unit
    factor = _OFFSET_TO_TICKS[transformer.unit].get(ref_time_unit)
    if factor is None:
        msg = (
            f"DurationToDatetime: cannot convert offset unit '{transformer.unit}' to "
            f"reference time unit '{ref_time_unit}' without fractional precision loss."
        )
        if errors == "raise":
            raise OnnxNotSupportedError(msg)
        # Emit each offset column as a stand-in for its datetime output so the graph stays valid.
        for col in transformer.subset:
            if col in input_names:
                dt_out = output_names.get(f"{col}__datetime", f"{col}__datetime")
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[dt_out]))
        return nodes, initializers

    factor_name = f"__dtd_factor_{transformer.unit}_{ref_time_unit}"
    initializers.append(
        onnx.numpy_helper.from_array(np.array([factor], dtype=np.int64), name=factor_name)
    )

    for col in transformer.subset:
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        out_name = output_names.get(f"{col}__datetime", f"{col}__datetime")
        p = f"{in_name}__DTD"

        if factor == 1:
            offset_us = in_name
        else:
            offset_us = f"{p}__offset_ticks"
            nodes.append(oh.make_node("Mul", inputs=[in_name, factor_name], outputs=[offset_us]))

        ref = ref_const if not transformer._is_column_reference else ref_input
        nodes.append(oh.make_node("Add", inputs=[ref, offset_us], outputs=[out_name]))

    return nodes, initializers


# ── HolidayFeatures ───────────────────────────────────────────────────────────

# Physical ticks per day for each Polars time unit (Date = already in days)
_US_PER_DAY: dict[str, int | None] = {
    "us": 86_400_000_000,
    "ms": 86_400_000,
    "ns": 86_400_000_000_000,
    "date": None,  # Date physical = days already, no division needed
}
# Sentinel: no holiday found in that direction → output -1 (mirrors Polars fill_null(-1))
_HF_BIG = np.iinfo(np.int64).max // 2
_HOLIDAY_EPOCH = _date(1970, 1, 1)


def _holiday_days_array(transformer: HolidayFeatures) -> np.ndarray:
    """Return sorted holiday dates as days-since-epoch int64 array shaped [1, M]."""
    return np.array(
        sorted((d - _HOLIDAY_EPOCH).days for d in transformer._holidays),
        dtype=np.int64,
    ).reshape(1, -1)


@get_output_columns.register(HolidayFeatures)
def _hf_output_columns(transformer: HolidayFeatures, input_columns: list[str]) -> list[str]:
    dist_feats = {"nearest_holiday_distance", "days_to_holiday", "days_from_holiday"}
    generated = [
        f"{col}__{feat}"
        for col in (transformer.subset or [])
        for feat in transformer.features
        # distance features are only generated when holidays were fitted (mirrors Polars)
        if feat not in dist_feats or transformer._holidays
    ]
    if transformer.drop_columns:
        dropped = set(transformer.subset or [])
        return [c for c in input_columns if c not in dropped] + generated
    return list(input_columns) + generated


@get_input_onnx_type.register(HolidayFeatures)
def _hf_input_type(transformer: HolidayFeatures, col: str) -> int:
    if col in (transformer.subset or []):
        return TensorProto.INT64  # datetime/date → physical int64
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@get_output_onnx_type.register(HolidayFeatures)
def _hf_output_type(transformer: HolidayFeatures, col: str) -> int:
    # Not using _output_dtypes here: features are hardcoded to DOUBLE regardless of
    # float_datatype, while _output_dtypes declares Float64.
    subset = set(transformer.subset or [])
    for dt_col in subset:
        if col == dt_col:
            return TensorProto.INT64
        if col == f"{dt_col}__is_holiday":
            return TensorProto.DOUBLE
        for feat in ("days_to_holiday", "days_from_holiday", "nearest_holiday_distance"):
            if col == f"{dt_col}__{feat}":
                return TensorProto.DOUBLE
    dtype = getattr(transformer, "_input_dtypes", {}).get(col)
    if dtype in (pl.String, pl.Utf8):
        return TensorProto.STRING
    return _POLARS_TO_ONNX.get(dtype, TensorProto.FLOAT)


@to_onnx_nodes.register(HolidayFeatures)
def _hf_to_onnx_nodes(
    transformer: HolidayFeatures,
    input_names: dict[str, str],
    output_names: dict[str, str],
    errors: str = "raise",
) -> tuple[list, list]:
    """Holiday feature extraction via broadcast arithmetic on pre-computed holiday days.

    Algorithm for each datetime column (fed as int64 physical ticks):
        date_days [N, 1]  =  Div(ts, ticks_per_day)  then Unsqueeze
        holidays  [1, M]  =  constant array of holiday days since epoch
        diff      [N, M]  =  Sub(date_days, holidays)

    Then per feature:
        is_holiday               = ReduceMin(Abs(diff), axis=1) == 0
        nearest_holiday_distance = ReduceMin(Abs(diff), axis=1)
        days_to_holiday          = ReduceMin(Where(neg_diff < 0, BIG, neg_diff), axis=1);  -1 if BIG
        days_from_holiday        = ReduceMin(Where(diff < 0, BIG, diff), axis=1);           -1 if BIG
    """
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    subset_set = set(transformer.subset or [])
    all_feats = set(transformer.features)
    generated_set = {f"{col}__{feat}" for col in subset_set for feat in all_feats}

    # Passthrough columns
    for col, in_name in input_names.items():
        if col not in subset_set and col not in generated_set and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

    if not transformer._holidays:
        # No holidays fitted: is_holiday → all False; distance features not generated (mirrors Polars)
        for col in (transformer.subset or []):
            if col not in input_names:  # pragma: no cover
                continue
            if not transformer.drop_columns and col in output_names:
                nodes.append(oh.make_node("Identity", inputs=[input_names[col]], outputs=[output_names[col]]))
            if "is_holiday" in all_feats:
                # Sub(ts, ts) = all zeros; Cast(zeros, BOOL) then Cast to DOUBLE = all 0.0
                zeros = f"__hf_{col}__zeros"
                zeros_bool = f"__hf_{col}__zeros_bool"
                out_name = output_names.get(f"{col}__is_holiday", f"{col}__is_holiday")
                nodes.append(oh.make_node("Sub", inputs=[input_names[col], input_names[col]], outputs=[zeros]))
                nodes.append(oh.make_node("Cast", inputs=[zeros], outputs=[zeros_bool], to=TensorProto.BOOL))
                nodes.append(oh.make_node("Cast", inputs=[zeros_bool], outputs=[out_name], to=TensorProto.DOUBLE))
        return nodes, initializers

    # Shared constants (added once, reused across all subset columns)
    ax1_name  = "__hf_axes1"
    zero_name = "__hf_zero"
    big_name  = "__hf_big"
    neg1_name = "__hf_neg1"
    initializers += [
        onnx.numpy_helper.from_array(np.array([1],         dtype=np.int64), name=ax1_name),
        onnx.numpy_helper.from_array(np.array([0],         dtype=np.int64), name=zero_name),
        onnx.numpy_helper.from_array(np.array([_HF_BIG],   dtype=np.int64), name=big_name),
        onnx.numpy_helper.from_array(np.array([-1],        dtype=np.int64), name=neg1_name),
    ]

    holiday_arr = _holiday_days_array(transformer)  # [1, M]

    for col in (transformer.subset or []):
        if col not in input_names:  # pragma: no cover
            continue
        in_name = input_names[col]
        p = f"{in_name}__HF"

        if not transformer.drop_columns and col in output_names:
            nodes.append(oh.make_node("Identity", inputs=[in_name], outputs=[output_names[col]]))

        # Determine ticks-per-day divisor from fitted dtype
        dtype = (getattr(transformer, "_input_dtypes", {}) or {}).get(col)
        if dtype == pl.Date:
            tpd = None  # Date physical is already days since epoch
        else:
            time_unit = getattr(dtype, "time_unit", None) or "us"
            tpd = _US_PER_DAY.get(time_unit, _US_PER_DAY["us"])

        # Convert ticks → days
        if tpd is not None:
            tpd_name = f"{p}__tpd"
            initializers.append(onnx.numpy_helper.from_array(np.array([tpd], dtype=np.int64), name=tpd_name))
            days_name = f"{p}__days"
            nodes.append(oh.make_node("Div", inputs=[in_name, tpd_name], outputs=[days_name]))
        else:
            days_name = in_name  # Date is already in days

        # holiday array constant (unique per column to avoid name collision)
        hdays_name = f"{p}__hdays"
        initializers.append(onnx.numpy_helper.from_array(holiday_arr, name=hdays_name))

        # date_days [N, 1]
        days_col = f"{p}__days_col"
        nodes.append(oh.make_node("Unsqueeze", inputs=[days_name, ax1_name], outputs=[days_col]))

        # diff [N, M] = date_days - holiday_days  (positive = day is AFTER the holiday)
        diff = f"{p}__diff"
        nodes.append(oh.make_node("Sub", inputs=[days_col, hdays_name], outputs=[diff]))

        # abs_diff for nearest and is_holiday
        if "nearest_holiday_distance" in all_feats or "is_holiday" in all_feats:
            abs_diff = f"{p}__abs_diff"
            nearest  = f"{p}__nearest"
            nodes.append(oh.make_node("Abs",       inputs=[diff], outputs=[abs_diff]))
            nodes.append(oh.make_node("ReduceMin", inputs=[abs_diff, ax1_name], outputs=[nearest], keepdims=0))

            if "is_holiday" in all_feats:
                out_name = output_names.get(f"{col}__is_holiday", f"{col}__is_holiday")
                eq_out = f"{p}__is_holiday_bool"
                nodes.append(oh.make_node("Equal", inputs=[nearest, zero_name], outputs=[eq_out]))
                nodes.append(oh.make_node("Cast", inputs=[eq_out], outputs=[out_name], to=TensorProto.DOUBLE))

            if "nearest_holiday_distance" in all_feats:
                out_name = output_names.get(f"{col}__nearest_holiday_distance", f"{col}__nearest_holiday_distance")
                nodes.append(oh.make_node("Cast", inputs=[nearest], outputs=[out_name], to=TensorProto.DOUBLE))

        # days_to_holiday: min of (holiday - date) for holidays in the future (neg_diff >= 0)
        if "days_to_holiday" in all_feats:
            neg_diff        = f"{p}__neg_diff"
            neg_diff_lt0    = f"{p}__neg_lt0"
            neg_diff_masked = f"{p}__neg_masked"
            to_raw          = f"{p}__to_raw"
            to_is_big       = f"{p}__to_is_big"
            out_name = output_names.get(f"{col}__days_to_holiday", f"{col}__days_to_holiday")
            to_int_out = f"{p}__to_int"
            nodes += [
                oh.make_node("Neg",       inputs=[diff],                           outputs=[neg_diff]),
                oh.make_node("Less",      inputs=[neg_diff, zero_name],            outputs=[neg_diff_lt0]),
                oh.make_node("Where",     inputs=[neg_diff_lt0, big_name, neg_diff], outputs=[neg_diff_masked]),
                oh.make_node("ReduceMin", inputs=[neg_diff_masked, ax1_name], outputs=[to_raw], keepdims=0),
                oh.make_node("Equal",     inputs=[to_raw, big_name],               outputs=[to_is_big]),
                oh.make_node("Where",     inputs=[to_is_big, neg1_name, to_raw],   outputs=[to_int_out]),
                oh.make_node("Cast",      inputs=[to_int_out], outputs=[out_name], to=TensorProto.DOUBLE),
            ]

        # days_from_holiday: min of (date - holiday) for holidays in the past (diff >= 0)
        if "days_from_holiday" in all_feats:
            diff_lt0       = f"{p}__diff_lt0"
            diff_masked    = f"{p}__diff_masked"
            from_raw       = f"{p}__from_raw"
            from_is_big    = f"{p}__from_is_big"
            out_name = output_names.get(f"{col}__days_from_holiday", f"{col}__days_from_holiday")
            from_int_out = f"{p}__from_int"
            nodes += [
                oh.make_node("Less",      inputs=[diff, zero_name],              outputs=[diff_lt0]),
                oh.make_node("Where",     inputs=[diff_lt0, big_name, diff],     outputs=[diff_masked]),
                oh.make_node("ReduceMin", inputs=[diff_masked, ax1_name], outputs=[from_raw], keepdims=0),
                oh.make_node("Equal",     inputs=[from_raw, big_name],           outputs=[from_is_big]),
                oh.make_node("Where",     inputs=[from_is_big, neg1_name, from_raw], outputs=[from_int_out]),
                oh.make_node("Cast",      inputs=[from_int_out], outputs=[out_name], to=TensorProto.DOUBLE),
            ]

    return nodes, initializers
