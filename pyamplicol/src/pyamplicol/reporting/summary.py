# SPDX-License-Identifier: 0BSD
"""Compact PrettyTable summaries for human-facing CLI results."""

from __future__ import annotations

import importlib
import math
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pyamplicol.api.results import (
        BenchmarkComponentTiming,
        BenchmarkProfileCounters,
        BenchmarkResult,
        BenchmarkStatistics,
        ResolvedEvaluation,
    )


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(entry) for key, entry in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(entry) for entry in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _value_text(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        if not value:
            return "none"
        scalar_items = tuple(
            (str(key), entry)
            for key, entry in value.items()
            if not isinstance(entry, (Mapping, list, tuple))
        )
        if len(scalar_items) == len(value) and len(value) <= 4:
            return "; ".join(
                f"{key}={_value_text(entry)}" for key, entry in scalar_items
            )
        labels = ", ".join(str(key).replace("_", " ") for key in value)
        return f"{len(value)} fields ({labels})"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = tuple(value)
        if not entries:
            return "none"
        if all(not isinstance(entry, (Mapping, list, tuple)) for entry in entries):
            return ", ".join(_value_text(entry) for entry in entries)
        return f"{len(entries)} entries"
    return str(value)


def _evaluation_scalar_text(value: Decimal | float | int) -> str:
    if value == 0:
        return "0"
    scientific = f"{value:.15e}"
    if "e" not in scientific:
        return scientific
    mantissa, raw_exponent = scientific.rsplit("e", maxsplit=1)
    return f"{mantissa}e{int(raw_exponent):+03d}"


def _evaluation_value_text(value: complex | Decimal | float | int) -> str:
    if isinstance(value, complex):
        real = _evaluation_scalar_text(value.real)
        if value.imag == 0.0:
            return real
        imaginary = f"{_evaluation_scalar_text(abs(value.imag))}j"
        if value.real == 0.0:
            return imaginary if value.imag > 0.0 else f"-{imaginary}"
        sign = "+" if value.imag > 0.0 else "-"
        return f"{real}{sign}{imaginary}"
    return _evaluation_scalar_text(value)


def _is_evaluation_value(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(
        value, (complex, Decimal, float, int)
    )


def _total_evaluation_summary(
    values: tuple[object, ...],
    *,
    prettytable: Any,
    color: bool,
    title: str = "Summed Evaluation",
) -> str:
    table = prettytable.PrettyTable(("point", "matrix element"))
    table.title = _paint(title, "CYAN", enabled=color)
    table.align["point"] = "r"
    table.align["matrix element"] = "r"
    table.hrules = prettytable.HRuleStyle.FRAME
    for point, raw in enumerate(values):
        if not _is_evaluation_value(raw):
            raise TypeError("evaluation values must be numeric")
        value = cast(complex | Decimal | float | int, raw)
        table.add_row(
            (
                point,
                _paint(
                    _evaluation_value_text(value),
                    "GREEN",
                    enabled=color,
                ),
            )
        )
    return cast(str, table.get_string())


def _resolved_evaluation_summary(
    evaluation: ResolvedEvaluation,
    *,
    prettytable: Any,
    color: bool,
) -> str:
    component_table = prettytable.PrettyTable(
        ("point", "helicity", "color", "matrix element")
    )
    component_table.title = _paint(
        f"Resolved Evaluation ({evaluation.color_accuracy.upper()})",
        "CYAN",
        enabled=color,
    )
    component_table.align["point"] = "r"
    component_table.align["helicity"] = "l"
    component_table.align["color"] = "l"
    component_table.align["matrix element"] = "r"
    component_table.hrules = prettytable.HRuleStyle.FRAME
    for point, helicities in enumerate(evaluation.values):
        for helicity, colors in zip(
            evaluation.helicity_ids,
            helicities,
            strict=True,
        ):
            for color_id, value in zip(
                evaluation.color_ids,
                colors,
                strict=True,
            ):
                component_table.add_row(
                    (
                        point,
                        _paint(helicity, "CYAN", enabled=color),
                        _paint(color_id, "MAGENTA", enabled=color),
                        _paint(
                            _evaluation_value_text(value),
                            "GREEN",
                            enabled=color,
                        ),
                    )
                )
    total_table = _total_evaluation_summary(
        cast(tuple[object, ...], evaluation.total()),
        prettytable=prettytable,
        color=color,
        title="Resolved Sum",
    )
    return f"{component_table.get_string()}\n\n{total_table}"


def _colored(text: str, *, key: str, enabled: bool) -> str:
    if not enabled:
        return text
    normalized = text.casefold()
    try:
        colorama: Any = importlib.import_module("colorama")
    except ImportError:
        return text
    if normalized in {
        "yes",
        "ok",
        "pass",
        "passed",
        "accepted",
        "complete",
        "completed",
        "valid",
        "verified",
    }:
        color = colorama.Fore.GREEN
    elif normalized in {"no", "fail", "failed", "error", "invalid"}:
        color = colorama.Fore.RED
    elif normalized == "warning" or key.casefold() in {
        "warning",
        "warnings",
        "adjustments",
    }:
        color = colorama.Fore.YELLOW
    else:
        return text
    return f"{color}{text}{colorama.Style.RESET_ALL}"


def _paint(
    text: str, color_name: str | None, *, enabled: bool, bold: bool = False
) -> str:
    """Colour ``text``; ``None`` keeps the terminal colour, for bold-only text."""

    if not enabled or not text or (color_name is None and not bold):
        return text
    try:
        colorama: Any = importlib.import_module("colorama")
    except ImportError:
        return text
    style = colorama.Style.BRIGHT if bold else ""
    fore = "" if color_name is None else getattr(colorama.Fore, color_name)
    return f"{style}{fore}{text}{colorama.Style.RESET_ALL}"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _byte_size(value: object) -> str:
    size = _integer(value, "payload size")
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(size)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable byte-size unit")


def _component_count(process: Mapping[str, object], prefix: str) -> str:
    physical = _integer(process[f"physical_{prefix}"], f"physical {prefix}")
    computed = _integer(process[f"computed_{prefix}"], f"computed {prefix}")
    if physical == computed:
        return str(physical)
    return f"{physical} ({computed} eval.)"


def _duration_unit(seconds: float) -> tuple[float, str]:
    magnitude = abs(seconds)
    if magnitude < 1.0e-6:
        return 1.0e9, "ns"
    if magnitude < 1.0e-3:
        return 1.0e6, "us"
    if magnitude < 1.0:
        return 1.0e3, "ms"
    return 1.0, "s"


def _timing_text(
    seconds: float,
    uncertainty: BenchmarkStatistics,
) -> str:
    scale, unit = _duration_unit(seconds)
    return (
        f"{seconds * scale:.6g} +/- "
        f"{uncertainty.standard_error * scale:.3g} {unit}/point (standard error)"
    )


def _benchmark_timing_text(
    seconds: float,
    uncertainty: BenchmarkStatistics,
    *,
    sample_count: int,
) -> str:
    if sample_count > 1:
        return _timing_text(seconds, uncertainty)
    return f"{_seconds_text(seconds)}/point (uncertainty needs at least 2 blocks)"


def _seconds_text(seconds: float) -> str:
    scale, unit = _duration_unit(seconds)
    return f"{seconds * scale:.6g} {unit}"


def _component_timing_text(timing: BenchmarkComponentTiming | None) -> str:
    if timing is None:
        return "N/A"
    scale, unit = _duration_unit(timing.mean_seconds_per_point)
    if timing.sample_count < 2:
        return f"{timing.mean_seconds_per_point * scale:.6g} {unit}/point (SE pending)"
    return (
        f"{timing.mean_seconds_per_point * scale:.6g} +/- "
        f"{timing.uncertainty.standard_error * scale:.3g} {unit}/point"
    )


def _component_timing_is_exact_zero(
    timing: BenchmarkComponentTiming | None,
) -> bool:
    return (
        timing is not None
        and timing.mean_seconds_per_point == 0.0
        and timing.uncertainty.standard_deviation == 0.0
        and timing.uncertainty.standard_error == 0.0
        and timing.uncertainty.relative_standard_error == 0.0
    )


def _component_timings_are_exact_zero(
    timings: tuple[BenchmarkComponentTiming | None, ...],
) -> bool:
    return all(_component_timing_is_exact_zero(timing) for timing in timings)


def _counter_mean_text(value: float, unit: str) -> str:
    rounded = round(value)
    number = (
        f"{rounded:,}"
        if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-12)
        else f"{value:,.6g}"
    )
    return f"{number} {unit}"


def _profile_counter_rows(
    counters: BenchmarkProfileCounters,
) -> tuple[tuple[str, str, float, str, str], ...]:
    definitions = (
        (
            "native input",
            "input components",
            counters.native_input_components_per_point,
            "components",
            "per point",
        ),
        (
            "native input",
            "packed input",
            counters.native_input_pack_bytes_per_point,
            "bytes",
            "per point",
        ),
        (
            "native input",
            "crossed input",
            counters.native_input_crossing_bytes_per_point,
            "bytes",
            "per point",
        ),
        (
            "native input",
            "input containers",
            counters.native_input_container_allocations_per_call,
            "allocations",
            "per runtime call",
        ),
        (
            "state setup",
            "state extent",
            counters.state_components_per_point,
            "components",
            "per point",
        ),
        (
            "state setup",
            "state clear",
            counters.state_clear_components_per_point,
            "components",
            "per point",
        ),
        (
            "state setup",
            "source fill",
            counters.source_components_per_point,
            "components",
            "per point",
        ),
        (
            "state setup",
            "momentum fill",
            counters.momentum_components_per_point,
            "components",
            "per point",
        ),
        (
            "state setup",
            "model parameter fill",
            counters.model_parameter_components_per_point,
            "components",
            "per point",
        ),
        (
            "compiled stages",
            "stage input copy",
            counters.stage_input_copy_components_per_point,
            "components",
            "per point",
        ),
        (
            "compiled stages",
            "stage leaf copy",
            counters.stage_leaf_input_copy_components_per_point,
            "components",
            "per point",
        ),
        (
            "compiled stages",
            "stage output gather",
            counters.stage_evaluator_output_gather_components_per_point,
            "components",
            "per point",
        ),
        (
            "compiled stages",
            "stage output assign",
            counters.stage_output_assign_components_per_point,
            "components",
            "per point",
        ),
        (
            "amplitude",
            "amplitude input copy",
            counters.amplitude_input_copy_components_per_point,
            "components",
            "per point",
        ),
        (
            "amplitude",
            "amplitude leaf copy",
            counters.amplitude_leaf_input_copy_components_per_point,
            "components",
            "per point",
        ),
        (
            "amplitude",
            "amplitude output gather",
            counters.amplitude_evaluator_output_gather_components_per_point,
            "components",
            "per point",
        ),
        (
            "amplitude",
            "amplitude output remap",
            counters.amplitude_output_remap_components_per_point,
            "components",
            "per point",
        ),
        (
            "backend",
            "evaluator backend calls",
            counters.evaluator_backend_calls_per_call,
            "calls",
            "per runtime call",
        ),
        (
            "compiled Direct-Arena",
            "Arena engines",
            counters.compiled_direct_arena_engines_per_call,
            "engines",
            "per runtime call",
        ),
        (
            "compiled Direct-Arena",
            "Arena calls",
            counters.compiled_direct_arena_calls_per_call,
            "calls",
            "per runtime call",
        ),
        (
            "compiled Direct-Arena",
            "boundary input",
            counters.compiled_direct_arena_boundary_input_bytes_per_call,
            "bytes",
            "per runtime call",
        ),
        (
            "compiled Direct-Arena",
            "boundary current output",
            counters.compiled_direct_arena_boundary_current_output_bytes_per_call,
            "bytes",
            "per runtime call",
        ),
        (
            "compiled Direct-Arena",
            "boundary amplitude output",
            counters.compiled_direct_arena_boundary_amplitude_output_bytes_per_call,
            "bytes",
            "per runtime call",
        ),
        (
            "recurrence",
            "momentum scalar fill",
            counters.recurrence_momentum_scalar_values_per_point,
            "values",
            "per point",
        ),
        (
            "recurrence",
            "schedule executions",
            counters.recurrence_schedule_executions_per_call,
            "executions",
            "per runtime call",
        ),
        (
            "recurrence",
            "replay schedule executions",
            counters.recurrence_replay_schedule_executions_per_call,
            "executions",
            "per runtime call",
        ),
        (
            "recurrence",
            "union schedule executions",
            counters.recurrence_union_schedule_executions_per_call,
            "executions",
            "per runtime call",
        ),
        (
            "recurrence",
            "union source rows",
            counters.recurrence_union_source_rows_per_call,
            "rows",
            "per runtime call",
        ),
        (
            "recurrence",
            "replay output mapping",
            counters.recurrence_replay_output_values_per_point,
            "values",
            "per point",
        ),
        (
            "recurrence kernels",
            "source calls",
            counters.recurrence_source_calls_per_call,
            "calls",
            "per runtime call",
        ),
        (
            "recurrence kernels",
            "source rows",
            counters.recurrence_source_rows_per_call,
            "rows",
            "per runtime call",
        ),
        (
            "recurrence kernels",
            "contribution calls",
            counters.recurrence_contribution_calls_per_call,
            "calls",
            "per runtime call",
        ),
        (
            "recurrence kernels",
            "contribution rows",
            counters.recurrence_contribution_rows_per_call,
            "rows",
            "per runtime call",
        ),
        (
            "recurrence kernels",
            "finalization calls",
            counters.recurrence_finalization_calls_per_call,
            "calls",
            "per runtime call",
        ),
        (
            "recurrence kernels",
            "finalization rows",
            counters.recurrence_finalization_rows_per_call,
            "rows",
            "per runtime call",
        ),
        (
            "recurrence kernels",
            "closure calls",
            counters.recurrence_closure_calls_per_call,
            "calls",
            "per runtime call",
        ),
        (
            "recurrence kernels",
            "closure rows",
            counters.recurrence_closure_rows_per_call,
            "rows",
            "per runtime call",
        ),
        (
            "reduction",
            "reduction input",
            counters.reduction_input_components_per_point,
            "components",
            "per point",
        ),
        (
            "selectors",
            "gathered rows",
            counters.selector_gather_points_per_point,
            "points",
            "per point",
        ),
        (
            "selectors",
            "gathered input",
            counters.selector_gather_bytes_per_point,
            "bytes",
            "per point",
        ),
        (
            "selectors",
            "scattered output",
            counters.selector_scatter_values_per_point,
            "values",
            "per point",
        ),
        (
            "materialization",
            "resolved components",
            counters.resolved_materialized_components_per_point,
            "components",
            "per point",
        ),
        (
            "materialization",
            "total values",
            counters.total_materialized_values_per_point,
            "values",
            "per point",
        ),
        (
            "materialization",
            "final output copy",
            counters.final_output_copy_values_per_point,
            "values",
            "per point",
        ),
        (
            "allocations",
            "observed scratch reallocations",
            counters.observed_scratch_reallocations_per_call,
            "reallocations",
            "per runtime call",
        ),
        (
            "allocations",
            "native output vectors",
            counters.native_output_allocations_per_call,
            "allocations",
            "per runtime call",
        ),
    )
    return tuple(
        (area, label, value, unit, basis)
        for area, label, value, unit, basis in definitions
        if value is not None
    )


def _benchmark_summary(
    result: BenchmarkResult,
    *,
    prettytable: Any,
    color: bool,
) -> str:
    environment = result.environment
    process = result.process_id or "N/A"
    if result.process_expression is not None:
        process = f"{process} ({result.process_expression})"
    elapsed_value = environment.get("elapsed_seconds")
    elapsed = (
        float(elapsed_value)
        if isinstance(elapsed_value, (float, int))
        and not isinstance(elapsed_value, bool)
        else None
    )
    evaluator_source = str(environment.get("evaluator_time_source", "unavailable"))
    evaluator_total_source = str(
        environment.get("evaluator_total_time_source", "unavailable")
    )
    timing_sample_contract = str(
        environment.get("timing_sample_contract", "unavailable")
    )
    shared_timing_samples = timing_sample_contract in {
        "shared_native_repeated_profile_v1",
        "shared_native_single_profile_v1",
    }
    paired_timing_samples = (
        timing_sample_contract == "paired_unprofiled_headline_profiled_attribution_v1"
    )
    separate_timing_samples = paired_timing_samples or (
        timing_sample_contract == "separate_native_profile_diagnostic_v1"
    )
    evaluator_sample_value = environment.get("evaluator_sample_count")
    evaluator_sample_count = (
        int(evaluator_sample_value)
        if isinstance(evaluator_sample_value, int)
        and not isinstance(evaluator_sample_value, bool)
        and evaluator_sample_value > 0
        else result.sample_count
    )
    evaluator_text = "N/A"
    if result.evaluator_time_per_point is not None:
        if result.evaluator_uncertainty is None:
            evaluator_text = f"{_seconds_text(result.evaluator_time_per_point)}/point"
        else:
            evaluator_text = _benchmark_timing_text(
                result.evaluator_time_per_point,
                result.evaluator_uncertainty,
                sample_count=evaluator_sample_count,
            )
    evaluator_total_text = "N/A"
    if result.evaluator_total_time_per_point is not None:
        evaluator_total_text = (
            f"{_seconds_text(result.evaluator_total_time_per_point)}/point"
        )

    relative_error = result.uncertainty.relative_standard_error
    if relative_error <= 0.01:
        uncertainty_color = "GREEN"
    elif relative_error <= 0.05:
        uncertainty_color = "YELLOW"
    else:
        uncertainty_color = "RED"
    status = (
        f"interrupted - partial statistics from {result.sample_count} complete blocks"
        if result.interrupted
        else "complete"
    )
    breakdown = result.timing_breakdown
    recurrence_timing = (
        breakdown is not None and breakdown.execution_mode == "recurrence"
    ) or environment.get("execution_mode") == "recurrence"
    on_the_fly_timing = (
        breakdown is not None and breakdown.execution_mode == "on-the-fly"
    ) or environment.get("execution_mode") == "on-the-fly"
    direct_arena_timing = environment.get("compiled_direct_arena_active") is True
    if recurrence_timing:
        profile_metric_label = "recurrence core"
    elif on_the_fly_timing:
        profile_metric_label = "OTF recurrence-schedule core"
    elif direct_arena_timing:
        profile_metric_label = "Direct-Arena runtime envelope"
    else:
        profile_metric_label = "evaluator core"
    profile_repetitions_value = environment.get("native_profile_repetitions_per_sample")
    profile_repetitions = (
        int(profile_repetitions_value)
        if isinstance(profile_repetitions_value, int)
        and not isinstance(profile_repetitions_value, bool)
        and profile_repetitions_value > 0
        else (result.repetitions_per_sample if shared_timing_samples else 1)
    )
    profile_points_value = environment.get("native_profile_points_per_sample")
    profile_points = (
        int(profile_points_value)
        if isinstance(profile_points_value, int)
        and not isinstance(profile_points_value, bool)
        and profile_points_value > 0
        else profile_repetitions * result.effective_config.batch_size
    )
    warmup_elapsed_value = environment.get("warmup_elapsed_seconds")
    warmup_elapsed = (
        float(warmup_elapsed_value)
        if isinstance(warmup_elapsed_value, (float, int))
        and not isinstance(warmup_elapsed_value, bool)
        and math.isfinite(float(warmup_elapsed_value))
        and float(warmup_elapsed_value) >= 0.0
        else None
    )
    first_warmup_value = environment.get("first_warmup_run_outer_wall_seconds")
    first_warmup = (
        float(first_warmup_value)
        if isinstance(first_warmup_value, (float, int))
        and not isinstance(first_warmup_value, bool)
        and math.isfinite(float(first_warmup_value))
        and float(first_warmup_value) >= 0.0
        else None
    )
    cold_warmup_value = environment.get("cold_warmup_elapsed_seconds")
    cold_warmup = (
        float(cold_warmup_value)
        if isinstance(cold_warmup_value, (float, int))
        and not isinstance(cold_warmup_value, bool)
        and math.isfinite(float(cold_warmup_value))
        and float(cold_warmup_value) >= 0.0
        else None
    )
    warmup_runs_value = environment.get("warmup_configured_run_count")
    warmup_runs = (
        int(warmup_runs_value)
        if isinstance(warmup_runs_value, int)
        and not isinstance(warmup_runs_value, bool)
        and warmup_runs_value >= 0
        else result.effective_config.warmup_runs
    )
    warmup_batch_value = environment.get("warmup_batch_size")
    warmup_batch = (
        int(warmup_batch_value)
        if isinstance(warmup_batch_value, int)
        and not isinstance(warmup_batch_value, bool)
        and warmup_batch_value > 0
        else result.effective_config.batch_size
    )
    warmup_point_value = environment.get("warmup_point_count")
    warmup_points = (
        int(warmup_point_value)
        if isinstance(warmup_point_value, int)
        and not isinstance(warmup_point_value, bool)
        and warmup_point_value >= 0
        else warmup_runs * warmup_batch
    )
    cold_warmup_batch_value = environment.get("cold_warmup_batch_size")
    cold_warmup_batch = (
        int(cold_warmup_batch_value)
        if isinstance(cold_warmup_batch_value, int)
        and not isinstance(cold_warmup_batch_value, bool)
        and cold_warmup_batch_value > 0
        else warmup_batch
    )
    warmup_rows: list[tuple[str, str, str | None]] = []
    state_evidence = environment.get("cold_warmup_runtime_state_evidence")
    runtime_was_cold = environment.get(
        "cold_warmup_runtime_cold_before_first_evaluation"
    )
    runtime_was_retained = environment.get(
        "cold_warmup_runtime_retained_before_first_evaluation"
    )
    runtime_is_retained = environment.get(
        "cold_warmup_runtime_retained_after_first_evaluation"
    )
    if (
        state_evidence == "authenticated-native-otf-census-v1"
        and isinstance(runtime_was_cold, bool)
        and isinstance(runtime_was_retained, bool)
        and isinstance(runtime_is_retained, bool)
    ):
        before_state = (
            "cold"
            if runtime_was_cold
            else "strictly retained"
            if runtime_was_retained
            else "partial"
        )
        after_state = "strictly retained" if runtime_is_retained else "partial"
        warmup_rows.append(
            (
                "OTF runtime state",
                f"authenticated native census: {before_state} -> {after_state}",
                "GREEN" if runtime_is_retained else "RED",
            )
        )
        state_before = environment.get("cold_warmup_runtime_state_before")
        state_after = environment.get("cold_warmup_runtime_state_after")
        if isinstance(state_before, Mapping) and isinstance(state_after, Mapping):
            cache_policy = state_after.get("family_cache_policy")
            cache_limit = state_after.get("family_cache_limit")
            before_families = state_before.get("retained_family_count")
            after_families = state_after.get("retained_family_count")
            if (
                isinstance(cache_policy, str)
                and isinstance(cache_limit, int)
                and not isinstance(cache_limit, bool)
                and isinstance(before_families, int)
                and not isinstance(before_families, bool)
                and isinstance(after_families, int)
                and not isinstance(after_families, bool)
            ):
                warmup_rows.append(
                    (
                        "OTF family cache",
                        (
                            f"{cache_policy}; limit {cache_limit}; retained families "
                            f"{before_families} -> {after_families}"
                        ),
                        "GREEN",
                    )
                )
    if cold_warmup is not None:
        warmup_rows.append(
            (
                "OTF initial preparation",
                (
                    f"{_seconds_text(cold_warmup)} outer wall for 1 requested-selector "
                    f"evaluation x {cold_warmup_batch} points; Runtime/artifact load "
                    "excluded"
                ),
                "YELLOW",
            )
        )
    if warmup_elapsed is not None:
        warmup_rows.append(
            (
                "warm-up wall (total)",
                (
                    f"{_seconds_text(warmup_elapsed)} across {warmup_runs} configured "
                    f"runs x {warmup_batch} points ({warmup_points} primary points)"
                ),
                "YELLOW",
            )
        )
    if first_warmup is not None:
        warmup_rows.append(
            (
                "first warm-up run",
                f"{_seconds_text(first_warmup)} outer wall",
                "YELLOW",
            )
        )
    wall_pass = _value_text(environment.get("wall_time_sample_pass"))
    evaluator_pass = _value_text(environment.get("evaluator_time_sample_pass"))
    rows: list[tuple[str, str, str | None]] = [
        ("process", process, None),
        ("artifact", _value_text(environment.get("target")), None),
        ("execution mode", _value_text(environment.get("execution_mode")), "CYAN"),
        ("color workload", _value_text(environment.get("color_workload")), None),
        (
            "helicity workload",
            _value_text(environment.get("helicity_workload")),
            None,
        ),
        ("status", status, "YELLOW" if result.interrupted else "GREEN"),
        *warmup_rows,
        (
            "outer wall",
            _benchmark_timing_text(
                result.wall_time_per_point,
                result.uncertainty,
                sample_count=result.sample_count,
            ),
            "GREEN",
        ),
        (
            "evaluator total",
            evaluator_total_text,
            "CYAN",
        ),
        (
            profile_metric_label,
            evaluator_text,
            "CYAN",
        ),
        (
            "wall variability",
            (
                (
                    f"SD {_seconds_text(result.uncertainty.standard_deviation)}/point; "
                    f"relative standard error {relative_error:.3%}"
                )
                if result.sample_count > 1
                else "not estimable from one complete block"
            ),
            uncertainty_color if result.sample_count > 1 else "YELLOW",
        ),
        (
            "sampling",
            (
                f"{result.sample_count} blocks x "
                f"{result.repetitions_per_sample} repetitions x "
                f"{result.effective_config.batch_size} points"
            ),
            None,
        ),
        ("timed evaluations", str(result.evaluation_count), None),
        ("timed points", str(result.evaluated_point_count), None),
        (
            "target / measured",
            (
                f"{result.effective_config.target_runtime:.6g} s / {elapsed:.6g} s"
                if elapsed is not None
                else f"{result.effective_config.target_runtime:.6g} s / N/A"
            ),
            None,
        ),
        ("precision", str(result.effective_config.precision), None),
        (
            "timing sources",
            (
                "outer wall="
                f"{environment.get('wall_time_source', 'unavailable')} "
                f"via {wall_pass}; evaluator total={evaluator_total_source}; "
                f"{profile_metric_label}={evaluator_source} via {evaluator_pass}"
            ),
            None,
        ),
        (
            "sample provenance",
            (
                (
                    f"one shared native pass: {result.sample_count} blocks x "
                    f"{profile_repetitions} repetitions x "
                    f"{result.effective_config.batch_size} points; outer wall, "
                    f"evaluator total, {profile_metric_label}, and breakdown "
                    "use the same blocks"
                )
                if shared_timing_samples
                else (
                    f"paired passes: unprofiled headline {result.sample_count} "
                    f"blocks x {result.repetitions_per_sample} repetitions x "
                    f"{result.effective_config.batch_size} points provides outer "
                    "wall and evaluator total; profiled "
                    f"{profile_metric_label} attribution uses "
                    f"{breakdown.sample_count} paired blocks with the identical "
                    "batch and repetition count"
                    if paired_timing_samples and breakdown is not None
                    else (
                        f"separate passes: headline outer wall and evaluator total "
                        f"{result.sample_count} "
                        f"blocks x {result.repetitions_per_sample} repetitions x "
                        f"{result.effective_config.batch_size} points; "
                        f"{profile_metric_label} and breakdown "
                        f"{breakdown.sample_count} diagnostic blocks x "
                        f"{profile_repetitions} repetitions "
                        f"({profile_points} points/block)"
                        if timing_sample_contract
                        == "separate_native_profile_diagnostic_v1"
                        and breakdown is not None
                        else "no native timing breakdown available"
                    )
                )
            ),
            "YELLOW" if separate_timing_samples else None,
        ),
        ("platform", _value_text(environment.get("platform")), None),
    ]

    table = prettytable.PrettyTable(("metric", "value"))
    table.title = _paint("Runtime Profile", "CYAN", enabled=color)
    table.align["metric"] = "l"
    table.align["value"] = "l"
    table.max_width["metric"] = 24
    table.max_width["value"] = 96
    table.hrules = prettytable.HRuleStyle.FRAME
    for metric, value, color_name in rows:
        table.add_row(
            (
                metric,
                _paint(value, color_name, enabled=color)
                if color_name is not None
                else value,
            )
        )
    sections = [cast(str, table.get_string())]
    if breakdown is None:
        return sections[0]

    component_table = prettytable.PrettyTable(
        ("component", "mean +/- standard error", "relative standard error")
    )
    eager_profile = breakdown.execution_mode == "eager"
    recurrence_profile = breakdown.execution_mode == "recurrence"
    on_the_fly_profile = breakdown.execution_mode == "on-the-fly"
    if eager_profile:
        breakdown_title = "Rusticol Eager Timing Breakdown"
    elif recurrence_profile:
        breakdown_title = "Rusticol Recurrence Timing Breakdown"
    elif on_the_fly_profile:
        breakdown_title = "Rusticol OTF Recurrence-Schedule Timing Breakdown"
    else:
        breakdown_title = "Rusticol Timing Breakdown"
    if separate_timing_samples:
        breakdown_title += (
            " (paired profiled attribution)"
            if paired_timing_samples
            else " (separate diagnostic samples)"
        )
    component_table.title = _paint(
        breakdown_title,
        "CYAN",
        enabled=color,
    )
    component_table.align["component"] = "l"
    component_table.align["mean +/- standard error"] = "r"
    component_table.align["relative standard error"] = "r"
    component_table.hrules = prettytable.HRuleStyle.FRAME
    detailed_eager_profile = eager_profile and any(
        timing is not None
        for timing in (
            breakdown.eager_gather_time,
            breakdown.eager_kernel_call_time,
            breakdown.eager_invocation_scatter_time,
            breakdown.eager_finalization_time,
            breakdown.eager_scatter_finalization_time,
            breakdown.eager_closure_time,
            breakdown.eager_initialize_time,
            breakdown.eager_copy_out_time,
        )
    )
    momentum_input_setup_time = (
        breakdown.momentum_input_setup_time or breakdown.momentum_setup_time
    )
    orchestration_component_label = (
        "Direct-Arena runtime envelope"
        if direct_arena_timing
        else "Runtime orchestration"
    )
    component_rows: tuple[tuple[str, BenchmarkComponentTiming | None], ...]
    if detailed_eager_profile:
        eager_phase_rows: tuple[tuple[str, BenchmarkComponentTiming | None], ...]
        if (
            breakdown.eager_invocation_scatter_time is not None
            or breakdown.eager_finalization_time is not None
        ):
            eager_phase_rows = (
                (
                    "Invocation scatter (exclusive)",
                    breakdown.eager_invocation_scatter_time,
                ),
                (
                    "Current finalization (exclusive)",
                    breakdown.eager_finalization_time,
                ),
            )
        else:
            eager_phase_rows = (
                (
                    "Scatter / finalization (exclusive)",
                    breakdown.eager_scatter_finalization_time,
                ),
            )
        component_rows = (
            (
                "Profile wall (headline)"
                if shared_timing_samples
                else (
                    "Profile wall (paired profiled pass)"
                    if paired_timing_samples
                    else "Profile wall (diagnostic pass)"
                ),
                breakdown.wall_time,
            ),
            ("Native input pack (exclusive)", breakdown.native_input_pack_time),
            (
                "Native input crossing (exclusive)",
                breakdown.native_input_crossing_time,
            ),
            ("Runtime orchestration (exclusive)", breakdown.orchestration_time),
            ("State prepare (exclusive)", breakdown.state_prepare_time),
            ("State clear (exclusive)", breakdown.state_clear_time),
            ("Source fill (exclusive)", breakdown.source_fill_time),
            (
                "Momentum input setup (exclusive)",
                momentum_input_setup_time,
            ),
            (
                "Model parameter setup (exclusive)",
                breakdown.model_parameter_setup_time,
            ),
            ("Eager execution (inclusive)", breakdown.eager_execution_time),
            ("Initialize (exclusive)", breakdown.eager_initialize_time),
            ("Gather (exclusive)", breakdown.eager_gather_time),
            ("Kernel calls (exclusive)", breakdown.eager_kernel_call_time),
            *eager_phase_rows,
            ("Amplitude closure (exclusive)", breakdown.eager_closure_time),
            ("Amplitude copy-out (exclusive)", breakdown.eager_copy_out_time),
            ("Reduction (exclusive)", breakdown.reduction_time),
            (
                "Total materialization (exclusive)",
                breakdown.total_materialization_time,
            ),
            ("Final output copy (exclusive)", breakdown.final_output_copy_time),
            ("Selector planning (exclusive)", breakdown.selector_planner_time),
            ("Selector gather (exclusive)", breakdown.selector_gather_time),
            ("Selector scatter (exclusive)", breakdown.selector_scatter_time),
            ("Other Rusticol core (exclusive)", breakdown.other_core_time),
        )
    elif eager_profile:
        component_rows = (
            (
                "Profile wall (headline)"
                if shared_timing_samples
                else (
                    "Profile wall (paired profiled pass)"
                    if paired_timing_samples
                    else "Profile wall (diagnostic pass)"
                ),
                breakdown.wall_time,
            ),
            ("Native input pack (exclusive)", breakdown.native_input_pack_time),
            (
                "Native input crossing (exclusive)",
                breakdown.native_input_crossing_time,
            ),
            ("Runtime orchestration (exclusive)", breakdown.orchestration_time),
            ("State prepare (exclusive)", breakdown.state_prepare_time),
            ("State clear (exclusive)", breakdown.state_clear_time),
            ("Source fill (exclusive)", breakdown.source_fill_time),
            (
                "Momentum input setup (exclusive)",
                momentum_input_setup_time,
            ),
            (
                "Model parameter setup (exclusive)",
                breakdown.model_parameter_setup_time,
            ),
            ("Eager execution (inclusive)", breakdown.eager_execution_time),
            (
                "Total materialization (exclusive)",
                breakdown.total_materialization_time,
            ),
            ("Final output copy (exclusive)", breakdown.final_output_copy_time),
            ("Selector planning (exclusive)", breakdown.selector_planner_time),
            ("Selector gather (exclusive)", breakdown.selector_gather_time),
            ("Selector scatter (exclusive)", breakdown.selector_scatter_time),
            ("Other Rusticol core (exclusive)", breakdown.other_core_time),
        )
    elif recurrence_profile or on_the_fly_profile:
        recurrence_schedule_label = (
            "OTF recurrence schedule (inclusive)"
            if on_the_fly_profile
            else "Recurrence schedule (inclusive)"
        )
        component_rows = (
            (
                "Profile wall (headline)"
                if shared_timing_samples
                else (
                    "Profile wall (paired profiled pass)"
                    if paired_timing_samples
                    else "Profile wall (diagnostic pass)"
                ),
                breakdown.wall_time,
            ),
            ("Native input pack (exclusive)", breakdown.native_input_pack_time),
            (
                "Native input crossing (exclusive)",
                breakdown.native_input_crossing_time,
            ),
            ("Runtime orchestration (exclusive)", breakdown.orchestration_time),
            ("State prepare (exclusive)", breakdown.state_prepare_time),
            ("State clear (exclusive)", breakdown.state_clear_time),
            ("Source preparation (exclusive)", breakdown.source_fill_time),
            (
                "External momentum flatten (exclusive)",
                momentum_input_setup_time,
            ),
            (
                "Model parameter setup (exclusive)",
                breakdown.model_parameter_setup_time,
            ),
            (
                "Momentum-form fill (exclusive)",
                breakdown.recurrence_momentum_fill_time,
            ),
            (
                "Union source fill (exclusive)",
                breakdown.recurrence_union_source_fill_time,
            ),
            (
                recurrence_schedule_label,
                breakdown.recurrence_schedule_time,
            ),
            (
                "Source kernels (schedule attribution; do not add)",
                breakdown.recurrence_source_kernel_time,
            ),
            (
                "Contribution kernels (schedule attribution; do not add)",
                breakdown.recurrence_contribution_kernel_time,
            ),
            (
                "Current finalization (schedule attribution; do not add)",
                breakdown.recurrence_finalization_time,
            ),
            (
                "Amplitude closure (schedule attribution; do not add)",
                breakdown.recurrence_closure_time,
            ),
            (
                "Replay output mapping (exclusive)",
                breakdown.recurrence_replay_output_mapping_time,
            ),
            ("Reduction (exclusive)", breakdown.reduction_time),
            (
                "Total materialization (exclusive)",
                breakdown.total_materialization_time,
            ),
            ("Final output copy (exclusive)", breakdown.final_output_copy_time),
            ("Selector planning (exclusive)", breakdown.selector_planner_time),
            ("Selector gather (exclusive)", breakdown.selector_gather_time),
            ("Selector scatter (exclusive)", breakdown.selector_scatter_time),
            ("Other Rusticol core (exclusive)", breakdown.other_core_time),
        )
    else:
        component_rows = (
            (
                "Profile wall (headline)"
                if shared_timing_samples
                else (
                    "Profile wall (paired profiled pass)"
                    if paired_timing_samples
                    else "Profile wall (diagnostic pass)"
                ),
                breakdown.wall_time,
            ),
            ("Native input pack", breakdown.native_input_pack_time),
            ("Native input crossing", breakdown.native_input_crossing_time),
            (orchestration_component_label, breakdown.orchestration_time),
            ("State prepare", breakdown.state_prepare_time),
            ("State clear", breakdown.state_clear_time),
            ("Source fill", breakdown.source_fill_time),
            ("Momentum input setup", momentum_input_setup_time),
            ("Model parameter setup", breakdown.model_parameter_setup_time),
            ("Stage input pack (top-level)", breakdown.stage_input_pack_time),
            (
                "Stage evaluator envelope (top-level)",
                breakdown.stage_evaluator_call_time,
            ),
            (
                "Stage leaf input pack (attribution)",
                breakdown.stage_leaf_input_pack_time,
            ),
            (
                "Stage backend calls (attribution)",
                breakdown.stage_backend_call_time,
            ),
            (
                "Stage output gather (attribution)",
                breakdown.stage_evaluator_output_gather_time,
            ),
            ("Output assign", breakdown.output_assign_time),
            (
                "Amplitude input pack (top-level)",
                breakdown.amplitude_input_pack_time,
            ),
            (
                "Amplitude evaluator envelope (top-level)",
                breakdown.amplitude_evaluator_call_time,
            ),
            (
                "Amplitude leaf input pack (attribution)",
                breakdown.amplitude_leaf_input_pack_time,
            ),
            (
                "Amplitude backend call (attribution)",
                breakdown.amplitude_backend_call_time,
            ),
            (
                "Amplitude output gather (attribution)",
                breakdown.amplitude_evaluator_output_gather_time,
            ),
            (
                "Amplitude output remap (attribution)",
                breakdown.amplitude_output_remap_time,
            ),
            ("Reduction", breakdown.reduction_time),
            ("Total materialization", breakdown.total_materialization_time),
            ("Final output copy", breakdown.final_output_copy_time),
            ("Selector planning", breakdown.selector_planner_time),
            ("Selector gather", breakdown.selector_gather_time),
            ("Selector scatter", breakdown.selector_scatter_time),
            ("Other Rusticol core", breakdown.other_core_time),
        )
    for label, timing in component_rows:
        if _component_timing_is_exact_zero(timing):
            continue
        relative = (
            "N/A"
            if timing is None
            else f"{timing.uncertainty.relative_standard_error:.3%}"
        )
        value = _component_timing_text(timing)
        if label in {
            "Stage evaluator envelope (top-level)",
            "Amplitude evaluator envelope (top-level)",
            "Eager execution (inclusive)",
            "Recurrence schedule (inclusive)",
            "OTF recurrence schedule (inclusive)",
            "Kernel calls (exclusive)",
            "Source kernels (schedule attribution; do not add)",
            "Contribution kernels (schedule attribution; do not add)",
            "Current finalization (schedule attribution; do not add)",
            "Amplitude closure (schedule attribution; do not add)",
        }:
            value = _paint(value, "CYAN", enabled=color)
        component_table.add_row((label, value, relative))
    sections.append(cast(str, component_table.get_string()))

    visible_stages = tuple(
        stage
        for stage in breakdown.stages
        if not _component_timings_are_exact_zero(
            (
                stage.input_pack_time,
                stage.evaluator_call_time,
                stage.output_assign_time,
            )
        )
    )
    if visible_stages:
        stage_table = prettytable.PrettyTable(
            ("stage", "input pack", "evaluator call", "output assign")
        )
        stage_table.title = _paint("Rusticol Stage Detail", "CYAN", enabled=color)
        stage_table.align["stage"] = "r"
        for column in ("input pack", "evaluator call", "output assign"):
            stage_table.align[column] = "r"
        stage_table.hrules = prettytable.HRuleStyle.FRAME
        for stage in visible_stages:
            stage_table.add_row(
                (
                    stage.stage_index,
                    _component_timing_text(stage.input_pack_time),
                    _paint(
                        _component_timing_text(stage.evaluator_call_time),
                        "CYAN",
                        enabled=color,
                    ),
                    _component_timing_text(stage.output_assign_time),
                )
            )
        sections.append(cast(str, stage_table.get_string()))

        visible_nested_stages = tuple(
            stage
            for stage in breakdown.stages
            if any(
                timing is not None
                for timing in (
                    stage.leaf_input_pack_time,
                    stage.backend_call_time,
                    stage.evaluator_output_gather_time,
                )
            )
            and not _component_timings_are_exact_zero(
                (
                    stage.leaf_input_pack_time,
                    stage.backend_call_time,
                    stage.evaluator_output_gather_time,
                )
            )
        )
        if visible_nested_stages:
            nested_stage_table = prettytable.PrettyTable(
                ("stage", "leaf input pack", "backend call", "output gather")
            )
            nested_stage_table.title = _paint(
                "Rusticol Stage Internal Attribution (do not add to top-level)",
                "CYAN",
                enabled=color,
            )
            nested_stage_table.align["stage"] = "r"
            for column in ("leaf input pack", "backend call", "output gather"):
                nested_stage_table.align[column] = "r"
            nested_stage_table.hrules = prettytable.HRuleStyle.FRAME
            for stage in visible_nested_stages:
                nested_stage_table.add_row(
                    (
                        stage.stage_index,
                        _component_timing_text(stage.leaf_input_pack_time),
                        _paint(
                            _component_timing_text(stage.backend_call_time),
                            "CYAN",
                            enabled=color,
                        ),
                        _component_timing_text(stage.evaluator_output_gather_time),
                    )
                )
            sections.append(cast(str, nested_stage_table.get_string()))

    counters = breakdown.counters
    if counters is not None:
        counter_table = prettytable.PrettyTable(
            ("area", "counter", "mean", "normalization")
        )
        counter_table.title = _paint(
            (
                "Native Work Counters "
                f"(mean across {counters.sample_count} profiled blocks)"
            ),
            "CYAN",
            enabled=color,
        )
        for column in ("area", "counter", "normalization"):
            counter_table.align[column] = "l"
        counter_table.align["mean"] = "r"
        counter_table.hrules = prettytable.HRuleStyle.FRAME
        previous_area: str | None = None
        for area, label, counter_value, unit, basis in _profile_counter_rows(counters):
            area_text = (
                _paint(area, "CYAN", enabled=color) if area != previous_area else ""
            )
            value_text = _counter_mean_text(counter_value, unit)
            if label == "observed scratch reallocations":
                value_text = _paint(
                    value_text,
                    "GREEN" if counter_value == 0.0 else "YELLOW",
                    enabled=color,
                )
            elif label == "evaluator backend calls":
                value_text = _paint(value_text, "CYAN", enabled=color)
            counter_table.add_row((area_text, label, value_text, basis))
            previous_area = area
        sections.append(cast(str, counter_table.get_string()))
    return "\n\n".join(sections)


# Width of the value column of the generation summary.
_SUMMARY_VALUE_WIDTH = 88


def _generated_process_groups(output: Path) -> Mapping[str, object] | None:
    """The stored process-group map of a written artifact, when readable."""

    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.artifacts.inspection import stored_process_groups

    try:
        groups = stored_process_groups(output)
    except (OSError, ValueError, ArtifactError):
        return None
    return None if groups is None else cast(Mapping[str, object], _plain(groups))


def _generated_group_texts(
    record: Mapping[str, object],
    *,
    color: bool,
) -> dict[str, str]:
    """The group column of each generated process of a grouped artifact.

    A representative shows its group size; a generated member, its
    representative and any factor other than 1 or charge conjugation.
    Processes alone in their group have no entry.
    """

    texts: dict[str, str] = {}
    for group in cast(list[Mapping[str, object]], record["groups"]):
        representative, *members = _group_members(group)
        if not members:
            continue
        routed = sum(not bool(member["generated"]) for member in members)
        size = _subprocess_count(len(members) + 1)
        texts[str(representative["id"])] = (
            f"{size} ({_paint(f'{routed} routed', _ROUTED_COLOR, enabled=color)})"
            if routed
            else size
        )
        for member in members:
            if not bool(member["generated"]):
                continue
            relation = []
            if str(member["factor"]) != "1":
                relation.append(_factor_text(member["factor"], color=color, prefix="x"))
            if bool(member["conjugated"]):
                relation.append(_conjugation_mark(color=color))
            texts[str(member["id"])] = (
                "member of "
                + _paint(str(representative["id"]), "CYAN", enabled=color)
                + (f" ({', '.join(relation)})" if relation else "")
            )
    return texts


def _generation_summary(
    result: object,
    *,
    prettytable: Any,
    color: bool,
) -> str:
    from pyamplicol.api.results import GenerationResult

    if not isinstance(result, GenerationResult):
        raise TypeError("generation summary requires a GenerationResult")
    file_count = len(result.files)
    total_size: int | None = None
    container: Mapping[str, object] | None = None
    try:
        from pyamplicol.artifacts import load_manifest

        manifest = load_manifest(result.output, verify_payloads=False)
        manifest_path = result.output / "artifact.json"
        file_count = len(manifest.payloads) + 1
        total_size = sum(payload.size_bytes for payload in manifest.payloads)
        total_size += manifest_path.stat().st_size
        raw_container = getattr(manifest, "extensions", {}).get(
            "evaluator_payload_container"
        )
        if isinstance(raw_container, Mapping):
            container = raw_container
    except (OSError, RuntimeError, ValueError):
        existing = tuple(path for path in result.files if path.is_file())
        if existing:
            total_size = sum(path.stat().st_size for path in existing)
    files = f"{file_count} files"
    if total_size is not None:
        files += f"; {_byte_size(total_size)} total"
    if container is not None:
        member_count = container.get("member_count")
        unpacked = container.get("unpacked_size_bytes")
        if isinstance(member_count, int) and isinstance(unpacked, int):
            files += (
                f"; {member_count} indexed evaluator payloads "
                f"({_byte_size(unpacked)} unpacked)"
            )
    request_count = len(result.processes.requests)
    alias_count = len(result.processes.aliases)
    process_count = f"{request_count} request{'s' if request_count != 1 else ''}"
    if alias_count:
        process_count += f"; {alias_count} aliases"
    grouping = _generated_process_groups(result.output)
    grouping_text = (
        ""
        if grouping is None
        else _process_grouping_text(grouping, color=color, width=_SUMMARY_VALUE_WIDTH)
    )
    rows = (
        ("output", str(result.output)),
        ("processes", process_count),
        ("existing-output policy", result.mode),
        ("schema version", str(result.schema_version)),
        ("files", files),
    )
    table = prettytable.PrettyTable(("field", "value"))
    table.align["field"] = "l"
    table.align["value"] = "l"
    table.max_width["field"] = 30
    table.max_width["value"] = _SUMMARY_VALUE_WIDTH
    table.hrules = prettytable.HRuleStyle.FRAME
    value_colors = {
        "output": "GREEN",
        "processes": "CYAN",
        "existing-output policy": "YELLOW",
        "schema version": "MAGENTA",
        "files": "GREEN",
    }
    for field, value in rows:
        table.add_row(
            (
                _paint(field, "CYAN", enabled=color),
                _paint(value, value_colors[field], enabled=color),
            )
        )
        if field == "processes" and grouping_text:
            # coloured like the `inspect` process-group overview
            table.add_row(
                (_paint("process grouping", "CYAN", enabled=color), grouping_text)
            )
    group_texts = (
        {}
        if grouping is None or not grouping_text
        else _generated_group_texts(grouping, color=color)
    )
    process_table = prettytable.PrettyTable(
        ("name", "process", "group") if group_texts else ("name", "process")
    )
    process_table.title = _paint("Generated Processes", "CYAN", enabled=color)
    process_table.align = "l"
    process_table.max_width["name"] = 34
    process_table.max_width["process"] = 76
    process_table.hrules = prettytable.HRuleStyle.FRAME
    for request in result.processes.requests:
        process_table.add_row(
            (
                _paint(request.name, "CYAN", enabled=color),
                _paint(request.expression, "GREEN", enabled=color),
            )
            + ((group_texts.get(request.name, ""),) if group_texts else ())
        )
    for alias in result.processes.aliases:
        process_table.add_row(
            (
                _paint(alias.name, "MAGENTA", enabled=color),
                f"alias of {alias.process_name}",
            )
            + (("",) if group_texts else ())
        )
    return f"{table.get_string()}\n\n{process_table.get_string()}"


# Maps with at most this many subprocesses list one row per member; larger ones
# list each group's members as a wrapped grid of process strings.
_PROCESS_GROUP_ROW_LIMIT = 80
# Target width of the process-group tables and notes.
_PROCESS_GROUP_WIDTH = 120
# Colours of the process-group views, shared by `inspect` and `generate`.
_GENERATED_COLOR = "GREEN"
_ROUTED_COLOR = "CYAN"
_CONJUGATED_COLOR = "MAGENTA"
_ATTENTION_COLOR = "YELLOW"  # factors other than 1, warnings
_BASIS_COLORS = {"flavor-blind": "LIGHTBLUE_EX", "numerical": "GREEN"}
_Member = Mapping[str, object]
# (routed, conjugated, factor, basis) of a member's relation
_Relation = tuple[bool, bool, Fraction, str]


@dataclass(frozen=True, slots=True)
class _GroupCounts:
    """Counts of a process-group map; ``bases`` counts the related members."""

    groups: int
    subprocesses: int
    generated: int
    conjugated: int
    non_unit: int
    warnings: int
    bases: tuple[tuple[str, int], ...]

    @property
    def routed(self) -> int:
        return self.subprocesses - self.generated

    @property
    def trivial(self) -> bool:
        return self.subprocesses == self.groups


def _count(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or f'{singular}s'}"


def _subprocess_count(count: int) -> str:
    return _count(count, "subprocess", "subprocesses")


def _own_groups_text(count: int) -> str:
    """The note of a trivial map: every process is its own group."""

    if count == 1:
        return "the process is its own group"
    return f"each of the {count} processes is its own group"


def _natural_key(text: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", text)
        if part
    )


def _group_members(group: Mapping[str, object]) -> list[Mapping[str, object]]:
    return cast(list[Mapping[str, object]], group["members"])


def _group_counts(groups: Sequence[Mapping[str, object]]) -> _GroupCounts:
    members = [member for group in groups for member in _group_members(group)]
    bases: dict[str, int] = {}
    for group in groups:
        for member in _group_members(group)[1:]:
            basis = str(member["basis"])
            bases[basis] = bases.get(basis, 0) + 1
    return _GroupCounts(
        groups=len(groups),
        subprocesses=len(members),
        generated=sum(bool(member["generated"]) for member in members),
        conjugated=sum(bool(member["conjugated"]) for member in members),
        non_unit=sum(str(member["factor"]) != "1" for member in members),
        warnings=sum(bool(member.get("warning")) for member in members),
        bases=tuple(sorted(bases.items())),
    )


def _ordered_group_members(
    group: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Representative first, then generated, routed and conjugated members."""

    representative, *members = _group_members(group)
    return [
        representative,
        *sorted(
            members,
            key=lambda member: (
                not bool(member["generated"]),
                bool(member["conjugated"]),
                _natural_key(str(member["id"])),
            ),
        ),
    ]


def _is_identity_permutation(permutation: object) -> bool:
    values = cast(Sequence[int], permutation)
    return list(values) == list(range(len(values)))


def _permutation_text(permutation: object) -> str:
    if _is_identity_permutation(permutation):
        return ""
    return (
        "[" + " ".join(str(index) for index in cast(Sequence[int], permutation)) + "]"
    )


def _factor_text(factor: object, *, color: bool, prefix: str = "") -> str:
    text = str(factor)
    return _paint(
        f"{prefix}{text}",
        _GENERATED_COLOR if text == "1" else _ATTENTION_COLOR,
        enabled=color,
    )


def _basis_text(basis: object, *, color: bool) -> str:
    text = str(basis)
    return _paint(text, _BASIS_COLORS.get(text), enabled=color)


def _role_text(member: Mapping[str, object], *, color: bool) -> str:
    if bool(member["generated"]):
        return _paint("generated", _GENERATED_COLOR, enabled=color)
    return _paint("routed", _ROUTED_COLOR, enabled=color)


def _conjugation_mark(*, color: bool) -> str:
    return _paint("C", _CONJUGATED_COLOR, enabled=color, bold=True)


def _warning_count(count: int, *, color: bool) -> str:
    return _paint(
        _count(count, "warning"),
        _ATTENTION_COLOR if count else _GENERATED_COLOR,
        enabled=color,
    )


def _analysis_labels(analyses: Sequence[Mapping[str, object]]) -> list[str]:
    labels: list[str] = []
    for analysis in analyses:
        flags = []
        if analysis.get("only_unit_multiplier") is True:
            flags.append("unit factor only")
        if analysis.get("consider_charge_conjugation") is False:
            flags.append("no C")
        text = str(analysis.get("variant", "analysis"))
        labels.append(f"{text} ({', '.join(flags)})" if flags else text)
    return labels


def _analysis_text(analysis: Mapping[str, object]) -> str:
    """One recorded analysis, such as ``numerical analysis: 3 points, ...``."""

    details: list[str] = []
    points = analysis.get("points")
    if isinstance(points, int) and not isinstance(points, bool):
        details.append(_count(points, "point"))
    tolerance = analysis.get("tolerance")
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
        details.append(f"tolerance {tolerance:g}")
    if analysis.get("only_unit_multiplier") is True:
        details.append("unit factors only")
    if analysis.get("consider_charge_conjugation") is False:
        details.append("without charge conjugation")
    text = f"{analysis.get('variant', 'grouping')} analysis"
    return f"{text}: {', '.join(details)}" if details else text


def _group_size_text(size: int, generated: int, conjugated: int, *, color: bool) -> str:
    """``N subprocesses: G generated, R routed, C conjugated``, coloured."""

    parts = [
        _paint(f"{generated} generated", _GENERATED_COLOR, enabled=color),
        _paint(f"{size - generated} routed", _ROUTED_COLOR, enabled=color),
    ]
    if conjugated:
        parts.append(
            _paint(f"{conjugated} conjugated", _CONJUGATED_COLOR, enabled=color)
        )
    return f"{_subprocess_count(size)}: {', '.join(parts)}"


def _process_grouping_text(
    record: Mapping[str, object],
    *,
    color: bool,
    width: int,
) -> str:
    """The grouping outcome of a process-group map, in lines of ``width``.

    The first line gives the reduction, the next the relations and warnings,
    and the last ones the recorded analyses. It is empty for a trivial map
    without analyses.
    """

    groups = cast(list[Mapping[str, object]], record["groups"])
    analyses = cast(list[Mapping[str, object]], record.get("analyses", []))
    counts = _group_counts(groups)
    lines: list[str] = []
    if counts.trivial:
        if not analyses:
            return ""
        lines.append(f"no relation found; {_own_groups_text(counts.groups)}")
    else:
        reduction = _paint(
            _subprocess_count(counts.subprocesses), None, enabled=color, bold=True
        )
        lines.append(
            f"{reduction} -> {_count(counts.groups, 'group')}: "
            + _paint(f"{counts.generated} generated", _GENERATED_COLOR, enabled=color)
            + ", "
            + _paint(f"{counts.routed} routed", _ROUTED_COLOR, enabled=color)
        )
        related = counts.subprocesses - counts.groups
        relations = " + ".join(
            f"{count} {_basis_text(basis, color=color)}"
            for basis, count in counts.bases
        )
        details: list[str] = []
        if counts.conjugated:
            details.append(
                _paint(
                    f"{counts.conjugated} charge-conjugated",
                    _CONJUGATED_COLOR,
                    enabled=color,
                )
            )
        if counts.non_unit:
            details.append(
                _paint(
                    f"{counts.non_unit} with factor != 1",
                    _ATTENTION_COLOR,
                    enabled=color,
                )
            )
        relation_text = f"{relations} {'relation' if related == 1 else 'relations'}"
        if details:
            relation_text += f" ({', '.join(details)})"
        lines.append(
            _legend(
                (relation_text, _warning_count(counts.warnings, color=color)),
                width=width,
            )
        )
    lines.extend(_analysis_text(analysis) for analysis in analyses)
    return "\n".join(lines)


def _wrapped_note(text: str, *, indent: str = "") -> str:
    return textwrap.fill(
        text,
        width=_PROCESS_GROUP_WIDTH,
        subsequent_indent=indent,
        break_on_hyphens=False,
    )


_ANSI_STYLE = re.compile(r"\x1b\[[0-9;]*m")


def _legend(items: Sequence[str], *, width: int = _PROCESS_GROUP_WIDTH) -> str:
    """Join ``items`` with semicolons, breaking lines only between items."""

    lines: list[tuple[int, str]] = []
    for item in items:
        size = len(_ANSI_STYLE.sub("", item))
        if lines and lines[-1][0] + size + 2 <= width:
            used, text = lines[-1]
            lines[-1] = (used + size + 2, f"{text}; {item}")
        else:
            lines.append((size, item))
    return "\n".join(text for _size, text in lines)


def _padded(plain: str, painted: str, width: int) -> str:
    return painted + " " * (width - len(plain))


def _process_groups_overview(
    groups: Sequence[Mapping[str, object]],
    analyses: Sequence[Mapping[str, object]],
    *,
    prettytable: Any,
    color: bool,
) -> str:
    counts = _group_counts(groups)
    table = prettytable.PrettyTable(
        (
            "groups",
            "subprocesses",
            "generated",
            "routed",
            "conjugated",
            "factor != 1",
            "bases",
            "analyses",
            "warnings",
        )
    )
    table.align = "r"
    table.align["bases"] = "l"
    table.align["analyses"] = "l"
    # One basis per line; long analysis descriptions wrap to keep 120 columns.
    table.max_width["analyses"] = 12
    table.hrules = prettytable.HRuleStyle.FRAME
    table.add_row(
        (
            _paint(str(counts.groups), None, enabled=color, bold=True),
            _paint(str(counts.subprocesses), None, enabled=color, bold=True),
            _paint(str(counts.generated), _GENERATED_COLOR, enabled=color),
            _paint(str(counts.routed), _ROUTED_COLOR, enabled=color),
            _paint(
                str(counts.conjugated),
                _CONJUGATED_COLOR,
                enabled=color and counts.conjugated > 0,
            ),
            _paint(
                str(counts.non_unit),
                _ATTENTION_COLOR if counts.non_unit else _GENERATED_COLOR,
                enabled=color,
            ),
            "\n".join(
                f"{_basis_text(basis, color=color)} {count}"
                for basis, count in counts.bases
            )
            or "none",
            "\n".join(_analysis_labels(analyses)) or "none",
            _paint(
                str(counts.warnings),
                _ATTENTION_COLOR if counts.warnings else _GENERATED_COLOR,
                enabled=color,
            ),
        )
    )
    return cast(str, table.get_string())


def _process_group_detail_table(
    groups: Sequence[Mapping[str, object]],
    footnotes: Mapping[str, int],
    *,
    prettytable: Any,
    color: bool,
) -> str:
    members = [member for group in groups for member in _group_members(group)]
    show_factor = any(str(member["factor"]) != "1" for member in members)
    show_conjugated = any(bool(member["conjugated"]) for member in members)
    show_permutation = not all(
        _is_identity_permutation(member["permutation"]) for member in members
    )
    headers = ["process ID", "process", "role"]
    if show_factor:
        headers.append("factor")
    if show_conjugated:
        headers.append("C")
    if show_permutation:
        headers.append("permutation")
    headers.append("basis")
    if footnotes:
        headers.append("note")
    table = prettytable.PrettyTable(headers)
    table.align = "l"
    if show_factor:
        table.align["factor"] = "r"
    if show_conjugated:
        table.align["C"] = "c"
    table.hrules = prettytable.HRuleStyle.FRAME
    for group_index, group in enumerate(groups):
        ordered = _ordered_group_members(group)
        representative = ordered[0]
        rows: list[list[str]] = [
            [
                _paint(str(representative["id"]), "CYAN", enabled=color, bold=True),
                _paint(
                    str(representative["process"]), "GREEN", enabled=color, bold=True
                ),
                _paint("representative", None, enabled=color, bold=True),
            ]
            + [""] * (len(headers) - 3)
        ]
        for index, member in enumerate(ordered[1:], start=2):
            branch = "`- " if index == len(ordered) else "|- "
            row = [
                branch + _paint(str(member["id"]), "CYAN", enabled=color),
                str(member["process"]),
                _role_text(member, color=color),
            ]
            if show_factor:
                row.append(_factor_text(member["factor"], color=color))
            if show_conjugated:
                row.append(
                    _conjugation_mark(color=color) if bool(member["conjugated"]) else ""
                )
            if show_permutation:
                row.append(_permutation_text(member["permutation"]))
            row.append(_basis_text(member["basis"], color=color))
            if footnotes:
                warning = member.get("warning")
                row.append(
                    _paint(
                        f"[{footnotes[str(warning)]}]", _ATTENTION_COLOR, enabled=color
                    )
                    if warning
                    else ""
                )
            rows.append(row)
        for index, row in enumerate(rows):
            table.add_row(
                row,
                divider=index == len(rows) - 1 and group_index < len(groups) - 1,
            )
    return cast(str, table.get_string())


def _process_group_compact_table(
    groups: Sequence[Mapping[str, object]],
    footnotes: Mapping[str, int],
    *,
    prettytable: Any,
    color: bool,
) -> str:
    """One row per group, its members as a grid of process strings.

    Members are listed in one block per relation, labelled with their count,
    role and conjugation. Factor and basis columns appear in the labels only
    when they vary; the overview reports them otherwise.
    """

    grouped = [group for group in groups if len(_group_members(group)) > 1]
    ungrouped = [
        _group_members(group)[0] for group in groups if len(_group_members(group)) == 1
    ]

    def token(member: _Member) -> tuple[str, str]:
        process = str(member["process"])
        warning = member.get("warning")
        if not warning:
            return process, process
        marker = f"[{footnotes[str(warning)]}]"
        return (
            f"{process} {marker}",
            f"{process} {_paint(marker, _ATTENTION_COLOR, enabled=color)}",
        )

    classes: list[list[tuple[_Relation, list[_Member]]]] = []
    for group in grouped:
        by_relation: dict[_Relation, list[_Member]] = {}
        for member in _ordered_group_members(group)[1:]:
            key = (
                not bool(member["generated"]),
                bool(member["conjugated"]),
                Fraction(str(member["factor"])),
                str(member["basis"]),
            )
            by_relation.setdefault(key, []).append(member)
        classes.append(sorted(by_relation.items(), key=lambda item: item[0]))
    relations = [(key, len(members)) for items in classes for key, members in items]
    count_width = max((len(str(count)) for _key, count in relations), default=1)
    role_width = max(
        (len("routed" if key[0] else "generated") for key, _count in relations),
        default=0,
    )
    show_factor = any(key[2] != 1 for key, _count in relations)
    show_basis = len({key[3] for key, _count in relations}) > 1
    factor_width = max((len(f"x{key[2]}") for key, _count in relations), default=0)
    basis_width = max((len(key[3]) for key, _count in relations), default=0)
    label_width = (
        count_width
        + role_width
        + 3
        + (factor_width + 1 if show_factor else 0)
        + (basis_width + 1 if show_basis else 0)
    )

    def label(key: _Relation, count: int) -> str:
        routed, conjugated, factor, basis = key
        role = "routed" if routed else "generated"
        parts = [
            f"{count:>{count_width}}",
            _padded(
                role,
                _paint(
                    role,
                    _ROUTED_COLOR if routed else _GENERATED_COLOR,
                    enabled=color,
                ),
                role_width,
            ),
            _conjugation_mark(color=color) if conjugated else " ",
        ]
        if show_factor:
            parts.append(
                _padded(
                    f"x{factor}",
                    _factor_text(factor, color=color, prefix="x"),
                    factor_width,
                )
            )
        if show_basis:
            parts.append(_padded(basis, _basis_text(basis, color=color), basis_width))
        return " ".join(parts)

    tokens = [
        token(member)
        for items in classes
        for _key, members in items
        for member in members
    ] + [token(member) for member in ungrouped]
    cell_width = max(len(plain) for plain, _painted in tokens) + 2
    group_width = max(
        [len("ungrouped"), len(_subprocess_count(len(ungrouped)))]
        + [
            len(text)
            for group in grouped
            for text in (
                str(group["representative"]),
                str(group["representative_id"]),
                _subprocess_count(len(_group_members(group))),
            )
        ]
    )
    members_width = _PROCESS_GROUP_WIDTH - group_width - 7
    per_line = max(1, (members_width - label_width) // cell_width)

    def grid(members: Sequence[tuple[str, str]], first: str) -> list[str]:
        lines: list[str] = []
        for start in range(0, len(members), per_line):
            chunk = members[start : start + per_line]
            cells = [_padded(plain, painted, cell_width) for plain, painted in chunk]
            prefix = first if start == 0 else " " * label_width
            lines.append(f"{prefix}  {''.join(cells).rstrip()}")
        return lines

    table = prettytable.PrettyTable(("representative", "members"))
    table.align = "l"
    table.hrules = prettytable.HRuleStyle.FRAME
    rows: list[tuple[str, str]] = []
    for group, items in zip(grouped, classes, strict=True):
        rows.append(
            (
                "\n".join(
                    (
                        _paint(
                            str(group["representative"]),
                            "GREEN",
                            enabled=color,
                            bold=True,
                        ),
                        _paint(str(group["representative_id"]), "CYAN", enabled=color),
                        _subprocess_count(len(_group_members(group))),
                    )
                ),
                "\n".join(
                    line
                    for key, members in items
                    for line in grid(
                        [token(member) for member in members],
                        label(key, len(members)),
                    )
                ),
            )
        )
    if ungrouped:
        rows.append(
            (
                _paint("ungrouped", None, enabled=color, bold=True)
                + "\n"
                + _subprocess_count(len(ungrouped)),
                "\n".join(
                    grid([token(member) for member in ungrouped], " " * label_width)
                ),
            )
        )
    for index, row in enumerate(rows):
        table.add_row(row, divider=index < len(rows) - 1)
    return cast(str, table.get_string())


def _process_groups_summary(
    record: Mapping[str, object],
    *,
    prettytable: Any,
    color: bool,
) -> str:
    groups = cast(list[Mapping[str, object]], record["groups"])
    analyses = cast(list[Mapping[str, object]], record.get("analyses", []))
    heading = _paint("Process groups", "CYAN", enabled=color)
    counts = _group_counts(groups)
    if counts.trivial:
        own = _own_groups_text(counts.groups)
        if not bool(record.get("stored", True)):
            return f"{heading}: no process-groups.json; {own}"
        if analyses:
            labels = "; ".join(_analysis_text(analysis) for analysis in analyses)
            return f"{heading}: no relation found ({labels}); {own}"
        return f"{heading}: none; {own}"
    members = [member for group in groups for member in _group_members(group)]
    footnotes: dict[str, int] = {}
    for group in groups:
        for member in _ordered_group_members(group):
            warning = member.get("warning")
            if warning and str(warning) not in footnotes:
                footnotes[str(warning)] = len(footnotes) + 1
    # One group, as selected by `--process REPRESENTATIVE`, is always detailed.
    detailed = counts.groups == 1 or counts.subprocesses <= _PROCESS_GROUP_ROW_LIMIT
    table = (_process_group_detail_table if detailed else _process_group_compact_table)(
        groups, footnotes, prettytable=prettytable, color=color
    )
    legend: list[str] = []
    if counts.routed:
        routed = _paint("routed", _ROUTED_COLOR, enabled=color)
        legend.append(f"{routed}: not generated, evaluated through its representative")
    if counts.conjugated:
        legend.append(f"{_conjugation_mark(color=color)}: charge-conjugated relation")
    if counts.non_unit:
        legend.append("factor F: ME = F x ME(representative)")
    if not detailed:
        legend.extend(
            (
                f"compact view of {_subprocess_count(counts.subprocesses)}",
                "IDs and leg permutations: --json",
                "one group row by row: --process REPRESENTATIVE",
            )
        )
    elif not all(_is_identity_permutation(member["permutation"]) for member in members):
        legend.append(
            "permutation p: representative leg r takes the momentum of member leg p[r]"
        )
    notes = [_legend(legend)] if legend else []
    notes.extend(map(_analysis_text, analyses))
    for warning, number in footnotes.items():
        marker = f"[{number}]"
        text = _wrapped_note(f"{marker} {warning}", indent=" " * (len(marker) + 1))
        notes.append(
            _paint(marker, _ATTENTION_COLOR, enabled=color) + text.removeprefix(marker)
        )
    sections = [
        heading,
        _process_groups_overview(
            groups, analyses, prettytable=prettytable, color=color
        ),
        table,
    ]
    if notes:
        sections.append("\n".join(notes))
    return "\n\n".join(sections)


def _process_group_selection_summary(
    record: Mapping[str, object],
    *,
    prettytable: Any,
    color: bool,
) -> str:
    role = str(record["role"])
    table = prettytable.PrettyTable(("field", "value"))
    table.title = _paint("Process Group", "CYAN", enabled=color)
    table.align = "l"
    table.max_width["value"] = 88
    table.hrules = prettytable.HRuleStyle.FRAME
    size = _integer(record["group_size"], "group size")
    routed = _integer(record["routed_count"], "routed count")
    conjugated = _integer(record["conjugated_count"], "conjugated count")
    group_text = _group_size_text(size, size - routed, conjugated, color=color)
    permutation = _permutation_text(record["permutation"])
    # `permutation` relates the representative to the requested leg order
    permutation_text = (
        f"p = {permutation}: representative leg r takes the momentum of leg p[r]"
        if permutation
        else "identity"
    )
    rows: list[tuple[str, str]]
    if role == "representative":
        rows = [
            (
                "role",
                _paint(
                    "representative: generated, evaluated by its own runtime",
                    _GENERATED_COLOR,
                    enabled=color,
                    bold=True,
                ),
            ),
            *((("permutation", permutation_text),) if permutation else ()),
            ("group", group_text),
        ]
    else:
        routed_member = role == "routed member"
        rows = [
            (
                "role",
                _paint(
                    "routed member: not generated, evaluated through the representative"
                    if routed_member
                    else "generated member: evaluated by its own runtime",
                    _ROUTED_COLOR if routed_member else _GENERATED_COLOR,
                    enabled=color,
                    bold=True,
                ),
            ),
            (
                "representative",
                _paint(str(record["representative_id"]), "CYAN", enabled=color)
                + "  "
                + _paint(str(record["representative"]), "GREEN", enabled=color),
            ),
            ("factor", _factor_text(record["factor"], color=color)),
            (
                "conjugated",
                _paint("yes (C)", _CONJUGATED_COLOR, enabled=color, bold=True)
                if bool(record["conjugated"])
                else "no",
            ),
            ("permutation", permutation_text),
            ("basis", _basis_text(record["basis"], color=color)),
            ("group", group_text),
        ]
        warning = record.get("warning")
        if warning:
            rows.append(
                (
                    "warning",
                    "\n".join(
                        _paint(line, _ATTENTION_COLOR, enabled=color)
                        for line in textwrap.wrap(str(warning), 88)
                    ),
                )
            )
    for field, value in rows:
        table.add_row((_paint(field, "CYAN", enabled=color), value))
    return cast(str, table.get_string())


def _artifact_inspection_summary(
    plain: Mapping[str, object],
    *,
    prettytable: Any,
    color: bool,
) -> str:
    processes = cast(list[Mapping[str, object]], plain["processes"])
    dependencies = cast(list[Mapping[str, object]], plain["dependencies"])
    aliases = [
        cast(Mapping[str, object], alias)
        for process in processes
        for alias in cast(list[object], process["aliases"])
    ]

    model = str(plain["model_name"])
    restriction = plain.get("model_restriction")
    if restriction is not None:
        model = f"{model} ({restriction})"
    target = str(plain["target"])
    features = cast(list[object], plain["cpu_features"])
    if features:
        target = f"{target} [{', '.join(str(feature) for feature in features)}]"
    runtime_capabilities = cast(
        Sequence[object],
        plain["runtime_capabilities"],
    )

    summary = prettytable.PrettyTable(("field", "value"))
    summary.align["field"] = "l"
    summary.align["value"] = "l"
    summary.max_width["field"] = 24
    summary.max_width["value"] = 88
    summary.hrules = prettytable.HRuleStyle.FRAME
    physical_file_count = plain.get("physical_file_count")
    if not isinstance(physical_file_count, int):
        payload_count = plain["payload_count"]
        if isinstance(payload_count, bool) or not isinstance(payload_count, int):
            raise ValueError("artifact payload count must be an integer")
        physical_file_count = payload_count + 1
    payload_text = (
        f"{physical_file_count} physical files; "
        f"{_byte_size(plain['payload_size_bytes'])}"
    )
    container_members = plain.get("evaluator_container_member_count")
    container_unpacked = plain.get("evaluator_container_unpacked_size_bytes")
    if isinstance(container_members, int) and isinstance(container_unpacked, int):
        payload_text += (
            f"; {container_members} indexed evaluators "
            f"({_byte_size(container_unpacked)} unpacked; "
            f"{plain.get('evaluator_container_abi')})"
        )
    contents = (
        f"{_count(len(processes), 'process', 'processes')}, "
        f"{_count(len(aliases), 'alias', 'aliases')}"
    )
    process_groups = plain.get("process_groups")
    if isinstance(process_groups, Mapping):
        counts = _group_counts(
            cast(list[Mapping[str, object]], process_groups["groups"])
        )
        if not counts.trivial:
            contents += (
                f"; {_subprocess_count(counts.subprocesses)} in "
                f"{_count(counts.groups, 'group')}"
            )
    summary_rows = (
        ("path", plain["path"]),
        ("artifact type", str(plain["artifact_kind"]).removeprefix("pyamplicol-")),
        ("artifact ID", plain["artifact_id"]),
        ("created", plain["created_utc"]),
        ("producer", f"pyamplicol {plain['producer_version']}"),
        ("target", target),
        ("model", f"{model}; {plain['model_source']}"),
        ("default process", plain.get("default_process_id")),
        ("contents", contents),
        (
            "runtime",
            f"{plain['runtime_engine']} {plain['runtime_version']}",
        ),
        (
            "capabilities",
            ", ".join(str(value) for value in runtime_capabilities),
        ),
        ("payloads", payload_text),
        ("integrity", plain["integrity"]),
    )
    for key, value in summary_rows:
        text = _value_text(value)
        summary.add_row(
            (
                _paint(key, "CYAN", enabled=color),
                _colored(text, key=key, enabled=color),
            )
        )

    process_table = prettytable.PrettyTable(
        (
            "default",
            "stable ID",
            "concrete process",
            "color",
            "helicities",
            "color outputs",
            "coverage (hel./color)",
            "aliases",
        )
    )
    process_table.align = "l"
    process_table.align["default"] = "c"
    process_table.align["color"] = "c"
    process_table.align["helicities"] = "r"
    process_table.align["color outputs"] = "r"
    process_table.align["aliases"] = "r"
    process_table.max_width["stable ID"] = 30
    process_table.max_width["concrete process"] = 48
    process_table.hrules = prettytable.HRuleStyle.FRAME
    accuracy_colors = {"lc": "CYAN", "nlc": "YELLOW", "full": "MAGENTA"}
    for process in processes:
        accuracy = str(process["color_accuracy"])
        marker = "*" if bool(process["default"]) else ""
        coverage = f"{process['helicity_coverage']} / {process['color_coverage']}"
        coverage_text = (
            _paint(coverage, "GREEN", enabled=color)
            if coverage == "complete / complete"
            else coverage
        )
        process_table.add_row(
            (
                _paint(marker, "GREEN", enabled=color),
                _paint(str(process["id"]), "CYAN", enabled=color),
                _paint(str(process["expression"]), "GREEN", enabled=color),
                _paint(
                    accuracy,
                    accuracy_colors.get(accuracy, "WHITE"),
                    enabled=color,
                ),
                _paint(
                    _component_count(process, "helicities"),
                    "MAGENTA",
                    enabled=color,
                ),
                _paint(
                    _component_count(process, "color_components"),
                    "MAGENTA",
                    enabled=color,
                ),
                coverage_text,
                _paint(
                    str(len(cast(list[object], process["aliases"]))),
                    "MAGENTA",
                    enabled=color,
                ),
            )
        )

    sections = [
        _paint("Artifact", "CYAN", enabled=color),
        cast(str, summary.get_string()),
        _paint("Processes", "CYAN", enabled=color),
        cast(str, process_table.get_string()),
    ]
    if isinstance(process_groups, Mapping):
        sections.append(
            _process_groups_summary(
                process_groups, prettytable=prettytable, color=color
            )
        )

    execution_table = prettytable.PrettyTable(("process", "execution field", "value"))
    execution_table.align = "l"
    execution_table.max_width["process"] = 30
    execution_table.max_width["execution field"] = 26
    execution_table.max_width["value"] = 72
    execution_table.hrules = prettytable.HRuleStyle.FRAME

    def add_execution_row(row: tuple[object, object, object]) -> None:
        process_id, field, value = row
        execution_table.add_row(
            (
                _paint(str(process_id), "CYAN", enabled=color),
                _paint(str(field), "MAGENTA", enabled=color),
                _paint(_value_text(value), "CYAN", enabled=color),
            )
        )

    for process in processes:
        process_id = str(process["id"])
        mode = str(process.get("execution_mode", "compiled"))
        backend = process.get("prepared_backend")
        mode_text = mode if backend is None else f"{mode} ({backend})"
        add_execution_row((process_id, "mode / backend", mode_text))
        lc_flow_layout = process.get("lc_flow_layout")
        if lc_flow_layout is not None:
            add_execution_row((process_id, "LC flow layout", str(lc_flow_layout)))
        union_sector_count = process.get("lc_union_sector_count")
        if union_sector_count is not None:
            add_execution_row(
                (
                    process_id,
                    "LC union sectors",
                    f"{union_sector_count} physical / materialized; 0 residual",
                )
            )
        phases = cast(Sequence[object], process.get("native_profile_phases", ()))
        add_execution_row(
            (
                process_id,
                "native profile phases",
                ", ".join(str(phase) for phase in phases) or "unavailable",
            )
        )
        selector_provenance = process.get("selector_provenance")
        if selector_provenance is not None:
            add_execution_row(
                (
                    process_id,
                    "runtime selectors",
                    (
                        f"helicity {process.get('helicity_runtime_contract')}; "
                        "color flow "
                        f"{process.get('color_flow_runtime_contract')}"
                    ),
                )
            )
            specialized_axes = cast(
                Sequence[object], process.get("generation_specialized_axes", ())
            )
            add_execution_row(
                (
                    process_id,
                    "generation specialization",
                    ", ".join(str(axis) for axis in specialized_axes) or "none",
                )
            )
            source_helicities = cast(
                Sequence[Sequence[object]],
                process.get("selected_source_helicities", ()),
            )
            color_sectors = cast(
                Sequence[object], process.get("selected_color_sector_ids", ())
            )
            selection_parts = []
            if source_helicities:
                selection_parts.append(
                    "helicities "
                    + ", ".join(f"{pair[0]}:{pair[1]}" for pair in source_helicities)
                )
            if color_sectors:
                selection_parts.append(
                    "color sectors " + ", ".join(str(value) for value in color_sectors)
                )
            add_execution_row(
                (
                    process_id,
                    "generation selection",
                    "; ".join(selection_parts) or "none",
                )
            )
            add_execution_row((process_id, "selector provenance", selector_provenance))
        if mode == "on-the-fly":
            add_execution_row(
                (
                    process_id,
                    "compact/query-local census",
                    (
                        f"{process.get('physical_helicities')} physical helicities; "
                        f"{process.get('physical_color_components')} physical LC "
                        "color flows"
                    ),
                )
            )
        elif process.get("lc_physical_sector_count") is not None:
            add_execution_row(
                (
                    process_id,
                    "LC replay sectors",
                    (
                        f"{process.get('lc_physical_sector_count')} physical; "
                        f"{process.get('lc_materialized_sector_count')} materialized; "
                        f"{process.get('lc_replayed_sector_count')} replayed; "
                        f"{process.get('lc_residual_sector_count')} residual"
                    ),
                )
            )
        helicity_status = process.get("helicity_recurrence_status")
        if helicity_status is not None:
            recurrence = str(helicity_status)
            optimized_currents = process.get("helicity_optimized_current_count")
            if optimized_currents is not None:
                recurrence += (
                    f"; {optimized_currents} optimized currents in "
                    f"{process.get('helicity_optimized_class_count')} classes; "
                    f"{process.get('helicity_residual_current_count')} residual"
                )
            optimized_amplitude_classes = process.get(
                "helicity_optimized_amplitude_class_count"
            )
            if optimized_amplitude_classes is not None:
                recurrence += (
                    f"; {optimized_amplitude_classes} optimized amplitude classes; "
                    f"{process.get('helicity_residual_amplitude_count')} residual roots"
                )
            materialized_currents = process.get("helicity_materialized_current_count")
            if materialized_currents is not None:
                recurrence += (
                    f"; materialized {materialized_currents} currents / "
                    f"{process.get('helicity_materialized_amplitude_count')} roots"
                )
            add_execution_row((process_id, "helicity recurrence", recurrence))
        if mode == "recurrence":
            add_execution_row(
                (
                    process_id,
                    "direct recurrence rows",
                    (
                        f"{process.get('direct_source_row_count')} sources; "
                        f"{process.get('direct_contribution_row_count')} "
                        "contributions; "
                        f"{process.get('direct_finalization_row_count')} "
                        "finalizations; "
                        f"{process.get('direct_closure_row_count')} closures"
                    ),
                )
            )
            if process.get("recurrence_color_storage") is not None:
                factorization = process.get("recurrence_color_factorization_kind")
                factorization_text = (
                    "none"
                    if factorization is None
                    else (
                        f"{factorization}, rank "
                        f"{process.get('recurrence_color_factorization_rank')}, "
                        f"{process.get('recurrence_color_factorization_coset_count')} "
                        "cosets"
                    )
                )
                add_execution_row(
                    (
                        process_id,
                        "contracted color",
                        (
                            f"{process.get('recurrence_color_accuracy')}; "
                            f"{process.get('recurrence_color_storage')}; "
                            f"{process.get('recurrence_color_active_sector_count')}/"
                            f"{process.get('recurrence_color_sector_count')} "
                            "active sectors; "
                            f"{process.get('recurrence_color_component_count')} "
                            "components; "
                            f"{process.get('recurrence_color_group_count')} groups; "
                            f"{process.get('recurrence_color_entry_count')}/"
                            f"{process.get('recurrence_color_logical_entry_count')} "
                            f"stored/logical entries; {factorization_text}"
                        ),
                    )
                )
            continue
        if mode == "on-the-fly":
            requested_tile = process.get("requested_point_tile_size")
            add_execution_row(
                (
                    process_id,
                    "point tile request",
                    f"{requested_tile} requested; not applied by on-the-fly execution",
                )
            )
            add_execution_row(
                (
                    process_id,
                    "query construction threads",
                    f"{process.get('requested_query_construction_threads')} requested",
                )
            )
            add_execution_row((process_id, "numeric runtime", "f64 only"))
            continue
        if mode != "eager":
            continue
        pack_kernels = process.get("prepared_kernel_count")
        referenced_kernels = process.get("referenced_kernel_count")
        add_execution_row(
            (
                process_id,
                "prepared kernels",
                f"{pack_kernels} in pack; {referenced_kernels} referenced",
            )
        )
        invocation_count = process.get("invocation_count")
        attachment_count = process.get("attachment_count")
        alias_count = process.get("evaluation_alias_count")
        maximum_fanout = process.get("maximum_fanout")
        add_execution_row(
            (
                process_id,
                "invocations / reuse",
                (
                    f"{invocation_count} canonical; {attachment_count} attachments; "
                    f"{alias_count} reused aliases; max fanout {maximum_fanout}"
                ),
            )
        )
        add_execution_row(
            (
                process_id,
                "finalization / closure",
                (
                    f"{process.get('finalization_count')} currents; "
                    f"{process.get('closure_count')} amplitudes"
                ),
            )
        )
        add_execution_row(
            (
                process_id,
                "selector closures",
                (
                    (
                        f"{process.get('selector_domain_count')} domains; "
                        f"{process.get('selector_domain_membership_count')} "
                        "memberships"
                    )
                    if bool(process.get("selector_closure_available"))
                    else "not emitted"
                ),
            )
        )
        requested_tile = process.get("requested_point_tile_size")
        effective_tile = process.get("effective_point_tile_size")
        add_execution_row(
            (
                process_id,
                "point tile requested / effective",
                (
                    f"{requested_tile} / "
                    + (
                        str(effective_tile)
                        if effective_tile is not None
                        else "available after runtime load"
                    )
                ),
            )
        )
        workspace_target = process.get("workspace_limit_bytes")
        workspace_used = process.get("workspace_bytes")
        add_execution_row(
            (
                process_id,
                "workspace target / allocated",
                (
                    f"{_byte_size(workspace_target)} / "
                    + (
                        _byte_size(workspace_used)
                        if workspace_used is not None
                        else "available after runtime load"
                    )
                ),
            )
        )
    sections.extend(
        (
            _paint("Execution", "CYAN", enabled=color),
            cast(str, execution_table.get_string()),
        )
    )

    if aliases:
        alias_table = prettytable.PrettyTable(
            ("stable alias ID", "concrete process", "representative ID")
        )
        alias_table.align = "l"
        alias_table.max_width["stable alias ID"] = 30
        alias_table.max_width["concrete process"] = 52
        alias_table.max_width["representative ID"] = 30
        alias_table.hrules = prettytable.HRuleStyle.FRAME
        for alias in aliases:
            alias_table.add_row(
                (
                    _paint(str(alias["id"]), "CYAN", enabled=color),
                    _paint(str(alias["expression"]), "GREEN", enabled=color),
                    _paint(
                        str(alias["representative_id"]),
                        "MAGENTA",
                        enabled=color,
                    ),
                )
            )
        sections.extend(
            (
                _paint("Aliases", "CYAN", enabled=color),
                cast(str, alias_table.get_string()),
            )
        )

    if dependencies:
        dependency_table = prettytable.PrettyTable(("dependency", "version", "license"))
        dependency_table.align = "l"
        dependency_table.max_width["license"] = 48
        dependency_table.hrules = prettytable.HRuleStyle.FRAME
        for dependency in dependencies:
            dependency_table.add_row(
                (
                    _paint(str(dependency["name"]), "CYAN", enabled=color),
                    _paint(str(dependency["version"]), "GREEN", enabled=color),
                    _paint(str(dependency["license"]), "YELLOW", enabled=color),
                )
            )
        sections.extend(
            (
                _paint("Dependencies", "CYAN", enabled=color),
                cast(str, dependency_table.get_string()),
            )
        )
    return "\n\n".join(sections)


def _human_result_title(value: object, plain: object) -> str:
    class_name = type(value).__name__
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and value
        and all(type(entry).__name__ == "ExampleEntry" for entry in value)
    ):
        return "Packaged Examples"
    if class_name == "DiagnosticReport":
        return "pyAmpliCol Diagnostics"
    if class_name == "UtilityResult":
        return "Operation Complete"
    if class_name == "LicenseRequestResult":
        return "Symbolica License Request"
    if isinstance(plain, (str, int, float, bool)) or plain is None:
        return "pyAmpliCol Result"
    if isinstance(plain, Mapping):
        keys = frozenset(plain)
        if {"requested", "effective", "adjustments"} <= keys:
            return "Resolved Configuration"
        if {"available", "model", "entries"} <= keys:
            return "Model Processes"
        if {"name", "supported", "contents"} <= keys:
            return "Model Inspection"
        if {"model", "supported", "output"} <= keys:
            return "Model Compilation"
        if {"process_id", "external_particles", "helicities"} <= keys:
            return "Process Physics"
    name = class_name.removesuffix("Result")
    words: list[str] = []
    current = ""
    for character in name:
        if current and character.isupper() and not current[-1].isupper():
            words.append(current)
            current = character
        else:
            current += character
    if current:
        words.append(current)
    if words and name not in {"dict", "tuple", "list"}:
        return " ".join(words)
    return "pyAmpliCol Result"


def _record_color(key: str, text: str, *, color: bool) -> str:
    normalized = key.casefold().replace("_", " ")
    if normalized in {"name", "check", "field", "id", "key"}:
        return _paint(text, "CYAN", enabled=color)
    if normalized in {"action", "kind", "request", "mode"}:
        return _paint(text, "MAGENTA", enabled=color)
    if normalized in {"status", "supported", "available", "ok"}:
        return _colored(text, key=key, enabled=color)
    return text


def _record_sequence_summary(
    records: Sequence[object],
    *,
    prettytable: Any,
    color: bool,
    title: str,
) -> str:
    mappings = tuple(entry for entry in records if isinstance(entry, Mapping))
    if len(mappings) != len(records):
        table = prettytable.PrettyTable(("#", "value"))
        table.align["#"] = "r"
        table.align["value"] = "l"
        table.max_width["value"] = 100
        for index, entry in enumerate(records, start=1):
            table.add_row((index, _value_text(entry)))
    else:
        ordered_keys: list[str] = []
        for record in mappings:
            for raw_key in record:
                key = str(raw_key)
                if key not in ordered_keys:
                    ordered_keys.append(key)
        visible_keys = ordered_keys[:6]
        has_remainder = len(ordered_keys) > len(visible_keys)
        headings = tuple(key.replace("_", " ") for key in visible_keys)
        if has_remainder:
            headings += ("other fields",)
        table = prettytable.PrettyTable(headings or ("value",))
        table.align = "l"
        for heading in headings:
            if heading == "name":
                table.max_width[heading] = 68
            elif heading in {"description", "detail"}:
                table.max_width[heading] = 60
            else:
                table.max_width[heading] = 34
        for record in mappings:
            row = [
                _record_color(
                    key,
                    _value_text(record.get(key)),
                    color=color,
                )
                for key in visible_keys
            ]
            if has_remainder:
                remainder = {
                    key: record.get(key) for key in ordered_keys[len(visible_keys) :]
                }
                row.append(_value_text(remainder))
            table.add_row(tuple(row) if row else ("N/A",))
    table.title = _paint(title, "CYAN", enabled=color)
    table.hrules = prettytable.HRuleStyle.FRAME
    return cast(str, table.get_string())


def _flatten_record(
    value: Mapping[str, object],
    *,
    prefix: str = "",
) -> tuple[tuple[str, object], ...]:
    rows: list[tuple[str, object]] = []
    for raw_key, entry in value.items():
        key = str(raw_key).replace("_", " ")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(entry, Mapping):
            if entry:
                rows.extend(_flatten_record(entry, prefix=path))
            else:
                rows.append((path, entry))
        else:
            rows.append((path, entry))
    return tuple(rows)


def _mapping_summary(
    value: object,
    plain: Mapping[str, object],
    *,
    prettytable: Any,
    color: bool,
) -> str:
    title = _human_result_title(value, plain)
    if title == "Resolved Configuration" and all(
        isinstance(plain.get(key), Mapping) for key in ("requested", "effective")
    ):
        sections: list[str] = []
        for key in ("requested", "effective"):
            configuration = cast(Mapping[str, object], plain[key])
            section = prettytable.PrettyTable(("field", "value"))
            section.title = _paint(
                f"{key.title()} Configuration",
                "CYAN",
                enabled=color,
            )
            section.align = "l"
            section.max_width["field"] = 42
            section.max_width["value"] = 74
            section.hrules = prettytable.HRuleStyle.FRAME
            for field, entry in _flatten_record(configuration):
                section.add_row(
                    (
                        _paint(field, "CYAN", enabled=color),
                        _record_color(
                            field.rsplit(".", maxsplit=1)[-1],
                            _value_text(entry),
                            color=color,
                        ),
                    )
                )
            sections.append(cast(str, section.get_string()))
        adjustments = plain.get("adjustments")
        if isinstance(adjustments, Sequence) and not isinstance(
            adjustments, (str, bytes)
        ):
            sections.append(
                _record_sequence_summary(
                    adjustments,
                    prettytable=prettytable,
                    color=color,
                    title="Adjustments",
                )
            )
        return "\n\n".join(sections)
    table = prettytable.PrettyTable(("field", "value"))
    table.title = _paint(title, "CYAN", enabled=color)
    table.align["field"] = "l"
    table.align["value"] = "l"
    table.max_width["field"] = 40
    table.max_width["value"] = 88
    table.hrules = prettytable.HRuleStyle.FRAME
    for key, entry in _flatten_record(plain):
        table.add_row(
            (
                _paint(key, "CYAN", enabled=color),
                _record_color(
                    key.rsplit(".", maxsplit=1)[-1],
                    _value_text(entry),
                    color=color,
                ),
            )
        )
    sections = [cast(str, table.get_string())]
    for raw_key, entry in plain.items():
        if (
            isinstance(entry, Sequence)
            and not isinstance(entry, (str, bytes))
            and entry
            and all(isinstance(item, Mapping) for item in entry)
        ):
            sections.append(
                _record_sequence_summary(
                    entry,
                    prettytable=prettytable,
                    color=color,
                    title=str(raw_key).replace("_", " ").title(),
                )
            )
    return "\n\n".join(sections)


def render_summary(value: object, *, color: bool = False) -> str | None:
    """Return colored human tables for every CLI result shape."""

    try:
        prettytable: Any = importlib.import_module("prettytable")
    except ImportError:
        return None
    from pyamplicol.api.results import (
        BenchmarkResult,
        GenerationResult,
        ResolvedEvaluation,
    )

    if isinstance(value, GenerationResult):
        return _generation_summary(value, prettytable=prettytable, color=color)
    if isinstance(value, ResolvedEvaluation):
        return _resolved_evaluation_summary(
            value,
            prettytable=prettytable,
            color=color,
        )
    if isinstance(value, tuple) and all(_is_evaluation_value(item) for item in value):
        return _total_evaluation_summary(
            cast(tuple[object, ...], value),
            prettytable=prettytable,
            color=color,
        )
    plain = _plain(value)
    if isinstance(value, BenchmarkResult):
        return _benchmark_summary(value, prettytable=prettytable, color=color)
    if (
        isinstance(plain, Mapping)
        and plain.get("kind") == "pyamplicol-artifact-inspection"
    ):
        return _artifact_inspection_summary(
            plain,
            prettytable=prettytable,
            color=color,
        )
    if isinstance(plain, Mapping) and isinstance(plain.get("process_group"), Mapping):
        # `inspect --process MEMBER --full-physics` of a grouped process
        physics = {key: entry for key, entry in plain.items() if key != "process_group"}
        return "\n\n".join(
            (
                _process_group_selection_summary(
                    cast(Mapping[str, object], plain["process_group"]),
                    prettytable=prettytable,
                    color=color,
                ),
                _mapping_summary(
                    physics, physics, prettytable=prettytable, color=color
                ),
            )
        )
    if isinstance(plain, Mapping):
        return _mapping_summary(
            value,
            plain,
            prettytable=prettytable,
            color=color,
        )
    if isinstance(plain, Sequence) and not isinstance(plain, (str, bytes)):
        return _record_sequence_summary(
            plain,
            prettytable=prettytable,
            color=color,
            title=_human_result_title(value, plain),
        )
    table = prettytable.PrettyTable(("field", "value"))
    table.title = _paint(_human_result_title(value, plain), "CYAN", enabled=color)
    table.align = "l"
    table.hrules = prettytable.HRuleStyle.FRAME
    table.add_row(
        (
            _paint("result", "CYAN", enabled=color),
            _paint(_value_text(plain), "GREEN", enabled=color),
        )
    )
    return cast(str, table.get_string())


__all__ = ["render_summary"]
