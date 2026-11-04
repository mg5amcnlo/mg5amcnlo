# SPDX-License-Identifier: 0BSD
"""Typed calibrated runtime profiling for Rusticol runtime backends."""

from __future__ import annotations

import math
import os
import platform
import statistics
import threading
import time
import tomllib
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

from pyamplicol.api.errors import EvaluationError
from pyamplicol.api.protocols import Momenta, RuntimeBackend
from pyamplicol.api.results import (
    BenchmarkComponentTiming,
    BenchmarkProfileCounters,
    BenchmarkResult,
    BenchmarkStageTiming,
    BenchmarkStatistics,
    BenchmarkTimingBreakdown,
    ProcessPhysics,
)
from pyamplicol.config import BenchmarkConfig, RunConfig
from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)

_MAX_SAMPLE_RUNTIME_SECONDS = 0.25
_MAX_CALIBRATION_BLOCKS = 2
_MAX_REPETITIONS_PER_SAMPLE = 1_000_000_000
# The calibrated runtime may be optimistic or the host may speed up after
# warmup. Keep extending complete blocks, but fail closed if timing changes so
# drastically that four times the initial plan still cannot reach the target.
_MAX_TARGET_RUNTIME_SAMPLE_FACTOR = 4
_CALIBRATION_LOWER_RATIO = 0.8
_CALIBRATION_UPPER_RATIO = 1.25
_MIN_CLOCK_INTERVAL_SECONDS = 1.0e-12
_MAX_NATIVE_PROFILE_SAMPLES = 8
_LC_TOPOLOGY_REPLAY_LAYOUT = "topology-replay"
_LC_ALL_FLOW_UNION_LAYOUT = "all-flow-union"
_LC_TOPOLOGY_REPLAY_PROFILE_RECOMMENDATION = (
    "this LC topology-replay artifact is being profiled outside its optimized "
    "single-flow/helicity-sum workload; pass exactly one --color-flow and no "
    "--helicity, or regenerate with --lc-flow-layout all-flow-union for the "
    "all-flows/single-helicity workload"
)
_LC_ALL_FLOW_UNION_PROFILE_RECOMMENDATION = (
    "this LC all-flow-union artifact is being profiled outside its optimized "
    "all-flows/single-helicity workload; pass exactly one --helicity and no "
    "--color-flow"
)
_LC_PROFILE_WARNING_ATTRIBUTE = "_pyamplicol_non_hot_profile_warning_emitted"
_LC_PROFILE_WARNING_LOCK = threading.Lock()
_LC_PROFILE_WARNING_FALLBACK_OWNERS: dict[int, object] = {}
_EVALUATOR_TOTAL_SAMPLE_CONTRACT = "accumulated-repeated-warmed-evaluator-total-v1"
_RECURRENCE_LIKE_EXECUTION_MODES = {"recurrence", "on-the-fly"}
_OTF_RUNTIME_STATE_CENSUS_KIND = "rusticol-on-the-fly-runtime-state-census-v1"
_OTF_FAMILY_CACHE_POLICY = "last-family-only"
_OTF_FAMILY_CACHE_LIMIT = 1
_OTF_RUNTIME_STATE_COUNT_FIELDS = (
    "process_preparation_count",
    "retained_family_count",
    "pending_family_count",
    "retained_selection_count",
    "retained_request_count",
    "retained_amplitude_destination_count",
    "retained_executor_handle_count",
    "retained_query_local_trace_count",
    "retained_embedded_lookup_key_count",
    "semantic_executor_binding_count",
)
_OTF_RETAINED_STATE_EXECUTABLE_COUNT_FIELDS = (
    "retained_amplitude_destination_count",
    "retained_executor_handle_count",
    "semantic_executor_binding_count",
)
_OTF_ACTIVE_FAMILY_COUNT_FIELDS = (
    "query_count",
    "union_unique_current_count",
    "union_unique_current_component_count",
    "union_source_rows",
    "union_contribution_rows",
    "union_finalization_rows",
    "union_closure_rows",
    "union_amplitude_destination_count",
    "union_source_executor_call_groups",
    "union_contribution_executor_call_groups",
    "union_finalization_executor_call_groups",
    "union_closure_executor_call_groups",
)
_OTF_OPERATION_ROLES = ("source", "contribution", "finalization", "closure")
_COMPILED_DIRECT_ARENA_COUNTER_KEYS = (
    "compiled_direct_arena_engine_count",
    "compiled_direct_arena_call_count",
    "compiled_direct_arena_boundary_input_bytes",
    "compiled_direct_arena_boundary_current_output_bytes",
    "compiled_direct_arena_boundary_amplitude_output_bytes",
)


@dataclass(frozen=True, slots=True)
class _Calibration:
    sample_count: int
    repetitions_per_sample: int
    target_sample_seconds: float
    probe_seconds: float
    block_count: int
    evaluation_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _CompactBenchmarkContext:
    process_id: str
    process_expression: str
    color_accuracy: str
    helicity_count: int
    color_count: int
    selected_color_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NativeProfileSample:
    execution_mode: str | None
    compiled_direct_arena_active: bool
    wall_time: float | None
    native_input_pack_time: float | None
    native_input_crossing_time: float | None
    orchestration_time: float | None
    state_prepare_time: float | None
    state_clear_time: float | None
    source_fill_time: float | None
    momentum_input_setup_time: float | None
    momentum_setup_time: float | None
    model_parameter_setup_time: float | None
    stage_input_pack_time: float | None
    stage_leaf_input_pack_time: float | None
    stage_evaluator_call_time: float
    stage_backend_call_time: float | None
    stage_evaluator_output_gather_time: float | None
    output_assign_time: float | None
    amplitude_input_pack_time: float | None
    amplitude_evaluator_call_time: float | None
    amplitude_leaf_input_pack_time: float | None
    amplitude_backend_call_time: float | None
    amplitude_evaluator_output_gather_time: float | None
    amplitude_output_remap_time: float | None
    reduction_time: float | None
    total_materialization_time: float | None
    final_output_copy_time: float | None
    selector_planner_time: float | None
    selector_gather_time: float | None
    selector_scatter_time: float | None
    other_core_time: float | None
    stage_input_pack_times: tuple[float, ...] | None
    stage_leaf_input_pack_times: tuple[float, ...] | None
    stage_evaluator_call_times: tuple[float, ...] | None
    stage_backend_call_times: tuple[float, ...] | None
    stage_evaluator_output_gather_times: tuple[float, ...] | None
    stage_output_assign_times: tuple[float, ...] | None
    eager_initialize_time: float | None
    eager_gather_time: float | None
    eager_kernel_call_time: float | None
    eager_invocation_scatter_time: float | None
    eager_finalization_time: float | None
    eager_scatter_finalization_time: float | None
    eager_closure_time: float | None
    eager_copy_out_time: float | None
    recurrence_momentum_fill_time: float | None
    recurrence_union_source_fill_time: float | None
    recurrence_schedule_time: float | None
    recurrence_source_kernel_time: float | None
    recurrence_contribution_kernel_time: float | None
    recurrence_finalization_time: float | None
    recurrence_closure_time: float | None
    recurrence_replay_output_mapping_time: float | None
    counters: _NativeProfileCounterSample | None


@dataclass(frozen=True, slots=True)
class _NativeProfileCounterSample:
    native_input_components_per_point: float | None
    native_input_pack_bytes_per_point: float | None
    native_input_crossing_bytes_per_point: float | None
    state_components_per_point: float | None
    state_clear_components_per_point: float | None
    source_components_per_point: float | None
    momentum_components_per_point: float | None
    model_parameter_components_per_point: float | None
    stage_input_copy_components_per_point: float | None
    stage_leaf_input_copy_components_per_point: float | None
    stage_evaluator_output_gather_components_per_point: float | None
    stage_output_assign_components_per_point: float | None
    amplitude_input_copy_components_per_point: float | None
    amplitude_leaf_input_copy_components_per_point: float | None
    amplitude_evaluator_output_gather_components_per_point: float | None
    amplitude_output_remap_components_per_point: float | None
    reduction_input_components_per_point: float | None
    selector_gather_points_per_point: float | None
    selector_gather_bytes_per_point: float | None
    selector_scatter_values_per_point: float | None
    resolved_materialized_components_per_point: float | None
    total_materialized_values_per_point: float | None
    final_output_copy_values_per_point: float | None
    native_input_container_allocations_per_call: float | None
    evaluator_backend_calls_per_call: float | None
    compiled_direct_arena_engines_per_call: float | None
    compiled_direct_arena_calls_per_call: float | None
    compiled_direct_arena_boundary_input_bytes_per_call: float | None
    compiled_direct_arena_boundary_current_output_bytes_per_call: float | None
    compiled_direct_arena_boundary_amplitude_output_bytes_per_call: float | None
    recurrence_momentum_scalar_values_per_point: float | None
    recurrence_schedule_executions_per_call: float | None
    recurrence_replay_schedule_executions_per_call: float | None
    recurrence_union_schedule_executions_per_call: float | None
    recurrence_union_source_rows_per_call: float | None
    recurrence_replay_output_values_per_point: float | None
    recurrence_source_calls_per_call: float | None
    recurrence_source_rows_per_call: float | None
    recurrence_contribution_calls_per_call: float | None
    recurrence_contribution_rows_per_call: float | None
    recurrence_finalization_calls_per_call: float | None
    recurrence_finalization_rows_per_call: float | None
    recurrence_closure_calls_per_call: float | None
    recurrence_closure_rows_per_call: float | None
    observed_scratch_reallocations_per_call: float | None
    native_output_allocations_per_call: float | None


def _benchmark_target_path(
    target: RuntimeBackend | os.PathLike[str] | str,
) -> Path | None:
    if not isinstance(target, (str, os.PathLike)):
        return None
    return Path(os.fspath(target)).expanduser().resolve(strict=False)


class BenchmarkBackend:
    """Measure the optimized summed runtime path without changing its semantics."""

    def __init__(
        self,
        config: BenchmarkConfig | RunConfig | None,
        progress: ProgressSink | None,
    ) -> None:
        self._run_config = config if isinstance(config, RunConfig) else None
        self._config = (
            config.benchmark
            if isinstance(config, RunConfig)
            else config or BenchmarkConfig()
        )
        self._progress = progress
        self._chunk_guard: Callable[[float | None, str], None] | None = None

    def set_chunk_guard(
        self,
        guard: Callable[[float | None, str], None],
    ) -> None:
        """Install a report-owned guard called before every profiling chunk."""

        if not callable(guard):
            raise TypeError("benchmark chunk guard must be callable")
        self._chunk_guard = guard

    def run(
        self,
        target: RuntimeBackend | os.PathLike[str] | str,
        *,
        points: Momenta | None = None,
    ) -> BenchmarkResult:
        target_path = _benchmark_target_path(target)
        runtime = self._runtime(target)
        if (
            str(getattr(runtime, "execution_mode", "compiled")) == "on-the-fly"
            and self._config.precision != 16
        ):
            raise EvaluationError(
                "on-the-fly benchmarking supports only precision=16 (native f64); "
                f"received precision={self._config.precision}"
            )
        compact_context = _on_the_fly_benchmark_context(
            runtime,
            self._config.color_flow_ids,
        )
        physics = None if compact_context is not None else runtime.physics
        if points is None:
            loader = getattr(runtime, "validation_momenta", None)
            points = loader() if callable(loader) else None
        if points is None or len(points) == 0:
            raise EvaluationError(
                "benchmarking requires at least one phase-space point and the "
                "selected runtime has no deterministic validation point"
            )
        batch = _benchmark_batch(points, self._config.batch_size)
        helicities = self._config.helicity_ids or None
        color_flows = (
            compact_context.selected_color_ids
            if compact_context is not None
            else _resolve_color_flow_ordinals(
                cast(ProcessPhysics, physics),
                self._config.color_flow_ids,
            )
        ) or None
        color_accuracy = (
            compact_context.color_accuracy
            if compact_context is not None
            else cast(ProcessPhysics, physics).color_accuracy
        )
        process_id = (
            compact_context.process_id
            if compact_context is not None
            else cast(ProcessPhysics, physics).process_id
        )
        process_expression = (
            compact_context.process_expression
            if compact_context is not None
            else cast(ProcessPhysics, physics).process
        )
        lc_flow_layout: str | None = None
        if compact_context is None:
            artifact_path = target_path or _runtime_artifact_path(runtime)
            lc_flow_layout = _artifact_lc_flow_layout(artifact_path)
            helicities, color_flows = _default_lc_profile_selectors(
                physics=cast(ProcessPhysics, physics),
                lc_flow_layout=lc_flow_layout,
                selected_helicity_ids=tuple(helicities or ()),
                selected_color_ids=tuple(color_flows or ()),
            )
        lc_flow_layout_recommendation = _lc_flow_layout_recommendation(
            color_accuracy=color_accuracy,
            lc_flow_layout=lc_flow_layout,
            selected_helicity_ids=tuple(helicities or ()),
            selected_color_ids=tuple(color_flows or ()),
        )
        if lc_flow_layout_recommendation is not None:
            _warn_non_hot_lc_profile_once(
                runtime,
                lc_flow_layout_recommendation,
            )
        profiler = _native_profiler(runtime) if self._config.precision == 16 else None
        repeated_profiler = (
            _native_repeated_profiler(runtime) if self._config.precision == 16 else None
        )
        native_wall_timer = (
            _native_wall_timer(runtime) if self._config.precision == 16 else None
        )
        timer_seconds_per_repetition: float | None = None
        profiler_seconds_per_repetition: float | None = None
        chunk_guard_failure: Exception | None = None

        def guard_chunk(estimated_seconds: float | None, description: str) -> None:
            nonlocal chunk_guard_failure
            if self._chunk_guard is not None:
                try:
                    self._chunk_guard(estimated_seconds, description)
                except Exception as error:
                    chunk_guard_failure = error
                    raise

        def evaluate_once() -> object:
            return runtime.evaluate(
                batch,
                helicities=helicities,
                color_flows=color_flows,
                precision=self._config.precision,
            )

        def measure_repetitions(repetitions: int) -> float:
            nonlocal timer_seconds_per_repetition
            estimate = (
                None
                if timer_seconds_per_repetition is None
                else timer_seconds_per_repetition * repetitions
            )
            guard_chunk(estimate, "runtime evaluator timing chunk")
            if native_wall_timer is not None:
                observed = native_wall_timer(
                    batch,
                    repetitions,
                    helicities=helicities,
                    color_flows=color_flows,
                    precision=self._config.precision,
                )
            else:
                observed = _timed_repetitions(evaluate_once, repetitions)
            timer_seconds_per_repetition = observed / repetitions
            return observed

        def profile_repetitions(
            repetitions: int,
            *,
            measure_elapsed: bool = True,
        ) -> tuple[object, float]:
            nonlocal profiler_seconds_per_repetition
            estimated_per_repetition = (
                profiler_seconds_per_repetition
                if profiler_seconds_per_repetition is not None
                else timer_seconds_per_repetition
            )
            estimate = (
                None
                if estimated_per_repetition is None
                else estimated_per_repetition * repetitions
            )
            guard_chunk(estimate, "runtime attribution chunk")
            started = time.perf_counter() if measure_elapsed else None
            if repeated_profiler is not None:
                result = repeated_profiler(
                    batch,
                    repetitions,
                    helicities=helicities,
                    color_flows=color_flows,
                    precision=self._config.precision,
                    include_values=False,
                )
                divisor = repetitions
            elif profiler is not None:
                result = profiler(
                    batch,
                    helicities=helicities,
                    color_flows=color_flows,
                    precision=self._config.precision,
                    include_values=False,
                )
                divisor = 1
            else:  # pragma: no cover - caller checks profiler availability.
                raise EvaluationError("runtime profiler is unavailable")
            elapsed_seconds = (
                (time.perf_counter() - started)
                if started is not None
                else (estimate or 0.0)
            )
            if elapsed_seconds > 0.0:
                profiler_seconds_per_repetition = elapsed_seconds / divisor
            return result, elapsed_seconds

        task_id = "runtime-benchmark"
        calibration_task_id = "runtime-profile-calibration"
        active_task_id = calibration_task_id
        cold_warmup_elapsed: float | None = None
        cold_warmup_state_before: dict[str, object] | None = None
        cold_warmup_state_after: dict[str, object] | None = None
        cold_warmup_runtime_was_cold: bool | None = None
        cold_warmup_runtime_was_retained: bool | None = None
        cold_warmup_runtime_is_retained: bool | None = None
        warmup_elapsed = 0.0
        warmup_run_outer_wall_seconds: list[float] = []
        calibration: _Calibration | None = None
        samples: list[float] = []
        evaluator_samples: list[float] | None = (
            [] if profiler is not None or repeated_profiler is not None else None
        )
        native_profile_samples: list[_NativeProfileSample] | None = (
            [] if profiler is not None or repeated_profiler is not None else None
        )
        elapsed = 0.0
        evaluator_total_elapsed = 0.0
        evaluator_elapsed = 0.0
        interrupted = False
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    calibration_task_id,
                    "Calibrating runtime profile",
                    total=None,
                )
            )
        try:
            if compact_context is not None:
                cold_warmup_state_before = _on_the_fly_runtime_state_census(
                    runtime,
                    expected_process_id=compact_context.process_id,
                )
                cold_warmup_runtime_was_cold = _on_the_fly_runtime_state_is_cold(
                    cold_warmup_state_before
                )
                cold_warmup_runtime_was_retained = (
                    _on_the_fly_runtime_state_is_retained(cold_warmup_state_before)
                )
                if not (
                    cold_warmup_runtime_was_cold or cold_warmup_runtime_was_retained
                ):
                    raise EvaluationError(
                        "on-the-fly runtime state before the first requested "
                        "evaluation is neither cold nor fully retained"
                    )
                cold_warmup_started = time.perf_counter()
                measure_repetitions(1)
                cold_warmup_elapsed = time.perf_counter() - cold_warmup_started
                cold_warmup_state_after = _on_the_fly_runtime_state_census(
                    runtime,
                    expected_process_id=compact_context.process_id,
                )
                cold_warmup_runtime_is_retained = _on_the_fly_runtime_state_is_retained(
                    cold_warmup_state_after
                )
                if not cold_warmup_runtime_is_retained:
                    raise EvaluationError(
                        "on-the-fly first requested evaluation did not leave a fully "
                        "retained runtime family"
                    )
                # The required first-selector evaluation can spend minutes
                # constructing a cold high-multiplicity family. It is reported
                # separately as OTF warm-up evidence and must not become the
                # estimate used to guard or calibrate retained evaluations.
                timer_seconds_per_repetition = None
            last_warmup_seconds: float | None = None
            for warmup_index in range(self._config.warmup_runs):
                warmup_started = time.perf_counter()
                last_warmup_seconds = measure_repetitions(1)
                if repeated_profiler is not None or profiler is not None:
                    profile_repetitions(1, measure_elapsed=False)
                warmup_run_elapsed = time.perf_counter() - warmup_started
                warmup_run_outer_wall_seconds.append(warmup_run_elapsed)
                warmup_elapsed += warmup_run_elapsed
                if self._progress is not None:
                    self._progress.emit(
                        ProgressUpdate(
                            calibration_task_id,
                            completed=warmup_index + 1,
                            total=None,
                            message="warmup",
                        )
                    )

            calibration_outer_started = time.perf_counter()
            calibration = _calibrate_repetitions(
                evaluate_once,
                self._config,
                initial_seconds=last_warmup_seconds,
                timer=measure_repetitions,
            )
            calibration_outer_elapsed = time.perf_counter() - calibration_outer_started
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(
                        calibration_task_id,
                        message=(
                            f"{calibration.sample_count} blocks x "
                            f"{calibration.repetitions_per_sample} repetitions"
                        ),
                    )
                )
                self._progress.emit(
                    ProgressStart(
                        task_id,
                        "Profiling runtime",
                        total=calibration.sample_count,
                    )
                )
            active_task_id = task_id
            repetitions = calibration.repetitions_per_sample
            initial_planned_sample_count = calibration.sample_count
            planned_sample_count = initial_planned_sample_count
            maximum_sample_count = (
                initial_planned_sample_count * _MAX_TARGET_RUNTIME_SAMPLE_FACTOR
            )
            native_profile_sample_limit = (
                planned_sample_count
                if repeated_profiler is not None
                else min(
                    planned_sample_count,
                    max(
                        1,
                        min(
                            self._config.minimum_samples,
                            _MAX_NATIVE_PROFILE_SAMPLES,
                        ),
                    ),
                )
            )
            try:
                sample_index = 0
                while sample_index < planned_sample_count:
                    native_sample: _NativeProfileSample | None = None
                    profile_duration = 0.0
                    sample_started = time.perf_counter()
                    evaluator_total_duration = measure_repetitions(repetitions)
                    wall_duration = time.perf_counter() - sample_started
                    sample_seconds_per_point = wall_duration / (
                        repetitions * len(batch)
                    )
                    if repeated_profiler is not None:
                        profile, profile_duration = profile_repetitions(repetitions)
                        native_sample = _native_profile_sample(
                            profile,
                            len(batch) * repetitions,
                            repetitions=repetitions,
                        )
                        if native_sample.wall_time is None:
                            raise EvaluationError(
                                "repeated native profile did not report core wall time"
                            )
                    elif (
                        profiler is not None
                        and evaluator_samples is not None
                        and native_profile_samples is not None
                        and sample_index < native_profile_sample_limit
                    ):
                        profile, profile_duration = profile_repetitions(1)
                        native_sample = _native_profile_sample(profile, len(batch))

                    samples.append(sample_seconds_per_point)
                    elapsed += wall_duration
                    evaluator_total_elapsed += evaluator_total_duration
                    if native_sample is not None:
                        assert evaluator_samples is not None
                        assert native_profile_samples is not None
                        native_profile_samples.append(native_sample)
                        if (
                            native_sample.execution_mode
                            in _RECURRENCE_LIKE_EXECUTION_MODES
                        ):
                            if native_sample.recurrence_schedule_time is None:
                                raise EvaluationError(
                                    "native recurrence schedule timing is unavailable"
                                )
                            evaluator_samples.append(
                                native_sample.recurrence_schedule_time
                            )
                        elif native_sample.compiled_direct_arena_active:
                            if native_sample.orchestration_time is None:
                                raise EvaluationError(
                                    "native compiled Direct-Arena orchestration "
                                    "timing is unavailable"
                                )
                            evaluator_samples.append(native_sample.orchestration_time)
                        else:
                            evaluator_samples.append(
                                native_sample.stage_evaluator_call_time
                                + (native_sample.amplitude_evaluator_call_time or 0.0)
                            )
                        evaluator_elapsed += profile_duration
                    sample_index += 1
                    if (
                        sample_index == planned_sample_count
                        and elapsed < self._config.target_runtime
                    ):
                        if planned_sample_count >= maximum_sample_count:
                            raise EvaluationError(
                                "runtime benchmark could not reach its target "
                                f"duration of {self._config.target_runtime:g}s "
                                f"within {maximum_sample_count} complete blocks"
                            )
                        mean_block_seconds = elapsed / len(samples)
                        additional_samples = max(
                            1,
                            math.ceil(
                                (self._config.target_runtime - elapsed)
                                / max(
                                    mean_block_seconds,
                                    _MIN_CLOCK_INTERVAL_SECONDS,
                                )
                            ),
                        )
                        planned_sample_count = min(
                            maximum_sample_count,
                            planned_sample_count + additional_samples,
                        )
                        if repeated_profiler is not None:
                            native_profile_sample_limit = planned_sample_count
                    if self._progress is not None:
                        message, details = _sample_progress_payload(
                            samples,
                            elapsed_seconds=elapsed,
                            target_seconds=self._config.target_runtime,
                            repetitions=repetitions,
                            batch_size=len(batch),
                        )
                        self._progress.emit(
                            ProgressUpdate(
                                task_id,
                                completed=len(samples),
                                total=planned_sample_count,
                                message=message,
                                details=details,
                            )
                        )
            except KeyboardInterrupt:
                if not samples:
                    raise
                interrupted = True
                if self._progress is not None:
                    self._progress.emit(
                        ProgressEnd(
                            task_id,
                            success=False,
                            message=(
                                f"interrupted after {len(samples)}/"
                                f"{planned_sample_count} complete blocks; "
                                "reporting partial statistics"
                            ),
                        )
                    )
        except KeyboardInterrupt:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(
                        active_task_id,
                        success=False,
                        message="interrupted before a complete timing block",
                    )
                )
            raise
        except Exception as exc:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(active_task_id, success=False, message=str(exc))
                )
            if isinstance(exc, EvaluationError):
                raise
            if exc is chunk_guard_failure:
                raise
            raise EvaluationError(f"runtime benchmark failed: {exc}") from exc
        if self._progress is not None and not interrupted:
            self._progress.emit(ProgressEnd(task_id))

        assert calibration is not None

        wall_samples = samples
        wall_time_source = "python_outer_perf_counter_wall_time"
        wall_sample_pass = "time.perf_counter[measure_repetitions]"
        evaluator_total_sample_pass = (
            "runtime._benchmark_f64_wall_time"
            if native_wall_timer is not None
            else "runtime.evaluate"
        )
        timing_sample_contract = (
            "paired_unprofiled_headline_profiled_attribution_v1"
            if repeated_profiler is not None
            else "separate_native_profile_diagnostic_v1"
        )
        mean, deviation, error, relative_error = _sample_statistics(wall_samples)
        uncertainty = BenchmarkStatistics(deviation, error, relative_error)
        compiled_direct_arena_active = False
        if evaluator_samples is None:
            evaluator_time_per_point = None
            evaluator_uncertainty = None
            timing_breakdown = None
            evaluator_environment: dict[str, object] = {
                "wall_time_source": wall_time_source,
                "wall_time_sample_pass": wall_sample_pass,
                "evaluator_time_source": "unavailable",
                "evaluator_time_sample_pass": "unavailable",
                "timing_sample_contract": "headline_only_no_breakdown_v1",
                "native_profile_unavailable_reason": (
                    "non_f64_precision" if self._config.precision != 16 else None
                ),
            }
        else:
            assert native_profile_samples is not None
            compiled_direct_arena_states = {
                sample.compiled_direct_arena_active for sample in native_profile_samples
            }
            if len(compiled_direct_arena_states) > 1:
                raise EvaluationError(
                    "native compiled Direct-Arena activity changed between samples"
                )
            compiled_direct_arena_active = next(
                iter(compiled_direct_arena_states),
                False,
            )
            (
                evaluator_time_per_point,
                evaluator_deviation,
                evaluator_error,
                evaluator_relative,
            ) = _sample_statistics(evaluator_samples)
            evaluator_uncertainty = BenchmarkStatistics(
                evaluator_deviation,
                evaluator_error,
                evaluator_relative,
            )
            timing_breakdown = _timing_breakdown(native_profile_samples)
            if timing_breakdown.execution_mode in _RECURRENCE_LIKE_EXECUTION_MODES:
                evaluator_time_source = "runtime_profile_core_recurrence_schedule_time"
            elif compiled_direct_arena_active:
                evaluator_time_source = (
                    "runtime_profile_core_compiled_direct_arena_orchestration_time"
                )
            else:
                evaluator_time_source = "runtime_profile_core_evaluator_call_time"
            native_profile_repetitions = (
                calibration.repetitions_per_sample
                if repeated_profiler is not None
                else 1
            )
            native_profile_warmup_calls = (
                self._config.warmup_runs
                if profiler is not None or repeated_profiler is not None
                else 0
            )
            paired_profile_attribution = repeated_profiler is not None
            evaluator_environment = {
                "wall_time_source": wall_time_source,
                "wall_time_sample_pass": wall_sample_pass,
                "evaluator_time_source": evaluator_time_source,
                "compiled_direct_arena_active": compiled_direct_arena_active,
                "evaluator_time_sample_pass": (
                    "runtime.profile_repeated"
                    if repeated_profiler is not None
                    else "runtime.profile"
                ),
                "timing_breakdown_sample_pass": (
                    "runtime.profile_repeated"
                    if repeated_profiler is not None
                    else "runtime.profile"
                ),
                "evaluator_sample_count": len(evaluator_samples),
                "evaluator_elapsed_seconds": evaluator_elapsed,
                "native_profile_sample_count": len(evaluator_samples),
                "native_profile_sample_limit": native_profile_sample_limit,
                "native_profile_repetitions_per_sample": (native_profile_repetitions),
                "native_profile_batch_size": len(batch),
                "native_profile_points_per_sample": (
                    native_profile_repetitions * len(batch)
                ),
                "native_profile_calls_per_block": (
                    len(evaluator_samples) / len(samples) if samples else 0.0
                ),
                "native_profile_warmup_call_count": native_profile_warmup_calls,
                "native_profile_total_call_count": (
                    native_profile_warmup_calls + len(evaluator_samples)
                ),
                "profile_attribution_paired_with_headline": (
                    paired_profile_attribution
                ),
                "profile_attribution_identical_batch": paired_profile_attribution,
                "profile_attribution_identical_repetitions": (
                    paired_profile_attribution
                ),
                "profile_attribution_evaluation_count": (
                    len(evaluator_samples) * native_profile_repetitions
                ),
                "profile_attribution_point_count": (
                    len(evaluator_samples) * native_profile_repetitions * len(batch)
                ),
                "timing_sample_contract": timing_sample_contract,
                "evaluator_standard_deviation_seconds_per_point": (evaluator_deviation),
                "evaluator_standard_error_seconds_per_point": evaluator_error,
                "evaluator_relative_standard_error": evaluator_relative,
            }
        raw_evaluator_time_per_point = evaluator_time_per_point
        evaluator_below_timer_resolution = (
            raw_evaluator_time_per_point is not None
            and compiled_direct_arena_active
            and raw_evaluator_time_per_point == 0.0
        )
        if evaluator_below_timer_resolution:
            evaluator_time_per_point = None
        evaluator_environment.update(
            {
                "evaluator_time_status": (
                    "below_timer_resolution"
                    if evaluator_below_timer_resolution
                    else (
                        "measured"
                        if raw_evaluator_time_per_point is not None
                        else "unavailable"
                    )
                ),
                "evaluator_time_ratio_eligible": (
                    raw_evaluator_time_per_point is not None
                    and not evaluator_below_timer_resolution
                    and raw_evaluator_time_per_point > 0.0
                ),
                "evaluator_time_raw_seconds_per_point": (raw_evaluator_time_per_point),
            }
        )
        execution_mode = (
            timing_breakdown.execution_mode
            if timing_breakdown is not None
            else str(getattr(runtime, "execution_mode", "unavailable"))
        )
        selected_helicity_ids = tuple(helicities or ())
        selected_color_ids = tuple(color_flows or ())
        effective_config = replace(
            self._config,
            helicity_ids=selected_helicity_ids,
            color_flow_ids=selected_color_ids,
        )
        measured_evaluations = len(samples) * calibration.repetitions_per_sample
        measured_point_count = measured_evaluations * len(batch)
        evaluator_total_time_per_point = evaluator_total_elapsed / measured_point_count
        layout_environment: dict[str, object] = {}
        if lc_flow_layout is not None:
            layout_environment = {
                "lc_flow_layout": lc_flow_layout,
                "lc_flow_layout_recommendation": lc_flow_layout_recommendation,
            }
        cold_warmup_environment: dict[str, object] = {}
        if cold_warmup_elapsed is not None:
            if (
                cold_warmup_state_before is None
                or cold_warmup_state_after is None
                or cold_warmup_runtime_was_cold is None
                or cold_warmup_runtime_was_retained is None
                or cold_warmup_runtime_is_retained is None
            ):
                raise EvaluationError(
                    "on-the-fly benchmark did not retain complete runtime state "
                    "evidence"
                )
            cold_warmup_environment = {
                "cold_warmup_elapsed_seconds": cold_warmup_elapsed,
                "cold_warmup_run_count": 1,
                "cold_warmup_batch_size": len(batch),
                "cold_warmup_point_count": len(batch),
                "cold_warmup_timer_source": "python_outer_time.perf_counter",
                "cold_warmup_timing_scope": (
                    "one initial requested-selector Runtime evaluation on the full "
                    "benchmark batch; artifact generation and Runtime/artifact load "
                    "are excluded"
                ),
                "cold_warmup_runtime_freshness": (
                    "authenticated-cold"
                    if cold_warmup_runtime_was_cold
                    else "authenticated-already-retained"
                ),
                "cold_warmup_runtime_state_evidence": (
                    "authenticated-native-otf-census-v1"
                ),
                "cold_warmup_runtime_state_before": cold_warmup_state_before,
                "cold_warmup_runtime_state_after": cold_warmup_state_after,
                "cold_warmup_runtime_cold_before_first_evaluation": (
                    cold_warmup_runtime_was_cold
                ),
                "cold_warmup_runtime_retained_before_first_evaluation": (
                    cold_warmup_runtime_was_retained
                ),
                "cold_warmup_runtime_retained_after_first_evaluation": (
                    cold_warmup_runtime_is_retained
                ),
                "cold_warmup_ratio_eligible": False,
                "cold_warmup_acceptance_eligible": False,
            }
        return BenchmarkResult(
            requested_config=self._config,
            effective_config=effective_config,
            sample_count=len(samples),
            wall_time_per_point=mean,
            evaluator_time_per_point=evaluator_time_per_point,
            uncertainty=uncertainty,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "target": None if target_path is None else str(target_path),
                "batch_size": len(batch),
                "precision": self._config.precision,
                "execution_mode": execution_mode,
                "color_accuracy": color_accuracy,
                "color_workload": (
                    _compact_color_workload_text(compact_context, selected_color_ids)
                    if compact_context is not None
                    else _color_workload_text(
                        cast(ProcessPhysics, physics),
                        selected_color_ids,
                    )
                ),
                "helicity_workload": (
                    _compact_helicity_workload_text(
                        compact_context,
                        selected_helicity_ids,
                    )
                    if compact_context is not None
                    else _helicity_workload_text(
                        cast(ProcessPhysics, physics),
                        selected_helicity_ids,
                    )
                ),
                "selected_color_ids": selected_color_ids,
                "selected_helicity_ids": selected_helicity_ids,
                "elapsed_seconds": elapsed,
                "interrupted": interrupted,
                "completed_sample_count": len(samples),
                "completion_fraction": len(samples) / planned_sample_count,
                "warmup_elapsed_seconds": warmup_elapsed,
                "warmup_configured_run_count": self._config.warmup_runs,
                "warmup_batch_size": len(batch),
                "warmup_point_count": self._config.warmup_runs * len(batch),
                "warmup_run_outer_wall_seconds": tuple(warmup_run_outer_wall_seconds),
                "first_warmup_run_outer_wall_seconds": (
                    warmup_run_outer_wall_seconds[0]
                    if warmup_run_outer_wall_seconds
                    else None
                ),
                "warmup_timer_source": "python_outer_time.perf_counter",
                "warmup_timing_scope": (
                    "configured benchmark warm-up iteration outer wall; includes "
                    "the headline evaluation and optional native-profile warm-up; "
                    "artifact generation and Runtime/artifact load are excluded"
                ),
                "planned_sample_count": planned_sample_count,
                "initial_planned_sample_count": initial_planned_sample_count,
                "adaptive_extension_sample_count": (
                    planned_sample_count - initial_planned_sample_count
                ),
                "repetitions_per_sample": calibration.repetitions_per_sample,
                "measured_evaluation_count": measured_evaluations,
                "measured_point_count": measured_point_count,
                "evaluator_total_time_raw_seconds_per_point": (
                    evaluator_total_time_per_point
                ),
                "evaluator_total_time_status": "measured",
                "evaluator_total_time_ratio_eligible": False,
                "evaluator_total_time_source": (
                    f"{evaluator_total_sample_pass}.accumulated"
                ),
                "evaluator_total_time_sample_contract": (
                    _EVALUATOR_TOTAL_SAMPLE_CONTRACT
                ),
                "evaluator_total_accumulated_seconds": evaluator_total_elapsed,
                "target_sample_seconds": calibration.target_sample_seconds,
                "calibration_probe_seconds": calibration.probe_seconds,
                "calibration_block_count": calibration.block_count,
                "calibration_evaluation_count": calibration.evaluation_count,
                "calibration_elapsed_seconds": calibration.elapsed_seconds,
                "calibration_outer_elapsed_seconds": calibration_outer_elapsed,
                **cold_warmup_environment,
                **layout_environment,
                **evaluator_environment,
            },
            interrupted=interrupted,
            repetitions_per_sample=calibration.repetitions_per_sample,
            evaluator_uncertainty=evaluator_uncertainty,
            evaluator_total_time_per_point=evaluator_total_time_per_point,
            process_id=process_id,
            process_expression=process_expression,
            timing_breakdown=timing_breakdown,
        )

    def _runtime(
        self,
        target: RuntimeBackend | os.PathLike[str] | str,
    ) -> RuntimeBackend:
        target_path = _benchmark_target_path(target)
        if target_path is None:
            return cast(RuntimeBackend, target)
        from pyamplicol.runtime import load_runtime_backend

        run = self._run_config
        process = None if run is None else run.evaluation.process
        path = target_path
        task_id = "process-output-load"
        started = time.perf_counter()
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    task_id,
                    f"Loading process output {path}",
                    details={"step": "loading process output", "path": str(path)},
                )
            )
        try:
            runtime = load_runtime_backend(
                path,
                process=process,
                model_parameters=None,
                mute_warnings=False,
            )
        except Exception as exc:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(
                        task_id,
                        success=False,
                        message=str(exc),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
            raise
        if self._progress is not None:
            self._progress.emit(
                ProgressEnd(
                    task_id,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
        return runtime


def _artifact_lc_flow_layout(target_path: Path | None) -> str | None:
    if target_path is None:
        return None
    effective_path = target_path / "config" / "effective.toml"
    try:
        payload = tomllib.loads(effective_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    color = payload.get("color")
    if not isinstance(color, Mapping) or color.get("accuracy") != "lc":
        return None
    layout = color.get("lc_flow_layout", _LC_TOPOLOGY_REPLAY_LAYOUT)
    if layout not in {_LC_TOPOLOGY_REPLAY_LAYOUT, _LC_ALL_FLOW_UNION_LAYOUT}:
        return None
    return str(layout)


def _runtime_artifact_path(runtime: RuntimeBackend) -> Path | None:
    """Recover the built-in runtime's artifact root without extending the protocol."""

    candidate: object | None = runtime
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        raw_path = getattr(candidate, "_artifact_path", None)
        if isinstance(raw_path, (str, os.PathLike)):
            return Path(os.fspath(raw_path)).expanduser().resolve(strict=False)
        candidate = getattr(candidate, "_backend", None)
    return None


def _warn_non_hot_lc_profile_once(
    runtime: RuntimeBackend,
    message: str,
) -> None:
    """Warn once for one loaded process, never once per timing sample."""

    owner: object = runtime
    seen: set[int] = set()
    while id(owner) not in seen:
        seen.add(id(owner))
        backend = getattr(owner, "_backend", None)
        if backend is None:
            break
        owner = backend
    with _LC_PROFILE_WARNING_LOCK:
        if bool(getattr(owner, _LC_PROFILE_WARNING_ATTRIBUTE, False)):
            return
        try:
            setattr(owner, _LC_PROFILE_WARNING_ATTRIBUTE, True)
        except (AttributeError, TypeError):
            # Keep an identity-stable fallback for immutable protocol
            # implementations so they cannot emit once per benchmark run.
            owner_id = id(owner)
            if _LC_PROFILE_WARNING_FALLBACK_OWNERS.get(owner_id) is owner:
                return
            _LC_PROFILE_WARNING_FALLBACK_OWNERS[owner_id] = owner
    warnings.warn(message, UserWarning, stacklevel=3)


def _default_lc_profile_selectors(
    *,
    physics: ProcessPhysics,
    lc_flow_layout: str | None,
    selected_helicity_ids: Sequence[str],
    selected_color_ids: Sequence[str],
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Choose the LC hot workload while preserving explicit subset intent."""

    helicities = tuple(selected_helicity_ids)
    color_flows = tuple(selected_color_ids)
    if physics.color_accuracy != "lc" or lc_flow_layout is None:
        return helicities or None, color_flows or None
    complete_helicities = _is_complete_selector_set(
        helicities, physics.helicity_ids
    )
    complete_color_flows = _is_complete_selector_set(
        color_flows, physics.color_ids
    )
    if lc_flow_layout == _LC_TOPOLOGY_REPLAY_LAYOUT:
        normalized_helicities = None if complete_helicities else helicities or None
        normalized_color_flows = (
            color_flows
            if color_flows and (not complete_color_flows or len(color_flows) == 1)
            else None
        )
        if color_flows or normalized_helicities is not None:
            return normalized_helicities, normalized_color_flows
        selectable_flows = tuple(
            flow.id for flow in physics.color_flows if flow.computed
        )
        if not selectable_flows:
            selectable_flows = physics.color_ids
        if not selectable_flows:
            raise EvaluationError("LC profiling requires at least one physical flow")
        return None, (selectable_flows[0],)
    if lc_flow_layout == _LC_ALL_FLOW_UNION_LAYOUT:
        normalized_color_flows = None if complete_color_flows else color_flows or None
        normalized_helicities = (
            helicities
            if helicities and (not complete_helicities or len(helicities) == 1)
            else None
        )
        if helicities or normalized_color_flows is not None:
            return normalized_helicities, normalized_color_flows
        selectable = tuple(
            helicity.id
            for helicity in physics.helicities
            if helicity.computed and not helicity.structural_zero
        )
        if not selectable:
            selectable = tuple(
                helicity.id for helicity in physics.helicities if helicity.computed
            )
        if not selectable:
            selectable = physics.helicity_ids
        if not selectable:
            raise EvaluationError(
                "LC profiling requires at least one physical helicity"
            )
        return (selectable[0],), None
    return None, None


def _is_complete_selector_set(
    selected: Sequence[str],
    available: Sequence[str],
) -> bool:
    """Return whether an explicit selector enumerates one complete physical axis."""

    return (
        bool(selected)
        and len(selected) == len(available)
        and set(selected) == set(available)
    )


def _lc_flow_layout_recommendation(
    *,
    color_accuracy: str,
    lc_flow_layout: str | None,
    selected_helicity_ids: Sequence[str],
    selected_color_ids: Sequence[str],
) -> str | None:
    if color_accuracy != "lc":
        return None
    if lc_flow_layout == _LC_TOPOLOGY_REPLAY_LAYOUT:
        if not selected_helicity_ids and len(selected_color_ids) == 1:
            return None
        return _LC_TOPOLOGY_REPLAY_PROFILE_RECOMMENDATION
    if lc_flow_layout == _LC_ALL_FLOW_UNION_LAYOUT:
        if len(selected_helicity_ids) == 1 and not selected_color_ids:
            return None
        return _LC_ALL_FLOW_UNION_PROFILE_RECOMMENDATION
    return None


def _benchmark_batch(points: Momenta, batch_size: int) -> Momenta:
    source = tuple(points)
    return tuple(source[index % len(source)] for index in range(batch_size))


def _on_the_fly_benchmark_context(
    runtime: RuntimeBackend,
    requested_color_ids: Sequence[str],
) -> _CompactBenchmarkContext | None:
    if str(getattr(runtime, "execution_mode", "compiled")) != "on-the-fly":
        return None
    loader = getattr(runtime, "_on_the_fly_benchmark_context", None)
    if not callable(loader):
        raise EvaluationError(
            "on-the-fly benchmark requires the installed compact runtime context"
        )
    raw = loader(tuple(requested_color_ids))
    if not isinstance(raw, Mapping):
        raise EvaluationError("on-the-fly benchmark context is unavailable")

    def required_text(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise EvaluationError(
                f"on-the-fly benchmark context field {key!r} is invalid"
            )
        return value

    def required_count(key: str) -> int:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EvaluationError(
                f"on-the-fly benchmark context field {key!r} is invalid"
            )
        return value

    selected = raw.get("selected_color_ids")
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise EvaluationError("on-the-fly benchmark selected color IDs are invalid")
    selected_color_ids = tuple(str(value) for value in selected)
    if len(selected_color_ids) != len(requested_color_ids) or any(
        not value for value in selected_color_ids
    ):
        raise EvaluationError(
            "on-the-fly benchmark selected color IDs do not match the request"
        )
    return _CompactBenchmarkContext(
        process_id=required_text("process_id"),
        process_expression=required_text("process_expression"),
        color_accuracy=required_text("color_accuracy"),
        helicity_count=required_count("helicity_count"),
        color_count=required_count("color_count"),
        selected_color_ids=selected_color_ids,
    )


def _on_the_fly_runtime_state_census(
    runtime: RuntimeBackend,
    *,
    expected_process_id: str,
) -> dict[str, object]:
    loader = getattr(runtime, "_on_the_fly_runtime_state_census", None)
    if not callable(loader):
        raise EvaluationError(
            "on-the-fly benchmark requires the installed compact runtime state census"
        )
    raw = loader()
    if not isinstance(raw, Mapping):
        raise EvaluationError("on-the-fly runtime state census is unavailable")
    value = deepcopy(dict(raw))
    if value.get("kind") != _OTF_RUNTIME_STATE_CENSUS_KIND:
        raise EvaluationError("on-the-fly runtime state census has an invalid kind")
    if value.get("process_id") != expected_process_id:
        raise EvaluationError(
            "on-the-fly runtime state census does not match the benchmark process"
        )
    if value.get("family_cache_policy") != _OTF_FAMILY_CACHE_POLICY:
        raise EvaluationError(
            "on-the-fly runtime state census has an invalid family cache policy"
        )
    cache_limit = value.get("family_cache_limit")
    if (
        isinstance(cache_limit, bool)
        or not isinstance(cache_limit, int)
        or cache_limit != _OTF_FAMILY_CACHE_LIMIT
    ):
        raise EvaluationError(
            "on-the-fly runtime state census has an invalid family cache limit"
        )

    counts: dict[str, int] = {}
    for field in _OTF_RUNTIME_STATE_COUNT_FIELDS:
        count = value.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EvaluationError(
                f"on-the-fly runtime state census field {field!r} is invalid"
            )
        counts[field] = count
    if counts["retained_family_count"] > _OTF_FAMILY_CACHE_LIMIT:
        raise EvaluationError(
            "on-the-fly runtime state census retained family count exceeds its "
            "cache limit"
        )

    active = value.get("active_family_union_census")
    if active is None:
        return value
    if not isinstance(active, Mapping):
        raise EvaluationError(
            "on-the-fly active-family runtime state census is invalid"
        )
    active_value = dict(active)
    if (
        active_value.get("basis") != "shared-query-family-union-v1"
        or active_value.get("scope") != "active-family-union"
    ):
        raise EvaluationError(
            "on-the-fly active-family runtime state census has an invalid identity"
        )
    active_counts: dict[str, int] = {}
    for field in _OTF_ACTIVE_FAMILY_COUNT_FIELDS:
        count = active_value.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EvaluationError(
                "on-the-fly active-family runtime state census field "
                f"{field!r} is invalid"
            )
        active_counts[field] = count
    if (
        active_counts["query_count"] < 1
        or active_counts["union_unique_current_count"]
        > active_counts["union_unique_current_component_count"]
        or active_counts["union_amplitude_destination_count"] < 1
        or active_counts["union_amplitude_destination_count"]
        > active_counts["query_count"]
        or active_counts["query_count"] > counts["retained_request_count"]
        or active_counts["union_amplitude_destination_count"]
        > counts["retained_amplitude_destination_count"]
    ):
        raise EvaluationError(
            "on-the-fly active-family runtime state census is inconsistent"
        )
    for role in _OTF_OPERATION_ROLES:
        if (
            active_counts[f"union_{role}_executor_call_groups"]
            > active_counts[f"union_{role}_rows"]
        ):
            raise EvaluationError(
                "on-the-fly active-family runtime state census has more "
                f"{role} executor call groups than rows"
            )
    value["active_family_union_census"] = active_value
    return value


def _on_the_fly_runtime_state_is_cold(census: Mapping[str, object]) -> bool:
    return all(census[field] == 0 for field in _OTF_RUNTIME_STATE_COUNT_FIELDS) and (
        census["active_family_union_census"] is None
    )


def _on_the_fly_runtime_state_is_retained(census: Mapping[str, object]) -> bool:
    if (
        census["pending_family_count"] != 0
        or census["process_preparation_count"] != 1
        or census["retained_family_count"] != _OTF_FAMILY_CACHE_LIMIT
        or census["retained_selection_count"] != _OTF_FAMILY_CACHE_LIMIT
        or cast(int, census["retained_request_count"]) < 1
    ):
        return False
    executable_counts = tuple(
        cast(int, census[field])
        for field in _OTF_RETAINED_STATE_EXECUTABLE_COUNT_FIELDS
    )
    active = census["active_family_union_census"]
    if isinstance(active, Mapping):
        return (
            executable_counts[0] > 0
            and executable_counts[1] == _OTF_FAMILY_CACHE_LIMIT
            and executable_counts[2] > 0
        )
    return active is None and all(count == 0 for count in executable_counts)


def _resolve_color_flow_ordinals(
    physics: ProcessPhysics,
    requested: Sequence[str],
) -> tuple[str, ...]:
    available = physics.color_ids
    resolved: list[str] = []
    for value in requested:
        if value in available:
            resolved.append(value)
            continue
        try:
            ordinal = int(value, 10)
        except ValueError:
            resolved.append(value)
            continue
        if str(ordinal) != value.strip() or ordinal < 1 or ordinal > len(available):
            maximum = len(available)
            raise EvaluationError(
                f"color-flow ordinal {value!r} is out of range; choose 1..{maximum} "
                "or a stable color component ID"
            )
        resolved.append(available[ordinal - 1])
    return tuple(resolved)


def _color_workload_text(
    physics: ProcessPhysics,
    selected: Sequence[str],
) -> str:
    if physics.color_accuracy != "lc":
        return f"contracted {physics.color_accuracy.upper()} color total"
    count = len(physics.color_ids)
    if not selected:
        return f"all {count} generated physical LC flows"
    return f"selected {len(selected)}/{count} physical LC flows: {', '.join(selected)}"


def _compact_color_workload_text(
    context: _CompactBenchmarkContext,
    selected: Sequence[str],
) -> str:
    if context.color_accuracy != "lc":
        return f"contracted {context.color_accuracy.upper()} color total"
    if not selected:
        return f"all {context.color_count} generated physical LC flows"
    return (
        f"selected {len(selected)}/{context.color_count} physical LC flows: "
        f"{', '.join(selected)}"
    )


def _helicity_workload_text(
    physics: ProcessPhysics,
    selected: Sequence[str],
) -> str:
    count = len(physics.helicity_ids)
    if not selected:
        structural = physics.structural_zero_helicity_count
        suffix = f"; {structural} structural zeros" if structural else ""
        return f"all {count} generated helicity configurations{suffix}"
    return f"selected {len(selected)}/{count} helicities: {', '.join(selected)}"


def _compact_helicity_workload_text(
    context: _CompactBenchmarkContext,
    selected: Sequence[str],
) -> str:
    if not selected:
        return f"all {context.helicity_count} generated helicity configurations"
    return (
        f"selected {len(selected)}/{context.helicity_count} helicities: "
        f"{', '.join(selected)}"
    )


def _sample_progress_payload(
    samples: list[float],
    *,
    elapsed_seconds: float,
    target_seconds: float,
    repetitions: int,
    batch_size: int,
) -> tuple[str, dict[str, str | int | float | None]]:
    mean, _deviation, error, relative_error = _sample_statistics(samples)
    if mean < 1.0e-6:
        scale, unit = 1.0e9, "ns"
    elif mean < 1.0e-3:
        scale, unit = 1.0e6, "us"
    elif mean < 1.0:
        scale, unit = 1.0e3, "ms"
    else:
        scale, unit = 1.0, "s"
    uncertainty = (
        "SE pending"
        if len(samples) < 2
        else (
            f"+/- {error * scale:.2g} {unit} "
            f"(relative standard error {relative_error:.2%})"
        )
    )
    message = (
        f"{elapsed_seconds:.3g}/{target_seconds:.3g}s; "
        f"wall {mean * scale:.5g} {unit}/point {uncertainty}; "
        f"{repetitions} calls x {batch_size} points"
    )
    return message, {
        "progress_kind": "benchmark-statistics",
        "elapsed_seconds": elapsed_seconds,
        "target_seconds": target_seconds,
        "wall_seconds_per_point": mean,
        "wall_standard_error_seconds_per_point": (error if len(samples) >= 2 else None),
        "wall_relative_standard_error": (relative_error if len(samples) >= 2 else None),
        "repetitions": repetitions,
        "batch_size": batch_size,
        "sample_count": len(samples),
    }


def _planned_sample_count(
    config: BenchmarkConfig,
    *,
    probe_seconds: float,
) -> int:
    if config.minimum_samples == 1:
        return 1
    runtime_samples = math.ceil(config.target_runtime / _MAX_SAMPLE_RUNTIME_SECONDS)
    desired = max(config.minimum_samples, runtime_samples)
    maximum_for_runtime = max(
        math.floor(
            config.target_runtime / max(probe_seconds, _MIN_CLOCK_INTERVAL_SECONDS)
        ),
        1,
    )
    return max(config.minimum_samples, min(desired, maximum_for_runtime))


def _timed_repetitions(callback: Callable[[], object], repetitions: int) -> float:
    started = time.perf_counter()
    for _ in range(repetitions):
        callback()
    return time.perf_counter() - started


def _estimated_repetitions(
    current: int,
    observed_seconds: float,
    target_seconds: float,
) -> int:
    observed = max(observed_seconds, _MIN_CLOCK_INTERVAL_SECONDS)
    estimate = math.ceil(current * target_seconds / observed)
    return min(max(estimate, 1), _MAX_REPETITIONS_PER_SAMPLE)


def _calibrate_repetitions(
    callback: Callable[[], object],
    config: BenchmarkConfig,
    *,
    initial_seconds: float | None,
    timer: Callable[[int], float] | None = None,
) -> _Calibration:
    measure = timer or (lambda repetitions: _timed_repetitions(callback, repetitions))
    block_count = 0
    evaluation_count = 0
    calibration_elapsed = 0.0
    if initial_seconds is None:
        initial_seconds = measure(1)
        block_count = 1
        evaluation_count = 1
        calibration_elapsed = initial_seconds

    probe_seconds = initial_seconds
    sample_count = _planned_sample_count(config, probe_seconds=probe_seconds)
    target_sample_seconds = config.target_runtime / sample_count
    if sample_count == 1:
        return _Calibration(
            sample_count=sample_count,
            repetitions_per_sample=_estimated_repetitions(
                1,
                probe_seconds,
                target_sample_seconds,
            ),
            target_sample_seconds=target_sample_seconds,
            probe_seconds=probe_seconds,
            block_count=block_count,
            evaluation_count=evaluation_count,
            elapsed_seconds=calibration_elapsed,
        )

    observed_seconds = initial_seconds
    repetitions = 1
    for _ in range(_MAX_CALIBRATION_BLOCKS):
        candidate = _estimated_repetitions(
            repetitions,
            observed_seconds,
            target_sample_seconds,
        )
        if candidate == repetitions:
            break
        observed_seconds = measure(candidate)
        block_count += 1
        evaluation_count += candidate
        calibration_elapsed += observed_seconds
        repetitions = candidate
        ratio = observed_seconds / target_sample_seconds
        if _CALIBRATION_LOWER_RATIO <= ratio <= _CALIBRATION_UPPER_RATIO:
            break
    else:
        repetitions = _estimated_repetitions(
            repetitions,
            observed_seconds,
            target_sample_seconds,
        )

    return _Calibration(
        sample_count=sample_count,
        repetitions_per_sample=repetitions,
        target_sample_seconds=target_sample_seconds,
        probe_seconds=probe_seconds,
        block_count=block_count,
        evaluation_count=evaluation_count,
        elapsed_seconds=calibration_elapsed,
    )


def _native_profiler(
    runtime: RuntimeBackend,
) -> Callable[..., Mapping[str, object]] | None:
    if getattr(runtime, "supports_profiling", None) is False:
        return None
    for name in ("profile", "evaluate_profile"):
        profiler = getattr(runtime, name, None)
        if callable(profiler):
            return cast(Callable[..., Mapping[str, object]], profiler)
    return None


def _native_repeated_profiler(
    runtime: RuntimeBackend,
) -> Callable[..., Mapping[str, object]] | None:
    if getattr(runtime, "supports_profiling", None) is False:
        return None
    profiler = getattr(runtime, "profile_repeated", None)
    return (
        cast(Callable[..., Mapping[str, object]], profiler)
        if callable(profiler)
        else None
    )


def _native_wall_timer(runtime: RuntimeBackend) -> Callable[..., float] | None:
    timer = getattr(runtime, "_benchmark_f64_wall_time", None)
    return timer if callable(timer) else None


def _profile_float(profile: Mapping[str, object], key: str) -> float:
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise EvaluationError(f"native runtime profile field {key!r} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise EvaluationError(f"native runtime profile field {key!r} is invalid")
    return value


def _profile_float_or_none(profile: Mapping[str, object], key: str) -> float | None:
    value = profile.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def _profile_count_or_none(
    profile: Mapping[str, object],
    key: str,
    *,
    denominator: int,
) -> float | None:
    if key not in profile:
        return None
    value = profile[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(
            f"native runtime profile counter {key!r} is not a non-negative integer"
        )
    return value / denominator


def _native_profile_counters(
    profile: Mapping[str, object],
    *,
    points: int,
    repetitions: int,
) -> _NativeProfileCounterSample | None:
    arena_counter_presence = tuple(
        key in profile for key in _COMPILED_DIRECT_ARENA_COUNTER_KEYS
    )
    if any(arena_counter_presence) and not all(arena_counter_presence):
        missing = tuple(
            key
            for key, present in zip(
                _COMPILED_DIRECT_ARENA_COUNTER_KEYS,
                arena_counter_presence,
                strict=True,
            )
            if not present
        )
        raise EvaluationError(
            "native compiled Direct-Arena profile counters must be jointly "
            f"available; missing {missing!r}"
        )

    def per_point(key: str) -> float | None:
        return _profile_count_or_none(profile, key, denominator=points)

    def per_call(key: str) -> float | None:
        return _profile_count_or_none(profile, key, denominator=repetitions)

    counters = _NativeProfileCounterSample(
        native_input_components_per_point=per_point("native_input_component_count"),
        native_input_pack_bytes_per_point=per_point("native_input_pack_bytes"),
        native_input_crossing_bytes_per_point=per_point("native_input_crossing_bytes"),
        state_components_per_point=per_point("state_component_count"),
        state_clear_components_per_point=per_point("state_clear_component_count"),
        source_components_per_point=per_point("source_component_count"),
        momentum_components_per_point=per_point("momentum_component_count"),
        model_parameter_components_per_point=per_point(
            "model_parameter_component_count"
        ),
        stage_input_copy_components_per_point=per_point(
            "stage_input_copy_component_count"
        ),
        stage_leaf_input_copy_components_per_point=per_point(
            "stage_leaf_input_copy_component_count"
        ),
        stage_evaluator_output_gather_components_per_point=per_point(
            "stage_evaluator_output_gather_component_count"
        ),
        stage_output_assign_components_per_point=per_point(
            "stage_output_assign_component_count"
        ),
        amplitude_input_copy_components_per_point=per_point(
            "amplitude_input_copy_component_count"
        ),
        amplitude_leaf_input_copy_components_per_point=per_point(
            "amplitude_leaf_input_copy_component_count"
        ),
        amplitude_evaluator_output_gather_components_per_point=per_point(
            "amplitude_evaluator_output_gather_component_count"
        ),
        amplitude_output_remap_components_per_point=per_point(
            "amplitude_output_remap_component_count"
        ),
        reduction_input_components_per_point=per_point(
            "reduction_input_component_count"
        ),
        selector_gather_points_per_point=per_point("selector_gather_point_count"),
        selector_gather_bytes_per_point=per_point("selector_gather_bytes"),
        selector_scatter_values_per_point=per_point("selector_scatter_value_count"),
        resolved_materialized_components_per_point=per_point(
            "resolved_materialized_component_count"
        ),
        total_materialized_values_per_point=per_point("total_materialized_value_count"),
        final_output_copy_values_per_point=per_point("final_output_copy_value_count"),
        native_input_container_allocations_per_call=per_call(
            "native_input_container_allocation_count"
        ),
        evaluator_backend_calls_per_call=per_call("evaluator_backend_call_count"),
        compiled_direct_arena_engines_per_call=per_call(
            "compiled_direct_arena_engine_count"
        ),
        compiled_direct_arena_calls_per_call=per_call(
            "compiled_direct_arena_call_count"
        ),
        compiled_direct_arena_boundary_input_bytes_per_call=per_call(
            "compiled_direct_arena_boundary_input_bytes"
        ),
        compiled_direct_arena_boundary_current_output_bytes_per_call=per_call(
            "compiled_direct_arena_boundary_current_output_bytes"
        ),
        compiled_direct_arena_boundary_amplitude_output_bytes_per_call=per_call(
            "compiled_direct_arena_boundary_amplitude_output_bytes"
        ),
        recurrence_momentum_scalar_values_per_point=per_point(
            "recurrence_momentum_scalar_value_count"
        ),
        recurrence_schedule_executions_per_call=per_call(
            "recurrence_schedule_execution_count"
        ),
        recurrence_replay_schedule_executions_per_call=per_call(
            "recurrence_replay_schedule_execution_count"
        ),
        recurrence_union_schedule_executions_per_call=per_call(
            "recurrence_union_schedule_execution_count"
        ),
        recurrence_union_source_rows_per_call=per_call(
            "recurrence_union_source_row_count"
        ),
        recurrence_replay_output_values_per_point=per_point(
            "recurrence_replay_output_value_count"
        ),
        recurrence_source_calls_per_call=per_call("recurrence_source_call_count"),
        recurrence_source_rows_per_call=per_call("recurrence_source_row_count"),
        recurrence_contribution_calls_per_call=per_call(
            "recurrence_contribution_call_count"
        ),
        recurrence_contribution_rows_per_call=per_call(
            "recurrence_contribution_row_count"
        ),
        recurrence_finalization_calls_per_call=per_call(
            "recurrence_finalization_call_count"
        ),
        recurrence_finalization_rows_per_call=per_call(
            "recurrence_finalization_row_count"
        ),
        recurrence_closure_calls_per_call=per_call("recurrence_closure_call_count"),
        recurrence_closure_rows_per_call=per_call("recurrence_closure_row_count"),
        observed_scratch_reallocations_per_call=per_call(
            "observed_scratch_reallocation_count"
        ),
        native_output_allocations_per_call=per_call("native_output_allocation_count"),
    )
    if not any(getattr(counters, field.name) is not None for field in fields(counters)):
        return None
    arena_engines = counters.compiled_direct_arena_engines_per_call
    if arena_engines is not None:
        arena_calls = counters.compiled_direct_arena_calls_per_call
        assert arena_calls is not None
        boundary_counters = (
            (
                "compiled_direct_arena_boundary_input_bytes",
                counters.compiled_direct_arena_boundary_input_bytes_per_call,
            ),
            (
                "compiled_direct_arena_boundary_current_output_bytes",
                counters.compiled_direct_arena_boundary_current_output_bytes_per_call,
            ),
            (
                "compiled_direct_arena_boundary_amplitude_output_bytes",
                counters.compiled_direct_arena_boundary_amplitude_output_bytes_per_call,
            ),
        )
        for name, value in boundary_counters:
            assert value is not None
            if value != 0.0:
                raise EvaluationError(
                    "native compiled Direct-Arena profile observed forbidden "
                    f"boundary traffic: {name}={value:g} bytes/runtime call"
                )
        if (arena_engines > 0.0) != (arena_calls > 0.0):
            raise EvaluationError(
                "native compiled Direct-Arena engine and call counters disagree"
            )
        if arena_engines > 0.0:
            evaluator_calls = counters.evaluator_backend_calls_per_call
            if evaluator_calls is None or evaluator_calls != arena_calls:
                raise EvaluationError(
                    "native compiled Direct-Arena calls do not cover every "
                    "evaluator backend call"
                )
    return counters


def _profile_float_sequence(
    profile: Mapping[str, object], key: str
) -> tuple[float, ...]:
    values = profile.get(key)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EvaluationError(
            f"native runtime profile field {key!r} is not a numeric sequence"
        )
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise EvaluationError(
                f"native runtime profile field {key!r} is not a numeric sequence"
            )
        entry = float(value)
        if not math.isfinite(entry) or entry < 0.0:
            raise EvaluationError(f"native runtime profile field {key!r} is invalid")
        result.append(entry)
    return tuple(result)


def _profile_float_sequence_or_none(
    profile: Mapping[str, object], key: str
) -> tuple[float, ...] | None:
    if key not in profile:
        return None
    values = _profile_float_sequence(profile, key)
    # Eager execution has no compiled-stage timing vector. Rusticol still
    # supplies the aggregate evaluator timer, so an empty vector means
    # "unavailable", not a measured zero.
    return values or None


def _profile_execution_mode(
    profile: Mapping[str, object],
    *,
    stage_vectors_present_but_empty: bool,
) -> str | None:
    value = profile.get("execution_mode")
    if value is not None:
        if value not in {"compiled", "eager", "recurrence", "on-the-fly"}:
            raise EvaluationError(
                "native runtime profile execution_mode must be compiled, eager, "
                "recurrence, or on-the-fly"
            )
        return str(value)
    stage_aggregate = profile.get("stage_evaluator_call_time_s")
    amplitude_aggregate = profile.get("amplitude_evaluator_call_time_s")
    if (
        stage_vectors_present_but_empty
        and isinstance(stage_aggregate, (float, int))
        and not isinstance(stage_aggregate, bool)
        and float(stage_aggregate) > 0.0
        and isinstance(amplitude_aggregate, (float, int))
        and not isinstance(amplitude_aggregate, bool)
        and float(amplitude_aggregate) == 0.0
    ):
        return "eager"
    return None


def _profile_point_count(profile: Mapping[str, object], fallback: int) -> int:
    value: Any = profile.get("points", fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationError("native runtime profile point count is invalid")
    return int(value)


def _native_profile_sample(
    profile: Mapping[str, object],
    fallback_points: int,
    *,
    repetitions: int = 1,
) -> _NativeProfileSample:
    points = _profile_point_count(profile, fallback_points)
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise EvaluationError("native runtime profile repetition count is invalid")
    if points % repetitions != 0:
        raise EvaluationError(
            "native runtime profile point count is not divisible by repetitions"
        )
    per_point = 1.0 / points
    counters = _native_profile_counters(
        profile,
        points=points,
        repetitions=repetitions,
    )
    stage_vector_keys = (
        "stage_input_pack_by_stage_time_s",
        "stage_evaluator_call_by_stage_time_s",
        "stage_output_assign_by_stage_time_s",
    )
    stage_vectors_present_but_empty = all(
        key in profile
        and isinstance(profile[key], Sequence)
        and not isinstance(profile[key], (str, bytes))
        and len(profile[key]) == 0  # type: ignore[arg-type]
        for key in stage_vector_keys
    )
    stage_input_pack = _profile_float_sequence_or_none(
        profile, "stage_input_pack_by_stage_time_s"
    )
    stage_leaf_input_pack = _profile_float_sequence_or_none(
        profile, "stage_leaf_input_pack_by_stage_time_s"
    )
    stage_evaluator_call = _profile_float_sequence_or_none(
        profile, "stage_evaluator_call_by_stage_time_s"
    )
    stage_backend_call = _profile_float_sequence_or_none(
        profile, "stage_backend_call_by_stage_time_s"
    )
    stage_evaluator_output_gather = _profile_float_sequence_or_none(
        profile, "stage_evaluator_output_gather_by_stage_time_s"
    )
    stage_output_assign = _profile_float_sequence_or_none(
        profile, "stage_output_assign_by_stage_time_s"
    )
    execution_mode = _profile_execution_mode(
        profile,
        stage_vectors_present_but_empty=stage_vectors_present_but_empty,
    )
    compiled_direct_arena_active = bool(
        counters is not None
        and counters.compiled_direct_arena_engines_per_call is not None
        and counters.compiled_direct_arena_engines_per_call > 0.0
    )
    if compiled_direct_arena_active and execution_mode != "compiled":
        raise EvaluationError(
            "native compiled Direct-Arena counters require compiled execution mode"
        )

    stage_input_pack_total = (
        sum(stage_input_pack)
        if stage_input_pack is not None
        else _profile_float_or_none(profile, "stage_input_pack_time_s")
    )
    stage_evaluator_call_total = (
        sum(stage_evaluator_call)
        if stage_evaluator_call is not None
        else _profile_float_or_none(profile, "stage_evaluator_call_time_s")
    )
    if (
        stage_evaluator_call_total is None
        and execution_mode not in _RECURRENCE_LIKE_EXECUTION_MODES
    ):
        raise EvaluationError("native runtime stage evaluator timing is unavailable")
    if stage_evaluator_call_total is None:
        stage_evaluator_call_total = 0.0
    output_assign_total = (
        sum(stage_output_assign)
        if stage_output_assign is not None
        else _profile_float_or_none(profile, "output_assign_time_s")
    )
    stage_leaf_input_pack_total = (
        sum(stage_leaf_input_pack)
        if stage_leaf_input_pack is not None
        else _profile_float_or_none(profile, "stage_leaf_input_pack_time_s")
    )
    stage_backend_call_total = (
        sum(stage_backend_call)
        if stage_backend_call is not None
        else _profile_float_or_none(profile, "stage_backend_call_time_s")
    )
    stage_evaluator_output_gather_total = (
        sum(stage_evaluator_output_gather)
        if stage_evaluator_output_gather is not None
        else _profile_float_or_none(profile, "stage_evaluator_output_gather_time_s")
    )
    amplitude_evaluator_call: float | None = None
    if execution_mode not in {"eager", *_RECURRENCE_LIKE_EXECUTION_MODES}:
        amplitude_evaluator_call = _profile_float_or_none(
            profile, "amplitude_evaluator_call_time_s"
        )
        if amplitude_evaluator_call is None:
            amplitude_evaluator_call = _profile_float(
                profile, "amplitude_evaluator_time_s"
            )

    def normalized(key: str) -> float | None:
        value = _profile_float_or_none(profile, key)
        return None if value is None else value * per_point

    def normalized_sequence(
        values: tuple[float, ...] | None,
    ) -> tuple[float, ...] | None:
        if values is None:
            return None
        return tuple(value * per_point for value in values)

    wall_time = normalized("wall_time_s")
    native_input_pack_time = normalized("native_input_pack_time_s")
    native_input_crossing_time = normalized("native_input_crossing_time_s")
    orchestration_time = normalized("orchestration_time_s")
    if compiled_direct_arena_active and orchestration_time is None:
        raise EvaluationError(
            "native compiled Direct-Arena orchestration timing is unavailable"
        )
    state_prepare_time = normalized("state_prepare_time_s")
    state_clear_time = normalized("state_clear_time_s")
    source_fill_time = normalized("source_fill_time_s")
    momentum_setup_time = normalized("momentum_setup_time_s")
    momentum_input_setup_time = normalized("momentum_input_setup_time_s")
    model_parameter_setup_time = normalized("model_parameter_setup_time_s")
    if momentum_input_setup_time is None:
        momentum_input_setup_time = momentum_setup_time
        if (
            momentum_input_setup_time is not None
            and model_parameter_setup_time is not None
        ):
            momentum_input_setup_time = max(
                momentum_input_setup_time - model_parameter_setup_time,
                0.0,
            )
    compiled_profile = execution_mode not in {
        "eager",
        *_RECURRENCE_LIKE_EXECUTION_MODES,
    }
    stage_input_pack_time = (
        None
        if not compiled_profile or stage_input_pack_total is None
        else stage_input_pack_total * per_point
    )
    stage_leaf_input_pack_time = (
        None
        if not compiled_profile or stage_leaf_input_pack_total is None
        else stage_leaf_input_pack_total * per_point
    )
    stage_evaluator_call_time = stage_evaluator_call_total * per_point
    stage_backend_call_time = (
        None
        if not compiled_profile or stage_backend_call_total is None
        else stage_backend_call_total * per_point
    )
    stage_evaluator_output_gather_time = (
        None
        if not compiled_profile or stage_evaluator_output_gather_total is None
        else stage_evaluator_output_gather_total * per_point
    )
    output_assign_time = (
        None
        if not compiled_profile or output_assign_total is None
        else output_assign_total * per_point
    )
    amplitude_input_pack_time = (
        normalized("amplitude_input_pack_time_s") if compiled_profile else None
    )
    amplitude_evaluator_call_time = (
        None
        if amplitude_evaluator_call is None
        else amplitude_evaluator_call * per_point
    )
    amplitude_leaf_input_pack_time = (
        normalized("amplitude_leaf_input_pack_time_s") if compiled_profile else None
    )
    amplitude_backend_call_time = (
        normalized("amplitude_backend_call_time_s") if compiled_profile else None
    )
    amplitude_evaluator_output_gather_time = (
        normalized("amplitude_evaluator_output_gather_time_s")
        if compiled_profile
        else None
    )
    amplitude_output_remap_time = (
        normalized("amplitude_output_remap_time_s") if compiled_profile else None
    )
    reduction_time = (
        normalized("eager_reduction_time_s")
        if execution_mode == "eager"
        else normalized("reduction_time_s")
    )
    total_materialization_time = normalized("total_materialization_time_s")
    final_output_copy_time = normalized("final_output_copy_time_s")
    selector_planner_time = normalized("selector_planner_time_s")
    selector_gather_time = normalized("selector_gather_time_s")
    selector_scatter_time = normalized("selector_scatter_time_s")
    eager_execution_time = (
        stage_evaluator_call_time if execution_mode == "eager" else None
    )
    eager_initialize_time = normalized("eager_initialize_time_s")
    eager_gather_time = normalized("eager_gather_time_s")
    eager_kernel_call_time = normalized("eager_kernel_call_time_s")
    eager_invocation_scatter_time = normalized("eager_invocation_scatter_time_s")
    eager_finalization_time = normalized("eager_finalization_time_s")
    eager_scatter_finalization_time = normalized("eager_scatter_finalization_time_s")
    eager_closure_time = normalized("eager_closure_time_s")
    eager_copy_out_time = normalized("eager_copy_out_time_s")
    recurrence_momentum_fill_time = normalized("recurrence_momentum_fill_time_s")
    recurrence_union_source_fill_time = normalized(
        "recurrence_union_source_fill_time_s"
    )
    recurrence_schedule_time = normalized("recurrence_schedule_time_s")
    recurrence_source_kernel_time = normalized("recurrence_source_kernel_time_s")
    recurrence_contribution_kernel_time = normalized(
        "recurrence_contribution_kernel_time_s"
    )
    recurrence_finalization_time = normalized("recurrence_finalization_time_s")
    recurrence_closure_time = normalized("recurrence_closure_time_s")
    recurrence_replay_output_mapping_time = normalized(
        "recurrence_replay_output_mapping_time_s"
    )
    if (
        execution_mode in _RECURRENCE_LIKE_EXECUTION_MODES
        and recurrence_schedule_time is None
    ):
        raise EvaluationError("native recurrence-like schedule timing is unavailable")

    common_accounted = (
        native_input_pack_time,
        native_input_crossing_time,
        orchestration_time,
        state_prepare_time,
        state_clear_time,
        source_fill_time,
        momentum_input_setup_time,
        model_parameter_setup_time,
    )
    if execution_mode == "eager":
        mode_accounted = (eager_execution_time,)
    elif execution_mode in _RECURRENCE_LIKE_EXECUTION_MODES:
        mode_accounted = (
            recurrence_momentum_fill_time,
            recurrence_union_source_fill_time,
            recurrence_schedule_time,
            recurrence_replay_output_mapping_time,
            reduction_time,
        )
    else:
        mode_accounted = (
            stage_input_pack_time,
            stage_evaluator_call_time,
            output_assign_time,
            amplitude_input_pack_time,
            amplitude_evaluator_call_time,
            reduction_time,
        )
    accounted = (
        *common_accounted,
        *mode_accounted,
        total_materialization_time,
        final_output_copy_time,
        selector_planner_time,
        selector_gather_time,
        selector_scatter_time,
    )
    accounted_total = sum(value or 0.0 for value in accounted)
    accounting_tolerance = (
        1.0e-12 if wall_time is None else max(1.0e-12, wall_time * 1.0e-12)
    )
    if wall_time is not None and accounted_total > wall_time + accounting_tolerance:
        raise EvaluationError(
            "native profile exclusive top-level phases account for "
            f"{accounted_total:.9e}s/point, exceeding wall time "
            f"{wall_time:.9e}s/point"
        )
    if execution_mode == "eager" and eager_execution_time is not None:
        scatter_phases = (
            (eager_invocation_scatter_time, eager_finalization_time)
            if eager_invocation_scatter_time is not None
            or eager_finalization_time is not None
            else (eager_scatter_finalization_time,)
        )
        exclusive_eager_phases = (
            eager_initialize_time,
            eager_gather_time,
            eager_kernel_call_time,
            *scatter_phases,
            eager_closure_time,
            reduction_time,
            eager_copy_out_time,
        )
        exclusive_eager_total = sum(value or 0.0 for value in exclusive_eager_phases)
        if exclusive_eager_total > eager_execution_time + accounting_tolerance:
            raise EvaluationError(
                "native eager profile exclusive execution phases account for "
                f"{exclusive_eager_total:.9e}s/point, exceeding the inclusive "
                f"eager execution time {eager_execution_time:.9e}s/point"
            )
    if (
        execution_mode in _RECURRENCE_LIKE_EXECUTION_MODES
        and recurrence_schedule_time is not None
    ):
        recurrence_attribution = (
            recurrence_source_kernel_time,
            recurrence_contribution_kernel_time,
            recurrence_finalization_time,
            recurrence_closure_time,
        )
        recurrence_attributed_total = sum(
            value or 0.0 for value in recurrence_attribution
        )
        if recurrence_attributed_total > (
            recurrence_schedule_time + accounting_tolerance
        ):
            raise EvaluationError(
                "native recurrence profile schedule sub-attribution accounts for "
                f"{recurrence_attributed_total:.9e}s/point, exceeding the "
                "inclusive recurrence schedule time "
                f"{recurrence_schedule_time:.9e}s/point"
            )
    other_core_time = (
        None if wall_time is None else max(wall_time - accounted_total, 0.0)
    )

    return _NativeProfileSample(
        execution_mode=execution_mode,
        compiled_direct_arena_active=compiled_direct_arena_active,
        wall_time=wall_time,
        native_input_pack_time=native_input_pack_time,
        native_input_crossing_time=native_input_crossing_time,
        orchestration_time=orchestration_time,
        state_prepare_time=state_prepare_time,
        state_clear_time=state_clear_time,
        source_fill_time=source_fill_time,
        momentum_input_setup_time=momentum_input_setup_time,
        momentum_setup_time=momentum_setup_time,
        model_parameter_setup_time=model_parameter_setup_time,
        stage_input_pack_time=stage_input_pack_time,
        stage_leaf_input_pack_time=stage_leaf_input_pack_time,
        stage_evaluator_call_time=stage_evaluator_call_time,
        stage_backend_call_time=stage_backend_call_time,
        stage_evaluator_output_gather_time=stage_evaluator_output_gather_time,
        output_assign_time=output_assign_time,
        amplitude_input_pack_time=amplitude_input_pack_time,
        amplitude_evaluator_call_time=amplitude_evaluator_call_time,
        amplitude_leaf_input_pack_time=amplitude_leaf_input_pack_time,
        amplitude_backend_call_time=amplitude_backend_call_time,
        amplitude_evaluator_output_gather_time=(amplitude_evaluator_output_gather_time),
        amplitude_output_remap_time=amplitude_output_remap_time,
        reduction_time=reduction_time,
        total_materialization_time=total_materialization_time,
        final_output_copy_time=final_output_copy_time,
        selector_planner_time=selector_planner_time,
        selector_gather_time=selector_gather_time,
        selector_scatter_time=selector_scatter_time,
        other_core_time=other_core_time,
        stage_input_pack_times=(
            normalized_sequence(stage_input_pack) if compiled_profile else None
        ),
        stage_leaf_input_pack_times=(
            normalized_sequence(stage_leaf_input_pack) if compiled_profile else None
        ),
        stage_evaluator_call_times=(
            normalized_sequence(stage_evaluator_call) if compiled_profile else None
        ),
        stage_backend_call_times=(
            normalized_sequence(stage_backend_call) if compiled_profile else None
        ),
        stage_evaluator_output_gather_times=normalized_sequence(
            stage_evaluator_output_gather if compiled_profile else None
        ),
        stage_output_assign_times=(
            normalized_sequence(stage_output_assign) if compiled_profile else None
        ),
        eager_initialize_time=eager_initialize_time,
        eager_gather_time=eager_gather_time,
        eager_kernel_call_time=eager_kernel_call_time,
        eager_invocation_scatter_time=eager_invocation_scatter_time,
        eager_finalization_time=eager_finalization_time,
        eager_scatter_finalization_time=eager_scatter_finalization_time,
        eager_closure_time=eager_closure_time,
        eager_copy_out_time=eager_copy_out_time,
        recurrence_momentum_fill_time=recurrence_momentum_fill_time,
        recurrence_union_source_fill_time=recurrence_union_source_fill_time,
        recurrence_schedule_time=recurrence_schedule_time,
        recurrence_source_kernel_time=recurrence_source_kernel_time,
        recurrence_contribution_kernel_time=recurrence_contribution_kernel_time,
        recurrence_finalization_time=recurrence_finalization_time,
        recurrence_closure_time=recurrence_closure_time,
        recurrence_replay_output_mapping_time=(recurrence_replay_output_mapping_time),
        counters=counters,
    )


def _component_timing(
    values: Sequence[float | None],
) -> BenchmarkComponentTiming | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    mean, deviation, error, relative_error = _sample_statistics(available)
    return BenchmarkComponentTiming(
        mean_seconds_per_point=mean,
        uncertainty=BenchmarkStatistics(deviation, error, relative_error),
        sample_count=len(available),
    )


def _stage_component_timing(
    samples: Sequence[_NativeProfileSample],
    attribute: str,
    stage_index: int,
) -> BenchmarkComponentTiming | None:
    values: list[float | None] = []
    for sample in samples:
        stage_values = getattr(sample, attribute)
        if not isinstance(stage_values, tuple) or stage_index >= len(stage_values):
            values.append(None)
        else:
            values.append(stage_values[stage_index])
    return _component_timing(values)


def _profile_counter_summary(
    samples: Sequence[_NativeProfileSample],
) -> BenchmarkProfileCounters | None:
    counter_samples = [
        sample.counters for sample in samples if sample.counters is not None
    ]
    if not counter_samples:
        return None
    if len(counter_samples) != len(samples):
        raise EvaluationError(
            "native runtime profile counter availability changed between samples"
        )

    def mean(attribute: str) -> float | None:
        values = [
            cast(float | None, getattr(sample, attribute)) for sample in counter_samples
        ]
        available = [value for value in values if value is not None]
        if available and len(available) != len(values):
            raise EvaluationError(
                f"native runtime profile counter {attribute!r} changed availability"
            )
        return statistics.fmean(available) if available else None

    return BenchmarkProfileCounters(
        sample_count=len(counter_samples),
        native_input_components_per_point=mean("native_input_components_per_point"),
        native_input_pack_bytes_per_point=mean("native_input_pack_bytes_per_point"),
        native_input_crossing_bytes_per_point=mean(
            "native_input_crossing_bytes_per_point"
        ),
        state_components_per_point=mean("state_components_per_point"),
        state_clear_components_per_point=mean("state_clear_components_per_point"),
        source_components_per_point=mean("source_components_per_point"),
        momentum_components_per_point=mean("momentum_components_per_point"),
        model_parameter_components_per_point=mean(
            "model_parameter_components_per_point"
        ),
        stage_input_copy_components_per_point=mean(
            "stage_input_copy_components_per_point"
        ),
        stage_leaf_input_copy_components_per_point=mean(
            "stage_leaf_input_copy_components_per_point"
        ),
        stage_evaluator_output_gather_components_per_point=mean(
            "stage_evaluator_output_gather_components_per_point"
        ),
        stage_output_assign_components_per_point=mean(
            "stage_output_assign_components_per_point"
        ),
        amplitude_input_copy_components_per_point=mean(
            "amplitude_input_copy_components_per_point"
        ),
        amplitude_leaf_input_copy_components_per_point=mean(
            "amplitude_leaf_input_copy_components_per_point"
        ),
        amplitude_evaluator_output_gather_components_per_point=mean(
            "amplitude_evaluator_output_gather_components_per_point"
        ),
        amplitude_output_remap_components_per_point=mean(
            "amplitude_output_remap_components_per_point"
        ),
        reduction_input_components_per_point=mean(
            "reduction_input_components_per_point"
        ),
        selector_gather_points_per_point=mean("selector_gather_points_per_point"),
        selector_gather_bytes_per_point=mean("selector_gather_bytes_per_point"),
        selector_scatter_values_per_point=mean("selector_scatter_values_per_point"),
        resolved_materialized_components_per_point=mean(
            "resolved_materialized_components_per_point"
        ),
        total_materialized_values_per_point=mean("total_materialized_values_per_point"),
        final_output_copy_values_per_point=mean("final_output_copy_values_per_point"),
        native_input_container_allocations_per_call=mean(
            "native_input_container_allocations_per_call"
        ),
        evaluator_backend_calls_per_call=mean("evaluator_backend_calls_per_call"),
        compiled_direct_arena_engines_per_call=mean(
            "compiled_direct_arena_engines_per_call"
        ),
        compiled_direct_arena_calls_per_call=mean(
            "compiled_direct_arena_calls_per_call"
        ),
        compiled_direct_arena_boundary_input_bytes_per_call=mean(
            "compiled_direct_arena_boundary_input_bytes_per_call"
        ),
        compiled_direct_arena_boundary_current_output_bytes_per_call=mean(
            "compiled_direct_arena_boundary_current_output_bytes_per_call"
        ),
        compiled_direct_arena_boundary_amplitude_output_bytes_per_call=mean(
            "compiled_direct_arena_boundary_amplitude_output_bytes_per_call"
        ),
        recurrence_momentum_scalar_values_per_point=mean(
            "recurrence_momentum_scalar_values_per_point"
        ),
        recurrence_schedule_executions_per_call=mean(
            "recurrence_schedule_executions_per_call"
        ),
        recurrence_replay_schedule_executions_per_call=mean(
            "recurrence_replay_schedule_executions_per_call"
        ),
        recurrence_union_schedule_executions_per_call=mean(
            "recurrence_union_schedule_executions_per_call"
        ),
        recurrence_union_source_rows_per_call=mean(
            "recurrence_union_source_rows_per_call"
        ),
        recurrence_replay_output_values_per_point=mean(
            "recurrence_replay_output_values_per_point"
        ),
        recurrence_source_calls_per_call=mean("recurrence_source_calls_per_call"),
        recurrence_source_rows_per_call=mean("recurrence_source_rows_per_call"),
        recurrence_contribution_calls_per_call=mean(
            "recurrence_contribution_calls_per_call"
        ),
        recurrence_contribution_rows_per_call=mean(
            "recurrence_contribution_rows_per_call"
        ),
        recurrence_finalization_calls_per_call=mean(
            "recurrence_finalization_calls_per_call"
        ),
        recurrence_finalization_rows_per_call=mean(
            "recurrence_finalization_rows_per_call"
        ),
        recurrence_closure_calls_per_call=mean("recurrence_closure_calls_per_call"),
        recurrence_closure_rows_per_call=mean("recurrence_closure_rows_per_call"),
        observed_scratch_reallocations_per_call=mean(
            "observed_scratch_reallocations_per_call"
        ),
        native_output_allocations_per_call=mean("native_output_allocations_per_call"),
    )


def _timing_breakdown(
    samples: Sequence[_NativeProfileSample],
) -> BenchmarkTimingBreakdown:
    if not samples:
        raise EvaluationError("native runtime profile returned no timing samples")
    stage_count = max(
        (
            len(values)
            for sample in samples
            for values in (
                sample.stage_input_pack_times,
                sample.stage_leaf_input_pack_times,
                sample.stage_evaluator_call_times,
                sample.stage_backend_call_times,
                sample.stage_evaluator_output_gather_times,
                sample.stage_output_assign_times,
            )
            if values is not None
        ),
        default=0,
    )
    stages: list[BenchmarkStageTiming] = []
    for stage_index in range(stage_count):
        input_pack = _stage_component_timing(
            samples, "stage_input_pack_times", stage_index
        )
        evaluator_call = _stage_component_timing(
            samples, "stage_evaluator_call_times", stage_index
        )
        leaf_input_pack = _stage_component_timing(
            samples, "stage_leaf_input_pack_times", stage_index
        )
        backend_call = _stage_component_timing(
            samples, "stage_backend_call_times", stage_index
        )
        evaluator_output_gather = _stage_component_timing(
            samples, "stage_evaluator_output_gather_times", stage_index
        )
        output_assign = _stage_component_timing(
            samples, "stage_output_assign_times", stage_index
        )
        if any(
            value is not None
            for value in (
                input_pack,
                evaluator_call,
                output_assign,
                leaf_input_pack,
                backend_call,
                evaluator_output_gather,
            )
        ):
            stages.append(
                BenchmarkStageTiming(
                    stage_index=stage_index + 1,
                    input_pack_time=input_pack,
                    evaluator_call_time=evaluator_call,
                    output_assign_time=output_assign,
                    leaf_input_pack_time=leaf_input_pack,
                    backend_call_time=backend_call,
                    evaluator_output_gather_time=evaluator_output_gather,
                )
            )

    execution_modes = {
        sample.execution_mode for sample in samples if sample.execution_mode is not None
    }
    if len(execution_modes) > 1:
        raise EvaluationError("native runtime profile changed execution mode")
    execution_mode = cast(
        Literal["compiled", "eager", "recurrence", "on-the-fly"],
        next(iter(execution_modes), "compiled"),
    )
    evaluator_call_time = _component_timing(
        [sample.stage_evaluator_call_time for sample in samples]
    )
    return BenchmarkTimingBreakdown(
        sample_count=len(samples),
        execution_mode=execution_mode,
        wall_time=_component_timing([sample.wall_time for sample in samples]),
        native_input_pack_time=_component_timing(
            [sample.native_input_pack_time for sample in samples]
        ),
        native_input_crossing_time=_component_timing(
            [sample.native_input_crossing_time for sample in samples]
        ),
        orchestration_time=_component_timing(
            [sample.orchestration_time for sample in samples]
        ),
        state_prepare_time=_component_timing(
            [sample.state_prepare_time for sample in samples]
        ),
        state_clear_time=_component_timing(
            [sample.state_clear_time for sample in samples]
        ),
        source_fill_time=_component_timing(
            [sample.source_fill_time for sample in samples]
        ),
        momentum_setup_time=_component_timing(
            [sample.momentum_setup_time for sample in samples]
        ),
        momentum_input_setup_time=_component_timing(
            [sample.momentum_input_setup_time for sample in samples]
        ),
        model_parameter_setup_time=_component_timing(
            [sample.model_parameter_setup_time for sample in samples]
        ),
        stage_input_pack_time=_component_timing(
            [sample.stage_input_pack_time for sample in samples]
        ),
        stage_leaf_input_pack_time=_component_timing(
            [sample.stage_leaf_input_pack_time for sample in samples]
        ),
        stage_evaluator_call_time=(
            None
            if execution_mode in {"eager", *_RECURRENCE_LIKE_EXECUTION_MODES}
            else evaluator_call_time
        ),
        stage_backend_call_time=_component_timing(
            [sample.stage_backend_call_time for sample in samples]
        ),
        stage_evaluator_output_gather_time=_component_timing(
            [sample.stage_evaluator_output_gather_time for sample in samples]
        ),
        output_assign_time=_component_timing(
            [sample.output_assign_time for sample in samples]
        ),
        amplitude_input_pack_time=_component_timing(
            [sample.amplitude_input_pack_time for sample in samples]
        ),
        amplitude_evaluator_call_time=_component_timing(
            [sample.amplitude_evaluator_call_time for sample in samples]
        ),
        amplitude_leaf_input_pack_time=_component_timing(
            [sample.amplitude_leaf_input_pack_time for sample in samples]
        ),
        amplitude_backend_call_time=_component_timing(
            [sample.amplitude_backend_call_time for sample in samples]
        ),
        amplitude_evaluator_output_gather_time=_component_timing(
            [sample.amplitude_evaluator_output_gather_time for sample in samples]
        ),
        amplitude_output_remap_time=_component_timing(
            [sample.amplitude_output_remap_time for sample in samples]
        ),
        reduction_time=_component_timing([sample.reduction_time for sample in samples]),
        total_materialization_time=_component_timing(
            [sample.total_materialization_time for sample in samples]
        ),
        final_output_copy_time=_component_timing(
            [sample.final_output_copy_time for sample in samples]
        ),
        selector_planner_time=_component_timing(
            [sample.selector_planner_time for sample in samples]
        ),
        selector_gather_time=_component_timing(
            [sample.selector_gather_time for sample in samples]
        ),
        selector_scatter_time=_component_timing(
            [sample.selector_scatter_time for sample in samples]
        ),
        other_core_time=_component_timing(
            [sample.other_core_time for sample in samples]
        ),
        eager_execution_time=(
            evaluator_call_time if execution_mode == "eager" else None
        ),
        eager_initialize_time=_component_timing(
            [sample.eager_initialize_time for sample in samples]
        ),
        eager_gather_time=_component_timing(
            [sample.eager_gather_time for sample in samples]
        ),
        eager_kernel_call_time=_component_timing(
            [sample.eager_kernel_call_time for sample in samples]
        ),
        eager_invocation_scatter_time=_component_timing(
            [sample.eager_invocation_scatter_time for sample in samples]
        ),
        eager_finalization_time=_component_timing(
            [sample.eager_finalization_time for sample in samples]
        ),
        eager_scatter_finalization_time=_component_timing(
            [sample.eager_scatter_finalization_time for sample in samples]
        ),
        eager_closure_time=_component_timing(
            [sample.eager_closure_time for sample in samples]
        ),
        eager_copy_out_time=_component_timing(
            [sample.eager_copy_out_time for sample in samples]
        ),
        recurrence_momentum_fill_time=_component_timing(
            [sample.recurrence_momentum_fill_time for sample in samples]
        ),
        recurrence_union_source_fill_time=_component_timing(
            [sample.recurrence_union_source_fill_time for sample in samples]
        ),
        recurrence_schedule_time=_component_timing(
            [sample.recurrence_schedule_time for sample in samples]
        ),
        recurrence_source_kernel_time=_component_timing(
            [sample.recurrence_source_kernel_time for sample in samples]
        ),
        recurrence_contribution_kernel_time=_component_timing(
            [sample.recurrence_contribution_kernel_time for sample in samples]
        ),
        recurrence_finalization_time=_component_timing(
            [sample.recurrence_finalization_time for sample in samples]
        ),
        recurrence_closure_time=_component_timing(
            [sample.recurrence_closure_time for sample in samples]
        ),
        recurrence_replay_output_mapping_time=_component_timing(
            [sample.recurrence_replay_output_mapping_time for sample in samples]
        ),
        stages=tuple(stages),
        counters=_profile_counter_summary(samples),
    )


def _sample_statistics(samples: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    error = deviation / math.sqrt(len(samples))
    relative_error = error / mean if mean > 0.0 else 0.0
    return mean, deviation, error, relative_error


def create_benchmark_backend(
    config: BenchmarkConfig | RunConfig | None,
    progress: ProgressSink | None,
) -> BenchmarkBackend:
    return BenchmarkBackend(config, progress)


__all__ = ["BenchmarkBackend", "create_benchmark_backend"]
