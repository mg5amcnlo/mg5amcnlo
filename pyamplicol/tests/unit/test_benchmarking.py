# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyamplicol.benchmarking as benchmark_module
from pyamplicol.api import (
    BenchmarkProfileCounters,
    BenchmarkTimingBreakdown,
    ColorFlow,
    HelicityConfiguration,
    PhysicsReduction,
    ProcessPhysics,
    ReductionGroup,
    ResolvedEvaluation,
)
from pyamplicol.api.errors import EvaluationError
from pyamplicol.benchmarking import BenchmarkBackend
from pyamplicol.config import BenchmarkConfig
from pyamplicol.reporting import (
    CallbackProgressSink,
    ProgressEnd,
    ProgressStart,
    ProgressUpdate,
)


class _Runtime:
    calls = 0
    last_options: dict[str, object] | None = None

    @property
    def physics(self) -> ProcessPhysics:
        return ProcessPhysics(
            "test",
            "d d~ > z",
            "lc",
            "all",
            "all",
            "flow",
            0,
            (),
            (HelicityConfiguration("h0", 0, (1, -1), True, False, "h0", 1.0),),
            (ColorFlow("c0", 0, (1,), True, "c0", 1.0),),
            (),
            PhysicsReduction(
                "lc-diagonal",
                (ReductionGroup("g0", "h0", "c0", ("h0",), ("c0",)),),
            ),
            (),
            (),
        )

    def evaluate(self, momenta: object, **selectors: object) -> tuple[complex, ...]:
        self.calls += 1
        self.last_options = selectors
        return tuple(1.0 + 0.0j for _ in momenta)  # type: ignore[union-attr]

    def evaluate_resolved(
        self, momenta: object, **selectors: object
    ) -> ResolvedEvaluation:
        del momenta, selectors
        return ResolvedEvaluation((((1.0 + 0.0j,),),), ("h0",), ("c0",))

    def set_model_parameters(self, mapping: object) -> None:
        del mapping

    def clear(self) -> None:
        pass

    def mute_warnings(self) -> None:
        pass

    def unmute_warnings(self) -> None:
        pass


class _RuntimeTwoAxes(_Runtime):
    @property
    def physics(self) -> ProcessPhysics:
        helicities = (
            HelicityConfiguration("h0", 0, (1, -1), True, False, "h0", 1.0),
            HelicityConfiguration("h1", 1, (-1, 1), True, False, "h1", 1.0),
        )
        flows = (
            ColorFlow("c0", 0, (1,), True, "c0", 1.0),
            ColorFlow("c1", 1, (2,), True, "c1", 1.0),
        )
        return ProcessPhysics(
            "test",
            "d d~ > z",
            "lc",
            "all",
            "all",
            "flow",
            0,
            (),
            helicities,
            flows,
            (),
            PhysicsReduction(
                "lc-diagonal",
                (
                    ReductionGroup(
                        "g0",
                        "h0",
                        "c0",
                        tuple(item.id for item in helicities),
                        tuple(item.id for item in flows),
                    ),
                ),
            ),
            (),
            (),
        )


class _ImmutableRuntime:
    __slots__ = ("_artifact_path", "_delegate")

    def __init__(self, delegate: _RuntimeTwoAxes, artifact_path: Path) -> None:
        self._delegate = delegate
        self._artifact_path = artifact_path

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    @property
    def physics(self) -> ProcessPhysics:
        return self._delegate.physics

    def evaluate(self, momenta: object, **selectors: object) -> tuple[complex, ...]:
        return self._delegate.evaluate(momenta, **selectors)

    def evaluate_resolved(
        self, momenta: object, **selectors: object
    ) -> ResolvedEvaluation:
        return self._delegate.evaluate_resolved(momenta, **selectors)

    def set_model_parameters(self, mapping: object) -> None:
        self._delegate.set_model_parameters(mapping)

    def clear(self) -> None:
        self._delegate.clear()

    def mute_warnings(self) -> None:
        self._delegate.mute_warnings()

    def unmute_warnings(self) -> None:
        self._delegate.unmute_warnings()


class _RuntimeWithValidation(_Runtime):
    def validation_momenta(self) -> object:
        return (((1.0, 0.0, 0.0, 1.0),),)


class _RuntimeWithProfile(_Runtime):
    profile_calls = 0

    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        self.profile_calls += 1
        assert kwargs["helicities"] == ("h0",)
        assert kwargs["color_flows"] is None
        assert kwargs["precision"] == 16
        assert kwargs["include_values"] is False
        return {
            "points": len(momenta),  # type: ignore[arg-type]
            "wall_time_s": 12.0e-6,
            "source_fill_time_s": 1.0e-6,
            "momentum_setup_time_s": 0.5e-6,
            "stage_input_pack_time_s": 1.0e-6,
            "stage_leaf_input_pack_time_s": 0.3e-6,
            "stage_evaluator_call_time_s": 2.0e-6,
            "stage_backend_call_time_s": 1.4e-6,
            "stage_evaluator_output_gather_time_s": 0.3e-6,
            "output_assign_time_s": 0.5e-6,
            "amplitude_input_pack_time_s": 0.5e-6,
            "amplitude_evaluator_call_time_s": 6.0e-6,
            "amplitude_leaf_input_pack_time_s": 0.5e-6,
            "amplitude_backend_call_time_s": 4.5e-6,
            "amplitude_evaluator_output_gather_time_s": 0.5e-6,
            "amplitude_output_remap_time_s": 0.5e-6,
            "reduction_time_s": 0.5e-6,
            "stage_input_pack_by_stage_time_s": [0.5e-6, 0.5e-6],
            "stage_leaf_input_pack_by_stage_time_s": [0.1e-6, 0.2e-6],
            "stage_evaluator_call_by_stage_time_s": [0.5e-6, 1.5e-6],
            "stage_backend_call_by_stage_time_s": [0.4e-6, 1.0e-6],
            "stage_evaluator_output_gather_by_stage_time_s": [0.0, 0.3e-6],
            "stage_output_assign_by_stage_time_s": [0.25e-6, 0.25e-6],
        }


class _RuntimeWithUnavailableProfile(_Runtime):
    supports_profiling = False

    def profile(self, _momenta: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("unavailable native profiler must not be called")

    def profile_repeated(
        self, _momenta: object, _repetitions: int, **_kwargs: object
    ) -> dict[str, object]:
        raise AssertionError("unavailable repeated native profiler must not be called")


class _RuntimeWithPerStageProfile(_RuntimeWithProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile["stage_evaluator_call_time_s"] = -1.0
        profile["stage_evaluator_call_by_stage_time_s"] = [0.5e-6, 1.5e-6]
        return profile


class _RuntimeWithEagerProfile(_RuntimeWithProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile["wall_time_s"] = 100.0e-6
        profile["stage_input_pack_by_stage_time_s"] = []
        profile["stage_evaluator_call_by_stage_time_s"] = []
        profile["stage_output_assign_by_stage_time_s"] = []
        profile["stage_evaluator_call_time_s"] = 80.0e-6
        profile["amplitude_evaluator_call_time_s"] = 0.0
        return profile


class _RuntimeWithDetailedEagerProfile(_RuntimeWithEagerProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile.update(
            {
                "execution_mode": "eager",
                "eager_initialize_time_s": 2.0e-6,
                "eager_gather_time_s": 4.0e-6,
                "eager_kernel_call_time_s": 24.0e-6,
                "eager_invocation_scatter_time_s": 12.0e-6,
                "eager_finalization_time_s": 20.0e-6,
                "eager_scatter_finalization_time_s": 32.0e-6,
                "eager_closure_time_s": 4.0e-6,
                "eager_copy_out_time_s": 6.0e-6,
            }
        )
        return profile


class _RuntimeWithAmplitudeOnlyCompiledProfile(_RuntimeWithProfile):
    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        profile = super().profile(momenta, **kwargs)
        profile["stage_input_pack_by_stage_time_s"] = []
        profile["stage_evaluator_call_by_stage_time_s"] = []
        profile["stage_output_assign_by_stage_time_s"] = []
        profile["stage_evaluator_call_time_s"] = 0.0
        return profile


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def perf_counter(self) -> float:
        return self.value


class _TimedRuntime(_Runtime):
    def __init__(self, clock: _Clock, duration: float = 1.0e-3) -> None:
        self.clock = clock
        self.duration = duration

    def evaluate(self, momenta: object, **selectors: object) -> tuple[complex, ...]:
        self.calls += 1
        self.last_options = selectors
        self.clock.value += self.duration
        return tuple(1.0 + 0.0j for _ in momenta)  # type: ignore[union-attr]


class _TimedRuntimeWithProfile(_TimedRuntime):
    def __init__(self, clock: _Clock) -> None:
        super().__init__(clock)
        self.profile_calls = 0

    def profile(self, momenta: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        self.profile_calls += 1
        return {
            "points": len(momenta),  # type: ignore[arg-type]
            "stage_evaluator_call_time_s": 2.0e-6,
            "amplitude_evaluator_call_time_s": 6.0e-6,
        }


class _TimedRuntimeWithNativeWall(_TimedRuntimeWithProfile):
    native_wall_calls = 0
    outer_seconds_per_repetition = 0.6e-3
    evaluator_total_seconds_per_repetition = 0.5e-3

    def _benchmark_f64_wall_time(
        self,
        momenta: object,
        repetitions: int,
        *,
        helicities: object,
        color_flows: object,
        precision: int,
    ) -> float:
        assert len(momenta) == 2  # type: ignore[arg-type]
        assert helicities is None
        assert color_flows is None
        assert precision == 16
        self.native_wall_calls += 1
        self.clock.value += repetitions * self.outer_seconds_per_repetition
        return repetitions * self.evaluator_total_seconds_per_repetition


class _TimedRuntimeWithCalibrationUndershoot(_TimedRuntimeWithNativeWall):
    def __init__(self, clock: _Clock) -> None:
        super().__init__(clock)
        self.native_wall_calls = 0

    def _benchmark_f64_wall_time(
        self,
        momenta: object,
        repetitions: int,
        *,
        helicities: object,
        color_flows: object,
        precision: int,
    ) -> float:
        assert len(momenta) == 2  # type: ignore[arg-type]
        assert helicities is None
        assert color_flows is None
        assert precision == 16
        self.native_wall_calls += 1
        duration = 1.0 if self.native_wall_calls == 1 else repetitions * 0.233
        self.clock.value += duration
        return duration


class _TimedRuntimeWithRepeatedProfile(_TimedRuntimeWithNativeWall):
    repeated_profile_calls = 0

    def profile_repeated(
        self,
        momenta: object,
        repetitions: int,
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs["helicities"] is None
        assert kwargs["color_flows"] is None
        assert kwargs["precision"] == 16
        assert kwargs["include_values"] is False
        self.repeated_profile_calls += 1
        points = len(momenta) * repetitions  # type: ignore[arg-type]
        return {
            "points": points,
            "wall_time_s": points * 4.0e-6,
            "stage_evaluator_call_time_s": points * 2.0e-6,
            "amplitude_evaluator_call_time_s": points * 0.5e-6,
            "reduction_time_s": points * 0.5e-6,
            "native_input_component_count": points * 4,
            "stage_input_copy_component_count": points * 7,
            "evaluator_backend_call_count": repetitions * 3,
            "observed_scratch_reallocation_count": 0,
            "native_output_allocation_count": repetitions,
        }


class _TimedRuntimeWithCompiledDirectArenaRepeatedProfile(
    _TimedRuntimeWithRepeatedProfile
):
    def profile_repeated(
        self,
        momenta: object,
        repetitions: int,
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs["helicities"] is None
        assert kwargs["color_flows"] is None
        assert kwargs["precision"] == 16
        assert kwargs["include_values"] is False
        self.repeated_profile_calls += 1
        points = len(momenta) * repetitions  # type: ignore[arg-type]
        return {
            "execution_mode": "compiled",
            "points": points,
            "wall_time_s": points * 6.0e-6,
            "orchestration_time_s": points * 3.5e-6,
            "stage_evaluator_call_time_s": 0.0,
            "amplitude_evaluator_call_time_s": 0.0,
            "evaluator_backend_call_count": repetitions * 3,
            "compiled_direct_arena_engine_count": repetitions * 2,
            "compiled_direct_arena_call_count": repetitions * 3,
            "compiled_direct_arena_boundary_input_bytes": 0,
            "compiled_direct_arena_boundary_current_output_bytes": 0,
            "compiled_direct_arena_boundary_amplitude_output_bytes": 0,
        }


class _TimedRuntimeWithBelowResolutionCompiledDirectArenaProfile(
    _TimedRuntimeWithCompiledDirectArenaRepeatedProfile
):
    def profile_repeated(
        self,
        momenta: object,
        repetitions: int,
        **kwargs: object,
    ) -> dict[str, object]:
        profile = super().profile_repeated(momenta, repetitions, **kwargs)
        profile["orchestration_time_s"] = 0.0
        return profile


class _TimedRuntimeWithRecurrenceRepeatedProfile(_TimedRuntimeWithRepeatedProfile):
    def profile_repeated(
        self,
        momenta: object,
        repetitions: int,
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs["helicities"] is None
        assert kwargs["color_flows"] is None
        assert kwargs["precision"] == 16
        assert kwargs["include_values"] is False
        self.repeated_profile_calls += 1
        points = len(momenta) * repetitions  # type: ignore[arg-type]
        per_point = {
            "wall_time_s": 20.0e-6,
            "native_input_pack_time_s": 1.0e-6,
            "native_input_crossing_time_s": 0.5e-6,
            "orchestration_time_s": 0.5e-6,
            "state_prepare_time_s": 0.5e-6,
            "state_clear_time_s": 0.5e-6,
            "source_fill_time_s": 1.0e-6,
            "momentum_input_setup_time_s": 0.5e-6,
            "model_parameter_setup_time_s": 0.5e-6,
            "recurrence_momentum_fill_time_s": 2.0e-6,
            "recurrence_union_source_fill_time_s": 0.0,
            "recurrence_schedule_time_s": 10.0e-6,
            "recurrence_source_kernel_time_s": 1.0e-6,
            "recurrence_contribution_kernel_time_s": 6.0e-6,
            "recurrence_finalization_time_s": 2.0e-6,
            "recurrence_closure_time_s": 1.0e-6,
            "recurrence_replay_output_mapping_time_s": 1.0e-6,
            "reduction_time_s": 1.0e-6,
            "total_materialization_time_s": 0.5e-6,
            "final_output_copy_time_s": 0.5e-6,
        }
        return {
            "execution_mode": "recurrence",
            "points": points,
            **{key: value * points for key, value in per_point.items()},
            "recurrence_momentum_scalar_value_count": points * 12,
            "recurrence_schedule_execution_count": repetitions,
            "recurrence_replay_schedule_execution_count": repetitions,
            "recurrence_union_schedule_execution_count": 0,
            "recurrence_union_source_row_count": 0,
            "recurrence_replay_output_value_count": points * 2,
            "recurrence_source_call_count": repetitions * 4,
            "recurrence_source_row_count": repetitions * 31,
            "recurrence_contribution_call_count": repetitions * 7,
            "recurrence_contribution_row_count": repetitions * 126,
            "recurrence_finalization_call_count": repetitions * 5,
            "recurrence_finalization_row_count": repetitions * 34,
            "recurrence_closure_call_count": repetitions * 2,
            "recurrence_closure_row_count": repetitions * 12,
        }


class _TimedRuntimeWithOnTheFlyRepeatedProfile(
    _TimedRuntimeWithRecurrenceRepeatedProfile
):
    def __init__(
        self,
        clock: _Clock,
        *,
        already_retained: bool = False,
        retain_after_evaluation: bool = True,
        structural_zero: bool = False,
    ) -> None:
        super().__init__(clock)
        self.native_wall_calls = 0
        self.repeated_profile_calls = 0
        self.otf_events: list[str] = []
        self.otf_context_access_count = 0
        self.structural_zero = structural_zero
        self.otf_census = _otf_runtime_state_census(
            already_retained,
            structural_zero=structural_zero,
        )
        self.retain_after_evaluation = retain_after_evaluation

    @property
    def execution_mode(self) -> str:
        return "on-the-fly"

    @property
    def physics(self) -> ProcessPhysics:
        raise AssertionError("on-the-fly benchmark opened dense physics metadata")

    def _on_the_fly_benchmark_context(
        self,
        color_flow_ids: tuple[str, ...],
    ) -> dict[str, object]:
        self.otf_context_access_count += 1
        assert color_flow_ids == ()
        return {
            "process_id": "otf-test",
            "process_expression": "d d~ > t t~ g g",
            "color_accuracy": "lc",
            "helicity_count": 64,
            "color_count": 24,
            "selected_color_ids": (),
        }

    def _on_the_fly_runtime_state_census(self) -> dict[str, object]:
        retained = self.otf_census["retained_family_count"] != 0
        self.otf_events.append(f"census:{'retained' if retained else 'cold'}")
        return self.otf_census

    def _benchmark_f64_wall_time(
        self,
        momenta: object,
        repetitions: int,
        *,
        helicities: object,
        color_flows: object,
        precision: int,
    ) -> float:
        self.otf_events.append("evaluate")
        if self.retain_after_evaluation:
            self.otf_census.clear()
            self.otf_census.update(
                _otf_runtime_state_census(
                    True,
                    structural_zero=self.structural_zero,
                )
            )
        return super()._benchmark_f64_wall_time(
            momenta,
            repetitions,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
        )

    def profile_repeated(
        self,
        momenta: object,
        repetitions: int,
        **kwargs: object,
    ) -> dict[str, object]:
        profile = super().profile_repeated(momenta, repetitions, **kwargs)
        profile["execution_mode"] = "on-the-fly"
        return profile


class _TimedRuntimeWithColdOnTheFlyConstruction(
    _TimedRuntimeWithOnTheFlyRepeatedProfile
):
    def _benchmark_f64_wall_time(
        self,
        momenta: object,
        repetitions: int,
        *,
        helicities: object,
        color_flows: object,
        precision: int,
    ) -> float:
        assert len(momenta) == 2  # type: ignore[arg-type]
        assert helicities is None
        assert color_flows is None
        assert precision == 16
        cold = self.otf_census["retained_family_count"] == 0
        self.native_wall_calls += 1
        self.otf_events.append("evaluate")
        self.otf_census.clear()
        self.otf_census.update(_otf_runtime_state_census(True))
        outer_seconds = 120.0 if cold else repetitions * 0.6e-3
        evaluator_seconds = 120.0 if cold else repetitions * 0.5e-3
        self.clock.value += outer_seconds
        return evaluator_seconds


def _otf_runtime_state_census(
    retained: bool,
    *,
    structural_zero: bool = False,
) -> dict[str, object]:
    count = int(retained)
    executable_count = int(retained and not structural_zero)
    active: dict[str, object] | None = None
    if executable_count:
        active = {
            "basis": "shared-query-family-union-v1",
            "scope": "active-family-union",
            "query_count": 1,
            "union_unique_current_count": 2,
            "union_unique_current_component_count": 3,
            "union_source_rows": 2,
            "union_contribution_rows": 2,
            "union_finalization_rows": 2,
            "union_closure_rows": 2,
            "union_amplitude_destination_count": 1,
            "union_source_executor_call_groups": 1,
            "union_contribution_executor_call_groups": 1,
            "union_finalization_executor_call_groups": 1,
            "union_closure_executor_call_groups": 1,
        }
    return {
        "kind": "rusticol-on-the-fly-runtime-state-census-v1",
        "process_id": "otf-test",
        "family_cache_policy": "last-family-only",
        "family_cache_limit": 1,
        "process_preparation_count": count,
        "retained_family_count": count,
        "pending_family_count": 0,
        "retained_selection_count": count,
        "retained_request_count": count,
        "retained_amplitude_destination_count": executable_count,
        "retained_executor_handle_count": executable_count,
        "retained_query_local_trace_count": 0,
        "retained_embedded_lookup_key_count": 0,
        "semantic_executor_binding_count": executable_count,
        "active_family_union_census": active,
    }


def test_on_the_fly_benchmark_uses_compact_context_without_opening_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithOnTheFlyRepeatedProfile(clock)
    runtime.repeated_profile_calls = 0
    runtime.native_wall_calls = 0
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.process_id == "otf-test"
    assert result.process_expression == "d d~ > t t~ g g"
    assert result.environment["execution_mode"] == "on-the-fly"
    assert result.environment["color_workload"] == "all 24 generated physical LC flows"
    assert result.environment["helicity_workload"] == (
        "all 64 generated helicity configurations"
    )
    assert result.evaluator_time_per_point == pytest.approx(10.0e-6)
    assert result.timing_breakdown is not None
    assert result.timing_breakdown.execution_mode == "on-the-fly"
    assert result.timing_breakdown.recurrence_schedule_time is not None
    assert runtime.otf_events[:4] == (
        ["census:cold", "evaluate", "census:retained", "evaluate"]
    )
    environment = result.environment
    assert environment["cold_warmup_runtime_state_evidence"] == (
        "authenticated-native-otf-census-v1"
    )
    assert environment["cold_warmup_runtime_freshness"] == "authenticated-cold"
    assert environment["cold_warmup_runtime_cold_before_first_evaluation"] is True
    assert environment["cold_warmup_runtime_retained_before_first_evaluation"] is False
    assert environment["cold_warmup_runtime_retained_after_first_evaluation"] is True
    before = environment["cold_warmup_runtime_state_before"]
    after = environment["cold_warmup_runtime_state_after"]
    assert isinstance(before, Mapping)
    assert isinstance(after, Mapping)
    assert before["retained_family_count"] == 0
    assert after["retained_family_count"] == 1


def test_on_the_fly_benchmark_authenticates_already_retained_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithOnTheFlyRepeatedProfile(clock, already_retained=True)
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert runtime.otf_events[:4] == (
        ["census:retained", "evaluate", "census:retained", "evaluate"]
    )
    environment = result.environment
    assert environment["cold_warmup_runtime_freshness"] == (
        "authenticated-already-retained"
    )
    assert environment["cold_warmup_runtime_cold_before_first_evaluation"] is False
    assert environment["cold_warmup_runtime_retained_before_first_evaluation"] is True
    assert environment["cold_warmup_runtime_retained_after_first_evaluation"] is True


def test_on_the_fly_benchmark_accepts_retained_structural_zero_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithOnTheFlyRepeatedProfile(
        clock,
        structural_zero=True,
    )
    result = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=0.1,
            batch_size=2,
            warmup_runs=1,
            minimum_samples=4,
        ),
        None,
    ).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    after = result.environment["cold_warmup_runtime_state_after"]
    assert isinstance(after, Mapping)
    assert after["retained_family_count"] == 1
    assert after["retained_selection_count"] == 1
    assert after["retained_request_count"] == 1
    assert after["retained_amplitude_destination_count"] == 0
    assert after["retained_executor_handle_count"] == 0
    assert after["semantic_executor_binding_count"] == 0
    assert after["active_family_union_census"] is None
    assert (
        result.environment["cold_warmup_runtime_retained_after_first_evaluation"]
        is True
    )
    for field in (
        "retained_amplitude_destination_count",
        "retained_executor_handle_count",
        "semantic_executor_binding_count",
    ):
        partial = dict(after)
        partial[field] = 1
        assert not benchmark_module._on_the_fly_runtime_state_is_retained(partial)


def test_on_the_fly_cold_construction_does_not_seed_warm_chunk_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithColdOnTheFlyConstruction(clock)
    guarded: list[tuple[float | None, str]] = []

    def guard(estimated_seconds: float | None, description: str) -> None:
        guarded.append((estimated_seconds, description))
        if estimated_seconds is not None and estimated_seconds > 1.0:
            raise AssertionError("cold construction estimate leaked into warm timing")

    backend = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=0.01,
            batch_size=2,
            warmup_runs=1,
            minimum_samples=4,
        ),
        None,
    )
    backend.set_chunk_guard(guard)
    result = backend.run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    evaluator_guards = [
        estimate
        for estimate, description in guarded
        if description == "runtime evaluator timing chunk"
    ]
    assert evaluator_guards[:2] == [None, None]
    assert result.environment["cold_warmup_elapsed_seconds"] == pytest.approx(120.0)
    assert result.environment["elapsed_seconds"] < 1.0


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("kind", "wrong-kind", "invalid kind"),
        ("process_id", "wrong-process", "does not match"),
        ("family_cache_policy", "all-seen", "cache policy"),
        ("family_cache_limit", True, "cache limit"),
        ("retained_family_count", 2, "exceeds its cache limit"),
        ("retained_family_count", -1, "retained_family_count"),
        ("pending_family_count", True, "pending_family_count"),
        ("active_family_union_census", "invalid", "active-family"),
        ("process_preparation_count", 1, "neither cold nor fully retained"),
    ),
)
def test_on_the_fly_benchmark_rejects_malformed_cold_state_before_evaluation(
    field: str,
    replacement: object,
    message: str,
) -> None:
    runtime = _TimedRuntimeWithOnTheFlyRepeatedProfile(_Clock())
    runtime.otf_census[field] = replacement

    with pytest.raises(EvaluationError, match=message):
        BenchmarkBackend(BenchmarkConfig(), None).run(
            runtime,
            points=(((1.0, 0.0, 0.0, 1.0),),),
        )

    assert runtime.native_wall_calls == 0
    assert "evaluate" not in runtime.otf_events


def test_on_the_fly_benchmark_requires_first_evaluation_to_retain_family() -> None:
    runtime = _TimedRuntimeWithOnTheFlyRepeatedProfile(
        _Clock(),
        retain_after_evaluation=False,
    )

    with pytest.raises(EvaluationError, match="did not leave a fully retained"):
        BenchmarkBackend(BenchmarkConfig(batch_size=2), None).run(
            runtime,
            points=(((1.0, 0.0, 0.0, 1.0),),),
        )

    assert runtime.native_wall_calls == 1
    assert runtime.otf_events == ["census:cold", "evaluate", "census:cold"]


def test_on_the_fly_benchmark_requires_runtime_state_census() -> None:
    runtime = _TimedRuntimeWithOnTheFlyRepeatedProfile(_Clock())
    runtime._on_the_fly_runtime_state_census = None  # type: ignore[method-assign]

    with pytest.raises(EvaluationError, match=r"requires.*runtime state census"):
        BenchmarkBackend(BenchmarkConfig(), None).run(
            runtime,
            points=(((1.0, 0.0, 0.0, 1.0),),),
        )

    assert runtime.native_wall_calls == 0
    assert runtime.otf_events == []


def test_on_the_fly_benchmark_rejects_non_f64_before_compact_state_access() -> None:
    runtime = _TimedRuntimeWithOnTheFlyRepeatedProfile(_Clock())

    with pytest.raises(EvaluationError, match="supports only precision=16"):
        BenchmarkBackend(BenchmarkConfig(precision=8), None).run(
            runtime,
            points=(((1.0, 0.0, 0.0, 1.0),),),
        )

    assert runtime.otf_context_access_count == 0
    assert runtime.native_wall_calls == 0
    assert runtime.otf_events == []


def test_benchmark_measures_minimum_samples_and_requested_batch() -> None:
    runtime = _Runtime()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=2,
        minimum_samples=5,
    )
    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )
    assert result.sample_count == 5
    assert runtime.calls == 7
    assert result.wall_time_per_point >= 0.0
    assert result.evaluator_time_per_point is None
    assert result.environment["batch_size"] == 4
    assert result.environment["wall_time_source"] == (
        "python_outer_perf_counter_wall_time"
    )
    assert result.environment["wall_time_sample_pass"] == (
        "time.perf_counter[measure_repetitions]"
    )
    assert result.environment["evaluator_time_source"] == "unavailable"
    assert result.environment["evaluator_time_status"] == "unavailable"
    assert result.repetitions_per_sample == 1
    assert result.evaluation_count == 5
    assert result.evaluated_point_count == 20
    assert result.evaluator_total_time_per_point == pytest.approx(
        result.environment["evaluator_total_accumulated_seconds"]
        / result.environment["measured_point_count"]
    )
    assert result.environment[
        "evaluator_total_time_raw_seconds_per_point"
    ] == pytest.approx(result.evaluator_total_time_per_point)
    assert (
        result.environment["evaluator_total_time_source"]
        == "runtime.evaluate.accumulated"
    )
    assert (
        result.environment["evaluator_total_time_sample_contract"]
        == "accumulated-repeated-warmed-evaluator-total-v1"
    )


def _write_effective_color_config(
    artifact: Path,
    *,
    accuracy: str = "lc",
    layout: str | None = "topology-replay",
) -> None:
    config = artifact / "config"
    config.mkdir(parents=True)
    layout_line = "" if layout is None else f'lc_flow_layout = "{layout}"\n'
    (config / "effective.toml").write_text(
        f'[color]\naccuracy = "{accuracy}"\n{layout_line}',
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("layout", "expected_helicities", "expected_color_flows"),
    (
        ("topology-replay", None, ("c0",)),
        ("all-flow-union", ("h0",), None),
    ),
)
def test_benchmark_defaults_to_generated_lc_hot_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    expected_helicities: tuple[str, ...] | None,
    expected_color_flows: tuple[str, ...] | None,
) -> None:
    artifact = tmp_path / "artifact"
    _write_effective_color_config(artifact, layout=layout)
    runtime = _RuntimeTwoAxes()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=1,
        warmup_runs=0,
        minimum_samples=1,
    )
    backend = BenchmarkBackend(config, None)
    monkeypatch.setattr(backend, "_runtime", lambda _target: runtime)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = backend.run(
            artifact,
            points=(((1.0, 0.0, 0.0, 1.0),),),
        )

    assert caught == []
    assert runtime.last_options == {
        "helicities": expected_helicities,
        "color_flows": expected_color_flows,
        "precision": 16,
    }
    assert result.requested_config.helicity_ids == ()
    assert result.requested_config.color_flow_ids == ()
    assert result.effective_config.helicity_ids == tuple(expected_helicities or ())
    assert result.effective_config.color_flow_ids == tuple(expected_color_flows or ())
    assert result.environment["selected_helicity_ids"] == tuple(
        expected_helicities or ()
    )
    assert result.environment["selected_color_ids"] == tuple(
        expected_color_flows or ()
    )


def test_loaded_runtime_recovers_artifact_layout_for_profile_default(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_effective_color_config(artifact, layout="topology-replay")
    runtime = _RuntimeTwoAxes()
    runtime._artifact_path = artifact

    result = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=1.0e-12,
            batch_size=1,
            warmup_runs=0,
            minimum_samples=1,
        ),
        None,
    ).run(runtime, points=(((1.0, 0.0, 0.0, 1.0),),))

    assert result.effective_config.color_flow_ids == ("c0",)
    assert runtime.last_options is not None
    assert runtime.last_options["color_flows"] == ("c0",)


def test_lc_profile_defaults_respect_complete_and_subset_axis_intent() -> None:
    physics = SimpleNamespace(
        color_accuracy="lc",
        color_flows=(
            ColorFlow("c0", 0, (1,), True, "c0", 1.0),
            ColorFlow("c1", 1, (2,), True, "c1", 1.0),
        ),
        color_ids=("c0", "c1"),
        helicities=(
            HelicityConfiguration("h0", 0, (1, -1), True, False, "h0", 1.0),
            HelicityConfiguration("h1", 1, (-1, 1), True, False, "h1", 1.0),
        ),
        helicity_ids=("h0", "h1"),
    )

    resolve = benchmark_module._default_lc_profile_selectors
    assert resolve(
        physics=physics,
        lc_flow_layout="topology-replay",
        selected_helicity_ids=("h1", "h0"),
        selected_color_ids=(),
    ) == (None, ("c0",))
    assert resolve(
        physics=physics,
        lc_flow_layout="all-flow-union",
        selected_helicity_ids=(),
        selected_color_ids=("c1", "c0"),
    ) == (("h0",), None)
    assert resolve(
        physics=physics,
        lc_flow_layout="topology-replay",
        selected_helicity_ids=("h0", "h1"),
        selected_color_ids=("c1",),
    ) == (None, ("c1",))
    assert resolve(
        physics=physics,
        lc_flow_layout="all-flow-union",
        selected_helicity_ids=("h1",),
        selected_color_ids=("c0", "c1"),
    ) == (("h1",), None)

    assert resolve(
        physics=physics,
        lc_flow_layout="topology-replay",
        selected_helicity_ids=(),
        selected_color_ids=("c0", "c1"),
    ) == (None, None)
    assert resolve(
        physics=physics,
        lc_flow_layout="all-flow-union",
        selected_helicity_ids=("h0", "h1"),
        selected_color_ids=(),
    ) == (None, None)

    assert resolve(
        physics=physics,
        lc_flow_layout="topology-replay",
        selected_helicity_ids=("h0",),
        selected_color_ids=(),
    ) == (("h0",), None)
    assert resolve(
        physics=physics,
        lc_flow_layout="all-flow-union",
        selected_helicity_ids=(),
        selected_color_ids=("c0",),
    ) == (None, ("c0",))


def test_benchmark_recommends_union_for_topology_replay_all_flow_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    _write_effective_color_config(artifact, layout=None)
    runtime = _RuntimeTwoAxes()
    backend = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=1.0e-12,
            batch_size=1,
            warmup_runs=0,
            minimum_samples=1,
            helicity_ids=("h0",),
        ),
        None,
    )
    monkeypatch.setattr(backend, "_runtime", lambda _target: runtime)

    with pytest.warns(
        UserWarning, match="--lc-flow-layout all-flow-union"
    ) as warning_records:
        result = backend.run(
            artifact,
            points=(((1.0, 0.0, 0.0, 1.0),),),
        )

    assert len(warning_records) == 1
    assert result.environment["lc_flow_layout"] == "topology-replay"
    recommendation = result.environment["lc_flow_layout_recommendation"]
    assert isinstance(recommendation, str)
    assert "--lc-flow-layout all-flow-union" in recommendation
    assert "\x1b" not in recommendation


def test_non_hot_profile_warning_is_once_per_loaded_process(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_effective_color_config(artifact, layout="topology-replay")
    runtime = _RuntimeTwoAxes()
    runtime._artifact_path = artifact
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=1,
        warmup_runs=0,
        minimum_samples=1,
        helicity_ids=("h0",),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            BenchmarkBackend(config, None).run(
                runtime,
                points=(((1.0, 0.0, 0.0, 1.0),),),
            )

    assert len(caught) == 1
    assert "single-flow/helicity-sum" in str(caught[0].message)


def test_non_hot_profile_warning_is_once_for_immutable_runtime(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_effective_color_config(artifact, layout="topology-replay")
    runtime = _ImmutableRuntime(_RuntimeTwoAxes(), artifact)
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=1,
        warmup_runs=0,
        minimum_samples=1,
        helicity_ids=("h0",),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            BenchmarkBackend(config, None).run(
                runtime,  # type: ignore[arg-type]
                points=(((1.0, 0.0, 0.0, 1.0),),),
            )

    assert len(caught) == 1


@pytest.mark.parametrize(
    ("layout", "color_accuracy", "helicity_ids", "color_flow_ids"),
    (
        ("all-flow-union", "lc", ("h0",), ()),
        ("topology-replay", "lc", (), ("c0",)),
        ("topology-replay", "nlc", ("h0",), ()),
        ("topology-replay", "full", ("h0",), ()),
    ),
)
def test_lc_flow_layout_recommendation_exclusions(
    layout: str,
    color_accuracy: str,
    helicity_ids: tuple[str, ...],
    color_flow_ids: tuple[str, ...],
) -> None:
    assert (
        benchmark_module._lc_flow_layout_recommendation(
            color_accuracy=color_accuracy,
            lc_flow_layout=layout,
            selected_helicity_ids=helicity_ids,
            selected_color_ids=color_flow_ids,
        )
        is None
    )


@pytest.mark.parametrize(
    ("layout", "helicity_ids", "color_flow_ids", "message"),
    (
        ("topology-replay", ("h0",), ("c0",), "single-flow/helicity-sum"),
        ("all-flow-union", (), ("c0",), "all-flows/single-helicity"),
    ),
)
def test_non_hot_lc_profile_shapes_have_one_run_level_warning(
    layout: str,
    helicity_ids: tuple[str, ...],
    color_flow_ids: tuple[str, ...],
    message: str,
) -> None:
    recommendation = benchmark_module._lc_flow_layout_recommendation(
        color_accuracy="lc",
        lc_flow_layout=layout,
        selected_helicity_ids=helicity_ids,
        selected_color_ids=color_flow_ids,
    )
    assert recommendation is not None
    assert message in recommendation


def test_runtime_backend_has_no_artifact_layout_recommendation() -> None:
    runtime = _Runtime()
    result = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=1.0e-12,
            batch_size=1,
            warmup_runs=0,
            minimum_samples=1,
            helicity_ids=("h0",),
        ),
        None,
    ).run(runtime, points=(((1.0, 0.0, 0.0, 1.0),),))

    assert "lc_flow_layout" not in result.environment
    assert "lc_flow_layout_recommendation" not in result.environment


def test_benchmark_calibrates_blocks_and_repetitions_toward_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntime(clock)
    events: list[object] = []
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, CallbackProgressSink(events.append)).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.sample_count == 4
    assert result.repetitions_per_sample == 25
    assert result.evaluation_count == 100
    assert result.evaluated_point_count == 200
    assert result.environment["elapsed_seconds"] == pytest.approx(0.1)
    assert result.environment["calibration_evaluation_count"] == 25
    assert runtime.calls == 126
    assert result.uncertainty.standard_error == pytest.approx(0.0)
    assert events[0] == ProgressStart(
        "runtime-profile-calibration", "Calibrating runtime profile"
    )
    assert (
        ProgressEnd(
            "runtime-profile-calibration",
            message="4 blocks x 25 repetitions",
        )
        in events
    )
    assert ProgressStart("runtime-benchmark", "Profiling runtime", 4) in events
    updates = [
        event
        for event in events
        if isinstance(event, ProgressUpdate) and event.task_id == "runtime-benchmark"
    ]
    assert updates
    assert all(
        event.details["progress_kind"] == "benchmark-statistics"
        and event.details["repetitions"] == 25
        and event.details["batch_size"] == 2
        for event in updates
    )
    assert all("\x1b[" not in (event.message or "") for event in updates)


def test_benchmark_single_sample_uses_probe_then_one_profile_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntime(clock)
    config = BenchmarkConfig(
        target_runtime=1.0,
        batch_size=2,
        warmup_runs=0,
        minimum_samples=1,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.sample_count == 1
    assert result.repetitions_per_sample == 1000
    assert result.evaluation_count == 1000
    assert runtime.calls == 1001
    assert result.environment["elapsed_seconds"] == pytest.approx(1.0)
    assert result.environment["initial_planned_sample_count"] == 1
    assert result.environment["adaptive_extension_sample_count"] == 0
    assert result.environment["calibration_evaluation_count"] == 1


def test_benchmark_chunk_guard_runs_before_the_next_calibrated_call() -> None:
    runtime = _RuntimeWithValidation()
    runtime.calls = 0
    calls: list[tuple[float | None, str]] = []

    class StageBudgetReached(RuntimeError):
        pass

    def guard(estimated_seconds: float | None, description: str) -> None:
        calls.append((estimated_seconds, description))
        if estimated_seconds is not None:
            raise StageBudgetReached

    backend = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=0.005,
            batch_size=1,
            warmup_runs=0,
            minimum_samples=5,
        ),
        None,
    )
    backend.set_chunk_guard(guard)

    with pytest.raises(StageBudgetReached):
        backend.run(runtime)

    assert calls[0] == (None, "runtime evaluator timing chunk")
    assert calls[1][0] is not None
    assert calls[1][1] == "runtime evaluator timing chunk"
    assert runtime.calls == 1


def test_benchmark_extends_complete_blocks_when_calibration_undershoots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithCalibrationUndershoot(clock)
    config = BenchmarkConfig(
        target_runtime=5.0,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=20,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.sample_count == 22
    assert result.environment["elapsed_seconds"] == pytest.approx(5.126)
    assert result.environment["elapsed_seconds"] >= config.target_runtime
    assert result.environment["initial_planned_sample_count"] == 20
    assert result.environment["adaptive_extension_sample_count"] == 2
    assert result.environment["completed_sample_count"] == 22
    assert result.environment["planned_sample_count"] == 22
    assert result.environment["completion_fraction"] == pytest.approx(1.0)


def test_benchmark_fails_closed_when_target_extension_cap_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(benchmark_module, "_MAX_TARGET_RUNTIME_SAMPLE_FACTOR", 1)
    runtime = _TimedRuntimeWithCalibrationUndershoot(clock)

    with pytest.raises(
        EvaluationError,
        match="could not reach its target duration of 5s within 20 complete blocks",
    ):
        BenchmarkBackend(
            BenchmarkConfig(
                target_runtime=5.0,
                batch_size=2,
                warmup_runs=1,
                minimum_samples=20,
            ),
            None,
        ).run(
            runtime,
            points=(((1.0, 0.0, 0.0, 1.0),),),
        )


def test_native_profile_calls_are_bounded_independently_of_wall_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithProfile(clock)
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.repetitions_per_sample == 25
    assert runtime.profile_calls == config.warmup_runs + result.sample_count
    assert result.environment["native_profile_calls_per_block"] == 1


def test_native_profile_calls_are_capped_for_long_target_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithNativeWall(clock)
    config = BenchmarkConfig(
        target_runtime=20.0,
        batch_size=2,
        warmup_runs=2,
        minimum_samples=5,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.sample_count == 80
    assert result.environment["native_profile_sample_count"] == 5
    assert result.environment["native_profile_sample_limit"] == 5
    assert result.environment["native_profile_calls_per_block"] == pytest.approx(5 / 80)
    assert result.environment["native_profile_repetitions_per_sample"] == 1
    assert result.environment["native_profile_points_per_sample"] == 2
    assert result.environment["timing_sample_contract"] == (
        "separate_native_profile_diagnostic_v1"
    )
    assert runtime.profile_calls == config.warmup_runs + 5


def test_benchmark_uses_repeated_native_rusticol_wall_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithNativeWall(clock)
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.wall_time_per_point == pytest.approx(0.3e-3)
    assert result.evaluator_total_time_per_point == pytest.approx(0.25e-3)
    assert result.environment["wall_time_source"] == (
        "python_outer_perf_counter_wall_time"
    )
    assert result.environment["elapsed_seconds"] == pytest.approx(0.12)
    assert result.environment["evaluator_total_accumulated_seconds"] == pytest.approx(
        0.1
    )
    assert runtime.native_wall_calls >= result.sample_count
    assert result.environment["timing_sample_contract"] == (
        "separate_native_profile_diagnostic_v1"
    )
    assert result.environment["wall_time_sample_pass"] == (
        "time.perf_counter[measure_repetitions]"
    )
    assert result.environment["evaluator_total_time_source"] == (
        "runtime._benchmark_f64_wall_time.accumulated"
    )
    assert result.environment["evaluator_time_sample_pass"] == "runtime.profile"
    assert result.environment["native_profile_repetitions_per_sample"] == 1
    assert result.repetitions_per_sample > 1


def test_repeated_native_profile_is_paired_with_unprofiled_headline_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithRepeatedProfile(clock)
    runtime.repeated_profile_calls = 0
    runtime.native_wall_calls = 0
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert runtime.repeated_profile_calls == config.warmup_runs + result.sample_count
    assert runtime.native_wall_calls >= config.warmup_runs + result.sample_count
    assert result.wall_time_per_point == pytest.approx(0.3e-3)
    assert result.evaluator_total_time_per_point == pytest.approx(0.25e-3)
    assert result.evaluator_time_per_point == pytest.approx(2.5e-6)
    assert result.timing_breakdown is not None
    assert result.timing_breakdown.wall_time is not None
    assert result.timing_breakdown.wall_time.mean_seconds_per_point == pytest.approx(
        4.0e-6
    )
    assert result.timing_breakdown.sample_count == result.sample_count
    counters = result.timing_breakdown.counters
    assert isinstance(counters, BenchmarkProfileCounters)
    assert counters.sample_count == result.sample_count
    assert counters.native_input_components_per_point == pytest.approx(4.0)
    assert counters.stage_input_copy_components_per_point == pytest.approx(7.0)
    assert counters.evaluator_backend_calls_per_call == pytest.approx(3.0)
    assert counters.observed_scratch_reallocations_per_call == pytest.approx(0.0)
    assert counters.native_output_allocations_per_call == pytest.approx(1.0)
    assert result.environment["native_profile_sample_count"] == result.sample_count
    assert result.environment["native_profile_repetitions_per_sample"] == (
        result.repetitions_per_sample
    )
    assert result.environment["native_profile_batch_size"] == config.batch_size
    assert result.environment["native_profile_points_per_sample"] == (
        result.repetitions_per_sample * config.batch_size
    )
    assert result.environment["wall_time_sample_pass"] == (
        "time.perf_counter[measure_repetitions]"
    )
    assert result.environment["evaluator_time_sample_pass"] == (
        "runtime.profile_repeated"
    )
    assert result.environment["timing_sample_contract"] == (
        "paired_unprofiled_headline_profiled_attribution_v1"
    )
    assert result.environment["profile_attribution_paired_with_headline"] is True
    assert result.environment["profile_attribution_identical_batch"] is True
    assert result.environment["profile_attribution_identical_repetitions"] is True
    assert result.environment["profile_attribution_evaluation_count"] == (
        result.evaluation_count
    )
    assert result.environment["profile_attribution_point_count"] == (
        result.evaluated_point_count
    )


def test_compiled_direct_arena_uses_fused_orchestration_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithCompiledDirectArenaRepeatedProfile(clock)
    runtime.repeated_profile_calls = 0
    runtime.native_wall_calls = 0
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.evaluator_time_per_point == pytest.approx(3.5e-6)
    assert result.environment["evaluator_time_source"] == (
        "runtime_profile_core_compiled_direct_arena_orchestration_time"
    )
    assert result.environment["compiled_direct_arena_active"] is True
    breakdown = result.timing_breakdown
    assert breakdown is not None
    assert breakdown.execution_mode == "compiled"
    assert breakdown.orchestration_time is not None
    assert breakdown.orchestration_time.mean_seconds_per_point == pytest.approx(3.5e-6)
    assert breakdown.stage_evaluator_call_time is not None
    assert breakdown.stage_evaluator_call_time.mean_seconds_per_point == 0.0
    assert breakdown.amplitude_evaluator_call_time is not None
    assert breakdown.amplitude_evaluator_call_time.mean_seconds_per_point == 0.0
    counters = breakdown.counters
    assert counters is not None
    assert counters.evaluator_backend_calls_per_call == pytest.approx(3.0)
    assert counters.compiled_direct_arena_engines_per_call == pytest.approx(2.0)
    assert counters.compiled_direct_arena_calls_per_call == pytest.approx(3.0)
    assert (
        counters.compiled_direct_arena_boundary_input_bytes_per_call
        == pytest.approx(0.0)
    )
    assert (
        counters.compiled_direct_arena_boundary_current_output_bytes_per_call
        == pytest.approx(0.0)
    )
    assert (
        counters.compiled_direct_arena_boundary_amplitude_output_bytes_per_call
        == pytest.approx(0.0)
    )


def test_compiled_direct_arena_zero_is_recorded_below_timer_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithBelowResolutionCompiledDirectArenaProfile(clock)
    runtime.repeated_profile_calls = 0
    runtime.native_wall_calls = 0
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.evaluator_time_per_point is None
    assert result.environment["evaluator_time_status"] == "below_timer_resolution"
    assert result.environment["evaluator_time_ratio_eligible"] is False
    assert result.environment["evaluator_time_raw_seconds_per_point"] == 0.0
    assert result.environment["compiled_direct_arena_active"] is True
    assert result.timing_breakdown is not None
    assert result.timing_breakdown.orchestration_time is not None
    assert result.timing_breakdown.orchestration_time.mean_seconds_per_point == 0.0


def test_recurrence_profile_uses_paired_schedule_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithRecurrenceRepeatedProfile(clock)
    runtime.repeated_profile_calls = 0
    runtime.native_wall_calls = 0
    config = BenchmarkConfig(
        target_runtime=0.1,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=4,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.wall_time_per_point == pytest.approx(0.3e-3)
    assert result.evaluator_total_time_per_point == pytest.approx(0.25e-3)
    assert result.evaluator_time_per_point == pytest.approx(10.0e-6)
    assert (
        len(
            {
                result.wall_time_per_point,
                result.evaluator_total_time_per_point,
                result.evaluator_time_per_point,
            }
        )
        == 3
    )
    assert result.environment["evaluator_time_source"] == (
        "runtime_profile_core_recurrence_schedule_time"
    )
    assert result.environment["timing_sample_contract"] == (
        "paired_unprofiled_headline_profiled_attribution_v1"
    )
    assert result.environment["native_profile_batch_size"] == config.batch_size
    assert result.environment[
        "evaluator_total_time_raw_seconds_per_point"
    ] == pytest.approx(0.25e-3)
    assert result.environment["evaluator_total_accumulated_seconds"] == pytest.approx(
        0.1
    )
    breakdown = result.timing_breakdown
    assert breakdown is not None
    assert breakdown.execution_mode == "recurrence"
    assert breakdown.stage_evaluator_call_time is None
    assert breakdown.amplitude_evaluator_call_time is None
    assert breakdown.stages == ()
    assert breakdown.recurrence_schedule_time is not None
    assert breakdown.recurrence_schedule_time.mean_seconds_per_point == pytest.approx(
        10.0e-6
    )
    assert breakdown.recurrence_contribution_kernel_time is not None
    contribution = breakdown.recurrence_contribution_kernel_time
    assert contribution.mean_seconds_per_point == pytest.approx(6.0e-6)
    assert breakdown.other_core_time is not None
    assert breakdown.other_core_time.mean_seconds_per_point == pytest.approx(0.0)
    assert breakdown.counters is not None
    assert breakdown.counters.recurrence_momentum_scalar_values_per_point == 12
    assert breakdown.counters.recurrence_schedule_executions_per_call == 1
    assert breakdown.counters.recurrence_source_rows_per_call == 31
    assert breakdown.counters.recurrence_contribution_rows_per_call == 126
    assert breakdown.counters.recurrence_finalization_rows_per_call == 34
    assert breakdown.counters.recurrence_closure_rows_per_call == 12


def test_recurrence_profile_rejects_sub_attribution_larger_than_schedule() -> None:
    profile = {
        "execution_mode": "recurrence",
        "points": 1,
        "wall_time_s": 20.0e-6,
        "recurrence_schedule_time_s": 10.0e-6,
        "recurrence_source_kernel_time_s": 4.0e-6,
        "recurrence_contribution_kernel_time_s": 7.0e-6,
    }

    with pytest.raises(EvaluationError, match="schedule sub-attribution"):
        benchmark_module._native_profile_sample(profile, fallback_points=1)


def test_keyboard_interrupt_returns_statistics_for_complete_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntimeWithNativeWall(clock)
    events: list[object] = []

    def interrupt_after_two_samples(event: object) -> None:
        events.append(event)
        if (
            isinstance(event, ProgressUpdate)
            and event.task_id == "runtime-benchmark"
            and event.completed == 2
        ):
            raise KeyboardInterrupt

    result = BenchmarkBackend(
        BenchmarkConfig(
            target_runtime=0.1,
            batch_size=2,
            warmup_runs=1,
            minimum_samples=4,
        ),
        CallbackProgressSink(interrupt_after_two_samples),
    ).run(runtime, points=(((1.0, 0.0, 0.0, 1.0),),))

    assert result.interrupted is True
    assert result.sample_count == 2
    assert result.environment["interrupted"] is True
    assert result.environment["completed_sample_count"] == 2
    assert result.environment["completion_fraction"] == pytest.approx(0.5)
    assert result.uncertainty.standard_error >= 0.0
    assert events[-1] == ProgressEnd(
        "runtime-benchmark",
        success=False,
        message=("interrupted after 2/4 complete blocks; reporting partial statistics"),
    )


def test_benchmark_reduces_planned_blocks_for_slow_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", clock.perf_counter)
    runtime = _TimedRuntime(clock, duration=2.0)
    config = BenchmarkConfig(
        target_runtime=10.0,
        batch_size=1,
        warmup_runs=1,
        minimum_samples=5,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.sample_count == 5
    assert result.repetitions_per_sample == 1
    assert result.environment["elapsed_seconds"] == pytest.approx(10.0)
    assert runtime.calls == 6


def test_benchmark_uses_native_profile_for_evaluator_time() -> None:
    runtime = _RuntimeWithProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=1,
        minimum_samples=3,
        helicity_ids=("h0",),
    )
    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )
    assert result.sample_count == 3
    assert runtime.calls == 4
    assert runtime.profile_calls == 4
    assert result.wall_time_per_point >= 0.0
    assert result.evaluator_time_per_point == pytest.approx(2.0e-6)
    assert result.environment["wall_time_source"] == (
        "python_outer_perf_counter_wall_time"
    )
    assert result.environment["wall_time_sample_pass"] == (
        "time.perf_counter[measure_repetitions]"
    )
    assert (
        result.environment["evaluator_time_source"]
        == "runtime_profile_core_evaluator_call_time"
    )
    assert result.environment["evaluator_sample_count"] == 3
    assert result.environment["timing_sample_contract"] == (
        "separate_native_profile_diagnostic_v1"
    )
    assert result.environment["native_profile_repetitions_per_sample"] == 1
    assert result.environment["precision"] == 16
    assert result.environment["execution_mode"] == "compiled"
    assert result.environment["color_workload"] == ("all 1 generated physical LC flows")
    assert result.environment["helicity_workload"] == "selected 1/1 helicities: h0"
    assert result.evaluator_uncertainty is not None
    assert result.process_id == "test"
    assert result.process_expression == "d d~ > z"
    assert result.timing_breakdown is not None
    assert result.timing_breakdown.sample_count == 3
    assert result.timing_breakdown.wall_time is not None
    assert result.timing_breakdown.wall_time.mean_seconds_per_point == pytest.approx(
        3.0e-6
    )
    assert result.timing_breakdown.source_fill_time is not None
    assert result.timing_breakdown.source_fill_time.mean_seconds_per_point == (
        pytest.approx(0.25e-6)
    )
    assert result.timing_breakdown.stage_leaf_input_pack_time is not None
    assert (
        result.timing_breakdown.stage_leaf_input_pack_time.mean_seconds_per_point
        == pytest.approx(0.075e-6)
    )
    assert result.timing_breakdown.stage_backend_call_time is not None
    assert (
        result.timing_breakdown.stage_backend_call_time.mean_seconds_per_point
        == pytest.approx(0.35e-6)
    )
    assert result.timing_breakdown.amplitude_output_remap_time is not None
    assert (
        result.timing_breakdown.amplitude_output_remap_time.mean_seconds_per_point
        == pytest.approx(0.125e-6)
    )
    assert result.timing_breakdown.other_core_time is not None
    assert result.timing_breakdown.other_core_time.mean_seconds_per_point == (
        pytest.approx(0.0)
    )
    assert len(result.timing_breakdown.stages) == 2
    assert result.timing_breakdown.stages[1].leaf_input_pack_time is not None
    assert result.timing_breakdown.stages[
        1
    ].leaf_input_pack_time.mean_seconds_per_point == pytest.approx(0.05e-6)
    assert result.timing_breakdown.stages[1].backend_call_time is not None
    assert result.timing_breakdown.stages[
        1
    ].backend_call_time.mean_seconds_per_point == pytest.approx(0.25e-6)


def test_native_profile_reports_unaccounted_rusticol_core_time() -> None:
    profile = {
        "points": 2,
        "wall_time_s": 20.0e-6,
        "source_fill_time_s": 1.0e-6,
        "momentum_setup_time_s": 1.0e-6,
        "stage_input_pack_time_s": 2.0e-6,
        "stage_evaluator_call_time_s": 4.0e-6,
        "output_assign_time_s": 2.0e-6,
        "amplitude_input_pack_time_s": 1.0e-6,
        "amplitude_evaluator_call_time_s": 3.0e-6,
        "reduction_time_s": 2.0e-6,
    }

    sample = benchmark_module._native_profile_sample(profile, fallback_points=2)

    assert sample.other_core_time == pytest.approx(2.0e-6)


def test_timing_breakdown_preserves_legacy_positional_field_order() -> None:
    legacy_fields = (
        "sample_count",
        "execution_mode",
        "wall_time",
        "source_fill_time",
        "momentum_setup_time",
        "stage_input_pack_time",
        "stage_evaluator_call_time",
        "output_assign_time",
        "amplitude_input_pack_time",
        "amplitude_evaluator_call_time",
        "reduction_time",
        "other_core_time",
        "eager_execution_time",
        "eager_initialize_time",
        "eager_gather_time",
        "eager_kernel_call_time",
        "eager_invocation_scatter_time",
        "eager_finalization_time",
        "eager_scatter_finalization_time",
        "eager_closure_time",
        "eager_copy_out_time",
        "stages",
    )

    assert tuple(BenchmarkTimingBreakdown.__dataclass_fields__)[
        : len(legacy_fields)
    ] == (legacy_fields)
    fields = tuple(BenchmarkTimingBreakdown.__dataclass_fields__)
    assert fields.index("counters") == len(fields) - 8
    assert fields[fields.index("counters") - 8 : fields.index("counters")] == (
        "recurrence_momentum_fill_time",
        "recurrence_union_source_fill_time",
        "recurrence_schedule_time",
        "recurrence_source_kernel_time",
        "recurrence_contribution_kernel_time",
        "recurrence_finalization_time",
        "recurrence_closure_time",
        "recurrence_replay_output_mapping_time",
    )
    assert fields[-7:] == (
        "stage_leaf_input_pack_time",
        "stage_backend_call_time",
        "stage_evaluator_output_gather_time",
        "amplitude_leaf_input_pack_time",
        "amplitude_backend_call_time",
        "amplitude_evaluator_output_gather_time",
        "amplitude_output_remap_time",
    )


def test_repeated_profile_counters_normalize_per_point_and_per_call() -> None:
    points = 24
    repetitions = 3
    per_point_counts = {
        "native_input_component_count": 4,
        "native_input_pack_bytes": 32,
        "native_input_crossing_bytes": 32,
        "state_component_count": 101,
        "state_clear_component_count": 99,
        "source_component_count": 8,
        "momentum_component_count": 12,
        "model_parameter_component_count": 5,
        "stage_input_copy_component_count": 41,
        "stage_leaf_input_copy_component_count": 43,
        "stage_evaluator_output_gather_component_count": 47,
        "stage_output_assign_component_count": 53,
        "amplitude_input_copy_component_count": 59,
        "amplitude_leaf_input_copy_component_count": 61,
        "amplitude_evaluator_output_gather_component_count": 67,
        "amplitude_output_remap_component_count": 71,
        "reduction_input_component_count": 73,
        "selector_gather_point_count": 1,
        "selector_gather_bytes": 32,
        "selector_scatter_value_count": 2,
        "resolved_materialized_component_count": 79,
        "total_materialized_value_count": 1,
        "final_output_copy_value_count": 83,
    }
    per_call_counts = {
        "native_input_container_allocation_count": 10,
        "evaluator_backend_call_count": 11,
        "observed_scratch_reallocation_count": 0,
        "native_output_allocation_count": 1,
    }
    sample = benchmark_module._native_profile_sample(
        {
            "execution_mode": "compiled",
            "points": points,
            "wall_time_s": points * 4.0e-6,
            "stage_evaluator_call_time_s": points * 2.0e-6,
            "amplitude_evaluator_call_time_s": points * 0.5e-6,
            **{key: value * points for key, value in per_point_counts.items()},
            **{key: value * repetitions for key, value in per_call_counts.items()},
        },
        fallback_points=points,
        repetitions=repetitions,
    )

    counters = sample.counters
    assert counters is not None
    for raw_name, expected in per_point_counts.items():
        field_name = {
            "native_input_component_count": "native_input_components_per_point",
            "native_input_pack_bytes": "native_input_pack_bytes_per_point",
            "native_input_crossing_bytes": "native_input_crossing_bytes_per_point",
            "state_component_count": "state_components_per_point",
            "state_clear_component_count": "state_clear_components_per_point",
            "source_component_count": "source_components_per_point",
            "momentum_component_count": "momentum_components_per_point",
            "model_parameter_component_count": "model_parameter_components_per_point",
            "stage_input_copy_component_count": (
                "stage_input_copy_components_per_point"
            ),
            "stage_leaf_input_copy_component_count": (
                "stage_leaf_input_copy_components_per_point"
            ),
            "stage_evaluator_output_gather_component_count": (
                "stage_evaluator_output_gather_components_per_point"
            ),
            "stage_output_assign_component_count": (
                "stage_output_assign_components_per_point"
            ),
            "amplitude_input_copy_component_count": (
                "amplitude_input_copy_components_per_point"
            ),
            "amplitude_leaf_input_copy_component_count": (
                "amplitude_leaf_input_copy_components_per_point"
            ),
            "amplitude_evaluator_output_gather_component_count": (
                "amplitude_evaluator_output_gather_components_per_point"
            ),
            "amplitude_output_remap_component_count": (
                "amplitude_output_remap_components_per_point"
            ),
            "reduction_input_component_count": ("reduction_input_components_per_point"),
            "selector_gather_point_count": "selector_gather_points_per_point",
            "selector_gather_bytes": "selector_gather_bytes_per_point",
            "selector_scatter_value_count": "selector_scatter_values_per_point",
            "resolved_materialized_component_count": (
                "resolved_materialized_components_per_point"
            ),
            "total_materialized_value_count": ("total_materialized_values_per_point"),
            "final_output_copy_value_count": ("final_output_copy_values_per_point"),
        }[raw_name]
        assert getattr(counters, field_name) == pytest.approx(expected)
    assert counters.native_input_container_allocations_per_call == pytest.approx(10.0)
    assert counters.evaluator_backend_calls_per_call == pytest.approx(11.0)
    assert counters.observed_scratch_reallocations_per_call == pytest.approx(0.0)
    assert counters.native_output_allocations_per_call == pytest.approx(1.0)

    summary = benchmark_module._timing_breakdown((sample, sample)).counters
    assert summary is not None
    assert summary.sample_count == 2
    assert summary.stage_leaf_input_copy_components_per_point == pytest.approx(43.0)
    assert summary.evaluator_backend_calls_per_call == pytest.approx(11.0)


def test_profile_counter_contract_rejects_ambiguous_repeated_totals() -> None:
    with pytest.raises(EvaluationError, match="not divisible by repetitions"):
        benchmark_module._native_profile_sample(
            {
                "points": 5,
                "stage_evaluator_call_time_s": 1.0e-6,
                "amplitude_evaluator_call_time_s": 0.0,
            },
            fallback_points=5,
            repetitions=2,
        )

    with pytest.raises(EvaluationError, match="non-negative integer"):
        benchmark_module._native_profile_sample(
            {
                "points": 2,
                "stage_evaluator_call_time_s": 1.0e-6,
                "amplitude_evaluator_call_time_s": 0.0,
                "stage_input_copy_component_count": 1.5,
            },
            fallback_points=2,
        )


def test_compiled_direct_arena_counters_must_be_jointly_available() -> None:
    with pytest.raises(EvaluationError, match="must be jointly available"):
        benchmark_module._native_profile_sample(
            {
                "execution_mode": "compiled",
                "points": 1,
                "wall_time_s": 2.0e-6,
                "orchestration_time_s": 1.0e-6,
                "stage_evaluator_call_time_s": 0.0,
                "amplitude_evaluator_call_time_s": 0.0,
                "evaluator_backend_call_count": 1,
                "compiled_direct_arena_engine_count": 1,
            },
            fallback_points=1,
        )


@pytest.mark.parametrize(
    "boundary_counter",
    (
        "compiled_direct_arena_boundary_input_bytes",
        "compiled_direct_arena_boundary_current_output_bytes",
        "compiled_direct_arena_boundary_amplitude_output_bytes",
    ),
)
def test_compiled_direct_arena_rejects_boundary_traffic(
    boundary_counter: str,
) -> None:
    profile = {
        "execution_mode": "compiled",
        "points": 1,
        "wall_time_s": 2.0e-6,
        "orchestration_time_s": 1.0e-6,
        "stage_evaluator_call_time_s": 0.0,
        "amplitude_evaluator_call_time_s": 0.0,
        "evaluator_backend_call_count": 1,
        "compiled_direct_arena_engine_count": 1,
        "compiled_direct_arena_call_count": 1,
        "compiled_direct_arena_boundary_input_bytes": 0,
        "compiled_direct_arena_boundary_current_output_bytes": 0,
        "compiled_direct_arena_boundary_amplitude_output_bytes": 0,
    }
    profile[boundary_counter] = 16

    with pytest.raises(EvaluationError, match="forbidden boundary traffic"):
        benchmark_module._native_profile_sample(profile, fallback_points=1)


def test_compiled_direct_arena_calls_must_cover_backend_calls() -> None:
    with pytest.raises(EvaluationError, match="cover every evaluator backend call"):
        benchmark_module._native_profile_sample(
            {
                "execution_mode": "compiled",
                "points": 1,
                "wall_time_s": 2.0e-6,
                "orchestration_time_s": 1.0e-6,
                "stage_evaluator_call_time_s": 0.0,
                "amplitude_evaluator_call_time_s": 0.0,
                "evaluator_backend_call_count": 2,
                "compiled_direct_arena_engine_count": 1,
                "compiled_direct_arena_call_count": 1,
                "compiled_direct_arena_boundary_input_bytes": 0,
                "compiled_direct_arena_boundary_current_output_bytes": 0,
                "compiled_direct_arena_boundary_amplitude_output_bytes": 0,
            },
            fallback_points=1,
        )


def test_compiled_direct_arena_requires_fused_orchestration_timing() -> None:
    with pytest.raises(EvaluationError, match="orchestration timing is unavailable"):
        benchmark_module._native_profile_sample(
            {
                "execution_mode": "compiled",
                "points": 1,
                "wall_time_s": 2.0e-6,
                "stage_evaluator_call_time_s": 0.0,
                "amplitude_evaluator_call_time_s": 0.0,
                "evaluator_backend_call_count": 1,
                "compiled_direct_arena_engine_count": 1,
                "compiled_direct_arena_call_count": 1,
                "compiled_direct_arena_boundary_input_bytes": 0,
                "compiled_direct_arena_boundary_current_output_bytes": 0,
                "compiled_direct_arena_boundary_amplitude_output_bytes": 0,
            },
            fallback_points=1,
        )


def test_compiled_direct_arena_public_counters_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        BenchmarkProfileCounters(
            sample_count=1,
            compiled_direct_arena_calls_per_call=-1.0,
        )


def test_native_profile_preserves_legacy_momentum_setup_aggregate() -> None:
    sample = benchmark_module._native_profile_sample(
        {
            "execution_mode": "compiled",
            "points": 1,
            "wall_time_s": 10.0e-6,
            "momentum_input_setup_time_s": 1.0e-6,
            "momentum_setup_time_s": 3.0e-6,
            "model_parameter_setup_time_s": 2.0e-6,
            "stage_evaluator_call_time_s": 1.0e-6,
            "amplitude_evaluator_call_time_s": 1.0e-6,
        },
        fallback_points=1,
    )

    assert sample.momentum_input_setup_time == pytest.approx(1.0e-6)
    assert sample.momentum_setup_time == pytest.approx(3.0e-6)
    assert sample.model_parameter_setup_time == pytest.approx(2.0e-6)
    assert sample.other_core_time == pytest.approx(5.0e-6)


def test_native_profile_accounts_selector_phases_exclusively() -> None:
    profile = {
        "execution_mode": "compiled",
        "points": 2,
        "wall_time_s": 20.0e-6,
        "stage_evaluator_call_time_s": 8.0e-6,
        "amplitude_evaluator_call_time_s": 2.0e-6,
        "selector_planner_time_s": 1.0e-6,
        "selector_gather_time_s": 3.0e-6,
        "selector_scatter_time_s": 2.0e-6,
    }

    sample = benchmark_module._native_profile_sample(profile, fallback_points=2)

    assert sample.selector_planner_time == pytest.approx(0.5e-6)
    assert sample.selector_gather_time == pytest.approx(1.5e-6)
    assert sample.selector_scatter_time == pytest.approx(1.0e-6)
    assert sample.other_core_time == pytest.approx(2.0e-6)


def test_native_eager_profile_rejects_overlapping_top_level_phases() -> None:
    profile = {
        "execution_mode": "eager",
        "points": 4,
        "wall_time_s": 10.0e-6,
        "source_fill_time_s": 8.0e-6,
        "momentum_setup_time_s": 1.0e-6,
        "stage_evaluator_call_time_s": 4.0e-6,
        "amplitude_evaluator_call_time_s": 0.0,
        "stage_input_pack_by_stage_time_s": [],
        "stage_evaluator_call_by_stage_time_s": [],
        "stage_output_assign_by_stage_time_s": [],
    }

    with pytest.raises(EvaluationError, match="exclusive top-level phases"):
        benchmark_module._native_profile_sample(profile, fallback_points=4)


def test_native_compiled_profile_rejects_overlapping_top_level_phases() -> None:
    profile = {
        "execution_mode": "compiled",
        "points": 1,
        "wall_time_s": 10.0e-6,
        "stage_input_pack_time_s": 4.0e-6,
        "stage_evaluator_call_time_s": 7.0e-6,
        "output_assign_time_s": 0.0,
        "amplitude_input_pack_time_s": 0.0,
        "amplitude_evaluator_call_time_s": 0.0,
        "reduction_time_s": 0.0,
    }

    with pytest.raises(EvaluationError, match="exclusive top-level phases"):
        benchmark_module._native_profile_sample(profile, fallback_points=1)


def test_native_eager_profile_rejects_children_exceeding_inclusive_execution() -> None:
    profile = {
        "execution_mode": "eager",
        "points": 1,
        "wall_time_s": 20.0e-6,
        "source_fill_time_s": 1.0e-6,
        "momentum_setup_time_s": 1.0e-6,
        "stage_evaluator_call_time_s": 10.0e-6,
        "amplitude_evaluator_call_time_s": 0.0,
        "eager_kernel_call_time_s": 11.0e-6,
        "stage_input_pack_by_stage_time_s": [],
        "stage_evaluator_call_by_stage_time_s": [],
        "stage_output_assign_by_stage_time_s": [],
    }

    with pytest.raises(EvaluationError, match="exclusive execution phases"):
        benchmark_module._native_profile_sample(profile, fallback_points=1)


def test_native_eager_shared_samples_partition_wall_time() -> None:
    samples = []
    for source_fill in (1.0e-6, 1.5e-6, 2.0e-6):
        sample = benchmark_module._native_profile_sample(
            {
                "execution_mode": "eager",
                "points": 1,
                "wall_time_s": 12.0e-6,
                "source_fill_time_s": source_fill,
                "momentum_setup_time_s": 1.0e-6,
                "stage_evaluator_call_time_s": 8.0e-6,
                "amplitude_evaluator_call_time_s": 0.0,
                "stage_input_pack_by_stage_time_s": [],
                "stage_evaluator_call_by_stage_time_s": [],
                "stage_output_assign_by_stage_time_s": [],
            },
            fallback_points=1,
        )
        assert sample.other_core_time is not None
        assert sample.source_fill_time is not None
        assert sample.momentum_setup_time is not None
        assert (
            sample.source_fill_time
            + sample.momentum_setup_time
            + sample.stage_evaluator_call_time
            + sample.other_core_time
        ) == pytest.approx(sample.wall_time)
        samples.append(sample)

    breakdown = benchmark_module._timing_breakdown(samples)
    assert breakdown.source_fill_time is not None
    assert breakdown.source_fill_time.uncertainty.standard_deviation > 0.0


def test_benchmark_falls_back_to_valid_per_stage_native_timings() -> None:
    runtime = _RuntimeWithPerStageProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=1,
        minimum_samples=3,
        helicity_ids=("h0",),
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.evaluator_time_per_point == pytest.approx(2.0e-6)


def test_eager_profile_uses_nonzero_aggregate_when_stage_vectors_are_empty() -> None:
    runtime = _RuntimeWithEagerProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=1,
        minimum_samples=3,
        helicity_ids=("h0",),
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.evaluator_time_per_point == pytest.approx(20.0e-6)
    assert result.timing_breakdown is not None
    assert result.timing_breakdown.execution_mode == "eager"
    assert result.timing_breakdown.stage_evaluator_call_time is None
    aggregate = result.timing_breakdown.eager_execution_time
    assert aggregate is not None
    assert aggregate.mean_seconds_per_point == pytest.approx(20.0e-6)
    assert result.timing_breakdown.stage_input_pack_time is None
    assert result.timing_breakdown.output_assign_time is None
    assert result.timing_breakdown.amplitude_input_pack_time is None
    assert result.timing_breakdown.amplitude_evaluator_call_time is None
    assert result.timing_breakdown.reduction_time is None
    assert result.timing_breakdown.stages == ()


def test_eager_profile_preserves_detailed_native_phase_timings() -> None:
    runtime = _RuntimeWithDetailedEagerProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=1,
        minimum_samples=3,
        helicity_ids=("h0",),
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    breakdown = result.timing_breakdown
    assert breakdown is not None
    assert breakdown.execution_mode == "eager"
    assert breakdown.eager_initialize_time is not None
    assert breakdown.eager_initialize_time.mean_seconds_per_point == pytest.approx(
        0.5e-6
    )
    assert breakdown.eager_gather_time is not None
    assert breakdown.eager_gather_time.mean_seconds_per_point == pytest.approx(1.0e-6)
    assert breakdown.eager_kernel_call_time is not None
    assert breakdown.eager_kernel_call_time.mean_seconds_per_point == pytest.approx(
        6.0e-6
    )
    assert breakdown.eager_invocation_scatter_time is not None
    invocation_scatter = breakdown.eager_invocation_scatter_time
    assert invocation_scatter.mean_seconds_per_point == pytest.approx(3.0e-6)
    assert breakdown.eager_finalization_time is not None
    assert breakdown.eager_finalization_time.mean_seconds_per_point == pytest.approx(
        5.0e-6
    )
    assert breakdown.eager_scatter_finalization_time is not None
    assert breakdown.eager_closure_time is not None
    assert breakdown.eager_copy_out_time is not None
    assert breakdown.eager_copy_out_time.mean_seconds_per_point == pytest.approx(1.5e-6)


def test_empty_stage_vectors_do_not_relabel_compiled_amplitude_only_profile() -> None:
    runtime = _RuntimeWithAmplitudeOnlyCompiledProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=4,
        warmup_runs=1,
        minimum_samples=3,
        helicity_ids=("h0",),
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert result.timing_breakdown is not None
    assert result.timing_breakdown.execution_mode == "compiled"
    assert result.evaluator_time_per_point == pytest.approx(1.5e-6)


def test_non_f64_benchmark_forwards_precision_without_native_profile() -> None:
    runtime = _RuntimeWithProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=2,
        precision=80,
        warmup_runs=1,
        minimum_samples=2,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert runtime.profile_calls == 0
    assert runtime.last_options is not None
    assert runtime.last_options["precision"] == 80
    assert result.environment["precision"] == 80
    assert result.environment["native_profile_unavailable_reason"] == (
        "non_f64_precision"
    )
    assert result.evaluator_time_per_point is None
    assert result.environment["evaluator_time_status"] == "unavailable"


def test_benchmark_falls_back_when_native_profile_is_unavailable() -> None:
    runtime = _RuntimeWithUnavailableProfile()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=2,
    )

    result = BenchmarkBackend(config, None).run(
        runtime,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert runtime.calls == 3
    assert result.environment["wall_time_source"] == (
        "python_outer_perf_counter_wall_time"
    )
    assert result.evaluator_time_per_point is None
    assert result.environment["evaluator_time_status"] == "unavailable"


def test_benchmark_uses_runtime_validation_point_when_points_are_omitted() -> None:
    runtime = _RuntimeWithValidation()
    config = BenchmarkConfig(
        target_runtime=1.0e-12,
        batch_size=2,
        warmup_runs=1,
        minimum_samples=2,
    )
    result = BenchmarkBackend(config, None).run(runtime)
    assert result.sample_count == 2
    assert runtime.calls == 3


def test_benchmark_without_points_or_artifact_fixture_is_explicit() -> None:
    with pytest.raises(EvaluationError, match="deterministic validation point"):
        BenchmarkBackend(BenchmarkConfig(), None).run(_Runtime())
