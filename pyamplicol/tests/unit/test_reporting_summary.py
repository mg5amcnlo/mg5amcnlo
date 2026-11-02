# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import itertools
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.api import (
    BenchmarkComponentTiming,
    BenchmarkProfileCounters,
    BenchmarkResult,
    BenchmarkStageTiming,
    BenchmarkStatistics,
    BenchmarkTimingBreakdown,
    GenerationResult,
    ProcessRequest,
    ProcessSet,
    ResolvedEvaluation,
)
from pyamplicol.artifacts import (
    ArtifactAliasInspection,
    ArtifactDependencyInspection,
    ArtifactInspection,
    ArtifactProcessInspection,
)
from pyamplicol.artifacts.inspection import (
    ArtifactProcessGroupInspection,
    ArtifactProcessGroupMemberInspection,
    ArtifactProcessGroupSelection,
    ArtifactProcessGroupsInspection,
)
from pyamplicol.cli.main import write_result
from pyamplicol.cli.utilities import ExampleEntry, UtilityResult
from pyamplicol.config import BenchmarkConfig
from pyamplicol.diagnostics import DiagnosticCheck, DiagnosticReport
from pyamplicol.reporting import render_summary


@dataclass(frozen=True)
class _Result:
    status: str
    generated_processes: int
    adjustments: tuple[str, ...]


def test_human_structured_results_use_aligned_prettytable() -> None:
    rendered = render_summary(_Result("complete", 2, ()), color=False)
    assert rendered is not None
    assert set(rendered.splitlines()[-1]) <= {"+", "-"}
    assert "field" in rendered
    assert "generated processes" in rendered
    assert "complete" in rendered


def test_examples_and_diagnostics_have_dedicated_colored_record_tables() -> None:
    examples = (
        ExampleEntry("otf_pp_zjj", "generate", "OTF p p > Z j j walkthrough"),
        ExampleEntry("benchmark", "benchmark", "Profile one process output"),
    )
    rendered_examples = render_summary(examples, color=True)
    assert rendered_examples is not None
    assert "Packaged Examples" in rendered_examples
    assert "name" in rendered_examples
    assert "action" in rendered_examples
    assert "description" in rendered_examples
    assert "\x1b[" in rendered_examples
    assert not rendered_examples.lstrip().startswith("[")

    diagnostics = DiagnosticReport(
        "0.1.4",
        (
            DiagnosticCheck("native-sdk", "pass", "ABI 1"),
            DiagnosticCheck("symbolica", "warning", "restricted single-core"),
        ),
    )
    rendered_diagnostics = render_summary(diagnostics, color=True)
    assert rendered_diagnostics is not None
    assert "pyAmpliCol Diagnostics" in rendered_diagnostics
    assert "native-sdk" in rendered_diagnostics
    assert "\x1b[32mpass" in rendered_diagnostics
    assert "\x1b[33mwarning" in rendered_diagnostics


def test_human_renderer_never_embeds_or_falls_back_to_json() -> None:
    result = {
        "status": "complete",
        "nested": {"worker_count": 4, "backend": "jit"},
        "entries": (
            {"name": "first", "process": "d d~ > z"},
            {"name": "second", "process": "u d~ > w+"},
        ),
    }
    stream = io.StringIO()
    write_result(result, format="human", stream=stream, color=False)
    output = stream.getvalue()
    assert "worker count" in output
    assert "Entries" in output
    assert '"status"' not in output
    assert "[{" not in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)

    operation = UtilityResult("examples copy", "complete", Path("examples"))
    rendered = render_summary(operation, color=False)
    assert rendered is not None
    assert "Operation Complete" in rendered
    assert "destination" in rendered


def test_generation_mode_is_labelled_as_existing_output_policy() -> None:
    rendered = render_summary(
        GenerationResult(
            output=Path("artifact"),
            processes=ProcessSet((ProcessRequest.parse("d d~ > z"),)),
            mode="error",
        ),
        color=True,
    )

    assert rendered is not None
    assert "existing-output policy" in rendered
    assert "error" in rendered
    assert "\x1b[36mexisting-output policy" in rendered
    assert "\x1b[33merror" in rendered
    assert "\x1b[31merror" not in rendered


def test_generation_summary_compacts_file_inventory(
    monkeypatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "artifact.json").write_bytes(b"x" * 24)
    manifest = SimpleNamespace(
        payloads=(
            SimpleNamespace(size_bytes=1000),
            SimpleNamespace(size_bytes=500),
        )
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.load_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    result = GenerationResult(
        output=artifact,
        processes=ProcessSet((ProcessRequest.parse("d d~ > z"),)),
        mode="error",
        files=(artifact / "one", artifact / "two", artifact / "artifact.json"),
    )

    rendered = render_summary(result, color=False)

    assert rendered is not None
    assert "3 files; 1.49 KiB total" in rendered
    assert str(result.files) not in rendered


def test_json_result_never_contains_table_or_color_sequences() -> None:
    stream = io.StringIO()
    write_result(_Result("complete", 2, ()), format="json", stream=stream, color=True)
    assert json.loads(stream.getvalue()) == {
        "adjustments": [],
        "generated_processes": 2,
        "status": "complete",
    }
    assert "\x1b[" not in stream.getvalue()


def test_total_evaluation_uses_a_colored_point_table_without_changing_json() -> None:
    result = (complex(1.25, 0.0), Decimal("2.5"))

    rendered = render_summary(result, color=True)

    assert rendered is not None
    assert "Summed Evaluation" in rendered
    assert "matrix element" in rendered
    assert "1.250000000000000e+00" in rendered
    assert "2.500000000000000e+00" in rendered
    assert "\x1b[" in rendered
    assert set(rendered.splitlines()[-1]) <= {"+", "-"}
    stream = io.StringIO()
    write_result(result, format="json", stream=stream, color=True)
    assert json.loads(stream.getvalue()) == [
        {"imag": 0.0, "real": 1.25},
        "2.5",
    ]
    assert "\x1b[" not in stream.getvalue()


def test_resolved_evaluation_uses_colored_component_and_sum_tables() -> None:
    result = ResolvedEvaluation(
        values=(
            (
                (complex(1.0, 2.0), complex(3.0, 0.0)),
                (complex(4.0, 0.0), complex(5.0, 0.0)),
            ),
        ),
        helicity_ids=("h:first", "h:second"),
        color_ids=("flow:first", "flow:second"),
    )

    rendered = render_summary(result, color=True)

    assert rendered is not None
    assert "Resolved Evaluation (LC)" in rendered
    assert "h:first" in rendered
    assert "flow:second" in rendered
    assert "1.000000000000000e+00+2.000000000000000e+00j" in rendered
    assert "Resolved Sum" in rendered
    assert "1.300000000000000e+01+2.000000000000000e+00j" in rendered
    assert "\x1b[" in rendered
    component_table, total_table = rendered.split("\n\n", maxsplit=1)
    for table in (component_table, total_table):
        assert set(table.splitlines()[-1]) <= {"+", "-"}
    stream = io.StringIO()
    write_result(result, format="json", stream=stream, color=True)
    payload = json.loads(stream.getvalue())
    assert payload["values"][0][0][0] == {"imag": 2.0, "real": 1.0}
    assert payload["helicity_ids"] == ["h:first", "h:second"]
    assert payload["color_ids"] == ["flow:first", "flow:second"]
    assert "\x1b[" not in stream.getvalue()


def test_evaluation_values_use_scientific_notation_except_exact_zero() -> None:
    rendered = render_summary((0.0002519743243773565, 0.0), color=False)

    assert rendered is not None
    assert "2.519743243773565e-04" in rendered
    zero_row = next(line for line in rendered.splitlines() if "|     1 |" in line)
    assert zero_row.split("|")[-2].strip() == "0"


def _benchmark_result() -> BenchmarkResult:
    config = BenchmarkConfig(target_runtime=1.0, batch_size=32, minimum_samples=8)
    wall_uncertainty = BenchmarkStatistics(0.2e-6, 0.05e-6, 0.02)
    evaluator_uncertainty = BenchmarkStatistics(0.1e-6, 0.025e-6, 0.0125)
    component = BenchmarkComponentTiming(0.5e-6, evaluator_uncertainty, 8)
    wall_component = BenchmarkComponentTiming(2.5e-6, wall_uncertainty, 8)
    evaluator_component = BenchmarkComponentTiming(2.0e-6, evaluator_uncertainty, 8)
    counters = BenchmarkProfileCounters(
        sample_count=8,
        native_input_components_per_point=20.0,
        native_input_pack_bytes_per_point=160.0,
        state_components_per_point=400.0,
        stage_input_copy_components_per_point=120.0,
        stage_leaf_input_copy_components_per_point=240.0,
        stage_evaluator_output_gather_components_per_point=60.0,
        stage_output_assign_components_per_point=80.0,
        amplitude_input_copy_components_per_point=40.0,
        amplitude_leaf_input_copy_components_per_point=42.0,
        amplitude_evaluator_output_gather_components_per_point=10.0,
        amplitude_output_remap_components_per_point=8.0,
        evaluator_backend_calls_per_call=12.0,
        compiled_direct_arena_engines_per_call=2.0,
        compiled_direct_arena_calls_per_call=12.0,
        compiled_direct_arena_boundary_input_bytes_per_call=0.0,
        compiled_direct_arena_boundary_current_output_bytes_per_call=0.0,
        compiled_direct_arena_boundary_amplitude_output_bytes_per_call=0.0,
        recurrence_schedule_executions_per_call=1.0,
        recurrence_contribution_calls_per_call=7.0,
        recurrence_contribution_rows_per_call=126.0,
        reduction_input_components_per_point=16.0,
        total_materialized_values_per_point=1.0,
        observed_scratch_reallocations_per_call=0.0,
        native_output_allocations_per_call=1.0,
    )
    breakdown = BenchmarkTimingBreakdown(
        sample_count=8,
        wall_time=wall_component,
        source_fill_time=component,
        momentum_setup_time=component,
        stage_input_pack_time=component,
        stage_evaluator_call_time=evaluator_component,
        output_assign_time=component,
        amplitude_input_pack_time=component,
        amplitude_evaluator_call_time=None,
        stage_leaf_input_pack_time=component,
        stage_backend_call_time=component,
        stage_evaluator_output_gather_time=component,
        amplitude_leaf_input_pack_time=component,
        amplitude_backend_call_time=component,
        amplitude_evaluator_output_gather_time=component,
        amplitude_output_remap_time=component,
        reduction_time=component,
        other_core_time=component,
        stages=(
            BenchmarkStageTiming(
                1,
                component,
                component,
                component,
                leaf_input_pack_time=component,
                backend_call_time=component,
                evaluator_output_gather_time=component,
            ),
        ),
        counters=counters,
    )
    return BenchmarkResult(
        requested_config=config,
        effective_config=config,
        sample_count=8,
        wall_time_per_point=2.5e-6,
        evaluator_time_per_point=2.0e-6,
        uncertainty=wall_uncertainty,
        environment={
            "target": "/tmp/artifact",
            "elapsed_seconds": 1.01,
            "platform": "test-platform",
            "wall_time_source": "python_outer_perf_counter_wall_time",
            "wall_time_sample_pass": "time.perf_counter[measure_repetitions]",
            "evaluator_total_time_source": (
                "runtime._benchmark_f64_wall_time.accumulated"
            ),
            "evaluator_time_source": "runtime_profile_core_evaluator_call_time",
            "evaluator_time_sample_pass": "runtime.profile_repeated",
            "execution_mode": "compiled",
            "color_workload": "all 1 generated physical LC flows",
            "helicity_workload": "all 24 generated helicity configurations",
            "timing_sample_contract": (
                "paired_unprofiled_headline_profiled_attribution_v1"
            ),
            "evaluator_sample_count": 8,
            "native_profile_repetitions_per_sample": 50,
            "native_profile_points_per_sample": 1600,
        },
        repetitions_per_sample=50,
        evaluator_uncertainty=evaluator_uncertainty,
        evaluator_total_time_per_point=2.25e-6,
        process_id="ddbar_zg",
        process_expression="d d~ > z g",
        timing_breakdown=breakdown,
    )


def test_benchmark_result_uses_clear_runtime_profile_table() -> None:
    rendered = render_summary(_benchmark_result(), color=False)

    assert rendered is not None
    assert "Runtime Profile" in rendered
    assert "ddbar_zg (d d~ > z g)" in rendered
    assert "execution mode" in rendered
    assert "compiled" in rendered
    assert "all 1 generated physical LC flows" in rendered
    assert "all 24 generated helicity configurations" in rendered
    assert "outer wall" in rendered
    assert "2.5 +/- 0.05 us/point (standard error)" in rendered
    assert "evaluator total" in rendered
    assert "2.25 us/point" in rendered
    assert "evaluator core" in rendered
    assert "8 blocks x 50 repetitions x 32 points" in rendered
    assert "timed points" in rendered
    assert "Rusticol Timing Breakdown" in rendered
    assert "Source fill" in rendered
    assert "Other Rusticol core" in rendered
    assert "Rusticol Stage Detail" in rendered
    assert "Stage evaluator envelope (top-level)" in rendered
    assert "Stage leaf input pack (attribution)" in rendered
    assert "Amplitude output remap (attribution)" in rendered
    assert "Rusticol Stage Internal Attribution (do not add to top-level)" in rendered
    assert "evaluator call" in rendered
    assert "Native Work Counters (mean across 8 profiled blocks)" in rendered
    assert "schedule executions" in rendered
    assert "contribution rows" in rendered
    assert "stage leaf copy" in rendered
    assert "240 components" in rendered
    assert "per runtime call" in rendered
    assert "compiled Direct-Arena" in rendered
    assert "Arena engines" in rendered
    assert "2 engines" in rendered
    assert "Arena calls" in rendered
    assert "boundary current output" in rendered
    assert "boundary amplitude output" in rendered
    assert "observed scratch reallocations" in rendered
    assert "paired passes: unprofiled headline" in rendered
    assert "profiled evaluator core attribution" in rendered
    assert "Profile wall (paired profiled pass)" in rendered
    assert "Rusticol Timing Breakdown (paired profiled attribution)" in rendered


def test_compiled_direct_arena_labels_the_inclusive_runtime_envelope() -> None:
    result = _benchmark_result()
    rendered = render_summary(
        replace(
            result,
            environment={
                **result.environment,
                "compiled_direct_arena_active": True,
                "evaluator_time_source": (
                    "runtime_profile_core_compiled_direct_arena_orchestration_time"
                ),
            },
        ),
        color=False,
    )

    assert rendered is not None
    assert rendered.count("Direct-Arena runtime envelope") == 2
    assert "evaluator core" not in rendered
    assert "Runtime orchestration" not in rendered


def test_timing_breakdown_omits_exact_zero_component_rows() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None
    exact_zero = BenchmarkComponentTiming(
        0.0,
        BenchmarkStatistics(0.0, 0.0, 0.0),
        breakdown.sample_count,
    )
    uncertain_zero = BenchmarkComponentTiming(
        0.0,
        BenchmarkStatistics(1.0e-9, 0.5e-9, 1.0),
        breakdown.sample_count,
    )
    zero_stage = BenchmarkStageTiming(
        2,
        exact_zero,
        exact_zero,
        exact_zero,
        leaf_input_pack_time=exact_zero,
        backend_call_time=exact_zero,
        evaluator_output_gather_time=exact_zero,
    )
    partially_unavailable_stage = BenchmarkStageTiming(
        3,
        None,
        exact_zero,
        None,
        backend_call_time=uncertain_zero,
    )

    rendered = render_summary(
        replace(
            result,
            timing_breakdown=replace(
                breakdown,
                state_clear_time=exact_zero,
                state_prepare_time=uncertain_zero,
                selector_planner_time=None,
                stages=(
                    *breakdown.stages,
                    zero_stage,
                    partially_unavailable_stage,
                ),
            ),
        ),
        color=False,
    )

    assert rendered is not None
    assert "| State clear" not in rendered
    assert "| State prepare" in rendered
    assert "| Selector planning" in rendered
    assert "|     2 |" not in rendered
    assert "|     3 |" in rendered
    assert "| Other Rusticol core" in rendered


def test_recurrence_profile_reports_paired_schedule_and_nested_attribution() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None
    component = breakdown.source_fill_time
    schedule = breakdown.stage_evaluator_call_time
    recurrence_breakdown = replace(
        breakdown,
        execution_mode="recurrence",
        stage_input_pack_time=None,
        stage_evaluator_call_time=None,
        output_assign_time=None,
        amplitude_input_pack_time=None,
        amplitude_evaluator_call_time=None,
        reduction_time=component,
        stages=(),
        recurrence_momentum_fill_time=component,
        recurrence_union_source_fill_time=component,
        recurrence_schedule_time=schedule,
        recurrence_source_kernel_time=component,
        recurrence_contribution_kernel_time=component,
        recurrence_finalization_time=component,
        recurrence_closure_time=component,
        recurrence_replay_output_mapping_time=component,
    )
    rendered = render_summary(
        replace(
            result,
            environment={
                **result.environment,
                "execution_mode": "recurrence",
                "evaluator_time_source": (
                    "runtime_profile_core_recurrence_schedule_time"
                ),
            },
            timing_breakdown=recurrence_breakdown,
        ),
        color=False,
    )

    assert rendered is not None
    assert "recurrence core" in rendered
    assert "evaluator core" not in rendered
    assert (
        "Rusticol Recurrence Timing Breakdown (paired profiled attribution)" in rendered
    )
    assert "Profile wall (paired profiled pass)" in rendered
    assert "Recurrence schedule (inclusive)" in rendered
    assert "Contribution kernels (schedule attribution; do not add)" in rendered
    assert "Current finalization (schedule attribution; do not add)" in rendered
    assert "Replay output mapping (exclusive)" in rendered
    assert "runtime_profile_core_recurrence_schedule_time" in rendered


def test_on_the_fly_profile_names_its_recurrence_schedule_attribution() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None
    component = breakdown.source_fill_time
    schedule = breakdown.stage_evaluator_call_time
    on_the_fly_breakdown = replace(
        breakdown,
        execution_mode="on-the-fly",
        stage_input_pack_time=None,
        stage_evaluator_call_time=None,
        output_assign_time=None,
        amplitude_input_pack_time=None,
        amplitude_evaluator_call_time=None,
        reduction_time=component,
        stages=(),
        recurrence_momentum_fill_time=component,
        recurrence_schedule_time=schedule,
        recurrence_source_kernel_time=component,
        recurrence_contribution_kernel_time=component,
        recurrence_finalization_time=component,
        recurrence_closure_time=component,
    )
    rendered = render_summary(
        replace(
            result,
            environment={
                **result.environment,
                "execution_mode": "on-the-fly",
                "evaluator_time_source": (
                    "runtime_profile_core_recurrence_schedule_time"
                ),
            },
            timing_breakdown=on_the_fly_breakdown,
        ),
        color=False,
    )

    assert rendered is not None
    assert "OTF recurrence-schedule core" in rendered
    assert "evaluator core" not in rendered
    assert (
        "Rusticol OTF Recurrence-Schedule Timing Breakdown "
        "(paired profiled attribution)" in rendered
    )
    assert "OTF recurrence schedule (inclusive)" in rendered
    assert "Contribution kernels (schedule attribution; do not add)" in rendered


def test_runtime_profile_renders_three_independent_clocks_with_full_precision() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None
    zero_uncertainty = BenchmarkStatistics(0.0, 0.0, 0.0)
    rendered = render_summary(
        replace(
            result,
            wall_time_per_point=218.105e-6,
            evaluator_total_time_per_point=217.812e-6,
            evaluator_time_per_point=10.1234e-6,
            uncertainty=zero_uncertainty,
            evaluator_uncertainty=zero_uncertainty,
            environment={
                **result.environment,
                "execution_mode": "recurrence",
                "evaluator_time_source": (
                    "runtime_profile_core_recurrence_schedule_time"
                ),
            },
            timing_breakdown=replace(breakdown, execution_mode="recurrence"),
        ),
        color=False,
    )

    assert rendered is not None
    lines = rendered.splitlines()
    assert any(
        "outer wall" in line and "218.105 +/- 0 us/point" in line for line in lines
    )
    assert any(
        "evaluator total" in line and "217.812 us/point" in line for line in lines
    )
    assert any(
        "recurrence core" in line and "10.1234 +/- 0 us/point" in line for line in lines
    )
    assert "evaluator total=runtime._benchmark_f64_wall_time.accumulated" in rendered
    assert "runtime_profile_core_recurrence_schedule_time" in rendered


def test_runtime_profile_marks_missing_total_and_core_unavailable() -> None:
    result = _benchmark_result()
    rendered = render_summary(
        replace(
            result,
            evaluator_total_time_per_point=None,
            evaluator_time_per_point=None,
            evaluator_uncertainty=None,
            environment={
                **result.environment,
                "execution_mode": "recurrence",
                "evaluator_time_source": "unavailable",
                "evaluator_total_time_source": "unavailable",
            },
            timing_breakdown=None,
        ),
        color=False,
    )

    assert rendered is not None
    lines = rendered.splitlines()
    assert any("evaluator total" in line and "N/A" in line for line in lines)
    assert any("recurrence core" in line and "N/A" in line for line in lines)


def test_separate_breakdown_samples_are_labeled_explicitly() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None
    environment = {
        **result.environment,
        "timing_sample_contract": "separate_native_profile_diagnostic_v1",
        "wall_time_sample_pass": "time.perf_counter[measure_repetitions]",
        "evaluator_time_sample_pass": "runtime.profile",
        "evaluator_sample_count": 3,
        "native_profile_repetitions_per_sample": 1,
        "native_profile_points_per_sample": 32,
        "evaluator_elapsed_seconds": 0.25,
    }
    rendered = render_summary(
        replace(
            result,
            environment=environment,
            timing_breakdown=replace(breakdown, sample_count=3),
        ),
        color=False,
    )

    assert rendered is not None
    assert "Rusticol Timing Breakdown (separate diagnostic samples)" in rendered
    assert "evaluator core" in rendered
    assert "separate passes" in rendered
    assert (
        "headline outer wall and evaluator total 8 blocks x 50 repetitions x 32 points"
        in rendered
    )
    assert "evaluator core and breakdown 3" in rendered
    assert "diagnostic blocks x 1 repetitions" in rendered
    assert "Profile wall (diagnostic pass)" in rendered


def test_separate_evaluator_uncertainty_uses_diagnostic_sample_count() -> None:
    result = _benchmark_result()
    environment = {
        **result.environment,
        "timing_sample_contract": "separate_native_profile_diagnostic_v1",
        "evaluator_sample_count": 1,
    }

    rendered = render_summary(replace(result, environment=environment), color=False)

    assert rendered is not None
    assert "evaluator core" in rendered
    assert "uncertainty needs at least 2 blocks" in rendered


def test_benchmark_profile_table_color_is_optional() -> None:
    rendered = render_summary(_benchmark_result(), color=True)

    assert rendered is not None
    assert "\x1b[" in rendered
    tables = tuple(
        section for section in rendered.split("\n\n") if section.startswith("+")
    )
    assert tables
    assert all(set(table.splitlines()[-1]) <= {"+", "-"} for table in tables)


def test_benchmark_profile_counters_remain_machine_readable() -> None:
    stream = io.StringIO()

    write_result(
        _benchmark_result(),
        format="json",
        stream=stream,
        color=True,
    )

    payload = json.loads(stream.getvalue())
    counters = payload["timing_breakdown"]["counters"]
    assert counters["normalization"] == ("mean_per_profiled_point_or_runtime_call_v1")
    assert counters["stage_leaf_input_copy_components_per_point"] == 240.0
    assert counters["evaluator_backend_calls_per_call"] == 12.0
    assert counters["compiled_direct_arena_engines_per_call"] == 2.0
    assert counters["compiled_direct_arena_calls_per_call"] == 12.0
    assert counters["compiled_direct_arena_boundary_input_bytes_per_call"] == 0.0
    assert (
        counters["compiled_direct_arena_boundary_current_output_bytes_per_call"] == 0.0
    )
    assert (
        counters["compiled_direct_arena_boundary_amplitude_output_bytes_per_call"]
        == 0.0
    )
    assert (
        payload["timing_breakdown"]["stage_backend_call_time"]["mean_seconds_per_point"]
        == 0.5e-6
    )
    assert (
        payload["timing_breakdown"]["amplitude_output_remap_time"][
            "mean_seconds_per_point"
        ]
        == 0.5e-6
    )
    assert "\x1b[" not in stream.getvalue()


def test_eager_benchmark_uses_eager_phase_labels() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None
    component = breakdown.stage_evaluator_call_time
    assert component is not None
    eager_breakdown = replace(
        breakdown,
        execution_mode="eager",
        stage_evaluator_call_time=None,
        eager_execution_time=component,
        eager_initialize_time=component,
        eager_gather_time=component,
        eager_kernel_call_time=component,
        eager_invocation_scatter_time=component,
        eager_finalization_time=component,
        eager_scatter_finalization_time=component,
        eager_closure_time=component,
        eager_copy_out_time=component,
        stages=(),
    )

    rendered = render_summary(
        replace(result, timing_breakdown=eager_breakdown), color=False
    )

    assert rendered is not None
    assert "Rusticol Eager Timing Breakdown" in rendered
    assert "Initialize (exclusive)" in rendered
    assert "Gather (exclusive)" in rendered
    assert "Kernel calls (exclusive)" in rendered
    assert "Invocation scatter (exclusive)" in rendered
    assert "Current finalization (exclusive)" in rendered
    assert "Amplitude closure (exclusive)" in rendered
    assert "Amplitude copy-out (exclusive)" in rendered
    assert "Eager execution (inclusive)" in rendered
    assert "Rusticol Stage Detail" not in rendered


def test_eager_benchmark_labels_aggregate_until_native_phases_exist() -> None:
    result = _benchmark_result()
    breakdown = result.timing_breakdown
    assert breakdown is not None

    rendered = render_summary(
        replace(
            result,
            timing_breakdown=replace(
                breakdown,
                execution_mode="eager",
                stage_evaluator_call_time=None,
                eager_execution_time=breakdown.stage_evaluator_call_time,
                stages=(),
            ),
        ),
        color=False,
    )

    assert rendered is not None
    assert "Eager execution (inclusive)" in rendered
    assert "Stage input pack" not in rendered
    assert "Output assign" not in rendered


def test_interrupted_benchmark_renders_partial_status() -> None:
    rendered = render_summary(
        replace(_benchmark_result(), interrupted=True), color=False
    )

    assert rendered is not None
    assert "interrupted - partial statistics from 8 complete blocks" in rendered


def _artifact_inspection() -> ArtifactInspection:
    alias = ArtifactAliasInspection(
        id="ddbar_zg_alias",
        expression="d d~ > g z",
        representative_id="ddbar_zg",
        external_pdgs=(1, -1, 21, 23),
    )
    process = ArtifactProcessInspection(
        id="ddbar_zg",
        expression="d d~ > z g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        default=True,
        physical_helicities=24,
        computed_helicities=12,
        physical_color_components=1,
        computed_color_components=1,
        helicity_coverage="complete",
        color_coverage="complete",
        aliases=(alias,),
        lc_flow_layout="all-flow-union",
        lc_union_sector_count=1,
    )
    return ArtifactInspection(
        kind="pyamplicol-artifact-inspection",
        path=Path("/tmp/artifact"),
        artifact_kind="pyamplicol-process-set",
        artifact_id="a" * 64,
        created_utc="2026-07-16T00:00:00Z",
        producer_version="0.1.0",
        target="x86_64-unknown-linux-gnu",
        cpu_features=(),
        model_name="sm",
        model_source="ufo-json",
        model_restriction="default",
        default_process_id="ddbar_zg",
        runtime_engine="rusticol",
        runtime_version="0.1.0",
        runtime_capabilities=("symjit.application.complex-f64.v1",),
        payload_count=12,
        payload_size_bytes=2048,
        integrity="verified",
        processes=(process,),
        dependencies=(
            ArtifactDependencyInspection(
                name="symbolica",
                version="2.1.0",
                license="Symbolica Software License Agreement",
                source="https://symbolica.io/",
            ),
        ),
    )


def test_artifact_inspection_uses_process_and_alias_tables() -> None:
    rendered = render_summary(_artifact_inspection(), color=False)

    assert rendered is not None
    assert "Artifact" in rendered
    assert "Processes" in rendered
    assert "Aliases" in rendered
    assert "Dependencies" in rendered
    assert "Execution" in rendered
    assert "ddbar_zg" in rendered
    assert "d d~ > z g" in rendered
    assert "24 (12 eval.)" in rendered
    assert "LC flow layout" in rendered
    assert "all-flow-union" in rendered
    assert "1 physical / materialized; 0 residual" in rendered


def test_eager_artifact_inspection_reports_execution_contract() -> None:
    inspection = _artifact_inspection()
    eager_process = replace(
        inspection.processes[0],
        execution_mode="eager",
        prepared_backend="jit",
        prepared_kernel_count=12,
        referenced_kernel_count=9,
        invocation_count=80,
        attachment_count=100,
        evaluation_alias_count=20,
        maximum_fanout=4,
        finalization_count=30,
        closure_count=12,
        selector_closure_available=True,
        selector_domain_count=12,
        selector_domain_membership_count=160,
        requested_point_tile_size=1024,
        effective_point_tile_size=None,
        workspace_limit_bytes=256 * 1024 * 1024,
        workspace_bytes=None,
        native_profile_phases=(
            "source-fill",
            "momentum-setup",
            "eager-execution-aggregate",
        ),
    )

    rendered = render_summary(
        replace(inspection, processes=(eager_process,)), color=False
    )

    assert rendered is not None
    assert "eager (jit)" in rendered
    assert "12 in pack; 9 referenced" in rendered
    assert "80 canonical; 100 attachments; 20 reused aliases; max fanout 4" in rendered
    assert "12 domains; 160 memberships" in rendered
    assert "1024 / available after runtime load" in rendered
    assert "256.00 MiB / available after runtime load" in rendered


def test_on_the_fly_artifact_inspection_reports_static_compact_contract() -> None:
    inspection = _artifact_inspection()
    on_the_fly_process = replace(
        inspection.processes[0],
        execution_mode="on-the-fly",
        physical_helicities=64,
        computed_helicities=64,
        physical_color_components=24,
        computed_color_components=24,
        lc_flow_layout="compact/query-local",
        lc_physical_sector_count=24,
        lc_union_sector_count=None,
        requested_point_tile_size=128,
        effective_point_tile_size=None,
        requested_query_construction_threads=4,
        native_profile_phases=(
            "selector-plan",
            "recurrence-direct-contribution",
            "recurrence-direct-closure",
        ),
    )

    rendered = render_summary(
        replace(inspection, processes=(on_the_fly_process,)), color=False
    )

    assert rendered is not None
    assert "compact/query-local" in rendered
    assert "64 physical helicities; 24 physical LC color flows" in rendered
    assert "128 requested; not applied by on-the-fly execution" in rendered
    assert "query construction threads" in rendered
    assert "4 requested" in rendered
    assert "f64 only" in rendered
    assert "topology-replay" not in rendered
    assert "LC replay sectors" not in rendered
    assert "None materialized" not in rendered
    assert "None replayed" not in rendered
    assert "None residual" not in rendered


def test_recurrence_artifact_inspection_reports_contracted_color() -> None:
    inspection = _artifact_inspection()
    recurrence_process = replace(
        inspection.processes[0],
        color_accuracy="full",
        execution_mode="recurrence",
        prepared_backend="jit",
        direct_source_row_count=12,
        direct_contribution_row_count=126,
        direct_finalization_row_count=69,
        direct_closure_row_count=24,
        recurrence_color_accuracy="full",
        recurrence_color_storage="repeated",
        recurrence_color_sector_count=36,
        recurrence_color_active_sector_count=6,
        recurrence_color_component_count=64,
        recurrence_color_group_count=384,
        recurrence_color_entry_count=21,
        recurrence_color_logical_entry_count=1344,
        recurrence_color_factorization_kind="klein-four-walsh",
        recurrence_color_factorization_rank=2,
        recurrence_color_factorization_coset_count=1,
    )

    rendered = render_summary(
        replace(inspection, processes=(recurrence_process,)), color=False
    )

    assert rendered is not None
    assert "recurrence (jit)" in rendered
    assert "12 sources; 126 contributions; 69 finalizations; 24 closures" in rendered
    assert "6/36 active sectors" in rendered
    assert "21/1344" in rendered
    assert "stored/logical entries" in rendered
    assert "klein-four-walsh, rank 2, 1 cosets" in rendered


def test_artifact_inspection_reports_runtime_selector_proofs() -> None:
    inspection = _artifact_inspection()
    reusable_process = replace(
        inspection.processes[0],
        selector_provenance="pyamplicol-runtime-selectors-v1",
        helicity_runtime_contract="complete-reusable",
        color_flow_runtime_contract="complete-reusable",
        generation_specialized_axes=(),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        helicity_recurrence_status="available",
        helicity_optimized_class_count=14,
        helicity_optimized_current_count=72,
        helicity_residual_current_count=0,
        helicity_optimized_amplitude_class_count=4,
        helicity_residual_amplitude_count=0,
        helicity_materialized_current_count=40,
        helicity_materialized_amplitude_count=6,
        lc_physical_sector_count=6,
        lc_materialized_sector_count=2,
        lc_replayed_sector_count=6,
        lc_residual_sector_count=0,
    )

    rendered = render_summary(
        replace(inspection, processes=(reusable_process,)), color=False
    )

    assert rendered is not None
    assert "helicity complete-reusable; color flow complete-reusable" in rendered
    assert "generation specialization" in rendered
    assert "generation selection" in rendered
    assert "pyamplicol-runtime-selectors-v1" in rendered
    assert "6 physical; 2 materialized; 6 replayed; 0 residual" in rendered
    assert "72 optimized currents in 14 classes; 0 residual" in rendered
    assert "amplitude classes; 0 residual roots" in rendered
    assert "materialized 40 currents / 6 roots" in rendered


def test_artifact_inspection_color_is_optional() -> None:
    rendered = render_summary(_artifact_inspection(), color=True)

    assert rendered is not None
    assert "\x1b[36mddbar_zg" in rendered
    assert "\x1b[32md d~ > z g" in rendered
    assert "\x1b[35mmode / backend" in rendered
    assert "\x1b[36mcompiled" in rendered
    assert "\x1b[36msymbolica" in rendered
    assert "\x1b[33mSymbolica Software License Agreement" in rendered


def _member(
    identifier: str,
    process: str,
    *,
    factor: str = "1",
    permutation: tuple[int, ...] = (0, 1, 2, 3),
    conjugated: bool = False,
    basis: str = "flavor-blind",
    generated: bool = False,
    warning: str | None = None,
) -> ArtifactProcessGroupMemberInspection:
    return ArtifactProcessGroupMemberInspection(
        id=identifier,
        process=process,
        pdgs=(1,) * len(permutation),
        initial_count=2,
        factor=factor,
        permutation=permutation,
        conjugated=conjugated,
        basis=basis,
        generated=generated,
        warning=warning,
    )


def _group(
    representative: str, process: str, *members: ArtifactProcessGroupMemberInspection
) -> ArtifactProcessGroupInspection:
    return ArtifactProcessGroupInspection(
        representative_id=representative,
        representative=process,
        members=(
            _member(representative, process, basis="identity", generated=True),
            *members,
        ),
    )


def _grouped_inspection() -> ArtifactInspection:
    """A flavour-blind map with a conjugated, a non-unit and a warned member."""

    groups = ArtifactProcessGroupsInspection(
        stored=True,
        groups=(
            _group(
                "ddbar_zg",
                "d d~ > z g",
                _member("uubar_zg", "u u~ > z g", warning="Z couplings differ"),
                _member(
                    "dbar_d_zg",
                    "d~ d > z g",
                    permutation=(1, 0, 2, 3),
                    conjugated=True,
                ),
                _member(
                    "ssbar_zg",
                    "s s~ > z g",
                    factor="16/9",
                    basis="numerical",
                    generated=True,
                ),
            ),
            _group("gg_zg", "g g > z g"),
        ),
        analyses=({"variant": "numerical", "only_unit_multiplier": True},),
    )
    return replace(_artifact_inspection(), process_groups=groups)


def test_process_group_map_renders_overview_rows_and_footnotes() -> None:
    rendered = render_summary(_grouped_inspection(), color=False)

    assert rendered is not None
    assert "\x1b[" not in rendered
    section = rendered[rendered.index("Process groups") : rendered.index("Execution")]
    lines = section.splitlines()
    assert lines[3].split("|")[1:7] == [
        " groups ",
        " subprocesses ",
        " generated ",
        " routed ",
        " conjugated ",
        " factor != 1 ",
    ]
    assert [cell.strip() for cell in lines[5].split("|")[1:10]] == [
        "2",
        "5",
        "3",
        "2",
        "1",
        "1",
        "flavor-blind 2",
        "numerical",
        "1",
    ]
    assert "| numerical 1    | (unit factor |" in lines[6]
    assert "| ddbar_zg     | d d~ > z g | representative |        |   |" in section
    assert "| |- ssbar_zg  | s s~ > z g | generated      |   16/9 |   |" in section
    assert "| |- uubar_zg  | u u~ > z g | routed         |      1 |   |" in section
    assert "| `- dbar_d_zg | d~ d > z g | routed         |      1 | C | [1 0 2 3]" in (
        section
    )
    assert "| flavor-blind | [1]  |" in section
    assert "| gg_zg        | g g > z g  | representative |" in section
    assert "\n[1] Z couplings differ" in section
    assert "C: charge-conjugated relation" in section
    assert "permutation p: representative leg r" in section
    assert "factor F: ME = F x ME(representative)" in section
    assert "\nnumerical analysis: unit factors only" in section
    assert "| contents        | 1 process, 1 alias; 5 subprocesses in 2 groups " in (
        rendered
    )
    assert all(len(line) <= 120 for line in lines)


def test_unit_factor_process_group_map_omits_the_factor_column() -> None:
    groups = ArtifactProcessGroupsInspection(
        stored=True,
        groups=(
            _group(
                "ddbar_zg",
                "d d~ > z g",
                _member("uubar_zg", "u u~ > z g"),
                _member("ssbar_zg", "s s~ > z g"),
            ),
        ),
    )

    rendered = render_summary(
        replace(_artifact_inspection(), process_groups=groups), color=False
    )

    assert rendered is not None
    section = rendered[rendered.index("Process groups") : rendered.index("Execution")]
    assert "| process ID  | process    | role           | basis        |" in section
    assert "| |- ssbar_zg | s s~ > z g | routed         | flavor-blind |" in section
    assert "| `- uubar_zg | u u~ > z g | routed         | flavor-blind |" in section
    assert "factor F" not in section
    assert "permutation" not in section
    assert "C: charge-conjugated" not in section

    # a numerical group of generated members needs no legend at all
    numerical = ArtifactProcessGroupsInspection(
        stored=True,
        groups=(
            _group(
                "ddbar_zg",
                "d d~ > z g",
                _member("ssbar_zg", "s s~ > z g", basis="numerical", generated=True),
            ),
        ),
    )
    rendered = render_summary(
        replace(_artifact_inspection(), process_groups=numerical), color=False
    )
    assert rendered is not None
    section = rendered[rendered.index("Process groups") : rendered.index("Execution")]
    assert section.endswith(
        "+-------------+------------+----------------+-----------+\n\n"
    )
    assert "| `- ssbar_zg | s s~ > z g | generated      | numerical |" in section


def test_process_group_map_colors_relations() -> None:
    rendered = render_summary(_grouped_inspection(), color=True)

    assert rendered is not None
    assert "\x1b[36mProcess groups" in rendered
    assert "\x1b[1m\x1b[32md d~ > z g" in rendered  # representative in bold
    # bold without a colour stays readable on light and dark terminals
    assert "\x1b[1mrepresentative" in rendered
    assert "\x1b[37m" not in rendered[rendered.index("Process groups") :]
    assert "\x1b[33m16/9" in rendered  # non-unit factor
    assert "\x1b[32m1\x1b[0m" in rendered  # unit factor
    assert "\x1b[1m\x1b[35mC" in rendered  # conjugation marker
    assert "\x1b[94mflavor-blind" in rendered
    assert "\x1b[32mnumerical" in rendered
    assert "\x1b[36mrouted" in rendered
    assert "\x1b[33m[1]" in rendered  # warning reference and footnote


def test_trivial_process_group_map_is_a_one_line_note() -> None:
    inspection = _artifact_inspection()
    identity = ArtifactProcessGroupsInspection(
        stored=True, groups=(_group("ddbar_zg", "d d~ > z g"),)
    )

    rendered = render_summary(replace(inspection, process_groups=identity), color=False)
    absent = render_summary(
        replace(inspection, process_groups=replace(identity, stored=False)),
        color=False,
    )

    assert rendered is not None and absent is not None
    assert "Process groups: none; the process is its own group" in rendered
    assert "subprocesses" not in rendered
    assert "Process groups: no process-groups.json" in absent


def _generation_result(*names: tuple[str, str]) -> GenerationResult:
    return GenerationResult(
        output=Path("artifact"),
        processes=ProcessSet(
            tuple(
                ProcessRequest.parse(expression, name=name)
                for name, expression in names
            )
        ),
        mode="error",
    )


def test_generation_summary_reports_the_process_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = _grouped_inspection().process_groups
    monkeypatch.setattr(
        "pyamplicol.artifacts.inspection.stored_process_groups",
        lambda _output: groups,
    )
    result = _generation_result(
        ("ddbar_zg", "d d~ > z g"), ("ssbar_zg", "s s~ > z g"), ("gg_zg", "g g > z g")
    )

    rendered = render_summary(result, color=False)
    colored = render_summary(result, color=True)

    assert rendered is not None and colored is not None
    lines = rendered.splitlines()
    row = next(
        index for index, line in enumerate(lines) if "| process grouping" in line
    )
    assert "| processes " in lines[row - 1]
    # the warning count moves to its own line rather than overflow 88 columns
    assert [line.split("|")[2].strip() for line in lines[row : row + 4]] == [
        "5 subprocesses -> 2 groups: 3 generated, 2 routed",
        "2 flavor-blind + 1 numerical relations (1 charge-conjugated, 1 with factor "
        "!= 1)",
        "1 warning",
        "numerical analysis: unit factors only",
    ]
    assert "| existing-output policy" in lines[row + 4]
    assert "| name     | process    | group                      |" in rendered
    assert "| ddbar_zg | d d~ > z g | 4 subprocesses (2 routed)  |" in rendered
    # a generated member names its representative and its non-unit factor
    assert "| ssbar_zg | s s~ > z g | member of ddbar_zg (x16/9) |" in rendered
    assert "| gg_zg    | g g > z g  |                            |" in rendered
    assert "\x1b[36mprocess grouping" in colored
    assert "\x1b[1m5 subprocesses" in colored
    assert "\x1b[32m3 generated" in colored
    assert "\x1b[36m2 routed" in colored
    assert "\x1b[94mflavor-blind" in colored
    assert "\x1b[32mnumerical" in colored
    assert "\x1b[35m1 charge-conjugated" in colored
    assert "\x1b[33m1 with factor != 1" in colored
    assert "\x1b[33m1 warning" in colored


def test_generation_summary_omits_a_trivial_process_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trivial = ArtifactProcessGroupsInspection(
        stored=True,
        groups=(_group("ddbar_zg", "d d~ > z g"), _group("gg_zg", "g g > z g")),
    )
    stored: list[ArtifactProcessGroupsInspection | None] = [trivial, None]
    monkeypatch.setattr(
        "pyamplicol.artifacts.inspection.stored_process_groups",
        lambda _output: stored.pop(0),
    )
    result = _generation_result(("ddbar_zg", "d d~ > z g"), ("gg_zg", "g g > z g"))

    for _ in range(2):
        rendered = render_summary(result, color=False)
        assert rendered is not None
        assert "process grouping" not in rendered
        assert "| name     | process    |\n" in rendered

    analysed = replace(trivial, analyses=({"variant": "numerical", "points": 3},))
    monkeypatch.setattr(
        "pyamplicol.artifacts.inspection.stored_process_groups",
        lambda _output: analysed,
    )
    rendered = render_summary(result, color=False)
    assert rendered is not None
    # worded as `inspect` words a trivial map
    assert (
        "| process grouping       | no relation found; each of the 2 processes is "
        "its own group" in rendered
    )
    assert "|                        | numerical analysis: 3 points " in rendered
    assert "| group" not in rendered


def test_large_process_group_map_lists_members_compactly() -> None:
    flavours = ("d", "u", "s", "c", "b")
    members = [
        _member(
            f"m{index}",
            f"{first} {second}~ > {first} {second}~ {third} {third}~",
            conjugated=index >= 60,
        )
        for index, (first, second, third) in enumerate(
            itertools.product(flavours, repeat=3)
        )
        if index < 100
    ]
    groups = ArtifactProcessGroupsInspection(
        stored=True,
        groups=(
            _group("ddbar_4j", "d d~ > d d~ d d~", *members),
            _group("gg_4g", "g g > g g g g"),
        ),
    )

    rendered = render_summary(
        replace(_artifact_inspection(), process_groups=groups), color=False
    )

    assert rendered is not None
    section = rendered[rendered.index("Process groups") : rendered.index("Execution")]
    lines = section.splitlines()
    assert "| representative   | members" in section
    assert any(
        line.startswith("| d d~ > d d~ d d~ | 60 routed    d d~ > d d~ d d~  ")
        for line in lines
    )
    assert (
        "| ddbar_4j         |              d d~ > d d~ b b~  d u~ > d u~ d d~"
        in section
    )
    assert "| 101 subprocesses |" in section
    assert "|                  | 40 routed C  " in section
    assert "| ungrouped        |              g g > g g g g" in section
    assert "| 1 subprocess     |" in section
    assert "compact view of 102 subprocesses" in section
    assert "--process REPRESENTATIVE" in section
    assert "|- " not in section
    # four members per line instead of one row each
    assert len(lines) < 50
    assert all(len(line) <= 120 for line in lines)

    # `--process REPRESENTATIVE` keeps one group, which is listed row by row
    selected = render_summary(
        replace(
            _artifact_inspection(),
            process_groups=replace(groups, groups=groups.groups[:1]),
        ),
        color=False,
    )
    assert selected is not None
    assert "| |- m0 " in selected
    assert "compact view" not in selected


def test_grouped_full_physics_renders_the_process_group_block() -> None:
    selection = ArtifactProcessGroupSelection(
        process_id="dbar_d_zg",
        role="routed member",
        representative_id="ddbar_zg",
        representative="d d~ > z g",
        factor="1",
        permutation=(1, 0, 2, 3),
        conjugated=True,
        basis="flavor-blind",
        warning=None,
        group_size=4,
        routed_count=2,
        conjugated_count=1,
    )
    physics = {
        "process_id": "dbar_d_zg",
        "process": "d~ d > z g",
        "external_particles": [],
        "helicities": [{"id": "h:-1,-1,-1,-1"}],
    }

    rendered = render_summary({**physics, "process_group": selection}, color=False)
    representative = render_summary(
        {
            **physics,
            "process_group": replace(
                selection, role="representative", conjugated=False
            ),
        },
        color=False,
    )

    assert rendered is not None and representative is not None
    assert rendered.index("Process Group") < rendered.index("Process Physics")
    assert "routed member: not generated, evaluated through the representative" in (
        rendered
    )
    assert "| representative | ddbar_zg  d d~ > z g" in rendered
    assert (
        "| permutation    | p = [1 0 2 3]: representative leg r takes the momentum "
        "of leg p[r]"
    ) in rendered
    assert "| conjugated     | yes (C)" in rendered
    assert "4 subprocesses: 2 generated, 2 routed, 1 conjugated" in rendered
    assert "representative: generated, evaluated by its own runtime" in (representative)
    # the representative requested in another leg order shows that order
    assert (
        "| permutation | p = [1 0 2 3]: representative leg r takes the momentum of "
        "leg p[r] |"
    ) in representative
    stored_order = render_summary(
        {
            **physics,
            "process_group": replace(
                selection,
                role="representative",
                conjugated=False,
                permutation=(0, 1, 2, 3),
            ),
        },
        color=False,
    )
    assert stored_order is not None
    assert "permutation" not in stored_order.split("Process Physics")[0]


def test_process_groups_json_is_complete_and_uncolored() -> None:
    stream = io.StringIO()
    write_result(_grouped_inspection(), format="json", stream=stream, color=True)

    assert "\x1b[" not in stream.getvalue()
    payload = json.loads(stream.getvalue())["process_groups"]
    assert payload["stored"] is True
    assert payload["analyses"] == [
        {"only_unit_multiplier": True, "variant": "numerical"}
    ]
    assert [len(group["members"]) for group in payload["groups"]] == [4, 1]
    assert payload["groups"][0]["members"][2] == {
        "basis": "flavor-blind",
        "conjugated": True,
        "factor": "1",
        "generated": False,
        "id": "dbar_d_zg",
        "initial_count": 2,
        "pdgs": [1, 1, 1, 1],
        "permutation": [1, 0, 2, 3],
        "process": "d~ d > z g",
        "warning": None,
    }
