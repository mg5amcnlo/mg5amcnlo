# SPDX-License-Identifier: 0BSD
"""Static external consumer of the installed inline public API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal, assert_type

from pyamplicol import (
    BenchmarkComponentTiming,
    BenchmarkConfig,
    BenchmarkProfileCounters,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkStageTiming,
    BenchmarkStatistics,
    BenchmarkTimingBreakdown,
    ColorCorrelator,
    ColorFlow,
    CompiledModel,
    ContractedColorComponent,
    CorrelatedRequest,
    CorrelatedValue,
    CorrelatorConfig,
    EvaluationConfig,
    GenerationConfig,
    GenerationPlan,
    GenerationResult,
    Generator,
    HelicityConfiguration,
    ModelSource,
    ProcessPhysics,
    ProcessRequest,
    ProcessSet,
    ResolvedEvaluation,
    RunConfig,
    Runtime,
    WarmUpResult,
    benchmark,
    generate,
    load,
)
from pyamplicol.api import GeneratorBackend, ModelParameters, Momenta, RuntimeBackend
from pyamplicol.config import (
    CONFIG_SECTIONS,
    FIELD_REGISTRY,
    Action,
    ClampRequest,
    ConfigClamp,
    ConfigField,
    ConfigOverride,
    ConfigResolution,
    config_to_dict,
    config_to_toml,
    get_config_field,
    load_config,
    parse_override,
    resolution_to_dict,
    resolve_config,
)
from pyamplicol.reporting import NullProgressSink

MOMENTA: Momenta = (((10.0, 0.0, 0.0, 10.0),),)
PARAMETERS: ModelParameters = {"normalization.alpha_s_me_check": 0.118}


def exercise_otf_cache(artifact: Path, cache: Path) -> None:
    runtime = Runtime.load(artifact)
    assert_type(runtime.warm_up(MOMENTA, n_cores=4), WarmUpResult)
    assert_type(runtime.warm_up(MOMENTA, n_cores=None), WarmUpResult)
    assert_type(runtime.save(cache), None)
    assert_type(runtime.load_cache(str(cache)), None)


def exercise_generator(artifact: Path) -> None:
    processes = ProcessSet.from_expressions(("d d~ > z",), names=("ddbar_z",))
    assert_type(processes, ProcessSet)
    compiled = ModelSource.built_in_sm().compile(use_cache=False)
    assert_type(compiled, CompiledModel)
    generator = Generator(
        RunConfig(
            action=Action.GENERATE,
            generation=GenerationConfig(workers=1),
        ),
        progress=NullProgressSink(),
    )
    plan = generator.plan(processes, model=compiled)
    assert_type(plan, GenerationPlan)
    generated = generator.generate(processes, artifact, mode="replace")
    assert_type(generated, GenerationResult)
    assert_type(
        generate(
            ProcessRequest.parse("d d~ > z"),
            artifact,
            model=compiled,
            mode="append",
            config=GenerationConfig(workers=1),
        ),
        GenerationResult,
    )


def exercise_runtime(artifact: Path) -> None:
    runtime = Runtime.load(artifact, model_parameters=PARAMETERS, mute_warnings=True)
    assert_type(runtime, Runtime)
    assert_type(load(artifact, process="ddbar_z"), Runtime)
    assert_type(runtime.artifact_id, str)
    assert_type(
        runtime.execution_mode,
        Literal["compiled", "eager", "recurrence", "on-the-fly"],
    )
    assert_type(runtime.inspect(), Mapping[str, object])
    physics = runtime.physics
    assert_type(physics, ProcessPhysics)
    assert_type(
        runtime.evaluate(
            MOMENTA,
            helicities=physics.helicities,
            color_flows=physics.color_flows,
        ),
        tuple[complex | Decimal, ...],
    )
    resolved = runtime.evaluate_resolved(MOMENTA, precision=32)
    assert_type(resolved, ResolvedEvaluation)
    assert_type(resolved.total(), tuple[complex | Decimal, ...])
    runtime.set_model_parameters(PARAMETERS)
    runtime.set_model_parameter("normalization.alpha_s_me_check", 0.119)
    runtime.mute_warnings()
    runtime.unmute_warnings()
    runtime.clear()
    assert_type(BenchmarkRunner(BenchmarkConfig()).run(runtime), BenchmarkResult)
    assert_type(benchmark(runtime, points=MOMENTA), BenchmarkResult)


def exercise_correlations(artifact: Path, momenta: Momenta) -> None:
    dipole = ColorCorrelator.dipole("T13", 1, 3)
    assert_type(dipole, ColorCorrelator)
    declarations = CorrelatorConfig(
        color_correlations=(dipole,), spin_correlations=((3,),)
    )
    assert_type(declarations, CorrelatorConfig)
    assert_type(
        Generator().generate("g g > g g", artifact, correlators=declarations),
        GenerationResult,
    )
    runtime = Runtime.load(artifact)
    runtime.set_spin_correlation_vectors({3: (0, 1, 0, 0)})
    requests = {
        "born": CorrelatedRequest("born", spin_vectors={}),
        "dipole13_spin": CorrelatedRequest("T13"),
    }
    assert_type(requests, dict[str, CorrelatedRequest])
    results = runtime.evaluate_correlated_many(momenta, requests, precision=40)
    assert_type(results, dict[str, tuple[CorrelatedValue, ...]])
    assert_type(results["dipole13_spin"][0].real, Decimal)
    assert_type(results["dipole13_spin"][0].imag, Decimal)
    assert_type(runtime.evaluate_correlated(momenta), tuple[CorrelatedValue, ...])
    runtime.set_spin_correlation_vectors(
        {3: (Decimal(0), (Decimal("1.00000000000000000001"), Decimal("1e-25")), 0, 0)}
    )
    assert_type(
        runtime.evaluate_correlated(momenta, arithmetic="double-double", precision=31),
        tuple[CorrelatedValue, ...],
    )
    assert_type(
        runtime.evaluate_correlated_many(
            momenta, requests, arithmetic="double-double", precision=31
        ),
        dict[str, tuple[CorrelatedValue, ...]],
    )
    runtime.set_spin_correlation_vectors(None)


def exercise_configuration(card: Path) -> None:
    generation = GenerationConfig(workers=1)
    evaluation = EvaluationConfig(artifact=card, precision=32, resolved=True)
    benchmark_config = BenchmarkConfig(minimum_samples=3)
    run = RunConfig(
        action=Action.EVALUATE,
        generation=generation,
        evaluation=evaluation,
        benchmark=benchmark_config,
    )
    assert_type(generation, GenerationConfig)
    assert_type(evaluation, EvaluationConfig)
    assert_type(benchmark_config, BenchmarkConfig)
    assert_type(run, RunConfig)
    override = parse_override("generation.workers=2", base_dir=card.parent)
    assert_type(override, ConfigOverride)
    dedicated: Mapping[str, object] = {"generation": {"workers": 2}}
    resolution = resolve_config(
        card,
        action=Action.GENERATE,
        dedicated=dedicated,
        overrides=(override,),
        clamps=(ClampRequest("generation.workers", 1, "resource limit"),),
    )
    assert_type(resolution, ConfigResolution)
    assert_type(load_config(card, overrides=(override,)), RunConfig)
    assert_type(config_to_dict(run), dict[str, object])
    assert_type(config_to_toml(run), str)
    assert_type(resolution_to_dict(resolution), dict[str, object])
    assert_type(get_config_field("generation.workers"), ConfigField)
    assert_type(FIELD_REGISTRY, Mapping[str, ConfigField])
    assert_type(CONFIG_SECTIONS, tuple[str, ...])


def exercise_results(
    plan: GenerationPlan,
    generated: GenerationResult,
    physics: ProcessPhysics,
    resolved: ResolvedEvaluation,
    result: BenchmarkResult,
) -> None:
    assert_type(plan.concrete_processes, tuple[ProcessRequest, ...])
    assert_type(plan.estimated_coverage, Mapping[str, object])
    assert_type(plan.requested_settings, RunConfig | GenerationConfig)
    assert_type(plan.effective_settings, RunConfig | GenerationConfig)
    assert_type(plan.adjustments, tuple[ConfigClamp, ...])
    assert_type(plan.unsupported_features, tuple[str, ...])
    assert_type(generated.output, Path)
    assert_type(generated.processes, ProcessSet)
    assert_type(generated.mode, Literal["error", "append", "replace"])
    assert_type(physics.helicities, tuple[HelicityConfiguration, ...])
    assert_type(physics.color_flows, tuple[ColorFlow, ...])
    assert_type(
        physics.contracted_color_components,
        tuple[ContractedColorComponent, ...],
    )
    assert_type(physics.helicity_ids, tuple[str, ...])
    assert_type(resolved.shape, tuple[int, int, int])
    assert_type(result.uncertainty, BenchmarkStatistics)
    assert_type(result.evaluator_uncertainty, BenchmarkStatistics | None)
    assert_type(result.repetitions_per_sample, int)
    assert_type(result.evaluation_count, int)
    assert_type(result.evaluated_point_count, int)
    assert_type(result.process_id, str | None)
    assert_type(result.process_expression, str | None)
    assert_type(result.timing_breakdown, BenchmarkTimingBreakdown | None)
    if result.timing_breakdown is not None:
        assert_type(
            result.timing_breakdown.stage_evaluator_call_time,
            BenchmarkComponentTiming | None,
        )
        assert_type(result.timing_breakdown.stages, tuple[BenchmarkStageTiming, ...])
        assert_type(
            result.timing_breakdown.stage_backend_call_time,
            BenchmarkComponentTiming | None,
        )
        if result.timing_breakdown.stages:
            assert_type(
                result.timing_breakdown.stages[0].leaf_input_pack_time,
                BenchmarkComponentTiming | None,
            )
        assert_type(
            result.timing_breakdown.counters,
            BenchmarkProfileCounters | None,
        )
        if result.timing_breakdown.counters is not None:
            assert_type(
                result.timing_breakdown.counters.stage_input_copy_components_per_point,
                float | None,
            )
            assert_type(
                result.timing_breakdown.counters.compiled_direct_arena_engines_per_call,
                float | None,
            )
            assert_type(
                result.timing_breakdown.counters.compiled_direct_arena_calls_per_call,
                float | None,
            )
            assert_type(
                result.timing_breakdown.counters.compiled_direct_arena_boundary_input_bytes_per_call,
                float | None,
            )
            assert_type(
                result.timing_breakdown.counters.compiled_direct_arena_boundary_current_output_bytes_per_call,
                float | None,
            )
            assert_type(
                result.timing_breakdown.counters.compiled_direct_arena_boundary_amplitude_output_bytes_per_call,
                float | None,
            )
    assert_type(result.environment, Mapping[str, object])


def exercise_protocols(
    generator: GeneratorBackend,
    runtime: RuntimeBackend,
    processes: ProcessSet,
) -> None:
    assert_type(generator.plan(processes), GenerationPlan)
    assert_type(generator.generate(processes, "artifact"), GenerationResult)
    assert_type(runtime.physics, ProcessPhysics)
    assert_type(runtime.evaluate(MOMENTA), Sequence[complex | Decimal])
    assert_type(runtime.evaluate_resolved(MOMENTA), ResolvedEvaluation)
