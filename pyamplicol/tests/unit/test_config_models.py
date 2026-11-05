# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass, replace

import pytest

from pyamplicol.config import (
    CONFIG_SECTIONS,
    FIELD_REGISTRY,
    Action,
    BenchmarkConfig,
    ColorAccuracy,
    ColorConfig,
    ColorContraction,
    ColorFFTBasis,
    ConfigurationError,
    CppConfig,
    EagerEvaluatorConfig,
    EvaluationConfig,
    EvaluatorBackend,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    LCFlowLayout,
    ModelConfig,
    OutputConfig,
    ProcessConfig,
    ProcessEntry,
    ProcessGroupAnalysis,
    RecurrenceEvaluatorConfig,
    RelationDiscoveryMode,
    RunConfig,
    SymbolicaConfig,
)


def test_schema_v1_registry_contains_every_contract_leaf() -> None:
    assert len(FIELD_REGISTRY) == 85
    assert "generation.umami_grouping" not in FIELD_REGISTRY
    assert "evaluator.jit.direct_translation" not in FIELD_REGISTRY
    assert FIELD_REGISTRY["action"].required
    assert FIELD_REGISTRY["generation.workers"].default == "auto"
    assert FIELD_REGISTRY["generation.validation.samples"].default == 2
    assert FIELD_REGISTRY["evaluator.jit.optimization_level"].choices == (
        0,
        1,
        2,
        3,
    )
    assert FIELD_REGISTRY["evaluator.jit.compress"].kind == "auto_bool"
    assert FIELD_REGISTRY["evaluator.jit.compress"].default == "auto"
    assert FIELD_REGISTRY["evaluator.execution_mode"].choices == (
        EvaluatorExecutionMode.COMPILED,
        EvaluatorExecutionMode.EAGER,
        EvaluatorExecutionMode.RECURRENCE,
        EvaluatorExecutionMode.ON_THE_FLY,
    )
    assert (
        FIELD_REGISTRY["evaluator.execution_mode"].default
        is EvaluatorExecutionMode.RECURRENCE
    )
    assert FIELD_REGISTRY["color.lc_flow_layout"].default == (
        LCFlowLayout.TOPOLOGY_REPLAY
    )
    assert FIELD_REGISTRY["color.lc_flow_layout"].choices == (
        LCFlowLayout.TOPOLOGY_REPLAY,
        LCFlowLayout.ALL_FLOW_UNION,
    )
    assert FIELD_REGISTRY["color.contraction"].default is ColorContraction.AUTO
    assert FIELD_REGISTRY["color.contraction"].choices == (
        ColorContraction.AUTO,
        ColorContraction.DIRECT,
        ColorContraction.SYMMETRIC_GROUP_FFT,
    )
    assert FIELD_REGISTRY["color.fft_basis"].default is ColorFFTBasis.ADJOINT
    assert FIELD_REGISTRY["color.fft_basis"].choices == tuple(ColorFFTBasis)
    assert FIELD_REGISTRY["evaluator.eager.point_tile_size"].default == 1024
    assert FIELD_REGISTRY["evaluator.eager.workspace_mib"].default == 256
    assert "evaluator.eager" in CONFIG_SECTIONS
    assert FIELD_REGISTRY["evaluator.recurrence.point_tile_size"].default == 1024
    assert FIELD_REGISTRY["evaluator.recurrence.workspace_mib"].default == 256
    assert "evaluator.recurrence" in CONFIG_SECTIONS
    assert FIELD_REGISTRY["generation.relation_discovery.mode"].default is (
        RelationDiscoveryMode.OFF
    )
    assert (
        FIELD_REGISTRY["generation.relation_discovery.precision_digits"].default == 96
    )
    assert (
        FIELD_REGISTRY["generation.relation_discovery.verification_probe_count"].default
        == 4
    )
    assert (
        FIELD_REGISTRY["generation.relation_discovery.relative_tolerance"].default
        == 1.0e-70
    )
    assert (
        FIELD_REGISTRY["generation.relation_discovery.absolute_tolerance"].default
        == 1.0e-80
    )
    assert "generation.relation_discovery" in CONFIG_SECTIONS
    assert FIELD_REGISTRY["process.multiparticles"].dynamic_kind == "list_str"
    assert FIELD_REGISTRY["process.entries"].kind == "process_entries"
    assert FIELD_REGISTRY["process.reference_color_order"].kind == "list_int"
    assert FIELD_REGISTRY["process.selected_color_sector_ids"].kind == "list_int"
    assert FIELD_REGISTRY["process.selected_source_helicities"].dynamic_kind == "int"
    assert FIELD_REGISTRY["process.flavor_blind"].default is False
    assert FIELD_REGISTRY["process.consider_charge_conjugation"].default is True
    assert FIELD_REGISTRY["process.only_unit_multiplier"].default is False
    assert FIELD_REGISTRY["process.analyze_process_groups"].choices == (
        ProcessGroupAnalysis.NONE,
        ProcessGroupAnalysis.NUMERICAL,
    )
    assert "process.requests" not in FIELD_REGISTRY
    assert "process.names" not in FIELD_REGISTRY
    assert "evaluator.stage_local_parameter_layout" not in FIELD_REGISTRY
    assert {
        "color.coverage",
        "color.flow_ids",
        "generation.validation.zero_current_filter",
        "generation.validation.current_merging",
    }.isdisjoint(FIELD_REGISTRY)


def test_all_public_configuration_dataclasses_are_frozen() -> None:
    classes = (
        BenchmarkConfig,
        ColorConfig,
        CppConfig,
        EagerEvaluatorConfig,
        EvaluationConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationRelationDiscoveryConfig,
        GenerationValidationConfig,
        JITConfig,
        ModelConfig,
        OutputConfig,
        ProcessConfig,
        ProcessEntry,
        RecurrenceEvaluatorConfig,
        RunConfig,
        SymbolicaConfig,
    )
    assert all(is_dataclass(cls) and cls.__dataclass_params__.frozen for cls in classes)
    config = RunConfig(action="generate")
    with pytest.raises(FrozenInstanceError):
        config.action = "evaluate"  # type: ignore[misc]


def test_configuration_collections_are_immutable() -> None:
    entry = ProcessEntry(expression="d d~ > z g", name="ddbar_zg")
    process = ProcessConfig(
        entries=(entry,),
        multiparticles={"j": ["u", "d", "g"]},
        max_coupling_orders={"QCD": 2},
    )
    assert process.entries == (entry,)
    assert process.multiparticles["j"] == ("u", "d", "g")
    with pytest.raises(FrozenInstanceError):
        entry.name = "renamed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        process.max_coupling_orders["QED"] = 1  # type: ignore[index]


def test_process_entries_validate_names_and_uniqueness() -> None:
    with pytest.raises(ConfigurationError, match="must start with a letter"):
        ProcessEntry(expression="d d~ > z", name="1-invalid")
    with pytest.raises(ConfigurationError, match="duplicates: 'same'"):
        ProcessConfig(
            entries=(
                ProcessEntry("d d~ > z", "same"),
                ProcessEntry("u u~ > z", "same"),
            )
        )


def test_process_grouping_settings_are_typed() -> None:
    process = ProcessConfig()
    assert not process.flavor_blind
    assert process.consider_charge_conjugation
    assert not process.only_unit_multiplier
    assert process.analyze_process_groups is ProcessGroupAnalysis.NONE
    assert (
        ProcessConfig(analyze_process_groups="numerical").analyze_process_groups  # type: ignore[arg-type]
        is ProcessGroupAnalysis.NUMERICAL
    )
    with pytest.raises(ConfigurationError, match=r"process\.flavor_blind"):
        ProcessConfig(flavor_blind=1)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match=r"process\.analyze_process_groups"):
        ProcessConfig(analyze_process_groups="symbolic")  # type: ignore[arg-type]
    assert RunConfig(action="group-processes").action is Action.GROUP_PROCESSES


def test_schema_and_jit_settings_reject_wrong_boolean_integer_types() -> None:
    with pytest.raises(ConfigurationError, match="schema_version"):
        RunConfig(action="generate", schema_version=True)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="optimization_level"):
        JITConfig(optimization_level=True)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="compress"):
        JITConfig(compress=1)  # type: ignore[arg-type]


def test_contract_defaults_are_typed() -> None:
    config = RunConfig(action="evaluate")
    assert config.action is Action.EVALUATE
    assert config.color.accuracy is ColorAccuracy.LC
    assert config.color.contraction is ColorContraction.AUTO
    assert config.color.fft_basis is ColorFFTBasis.ADJOINT
    assert config.resolved_color.contraction is ColorContraction.DIRECT
    assert config.resolved_color.fft_basis is ColorFFTBasis.TRACE
    assert config.color.lc_flow_layout is LCFlowLayout.TOPOLOGY_REPLAY
    assert config.evaluator.backend is EvaluatorBackend.JIT
    assert config.evaluator.execution_mode is EvaluatorExecutionMode.RECURRENCE
    assert config.evaluator.jit.compress == "auto"
    assert config.evaluator.resolved_jit_compress is False
    assert config.evaluator.eager == EagerEvaluatorConfig()
    assert config.evaluator.recurrence == RecurrenceEvaluatorConfig()
    assert config.schema_version == 1
    assert config.generation.validation.samples == 2
    assert config.generation.relation_discovery.mode is RelationDiscoveryMode.OFF
    assert config.generation.relation_discovery.precision_digits == 96
    assert config.generation.relation_discovery.probe_count == 4
    assert config.generation.relation_discovery.verification_probe_count == 4
    assert config.generation.relation_discovery.relative_tolerance == 1.0e-70
    assert config.generation.relation_discovery.absolute_tolerance == 1.0e-80
    assert config.evaluator.output_chunk_size == 512
    assert not config.evaluator.cpp.native_arch
    assert config.evaluator.optimization.max_common_pair_cache_entries == 5_000_000
    assert config.benchmark.target_runtime == 10.0
    assert config.benchmark.precision == 16
    assert config.output == OutputConfig()


def test_jit_compression_auto_tracks_python_configuration_overrides() -> None:
    recurrence = EvaluatorConfig()
    compiled = replace(recurrence, execution_mode="compiled")

    assert recurrence.jit.compress == compiled.jit.compress == "auto"
    assert recurrence.resolved_jit_compress is False
    assert compiled.resolved_jit_compress is True
    assert (
        replace(compiled, jit=JITConfig(optimization_level=3)).resolved_jit_compress
        is False
    )
    assert (
        replace(compiled, jit=JITConfig(compress=False)).resolved_jit_compress is False
    )


def test_relation_discovery_defaults_off_and_validates_certification_policy() -> None:
    assert GenerationRelationDiscoveryConfig().mode is RelationDiscoveryMode.OFF
    assert (
        GenerationRelationDiscoveryConfig(mode="off").mode is RelationDiscoveryMode.OFF
    )
    assert (
        GenerationRelationDiscoveryConfig(mode="diagnostic").mode
        is RelationDiscoveryMode.DIAGNOSTIC
    )
    assert (
        GenerationRelationDiscoveryConfig(mode="certified-reuse").mode
        is RelationDiscoveryMode.CERTIFIED_REUSE
    )
    with pytest.raises(ConfigurationError, match="precision_digits"):
        GenerationRelationDiscoveryConfig(precision_digits=79)
    with pytest.raises(ConfigurationError, match="probe_count"):
        GenerationRelationDiscoveryConfig(probe_count=1)
    with pytest.raises(ConfigurationError, match="verification_probe_count"):
        GenerationRelationDiscoveryConfig(verification_probe_count=1)
    with pytest.raises(ConfigurationError, match="relative_tolerance"):
        GenerationRelationDiscoveryConfig(relative_tolerance=-1.0)
    with pytest.raises(ConfigurationError, match="absolute_tolerance"):
        GenerationRelationDiscoveryConfig(absolute_tolerance=-1.0)
    with pytest.raises(ConfigurationError, match="negative zero"):
        GenerationRelationDiscoveryConfig(
            relative_tolerance=-0.0,
            absolute_tolerance=1.0e-80,
        )
    with pytest.raises(ConfigurationError, match="negative zero"):
        GenerationRelationDiscoveryConfig(
            relative_tolerance=1.0e-70,
            absolute_tolerance=-0.0,
        )
    subnormal = GenerationRelationDiscoveryConfig(
        relative_tolerance=float.fromhex("0x0.0000000000001p-1022"),
        absolute_tolerance=0.0,
    )
    assert subnormal.relative_tolerance.hex() == "0x0.0000000000001p-1022"
    with pytest.raises(ConfigurationError, match="cannot both be zero"):
        GenerationRelationDiscoveryConfig(
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        )


def test_all_flow_union_layout_requires_lc_accuracy() -> None:
    assert (
        ColorConfig(lc_flow_layout="all-flow-union").lc_flow_layout
        is LCFlowLayout.ALL_FLOW_UNION
    )
    with pytest.raises(ConfigurationError, match=r"requires color\.accuracy='lc'"):
        ColorConfig(accuracy="nlc", lc_flow_layout="all-flow-union")
    with pytest.raises(ConfigurationError, match=r"requires color\.accuracy='lc'"):
        ColorConfig(accuracy="full", lc_flow_layout="all-flow-union")


@pytest.mark.parametrize("fft_basis", tuple(ColorFFTBasis))
def test_symmetric_group_fft_requires_contracted_color_and_supported_lane(
    fft_basis: ColorFFTBasis,
) -> None:
    with pytest.raises(ConfigurationError, match=r"requires color\.accuracy"):
        ColorConfig(contraction="symmetric-group-fft", fft_basis=fft_basis)

    for accuracy in ("nlc", "full"):
        for execution_mode in ("recurrence", "on-the-fly"):
            config = RunConfig(
                action="generate",
                color=ColorConfig(
                    accuracy=accuracy,
                    contraction="symmetric-group-fft",
                    fft_basis=fft_basis,
                ),
                evaluator=EvaluatorConfig(execution_mode=execution_mode),
            )
            assert config.color.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
            assert config.color.fft_basis is fft_basis

    compiled_direct = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        evaluator=EvaluatorConfig(execution_mode="compiled"),
    )
    assert compiled_direct.resolved_color.contraction is ColorContraction.DIRECT

    for execution_mode in ("compiled", "eager"):
        with pytest.raises(
            ConfigurationError,
            match=r"requires evaluator\.execution_mode='recurrence' or 'on-the-fly'",
        ):
            RunConfig(
                action="generate",
                process=ProcessConfig(selected_source_helicities={"1": 1}),
                color=ColorConfig(
                    accuracy="full",
                    contraction="symmetric-group-fft",
                    fft_basis=fft_basis,
                ),
                evaluator=EvaluatorConfig(execution_mode=execution_mode),
            )


def test_direct_contraction_ignores_the_default_adjoint_basis() -> None:
    config = RunConfig(
        action="generate", color=ColorConfig(accuracy="full", contraction="direct")
    )
    assert config.resolved_color.contraction is ColorContraction.DIRECT
    assert config.resolved_color.fft_basis is ColorFFTBasis.TRACE


def test_automatic_color_tracks_python_configuration_overrides() -> None:
    config = RunConfig(action="generate")
    full = replace(config, color=replace(config.color, accuracy="full"))
    assert full.resolved_color.contraction is ColorContraction.AUTO
    assert full.resolved_color.fft_basis is ColorFFTBasis.ADJOINT
    compiled = replace(
        full, evaluator=replace(full.evaluator, execution_mode="compiled")
    )
    assert compiled.resolved_color.contraction is ColorContraction.DIRECT
    assert compiled.resolved_color.fft_basis is ColorFFTBasis.TRACE


def test_fft_basis_rejects_unknown_values() -> None:
    with pytest.raises(ConfigurationError, match=r"color\.fft_basis"):
        ColorConfig(accuracy="full", fft_basis="other")


def test_on_the_fly_execution_supports_every_color_accuracy() -> None:
    selected_flow = RunConfig(
        action="generate",
        evaluator=EvaluatorConfig(execution_mode="on-the-fly"),
    )
    all_flow = RunConfig(
        action="generate",
        color=ColorConfig(lc_flow_layout="all-flow-union"),
        evaluator=EvaluatorConfig(execution_mode="on-the-fly"),
    )
    assert selected_flow.evaluator.execution_mode is EvaluatorExecutionMode.ON_THE_FLY
    assert all_flow.evaluator.execution_mode is EvaluatorExecutionMode.ON_THE_FLY
    for accuracy in ("nlc", "full"):
        contracted = RunConfig(
            action="generate",
            color=ColorConfig(accuracy=accuracy),
            evaluator=EvaluatorConfig(execution_mode="on-the-fly"),
        )
        assert contracted.color.accuracy.value == accuracy
        assert contracted.evaluator.execution_mode is EvaluatorExecutionMode.ON_THE_FLY


@pytest.mark.parametrize(
    ("field_name", "value"),
    (("point_tile_size", 0), ("workspace_mib", -1), ("workspace_mib", True)),
)
def test_eager_evaluator_sizes_must_be_positive_integers(
    field_name: str, value: object
) -> None:
    with pytest.raises(ConfigurationError, match=rf"evaluator\.eager\.{field_name}"):
        EagerEvaluatorConfig(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    (("point_tile_size", 0), ("workspace_mib", -1), ("workspace_mib", True)),
)
def test_recurrence_evaluator_sizes_must_be_positive_integers(
    field_name: str, value: object
) -> None:
    with pytest.raises(
        ConfigurationError, match=rf"evaluator\.recurrence\.{field_name}"
    ):
        RecurrenceEvaluatorConfig(**{field_name: value})  # type: ignore[arg-type]


def test_cpp_flags_cannot_hide_target_cpu_requirements() -> None:
    assert CppConfig(extra_flags=("-fno-math-errno",)).extra_flags == (
        "-fno-math-errno",
    )
    with pytest.raises(ConfigurationError, match="unrecorded target CPU"):
        CppConfig(extra_flags=("-march=native",))
    with pytest.raises(ConfigurationError, match="unsupported compiler arguments"):
        CppConfig(extra_flags=("-mavx2",))
