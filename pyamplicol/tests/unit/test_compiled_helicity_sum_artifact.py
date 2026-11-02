# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pyamplicol.generation.artifact_writer as artifact_writer
import pyamplicol.generation.service as service_module
import pyamplicol.models.loading as model_loading
from pyamplicol._internal.versions import (
    COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY,
    COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY,
    COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
    COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
    COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
    COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
    COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    EVALUATOR_RUNTIME_CAPABILITIES,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
    SYMJIT_PLANE_APPLICATION_ABI,
)
from pyamplicol.api import ModelSource, ProcessRequest
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.manifest import PORTABLE_64LE_TARGET
from pyamplicol.config import (
    ColorConfig,
    ConfigurationError,
    EvaluatorConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    RunConfig,
)
from pyamplicol.generation.artifact_writer import (
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from pyamplicol.generation.contracts import RuntimeExpressionSchema
from pyamplicol.generation.evaluator_container import PacbinReader
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.service import _ProcessSelection
from pyamplicol.generation.stage_artifacts import _compiled_plane_arena_stage
from pyamplicol.models import BuiltinSMModel, compile_model_source
from pyamplicol.models.builtin.process_ir import build_process_ir


def _evaluator_process(
    expression: str = "d d~ > z g",
    *,
    selection: _ProcessSelection | None = None,
    config: GenerationConfig | RunConfig | None = None,
) -> tuple[
    service_module.GenerationBackend,
    BuiltinSMModel,
    service_module._EvaluatorProcess,
]:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        GenerationConfig() if config is None else config,
        None,
        process_selection=selection,
    )
    color_accuracy = (
        config.color.accuracy.value if isinstance(config, RunConfig) else "lc"
    )
    process_ir = build_process_ir(expression, color_accuracy=color_accuracy)
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(
            expanded=service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name="dual_lane"),
                process_ir=process_ir,
            ),
            dag=dag,
            coverage=coverage,
        ),
        model,
        index=0,
        phase=PhaseHandle("test", None, 1),
    )
    evaluator = backend._construct_evaluator(
        prepared,
        model,
        PhaseHandle("test", None, 1),
    )
    return backend, model, evaluator


def _symjit_stage_manifest(
    root: Path,
    *,
    label: str,
    optimization_level: int = 3,
    value_component_count: int = 0,
) -> dict[str, object]:
    evaluator_dir = root / "evaluators"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    application = evaluator_dir / f"{label}.symjit"
    plane_application = evaluator_dir / f"{label}.plane.symjit"
    state = evaluator_dir / f"{label}.evaluator.bin"
    application.write_bytes(f"application:{label}".encode())
    plane_application.write_bytes(f"plane-application:{label}".encode())
    state.write_bytes(f"state:{label}".encode())
    plane_source_digest = hashlib.sha256(f"instructions:{label}".encode()).hexdigest()
    evaluator = {
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
        "input_len": 1,
        "output_len": 1,
        "application_path": application.relative_to(root).as_posix(),
        "application_abi": SYMJIT_APPLICATION_ABI,
        "plane_application": {
            "application_path": plane_application.relative_to(root).as_posix(),
            "application_abi": SYMJIT_PLANE_APPLICATION_ABI,
            "storage_abi": SYMJIT_APPLICATION_ABI,
            "element_layout": "split-complex-plane-major",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "input_complex_count": 1,
            "output_complex_count": 1,
            "input_plane_count": 2,
            "output_plane_count": 2,
            "compiler_type": "native",
            "translation_mode": "symbolica-structured-instructions",
            "optimization_level": optimization_level,
            "simd": True,
            "complex": True,
            "fast_math": True,
            "fast_complex": False,
            "compression": False,
            "threading": False,
            "direct_arena": True,
            "source_digest": plane_source_digest,
            "target": {"word_bits": 64, "endianness": "little"},
        },
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": optimization_level,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": state.relative_to(root).as_posix(),
        "evaluator_state_runtime_capability": (SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY),
    }
    input_component = {
        "kind": "momentum",
        "source_id": 0,
        "component": 0,
        "global_component": value_component_count,
        "parameter_index": 0,
        "real_valued": True,
    }
    amplitude_stage = {
        "stage_index": 0,
        "stage_kind": "amplitude",
        "subset_size": None,
        "evaluator_label": label,
        "parameter_layout": "stage-local-value-momentum",
        "output_length": 1,
        "output_slots": [
            {
                "value_slot_id": -1,
                "current_id": -1,
                "variant": "amplitude-root",
                "component_start": 0,
                "component_stop": 1,
                "output_start": 0,
                "output_stop": 1,
                "color_selector_domain_ids": [],
            }
        ],
        "input_value_slot_ids": [],
        "output_value_slot_ids": [],
        "interaction_ids": [],
        "input_components": [input_component],
        "parameter_count": 1,
        "value_parameter_count": 0,
        "momentum_parameter_count": 1,
        "model_parameter_count": 0,
        "real_valued_inputs": [0],
        "expression_ready": True,
        "blockers": [],
        "evaluator": evaluator,
    }
    direct = _compiled_plane_arena_stage(amplitude_stage)
    assert direct is not None
    amplitude_stage["compiled_plane_arena"] = direct
    return {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "parameter_count": 0,
        "value_parameter_count": 0,
        "momentum_parameter_count": 0,
        "model_parameter_count": 0,
        "real_valued_inputs": [],
        "parameter_layout": "stage-local-value-momentum",
        "stage_count": 1,
        "required_runtime_capabilities": [
            COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            SYMJIT_F64_RUNTIME_CAPABILITY,
        ],
        "stages": [],
        "amplitude_stage": amplitude_stage,
    }


def _materialize_without_symbolica(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    expression: str = "d d~ > z g",
    selection: _ProcessSelection | None = None,
    config: GenerationConfig | RunConfig | None = None,
    jit_optimization_level: int = 3,
) -> service_module.CompiledProcessArtifact:
    backend, model, evaluator = _evaluator_process(
        expression,
        selection=selection,
        config=config,
    )
    calls: list[tuple[Path, int]] = []

    def compile_stages(
        stage_input: object,
        runtime_schema: dict[str, object],
        root: Path,
        **_kwargs: object,
    ) -> tuple[object, dict[str, object]]:
        lane = "-".join(root.relative_to(tmp_path / "build").parts).replace(
            ".",
            "",
        )
        calls.append((root, len(stage_input.dag.currents)))  # type: ignore[attr-defined]
        layout = runtime_schema["parameter_layout"]
        assert isinstance(layout, dict)
        return object(), _symjit_stage_manifest(
            root,
            label=lane,
            optimization_level=jit_optimization_level,
            value_component_count=int(layout["value_component_count"]),
        )

    monkeypatch.setattr(
        service_module,
        "build_and_write_generic_stage_evaluator_artifacts",
        compile_stages,
    )
    monkeypatch.setattr(
        service_module,
        "write_model_parameter_evaluator_artifact",
        lambda *_args, **_kwargs: None,
    )
    artifact = backend._materialize_evaluator_unlocked(
        evaluator,
        model,
        tmp_path / "build",
        PhaseHandle("jit", None, None),
        PhaseHandle("process", None, None),
        backend="JIT",
    )

    expected_calls = [
        (tmp_path / "build" / "dual_lane", len(evaluator.compiled.dag.currents))
    ]
    if evaluator.compiled.helicity_sum_dag is not None:
        expected_calls.append(
            (
                tmp_path / "build" / ".helicity-sum" / "dual_lane",
                len(evaluator.compiled.helicity_sum_dag.currents),
            )
        )

    def add_selector_lane_call(
        lane: service_module._HelicitySelectorLane,
        root: Path,
    ) -> None:
        expected_calls.append((root, len(lane.dag.currents)))
        for child_index, child in enumerate(lane.child_lanes):
            add_selector_lane_call(
                child,
                root.parent / f"{root.name}-closure-{child_index}",
            )

    selector_root = tmp_path / "build" / ".helicity-selector-union" / "dual_lane"
    for lane_index, lane in enumerate(evaluator.helicity_selector_lanes):
        add_selector_lane_call(lane, selector_root / f"class-{lane_index}")
    selector_root = (
        ".helicity-sum-color-selector"
        if evaluator.compiled.helicity_sum_dag is not None
        else ".color-selector"
    )
    expected_calls.extend(
        (
            tmp_path
            / "build"
            / selector_root
            / "dual_lane"
            / f"sector-{lane.materialized_sector_id}",
            len(lane.dag.currents),
        )
        for lane in evaluator.color_selector_lanes
    )
    assert calls == expected_calls
    return artifact


@pytest.mark.parametrize("value_component_count", [0, 12])
def test_mock_symjit_manifest_preserves_the_compiled_arena_invariant(
    tmp_path: Path,
    value_component_count: int,
) -> None:
    manifest = _symjit_stage_manifest(
        tmp_path,
        label="authenticated",
        value_component_count=value_component_count,
    )
    amplitude_stage = manifest["amplitude_stage"]
    assert isinstance(amplitude_stage, dict)
    evaluator = amplitude_stage["evaluator"]
    assert isinstance(evaluator, dict)
    direct = amplitude_stage["compiled_plane_arena"]
    assert isinstance(direct, dict)
    leaves = direct["leaves"]
    assert isinstance(leaves, list)
    plane_application = evaluator["plane_application"]
    assert isinstance(plane_application, dict)
    assert leaves == [
        {
            "application_path": plane_application["application_path"],
            "source_application_abi": plane_application["application_abi"],
            "optimization_level": plane_application["optimization_level"],
            "direct_codegen_optimization_level": plane_application[
                "optimization_level"
            ],
            "input_len": evaluator["input_len"],
            "output_len": evaluator["output_len"],
            "input_indices": [0],
            "output_start": 0,
            "output_stop": 1,
        }
    ]
    assert (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY
        in manifest["required_runtime_capabilities"]
    )
    assert artifact_writer._compiled_momentum_slot_ids(
        manifest,
        value_component_count=value_component_count,
        momentum_component_count=4,
    ) == [0]

    del amplitude_stage["compiled_plane_arena"]
    with pytest.raises(ValueError, match="compiled f64 artifacts require"):
        artifact_writer._stage_evaluator_set(manifest)


def _payload_paths(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "application_path",
                "evaluator_state_path",
                "library_path",
            } and isinstance(item, str):
                result.add(item)
            else:
                result.update(_payload_paths(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_payload_paths(item))
    return result


def test_generation_retains_selector_and_fused_helicity_sum_dags() -> None:
    _backend, _model, evaluator = _evaluator_process()
    prepared = evaluator.compiled
    assert prepared.helicity_sum_dag is not None
    assert prepared.helicity_sum_dag.helicity_recurrence is None
    assert prepared.helicity_sum_dag.helicity_materialization is None
    assert prepared.dag.helicity_materialization is not None
    assert prepared.dag.helicity_materialization.strategy == "quotient"
    assert prepared.dag.helicity_materialization.materialized_current_count == len(
        prepared.dag.currents
    )
    assert prepared.dag.helicity_materialization.materialized_root_count == len(
        prepared.dag.amplitude_roots
    )

    primary = evaluator.runtime_schema.to_mapping()
    assert evaluator.helicity_sum_runtime_schema is not None
    assert evaluator.helicity_sum_stage_input is not None
    assert "materialization" in primary["helicity_recurrence"]
    assert primary["helicity_recurrence"]["materialization"]["strategy"] == "quotient"
    assert "helicity_recurrence" not in (
        evaluator.helicity_sum_runtime_schema.to_mapping()
    )

    assert evaluator.helicity_selector_lanes
    materialization = prepared.dag.helicity_materialization
    assert materialization is not None
    nonzero_domains = {
        schedule.selector_domain_id
        for schedule in materialization.selector_schedules
        if not schedule.structural_zero
    }
    assert {
        domain_id
        for lane in evaluator.helicity_selector_lanes
        for domain_id in lane.selector_domain_ids
    } == nonzero_domains
    assert len(evaluator.helicity_selector_lanes) == len(
        {
            (schedule.active_current_ids, schedule.active_root_ids)
            for schedule in materialization.selector_schedules
            if not schedule.structural_zero
        }
    )
    for lane in evaluator.helicity_selector_lanes:
        selector_input = lane.stage_input
        assert selector_input.dag.currents == prepared.dag.currents
        assert selector_input.dag.interactions == prepared.dag.interactions
        assert selector_input.dag.amplitude_roots == prepared.dag.amplitude_roots
        assert selector_input.dag.helicity_recurrence is None
        assert selector_input.dag.helicity_materialization is None
        selector_schema = lane.runtime_schema.to_mapping()
        assert "helicity_recurrence" not in selector_schema
        assert all(
            int(stage["interaction_count"]) < int(parent_stage["interaction_count"])
            for stage, parent_stage in zip(
                selector_schema["stages"],
                primary["stages"],
                strict=True,
            )
        )


def test_all_flow_union_artifact_omits_duplicate_execution_lanes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="d d~ > z g g",
        config=RunConfig(
            action="generate",
            color=ColorConfig(lc_flow_layout="all-flow-union"),
            evaluator=EvaluatorConfig(execution_mode="compiled"),
            generation=GenerationConfig(
                emit_api_bundle=False,
            ),
        ),
    )

    assert artifact.helicity_sum_execution is None
    assert artifact.helicity_selector_executions
    assert all(
        record.schedule_mode == "parent-closure"
        for record in artifact.helicity_selector_executions
    )
    assert artifact.color_selector_executions == ()
    capabilities = artifact_writer._compiled_process_runtime_capabilities(artifact)
    assert COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY in capabilities
    execution = artifact_writer._execution_manifest(
        artifact,
        artifact.runtime_schema.to_mapping(),
    )
    assert (
        COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
        in execution["required_runtime_capabilities"]
    )
    assert (
        COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
        not in execution["compiled"]["stage_evaluators"][
            "required_runtime_capabilities"
        ]
    )


def test_color_topology_lane_capability_is_publicly_supported() -> None:
    assert COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY in EVALUATOR_RUNTIME_CAPABILITIES


def test_walsh_color_contraction_capability_is_publicly_supported() -> None:
    assert COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY in EVALUATOR_RUNTIME_CAPABILITIES
    assert (
        COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY
        in EVALUATOR_RUNTIME_CAPABILITIES
    )


def test_helicity_selector_union_capability_is_publicly_supported() -> None:
    assert COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY in EVALUATOR_RUNTIME_CAPABILITIES


def test_primary_helicity_recurrence_capability_is_publicly_supported() -> None:
    assert (
        COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
        in EVALUATOR_RUNTIME_CAPABILITIES
    )


def test_complete_lc_union_owns_childless_materialized_selector_lane() -> None:
    _backend, model, evaluator = _evaluator_process("d d~ > z g g g")

    assert len(evaluator.helicity_selector_lanes) == 1
    union = evaluator.helicity_selector_lanes[0]
    assert union.schedule_mode == "nested-runtime"
    assert union.child_lanes == ()
    assert union.dag.helicity_materialization is not None
    assert union.dag.lc_topology_replay is None
    assert union.dag.color_topology_replay is None
    assert (
        service_module._compiled_helicity_selector_closure_lanes(
            evaluator.compiled.dag,
            evaluator.runtime_schema,
            model,
        )
        == ()
    )


@pytest.mark.parametrize("accuracy", ["nlc", "full"])
def test_replayed_contracted_color_uses_physical_childless_selector_companion(
    accuracy: str,
) -> None:
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy),
        evaluator=EvaluatorConfig(execution_mode="compiled"),
        generation=GenerationConfig(
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
        ),
    )
    _backend, _model, evaluator = _evaluator_process(
        "g g > g g",
        config=config,
    )

    primary = evaluator.compiled.dag
    assert primary.color_topology_replay is not None
    assert len(evaluator.helicity_selector_lanes) == 1
    companion = evaluator.helicity_selector_lanes[0]
    assert companion.schedule_mode == "nested-runtime"
    assert companion.child_lanes == ()
    assert companion.dag.process.color_accuracy == accuracy
    assert companion.dag.selected_source_helicities == ()
    assert companion.dag.helicity_coverage == "complete"
    assert companion.dag.color_coverage == "complete"
    assert companion.dag.lc_topology_replay is None
    assert companion.dag.color_topology_replay is None
    assert companion.dag.helicity_materialization is not None
    assert companion.dag.color_plan.sectors == primary.color_plan.sectors
    service_module._validate_matching_helicity_selector_domains(
        primary,
        companion.dag,
    )


def test_childless_nested_helicity_selector_is_written_with_owned_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="d d~ > z g g g",
        jit_optimization_level=2,
    )
    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: (
            {"triple": "aarch64-apple-darwin", "cpu_features": []},
            1,
        ),
    )
    monkeypatch.setattr(model_loading, "package_version", lambda: "0.1.4")
    monkeypatch.setattr(artifact_writer, "package_version", lambda: "0.1.4")
    monkeypatch.setattr(
        artifact_writer,
        "active_native_build_inputs_sha256",
        lambda: "0" * 64,
    )
    output = tmp_path / "artifact"
    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.built_in_sm(),
        compiled_model=compile_model_source("built-in-sm", use_cache=False),
        configuration=_GenerationConfigProvenance.from_config(
            GenerationConfig(emit_api_bundle=False)
        ),
        processes=(artifact,),
        timings={"total": 0.1},
        api_bundle_hook=None,
    )

    execution = json.loads(
        (output / "processes/dual_lane/execution.json").read_text(encoding="utf-8")
    )
    outer = execution["helicity_selector_executions"][0]
    assert outer["schedule_mode"] == "nested-runtime"
    assert "helicity_selector_executions" not in outer["execution"]

    referenced = _payload_paths(execution)
    nested_prefix = "helicity-selector-union/class-0/"
    assert any(path.startswith(nested_prefix) for path in referenced)
    assert not any(
        path.startswith(f"{nested_prefix}helicity-selector-union/")
        for path in referenced
    )
    manifest = load_manifest(output)
    portable_target = {
        "triple": PORTABLE_64LE_TARGET,
        "cpu_features": (),
    }
    assert manifest.producer["target"] == portable_target
    targeted_payloads = [
        record for record in manifest.payloads if record.target is not None
    ]
    assert targeted_payloads
    assert all(record.target == portable_target for record in targeted_payloads)
    declared = {record.path for record in manifest.payloads}
    assert "evaluators.pacbin" in declared
    with PacbinReader.open(output / "evaluators.pacbin") as container:
        members = {member.logical_path for member in container.members}
    for relative in referenced:
        assert f"processes/dual_lane/{relative}" in members


def test_compiled_symmetric_group_is_rejected_before_generation() -> None:
    with pytest.raises(
        ConfigurationError,
        match=r"requires evaluator\.execution_mode='recurrence' or 'on-the-fly'",
    ):
        RunConfig(
            action="generate",
            color=ColorConfig(
                accuracy="full",
                contraction="symmetric-group-fft",
            ),
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        )


def test_generation_specialized_color_artifact_has_no_topology_lanes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="d d~ > z g g",
        selection=_ProcessSelection(selected_color_sector_ids=frozenset({0})),
    )

    assert artifact.color_selector_executions == ()
    assert artifact.helicity_selector_executions
    if artifact.helicity_sum_execution is not None:
        assert artifact.helicity_sum_execution.color_selector_executions == ()
    manifest = artifact_writer._execution_manifest(
        artifact,
        artifact.runtime_schema.to_mapping(),
    )
    assert "color_selector_executions" not in manifest
    auxiliary = manifest.get("helicity_sum_execution")
    if isinstance(auxiliary, dict):
        assert "color_selector_executions" not in auxiliary
    assert (
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY
        not in manifest["required_runtime_capabilities"]
    )


def test_complete_lc_artifact_serializes_every_materialized_sector_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _backend, _model, evaluator = _evaluator_process("g g > g g")
    target_dag = evaluator.compiled.helicity_sum_dag or evaluator.compiled.dag
    expected_sector_ids = tuple(
        sorted(
            {
                int(root.color_sector_id)
                for root in target_dag.amplitude_roots
                if root.color_sector_id is not None
            }
        )
    )
    assert expected_sector_ids
    assert (
        tuple(lane.materialized_sector_id for lane in evaluator.color_selector_lanes)
        == expected_sector_ids
    )

    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="g g > g g",
    )
    manifest = artifact_writer._execution_manifest(
        artifact,
        artifact.runtime_schema.to_mapping(),
    )
    parent = manifest.get("helicity_sum_execution", manifest)
    records = parent["color_selector_executions"]
    assert (
        tuple(record["materialized_sector_id"] for record in records)
        == expected_sector_ids
    )
    assert all(
        "color_selector_executions" not in record["execution"]
        and "helicity_sum_execution" not in record["execution"]
        and "physics_reduction" in record["execution"]
        for record in records
    )


def test_fixed_helicity_complete_lc_attaches_color_lanes_to_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(
        monkeypatch,
        tmp_path,
        expression="d d~ > z g g",
        selection=_ProcessSelection(
            selected_source_helicities={
                1: -1,
                2: 1,
                3: -1,
                4: 1,
                5: -1,
            }
        ),
    )

    assert artifact.helicity_sum_execution is None
    assert artifact.helicity_selector_executions == ()
    assert artifact.color_selector_executions
    manifest = artifact_writer._execution_manifest(
        artifact,
        artifact.runtime_schema.to_mapping(),
    )
    assert "helicity_sum_execution" not in manifest
    assert manifest["color_selector_executions"]
    assert (
        COMPILED_RUNTIME_SELECTORS_CAPABILITY
        in manifest["required_runtime_capabilities"]
    )


def test_eager_plan_v3_artifacts_have_no_compiled_selector_union_lane() -> None:
    assert "helicity_selector_executions" not in (
        service_module.EagerPlanV3ProcessArtifact.__dataclass_fields__
    )


def test_replayed_physical_flows_compile_only_materialized_sector_lanes() -> None:
    _backend, _model, evaluator = _evaluator_process("d d~ > z g g")
    target_dag = evaluator.compiled.helicity_sum_dag or evaluator.compiled.dag
    replay = target_dag.lc_topology_replay

    assert replay is not None
    assert len(replay.physical_sector_ids) > len(replay.materialized_sector_ids)
    assert (
        tuple(lane.materialized_sector_id for lane in evaluator.color_selector_lanes)
        == replay.materialized_sector_ids
    )


def test_color_selector_execution_lanes_cannot_nest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    assert artifact.helicity_sum_execution is not None
    record = artifact.helicity_sum_execution.color_selector_executions[0]
    recursive_record = replace(
        record,
        execution=replace(
            record.execution,
            color_selector_executions=(record,),
        ),
    )
    invalid = replace(
        artifact,
        helicity_sum_execution=replace(
            artifact.helicity_sum_execution,
            color_selector_executions=(recursive_record,),
        ),
    )

    with pytest.raises(ValueError, match="cannot nest"):
        artifact_writer._execution_manifest(
            invalid,
            invalid.runtime_schema.to_mapping(),
        )


def test_compiled_materialization_builds_primary_sum_and_selector_union_lanes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    assert artifact.helicity_sum_execution is not None
    assert artifact.helicity_selector_executions
    assert (
        artifact.dag_summary["current_count"]
        == (
            artifact.runtime_schema.to_mapping()["helicity_recurrence"][
                "materialization"
            ]["materialized_current_count"]
        )
    )
    assert "helicity_recurrence" not in (
        artifact.helicity_sum_execution.runtime_schema.to_mapping()
    )
    primary_schema = artifact.runtime_schema.to_mapping()
    for record in artifact.helicity_selector_executions:
        assert record.schedule_mode == "parent-closure"
        selector = record.execution
        selector_schema = selector.runtime_schema.to_mapping()
        assert "helicity_recurrence" not in selector_schema
        assert selector.helicity_selector_executions == ()
        assert (
            selector_schema["physics"]["reduction"]
            == (primary_schema["physics"]["reduction"])
        )
        assert selector.dag_summary["current_count"] == len(
            selector_schema["current_storage"]["current_slots"]
        )
        assert (
            selector.dag_summary["amplitude_root_count"]
            == (selector_schema["amplitude_stage"]["output_count"])
        )
        assert (
            COMPILED_RUNTIME_SELECTORS_CAPABILITY
            not in (selector.stage_manifest["required_runtime_capabilities"])
        )
    execution_manifest = artifact_writer._execution_manifest(
        artifact,
        primary_schema,
    )
    assert execution_manifest["materialization_census"]["final"] == {
        name: value
        for name, value in execution_manifest["dag_summary"].items()
        if name.endswith("_count")
    }
    assert execution_manifest["helicity_sum_execution"]["materialization_census"][
        "final"
    ] == {
        name: value
        for name, value in execution_manifest["helicity_sum_execution"][
            "dag_summary"
        ].items()
        if name.endswith("_count")
    }
    selector_records = execution_manifest["helicity_selector_executions"]
    assert len(selector_records) == len(artifact.helicity_selector_executions)
    for selector_record in selector_records:
        assert selector_record["selector_domain_ids"]
        assert selector_record["schedule_mode"] == "parent-closure"
        selector_manifest = selector_record["execution"]
        assert selector_manifest["kind"] == "pyamplicol-runtime-execution"
        assert selector_manifest["materialization_census"]["final"] == {
            name: value
            for name, value in selector_manifest["dag_summary"].items()
            if name.endswith("_count")
        }
        assert (
            selector_manifest["physics_reduction"]
            == (primary_schema["physics"]["reduction"])
        )
        assert "helicity_sum_execution" not in selector_manifest
        assert "helicity_selector_executions" not in selector_manifest
        assert "color_selector_executions" not in selector_manifest


def test_helicity_selector_union_execution_cannot_nest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    assert artifact.helicity_selector_executions
    assert artifact.helicity_sum_execution is not None
    nested_color_lane = artifact.helicity_sum_execution.color_selector_executions[0]
    first = artifact.helicity_selector_executions[0]
    invalid_first = replace(
        first,
        execution=replace(
            first.execution,
            color_selector_executions=(nested_color_lane,),
        ),
    )
    invalid = replace(
        artifact,
        helicity_selector_executions=(
            invalid_first,
            *artifact.helicity_selector_executions[1:],
        ),
    )

    with pytest.raises(ValueError, match="cannot nest"):
        artifact_writer._execution_manifest(
            invalid,
            invalid.runtime_schema.to_mapping(),
        )


def test_compiled_process_capabilities_use_primary_execution_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    assert artifact_writer._compiled_process_runtime_capabilities(artifact) == (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
    )


@pytest.mark.parametrize(
    ("factorized_block", "walsh_capability"),
    [
        (
            {
                "kind": "klein-four-walsh",
                "cosets": [[0, 1, 2, 3]],
            },
            COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY,
        ),
        (
            {
                "kind": "elementary-abelian-walsh",
                "rank": 3,
                "cosets": [list(range(8))],
            },
            COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY,
        ),
    ],
)
def test_walsh_color_contraction_capability_is_lane_local_and_parent_aggregated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    factorized_block: dict[str, object],
    walsh_capability: str,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)

    def with_walsh(
        schema: RuntimeExpressionSchema,
    ) -> RuntimeExpressionSchema:
        payload = schema.to_mapping()
        amplitude_stage = payload["amplitude_stage"]
        assert isinstance(amplitude_stage, dict)
        cosets = factorized_block["cosets"]
        assert isinstance(cosets, list)
        local_group_count = 1 + max(int(index) for coset in cosets for index in coset)
        group_count = 2 * local_group_count
        amplitude_stage["color_contraction"] = {
            "color_accuracy": "full",
            "supported": True,
            "reason": None,
            "group_count": group_count,
            "includes_color_factor": True,
            "entry_count": 0,
            "logical_entry_count": 0,
            "storage": "test",
            "entries": [],
            "repeated_block": {
                "component_count": 2,
                "component_group_ids": list(range(group_count)),
                "entries": [],
                "factorized_block": factorized_block,
            },
        }
        return RuntimeExpressionSchema.from_mapping(payload)

    old_manifest = artifact_writer._execution_manifest(
        artifact,
        artifact.runtime_schema.to_mapping(),
    )
    assert walsh_capability not in old_manifest["required_runtime_capabilities"]

    primary = replace(artifact, runtime_schema=with_walsh(artifact.runtime_schema))
    primary_manifest = artifact_writer._execution_manifest(
        primary,
        primary.runtime_schema.to_mapping(),
    )
    assert walsh_capability in primary_manifest["required_runtime_capabilities"]
    assert (
        walsh_capability
        not in primary_manifest["compiled"]["stage_evaluators"][
            "required_runtime_capabilities"
        ]
    )
    assert walsh_capability in artifact_writer._compiled_process_runtime_capabilities(
        primary
    )

    assert artifact.helicity_sum_execution is not None
    auxiliary = replace(
        artifact.helicity_sum_execution,
        runtime_schema=with_walsh(artifact.helicity_sum_execution.runtime_schema),
    )
    parent = replace(artifact, helicity_sum_execution=auxiliary)
    parent_manifest = artifact_writer._execution_manifest(
        parent,
        parent.runtime_schema.to_mapping(),
    )
    auxiliary_manifest = parent_manifest["helicity_sum_execution"]
    assert walsh_capability in auxiliary_manifest["required_runtime_capabilities"]
    assert walsh_capability in parent_manifest["required_runtime_capabilities"]
    assert walsh_capability in artifact_writer._compiled_process_runtime_capabilities(
        parent
    )


def test_writer_emits_selector_and_fused_sum_lanes_and_owns_all_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _materialize_without_symbolica(monkeypatch, tmp_path)
    primary_schema = artifact.runtime_schema.to_mapping()
    execution_manifest = artifact_writer._execution_manifest(
        artifact,
        primary_schema,
    )
    assert "helicity_sum_execution" in execution_manifest
    assert (
        execution_manifest["runtime_schema"]["helicity_recurrence"]["materialization"][
            "strategy"
        ]
        == "quotient"
    )

    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: (
            {"triple": "aarch64-apple-darwin", "cpu_features": []},
            1,
        ),
    )
    output = tmp_path / "artifact"
    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.built_in_sm(),
        compiled_model=compile_model_source("built-in-sm", use_cache=False),
        configuration=_GenerationConfigProvenance.from_config(
            GenerationConfig(emit_api_bundle=False)
        ),
        processes=(artifact,),
        timings={"total": 0.1},
        api_bundle_hook=None,
    )

    execution = json.loads(
        (output / "processes/dual_lane/execution.json").read_text(encoding="utf-8")
    )
    assert "helicity_sum_execution" in execution
    assert "helicity_selector_executions" in execution
    assert execution["runtime_schema"]["process_key"] == "dual_lane"
    assert (
        execution["runtime_schema"]["helicity_recurrence"]["materialization"][
            "strategy"
        ]
        == "quotient"
    )
    primary_evaluator = execution["compiled"]["stage_evaluators"]["amplitude_stage"][
        "evaluator"
    ]
    assert primary_evaluator["optimization_level"] == 3
    helicity_sum_evaluator = execution["helicity_sum_execution"]["compiled"][
        "stage_evaluators"
    ]["amplitude_stage"]["evaluator"]
    assert helicity_sum_evaluator["optimization_level"] == 3
    assert (
        "helicity_recurrence"
        not in execution["helicity_sum_execution"]["runtime_schema"]
    )
    selector_records = execution["helicity_selector_executions"]
    assert selector_records
    for record in selector_records:
        selector_lane = record["execution"]
        assert record["selector_domain_ids"]
        assert record["schedule_mode"] == "parent-closure"
        assert "helicity_recurrence" not in selector_lane["runtime_schema"]
        assert "helicity_selector_executions" not in selector_lane
        assert (
            selector_lane["physics_reduction"] == primary_schema["physics"]["reduction"]
        )
        assert (
            COMPILED_RUNTIME_SELECTORS_CAPABILITY
            not in selector_lane["required_runtime_capabilities"]
        )
    color_selector_records = execution["helicity_sum_execution"][
        "color_selector_executions"
    ]
    assert artifact.helicity_sum_execution is not None
    assert [record["materialized_sector_id"] for record in color_selector_records] == [
        0
    ]
    color_selector_execution = color_selector_records[0]["execution"]
    assert "helicity_sum_execution" not in color_selector_execution
    assert "color_selector_executions" not in color_selector_execution
    assert (
        color_selector_execution["physics_reduction"]
        == (
            artifact.helicity_sum_execution.color_selector_executions[
                0
            ].execution.runtime_schema.to_mapping()["physics"]["reduction"]
        )
    )

    manifest = load_manifest(output)
    declared = {record.path for record in manifest.payloads}
    referenced = _payload_paths(execution)
    assert referenced
    assert any(path.startswith("helicity-sum/") for path in referenced)
    assert any(path.startswith("helicity-selector-union/") for path in referenced)
    assert any(
        path.startswith("helicity-sum/color-selector/sector-0/") for path in referenced
    )
    assert "evaluators.pacbin" in declared
    with PacbinReader.open(output / "evaluators.pacbin") as container:
        members = {member.logical_path for member in container.members}
    for relative in referenced:
        assert f"processes/dual_lane/{relative}" in members
    assert execution["required_runtime_capabilities"] == [
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
    ]
    generation = manifest.extensions["generation"]["concrete_processes"][0]
    assert generation["runtime_schema_sha256"] == artifact.runtime_schema.sha256
    assert artifact.helicity_sum_execution is not None
    assert generation["helicity_sum_runtime_schema_sha256"] == (
        artifact.helicity_sum_execution.runtime_schema.sha256
    )
    assert artifact.helicity_selector_executions
    assert generation["helicity_selector_runtime_schema_sha256s"] == [
        record.execution.runtime_schema.sha256
        for record in artifact.helicity_selector_executions
    ]
