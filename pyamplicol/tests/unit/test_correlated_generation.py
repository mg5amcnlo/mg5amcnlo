# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest

import pyamplicol.generation.correlated as correlated
import pyamplicol.generation.service as service
from pyamplicol.api import ProcessAlias, ProcessRequest, ProcessSet
from pyamplicol.api.errors import GenerationError
from pyamplicol.artifacts import ArtifactBuilder, load_manifest
from pyamplicol.color.plan import build_color_plan
from pyamplicol.config import (
    Action,
    ColorConfig,
    ConfigClamp,
    ConfigResolution,
    GenerationConfig,
    GenerationValidationConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.correlators import ColorCorrelator, CorrelatorConfig
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def test_correlated_settings_are_explicit_and_do_not_change_default_factory() -> None:
    ordinary = service.create_generator_backend(None, None)
    assert type(ordinary) is service.GenerationBackend
    assert ordinary._color_accuracy == "lc"
    settings = correlated.correlated_configuration(None)
    assert settings.requested.color.accuracy == "lc"
    assert settings.effective.color.accuracy == "lc"
    assert settings.effective.color.contraction == "direct"
    assert settings.effective.color.fft_basis == "trace"
    assert settings.effective.evaluator.execution_mode == "compiled"
    assert settings.effective.evaluator.jit.compress is True
    assert settings.effective.generation.relation_discovery.mode == "off"
    assert "color.accuracy" not in {change.path for change in settings.clamps}
    assert "evaluator.execution_mode" in {change.path for change in settings.clamps}
    assert not hasattr(ordinary, "declarations")


@pytest.mark.parametrize("accuracy", ("lc", "nlc", "full"))
def test_output_accuracy_is_retained_but_underlying_generation_stays_full(accuracy):
    config = RunConfig(action=Action.GENERATE, color=ColorConfig(accuracy=accuracy))
    settings = correlated.correlated_configuration(config)
    assert (
        settings.requested.color.accuracy
        == settings.effective.color.accuracy
        == accuracy
    )
    backend = correlated.CorrelatedGenerationBackend(
        config, None, declarations=CorrelatorConfig()
    )
    assert backend._run_config.color.accuracy == accuracy
    assert backend._run_config.color.contraction == "direct"
    assert backend._run_config.color.fft_basis == "trace"
    assert backend._color_accuracy == "full"
    process = build_process_ir("d d~ > z", color_accuracy="full")
    plan = build_color_plan(process, color_accuracy="full")
    groups = [
        {"helicities": (1, -1, 0), "color_sector_id": sector.id}
        for sector in plan.sectors
    ]
    backend._correlated_processes["process"] = correlated.CorrelatedProcessMetadata(
        process,
        plan,
        SimpleNamespace(
            to_mapping=lambda: {"amplitude_stage": {"coherent_groups": groups}}
        ),
        (),
    )
    matrices = backend.correlator_payload()["processes"]["process"]["matrices"]
    assert matrices[0]["color_accuracy"] == accuracy


def test_correlated_configuration_preserves_resources_and_existing_clamps() -> None:
    requested = RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(
            accuracy="full", contraction="symmetric-group-fft", fft_basis="trace"
        ),
        generation=GenerationConfig(
            workers=1,
            validation=GenerationValidationConfig(seed=19, samples=3),
        ),
    )
    effective = replace(requested, generation=replace(requested.generation, workers=2))
    clamp = ConfigClamp("generation.workers", 1, 2, "resource selection")
    resolution = correlated.correlated_configuration(
        ConfigResolution(requested, effective, (clamp,))
    )
    assert resolution.requested is requested
    assert resolution.clamps[0] is clamp
    assert resolution.effective.generation.workers == 2
    assert resolution.effective.generation.validation is effective.generation.validation
    assert resolution.effective.evaluator.backend is effective.evaluator.backend
    assert resolution.effective.color.contraction == "direct"
    assert correlated.correlated_configuration(resolution).clamps == resolution.clamps


def test_correlated_configuration_still_rejects_forced_adjoint_fft() -> None:
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full", contraction="symmetric-group-fft"),
    )
    with pytest.raises(ValueError, match="adjoint FFT is not supported"):
        correlated.correlated_configuration(config)


@pytest.mark.parametrize(
    "compress, expected", (("auto", True), (False, False), (True, True))
)
def test_correlated_configuration_reselects_compiled_compression_default(
    compress, expected
) -> None:
    from pyamplicol.config import resolve_config

    recurrence = resolve_config(
        {"action": "generate", "evaluator": {"jit": {"compress": compress}}}
    )
    assert recurrence.effective.evaluator.jit.compress is (compress is True)
    compiled = correlated.correlated_configuration(recurrence)
    assert compiled.effective.evaluator.jit.compress is expected


def test_correlated_configuration_preserves_compression_clamp() -> None:
    from pyamplicol.config import ClampRequest, resolve_config

    recurrence = resolve_config(
        {"action": "generate"},
        clamps=(ClampRequest("evaluator.jit.compress", False, "explicit pack policy"),),
    )
    compiled = correlated.correlated_configuration(recurrence)
    assert compiled.effective.evaluator.jit.compress is False


def test_automatic_catalogues_expand_for_each_generated_process() -> None:
    declarations = CorrelatorConfig.all_color(through_order=2)
    backend = correlated.CorrelatedGenerationBackend(
        RunConfig(action="generate", color=ColorConfig(accuracy="full")),
        None,
        declarations=declarations,
    )
    cases = (("annihilation", "d d~ > z"), ("pair", "e+ e- > d d~"))
    for process_id, specification in cases:
        process = build_process_ir(specification, color_accuracy="full")
        plan = build_color_plan(process, color_accuracy="full")
        groups = [
            {"helicities": (1,), "color_sector_id": sector.id}
            for sector in plan.sectors
        ]
        metadata = correlated.CorrelatedProcessMetadata(
            process,
            plan,
            SimpleNamespace(
                to_mapping=lambda groups=groups: {
                    "amplitude_stage": {"coherent_groups": groups}
                }
            ),
            (),
        )
        backend._correlated_processes[process_id] = metadata
    payload = backend.correlator_payload()
    assert payload["declarations"]["all_color_through_order"] == 2
    for process_id, expected_labels in (("annihilation", {1, 2}), ("pair", {3, 4})):
        process_payload = payload["processes"][process_id]
        resolved = CorrelatorConfig.from_json_dict(process_payload["declarations"])
        assert resolved.all_color_through_order is None
        assert len(resolved.color_requests) == 113
        assert [matrix["id"] for matrix in process_payload["matrices"]] == [
            request.id for request in resolved.color_requests
        ]
        nlo = [request for request in resolved.color_correlations if request.order == 1]
        assert {request.bra[0].emitter_label for request in nlo} == expected_labels
    assert backend.declarations is declarations


@pytest.mark.parametrize(
    "selection",
    [
        ProcessConfig(max_color_sectors=1),
        ProcessConfig(reference_color_order=(1, 2, 3, 4)),
        ProcessConfig(selected_color_sector_ids=(0,)),
        ProcessConfig(selected_source_helicities={"1": -1}),
    ],
)
def test_partial_source_or_colour_selection_is_rejected(
    selection: ProcessConfig,
) -> None:
    with pytest.raises(GenerationError, match="complete colour and helicity"):
        correlated.CorrelatedGenerationBackend(
            RunConfig(action=Action.GENERATE, process=selection),
            None,
            declarations=CorrelatorConfig(),
        )


def test_append_rejected_before_entering_generation(monkeypatch, tmp_path) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary pipeline must not run")

    monkeypatch.setattr(service.GenerationBackend, "generate", forbidden)
    with pytest.raises(GenerationError, match="append"):
        correlated.generate_correlated(
            ProcessSet((ProcessRequest.parse("g g > g g"),)),
            tmp_path / "not-created",
            declarations=CorrelatorConfig(),
            mode="append",
        )
    assert not (tmp_path / "not-created").exists()


def test_nonidentity_alias_is_rejected_before_entering_generation(
    monkeypatch, tmp_path
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary pipeline must not run")

    monkeypatch.setattr(service.GenerationBackend, "generate", forbidden)
    with pytest.raises(GenerationError, match="identity process aliases"):
        correlated.generate_correlated(
            ProcessSet(
                (ProcessRequest.parse("g g > g g", name="gggg"),),
                (ProcessAlias("swapped", "gggg", (1, 0, 2, 3)),),
            ),
            tmp_path / "not-created",
            declarations=CorrelatorConfig(),
        )
    assert not (tmp_path / "not-created").exists()


@pytest.fixture
def complete_gggg():
    backend = correlated.CorrelatedGenerationBackend(
        RunConfig(action=Action.GENERATE, color=ColorConfig(accuracy="full")),
        None,
        declarations=CorrelatorConfig(
            color_correlations=(ColorCorrelator.dipole("T1.T3", 1, 3),),
            spin_correlations=((3,), (3, 4)),
        ),
    )
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g", color_accuracy="full")
    dag, coverage = backend._compile_concrete_process(process, model)
    expanded = service._ExpandedProcess(
        request=ProcessRequest.parse(process.process, name="gggg"),
        process_ir=process,
    )
    return backend, model, service._DagProcess(expanded, dag, coverage)


def test_generic_compiler_keeps_complete_complex_source_basis(
    complete_gggg, monkeypatch
) -> None:
    backend, model, process = complete_gggg

    def forbidden(*args, **kwargs):
        raise AssertionError("physical-helicity reductions must not run")

    for name in (
        "prune_global_helicity_flip_equivalent_roots",
        "prune_dag_to_amplitude_roots",
        "project_rectangular_dynamic_color_classes",
        "build_helicity_recurrence_plan",
        "materialize_helicity_recurrence",
        "run_generic_dag_numerical_current_warmup",
    ):
        monkeypatch.setattr(service, name, forbidden)
    prepared = backend._prepare_warmup_process(
        process, model, index=0, phase=PhaseHandle("prepare", None, 1)
    )
    assert prepared.dag is process.dag
    assert prepared.helicity_sum_dag is None
    assert prepared.helicity_selector_union_dag is None
    assert prepared.filters["structural_helicity_reduction"]["applied"] is False
    assert prepared.filters["relation_discovery"]["effective_mode"] == "off"
    assert len(prepared.validation_points) == 1
    dag = prepared.dag
    assert dag.color_coverage == dag.helicity_coverage == "complete"
    assert not dag.color_plan.trace_reflections_folded
    assert dag.lc_topology_replay is dag.color_topology_replay is None
    assert dag.helicity_recurrence is dag.helicity_materialization is None
    assert all(root.helicity_weight == 1 for root in dag.amplitude_roots)

    evaluator = backend._construct_evaluator(
        prepared, model, PhaseHandle("schema", None, 1)
    )
    schema = evaluator.runtime_schema.to_mapping()
    assert "runtime_selectors" not in schema["physics"]["extensions"]
    assert evaluator.stage_input.runtime_schema is evaluator.runtime_schema
    ordinary_schema = service.build_runtime_expression_schema(
        dag, model, process_id="ordinary"
    ).to_mapping()
    assert (
        ordinary_schema["physics"]["extensions"]["runtime_selectors"]["axes"][
            "helicity"
        ]["runtime_contract"]
        == "complete-reusable"
    )
    groups = schema["amplitude_stage"]["coherent_groups"]
    assert {tuple(group["helicities"]) for group in groups} == set(
        product((-1, 1), repeat=4)
    )
    assert {group["color_sector_id"] for group in groups} == {
        sector.id for sector in dag.color_plan.sectors
    }
    assert all(
        group["helicity_weight"] == group["all_sector_weight"] == 1 for group in groups
    )
    for source in schema["source_fill"]["sources"]:
        assert source["source_parameter_stop"] - source["source_parameter_start"] == 4
        assert source["source_ir"]["component_dimension"] == 4
        assert (
            source["value_slot"]["component_stop"]
            - source["value_slot"]["component_start"]
            == 4
        )
    metadata = backend.process_metadata["gggg"]
    assert metadata.color_plan is dag.color_plan
    assert metadata.process is dag.process
    assert metadata.runtime_schema is evaluator.runtime_schema

    payload = backend.correlator_payload()
    assert payload["complete_source_basis"] is True
    assert payload["declarations"] == backend.declarations.to_json_dict()
    catalog = payload["processes"]["gggg"]
    assert catalog["declarations"] == backend.declarations.to_json_dict()
    assert catalog["spin_legs"] == [3, 4]
    assert catalog["coherent_groups"] == groups
    assert [matrix["id"] for matrix in catalog["matrices"]] == ["born", "T1.T3"]
    for matrix in catalog["matrices"]:
        assert matrix["color_accuracy"] == "full"
        assert matrix["storage"] == "sparse-directed"
        assert matrix["sector_ids"] == [sector.id for sector in dag.color_plan.sectors]


@pytest.mark.parametrize("spin_leg", [1, 4])
def test_only_present_four_vector_legs_can_be_declared(spin_leg: int) -> None:
    backend = correlated.CorrelatedGenerationBackend(
        None, None, declarations=CorrelatorConfig(spin_correlations=((spin_leg,),))
    )
    with pytest.raises(GenerationError, match=r"(four-component vector|absent)"):
        backend._prepare_process_construction(
            build_process_ir("d d~ > z", color_accuracy="full"), BuiltinSMModel()
        )


def test_multi_quark_catalogue_uses_retained_canonical_colour_owners() -> None:
    backend = correlated.CorrelatedGenerationBackend(
        None, None, declarations=CorrelatorConfig(spin_correlations=((5,),))
    )
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > u u~ g", color_accuracy="full")
    dag, coverage = backend._compile_concrete_process(process, model)
    expanded = service._ExpandedProcess(
        request=ProcessRequest.parse(process.process, name="qqqqg"), process_ir=process
    )
    compiled = backend._prepare_warmup_process(
        service._DagProcess(expanded, dag, coverage),
        model,
        index=0,
        phase=PhaseHandle("prepare", None, 1),
    )
    backend._construct_evaluator(compiled, model, PhaseHandle("schema", None, 1))
    catalog = backend.correlator_payload()["processes"]["qqqqg"]
    owners = catalog["matrices"][0]["sector_ids"]
    assert owners == [0, 2, 4, 6]
    assert len(owners) < dag.color_plan.sector_count
    basis_by_helicity = {}
    for group in catalog["coherent_groups"]:
        basis_by_helicity.setdefault(tuple(group["helicities"]), set()).add(
            group["color_sector_id"]
        )
    assert all(basis == set(owners) for basis in basis_by_helicity.values())


def test_opt_in_payload_is_added_through_the_existing_writer(monkeypatch) -> None:
    backend = correlated.CorrelatedGenerationBackend(
        None, None, declarations=CorrelatorConfig()
    )
    calls = []

    class Builder:
        def add_json(self, path, payload, *, role):
            calls.append((path, payload, role))

    sentinel = object()

    def writer(self, destination, **kwargs):
        assert destination == "destination"
        extensions = kwargs["payload_hook"](Builder())
        assert extensions == {
            "correlators": {"schema_version": 1, "path": "correlators.json"}
        }
        return sentinel

    monkeypatch.setattr(service.GenerationBackend, "_write_artifact", writer)
    assert backend._write_artifact("destination") is sentinel
    assert calls == [
        ("correlators.json", backend.correlator_payload(), "runtime-physics")
    ]


def test_sidecar_and_extension_are_ordinary_manifest_payloads(tmp_path) -> None:
    backend = correlated.CorrelatedGenerationBackend(
        None, None, declarations=CorrelatorConfig(spin_correlations=((3,),))
    )
    output = tmp_path / "artifact"
    with ArtifactBuilder(output) as builder:
        builder.add_json(
            "physics/process.json", {}, role="runtime-physics", process_id="gggg"
        )
        extension = backend._write_correlator_payload(builder)
        builder.finalize(
            kind="pyamplicol-process",
            producer={
                "distribution": "pyamplicol",
                "version": "0.1.0",
                "versions": {
                    "python_api": 1,
                    "toml": 1,
                    "compiled_model": 1,
                    "process_artifact": 3,
                    "runtime_physics": 1,
                    "symbolica_serialization": "test",
                    "c_abi": 1,
                },
                "target": {"triple": "test-target", "cpu_features": []},
            },
            model={
                "name": "built-in-sm",
                "source_kind": "built-in-sm",
                "content_sha256": "1" * 64,
                "compiled_schema_version": 1,
            },
            configuration={
                "toml_schema_version": 1,
                "requested_path": "config/requested.toml",
                "effective_path": "config/effective.toml",
                "adjustments": [],
            },
            processes=[
                {
                    "id": "gggg",
                    "expression": "g g > g g",
                    "color_accuracy": "full",
                    "external_pdgs": [21, 21, 21, 21],
                    "physics_path": "physics/process.json",
                    "required_runtime_capabilities": [
                        "symjit.application.complex-f64.v1"
                    ],
                    "aliases": [],
                }
            ],
            runtime={
                "engine": "rusticol",
                "engine_version": "0.1.0",
                "evaluator_manifest_path": "physics/process.json",
                "api_bundle_path": None,
                "required_runtime_capabilities": ["symjit.application.complex-f64.v1"],
            },
            extensions=extension,
        )
        assert not output.exists()
    manifest = load_manifest(output)
    assert manifest.extensions["correlators"] == extension["correlators"]
    records = {record.path: record for record in manifest.payloads}
    assert records["correlators.json"].role == "runtime-physics"
    assert (
        json.loads((output / "correlators.json").read_text())
        == backend.correlator_payload()
    )


def test_default_writer_wrapper_passes_through_without_a_payload_hook(
    monkeypatch,
) -> None:
    calls = []
    sentinel = object()

    def writer(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(service, "write_schema_v3_artifact", writer)
    backend = service.create_generator_backend(None, None)
    assert backend._write_artifact("output", mode="error") is sentinel
    assert calls == [(("output",), {"mode": "error"})]
