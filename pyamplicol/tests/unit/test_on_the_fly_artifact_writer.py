# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyamplicol.generation.artifact_writer as artifact_writer
import pyamplicol.generation.service as service_module
from pyamplicol.api.errors import ArtifactError, GenerationError
from pyamplicol.api.requests import ModelSource, ProcessRequest, ProcessSet
from pyamplicol.api.services import Generator
from pyamplicol.artifacts import inspect_artifact, load_manifest
from pyamplicol.artifacts import inspection as artifact_inspection
from pyamplicol.artifacts.process_groups import (
    initial_process_groups,
    read_process_groups,
)
from pyamplicol.color import build_color_plan
from pyamplicol.config import (
    Action,
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    RunConfig,
)
from pyamplicol.generation.artifact_writer import (
    OnTheFlyProcessArtifact,
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from pyamplicol.generation.evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    PacbinReader,
    write_pacbin_atomic,
)
from pyamplicol.generation.process_grouping import GroupingBasis, GroupMember
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.recurrence_physics import (
    build_on_the_fly_runtime_metadata,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.loading import load_compiled_model
from pyamplicol.models.prepared import load_prepared_model_bundle
from pyamplicol.models.prepared_target import canonical_architecture

ROOT = Path(__file__).resolve().parents[2]
_PROCESS_ID = "d_dbar_to_z"
_SEED_MEMBER = "on-the-fly/process-seed-v1.bin"


def _seed_identity(
    *,
    process_digest: str = "1" * 64,
    external_leg_count: int = 3,
) -> dict[str, object]:
    return {
        "abi": "pyamplicol-on-the-fly-process-seed-identity-v1",
        "process_digest": process_digest,
        "compiled_model_digest": "2" * 64,
        "recurrence_template_catalog_digest": "3" * 64,
        "prepared_kernel_pack_digest": "4" * 64,
        "recurrence_direct_template_catalog_digest": "5" * 64,
        "semantic_digest": "6" * 64,
        "external_permutation": list(range(external_leg_count)),
        "external_sources": [
            {
                "source_slot": slot,
                "public_label": slot + 1,
                "is_initial": slot < 2,
                "states": [
                    {
                        "state_index": 0,
                        "public_helicity": 0,
                        "prepared_mass_parameter_slot": None,
                    }
                ],
            }
            for slot in range(external_leg_count)
        ],
    }


def _seed_identity_from_source_projection(
    source_projection: dict[str, object],
    *,
    process_digest: str | None = None,
) -> dict[str, object]:
    projected_digest = source_projection["process_digest"]
    permutation = source_projection["external_permutation"]
    sources = source_projection["external_sources"]
    assert isinstance(projected_digest, str)
    assert isinstance(permutation, list)
    assert isinstance(sources, list)
    identity = _seed_identity(
        process_digest=(projected_digest if process_digest is None else process_digest)
    )
    identity["external_permutation"] = list(permutation)
    compact_sources: list[dict[str, object]] = []
    for source in sources:
        assert isinstance(source, dict)
        states = source["states"]
        assert isinstance(states, list)
        compact_sources.append(
            {
                "source_slot": source["source_slot"],
                "public_label": source["public_label"],
                "is_initial": source["is_initial"],
                "states": [
                    {
                        "state_index": state["state_index"],
                        "public_helicity": state["public_helicity"],
                        "prepared_mass_parameter_slot": None,
                    }
                    for state in states
                    if isinstance(state, dict)
                ],
            }
        )
    identity["external_sources"] = compact_sources
    return identity


def _prepared_model_path() -> Path:
    architecture = canonical_architecture()
    return (
        ROOT
        / "src"
        / "pyamplicol"
        / "assets"
        / "prepared_models"
        / f"built-in-sm-jit-o2-{architecture}.pyamplicol-model"
    )


def _configuration(
    color_accuracy: str = "lc",
    *,
    contraction: str = "direct",
    fft_basis: str = "trace",
) -> _GenerationConfigProvenance:
    return _GenerationConfigProvenance.from_config(
        RunConfig(
            action=Action.GENERATE,
            color=ColorConfig(
                accuracy=color_accuracy,
                contraction=contraction,
                fft_basis=fft_basis,
            ),
            generation=GenerationConfig(
                emit_api_bundle=False,
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(
                execution_mode="on-the-fly",
                optimization=EvaluatorOptimizationConfig(cores=1),
            ),
        )
    )


def _write_seed_container(path: Path, payload: bytes = b"compact-process-seed"):
    return write_pacbin_atomic(
        path,
        (
            PacbinMemberSource(
                _SEED_MEMBER,
                PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                source=io.BytesIO(payload),
            ),
        ),
    )


def _process_artifact(
    tmp_path: Path,
    *,
    runtime_name: str = "runtime.pacbin",
    process_id: str = _PROCESS_ID,
    expression: str = "d d~ > z",
) -> OnTheFlyProcessArtifact:
    runtime_path = tmp_path / runtime_name
    runtime_index = _write_seed_container(runtime_path)
    bundle = load_prepared_model_bundle(_prepared_model_path())
    process = build_process_ir(expression, color_accuracy="full")
    return OnTheFlyProcessArtifact(
        process_id=process_id,
        expression=expression,
        color_accuracy="lc",
        external_pdgs=tuple(leg.pdg for leg in process.legs),
        aliases=(),
        physics={
            "schema_version": 1,
            "kind": artifact_writer.ON_THE_FLY_PUBLIC_METADATA_KIND,
            "process_id": process_id,
            "process": expression,
            "color_accuracy": "lc",
            "external_particles": [
                {
                    "index": index,
                    "label": index + 1,
                    "particle": leg.particle,
                    "pdg": leg.pdg,
                    "role": "initial" if index < 2 else "final",
                    "momentum_slot": index,
                    "momentum_components": ["E", "px", "py", "pz"],
                }
                for index, leg in enumerate(process.legs)
            ],
            "model_parameters": [],
        },
        runtime_path=runtime_path,
        runtime_size_bytes=runtime_index.file_size,
        runtime_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        runtime_member_count=1,
        runtime_unpacked_size_bytes=sum(
            member.length for member in runtime_index.members
        ),
        runtime_index_sha256=runtime_index.index_sha256,
        referenced_kernel_ids=frozenset(
            kernel.kernel_id for kernel in bundle.kernel_pack.kernels
        ),
        runtime_metadata={
            "runtime_parameters": [],
            "prepared_parameter_defaults": [],
            "parameter_projection": [],
            "external_legs": [],
            "particle_masses": [],
            "normalization": {},
            "process_seed_identity": _seed_identity(
                external_leg_count=len(process.legs)
            ),
        },
        selector_policy={
            "color_coverage": "complete",
            "reference_color_word": None,
            "trace_reflections_folded": False,
            "selector_census": {
                "physical_helicity_count": 1,
                "physical_color_flow_count": 1,
            },
        },
        color_contraction_payload=None,
        color_contraction_summary=None,
        point_tile_size=128,
        query_construction_threads=3,
        validation_point=ValidationPointRecord(
            process_id=process_id,
            process=expression,
            seed=7,
            error="not sampled in writer test",
        ),
        generation_filters={"on_the_fly": {"selector_coverage": "complete"}},
    )


def _contracted_process_artifact(
    tmp_path: Path,
    *,
    color_accuracy: str,
    contraction: str = "direct",
    fft_basis: str = "trace",
    expression: str = "d d~ > z",
    process_id: str = _PROCESS_ID,
) -> OnTheFlyProcessArtifact:
    process = build_process_ir(expression, color_accuracy=color_accuracy)
    color_plan = build_color_plan(
        process,
        basis=fft_basis,
        color_accuracy=color_accuracy,
        fold_trace_reflections=False,
    )
    contracted = service_module._build_on_the_fly_contracted_color_payload_v1(
        color_plan,
        contraction=contraction,
    )
    base = _process_artifact(
        tmp_path,
        runtime_name=f"{process_id}-{contraction}-runtime.pacbin",
        process_id=process_id,
        expression=expression,
    )
    return replace(
        base,
        color_accuracy=color_accuracy,
        physics={**base.physics, "color_accuracy": color_accuracy},
        runtime_metadata={
            **base.runtime_metadata,
            "normalization": {
                "color_accuracy": color_accuracy,
                "color_factor": 3.0,
            },
        },
        selector_policy={
            **base.selector_policy,
            **({"color_basis": fft_basis} if fft_basis != "trace" else {}),
            "color_coverage": "contracted",
            "trace_reflections_folded": False,
            "selector_census": {
                "physical_helicity_count": 1,
                "physical_color_flow_count": 1,
            },
        },
        color_contraction_payload=contracted.payload,
        color_contraction_summary=contracted.summary,
    )


def _prepare_writer_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    source_revision = "a" * 40
    native_inputs = "b" * 64
    monkeypatch.setattr(
        artifact_writer,
        "active_source_revision",
        lambda: source_revision,
    )
    monkeypatch.setattr(
        artifact_writer,
        "active_native_build_inputs_sha256",
        lambda: native_inputs,
    )
    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: (
            {"triple": "aarch64-apple-darwin", "cpu_features": []},
            1,
        ),
    )
    monkeypatch.setattr(
        artifact_writer,
        "_derive_eager_direct_descriptor",
        lambda source, **_widths: b"direct-table:" + source,
    )


def test_producer_keeps_authenticated_native_digest_without_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_inputs = "b" * 64
    monkeypatch.setattr(artifact_writer, "package_version", lambda: "0.1.0")
    monkeypatch.setattr(artifact_writer, "active_source_revision", lambda: None)
    monkeypatch.setattr(
        artifact_writer,
        "active_native_build_inputs_sha256",
        lambda: native_inputs,
    )
    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: (
            {"triple": "aarch64-apple-darwin", "cpu_features": []},
            1,
        ),
    )

    producer = artifact_writer._producer_metadata(RunConfig(action=Action.GENERATE))

    assert producer["native_build_inputs_sha256"] == native_inputs
    assert "git_revision" not in producer


def test_append_rejects_native_build_input_mismatch() -> None:
    existing = SimpleNamespace(
        runtime={"required_runtime_capabilities": []},
        producer={
            "distribution": "pyamplicol",
            "version": "0.1.0",
            "native_build_inputs_sha256": "a" * 64,
        },
    )

    with pytest.raises(ValueError, match="native build-input digest differs"):
        artifact_writer._validate_append_compatibility(
            existing,
            producer={
                "distribution": "pyamplicol",
                "version": "0.1.0",
                "native_build_inputs_sha256": "b" * 64,
            },
            model={},
            eager_pack_identity=None,
            requested_bytes=b"",
            effective_bytes=b"",
            adjustments=(),
            processes=(),
        )


def _payload_inventory(root: Path) -> tuple[tuple[object, ...], ...]:
    manifest = load_manifest(root)
    return tuple(
        sorted(
            (
                record.path,
                record.role,
                record.process_id,
            )
            for record in manifest.payloads
        )
    )


def test_on_the_fly_writer_publishes_one_seed_and_compact_metadata_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    compiled = load_compiled_model(source_path)
    first_process = _process_artifact(tmp_path, runtime_name="first.pacbin")
    second_process = _process_artifact(tmp_path, runtime_name="second.pacbin")
    first = tmp_path / "first-artifact"
    second = tmp_path / "second-artifact"

    for output, process in ((first, first_process), (second, second_process)):
        write_schema_v3_artifact(
            output,
            mode="error",
            source=ModelSource.from_path(source_path),
            compiled_model=compiled,
            configuration=_configuration(),
            processes=(process,),
            timings={"total": 0.1},
            api_bundle_hook=None,
        )

    runtime_relative = f"processes/{_PROCESS_ID}/on-the-fly-runtime.pacbin"
    for output in (first, second):
        manifest = load_manifest(output)
        structural_proof_path = f"processes/{_PROCESS_ID}/structural-source-proof.json"
        assert [
            (record.path, record.role)
            for record in manifest.payloads
            if record.path == structural_proof_path
            or record.role == artifact_writer.STRUCTURAL_SOURCE_PROOF_ROLE
        ] == []
        assert not (output / structural_proof_path).exists()
        runtime = output / runtime_relative
        with PacbinReader.open(runtime, verify_payloads=True) as reader:
            assert [
                (member.logical_path, member.kind) for member in reader.members
            ] == [(_SEED_MEMBER, PacbinMemberKind.ON_THE_FLY_PROCESS_SEED)]
        physics = json.loads(
            (output / f"processes/{_PROCESS_ID}/physics.json").read_text(
                encoding="utf-8"
            )
        )
        assert physics["kind"] == artifact_writer.ON_THE_FLY_PUBLIC_METADATA_KIND
        assert len(physics["external_particles"]) == 3
        assert not {
            "color_components",
            "helicities",
            "lc_topology_replay",
            "recurrence",
            "stages",
        }.intersection(physics)
        execution = json.loads(
            (output / f"processes/{_PROCESS_ID}/execution.json").read_text(
                encoding="utf-8"
            )
        )
        assert execution["kind"] == artifact_writer.ON_THE_FLY_RUNTIME_KIND
        assert "color_contraction" not in execution["runtime_metadata"]
        assert not (output / f"processes/{_PROCESS_ID}/on-the-fly-color.bin").exists()
        assert execution["selector_policy"]["selector_census"] == {
            "physical_helicity_count": 1,
            "physical_color_flow_count": 1,
        }
        assert execution["runtime_options"] == {
            "point_tile_size": 128,
            "query_construction_threads": 3,
        }
        assert execution["runtime_container"] == {
            "kind": artifact_writer.ON_THE_FLY_RUNTIME_CONTAINER_KIND,
            "path": "on-the-fly-runtime.pacbin",
            "schema_version": 1,
            "seed_member_path": _SEED_MEMBER,
            "storage_abi": "pacbin-v1",
        }
        inspected = inspect_artifact(output).processes[0]
        assert inspected.execution_mode == "on-the-fly"
        assert inspected.lc_flow_layout == "compact/query-local"
        assert inspected.requested_point_tile_size == 128
        assert inspected.effective_point_tile_size is None
        assert inspected.requested_query_construction_threads == 3
        assert inspected.physical_helicities == inspected.computed_helicities == 1
        assert inspected.physical_color_components == 1
        assert inspected.computed_color_components == 1
        assert inspected.helicity_coverage == inspected.color_coverage == "complete"
        assert inspected.generation_specialized_axes == ()
        assert inspected.lc_physical_sector_count == 1
        assert inspected.lc_materialized_sector_count is None
        assert inspected.lc_replayed_sector_count is None
        assert inspected.lc_residual_sector_count is None

    assert isinstance(execution, dict)
    process_record = manifest.processes[0]
    capabilities = process_record["required_runtime_capabilities"]
    assert isinstance(capabilities, tuple)
    for index, mutation in enumerate(({"key": "wrong-process"}, {"invented": True})):
        mutated_execution = {**execution, **mutation}
        mutated_path = tmp_path / f"mutated-execution-{index}.json"
        mutated_path.write_text(json.dumps(mutated_execution), encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"outer process|root fields"):
            artifact_inspection._execution_inspection(
                manifest,
                mutated_path,
                capabilities,
                process=process_record,
            )

    assert (first / runtime_relative).read_bytes() == (
        second / runtime_relative
    ).read_bytes()
    assert _payload_inventory(first) == _payload_inventory(second)


@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_on_the_fly_writer_publishes_authenticated_contracted_color(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    color_accuracy: str,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    output = tmp_path / f"{color_accuracy}-artifact"
    process = _contracted_process_artifact(
        tmp_path,
        color_accuracy=color_accuracy,
    )

    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.from_path(source_path),
        compiled_model=load_compiled_model(source_path),
        configuration=_configuration(color_accuracy),
        processes=(process,),
        timings={"total": 0.1},
        api_bundle_hook=None,
    )

    manifest = load_manifest(output)
    color_path = f"processes/{_PROCESS_ID}/on-the-fly-color.bin"
    color_record = next(
        record for record in manifest.payloads if record.path == color_path
    )
    color_payload = (output / color_path).read_bytes()
    execution = json.loads(
        (output / f"processes/{_PROCESS_ID}/execution.json").read_text(encoding="utf-8")
    )
    reference = execution["runtime_metadata"]["color_contraction"]

    assert color_payload == process.color_contraction_payload
    assert color_record.sha256 == hashlib.sha256(color_payload).hexdigest()
    assert reference == {
        **process.color_contraction_summary,
        "path": "on-the-fly-color.bin",
        "size_bytes": len(color_payload),
        "sha256": color_record.sha256,
    }
    assert execution["required_runtime_capabilities"] == sorted(
        (
            "rusticol.on-the-fly.complex-f64.v1",
            "rusticol.on-the-fly.contracted-color.v1",
        )
    )
    assert execution["selector_policy"] == {
        "color_coverage": "contracted",
        "reference_color_word": None,
        "trace_reflections_folded": False,
        "selector_census": {
            "physical_helicity_count": 1,
            "physical_color_flow_count": 1,
        },
    }
    inspected = inspect_artifact(output).processes[0]
    assert inspected.execution_mode == "on-the-fly"
    assert inspected.color_accuracy == color_accuracy
    assert inspected.physical_color_components == 1
    assert inspected.computed_color_components == 1
    assert inspected.color_coverage == "contracted"
    assert inspected.color_flow_runtime_contract is None
    assert inspected.recurrence_color_accuracy == color_accuracy
    assert inspected.recurrence_color_storage == "expanded"
    assert inspected.recurrence_color_component_count == 1
    assert inspected.recurrence_color_group_count == 1
    assert inspected.recurrence_color_destination_count == 1


@pytest.mark.parametrize(
    ("color_accuracy", "color_basis"),
    (("lc", "trace"), ("nlc", "adjoint"), ("full", "adjoint")),
)
def test_on_the_fly_writer_and_inspector_validate_optional_color_basis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    color_accuracy: str,
    color_basis: str,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    contracted = color_accuracy != "lc"
    contraction = "symmetric-group-fft" if contracted else "direct"
    process = (
        _contracted_process_artifact(
            tmp_path,
            color_accuracy=color_accuracy,
            contraction=contraction,
            fft_basis=color_basis,
            expression="g g > g g",
        )
        if contracted
        else _process_artifact(tmp_path)
    )
    process = replace(
        process, selector_policy={**process.selector_policy, "color_basis": color_basis}
    )
    output = tmp_path / "artifact"
    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.from_path(source_path),
        compiled_model=load_compiled_model(source_path),
        configuration=_configuration(
            color_accuracy, contraction=contraction, fft_basis=color_basis
        ),
        processes=(process,),
        timings={},
        api_bundle_hook=None,
    )
    manifest = load_manifest(output)
    execution_path = output / f"processes/{_PROCESS_ID}/execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution["selector_policy"]["color_basis"] == color_basis
    assert inspect_artifact(output).processes[0].color_accuracy == color_accuracy
    color_record = next(
        (
            record
            for record in manifest.payloads
            if record.path == f"processes/{_PROCESS_ID}/on-the-fly-color.bin"
        ),
        None,
    )
    mutations = [{"color_basis": value} for value in ("other", None, True, [])]
    mutations.append(
        {"trace_reflections_folded": True}
        if contracted
        else {"color_basis": "adjoint"}
    )
    if contracted:
        mutations.append({"color_coverage": "complete"})
    for index, mutation in enumerate(mutations):
        policy = {**process.selector_policy, **mutation}
        with pytest.raises(ValueError, match=r"basis|contracted|selector policy"):
            artifact_writer._on_the_fly_execution_summary(
                replace(process, selector_policy=policy),
                color_contraction_record=color_record,
            )
        invalid_execution = {**execution, "selector_policy": policy}
        invalid_path = tmp_path / f"invalid-basis-{index}.json"
        invalid_path.write_text(json.dumps(invalid_execution), encoding="utf-8")
        with pytest.raises(ArtifactError, match=r"basis|contracted|selector_policy"):
            artifact_inspection._execution_inspection(
                manifest,
                invalid_path,
                execution["required_runtime_capabilities"],
                process=manifest.processes[0],
            )


def test_color_contraction_regenerates_only_the_process_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    source = ModelSource.from_path(source_path)
    compiled = load_compiled_model(source_path)
    expression = "g g > g g"
    process_id = "g_g_to_g_g"
    manifests = {}
    color_payload_digests = {}

    for contraction in ("direct", "symmetric-group-fft"):
        output = tmp_path / contraction
        process = _contracted_process_artifact(
            tmp_path,
            color_accuracy="full",
            contraction=contraction,
            expression=expression,
            process_id=process_id,
        )
        write_schema_v3_artifact(
            output,
            mode="error",
            source=source,
            compiled_model=compiled,
            configuration=_configuration(
                "full",
                contraction=contraction,
            ),
            processes=(process,),
            timings={"total": 0.1},
            api_bundle_hook=None,
        )
        manifest = load_manifest(output)
        manifests[contraction] = manifest
        color_payload_digests[contraction] = next(
            record.sha256
            for record in manifest.payloads
            if record.path == f"processes/{process_id}/on-the-fly-color.bin"
        )

    direct = manifests["direct"]
    fft = manifests["symmetric-group-fft"]
    assert (
        color_payload_digests["direct"] != color_payload_digests["symmetric-group-fft"]
    )
    assert direct.artifact_id != fft.artifact_id

    # The setting selects a process-owned reduction payload. It must not
    # invalidate either the prepared model or its authenticated native kernels.
    assert direct.model == fft.model
    assert (
        direct.extensions["eager_prepared_pack"]
        == fft.extensions["eager_prepared_pack"]
    )
    assert (
        direct.producer["native_build_inputs_sha256"]
        == fft.producer["native_build_inputs_sha256"]
    )


def test_on_the_fly_writer_append_skips_structural_proofs_for_complete_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    source = ModelSource.from_path(source_path)
    compiled = load_compiled_model(source_path)
    artifact = tmp_path / "artifact"
    first = _process_artifact(tmp_path, runtime_name="first.pacbin")
    second_process_id = "d_dbar_to_z_copy"
    second = _process_artifact(
        tmp_path,
        runtime_name="second.pacbin",
        process_id=second_process_id,
    )

    write_schema_v3_artifact(
        artifact,
        mode="error",
        source=source,
        compiled_model=compiled,
        configuration=_configuration(),
        processes=(first,),
        timings={"total": 0.1},
        api_bundle_hook=None,
    )
    write_schema_v3_artifact(
        artifact,
        mode="append",
        source=source,
        compiled_model=compiled,
        configuration=_configuration(),
        processes=(second,),
        timings={"total": 0.2},
        api_bundle_hook=None,
    )

    manifest = load_manifest(artifact)
    assert tuple(record["id"] for record in manifest.processes) == (
        _PROCESS_ID,
        second_process_id,
    )
    assert [
        record.path
        for record in manifest.payloads
        if record.role == artifact_writer.STRUCTURAL_SOURCE_PROOF_ROLE
    ] == []
    for process_id in (_PROCESS_ID, second_process_id):
        assert not (
            artifact / f"processes/{process_id}/structural-source-proof.json"
        ).exists()
    assert [
        process.execution_mode for process in inspect_artifact(artifact).processes
    ] == ["on-the-fly", "on-the-fly"]


def test_writer_records_process_groups_and_appended_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    source = ModelSource.from_path(source_path)
    compiled = load_compiled_model(source_path)
    artifact = tmp_path / "artifact"
    assumed = GroupMember(
        process_id="u_ubar_to_z",
        expression="u u~ > z",
        pdgs=(2, -2, 23),
        initial_count=2,
        permutation=(0, 1, 2),
        basis=GroupingBasis.FLAVOR_BLIND,
        generated=False,
    )
    write_schema_v3_artifact(
        artifact,
        mode="error",
        source=source,
        compiled_model=compiled,
        configuration=_configuration(),
        processes=(_process_artifact(tmp_path, runtime_name="first.pacbin"),),
        timings={"total": 0.1},
        api_bundle_hook=None,
        process_groups=initial_process_groups(
            [(_PROCESS_ID, "d d~ > z", (1, -1, 23))], {_PROCESS_ID: [assumed]}
        ),
    )
    stored = read_process_groups(artifact)
    assert stored is not None
    assert [(member.process_id, member.generated) for member in stored.members()] == [
        (_PROCESS_ID, True),
        ("u_ubar_to_z", False),
    ]

    # Generating an assumed member makes it a representative of its own.
    write_schema_v3_artifact(
        artifact,
        mode="append",
        source=source,
        compiled_model=compiled,
        configuration=_configuration(),
        processes=(
            _process_artifact(
                tmp_path,
                runtime_name="second.pacbin",
                process_id="u_ubar_to_z",
                expression="u u~ > z",
            ),
        ),
        timings={"total": 0.2},
        api_bundle_hook=None,
    )
    manifest = load_manifest(artifact, verify_payloads=True)
    assert [
        (record.role, record.media_type)
        for record in manifest.payloads
        if record.path == "process-groups.json"
    ] == [("process-groups", "application/json")]
    appended = read_process_groups(artifact)
    assert appended is not None
    assert {
        identifier: [
            (member.process_id, member.factor, member.permutation, member.basis)
            for member in group.members
        ]
        for identifier, group in appended.groups.items()
    } == {
        _PROCESS_ID: [(_PROCESS_ID, 1, (0, 1, 2), GroupingBasis.IDENTITY)],
        "u_ubar_to_z": [("u_ubar_to_z", 1, (0, 1, 2), GroupingBasis.IDENTITY)],
    }


@pytest.mark.parametrize(
    ("roles", "expected_color_flows"),
    (
        (("singlet",) * 6, 1),
        (("adjoint",) * 4, 6),
        (
            (
                "fundamental",
                "antifundamental",
                "fundamental",
                "antifundamental",
                "adjoint",
                "adjoint",
            ),
            12,
        ),
        (
            (
                "fundamental",
                "antifundamental",
                "fundamental",
                "antifundamental",
                "fundamental",
                "antifundamental",
            ),
            6,
        ),
    ),
)
def test_on_the_fly_selector_census_matches_compact_adapter_formulas(
    roles: tuple[str, ...],
    expected_color_flows: int,
) -> None:
    projection = SimpleNamespace(
        external_sources=tuple(
            SimpleNamespace(source_states=(object(), object())) for _ in roles
        )
    )
    labels = tuple(range(1, len(roles) + 1))
    process = SimpleNamespace(
        legs=tuple(SimpleNamespace(color_role=role) for role in roles),
        fundamental_labels=tuple(
            label
            for label, role in zip(labels, roles, strict=True)
            if role == "fundamental"
        ),
        antifundamental_labels=tuple(
            label
            for label, role in zip(labels, roles, strict=True)
            if role == "antifundamental"
        ),
        adjoint_labels=tuple(
            label
            for label, role in zip(labels, roles, strict=True)
            if role == "adjoint"
        ),
    )

    census = service_module._on_the_fly_selector_census_v1(
        projection,  # type: ignore[arg-type]
        process,  # type: ignore[arg-type]
    )
    assert census["physical_helicity_count"] == 2 ** len(roles)
    assert census["physical_color_flow_count"] == expected_color_flows


def test_on_the_fly_writer_rejects_changed_or_noncanonical_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    source_path = _prepared_model_path()
    compiled = load_compiled_model(source_path)
    changed = _process_artifact(tmp_path, runtime_name="changed.pacbin")
    changed.runtime_path.write_bytes(changed.runtime_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="payload changed"):
        write_schema_v3_artifact(
            tmp_path / "changed-artifact",
            mode="error",
            source=ModelSource.from_path(source_path),
            compiled_model=compiled,
            configuration=_configuration(),
            processes=(changed,),
            timings={},
            api_bundle_hook=None,
        )

    extra_path = tmp_path / "extra.pacbin"
    extra_index = write_pacbin_atomic(
        extra_path,
        (
            PacbinMemberSource(
                _SEED_MEMBER,
                PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                source=io.BytesIO(b"seed"),
            ),
            PacbinMemberSource(
                "on-the-fly/extra.bin",
                PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
                source=io.BytesIO(b"extra"),
            ),
        ),
    )
    noncanonical = replace(
        changed,
        runtime_path=extra_path,
        runtime_size_bytes=extra_index.file_size,
        runtime_sha256=hashlib.sha256(extra_path.read_bytes()).hexdigest(),
        runtime_member_count=len(extra_index.members),
        runtime_unpacked_size_bytes=sum(
            member.length for member in extra_index.members
        ),
        runtime_index_sha256=extra_index.index_sha256,
    )
    with pytest.raises(ValueError, match="exactly one canonical process seed"):
        write_schema_v3_artifact(
            tmp_path / "extra-artifact",
            mode="error",
            source=ModelSource.from_path(source_path),
            compiled_model=compiled,
            configuration=_configuration(),
            processes=(noncanonical,),
            timings={},
            api_bundle_hook=None,
        )

    missing_color = _process_artifact(tmp_path, runtime_name="missing-color.pacbin")
    missing_color = replace(
        missing_color,
        color_accuracy="nlc",
        physics={**missing_color.physics, "color_accuracy": "nlc"},
        selector_policy={
            **missing_color.selector_policy,
            "color_coverage": "contracted",
        },
    )
    with pytest.raises(ValueError, match="no color payload"):
        write_schema_v3_artifact(
            tmp_path / "nlc-artifact",
            mode="error",
            source=ModelSource.from_path(source_path),
            compiled_model=compiled,
            configuration=_configuration(),
            processes=(missing_color,),
            timings={},
            api_bundle_hook=None,
        )

    valid = _process_artifact(tmp_path, runtime_name="valid-census.pacbin")
    for index, selector_policy in enumerate(
        (
            {
                "color_coverage": "complete",
                "reference_color_word": None,
                "trace_reflections_folded": False,
            },
            {
                **valid.selector_policy,
                "selector_census": {
                    "physical_helicity_count": 0,
                    "physical_color_flow_count": 1,
                },
            },
        )
    ):
        with pytest.raises(ValueError, match=r"selector census|selector policy"):
            write_schema_v3_artifact(
                tmp_path / f"invalid-census-{index}",
                mode="error",
                source=ModelSource.from_path(source_path),
                compiled_model=compiled,
                configuration=_configuration(),
                processes=(replace(valid, selector_policy=selector_policy),),
                timings={},
                api_bundle_hook=None,
            )


@pytest.mark.parametrize(
    "selection",
    (
        service_module._ProcessSelection(max_color_sectors=1),
        service_module._ProcessSelection(selected_color_sector_ids=frozenset({0})),
        service_module._ProcessSelection(selected_source_helicities={1: -1}),
    ),
)
def test_on_the_fly_construction_rejects_generation_time_selector_trimming(
    tmp_path: Path,
    selection: service_module._ProcessSelection,
) -> None:
    backend = service_module.GenerationBackend(
        _configuration().effective,
        None,
        process_selection=selection,
    )
    process = build_process_ir("d d~ > z")
    expanded = service_module._ExpandedProcess(
        ProcessRequest.parse(process.process, name=_PROCESS_ID),
        process,
    )

    with pytest.raises(
        GenerationError,
        match="retain complete runtime selector coverage",
    ):
        backend._project_on_the_fly_process(
            expanded,
            BuiltinSMModel(),
            None,  # type: ignore[arg-type]
            index=0,
        )


def test_on_the_fly_construction_never_enters_materialized_process_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = service_module.GenerationBackend(_configuration().effective, None)
    source_path = _prepared_model_path()
    resolved = backend._resolve_model(ModelSource.from_path(source_path))
    model = resolved.model
    assert model is not None
    model_inputs = backend._recurrence_model_inputs(resolved)
    process = build_process_ir("d d~ > z")
    expanded = service_module._ExpandedProcess(
        ProcessRequest.parse(process.process, name=_PROCESS_ID),
        process,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("on-the-fly construction materialized a process lane")

    for name in (
        "build_color_plan",
        "compile_generic_dag",
        "_invoke_rust_recurrence_lowering_v2",
        "run_generic_dag_numerical_current_warmup",
        "run_recurrence_numerical_current_warmup",
    ):
        monkeypatch.setattr(service_module, name, forbidden)
    monkeypatch.setattr(backend, "_prepare_process_construction", forbidden)
    projected = backend._project_on_the_fly_process(
        expanded,
        model,
        model_inputs,
        index=0,
    )
    seed_identity = _seed_identity_from_source_projection(
        projected.projection.seed.to_json_dict()
    )
    wrong_seed_identity = _seed_identity_from_source_projection(
        projected.projection.seed.to_json_dict(),
        process_digest="f" * 64,
    )
    assert wrong_seed_identity["process_digest"] != (
        projected.projection.seed.process_digest
    )
    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_inspector_v1",
        lambda _payload: wrong_seed_identity,
    )
    with pytest.raises(GenerationError, match="process digest disagrees"):
        backend._construct_on_the_fly_artifact(
            projected,
            b"native-compact-seed",
            model,
            model_inputs,
            tmp_path,
            phase=PhaseHandle("test", None, 1),
        )
    assert not (tmp_path / "on-the-fly-runtimes").exists()

    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_inspector_v1",
        lambda _payload: seed_identity,
    )
    generated = backend._construct_on_the_fly_artifact(
        projected,
        b"native-compact-seed",
        model,
        model_inputs,
        tmp_path,
        phase=PhaseHandle("test", None, 1),
    )

    assert generated.artifact.process_id == _PROCESS_ID
    assert generated.artifact.runtime_path.parent == (
        tmp_path / "on-the-fly-runtimes" / _PROCESS_ID
    )
    source_projection = projected.projection.seed.to_json_dict()
    assert not {"dag", "color_plan", "recurrence", "direct_plan"}.intersection(
        source_projection
    )
    with PacbinReader.open(generated.artifact.runtime_path) as reader:
        member = reader.members[0]
        assert (member.logical_path, member.kind) == (
            _SEED_MEMBER,
            PacbinMemberKind.ON_THE_FLY_PROCESS_SEED,
        )
        assert reader.read_member(_SEED_MEMBER, length=member.length) == (
            b"native-compact-seed"
        )
    assert set(generated.artifact.runtime_metadata) == {
        "runtime_parameters",
        "prepared_parameter_defaults",
        "parameter_projection",
        "external_legs",
        "particle_masses",
        "normalization",
        "process_seed_identity",
    }


def test_multi_process_on_the_fly_writer_uses_one_ordered_native_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)
    calls: list[dict[str, object]] = []

    def batch_builder(
        ordered_sources: object,
        recurrence_json: object,
        direct_json: object,
        pack_digest: object,
        *,
        process_identities: object = (),
    ) -> tuple[bytes, ...]:
        sources = tuple(ordered_sources)  # type: ignore[arg-type]
        identities = tuple(process_identities)  # type: ignore[arg-type]
        calls.append(
            {
                "sources": sources,
                "recurrence_json": recurrence_json,
                "direct_json": direct_json,
                "pack_digest": pack_digest,
                "identities": identities,
            }
        )
        return tuple(
            f"native-compact-seed-{index}".encode() for index in range(len(sources))
        )

    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_batch_builder_v1",
        batch_builder,
    )

    def inspect_seed(payload: bytes) -> dict[str, object]:
        index = int(payload.rsplit(b"-", maxsplit=1)[1])
        assert calls
        ordered_sources = calls[0]["sources"]
        assert isinstance(ordered_sources, tuple)
        source_projection = json.loads(ordered_sources[index])
        assert isinstance(source_projection, dict)
        return _seed_identity_from_source_projection(source_projection)

    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_inspector_v1",
        inspect_seed,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("on-the-fly generation materialized a process lane")

    for name in (
        "build_color_plan",
        "compile_generic_dag",
        "_invoke_rust_recurrence_lowering_v2",
        "run_generic_dag_numerical_current_warmup",
        "run_recurrence_numerical_current_warmup",
    ):
        monkeypatch.setattr(service_module, name, forbidden)
    monkeypatch.setattr(
        service_module.GenerationBackend,
        "_prepare_process_construction",
        forbidden,
    )

    process_ids = ("d_dbar_to_z", "u_ubar_to_z")
    processes = ProcessSet.from_expressions(
        ("d d~ > z", "u u~ > z"),
        names=process_ids,
    )
    artifact = tmp_path / "multi-process-artifact"
    Generator(_configuration().effective).generate(
        processes,
        artifact,
        model=ModelSource.from_path(_prepared_model_path()),
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["identities"] == process_ids
    sources = call["sources"]
    assert isinstance(sources, tuple) and len(sources) == 2
    decoded_sources = tuple(json.loads(source) for source in sources)
    assert [source["schema_version"] for source in decoded_sources] == [1, 1]
    assert decoded_sources[0]["process_digest"] != decoded_sources[1]["process_digest"]
    for source in decoded_sources:
        assert not {"dag", "color_plan", "recurrence", "direct_plan"}.intersection(
            source
        )

    for index, process_id in enumerate(process_ids):
        runtime = artifact / "processes" / process_id / "on-the-fly-runtime.pacbin"
        with PacbinReader.open(runtime, verify_payloads=True) as reader:
            member = reader.members[0]
            assert reader.read_member(_SEED_MEMBER, length=member.length) == (
                f"native-compact-seed-{index}".encode()
            )


def test_public_generator_appends_on_the_fly_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_writer_identity(monkeypatch)

    def batch_builder(
        ordered_sources: object,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[bytes, ...]:
        return tuple(
            b"native-compact-seed:" + source
            for source in tuple(ordered_sources)  # type: ignore[arg-type]
        )

    def inspect_seed(payload: bytes) -> dict[str, object]:
        prefix = b"native-compact-seed:"
        assert payload.startswith(prefix)
        source_projection = json.loads(payload.removeprefix(prefix))
        assert isinstance(source_projection, dict)
        return _seed_identity_from_source_projection(source_projection)

    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_batch_builder_v1",
        batch_builder,
    )
    monkeypatch.setattr(
        service_module,
        "_invoke_rust_on_the_fly_seed_inspector_v1",
        inspect_seed,
    )

    generator = Generator(_configuration().effective)
    model = ModelSource.from_path(_prepared_model_path())
    artifact = tmp_path / "public-append-artifact"
    generator.generate("d d~ > z", artifact, model=model)
    generator.generate("u u~ > z", artifact, model=model, mode="append")

    manifest = load_manifest(artifact)
    assert tuple(record["id"] for record in manifest.processes) == (
        "d_dbar_to_z",
        "u_ubar_to_z",
    )
    assert [
        record.path
        for record in manifest.payloads
        if record.role == artifact_writer.STRUCTURAL_SOURCE_PROOF_ROLE
    ] == []
    assert [
        process.execution_mode for process in inspect_artifact(artifact).processes
    ] == ["on-the-fly", "on-the-fly"]


def test_on_the_fly_runtime_metadata_reuses_recurrence_support_contract() -> None:
    backend = service_module.GenerationBackend(_configuration().effective, None)
    resolved = backend._resolve_model(ModelSource.from_path(_prepared_model_path()))
    model = resolved.model
    assert model is not None
    inputs = backend._recurrence_model_inputs(resolved)
    process = build_process_ir("d d~ > z")
    projection = service_module.project_on_the_fly_process_seed_v1(
        process,
        inputs.catalog,
        model,
        coupling_order_policy="minimal",
        coupling_order_limits={},
    )

    metadata = build_on_the_fly_runtime_metadata(
        projection.seed.external_sources,
        projection.seed.parameter_projection,
        inputs.catalog,
        model,
        projection.runtime_normalization,
    )

    assert metadata["external_legs"] == [
        {
            "source_slot": leg.source_slot,
            "public_label": leg.public_label,
            "physical_pdg": leg.physical_pdg,
            "outgoing_pdg": leg.outgoing_pdg,
            "is_initial": leg.is_initial,
        }
        for leg in projection.seed.external_sources
    ]
    assert metadata["parameter_projection"] == [
        {
            "runtime_slot": row.runtime_slot,
            "runtime_name": row.runtime_name,
            "parameter_template_id": row.parameter_template_id,
            "prepared_parameter_id": row.prepared_parameter_id,
            "component": row.component,
        }
        for row in projection.seed.parameter_projection
    ]
    assert metadata["normalization"] == projection.runtime_normalization
    assert "source_templates" not in metadata
    assert "public_color_flows" not in metadata


def test_on_the_fly_singleton_seed_bridge_wraps_plural_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def binding(*args: object) -> list[bytes]:
        calls.append(args)
        return [b"stable-seed-bytes"]

    module = SimpleNamespace(_build_on_the_fly_process_seeds_v1=binding)
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    result = service_module._invoke_rust_on_the_fly_seed_builder_v1(
        b"source",
        b"recurrence",
        b"direct",
        "a" * 64,
    )

    assert result == b"stable-seed-bytes"
    assert calls == [((b"source",), b"recurrence", b"direct", "a" * 64)]


def test_on_the_fly_native_seed_inspector_returns_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _seed_identity()
    module = SimpleNamespace(
        _inspect_on_the_fly_process_seed_v1=lambda payload: (
            json.dumps(identity) if payload == b"seed" else "{}"
        )
    )
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    assert service_module._invoke_rust_on_the_fly_seed_inspector_v1(b"seed") == (
        identity
    )


def test_on_the_fly_native_seed_identity_is_bound_to_source_projection() -> None:
    projection_json: dict[str, object] = {
        "process_digest": "a" * 64,
        "external_permutation": [0, 1],
        "external_sources": [
            {
                "source_slot": 0,
                "public_label": 1,
                "is_initial": True,
                "states": [{"state_index": 0, "public_helicity": -1}],
            },
            {
                "source_slot": 1,
                "public_label": 2,
                "is_initial": False,
                "states": [{"state_index": 0, "public_helicity": 1}],
            },
        ],
    }
    projection = SimpleNamespace(
        process_digest="a" * 64,
        external_permutation=(0, 1),
        external_sources=(
            SimpleNamespace(
                source_slot=0,
                public_label=1,
                is_initial=True,
                source_states=(SimpleNamespace(state_index=0, public_helicity=-1),),
            ),
            SimpleNamespace(
                source_slot=1,
                public_label=2,
                is_initial=False,
                source_states=(SimpleNamespace(state_index=0, public_helicity=1),),
            ),
        ),
    )
    identity = _seed_identity_from_source_projection(projection_json)
    service_module._bind_on_the_fly_process_seed_identity_v1(
        identity,
        projection,  # type: ignore[arg-type]
    )

    wrong_digest = {**identity, "process_digest": "f" * 64}
    wrong_permutation = {**identity, "external_permutation": [1, 0]}
    candidates: list[tuple[dict[str, object], str]] = [
        (wrong_digest, "process digest disagrees"),
        (wrong_permutation, "external permutation disagrees"),
    ]
    for field, replacement in (
        ("source_slot", 7),
        ("public_label", 7),
        ("is_initial", False),
    ):
        candidate = copy.deepcopy(identity)
        sources = candidate["external_sources"]
        assert isinstance(sources, list) and isinstance(sources[0], dict)
        sources[0][field] = replacement
        candidates.append((candidate, "external source domain disagrees"))
    for field, replacement in (
        ("state_index", 7),
        ("public_helicity", 7),
    ):
        candidate = copy.deepcopy(identity)
        sources = candidate["external_sources"]
        assert isinstance(sources, list) and isinstance(sources[0], dict)
        states = sources[0]["states"]
        assert isinstance(states, list) and isinstance(states[0], dict)
        states[0][field] = replacement
        candidates.append((candidate, "external source domain disagrees"))

    for candidate, message in candidates:
        with pytest.raises(GenerationError, match=message):
            service_module._bind_on_the_fly_process_seed_identity_v1(
                candidate,
                projection,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        (None, "does not provide the private"),
        (lambda _payload: b"not-json-text", "inspection failed"),
        (lambda _payload: "not-json", "inspection failed"),
        (
            lambda _payload: json.dumps(
                {**_seed_identity(), "abi": "stale-seed-identity-v0"}
            ),
            "unsupported ABI",
        ),
        (
            lambda _payload: json.dumps({**_seed_identity(), "invented": True}),
            "wrong fields",
        ),
    ),
)
def test_on_the_fly_native_seed_inspector_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    binding: object,
    message: str,
) -> None:
    module = SimpleNamespace()
    if binding is not None:
        module._inspect_on_the_fly_process_seed_v1 = binding
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    with pytest.raises(GenerationError, match=message):
        service_module._invoke_rust_on_the_fly_seed_inspector_v1(b"seed")


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        (None, "does not provide the private"),
        (lambda *_args: [], "wrong process count"),
        (lambda *_args: [b""], "empty payload for process index 0"),
        (lambda *_args: ["not-bytes"], "empty payload for process index 0"),
        (
            lambda *_args: (_ for _ in ()).throw(RuntimeError("native failure")),
            "process-seed construction failed: native failure",
        ),
    ),
)
def test_on_the_fly_native_seed_bridge_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    binding: object,
    message: str,
) -> None:
    module = SimpleNamespace()
    if binding is not None:
        module._build_on_the_fly_process_seeds_v1 = binding
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    with pytest.raises(GenerationError, match=message):
        service_module._invoke_rust_on_the_fly_seed_builder_v1(
            b"{}",
            b"{}",
            b"{}",
            "a" * 64,
        )


def test_on_the_fly_native_seed_bridge_preserves_indexed_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def binding(*_args: object) -> object:
        raise RuntimeError(
            "on-the-fly process seed at index 1 "
            '(process digest "abc") failed: stale source'
        )

    module = SimpleNamespace(_build_on_the_fly_process_seeds_v1=binding)
    monkeypatch.setattr(service_module.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(service_module, "verify_native_module", lambda _module: None)

    with pytest.raises(
        GenerationError,
        match=r"process index 1 'u_ubar_to_z'.*process digest.*stale source",
    ):
        service_module._invoke_rust_on_the_fly_seed_batch_builder_v1(
            (b"{}", b"{}"),
            b"{}",
            b"{}",
            "a" * 64,
            process_identities=("d_dbar_to_z", "u_ubar_to_z"),
        )
