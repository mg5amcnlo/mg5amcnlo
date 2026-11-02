# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import get_args

import pytest

import pyamplicol.generation.artifact_writer as artifact_writer
from pyamplicol._internal.versions import (
    EAGER_DAG_F64_RUNTIME_CAPABILITY,
    EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
    EVALUATOR_RUNTIME_CAPABILITIES,
    KNOWN_EVALUATOR_RUNTIME_CAPABILITIES,
)
from pyamplicol.api.requests import ModelSource
from pyamplicol.artifacts import ArtifactBuilder, load_manifest
from pyamplicol.artifacts.manifest import PORTABLE_64LE_TARGET
from pyamplicol.config import (
    Action,
    EvaluatorConfig,
    GenerationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.generation.artifact_writer import (
    EagerPlanV3ProcessArtifact,
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from pyamplicol.generation.evaluator_container import PacbinReader
from pyamplicol.generation.structural_source_proof import (
    ROLE as STRUCTURAL_SOURCE_PROOF_ROLE,
)
from pyamplicol.generation.structural_source_proof import (
    validate_generation_structural_proof,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.prepared import (
    PREPARED_KERNEL_VARIANT_ABI,
    PreparedKernelPack,
    PreparedKernelRecord,
    PreparedKernelVariantRecord,
    prepared_expression_digest,
    prepared_input_contract_digest,
    prepared_kernel_pack_manifest_identity_sha256,
    prepared_optimization_settings_digest,
    prepared_output_contract_digest,
    write_prepared_model_bundle,
)


def _plane_application(
    root: str,
    *,
    input_arity: int,
    output_arity: int,
) -> dict[str, object]:
    return {
        "application_path": f"{root}/plane-application.symjit",
        "application_abi": artifact_writer.SYMJIT_PLANE_APPLICATION_ABI,
        "storage_abi": artifact_writer.SYMJIT_APPLICATION_ABI,
        "element_layout": "split-complex-plane-major",
        "descriptor_order": "inputs-re-im-then-outputs-re-im",
        "input_complex_count": input_arity,
        "output_complex_count": output_arity,
        "input_plane_count": 2 * input_arity,
        "output_plane_count": 2 * output_arity,
        "compiler_type": "native",
        "translation_mode": "symbolica-structured-instructions",
        "optimization_level": 2,
        "simd": True,
        "complex": True,
        "fast_math": True,
        "fast_complex": False,
        "compression": True,
        "threading": False,
        "direct_arena": True,
        "source_digest": "0" * 64,
        "target": {"triple": "test-native", "cpu_features": []},
    }


def _kernel(kernel_id: int, signature: str) -> PreparedKernelRecord:
    root = f"kernels/{kernel_id}"
    return PreparedKernelRecord(
        kernel_id=kernel_id,
        contract_kind="vertex",
        canonical_signature=signature,
        input_arity=1,
        output_arity=1,
        input_layout=("input",),
        input_contracts=(
            {
                "role": "current",
                "component": 0,
                "symbol": "pyamplicol::input",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
        ),
        output_layout=("output",),
        exact_expressions=("pyamplicol::input",),
        exact_evaluator_state_path=f"{root}/exact.evaluator.bin",
        f64_evaluator_manifest={
            "kind": "symjit-application-evaluator",
            "application_abi": artifact_writer.SYMJIT_APPLICATION_ABI,
            "optimization_level": 2,
            "settings": {"jit_optimization_level": 2},
            "input_len": 1,
            "output_len": 1,
            "application_path": f"{root}/application.symjit",
            "plane_application": _plane_application(
                root,
                input_arity=1,
                output_arity=1,
            ),
            "evaluator_state_path": f"{root}/exact.evaluator.bin",
        },
    )


def _variant(kernel: PreparedKernelRecord) -> PreparedKernelVariantRecord:
    settings = {"jit_optimization_level": 2}
    root = f"kernels/{kernel.kernel_id}/variants/independent-block-4"
    return PreparedKernelVariantRecord(
        variant_id="independent-block-4",
        variant_abi=PREPARED_KERNEL_VARIANT_ABI,
        kind="independent-block",
        block_size=4,
        lane_layout="lane-major",
        base_kernel_id=kernel.kernel_id,
        base_canonical_signature=kernel.canonical_signature,
        base_expression_digest=prepared_expression_digest(kernel.exact_expressions),
        base_input_contract_digest=prepared_input_contract_digest(
            kernel.input_layout,
            kernel.input_contracts,
        ),
        base_output_contract_digest=prepared_output_contract_digest(
            kernel.output_layout
        ),
        backend="jit",
        optimization_settings_digest=prepared_optimization_settings_digest(settings),
        input_arity=4,
        output_arity=4,
        input_lane_stride=1,
        output_lane_stride=1,
        input_layout=tuple(f"lane:{lane}:input" for lane in range(4)),
        output_layout=tuple(f"lane:{lane}:output" for lane in range(4)),
        f64_evaluator_manifest={
            "kind": "symjit-application-evaluator",
            "application_abi": artifact_writer.SYMJIT_APPLICATION_ABI,
            "optimization_level": 2,
            "settings": settings,
            "input_len": 4,
            "output_len": 4,
            "application_path": f"{root}/application.symjit",
            "plane_application": _plane_application(
                root,
                input_arity=4,
                output_arity=4,
            ),
            "evaluator_state_path": f"{root}/exact.evaluator.bin",
        },
    )


def _prepared_model(
    tmp_path: Path,
    *,
    signatures: dict[int, str],
    bundle_name: str,
) -> tuple[Path, object]:
    kernels = tuple(
        _kernel(kernel_id, signature)
        for kernel_id, signature in sorted(signatures.items())
    )
    settings = {"jit_optimization_level": 2}
    pack = PreparedKernelPack(
        backend="jit",
        optimization_settings=settings,
        producer={"distribution": "pyamplicol", "version": "test"},
        dependency_abis={
            "symjit_application": artifact_writer.SYMJIT_APPLICATION_ABI,
            "symjit_plane_application": "pyamplicol-symjit-plane-application-v2",
        },
        provenance={"compiled_model": "test"},
        target={
            "portable": True,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "symjit-storage-v3-portable",
            "cpu_features": [],
        },
        resolver_manifest={
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "built-in-sm",
        },
        kernels=kernels,
        kernel_variants=tuple(_variant(kernel) for kernel in kernels),
    )
    source_model = compile_model_source("built-in-sm", use_cache=False)
    bundle = write_prepared_model_bundle(
        tmp_path / bundle_name,
        compiled_model=source_model.to_dict(),
        kernel_pack=pack,
        payloads={
            path: f"payload:{path}".encode() for path in pack.referenced_payload_paths
        },
    )
    return bundle, compile_model_source(bundle, use_cache=False)


def _v3_process(
    tmp_path: Path,
    *,
    process_id: str,
    kernel_ids: frozenset[int],
) -> EagerPlanV3ProcessArtifact:
    runtime = tmp_path / f"{process_id}.pacbin"
    runtime_bytes = f"native-eager-runtime:{process_id}".encode()
    runtime.write_bytes(runtime_bytes)
    return EagerPlanV3ProcessArtifact(
        process_id=process_id,
        expression="d d~ > z",
        color_accuracy="full",
        external_pdgs=(1, -1, 23),
        aliases=(),
        physics={
            "schema_version": 1,
            "kind": "pyamplicol-resolved-physics",
            "process_id": process_id,
        },
        eager_runtime_path=runtime,
        eager_runtime_size_bytes=len(runtime_bytes),
        eager_runtime_sha256=hashlib.sha256(runtime_bytes).hexdigest(),
        eager_runtime_member_count=3,
        eager_runtime_unpacked_size_bytes=256,
        eager_runtime_index_sha256="3" * 64,
        lowering_input_sha256=hashlib.sha256(process_id.encode()).hexdigest(),
        referenced_kernel_ids=kernel_ids,
        inspection_summary={
            "stage_count": 2,
            "invocation_count": 3,
            "attachment_count": 4,
            "finalization_count": 2,
            "closure_count": 1,
            "selector_domain_count": 0,
        },
        point_tile_size=128,
        workspace_mib=64,
        dag_summary={
            "current_count": 4,
            "source_count": 3,
            "interaction_count": 3,
            "amplitude_root_count": 1,
            "truncated": False,
        },
        validation_point=ValidationPointRecord(
            process_id=process_id,
            process="d d~ > z",
            seed=7,
            error="not sampled in writer test",
        ),
        generation_filters={},
    )


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _inject_stale_eager_payload(output: Path) -> str:
    stale_path = "model/eager-kernels/superseded/application.symjit"
    manifest = load_manifest(output)
    with ArtifactBuilder(output, mode="append") as builder:
        builder.add_bytes(
            stale_path,
            b"superseded evaluator payload",
            role="evaluator-state",
            media_type="application/vnd.symjit.application",
            target=manifest.producer["target"],
        )
        builder.finalize(
            kind=manifest.kind,
            producer=manifest.producer,
            model=manifest.model,
            configuration=manifest.configuration,
            processes=manifest.processes,
            default_process_id=manifest.default_process_id,
            runtime=manifest.runtime,
            dependencies=manifest.dependencies,
            extensions=manifest.extensions,
        )
    return stale_path


def test_schema_v3_writer_has_no_legacy_eager_process_variant() -> None:
    assert not hasattr(artifact_writer, "EagerProcessArtifact")
    assert "EagerProcessArtifact" not in artifact_writer.__all__
    assert {
        variant.__name__ for variant in get_args(artifact_writer.ProcessArtifact)
    } == {
        "CompiledProcessArtifact",
        "EagerPlanV3ProcessArtifact",
        "OnTheFlyProcessArtifact",
        "RecurrenceProcessArtifact",
    }


def test_legacy_eager_capabilities_are_known_but_not_supported() -> None:
    legacy = {
        EAGER_DAG_F64_RUNTIME_CAPABILITY,
        EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
    }
    assert legacy <= KNOWN_EVALUATOR_RUNTIME_CAPABILITIES
    assert legacy.isdisjoint(EVALUATOR_RUNTIME_CAPABILITIES)


def test_plan_v3_writer_filters_pack_and_appends_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: ({"triple": "aarch64-apple-darwin", "cpu_features": []}, 1),
    )
    monkeypatch.setattr(
        artifact_writer,
        "_derive_eager_direct_descriptor",
        lambda source, **_widths: b"direct-table:" + source,
    )
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
    signatures = {
        10: "a" * 64,
        20: "b" * 64,
        30: "c" * 64,
    }
    bundle, compiled_model = _prepared_model(
        tmp_path,
        signatures=signatures,
        bundle_name="prepared",
    )
    configuration = _GenerationConfigProvenance.from_config(
        RunConfig(
            action=Action.GENERATE,
            generation=GenerationConfig(emit_api_bundle=False),
            evaluator=EvaluatorConfig(
                execution_mode="eager",
                jit=JITConfig(optimization_level=2),
            ),
        )
    )
    first = _v3_process(
        tmp_path,
        process_id="d_dbar_to_z",
        kernel_ids=frozenset({10}),
    )
    output = tmp_path / "artifact"
    progress: list[dict[str, object]] = []

    write_schema_v3_artifact(
        output,
        mode="error",
        source=ModelSource.from_path(bundle),
        compiled_model=compiled_model,
        configuration=configuration,
        processes=(first,),
        timings={"total": 0.1},
        api_bundle_hook=None,
        progress_callback=progress.append,
    )

    assert [event["step"] for event in progress] == [
        "global payloads",
        "prepared kernel pack",
        "process payloads",
        "publishing artifact",
    ]
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
    structural_records = [
        record
        for record in manifest.payloads
        if record.role == STRUCTURAL_SOURCE_PROOF_ROLE
    ]
    assert len(structural_records) == 1
    structural = json.loads(
        (output / structural_records[0].path).read_text(encoding="utf-8")
    )
    validate_generation_structural_proof(
        structural,
        artifact_root=output,
        expected_process_id="d_dbar_to_z",
        expected_source_revision=source_revision,
        expected_native_build_inputs_sha256=native_inputs,
    )
    capabilities = {
        artifact_writer.EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        artifact_writer.EAGER_PLAN_V3_RUNTIME_CAPABILITY,
    }
    assert set(manifest.runtime["required_runtime_capabilities"]) == capabilities
    execution = json.loads(
        (output / "processes/d_dbar_to_z/execution.json").read_text(encoding="utf-8")
    )
    assert execution["eager_plan_abi"] == artifact_writer.EAGER_PLAN_V3_ABI
    assert set(execution["required_runtime_capabilities"]) == capabilities
    assert execution["plan"]["inspection_summary"] == first.inspection_summary

    pack_identity = manifest.extensions["eager_prepared_pack"]
    assert pack_identity["kind"] == "pyamplicol-prepared-kernel-pack-identity"
    assert pack_identity["abi"] == "pyamplicol-prepared-kernel-pack-identity-v2"
    assert pack_identity["kernel_count"] == 3
    assert pack_identity["identity_sha256"] == (
        prepared_kernel_pack_manifest_identity_sha256(
            compiled_model.prepared_bundle.manifest  # type: ignore[union-attr]
        )
    )
    pack = json.loads(
        (output / "model/eager-kernel-pack.json").read_text(encoding="utf-8")
    )
    assert {kernel["kernel_id"] for kernel in pack["kernels"]} == {10}
    assert {variant["base_kernel_id"] for variant in pack["kernel_variants"]} == {10}
    assert pack["recurrence_template"] is None
    assert pack["recurrence_direct_template"] is None
    assert (
        pack["kernels"][0]["f64_evaluator_manifest"]["direct_table"]["capability"]
        == artifact_writer.EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY
    )

    declared = {record.path for record in manifest.payloads}
    assert "model/eager-kernel-pack.json" in declared
    assert "model/eager-kernels/kernels/10/eager-direct-table-descriptor-v1.bin" in (
        declared
    )
    assert not any(path.endswith((".symjit", ".evaluator.bin")) for path in declared)
    container_extension = manifest.extensions["evaluator_payload_container"]
    with PacbinReader.open(output / "evaluators.pacbin") as container:
        members = {member.logical_path for member in container.members}
        assert container_extension == {
            "kind": "pyamplicol-evaluator-payload-container",
            "schema_version": 1,
            "storage_abi": "pacbin-v1",
            "path": "evaluators.pacbin",
            "member_count": len(container.members),
            "unpacked_size_bytes": sum(member.length for member in container.members),
            "index_sha256": container.index.index_sha256,
        }
        assert any(
            path.startswith("model/eager-kernels/kernels/10/") for path in members
        )
        assert not any(
            path.startswith("model/eager-kernels/kernels/20/")
            or path.startswith("model/eager-kernels/kernels/30/")
            for path in members
        )

    stale_path = _inject_stale_eager_payload(output)
    second = _v3_process(
        tmp_path,
        process_id="u_ubar_to_z",
        kernel_ids=frozenset({20}),
    )
    write_schema_v3_artifact(
        output,
        mode="append",
        source=ModelSource.from_path(bundle),
        compiled_model=compiled_model,
        configuration=configuration,
        processes=(second,),
        timings={"total": 0.2},
        api_bundle_hook=None,
    )

    appended = load_manifest(output)
    appended_structural_records = [
        record
        for record in appended.payloads
        if record.role == STRUCTURAL_SOURCE_PROOF_ROLE
    ]
    assert {record.process_id for record in appended_structural_records} == {
        "d_dbar_to_z",
        "u_ubar_to_z",
    }
    for record in appended_structural_records:
        structural = json.loads((output / record.path).read_text(encoding="utf-8"))
        validate_generation_structural_proof(
            structural,
            artifact_root=output,
            expected_process_id=str(record.process_id),
            expected_source_revision=source_revision,
            expected_native_build_inputs_sha256=native_inputs,
        )
    assert {str(record["id"]) for record in appended.processes} == {
        "d_dbar_to_z",
        "u_ubar_to_z",
    }
    assert appended.extensions["eager_prepared_pack"] == pack_identity
    appended_pack = json.loads(
        (output / "model/eager-kernel-pack.json").read_text(encoding="utf-8")
    )
    assert {kernel["kernel_id"] for kernel in appended_pack["kernels"]} == {10, 20}
    assert {
        variant["base_kernel_id"] for variant in appended_pack["kernel_variants"]
    } == {10, 20}
    assert stale_path not in {record.path for record in appended.payloads}
    assert not (output / stale_path).exists()
    with PacbinReader.open(output / "evaluators.pacbin") as container:
        members = {member.logical_path for member in container.members}
        assert stale_path not in members
        assert any(
            path.startswith("model/eager-kernels/kernels/10/") for path in members
        )
        assert any(
            path.startswith("model/eager-kernels/kernels/20/") for path in members
        )
        assert not any(
            path.startswith("model/eager-kernels/kernels/30/") for path in members
        )

    shifted_bundle, shifted_model = _prepared_model(
        tmp_path,
        signatures={
            10: "d" * 64,
            20: "e" * 64,
            30: "f" * 64,
        },
        bundle_name="shifted",
    )
    rejected = _v3_process(
        tmp_path,
        process_id="rejected",
        kernel_ids=frozenset({30}),
    )
    before = _tree_snapshot(output)
    with pytest.raises(ValueError, match="prepared kernel pack identity differs"):
        write_schema_v3_artifact(
            output,
            mode="append",
            source=ModelSource.from_path(shifted_bundle),
            compiled_model=shifted_model,
            configuration=configuration,
            processes=(rejected,),
            timings={"total": 0.3},
            api_bundle_hook=None,
        )
    assert _tree_snapshot(output) == before
    assert not tuple(tmp_path.glob(".artifact.staging-*"))


def test_append_rejects_legacy_eager_plan_v2_before_pack_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "legacy-eager"
    legacy_capability = "rusticol.eager-dag.complex-f64.v1"
    with ArtifactBuilder(output) as builder:
        builder.add_bytes(
            "config/requested.toml",
            b"",
            role="configuration-requested",
            media_type="application/toml",
        )
        builder.add_bytes(
            "config/effective.toml",
            b"",
            role="configuration-effective",
            media_type="application/toml",
        )
        builder.add_json(
            "processes/legacy/physics.json",
            {"schema_version": 1},
            role="runtime-physics",
            process_id="legacy",
        )
        builder.add_json(
            "runtime/evaluators.json",
            {"schema_version": 3, "kind": "pyamplicol-runtime-execution-set"},
            role="evaluator-manifest",
        )
        builder.finalize(
            kind="pyamplicol-process",
            producer={
                "distribution": "pyamplicol",
                "version": "test",
                "versions": {
                    "python_api": 1,
                    "toml": 1,
                    "compiled_model": 9,
                    "process_artifact": 3,
                    "runtime_physics": 1,
                    "symbolica_serialization": "test",
                    "c_abi": 1,
                },
                "target": {"triple": "aarch64-apple-darwin", "cpu_features": []},
            },
            model={
                "name": "built-in-sm",
                "source_kind": "built-in-sm",
                "content_sha256": "1" * 64,
                "compiled_schema_version": 9,
                "restriction": None,
            },
            configuration={
                "toml_schema_version": 1,
                "requested_path": "config/requested.toml",
                "effective_path": "config/effective.toml",
                "adjustments": [],
            },
            processes=(
                {
                    "id": "legacy",
                    "expression": "d d~ > z",
                    "color_accuracy": "full",
                    "external_pdgs": [1, -1, 23],
                    "physics_path": "processes/legacy/physics.json",
                    "required_runtime_capabilities": [legacy_capability],
                    "aliases": [],
                },
            ),
            default_process_id="legacy",
            runtime={
                "engine": "rusticol",
                "engine_version": "test",
                "evaluator_manifest_path": "runtime/evaluators.json",
                "api_bundle_path": None,
                "required_runtime_capabilities": [legacy_capability],
            },
        )

    assert load_manifest(output).runtime["required_runtime_capabilities"] == (
        legacy_capability,
    )
    monkeypatch.setattr(
        artifact_writer,
        "_default_api_bundle_hook",
        lambda: pytest.fail("legacy rejection must precede append preparation"),
    )
    opaque = object()

    with pytest.raises(
        ValueError,
        match=r"legacy eager plan-v2 artifacts cannot be extended.*replace mode",
    ):
        artifact_writer.write_schema_v3_artifact(
            output,
            mode="append",
            source=opaque,
            compiled_model=opaque,
            configuration=opaque,
            processes=(opaque,),
            timings={},
        )
