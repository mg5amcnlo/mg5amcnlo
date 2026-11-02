# SPDX-License-Identifier: 0BSD
"""Python metadata contracts for direct-arena recurrence artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol._internal import versions
from pyamplicol.api.errors import ArtifactError
from pyamplicol.artifacts import inspection, load_manifest
from pyamplicol.artifacts.writer import ArtifactBuilder
from pyamplicol.generation import artifact_writer, recurrence_physics, service
from pyamplicol.generation.recurrence_columnar import RecurrenceSemanticDigestV1
from pyamplicol.generation.recurrence_schedule_sharing import (
    RecurrenceProcessExecutorPack,
    RecurrenceProcessRemap,
    recurrence_helicity_selector_schedule_digest,
)
from pyamplicol.generation.validation import ValidationPointRecord


def _digest(character: str) -> str:
    return character * 64


def test_recurrence_bootstrap_publishes_exact_authenticated_runtime_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_id = "g_g_to_g_g"
    prefix = f"processes/{process_id}"
    execution_path = f"{prefix}/execution.json"
    physics_path = f"{prefix}/physics.json"
    bootstrap_path = f"{prefix}/recurrence-bootstrap.bin"
    execution = b'{"kind":"pyamplicol-recurrence-runtime","schema_version":1}\n'
    physics = b'{"kind":"pyamplicol-resolved-physics","schema_version":1}\n'
    target = {"triple": "portable-64le", "cpu_features": []}
    capability = artifact_writer.RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY
    output = tmp_path / "artifact"
    image = b"native-recurrence-bootstrap-image-v1"
    captured: dict[str, object] = {}

    def build(
        context_json: bytes,
        execution_json: bytes,
        physics_json: bytes,
    ) -> bytes:
        captured.update(
            context=json.loads(context_json),
            execution=execution_json,
            physics=physics_json,
        )
        return image

    monkeypatch.setattr(
        artifact_writer,
        "_recurrence_bootstrap_operation",
        lambda: build,
    )
    producer = {
        "distribution": "pyamplicol",
        "version": "0.1.0",
        "versions": {
            "python_api": 1,
            "toml": 1,
            "compiled_model": 1,
            "process_artifact": 3,
            "runtime_physics": 1,
            "symbolica_serialization": "1",
            "c_abi": 1,
        },
        "target": target,
    }
    process_record = {
        "id": process_id,
        "expression": "g g > g g",
        "color_accuracy": "full",
        "external_pdgs": [21, 21, 21, 21],
        "physics_path": physics_path,
        "required_runtime_capabilities": [capability],
        "aliases": [],
    }
    runtime = {
        "engine": "rusticol",
        "engine_version": "0.1.0",
        "evaluator_manifest_path": execution_path,
        "api_bundle_path": None,
        "required_runtime_capabilities": [capability],
    }
    container = {
        "kind": "pyamplicol-evaluator-payload-container",
        "schema_version": 1,
        "storage_abi": "pacbin-v1",
        "path": "evaluators.pacbin",
        "member_count": 0,
        "unpacked_size_bytes": 0,
        "index_sha256": _digest("b"),
    }

    with ArtifactBuilder(output) as builder:
        builder.add_bytes(
            execution_path,
            execution,
            role="evaluator-manifest",
            media_type="application/json",
            process_id=process_id,
        )
        builder.add_bytes(
            physics_path,
            physics,
            role="runtime-physics",
            media_type="application/json",
            process_id=process_id,
        )
        builder.add_bytes(
            "evaluators.pacbin",
            b"container",
            role="evaluator-state",
            media_type="application/octet-stream",
            target=target,
        )
        artifact_writer._write_recurrence_bootstraps(
            builder,
            processes=(SimpleNamespace(process_id=process_id),),  # type: ignore[arg-type]
            process_records=(process_record,),
            producer=producer,
            runtime=runtime,
            evaluator_payload_container=container,
            target=target,
        )
        builder.finalize(
            kind="pyamplicol-process",
            producer=producer,
            model={
                "name": "sm",
                "source_kind": "built-in-sm",
                "content_sha256": _digest("a"),
                "compiled_schema_version": 1,
            },
            configuration={
                "toml_schema_version": 1,
                "requested_path": "configuration/requested.toml",
                "effective_path": "configuration/effective.toml",
                "adjustments": [],
            },
            processes=(process_record,),
            runtime=runtime,
            default_process_id=process_id,
        )

    manifest = load_manifest(output, verify_payloads=True)
    bootstrap_record = next(
        record for record in manifest.payloads if record.path == bootstrap_path
    )
    assert bootstrap_record.role == "evaluator-state"
    assert bootstrap_record.process_id == process_id
    assert bootstrap_record.target == {
        "triple": "portable-64le",
        "cpu_features": (),
    }
    bootstrap_bytes = (output / bootstrap_path).read_bytes()
    assert bootstrap_bytes == image
    assert hashlib.sha256(bootstrap_bytes).hexdigest() == bootstrap_record.sha256
    assert captured["execution"] == execution
    assert captured["physics"] == physics
    context = captured["context"]
    assert isinstance(context, dict)
    assert context["process"] == process_record
    assert context["producer"] == producer
    assert context["runtime"] == runtime
    assert context["evaluator_payload_container"] == container
    assert Path(context["staged_root"]).is_absolute()
    inventory = context["payloads"]
    assert isinstance(inventory, list)
    assert bootstrap_path not in {record["path"] for record in inventory}
    assert {record["path"] for record in inventory} == {
        execution_path,
        physics_path,
        "evaluators.pacbin",
    }


def test_recurrence_bootstrap_v1_skips_multi_process_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_writer,
        "_recurrence_bootstrap_operation",
        lambda: pytest.fail("multi-process artifacts must not build a v1 bootstrap"),
    )
    # The multi-process guard returns before consulting the builder at all.
    artifact_writer._write_recurrence_bootstraps(
        SimpleNamespace(),  # type: ignore[arg-type]
        processes=(SimpleNamespace(process_id="recurrence"),),  # type: ignore[arg-type]
        process_records=(
            {"id": "recurrence"},
            {"id": "compiled"},
        ),
        producer={},
        runtime={},
        evaluator_payload_container=None,
        target={},
    )


def test_recurrence_inspection_summary_binds_public_process_alias() -> None:
    native_summary = {
        "execution_mode": "recurrence",
        "process_id": "g_g_to_g_g",
        "semantic_digest": _digest("1"),
        "schedule_digest": _digest("2"),
    }

    summary = service._process_scoped_recurrence_inspection_summary(
        native_summary,
        process_id="gg_n2",
        semantic_digest=_digest("3"),
        schedule_digest=_digest("4"),
    )

    assert summary["process_id"] == "gg_n2"
    assert summary["semantic_digest"] == _digest("3")
    assert summary["schedule_digest"] == _digest("4")
    assert native_summary["process_id"] == "g_g_to_g_g"


def _recurrence_process() -> SimpleNamespace:
    return SimpleNamespace(
        expression="d d~ > z g",
        process_id="d_dbar_to_z_g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        recurrence_schedule_size_bytes=256,
        recurrence_schedule_sha256=_digest("a"),
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=192,
        recurrence_schedule_index_sha256=_digest("b"),
        builder_input_sha256=_digest("c"),
        prepared_kernel_pack_digest=_digest("d"),
        direct_template_catalog_digest=_digest("e"),
        recurrence_process_executor_pack=SimpleNamespace(
            compiled_model_digest=_digest("2"),
            recurrence_template_catalog_digest=_digest("3"),
            prepared_kernel_pack_digest=_digest("d"),
            direct_template_catalog_digest=_digest("e"),
        ),
        inspection_summary={
            "execution_mode": "recurrence",
            "process_id": "d_dbar_to_z_g",
            "semantic_digest": _digest("c"),
            "schedule_digest": _digest("f"),
            "prepared_kernel_count": 2,
            "schedule": {
                "source_row_count": 4,
                "contribution_count": 34,
                "finalization_count": 22,
                "closure_term_count": 12,
            },
            "direct_arena": {
                "semantic_component_count": 48,
                "current_arena_components": 32,
                "arena_component_reuse_count": 16,
                "momentum_form_count": 7,
                "row_group_count": 11,
                "packed_input_bytes": 0,
                "packed_output_bytes": 0,
                "scatter_bytes": 0,
            },
        },
        runtime_metadata={},
        point_tile_size=1024,
        workspace_mib=256,
        recurrence_summary={"lc_flow_layout": "topology-replay"},
        color_contraction_summary=None,
        helicity_selector_companion=None,
        process_digest=_digest("1"),
    )


def _selector_companion() -> artifact_writer.RecurrenceHelicitySelectorPlanArtifact:
    base_schedule_digest = _digest("6")
    dispatch_sha256 = _digest("4")
    return artifact_writer.RecurrenceHelicitySelectorPlanArtifact(
        recurrence_schedule_path=Path("selector-runtime.pacbin"),
        recurrence_base_schedule_digest=base_schedule_digest,
        recurrence_schedule_digest=recurrence_helicity_selector_schedule_digest(
            base_schedule_digest,
            dispatch_sha256,
        ),
        recurrence_native_schedule_semantic_digest=base_schedule_digest,
        recurrence_schedule_size_bytes=512,
        recurrence_schedule_sha256=_digest("8"),
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=448,
        recurrence_schedule_index_sha256=_digest("a"),
        helicity_dispatch_path=Path("selector-dispatch.bin"),
        helicity_dispatch_size_bytes=96,
        helicity_dispatch_sha256=dispatch_sha256,
        helicity_dispatch_base_runtime_layout_digest=_digest("5"),
        helicity_dispatch_resolved_helicity_count=8,
        physical_destination_count=3,
        builder_input_sha256=_digest("b"),
        inspection_summary={
            "execution_mode": "recurrence",
            "runtime_layout_digest": _digest("5"),
            "schedule_digest": base_schedule_digest,
        },
        referenced_kernel_ids=frozenset({0, 1}),
        recurrence_process_remap=SimpleNamespace(),  # type: ignore[arg-type]
        recurrence_process_executor_pack=SimpleNamespace(
            compiled_model_digest=_digest("2"),
            recurrence_template_catalog_digest=_digest("3"),
            prepared_kernel_pack_digest=_digest("d"),
            direct_template_catalog_digest=_digest("e"),
        ),  # type: ignore[arg-type]
    )


def _identity_remap(*, physical_sector_count: int) -> RecurrenceProcessRemap:
    return RecurrenceProcessRemap(
        source_slots=(0,),
        source_momentum_signs=(1,),
        source_helicity_signs=(1,),
        source_state_offsets=(0, 1),
        source_state_indices=(0,),
        public_flow_ids=(),
        physical_sector_ids=tuple(range(physical_sector_count)),
        state_template_count=0,
        source_template_count=0,
        direct_executor_count=1,
        parameter_slot_count=0,
        bijection_digest=_digest("7"),
    )


def _executor_pack(runtime_layout_digest: str) -> RecurrenceProcessExecutorPack:
    return RecurrenceProcessExecutorPack(
        compiled_model_digest=_digest("2"),
        recurrence_template_catalog_digest=_digest("3"),
        prepared_kernel_pack_digest=_digest("d"),
        direct_template_catalog_digest=_digest("e"),
        runtime_layout_digest=runtime_layout_digest,
        backend="jit",
        target_triple="symjit-storage-v3-portable",
        portable=True,
        cpu_features=(),
        catalog_executor_count=1,
        executor_ids=(0,),
        descriptor_payloads=(bytes.fromhex("10000000000000000000000001000000"),),
    )


def _execution_manifest(process: SimpleNamespace | None = None) -> dict[str, object]:
    process = process or _recurrence_process()
    return artifact_writer._recurrence_execution_manifest(
        process,
        schedule_path=f"recurrence/schedules/{_digest('f')}/recurrence-runtime.pacbin",
        binding={
            "abi": "pyamplicol-recurrence-process-binding-v4",
            "process_id": process.process_id,
            "schedule_digest": _digest("f"),
            "process_digest": process.process_digest,
            "process_semantic_digest": _digest("c"),
            "process_support_words": [1],
            "path": "recurrence-binding.bin",
            "size_bytes": 128,
            "sha256": _digest("9"),
        },
        color_contraction_record=None,
    )


def _recurrence_execution_summary_with_size(
    monkeypatch: pytest.MonkeyPatch,
    target_size: int,
) -> bytes:
    empty_summary_size = len(b'{"padding":""}\n')
    assert target_size >= empty_summary_size

    def padded_manifest(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        return {"padding": "x" * (target_size - empty_summary_size)}

    monkeypatch.setattr(
        artifact_writer,
        "_recurrence_execution_manifest",
        padded_manifest,
    )
    return artifact_writer._bounded_recurrence_execution_summary(
        _recurrence_process(),
        schedule_path="recurrence-runtime.pacbin",
        binding={},
        color_contraction_record=None,
    )


def _write_prepared_pack(root: Path) -> None:
    pack_path = root / "model" / "eager-kernel-pack.json"
    pack_path.parent.mkdir(parents=True)
    pack_path.write_text(
        json.dumps(
            {
                "backend": "jit",
                "kernels": [{"kernel_id": 0}, {"kernel_id": 1}],
                "recurrence_direct_template": {
                    "templates": [
                        {"payload_binding": {"prepared_kernel_id": 0}},
                        {"payload_binding": {"prepared_kernel_id": 1}},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_direct_recurrence_versions_replace_packet_abi() -> None:
    assert (
        versions.RECURRENCE_BUILDER_INPUT_ABI
        == "pyamplicol-recurrence-builder-input-v2"
    )
    assert versions.RECURRENCE_PLAN_ABI == "pyamplicol-recurrence-plan-v2"
    assert (
        versions.RECURRENCE_RUNTIME_LAYOUT_ABI
        == "pyamplicol-recurrence-runtime-layout-v2"
    )
    assert (
        versions.RECURRENCE_DIRECT_TEMPLATE_ABI
        == "pyamplicol-recurrence-direct-template-v1"
    )
    assert (
        versions.RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY
        == "rusticol.recurrence-direct-arena.complex-f64.v1"
    )
    assert not hasattr(versions, "RECURRENCE_RUNTIME_CAPABILITY")


def test_recurrence_execution_manifest_publishes_only_direct_arena_contract() -> None:
    manifest = _execution_manifest()

    assert manifest["builder_input_abi"] == versions.RECURRENCE_BUILDER_INPUT_ABI
    assert manifest["recurrence_plan_abi"] == versions.RECURRENCE_PLAN_ABI
    assert manifest["runtime_layout_abi"] == versions.RECURRENCE_RUNTIME_LAYOUT_ABI
    assert manifest["direct_template_abi"] == versions.RECURRENCE_DIRECT_TEMPLATE_ABI
    assert manifest["direct_backend_abi"] == versions.RECURRENCE_DIRECT_BACKEND_ABI
    assert manifest["prepared_kernel_pack_digest"] == _digest("d")
    assert manifest["direct_template_catalog_digest"] == _digest("e")
    assert manifest["compiled_model_digest"] == _digest("2")
    assert manifest["recurrence_template_catalog_digest"] == _digest("3")
    assert manifest["process_digest"] is None
    assert manifest["helicity_selector_companion"] is None
    assert manifest["required_runtime_capabilities"] == [
        versions.RECURRENCE_COLOR_RUNTIME_CAPABILITY,
        versions.RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
    ]
    plan = manifest["plan"]
    assert isinstance(plan, dict)
    assert plan["builder_input_abi"] == versions.RECURRENCE_BUILDER_INPUT_ABI
    assert plan["runtime_layout_abi"] == versions.RECURRENCE_RUNTIME_LAYOUT_ABI
    assert plan["prepared_kernel_pack_digest"] == _digest("d")
    assert plan["direct_template_catalog_digest"] == _digest("e")
    container = plan["runtime_schedule"]
    assert isinstance(container, dict)
    assert container["plan_member_path"] == (
        "schedule/recurrence-direct-schedule-v2.bin"
    )
    encoded = json.dumps(manifest)
    assert "pyamplicol-recurrence-plan-v1" not in encoded
    assert "pyamplicol-recurrence-runtime-layout-v1" not in encoded
    assert "rusticol.recurrence-runtime.complex-f64.v1" not in encoded


def test_recurrence_selector_v2_manifest_authenticates_persisted_direct_plan() -> None:
    process = _recurrence_process()
    process.color_accuracy = "full"
    process.color_contraction_summary = {
        "abi": "pyamplicol-recurrence-color-contraction-v3",
        "color_accuracy": "full",
        "storage": "repeated",
        "group_count": 24,
        "component_count": 8,
        "factorization": None,
    }
    companion = _selector_companion()
    process.helicity_selector_companion = companion
    binding = {
        "abi": "pyamplicol-recurrence-process-binding-v4",
        "process_id": process.process_id,
        "schedule_digest": companion.recurrence_schedule_digest,
        "native_schedule_semantic_digest": (
            companion.recurrence_native_schedule_semantic_digest
        ),
        "process_digest": process.process_digest,
        "process_semantic_digest": companion.builder_input_sha256,
        "process_support_words": [1],
        "path": "helicity-selector-binding.bin",
        "size_bytes": 256,
        "sha256": _digest("9"),
    }

    manifest = artifact_writer._recurrence_execution_manifest(
        process,
        schedule_path=f"recurrence/schedules/{_digest('f')}/recurrence-runtime.pacbin",
        binding={
            **binding,
            "schedule_digest": _digest("f"),
            "process_semantic_digest": _digest("c"),
            "path": "recurrence-binding.bin",
        },
        color_contraction_record=SimpleNamespace(
            size_bytes=5,
            sha256=_digest("0"),
        ),
        companion_binding=binding,
    )

    selector = manifest["helicity_selector_companion"]
    assert isinstance(selector, dict)
    assert selector == {
        "schema_version": 2,
        "kind": "pyamplicol-recurrence-helicity-selector-companion-v2",
        "process_digest": process.process_digest,
        "plan": selector["plan"],
        "color_contraction": selector["color_contraction"],
    }
    plan = selector["plan"]
    assert isinstance(plan, dict)
    assert plan["builder_input_sha256"] == companion.builder_input_sha256
    assert plan["process_binding"] == binding
    schedule = plan["runtime_schedule"]
    assert isinstance(schedule, dict)
    assert schedule["path"] == (
        f"recurrence/schedules/{companion.recurrence_schedule_digest}/"
        "recurrence-runtime.pacbin"
    )
    dispatch = plan["helicity_dispatch"]
    assert isinstance(dispatch, dict)
    assert dispatch == {
        "abi": "pyamplicol-recurrence-helicity-dispatch-v1",
        "path": (
            f"recurrence/schedules/{companion.recurrence_schedule_digest}/"
            "recurrence-helicity-dispatch-v1.bin"
        ),
        "size_bytes": companion.helicity_dispatch_size_bytes,
        "sha256": companion.helicity_dispatch_sha256,
        "base_runtime_layout_digest": (
            companion.helicity_dispatch_base_runtime_layout_digest
        ),
        "resolved_helicity_count": 8,
    }
    color = selector["color_contraction"]
    assert isinstance(color, dict)
    assert color == {
        "source": "primary",
        "view": "primary-local-color-view-v1",
    }
    assert "runtime_container" not in selector
    assert "query_construction_threads" not in selector
    assert "selector_policy" not in selector
    assert "on-the-fly" not in json.dumps(manifest)

    process.helicity_selector_companion = replace(
        companion,
        inspection_summary={
            **companion.inspection_summary,
            "schedule_digest": companion.recurrence_schedule_digest,
        },
    )
    with pytest.raises(ValueError, match="native Direct-plan identity"):
        artifact_writer._recurrence_execution_manifest(
            process,
            schedule_path=(
                f"recurrence/schedules/{_digest('f')}/recurrence-runtime.pacbin"
            ),
            binding={
                **binding,
                "schedule_digest": _digest("f"),
                "process_semantic_digest": _digest("c"),
                "path": "recurrence-binding.bin",
            },
            color_contraction_record=SimpleNamespace(
                size_bytes=5,
                sha256=_digest("0"),
            ),
            companion_binding=binding,
        )

    process.helicity_selector_companion = companion
    process.color_contraction_summary = {
        **process.color_contraction_summary,
        "storage": "expanded",
    }
    with pytest.raises(ValueError, match="direct, Walsh, or symmetric-group"):
        artifact_writer._recurrence_execution_manifest(
            process,
            schedule_path=(
                f"recurrence/schedules/{_digest('f')}/recurrence-runtime.pacbin"
            ),
            binding={
                **binding,
                "schedule_digest": _digest("f"),
                "process_semantic_digest": _digest("c"),
                "path": "recurrence-binding.bin",
            },
            color_contraction_record=SimpleNamespace(
                size_bytes=5,
                sha256=_digest("0"),
            ),
            companion_binding=binding,
        )


def test_recurrence_rejects_retired_on_the_fly_selector_companion() -> None:
    process = _recurrence_process()
    process.helicity_selector_companion = SimpleNamespace(kind="retired-otf-seed")

    with pytest.raises(TypeError, match="persisted Direct plan"):
        _execution_manifest(process)


def test_recurrence_selector_publication_emits_only_v2_plan_sidecars(
    tmp_path: Path,
) -> None:
    process_id = "d_dbar_to_z_g"
    base_digest = _digest("6")
    schedule_payload = b"selector direct schedule"
    dispatch_payload = b"exact helicity dispatch"
    schedule_path = tmp_path / "selector-runtime.pacbin"
    dispatch_path = tmp_path / "selector-dispatch.bin"
    schedule_path.write_bytes(schedule_payload)
    dispatch_path.write_bytes(dispatch_payload)
    dispatch_sha256 = hashlib.sha256(dispatch_payload).hexdigest()
    selector_digest = recurrence_helicity_selector_schedule_digest(
        base_digest,
        dispatch_sha256,
    )
    primary_pack = _executor_pack(_digest("4"))
    companion = replace(
        _selector_companion(),
        recurrence_schedule_path=schedule_path,
        recurrence_base_schedule_digest=base_digest,
        recurrence_schedule_digest=selector_digest,
        recurrence_native_schedule_semantic_digest=base_digest,
        recurrence_schedule_size_bytes=len(schedule_payload),
        recurrence_schedule_sha256=hashlib.sha256(schedule_payload).hexdigest(),
        helicity_dispatch_path=dispatch_path,
        helicity_dispatch_size_bytes=len(dispatch_payload),
        helicity_dispatch_sha256=dispatch_sha256,
        recurrence_process_remap=_identity_remap(physical_sector_count=3),
        recurrence_process_executor_pack=_executor_pack(_digest("5")),
    )
    process = artifact_writer.RecurrenceProcessArtifact(
        process_id=process_id,
        expression="d d~ > z g",
        color_accuracy="full",
        external_pdgs=(1, -1, 23, 21),
        aliases=(),
        physics={
            "schema_version": 1,
            "kind": "pyamplicol-resolved-physics",
            "process_id": process_id,
        },
        recurrence_schedule_path=tmp_path / "primary-runtime.pacbin",
        recurrence_schedule_digest=_digest("f"),
        recurrence_native_schedule_semantic_digest=_digest("f"),
        recurrence_schedule_size_bytes=1,
        recurrence_schedule_sha256=_digest("a"),
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=1,
        recurrence_schedule_index_sha256=_digest("b"),
        builder_input_sha256=_digest("c"),
        prepared_kernel_pack_digest=_digest("d"),
        direct_template_catalog_digest=_digest("e"),
        referenced_kernel_ids=frozenset({0}),
        inspection_summary={
            "runtime_layout_digest": _digest("4"),
            "schedule_digest": _digest("f"),
        },
        runtime_metadata={},
        color_contraction_payload=b"primary color reducer",
        color_contraction_summary={
            "abi": "pyamplicol-recurrence-color-contraction-v3",
            "color_accuracy": "full",
            "storage": "repeated",
            "group_count": 24,
            "component_count": 8,
            "factorization": None,
        },
        point_tile_size=1,
        workspace_mib=1,
        recurrence_summary={"lc_flow_layout": "contracted-color-union"},
        validation_point=ValidationPointRecord(
            process_id=process_id,
            process="d d~ > z g",
            seed=1,
            error="not needed",
        ),
        generation_filters={},
        generation_profile={},
        recurrence_process_remap=_identity_remap(physical_sector_count=3),
        recurrence_process_executor_pack=primary_pack,
        process_digest=_digest("1"),
        helicity_selector_companion=companion,
    )

    class PayloadSink:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def add_file(
            self,
            relative: str,
            source: Path,
            *,
            process_id: str | None,
        ) -> SimpleNamespace:
            del process_id
            payload = source.read_bytes()
            self.paths.append(relative)
            return SimpleNamespace(
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )

        def add_bytes(
            self,
            relative: str,
            content: bytes,
            *,
            process_id: str | None,
            media_type: str | None = None,
        ) -> SimpleNamespace:
            del process_id, media_type
            self.paths.append(relative)
            return SimpleNamespace(
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )

    class ManifestSink:
        def __init__(self) -> None:
            self.execution: dict[str, object] | None = None

        def add_json(self, relative: str, value: object, **_kwargs: object) -> None:
            del relative, value

        def add_bytes(
            self,
            relative: str,
            content: bytes,
            **_kwargs: object,
        ) -> SimpleNamespace:
            if relative.endswith("/execution.json"):
                decoded = json.loads(content)
                assert isinstance(decoded, dict)
                self.execution = decoded
            return SimpleNamespace(sha256=hashlib.sha256(content).hexdigest())

    primary_binding_payload = b"primary binding"
    primary_binding_mapping = {
        "abi": "pyamplicol-recurrence-process-binding-v4",
        "process_id": process_id,
        "schedule_digest": process.recurrence_schedule_digest,
        "native_schedule_semantic_digest": (
            process.recurrence_native_schedule_semantic_digest
        ),
        "process_digest": process.process_digest,
        "process_semantic_digest": process.builder_input_sha256,
        "process_support_words": [1],
        "path": "recurrence-binding.bin",
        "size_bytes": len(primary_binding_payload),
        "sha256": hashlib.sha256(primary_binding_payload).hexdigest(),
    }
    primary_binding = SimpleNamespace(
        artifact_path=f"processes/{process_id}/recurrence-binding.bin",
        schedule_digest=process.recurrence_schedule_digest,
        payload=primary_binding_payload,
        size_bytes=len(primary_binding_payload),
        sha256=hashlib.sha256(primary_binding_payload).hexdigest(),
        to_mapping=lambda: primary_binding_mapping,
    )
    sharing = SimpleNamespace(
        binding=lambda requested: primary_binding,
        schedule=lambda _digest_value: SimpleNamespace(
            artifact_path=(
                f"recurrence/schedules/{process.recurrence_schedule_digest}/"
                "recurrence-runtime.pacbin"
            )
        ),
    )
    evaluator_payloads = PayloadSink()
    artifact_writer._write_recurrence_helicity_selector_schedule_roots(
        evaluator_payloads,  # type: ignore[arg-type]
        (process,),
        recurrence_sharing=None,
    )
    manifest_sink = ManifestSink()
    artifact_writer._write_process_payloads(
        manifest_sink,  # type: ignore[arg-type]
        process,
        evaluator_payloads=evaluator_payloads,  # type: ignore[arg-type]
        recurrence_sharing=sharing,  # type: ignore[arg-type]
    )

    expected_root = f"recurrence/schedules/{selector_digest}"
    assert sorted(evaluator_payloads.paths) == sorted(
        (
            f"{expected_root}/recurrence-runtime.pacbin",
            f"{expected_root}/recurrence-helicity-dispatch-v1.bin",
            f"processes/{process_id}/recurrence-binding.bin",
            f"processes/{process_id}/recurrence-color.bin",
            f"processes/{process_id}/helicity-selector-binding.bin",
        )
    )
    assert not any("on-the-fly" in path for path in evaluator_payloads.paths)
    assert manifest_sink.execution is not None
    encoded_execution = json.dumps(manifest_sink.execution)
    assert "pyamplicol-recurrence-helicity-selector-companion-v2" in encoded_execution
    assert "primary-local-color-view-v1" in encoded_execution
    assert "on-the-fly" not in encoded_execution


def test_recurrence_execution_summary_accepts_observed_large_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_cluster_size = 40_826_959

    content = _recurrence_execution_summary_with_size(
        monkeypatch,
        observed_cluster_size,
    )

    assert len(content) == observed_cluster_size


def test_recurrence_physics_identifies_direct_plan_and_runtime_layout() -> None:
    exact_one = SimpleNamespace(
        real_numerator=1,
        real_denominator=1,
        imag_numerator=0,
        imag_denominator=1,
    )
    source_state = SimpleNamespace(state_index=0, public_helicity=-1)
    external_leg = SimpleNamespace(
        source_slot=0,
        public_label=1,
        source_states=(source_state,),
    )
    logical = SimpleNamespace(
        process_id="d_dbar_to_z_g",
        layout="topology-replay",
        selected_source_coverage=None,
        selected_public_flow_ids=None,
        external_legs=(external_leg,),
        public_flows=(
            SimpleNamespace(
                flow_id=0,
                public_id="flow:1",
                word_source_slots=(0,),
                reduction_weight=exact_one,
            ),
        ),
    )
    process = SimpleNamespace(
        key="d_dbar_to_z_g",
        process="d d~ > z g",
        color_accuracy="lc",
        legs=(
            SimpleNamespace(
                label=1,
                particle="d",
                pdg=1,
                is_initial=True,
            ),
        ),
    )
    physics = recurrence_physics.build_recurrence_physics(
        process,
        logical,
        SimpleNamespace(parameters=()),
        process_id="d_dbar_to_z_g",
        resolved_helicities=((-1,),),
        normalization={},
        selected_color_sector_ids=None,
    )

    extensions = physics["extensions"]
    assert isinstance(extensions, dict)
    selectors = extensions["runtime_selectors"]
    assert isinstance(selectors, dict)
    assert selectors["provenance"] == versions.RECURRENCE_PLAN_ABI
    recurrence = extensions["recurrence"]
    assert isinstance(recurrence, dict)
    assert recurrence == {
        "builder_input_abi": versions.RECURRENCE_BUILDER_INPUT_ABI,
        "plan_abi": versions.RECURRENCE_PLAN_ABI,
        "runtime_layout_abi": versions.RECURRENCE_RUNTIME_LAYOUT_ABI,
        "direct_template_abi": versions.RECURRENCE_DIRECT_TEMPLATE_ABI,
        "lc_flow_layout": "topology-replay",
    }
    reduction = extensions["recurrence_runtime_reduction"]
    assert isinstance(reduction, dict)
    assert reduction["kind"] == "pyamplicol-recurrence-native-reduction-v2"
    assert reduction["plan_member_path"] == (
        "schedule/recurrence-direct-schedule-v2.bin"
    )


def test_recurrence_physics_keeps_sector_ids_distinct_from_flow_ids() -> None:
    exact_one = SimpleNamespace(
        real_numerator=1,
        real_denominator=1,
        imag_numerator=0,
        imag_denominator=1,
    )
    external_leg = SimpleNamespace(
        source_slot=0,
        public_label=1,
        source_states=(SimpleNamespace(state_index=0, public_helicity=-1),),
    )
    logical = SimpleNamespace(
        process_id="g_g_to_g_g",
        layout="topology-replay",
        selected_source_coverage=None,
        selected_public_flow_ids=(0, 1),
        external_legs=(external_leg,),
        public_flows=tuple(
            SimpleNamespace(
                flow_id=index,
                public_id=f"flow:{index}",
                word_source_slots=(0,),
                reduction_weight=exact_one,
            )
            for index in range(2)
        ),
    )
    process = SimpleNamespace(
        key="g_g_to_g_g",
        process="g g > g g",
        color_accuracy="lc",
        legs=(
            SimpleNamespace(
                label=1,
                particle="g",
                pdg=21,
                is_initial=True,
            ),
        ),
    )

    physics = recurrence_physics.build_recurrence_physics(
        process,
        logical,
        SimpleNamespace(parameters=()),
        process_id="g_g_to_g_g",
        resolved_helicities=((-1,),),
        normalization={},
        selected_color_sector_ids=(7,),
    )

    color_axis = physics["extensions"]["runtime_selectors"]["axes"]["color_flow"]
    assert color_axis["generation_coverage"] == "selected"
    assert color_axis["generation_selection"] == [7]
    assert tuple(component["id"] for component in physics["color_components"]) == (
        "flow:0",
        "flow:1",
    )


def test_recurrence_physics_expands_global_helicity_flip_aliases() -> None:
    exact_one = SimpleNamespace(
        real_numerator=1,
        real_denominator=1,
        imag_numerator=0,
        imag_denominator=1,
    )
    external_leg = SimpleNamespace(
        source_slot=0,
        public_label=1,
        source_states=(
            SimpleNamespace(state_index=0, public_helicity=-1),
            SimpleNamespace(state_index=1, public_helicity=1),
        ),
    )
    proof_digest = _digest("7")
    logical = SimpleNamespace(
        process_id="d_dbar_to_z_g",
        layout="topology-replay",
        semantic_digests=(
            RecurrenceSemanticDigestV1(
                "helicity-equivalence:global-flip-v1",
                proof_digest,
            ),
        ),
        selected_source_coverage=None,
        selected_public_flow_ids=None,
        external_legs=(external_leg,),
        public_flows=(
            SimpleNamespace(
                flow_id=0,
                public_id="flow:1",
                word_source_slots=(0,),
                reduction_weight=exact_one,
            ),
        ),
    )
    process = SimpleNamespace(
        key="d_dbar_to_z_g",
        process="d d~ > z g",
        color_accuracy="lc",
        legs=(
            SimpleNamespace(
                label=1,
                particle="d",
                pdg=1,
                is_initial=True,
            ),
        ),
    )

    physics = recurrence_physics.build_recurrence_physics(
        process,
        logical,
        SimpleNamespace(parameters=()),
        process_id="d_dbar_to_z_g",
        resolved_helicities=((-1,),),
        normalization={},
        selected_color_sector_ids=None,
    )

    assert physics["coverage"]["structural_zero_helicity_count"] == 0
    negative, positive = physics["helicities"]
    assert negative["computed"] is True
    assert negative["representative_id"] == "h:-1"
    assert positive["computed"] is False
    assert positive["structural_zero"] is False
    assert positive["representative_id"] == "h:-1"
    reduction = physics["extensions"]["global_helicity_flip_reduction"]
    assert reduction == {
        "kind": "pyamplicol-global-helicity-flip-reduction-v1",
        "proof_role": "helicity-equivalence:global-flip-v1",
        "proof_digest": proof_digest,
        "physical_nonzero_helicity_count": 2,
        "representative_helicity_count": 1,
        "aliases": [
            {
                "physical_id": "h:+1",
                "representative_id": "h:-1",
            }
        ],
    }


def test_inspection_rejects_retired_recurrence_before_strict_manifest_load(
    tmp_path: Path,
) -> None:
    (tmp_path / "artifact.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "required_runtime_capabilities": [
                        "rusticol.recurrence-runtime.complex-f64.v1"
                    ]
                },
                "processes": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=r"retired packet ABI.*regenerate"):
        inspection.inspect_artifact(tmp_path)


def test_direct_recurrence_inspection_exposes_arena_counters(tmp_path: Path) -> None:
    _write_prepared_pack(tmp_path)
    execution = _execution_manifest()
    result = inspection._recurrence_execution_inspection(
        SimpleNamespace(root=tmp_path),
        execution,
    )

    assert result.invocation_count == 34
    assert result.attachment_count == 0
    assert result.arena_semantic_component_count == 48
    assert result.arena_component_count == 32
    assert result.arena_component_reuse_count == 16
    assert result.momentum_form_count == 7
    assert result.direct_source_row_count == 4
    assert result.direct_contribution_row_count == 34
    assert result.direct_finalization_row_count == 22
    assert result.direct_closure_row_count == 12
    assert result.direct_row_group_count == 11
    assert result.packed_input_bytes == 0
    assert result.packed_output_bytes == 0
    assert result.scatter_bytes == 0


def test_direct_recurrence_inspection_exposes_contracted_color_summary(
    tmp_path: Path,
) -> None:
    _write_prepared_pack(tmp_path)
    execution = _execution_manifest()
    execution["runtime_metadata"]["color_contraction"] = {
        "abi": "pyamplicol-recurrence-color-contraction-v3",
        "color_accuracy": "full",
        "storage": "repeated",
        "includes_color_factor": True,
        "group_count": 384,
        "sector_count": 36,
        "active_sector_count": 6,
        "component_count": 64,
        "destination_count": 384,
        "entry_count": 21,
        "logical_entry_count": 1344,
        "semantic_digest": _digest("7"),
        "factorization": {
            "kind": "klein-four-walsh",
            "rank": 2,
            "coset_count": 1,
        },
        "path": "recurrence-color.bin",
        "size_bytes": 4096,
        "sha256": _digest("8"),
    }

    result = inspection._recurrence_execution_inspection(
        SimpleNamespace(root=tmp_path),
        execution,
    )

    assert result.recurrence_color_accuracy == "full"
    assert result.recurrence_color_storage == "repeated"
    assert result.recurrence_color_sector_count == 36
    assert result.recurrence_color_active_sector_count == 6
    assert result.recurrence_color_component_count == 64
    assert result.recurrence_color_group_count == 384
    assert result.recurrence_color_destination_count == 384
    assert result.recurrence_color_entry_count == 21
    assert result.recurrence_color_logical_entry_count == 1344
    assert result.recurrence_color_factorization_kind == "klein-four-walsh"
    assert result.recurrence_color_factorization_rank == 2
    assert result.recurrence_color_factorization_coset_count == 1


def test_inspection_requires_v2_metadata_without_packet_fallback(
    tmp_path: Path,
) -> None:
    _write_prepared_pack(tmp_path)
    execution = _execution_manifest()
    execution["recurrence_plan_abi"] = "pyamplicol-recurrence-plan-v1"

    with pytest.raises(ArtifactError, match="regenerate the recurrence artifact"):
        inspection._recurrence_execution_inspection(
            SimpleNamespace(root=tmp_path),
            execution,
        )


def test_v2_inspection_tolerates_pending_direct_counter_wiring(
    tmp_path: Path,
) -> None:
    _write_prepared_pack(tmp_path)
    process = _recurrence_process()
    process.inspection_summary.pop("direct_arena")
    execution = _execution_manifest(process)

    result = inspection._recurrence_execution_inspection(
        SimpleNamespace(root=tmp_path),
        execution,
    )

    assert result.arena_component_count is None
    assert result.momentum_form_count is None
    assert result.direct_row_group_count is None
    assert result.packed_input_bytes == 0
    assert result.packed_output_bytes == 0
    assert result.scatter_bytes == 0
