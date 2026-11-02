# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import pytest

import pyamplicol.artifacts.manifest as manifest_module
import pyamplicol.generation.artifact_writer as generation_artifact_writer
from pyamplicol import ArtifactError, CompatibilityError
from pyamplicol._internal.versions import (
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
)
from pyamplicol.artifacts import (
    ArtifactBuilder,
    ArtifactTransaction,
    load_manifest,
    normalize_relative_path,
)
from pyamplicol.artifacts.manifest import PORTABLE_64LE_TARGET
from pyamplicol.artifacts.process_groups import (
    initial_process_groups,
    read_process_groups,
    write_process_groups,
)
from pyamplicol.generation.evaluator_container import PacbinReader
from pyamplicol.generation.process_grouping import (
    GroupingBasis,
    GroupMember,
    ProcessGroupMap,
)


def _producer(
    *,
    triple: str = "test-target",
    cpu_features: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
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
        "target": {"triple": triple, "cpu_features": list(cpu_features)},
    }


def _model() -> dict[str, object]:
    return {
        "name": "built-in-sm",
        "source_kind": "built-in-sm",
        "content_sha256": "1" * 64,
        "compiled_schema_version": 1,
    }


def _configuration() -> dict[str, object]:
    return {
        "toml_schema_version": 1,
        "requested_path": "config/requested.toml",
        "effective_path": "config/effective.toml",
        "adjustments": [],
    }


def _process(
    identifier: str = "dd_to_z",
    *,
    runtime_capabilities: tuple[str, ...] = (
        "symjit.application.complex-f64.v1",
    ),
) -> dict[str, object]:
    return {
        "id": identifier,
        "expression": "d d~ > z",
        "color_accuracy": "lc",
        "external_pdgs": [1, -1, 23],
        "physics_path": "physics/process.json",
        "required_runtime_capabilities": list(runtime_capabilities),
        "aliases": [],
    }


def _runtime(
    *,
    runtime_capabilities: tuple[str, ...] = (
        "symjit.application.complex-f64.v1",
    ),
) -> dict[str, object]:
    return {
        "engine": "rusticol",
        "engine_version": "0.1.0",
        "evaluator_manifest_path": "physics/process.json",
        "api_bundle_path": None,
        "required_runtime_capabilities": list(runtime_capabilities),
    }


def _build(
    path: Path,
    *,
    mode: str = "error",
    triple: str = "test-target",
    cpu_features: tuple[str, ...] = (),
    runtime_capabilities: tuple[str, ...] = (
        "symjit.application.complex-f64.v1",
    ),
    evaluator_state_target: dict[str, object] | None = None,
) -> None:
    with ArtifactBuilder(path, mode=mode) as builder:  # type: ignore[arg-type]
        builder.add_json(
            "physics/process.json",
            {"schema_version": 1},
            role="runtime-physics",
            process_id="dd_to_z",
        )
        if evaluator_state_target is not None:
            builder.add_bytes(
                "evaluators/state.symjit",
                b"portable-state",
                role="evaluator-state",
                media_type="application/vnd.symjit.application",
                target=evaluator_state_target,
                process_id="dd_to_z",
            )
        builder.finalize(
            kind="pyamplicol-process",
            producer=_producer(triple=triple, cpu_features=cpu_features),
            model=_model(),
            configuration=_configuration(),
            processes=[
                _process(runtime_capabilities=runtime_capabilities)
            ],
            default_process_id="dd_to_z",
            runtime=_runtime(runtime_capabilities=runtime_capabilities),
        )


def test_builder_writes_compact_machine_json(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    with ArtifactBuilder(root) as builder:
        builder.add_json(
            "physics/process.json",
            {"beta": [1, 2], "alpha": True},
            role="runtime-physics",
            process_id="dd_to_z",
            compact=True,
        )
        builder.finalize(
            kind="pyamplicol-process",
            producer=_producer(),
            model=_model(),
            configuration=_configuration(),
            processes=[_process()],
            default_process_id="dd_to_z",
            runtime=_runtime(),
        )

    assert (root / "physics/process.json").read_text(encoding="utf-8") == (
        '{"alpha":true,"beta":[1,2]}\n'
    )


def test_builder_streams_files_and_registers_staged_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    source = tmp_path / "large.bin"
    source.write_bytes(b"streamed" * 4096)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == source:
            raise AssertionError("ArtifactBuilder.add_file must not call read_bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with ArtifactBuilder(root) as builder:
        builder.add_file(
            "payloads/streamed.bin",
            source,
            role="runtime-physics",
            media_type="application/octet-stream",
        )
        staged = builder.staged_path("payloads/registered.bin", create_parent=True)
        staged.write_bytes(b"registered")
        builder.register_staged_file(
            "payloads/registered.bin",
            role="runtime-physics",
            media_type="application/octet-stream",
        )
        builder.add_stream(
            "physics/process.json",
            io.BytesIO(b'{}\n'),
            role="runtime-physics",
            media_type="application/json",
            process_id="dd_to_z",
            chunk_size=2,
        )
        builder.finalize(
            kind="pyamplicol-process",
            producer=_producer(),
            model=_model(),
            configuration=_configuration(),
            processes=[_process()],
            default_process_id="dd_to_z",
            runtime=_runtime(),
        )

    assert (root / "payloads/streamed.bin").stat().st_size == source.stat().st_size
    assert (root / "payloads/registered.bin").read_bytes() == b"registered"


def test_builder_round_trip_and_explicit_tamper_detection(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    manifest = load_manifest(
        root,
        expected_target="test-target",
        verify_payloads=True,
    )
    assert manifest.default_process_id == "dd_to_z"
    assert manifest.payloads[0].path == "physics/process.json"
    assert manifest.runtime["required_runtime_capabilities"] == (
        "symjit.application.complex-f64.v1",
    )

    (root / "physics/process.json").write_text("modified\n", encoding="utf-8")
    assert load_manifest(root).artifact_id == manifest.artifact_id
    with pytest.raises(ArtifactError, match=r"(size|digest) mismatch"):
        load_manifest(root, verify_payloads=True)


def test_artifact_id_tracks_only_runtime_bearing_payload_records() -> None:
    def record(path: str, role: str, digest: str) -> dict[str, object]:
        return {
            "path": path,
            "role": role,
            "media_type": "application/octet-stream",
            "size_bytes": 1,
            "sha256": digest * 64,
            "executable": False,
        }

    manifest = {
        "artifact_id": "0" * 64,
        "created_utc": "2026-07-31T00:00:00Z",
        "payloads": [
            record("config/requested.toml", "configuration-requested", "1"),
            record("config/effective.toml", "configuration-effective", "2"),
            record(
                "processes/p0/validation-momenta.json",
                "validation-momenta",
                "3",
            ),
            record("processes/p0/physics.json", "runtime-physics", "4"),
        ],
    }
    identity = manifest_module.compute_artifact_id(manifest)

    provenance_changed = json.loads(json.dumps(manifest))
    for item in provenance_changed["payloads"][:3]:
        item["sha256"] = "f" * 64
        item["size_bytes"] = 999
    provenance_changed["created_utc"] = "2030-01-01T00:00:00Z"
    assert manifest_module.compute_artifact_id(provenance_changed) == identity

    runtime_changed = json.loads(json.dumps(manifest))
    runtime_changed["payloads"][3]["sha256"] = "f" * 64
    assert manifest_module.compute_artifact_id(runtime_changed) != identity


def test_artifact_without_current_identity_contract_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["extensions"]["artifact_identity"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CompatibilityError, match="regenerate"):
        load_manifest(root)


def test_python_loader_checks_cpu_features_before_payload_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    _build(root, cpu_features=("avx2",))
    (root / "physics/process.json").write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(
        manifest_module,
        "_runtime_target_metadata",
        lambda: ("test-target", ("sse2",)),
    )
    with pytest.raises(CompatibilityError, match="unavailable CPU features: avx2"):
        load_manifest(root, verify_payloads=True)


def test_python_loader_accepts_available_canonical_cpu_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    _build(root, cpu_features=("avx2", "fma"))
    monkeypatch.setattr(
        manifest_module,
        "_runtime_target_metadata",
        lambda: ("test-target", ("avx", "avx2", "fma", "sse2")),
    )
    assert load_manifest(root, verify_payloads=True).producer["target"] == {
        "triple": "test-target",
        "cpu_features": ("avx2", "fma"),
    }


@pytest.mark.parametrize(
    "runtime_target",
    (
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
    ),
)
def test_python_loader_accepts_portable_o2_artifact_for_every_supported_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_target: str,
) -> None:
    root = tmp_path / "artifact"
    target = {"triple": PORTABLE_64LE_TARGET, "cpu_features": []}
    _build(
        root,
        triple=PORTABLE_64LE_TARGET,
        evaluator_state_target=target,
    )
    monkeypatch.setattr(
        manifest_module,
        "_runtime_target_metadata",
        lambda: (_ for _ in ()).throw(
            AssertionError("portable artifact inspection must not import Rusticol")
        ),
    )

    manifest = load_manifest(
        root,
        expected_target=runtime_target,
        verify_payloads=True,
    )

    assert manifest.producer["target"] == {
        "triple": PORTABLE_64LE_TARGET,
        "cpu_features": (),
    }


def test_python_loader_inspects_portable_artifact_without_native_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _build(root, triple=PORTABLE_64LE_TARGET)
    monkeypatch.setattr(
        manifest_module,
        "_runtime_target_metadata",
        lambda: (_ for _ in ()).throw(ImportError("native runtime unavailable")),
    )

    assert load_manifest(root).producer["target"]["triple"] == PORTABLE_64LE_TARGET


def test_python_loader_rejects_portable_artifact_for_unsupported_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(root, triple=PORTABLE_64LE_TARGET)

    with pytest.raises(CompatibilityError, match="aarch64-unknown-linux-gnu"):
        load_manifest(root, expected_target="aarch64-unknown-linux-gnu")


def test_python_loader_rejects_portable_artifact_with_cpu_features(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(
        root,
        triple=PORTABLE_64LE_TARGET,
        cpu_features=("avx2",),
    )

    with pytest.raises(CompatibilityError, match="may not require CPU features"):
        load_manifest(root)


@pytest.mark.parametrize(
    "runtime_capability",
    (SYMBOLICA_CPP_RUNTIME_CAPABILITY, SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY),
)
def test_python_loader_rejects_target_specific_capability_under_portable_target(
    tmp_path: Path,
    runtime_capability: str,
) -> None:
    root = tmp_path / "artifact"
    _build(
        root,
        triple=PORTABLE_64LE_TARGET,
        runtime_capabilities=(runtime_capability,),
    )

    with pytest.raises(CompatibilityError, match="target-specific evaluator"):
        load_manifest(root)


def test_python_loader_rejects_mismatched_portable_payload_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(
        root,
        triple=PORTABLE_64LE_TARGET,
        evaluator_state_target={
            "triple": "aarch64-apple-darwin",
            "cpu_features": [],
        },
    )

    with pytest.raises(CompatibilityError, match="differs from producer target"):
        load_manifest(root)


def test_python_loader_rejects_portable_artifact_on_non_64_bit_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _build(root, triple=PORTABLE_64LE_TARGET)
    monkeypatch.setattr(manifest_module.struct, "calcsize", lambda _format: 4)

    with pytest.raises(CompatibilityError, match="64-bit little-endian"):
        load_manifest(root)


@pytest.mark.parametrize(
    "features",
    (("fma", "avx2"), ("+avx2",), ("AVX2",), ("avx_2",)),
)
def test_manifest_rejects_noncanonical_cpu_feature_ids(
    tmp_path: Path, features: tuple[str, ...]
) -> None:
    root = tmp_path / "artifact"
    _build(root, cpu_features=features)
    with pytest.raises(ArtifactError, match=r"(sorted|non-canonical)"):
        load_manifest(root, verify_payloads=False)


@pytest.mark.parametrize(
    "value",
    ("../escape", "/absolute", "a/../b", r"a\b", "./payload"),
)
def test_relative_paths_reject_escape_and_nonportable_forms(value: str) -> None:
    with pytest.raises(ArtifactError):
        normalize_relative_path(value)


def test_explicit_validation_rejects_payload_symlink(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    payload = root / "physics/process.json"
    target = tmp_path / "outside.json"
    target.write_text("{}\n", encoding="utf-8")
    payload.unlink()
    payload.symlink_to(target)
    with pytest.raises(ArtifactError, match="symlink"):
        load_manifest(root, verify_payloads=True)


def test_explicit_validation_rejects_undeclared_symlink(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    target = tmp_path / "outside.txt"
    target.write_text("outside\n", encoding="utf-8")
    (root / "extra-link").symlink_to(target)
    with pytest.raises(ArtifactError, match="undeclared symlink"):
        load_manifest(root, verify_payloads=True)


def test_explicit_validation_rejects_undeclared_executable(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    executable = root / "injected"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    with pytest.raises(ArtifactError, match="undeclared executable"):
        load_manifest(root, verify_payloads=True)


def test_evaluator_state_requires_target_metadata(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    with ArtifactBuilder(root) as builder:
        builder.add_bytes(
            "evaluators/state.bin",
            b"state",
            role="evaluator-state",
            media_type="application/octet-stream",
        )
        builder.finalize(
            kind="pyamplicol-process",
            producer=_producer(),
            model=_model(),
            configuration=_configuration(),
            processes=[_process()],
            runtime=_runtime(),
        )
    with pytest.raises(ArtifactError, match="no target metadata"):
        load_manifest(root, verify_payloads=True)


def test_artifact_root_must_be_a_directory(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="not a directory"):
        load_manifest(artifact)


def test_old_schema_has_actionable_regeneration_error(tmp_path: Path) -> None:
    root = tmp_path / "old"
    root.mkdir()
    (root / "artifact.json").write_text(
        json.dumps({"schema_version": 2}) + "\n", encoding="utf-8"
    )
    with pytest.raises(CompatibilityError, match="regenerate"):
        load_manifest(root, verify_payloads=False)


def test_transaction_rolls_back_failed_replace(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    (destination / "sentinel").write_text("old\n", encoding="utf-8")
    with (
        pytest.raises(RuntimeError, match="stop"),
        ArtifactTransaction(destination, mode="replace") as staging,
    ):
        (staging / "sentinel").write_text("new\n", encoding="utf-8")
        raise RuntimeError("stop")
    assert (destination / "sentinel").read_text(encoding="utf-8") == "old\n"


def test_transaction_atomically_replaces_existing_directory(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    (destination / "sentinel").write_text("old\n", encoding="utf-8")

    with ArtifactTransaction(destination, mode="replace") as staging:
        (staging / "sentinel").write_text("new\n", encoding="utf-8")

    assert (destination / "sentinel").read_text(encoding="utf-8") == "new\n"
    assert not tuple(tmp_path.glob(".artifact.staging-*"))


def test_append_starts_from_existing_snapshot(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    (destination / "old").write_text("preserved\n", encoding="utf-8")
    with ArtifactTransaction(destination, mode="append") as staging:
        (staging / "new").write_text("added\n", encoding="utf-8")
    assert (destination / "old").read_text(encoding="utf-8") == "preserved\n"
    assert (destination / "new").read_text(encoding="utf-8") == "added\n"


def test_builder_append_retains_declared_payloads_and_adds_new_ones(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"
    _build(destination)

    with ArtifactBuilder(destination, mode="append") as builder:
        builder.add_json(
            "physics/additional.json",
            {"schema_version": 1},
            role="runtime-physics",
            process_id="dd_to_zg",
        )
        builder.finalize(
            kind="pyamplicol-process-set",
            producer=_producer(),
            model=_model(),
            configuration=_configuration(),
            processes=[_process(), _process("dd_to_zg")],
            default_process_id="dd_to_z",
            runtime=_runtime(),
        )

    manifest = load_manifest(destination)
    assert [record.path for record in manifest.payloads] == [
        "physics/additional.json",
        "physics/process.json",
    ]
    assert (destination / "physics/process.json").is_file()


def test_evaluator_container_append_migrates_loose_and_reuses_old_members(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"
    target = _producer()["target"]
    with ArtifactBuilder(destination) as builder:
        builder.add_json(
            "physics/process.json",
            {"schema_version": 1},
            role="runtime-physics",
            process_id="dd_to_z",
        )
        for relative, content in (
            ("processes/old/application.symjit", b"old-application"),
            ("processes/old/state.evaluator.bin", b"old-state"),
            ("processes/old/library.dylib", b"native-library"),
        ):
            builder.add_bytes(
                relative,
                content,
                role="evaluator-state",
                media_type="application/octet-stream",
                target=target,
                process_id="dd_to_z",
            )
        builder.finalize(
            kind="pyamplicol-process",
            producer=_producer(),
            model=_model(),
            configuration=_configuration(),
            processes=[_process()],
            default_process_id="dd_to_z",
            runtime=_runtime(),
        )

    replacement = tmp_path / "replacement.symjit"
    replacement.write_bytes(b"replacement-application")
    added = tmp_path / "added.evaluator.bin"
    added.write_bytes(b"added-state")
    existing = load_manifest(destination)
    with ArtifactBuilder(
        destination,
        mode="append",
        expected_artifact_id=existing.artifact_id,
    ) as builder:
        collector = generation_artifact_writer._EvaluatorPayloadCollector(
            builder,
            existing=existing,
            target=target,
        )
        collector.add_file(
            "processes/old/application.symjit",
            replacement,
            process_id="dd_to_z",
        )
        collector.add_file(
            "processes/new/state.evaluator.bin",
            added,
            process_id="dd_to_z",
        )
        extension = collector.publish()
        assert extension is not None
        builder.finalize(
            kind="pyamplicol-process",
            producer=existing.producer,
            model=existing.model,
            configuration=existing.configuration,
            processes=existing.processes,
            default_process_id=existing.default_process_id,
            runtime=existing.runtime,
            dependencies=existing.dependencies,
            extensions={"evaluator_payload_container": extension},
        )

    migrated = load_manifest(destination)
    declared = {record.path for record in migrated.payloads}
    assert "evaluators.pacbin" in declared
    assert "processes/old/library.dylib" in declared
    assert "processes/old/application.symjit" not in declared
    assert "processes/old/state.evaluator.bin" not in declared
    with PacbinReader.open(destination / "evaluators.pacbin") as container:
        assert container.read_member(
            "processes/old/application.symjit",
            length=len(b"replacement-application"),
        ) == b"replacement-application"
        assert container.read_member(
            "processes/old/state.evaluator.bin",
            length=len(b"old-state"),
        ) == b"old-state"
        assert container.read_member(
            "processes/new/state.evaluator.bin",
            length=len(b"added-state"),
        ) == b"added-state"

    second = tmp_path / "second.symjit"
    second.write_bytes(b"second-application")
    existing = load_manifest(destination)
    with ArtifactBuilder(
        destination,
        mode="append",
        expected_artifact_id=existing.artifact_id,
    ) as builder:
        collector = generation_artifact_writer._EvaluatorPayloadCollector(
            builder,
            existing=existing,
            target=target,
        )
        collector.add_file(
            "processes/second/application.symjit",
            second,
            process_id="dd_to_z",
        )
        extension = collector.publish()
        assert extension is not None
        builder.finalize(
            kind="pyamplicol-process",
            producer=existing.producer,
            model=existing.model,
            configuration=existing.configuration,
            processes=existing.processes,
            default_process_id=existing.default_process_id,
            runtime=existing.runtime,
            dependencies=existing.dependencies,
            extensions={"evaluator_payload_container": extension},
        )
    with PacbinReader.open(destination / "evaluators.pacbin") as container:
        assert {member.logical_path for member in container.members} == {
            "processes/new/state.evaluator.bin",
            "processes/old/application.symjit",
            "processes/old/state.evaluator.bin",
            "processes/second/application.symjit",
        }


def test_evaluator_container_failed_repack_preserves_published_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact"
    _build(destination)
    before = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    source = tmp_path / "application.symjit"
    source.write_bytes(b"application")
    existing = load_manifest(destination)

    def fail_after_partial_write(path: Path, *_args: object, **_kwargs: object) -> None:
        path.write_bytes(b"malformed-partial-container")
        raise RuntimeError("injected container interruption")

    monkeypatch.setattr(
        generation_artifact_writer,
        "write_pacbin_atomic",
        fail_after_partial_write,
    )
    with (
        pytest.raises(RuntimeError, match="injected container interruption"),
        ArtifactBuilder(
            destination,
            mode="append",
            expected_artifact_id=existing.artifact_id,
        ) as builder,
    ):
        collector = generation_artifact_writer._EvaluatorPayloadCollector(
            builder,
            existing=existing,
            target=existing.producer["target"],
        )
        collector.add_file(
            "processes/new/application.symjit",
            source,
            process_id="dd_to_z",
        )
        collector.publish()

    after = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_evaluator_container_snapshots_registered_files(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    source = tmp_path / "reused.symjit"
    source.write_bytes(b"first-application")
    target = _producer()["target"]

    with ArtifactBuilder(destination) as builder:
        collector = generation_artifact_writer._EvaluatorPayloadCollector(
            builder,
            existing=None,
            target=target,
        )
        collector.add_file(
            "processes/first/application.symjit",
            source,
            process_id="dd_to_z",
        )
        source.write_bytes(b"second-application")
        collector.add_file(
            "processes/second/application.symjit",
            source,
            process_id="dd_to_z",
        )
        extension = collector.publish()
        assert extension is not None
        builder.finalize(
            kind="pyamplicol-process",
            producer=_producer(),
            model=_model(),
            configuration=_configuration(),
            processes=[_process()],
            default_process_id="dd_to_z",
            runtime=_runtime(),
            extensions={"evaluator_payload_container": extension},
        )

    with PacbinReader.open(destination / "evaluators.pacbin") as container:
        assert container.read_member(
            "processes/first/application.symjit",
            length=len(b"first-application"),
        ) == b"first-application"
        assert container.read_member(
            "processes/second/application.symjit",
            length=len(b"second-application"),
        ) == b"second-application"


def test_manifest_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["producer"]["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="unknown fields"):
        load_manifest(root, verify_payloads=False)


def test_manifest_accepts_native_digest_without_git_revision(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["producer"]["native_build_inputs_sha256"] = "a" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    producer = load_manifest(root, verify_payloads=False).producer

    assert producer["native_build_inputs_sha256"] == "a" * 64
    assert "git_revision" not in producer


def test_manifest_rejects_noncanonical_runtime_capabilities(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime"]["required_runtime_capabilities"] = [
        "symjit.application.complex-f64.v1",
        "symbolica.compiled-cpp.complex-f64.v1",
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="must be sorted"):
        load_manifest(root, verify_payloads=False)


def test_manifest_rejects_unknown_runtime_capability(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime"]["required_runtime_capabilities"] = ["unknown.runtime.v1"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="unsupported capabilities"):
        load_manifest(root, verify_payloads=False)


def test_manifest_rejects_runtime_capabilities_not_owned_by_a_process(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime"]["required_runtime_capabilities"] = [
        "symbolica.compiled-cpp.complex-f64.v1",
        "symjit.application.complex-f64.v1",
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="must equal the union"):
        load_manifest(root, verify_payloads=False)


@pytest.mark.parametrize(
    ("permutation", "message"),
    (
        ([0, 1], "complete permutation"),
        ([0, 0, 2], "complete permutation"),
        ([0, 2, 1], "incoming and outgoing sides"),
    ),
)
def test_manifest_rejects_invalid_alias_permutations(
    tmp_path: Path,
    permutation: list[int],
    message: str,
) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["processes"][0]["aliases"] = [
        {
            "id": "alias",
            "expression": "d d~ > z",
            "external_pdgs": [1, -1, 23],
            "external_permutation": permutation,
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match=message):
        load_manifest(root, verify_payloads=False)


def test_manifest_accepts_alias_permutations_on_both_sides(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["processes"][0]["aliases"] = [
        {
            "id": "incoming_swap",
            "expression": "d~ d > z",
            "external_pdgs": [-1, 1, 23],
            "external_permutation": [1, 0, 2],
        }
    ]
    payload["artifact_id"] = manifest_module.compute_artifact_id(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_manifest(root, verify_payloads=False)

    assert manifest.processes[0]["aliases"][0]["external_permutation"] == (1, 0, 2)


def test_manifest_alias_records_permuted_public_particle_order(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["processes"][0].update(
        {
            "expression": "d d~ > z g a",
            "external_pdgs": [1, -1, 23, 21, 22],
            "aliases": [
                {
                    "id": "cycled",
                    "expression": "d d~ > a z g",
                    "external_pdgs": [1, -1, 22, 23, 21],
                    "external_permutation": [0, 1, 3, 4, 2],
                }
            ],
        }
    )
    payload["artifact_id"] = manifest_module.compute_artifact_id(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_manifest(root, verify_payloads=False)

    assert manifest.processes[0]["aliases"][0] == {
        "id": "cycled",
        "expression": "d d~ > a z g",
        "external_pdgs": (1, -1, 22, 23, 21),
        "external_permutation": (0, 1, 3, 4, 2),
    }


def test_manifest_rejects_alias_pdg_order_inconsistent_with_permutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(root)
    path = root / "artifact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["processes"][0]["aliases"] = [
        {
            "id": "alias",
            "expression": "d d~ > z",
            "external_pdgs": [1, -1, 22],
            "external_permutation": [0, 1, 2],
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError, match="does not match external_permutation"):
        load_manifest(root, verify_payloads=False)


def _flavor_blind_group_map() -> ProcessGroupMap:
    return initial_process_groups(
        [("dd_to_z", "d d~ > z", (1, -1, 23))],
        {
            "dd_to_z": [
                GroupMember(
                    process_id="uu_to_z",
                    expression="u u~ > z",
                    pdgs=(2, -2, 23),
                    initial_count=2,
                    permutation=(0, 1, 2),
                    basis=GroupingBasis.FLAVOR_BLIND,
                    generated=False,
                    warning="Z couplings differ between d and u",
                )
            ]
        },
    )


def test_process_groups_are_rewritten_in_place_without_changing_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(root)
    assert read_process_groups(root) is None
    before = load_manifest(root)

    group_map = _flavor_blind_group_map()
    write_process_groups(root, group_map)
    first = load_manifest(root, verify_payloads=True)
    stored = read_process_groups(root)
    assert stored is not None
    assert stored.as_document() == group_map.as_document()
    assert first.artifact_id == before.artifact_id
    assert [
        (record.path, record.role, record.media_type) for record in first.payloads
    ] == [
        ("physics/process.json", "runtime-physics", "application/json"),
        ("process-groups.json", "process-groups", "application/json"),
    ]

    stored.analyses.append({"analysis": "numerical", "merged": 0})
    write_process_groups(root, stored)
    second = load_manifest(root, verify_payloads=True)
    assert second.artifact_id == before.artifact_id
    assert second.payloads[1].sha256 != first.payloads[1].sha256
    reread = read_process_groups(root)
    assert reread is not None
    assert reread.analyses == [{"analysis": "numerical", "merged": 0}]


def test_process_groups_must_list_each_artifact_process_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(root)
    unrelated = initial_process_groups([("uu_to_z", "u u~ > z", (2, -2, 23))])
    with pytest.raises(ArtifactError, match="missing: \\['dd_to_z'\\]"):
        write_process_groups(root, unrelated)
    assert read_process_groups(root) is None


def test_process_group_update_waits_for_the_generation_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _build(root)
    writer = threading.Thread(
        target=write_process_groups, args=(root, _flavor_blind_group_map())
    )
    with ArtifactTransaction(root, mode="append"):
        writer.start()
        writer.join(timeout=0.2)
        assert writer.is_alive()
        assert read_process_groups(root) is None
    writer.join()
    assert read_process_groups(root) is not None
    load_manifest(root, verify_payloads=True)
