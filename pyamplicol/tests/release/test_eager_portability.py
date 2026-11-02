# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pytest

from tools.ci import eager_portability as portability
from tools.ci.eager_portability_contract import symjit_storage_v3_target

ROOT = Path(__file__).resolve().parents[2]
# The workflow lives in the MadGraph7 repository that contains pyamplicol/.
WORKFLOW = ROOT.parent / ".github" / "workflows" / "pyamplicol-eager-portability.yml"
requires_workflow = pytest.mark.skipif(
    not WORKFLOW.is_file(), reason="outside the MadGraph7 repository"
)


@pytest.fixture
def contracts() -> portability.RuntimeContracts:
    return portability.RuntimeContracts(
        bundle_kind="pyamplicol-prepared-model",
        bundle_schema_version=1,
        eager_kernel_abi="pyamplicol-eager-kernel-v1",
        compiled_model_schema_version=9,
        symbolica_serialization_abi="symbolica-bincode2-v1",
        symjit_application_abi="symjit-application-storage-v3",
        symjit_plane_application_abi="pyamplicol-symjit-plane-application-v2",
        symjit_runtime_capability="symjit.application.complex-f64.v1",
        eager_runtime_capability="rusticol.eager-runtime-layout.complex-f64.v1",
        package_version="0.1.0.dev0+candidate.portability",
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bundle(
    path: Path,
    contracts: portability.RuntimeContracts,
    *,
    architecture: str = "x86_64",
    model: str = "built-in-sm",
    mutate: Callable[[dict[str, object], dict[str, bytes]], None] | None = None,
) -> Path:
    application_path = "kernels/000000/application-0.symjit"
    plane_application_path = "kernels/000000/application-0.plane.symjit"
    state_path = "kernels/000000/evaluator-state-0.evaluator.bin"
    model_path = "model/model.pyAmplicol-model.json"
    payloads = {
        application_path: b"portable-symjit-storage-v3-fixture\x00\x01",
        plane_application_path: b"portable-symjit-plane-storage-v3-fixture\x00\x03",
        state_path: b"architecture-neutral-symbolica-state-fixture\x00\x02",
        model_path: json.dumps(
            {
                "schema_version": contracts.compiled_model_schema_version,
                "source": {"kind": model},
            },
            sort_keys=True,
        ).encode("utf-8"),
    }
    f64_manifest: dict[str, object] = {
        "application_abi": contracts.symjit_application_abi,
        "application_path": application_path,
        "backend": "jit",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "element_layout": "complex-f64",
        "endianness": "little",
        "evaluator_state_path": state_path,
        "input_len": 1,
        "kind": "symjit-application-evaluator",
        "optimization_level": 2,
        "output_len": 1,
        "plane_application": {
            "application_abi": contracts.symjit_plane_application_abi,
            "application_path": plane_application_path,
            "complex": True,
            "compression": False,
            "compiler_type": "native",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "direct_arena": True,
            "element_layout": "split-complex-plane-major",
            "fast_complex": False,
            "fast_math": True,
            "input_complex_count": 1,
            "input_plane_count": 2,
            "optimization_level": 2,
            "output_complex_count": 1,
            "output_plane_count": 2,
            "simd": True,
            "source_digest": _sha256(b"synthetic structured instructions"),
            "storage_abi": contracts.symjit_application_abi,
            "target": {
                "endianness": "little",
                "word_bits": 64,
            },
            "threading": False,
            "translation_mode": "symbolica-structured-instructions",
        },
        "required_defuns": [],
        "runtime_capability": contracts.symjit_runtime_capability,
        "settings": {
            "backend": "jit",
            "compiled_inline_asm": "none",
            "compiled_native": False,
            "compiler_path": None,
            "effective_compiler_flags": [],
            "jit_optimization_level": 2,
        },
        "translation_mode": "indirect",
        "word_bits": 64,
    }
    manifest: dict[str, object] = {
        "eager_kernel_abi": contracts.eager_kernel_abi,
        "kernel_pack": {
            "backend": "jit",
            "dependency_abis": {
                "symbolica_serialization": contracts.symbolica_serialization_abi,
                "symjit_application": contracts.symjit_application_abi,
                "symjit_plane_application": contracts.symjit_plane_application_abi,
            },
            "kernels": [
                {
                    "f64_evaluator_manifest": f64_manifest,
                    "kernel_id": 0,
                }
            ],
            "kernel_variants": [],
            "optimization_settings": {
                "backend": "jit",
                "compiled_inline_asm": "none",
                "compiled_native": False,
                "compiler_path": None,
                "effective_compiler_flags": [],
                "jit_optimization_level": 2,
            },
            "producer": {
                "compiled_model_schema": contracts.compiled_model_schema_version,
                "distribution": "pyamplicol",
                "model_compiler_version": 13,
                "version": contracts.package_version,
            },
            "provenance": {
                "compiled_model_digest": "a" * 64,
                "model_name": model,
                "model_source": {"kind": model},
            },
            "target": symjit_storage_v3_target(architecture),
        },
        "kind": contracts.bundle_kind,
        "schema_version": contracts.bundle_schema_version,
    }
    if mutate is not None:
        mutate(manifest, payloads)
    manifest["members"] = [
        {
            "path": member,
            "sha256": _sha256(payload),
            "size": len(payload),
        }
        for member, payload in sorted(payloads.items())
    ]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in {
            "manifest.json": json.dumps(manifest, sort_keys=True).encode("utf-8"),
            **payloads,
        }.items():
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return path


def _write_transfer_fixture(
    directory: Path,
    contracts: portability.RuntimeContracts,
    *,
    architecture: str,
) -> Path:
    directory.mkdir()
    bundle = _write_bundle(
        directory / portability.DEFAULT_BUNDLE_NAME,
        contracts,
        architecture=architecture,
    )
    fixture = {
        "bundle": {
            "architecture_class": "portable",
            "bundle_sha256": portability._sha256_file(bundle),
            "filename": bundle.name,
            "preflight_evaluator_count": 1,
        },
        "expected": {
            "atol": portability.DEFAULT_ATOL,
            "imaginary": 0.0,
            "real": 1.0,
            "rtol": portability.DEFAULT_RTOL,
        },
        "kind": portability.TRANSFER_KIND,
        "process": {
            "expression": portability.DEFAULT_PROCESS,
            "id": portability.DEFAULT_PROCESS_ID,
            "momenta": [],
        },
        "producer": {
            "architecture_class": architecture,
            "git_commit": "fixture-commit",
            "machine": "x86_64" if architecture == "x86_64" else "arm64",
            "system": "Linux",
        },
        "schema_version": portability.TRANSFER_SCHEMA_VERSION,
    }
    (directory / portability.DEFAULT_FIXTURE_NAME).write_text(
        json.dumps(fixture),
        encoding="utf-8",
    )
    return directory


def _write_packed_consumer_artifact(
    artifact: Path,
    bundle: Path,
    contracts: portability.RuntimeContracts,
    *,
    include_plane_payload: bool = True,
    plane_payload_override: bytes | None = None,
) -> Path:
    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        write_pacbin_atomic,
    )

    artifact.mkdir()
    bundle_manifest, bundle_payloads = portability._archive_manifest(bundle)
    kernel_pack = bundle_manifest["kernel_pack"]
    assert isinstance(kernel_pack, dict)
    kernels = kernel_pack["kernels"]
    assert isinstance(kernels, list) and len(kernels) == 1
    kernel = kernels[0]
    assert isinstance(kernel, dict)
    f64 = kernel["f64_evaluator_manifest"]
    assert isinstance(f64, dict)

    filtered_pack = dict(kernel_pack)
    filtered_pack["kernels"] = [kernel]
    filtered_pack["kernel_variants"] = []
    pack_path = artifact / "model" / "eager-kernel-pack.json"
    pack_path.parent.mkdir(parents=True)
    pack_path.write_text(json.dumps(filtered_pack), encoding="utf-8")

    execution_path = (
        artifact / "processes" / portability.DEFAULT_PROCESS_ID / "execution.json"
    )
    execution_path.parent.mkdir(parents=True)
    execution_path.write_text(
        json.dumps({"kind": "pyamplicol-runtime-eager-execution"}),
        encoding="utf-8",
    )

    plane = f64["plane_application"]
    assert isinstance(plane, dict)
    packed_payloads: list[tuple[str, PacbinMemberKind, bytes | None]] = [
        (
            str(f64["application_path"]),
            PacbinMemberKind.SYMJIT_APPLICATION,
            None,
        ),
        (
            str(f64["evaluator_state_path"]),
            PacbinMemberKind.SYMBOLICA_EXACT_STATE,
            None,
        ),
    ]
    if include_plane_payload:
        packed_payloads.append(
            (
                str(plane["application_path"]),
                PacbinMemberKind.SYMJIT_APPLICATION,
                plane_payload_override,
            )
        )
    packed_sources = []
    for source_path, kind, override in packed_payloads:
        packed_sources.append(
            PacbinMemberSource(
                f"model/eager-kernels/{source_path}",
                kind,
                io.BytesIO(
                    bundle_payloads[source_path] if override is None else override
                ),
            )
        )
    container_path = artifact / "evaluators.pacbin"
    container_index = write_pacbin_atomic(container_path, packed_sources)

    payload_paths = (
        "evaluators.pacbin",
        "model/eager-kernel-pack.json",
        f"processes/{portability.DEFAULT_PROCESS_ID}/execution.json",
    )
    payloads = []
    for relative in payload_paths:
        path = artifact / relative
        payloads.append(
            {
                "executable": False,
                "path": relative,
                "sha256": portability._sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "artifact_id": "packed-portability-fixture",
        "extensions": {
            "eager_prepared_pack": {
                "backend": "jit",
                "eager_kernel_abi": contracts.eager_kernel_abi,
                "kernel_count": 1,
            },
            "evaluator_payload_container": {
                "index_sha256": container_index.index_sha256,
                "member_count": len(container_index.members),
                "path": "evaluators.pacbin",
                "storage_abi": "pacbin-v1",
            },
        },
        "kind": "pyamplicol-process",
        "payloads": payloads,
        "processes": [
            {
                "expression": portability.DEFAULT_PROCESS,
                "id": portability.DEFAULT_PROCESS_ID,
                "required_runtime_capabilities": [contracts.eager_runtime_capability],
            }
        ],
        "producer": {"version": contracts.package_version},
        "schema_version": 3,
    }
    (artifact / "artifact.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return artifact


@pytest.mark.parametrize("architecture", ("x86_64", "aarch64"))
def test_architecture_jit_bundle_audit_accepts_portable_storage_v3_pack(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    architecture: str,
) -> None:
    bundle = _write_bundle(
        tmp_path / "model.pyamplicol-model",
        contracts,
        architecture=architecture,
    )

    result = portability.audit_architecture_jit_bundle(
        bundle,
        contracts=contracts,
        expected_sha256=portability._sha256_file(bundle),
        expected_architecture_class=architecture,
    )

    assert result["backend"] == "jit"
    assert result["kernel_count"] == 1
    assert result["symjit_application_count"] == 1
    assert result["symjit_plane_application_count"] == 1
    assert result["exact_state_count"] == 1
    assert result["architecture_class"] == "portable"
    assert result["target"] == symjit_storage_v3_target(architecture)


def test_architecture_jit_bundle_audit_accepts_explicit_heft_model(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    bundle = _write_bundle(
        tmp_path / "heft.pyamplicol-model",
        contracts,
        model="built-in-sm-heft",
    )

    result = portability.audit_architecture_jit_bundle(
        bundle,
        contracts=contracts,
        expected_model_source="built-in-sm-heft",
    )

    assert result["kernel_count"] == 1


def test_architecture_jit_bundle_audit_rejects_wrong_storage_abi(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    def mutate(manifest: dict[str, object], _payloads: dict[str, bytes]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        dependencies = pack["dependency_abis"]
        assert isinstance(dependencies, dict)
        dependencies["symjit_application"] = "symjit-application-storage-v999"

    bundle = _write_bundle(
        tmp_path / "wrong-abi.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match="application storage ABI mismatch",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("application_abi", "pyamplicol-symjit-plane-application-v999"),
        ("storage_abi", "symjit-application-storage-v999"),
    ),
)
def test_architecture_jit_bundle_audit_rejects_wrong_plane_application_abi(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    field: str,
    value: str,
) -> None:
    def mutate(manifest: dict[str, object], _payloads: dict[str, bytes]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        kernels = pack["kernels"]
        assert isinstance(kernels, list)
        kernel = kernels[0]
        assert isinstance(kernel, dict)
        f64 = kernel["f64_evaluator_manifest"]
        assert isinstance(f64, dict)
        plane = f64["plane_application"]
        assert isinstance(plane, dict)
        plane[field] = value

    bundle = _write_bundle(
        tmp_path / "wrong-plane-abi.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match=rf"incompatible plane-application {field!r}",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


def test_architecture_jit_bundle_audit_rejects_missing_plane_payload(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    def mutate(_manifest: dict[str, object], payloads: dict[str, bytes]) -> None:
        payloads.pop("kernels/000000/application-0.plane.symjit")

    bundle = _write_bundle(
        tmp_path / "missing-plane.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match="omits referenced plane-application payload",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("input_plane_count", 1),
        ("output_plane_count", 3),
        ("descriptor_order", "outputs-then-inputs"),
        ("direct_arena", False),
        ("optimization_level", 3),
    ),
)
def test_architecture_jit_bundle_audit_rejects_wrong_plane_descriptor_contract(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    field: str,
    value: object,
) -> None:
    def mutate(manifest: dict[str, object], _payloads: dict[str, bytes]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        kernels = pack["kernels"]
        assert isinstance(kernels, list)
        kernel = kernels[0]
        assert isinstance(kernel, dict)
        f64 = kernel["f64_evaluator_manifest"]
        assert isinstance(f64, dict)
        plane = f64["plane_application"]
        assert isinstance(plane, dict)
        plane[field] = value

    bundle = _write_bundle(
        tmp_path / "wrong-plane-descriptor.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match=rf"incompatible plane-application {field!r}",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b"\x7fELF" + b"\0" * 64, "contains ELF image"),
        (b"!<arch>\n" + b"\0" * 64, "contains static archive"),
        (b"\xcf\xfa\xed\xfe" + b"\0" * 64, "contains Mach-O image"),
    ),
)
def test_architecture_jit_bundle_audit_rejects_native_machine_code(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    payload: bytes,
    message: str,
) -> None:
    def mutate(_manifest: dict[str, object], payloads: dict[str, bytes]) -> None:
        payloads["kernels/000000/application-0.symjit"] = payload

    bundle = _write_bundle(
        tmp_path / "native.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(portability.PortabilityError, match=message):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


@pytest.mark.parametrize(
    ("member", "payload"),
    (
        ("kernels/000000/generated.cpp", b"double kernel(double x);\n"),
        ("kernels/000000/generated.asm", b"section .text\n"),
        ("kernels/000000/kernel.o", b"opaque native object fixture"),
    ),
)
def test_architecture_jit_bundle_audit_rejects_native_source_or_object_payload(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    member: str,
    payload: bytes,
) -> None:
    def mutate(_manifest: dict[str, object], payloads: dict[str, bytes]) -> None:
        payloads[member] = payload

    bundle = _write_bundle(
        tmp_path / "native-source.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match="native/source payload",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


def test_architecture_jit_bundle_audit_rejects_legacy_target_spelling(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    def mutate(manifest: dict[str, object], _payloads: dict[str, bytes]) -> None:
        pack = manifest["kernel_pack"]
        assert isinstance(pack, dict)
        target = pack["target"]
        assert isinstance(target, dict)
        target["portable"] = True
        target["target_triple"] = "portable-symjit-mir"

    bundle = _write_bundle(
        tmp_path / "falsely-portable.pyamplicol-model",
        contracts,
        mutate=mutate,
    )

    with pytest.raises(
        portability.PortabilityError,
        match="must be portable 64-bit little-endian",
    ):
        portability.audit_architecture_jit_bundle(bundle, contracts=contracts)


def test_architecture_jit_bundle_audit_accepts_cross_architecture_load(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    bundle = _write_bundle(
        tmp_path / "x86-pack.pyamplicol-model",
        contracts,
        architecture="x86_64",
    )

    result = portability.audit_architecture_jit_bundle(
        bundle,
        contracts=contracts,
        expected_architecture_class="aarch64",
    )

    assert result["architecture_class"] == "portable"
    assert result["target"] == symjit_storage_v3_target("aarch64")


@pytest.mark.parametrize(
    ("machine", "expected"),
    (("AMD64", "x86_64"), ("x86_64", "x86_64"), ("arm64", "aarch64")),
)
def test_machine_names_normalize_to_storage_v3_architecture_classes(
    machine: str,
    expected: str,
) -> None:
    assert portability.architecture_class(machine) == expected


def test_consumer_generation_command_uses_eager_pack_without_model_compile(
    tmp_path: Path,
) -> None:
    command = portability._generation_command(
        Path("/python"),
        model=tmp_path / "transferred.pyamplicol-model",
        output=tmp_path / "artifact",
        execution_mode="eager",
    )

    assert command[:4] == ["/python", "-m", "pyamplicol", "generate"]
    assert "compile" not in command
    assert command[command.index("--execution-mode") + 1] == "eager"
    assert command[command.index("--model") + 1].endswith(
        "transferred.pyamplicol-model"
    )
    assert "--no-post-build-validation" in command
    assert "--no-emit-api-bundle" in command


def test_consumer_verifier_accepts_byte_identical_packed_kernel_payloads(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    bundle = _write_bundle(tmp_path / "model.pyamplicol-model", contracts)
    artifact = _write_packed_consumer_artifact(
        tmp_path / "artifact",
        bundle,
        contracts,
    )

    result = portability.verify_consumer_artifact(
        artifact,
        bundle,
        contracts=contracts,
        bundle_audit={"kernel_count": 1},
    )

    assert result["copied_kernel_payload_count"] == 3
    assert result["filtered_kernel_count"] == 1
    assert result["filtered_kernel_variant_count"] == 0


def test_consumer_verifier_rejects_missing_packed_plane_payload(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    bundle = _write_bundle(tmp_path / "model.pyamplicol-model", contracts)
    artifact = _write_packed_consumer_artifact(
        tmp_path / "artifact",
        bundle,
        contracts,
        include_plane_payload=False,
    )

    with pytest.raises(
        portability.PortabilityError,
        match=r"omitted transferred kernel payload.*plane\.symjit",
    ):
        portability.verify_consumer_artifact(
            artifact,
            bundle,
            contracts=contracts,
            bundle_audit={"kernel_count": 1},
        )


def test_consumer_verifier_rejects_changed_packed_plane_payload(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
) -> None:
    bundle = _write_bundle(tmp_path / "model.pyamplicol-model", contracts)
    artifact = _write_packed_consumer_artifact(
        tmp_path / "artifact",
        bundle,
        contracts,
        plane_payload_override=b"changed-plane-application",
    )

    with pytest.raises(
        portability.PortabilityError,
        match=r"rebuilt or changed kernel payload.*plane\.symjit",
    ):
        portability.verify_consumer_artifact(
            artifact,
            bundle,
            contracts=contracts,
            bundle_audit={"kernel_count": 1},
        )


def test_python_executable_preserves_virtual_environment_symlink(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "python"
    launcher.symlink_to(Path(sys.executable))

    result = portability._python_executable(launcher)

    assert result == launcher.absolute()
    assert result.is_symlink()


def test_runtime_contracts_do_not_initialize_the_driver_runtime(
    contracts: portability.RuntimeContracts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def child(command, **kwargs):
        assert command[:2] == [sys.executable, "-c"]
        assert "_runtime_contracts_in_process()" in command[2]
        assert kwargs["check"] and kwargs["capture_output"]
        return subprocess.CompletedProcess(command, 0, json.dumps(asdict(contracts)))

    monkeypatch.setattr(portability.subprocess, "run", child)
    monkeypatch.setattr(
        portability,
        "_runtime_contracts_in_process",
        lambda: pytest.fail("parent must not initialize Symbolica before CLI children"),
    )

    assert portability._runtime_contracts() == contracts


def test_producer_finishes_all_cli_children_before_native_parent_work(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children: list[list[str]] = []
    native_work: list[str] = []
    monkeypatch.setattr(portability.platform, "system", lambda: "Linux")
    monkeypatch.setattr(portability.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(portability, "_runtime_contracts", lambda: contracts)
    monkeypatch.setattr(portability, "_git_commit", lambda: "fixture-commit")
    monkeypatch.setattr(portability, "_validation_momenta", lambda _artifact: [])

    def child(command, **_kwargs):
        assert not native_work, "CLI child overlaps a parent Symbolica runtime"
        children.append(command)
        if command[3:5] == ["model", "compile"]:
            _write_bundle(Path(command[6]), contracts, model=command[5])

    def preflight(_bundle):
        assert len(children) == 4
        native_work.append("preflight")
        return 1

    def write_asset(bundle, output, *, identifier="built-in-sm-jit-o2", **_kwargs):
        assert len(children) == 4
        native_work.append("write")
        output.mkdir(parents=True, exist_ok=True)
        retained = output / f"{identifier}.pyamplicol-model"
        retained.write_bytes(bundle.read_bytes())
        return output / f"{identifier}.metadata.json", retained

    def evaluate(_artifact, _momenta):
        assert len(children) == 4
        native_work.append("evaluate")
        return 1 + 0j

    monkeypatch.setattr(portability, "_run", child)
    monkeypatch.setattr(portability, "_preflight_all_prepared_applications", preflight)
    monkeypatch.setattr(portability, "_write_source_ready_asset", write_asset)
    monkeypatch.setattr(portability, "_evaluate_artifact", evaluate)

    report = portability.produce_transfer(
        tmp_path / "transfer",
        python=Path(sys.executable),
        expected_system="Linux",
        expected_machine="x86_64",
        asset_mode="release",
    )

    assert [command[3] for command in children] == [
        "model", "model", "generate", "generate"
    ]
    assert native_work == [
        "preflight", "write", "preflight", "write", "evaluate", "evaluate"
    ]
    assert report["expected"] == {"real": 1.0, "imaginary": 0.0}


def test_producer_rejects_an_unexpected_architecture_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(portability.platform, "system", lambda: "Linux")
    monkeypatch.setattr(portability.platform, "machine", lambda: "aarch64")

    with pytest.raises(
        portability.PortabilityError,
        match=r"producer architecture class.*expected 'x86_64'",
    ):
        portability.produce_transfer(
            tmp_path / "transfer",
            python=Path(sys.executable),
            expected_system="Linux",
            expected_machine="x86_64",
        )

    assert not (tmp_path / "transfer").exists()


def test_producer_cli_accepts_explicit_release_asset_mode(tmp_path: Path) -> None:
    arguments = portability._parser().parse_args(
        [
            "produce",
            str(tmp_path / "transfer"),
            "--asset-mode",
            "release",
        ]
    )

    assert arguments.asset_mode == "release"


def test_release_asset_output_uses_source_only_store_layout(tmp_path: Path) -> None:
    output = tmp_path / "transfer"

    assert portability._source_ready_asset_directory(output, "candidate") == output
    assert (
        portability._source_ready_asset_directory(
            output,
            "release",
        )
        == output / "release_assets" / "prepared_models"
    )
    with pytest.raises(
        portability.PortabilityError,
        match="unsupported prepared-model asset mode",
    ):
        portability._source_ready_asset_directory(output, "mixed")


def test_transfer_bundle_member_accepts_only_root_or_release_store() -> None:
    assert portability._transfer_bundle_member("built-in-sm.pyamplicol-model") == Path(
        "built-in-sm.pyamplicol-model"
    )
    assert portability._transfer_bundle_member(
        "release_assets/prepared_models/built-in-sm.pyamplicol-model"
    ) == Path("release_assets/prepared_models/built-in-sm.pyamplicol-model")
    for value in (
        "nested/built-in-sm.pyamplicol-model",
        "release_assets/built-in-sm.pyamplicol-model",
        "release_assets/prepared_models/nested/built-in-sm.pyamplicol-model",
    ):
        with pytest.raises(
            portability.PortabilityError,
            match="exact release prepared-model source store",
        ):
            portability._transfer_bundle_member(value)


def test_release_asset_dispatch_reaches_real_writer(tmp_path: Path) -> None:
    bundle = tmp_path / "invalid.pyamplicol-model"
    bundle.write_bytes(b"not a prepared-model bundle")

    with pytest.raises(RuntimeError, match="prepared-model bundle is invalid"):
        portability._write_source_ready_asset(
            bundle,
            tmp_path / "output",
            architecture="x86_64",
            asset_mode="release",
        )


def test_consumer_accepts_portable_pack_across_architectures(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = _write_transfer_fixture(
        tmp_path / "transfer",
        contracts,
        architecture="x86_64",
    )
    monkeypatch.setattr(portability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(portability.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(portability, "_runtime_contracts", lambda: contracts)
    monkeypatch.setattr(portability, "_git_commit", lambda: "fixture-commit")
    monkeypatch.setattr(
        portability,
        "_preflight_all_prepared_applications",
        lambda _bundle: 1,
    )

    class GenerationStarted(RuntimeError):
        pass

    def stop_at_generation(*_args: object, **_kwargs: object) -> None:
        raise GenerationStarted

    monkeypatch.setattr(portability, "_run", stop_at_generation)

    with pytest.raises(GenerationStarted):
        portability.consume_transfer(
            transfer,
            python=Path(sys.executable),
            report_path=tmp_path / "report.json",
            expected_system="Darwin",
            expected_machine="arm64",
        )


def test_consumer_rejects_inconsistent_producer_host_provenance(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = _write_transfer_fixture(
        tmp_path / "transfer",
        contracts,
        architecture="x86_64",
    )
    fixture_path = transfer / portability.DEFAULT_FIXTURE_NAME
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["producer"]["machine"] = "arm64"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr(portability.platform, "system", lambda: "Linux")
    monkeypatch.setattr(portability.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(portability, "_runtime_contracts", lambda: contracts)
    monkeypatch.setattr(
        portability,
        "_preflight_all_prepared_applications",
        lambda _bundle: 1,
    )

    with pytest.raises(
        portability.PortabilityError,
        match="machine and architecture-class provenance differ",
    ):
        portability.consume_transfer(
            transfer,
            python=Path(sys.executable),
            report_path=tmp_path / "report.json",
            expected_system="Linux",
            expected_machine="x86_64",
        )


def test_consumer_accepts_same_architecture_pack_across_operating_systems(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = _write_transfer_fixture(
        tmp_path / "transfer",
        contracts,
        architecture="x86_64",
    )
    monkeypatch.setattr(portability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(portability.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(portability, "_runtime_contracts", lambda: contracts)
    monkeypatch.setattr(portability, "_git_commit", lambda: "fixture-commit")
    monkeypatch.setattr(
        portability,
        "_preflight_all_prepared_applications",
        lambda _bundle: 1,
    )

    class GenerationStarted(RuntimeError):
        pass

    def stop_at_generation(*_args: object, **_kwargs: object) -> None:
        raise GenerationStarted

    monkeypatch.setattr(portability, "_run", stop_at_generation)

    with pytest.raises(GenerationStarted):
        portability.consume_transfer(
            transfer,
            python=Path(sys.executable),
            report_path=tmp_path / "report.json",
            expected_system="Darwin",
            expected_machine="AMD64",
        )


def test_consumer_finishes_generation_before_native_parent_work(
    tmp_path: Path,
    contracts: portability.RuntimeContracts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = _write_transfer_fixture(
        tmp_path / "transfer", contracts, architecture="x86_64"
    )
    events: list[str] = []
    monkeypatch.setattr(portability.platform, "system", lambda: "Linux")
    monkeypatch.setattr(portability.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(portability, "_runtime_contracts", lambda: contracts)
    monkeypatch.setattr(portability, "_git_commit", lambda: "fixture-commit")

    def child(*_args, **_kwargs):
        assert not events, "CLI child overlaps a parent Symbolica runtime"
        events.append("generation")

    def preflight(_bundle):
        assert events == ["generation"]
        events.append("preflight")
        return 1

    def verify(*_args, **_kwargs):
        events.append("verify")
        return {}

    def evaluate(*_args):
        events.append("evaluate")
        return 1 + 0j

    monkeypatch.setattr(portability, "_run", child)
    monkeypatch.setattr(portability, "_preflight_all_prepared_applications", preflight)
    monkeypatch.setattr(portability, "verify_consumer_artifact", verify)
    monkeypatch.setattr(portability, "_evaluate_artifact", evaluate)

    portability.consume_transfer(
        transfer,
        python=Path(sys.executable),
        report_path=tmp_path / "report.json",
        expected_system="Linux",
        expected_machine="x86_64",
    )

    assert events == ["generation", "preflight", "verify", "evaluate"]


def test_compiler_guard_records_and_denies_external_tool_execution() -> None:
    with portability.compiler_guard() as (environment, marker):
        completed = subprocess.run(
            ["c++", "--version"],
            check=False,
            capture_output=True,
            env=environment,
        )
        assert completed.returncode == 97
        assert marker.read_text(encoding="utf-8").strip().endswith("c++ --version")


@requires_workflow
def test_portability_workflow_transfers_portable_packs() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    trigger = workflow.split("on:\n", maxsplit=1)[1].split(
        "\npermissions:\n", maxsplit=1
    )[0]

    assert "pull_request:" in trigger
    assert "push:" in trigger
    assert "workflow_dispatch:" in trigger
    assert "rust/crates/rusticol-core/src/eager_runtime/**" in trigger
    assert "rust/crates/rusticol-core/src/engine/**" in trigger
    assert "rust/crates/rusticol-core/src/evaluator/symjit.rs" in trigger
    for adapter in (
        "symjit_plane.rs",
        "symjit_direct.rs",
        "symjit_eager_direct.rs",
        "symjit_compiled_direct.rs",
    ):
        assert workflow.count(f"rust/crates/rusticol-core/src/evaluator/{adapter}") == 2
    assert "dependencies/release-lock.toml" in trigger
    assert "src/pyamplicol/evaluators/symbolica*.py" in trigger
    assert workflow.count("eager_portability.py produce") == 1
    assert workflow.count("eager_portability.py consume") == 1
    assert workflow.count("pyamplicol-eager-jit-transfer-${{") == 2
    assert "needs: produce-architecture-packs" in workflow
    assert "ubuntu-24.04" in workflow
    assert "macos-15-intel" in workflow
    assert "macos-15" in workflow
    assert "target: linux-x86-64" in workflow
    assert "target: macos-x86-64" in workflow
    assert "target: macos-arm64" in workflow
    assert workflow.count("pack_architecture: x86_64") == 3
    assert workflow.count("pack_architecture: aarch64") == 2
    assert (
        "target: macos-x86-64\n"
        "            system: Darwin\n"
        "            machine: x86_64\n"
        "            pack_architecture: x86_64"
    ) in workflow
    assert (
        "target: macos-arm64\n"
        "            system: Darwin\n"
        "            machine: arm64\n"
        "            pack_architecture: aarch64"
    ) in workflow
    assert "--expected-system ${{ matrix.system }}" in workflow
    assert "--expected-machine ${{ matrix.machine }}" in workflow
    assert workflow.count("tools/ci/memory_watchdog.py --limit-gib 30 --") == 4
    assert "dependencies/install_dependencies.py" not in workflow
    assert workflow.count("prepare_release_prepared_models.py") == 2
    assert workflow.count("bootstrap-wheel .artifacts/release-prepared-bootstrap") == 2
    assert workflow.count("python -m venv .venv") == 2
    assert workflow.count("--asset-mode release") == 1
    assert "--with-legacy-amplicol" not in workflow
    assert "--with-reference-fft" not in workflow
    assert "--without-legacy-amplicol" not in workflow
    assert "--without-reference-fft" not in workflow
    assert "contents: read" in workflow
    assert "continue-on-error" not in workflow
    assert "publish-pypi" not in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert workflow.count("PYAMPLICOL_BUILD_MODE: release") == 2
    assert "PYAMPLICOL_BUILD_MODE: candidate" not in workflow
    assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in workflow
    assert "dependencies/candidate-Cargo.lock" not in workflow
    assert workflow.count("      - pyamplicol/dependencies/release-lock.toml\n") == 2
    assert (
        "actions/download-artifact@634f93cb2916e3fdff6788551b99b062d0335ce0" in workflow
    )
    assert (
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
    )


@requires_workflow
def test_portability_workflow_does_not_expose_credentials() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "persist-credentials: false" in workflow
    assert "id-token: write" not in workflow
    assert "PYPI" not in workflow
    assert "secrets." not in workflow
    assert os.fspath(ROOT) not in workflow
