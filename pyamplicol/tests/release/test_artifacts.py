# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))
_LOCK = tomllib.loads(
    (ROOT / "dependencies" / "release-lock.toml").read_text(encoding="utf-8")
)
_RELEASE_VERSION = str(_LOCK["project"]["version"])
_CONTRIBUTOR_LOCK = tomllib.loads(
    (ROOT / "dependencies" / "contributor-lock.toml").read_text(encoding="utf-8")
)
_DEFAULT_REQUIREMENTS = [
    f"{entry['distribution']}=={entry['version']}"
    for entry in _LOCK["python_dependencies"]
]
_CANDIDATE_REQUIREMENTS = [
    (
        f"{entry['distribution']}=="
        f"{_CONTRIBUTOR_LOCK['symbolica']['candidate_version']}"
        if entry["distribution"] == _LOCK["symbolica"]["python_distribution"]
        else f"{entry['distribution']}=={entry['version']}"
    )
    for entry in _LOCK["python_dependencies"]
]
_LEGAL_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "licenses/Symbolica.txt",
    "licenses/SymJIT.txt",
)
_COMPILED_MODEL_KIND = "pyamplicol-compiled-model"
_BUILTIN_MODEL_SOURCE_KIND = "built-in-sm"
_MODEL_COMPILER_VERSION = 11
_RUNTIME_ARTIFACT_ID_PAYLOAD_ROLES = frozenset(
    {
        "compiled-model",
        "evaluator-manifest",
        "evaluator-state",
        "model-parameters",
        "runtime-physics",
    }
)
_MODEL_COMPILER_SOURCE_FILES = {
    "pyamplicol/models/__init__.py": b'"""Synthetic model package."""\n',
    "pyamplicol/models/compiler.py": b"def compile_model():\n    return 'compiled'\n",
    "pyamplicol/models/loading.py": b"MODEL_COMPILER_VERSION = 11\n",
    "pyamplicol/_internal/physics/__init__.py": b'"""Synthetic physics."""\n',
    "pyamplicol/_internal/physics/rules.py": b"PROPAGATOR = 'feynman'\n",
    "pyamplicol/processes/core_syntax.py": b"CORE_SYNTAX_VERSION = 1\n",
}
_BUILTIN_MODEL_SOURCE_FILES = {
    "pyamplicol/models/builtin/__init__.py": b'"""Synthetic built-in model."""\n',
    "pyamplicol/models/builtin/adapters.py": b"def source_digest():\n    pass\n",
    "pyamplicol/models/builtin/model.py": b"MODEL_NAME = 'sm'\n",
}
_PREPARED_PACK_COMPILER_SOURCE_FILES = {
    **{
        f"pyamplicol/models/{name}.py": f"# synthetic {name}\n".encode()
        for name in (
            "prepared",
            "prepared_catalog",
            "prepared_catalog_builder",
            "prepared_catalog_helpers",
            "prepared_compile",
            "prepared_target",
        )
    },
    **{
        f"pyamplicol/evaluators/{name}.py": f"# synthetic {name}\n".encode()
        for name in (
            "symbolica",
            "symbolica_adapters",
            "symbolica_compile",
            "symbolica_helpers",
            "symbolica_settings",
        )
    },
    **{
        f"pyamplicol/config/{name}.py": f"# synthetic config {name}\n".encode()
        for name in ("__init__", "errors", "models", "registry", "resolver")
    },
    "pyamplicol/_internal/physics/symbols.py": b"# synthetic symbols\n",
    "pyamplicol/_internal/versions.py": b"# synthetic versions\n",
}


def _model_compiler_digest() -> str:
    digest = hashlib.sha256()
    sources = dict(_MODEL_COMPILER_SOURCE_FILES)
    sources.update(
        {
            name: data
            for name, data in _PREPARED_PACK_COMPILER_SOURCE_FILES.items()
            if PurePosixPath(name).parent
            in {
                PurePosixPath("pyamplicol/models"),
                PurePosixPath("pyamplicol/_internal/physics"),
            }
        }
    )
    for name, data in sorted(sources.items()):
        relative = PurePosixPath(name).relative_to("pyamplicol").as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def _builtin_model_source_digest() -> str:
    inner = hashlib.sha256()
    for name, data in sorted(_BUILTIN_MODEL_SOURCE_FILES.items()):
        inner.update(PurePosixPath(name).name.encode("utf-8") + b"\0")
        inner.update(data)
        inner.update(b"\0")
    outer = hashlib.sha256()
    outer.update(b"built-in-sm\0")
    outer.update(inner.hexdigest().encode("ascii"))
    return outer.hexdigest()


import _artifacts as artifacts  # noqa: E402
from _artifacts import (  # noqa: E402
    _REQUIRED_WHEEL_PACKAGE_MEMBERS,
    MAX_WHEEL_BYTES,
    ArtifactError,
    _scan_capi_archive,
    _scan_static_runtime_families,
    audit_sdist,
    audit_wheel,
)
from audit_sdist import (  # noqa: E402
    PREPARED_MODEL_ARCHITECTURES,
    PREPARED_MODEL_ASSET_BASENAME,
    PREPARED_MODEL_ASSETS,
    REQUIRED_SDIST_MEMBERS,
    prepared_model_asset_members,
)
from prepare_selftest_fixture import ARTIFACT_IDENTITY_CONTRACT  # noqa: E402


@pytest.fixture
def candidate_dependency_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contributor_path = tmp_path / "contributor-lock.toml"
    state_path = tmp_path / "install-state.json"
    contributor_data = (ROOT / "dependencies" / "contributor-lock.toml").read_bytes()
    contributor_path.write_bytes(contributor_data)
    symbolica = _CONTRIBUTOR_LOCK["symbolica"]
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "sources": {
                    "symbolica": {
                        "url": symbolica["source_url"],
                        "revision": symbolica["candidate_revision"],
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        artifacts,
        "_CONTRIBUTOR_DEPENDENCY_LOCK",
        contributor_path,
    )
    monkeypatch.setattr(artifacts, "_CANDIDATE_INSTALL_STATE", state_path)


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _prepared_model_files(
    prefix: str,
    *,
    mode: str = "release",
) -> dict[str, bytes]:
    root = prefix.rstrip("/")
    files = {f"{root}/__init__.py": b'"""Synthetic prepared models."""\n'}
    for identifier, model_source in PREPARED_MODEL_ASSETS.items():
        for architecture in PREPARED_MODEL_ARCHITECTURES:
            stem = f"{identifier}-{architecture}"
            bundle_name = f"{stem}.pyamplicol-model"
            bundle = f"synthetic portable {stem} prepared model\n".encode()
            metadata = {
                "schema_version": 1,
                "prepared_model_bundle_schema": 1,
                "eager_kernel_abi": "pyamplicol-eager-kernel-v1",
                "id": identifier,
                "model": model_source,
                "backend": "jit",
                "jit_optimization_level": 2,
                "bundle": bundle_name,
                "bundle_size": len(bundle),
                "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
                "dependencies": {
                    "symbolica_serialization_abi": "symbolica-bincode2-v1",
                    "symjit_application_abi": "symjit-application-storage-v3",
                    "symjit_plane_application_abi": (
                        "pyamplicol-symjit-plane-application-v2"
                    ),
                },
                "build_contract": {"candidate_fingerprint": None, "mode": mode},
                "producer": {},
                "target": {
                    "portable": True,
                    "word_bits": 64,
                    "endianness": "little",
                    "target_triple": "symjit-storage-v3-portable",
                    "cpu_features": [],
                },
            }
            files[f"{root}/{bundle_name}"] = bundle
            files[f"{root}/{stem}.metadata.json"] = (
                json.dumps(metadata, sort_keys=True) + "\n"
            ).encode()
    return files


def _modified_prepared_metadata(
    prefix: str,
    architecture: str,
    *,
    values: dict[str, object] | None = None,
    target_values: dict[str, object] | None = None,
) -> tuple[str, bytes]:
    root = prefix.rstrip("/")
    stem = f"{PREPARED_MODEL_ASSET_BASENAME}-{architecture}"
    metadata_name = f"{root}/{stem}.metadata.json"
    metadata = json.loads(_prepared_model_files(prefix)[metadata_name])
    if values is not None:
        metadata.update(values)
    if target_values is not None:
        metadata["target"].update(target_values)
    return metadata_name, (json.dumps(metadata, sort_keys=True) + "\n").encode()


def _selftest_arena_stage(
    *,
    amplitude: bool,
    direct_codegen_optimization_level: int,
) -> dict[str, object]:
    application_path = "evaluator.symjit"
    plane_application_path = "evaluator.plane.symjit"
    arena = "amplitude" if amplitude else "current"
    output_component = 0 if amplitude else 2
    input_component = {
        "parameter_index": 0,
        "kind": "value",
        "source_id": 0,
        "component": 0,
        "global_component": 0,
        "real_valued": False,
    }
    evaluator = {
        "kind": "symjit-application-evaluator",
        "runtime_capability": "symjit.application.complex-f64.v1",
        "application_abi": "symjit-application-storage-v3",
        "application_path": application_path,
        "plane_application": {
            "application_path": plane_application_path,
            "application_abi": "pyamplicol-symjit-plane-application-v2",
            "storage_abi": "symjit-application-storage-v3",
            "element_layout": "split-complex-plane-major",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "input_complex_count": 1,
            "output_complex_count": 1,
            "input_plane_count": 2,
            "output_plane_count": 2,
            "compiler_type": "native",
            "translation_mode": "symbolica-structured-instructions",
            "optimization_level": 2,
            "simd": True,
            "complex": True,
            "fast_math": True,
            "fast_complex": False,
            "compression": False,
            "threading": False,
            "direct_arena": True,
            "source_digest": hashlib.sha256(
                b"synthetic structured instructions"
            ).hexdigest(),
            "target": {"word_bits": 64, "endianness": "little"},
        },
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": 2,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "input_len": 1,
        "output_len": 1,
    }
    return {
        "stage_kind": "amplitude-roots" if amplitude else "current-combine",
        "parameter_layout": "stage-local-value-momentum",
        "parameter_count": 1,
        "output_length": 1,
        "input_components": [input_component],
        "output_slots": [
            {
                "output_start": 0,
                "output_stop": 1,
                "component_start": output_component,
                "component_stop": output_component + 1,
            }
        ],
        "evaluator": evaluator,
        "compiled_plane_arena": {
            "schema_version": 1,
            "kind": "compiled-plane-arena-stage",
            "application_abi": "pyamplicol-compiled-plane-kernel-v2",
            "source_application_abi": "pyamplicol-symjit-plane-application-v2",
            "element_layout": "split-complex-component-major",
            "output_operation": "overwrite",
            "output_factor": "identity",
            "input_output_aliasing": "forbidden",
            "output_output_aliasing": "forbidden",
            "input_bindings": [input_component],
            "output_bindings": [
                {
                    "output_index": 0,
                    "arena": arena,
                    "component": output_component,
                }
            ],
            "leaves": [
                {
                    "application_path": plane_application_path,
                    "source_application_abi": (
                        "pyamplicol-symjit-plane-application-v2"
                    ),
                    "optimization_level": 2,
                    "direct_codegen_optimization_level": (
                        direct_codegen_optimization_level
                    ),
                    "input_len": 1,
                    "output_len": 1,
                    "input_indices": [0],
                    "output_start": 0,
                    "output_stop": 1,
                }
            ],
        },
    }


def _selftest_execution(
    *,
    missing_arena_capability: str | None = None,
    missing_symjit_capability: str | None = None,
    direct_codegen_optimization_level: int = 2,
) -> dict[str, object]:
    arena_capability = "compiled-plane-arena-v1"
    symjit_capability = "symjit.application.complex-f64.v1"
    execution_capabilities: list[str] = []
    stage_capabilities: list[str] = []
    if missing_arena_capability != "execution":
        execution_capabilities.append(arena_capability)
    if missing_symjit_capability != "execution":
        execution_capabilities.append(symjit_capability)
    if missing_arena_capability != "stage_evaluators":
        stage_capabilities.append(arena_capability)
    if missing_symjit_capability != "stage_evaluators":
        stage_capabilities.append(symjit_capability)
    return {
        "required_runtime_capabilities": execution_capabilities,
        "compiled": {
            "kind": "generic-dag-stage-blueprint",
            "stage_evaluators": {
                "kind": "generic-dag-stage-evaluator-artifacts",
                "required_runtime_capabilities": stage_capabilities,
                "stage_count": 2,
                "stages": [
                    _selftest_arena_stage(
                        amplitude=False,
                        direct_codegen_optimization_level=(
                            direct_codegen_optimization_level
                        ),
                    )
                ],
                "amplitude_stage": _selftest_arena_stage(
                    amplitude=True,
                    direct_codegen_optimization_level=(
                        direct_codegen_optimization_level
                    ),
                ),
            },
        },
    }


def _selftest_files(
    rust_target: str,
    version: str,
    compiled_model_schema: int = _LOCK["abis"]["compiled_model"],
    *,
    producer_compiled_model_schema: int | None = None,
    model_compiled_model_schema: int | None = None,
    compiled_model_kind: str = _COMPILED_MODEL_KIND,
    model_compiler_version: int = _MODEL_COMPILER_VERSION,
    compiled_model_producer_schema: int | None = None,
    compiled_model_producer_version: str | None = None,
    compiled_model_producer_compiler_version: int | None = None,
    compiled_model_compiler_sha256: str | None = None,
    compiled_model_source_kind: str = _BUILTIN_MODEL_SOURCE_KIND,
    compiled_model_source_digest: str | None = None,
    omitted_api_payload: str | None = None,
    missing_arena_capability: str | None = None,
    missing_symjit_capability: str | None = None,
    direct_codegen_optimization_level: int = 2,
) -> dict[str, bytes]:
    payload_path = "processes/smoke/evaluator.symjit"
    payload = b"synthetic trusted SymJIT application"
    plane_payload_path = "processes/smoke/evaluator.plane.symjit"
    plane_payload = b"synthetic trusted SymJIT plane application"
    execution_path = "processes/smoke/execution.json"
    execution = (
        json.dumps(
            _selftest_execution(
                missing_arena_capability=missing_arena_capability,
                missing_symjit_capability=missing_symjit_capability,
                direct_codegen_optimization_level=(direct_codegen_optimization_level),
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    compiled_model_path = "model/compiled-model.json"
    compiled_model = (
        json.dumps(
            {
                "kind": compiled_model_kind,
                "schema_version": compiled_model_schema,
                "model_compiler_version": model_compiler_version,
                "source": {
                    "kind": compiled_model_source_kind,
                    "digest": (
                        _builtin_model_source_digest()
                        if compiled_model_source_digest is None
                        else compiled_model_source_digest
                    ),
                },
                "producer": {
                    "pyamplicol": (
                        version
                        if compiled_model_producer_version is None
                        else compiled_model_producer_version
                    ),
                    "compiled_model_schema_version": (
                        _LOCK["abis"]["compiled_model"]
                        if compiled_model_producer_schema is None
                        else compiled_model_producer_schema
                    ),
                    "model_compiler_version": (
                        model_compiler_version
                        if compiled_model_producer_compiler_version is None
                        else compiled_model_producer_compiler_version
                    ),
                    "model_compiler_sha256": (
                        _model_compiler_digest()
                        if compiled_model_compiler_sha256 is None
                        else compiled_model_compiler_sha256
                    ),
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    api_payloads = {
        "API/validation_points.dat": (
            b"RUSTICOL_VALIDATION_POINTS_V1\n",
            "validation-momenta",
            "text/tab-separated-values",
        ),
        "API/python/check_standalone.py": (
            b"# synthetic Python driver\n",
            "api-source",
            "text/x-python",
        ),
        "API/c/Makefile": (
            b"# synthetic C Makefile\n",
            "api-build-file",
            "text/x-makefile",
        ),
        "API/c/check_standalone.c": (
            b"/* synthetic C driver */\n",
            "api-source",
            "text/x-csrc",
        ),
        "API/cpp/Makefile": (
            b"# synthetic C++ Makefile\n",
            "api-build-file",
            "text/x-makefile",
        ),
        "API/cpp/check_standalone.cpp": (
            b"// synthetic C++ driver\n",
            "api-source",
            "text/x-c++src",
        ),
        "API/fortran/Makefile": (
            b"# synthetic Fortran Makefile\n",
            "api-build-file",
            "text/x-makefile",
        ),
        "API/fortran/check_standalone.f90": (
            b"! synthetic Fortran driver\n",
            "api-source",
            "text/x-fortran",
        ),
        "API/rust/Makefile": (
            b"# synthetic Rust Makefile\n",
            "api-build-file",
            "text/x-makefile",
        ),
        "API/rust/check_standalone.rs": (
            b"// synthetic Rust driver\n",
            "api-source",
            "text/x-rust",
        ),
    }
    if omitted_api_payload is not None:
        api_payloads.pop(omitted_api_payload)
    producer_schema = (
        _LOCK["abis"]["compiled_model"]
        if producer_compiled_model_schema is None
        else producer_compiled_model_schema
    )
    model_schema = (
        _LOCK["abis"]["compiled_model"]
        if model_compiled_model_schema is None
        else model_compiled_model_schema
    )
    artifact_capabilities = []
    process_capabilities = []
    if missing_arena_capability != "artifact_runtime":
        artifact_capabilities.append("compiled-plane-arena-v1")
    if missing_symjit_capability != "artifact_runtime":
        artifact_capabilities.append("symjit.application.complex-f64.v1")
    if missing_arena_capability != "process":
        process_capabilities.append("compiled-plane-arena-v1")
    if missing_symjit_capability != "process":
        process_capabilities.append("symjit.application.complex-f64.v1")
    manifest = {
        "schema_version": 3,
        "artifact_id": "0" * 64,
        "extensions": {
            "artifact_identity": dict(ARTIFACT_IDENTITY_CONTRACT),
        },
        "producer": {
            "distribution": "pyamplicol",
            "version": version,
            "target": {"triple": rust_target, "cpu_features": []},
            "versions": {"compiled_model": producer_schema},
        },
        "model": {"compiled_schema_version": model_schema},
        "runtime": {
            "engine": "rusticol",
            "engine_version": version,
            "required_runtime_capabilities": artifact_capabilities,
        },
        "processes": [
            {
                "id": "smoke",
                "required_runtime_capabilities": process_capabilities,
            }
        ],
        "payloads": [
            {
                "path": compiled_model_path,
                "role": "compiled-model",
                "media_type": "application/json",
                "size_bytes": len(compiled_model),
                "sha256": hashlib.sha256(compiled_model).hexdigest(),
            },
            {
                "path": payload_path,
                "role": "evaluator-state",
                "media_type": "application/vnd.symjit.application",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "target": {"triple": rust_target, "cpu_features": []},
            },
            {
                "path": plane_payload_path,
                "role": "evaluator-state",
                "media_type": "application/vnd.symjit.application",
                "size_bytes": len(plane_payload),
                "sha256": hashlib.sha256(plane_payload).hexdigest(),
                "target": {"triple": rust_target, "cpu_features": []},
            },
            {
                "path": execution_path,
                "role": "evaluator-manifest",
                "media_type": "application/json",
                "size_bytes": len(execution),
                "sha256": hashlib.sha256(execution).hexdigest(),
            },
            *[
                {
                    "path": path,
                    "role": role,
                    "media_type": media_type,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for path, (data, role, media_type) in sorted(api_payloads.items())
            ],
        ],
    }
    runtime_payloads = [
        dict(payload)
        for payload in manifest["payloads"]
        if payload.get("role") in _RUNTIME_ARTIFACT_ID_PAYLOAD_ROLES
    ]
    runtime_payloads.sort(key=lambda payload: str(payload.get("path", "")))
    canonical = (
        json.dumps(
            {
                "kind": "pyamplicol-runtime-payload-identity",
                "schema_version": 1,
                "payloads": runtime_payloads,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    manifest["artifact_id"] = hashlib.sha256(canonical).hexdigest()
    prefix = f"pyamplicol/assets/selftest/{rust_target}"
    files = {
        f"{prefix}/expected.json": json.dumps(
            {
                "schema_version": 1,
                "target": rust_target,
                "artifact_path": "artifact",
            }
        ).encode(),
        f"{prefix}/artifact/artifact.json": (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode(),
        f"{prefix}/artifact/{compiled_model_path}": compiled_model,
        f"{prefix}/artifact/{payload_path}": payload,
        f"{prefix}/artifact/{plane_payload_path}": plane_payload,
        f"{prefix}/artifact/{execution_path}": execution,
    }
    files.update(
        {
            f"{prefix}/artifact/{path}": data
            for path, (data, _role, _media_type) in api_payloads.items()
        }
    )
    return files


def _wheel(
    directory: Path,
    *,
    platform_tag: str = "manylinux_2_28_x86_64",
    rust_target: str = "x86_64-unknown-linux-gnu",
    version: str = _RELEASE_VERSION,
    candidate: bool = False,
    extension: bytes = b"synthetic abi3 extension",
    bad_record: bool = False,
    requirement: str | None = None,
    requirements: list[str] | None = None,
    system_libraries: list[str] | None = None,
    sdk_version: str | None = None,
    notice: bytes | None = b"third-party notices\n",
    license_files: tuple[str, ...] | None = None,
    omitted_member: str | None = None,
    extra_files: dict[str, bytes] | None = None,
    compiled_model_schema: int = _LOCK["abis"]["compiled_model"],
    producer_compiled_model_schema: int | None = None,
    model_compiled_model_schema: int | None = None,
    compiled_model_kind: str = _COMPILED_MODEL_KIND,
    model_compiler_version: int = _MODEL_COMPILER_VERSION,
    compiled_model_producer_schema: int | None = None,
    compiled_model_producer_version: str | None = None,
    compiled_model_producer_compiler_version: int | None = None,
    compiled_model_compiler_sha256: str | None = None,
    compiled_model_source_kind: str = _BUILTIN_MODEL_SOURCE_KIND,
    compiled_model_source_digest: str | None = None,
    omitted_selftest_api_payload: str | None = None,
    selftest_missing_arena_capability: str | None = None,
    selftest_missing_symjit_capability: str | None = None,
    selftest_direct_codegen_optimization_level: int = 2,
    selftest_fixture_bootstrap: bool = False,
    metadata_description: str = "",
) -> Path:
    if requirement is not None and requirements is not None:
        raise ValueError("use requirement or requirements, not both")
    runtime_requirements = (
        [requirement]
        if requirement is not None
        else list(
            (_CANDIDATE_REQUIREMENTS if candidate else _DEFAULT_REQUIREMENTS)
            if requirements is None
            else requirements
        )
    )
    declared_legal_files = _LEGAL_FILES if license_files is None else license_files
    dist_info = f"pyamplicol-{version}.dist-info"
    sdk_archive = b"synthetic static archive"
    files = {
        name: b"synthetic required wheel member\n"
        for name in _REQUIRED_WHEEL_PACKAGE_MEMBERS
    }
    files.update(
        {
            "pyamplicol/__init__.py": b"",
            "pyamplicol/_examples/README.md": b"load ../models/example.json\n",
            "pyamplicol/_rusticol.abi3.so": extension,
            "pyamplicol/_sdk/config.py": b"",
            "pyamplicol/_sdk/include/rusticol.h": b"void rusticol(void);\n",
            "pyamplicol/_sdk/include/rusticol.hpp": b"#pragma once\n",
            "pyamplicol/_sdk/fortran/rusticol.f90": b"module rusticol\nend module\n",
            "pyamplicol/_sdk/rust/rusticol.rs": b"// safe Rust wrapper\n",
            "pyamplicol/_sdk/lib/librusticol_capi.a": sdk_archive,
            "pyamplicol/_sdk/metadata.json": json.dumps(
                {
                    "schema_version": 1,
                    "abi_version": 1,
                    "version": sdk_version or version,
                    "target": rust_target,
                    "archive": "lib/librusticol_capi.a",
                    "rust_source": "rust/rusticol.rs",
                }
            ).encode(),
            "pyamplicol/_sdk/link.json": json.dumps(
                {
                    "schema_version": 1,
                    "target": rust_target,
                    "system_libraries": system_libraries or ["c", "m"],
                    "frameworks": [],
                }
            ).encode(),
            f"{dist_info}/METADATA": (
                "Metadata-Version: 2.4\n"
                "Name: pyamplicol\n"
                f"Version: {version}\n"
                "Requires-Python: >=3.11\n"
                "License-Expression: 0BSD\n"
                + "".join(f"License-File: {item}\n" for item in declared_legal_files)
                + "".join(f"Requires-Dist: {item}\n" for item in runtime_requirements)
                + f"\n{metadata_description}"
            ).encode(),
            f"{dist_info}/WHEEL": (
                "Wheel-Version: 1.0\n"
                "Generator: synthetic\n"
                "Root-Is-Purelib: false\n"
                f"Tag: cp311-abi3-{platform_tag}\n\n"
            ).encode(),
            f"{dist_info}/entry_points.txt": (
                b"[console_scripts]\n"
                b"pyamplicol = pyamplicol.cli:main\n"
                b"rusticol-config = rusticol_config:main\n"
            ),
        }
    )
    files.update(_MODEL_COMPILER_SOURCE_FILES)
    files.update(_BUILTIN_MODEL_SOURCE_FILES)
    files.update(_PREPARED_PACK_COMPILER_SOURCE_FILES)
    files.update(
        _prepared_model_files(
            "pyamplicol/assets/prepared_models",
            mode="candidate" if candidate else "release",
        )
    )
    files.update(
        _selftest_files(
            rust_target,
            version,
            compiled_model_schema,
            producer_compiled_model_schema=producer_compiled_model_schema,
            model_compiled_model_schema=model_compiled_model_schema,
            compiled_model_kind=compiled_model_kind,
            model_compiler_version=model_compiler_version,
            compiled_model_producer_schema=compiled_model_producer_schema,
            compiled_model_producer_version=compiled_model_producer_version,
            compiled_model_producer_compiler_version=(
                compiled_model_producer_compiler_version
            ),
            compiled_model_compiler_sha256=compiled_model_compiler_sha256,
            compiled_model_source_kind=compiled_model_source_kind,
            compiled_model_source_digest=compiled_model_source_digest,
            omitted_api_payload=omitted_selftest_api_payload,
            missing_arena_capability=selftest_missing_arena_capability,
            missing_symjit_capability=selftest_missing_symjit_capability,
            direct_codegen_optimization_level=(
                selftest_direct_codegen_optimization_level
            ),
        )
    )
    for relative in _LEGAL_FILES:
        if relative == "THIRD_PARTY_NOTICES.md" and notice is None:
            continue
        data = (
            notice
            if relative == "THIRD_PARTY_NOTICES.md"
            else (ROOT / relative).read_bytes()
        )
        assert data is not None
        files[f"{dist_info}/licenses/{relative}"] = data
    for name in (
        "README.md",
        "artifact-manifest-v3.schema.json",
        "runtime-physics-v1.schema.json",
    ):
        files[f"pyamplicol/assets/schemas/{name}"] = (
            ROOT / "schemas" / name
        ).read_bytes()
    if candidate:
        files["pyamplicol/_build_info.json"] = json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "selftest_fixture_bootstrap": selftest_fixture_bootstrap,
                "source_checkout": "/Users/developer/pyamplicol",
                "version": version,
            }
        ).encode()
    else:
        files["pyamplicol/_build_info.json"] = json.dumps(
            {
                "schema_version": 1,
                "publishable": True,
                "selftest_fixture_bootstrap": False,
                "source_revision": "a" * 40,
                "version": version,
            }
        ).encode()
    if extra_files is not None:
        files.update(extra_files)
    if omitted_member is not None:
        files.pop(omitted_member)
    record_name = f"{dist_info}/RECORD"
    rows = [
        [name, _record_hash(data), str(len(data))]
        for name, data in sorted(files.items())
    ]
    rows.append([record_name, "", ""])
    if bad_record:
        rows[0][1] = "sha256=wrong"
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[record_name] = record.getvalue().encode()

    filename = f"pyamplicol-{version}-cp311-abi3-{platform_tag}.whl"
    path = directory / filename
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return path


def _sdist(
    directory: Path,
    *,
    version: str = _RELEASE_VERSION,
    candidate: bool = False,
    manifest_path: str | None = None,
    project_requirement: str | None = None,
    omitted_member: str | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    python_version = version.replace("-dev.0", ".dev0")
    root = f"pyamplicol-{python_version}"
    release_cargo_lock = (ROOT / "Cargo.lock").read_bytes()
    files = {
        name: b"synthetic required sdist member\n"
        for name in {*REQUIRED_SDIST_MEMBERS, "PKG-INFO"}
    }
    files.update(
        {
            "Cargo.lock": release_cargo_lock,
            "Cargo.toml": f'[workspace.package]\nversion = "{version}"\n'.encode(),
            "LICENSE": b"0BSD\n",
            "README.md": b"pyamplicol\n",
            "THIRD_PARTY_NOTICES.md": b"notices\n",
            "build_backend/_pyamplicol_build.py": b"",
            "dependencies/release-lock.toml": (
                ROOT / "dependencies" / "release-lock.toml"
            ).read_bytes(),
            "pyproject.toml": (
                '[project]\nname = "pyamplicol"\n'
                + (
                    f"dependencies = [{json.dumps(project_requirement)}]\n"
                    if project_requirement is not None
                    else ""
                )
            ).encode(),
            "tools/release/_artifacts.py": b"",
            "tools/release/_common.py": b"",
            "tools/release/build_from_sdist.py": b"",
            "tools/release/build_release_artifacts.py": b"",
            "tools/release/check_dependencies.py": b"",
            "tools/release/install_wheel.py": b"",
            "tools/release/publish_dry_run.py": b"",
            "tools/release/test_deployment.py": b"",
        }
    )
    files.update(
        {
            f"src/{name}": data
            for name, data in _PREPARED_PACK_COMPILER_SOURCE_FILES.items()
        }
    )
    files.update(_prepared_model_files("src/pyamplicol/assets/prepared_models"))
    if candidate:
        files["src/pyamplicol/_build_info.json"] = json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "version": python_version,
            }
        ).encode()
    else:
        files["src/pyamplicol/_build_info.json"] = json.dumps(
            {
                "schema_version": 1,
                "publishable": True,
                "selftest_fixture_bootstrap": False,
                "source_revision": "a" * 40,
                "version": python_version,
            }
        ).encode()
    if manifest_path is not None:
        files["docs/development/BUILD_MANIFEST.toml"] = (
            f'repository = "{manifest_path}"\n'.encode()
        )
    if extra_files is not None:
        files.update(extra_files)
    if omitted_member is not None:
        files.pop(omitted_member)
    path = directory / f"pyamplicol-{python_version}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_required_sdist_keeps_the_portable_source_selftest() -> None:
    members = REQUIRED_SDIST_MEMBERS

    assert (
        "src/pyamplicol/assets/selftest/portable-64le/artifact/artifact.json" in members
    )
    assert {
        "examples/data/pp_zjj_momenta.json",
        "src/pyamplicol/assets/api_templates/rust/Makefile",
        "src/pyamplicol/assets/api_templates/rust/check_standalone.rs",
        "tools/release/prepare_selftest_fixture.py",
        "tests/fixtures/reference/analytic-oracles-v2.json",
        "tests/fixtures/reference/legacy-fortran-v2.json",
        "tests/fixtures/reference/physics-v2.json",
        "tests/fixtures/reference/reference-fixture-v2.manifest.json",
    } <= members


def test_generated_umami_templates_are_required_in_wheel_and_sdist() -> None:
    resources = {
        f"pyamplicol/assets/api_templates/umami/{name}"
        for name in (
            "Makefile",
            "README.md",
            "UMAMI_LICENSE",
            "umami.h",
            "umami.c",
            "umami_provider.h",
            "umami_driver.c",
            "umami.exports",
            "umami.exports.macos",
        )
    }
    assert resources <= set(_REQUIRED_WHEEL_PACKAGE_MEMBERS)
    assert {f"src/{name}" for name in resources} <= REQUIRED_SDIST_MEMBERS


def test_standalone_sdk_query_package_is_required_in_wheel_and_sdist() -> None:
    resources = {"rusticol_config/__init__.py", "rusticol_config/__main__.py"}
    assert resources <= set(_REQUIRED_WHEEL_PACKAGE_MEMBERS)
    assert {f"src/{name}" for name in resources} <= REQUIRED_SDIST_MEMBERS


def test_required_sdist_keeps_both_prepared_model_architectures() -> None:
    assert (
        prepared_model_asset_members("src/pyamplicol/assets/prepared_models")
        <= REQUIRED_SDIST_MEMBERS
    )


@pytest.mark.parametrize(
    "missing_member",
    sorted(
        prepared_model_asset_members("pyamplicol/assets/prepared_models")
        - {"pyamplicol/assets/prepared_models/__init__.py"}
    ),
)
def test_wheel_requires_both_prepared_model_asset_pairs(
    tmp_path: Path,
    missing_member: str,
) -> None:
    wheel = _wheel(tmp_path, omitted_member=missing_member)

    with pytest.raises(ArtifactError, match="prepared-model asset inventory"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_rejects_generic_or_extra_prepared_model_assets(tmp_path: Path) -> None:
    prefix = "pyamplicol/assets/prepared_models"
    wheel = _wheel(
        tmp_path,
        extra_files={
            f"{prefix}/{PREPARED_MODEL_ASSET_BASENAME}.pyamplicol-model": (
                b"legacy generic prepared model"
            )
        },
    )

    with pytest.raises(ArtifactError, match=r"prepared-model.*extra=.*built-in-sm"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_rejects_wrong_prepared_model_target_class(tmp_path: Path) -> None:
    prefix = "pyamplicol/assets/prepared_models"
    metadata_name, metadata = _modified_prepared_metadata(
        prefix,
        "x86_64",
        target_values={
            "target_triple": "symjit-storage-v3-x86_64",
            "portable": False,
        },
    )
    wheel = _wheel(tmp_path, extra_files={metadata_name: metadata})

    with pytest.raises(ArtifactError, match="target class is invalid"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_rejects_prepared_model_o3_drift(tmp_path: Path) -> None:
    prefix = "pyamplicol/assets/prepared_models"
    metadata_name, metadata = _modified_prepared_metadata(
        prefix,
        "aarch64",
        values={"jit_optimization_level": 3},
    )
    wheel = _wheel(tmp_path, extra_files={metadata_name: metadata})

    with pytest.raises(ArtifactError, match="metadata identity is invalid"):
        audit_wheel(wheel, mode="release", native_scan=False)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({"bundle_size": 1}, id="size"),
        pytest.param({"bundle_sha256": "0" * 64}, id="sha256"),
    ],
)
def test_wheel_rejects_prepared_model_bundle_identity_drift(
    tmp_path: Path,
    values: dict[str, object],
) -> None:
    prefix = "pyamplicol/assets/prepared_models"
    metadata_name, metadata = _modified_prepared_metadata(
        prefix,
        "aarch64",
        values=values,
    )
    wheel = _wheel(tmp_path, extra_files={metadata_name: metadata})

    with pytest.raises(ArtifactError, match="bundle hash/size is invalid"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_accepts_independently_serialized_portable_prepared_model_assets(
    tmp_path: Path,
) -> None:
    wheel = _wheel(tmp_path)

    report = audit_wheel(wheel, mode="release", native_scan=False)

    assert report.target == "manylinux_2_28_x86_64"


def test_release_and_candidate_wheels_are_distinct_and_audited(
    tmp_path: Path,
    candidate_dependency_provenance: None,
) -> None:
    release = _wheel(tmp_path)
    report = audit_wheel(release, mode="release", native_scan=False)
    assert report.target == "manylinux_2_28_x86_64"
    assert report.abi_tag == "abi3"

    candidate_version = f"{_RELEASE_VERSION}.dev0+candidate.0123456789ab"
    candidate = _wheel(
        tmp_path,
        version=candidate_version,
        candidate=True,
    )
    candidate_report = audit_wheel(candidate, mode="candidate", native_scan=False)
    assert candidate_report.version == candidate_version
    with pytest.raises(ArtifactError, match="release wheel has invalid version"):
        audit_wheel(candidate, mode="release", native_scan=False)


def test_release_metadata_may_document_candidate_workflows(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        metadata_description="Contributor candidate setup is non-publishable.\n",
    )

    assert (
        audit_wheel(wheel, mode="release", native_scan=False).version
        == _RELEASE_VERSION
    )


def test_release_wheel_requires_the_minimal_build_marker(tmp_path: Path) -> None:
    missing = _wheel(
        tmp_path,
        omitted_member="pyamplicol/_build_info.json",
    )
    with pytest.raises(ArtifactError, match="one _build_info"):
        audit_wheel(missing, mode="release", native_scan=False)

    missing.unlink()
    invalid = _wheel(
        tmp_path,
        extra_files={
            "pyamplicol/_build_info.json": json.dumps(
                {
                    "schema_version": 1,
                    "publishable": True,
                    "selftest_fixture_bootstrap": False,
                    "source_revision": "a" * 40,
                    "source_checkout": "/private/source",
                    "version": _RELEASE_VERSION,
                }
            ).encode()
        },
    )
    with pytest.raises(ArtifactError, match="release build marker"):
        audit_wheel(invalid, mode="release", native_scan=False)


def test_candidate_and_release_symbolica_wheel_pins_can_match(
    tmp_path: Path,
    candidate_dependency_provenance: None,
) -> None:
    candidate_version = f"{_RELEASE_VERSION}.dev0+candidate.0123456789ab"
    release_pin = _wheel(
        tmp_path,
        version=candidate_version,
        candidate=True,
        requirements=_DEFAULT_REQUIREMENTS,
    )
    candidate_report = audit_wheel(
        release_pin,
        mode="candidate",
        native_scan=False,
    )
    assert candidate_report.version == candidate_version

    release_pin.unlink()
    candidate_pin = _wheel(
        tmp_path,
        requirements=_CANDIDATE_REQUIREMENTS,
    )
    release_report = audit_wheel(
        candidate_pin,
        mode="release",
        native_scan=False,
    )
    assert release_report.version == _LOCK["project"]["version"]


def test_candidate_dependency_provenance_uses_compact_source_map(
    candidate_dependency_provenance: None,
) -> None:
    assert artifacts._candidate_dependency_overrides() == {
        "symbolica": _CONTRIBUTOR_LOCK["symbolica"]["candidate_version"],
        "ufo-model-loader": _LOCK["ufo_model_loader"]["required_version"],
    }


def test_candidate_dependency_provenance_rejects_extra_attestations(
    tmp_path: Path,
    candidate_dependency_provenance: None,
) -> None:
    state_path = artifacts.ROOT / artifacts._CANDIDATE_INSTALL_STATE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["created_utc"] = "2026-07-31T00:00:00Z"
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    candidate = _wheel(
        tmp_path,
        version=f"{_RELEASE_VERSION}.dev0+candidate.0123456789ab",
        candidate=True,
    )

    with pytest.raises(ArtifactError, match="only its source revisions"):
        audit_wheel(candidate, mode="candidate", native_scan=False)


def test_candidate_dependency_provenance_rejects_heavy_source_attestations(
    tmp_path: Path,
    candidate_dependency_provenance: None,
) -> None:
    state_path = artifacts.ROOT / artifacts._CANDIDATE_INSTALL_STATE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["sources"]["symbolica"]["worktree_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    candidate = _wheel(
        tmp_path,
        version=f"{_RELEASE_VERSION}.dev0+candidate.0123456789ab",
        candidate=True,
    )

    with pytest.raises(ArtifactError, match="invalid descriptor"):
        audit_wheel(candidate, mode="candidate", native_scan=False)


def test_candidate_wheel_rejects_selftest_fixture_bootstrap_marker(
    tmp_path: Path,
    candidate_dependency_provenance: None,
) -> None:
    candidate = _wheel(
        tmp_path,
        version=f"{_RELEASE_VERSION}.dev0+candidate.0123456789ab",
        candidate=True,
        selftest_fixture_bootstrap=True,
    )

    with pytest.raises(ArtifactError, match="bootstrap wheels are not deployable"):
        audit_wheel(candidate, mode="candidate", native_scan=False)


def test_host_native_linux_tag_is_candidate_only(
    tmp_path: Path,
    candidate_dependency_provenance: None,
) -> None:
    candidate = _wheel(
        tmp_path,
        version=f"{_RELEASE_VERSION}.dev0+candidate.0123456789ab",
        candidate=True,
        platform_tag="linux_x86_64",
    )

    report = audit_wheel(candidate, mode="candidate", native_scan=False)
    assert report.target == "manylinux_2_28_x86_64"
    assert report.rust_target == "x86_64-unknown-linux-gnu"

    with pytest.raises(ArtifactError, match="wheel platform tag"):
        audit_wheel(candidate, mode="release", native_scan=False)


def test_candidate_allows_only_rustup_standard_library_source_paths(
    tmp_path: Path,
    candidate_dependency_provenance: None,
) -> None:
    candidate_version = f"{_RELEASE_VERSION}.dev0+candidate.0123456789ab"
    rustup_location = (
        b"/Users/developer/.rustup/toolchains/stable-aarch64-apple-darwin/"
        b"lib/rustlib/src/rust/library/core/src/slice/index.rs"
    )
    candidate = _wheel(
        tmp_path,
        version=candidate_version,
        candidate=True,
        extension=rustup_location,
    )
    assert audit_wheel(candidate, mode="candidate", native_scan=False).version == (
        candidate_version
    )

    release = _wheel(tmp_path, extension=rustup_location)
    with pytest.raises(ArtifactError, match="non-relocatable path marker"):
        audit_wheel(release, mode="release", native_scan=False)

    other_local_path = _wheel(
        tmp_path,
        version=f"{_RELEASE_VERSION}.dev0+candidate.abcdef012345",
        candidate=True,
        extension=b"/Users/developer/project/private.rs",
    )
    with pytest.raises(ArtifactError, match="non-relocatable path marker"):
        audit_wheel(other_local_path, mode="candidate", native_scan=False)


def test_short_repository_root_is_matched_only_at_a_path_token_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = b"/io"
    monkeypatch.setattr(artifacts, "_REPOSITORY_PATH_MARKER", marker)

    assert not artifacts._contains_forbidden_path(
        b"/rustc/hash/library/std/src/io/error.rs",
        marker,
    )
    assert not artifacts._contains_forbidden_path(
        b"symbolic/path/to/io/helpers.rs",
        marker,
    )
    assert not artifacts._contains_forbidden_path(
        b"\0native-instruction-bytes:\x8f/ioX\x91",
        marker,
    )
    assert artifacts._contains_forbidden_path(
        b"compiled from /io/dependencies/checkouts/symjit/rust/lib.rs",
        marker,
    )
    assert artifacts._contains_forbidden_path(
        b"\0/io/source/rust/crates/rusticol-core/src/lib.rs",
        marker,
    )
    assert artifacts._contains_forbidden_path(b"checkout root: /io\0", marker)


def test_native_scan_requires_short_repository_directory_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = b"/io"
    monkeypatch.setattr(artifacts, "_REPOSITORY_PATH_MARKER", marker)
    monkeypatch.setattr(artifacts, "_FORBIDDEN_PATH_MARKERS", (b"/tmp/", marker))

    artifacts._scan_embedded_paths({"pyamplicol/_rusticol.abi3.so": b"\x8f/io\x91"})
    with pytest.raises(ArtifactError, match="non-relocatable path marker '/io/'"):
        artifacts._scan_embedded_paths(
            {"pyamplicol/_rusticol.abi3.so": b"\0/io/source.rs\0"}
        )


def test_temporary_path_marker_is_matched_only_at_a_path_token_boundary() -> None:
    marker = b"/tmp/"

    assert not artifacts._contains_forbidden_path(
        b"/pyamplicol/checkout/.artifacts/dev-install/tmp/build/source.rs",
        marker,
    )
    assert artifacts._contains_forbidden_path(
        b"compiled from /tmp/build/source.rs",
        marker,
    )
    assert artifacts._contains_forbidden_path(b"\0/tmp/build/source.rs", marker)


def test_release_allows_rust_standard_library_backtrace_source_paths(
    tmp_path: Path,
) -> None:
    standard_library_location = (
        b"/rustc/59807616e1fa2540724bfbac14d7976d7e4a3860/library/std/src/"
        b"../../backtrace/src/symbolize/gimli.rs"
    )
    wheel = _wheel(tmp_path, extension=standard_library_location)
    assert (
        audit_wheel(wheel, mode="release", native_scan=False).version
        == _RELEASE_VERSION
    )


def test_release_allows_remapped_rust_standard_library_backtrace_paths(
    tmp_path: Path,
) -> None:
    remapped_standard_library_location = (
        b"library/std/src/../../backtrace/src/symbolize/gimli.rs"
    )
    wheel = _wheel(tmp_path, extension=remapped_standard_library_location)
    assert (
        audit_wheel(wheel, mode="release", native_scan=False).version
        == _RELEASE_VERSION
    )


@pytest.mark.parametrize("marker", [b"__gmpz_init", b"mpfr_set", b"malachite-q"])
def test_native_scan_rejects_static_arithmetic_runtime_markers(
    tmp_path: Path,
    marker: bytes,
) -> None:
    binary = tmp_path / "native-artifact"
    binary.write_bytes(b"prefix\x00" + marker + b"\x00suffix")
    with pytest.raises(ArtifactError, match="forbidden static arithmetic runtime"):
        _scan_static_runtime_families(binary, "test artifact")


@pytest.mark.parametrize(
    "symbol",
    ["_PyErr_SetString", "PyLong_FromLong", "_Py_Dealloc"],
)
def test_capi_archive_scan_rejects_undefined_python_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"synthetic archive")
    monkeypatch.delenv("LLVM_NM", raising=False)
    monkeypatch.setattr(artifacts.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout=f"                 U {symbol}\n", stderr=""
        ),
    )

    with pytest.raises(ArtifactError, match="undefined Python"):
        _scan_capi_archive(archive, allow_local_rustup=False)


def test_capi_archive_scan_ignores_raw_pyo3_marker_with_clean_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"synthetic compressed archive bytes containing pyo3")
    monkeypatch.delenv("LLVM_NM", raising=False)
    monkeypatch.setattr(artifacts.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout="", stderr=""
        ),
    )

    assert _scan_capi_archive(archive, allow_local_rustup=False)


def test_capi_archive_scan_defers_incompatible_llvm_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"synthetic archive")
    monkeypatch.delenv("LLVM_NM", raising=False)
    monkeypatch.setattr(artifacts.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"],
            1,
            stdout="",
            stderr="Unknown attribute kind (91) from LLVM 20",
        ),
    )

    assert not _scan_capi_archive(archive, allow_local_rustup=False)


def test_record_direct_dependency_and_size_fail_closed(tmp_path: Path) -> None:
    bad_record = _wheel(tmp_path, bad_record=True)
    with pytest.raises(ArtifactError, match="RECORD hash"):
        audit_wheel(bad_record, mode="release", native_scan=False)

    bad_record.unlink()
    direct = _wheel(tmp_path, requirement="numpy @ file:///tmp/numpy.whl")
    with pytest.raises(ArtifactError, match="direct URL or path"):
        audit_wheel(direct, mode="release", native_scan=False)

    direct.unlink()
    parent_reference = _wheel(tmp_path, extension=b"built from ../parent/source")
    with pytest.raises(ArtifactError, match="non-relocatable"):
        audit_wheel(parent_reference, mode="release", native_scan=False)

    oversized = (
        tmp_path / f"pyamplicol-{_RELEASE_VERSION}-cp311-abi3-manylinux_2_28_x86_64.whl"
    )
    with oversized.open("wb") as stream:
        stream.truncate(MAX_WHEEL_BYTES + 1)
    with pytest.raises(ArtifactError, match="limit"):
        audit_wheel(oversized, mode="release", native_scan=False)


def test_wheel_filename_version_must_match_metadata(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    renamed = wheel.with_name(wheel.name.replace(f"-{_RELEASE_VERSION}-", "-9.9.9-"))
    wheel.rename(renamed)
    with pytest.raises(ArtifactError, match="filename version"):
        audit_wheel(renamed, mode="release", native_scan=False)


def test_wheel_selftest_compiled_model_matches_release_schema(tmp_path: Path) -> None:
    expected = _LOCK["abis"]["compiled_model"]
    wheel = _wheel(tmp_path, compiled_model_schema=expected - 1)

    with pytest.raises(ArtifactError, match=f"release schema {expected}"):
        audit_wheel(wheel, mode="release", native_scan=False)


@pytest.mark.parametrize(
    ("owner", "contract"),
    (
        ("artifact_runtime", "artifact"),
        ("process", "artifact"),
        ("execution", "execution"),
        ("stage_evaluators", "execution"),
    ),
)
def test_wheel_selftest_requires_complete_arena_capability(
    tmp_path: Path,
    owner: str,
    contract: str,
) -> None:
    wheel = _wheel(
        tmp_path,
        selftest_missing_arena_capability=owner,
    )

    with pytest.raises(
        ArtifactError,
        match=rf"Arena {contract} contract.*must require compiled-plane-arena-v1",
    ):
        audit_wheel(wheel, mode="release", native_scan=False)


@pytest.mark.parametrize(
    ("owner", "contract"),
    (
        ("artifact_runtime", "artifact"),
        ("process", "artifact"),
        ("execution", "execution"),
        ("stage_evaluators", "execution"),
    ),
)
def test_wheel_selftest_requires_complete_symjit_capability(
    tmp_path: Path,
    owner: str,
    contract: str,
) -> None:
    wheel = _wheel(
        tmp_path,
        selftest_missing_symjit_capability=owner,
    )

    with pytest.raises(
        ArtifactError,
        match=(
            rf"Arena {contract} contract.*must require "
            r"symjit\.application\.complex-f64\.v1"
        ),
    ):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_selftest_requires_plane_codegen_optimization_match(
    tmp_path: Path,
) -> None:
    wheel = _wheel(
        tmp_path,
        selftest_direct_codegen_optimization_level=3,
    )

    with pytest.raises(
        ArtifactError,
        match=r"Arena execution contract.*direct codegen optimization.*plane",
    ):
        audit_wheel(wheel, mode="release", native_scan=False)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        pytest.param(
            {"compiled_model_kind": "other-compiled-model"},
            "compiled model kind is invalid",
            id="kind",
        ),
        pytest.param(
            {
                "model_compiler_version": _MODEL_COMPILER_VERSION + 1,
                "compiled_model_producer_compiler_version": (_MODEL_COMPILER_VERSION),
            },
            "compiler version does not match producer",
            id="compiler-version",
        ),
        pytest.param(
            {"compiled_model_producer_schema": (_LOCK["abis"]["compiled_model"] - 1)},
            "producer does not match release schema",
            id="producer-schema",
        ),
        pytest.param(
            {"compiled_model_producer_version": "9.9.9"},
            "producer does not match wheel version",
            id="producer-version",
        ),
        pytest.param(
            {"compiled_model_source_kind": "ufo"},
            "source is not built-in-sm",
            id="source-kind",
        ),
    ],
)
def test_wheel_selftest_compiled_model_matches_packaged_producer(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    wheel = _wheel(tmp_path, **override)

    with pytest.raises(ArtifactError, match=message):
        audit_wheel(wheel, mode="release", native_scan=False)


@pytest.mark.parametrize(
    ("source_name", "replacement"),
    [
        pytest.param(
            "pyamplicol/models/compiler.py",
            b"def compile_model():\n    return 'changed'\n",
            id="models",
        ),
        pytest.param(
            "pyamplicol/_internal/physics/rules.py",
            b"PROPAGATOR = 'changed'\n",
            id="physics",
        ),
        pytest.param(
            "pyamplicol/processes/core_syntax.py",
            b"CORE_SYNTAX_VERSION = 2\n",
            id="core-syntax",
        ),
        pytest.param(
            "pyamplicol/models/builtin/model.py",
            b"MODEL_NAME = 'changed'\n",
            id="built-in",
        ),
    ],
)
def test_wheel_selftest_accepts_source_edits_with_stable_runtime_contract(
    tmp_path: Path,
    source_name: str,
    replacement: bytes,
) -> None:
    wheel = _wheel(tmp_path, extra_files={source_name: replacement})

    assert (
        audit_wheel(wheel, mode="release", native_scan=False).version
        == _RELEASE_VERSION
    )


def test_wheel_selftest_digests_ignore_nested_model_sources(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        extra_files={
            "pyamplicol/models/nested/ignored.py": b"MODEL_COMPILER_VERSION = 99\n",
            "pyamplicol/_internal/physics/nested/ignored.py": b"PHYSICS = 99\n",
            "pyamplicol/models/builtin/nested/ignored.py": b"MODEL = 99\n",
        },
    )

    assert (
        audit_wheel(wheel, mode="release", native_scan=False).version
        == _RELEASE_VERSION
    )


@pytest.mark.parametrize(
    "schema_override",
    [
        {"producer_compiled_model_schema": _LOCK["abis"]["compiled_model"] - 1},
        {"model_compiled_model_schema": _LOCK["abis"]["compiled_model"] - 1},
    ],
)
def test_wheel_selftest_producer_and_model_metadata_match_release_schema(
    tmp_path: Path,
    schema_override: dict[str, int],
) -> None:
    expected = _LOCK["abis"]["compiled_model"]
    wheel = _wheel(tmp_path, **schema_override)

    with pytest.raises(
        ArtifactError,
        match=f"producer/model metadata.*schema {expected}",
    ):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_selftest_requires_the_complete_five_language_api_bundle(
    tmp_path: Path,
) -> None:
    wheel = _wheel(
        tmp_path,
        omitted_selftest_api_payload="API/rust/check_standalone.rs",
    )

    with pytest.raises(ArtifactError, match=r"five-language API.*API/rust"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_sdk_link_metadata_is_target_allowlisted(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, system_libraries=["python3.11"])
    with pytest.raises(ArtifactError, match="target-allowlisted"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_runtime_requirements_must_agree_with_release_contract(tmp_path: Path) -> None:
    missing = _wheel(
        tmp_path,
        requirements=[
            item for item in _DEFAULT_REQUIREMENTS if not item.startswith("colorama")
        ],
    )
    with pytest.raises(ArtifactError, match="inventory disagrees"):
        audit_wheel(missing, mode="release", native_scan=False)

    missing.unlink()
    incompatible = _wheel(
        tmp_path,
        requirements=[
            "numpy<2" if item.startswith("numpy") else item
            for item in _DEFAULT_REQUIREMENTS
        ],
    )
    with pytest.raises(ArtifactError, match="excludes locked version"):
        audit_wheel(incompatible, mode="release", native_scan=False)

    incompatible.unlink()
    ordinary_range = _wheel(
        tmp_path,
        requirements=[
            "numpy>=1.26" if item.startswith("numpy") else item
            for item in _DEFAULT_REQUIREMENTS
        ],
    )
    with pytest.raises(ArtifactError, match=r"pin numpy==2\.4\.6 exactly"):
        audit_wheel(ordinary_range, mode="release", native_scan=False)

    ordinary_range.unlink()
    symbolica_version = _LOCK["symbolica"]["python_version"]
    non_exact = _wheel(
        tmp_path,
        requirements=[
            f"symbolica>={symbolica_version}" if item.startswith("symbolica") else item
            for item in _DEFAULT_REQUIREMENTS
        ],
    )
    with pytest.raises(
        ArtifactError,
        match=rf"pin symbolica=={re.escape(symbolica_version)} exactly",
    ):
        audit_wheel(non_exact, mode="release", native_scan=False)

    non_exact.unlink()
    marker_string = _wheel(
        tmp_path,
        requirements=[
            *_DEFAULT_REQUIREMENTS,
            "unexpected==1; python_version == 'extra'",
        ],
    )
    with pytest.raises(ArtifactError, match="may not use environment markers"):
        audit_wheel(marker_string, mode="release", native_scan=False)

    marker_string.unlink()
    optional = _wheel(
        tmp_path,
        requirements=[*_DEFAULT_REQUIREMENTS, "build>=1; extra == 'test'"],
    )
    assert (
        audit_wheel(optional, mode="release", native_scan=False).version
        == _RELEASE_VERSION
    )


def test_wheel_uses_metadata_not_packaged_dependency_inventories(
    tmp_path: Path,
) -> None:
    wheel = _wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        assert not any(
            name.startswith("pyamplicol/assets/release/") for name in archive.namelist()
        )

    wheel.unlink()
    no_notice = _wheel(tmp_path, notice=None)
    with pytest.raises(ArtifactError, match="missing nonempty legal member"):
        audit_wheel(no_notice, mode="release", native_scan=False)

    no_notice.unlink()
    empty_notice = _wheel(tmp_path, notice=b"\n")
    with pytest.raises(ArtifactError, match="missing nonempty legal member"):
        audit_wheel(empty_notice, mode="release", native_scan=False)


def test_wheel_rejects_resources_installed_at_site_packages_root(
    tmp_path: Path,
) -> None:
    wheel = _wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["schemas/artifact-manifest-v3.schema.json"] = b"{}\n"
    record_name = next(name for name in files if name.endswith(".dist-info/RECORD"))
    rows = [
        [name, _record_hash(data), str(len(data))]
        for name, data in sorted(files.items())
        if name != record_name
    ]
    rows.append([record_name, "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[record_name] = record.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)

    with pytest.raises(ArtifactError, match="outside pyamplicol"):
        audit_wheel(wheel, mode="release", native_scan=False)


@pytest.mark.parametrize(
    "resource",
    [
        "pyamplicol/assets/schemas/README.md",
        "pyamplicol/assets/schemas/artifact-manifest-v3.schema.json",
        "pyamplicol/assets/schemas/runtime-physics-v1.schema.json",
    ],
)
def test_wheel_requires_all_package_owned_schema_resources(
    tmp_path: Path, resource: str
) -> None:
    wheel = _wheel(tmp_path, omitted_member=resource)
    with pytest.raises(ArtifactError, match=re.escape(resource)):
        audit_wheel(wheel, mode="release", native_scan=False)


@pytest.mark.parametrize(
    "resource",
    [
        "pyamplicol/_examples/data/pp_zjj_momenta.json",
        "pyamplicol/assets/api_templates/rust/Makefile",
        "pyamplicol/assets/api_templates/rust/check_standalone.rs",
        "pyamplicol/assets/api_templates/umami/umami.c",
        "pyamplicol/_sdk/rust/rusticol.rs",
    ],
)
def test_wheel_requires_primary_example_data_and_rust_api_templates(
    tmp_path: Path, resource: str
) -> None:
    wheel = _wheel(tmp_path, omitted_member=resource)
    with pytest.raises(ArtifactError, match=re.escape(resource)):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_rejects_an_unknown_top_level_root(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, extra_files={"docs/leak.txt": b"not wheel data\n"})
    with pytest.raises(ArtifactError, match=r"outside pyamplicol.*docs/leak\.txt"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_rejects_generated_sboms(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        extra_files={
            f"pyamplicol-{_RELEASE_VERSION}.dist-info/"
            "sboms/rusticol-python.cyclonedx.json": b"{}\n"
        },
    )
    with pytest.raises(ArtifactError, match="must not contain generated SBOMs"):
        audit_wheel(wheel, mode="release", native_scan=False)


def test_wheel_rejects_non_library_repair_root_files(tmp_path: Path) -> None:
    repair_secret = _wheel(
        tmp_path,
        extra_files={"pyamplicol.libs/token.txt": b"not a repaired library\n"},
    )
    with pytest.raises(ArtifactError, match="non-library member"):
        audit_wheel(repair_secret, mode="release", native_scan=False)


def test_wheel_requires_standard_license_metadata_and_allows_extra_notices(
    tmp_path: Path,
) -> None:
    supplementary = "licenses/SymJIT.txt"
    dist_info = f"pyamplicol-{_RELEASE_VERSION}.dist-info"
    wheel = _wheel(
        tmp_path,
        omitted_member=f"{dist_info}/licenses/{supplementary}",
    )
    with pytest.raises(ArtifactError, match=re.escape(supplementary)):
        audit_wheel(wheel, mode="release", native_scan=False)

    wheel.unlink()
    wheel = _wheel(
        tmp_path,
        license_files=tuple(item for item in _LEGAL_FILES if item != supplementary),
    )
    with pytest.raises(ArtifactError, match=r"omits required.*SymJIT"):
        audit_wheel(wheel, mode="release", native_scan=False)

    wheel.unlink()
    additional = "licenses/ADDITIONAL.txt"
    wheel = _wheel(
        tmp_path,
        license_files=(*_LEGAL_FILES, additional),
        extra_files={f"{dist_info}/licenses/{additional}": b"additional attribution\n"},
    )
    assert (
        audit_wheel(wheel, mode="release", native_scan=False).version
        == _RELEASE_VERSION
    )


def test_candidate_sdk_version_must_match_staged_package_exactly(
    tmp_path: Path,
    candidate_dependency_provenance: None,
) -> None:
    version = f"{_RELEASE_VERSION}.dev0+candidate.0123456789ab"
    wheel = _wheel(
        tmp_path,
        version=version,
        candidate=True,
        sdk_version="0.1.0",
    )
    with pytest.raises(ArtifactError, match="SDK version"):
        audit_wheel(wheel, mode="candidate", native_scan=False)


def test_release_sdist_identity_and_path_scan(tmp_path: Path) -> None:
    release = _sdist(tmp_path)
    assert audit_sdist(release, mode="release").version == _RELEASE_VERSION

    candidate = _sdist(
        tmp_path,
        version=f"{_RELEASE_VERSION}-dev.0+candidate.0123456789ab",
        candidate=True,
    )
    with pytest.raises(ArtifactError, match="candidate source distributions"):
        audit_sdist(candidate, mode="candidate")

    release.unlink()
    leaked = _sdist(tmp_path, manifest_path="/Users/build/parent-project")
    with pytest.raises(ArtifactError, match="non-relocatable"):
        audit_sdist(leaked, mode="release")

    leaked.unlink()
    direct = _sdist(
        tmp_path,
        project_requirement="numpy @ https://example.invalid/numpy.whl",
    )
    with pytest.raises(ArtifactError, match="direct URL or path"):
        audit_sdist(direct, mode="release")

    direct.unlink()
    fixture_leak = _sdist(
        tmp_path,
        extra_files={
            "tests/fixtures/reference/physics-v2.json": (
                b'{"capture_root":"/Users/build/private-checkout"}\n'
            )
        },
    )
    with pytest.raises(ArtifactError, match="non-relocatable"):
        audit_sdist(fixture_leak, mode="release")


def test_release_sdist_requires_the_minimal_build_marker(tmp_path: Path) -> None:
    missing = _sdist(
        tmp_path,
        omitted_member="src/pyamplicol/_build_info.json",
    )
    with pytest.raises(ArtifactError, match="missing required files"):
        audit_sdist(missing, mode="release")

    missing.unlink()
    invalid = _sdist(
        tmp_path,
        extra_files={
            "src/pyamplicol/_build_info.json": json.dumps(
                {
                    "schema_version": 1,
                    "publishable": True,
                    "selftest_fixture_bootstrap": False,
                    "source_revision": "not-a-revision",
                    "version": _RELEASE_VERSION,
                }
            ).encode()
        },
    )
    with pytest.raises(ArtifactError, match="release sdist build marker"):
        audit_sdist(invalid, mode="release")


@pytest.mark.parametrize(
    "missing_member",
    sorted(
        prepared_model_asset_members("src/pyamplicol/assets/prepared_models")
        - {"src/pyamplicol/assets/prepared_models/__init__.py"}
    ),
)
def test_sdist_requires_both_prepared_model_asset_pairs(
    tmp_path: Path,
    missing_member: str,
) -> None:
    sdist = _sdist(tmp_path, omitted_member=missing_member)

    with pytest.raises(ArtifactError, match="sdist is missing required files"):
        audit_sdist(sdist, mode="release")


def test_sdist_rejects_generic_or_extra_prepared_model_assets(tmp_path: Path) -> None:
    prefix = "src/pyamplicol/assets/prepared_models"
    sdist = _sdist(
        tmp_path,
        extra_files={
            f"{prefix}/{PREPARED_MODEL_ASSET_BASENAME}.metadata.json": b"{}\n"
        },
    )

    with pytest.raises(ArtifactError, match=r"prepared-model.*extra=.*built-in-sm"):
        audit_sdist(sdist, mode="release")


def test_sdist_validates_prepared_model_target_and_bundle_identity(
    tmp_path: Path,
) -> None:
    prefix = "src/pyamplicol/assets/prepared_models"
    metadata_name, metadata = _modified_prepared_metadata(
        prefix,
        "aarch64",
        target_values={"target_triple": "symjit-storage-v3-x86_64"},
    )
    wrong_target = _sdist(tmp_path, extra_files={metadata_name: metadata})
    with pytest.raises(ArtifactError, match="target class is invalid"):
        audit_sdist(wrong_target, mode="release")

    wrong_target.unlink()
    metadata_name, metadata = _modified_prepared_metadata(
        prefix,
        "x86_64",
        values={"bundle_sha256": "f" * 64},
    )
    wrong_digest = _sdist(tmp_path, extra_files={metadata_name: metadata})
    with pytest.raises(ArtifactError, match="bundle hash/size is invalid"):
        audit_sdist(wrong_digest, mode="release")


def test_sdist_rejects_candidate_prepared_model_assets(tmp_path: Path) -> None:
    prefix = "src/pyamplicol/assets/prepared_models"
    metadata_name, metadata_bytes = _modified_prepared_metadata(
        prefix,
        "aarch64",
    )
    metadata = json.loads(metadata_bytes)
    metadata["build_contract"]["mode"] = "candidate"
    sdist = _sdist(
        tmp_path,
        extra_files={
            metadata_name: (json.dumps(metadata, sort_keys=True) + "\n").encode()
        },
    )

    with pytest.raises(ArtifactError, match="build mode is invalid"):
        audit_sdist(sdist, mode="release")


def test_sdist_accepts_prepared_payload_compiler_edits(tmp_path: Path) -> None:
    sdist = _sdist(
        tmp_path,
        extra_files={"src/pyamplicol/evaluators/symbolica_compile.py": b"# drift\n"},
    )

    assert audit_sdist(sdist, mode="release").version == _RELEASE_VERSION


@pytest.mark.parametrize(
    "forbidden",
    [
        ".cargo/config.toml",
        "build_backend/python_lock.py",
        "dependencies/contributor-lock.toml",
        "dependencies/install_dependencies.py",
        "dependencies/patches/symbolica/fix.patch",
        "dependencies/python-runtime-lock.toml",
        "release_assets/prepared_models/built-in-sm-jit-o2-aarch64.metadata.json",
    ],
)
def test_sdist_rejects_contributor_dependency_material(
    tmp_path: Path, forbidden: str
) -> None:
    sdist = _sdist(
        tmp_path,
        extra_files={forbidden: b"contributor-only\n"},
    )
    with pytest.raises(ArtifactError, match="contributor-only dependency inputs"):
        audit_sdist(sdist, mode="release")


def test_sdist_rejects_extra_symjit_source_ceremony(tmp_path: Path) -> None:
    lock_member = "dependencies/release-lock.toml"
    lock_text = (ROOT / lock_member).read_text(encoding="utf-8")
    symjit = _LOCK["symjit"]
    contract = f'[symjit]\nversion = "{symjit["version"]}"\n'
    replacement = contract + 'source_url = "https://example.invalid/archive.tar.gz"\n'
    assert lock_text.count(contract) == 1
    sdist = _sdist(
        tmp_path,
        extra_files={lock_member: lock_text.replace(contract, replacement).encode()},
    )

    with pytest.raises(ArtifactError, match="published SymJIT version contract"):
        audit_sdist(sdist, mode="release")


def test_sdist_rejects_mismatched_symjit_cargo_source(tmp_path: Path) -> None:
    symjit = _LOCK["symjit"]
    cargo_lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    expected = (
        'name = "symjit"\n'
        f'version = "{symjit["version"]}"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"'
    )
    assert cargo_lock.count(expected) == 1
    mismatched = cargo_lock.replace(
        expected,
        expected.replace(
            "registry+https://github.com/rust-lang/crates.io-index",
            "git+https://example.invalid/symjit",
        ),
    )
    sdist = _sdist(
        tmp_path,
        extra_files={"Cargo.lock": mismatched.encode("utf-8")},
    )

    with pytest.raises(ArtifactError, match="published SymJIT dependency"):
        audit_sdist(sdist, mode="release")


@pytest.mark.parametrize(
    "missing_member",
    [
        "build_backend/_pyamplicol_build.py",
        "build_backend/sdk.py",
        "docs/user/installation.md",
        "examples/data/pp_zjj_momenta.json",
        "examples/python/typed_generation.py",
        "rust/crates/rusticol-capi/Cargo.toml",
        "rust/crates/rusticol-capi/include/rusticol.h",
        "rust/crates/rusticol-python/stubs/pyamplicol/_rusticol.pyi",
        "schemas/README.md",
        "schemas/artifact-manifest-v3.schema.json",
        "src/pyamplicol/assets/selftest/portable-64le/expected.json",
        "src/pyamplicol/assets/selftest/portable-64le/artifact/artifact.json",
        "tools/release/prepare_selftest_fixture.py",
        "src/pyamplicol/assets/api_templates/rust/check_standalone.rs",
        "tests/integration/test_examples.py",
        "tools/release/test_deployment.py",
    ],
)
def test_sdist_requires_nested_release_content(
    tmp_path: Path, missing_member: str
) -> None:
    sdist = _sdist(tmp_path, omitted_member=missing_member)
    with pytest.raises(ArtifactError, match=re.escape(missing_member)):
        audit_sdist(sdist, mode="release")
