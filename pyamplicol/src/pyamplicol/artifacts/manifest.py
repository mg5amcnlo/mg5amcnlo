# SPDX-License-Identifier: 0BSD
"""Typed schema-v3 manifest loading and payload validation."""

from __future__ import annotations

import importlib
import json
import re
import struct
import sys
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from pyamplicol._internal.versions import (
    KNOWN_EVALUATOR_RUNTIME_CAPABILITIES,
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    verify_native_module,
)
from pyamplicol.api.errors import ArtifactError, CompatibilityError

from .security import (
    confined_path,
    executable_bit,
    normalize_relative_path,
    sha256_file,
)

MANIFEST_NAME = "artifact.json"
SCHEMA_VERSION = 3
ARTIFACT_IDENTITY_EXTENSION = "artifact_identity"
ARTIFACT_IDENTITY_KIND = "pyamplicol-runtime-payload-identity"
ARTIFACT_IDENTITY_SCHEMA_VERSION = 1
PORTABLE_64LE_TARGET = "portable-64le"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_GIT_REVISION = re.compile(r"^[a-f0-9]{40}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+,~-]{0,254}$")
_CPU_FEATURE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_SUPPORTED_ARTIFACT_TARGETS = frozenset(
    {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
    }
)
_TARGET_SPECIFIC_EVALUATOR_RUNTIME_CAPABILITIES = frozenset(
    {
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    }
)
_EXECUTABLE_ROLES = {"api-source", "api-build-file", "evaluator-state"}
_PAYLOAD_ROLES = {
    "configuration-requested",
    "configuration-effective",
    "compiled-model",
    "runtime-physics",
    "evaluator-manifest",
    "evaluator-state",
    "model-parameters",
    "validation-momenta",
    "structural-source-proof",
    "api-source",
    "api-build-file",
    "sdk-metadata",
    "process-groups",
}
_RUNTIME_IDENTITY_PAYLOAD_ROLES = frozenset(
    {
        "compiled-model",
        "evaluator-manifest",
        "evaluator-state",
        "model-parameters",
        "runtime-physics",
    }
)


def artifact_identity_extension() -> dict[str, object]:
    """Return the sole supported artifact-ID interpretation contract."""

    return {
        "kind": ARTIFACT_IDENTITY_KIND,
        "schema_version": ARTIFACT_IDENTITY_SCHEMA_VERSION,
    }


def _validate_artifact_identity_extension(
    extensions: Mapping[str, object],
) -> None:
    raw = extensions.get(ARTIFACT_IDENTITY_EXTENSION)
    if raw is None:
        raise CompatibilityError(
            "artifact predates the required runtime-payload identity contract; "
            "regenerate it with the current pyAmpliCol"
        )
    policy = _mapping(raw, f"extensions.{ARTIFACT_IDENTITY_EXTENSION}")
    _keys(
        policy,
        f"extensions.{ARTIFACT_IDENTITY_EXTENSION}",
        required={"kind", "schema_version"},
    )
    if (
        policy.get("kind") != ARTIFACT_IDENTITY_KIND
        or policy.get("schema_version") != ARTIFACT_IDENTITY_SCHEMA_VERSION
    ):
        raise CompatibilityError(
            "artifact uses an unsupported artifact identity contract; "
            "regenerate it with the current pyAmpliCol"
        )


def _keys(
    value: Mapping[str, object],
    context: str,
    *,
    required: Set[str],
    optional: Set[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ArtifactError(
            f"{context} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ArtifactError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ArtifactError(f"{context} keys must be strings")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} must be an array")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{context} must be a non-empty string")
    return value


def _public_id(value: object, context: str) -> str:
    identifier = _string(value, context)
    if _PUBLIC_ID.fullmatch(identifier) is None:
        raise ArtifactError(f"{context} is not a valid public identifier")
    return identifier


def _string_array(value: object, context: str) -> tuple[str, ...]:
    items = tuple(_sequence(value, context))
    result = tuple(
        _string(item, f"{context}[{index}]") for index, item in enumerate(items)
    )
    if len(set(result)) != len(result):
        raise ArtifactError(f"{context} must not contain duplicates")
    return result


def _runtime_capabilities(value: object, context: str) -> tuple[str, ...]:
    result = _string_array(value, context)
    if not result:
        raise ArtifactError(f"{context} must contain at least one capability")
    if result != tuple(sorted(result)):
        raise ArtifactError(f"{context} must be sorted")
    unknown = set(result) - KNOWN_EVALUATOR_RUNTIME_CAPABILITIES
    if unknown:
        raise ArtifactError(
            f"{context} contains unsupported capabilities: "
            + ", ".join(sorted(unknown))
        )
    return result


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if _SHA256.fullmatch(digest) is None:
        raise ArtifactError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactError(f"{context} must be an integer >= {minimum}")
    return value


def _target(value: object, context: str) -> Mapping[str, object]:
    raw = _mapping(value, context)
    _keys(raw, context, required={"triple", "cpu_features"})
    cpu_features = _string_array(raw.get("cpu_features"), f"{context}.cpu_features")
    if cpu_features != tuple(sorted(cpu_features)):
        raise ArtifactError(f"{context}.cpu_features must be sorted")
    invalid = tuple(
        feature
        for feature in cpu_features
        if _CPU_FEATURE_ID.fullmatch(feature) is None
    )
    if invalid:
        raise ArtifactError(
            f"{context}.cpu_features contains non-canonical IDs: "
            + ", ".join(repr(feature) for feature in invalid)
        )
    return MappingProxyType(
        {
            "triple": _string(raw.get("triple"), f"{context}.triple"),
            "cpu_features": cpu_features,
        }
    )


def _producer(value: object) -> Mapping[str, object]:
    raw = _mapping(value, "producer")
    _keys(
        raw,
        "producer",
        required={"distribution", "version", "versions", "target"},
        optional={"git_revision", "native_build_inputs_sha256"},
    )
    if raw.get("distribution") != "pyamplicol":
        raise ArtifactError("producer.distribution must be 'pyamplicol'")
    versions = _mapping(raw.get("versions"), "producer.versions")
    version_fields = {
        "python_api",
        "toml",
        "compiled_model",
        "process_artifact",
        "runtime_physics",
        "symbolica_serialization",
        "c_abi",
    }
    _keys(versions, "producer.versions", required=version_fields)
    normalized_versions: dict[str, object] = {
        name: _integer(versions.get(name), f"producer.versions.{name}", minimum=1)
        for name in version_fields - {"symbolica_serialization"}
    }
    if normalized_versions["process_artifact"] != 3:
        raise ArtifactError("producer.versions.process_artifact must be 3")
    if normalized_versions["runtime_physics"] != 1:
        raise ArtifactError("producer.versions.runtime_physics must be 1")
    normalized_versions["symbolica_serialization"] = _string(
        versions.get("symbolica_serialization"),
        "producer.versions.symbolica_serialization",
    )
    result: dict[str, object] = {
        "distribution": "pyamplicol",
        "version": _string(raw.get("version"), "producer.version"),
        "versions": MappingProxyType(normalized_versions),
        "target": _target(raw.get("target"), "producer.target"),
    }
    revision = raw.get("git_revision")
    if revision is not None:
        revision_value = _string(revision, "producer.git_revision")
        if _GIT_REVISION.fullmatch(revision_value) is None:
            raise ArtifactError("producer.git_revision must be a Git SHA-1")
        result["git_revision"] = revision_value
    native_inputs = raw.get("native_build_inputs_sha256")
    if native_inputs is not None:
        native_inputs_value = _string(
            native_inputs,
            "producer.native_build_inputs_sha256",
        )
        if _SHA256.fullmatch(native_inputs_value) is None:
            raise ArtifactError("producer.native_build_inputs_sha256 must be a SHA-256")
        result["native_build_inputs_sha256"] = native_inputs_value
    return MappingProxyType(result)


def _model(value: object) -> Mapping[str, object]:
    raw = _mapping(value, "model")
    _keys(
        raw,
        "model",
        required={"name", "source_kind", "content_sha256", "compiled_schema_version"},
        optional={"restriction"},
    )
    source_kind = raw.get("source_kind")
    if source_kind not in {
        "built-in-sm",
        "built-in-sm-heft",
        "ufo",
        "ufo-json",
        "compiled-model",
    }:
        raise ArtifactError(f"unsupported model.source_kind {source_kind!r}")
    result: dict[str, object] = {
        "name": _string(raw.get("name"), "model.name"),
        "source_kind": source_kind,
        "content_sha256": _sha256(raw.get("content_sha256"), "model.content_sha256"),
        "compiled_schema_version": _integer(
            raw.get("compiled_schema_version"),
            "model.compiled_schema_version",
            minimum=1,
        ),
    }
    if "restriction" in raw:
        restriction = raw.get("restriction")
        result["restriction"] = (
            None if restriction is None else _string(restriction, "model.restriction")
        )
    return MappingProxyType(result)


def _configuration(value: object) -> Mapping[str, object]:
    raw = _mapping(value, "configuration")
    _keys(
        raw,
        "configuration",
        required={
            "toml_schema_version",
            "requested_path",
            "effective_path",
            "adjustments",
        },
    )
    if raw.get("toml_schema_version") != 1:
        raise ArtifactError("configuration.toml_schema_version must be 1")
    adjustments: list[Mapping[str, object]] = []
    for index, item in enumerate(
        _sequence(raw.get("adjustments"), "configuration.adjustments")
    ):
        adjustment = _mapping(item, f"configuration.adjustments[{index}]")
        _keys(
            adjustment,
            f"configuration.adjustments[{index}]",
            required={"path", "reason"},
        )
        adjustments.append(
            MappingProxyType(
                {
                    "path": _string(
                        adjustment.get("path"),
                        f"configuration.adjustments[{index}].path",
                    ),
                    "reason": _string(
                        adjustment.get("reason"),
                        f"configuration.adjustments[{index}].reason",
                    ),
                }
            )
        )
    return MappingProxyType(
        {
            "toml_schema_version": 1,
            "requested_path": normalize_relative_path(
                _string(raw.get("requested_path"), "configuration.requested_path")
            ),
            "effective_path": normalize_relative_path(
                _string(raw.get("effective_path"), "configuration.effective_path")
            ),
            "adjustments": tuple(adjustments),
        }
    )


def _process(value: object, index: int) -> Mapping[str, object]:
    context = f"processes[{index}]"
    raw = _mapping(value, context)
    _keys(
        raw,
        context,
        required={
            "id",
            "expression",
            "color_accuracy",
            "external_pdgs",
            "physics_path",
            "required_runtime_capabilities",
            "aliases",
        },
    )
    accuracy = raw.get("color_accuracy")
    if accuracy not in {"lc", "nlc", "full"}:
        raise ArtifactError(f"{context}.color_accuracy is invalid")
    pdgs = tuple(_sequence(raw.get("external_pdgs"), f"{context}.external_pdgs"))
    if len(pdgs) < 3 or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in pdgs
    ):
        raise ArtifactError(
            f"{context}.external_pdgs must contain at least three integers"
        )
    process_initial_count = _process_initial_count(
        _string(raw.get("expression"), f"{context}.expression"),
        context,
        len(pdgs),
    )
    aliases: list[Mapping[str, object]] = []
    for alias_index, item in enumerate(
        _sequence(raw.get("aliases"), f"{context}.aliases")
    ):
        alias_context = f"{context}.aliases[{alias_index}]"
        alias = _mapping(item, alias_context)
        _keys(
            alias,
            alias_context,
            required={"id", "expression", "external_pdgs", "external_permutation"},
        )
        permutation = tuple(
            _sequence(
                alias.get("external_permutation"),
                f"{alias_context}.external_permutation",
            )
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in permutation
        ):
            raise ArtifactError(f"{alias_context}.external_permutation is invalid")
        if len(permutation) != len(pdgs) or sorted(permutation) != list(
            range(len(pdgs))
        ):
            raise ArtifactError(
                f"{alias_context}.external_permutation must be a complete "
                f"permutation of {len(pdgs)} external particles"
            )
        alias_initial_count = _process_initial_count(
            _string(alias.get("expression"), f"{alias_context}.expression"),
            alias_context,
            len(pdgs),
        )
        if alias_initial_count != process_initial_count or any(
            (representative_index < process_initial_count)
            != (public_index < process_initial_count)
            for representative_index, public_index in enumerate(permutation)
        ):
            raise ArtifactError(
                f"{alias_context}.external_permutation may only permute "
                "particles within the incoming and outgoing sides"
            )
        alias_pdgs = tuple(
            _sequence(alias.get("external_pdgs"), f"{alias_context}.external_pdgs")
        )
        if len(alias_pdgs) != len(pdgs) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in alias_pdgs
        ):
            raise ArtifactError(
                f"{alias_context}.external_pdgs must contain {len(pdgs)} integers"
            )
        expected_pdgs: list[int | None] = [None] * len(pdgs)
        for representative_index, alias_index in enumerate(permutation):
            expected_pdgs[alias_index] = int(pdgs[representative_index])
        if tuple(expected_pdgs) != alias_pdgs:
            raise ArtifactError(
                f"{alias_context}.external_pdgs does not match external_permutation"
            )
        aliases.append(
            MappingProxyType(
                {
                    "id": _public_id(alias.get("id"), f"{alias_context}.id"),
                    "expression": _string(
                        alias.get("expression"), f"{alias_context}.expression"
                    ),
                    "external_pdgs": alias_pdgs,
                    "external_permutation": permutation,
                }
            )
        )
    return MappingProxyType(
        {
            "id": _public_id(raw.get("id"), f"{context}.id"),
            "expression": _string(raw.get("expression"), f"{context}.expression"),
            "color_accuracy": accuracy,
            "external_pdgs": pdgs,
            "physics_path": normalize_relative_path(
                _string(raw.get("physics_path"), f"{context}.physics_path")
            ),
            "required_runtime_capabilities": _runtime_capabilities(
                raw.get("required_runtime_capabilities"),
                f"{context}.required_runtime_capabilities",
            ),
            "aliases": tuple(aliases),
        }
    )


def _process_initial_count(expression: str, context: str, particle_count: int) -> int:
    sides = expression.split(">")
    if len(sides) != 2:
        raise ArtifactError(
            f"{context}.expression must contain exactly one '>' separator"
        )
    initial_count = len(sides[0].split())
    final_count = len(sides[1].split())
    if (
        initial_count < 1
        or final_count < 1
        or initial_count + final_count != particle_count
    ):
        raise ArtifactError(
            f"{context}.expression particle count does not match external_pdgs"
        )
    return initial_count


def _runtime(value: object) -> Mapping[str, object]:
    raw = _mapping(value, "runtime")
    _keys(
        raw,
        "runtime",
        required={
            "engine",
            "engine_version",
            "evaluator_manifest_path",
            "api_bundle_path",
            "required_runtime_capabilities",
        },
    )
    if raw.get("engine") != "rusticol":
        raise ArtifactError("runtime.engine must be 'rusticol'")
    api_path = raw.get("api_bundle_path")
    return MappingProxyType(
        {
            "engine": "rusticol",
            "engine_version": _string(
                raw.get("engine_version"), "runtime.engine_version"
            ),
            "evaluator_manifest_path": normalize_relative_path(
                _string(
                    raw.get("evaluator_manifest_path"),
                    "runtime.evaluator_manifest_path",
                )
            ),
            "api_bundle_path": None
            if api_path is None
            else normalize_relative_path(_string(api_path, "runtime.api_bundle_path")),
            "required_runtime_capabilities": _runtime_capabilities(
                raw.get("required_runtime_capabilities"),
                "runtime.required_runtime_capabilities",
            ),
        }
    )


def _dependency(value: object, index: int) -> Mapping[str, object]:
    context = f"dependencies[{index}]"
    raw = _mapping(value, context)
    _keys(
        raw,
        context,
        required={"name", "version", "source", "license"},
        optional={"content_sha256", "revision", "patch_sha256"},
    )
    result: dict[str, object] = {
        name: _string(raw.get(name), f"{context}.{name}")
        for name in ("name", "version", "source", "license")
    }
    for name in ("content_sha256", "patch_sha256"):
        if name in raw:
            result[name] = _sha256(raw.get(name), f"{context}.{name}")
    if "revision" in raw:
        result["revision"] = _string(raw.get("revision"), f"{context}.revision")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class PayloadRecord:
    path: str
    role: str
    media_type: str
    size_bytes: int
    sha256: str
    executable: bool
    target: Mapping[str, object] | None = None
    process_id: str | None = None

    @classmethod
    def from_mapping(cls, value: object, index: int) -> PayloadRecord:
        raw = _mapping(value, f"payloads[{index}]")
        _keys(
            raw,
            f"payloads[{index}]",
            required={
                "path",
                "role",
                "media_type",
                "size_bytes",
                "sha256",
                "executable",
            },
            optional={"target", "process_id"},
        )
        executable = raw.get("executable")
        if not isinstance(executable, bool):
            raise ArtifactError(f"payloads[{index}].executable must be a boolean")
        target = raw.get("target")
        process_id = raw.get("process_id")
        role = _string(raw.get("role"), f"payloads[{index}].role")
        if role not in _PAYLOAD_ROLES:
            raise ArtifactError(f"payloads[{index}].role is unsupported: {role!r}")
        return cls(
            path=normalize_relative_path(
                _string(raw.get("path"), f"payloads[{index}].path")
            ),
            role=role,
            media_type=_string(raw.get("media_type"), f"payloads[{index}].media_type"),
            size_bytes=_integer(raw.get("size_bytes"), f"payloads[{index}].size_bytes"),
            sha256=_sha256(raw.get("sha256"), f"payloads[{index}].sha256"),
            executable=executable,
            target=(
                _target(target, f"payloads[{index}].target")
                if target is not None
                else None
            ),
            process_id=(
                _public_id(process_id, f"payloads[{index}].process_id")
                if process_id is not None
                else None
            ),
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path": self.path,
            "role": self.role,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "executable": self.executable,
        }
        if self.target is not None:
            result["target"] = dict(self.target)
        if self.process_id is not None:
            result["process_id"] = self.process_id
        return result


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    root: Path
    kind: Literal["pyamplicol-process", "pyamplicol-process-set"]
    artifact_id: str
    created_utc: str
    producer: Mapping[str, object]
    model: Mapping[str, object]
    configuration: Mapping[str, object]
    processes: tuple[Mapping[str, object], ...]
    default_process_id: str | None
    runtime: Mapping[str, object]
    payloads: tuple[PayloadRecord, ...]
    dependencies: tuple[Mapping[str, object], ...]
    extensions: Mapping[str, object]

    @classmethod
    def from_mapping(cls, root: Path, value: object) -> ArtifactManifest:
        raw = _mapping(value, "artifact manifest")
        schema = raw.get("schema_version")
        if schema in (1, 2):
            raise CompatibilityError(
                f"process-artifact schema v{schema} is unsupported; regenerate "
                "the process with pyAmpliCol 0.1.0 or newer"
            )
        if schema != SCHEMA_VERSION:
            raise CompatibilityError(
                f"unsupported process-artifact schema {schema!r}; expected v3"
            )
        if "extensions" not in raw:
            raise CompatibilityError(
                "artifact predates the required runtime-payload identity contract; "
                "regenerate it with the current pyAmpliCol"
            )
        _keys(
            raw,
            "artifact manifest",
            required={
                "schema_version",
                "kind",
                "artifact_id",
                "created_utc",
                "producer",
                "model",
                "configuration",
                "processes",
                "runtime",
                "payloads",
                "dependencies",
                "extensions",
            },
            optional={"default_process_id"},
        )
        kind = raw.get("kind")
        if kind not in ("pyamplicol-process", "pyamplicol-process-set"):
            raise ArtifactError(f"unsupported artifact kind {kind!r}")
        process_values = _sequence(raw.get("processes"), "processes")
        if not process_values:
            raise ArtifactError("processes must contain at least one process")
        payload_values = _sequence(raw.get("payloads"), "payloads")
        if not payload_values:
            raise ArtifactError("payloads must contain at least one payload")
        dependency_values = _sequence(raw.get("dependencies"), "dependencies")
        default_process = raw.get("default_process_id")
        extensions = _mapping(raw.get("extensions"), "extensions")
        _validate_artifact_identity_extension(extensions)
        created_utc = _string(raw.get("created_utc"), "created_utc")
        try:
            datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ArtifactError("created_utc must be an ISO-8601 date-time") from exc
        processes = tuple(
            _process(item, index) for index, item in enumerate(process_values)
        )
        process_ids = tuple(str(item["id"]) for item in processes)
        if len(set(process_ids)) != len(process_ids):
            raise ArtifactError("process IDs must be unique")
        default_process_id = (
            _public_id(default_process, "default_process_id")
            if default_process is not None
            else None
        )
        if default_process_id is not None and default_process_id not in process_ids:
            raise ArtifactError("default_process_id does not identify a process")
        runtime = _runtime(raw.get("runtime"))
        process_capabilities = {
            str(capability)
            for process in processes
            for capability in _sequence(
                process["required_runtime_capabilities"],
                "process.required_runtime_capabilities",
            )
        }
        runtime_capabilities = set(
            _sequence(
                runtime["required_runtime_capabilities"],
                "runtime.required_runtime_capabilities",
            )
        )
        if runtime_capabilities != process_capabilities:
            raise ArtifactError(
                "runtime.required_runtime_capabilities must equal the union of "
                "process capability declarations"
            )
        return cls(
            root=root.resolve(strict=True),
            kind=kind,
            artifact_id=_sha256(raw.get("artifact_id"), "artifact_id"),
            created_utc=created_utc,
            producer=_producer(raw.get("producer")),
            model=_model(raw.get("model")),
            configuration=_configuration(raw.get("configuration")),
            processes=processes,
            default_process_id=default_process_id,
            runtime=runtime,
            payloads=tuple(
                PayloadRecord.from_mapping(item, index)
                for index, item in enumerate(payload_values)
            ),
            dependencies=tuple(
                _dependency(item, index) for index, item in enumerate(dependency_values)
            ),
            extensions=MappingProxyType(dict(extensions)),
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "created_utc": self.created_utc,
            "producer": _plain(self.producer),
            "model": _plain(self.model),
            "configuration": _plain(self.configuration),
            "processes": _plain(self.processes),
            "runtime": _plain(self.runtime),
            "payloads": [payload.as_dict() for payload in self.payloads],
            "dependencies": _plain(self.dependencies),
            "extensions": _plain(self.extensions),
        }
        if self.default_process_id is not None:
            result["default_process_id"] = self.default_process_id
        return result


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def canonical_manifest_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _artifact_identity_content(value: Mapping[str, object]) -> dict[str, object]:
    payloads = _sequence(value.get("payloads"), "payloads")
    runtime_records: list[dict[str, object]] = []
    for index, item in enumerate(payloads):
        record = _mapping(item, f"payloads[{index}]")
        role = _string(record.get("role"), f"payloads[{index}].role")
        if role in _RUNTIME_IDENTITY_PAYLOAD_ROLES:
            runtime_records.append(cast(dict[str, object], _plain(record)))
    runtime_records.sort(key=lambda record: str(record.get("path", "")))
    return {
        "kind": "pyamplicol-runtime-payload-identity",
        "schema_version": 1,
        "payloads": runtime_records,
    }


def compute_artifact_id(value: Mapping[str, object]) -> str:
    import hashlib

    content = _artifact_identity_content(value)
    return hashlib.sha256(canonical_manifest_bytes(content)).hexdigest()


def _validate_target_compatibility(
    manifest: ArtifactManifest,
    *,
    expected_target: str | None = None,
) -> None:
    producer_target = _target(manifest.producer.get("target"), "producer.target")
    target_triple = _string(producer_target.get("triple"), "producer.target.triple")
    is_portable_64le = target_triple == PORTABLE_64LE_TARGET
    if (
        expected_target is not None
        and not is_portable_64le
        and target_triple != expected_target
    ):
        raise CompatibilityError(
            f"artifact target {target_triple!r} is incompatible with "
            f"runtime target {expected_target!r}"
        )
    seen: set[str] = set()
    for record in manifest.payloads:
        if record.path in seen:
            raise ArtifactError(f"duplicate artifact payload path: {record.path}")
        seen.add(record.path)
        if record.role == "evaluator-state" and record.target is None:
            raise ArtifactError(
                f"evaluator-state payload has no target metadata: {record.path}"
            )
        if record.target is not None and dict(record.target) != dict(producer_target):
            raise CompatibilityError(
                f"payload {record.path} target metadata differs from producer target"
            )

    required_features = cast(tuple[str, ...], producer_target["cpu_features"])
    if is_portable_64le:
        if required_features:
            raise CompatibilityError(
                f"{PORTABLE_64LE_TARGET} artifacts may not require CPU features"
            )
        runtime_capabilities = set(
            cast(tuple[str, ...], manifest.runtime["required_runtime_capabilities"])
        )
        target_specific_capabilities = sorted(
            runtime_capabilities & _TARGET_SPECIFIC_EVALUATOR_RUNTIME_CAPABILITIES
        )
        if target_specific_capabilities:
            raise CompatibilityError(
                f"{PORTABLE_64LE_TARGET} artifact declares target-specific "
                "evaluator capabilities: "
                + ", ".join(target_specific_capabilities)
            )
        if (
            expected_target is not None
            and expected_target not in _SUPPORTED_ARTIFACT_TARGETS
        ):
            raise CompatibilityError(
                f"artifact target {PORTABLE_64LE_TARGET!r} is incompatible with "
                f"runtime target {expected_target!r}"
            )
        if sys.byteorder != "little" or struct.calcsize("P") * 8 != 64:
            raise CompatibilityError(
                f"{PORTABLE_64LE_TARGET} artifacts require a 64-bit "
                "little-endian runtime"
            )
        return
    if required_features:
        runtime_triple, runtime_features = _runtime_target_metadata()
        if target_triple != runtime_triple:
            raise CompatibilityError(
                f"artifact target {target_triple!r} is incompatible with "
                f"runtime target {runtime_triple!r}"
            )
        unavailable = sorted(set(required_features) - set(runtime_features))
        if unavailable:
            raise CompatibilityError(
                "artifact requires unavailable CPU features: " + ", ".join(unavailable)
            )


def validate_payloads(
    manifest: ArtifactManifest,
    *,
    expected_target: str | None = None,
) -> None:
    if compute_artifact_id(manifest.as_dict()) != manifest.artifact_id:
        raise ArtifactError("artifact manifest identity digest mismatch")
    _validate_target_compatibility(manifest, expected_target=expected_target)
    seen = {record.path for record in manifest.payloads}
    for record in manifest.payloads:
        path = confined_path(manifest.root, record.path)
        if path.stat().st_size != record.size_bytes:
            raise ArtifactError(f"artifact payload size mismatch: {record.path}")
        if sha256_file(path) != record.sha256:
            raise ArtifactError(f"artifact payload digest mismatch: {record.path}")
        actual_executable = executable_bit(path)
        if actual_executable != record.executable:
            raise ArtifactError(
                f"artifact payload executable-bit mismatch: {record.path}"
            )
        if record.executable and record.role not in _EXECUTABLE_ROLES:
            raise ArtifactError(
                f"artifact payload role {record.role!r} may not be executable"
            )

    expected_files = seen | {MANIFEST_NAME}
    for path in manifest.root.rglob("*"):
        relative = path.relative_to(manifest.root).as_posix()
        if path.is_symlink():
            raise ArtifactError(
                f"artifact tree contains an undeclared symlink: {relative}"
            )
        if not path.is_file() or relative in expected_files:
            continue
        if executable_bit(path):
            raise ArtifactError(
                f"artifact tree contains an undeclared executable: {relative}"
            )


def _runtime_target_metadata() -> tuple[str, tuple[str, ...]]:
    try:
        rusticol = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(rusticol)
        info = rusticol.target_info()
        triple = str(info.triple)
        features = tuple(str(feature) for feature in info.cpu_features)
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        raise CompatibilityError(
            "artifact CPU-feature requirements cannot be checked because the "
            "Rusticol native runtime is unavailable"
        ) from error
    if not triple:
        raise CompatibilityError("Rusticol returned an empty runtime target triple")
    if triple not in _SUPPORTED_ARTIFACT_TARGETS:
        raise CompatibilityError(
            f"Rusticol process artifacts are not supported on target {triple!r}"
        )
    if features != tuple(sorted(set(features))) or any(
        _CPU_FEATURE_ID.fullmatch(feature) is None for feature in features
    ):
        raise CompatibilityError(
            "Rusticol returned invalid runtime CPU-feature metadata"
        )
    return triple, features


def load_manifest(
    artifact: str | Path,
    *,
    expected_target: str | None = None,
    verify_payloads: bool = False,
) -> ArtifactManifest:
    root = Path(artifact).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ArtifactError(f"artifact root is not a directory: {root}")
    manifest_path = confined_path(root, MANIFEST_NAME)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid artifact manifest JSON: {exc}") from exc
    manifest = ArtifactManifest.from_mapping(root, raw)
    if verify_payloads:
        validate_payloads(manifest, expected_target=expected_target)
    else:
        _validate_target_compatibility(manifest, expected_target=expected_target)
    return manifest


__all__ = [
    "ARTIFACT_IDENTITY_EXTENSION",
    "ARTIFACT_IDENTITY_KIND",
    "ARTIFACT_IDENTITY_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "PORTABLE_64LE_TARGET",
    "SCHEMA_VERSION",
    "ArtifactManifest",
    "PayloadRecord",
    "artifact_identity_extension",
    "canonical_manifest_bytes",
    "compute_artifact_id",
    "load_manifest",
    "validate_payloads",
]
