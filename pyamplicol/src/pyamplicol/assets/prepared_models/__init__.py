# SPDX-License-Identifier: 0BSD
"""Discovery and validation for wheel-owned prepared model bundles."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pyamplicol.models.prepared import PreparedModelBundle

BUILTIN_SM_JIT_O2 = "built-in-sm-jit-o2"
BUILTIN_SM_HEFT_JIT_O2 = "built-in-sm-heft-jit-o2"
_PACKAGED_ARCHITECTURES = ("aarch64", "x86_64")
_MODEL_SOURCE_BY_ID = {
    BUILTIN_SM_JIT_O2: "built-in-sm",
    BUILTIN_SM_HEFT_JIT_O2: "built-in-sm-heft",
}
_KNOWN_MODELS = tuple(_MODEL_SOURCE_BY_ID)
_METADATA_KEYS = frozenset(
    {
        "backend",
        "bundle",
        "bundle_sha256",
        "bundle_size",
        "build_contract",
        "dependencies",
        "eager_kernel_abi",
        "id",
        "jit_optimization_level",
        "kernel_count",
        "model",
        "prepared_model_bundle_schema",
        "producer",
        "schema_version",
        "target",
    }
)


class PackagedPreparedModelError(RuntimeError):
    """Raised when an installed prepared-model resource is absent or stale."""


def available_prepared_models() -> tuple[str, ...]:
    """Return stable identifiers for prepared models shipped in this wheel."""

    return _KNOWN_MODELS


@contextmanager
def packaged_prepared_model_path(identifier: str) -> Iterator[Path]:
    """Materialize and validate one packaged bundle for path-based consumers."""

    if identifier not in _KNOWN_MODELS:
        choices = ", ".join(_KNOWN_MODELS)
        raise PackagedPreparedModelError(
            f"unknown packaged prepared model {identifier!r}; available: {choices}"
        )
    from pyamplicol.models.prepared_target import (
        PreparedTargetError,
        canonical_architecture,
    )

    try:
        architecture = canonical_architecture()
    except PreparedTargetError as error:
        raise PackagedPreparedModelError(str(error)) from error
    if architecture not in _PACKAGED_ARCHITECTURES:  # pragma: no cover
        raise PackagedPreparedModelError(
            f"no wheel-owned prepared model supports {architecture!r}"
        )
    metadata_name, bundle_name = _asset_names(identifier, architecture)
    root = resources.files(__package__)
    metadata_resource = root.joinpath(metadata_name)
    bundle_resource = root.joinpath(bundle_name)
    try:
        metadata = _metadata(
            json.loads(metadata_resource.read_text(encoding="utf-8")),
            identifier=identifier,
            bundle_name=bundle_name,
        )
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise PackagedPreparedModelError(
            f"cannot read packaged prepared-model metadata: {error}"
        ) from error
    with resources.as_file(bundle_resource) as path:
        bundle_path = Path(path)
        if not bundle_path.is_file() or bundle_path.is_symlink():
            raise PackagedPreparedModelError(
                "packaged prepared-model bundle is not a regular file"
            )
        data = bundle_path.read_bytes()
        if metadata.get("bundle_size") != len(data):
            raise PackagedPreparedModelError(
                "packaged prepared-model bundle size does not match metadata"
            )
        if metadata.get("bundle_sha256") != hashlib.sha256(data).hexdigest():
            raise PackagedPreparedModelError(
                "packaged prepared-model bundle SHA-256 does not match metadata"
            )
        _validate_bundle(
            bundle_path,
            metadata,
            identifier=identifier,
            architecture=architecture,
        )
        yield bundle_path


@contextmanager
def open_packaged_prepared_model(
    identifier: str = BUILTIN_SM_JIT_O2,
) -> Iterator[PreparedModelBundle]:
    """Open a validated bundle while keeping zip-installed resources alive."""

    from pyamplicol.models.prepared import load_prepared_model_bundle

    with packaged_prepared_model_path(identifier) as path:
        yield load_prepared_model_bundle(path)


def materialize_packaged_prepared_model(
    identifier: str = BUILTIN_SM_JIT_O2,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Return a stable path for a validated wheel-owned prepared model.

    Installed wheels normally expose resources as ordinary files, but the
    resource API also permits transient extraction from archive importers.
    Eager generation retains the bundle path while it copies referenced
    kernels into the process artifact, so materialize one content-addressed
    cache entry instead of leaking a temporary resource path.
    """

    with packaged_prepared_model_path(identifier) as source:
        bundle_name = source.name
        data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if cache_dir is None:
        from platformdirs import user_cache_path

        root = user_cache_path("pyamplicol") / "prepared_models" / digest
    else:
        root = Path(cache_dir).expanduser().resolve(strict=False) / digest
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise PackagedPreparedModelError(
            "packaged prepared-model cache must be a regular directory"
        )
    output = root / bundle_name
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise PackagedPreparedModelError(
                "packaged prepared-model cache entry must be a regular file"
            )
        if hashlib.sha256(output.read_bytes()).hexdigest() == digest:
            return output

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{bundle_name}.",
        suffix=".tmp",
        dir=root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    if hashlib.sha256(output.read_bytes()).hexdigest() != digest:
        raise PackagedPreparedModelError(
            "materialized prepared-model cache entry has a SHA-256 mismatch"
        )
    return output


def _validate_bundle(
    path: Path,
    metadata: Mapping[str, object],
    *,
    identifier: str,
    architecture: str,
) -> None:
    from pyamplicol._internal.versions import (
        COMPILED_MODEL_SCHEMA_VERSION,
        SYMBOLICA_SERIALIZATION_ABI,
        SYMJIT_APPLICATION_ABI,
        SYMJIT_PLANE_APPLICATION_ABI,
    )
    from pyamplicol.models.loading import MODEL_COMPILER_VERSION, compiler_fingerprint
    from pyamplicol.models.prepared import (
        EAGER_KERNEL_ABI,
        PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
        load_prepared_model_bundle,
    )
    from pyamplicol.models.prepared_target import symjit_storage_v3_target

    try:
        bundle = load_prepared_model_bundle(path)
    except (OSError, TypeError, ValueError) as error:
        raise PackagedPreparedModelError(
            f"packaged prepared-model bundle is invalid: {error}"
        ) from error
    pack = bundle.kernel_pack
    producer = _mapping(metadata.get("producer"), "metadata.producer")
    dependencies = _mapping(metadata.get("dependencies"), "metadata.dependencies")
    compiled_producer = _mapping(
        bundle.compiled_model.get("producer"), "compiled_model.producer"
    )
    compiled_source = _mapping(
        bundle.compiled_model.get("source"), "compiled_model.source"
    )
    fingerprint = compiler_fingerprint()
    expected = {
        "compiled_model_schema": COMPILED_MODEL_SCHEMA_VERSION,
        "model_compiler_version": MODEL_COMPILER_VERSION,
    }
    for key, value in expected.items():
        if producer.get(key) != value:
            raise PackagedPreparedModelError(
                f"packaged prepared-model producer {key} is stale: "
                f"expected {value!r}, got {producer.get(key)!r}"
            )
    packaged_package_version = producer.get("package_version")
    if compiled_producer.get("pyamplicol") != packaged_package_version:
        raise PackagedPreparedModelError(
            "prepared compiled-model package version disagrees with metadata"
        )
    if (
        compiled_producer.get("compiled_model_schema_version")
        != expected["compiled_model_schema"]
    ):
        raise PackagedPreparedModelError("prepared compiled-model schema is stale")
    if (
        compiled_producer.get("model_compiler_version")
        != expected["model_compiler_version"]
    ):
        raise PackagedPreparedModelError("prepared model compiler version is stale")
    model_source = _MODEL_SOURCE_BY_ID[identifier]
    if compiled_source.get("kind") != model_source:
        raise PackagedPreparedModelError(f"prepared model source is not {model_source}")
    if pack.producer.get("version") != packaged_package_version:
        raise PackagedPreparedModelError(
            "prepared kernel-pack package version disagrees with metadata"
        )
    dependency_expected = {
        "symbolica_version": fingerprint["symbolica"],
        "ufo_model_loader_version": fingerprint["ufo_model_loader"],
        "symbolica_serialization_abi": SYMBOLICA_SERIALIZATION_ABI,
        "symjit_application_abi": SYMJIT_APPLICATION_ABI,
        "symjit_plane_application_abi": SYMJIT_PLANE_APPLICATION_ABI,
    }
    for key, value in dependency_expected.items():
        if dependencies.get(key) != value:
            raise PackagedPreparedModelError(
                f"packaged prepared-model dependency {key} is stale"
            )
    pack_dependencies = {
        "symbolica_version": dependencies["symbolica_version"],
        "symbolica_serialization": dependencies["symbolica_serialization_abi"],
        "symjit_application": dependencies["symjit_application_abi"],
        "symjit_plane_application": dependencies["symjit_plane_application_abi"],
    }
    for key, value in pack_dependencies.items():
        if pack.dependency_abis.get(key) != value:
            raise PackagedPreparedModelError(
                f"prepared kernel-pack dependency {key} is stale"
            )
    if metadata.get("eager_kernel_abi") != EAGER_KERNEL_ABI:
        raise PackagedPreparedModelError("packaged eager kernel ABI is stale")
    if (
        metadata.get("prepared_model_bundle_schema")
        != PREPARED_MODEL_BUNDLE_SCHEMA_VERSION
    ):
        raise PackagedPreparedModelError("packaged model bundle schema is stale")
    if metadata.get("backend") != "jit" or bundle.backend != "jit":
        raise PackagedPreparedModelError("packaged built-in model is not JIT-backed")
    if metadata.get("jit_optimization_level") != 2:
        raise PackagedPreparedModelError(
            "packaged built-in model is not portable JIT O2"
        )
    if pack.optimization_settings.get("jit_optimization_level") != 2:
        raise PackagedPreparedModelError("prepared kernel pack is not portable JIT O2")
    if metadata.get("kernel_count") != len(pack.kernels) or not pack.kernels:
        raise PackagedPreparedModelError(
            "packaged prepared-model kernel count is invalid"
        )
    target = _plain_json(pack.target)
    expected_target = symjit_storage_v3_target(machine=architecture)
    if target != metadata.get("target") or target != expected_target:
        raise PackagedPreparedModelError(
            "packaged prepared model target does not match its architecture asset"
        )


def _metadata(
    value: object,
    *,
    identifier: str,
    bundle_name: str,
) -> Mapping[str, object]:
    metadata = _mapping(value, "prepared-model metadata")
    missing = _METADATA_KEYS - metadata.keys()
    unknown = metadata.keys() - _METADATA_KEYS
    if missing or unknown:
        raise PackagedPreparedModelError(
            "packaged prepared-model metadata fields are invalid"
        )
    if metadata.get("schema_version") != 1:
        raise PackagedPreparedModelError(
            "unsupported packaged prepared-model metadata schema"
        )
    if metadata.get("id") != identifier:
        raise PackagedPreparedModelError("packaged prepared-model identity is invalid")
    if (
        metadata.get("model") != _MODEL_SOURCE_BY_ID[identifier]
        or metadata.get("bundle") != bundle_name
    ):
        raise PackagedPreparedModelError("packaged prepared-model resource is invalid")
    return metadata


def _asset_names(identifier: str, architecture: str) -> tuple[str, str]:
    stem = f"{identifier}-{architecture}"
    return f"{stem}.metadata.json", f"{stem}.pyamplicol-model"


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PackagedPreparedModelError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(item) for item in value]
    return value


__all__ = [
    "BUILTIN_SM_HEFT_JIT_O2",
    "BUILTIN_SM_JIT_O2",
    "PackagedPreparedModelError",
    "available_prepared_models",
    "materialize_packaged_prepared_model",
    "open_packaged_prepared_model",
    "packaged_prepared_model_path",
]
