#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Cross-architecture prepared-JIT contracts for the CI transfer harness.

SymJIT application storage v3 at optimization level 2 can be reloaded across
the supported x86-64 and AArch64 hosts. These checks intentionally run before
the consumer asks SymJIT to load an application.
"""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

FORBIDDEN_SUFFIXES = frozenset(
    {
        ".a",
        ".asm",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".dll",
        ".dylib",
        ".exe",
        ".lib",
        ".o",
        ".obj",
        ".s",
        ".so",
    }
)
_FORBIDDEN_MANIFEST_PATH_FIELDS = frozenset(
    {
        "assembly_path",
        "library_path",
        "object_path",
        "source_path",
    }
)
_NATIVE_MAGICS = (
    (b"\x7fELF", "ELF image"),
    (b"!<arch>\n", "static archive"),
    (b"BC\xc0\xde", "LLVM bitcode"),
    (b"\x00asm", "WebAssembly image"),
    (b"\xfe\xed\xfa\xce", "Mach-O image"),
    (b"\xce\xfa\xed\xfe", "Mach-O image"),
    (b"\xfe\xed\xfa\xcf", "Mach-O image"),
    (b"\xcf\xfa\xed\xfe", "Mach-O image"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal image"),
    (b"\xbe\xba\xfe\xca", "Mach-O universal image"),
)
_COFF_MACHINE_IDS = frozenset(
    {
        0x014C,  # i386
        0x01C0,  # ARM
        0x01C4,  # ARMv7
        0x8664,  # x86-64
        0xAA64,  # ARM64
    }
)
SYMJIT_STORAGE_V3_APPLICATION_ABI = "symjit-application-storage-v3"
SYMJIT_PLANE_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"
SYMJIT_STORAGE_V3_ARCHITECTURES = ("aarch64", "x86_64")
_MACHINE_ARCHITECTURE_ALIASES = {
    "aarch64": "aarch64",
    "amd64": "x86_64",
    "arm64": "aarch64",
    "x64": "x86_64",
    "x86-64": "x86_64",
    "x86_64": "x86_64",
}


class PortabilityError(RuntimeError):
    """Raised when the transfer or portability contract is violated."""


@dataclass(frozen=True, slots=True)
class RuntimeContracts:
    """Installed package contracts relevant to a transferred JIT pack."""

    bundle_kind: str
    bundle_schema_version: int
    eager_kernel_abi: str
    compiled_model_schema_version: int
    symbolica_serialization_abi: str
    symjit_application_abi: str
    symjit_plane_application_abi: str
    symjit_runtime_capability: str
    eager_runtime_capability: str
    package_version: str


def architecture_class(machine: str) -> str:
    """Normalize a host machine name to a supported storage-v3 class."""

    normalized = machine.strip().lower()
    architecture = _MACHINE_ARCHITECTURE_ALIASES.get(normalized)
    if architecture is None:
        supported = ", ".join(SYMJIT_STORAGE_V3_ARCHITECTURES)
        raise PortabilityError(
            f"unsupported SymJIT storage-v3 machine {machine!r}; "
            f"supported architecture classes are {supported}"
        )
    return architecture


def symjit_storage_v3_target(architecture: str) -> dict[str, object]:
    """Return the canonical portable target after checking the host class."""

    if architecture not in SYMJIT_STORAGE_V3_ARCHITECTURES:
        supported = ", ".join(SYMJIT_STORAGE_V3_ARCHITECTURES)
        raise PortabilityError(
            f"unsupported SymJIT storage-v3 architecture class {architecture!r}; "
            f"expected one of {supported}"
        )
    return {
        "cpu_features": [],
        "endianness": "little",
        "portable": True,
        "target_triple": "symjit-storage-v3-portable",
        "word_bits": 64,
    }


def _target_architecture_class(target: dict[str, object]) -> str:
    """Return the storage class encoded by a prepared target.

    Storage-v3 O2 applications contain portable MIR rather than host machine
    code.  The producing and consuming hosts must each be supported, but their
    architecture classes need not match.
    """

    if target == symjit_storage_v3_target("x86_64"):
        return "portable"
    raise PortabilityError(
        "prepared JIT target must be portable 64-bit little-endian SymJIT storage-v3"
    )


def object_value(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PortabilityError(f"{context} must be an object with string keys")
    return value


def array_value(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise PortabilityError(f"{context} must be an array")
    return value


def string_value(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PortabilityError(f"{context} must be a non-empty string")
    return value


def integer_value(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PortabilityError(f"{context} must be an integer")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_member_path(value: object, context: str) -> str:
    path = string_value(value, context)
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in path
    ):
        raise PortabilityError(f"{context} is not a normalized relative path")
    return pure.as_posix()


def native_payload_kind(payload: bytes) -> str | None:
    for magic, description in _NATIVE_MAGICS:
        if payload.startswith(magic):
            return description

    if len(payload) >= 64 and payload.startswith(b"MZ"):
        pe_offset = int.from_bytes(payload[0x3C:0x40], "little")
        if (
            0 <= pe_offset <= len(payload) - 4
            and payload[pe_offset : pe_offset + 4] == b"PE\0\0"
        ):
            return "PE image"

    if len(payload) >= 20:
        machine = int.from_bytes(payload[0:2], "little")
        section_count = int.from_bytes(payload[2:4], "little")
        optional_header_size = int.from_bytes(payload[16:18], "little")
        if (
            machine in _COFF_MACHINE_IDS
            and 0 < section_count < 256
            and optional_header_size <= len(payload)
        ):
            return "COFF object"
    return None


def archive_manifest(path: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    if not path.is_file() or path.is_symlink():
        raise PortabilityError("prepared JIT bundle must be a regular file")
    if not path.name.lower().endswith(".pyamplicol-model"):
        raise PortabilityError("prepared JIT bundle must end with .pyamplicol-model")

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise PortabilityError("prepared JIT bundle has duplicate members")
            if "manifest.json" not in names:
                raise PortabilityError("prepared JIT bundle is missing manifest.json")

            payloads: dict[str, bytes] = {}
            for info in infos:
                member = canonical_member_path(info.filename, "archive member")
                if info.is_dir():
                    raise PortabilityError(
                        f"prepared JIT bundle contains directory member {member!r}"
                    )
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type not in {0, stat.S_IFREG}:
                    raise PortabilityError(
                        f"prepared JIT bundle member {member!r} is not regular"
                    )
                if mode & 0o111:
                    raise PortabilityError(
                        f"prepared JIT bundle member {member!r} is executable"
                    )
                payloads[member] = archive.read(info)
    except (OSError, zipfile.BadZipFile) as error:
        raise PortabilityError(
            f"could not read prepared JIT bundle: {error}"
        ) from error

    try:
        manifest = json.loads(payloads["manifest.json"])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortabilityError(
            "prepared JIT manifest is not valid UTF-8 JSON"
        ) from error
    return object_value(manifest, "manifest"), payloads


def audit_architecture_jit_bundle(
    path: Path,
    *,
    contracts: RuntimeContracts,
    expected_sha256: str | None = None,
    expected_architecture_class: str | None = None,
    expected_model_source: str = "built-in-sm",
) -> dict[str, object]:
    """Validate archive integrity and the portable storage-v3 contract."""

    bundle_path = path.expanduser().resolve(strict=True)
    bundle_sha256 = sha256_file(bundle_path)
    if expected_sha256 is not None and bundle_sha256 != expected_sha256:
        raise PortabilityError("transferred prepared JIT bundle SHA-256 mismatch")

    manifest, archive_payloads = archive_manifest(bundle_path)
    if manifest.get("kind") != contracts.bundle_kind:
        raise PortabilityError("invalid prepared JIT bundle kind")
    if manifest.get("schema_version") != contracts.bundle_schema_version:
        raise PortabilityError("unsupported prepared JIT bundle schema")
    if manifest.get("eager_kernel_abi") != contracts.eager_kernel_abi:
        raise PortabilityError("prepared JIT eager-kernel ABI mismatch")

    member_records = array_value(manifest.get("members"), "manifest.members")
    recorded_members: dict[str, dict[str, object]] = {}
    for index, raw_record in enumerate(member_records):
        record = object_value(raw_record, f"manifest.members[{index}]")
        member_path = canonical_member_path(
            record.get("path"), f"manifest.members[{index}].path"
        )
        if member_path in recorded_members:
            raise PortabilityError(f"duplicate manifest member {member_path!r}")
        payload = archive_payloads.get(member_path)
        if payload is None:
            raise PortabilityError(f"manifest member {member_path!r} is missing")
        if integer_value(record.get("size"), f"member {member_path!r} size") != len(
            payload
        ):
            raise PortabilityError(f"manifest member {member_path!r} size mismatch")
        if string_value(
            record.get("sha256"), f"member {member_path!r} sha256"
        ) != _sha256_bytes(payload):
            raise PortabilityError(f"manifest member {member_path!r} SHA-256 mismatch")
        recorded_members[member_path] = record

    expected_members = set(recorded_members) | {"manifest.json"}
    if set(archive_payloads) != expected_members:
        unexpected = sorted(set(archive_payloads) - expected_members)
        missing = sorted(expected_members - set(archive_payloads))
        raise PortabilityError(
            "prepared JIT archive member set differs from its manifest "
            f"(unexpected={unexpected}, missing={missing})"
        )

    for member_path, payload in archive_payloads.items():
        if member_path == "manifest.json":
            continue
        if PurePosixPath(member_path).suffix.lower() in FORBIDDEN_SUFFIXES:
            raise PortabilityError(
                f"prepared JIT bundle contains native/source payload {member_path!r}"
            )
        native_kind = native_payload_kind(payload)
        if native_kind is not None:
            raise PortabilityError(
                f"prepared JIT bundle member {member_path!r} contains {native_kind}"
            )

    kernel_pack = object_value(manifest.get("kernel_pack"), "manifest.kernel_pack")
    if kernel_pack.get("backend") != "jit":
        raise PortabilityError("prepared pack backend must be jit")
    target = object_value(kernel_pack.get("target"), "kernel_pack.target")
    bundle_architecture = _target_architecture_class(target)
    if expected_architecture_class is not None:
        # Validate that the consumer belongs to a supported host class.  There
        # is deliberately no equality check against ``bundle_architecture``:
        # portable storage-v3 O2 applications are recompiled when loaded.
        architecture_class(expected_architecture_class)
    expected_target = symjit_storage_v3_target("x86_64")

    dependency_abis = object_value(
        kernel_pack.get("dependency_abis"), "kernel_pack.dependency_abis"
    )
    if contracts.symjit_application_abi != SYMJIT_STORAGE_V3_APPLICATION_ABI:
        raise PortabilityError(
            "this architecture-class transfer harness applies only to "
            "SymJIT application storage v3"
        )
    if (
        dependency_abis.get("symbolica_serialization")
        != contracts.symbolica_serialization_abi
    ):
        raise PortabilityError("prepared JIT Symbolica serialization ABI mismatch")
    if dependency_abis.get("symjit_application") != contracts.symjit_application_abi:
        raise PortabilityError("prepared JIT application storage ABI mismatch")
    if (
        dependency_abis.get("symjit_plane_application")
        != contracts.symjit_plane_application_abi
    ):
        raise PortabilityError("prepared JIT plane-application ABI mismatch")

    optimization = object_value(
        kernel_pack.get("optimization_settings"),
        "kernel_pack.optimization_settings",
    )
    if (
        optimization.get("backend") != "jit"
        or optimization.get("jit_optimization_level") != 2
    ):
        raise PortabilityError("prepared JIT pack must use portable JIT O2")
    if optimization.get("compiled_native") is not False:
        raise PortabilityError("prepared JIT pack requests final native compilation")
    if optimization.get("compiled_inline_asm") not in {None, "none"}:
        raise PortabilityError("prepared JIT pack requests inline assembly")
    if optimization.get("compiler_path") is not None:
        raise PortabilityError("prepared JIT pack records an external compiler")
    effective_flags = optimization.get("effective_compiler_flags")
    if effective_flags is not None and effective_flags != []:
        raise PortabilityError("prepared JIT pack records target compiler flags")

    provenance = object_value(kernel_pack.get("provenance"), "kernel_pack.provenance")
    model_source = object_value(provenance.get("model_source"), "pack model source")
    expected_model = string_value(expected_model_source, "expected model source")
    if (
        provenance.get("model_name") != expected_model
        or model_source.get("kind") != expected_model
    ):
        raise PortabilityError(
            f"portability transfer must contain model {expected_model!r}"
        )
    producer = object_value(kernel_pack.get("producer"), "kernel_pack.producer")
    if producer.get("distribution") != "pyamplicol":
        raise PortabilityError("prepared JIT producer distribution is not pyamplicol")
    if producer.get("compiled_model_schema") != contracts.compiled_model_schema_version:
        raise PortabilityError("prepared JIT compiled-model schema mismatch")
    producer_version = string_value(
        producer.get("version"), "kernel_pack.producer.version"
    )

    kernels = array_value(kernel_pack.get("kernels"), "kernel_pack.kernels")
    if not kernels:
        raise PortabilityError("prepared JIT pack contains no kernels")
    variants = array_value(
        kernel_pack.get("kernel_variants", []),
        "kernel_pack.kernel_variants",
    )
    evaluator_records = [
        (f"kernel {index}", object_value(raw, f"kernel_pack.kernels[{index}]"))
        for index, raw in enumerate(kernels)
    ]
    evaluator_records.extend(
        (
            f"kernel variant {index}",
            object_value(raw, f"kernel_pack.kernel_variants[{index}]"),
        )
        for index, raw in enumerate(variants)
    )
    application_paths: set[str] = set()
    plane_application_paths: set[str] = set()
    exact_state_paths: set[str] = set()
    for context, kernel in evaluator_records:
        f64 = object_value(
            kernel.get("f64_evaluator_manifest"),
            f"{context}.f64_evaluator_manifest",
        )
        required = {
            "application_abi": contracts.symjit_application_abi,
            "backend": "jit",
            "batch_layout": "row-major",
            "compiler_type": "native",
            "element_layout": "complex-f64",
            "endianness": "little",
            "kind": "symjit-application-evaluator",
            "optimization_level": 2,
            "runtime_capability": contracts.symjit_runtime_capability,
            "translation_mode": "indirect",
            "word_bits": 64,
        }
        for key, expected in required.items():
            if f64.get(key) != expected:
                raise PortabilityError(f"prepared {context} has incompatible {key!r}")
        if f64.get("required_defuns") != []:
            raise PortabilityError(f"prepared {context} requires external functions")
        kernel_settings = object_value(
            f64.get("settings"),
            f"{context}.f64_evaluator_manifest.settings",
        )
        if kernel_settings.get("backend") != "jit":
            raise PortabilityError(
                f"prepared {context} settings select a non-JIT backend"
            )
        if kernel_settings.get("jit_optimization_level") != 2:
            raise PortabilityError(
                f"prepared {context} settings do not select portable JIT O2"
            )
        if kernel_settings.get("compiled_native") is not False:
            raise PortabilityError(f"prepared {context} settings request native code")
        if kernel_settings.get("compiled_inline_asm") not in {None, "none"}:
            raise PortabilityError(
                f"prepared {context} settings request inline assembly"
            )
        if kernel_settings.get("compiler_path") is not None:
            raise PortabilityError(
                f"prepared {context} settings record an external compiler"
            )
        kernel_flags = kernel_settings.get("effective_compiler_flags")
        if kernel_flags is not None and kernel_flags != []:
            raise PortabilityError(f"prepared {context} settings record compiler flags")
        for field in _FORBIDDEN_MANIFEST_PATH_FIELDS:
            if f64.get(field) not in {None, ""}:
                raise PortabilityError(f"prepared {context} records forbidden {field}")
        application_path = canonical_member_path(
            f64.get("application_path"),
            f"{context} application_path",
        )
        if not application_path.endswith(".symjit"):
            raise PortabilityError(
                f"prepared {context} application is not SymJIT storage"
            )
        exact_state_path = canonical_member_path(
            f64.get("evaluator_state_path"),
            f"{context} evaluator_state_path",
        )
        if not exact_state_path.endswith(".evaluator.bin"):
            raise PortabilityError(
                f"prepared {context} exact state has an unexpected format"
            )
        if (
            application_path not in recorded_members
            or exact_state_path not in recorded_members
        ):
            raise PortabilityError(f"prepared {context} omits referenced payloads")

        plane = object_value(
            f64.get("plane_application"),
            f"{context}.f64_evaluator_manifest.plane_application",
        )
        input_complex_count = integer_value(
            f64.get("input_len"),
            f"{context}.f64_evaluator_manifest.input_len",
        )
        output_complex_count = integer_value(
            f64.get("output_len"),
            f"{context}.f64_evaluator_manifest.output_len",
        )
        if input_complex_count < 0 or output_complex_count < 1:
            raise PortabilityError(
                f"prepared {context} has invalid evaluator input/output counts"
            )
        plane_required = {
            "application_abi": contracts.symjit_plane_application_abi,
            "storage_abi": contracts.symjit_application_abi,
            "element_layout": "split-complex-plane-major",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "input_complex_count": input_complex_count,
            "output_complex_count": output_complex_count,
            "input_plane_count": 2 * input_complex_count,
            "output_plane_count": 2 * output_complex_count,
            "compiler_type": "native",
            "translation_mode": "symbolica-structured-instructions",
            "optimization_level": 2,
            "simd": True,
            "complex": True,
            "fast_math": True,
            "fast_complex": False,
            "threading": False,
            "direct_arena": True,
        }
        for key, expected in plane_required.items():
            if plane.get(key) != expected:
                raise PortabilityError(
                    f"prepared {context} has incompatible plane-application {key!r}"
                )
        if not isinstance(plane.get("compression"), bool):
            raise PortabilityError(
                f"prepared {context} has invalid plane-application compression"
            )
        source_digest = plane.get("source_digest")
        if (
            not isinstance(source_digest, str)
            or len(source_digest) != 64
            or any(character not in "0123456789abcdef" for character in source_digest)
        ):
            raise PortabilityError(
                f"prepared {context} has invalid plane-application source digest"
            )
        plane_target = object_value(
            plane.get("target"),
            f"{context}.f64_evaluator_manifest.plane_application.target",
        )
        if (
            plane_target.get("word_bits") != 64
            or plane_target.get("endianness") != "little"
        ):
            raise PortabilityError(
                f"prepared {context} has incompatible plane-application target"
            )
        plane_application_path = canonical_member_path(
            plane.get("application_path"),
            f"{context} plane-application path",
        )
        if not plane_application_path.endswith(".plane.symjit"):
            raise PortabilityError(
                f"prepared {context} plane application is not canonical SymJIT storage"
            )
        if plane_application_path not in recorded_members:
            raise PortabilityError(
                f"prepared {context} omits referenced plane-application payload"
            )
        application_paths.add(application_path)
        plane_application_paths.add(plane_application_path)
        exact_state_paths.add(exact_state_path)

    compiled_model_path = "model/model.pyAmplicol-model.json"
    expected_payload_members = (
        application_paths
        | plane_application_paths
        | exact_state_paths
        | {compiled_model_path}
    )
    if set(recorded_members) != expected_payload_members:
        unexpected = sorted(set(recorded_members) - expected_payload_members)
        missing = sorted(expected_payload_members - set(recorded_members))
        raise PortabilityError(
            "prepared JIT payload set differs from canonical application/state "
            "members "
            f"(unexpected={unexpected}, missing={missing})"
        )
    try:
        compiled_model = object_value(
            json.loads(archive_payloads[compiled_model_path]),
            "compiled model",
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortabilityError("prepared bundle compiled model is invalid") from error
    if compiled_model.get("schema_version") != contracts.compiled_model_schema_version:
        raise PortabilityError("prepared bundle compiled-model payload is incompatible")
    compiled_source = object_value(
        compiled_model.get("source"), "compiled model source"
    )
    if compiled_source.get("kind") != expected_model:
        raise PortabilityError(
            f"prepared bundle compiled model is not {expected_model!r}"
        )

    return {
        "backend": "jit",
        "architecture_class": bundle_architecture,
        "bundle_sha256": bundle_sha256,
        "bundle_size": bundle_path.stat().st_size,
        "compiled_model_digest": provenance.get("compiled_model_digest"),
        "eager_kernel_abi": contracts.eager_kernel_abi,
        "exact_state_count": len(exact_state_paths),
        "kernel_count": len(kernels),
        "kernel_variant_count": len(variants),
        "model_compiler_version": producer.get("model_compiler_version"),
        "producer_version": producer_version,
        "symjit_application_count": len(application_paths),
        "symjit_application_abi": contracts.symjit_application_abi,
        "symjit_plane_application_count": len(plane_application_paths),
        "symjit_plane_application_abi": contracts.symjit_plane_application_abi,
        "target": expected_target,
    }


__all__ = [
    "FORBIDDEN_SUFFIXES",
    "SYMJIT_PLANE_APPLICATION_ABI",
    "SYMJIT_STORAGE_V3_APPLICATION_ABI",
    "SYMJIT_STORAGE_V3_ARCHITECTURES",
    "PortabilityError",
    "RuntimeContracts",
    "architecture_class",
    "archive_manifest",
    "array_value",
    "audit_architecture_jit_bundle",
    "canonical_member_path",
    "integer_value",
    "native_payload_kind",
    "object_value",
    "sha256_file",
    "string_value",
    "symjit_storage_v3_target",
]
