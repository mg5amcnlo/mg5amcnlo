#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Audit and compare pyAmpliCol release artifacts."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from packaging.markers import Variable
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:  # pragma: no cover - pip vendors the build fallback
    from pip._vendor.packaging.markers import Variable  # type: ignore[no-redef]
    from pip._vendor.packaging.requirements import (  # type: ignore[no-redef]
        InvalidRequirement,
        Requirement,
    )
    from pip._vendor.packaging.version import (  # type: ignore[no-redef]
        InvalidVersion,
        Version,
    )

from _common import (
    ROOT,
    ReleaseError,
    external_temporary_directory,
    run,
    safe_extract_sdist,
    sha256,
)
from audit_sdist import (
    PREPARED_MODEL_ARCHITECTURES,
    PREPARED_MODEL_ASSETS,
    REQUIRED_SDIST_MEMBERS,
    prepared_model_asset_members,
)
from prepare_selftest_fixture import (
    _artifact_id,
    validate_artifact_identity_contract,
    validate_portable_artifact_capabilities,
    validate_portable_execution_manifest,
)

MAX_WHEEL_BYTES = 95_000_000
EXPECTED_DISTRIBUTION = "pyamplicol"
with (ROOT / "dependencies" / "release-lock.toml").open("rb") as _release_stream:
    EXPECTED_RELEASE_VERSION = str(tomllib.load(_release_stream)["project"]["version"])
EXPECTED_PYTHON_TAG = "cp311"
EXPECTED_ABI_TAG = "abi3"
RELEASE_TARGETS = {
    "macosx_11_0_arm64": "aarch64-apple-darwin",
    "macosx_11_0_x86_64": "x86_64-apple-darwin",
    "manylinux_2_28_x86_64": "x86_64-unknown-linux-gnu",
}
_LINUX_PLATFORM_TAGS = {
    "manylinux_2_28_x86_64",
}
_REQUIRED_SDK_PATHS = {
    "rusticol_config/__init__.py",
    "rusticol_config/__main__.py",
    "pyamplicol/_sdk/include/rusticol.h",
    "pyamplicol/_sdk/include/rusticol.hpp",
    "pyamplicol/_sdk/fortran/rusticol.f90",
    "pyamplicol/_sdk/rust/rusticol.rs",
    "pyamplicol/_sdk/lib/librusticol_capi.a",
    "pyamplicol/_sdk/config.py",
    "pyamplicol/_sdk/metadata.json",
    "pyamplicol/_sdk/link.json",
}


_FORBIDDEN_MEMBER_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "PYPI_DEPLOYMENT_TEST",
    "target",
}
_REPOSITORY_PATH_MARKER = os.fsencode(str(ROOT.resolve()))
_FORBIDDEN_PATH_MARKERS = (
    b"/tmp/",
    b"/var/tmp/",
    b"/Users/",
    b"/home/",
    b"/private/var/folders/",
    b"/opt/homebrew/",
    b"/opt/local/",
    b"/opt/python/",
    b"/github/workspace/",
    b"C:\\Users\\",
    _REPOSITORY_PATH_MARKER,
)
_FORBIDDEN_NATIVE_RELATIVE_MARKERS = (b"../", b"..\\")
_NATIVE_MEMBER_SUFFIXES = (".a", ".dylib", ".lib", ".pyd", ".so")
_PATH_TOKEN_BYTES = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~+%-/\\"
)
_FORBIDDEN_CAPI_SYMBOLS = (
    "pyobject",
    "pyexc_",
    "pygilstate",
    "pyerr_",
    "pylong_",
    "pyunicode_",
    "pymem_",
    "pytype_",
    "pytuple_",
    "pylist_",
    "pydict_",
    "pymodule_",
    "pyimport_",
    "pybytes_",
    "pycapsule_",
    "pyfloat_",
    "pybool_",
    "pythread_",
    "pyo3",
    "numpy",
    "python3",
)
_FORBIDDEN_CAPI_UNDEFINED_PREFIXES = (
    *_FORBIDDEN_CAPI_SYMBOLS,
    "py_",
    "python",
    "libpython",
)
_FORBIDDEN_STATIC_RUNTIME_MARKERS = (
    "__gmp",
    "gmp-mpfr-sys",
    "malachite-base",
    "malachite-nz",
    "malachite-q",
    "mpfr_",
)
_COMMON_SYSTEM_LIBRARIES = {"c", "dl", "m", "pthread", "resolv", "rt", "util"}
_SYSTEM_LIBRARIES = {
    "aarch64-apple-darwin": _COMMON_SYSTEM_LIBRARIES | {"System", "gcc_s", "iconv"},
    "x86_64-apple-darwin": _COMMON_SYSTEM_LIBRARIES | {"System", "gcc_s", "iconv"},
    "x86_64-unknown-linux-gnu": _COMMON_SYSTEM_LIBRARIES | {"gcc_s"},
}
_MACOS_FRAMEWORKS = {"CoreFoundation", "IOKit", "Security", "SystemConfiguration"}
_CANONICAL_DEPENDENCY_LOCK = Path("dependencies/release-lock.toml")
_CONTRIBUTOR_DEPENDENCY_LOCK = Path("dependencies/contributor-lock.toml")
_CANDIDATE_INSTALL_STATE = Path("dependencies/install-state.json")
_REQUIRED_PACKAGE_RESOURCES = {
    "pyamplicol/assets/schemas/README.md",
    "pyamplicol/assets/schemas/artifact-manifest-v3.schema.json",
    "pyamplicol/assets/schemas/runtime-physics-v1.schema.json",
}
_REQUIRED_PACKAGED_EXAMPLE_MEMBERS = {
    "pyamplicol/_examples/data/pp_zjj_momenta.json",
}
_REQUIRED_API_TEMPLATE_MEMBERS = {
    "pyamplicol/assets/api_templates/c/Makefile",
    "pyamplicol/assets/api_templates/c/check_standalone.c",
    "pyamplicol/assets/api_templates/cpp/Makefile",
    "pyamplicol/assets/api_templates/cpp/check_standalone.cpp",
    "pyamplicol/assets/api_templates/fortran/Makefile",
    "pyamplicol/assets/api_templates/fortran/check_standalone.f90",
    "pyamplicol/assets/api_templates/python/check_standalone.py",
    "pyamplicol/assets/api_templates/rust/Makefile",
    "pyamplicol/assets/api_templates/rust/check_standalone.rs",
    *(
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
    ),
}
_PREPARED_MODEL_WHEEL_PREFIX = "pyamplicol/assets/prepared_models"
_PREPARED_MODEL_SDIST_PREFIX = "src/pyamplicol/assets/prepared_models"
_REQUIRED_PREPARED_MODEL_WHEEL_MEMBERS = prepared_model_asset_members(
    _PREPARED_MODEL_WHEEL_PREFIX
)
_REQUIRED_WHEEL_PACKAGE_MEMBERS = {
    "pyamplicol/__init__.py",
    "pyamplicol/_build_info.json",
    "pyamplicol/_rusticol.pyi",
    "pyamplicol/py.typed",
    *_REQUIRED_API_TEMPLATE_MEMBERS,
    *_REQUIRED_PACKAGED_EXAMPLE_MEMBERS,
    *_REQUIRED_PACKAGE_RESOURCES,
    *_REQUIRED_PREPARED_MODEL_WHEEL_MEMBERS,
    *_REQUIRED_SDK_PATHS,
}
_REQUIRED_SELFTEST_API_PAYLOADS = {
    "API/validation_points.dat",
    "API/python/check_standalone.py",
    "API/c/Makefile",
    "API/c/check_standalone.c",
    "API/cpp/Makefile",
    "API/cpp/check_standalone.cpp",
    "API/fortran/Makefile",
    "API/fortran/check_standalone.f90",
    "API/rust/Makefile",
    "API/rust/check_standalone.rs",
}
_COMPILED_MODEL_KIND = "pyamplicol-compiled-model"
_BUILTIN_MODEL_SOURCE_KIND = "built-in-sm"
_ALLOWED_REPAIR_ROOTS = {"pyamplicol.libs"}
_FORBIDDEN_SDIST_MEMBERS = {
    ".cargo/config.toml",
    "build_backend/python_lock.py",
    "dependencies/candidate-Cargo.lock",
    "dependencies/candidate-cargo-config.toml",
    "dependencies/contributor-lock.toml",
    "dependencies/install-state.json",
    "dependencies/install_dependencies.py",
    "dependencies/python-runtime-lock.toml",
    "dependencies/symbolica_patches.tar.gz",
}
_FORBIDDEN_SDIST_PREFIXES = (
    "dependencies/checkouts/",
    "dependencies/patches/",
    "release_assets/",
)
_RELEASE_SYMJIT_FIELDS = {"version"}
_RELEASE_BUILD_INFO_FIELDS = {
    "publishable",
    "schema_version",
    "selftest_fixture_bootstrap",
    "source_revision",
    "version",
}


class ArtifactError(ReleaseError):
    """An artifact violates the packaging contract."""


@dataclass(frozen=True)
class WheelReport:
    filename: str
    version: str
    size: int
    sha256: str
    python_tag: str
    abi_tag: str
    target: str
    rust_target: str
    native_scan: bool


@dataclass(frozen=True)
class SdistReport:
    filename: str
    version: str
    size: int
    sha256: str


def canonicalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _safe_member_name(name: str, *, archive: str) -> PurePosixPath:
    stripped = name.rstrip("/")
    if not stripped or "\\" in stripped:
        raise ArtifactError(f"{archive} contains an unsafe member name: {name!r}")
    relative = PurePosixPath(stripped)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ArtifactError(f"{archive} contains an unsafe member path: {name}")
    if _FORBIDDEN_MEMBER_PARTS.intersection(relative.parts):
        raise ArtifactError(f"{archive} contains forbidden build state: {name}")
    return relative


def _wheel_entries(path: Path) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ArtifactError(f"invalid wheel archive {path}: {error}") from error
    with archive:
        for info in archive.infolist():
            relative = _safe_member_name(info.filename, archive="wheel")
            name = relative.as_posix()
            if info.is_dir():
                continue
            if name in entries:
                raise ArtifactError(f"wheel contains duplicate member: {name}")
            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ArtifactError(f"wheel contains a symbolic link: {name}")
            if info.flag_bits & 0x1:
                raise ArtifactError(f"wheel contains an encrypted member: {name}")
            entries[name] = archive.read(info)
    return entries


def _single_name(entries: dict[str, bytes], suffix: str) -> str:
    matches = sorted(name for name in entries if name.endswith(suffix))
    if len(matches) != 1:
        raise ArtifactError(
            f"wheel must contain exactly one {suffix}, found {len(matches)}"
        )
    return matches[0]


def _parse_wheel_filename(path: Path) -> tuple[str, str, str, str]:
    if path.suffix != ".whl":
        raise ArtifactError(f"not a wheel filename: {path.name}")
    fields = path.name[:-4].rsplit("-", 3)
    if len(fields) != 4:
        raise ArtifactError(f"invalid wheel filename: {path.name}")
    prefix, python_tag, abi_tag, platform_tag = fields
    distribution_version = prefix.rsplit("-", 1)
    if len(distribution_version) != 2:
        raise ArtifactError(f"invalid wheel distribution/version: {path.name}")
    distribution, filename_version = distribution_version
    if canonicalize_name(distribution) != EXPECTED_DISTRIBUTION:
        raise ArtifactError(f"wheel distribution is not pyamplicol: {path.name}")
    return filename_version, python_tag, abi_tag, platform_tag


def _canonical_target(platform_tag: str, *, mode: str) -> tuple[str, str]:
    components = set(platform_tag.split("."))
    if components and components <= _LINUX_PLATFORM_TAGS:
        target = "manylinux_2_28_x86_64"
        return target, RELEASE_TARGETS[target]
    if mode == "candidate" and platform_tag == "linux_x86_64":
        target = "manylinux_2_28_x86_64"
        return target, RELEASE_TARGETS[target]
    if len(components) == 1:
        target = next(iter(components))
        rust_target = RELEASE_TARGETS.get(target)
        if rust_target is not None:
            return target, rust_target
    raise ArtifactError(
        "wheel platform tag must be macosx_11_0_arm64, "
        "macosx_11_0_x86_64, or manylinux_2_28_x86_64; "
        f"found {platform_tag}"
    )


def _message(data: bytes, description: str):
    try:
        return BytesParser(policy=policy.default).parsebytes(data)
    except Exception as error:  # pragma: no cover - email parser is defensive
        raise ArtifactError(f"could not parse {description}: {error}") from error


def _reject_direct_requirements(
    requirements: list[str], *, description: str = "wheel metadata"
) -> None:
    direct = re.compile(
        r"(?i)(?:\s@\s|(?:file|git\+[^:]+|https?)://|(?:^|[ (])\.\.?[/\\])"
    )
    invalid = [
        requirement for requirement in requirements if direct.search(requirement)
    ]
    if invalid:
        raise ArtifactError(
            f"{description} contains direct URL or path dependencies: "
            + ", ".join(invalid)
        )


def _project_requirements(pyproject: dict[str, Any]) -> list[str]:
    requirements: list[str] = []
    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        raise ArtifactError("sdist pyproject.toml [project] must be a table")
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise ArtifactError("sdist project dependencies must be strings")
    requirements.extend(dependencies)
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict) or not all(
        isinstance(group, list) and all(isinstance(item, str) for item in group)
        for group in optional.values()
    ):
        raise ArtifactError("sdist optional dependencies must be string lists")
    for group in optional.values():
        requirements.extend(group)
    build_system = pyproject.get("build-system", {})
    if not isinstance(build_system, dict):
        raise ArtifactError("sdist pyproject.toml [build-system] must be a table")
    build_requirements = build_system.get("requires", [])
    if not isinstance(build_requirements, list) or not all(
        isinstance(item, str) for item in build_requirements
    ):
        raise ArtifactError("sdist build requirements must be strings")
    requirements.extend(build_requirements)
    return requirements


@lru_cache(maxsize=1)
def _dependency_lock() -> dict[str, Any]:
    canonical_path = ROOT / _CANONICAL_DEPENDENCY_LOCK
    if not canonical_path.is_file():
        raise ArtifactError(f"canonical dependency lock is missing: {canonical_path}")
    try:
        with canonical_path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(f"dependency lock is invalid: {error}") from error
    if payload.get("schema_version") != 1:
        raise ArtifactError("dependency lock must use schema_version = 1")
    return payload


def _validate_wheel_resource_layout(entries: dict[str, bytes]) -> None:
    dist_info_roots = {
        PurePosixPath(name).parts[0]
        for name in entries
        if PurePosixPath(name).parts[0].endswith(".dist-info")
    }
    if len(dist_info_roots) != 1:
        raise ArtifactError(
            "wheel must contain exactly one top-level .dist-info directory"
        )
    allowed_roots = {
        "pyamplicol",
        "rusticol_config",
        *dist_info_roots,
        *_ALLOWED_REPAIR_ROOTS,
    }
    misplaced = sorted(
        name for name in entries if PurePosixPath(name).parts[0] not in allowed_roots
    )
    if misplaced:
        raise ArtifactError(
            "wheel contains members outside pyamplicol, rusticol_config, its "
            ".dist-info, or an approved repair-library root: " + ", ".join(misplaced)
        )
    sboms = sorted(
        name
        for name in entries
        if len(PurePosixPath(name).parts) > 1
        and PurePosixPath(name).parts[0] in dist_info_roots
        and PurePosixPath(name).parts[1] == "sboms"
    )
    if sboms:
        raise ArtifactError(
            "lean release wheels must not contain generated SBOMs: " + ", ".join(sboms)
        )
    missing = sorted(_REQUIRED_PACKAGE_RESOURCES - entries.keys())
    if missing:
        raise ArtifactError(
            "wheel is missing required package-owned resources: " + ", ".join(missing)
        )


def _required_legal_files() -> tuple[str, ...]:
    return (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "licenses/Symbolica.txt",
        "licenses/SymJIT.txt",
    )


def _locked_python_dependencies(lock: dict[str, Any]) -> dict[str, str]:
    raw_dependencies = lock.get("python_dependencies")
    if not isinstance(raw_dependencies, list):
        raise ArtifactError("wheel dependency lock has no Python dependency list")
    dependencies: dict[str, str] = {}
    for entry in raw_dependencies:
        if not isinstance(entry, dict):
            raise ArtifactError("wheel dependency lock entries must be tables")
        distribution = entry.get("distribution")
        version = entry.get("version")
        if not isinstance(distribution, str) or not isinstance(version, str):
            raise ArtifactError("wheel dependency lock entries need name and version")
        name = canonicalize_name(distribution)
        if name in dependencies:
            raise ArtifactError(f"wheel dependency lock repeats {name}")
        try:
            normalized_version = str(Version(version))
        except InvalidVersion as error:
            raise ArtifactError(
                f"wheel dependency lock has invalid version for {name}: {version}"
            ) from error
        dependencies[name] = normalized_version
    return dependencies


def _candidate_dependency_overrides() -> dict[str, str]:
    contributor_path = ROOT / _CONTRIBUTOR_DEPENDENCY_LOCK
    state_path = ROOT / _CANDIDATE_INSTALL_STATE
    missing = [path for path in (contributor_path, state_path) if not path.is_file()]
    if missing:
        raise ArtifactError(
            "candidate dependency provenance is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    try:
        with contributor_path.open("rb") as stream:
            contributor = tomllib.load(stream)
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(
            f"candidate dependency provenance is invalid: {error}"
        ) from error
    if contributor.get("schema_version") != 1:
        raise ArtifactError(
            "candidate dependency provenance has an unsupported contributor lock"
        )
    if (
        not isinstance(state, dict)
        or set(state) != {"schema_version", "publishable", "sources"}
        or state.get("schema_version") != 1
        or state.get("publishable") is not False
    ):
        raise ArtifactError(
            "candidate dependency provenance requires a non-publishable schema-v1 "
            "installer state containing only its source revisions"
        )
    sources = state["sources"]
    if not isinstance(sources, dict) or not sources:
        raise ArtifactError("candidate dependency provenance has no source map")
    for source_name, source in sources.items():
        allowed = {"url", "revision"}
        if isinstance(source, dict) and "branch" in source:
            allowed.add("branch")
        if (
            not isinstance(source_name, str)
            or not source_name
            or not isinstance(source, dict)
            or set(source) != allowed
            or not all(isinstance(source[key], str) and source[key] for key in allowed)
            or re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None
        ):
            raise ArtifactError(
                f"candidate dependency source {source_name!r} has an invalid descriptor"
            )

    release = _dependency_lock()
    dependencies = _locked_python_dependencies(release)
    release_symbolica = release.get("symbolica")
    candidate_symbolica = contributor.get("symbolica")
    if not isinstance(release_symbolica, dict) or not isinstance(
        candidate_symbolica, dict
    ):
        raise ArtifactError("candidate dependency provenance has no Symbolica contract")
    distribution = release_symbolica.get("python_distribution")
    release_version = release_symbolica.get("python_version")
    candidate_version = candidate_symbolica.get("candidate_version")
    candidate_revision = candidate_symbolica.get("candidate_revision")
    source_url = candidate_symbolica.get("source_url")
    if not all(
        isinstance(value, str) and value
        for value in (
            distribution,
            release_version,
            candidate_version,
            candidate_revision,
            source_url,
        )
    ):
        raise ArtifactError(
            "candidate dependency provenance has an invalid Symbolica contract"
        )
    name = canonicalize_name(distribution)
    try:
        normalized_release = str(Version(release_version))
        normalized_candidate = str(Version(candidate_version))
    except InvalidVersion as error:
        raise ArtifactError(
            "candidate dependency provenance has an invalid Symbolica version"
        ) from error
    if dependencies.get(name) != normalized_release:
        raise ArtifactError(
            "release dependency inventory disagrees with its Symbolica contract"
        )
    source = sources.get("symbolica")
    if (
        not isinstance(source, dict)
        or source.get("revision") != candidate_revision
        or source.get("url") != source_url
    ):
        raise ArtifactError(
            "candidate dependency provenance is not bound to the pinned Symbolica "
            "source"
        )
    return {
        name: normalized_candidate,
        "ufo-model-loader": dependencies["ufo-model-loader"],
    }


def _validate_runtime_requirements(
    raw_requirements: list[str],
    lock: dict[str, Any],
    *,
    mode: str,
) -> None:
    dependencies = _locked_python_dependencies(lock)
    if mode == "candidate":
        dependencies.update(_candidate_dependency_overrides())
    elif mode != "release":
        raise ArtifactError(f"unsupported dependency contract mode: {mode}")
    requirements: dict[str, Requirement] = {}

    def marker_uses_extra(items: Any) -> bool:
        if isinstance(items, tuple):
            return any(
                isinstance(item, Variable) and item.value == "extra" for item in items
            )
        if isinstance(items, list):
            return any(marker_uses_extra(item) for item in items)
        return False

    for raw in raw_requirements:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as error:
            raise ArtifactError(f"wheel has invalid Requires-Dist: {raw}") from error
        if requirement.marker is not None and marker_uses_extra(
            requirement.marker._markers
        ):
            continue
        if requirement.marker is not None:
            raise ArtifactError(
                "locked runtime Requires-Dist entries may not use environment markers: "
                f"{raw}"
            )
        name = canonicalize_name(requirement.name)
        if name in requirements:
            raise ArtifactError(f"wheel metadata repeats runtime dependency {name}")
        if requirement.extras or requirement.url:
            raise ArtifactError(
                f"wheel runtime dependency must be a plain version requirement: {raw}"
            )
        requirements[name] = requirement
    if set(requirements) != set(dependencies):
        missing = sorted(set(dependencies) - set(requirements))
        extra = sorted(set(requirements) - set(dependencies))
        raise ArtifactError(
            "wheel Requires-Dist inventory disagrees with dependency lock; "
            f"missing={missing}, extra={extra}"
        )
    for name, locked_version in dependencies.items():
        requirement = requirements[name]
        if (
            not requirement.specifier
            or Version(locked_version) not in requirement.specifier
        ):
            raise ArtifactError(
                f"wheel Requires-Dist for {name} excludes locked version "
                f"{locked_version}"
            )
        specifiers = list(requirement.specifier)
        exact_match = False
        if len(specifiers) == 1 and specifiers[0].operator == "==":
            try:
                exact_match = str(Version(specifiers[0].version)) == locked_version
            except InvalidVersion:
                exact_match = False
        if not exact_match:
            raise ArtifactError(
                f"wheel Requires-Dist must pin {name}=={locked_version} exactly"
            )


def _validate_legal_members(
    entries: dict[str, bytes], metadata_name: str, metadata: Any
) -> None:
    license_files = list(metadata.get_all("License-File", []))
    declared: list[str] = []
    for raw_name in license_files:
        relative = _safe_member_name(str(raw_name), archive="License-File")
        declared.append(relative.as_posix())
    if len(declared) != len(set(declared)):
        raise ArtifactError("wheel metadata repeats a License-File declaration")
    required = set(_required_legal_files())
    if not required <= set(declared):
        missing = sorted(required - set(declared))
        raise ArtifactError(
            "wheel metadata omits required license/notice files: " + ", ".join(missing)
        )
    dist_info = metadata_name.removesuffix("METADATA")
    for relative in declared:
        member = f"{dist_info}licenses/{relative}"
        if member not in entries or not entries[member].strip():
            raise ArtifactError(f"wheel is missing nonempty legal member: {member}")


def _record_digest(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _validate_record(entries: dict[str, bytes], record_name: str) -> None:
    try:
        rows = list(csv.reader(io.StringIO(entries[record_name].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ArtifactError(f"invalid wheel RECORD: {error}") from error
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise ArtifactError(f"wheel RECORD row must have three fields: {row!r}")
        name, digest, size = row
        safe_name = _safe_member_name(name, archive="RECORD").as_posix()
        if safe_name in records:
            raise ArtifactError(f"wheel RECORD repeats {safe_name}")
        records[safe_name] = (digest, size)
    if set(records) != set(entries):
        missing = sorted(set(entries) - set(records))
        extra = sorted(set(records) - set(entries))
        raise ArtifactError(
            f"wheel RECORD inventory mismatch; missing={missing}, extra={extra}"
        )
    for name, data in entries.items():
        digest, size = records[name]
        if name == record_name:
            if digest or size:
                raise ArtifactError("wheel RECORD must not hash itself")
            continue
        expected_digest = f"sha256={_record_digest(data)}"
        if digest != expected_digest or size != str(len(data)):
            raise ArtifactError(f"wheel RECORD hash/size mismatch for {name}")


def _json_object(entries: dict[str, bytes], name: str) -> dict[str, Any]:
    try:
        payload = json.loads(entries[name])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid JSON resource {name}: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactError(f"JSON resource must be an object: {name}")
    return payload


def _validate_prepared_model_assets(
    entries: dict[str, bytes],
    *,
    prefix: str,
    mode: str,
) -> None:
    abis = _dependency_lock().get("abis")
    if not isinstance(abis, dict):
        raise ArtifactError("dependency lock has no ABI inventory")
    expected = prepared_model_asset_members(prefix)
    prepared_prefix = f"{prefix.rstrip('/')}/"
    actual = {name for name in entries if name.startswith(prepared_prefix)}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactError(
            f"prepared-model asset inventory mismatch; missing={missing}, extra={extra}"
        )

    for identifier, model_source in PREPARED_MODEL_ASSETS.items():
        for architecture in PREPARED_MODEL_ARCHITECTURES:
            stem = f"{identifier}-{architecture}"
            metadata_name = f"{prepared_prefix}{stem}.metadata.json"
            bundle_name = f"{prepared_prefix}{stem}.pyamplicol-model"
            metadata = _json_object(entries, metadata_name)
            if (
                metadata.get("schema_version") != 1
                or metadata.get("prepared_model_bundle_schema")
                != abis.get("prepared_model_bundle")
                or metadata.get("eager_kernel_abi") != abis.get("eager_kernel")
                or metadata.get("id") != identifier
                or metadata.get("model") != model_source
                or metadata.get("backend") != "jit"
                or metadata.get("jit_optimization_level") != 2
                or metadata.get("bundle") != PurePosixPath(bundle_name).name
            ):
                raise ArtifactError(
                    f"prepared-model metadata identity is invalid: {metadata_name}"
                )

            dependencies = metadata.get("dependencies")
            if (
                not isinstance(dependencies, dict)
                or dependencies.get("symjit_application_abi")
                != abis.get("symjit_application")
                or dependencies.get("symjit_plane_application_abi")
                != abis.get("symjit_plane_application")
                or dependencies.get("symbolica_serialization_abi")
                != abis.get("symbolica_serialization")
            ):
                raise ArtifactError(
                    f"prepared-model SymJIT storage ABI is invalid: {metadata_name}"
                )

            build_contract = metadata.get("build_contract")
            if (
                not isinstance(build_contract, dict)
                or build_contract.get("mode") != mode
            ):
                raise ArtifactError(
                    f"prepared-model build mode is invalid: {metadata_name}"
                )
            producer = metadata.get("producer")
            if not isinstance(producer, dict):
                raise ArtifactError(
                    f"prepared-model producer metadata is invalid: {metadata_name}"
                )

            expected_target = {
                "portable": True,
                "word_bits": 64,
                "endianness": "little",
                "target_triple": "symjit-storage-v3-portable",
                "cpu_features": [],
            }
            if metadata.get("target") != expected_target:
                raise ArtifactError(
                    f"prepared-model target class is invalid: {metadata_name}"
                )

            bundle = entries[bundle_name]
            claimed_size = metadata.get("bundle_size")
            claimed_digest = metadata.get("bundle_sha256")
            if (
                not bundle
                or type(claimed_size) is not int
                or claimed_size != len(bundle)
                or not isinstance(claimed_digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", claimed_digest)
                or claimed_digest != hashlib.sha256(bundle).hexdigest()
            ):
                raise ArtifactError(
                    f"prepared-model bundle hash/size is invalid: {bundle_name}"
                )


def _sanitized_native_bytes(data: bytes, *, allow_local_rustup: bool) -> bytes:
    scanned = re.sub(
        rb"/rustc/[0-9a-f]+/library/std/src/\.\./\.\./backtrace/",
        b"/rustc/standard-library-backtrace/",
        data,
    )
    scanned = re.sub(
        rb"library/std/src/\.\./\.\./backtrace/",
        b"library/standard-library-backtrace/",
        scanned,
    )
    scanned = re.sub(
        rb"/(?:Users|home)/runner/work/rust/rust/",
        b"/rust/toolchain-source/",
        scanned,
    )
    scanned = re.sub(
        rb"/(?:private/)?var/folders/[^/\x00]+/[^/\x00]+/T/"
        rb"pyamplicol-build-[^/\x00]+/cargo-home/registry/src/",
        b"/cargo/registry/src/",
        scanned,
    )
    scanned = re.sub(
        rb"/(?:tmp|var/tmp)/pyamplicol-build-[^/\x00]+/"
        rb"cargo-home/registry/src/",
        b"/cargo/registry/src/",
        scanned,
    )
    if allow_local_rustup:
        scanned = re.sub(
            rb"/(?:Users|home)/[^/\x00]+/\.rustup/toolchains/[^/\x00]+/"
            rb"lib/rustlib/src/rust/library/",
            b"/rust/sysroot/library/",
            scanned,
        )
    return scanned


def _contains_forbidden_path(data: bytes, marker: bytes) -> bool:
    """Return whether *marker* occurs as the forbidden path it represents.

    Require every absolute marker to occupy a complete path token. A direct
    substring check would misclassify deterministic remapped paths such as
    ``/pyamplicol/checkout/.artifacts/dev-install/tmp/...`` as host ``/tmp``
    paths, and a manylinux checkout rooted at ``/io`` would collide with
    ordinary ``.../src/io/...`` paths or arbitrary native instruction bytes.
    Real absolute paths remain rejected at token boundaries.
    """

    needle = marker.lower()
    if not needle:
        return False
    lowered = data.lower()
    offset = 0
    while (index := lowered.find(needle, offset)) >= 0:
        starts_token = index == 0 or lowered[index - 1] not in _PATH_TOKEN_BYTES
        end = index + len(needle)
        ends_token = (
            needle.endswith((b"/", b"\\"))
            or end == len(lowered)
            or lowered[end] not in _PATH_TOKEN_BYTES
            or lowered[end] in b"/\\"
        )
        if starts_token and ends_token:
            return True
        offset = index + 1
    return False


def _native_forbidden_path_markers() -> tuple[bytes, ...]:
    """Use directory prefixes for the dynamic checkout root in native bytes."""

    root = _REPOSITORY_PATH_MARKER.rstrip(b"/\\")
    fixed = tuple(
        marker
        for marker in _FORBIDDEN_PATH_MARKERS
        if marker != _REPOSITORY_PATH_MARKER
    )
    return (*fixed, root + b"/", root + b"\\")


def _scan_embedded_paths(
    entries: dict[str, bytes],
    *,
    allow_local_rustup: bool = False,
    allow_candidate_source_checkout: bool = False,
) -> None:
    for name, data in entries.items():
        scanned = data
        if allow_candidate_source_checkout and name == "pyamplicol/_build_info.json":
            build_info = _json_object(entries, name)
            build_info.pop("source_checkout", None)
            scanned = json.dumps(build_info, sort_keys=True).encode("utf-8")
        native_member = name.lower().endswith(_NATIVE_MEMBER_SUFFIXES)
        if native_member:
            scanned = _sanitized_native_bytes(
                data, allow_local_rustup=allow_local_rustup
            )
        lowered = scanned.lower()
        markers = (
            _native_forbidden_path_markers()
            if native_member
            else _FORBIDDEN_PATH_MARKERS
        )
        if name.lower().endswith((".dylib", ".pyd", ".so")):
            markers += _FORBIDDEN_NATIVE_RELATIVE_MARKERS
        for marker in markers:
            if _contains_forbidden_path(lowered, marker):
                raise ArtifactError(
                    f"wheel member {name} embeds non-relocatable path marker "
                    f"{os.fsdecode(marker)!r}"
                )


def _validate_sdk(
    entries: dict[str, bytes],
    *,
    version: str,
    rust_target: str,
) -> None:
    missing = sorted(_REQUIRED_SDK_PATHS - set(entries))
    if missing:
        raise ArtifactError("wheel is missing SDK resources: " + ", ".join(missing))
    metadata = _json_object(entries, "pyamplicol/_sdk/metadata.json")
    link = _json_object(entries, "pyamplicol/_sdk/link.json")
    if metadata.get("schema_version") != 1 or link.get("schema_version") != 1:
        raise ArtifactError("unsupported SDK metadata schema")
    if metadata.get("abi_version") != 1:
        raise ArtifactError("Rusticol SDK must provide C ABI version 1")
    if metadata.get("target") != rust_target or link.get("target") != rust_target:
        raise ArtifactError("SDK target does not match the wheel platform")
    metadata_version = str(metadata.get("version", ""))
    if version != metadata_version:
        raise ArtifactError("SDK version does not match wheel metadata")
    archive = PurePosixPath(str(metadata.get("archive", "")))
    if archive.is_absolute() or any(part in {"", ".", ".."} for part in archive.parts):
        raise ArtifactError("SDK archive path must be relative and confined")
    archive_name = (PurePosixPath("pyamplicol/_sdk") / archive).as_posix()
    if archive_name != "pyamplicol/_sdk/lib/librusticol_capi.a":
        raise ArtifactError(f"unexpected SDK archive path: {archive}")
    rust_source = PurePosixPath(str(metadata.get("rust_source", "")))
    if rust_source.is_absolute() or any(
        part in {"", ".", ".."} for part in rust_source.parts
    ):
        raise ArtifactError("SDK Rust source path must be relative and confined")
    rust_source_name = (PurePosixPath("pyamplicol/_sdk") / rust_source).as_posix()
    if rust_source_name != "pyamplicol/_sdk/rust/rusticol.rs":
        raise ArtifactError(f"unexpected SDK Rust source path: {rust_source}")
    typed_values: dict[str, list[str]] = {}
    for key in ("system_libraries", "frameworks"):
        values = link.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ArtifactError(f"SDK {key} must be a list of nonempty strings")
        if len(values) != len(set(values)):
            raise ArtifactError(f"SDK {key} contains duplicate entries")
        if any(
            "/" in value
            or "\\" in value
            or value.startswith("-")
            or any(character.isspace() for character in value)
            for value in values
        ):
            raise ArtifactError(f"SDK {key} contains a path")
        typed_values[key] = values
    unexpected_libraries = (
        set(typed_values["system_libraries"]) - _SYSTEM_LIBRARIES[rust_target]
    )
    allowed_frameworks = _MACOS_FRAMEWORKS if "apple-darwin" in rust_target else set()
    unexpected_frameworks = set(typed_values["frameworks"]) - allowed_frameworks
    if unexpected_libraries or unexpected_frameworks:
        raise ArtifactError(
            "SDK link metadata is not target-allowlisted: "
            f"libraries={sorted(unexpected_libraries)}, "
            f"frameworks={sorted(unexpected_frameworks)}"
        )


def _validate_selftest_fixture(
    entries: dict[str, bytes], *, version: str, rust_target: str
) -> None:
    root = PurePosixPath("pyamplicol/assets/selftest")
    fixture_members = [
        PurePosixPath(name)
        for name in entries
        if PurePosixPath(name).is_relative_to(root)
    ]
    targets = {
        path.parts[len(root.parts)]
        for path in fixture_members
        if len(path.parts) > len(root.parts)
    }
    if targets != {rust_target}:
        raise ArtifactError(
            "wheel must contain only its target self-test fixture; "
            f"expected {rust_target!r}, found {sorted(targets)!r}"
        )
    prefix = (root / rust_target).as_posix()
    expected_name = f"{prefix}/expected.json"
    manifest_name = f"{prefix}/artifact/artifact.json"
    expected = _json_object(entries, expected_name)
    manifest = _json_object(entries, manifest_name)
    if (
        expected.get("schema_version") != 1
        or expected.get("target") != rust_target
        or expected.get("artifact_path") != "artifact"
    ):
        raise ArtifactError("wheel self-test expectation is invalid")
    if manifest.get("schema_version") != 3:
        raise ArtifactError("wheel self-test artifact must use schema v3")
    producer = manifest.get("producer")
    runtime = manifest.get("runtime")
    if not isinstance(producer, dict) or not isinstance(runtime, dict):
        raise ArtifactError("wheel self-test producer/runtime metadata is invalid")
    try:
        validate_portable_artifact_capabilities(
            manifest,
            context="wheel self-test artifact manifest",
        )
    except RuntimeError as error:
        raise ArtifactError(
            f"wheel self-test Arena artifact contract is invalid: {error}"
        ) from error
    target = producer.get("target")
    if (
        producer.get("version") != version
        or runtime.get("engine_version") != version
        or not isinstance(target, dict)
        or target.get("triple") != rust_target
    ):
        raise ArtifactError("wheel self-test artifact version/target is inconsistent")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise ArtifactError("wheel self-test payload inventory is invalid")
    abis = _dependency_lock().get("abis")
    expected_compiled_model_schema = (
        abis.get("compiled_model") if isinstance(abis, dict) else None
    )
    if not isinstance(expected_compiled_model_schema, int):
        raise ArtifactError("dependency lock has no compiled-model schema version")
    compiled_model = _json_object(
        entries, f"{prefix}/artifact/model/compiled-model.json"
    )
    model = manifest.get("model")
    producer_versions = producer.get("versions")
    if (
        not isinstance(model, dict)
        or model.get("compiled_schema_version") != expected_compiled_model_schema
        or not isinstance(producer_versions, dict)
        or producer_versions.get("compiled_model") != expected_compiled_model_schema
    ):
        raise ArtifactError(
            "wheel self-test producer/model metadata does not match release "
            f"compiled-model schema {expected_compiled_model_schema}"
        )
    if compiled_model.get("kind") != _COMPILED_MODEL_KIND:
        raise ArtifactError("wheel self-test compiled model kind is invalid")
    if compiled_model.get("schema_version") != expected_compiled_model_schema:
        raise ArtifactError(
            "wheel self-test compiled model does not match release schema "
            f"{expected_compiled_model_schema}"
        )
    compiled_producer = compiled_model.get("producer")
    compiled_source = compiled_model.get("source")
    if not isinstance(compiled_producer, dict) or not isinstance(compiled_source, dict):
        raise ArtifactError(
            "wheel self-test compiled model provenance metadata is invalid"
        )
    model_compiler_version = compiled_model.get("model_compiler_version")
    producer_model_compiler_version = compiled_producer.get("model_compiler_version")
    if (
        not isinstance(model_compiler_version, int)
        or isinstance(model_compiler_version, bool)
        or not isinstance(producer_model_compiler_version, int)
        or isinstance(producer_model_compiler_version, bool)
        or producer_model_compiler_version != model_compiler_version
    ):
        raise ArtifactError(
            "wheel self-test model compiler version does not match producer"
        )
    if (
        compiled_producer.get("compiled_model_schema_version")
        != expected_compiled_model_schema
    ):
        raise ArtifactError(
            "wheel self-test compiled model producer does not match release "
            f"schema {expected_compiled_model_schema}"
        )
    if compiled_producer.get("pyamplicol") != version:
        raise ArtifactError(
            "wheel self-test compiled model producer does not match wheel version"
        )
    if compiled_source.get("kind") != _BUILTIN_MODEL_SOURCE_KIND:
        raise ArtifactError("wheel self-test compiled model source is not built-in-sm")
    tagged_payloads = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("target") is not None
    ]
    if not tagged_payloads or any(
        not isinstance(payload.get("target"), dict)
        or payload["target"].get("triple") != rust_target
        or payload["target"].get("cpu_features") != []
        for payload in tagged_payloads
    ):
        raise ArtifactError("wheel self-test evaluator target metadata is invalid")
    validate_artifact_identity_contract(
        manifest,
        context="wheel self-test artifact",
    )
    claimed_id = manifest.get("artifact_id")
    if claimed_id != _artifact_id(manifest):
        raise ArtifactError("wheel self-test artifact identity is invalid")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise ArtifactError("wheel self-test artifact has no payload inventory")
    declared: set[str] = set()
    direct_symjit = 0
    execution_manifests: list[tuple[str, str]] = []
    artifact_prefix = f"{prefix}/artifact/"
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ArtifactError(f"wheel self-test payload {index} is invalid")
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise ArtifactError(f"wheel self-test payload {index} has no path")
        path = PurePosixPath(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ArtifactError(f"wheel self-test payload path is unsafe: {relative}")
        if relative in declared:
            raise ArtifactError(f"wheel self-test payload is repeated: {relative}")
        declared.add(relative)
        member = f"{artifact_prefix}{relative}"
        data = entries.get(member)
        if data is None:
            raise ArtifactError(f"wheel self-test payload is missing: {relative}")
        if (
            payload.get("size_bytes") != len(data)
            or payload.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ArtifactError(
                f"wheel self-test payload digest is invalid: {relative}"
            )
        if relative.endswith(".evaluator.bin"):
            raise ArtifactError(
                "wheel self-test must not retain Symbolica fallback state"
            )
        if payload.get("media_type") == "application/vnd.symjit.application":
            direct_symjit += 1
        if payload.get("role") == "evaluator-manifest" and relative.endswith(
            "/execution.json"
        ):
            execution_manifests.append((relative, member))
    container = manifest.get("extensions", {}).get("evaluator_payload_container")
    if container is not None:
        direct_symjit += _validate_selftest_evaluator_container(
            entries,
            artifact_prefix=artifact_prefix,
            container=container,
        )
    actual = {
        name.removeprefix(artifact_prefix)
        for name in entries
        if name.startswith(artifact_prefix) and name != manifest_name
    }
    if actual != declared:
        raise ArtifactError("wheel self-test artifact payload inventory is incomplete")
    missing_api_payloads = sorted(_REQUIRED_SELFTEST_API_PAYLOADS - declared)
    if missing_api_payloads:
        raise ArtifactError(
            "wheel self-test artifact is missing five-language API payloads: "
            + ", ".join(missing_api_payloads)
        )
    if direct_symjit == 0:
        raise ArtifactError("wheel self-test artifact has no direct SymJIT application")
    if not execution_manifests:
        raise ArtifactError(
            "wheel self-test artifact has no compiled execution manifest"
        )
    for relative, member in execution_manifests:
        execution = _json_object(entries, member)
        try:
            validate_portable_execution_manifest(
                execution,
                context=f"wheel self-test execution manifest {relative}",
            )
        except RuntimeError as error:
            raise ArtifactError(
                f"wheel self-test Arena execution contract is invalid: {error}"
            ) from error


@lru_cache(maxsize=1)
def _evaluator_container_codec() -> Any:
    """Load the source-tree pacbin codec without importing an installed package."""

    source = ROOT / "src" / "pyamplicol" / "generation" / "evaluator_container.py"
    name = "_pyamplicol_release_evaluator_container"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ArtifactError("could not load the evaluator container codec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_selftest_evaluator_container(
    entries: dict[str, bytes],
    *,
    artifact_prefix: str,
    container: object,
) -> int:
    if not isinstance(container, dict):
        raise ArtifactError("wheel self-test evaluator container metadata is invalid")
    relative = container.get("path")
    if not isinstance(relative, str):
        raise ArtifactError("wheel self-test evaluator container has no path")
    data = entries.get(f"{artifact_prefix}{relative}")
    if data is None:
        raise ArtifactError("wheel self-test evaluator container is missing")

    codec = _evaluator_container_codec()
    try:
        with codec.PacbinReader.open(io.BytesIO(data)) as reader:
            index = reader.index
            expected = {
                "member_count": len(index.members),
                "unpacked_size_bytes": sum(member.length for member in index.members),
                "index_sha256": index.index_sha256,
            }
            if any(container.get(key) != value for key, value in expected.items()):
                raise ArtifactError(
                    "wheel self-test evaluator container metadata is inconsistent"
                )
            if any(
                member.kind is codec.PacbinMemberKind.SYMBOLICA_EXACT_STATE
                for member in index.members
            ):
                raise ArtifactError(
                    "wheel self-test must not retain packed Symbolica fallback state"
                )
            return sum(
                member.kind is codec.PacbinMemberKind.SYMJIT_APPLICATION
                for member in index.members
            )
    except ArtifactError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactError(
            f"wheel self-test evaluator container is invalid: {error}"
        ) from error


def _validate_wheel_inventory(
    entries: dict[str, bytes],
    *,
    metadata_name: str,
    wheel_name: str,
    record_name: str,
    extension_name: str,
    rust_target: str,
    mode: str,
) -> None:
    _validate_prepared_model_assets(
        entries,
        prefix=_PREPARED_MODEL_WHEEL_PREFIX,
        mode=mode,
    )
    expected = set(_REQUIRED_WHEEL_PACKAGE_MEMBERS)
    expected.add(extension_name)
    selftest_root = f"pyamplicol/assets/selftest/{rust_target}/"
    expected.update(name for name in entries if name.startswith(selftest_root))

    dist_info = metadata_name.removesuffix("METADATA")
    expected.update(
        {
            metadata_name,
            wheel_name,
            record_name,
            f"{dist_info}entry_points.txt",
        }
    )
    expected.update(
        f"{dist_info}licenses/{relative}" for relative in _required_legal_files()
    )

    repair_members: set[str] = set()
    for name in entries:
        root = PurePosixPath(name).parts[0]
        if root not in _ALLOWED_REPAIR_ROOTS:
            continue
        basename = PurePosixPath(name).name
        if not re.search(r"(?:\.dylib|\.dll|\.so(?:\.[0-9]+)*)$", basename):
            raise ArtifactError(
                f"wheel repair-library root contains a non-library member: {name}"
            )
        repair_members.add(name)

    actual = set(entries) - repair_members
    missing = sorted(expected - actual)
    if missing:
        raise ArtifactError(
            "wheel is missing required package members: " + ", ".join(missing)
        )


def _host_matches(target: str) -> bool:
    machine = platform.machine().lower()
    if target == "macosx_11_0_arm64":
        return sys.platform == "darwin" and machine in {"arm64", "aarch64"}
    if target == "macosx_11_0_x86_64":
        return sys.platform == "darwin" and machine in {"x86_64", "amd64"}
    return sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}


def _command_output(command: list[str], *, cwd: Path) -> str:
    completed = run(command, cwd=cwd, capture_output=True)
    return completed.stdout + completed.stderr


def _reject_native_paths(output: str, description: str) -> None:
    encoded = os.fsencode(output)
    for marker in _FORBIDDEN_PATH_MARKERS:
        decoded = os.fsdecode(marker).lower()
        if _contains_forbidden_path(encoded, marker):
            raise ArtifactError(f"{description} embeds non-relocatable path {decoded}")


def _reject_native_path_bytes(data: bytes, description: str) -> None:
    for marker in _native_forbidden_path_markers():
        if _contains_forbidden_path(data, marker):
            raise ArtifactError(
                f"{description} embeds non-relocatable path "
                f"{os.fsdecode(marker).lower()}"
            )


def _scan_capi_archive(path: Path, *, allow_local_rustup: bool) -> bool:
    """Validate relocation paths and authoritative undefined-symbol linkage.

    The archive link probe and ``nm -u`` inspection own the Python-linkage
    invariant. Searching arbitrary compressed archive bytes for short strings
    such as ``pyo3`` can match unrelated binary data and reject a valid wheel.
    """

    archive = path.read_bytes()
    scanned = _sanitized_native_bytes(archive, allow_local_rustup=allow_local_rustup)
    _reject_native_path_bytes(scanned, "Rusticol C API archive")

    candidates: list[str] = []
    for candidate in (
        os.environ.get("LLVM_NM"),
        shutil.which("llvm-nm"),
        shutil.which("nm"),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for nm in candidates:
        completed = subprocess.run(
            [nm, "-u", str(path)],
            cwd=path.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        forbidden_symbols: list[str] = []
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) < 2 or fields[-2] not in {"U", "u", "W", "w", "V", "v"}:
                continue
            symbol = fields[-1]
            normalized = symbol.removeprefix("_").lower()
            if normalized.startswith(_FORBIDDEN_CAPI_UNDEFINED_PREFIXES):
                forbidden_symbols.append(symbol)
        if forbidden_symbols:
            raise ArtifactError(
                "Rusticol C API archive has undefined Python/PyO3/NumPy symbols: "
                + ", ".join(sorted(set(forbidden_symbols)))
            )
        if completed.returncode == 0:
            return True
    return False


def _native_c_compiler(rust_target: str) -> list[str]:
    configured = os.environ.get("CC")
    if configured:
        command = shlex.split(configured)
        if command:
            return command
    names = ("clang", "cc") if "apple-darwin" in rust_target else ("cc", "clang")
    for name in names:
        compiler = shutil.which(name)
        if compiler:
            return [compiler]
    raise ArtifactError("a C compiler is required for the native SDK audit")


def _validate_capi_linkage(
    entries: dict[str, bytes],
    *,
    archive: Path,
    directory: Path,
    rust_target: str,
) -> None:
    header_data = entries["pyamplicol/_sdk/include/rusticol.h"]
    exports = tuple(
        dict.fromkeys(re.findall(rb"\b(rusticol_[a-z0-9_]+)\s*\(", header_data))
    )
    decoded_exports = tuple(item.decode("ascii") for item in exports)
    if not decoded_exports or "rusticol_abi_version" not in decoded_exports:
        raise ArtifactError("wheel Rusticol C header has no complete public ABI")
    header = directory / "rusticol.h"
    source = directory / "probe.c"
    binary = directory / "probe"
    header.write_bytes(header_data)
    table = ",\n".join(f"    (rusticol_probe_fn){name}" for name in decoded_exports)
    source.write_text(
        f"""\
#include <stdint.h>
#include \"rusticol.h\"

typedef void (*rusticol_probe_fn)(void);
static rusticol_probe_fn volatile rusticol_api[] = {{
{table}
}};

int main(void) {{
    uint32_t (*abi_version)(void) = (uint32_t (*)(void))rusticol_api[0];
    return abi_version() == RUSTICOL_ABI_VERSION ? 0 : 1;
}}
""",
        encoding="utf-8",
    )
    link = _json_object(entries, "pyamplicol/_sdk/link.json")
    if link.get("target") != rust_target:
        raise ArtifactError("wheel SDK link metadata targets the wrong platform")
    flags = [f"-l{item}" for item in link.get("system_libraries", [])]
    for framework in link.get("frameworks", []):
        flags.extend(("-framework", str(framework)))
    completed = subprocess.run(
        [
            *_native_c_compiler(rust_target),
            "-std=c11",
            "-O0",
            "-I",
            str(directory),
            str(source),
            str(archive),
            *flags,
            "-o",
            str(binary),
        ],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise ArtifactError(
            "wheel Rusticol archive failed the complete C ABI link probe" + suffix
        )
    completed = subprocess.run(
        [str(binary)],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ArtifactError("wheel Rusticol archive failed the C ABI execution probe")


def _scan_static_runtime_families(path: Path, description: str) -> None:
    lowered = path.read_bytes().lower()
    found = [
        marker
        for marker in _FORBIDDEN_STATIC_RUNTIME_MARKERS
        if marker.encode() in lowered
    ]
    if found:
        raise ArtifactError(
            f"{description} contains a forbidden static arithmetic runtime: "
            + ", ".join(found)
        )


def _allowed_runtime_path(value: str) -> bool:
    return value.startswith(
        ("$ORIGIN", "@loader_path", "@rpath", "/usr/lib/", "/System/Library/", "/lib/")
    )


def _scan_extension(path: Path, target: str) -> None:
    file_output = _command_output(["file", str(path)], cwd=path.parent).lower()
    if target.endswith("arm64") and not any(
        marker in file_output for marker in ("arm64", "aarch64")
    ):
        raise ArtifactError(
            "extension binary does not contain the required arm64 slice"
        )
    if target.endswith("x86_64") and not any(
        marker in file_output for marker in ("x86-64", "x86_64")
    ):
        raise ArtifactError("extension binary is not x86_64")

    if target.startswith("macosx"):
        libraries = _command_output(["otool", "-L", str(path)], cwd=path.parent)
        for line in libraries.splitlines()[1:]:
            dependency = line.strip().split(" ", 1)[0]
            if dependency and not _allowed_runtime_path(dependency):
                raise ArtifactError(f"extension links a non-system path: {dependency}")
            if "python" in dependency.lower():
                raise ArtifactError(f"extension links Python directly: {dependency}")
        load_commands = _command_output(["otool", "-l", str(path)], cwd=path.parent)
        lines = load_commands.splitlines()
        for index, line in enumerate(lines):
            if line.strip() == "cmd LC_RPATH":
                for candidate in lines[index + 1 : index + 5]:
                    match = re.match(r"\s*path\s+(\S+)", candidate)
                    if match and not _allowed_runtime_path(match.group(1)):
                        raise ArtifactError(
                            f"extension has a non-system RPATH: {match.group(1)}"
                        )
                        break
        inspected = (libraries + load_commands).replace(str(path), "<extension>")
        _reject_native_paths(inspected, "extension load metadata")
        return

    dynamic = _command_output(["readelf", "-d", str(path)], cwd=path.parent)
    for line in dynamic.splitlines():
        if "(RPATH)" in line or "(RUNPATH)" in line:
            match = re.search(r"\[([^]]*)\]", line)
            if match:
                for value in match.group(1).split(":"):
                    if value and not _allowed_runtime_path(value):
                        raise ArtifactError(
                            f"extension has a non-system RPATH: {value}"
                        )
        if "(NEEDED)" in line:
            match = re.search(r"\[([^]]+)\]", line)
            if match and ("/" in match.group(1) or "python" in match.group(1).lower()):
                raise ArtifactError(
                    f"extension has a forbidden dynamic dependency: {match.group(1)}"
                )
    _reject_native_paths(
        dynamic.replace(str(path), "<extension>"), "extension dynamic metadata"
    )


def _native_scan(
    entries: dict[str, bytes],
    extension: str,
    target: str,
    *,
    allow_local_rustup: bool,
) -> None:
    with external_temporary_directory("pyamplicol-native-audit-") as temporary:
        extension_path = temporary / Path(extension).name
        archive_path = temporary / "librusticol_capi.a"
        extension_path.write_bytes(entries[extension])
        archive_path.write_bytes(entries["pyamplicol/_sdk/lib/librusticol_capi.a"])
        _scan_extension(extension_path, target)
        _scan_capi_archive(archive_path, allow_local_rustup=allow_local_rustup)
        link = _json_object(entries, "pyamplicol/_sdk/link.json")
        rust_target = str(link.get("target", ""))
        _validate_capi_linkage(
            entries,
            archive=archive_path,
            directory=temporary,
            rust_target=rust_target,
        )
        _scan_static_runtime_families(extension_path, "Rusticol Python extension")
        _scan_static_runtime_families(archive_path, "Rusticol C API archive")


def audit_wheel(
    path: Path,
    *,
    mode: str,
    native_scan: bool | None = None,
) -> WheelReport:
    """Validate a wheel's metadata, content inventory, SDK, and native binaries."""

    path = path.resolve()
    if mode not in {"candidate", "release"}:
        raise ArtifactError(f"unsupported wheel audit mode: {mode}")
    size = path.stat().st_size
    if size > MAX_WHEEL_BYTES:
        raise ArtifactError(
            f"compressed wheel is {size} bytes; limit is {MAX_WHEEL_BYTES} bytes"
        )
    filename_version, python_tag, abi_tag, platform_tag = _parse_wheel_filename(path)
    if python_tag != EXPECTED_PYTHON_TAG or abi_tag != EXPECTED_ABI_TAG:
        raise ArtifactError(
            f"wheel must use {EXPECTED_PYTHON_TAG}-{EXPECTED_ABI_TAG}, found "
            f"{python_tag}-{abi_tag}"
        )
    target, rust_target = _canonical_target(platform_tag, mode=mode)
    entries = _wheel_entries(path)
    _validate_wheel_resource_layout(entries)
    metadata_name = _single_name(entries, ".dist-info/METADATA")
    wheel_name = _single_name(entries, ".dist-info/WHEEL")
    record_name = _single_name(entries, ".dist-info/RECORD")
    metadata = _message(entries[metadata_name], "METADATA")
    name = canonicalize_name(str(metadata.get("Name", "")))
    version = str(metadata.get("Version", ""))
    if name != EXPECTED_DISTRIBUTION:
        raise ArtifactError(f"wheel metadata Name is {name!r}, expected pyamplicol")
    if filename_version != version:
        raise ArtifactError(
            "wheel filename version disagrees with metadata: "
            f"{filename_version!r} != {version!r}"
        )
    raw_requirements = list(metadata.get_all("Requires-Dist", []))
    _reject_direct_requirements(raw_requirements)
    dependency_lock = _dependency_lock()
    _validate_runtime_requirements(raw_requirements, dependency_lock, mode=mode)
    _validate_legal_members(entries, metadata_name, metadata)
    if str(metadata.get("License-Expression", "")) != "0BSD":
        raise ArtifactError("wheel metadata must declare License-Expression: 0BSD")
    if str(metadata.get("Requires-Python", "")) != ">=3.11":
        raise ArtifactError("wheel metadata must require Python >=3.11")

    build_info_names = [name for name in entries if name.endswith("/_build_info.json")]
    if mode == "candidate":
        candidate_pattern = (
            re.escape(EXPECTED_RELEASE_VERSION) + r"\.dev0\+candidate\.[0-9a-f]{12}"
        )
        if re.fullmatch(candidate_pattern, version) is None:
            raise ArtifactError(f"candidate wheel has invalid version: {version}")
        if build_info_names != ["pyamplicol/_build_info.json"]:
            raise ArtifactError("candidate wheel must contain one _build_info.json")
        build_info = _json_object(entries, build_info_names[0])
        if build_info.get("selftest_fixture_bootstrap") is not False:
            raise ArtifactError(
                "candidate self-test fixture bootstrap wheels are not deployable"
            )
        if (
            build_info.get("publishable") is not False
            or build_info.get("version") != version
        ):
            raise ArtifactError("candidate build marker is missing or inconsistent")
    else:
        if version != EXPECTED_RELEASE_VERSION:
            raise ArtifactError(f"release wheel has invalid version: {version}")
        if build_info_names != ["pyamplicol/_build_info.json"]:
            raise ArtifactError("release wheel must contain one _build_info.json")
        build_info = _json_object(entries, build_info_names[0])
        source_revision = build_info.get("source_revision")
        if (
            set(build_info) != _RELEASE_BUILD_INFO_FIELDS
            or build_info.get("schema_version") != 1
            or build_info.get("publishable") is not True
            or build_info.get("selftest_fixture_bootstrap") is not False
            or build_info.get("version") != version
            or not isinstance(source_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
        ):
            raise ArtifactError("release build marker is missing or inconsistent")
    wheel_metadata = _message(entries[wheel_name], "WHEEL")
    if str(wheel_metadata.get("Root-Is-Purelib", "")).lower() != "false":
        raise ArtifactError("binary wheel must set Root-Is-Purelib: false")
    wheel_tags = list(wheel_metadata.get_all("Tag", []))
    if not wheel_tags:
        raise ArtifactError("WHEEL metadata contains no Tag fields")
    for tag in wheel_tags:
        fields = tag.rsplit("-", 2)
        if len(fields) != 3 or fields[0] != python_tag or fields[1] != abi_tag:
            raise ArtifactError(f"WHEEL metadata has an unexpected tag: {tag}")
        tag_target, _ = _canonical_target(fields[2], mode=mode)
        if tag_target != target:
            raise ArtifactError(f"WHEEL tag disagrees with filename: {tag}")

    extensions = sorted(
        name
        for name in entries
        if re.fullmatch(r"pyamplicol/_rusticol(?:\.[^.]+)*\.so", name)
    )
    if len(extensions) != 1 or "abi3" not in Path(extensions[0]).name:
        raise ArtifactError(
            "wheel must contain exactly one pyamplicol/_rusticol*.abi3.so"
        )
    _validate_sdk(
        entries,
        version=version,
        rust_target=rust_target,
    )
    _validate_selftest_fixture(entries, version=version, rust_target=rust_target)
    _validate_wheel_inventory(
        entries,
        metadata_name=metadata_name,
        wheel_name=wheel_name,
        record_name=record_name,
        extension_name=extensions[0],
        rust_target=rust_target,
        mode=mode,
    )
    _validate_record(entries, record_name)
    _scan_embedded_paths(
        entries,
        allow_local_rustup=mode == "candidate",
        allow_candidate_source_checkout=mode == "candidate",
    )

    perform_native_scan = _host_matches(target) if native_scan is None else native_scan
    if perform_native_scan:
        if not _host_matches(target):
            raise ArtifactError(f"cannot perform native scan for {target} on this host")
        _native_scan(
            entries,
            extensions[0],
            target,
            allow_local_rustup=mode == "candidate",
        )
    return WheelReport(
        filename=path.name,
        version=version,
        size=size,
        sha256=sha256(path),
        python_tag=python_tag,
        abi_tag=abi_tag,
        target=target,
        rust_target=rust_target,
        native_scan=perform_native_scan,
    )


def _validate_sdist_inventory(members: set[str]) -> None:
    required = {*REQUIRED_SDIST_MEMBERS, "PKG-INFO"}
    missing = sorted(required - members)
    if missing:
        raise ArtifactError("sdist is missing required files: " + ", ".join(missing))
    forbidden = sorted(
        name
        for name in members
        if name in _FORBIDDEN_SDIST_MEMBERS
        or name.startswith(_FORBIDDEN_SDIST_PREFIXES)
    )
    if forbidden:
        raise ArtifactError(
            "sdist contains contributor-only dependency inputs: " + ", ".join(forbidden)
        )


def _release_sdist_symjit_contract(
    relative_files: Mapping[str, Path],
) -> dict[str, str]:
    """Validate the published SymJIT version against ordinary Cargo resolution."""

    lock_member = "dependencies/release-lock.toml"
    try:
        with relative_files[lock_member].open("rb") as stream:
            release_lock = tomllib.load(stream)
        symjit = release_lock["symjit"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(
            f"sdist has an invalid release SymJIT source contract: {error}"
        ) from error
    if (
        not isinstance(symjit, dict)
        or set(symjit) != _RELEASE_SYMJIT_FIELDS
        or not all(
            isinstance(symjit.get(field), str) and symjit[field]
            for field in _RELEASE_SYMJIT_FIELDS
        )
    ):
        raise ArtifactError("sdist has an invalid published SymJIT version contract")

    expected_source = "registry+https://github.com/rust-lang/crates.io-index"
    try:
        with relative_files["Cargo.lock"].open("rb") as stream:
            cargo_lock = tomllib.load(stream)
        packages = cargo_lock["package"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ArtifactError(f"sdist has an invalid Cargo.lock: {error}") from error
    matches = [
        package
        for package in packages
        if isinstance(package, dict) and package.get("name") == "symjit"
    ]
    if (
        len(matches) != 1
        or matches[0].get("version") != symjit["version"]
        or matches[0].get("source") != expected_source
        or re.fullmatch(r"[0-9a-f]{64}", str(matches[0].get("checksum", ""))) is None
    ):
        raise ArtifactError(
            "sdist Cargo.lock does not use the published SymJIT dependency"
        )
    return {field: str(symjit[field]) for field in _RELEASE_SYMJIT_FIELDS}


def audit_sdist(path: Path, *, mode: str) -> SdistReport:
    """Validate a release sdist's structure and dependency boundary."""

    path = path.resolve()
    if mode != "release":
        raise ArtifactError("candidate source distributions are not supported")
    with external_temporary_directory("pyamplicol-sdist-audit-") as temporary:
        source = safe_extract_sdist(path, temporary / "source")
        relative_files = {
            item.relative_to(source).as_posix(): item
            for item in source.rglob("*")
            if item.is_file()
        }
        _release_sdist_symjit_contract(relative_files)
        _validate_sdist_inventory(set(relative_files))
        _validate_prepared_model_assets(
            {name: item.read_bytes() for name, item in relative_files.items()},
            prefix=_PREPARED_MODEL_SDIST_PREFIX,
            mode="release",
        )
        cargo = relative_files["Cargo.toml"].read_text(encoding="utf-8")
        match = re.search(r'(?m)^version = "([^"]+)"$', cargo)
        if match is None:
            raise ArtifactError("sdist Cargo.toml has no workspace version")
        cargo_version = match.group(1)
        if cargo_version != EXPECTED_RELEASE_VERSION:
            raise ArtifactError("release sdist contains a non-release version")
        version = EXPECTED_RELEASE_VERSION

        try:
            build_info = json.loads(
                relative_files["src/pyamplicol/_build_info.json"].read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, TypeError, ValueError) as error:
            raise ArtifactError("release sdist build marker is unreadable") from error
        source_revision = (
            build_info.get("source_revision") if isinstance(build_info, dict) else None
        )
        if (
            not isinstance(build_info, dict)
            or set(build_info) != _RELEASE_BUILD_INFO_FIELDS
            or build_info.get("schema_version") != 1
            or build_info.get("publishable") is not True
            or build_info.get("selftest_fixture_bootstrap") is not False
            or build_info.get("version") != version
            or not isinstance(source_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
        ):
            raise ArtifactError("release sdist build marker is missing or inconsistent")

        pyproject = relative_files["pyproject.toml"].read_text(encoding="utf-8")
        try:
            pyproject_data = tomllib.loads(pyproject)
        except tomllib.TOMLDecodeError as error:
            raise ArtifactError(
                f"sdist contains invalid pyproject.toml: {error}"
            ) from error
        _reject_direct_requirements(
            _project_requirements(pyproject_data),
            description="sdist project metadata",
        )
        scan_names = {
            name
            for name in relative_files
            if name
            in {
                "Cargo.lock",
                "Cargo.toml",
                "dependencies/release-lock.toml",
                "pyproject.toml",
                "src/pyamplicol/_build_info.json",
            }
            or ("MANIFEST" in Path(name).name and name.endswith(".toml"))
            or (name.startswith("tests/fixtures/reference/") and name.endswith(".json"))
            or name.startswith(".cargo/")
        }
        for name in sorted(scan_names):
            item = relative_files[name]
            data = item.read_bytes()
            for marker in _FORBIDDEN_PATH_MARKERS:
                if _contains_forbidden_path(data, marker):
                    raise ArtifactError(
                        f"sdist member {name} embeds non-relocatable path marker "
                        f"{os.fsdecode(marker)!r}"
                    )
        expected_base = f"pyamplicol-{version}"
        if source.name != expected_base or path.name != f"{expected_base}.tar.gz":
            raise ArtifactError(
                "sdist filename/root does not match package identity: "
                f"{path.name}, {source.name}"
            )
    return SdistReport(
        filename=path.name,
        version=version,
        size=path.stat().st_size,
        sha256=sha256(path),
    )


__all__ = [
    "MAX_WHEEL_BYTES",
    "RELEASE_TARGETS",
    "ArtifactError",
    "SdistReport",
    "WheelReport",
    "audit_sdist",
    "audit_wheel",
]
