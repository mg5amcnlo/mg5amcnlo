# SPDX-License-Identifier: 0BSD
"""Typed parser for the immutable Python runtime artifact lock."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_NAME = re.compile(r"[-_.]+")


def canonicalize_name(value: str) -> str:
    """Return the normalized distribution name used by package indexes."""

    return _CANONICAL_NAME.sub("-", value).lower()


@dataclass(frozen=True, slots=True)
class PythonArtifact:
    """One published wheel admitted by the runtime dependency contract."""

    filename: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PythonPackage:
    """One exact package in the resolved runtime dependency graph."""

    distribution: str
    version: str
    license: str
    direct: bool
    dependencies: tuple[str, ...]
    artifacts: tuple[PythonArtifact, ...]

    @property
    def name(self) -> str:
        return canonicalize_name(self.distribution)


@dataclass(frozen=True, slots=True)
class PythonRuntimeLock:
    """Complete Python runtime graph and its admitted binary artifacts."""

    requires_python: str
    supported_python: tuple[str, ...]
    packages: tuple[PythonPackage, ...]

    @property
    def by_name(self) -> dict[str, PythonPackage]:
        return {package.name: package for package in self.packages}

    @property
    def direct_packages(self) -> tuple[PythonPackage, ...]:
        return tuple(package for package in self.packages if package.direct)


def _required_string(table: dict[str, object], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires a nonempty {key!r} string")
    return value


def _string_list(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{context} must be a list of nonempty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicate entries")
    return result


def _parse_artifacts(raw: object, context: str) -> tuple[PythonArtifact, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{context}.artifacts must be an array of tables")
    artifacts: list[PythonArtifact] = []
    for index, item in enumerate(raw):
        item_context = f"{context}.artifacts[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_context} must be a table")
        filename = _required_string(item, "filename", item_context)
        digest = _required_string(item, "sha256", item_context)
        if filename != Path(filename).name or not filename.endswith(".whl"):
            raise ValueError(f"{item_context}.filename must be a plain wheel filename")
        if _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{item_context}.sha256 must be a lowercase SHA-256")
        artifacts.append(PythonArtifact(filename=filename, sha256=digest))
    filenames = [artifact.filename for artifact in artifacts]
    if filenames != sorted(filenames) or len(filenames) != len(set(filenames)):
        raise ValueError(f"{context}.artifacts must be unique and filename-sorted")
    return tuple(artifacts)


def load_python_runtime_lock(path: Path) -> PythonRuntimeLock:
    """Load and structurally validate a schema-v1 Python runtime lock."""

    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} must use schema_version = 1")
    requires_python = _required_string(payload, "requires_python", str(path))
    supported_python = _string_list(
        payload.get("supported_python"), f"{path}.supported_python"
    )
    if supported_python != tuple(sorted(supported_python)):
        raise ValueError(f"{path}.supported_python must be sorted")
    raw_packages = payload.get("packages")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ValueError(f"{path}.packages must be a nonempty array of tables")

    packages: list[PythonPackage] = []
    for index, item in enumerate(raw_packages):
        context = f"{path}.packages[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{context} must be a table")
        distribution = _required_string(item, "distribution", context)
        version = _required_string(item, "version", context)
        license_expression = _required_string(item, "license", context)
        direct = item.get("direct")
        if not isinstance(direct, bool):
            raise ValueError(f"{context}.direct must be a boolean")
        dependencies = tuple(
            canonicalize_name(value)
            for value in _string_list(
                item.get("dependencies"), f"{context}.dependencies"
            )
        )
        if dependencies != tuple(sorted(dependencies)):
            raise ValueError(f"{context}.dependencies must be canonical and sorted")
        package = PythonPackage(
            distribution=distribution,
            version=version,
            license=license_expression,
            direct=direct,
            dependencies=dependencies,
            artifacts=_parse_artifacts(item.get("artifacts"), context),
        )
        packages.append(package)

    names = [package.name for package in packages]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(f"{path}.packages must be unique and canonical-name-sorted")
    by_name = {package.name: package for package in packages}
    for package in packages:
        missing = set(package.dependencies) - set(by_name)
        if missing:
            raise ValueError(
                f"{path} package {package.name} references missing dependencies: "
                f"{sorted(missing)}"
            )
        if package.name in package.dependencies:
            raise ValueError(f"{path} package {package.name} depends on itself")

    reachable: set[str] = set()
    pending = [package.name for package in packages if package.direct]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(by_name[name].dependencies)
    if reachable != set(by_name):
        raise ValueError(
            f"{path} contains unreachable packages: {sorted(set(by_name) - reachable)}"
        )

    filenames = [
        artifact.filename for package in packages for artifact in package.artifacts
    ]
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"{path} repeats a wheel filename across packages")
    return PythonRuntimeLock(
        requires_python=requires_python,
        supported_python=supported_python,
        packages=tuple(packages),
    )


__all__ = [
    "PythonArtifact",
    "PythonPackage",
    "PythonRuntimeLock",
    "canonicalize_name",
    "load_python_runtime_lock",
]
