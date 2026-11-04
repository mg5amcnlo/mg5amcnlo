# SPDX-License-Identifier: 0BSD
"""Package-version checks sourced from the Cargo workspace manifest."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            f"cannot read package version contract {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"package version contract {path} is not a TOML table")
    return payload


def _required_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} must be a nonempty string")
    return value


def canonical_package_version(root: Path) -> str:
    """Return Cargo's package version after checking release-facing metadata."""

    cargo = _load_toml(root / "Cargo.toml")
    pyproject = _load_toml(root / "pyproject.toml")
    release = _load_toml(root / "dependencies" / "release-lock.toml")
    try:
        cargo_version = _required_string(
            cargo["workspace"]["package"]["version"],
            "Cargo.toml [workspace.package].version",
        )
        project = pyproject["project"]
        locked_version = _required_string(
            release["project"]["version"],
            "dependencies/release-lock.toml [project].version",
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError("package version contract is incomplete") from error

    if not isinstance(project, dict):
        raise RuntimeError("pyproject.toml [project] must be a table")
    dynamic = project.get("dynamic")
    if (
        "version" in project
        or not isinstance(dynamic, list)
        or "version" not in dynamic
    ):
        raise RuntimeError(
            "pyproject.toml must declare project.version as dynamic; "
            "Cargo.toml [workspace.package].version is canonical"
        )
    if locked_version != cargo_version:
        raise RuntimeError(
            "package version mismatch: Cargo.toml declares "
            f"{cargo_version!r}, but dependencies/release-lock.toml declares "
            f"{locked_version!r}"
        )
    return cargo_version


def check_contributor_lock_consistency(root: Path) -> None:
    """Check dependency identities shared by release and contributor locks."""

    release = _load_toml(root / "dependencies" / "release-lock.toml")
    contributor = _load_toml(root / "dependencies" / "contributor-lock.toml")
    try:
        values = {
            "release [abis]": _required_string(
                release["abis"]["symbolica_serialization"],
                "release-lock [abis].symbolica_serialization",
            ),
            "release [symbolica]": _required_string(
                release["symbolica"]["serialization_abi"],
                "release-lock [symbolica].serialization_abi",
            ),
            "contributor [abis]": _required_string(
                contributor["abis"]["symbolica_serialization"],
                "contributor-lock [abis].symbolica_serialization",
            ),
        }
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "dependency lock compatibility contract is incomplete"
        ) from error
    if len(set(values.values())) != 1:
        rendered = ", ".join(f"{name}={value!r}" for name, value in values.items())
        raise RuntimeError(
            "release and contributor dependency locks disagree on the Symbolica "
            f"serialization ABI: {rendered}"
        )

    try:
        plane_values = {
            "release [abis]": _required_string(
                release["abis"]["symjit_plane_application"],
                "release-lock [abis].symjit_plane_application",
            ),
            "contributor [abis]": _required_string(
                contributor["abis"]["symjit_plane_application"],
                "contributor-lock [abis].symjit_plane_application",
            ),
        }
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "dependency lock compatibility contract is incomplete"
        ) from error
    if len(set(plane_values.values())) != 1:
        rendered = ", ".join(
            f"{name}={value!r}" for name, value in plane_values.items()
        )
        raise RuntimeError(
            "release and contributor dependency locks disagree on the SymJIT "
            f"plane-application ABI: {rendered}"
        )
