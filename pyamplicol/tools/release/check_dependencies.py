#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Validate the small published or contributor dependency contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:  # pragma: no cover - pip vendors the build fallback
    from pip._vendor.packaging.requirements import (  # type: ignore[no-redef]
        InvalidRequirement,
        Requirement,
    )
    from pip._vendor.packaging.version import (  # type: ignore[no-redef]
        InvalidVersion,
        Version,
    )

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "dependencies" / "release-lock.toml"
CONTRIBUTOR_LOCK_PATH = ROOT / "dependencies" / "contributor-lock.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
CARGO_LOCK_PATH = ROOT / "Cargo.lock"
RUST_TOOLCHAIN_PATH = ROOT / "rust-toolchain.toml"
STATE_PATH = ROOT / "dependencies" / "install-state.json"
CANDIDATE_LOCK_PATH = ROOT / "dependencies" / "candidate-Cargo.lock"
CARGO_CONFIG_PATH = ROOT / "dependencies" / "candidate-cargo-config.toml"
CHECKOUTS_PATH = ROOT / "dependencies" / "checkouts"

_REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
_LOCAL_CRATES = {"rusticol-capi", "rusticol-core", "rusticol-python"}
_CANDIDATE_LOCAL_CRATES = {
    *_LOCAL_CRATES,
    "graphica",
    "numerica",
    "symbolica",
    "symjit",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_CANONICAL_NAME = re.compile(r"[-_.]+")
_CANDIDATE_ABIS = {
    "symbolica_serialization": "symbolica-bincode2-v1",
    "symjit_application": "symjit-application-storage-v3",
    "symjit_plane_application": "pyamplicol-symjit-plane-application-v2",
}


@dataclass(frozen=True)
class GateIssue:
    code: str
    message: str


def canonicalize_name(value: str) -> str:
    return _CANONICAL_NAME.sub("-", value).lower()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a TOML table")
    return payload


def _load_lock() -> dict[str, Any]:
    payload = _load_toml(LOCK_PATH)
    if payload.get("schema_version") != 1:
        raise ValueError("dependencies/release-lock.toml must use schema_version = 1")
    return payload


def _load_contributor_lock() -> dict[str, Any]:
    payload = _load_toml(CONTRIBUTOR_LOCK_PATH)
    if payload.get("schema_version") != 1:
        raise ValueError(
            "dependencies/contributor-lock.toml must use schema_version = 1"
        )
    return payload


def _locked_python_dependencies(lock: dict[str, Any]) -> dict[str, str]:
    raw = lock.get("python_dependencies")
    if not isinstance(raw, list) or not raw:
        raise ValueError("release lock must list exact Python dependencies")
    dependencies: dict[str, str] = {}
    ordered_names: list[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != {"distribution", "version"}:
            raise ValueError(
                f"python_dependencies[{index}] must contain only distribution/version"
            )
        distribution = entry["distribution"]
        version = entry["version"]
        if not isinstance(distribution, str) or not isinstance(version, str):
            raise ValueError(
                f"python_dependencies[{index}] needs string distribution/version"
            )
        name = canonicalize_name(distribution)
        try:
            normalized_version = str(Version(version))
        except InvalidVersion as error:
            raise ValueError(f"invalid locked version for {name}: {version}") from error
        if name in dependencies:
            raise ValueError(f"release lock repeats Python dependency {name}")
        ordered_names.append(name)
        dependencies[name] = normalized_version
    if ordered_names != sorted(ordered_names):
        raise ValueError("release-lock Python dependencies must be name-sorted")
    return dependencies


def _project_python_dependencies() -> dict[str, str]:
    project = _load_toml(PYPROJECT_PATH).get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml has no [project] table")
    raw = project.get("dependencies")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("pyproject runtime dependencies must be strings")
    dependencies: dict[str, str] = {}
    for value in raw:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as error:
            raise ValueError(f"invalid pyproject dependency: {value}") from error
        name = canonicalize_name(requirement.name)
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.marker is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
        ):
            raise ValueError(
                f"release runtime dependency must be one exact published pin: {value}"
            )
        try:
            version = str(Version(specifiers[0].version))
        except InvalidVersion as error:
            raise ValueError(
                f"invalid pyproject dependency version: {value}"
            ) from error
        if name in dependencies:
            raise ValueError(f"pyproject repeats runtime dependency {name}")
        dependencies[name] = version
    return dependencies


def _release_contract_issues(lock: dict[str, Any]) -> list[GateIssue]:
    issues: list[GateIssue] = []
    abis = lock.get("abis")
    if not isinstance(abis, dict) or any(
        abis.get(name) != value for name, value in _CANDIDATE_ABIS.items()
    ):
        issues.append(
            GateIssue(
                "release-abi-contract",
                "release-lock.toml must pin the Symbolica serialization, SymJIT "
                "storage, and SymJIT plane-application ABIs",
            )
        )
    forbidden_tables = {"legal_status", "python_runtime_lock"}.intersection(lock)
    if forbidden_tables:
        issues.append(
            GateIssue(
                "release-lock-scope",
                "release-lock.toml contains retired release machinery: "
                + ", ".join(sorted(forbidden_tables)),
            )
        )
    try:
        locked = _locked_python_dependencies(lock)
        project = _project_python_dependencies()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [*issues, GateIssue("python-dependency-contract", str(error))]
    if locked != project:
        issues.append(
            GateIssue(
                "python-dependency-contract",
                "pyproject exact runtime pins disagree with release-lock.toml; "
                f"lock={locked}, pyproject={project}",
            )
        )

    symbolica = lock.get("symbolica")
    symjit = lock.get("symjit")
    loader = lock.get("ufo_model_loader")
    if (
        not isinstance(symbolica, dict)
        or not isinstance(symjit, dict)
        or not isinstance(loader, dict)
    ):
        issues.append(
            GateIssue(
                "published-dependency-contract",
                "release lock needs Symbolica, SymJIT, and ufo-model-loader "
                "compatibility data",
            )
        )
        return issues
    allowed_symbolica = {
        "python_distribution",
        "python_version",
        "rust_crate",
        "rust_version",
        "serialization_abi",
    }
    allowed_symjit = {"version"}
    allowed_loader = {
        "python_distribution",
        "required_version",
    }
    if (
        set(symbolica) != allowed_symbolica
        or set(symjit) != allowed_symjit
        or set(loader) != allowed_loader
    ):
        issues.append(
            GateIssue(
                "release-lock-scope",
                "release dependency sections contain fields outside their approved "
                "compatibility and source contracts",
            )
        )
    symbolica_name = canonicalize_name(str(symbolica.get("python_distribution", "")))
    loader_name = canonicalize_name(str(loader.get("python_distribution", "")))
    if locked.get(symbolica_name) != str(symbolica.get("python_version", "")):
        issues.append(
            GateIssue(
                "symbolica-pin",
                "Symbolica compatibility data disagrees with the exact Python pin",
            )
        )
    if locked.get(loader_name) != str(loader.get("required_version", "")):
        issues.append(
            GateIssue(
                "ufo-loader-pin",
                "ufo-model-loader compatibility data disagrees with the exact pin",
            )
        )
    return issues


def _cargo_packages(path: Path) -> list[dict[str, Any]]:
    payload = _load_toml(path)
    packages = payload.get("package")
    if not isinstance(packages, list) or not all(
        isinstance(package, dict) for package in packages
    ):
        raise ValueError(f"{path} has no Cargo package array")
    return packages


def _registry_source_issues(
    packages: list[dict[str, Any]],
    *,
    local_crates: set[str],
    prefix: str,
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    for package in packages:
        name = str(package.get("name", ""))
        source = package.get("source")
        if name in local_crates:
            if source is not None:
                issues.append(
                    GateIssue(
                        f"{prefix}-cargo-local-source",
                        f"local crate {name} unexpectedly has source {source}",
                    )
                )
            continue
        if source != _REGISTRY_SOURCE:
            issues.append(
                GateIssue(
                    f"{prefix}-cargo-nonregistry",
                    f"Cargo.lock package {name} is not an exact crates.io package",
                )
            )
            continue
        checksum = package.get("checksum")
        if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
            issues.append(
                GateIssue(
                    f"{prefix}-cargo-checksum",
                    f"Cargo.lock package {name} has no crates.io checksum",
                )
            )
    return issues


def _cargo_manifest_pin_issues(lock: dict[str, Any]) -> list[GateIssue]:
    symbolica = lock["symbolica"]
    symjit = lock["symjit"]
    try:
        root = _load_toml(ROOT / "Cargo.toml")
        core = _load_toml(ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml")
        root_dependencies = root["workspace"]["dependencies"]
        core_dependencies = core["dependencies"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-cargo-manifest", f"invalid Cargo manifest: {error}")]
    issues: list[GateIssue] = []
    if root.get("patch"):
        issues.append(
            GateIssue(
                "release-cargo-patch",
                "release Cargo.toml must use published crates without source patches",
            )
        )
    symbolica_version = f"={symbolica['rust_version']}"
    for table in (root_dependencies, core_dependencies):
        entry = table.get("symbolica") if isinstance(table, dict) else None
        if not isinstance(entry, dict) or entry.get("version") != symbolica_version:
            issues.append(
                GateIssue(
                    "release-cargo-pin",
                    "Cargo manifest must require "
                    f"symbolica {symbolica_version} exactly",
                )
            )
        elif "git" in entry or "path" in entry:
            issues.append(
                GateIssue(
                    "release-cargo-source",
                    "release Cargo dependency symbolica must use crates.io",
                )
            )
    symjit_entry = (
        core_dependencies.get("symjit") if isinstance(core_dependencies, dict) else None
    )
    if (
        not isinstance(symjit_entry, dict)
        or symjit_entry.get("version") != f"={symjit['version']}"
        or "git" in symjit_entry
        or "rev" in symjit_entry
        or "path" in symjit_entry
    ):
        issues.append(
            GateIssue(
                "release-cargo-source",
                "rusticol-core must require the exact release-lock SymJIT version "
                "from crates.io",
            )
        )
    return issues


def _release_cargo_lock_issues(lock: dict[str, Any]) -> list[GateIssue]:
    try:
        packages = _cargo_packages(CARGO_LOCK_PATH)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-cargo-lock", str(error))]
    issues = _registry_source_issues(
        packages,
        local_crates=_LOCAL_CRATES,
        prefix="release",
    )
    symbolica = lock.get("symbolica", {})
    for name, version in (
        (
            str(symbolica.get("rust_crate", "symbolica")),
            str(symbolica.get("rust_version", "")),
        ),
        ("symjit", str(lock.get("symjit", {}).get("version", ""))),
    ):
        matches = [
            package
            for package in packages
            if package.get("name") == name and package.get("version") == version
        ]
        if len(matches) != 1 or matches[0].get("source") != _REGISTRY_SOURCE:
            issues.append(
                GateIssue(
                    "release-cargo-pin",
                    f"Cargo.lock must resolve published {name}=={version} exactly",
                )
            )
    return [*issues, *_cargo_manifest_pin_issues(lock)]


def _toolchain_issues(lock: dict[str, Any]) -> list[GateIssue]:
    toolchain = lock.get("toolchain")
    if not isinstance(toolchain, dict):
        return [GateIssue("toolchain-contract", "release lock has no toolchain table")]
    issues: list[GateIssue] = []
    try:
        rust = _load_toml(RUST_TOOLCHAIN_PATH)["toolchain"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [
            GateIssue("toolchain-contract", f"invalid rust-toolchain.toml: {error}")
        ]
    if not isinstance(rust, dict) or rust.get("channel") != toolchain.get(
        "rust_toolchain"
    ):
        issues.append(
            GateIssue(
                "toolchain-contract",
                "rust-toolchain.toml disagrees with release-lock.toml",
            )
        )
    digest = toolchain.get("manylinux_image_digest")
    image = toolchain.get("manylinux_image")
    if (
        not isinstance(image, str)
        or not image
        or not isinstance(digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
    ):
        issues.append(
            GateIssue(
                "toolchain-contract",
                "manylinux image and digest must be pinned exactly",
            )
        )
    build_system = _load_toml(PYPROJECT_PATH).get("build-system", {})
    requirements = (
        build_system.get("requires", []) if isinstance(build_system, dict) else []
    )
    expected = {
        "maturin": str(toolchain.get("maturin", "")),
        "packaging": str(toolchain.get("packaging", "")),
    }
    observed: dict[str, str] = {}
    for raw in requirements if isinstance(requirements, list) else []:
        if not isinstance(raw, str):
            continue
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        specifiers = list(requirement.specifier)
        if len(specifiers) == 1 and specifiers[0].operator == "==":
            observed[canonicalize_name(requirement.name)] = specifiers[0].version
    if observed != expected:
        issues.append(
            GateIssue(
                "toolchain-contract",
                f"build-system pins disagree with release lock: {observed}",
            )
        )
    return issues


def _candidate_sources(
    contributor: dict[str, Any],
    release: dict[str, Any],
) -> dict[str, tuple[str, str]]:
    return {
        "gammaloop": (
            str(contributor["gammaloop_candidate"]["source_url"]),
            str(contributor["gammaloop_candidate"]["revision"]),
        ),
        "symbolica": (
            str(contributor["symbolica"]["source_url"]),
            str(contributor["symbolica"]["candidate_revision"]),
        ),
        "symbolica-community": (
            str(contributor["symbolica"]["community_url"]),
            str(contributor["symbolica"]["community_revision"]),
        ),
        "symbolica-integrate": (
            str(contributor["symbolica_integrate"]["source_url"]),
            str(contributor["symbolica_integrate"]["revision"]),
        ),
        "symjit": (
            str(contributor["symjit"]["repository"]),
            str(contributor["symjit"]["revision"]),
        ),
        "ufo-model-loader": (
            str(contributor["ufo_model_loader"]["source_url"]),
            str(contributor["ufo_model_loader"]["candidate_revision"]),
        ),
    }


def _git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _symjit_manifest_issues(path: Path, *, expected_version: str) -> list[GateIssue]:
    try:
        manifest = _load_toml(path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("candidate-symjit-manifest", str(error))]
    package = manifest.get("package")
    library = manifest.get("lib")
    if (
        not isinstance(package, dict)
        or package.get("name") != "symjit"
        or package.get("version") != expected_version
        or not isinstance(library, dict)
        or library.get("crate-type") != ["rlib"]
    ):
        return [
            GateIssue(
                "candidate-symjit-manifest",
                "managed SymJIT must be the locked version and expose an "
                "rlib-only library",
            )
        ]
    return []


def _candidate_config_issues(path_overrides: dict[str, Path]) -> list[GateIssue]:
    checkout_root = CHECKOUTS_PATH.resolve()
    expected_paths = {
        "graphica": checkout_root / "symbolica" / "lib" / "graphica",
        "numerica": checkout_root / "symbolica" / "lib" / "numerica",
        "symbolica": checkout_root / "symbolica",
        "symjit": checkout_root / "symjit",
        **path_overrides,
    }
    try:
        config = _load_toml(CARGO_CONFIG_PATH)
        patch_tables = config["patch"]
        registry_patches = patch_tables["crates-io"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return [
            GateIssue("candidate-cargo-config", f"invalid Cargo patch config: {error}")
        ]
    if (
        not isinstance(config, dict)
        or set(config) != {"patch"}
        or not isinstance(patch_tables, dict)
        or set(patch_tables) != {"crates-io"}
        or not isinstance(registry_patches, dict)
        or set(registry_patches) != set(expected_paths)
    ):
        return [
            GateIssue(
                "candidate-cargo-config",
                "candidate Cargo patch table does not match the locked source "
                "overrides",
            )
        ]
    issues: list[GateIssue] = []
    patches = registry_patches
    for name, entry in patches.items():
        path_value = entry.get("path") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path"}
            or not isinstance(path_value, str)
        ):
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} must contain exactly one local path",
                )
            )
            continue
        path = Path(path_value).resolve()
        if path != expected_paths[name]:
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} does not resolve to its "
                    "declared checkout",
                )
            )
            continue
        if not path.exists():
            issues.append(
                GateIssue(
                    "candidate-cargo-config",
                    f"candidate Cargo patch {name} path is missing: {path}",
                )
            )
    return issues


def _candidate_issues(release_lock: dict[str, Any]) -> list[GateIssue]:
    required = (
        CONTRIBUTOR_LOCK_PATH,
        STATE_PATH,
        CANDIDATE_LOCK_PATH,
        CARGO_CONFIG_PATH,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        return [
            GateIssue(
                "candidate-input-missing",
                "run 'just dev-install' before a candidate build; missing: "
                + ", ".join(str(path) for path in missing),
            )
        ]
    try:
        contributor = _load_contributor_lock()
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        packages = _cargo_packages(CANDIDATE_LOCK_PATH)
        root_patches = (
            _load_toml(ROOT / "Cargo.toml").get("patch", {}).get("crates-io", {})
        )
        path_overrides = {
            name: (ROOT / entry["path"]).resolve()
            for name, entry in root_patches.items()
            if isinstance(entry, dict) and "path" in entry
        }
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        return [GateIssue("candidate-input-invalid", str(error))]
    issues = _registry_source_issues(
        packages,
        local_crates=_CANDIDATE_LOCAL_CRATES | path_overrides.keys(),
        prefix="candidate",
    )
    if contributor.get("abis") != _CANDIDATE_ABIS:
        issues.append(
            GateIssue(
                "candidate-abi-contract",
                "contributor-lock.toml has unexpected candidate ABI identities",
            )
        )
    if "patches" in contributor:
        issues.append(
            GateIssue(
                "candidate-symjit-patch",
                "contributor-lock.toml must not carry a local SymJIT patch contract",
            )
        )
    patch_root = CONTRIBUTOR_LOCK_PATH.parent / "patches" / "symjit"
    if patch_root.is_symlink() or (
        patch_root.is_dir() and any(path.is_file() for path in patch_root.rglob("*"))
    ):
        issues.append(
            GateIssue(
                "candidate-symjit-patch",
                "local SymJIT patch inventory must be removed",
            )
        )
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        issues.append(
            GateIssue("candidate-state-invalid", "installer state must use schema 1")
        )
        return issues
    if state.get("publishable") is not False:
        issues.append(
            GateIssue(
                "candidate-state-publishable",
                "candidate installer state must explicitly be non-publishable",
            )
        )
    if set(state) != {"schema_version", "publishable", "sources"}:
        issues.append(
            GateIssue(
                "candidate-state-invalid",
                "installer state may contain only schema_version, publishable, "
                "and sources",
            )
        )
    sources = state.get("sources")
    try:
        expected_sources = _candidate_sources(contributor, release_lock)
    except (KeyError, TypeError) as error:
        issues.append(
            GateIssue(
                "candidate-source-contract",
                f"candidate source contract is incomplete: {error}",
            )
        )
        expected_sources = {}
    if not isinstance(sources, dict):
        issues.append(
            GateIssue("candidate-state-invalid", "installer state has no source map")
        )
    else:
        local_sources = {
            name: path_overrides[crate]
            for name, crate in (
                ("symjit", "symjit"),
                ("symbolica-integrate", "symbolica-integrate"),
            )
            if crate in path_overrides
        }
        if "spenso" in path_overrides:
            local_sources["gammaloop"] = path_overrides["spenso"].parent.parent
        for source_name, entry in sources.items():
            allowed = {"url", "revision"}
            if isinstance(entry, dict) and "branch" in entry:
                allowed.add("branch")
            if (
                not isinstance(source_name, str)
                or not source_name
                or not isinstance(entry, dict)
                or set(entry) != allowed
                or not all(
                    isinstance(entry[key], str) and entry[key] for key in allowed
                )
                or _GIT_REVISION.fullmatch(entry["revision"]) is None
            ):
                issues.append(
                    GateIssue(
                        "candidate-state-invalid",
                        f"candidate source {source_name!r} has an invalid descriptor",
                    )
                )
        for name, (url, revision) in expected_sources.items():
            entry = sources.get(name)
            checkout = local_sources.get(name, CHECKOUTS_PATH / name)
            # Explicit Cargo path overrides are user-owned, not managed installs.
            # Their declared Git base is checked below; local edits are permitted.
            if name not in local_sources and (
                not isinstance(entry, dict)
                or entry.get("url") != url
                or entry.get("revision") != revision
            ):
                issues.append(
                    GateIssue(
                        "candidate-source-revision",
                        f"candidate source {name} does not match contributor-lock.toml",
                    )
                )
            if not checkout.is_dir() or _git_head(checkout) != revision:
                issues.append(
                    GateIssue(
                        "candidate-source-revision",
                        f"candidate checkout {name} is not at {revision}",
                    )
                )
        symjit_checkout = local_sources.get("symjit", CHECKOUTS_PATH / "symjit")
        if symjit_checkout.is_dir():
            issues.extend(
                _symjit_manifest_issues(
                    symjit_checkout / "Cargo.toml",
                    expected_version=str(contributor["symjit"]["candidate_version"]),
                )
            )
    return [*issues, *_candidate_config_issues(path_overrides)]


def check(*, candidate: bool) -> list[GateIssue]:
    try:
        lock = _load_lock()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [GateIssue("release-lock-invalid", str(error))]
    issues = [
        *_release_contract_issues(lock),
        *_toolchain_issues(lock),
    ]
    if candidate:
        issues.extend(_candidate_issues(lock))
    else:
        issues.extend(_release_cargo_lock_issues(lock))
    return issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="validate source-checkout candidate inputs as non-publishable",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    issues = check(candidate=args.candidate)
    payload = {
        "mode": "candidate" if args.candidate else "release",
        "ready": not issues,
        "issues": [{"code": issue.code, "message": issue.message} for issue in issues],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif issues:
        for issue in issues:
            print(f"[{issue.code}] {issue.message}", file=sys.stderr)
    else:
        print(f"{payload['mode']} dependency gate passed")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
