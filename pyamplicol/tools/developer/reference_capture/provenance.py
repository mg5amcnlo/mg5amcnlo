# SPDX-License-Identifier: 0BSD
"""Tracked-source and dependency provenance for reference capture."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import tomllib
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from .common import (
    CONTRIBUTOR_LOCK,
    INSTALL_STATE,
    RELEASE_LOCK,
    ROOT,
    CaptureError,
    DependencySnapshot,
    RuntimeSnapshot,
    SourceSnapshot,
    as_mapping,
    canonical_sha256,
    developer_module,
    sha256_file,
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_GIT_REVISION_RE = re.compile(r"^[a-f0-9]{40}$")
_OFFICIAL_SYMJIT_SOURCE_URL = "https://github.com/siravan/symjit-crate.git"
with RELEASE_LOCK.open("rb") as _release_stream:
    _RELEASE_VERSION = str(tomllib.load(_release_stream)["project"]["version"])


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({"LC_ALL": "C", "LANG": "C"})
    return environment


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        env=_git_environment(),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CaptureError(
            f"git {' '.join(arguments)} failed in {root}"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout


def require_clean_tracked_tree(root: Path) -> str:
    """Require a clean checkout, including non-ignored untracked files."""

    repository = Path(
        _git_bytes(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    if repository != root.resolve():
        raise CaptureError(
            f"capture root {root.resolve()} is not Git top-level {repository}"
        )
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status:
        entries = [
            entry.decode("utf-8", errors="replace")
            for entry in status.split(b"\0")
            if entry
        ]
        rendered = ", ".join(entries[:8])
        if len(entries) > 8:
            rendered += f", and {len(entries) - 8} more"
        raise CaptureError(
            "fixture capture requires a clean source tree; commit, move, or "
            f"ignore these changes first: {rendered}"
        )
    revision = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
    if _GIT_REVISION_RE.fullmatch(revision) is None:
        raise CaptureError(f"HEAD is not a full Git SHA-1: {revision!r}")
    return revision


def tracked_source_tree_sha256(root: Path, revision: str = "HEAD") -> str:
    """Hash tracked path, mode, type, and blob bytes in deterministic tree order."""

    listing = _git_bytes(root, "ls-tree", "-r", "-z", "--full-tree", revision)
    digest = hashlib.sha256(b"pyamplicol-tracked-source-tree-v1\0")
    for raw_record in listing.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, path = raw_record.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
        except ValueError as error:
            raise CaptureError("git ls-tree returned an invalid record") from error
        content = (
            _git_bytes(root, "cat-file", "blob", object_id.decode("ascii"))
            if object_type == b"blob"
            else object_id
        )
        for field in (mode, object_type, path, content):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def github_https_uri(origin: str) -> str:
    """Normalize GitHub SSH remotes to a schema-compliant HTTPS URI."""

    scp_prefix = "git@github.com:"
    if origin.startswith(scp_prefix):
        return "https://github.com/" + origin.removeprefix(scp_prefix)
    parsed = urlparse(origin)
    if parsed.scheme == "ssh" and parsed.hostname == "github.com":
        return "https://github.com/" + parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https", "ssh", "git"} and parsed.netloc:
        return origin
    raise CaptureError(
        f"Git origin {origin!r} is not an absolute repository URI; configure origin"
    )


def collect_source_snapshot(root: Path = ROOT) -> SourceSnapshot:
    revision = require_clean_tracked_tree(root)
    origin = _git_bytes(root, "remote", "get-url", "origin").decode("utf-8").strip()
    return SourceSnapshot(
        repository_uri=github_https_uri(origin),
        revision=revision,
        tree_sha256=tracked_source_tree_sha256(root, revision),
    )


def assert_source_snapshot_unchanged(
    snapshot: SourceSnapshot,
    root: Path = ROOT,
) -> None:
    revision = require_clean_tracked_tree(root)
    digest = tracked_source_tree_sha256(root, revision)
    if revision != snapshot.revision or digest != snapshot.tree_sha256:
        raise CaptureError(
            "tracked source changed during capture; retained artifacts were not "
            "removed, but fixture files were not published"
        )


def _distribution_sha256(distribution: metadata.Distribution) -> str:
    files = distribution.files
    if files is None:
        raise CaptureError("installed pyamplicol distribution has no file inventory")
    digest = hashlib.sha256(b"pyamplicol-installed-distribution-v1\0")
    included = 0
    for package_path in sorted(files, key=lambda item: str(item)):
        relative = Path(str(package_path))
        if (
            "__pycache__" in relative.parts
            or relative.suffix in {".pyc", ".pyo"}
            or relative.name in {"INSTALLER", "RECORD", "REQUESTED", "direct_url.json"}
        ):
            continue
        path = Path(str(distribution.locate_file(package_path)))
        if not path.is_file():
            raise CaptureError(
                f"installed pyamplicol distribution member is missing: {relative}"
            )
        encoded = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        included += 1
    if included == 0:
        raise CaptureError("installed pyamplicol distribution inventory is empty")
    return digest.hexdigest()


def collect_runtime_snapshot(source: SourceSnapshot) -> RuntimeSnapshot:
    """Bind capture to one installed candidate built from ``source`` exactly."""

    try:
        package = importlib.import_module("pyamplicol")
        raw_package_path = getattr(package, "__file__", None)
        if not isinstance(raw_package_path, str):
            raise CaptureError("installed pyamplicol module has no filesystem path")
        package_path = Path(raw_package_path).resolve(strict=True)
        distribution = metadata.distribution("pyamplicol")
    except (
        AttributeError,
        ImportError,
        metadata.PackageNotFoundError,
        OSError,
    ) as error:
        raise CaptureError(
            "reference capture requires an installed pyamplicol candidate wheel"
        ) from error
    source_package = (ROOT / "src" / "pyamplicol").resolve()
    if package_path.is_relative_to(source_package):
        raise CaptureError(
            "reference capture may not import pyamplicol directly from the source "
            "tree; "
            "install a candidate wheel built from the clean revision"
        )
    build_info_path = package_path.parent / "_build_info.json"
    try:
        build_info = as_mapping(
            json.loads(build_info_path.read_text(encoding="utf-8")),
            "installed pyamplicol _build_info.json",
        )
    except (OSError, json.JSONDecodeError) as error:
        raise CaptureError(
            "installed pyamplicol is not a provenance-marked candidate wheel"
        ) from error
    version = str(build_info.get("version", ""))
    fingerprint = str(build_info.get("candidate_fingerprint", ""))
    revision = str(build_info.get("source_revision", ""))
    if (
        build_info.get("schema_version") != 1
        or build_info.get("publishable") is not False
        or re.fullmatch(r"[a-f0-9]{12}", fingerprint) is None
        or version != f"{_RELEASE_VERSION}.dev0+candidate.{fingerprint}"
    ):
        raise CaptureError(
            "installed pyamplicol has invalid candidate build provenance"
        )
    if revision != source.revision:
        raise CaptureError(
            "installed pyamplicol candidate was not built from the clean capture "
            f"revision {source.revision}; found {revision or 'no source revision'}"
        )
    if str(getattr(package, "__version__", "")) != version:
        raise CaptureError(
            "installed pyamplicol module version differs from its candidate marker"
        )
    if distribution.version != version:
        raise CaptureError(
            "installed pyamplicol distribution version differs from its candidate "
            "marker"
        )
    return RuntimeSnapshot(
        version=version,
        candidate_fingerprint=fingerprint,
        source_revision=revision,
        distribution_sha256=_distribution_sha256(distribution),
        build_info_sha256=sha256_file(build_info_path),
    )


def assert_runtime_snapshot_unchanged(
    snapshot: RuntimeSnapshot,
    source: SourceSnapshot,
) -> None:
    if collect_runtime_snapshot(source) != snapshot:
        raise CaptureError("installed pyamplicol candidate changed during capture")


def collect_dependency_snapshot(runtime: RuntimeSnapshot) -> DependencySnapshot:
    """Bind dependency records to the release lock and installed source state."""

    try:
        with RELEASE_LOCK.open("rb") as stream:
            release = cast(Mapping[str, object], tomllib.load(stream))
        with CONTRIBUTOR_LOCK.open("rb") as stream:
            contributor = cast(Mapping[str, object], tomllib.load(stream))
        state = cast(
            Mapping[str, object],
            json.loads(INSTALL_STATE.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise CaptureError(f"cannot read dependency provenance: {error}") from error
    if (
        release.get("schema_version") != 1
        or contributor.get("schema_version") != 1
        or state.get("schema_version") != 1
    ):
        raise CaptureError(
            "release-lock, contributor-lock, and install-state schema versions "
            "must be 1"
        )
    if state.get("publishable") is not False:
        raise CaptureError("reference capture requires non-publishable install state")

    release_digest = sha256_file(RELEASE_LOCK)
    sources = as_mapping(state.get("sources"), "install-state.sources")
    symbolica = as_mapping(
        contributor.get("symbolica"),
        "contributor-lock.symbolica",
    )
    symjit = as_mapping(contributor.get("symjit"), "contributor-lock.symjit")
    loader = {
        **as_mapping(
            release.get("ufo_model_loader"),
            "release-lock.ufo_model_loader",
        ),
        **as_mapping(
            contributor.get("ufo_model_loader"),
            "contributor-lock.ufo_model_loader",
        ),
    }
    legacy = as_mapping(
        contributor.get("legacy_amplicol"),
        "contributor-lock.legacy_amplicol",
    )
    release_abis = as_mapping(release.get("abis"), "release-lock.abis")
    contributor_abis = as_mapping(
        contributor.get("abis"),
        "contributor-lock.abis",
    )
    project = as_mapping(release.get("project"), "release-lock.project")
    symbolica_source = as_mapping(sources.get("symbolica"), "source symbolica")
    symjit_source = as_mapping(sources.get("symjit"), "source symjit")

    symbolica_descriptor = {
        "url": symbolica.get("source_url"),
        "revision": symbolica.get("candidate_revision"),
    }
    symjit_descriptor = {
        "url": symjit.get("repository"),
        "revision": symjit.get("revision"),
    }
    for name, descriptor, installed in (
        ("Symbolica", symbolica_descriptor, symbolica_source),
        ("SymJIT", symjit_descriptor, symjit_source),
    ):
        source_url = descriptor["url"]
        revision = descriptor["revision"]
        if (
            not isinstance(source_url, str)
            or not source_url
            or not isinstance(revision, str)
            or _GIT_REVISION_RE.fullmatch(revision) is None
        ):
            raise CaptureError(f"{name} dependency source contract is invalid")
        if installed.get("url") != source_url or installed.get("revision") != revision:
            raise CaptureError(
                f"installed {name} source does not match its dependency contract"
            )
    if symjit_descriptor["url"] != _OFFICIAL_SYMJIT_SOURCE_URL:
        raise CaptureError("release lock does not use the official SymJIT repository")

    legacy_module = developer_module("legacy_amplicol")
    legacy_digest = canonical_sha256(
        {
            "branch": legacy_module.checkout_branch(),
            "revision": legacy_module.expected_revision(),
            "source_url": str(legacy["source_url"]),
        }
    )
    analytic_path = ROOT / "tools" / "developer" / "analytic_oracles.py"
    payloads: tuple[dict[str, object], ...] = (
        {
            "id": "dependency:pyamplicol-candidate",
            "name": "Installed pyAmpliCol capture candidate",
            "version": runtime.version,
            "revision": runtime.source_revision,
            "content_sha256": runtime.distribution_sha256,
            "serialization_abi": str(release_abis["symbolica_serialization"]),
            "license": str(project["license"]),
        },
        {
            "id": "dependency:release-lock",
            "name": "pyAmpliCol dependency release lock",
            "version": "schema-1",
            "revision": None,
            "content_sha256": release_digest,
            "serialization_abi": str(release_abis["symbolica_serialization"]),
            "license": str(project["license"]),
        },
        {
            "id": "dependency:install-state",
            "name": "pyAmpliCol installed dependency state",
            "version": "schema-1",
            "revision": None,
            "content_sha256": sha256_file(INSTALL_STATE),
            "serialization_abi": str(release_abis["symbolica_serialization"]),
            "license": str(project["license"]),
        },
        {
            "id": "dependency:symbolica",
            "name": "Symbolica",
            "version": str(symbolica["candidate_version"]),
            "revision": str(symbolica_descriptor["revision"]),
            "content_sha256": canonical_sha256(symbolica_descriptor),
            "serialization_abi": str(contributor_abis["symbolica_serialization"]),
            "license": "Symbolica proprietary",
        },
        {
            "id": "dependency:symjit",
            "name": "Symjit",
            "version": str(symjit["candidate_version"]),
            "revision": str(symjit_descriptor["revision"]),
            "content_sha256": canonical_sha256(symjit_descriptor),
            "serialization_abi": str(release_abis["symjit_application"]),
            "license": "MIT",
        },
        {
            "id": "dependency:ufo-model-loader",
            "name": "ufo-model-loader",
            "version": str(loader["required_version"]),
            "revision": str(loader["published_revision"]),
            "content_sha256": str(loader["wheel_sha256"]),
            "serialization_abi": None,
            "license": str(loader["license"]),
        },
        {
            "id": "dependency:legacy-amplicol",
            "name": "Pinned legacy Fortran AmpliCol",
            "version": str(legacy["release_status"]),
            "revision": str(legacy["revision"]),
            "content_sha256": legacy_digest,
            "serialization_abi": None,
            "license": str(legacy["license"]),
        },
        {
            "id": "dependency:analytic-oracles",
            "name": "pyAmpliCol independent analytic oracles",
            "version": "1",
            "revision": None,
            "content_sha256": sha256_file(analytic_path),
            "serialization_abi": None,
            "license": "0BSD",
        },
    )
    for payload in payloads:
        digest = str(payload["content_sha256"])
        if _SHA256_RE.fullmatch(digest) is None or digest == "0" * 64:
            raise CaptureError(
                f"dependency {payload['id']} has invalid content digest {digest!r}"
            )
    return DependencySnapshot(payloads, release, contributor, state)


def default_artifact_root(revision: str) -> Path:
    if _GIT_REVISION_RE.fullmatch(revision) is None:
        raise CaptureError("default artifact roots require a full Git revision")
    return ROOT / ".artifacts" / "reference-fixture-v2" / revision
