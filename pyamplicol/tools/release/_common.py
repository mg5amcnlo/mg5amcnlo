#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Shared safety and subprocess helpers for release tooling."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
CANDIDATE_ARTIFACTS = ROOT / ".artifacts" / "candidate"
DEPLOYMENT_ROOT = ROOT / "PYPI_DEPLOYMENT_TEST"
DEPENDENCY_WHEELHOUSE = ROOT / "dependencies" / "wheelhouse"
DEPENDENCY_GATE = ROOT / "tools" / "release" / "check_dependencies.py"
MAX_ARCHIVE_FILES = 100_000
MAX_UNPACKED_SDIST_BYTES = 2 * 1024 * 1024 * 1024


class ReleaseError(RuntimeError):
    """A release invariant was not satisfied."""


def build_mode(*, candidate: bool = False) -> str:
    """Resolve and validate candidate/release mode without silent conflicts."""

    configured = os.environ.get("PYAMPLICOL_BUILD_MODE")
    if configured is not None and configured not in {"candidate", "release"}:
        raise ReleaseError("PYAMPLICOL_BUILD_MODE must be 'candidate' or 'release'")
    requested = "candidate" if candidate else configured or "release"
    if candidate and configured == "release":
        raise ReleaseError("--candidate conflicts with PYAMPLICOL_BUILD_MODE=release")
    return requested


_PIP_INJECTION_VARIABLES = {
    "PIP_CONFIG_FILE",
    "PIP_CONSTRAINT",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_NO_INDEX",
    "PIP_PREFIX",
    "PIP_REQUIREMENT",
    "PIP_TARGET",
    "PIP_TRUSTED_HOST",
    "PIP_USER",
}

_NATIVE_BUILD_INJECTION_VARIABLES = {
    "AR",
    "C_INCLUDE_PATH",
    "CC",
    "CFLAGS",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CXX",
    "CXXFLAGS",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LDFLAGS",
    "LIBRARY_PATH",
    "MACOSX_DEPLOYMENT_TARGET",
    "OBJC_INCLUDE_PATH",
    "PKG_CONFIG_PATH",
    "RUSTFLAGS",
    "SDKROOT",
}


def clean_environment(
    updates: Mapping[str, str] | None = None,
    *,
    mode: str | None = None,
    virtual_env: Path | None = None,
) -> dict[str, str]:
    """Return an environment with Python and pip path injection disabled."""

    environment = dict(os.environ)
    for name in (
        _PIP_INJECTION_VARIABLES
        | _NATIVE_BUILD_INJECTION_VARIABLES
        | {
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "_OLD_VIRTUAL_PATH",
        }
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    if mode is not None:
        if mode not in {"candidate", "release"}:
            raise ReleaseError(f"unsupported build mode: {mode}")
        environment["PYAMPLICOL_BUILD_MODE"] = mode
    if virtual_env is not None:
        binary = virtual_env / ("Scripts" if os.name == "nt" else "bin")
        environment["VIRTUAL_ENV"] = str(virtual_env)
        environment["PATH"] = str(binary) + os.pathsep + environment.get("PATH", "")
    if updates:
        environment.update(updates)
    return environment


def runtime_environment(virtual_env: Path) -> dict[str, str]:
    """Create a runtime environment that cannot discover a Rust toolchain."""

    environment = clean_environment(virtual_env=virtual_env)
    binary = virtual_env / ("Scripts" if os.name == "nt" else "bin")
    system_paths = (
        [Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"]
        if os.name == "nt"
        else [Path("/usr/bin"), Path("/bin")]
    )
    environment["PATH"] = os.pathsep.join(
        str(path) for path in (binary, *system_paths) if path.is_dir()
    )
    environment.pop("CARGO_HOME", None)
    environment.pop("RUSTUP_HOME", None)
    return environment


def run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command visibly and fail with a concise release-tool error."""

    rendered = [os.fspath(part) for part in command]
    suffix = f"  # cwd={cwd}" if cwd is not None else ""
    print(f"$ {shlex.join(rendered)}{suffix}")
    if dry_run:
        return subprocess.CompletedProcess(rendered, 0, "", "")
    completed = subprocess.run(
        rendered,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        capture_output=capture_output,
        text=True,
    )
    if completed.returncode != 0:
        if capture_output:
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
        raise ReleaseError(
            f"command exited with {completed.returncode}: {shlex.join(rendered)}"
        )
    return completed


def check_dependency_gate(mode: str) -> None:
    """Call the normative dependency gate in the requested mode."""

    if mode not in {"candidate", "release"}:
        raise ReleaseError(f"unsupported dependency-gate mode: {mode}")
    command: list[str | os.PathLike[str]] = [sys.executable, DEPENDENCY_GATE]
    if mode == "candidate":
        command.append("--candidate")
    run(command, cwd=ROOT, env=clean_environment(mode=mode))


def require_clean_checkout(*, allow_dirty_candidate: bool, mode: str) -> None:
    """Require a clean checkout for candidate builds unless explicitly allowed."""

    if mode != "candidate" or allow_dirty_candidate:
        return
    completed = run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", "."],
        cwd=ROOT,
        env=clean_environment(),
        capture_output=True,
    )
    dirty = completed.stdout.strip()
    if dirty:
        preview = "\n".join(dirty.splitlines()[:20])
        raise ReleaseError(
            "candidate artifacts require a clean checkout; Git reported:\n" + preview
        )


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def containing_workspace() -> Path:
    """Return the checkout's containing workspace without treating / as one."""

    parent = ROOT.parent.resolve()
    filesystem_root = Path(parent.anchor)
    if parent == filesystem_root:
        return ROOT.resolve()
    if parent.name == "PREPARE_STANDALONE_PYAMPLICOL":
        return parent.parent
    return parent


def _release_temporary_path_is_isolated(path: Path) -> bool:
    """Accept external scratch or the checkout's dedicated artifact tree."""

    workspace = containing_workspace()
    if not is_relative_to(path, workspace):
        return True
    artifact_root = (ROOT / ".artifacts").resolve()
    return is_relative_to(path, artifact_root)


@contextmanager
def external_temporary_directory(prefix: str) -> Iterator[Path]:
    """Create scratch outside tracked source, including under ``.artifacts``."""

    with tempfile.TemporaryDirectory(prefix=prefix) as raw:
        path = Path(raw).resolve()
        if not _release_temporary_path_is_isolated(path):
            raise ReleaseError(
                "temporary release workspace must be outside "
                f"{containing_workspace()} or below {ROOT / '.artifacts'}: {path}"
            )
        yield path


def _safe_archive_name(name: str) -> PurePosixPath:
    if not name or "\\" in name:
        raise ReleaseError(f"unsafe sdist member name: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ReleaseError(f"unsafe sdist member path: {name}")
    return relative


def safe_extract_sdist(sdist: Path, destination: Path) -> Path:
    """Extract one regular-file sdist tree without trusting tar paths or links."""

    if not sdist.is_file():
        raise ReleaseError(f"sdist does not exist: {sdist}")
    destination.mkdir(parents=True, exist_ok=True)
    roots: set[str] = set()
    seen: set[PurePosixPath] = set()
    unpacked_size = 0
    with tarfile.open(sdist, mode="r:*") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_FILES:
            raise ReleaseError(f"sdist contains too many members: {len(members)}")
        for member in members:
            relative = _safe_archive_name(member.name.rstrip("/"))
            if relative in seen:
                raise ReleaseError(f"sdist contains duplicate member: {relative}")
            seen.add(relative)
            roots.add(relative.parts[0])
            if not (member.isdir() or member.isreg()):
                raise ReleaseError(
                    f"sdist may contain only directories and regular files: {relative}"
                )
            unpacked_size += member.size
            if unpacked_size > MAX_UNPACKED_SDIST_BYTES:
                raise ReleaseError("sdist exceeds the unpacked size safety limit")

            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ReleaseError(f"could not read sdist member: {relative}")
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            target.chmod((member.mode & 0o777) or 0o644)
    if len(roots) != 1:
        raise ReleaseError(
            "sdist must contain exactly one top-level directory, found "
            + ", ".join(sorted(roots))
        )
    root = destination / next(iter(roots))
    if not root.is_dir():
        raise ReleaseError("sdist top-level member is not a directory")
    return root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def exactly_one(paths: Sequence[Path], description: str) -> Path:
    candidates = sorted(path.resolve() for path in paths)
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise ReleaseError(
            f"expected exactly one {description}, found {len(candidates)}: {rendered}"
        )
    return candidates[0]


__all__ = [
    "CANDIDATE_ARTIFACTS",
    "DEPENDENCY_GATE",
    "DEPENDENCY_WHEELHOUSE",
    "DEPLOYMENT_ROOT",
    "DIST",
    "ROOT",
    "ReleaseError",
    "build_mode",
    "check_dependency_gate",
    "clean_environment",
    "containing_workspace",
    "exactly_one",
    "external_temporary_directory",
    "require_clean_checkout",
    "run",
    "runtime_environment",
    "safe_extract_sdist",
    "sha256",
]
