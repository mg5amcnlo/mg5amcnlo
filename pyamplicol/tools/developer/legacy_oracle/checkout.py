# SPDX-License-Identifier: 0BSD
"""Contributor-lock, Git checkout, and compiler provenance helpers."""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .model import (
    LOCK,
    PINNED_REFERENCE_BRANCH,
    PINNED_REFERENCE_REVISION,
    PINNED_SOURCE_URL,
    CompilerProvenance,
    LegacyOracleError,
)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    capture: bool = True,
    subprocess_module: Any = subprocess,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess_module.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=capture,
    )
    if completed.returncode != 0:
        detail = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        )
        raise LegacyOracleError(
            f"command exited with {completed.returncode}: {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return cast(subprocess.CompletedProcess[str], completed)


def _contributor_lock(*, lock_path: Path = LOCK) -> dict[str, Any]:
    try:
        with lock_path.open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError:
        # Published profiling runtimes do not redistribute contributor locks.
        # Manual campaigns inject the capability checkout's actual revision;
        # this record retains a usable reference for standalone oracle helpers.
        return {
            "legacy_amplicol": {
                "branch": PINNED_REFERENCE_BRANCH,
                "revision": PINNED_REFERENCE_REVISION,
                "source_url": PINNED_SOURCE_URL,
            }
        }


def expected_revision(*, lock: Mapping[str, Any] | None = None) -> str:
    contributor = _contributor_lock() if lock is None else lock
    return str(contributor["legacy_amplicol"]["revision"])


def checkout_branch(*, lock: Mapping[str, Any] | None = None) -> str:
    contributor = _contributor_lock() if lock is None else lock
    return str(contributor["legacy_amplicol"]["branch"])


def checkout_url(*, lock: Mapping[str, Any] | None = None) -> str:
    contributor = _contributor_lock() if lock is None else lock
    source = str(contributor["legacy_amplicol"]["source_url"])
    prefix = "git@github.com:"
    if source.startswith(prefix):
        return "https://github.com/" + source.removeprefix(prefix)
    return source


def validate_checkout(
    repository: Path,
    *,
    run: Any = _run,
    revision: Any = expected_revision,
) -> None:
    repository = repository.resolve()
    if not (repository / ".git").exists():
        raise LegacyOracleError(
            f"legacy AmpliCol checkout is absent: {repository}; run "
            "`just dev-install --with-legacy-amplicol`"
        )
    actual_revision = run(["git", "rev-parse", "HEAD"], cwd=repository).stdout.strip()
    if actual_revision != revision():
        raise LegacyOracleError(
            f"legacy AmpliCol is at {actual_revision}, expected {revision()}"
        )
    tracked_changes = run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repository,
    ).stdout.strip()
    if tracked_changes:
        raise LegacyOracleError(
            "legacy AmpliCol checkout contains tracked edits; the developer "
            "oracle must use the clean pinned branch revision"
        )


def prepare_checkout(
    repository: Path,
    *,
    run: Any = _run,
    validate: Any = validate_checkout,
    url: Any = checkout_url,
    branch: Any = checkout_branch,
    revision: Any = expected_revision,
) -> None:
    repository = repository.resolve()
    if repository.exists():
        validate(repository)
        return
    repository.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "init", str(repository)], cwd=repository.parent)
    run(["git", "remote", "add", "origin", url()], cwd=repository)
    run(
        ["git", "fetch", "--depth=1", "origin", branch()],
        cwd=repository,
        capture=False,
    )
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repository)
    validate(repository)


def _make_database_variable(database: str, name: str) -> str:
    matches = re.findall(rf"^{re.escape(name)}\s*[:?+]?=\s*(.*?)\s*$", database, re.M)
    if not matches or not matches[-1]:
        raise LegacyOracleError(f"legacy makefile does not define {name}")
    return cast(str, matches[-1])


def _compiler_provenance(
    repository: Path,
    *,
    run: Any = _run,
    shutil_module: Any = shutil,
) -> CompilerProvenance:
    database = run(
        [
            "make",
            "-f",
            "makefile",
            "PDF_BACKEND=internal",
            "-pn",
            "amplicol_color_probe",
        ],
        cwd=repository,
    ).stdout
    compiler_tokens = shlex.split(_make_database_variable(database, "FC"))
    if not compiler_tokens:
        raise LegacyOracleError("legacy makefile defines an empty Fortran compiler")
    compiler = shutil_module.which(compiler_tokens[0])
    if compiler is None:
        candidate = repository / compiler_tokens[0]
        if candidate.is_file():
            compiler = str(candidate.resolve())
    if compiler is None:
        raise LegacyOracleError(
            f"cannot resolve legacy Fortran compiler {compiler_tokens[0]!r}"
        )
    compiler_command = [compiler, *compiler_tokens[1:]]
    version_result = run([*compiler_command, "--version"], cwd=repository)
    version_lines = tuple(
        line.strip()
        for line in (version_result.stdout + "\n" + version_result.stderr).splitlines()
        if line.strip()
    )
    if not version_lines:
        raise LegacyOracleError("Fortran compiler emitted no version identity")
    target_result = run([*compiler_command, "-dumpmachine"], cwd=repository)
    target = target_result.stdout.strip()
    if not target:
        raise LegacyOracleError("Fortran compiler emitted no target triple")
    executable = repository / "amplicol_color_probe"
    if not executable.is_file():
        raise LegacyOracleError("amplicol_color_probe has not been built")
    return CompilerProvenance(
        identity=Path(compiler).name,
        version=version_lines[0],
        flags=tuple(shlex.split(_make_database_variable(database, "FFLAGS"))),
        target=target,
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
