#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Build audited candidate or release artifacts without publishing them."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from _artifacts import audit_sdist, audit_wheel
from _common import (
    CANDIDATE_ARTIFACTS,
    DIST,
    ROOT,
    ReleaseError,
    build_mode,
    check_dependency_gate,
    clean_environment,
    exactly_one,
    external_temporary_directory,
    require_clean_checkout,
    run,
)
from build_from_sdist import build_wheel_from_sdist


def _prepare_output(directory: Path) -> Path:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for pattern in (
            "pyamplicol-*.whl",
            "pyamplicol-*.tar.gz",
        )
        for path in directory.glob(pattern)
    ]
    if existing:
        raise ReleaseError(
            "artifact output must not contain a previous release bundle: "
            + ", ".join(path.name for path in sorted(existing))
        )
    return directory


def _build(
    python: Path,
    destination: Path,
    *,
    mode: str,
    wheel: bool,
    sdist: bool,
) -> None:
    command: list[str | Path] = [python, "-m", "build"]
    if wheel:
        command.append("--wheel")
    if sdist:
        command.append("--sdist")
    command.extend(("--outdir", destination))
    run(command, cwd=ROOT, env=clean_environment(mode=mode))


def build_release_artifacts(
    output_directory: Path,
    *,
    mode: str,
    python: Path,
    allow_dirty_candidate: bool,
    sdist_only: bool,
    retained_sdist_path: Path | None,
) -> list[Path]:
    python = python.resolve()
    if mode == "release" and allow_dirty_candidate:
        raise ReleaseError("--allow-dirty is available only for candidate builds")
    if sdist_only and retained_sdist_path is not None:
        raise ReleaseError("--sdist-only and --retained-sdist are mutually exclusive")
    if mode == "candidate" and sdist_only:
        raise ReleaseError("candidate artifact production is wheel-only")
    if mode == "candidate" and retained_sdist_path is not None:
        raise ReleaseError(
            "candidate sdists are non-publishable and are not release parity inputs"
        )
    check_dependency_gate(mode)
    if mode == "candidate":
        require_clean_checkout(
            allow_dirty_candidate=allow_dirty_candidate,
            mode=mode,
        )
    output_directory = _prepare_output(output_directory)

    with external_temporary_directory("pyamplicol-release-build-") as temporary:
        direct_directory = temporary / "direct"
        rebuilt_directory = temporary / "rebuilt"
        direct_directory.mkdir()
        rebuilt_directory.mkdir()

        if sdist_only:
            _build(python, direct_directory, mode=mode, wheel=False, sdist=True)
            sdist = exactly_one(
                list(direct_directory.glob("pyamplicol-*.tar.gz")), "built sdist"
            )
            audit_sdist(sdist, mode=mode)
            retained = output_directory / sdist.name
            shutil.copy2(sdist, retained)
            return [retained]

        _build(
            python,
            direct_directory,
            mode=mode,
            wheel=True,
            sdist=mode == "release" and retained_sdist_path is None,
        )
        direct_wheel = exactly_one(
            list(direct_directory.glob("pyamplicol-*.whl")),
            "direct-source wheel",
        )
        audit_wheel(direct_wheel, mode=mode)

        if mode == "candidate":
            candidate_wheel = output_directory / direct_wheel.name
            shutil.copy2(direct_wheel, candidate_wheel)
            return [candidate_wheel]

        if retained_sdist_path is None:
            retained_sdist_path = exactly_one(
                list(direct_directory.glob("pyamplicol-*.tar.gz")), "built sdist"
            )
        retained_sdist_path = retained_sdist_path.resolve()
        audit_sdist(retained_sdist_path, mode="release")
        rebuilt_wheel = build_wheel_from_sdist(
            retained_sdist_path,
            direct_wheel,
            rebuilt_directory,
            mode="release",
            python=python,
        )
        audit_wheel(rebuilt_wheel, mode="release")

        copied_wheel = output_directory / rebuilt_wheel.name
        shutil.copy2(rebuilt_wheel, copied_wheel)
        copied: list[Path] = [copied_wheel]
        if retained_sdist_path.parent == direct_directory:
            copied_sdist = output_directory / retained_sdist_path.name
            shutil.copy2(retained_sdist_path, copied_sdist)
            copied.append(copied_sdist)
        return copied


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a candidate build from a dirty checkout",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sdist-only", action="store_true")
    group.add_argument("--retained-sdist", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = build_mode(candidate=args.candidate)
    output = args.output_dir or (CANDIDATE_ARTIFACTS if mode == "candidate" else DIST)
    artifacts = build_release_artifacts(
        output,
        mode=mode,
        python=args.python,
        allow_dirty_candidate=args.allow_dirty,
        sdist_only=args.sdist_only,
        retained_sdist_path=args.retained_sdist,
    )
    heading = (
        "Retained NON-PUBLISHABLE candidate artifacts:"
        if mode == "candidate"
        else "Retained validated release artifacts:"
    )
    print(heading)
    for artifact in artifacts:
        print(f"  {artifact}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
