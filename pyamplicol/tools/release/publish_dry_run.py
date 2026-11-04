#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Audit ordinary package artifacts and print, but never run, an upload."""

from __future__ import annotations

import argparse
import shlex
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from _artifacts import RELEASE_TARGETS, ArtifactError, audit_sdist, audit_wheel
from _common import (
    CANDIDATE_ARTIFACTS,
    DIST,
    ROOT,
    ReleaseError,
    build_mode,
    check_dependency_gate,
    clean_environment,
    exactly_one,
    run,
)
from install_wheel import interpreter_tags, select_compatible_wheel


def _collect_artifacts(directory: Path) -> list[Path]:
    """Find package files recursively and reject ambiguous duplicate names."""

    directory = directory.resolve()
    by_name: dict[str, Path] = {}
    for pattern in ("pyamplicol-*.whl", "pyamplicol-*.tar.gz"):
        for path in sorted(directory.rglob(pattern)):
            existing = by_name.get(path.name)
            if existing is not None:
                raise ArtifactError(
                    f"multiple artifacts share filename {path.name}: "
                    f"{existing} and {path}"
                )
            by_name[path.name] = path.resolve()
    if not by_name:
        raise ArtifactError(f"no pyamplicol wheels or sdist found under {directory}")
    return [by_name[name] for name in sorted(by_name)]


def _validated_artifacts(
    directory: Path,
    *,
    mode: str,
    require_all_targets: bool,
) -> list[Path]:
    """Audit package files and enforce the expected release inventory."""

    artifacts = _collect_artifacts(directory)
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    if not wheels:
        raise ArtifactError("artifact directory contains no wheels")
    if mode == "candidate":
        if sdists:
            raise ArtifactError(
                "candidate artifacts are wheel-only and non-publishable"
            )
    else:
        exactly_one(sdists, "release sdist")

    if sdists:
        audit_sdist(sdists[0], mode=mode)
    reports = [
        audit_wheel(path, mode=mode, native_scan=False)
        for path in wheels
    ]
    targets = [report.target for report in reports]
    if len(set(targets)) != len(targets):
        raise ArtifactError(
            "artifact directory contains multiple wheels for one target"
        )
    if require_all_targets and set(targets) != set(RELEASE_TARGETS):
        raise ArtifactError(
            "release wheel target set is incomplete: " + ", ".join(sorted(targets))
        )
    return artifacts


def _stage_artifacts(artifacts: list[Path], destination: Path) -> list[Path]:
    """Copy ordinary wheel and sdist files into one upload directory."""

    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ArtifactError(f"artifact destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for source in artifacts:
        target = destination / source.name
        shutil.copy2(source, target)
        staged.append(target)
    return staged


def _build_default_artifacts(mode: str) -> Path:
    command: list[str | Path] = [
        sys.executable,
        ROOT / "tools" / "release" / "build_release_artifacts.py",
    ]
    if mode == "candidate":
        command.append("--candidate")
    run(command, cwd=ROOT, env=clean_environment(mode=mode))
    return CANDIDATE_ARTIFACTS if mode == "candidate" else DIST


def _run_clean_install(wheels: list[Path]) -> None:
    wheel = select_compatible_wheel(wheels, interpreter_tags(Path(sys.executable)))
    run(
        [
            sys.executable,
            ROOT / "tools" / "release" / "test_deployment.py",
            "--python",
            sys.executable,
            "--wheel",
            wheel,
            "--no-build",
        ],
        cwd=ROOT,
        env=clean_environment(mode="release"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="existing flat or recursive directory containing wheels and an sdist",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="empty directory in which to stage ordinary upload files",
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--require-all-targets", action="store_true")
    parser.add_argument("--skip-twine-check", action="store_true")
    parser.add_argument(
        "--skip-clean-install",
        action="store_true",
        help="skip only when every platform artifact was already deployment-tested",
    )
    parser.add_argument("--repository", choices=("pypi", "testpypi"), default="pypi")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = build_mode(candidate=args.candidate)
    check_dependency_gate(mode)
    if args.artifact_dir is None:
        source = (
            CANDIDATE_ARTIFACTS if mode == "candidate" else DIST
        ) if args.no_build else _build_default_artifacts(mode)
    else:
        source = args.artifact_dir

    artifacts = _validated_artifacts(
        source,
        mode=mode,
        require_all_targets=args.require_all_targets,
    )
    if args.output_dir is not None:
        if args.output_dir.resolve() == source.resolve():
            raise ArtifactError("artifact source and output directories must differ")
        artifacts = _stage_artifacts(artifacts, args.output_dir)

    if not args.skip_twine_check:
        run(
            [sys.executable, "-m", "twine", "check", *artifacts],
            cwd=ROOT,
            env=clean_environment(),
        )
    if mode == "release" and not args.skip_clean_install:
        _run_clean_install([path for path in artifacts if path.suffix == ".whl"])
    print("Validated package artifacts:")
    for artifact in artifacts:
        print(f"  {artifact.resolve()}")
    if mode == "candidate":
        print("Candidate artifacts are non-publishable; no upload command is emitted.")
        return 0
    command = [
        sys.executable,
        "-m",
        "twine",
        "upload",
        "--repository",
        args.repository,
        "--",
        *(str(path.resolve()) for path in artifacts),
    ]
    print("Dry run only; upload command (not executed):")
    print(shlex.join(command))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
