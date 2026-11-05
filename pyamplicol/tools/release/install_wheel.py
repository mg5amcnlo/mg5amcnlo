#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Install the one audited pyAmpliCol wheel matching a target interpreter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from _artifacts import audit_wheel
from _common import (
    CANDIDATE_ARTIFACTS,
    DEPENDENCY_WHEELHOUSE,
    DIST,
    ROOT,
    ReleaseError,
    build_mode,
    clean_environment,
    run,
)

_TAG_SCRIPT = r"""
import json
try:
    from packaging.tags import sys_tags
except ImportError:
    from pip._vendor.packaging.tags import sys_tags
print(json.dumps([str(tag) for tag in sys_tags()]))
"""


def interpreter_tags(python: Path) -> list[str]:
    completed = run(
        [python, "-I", "-c", _TAG_SCRIPT],
        env=clean_environment(),
        capture_output=True,
    )
    try:
        tags = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(
            f"target interpreter did not report valid packaging tags: {python}"
        ) from error
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ReleaseError(f"target interpreter reported invalid tags: {python}")
    return tags


def wheel_tags(path: Path) -> set[str]:
    fields = path.name[:-4].rsplit("-", 3)
    if path.suffix != ".whl" or len(fields) != 4:
        raise ReleaseError(f"invalid wheel filename: {path.name}")
    _, python_tags, abi_tags, platform_tags = fields
    return {
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag in python_tags.split(".")
        for abi_tag in abi_tags.split(".")
        for platform_tag in platform_tags.split(".")
    }


def select_compatible_wheel(
    wheels: Sequence[Path], supported_tags: Sequence[str]
) -> Path:
    ranking = {tag: index for index, tag in enumerate(supported_tags)}
    candidates: list[tuple[int, Path]] = []
    for wheel in wheels:
        matches = [ranking[tag] for tag in wheel_tags(wheel) if tag in ranking]
        if matches:
            candidates.append((min(matches), wheel.resolve()))
    if not candidates:
        raise ReleaseError("no pyAmpliCol wheel matches the target interpreter")
    candidates.sort(key=lambda item: (item[0], item[1].name))
    best_rank = candidates[0][0]
    best = [path for rank, path in candidates if rank == best_rank]
    if len(best) != 1:
        raise ReleaseError(
            "multiple wheels are equally compatible with the target interpreter: "
            + ", ".join(path.name for path in best)
        )
    return best[0]


def wheelhouse_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        directory.resolve()
        for directory in {path.parent for path in root.rglob("*.whl")}
    )


def build_if_missing(mode: str, artifact_directory: Path) -> None:
    if list(artifact_directory.glob("pyamplicol-*.whl")):
        return
    command: list[str | os.PathLike[str]] = [
        sys.executable,
        ROOT / "tools" / "release" / "build_release_artifacts.py",
        "--output-dir",
        artifact_directory,
    ]
    if mode == "candidate":
        command.append("--candidate")
    run(command, cwd=ROOT, env=clean_environment(mode=mode))


def install_wheel(
    wheel: Path,
    *,
    python: Path,
    mode: str,
    no_dependencies: bool,
    wheelhouses: Sequence[Path],
    dry_run: bool,
) -> None:
    audit_wheel(wheel, mode=mode)
    command: list[str | os.PathLike[str]] = [
        python,
        "-I",
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--only-binary=:all:",
        "--index-url",
        "https://pypi.org/simple",
    ]
    if no_dependencies:
        command.append("--no-deps")
    if mode == "candidate":
        for directory in wheelhouses:
            command.extend(("--find-links", directory))
    elif wheelhouses:
        raise ReleaseError("release installation cannot use a local wheelhouse")
    command.append(wheel.resolve())
    run(command, env=clean_environment(), dry_run=dry_run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--wheelhouse", type=Path, action="append", default=[])
    parser.add_argument("--no-deps", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = build_mode(candidate=args.candidate)
    artifact_directory = args.artifact_dir or (
        CANDIDATE_ARTIFACTS if mode == "candidate" else DIST
    )
    if args.wheel is not None:
        wheels = [args.wheel]
    else:
        if not args.no_build:
            build_if_missing(mode, artifact_directory)
        wheels = sorted(artifact_directory.glob("pyamplicol-*.whl"))
    wheel = select_compatible_wheel(wheels, interpreter_tags(args.python))
    wheelhouses = [path.resolve() for path in args.wheelhouse]
    if mode == "candidate" and not wheelhouses:
        wheelhouses = wheelhouse_directories(DEPENDENCY_WHEELHOUSE)
    install_wheel(
        wheel,
        python=args.python,
        mode=mode,
        no_dependencies=args.no_deps,
        wheelhouses=wheelhouses,
        dry_run=args.dry_run,
    )
    print(f"Installed {wheel.name} with {args.python}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
