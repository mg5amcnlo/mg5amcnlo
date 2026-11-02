#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run Cargo against the clean release or candidate source overlay."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import _pyamplicol_build as backend  # noqa: E402


def _python_test_loader_updates(arguments: list[str]) -> dict[str, str]:
    """Expose only the selected interpreter's shared library to Rust tests."""

    if arguments[:1] != ["test"] or not sys.platform.startswith("linux"):
        return {}

    directories: list[Path] = []
    for variable in ("LIBDIR", "LIBPL"):
        value = sysconfig.get_config_var(variable)
        if not value:
            continue
        directory = Path(str(value)).resolve()
        if directory.is_dir() and directory not in directories:
            directories.append(directory)
    if not directories:
        raise RuntimeError(
            "the selected Python interpreter does not report a usable shared-library "
            "directory; Rust PyO3 tests cannot be launched"
        )
    return {"LD_LIBRARY_PATH": os.pathsep.join(map(str, directories))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("candidate", "release"), required=True)
    parser.add_argument("cargo_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = list(args.cargo_arguments)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    if not arguments:
        parser.error("a Cargo subcommand is required after '--'")

    with backend._overlay(args.mode) as (overlay, target):
        updates = {
            "CARGO_HOME": str(target.parent / "cargo-home"),
            "CARGO_TARGET_DIR": str(target),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        updates.update(_python_test_loader_updates(arguments))
        updates.update(backend._macos_native_build_updates())
        environment = backend._clean_environment(updates)
        completed = subprocess.run(
            ["cargo", *arguments],
            cwd=overlay,
            env=environment,
            check=False,
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
