#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Capture strict fixture-v2 data from retained schema-v3 artifacts.

Run this developer-only command through a repository-external 30 GB memory
watchdog. The command deliberately contains no process killer or memory monitor.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

capture = importlib.import_module("tools.developer.reference_capture")
common = importlib.import_module("tools.developer.reference_capture.common")
provenance = importlib.import_module("tools.developer.reference_capture.provenance")


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external:{resolved.name}>"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the strict compact reference fixture v2 from retained "
            "schema-v3 artifacts and independent oracles."
        ),
        epilog=(
            "This command has no internal memory killer. Invoke it through a "
            "repository-external watchdog limited to 30 GB, then pass "
            "--external-watchdog-gb 30 to record that requirement."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--generate",
        dest="artifact_mode",
        action="store_const",
        const="generate",
        help="generate new retained artifacts; fail if any destination exists",
    )
    mode.add_argument(
        "--reuse-artifacts",
        "--reuse",
        dest="artifact_mode",
        action="store_const",
        const="reuse",
        help="explicitly reuse the complete retained schema-v3 artifact set",
    )
    parser.add_argument(
        "--output-directory",
        "--output-dir",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "reference",
        help=(
            "directory for the fixture, two evidence documents, and final "
            "bundle manifest (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "retained artifact directory; default is revision-scoped under "
            ".artifacts/reference-fixture-v2"
        ),
    )
    parser.add_argument(
        "--legacy-repository",
        type=Path,
        default=common.DEPENDENCIES / "checkouts" / "legacy-amplicol",
        help="pinned legacy AmpliCol checkout (default: %(default)s)",
    )
    parser.add_argument(
        "--legacy-jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="parallel build jobs for the legacy Fortran oracle (default: %(default)s)",
    )
    parser.add_argument(
        "--external-watchdog-gb",
        type=int,
        required=True,
        help="confirm the repository-external watchdog limit; must be exactly 30",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    options = parser.parse_args(arguments)
    try:
        revision = provenance.require_clean_tracked_tree(ROOT)
        artifact_root = (
            provenance.default_artifact_root(revision)
            if options.artifact_root is None
            else options.artifact_root.expanduser().resolve(strict=False)
        )
        config = capture.CaptureConfig(
            output_directory=options.output_directory.expanduser().resolve(
                strict=False
            ),
            artifact_root=artifact_root,
            legacy_repository=options.legacy_repository.expanduser().resolve(
                strict=False
            ),
            legacy_jobs=options.legacy_jobs,
            artifact_mode=options.artifact_mode,
            capture_command=(
                "python",
                "tools/developer/capture_reference_fixture_v2.py",
                (
                    "--generate"
                    if options.artifact_mode == "generate"
                    else "--reuse-artifacts"
                ),
                "--output-directory",
                _portable_path(options.output_directory),
                "--artifact-root",
                _portable_path(artifact_root),
                "--legacy-repository",
                _portable_path(options.legacy_repository),
                "--legacy-jobs",
                str(options.legacy_jobs),
                "--external-watchdog-gb",
                str(options.external_watchdog_gb),
            ),
            external_watchdog_gb=options.external_watchdog_gb,
        )
        result = capture.run_capture(config)
    except (capture.CaptureError, OSError, ValueError) as error:
        parser.exit(2, f"capture_reference_fixture_v2.py: error: {error}\n")
    print(f"wrote {result.fixture_path}")
    for path in result.evidence_paths:
        print(f"wrote {path}")
    print(f"wrote {result.bundle_manifest_path}")
    for path in result.artifact_paths:
        print(f"retained artifact {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
