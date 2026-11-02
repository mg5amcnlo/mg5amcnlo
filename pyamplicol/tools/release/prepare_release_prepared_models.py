# SPDX-License-Identifier: 0BSD
"""Build and audit the non-publishable release prepared-model bootstrap wheel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tomllib
import zipfile
from collections.abc import Sequence
from email.parser import BytesParser
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_CONTEXT = "release-prepared-model-producer-v1"
with (ROOT / "dependencies" / "release-lock.toml").open("rb") as _release_stream:
    EXPECTED_VERSION = str(tomllib.load(_release_stream)["project"]["version"])


class ReleasePreparedModelError(RuntimeError):
    """The release prepared-model producer contract was not satisfied."""


def _load_build_backend() -> ModuleType:
    path = ROOT / "build_backend" / "_pyamplicol_build.py"
    build_backend = str(path.parent)
    inserted = False
    if build_backend not in sys.path:
        sys.path.insert(0, build_backend)
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location(
            "_pyamplicol_release_prepared_model_build",
            path,
        )
        if spec is None or spec.loader is None:
            raise ReleasePreparedModelError(f"cannot load build backend from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(build_backend)


def _json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(archive.read(name))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleasePreparedModelError(
            f"bootstrap wheel member is invalid: {name}"
        ) from error
    if not isinstance(payload, dict):
        raise ReleasePreparedModelError(
            f"bootstrap wheel member is not an object: {name}"
        )
    return payload


def audit_bootstrap_wheel(wheel: Path) -> dict[str, object]:
    """Check that a bootstrap wheel is useful for production but not publication."""

    path = wheel.resolve(strict=True)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        build_info_names = [
            name for name in names if name == "pyamplicol/_build_info.json"
        ]
        if build_info_names != ["pyamplicol/_build_info.json"]:
            raise ReleasePreparedModelError(
                "bootstrap wheel must contain one non-publishable build marker"
            )
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ReleasePreparedModelError(
                "bootstrap wheel must contain one distribution METADATA member"
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata.get("Name") != "pyamplicol":
            raise ReleasePreparedModelError(
                "bootstrap wheel distribution name is not pyamplicol"
            )
        if metadata.get("Version") != EXPECTED_VERSION:
            raise ReleasePreparedModelError(
                f"bootstrap wheel must retain release version {EXPECTED_VERSION!r}"
            )
        marker = _json_member(archive, build_info_names[0])
        expected_marker = {
            "schema_version": 1,
            "publishable": False,
            "candidate_fingerprint": None,
            "release_prepared_model_bootstrap": True,
            "selftest_fixture_bootstrap": True,
            "version": EXPECTED_VERSION,
        }
        for key, expected in expected_marker.items():
            if marker.get(key) != expected:
                raise ReleasePreparedModelError(
                    f"bootstrap wheel marker {key} is invalid"
                )
        native_digest = marker.get("native_build_inputs_sha256")
        if (
            not isinstance(native_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", native_digest) is None
        ):
            raise ReleasePreparedModelError(
                "bootstrap wheel native source identity is invalid"
            )
        source_checkout = marker.get("source_checkout")
        if source_checkout != str(ROOT.resolve()):
            raise ReleasePreparedModelError(
                "bootstrap wheel source checkout is not the active repository"
            )
        prepared_prefix = "pyamplicol/assets/prepared_models/"
        prepared_payloads = [
            name
            for name in names
            if name.startswith(prepared_prefix)
            and name != f"{prepared_prefix}__init__.py"
        ]
        if prepared_payloads:
            raise ReleasePreparedModelError(
                "bootstrap wheel contains stale prepared-model payloads"
            )
        if any(name.startswith("pyamplicol/assets/selftest/") for name in names):
            raise ReleasePreparedModelError(
                "bootstrap wheel contains a stale self-test fixture"
            )
        if any(
            name == "release_assets" or name.startswith("release_assets/")
            for name in names
        ):
            raise ReleasePreparedModelError(
                "bootstrap wheel contains the release prepared-model source store"
            )
    return {
        "candidate_fingerprint": None,
        "native_build_inputs_sha256": native_digest,
        "path": str(path),
        "publishable": False,
        "release_prepared_model_bootstrap": True,
        "selftest_fixture_bootstrap": True,
        "version": EXPECTED_VERSION,
    }


def build_bootstrap_wheel(output_directory: Path) -> dict[str, object]:
    """Build the sole wheel authorized to produce release prepared-model packs."""

    if os.environ.get("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", "0") != "0":
        raise ReleasePreparedModelError(
            "candidate prepared-model bootstrap cannot be combined with "
            "release production"
        )
    if os.environ.get("PYAMPLICOL_BUILD_MODE", "release") != "release":
        raise ReleasePreparedModelError(
            "release prepared-model production requires PYAMPLICOL_BUILD_MODE=release"
        )
    destination = output_directory.expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ReleasePreparedModelError(
            f"bootstrap output directory is not empty: {destination}"
        )
    backend = _load_build_backend()
    filename = backend.build_release_prepared_model_bootstrap_wheel(
        str(destination),
        bootstrap_context=BOOTSTRAP_CONTEXT,
    )
    return audit_bootstrap_wheel(destination / filename)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or audit the explicitly non-publishable release prepared-model "
            "bootstrap wheel."
        )
    )
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("bootstrap-wheel")
    build.add_argument("output_directory", type=Path)
    audit = commands.add_parser("audit-bootstrap-wheel")
    audit.add_argument("wheel", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "bootstrap-wheel":
            result = build_bootstrap_wheel(arguments.output_directory)
        else:
            result = audit_bootstrap_wheel(arguments.wheel)
    except (OSError, ReleasePreparedModelError, RuntimeError) as error:
        print(f"release-prepared-models: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
