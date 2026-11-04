#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Keep the pyAmpliCol version equal to the MadGraph7 VERSION file.

Cargo needs a literal version, so the MadGraph7 version is written into every
file that carries it. ``--check`` only reports a mismatch (exit status 1).

    sync_version.py [--check] [--version V]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MG_VERSION_FILE = ROOT.parent / "VERSION"
WORKSPACE_CRATES = ("rusticol-capi", "rusticol-core", "rusticol-python")


def madgraph_version() -> str:
    text = MG_VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r"(?m)^version\s*=\s*(\S+)", text)
    if match is None:
        raise RuntimeError(f"no version in {MG_VERSION_FILE}")
    return match.group(1)


def _section_value(text: str, section: str, version: str) -> str:
    # Replace `version = "..."` in the first line of the given TOML section.
    pattern = re.compile(
        rf'(?ms)(^\[{re.escape(section)}\]\n(?:(?!^\[).)*?^version = )"[^"]*"'
    )
    return pattern.sub(rf'\g<1>"{version}"', text, count=1)


def _cargo_lock(text: str, version: str) -> str:
    for crate in WORKSPACE_CRATES:
        text = re.sub(
            rf'(?m)(^name = "{crate}"\nversion = )"[^"]*"',
            rf'\g<1>"{version}"',
            text,
        )
    return text


def _versions_py(text: str, version: str) -> str:
    return re.sub(
        r'(def package_version\(default: str = )"[^"]*"',
        rf'\g<1>"{version}"',
        text,
    )


EDITS = {
    Path("Cargo.toml"): lambda text, v: _section_value(text, "workspace.package", v),
    Path("Cargo.lock"): _cargo_lock,
    Path("dependencies/release-lock.toml"): lambda text, v: _section_value(
        text, "project", v
    ),
    Path("src/pyamplicol/_internal/versions.py"): _versions_py,
}


def sync(version: str, *, check: bool) -> list[Path]:
    """Return the files whose version differs, updating them unless checking."""

    stale: list[Path] = []
    for relative, edit in EDITS.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        updated = edit(text, version)
        if updated != text:
            stale.append(relative)
            if not check:
                path.write_text(updated, encoding="utf-8")
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--version", help="default: the MadGraph7 VERSION file")
    args = parser.parse_args(argv)
    version = args.version or madgraph_version()
    stale = sync(version, check=args.check)
    if args.check and stale:
        names = ", ".join(path.as_posix() for path in stale)
        print(f"pyamplicol version is not {version} in: {names}", file=sys.stderr)
        return 1
    for path in stale:
        print(f"set version {version} in {path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
