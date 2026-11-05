# SPDX-License-Identifier: 0BSD
"""Copy pyAmpliCol's wheel-owned example models into an editable workspace."""

from __future__ import annotations

import argparse
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=Path("models"),
        help="destination directory (default: ./models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="merge into a non-empty destination without removing other files",
    )
    return parser


def _copy_resource_tree(source: Traversable, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "__pycache__":
            continue
        target = destination / child.name
        if child.is_dir():
            _copy_resource_tree(child, target)
        elif child.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(child.read_bytes())


def main() -> int:
    args = _parser().parse_args()
    destination = args.destination.expanduser().resolve(strict=False)
    if destination.exists() and not destination.is_dir():
        raise SystemExit(f"destination is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()) and not args.force:
        raise SystemExit(
            f"destination is not empty: {destination}; pass --force to merge"
        )

    model_assets = resources.files("pyamplicol.assets").joinpath("models")
    if not model_assets.is_dir():
        raise SystemExit("installed pyAmpliCol package contains no model assets")
    _copy_resource_tree(model_assets, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
