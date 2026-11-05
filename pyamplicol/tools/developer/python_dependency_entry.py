#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run a module with appended dependencies and fail closed on package shadowing."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path


class EntryError(RuntimeError):
    """Raised when dependency activation would change the measured package."""


def _pyamplicol_origin() -> Path:
    specification = importlib.util.find_spec("pyamplicol")
    origin = None if specification is None else specification.origin
    if origin is None:
        raise EntryError("the selected interpreter has no pyamplicol package")
    return Path(origin).resolve(strict=True)


def activate(path: Path) -> dict[str, str]:
    dependency_site = path.expanduser().resolve(strict=True)
    if not dependency_site.is_dir():
        raise EntryError(f"dependency site is not a directory: {dependency_site}")
    before = _pyamplicol_origin()
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(dependency_site))
        for package in ("numpy", "symbolica", "ufo_model_loader"):
            importlib.import_module(package)
    except ImportError as error:
        raise EntryError(
            f"dependency site does not provide required package {error.name!r}"
        ) from error
    finally:
        sys.path[:] = original_path
    if str(dependency_site) not in sys.path:
        sys.path.append(str(dependency_site))
    after = _pyamplicol_origin()
    distribution = importlib.metadata.distribution("pyamplicol")
    distribution_origin = Path(
        str(distribution.locate_file("pyamplicol/__init__.py"))
    ).resolve(strict=True)
    if before != after or after != distribution_origin:
        raise EntryError(
            "dependency activation changed the measured pyamplicol origin: "
            f"before={before}, after={after}, distribution={distribution_origin}"
        )
    for package in ("numpy", "symbolica", "ufo_model_loader"):
        specification = importlib.util.find_spec(package)
        origin = None if specification is None else specification.origin
        if origin is None:
            raise EntryError(f"required dependency {package!r} has no import origin")
        resolved_origin = Path(origin).resolve(strict=True)
        try:
            resolved_origin.relative_to(dependency_site)
        except ValueError as error:
            raise EntryError(
                f"required dependency {package!r} was not loaded from "
                f"{dependency_site}: {resolved_origin}"
            ) from error
    return {
        "dependency_site": str(dependency_site),
        "pyamplicol_origin": str(after),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dependency-site", type=Path, required=True)
    result.add_argument("--module", required=True)
    result.add_argument("arguments", nargs=argparse.REMAINDER)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        activate(arguments.dependency_site)
    except (EntryError, OSError) as error:
        print(f"python-dependency-entry: {error}", file=sys.stderr)
        return 2
    target_arguments = list(arguments.arguments)
    if target_arguments[:1] == ["--"]:
        target_arguments.pop(0)
    sys.argv = [arguments.module, *target_arguments]
    runpy.run_module(arguments.module, run_name="__main__", alter_sys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
