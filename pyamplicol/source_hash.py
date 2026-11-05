# SPDX-License-Identifier: 0BSD
"""Hash of the pyAmpliCol sources that end up in an installed package.

The build stores it as ``pyamplicol._source_hash.SOURCE_HASH``, so MadGraph7
can warn when the installed package no longer matches the source tree (as for
madspace). Generated files (native libraries, staged SDK, prepared-model packs
from the release archive, build provenance) are left out.
"""

import hashlib
from pathlib import Path, PurePosixPath

_ROOT = Path(__file__).resolve().parent

_SOURCE_DIRS = ("src", "rust")
_SOURCE_FILES = ("Cargo.toml", "Cargo.lock", "pyproject.toml", "rust-toolchain.toml")

_IGNORED_DIR_NAMES = {"__pycache__", "target"}
_IGNORED_FILE_NAMES = {".DS_Store", "_build_info.json", "_source_hash.py"}
_IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".pyd", ".pyamplicol-model"}
_IGNORED_TREES = tuple(
    PurePosixPath(path)
    for path in (
        "src/pyamplicol/_sdk/lib",
        "src/pyamplicol/_sdk/include",
        "src/pyamplicol/_sdk/fortran",
        "src/pyamplicol/_sdk/sboms",
        "src/pyamplicol/_sdk/link.json",
        "src/pyamplicol/_sdk/metadata.json",
        "src/pyamplicol/_examples",
    )
)


def _ignored(relative: PurePosixPath) -> bool:
    if _IGNORED_DIR_NAMES & set(relative.parts):
        return True
    if relative.name in _IGNORED_FILE_NAMES or relative.suffix in _IGNORED_SUFFIXES:
        return True
    if (
        relative.name.endswith(".metadata.json")
        and relative.parent.name == "prepared_models"
    ):
        return True
    # Self-test fixtures other than the tracked portable one are generated.
    parts = relative.parts
    selftest = parts[:4] == ("src", "pyamplicol", "assets", "selftest")
    if selftest and len(parts) > 5 and parts[4] != "portable-64le":
        return True
    return any(relative == tree or tree in relative.parents for tree in _IGNORED_TREES)


def _iter_source_files(root: Path):
    for dir_name in _SOURCE_DIRS:
        for path in (root / dir_name).rglob("*"):
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if path.is_file() and not _ignored(relative):
                yield path
    for file_name in _SOURCE_FILES:
        yield root / file_name


def compute_source_hash(root: Path = _ROOT) -> str:
    root = Path(root).resolve()
    paths = sorted(
        _iter_source_files(root), key=lambda p: p.relative_to(root).as_posix()
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as f:
            digest.update(hashlib.file_digest(f, "sha256").digest())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    print(compute_source_hash())
