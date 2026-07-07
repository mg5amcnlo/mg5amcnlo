import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

_SOURCE_DIRS = ("src", "include", "madspace")
_SOURCE_FILES = ("CMakeLists.txt", "instruction_set.yaml", "pyproject.toml")

# Artifacts that can appear on disk (e.g. compiled bytecode) but are not
# actually part of the source, and would otherwise make the hash depend on
# the local environment instead of just the source tree.
_IGNORED_DIR_NAMES = {"__pycache__"}
_IGNORED_FILE_NAMES = {".DS_Store"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _iter_source_files():
    for dir_name in _SOURCE_DIRS:
        for path in (_ROOT / dir_name).rglob("*"):
            if not path.is_file():
                continue
            if _IGNORED_DIR_NAMES & set(path.relative_to(_ROOT).parts):
                continue
            if path.name in _IGNORED_FILE_NAMES or path.suffix in _IGNORED_SUFFIXES:
                continue
            yield path
    for file_name in _SOURCE_FILES:
        yield _ROOT / file_name


def compute_source_hash() -> str:
    paths = sorted(_iter_source_files(), key=lambda p: p.relative_to(_ROOT).as_posix())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as f:
            digest.update(hashlib.file_digest(f, "sha256").digest())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    print(compute_source_hash())
