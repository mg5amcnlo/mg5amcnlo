"""Is this madtrex build compatible with the madspace next to it?

_madtrex_py is compiled against madspace's installed headers: its classes
(MatrixElementApi, EventFile, ...) and their inline methods are baked into
it. A madspace reinstalled with other headers after madtrex was built (a
newer checkout, 'install madspace --bin', ...) is a silent binary mismatch,
so CMake records a digest of the headers it compiled against in
_build_info.json, and this module compares it with the madspace that would
be imported.

Pure Python and standard library only: MadGraph loads this file directly
(madgraph/various/reweight_backend.py) to tell whether MadtRex can run,
without importing madtrex, madspace or pyrex.
"""

import hashlib
import importlib.util
import json
import os

BUILD_INFO = "_build_info.json"

REBUILD_HINT = (
    "rebuild it: python madtrex/install.py --clean --yes "
    "(or 'install madtrex --clean -y' in MadGraph)"
)


def headers_digest(include_dir):
    """sha256 over the files under include_dir: one "<relative path>
    <sha256>" line per file, in sorted path order. madtrex/CMakeLists.txt
    computes the same digest at build time."""
    lines = []
    for root, _dirs, files in os.walk(include_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, include_dir).replace(os.sep, "/")
            with open(path, "rb") as f:
                lines.append((rel, hashlib.sha256(f.read()).hexdigest()))
    text = "".join("%s %s\n" % line for line in sorted(lines))
    return hashlib.sha256(text.encode()).hexdigest()


def read_build_info(package_dir):
    """The _build_info.json of the madtrex package in package_dir, or None."""
    try:
        with open(os.path.join(package_dir, BUILD_INFO)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_madspace_dir():
    """The directory of the madspace package that 'import madspace' would
    load, without importing it, or None."""
    spec = importlib.util.find_spec("madspace")
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(os.path.abspath(spec.origin))


def build_problem(package_dir, madspace_dir):
    """Why the madtrex package in package_dir cannot be used with the
    madspace package in madspace_dir, or None when it can (or when there is
    nothing to compare: a build from before the digest was recorded)."""
    info = read_build_info(package_dir)
    if not info or not info.get("madspace_headers"):
        return None
    include_dir = os.path.join(madspace_dir, "include")
    if not os.path.isdir(include_dir):
        return None
    if headers_digest(include_dir) == info["madspace_headers"]:
        return None
    return (
        "madtrex (%s) was built against another madspace than the one in %s "
        "(madspace was reinstalled or updated since): %s"
        % (package_dir, madspace_dir, REBUILD_HINT)
    )
