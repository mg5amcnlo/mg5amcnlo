"""One-off madspace (and pyamplicol) bootstrap for the mg7 output.

The mg7 launcher needs the compiled ``madspace`` package. A git checkout ships
only the sources, so the first run has to build and install it. That bootstrap
used to live at the top of :mod:`launch.py`, executed as an import side effect.
That was fine while ``launch`` was only ever imported by ``bin/generate_events``
in a throw-away subprocess whose ``sys.argv``/``sys.stdin`` described the run
itself.

``launch`` is now also imported *in process* by MG5's ``launch`` command, where
neither of those is true: ``sys.argv`` is MG5's own command line (so a ``-f``
meant for, say, ``mg5_aMC -f`` would be misread as the run's force flag, and
conversely a ``launch -f`` would not be seen at all) and ``sys.stdin`` is MG5's,
which may be a terminal even when the run itself is scripted.

So the decision is made by the caller and passed in explicitly. Keeping it in a
module of its own -- one that must never import ``madspace`` -- lets MG5 run the
bootstrap *before* importing :mod:`launch`, which is what makes the heavy import
safe to do lazily.
"""

import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Locate the madspace installation bundled alongside MadGraph.
# madgraph/__init__.py lives one level below the MadGraph root, so .parents[1]
# reaches the root and then "madspace/install" is the local install prefix.
import madgraph as _mg_pkg

MG_ROOT = Path(_mg_pkg.__file__).parents[1]
MADSPACE_DIR = MG_ROOT / "madspace"
INSTALL_DIR = MADSPACE_DIR / "install"
PYAMPLICOL_DIR = MG_ROOT / "pyamplicol"
PYAMPLICOL_INSTALL_DIR = PYAMPLICOL_DIR / "install"
# MadtRex (event reweighting) is installed separately, on top of madspace;
# nothing here installs it -- see madtrex/install.py.
MADTREX_DIR = MG_ROOT / "madtrex"
MADTREX_INSTALL_DIR = MADTREX_DIR / "install"


def madspace_is_installed() -> bool:
    """True when a compiled madspace is already available to import."""
    return (INSTALL_DIR / "madspace").is_dir()


def _pyamplicol_package():
    """The directory of the pyamplicol package that ``import pyamplicol`` would
    load, or None. Found, not imported."""
    spec = importlib.util.find_spec("pyamplicol")
    if spec is None or not spec.origin:
        return None  # e.g. the source directory as a namespace package
    package = Path(spec.origin).parent
    if (package.resolve().is_relative_to(PYAMPLICOL_DIR.resolve())
            and not package.resolve().is_relative_to(PYAMPLICOL_INSTALL_DIR.resolve())):
        return None
    return package


def pyamplicol_is_installed() -> bool:
    """True when a compiled pyamplicol is already available to import, in
    pyamplicol/install or system-wide (``install pyamplicol --system``)."""
    return (PYAMPLICOL_INSTALL_DIR / "pyamplicol").is_dir() or _pyamplicol_package() is not None


def installed_pyamplicol() -> dict:
    """The version and source hash of the pyamplicol that would be imported
    (None where unknown), read without importing it."""
    package = _pyamplicol_package()
    if package is None:
        return {"version": None, "source_hash": None}
    source_hash = None
    hash_file = package / "_source_hash.py"
    if hash_file.is_file():
        match = re.search(r'SOURCE_HASH\s*=\s*"([0-9a-f]*)"', hash_file.read_text())
        source_hash = match.group(1) if match else None
    dist = next(importlib.metadata.distributions(
        name="pyamplicol", path=[str(package.parent)]), None)
    return {"version": dist.version if dist else None, "source_hash": source_hash}


def _run_installer(name, source_dir, interactive, jobs) -> None:
    """Run ``<source_dir>/install.py`` as the first-use bootstrap does."""
    if interactive is None:
        interactive = sys.stdin.isatty()
    print()
    print(f"You don't have {name} installed for this madgraph instance")
    print(f"Running the {name} installation script")
    print()

    install_cmd = [sys.executable, str(source_dir / "install.py")]
    # When the run is non-interactive (scripted / piped), install
    # non-interactively with a source build and default options
    # (--source --yes), and keep the installer away from our stdin (which
    # may carry the run's scripted card-editing commands); when
    # interactive, let it share the terminal so the user can answer.
    install_stdin = None if interactive else subprocess.DEVNULL
    if not interactive:
        install_cmd += ["--source", "--yes"]
    if jobs:
        install_cmd += ["-j", str(jobs)]
    # Expose madgraph on PYTHONPATH so the installer subprocess can import
    # cmd.ask for its prompts.
    install_env = os.environ.copy()
    install_env["PYTHONPATH"] = os.pathsep.join(
        [str(MG_ROOT)]
        + ([install_env["PYTHONPATH"]] if install_env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(install_cmd, env=install_env, stdin=install_stdin)
    if result.returncode != 0:
        raise RuntimeError(f"{name} installation failed — see output above")


def ensure_madspace(interactive=None, jobs=None) -> None:
    """Install madspace if needed, then put it on ``sys.path``.

    ``interactive`` says whether the installer may take over the terminal and
    ask questions. ``None`` falls back to auto-detection from ``sys.stdin``,
    which is only correct for ``bin/generate_events``; every in-process caller
    should pass the value explicitly.

    ``jobs`` is the number of parallel compilation jobs for the source build
    (MG5's ``nb_core``); ``None`` lets the installer use every core.
    """
    if not madspace_is_installed():
        _run_installer("madspace", MADSPACE_DIR, interactive, jobs)

    if str(INSTALL_DIR) not in sys.path:
        sys.path.insert(0, str(INSTALL_DIR))


def ensure_pyamplicol(interactive=None, jobs=None) -> None:
    """Install pyamplicol if needed, then put it first on ``sys.path``.

    Same arguments as :func:`ensure_madspace`. The install directory must come
    before the MadGraph root: there ``pyamplicol/`` is the source directory,
    which would otherwise be imported as an empty namespace package.
    """
    if not pyamplicol_is_installed():
        _run_installer("pyamplicol", PYAMPLICOL_DIR, interactive, jobs)

    while str(PYAMPLICOL_INSTALL_DIR) in sys.path:
        sys.path.remove(str(PYAMPLICOL_INSTALL_DIR))
    sys.path.insert(0, str(PYAMPLICOL_INSTALL_DIR))


def check_pyamplicol_source_hash() -> dict:
    """Fail unless the installed pyamplicol was built from the sources in
    pyamplicol/, and return :func:`installed_pyamplicol`.

    Must run after :func:`ensure_pyamplicol`.
    """
    installed = installed_pyamplicol()
    result = subprocess.run(
        [sys.executable, str(PYAMPLICOL_DIR / "source_hash.py")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "cannot compute the hash of the pyamplicol sources:\n"
            + result.stdout + result.stderr
        )
    if installed["source_hash"] != result.stdout.strip():
        raise RuntimeError(
            "the installed pyamplicol was not built from the sources in "
            f"{PYAMPLICOL_DIR} (source hash mismatch): run `install pyamplicol` "
            "to rebuild it"
        )
    return installed


def check_pyamplicol_build(record_path) -> None:
    """Exit with an error unless the pyamplicol that would be imported is the
    one recorded in record_path by the output. Run by SubProcesses/check-pyamplicol
    before the [amplicol] libraries are compiled."""
    with open(record_path) as stream:
        recorded = json.load(stream)
    installed = installed_pyamplicol()
    if installed != {key: recorded.get(key) for key in installed}:
        sys.exit(
            f"The [amplicol] matrix elements of this output were generated with "
            f"pyamplicol {recorded.get('version')} (source hash "
            f"{recorded.get('source_hash')}), but the installed one is "
            f"{installed['version']} (source hash {installed['source_hash']}): "
            "output the process again, or reinstall the pyamplicol it was "
            "generated with (`install pyamplicol`)."
        )


def check_pyamplicol_license(pythonpath=None) -> None:
    """Fail unless Symbolica accepts pyamplicol's package-scoped library key.

    The key is registered on ``import pyamplicol`` and only licenses calls made
    from the ``pyamplicol`` package, so the check runs in a fresh process with
    the same ``PYTHONPATH`` as the generation worker and without a personal
    ``SYMBOLICA_LICENSE``. It also checks that the import resolves to the
    installed package, not to the source directory.
    """
    env = os.environ.copy()
    env.pop("SYMBOLICA_LICENSE", None)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath if pythonpath is not None else sys.path)
    code = (
        "import json, pyamplicol\n"
        "from pyamplicol.licensing import detect_symbolica_license\n"
        "print(json.dumps({'file': pyamplicol.__file__,\n"
        "    'licensed': detect_symbolica_license(suggest=False).licensed}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=MG_ROOT, env=env,
        capture_output=True, text=True,
    )
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise RuntimeError(
            "pyamplicol cannot be imported:\n" + result.stdout + result.stderr
        ) from None
    location = Path(info["file"]).resolve()
    if (location.is_relative_to(PYAMPLICOL_DIR.resolve())
            and not location.is_relative_to(PYAMPLICOL_INSTALL_DIR.resolve())):
        raise RuntimeError(
            f"pyamplicol resolves to its source directory ({PYAMPLICOL_DIR}), "
            "not to an installed package"
        )
    if not info["licensed"]:
        raise RuntimeError(
            "Symbolica does not accept pyamplicol's library license key; "
            "generation would run in Symbolica's restricted mode"
        )


def drop_install_path() -> None:
    """Take the madspace install directory back off ``sys.path``.

    It is not only the madspace package: the installer puts madspace's build
    dependencies in the same target, so the directory also carries ``yaml``,
    ``packaging``, ``pathspec``, ``scikit_build_core`` and ``pybind11_stubgen``.
    Prepending it therefore shadows any of those the caller already had.

    That was harmless while the launcher only ever ran in a throw-away
    ``bin/generate_events`` process. Now that it is imported into MG5's own
    long-lived session, the shadowing would outlast the run, so drop the entry
    once madspace itself is imported. The madspace package keeps working: its
    submodules resolve through its own ``__path__``, not through ``sys.path``.
    """
    while str(INSTALL_DIR) in sys.path:
        sys.path.remove(str(INSTALL_DIR))
