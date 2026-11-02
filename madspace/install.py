#!/usr/bin/env python3
"""Install madspace either using pre-compiled binaries or built from source.

The pre-compiled PyPI wheel (--bin) is only offered/defaulted-to in an
official MadGraph release tarball, where it is guaranteed to match the
bundled source; a plain git checkout always builds from source.

Interactive usage (no arguments):  python install.py
Non-interactive examples:
  python install.py --bin
  python install.py --source
  python install.py --source --cuda --cuda-arch "75;80;86"
  python install.py --source -j 8
  python install.py --source --cuda --hip --simd --debug
  python install.py --source --yes --cuda --cuda-arch 80
  python install.py --source --clean          # rebuild from scratch

Source-build options (--cuda, --hip, --openblas, --simd, --debug, ...,
--cuda-arch, --hip-arch) are resolved per option: a flag given on the command
line always wins; otherwise --yes reuses the value saved by the previous
source build, else the platform default. Without --yes, compile flags describe
the whole build: the options they leave out take the platform default.

Rebuilds reuse the CMake tree in build/ and are therefore incremental. --clean
deletes build/ and install/ first, for the rare cases where that tree is in the
way; the saved settings live outside both and survive it.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
INSTALL_DIR = SCRIPT_DIR / "install"
BUILD_DIR = SCRIPT_DIR / "build"
# Deliberately outside both directories, so --clean resets the build without
# also forgetting how the user wants madspace built. _LEGACY_SETTINGS_FILE is
# where it used to live; still read so an existing installation keeps its
# choices.
SETTINGS_FILE = SCRIPT_DIR / "install_settings.json"
_LEGACY_SETTINGS_FILE = BUILD_DIR / "install_settings.json"

PACKAGE_NAME = "madspace"

DEFAULT_CUDA_ARCH = "75"
DEFAULT_HIP_ARCH = "gfx900"
_DEFAULT_GPU_ARCH = {"cuda": DEFAULT_CUDA_ARCH, "hip": DEFAULT_HIP_ARCH}

# Platform-aware defaults for source-build options (mirrors CMakeLists.txt logic)
_IS_APPLE = platform.system() == "Darwin"
_PLATFORM_SOURCE_DEFAULTS: dict = {
    "cuda": False,
    "hip": False,
    "openblas": not _IS_APPLE,
    "simd": False,
    "build_type": "Release",
}


# Interactive helpers
#
# Prompts are routed through MadGraph's cmd.ask when it is importable, so that
# non-interactive / scripted runs behave consistently with the rest of MG5
# (piped stdin, the launcher's -f flag, input command files): the default is
# taken without blocking instead of hanging on input(). When madgraph is not
# importable (truly standalone use) we fall back to input(), still honouring
# the non-interactive flag.

_NONINTERACTIVE = (not sys.stdin.isatty()) or any(
    a in ("-y", "--yes") for a in sys.argv[1:]
)


def _set_noninteractive(flag: bool) -> None:
    """Force non-interactive mode (called from main once args are parsed)."""
    global _NONINTERACTIVE
    if flag:
        _NONINTERACTIVE = True


class _Prompter:
    """Ask a question through MG5's cmd.ask when available, else input()."""

    _cmd = False  # False = not tried yet, None = unavailable

    @classmethod
    def _get_cmd(cls):
        if cls._cmd is False:
            try:
                from madgraph.interface.extended_cmd import Cmd

                cls._cmd = Cmd()
            except Exception:
                cls._cmd = None
        return cls._cmd

    @classmethod
    def ask(cls, question: str, default: str, choices=None) -> str:
        default = str(default)
        cmd = cls._get_cmd()
        if cmd is not None:
            try:
                ans = cmd.ask(
                    question,
                    default,
                    choices=list(choices or []),
                    timeout=0,
                    force=_NONINTERACTIVE,
                )
                return default if ans is None else str(ans)
            except Exception:
                pass
        # standalone fallback (no madgraph on path)
        if _NONINTERACTIVE:
            return default
        try:
            raw = input(f"{question} [{default}]: ").strip()
        except EOFError:
            return default
        return raw if raw else default


def _ask(question, default, choices=None) -> str:
    return _Prompter.ask(question, default, choices)


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_s = "y" if default else "n"
    while True:
        raw = _ask(prompt, default_s, choices=["y", "n"]).strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        if _NONINTERACTIVE:
            return default
        print("  Please enter 'y' or 'n'.")


def ask_string(prompt: str, default: str) -> str:
    raw = _ask(prompt, default).strip()
    return raw if raw else default


def ask_compile_options(
    saved: dict | None = None,
    from_saved: bool = False,
    keys: tuple[str, ...] | None = None,
    show_expert_option: bool = False,
) -> dict[str, bool]:
    """Multi-select menu for compile options; returns {output_key: bool}.

    Each entry is (menu_key, label, output_key, invert).  When invert=True,
    selecting the item sets output_key=False; not selecting it sets it True.
    This lets the BLAS item be opt-in on Apple ("build OpenBLAS") and opt-out
    on Linux ("use system BLAS") while the default behavior of pressing Enter
    always matches the platform default.

    *keys*, if given, restricts the menu to entries whose output_key is in
    it (used to split the compile-options prompt into a basic and an expert
    stage). *show_expert_option* appends a pseudo entry ("_expert") used to
    let the user opt into seeing the expert stage.

    *from_saved* controls the Enter hint wording.
    """
    saved = saved or {}

    if _IS_APPLE:
        blas_entry = (
            "openblas",
            "Build OpenBLAS from source (system BLAS used by default)",
            "openblas",
            False,
        )
    else:
        blas_entry = (
            "system_blas",
            "Use system BLAS library (default: build OpenBLAS from source)",
            "openblas",
            True,
        )

    # (menu_key, label, output_key, invert)
    all_entries = [
        ("cuda", "Build CUDA backend", "cuda", False),
        ("hip", "Build HIP/ROCm backend", "hip", False),
        blas_entry,
        (
            "simd",
            "Build SIMD backend (experimental — not required to run SIMD matrix elements)",
            "simd",
            False,
        ),
    ]
    # CUDA and HIP are not available on Apple; hide them in interactive mode
    entries = [e for e in all_entries if not (_IS_APPLE and e[0] in ("cuda", "hip"))]
    if keys is not None:
        entries = [e for e in entries if e[2] in keys]
    if show_expert_option:
        entries.append(("expert", "Show expert/developer options", "_expert", False))

    # Derive the checkbox state for each menu item from the saved output values.
    # For normal items: checked = saved output value.
    # For inverted items: checked = NOT saved output value
    #   (e.g. if openblas=True was saved, "use system BLAS" should be unchecked).
    def _checked(output_key, invert):
        val = saved.get(output_key, False)
        return (not val) if invert else val

    prev = {mk: _checked(ok, inv) for mk, _, ok, inv in entries}
    has_any = any(prev.values())

    print()
    print("Compile options:")
    for i, (mk, label, _, _) in enumerate(entries, 1):
        marker = " [*]" if prev[mk] else ""
        print(f"  {i}. {label}{marker}")

    if from_saved:
        hint = "Enter to keep previous selection"
    elif has_any:
        hint = "Enter to keep defaults"
    else:
        hint = "Enter for none"

    def _defaults():
        # Convert checkbox state back to output values
        return {ok: (prev[mk] != inv) for mk, _, ok, inv in entries}

    while True:
        raw = _ask(
            f"Enter numbers separated by commas/spaces, or press Enter ({hint})", ""
        ).strip()
        if not raw:
            return _defaults()
        try:
            chosen = {int(x) for x in raw.replace(",", " ").split()}
        except ValueError:
            if _NONINTERACTIVE:
                return _defaults()
            print("  Invalid input — please enter numbers, e.g. 1,3 or 1 3")
            continue
        if not all(1 <= c <= len(entries) for c in chosen):
            if _NONINTERACTIVE:
                return _defaults()
            print(f"  Numbers must be between 1 and {len(entries)}.")
            continue
        return {
            ok: ((idx in chosen) != inv)
            for idx, (mk, _, ok, inv) in enumerate(entries, 1)
        }


def _saved_build_type(saved: dict) -> str:
    """Return the saved CMake build type, migrating the legacy 'debug' boolean if needed."""
    if "build_type" in saved:
        return saved["build_type"]
    return "RelWithDebInfo" if saved.get("debug", False) else "Release"


def ask_build_type(saved: dict) -> str:
    """Interactive single-select for CMake build type; returns one of the CMAKE_BUILD_TYPE strings."""
    options = [
        ("Release", "Optimized build, no debug symbols (default)"),
        ("RelWithDebInfo", "Optimized with debug symbols"),
        ("Debug", "Debug build, no optimization"),
    ]
    current = _saved_build_type(saved)
    print()
    print("Build type:")
    for i, (key, label) in enumerate(options, 1):
        marker = " [*]" if key == current else ""
        print(f"  {i}. {label}{marker}")
    hint = f"keep {current}" if current != "Release" else "Release"
    while True:
        raw = _ask(f"Choose (1-{len(options)}), or press Enter ({hint})", "").strip()
        if not raw:
            return current
        try:
            idx = int(raw)
        except ValueError:
            if _NONINTERACTIVE:
                return current
            print(f"  Please enter a number between 1 and {len(options)}.")
            continue
        if not 1 <= idx <= len(options):
            if _NONINTERACTIVE:
                return current
            print(f"  Please enter a number between 1 and {len(options)}.")
            continue
        return options[idx - 1][0]


# Source-build option resolution
#
# Per option, the first of these that applies wins:
#   1. a flag given on the command line;
#   2. under --yes, the value saved by the previous source build;
#   3. the platform default.
# Without --yes, step 2 is skipped for the compile options: the flags describe
# the whole build. When neither --yes nor any compile flag is given, the
# interactive menu decides instead.

COMPILE_OPTIONS = ("cuda", "hip", "openblas", "simd", "build_type")


def resolve_compile_options(args: argparse.Namespace, saved: dict) -> dict | None:
    """Return {option: value} for COMPILE_OPTIONS without prompting, or None
    when the interactive menu should ask for them."""
    if not args.yes and all(getattr(args, k) is None for k in COMPILE_OPTIONS):
        return None
    fallback = dict(_PLATFORM_SOURCE_DEFAULTS)
    if args.yes:
        fallback.update(
            (k, saved[k]) for k in ("cuda", "hip", "openblas", "simd") if k in saved
        )
        fallback["build_type"] = _saved_build_type(saved)
    return {
        k: fallback[k] if getattr(args, k) is None else getattr(args, k)
        for k in COMPILE_OPTIONS
    }


def resolve_gpu_arch(args: argparse.Namespace, saved: dict, backend: str) -> str:
    """Architectures for *backend* ("cuda" or "hip"): --cuda-arch/--hip-arch,
    else the saved value, else the default. Without --yes this is only the
    prompt default when the flag was not given."""
    cli_value = getattr(args, f"{backend}_arch")
    if cli_value is not None:
        return cli_value
    return saved.get(f"{backend}_arch", _DEFAULT_GPU_ARCH[backend])


# CMake discovery
#
# The source build (scikit-build-core) needs cmake >= 3.15 on PATH. When it is
# missing (or too old) we fall back to a cmake installed through MadGraph's
# HEPTools installer (`install cmake` in MG5), which lives next to this package
# under <MG5>/HEPTools. An explicit MADGRAPH_CMAKE / CMAKE env var wins.

CMAKE_MIN_VERSION = (3, 15)


def _cmake_version_ok(path, minimum=CMAKE_MIN_VERSION) -> bool:
    try:
        out = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return False
    m = re.search(r"(\d+)\.(\d+)", out)
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= minimum


def _heptools_dir_from_config() -> str | None:
    """Read heptools_install_dir from the MadGraph configuration files (same
    locations MadGraph itself uses -- see misc.user_config_file), so the value
    is available even when the installer is run directly rather than launched
    from MadGraph. This installation's own config wins over the per-user one."""
    candidates = [str(SCRIPT_DIR.parent / "input" / "mg7_configuration.txt")]
    xdg = os.environ.get("XDG_CONFIG_HOME")
    home = os.environ.get("HOME")
    if xdg:
        candidates.append(os.path.join(xdg, "mg7", "mg7_configuration.txt"))
    elif home:
        candidates.append(os.path.join(home, ".mg7", "mg7_configuration.txt"))
    for cfg in candidates:
        try:
            with open(cfg) as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if line.startswith("heptools_install_dir"):
                        val = line.partition("=")[2].strip()
                        if val:
                            if not os.path.isabs(val):
                                val = os.path.join(str(SCRIPT_DIR.parent), val)
                            return val
        except OSError:
            continue
    return None


def find_heptools_cmake() -> str | None:
    """find cmake installed via MadGraph's HEPTools installer. The HEPTools
    location is configurable in MG5 (heptools_install_dir); MG5 passes it
    down as MADGRAPH_HEPTOOLS_DIR. Fall back to the default <MG5>/HEPTools.
    """
    hep_dirs = []
    env_hep = os.environ.get("MADGRAPH_HEPTOOLS_DIR")
    if env_hep:
        hep_dirs.append(Path(env_hep))
    cfg_hep = _heptools_dir_from_config()
    if cfg_hep:
        hep_dirs.append(Path(cfg_hep))
    hep_dirs.append(SCRIPT_DIR.parent / "HEPTools")
    seen = set()
    for heptools in hep_dirs:
        for pattern in (
            "cmake/bin/cmake",
            "bin/cmake",
            "cmake",
            "cmake*/bin/cmake",
            "*/bin/cmake",
        ):
            for cand in sorted(heptools.glob(pattern)):
                cand = cand.resolve()
                if cand in seen:
                    continue
                seen.add(cand)
                if cand.is_file() and _cmake_version_ok(cand):
                    return str(cand)
    return None


def add_cmake_to_path(env: dict) -> dict:
    """Ensure a suitable cmake (and co-located tools such as ninja) is on the
    PATH of the build environment."""
    cmake = shutil.which("cmake")
    if cmake and _cmake_version_ok(cmake):
        print(f"Using cmake: {cmake}")
        return env

    cmake = find_heptools_cmake()
    if cmake:
        cmake_dir = os.path.dirname(os.path.abspath(cmake))
        parts = [cmake_dir]
        if existing := env.get("PATH"):
            parts.append(existing)
        env["PATH"] = os.pathsep.join(parts)
        print(f"Using cmake: {cmake}")
        return env

    print(
        "WARNING: no cmake >= %d.%d found. The source build will likely fail.\n"
        "  Install one with MG5 ('install cmake'), via your package manager,\n"
        "  or point to it with MADGRAPH_CMAKE=/path/to/cmake." % CMAKE_MIN_VERSION
    )
    return env


def default_jobs() -> int:
    """Number of compilation jobs to use when none was requested."""
    try:
        return len(os.sched_getaffinity(0))  # Linux: respects cgroup/taskset
    except AttributeError:
        return os.cpu_count() or 1


def set_build_parallelism(env: dict, jobs: int | None) -> dict:
    """Tell CMake how many compilation jobs it may run in parallel.

    scikit-build-core calls ``cmake --build`` without any ``-j``, so the job
    count comes entirely from the environment. With the Ninja generator that
    goes unnoticed (ninja parallelises on its own), but whenever ninja is
    missing scikit-build-core falls back to "Unix Makefiles", and make without
    ``-j`` compiles one file at a time -- which is why a gcc/make build took
    minutes while the ninja one did not.

    ``CMAKE_BUILD_PARALLEL_LEVEL`` is read by ``cmake --build`` itself, so it
    works for either generator, and being an environment variable it is also
    inherited by the nested OpenBLAS build.
    """
    if jobs is not None and jobs <= 0:
        jobs = None  # 0 / negative = "as many as there are cores"
    if jobs is None:
        if env.get("CMAKE_BUILD_PARALLEL_LEVEL"):
            return env  # an explicit setting in the caller's environment wins
        jobs = default_jobs()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    print(f"Compiling with {jobs} parallel job(s) (-j to change).")
    return env


# Command execution


def run(cmd: list, env: dict | None = None) -> None:
    display = " ".join(str(c) for c in cmd)
    print(f"\n$ {display}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def install_build_deps(system: bool = False) -> dict:
    """Install build-system dependencies and return an updated env.

    Reads the list from pyproject.toml's [build-system] requires, which now
    also includes pyyaml and pybind11-stubgen for the CMake-driven code
    generation (see CMakeLists.txt).

    When *system* is False (default) deps are installed into INSTALL_DIR and
    PYTHONPATH is set so the subsequent pip invocation picks them up.
    When *system* is True deps go to the default site-packages and PYTHONPATH
    is left unchanged.
    """
    pyproject_path = SCRIPT_DIR / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    requires = config.get("build-system", {}).get("requires", [])
    if requires:
        print()
        print("Installing build dependencies...")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *requires]
        if not system:
            cmd.append(f"--target={INSTALL_DIR}")
        run(cmd)

    env = os.environ.copy()
    if not system:
        pythonpath_parts = [str(INSTALL_DIR)]
        if existing := env.get("PYTHONPATH"):
            pythonpath_parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    # make sure the source build can find a recent-enough cmake
    env = add_cmake_to_path(env)
    return env


def _madspace_version() -> str:
    """The version a wheel built from this checkout would carry. Comes from the
    MadGraph VERSION file, the same source pyproject.toml's dynamic version
    reads, so madspace and MadGraph are always released in lockstep."""
    for line in (SCRIPT_DIR.parent / "VERSION").read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "version":
            return value.strip()
    raise RuntimeError("no 'version' line in the MadGraph VERSION file")


def _release_info() -> dict[str, str]:
    """Parse input/.release, the marker a MadGraph release tarball carries
    (written by bin/create_release.py), or {} in a plain git checkout."""
    marker = SCRIPT_DIR.parent / "input" / ".release"
    if not marker.is_file():
        return {}
    info = {}
    for line in marker.read_text().splitlines():
        name, _, value = line.partition("=")
        info[name.strip()] = value.strip()
    return info


def _release_version() -> str | None:
    """Version recorded in input/.release, or None in a plain git checkout.
    Used to decide whether the PyPI wheel -- built from the exact same
    release -- may be offered instead of a source build."""
    return _release_info().get("version")


def _release_wheel_available() -> bool:
    """Whether input/.release lists a wheel matching this exact interpreter
    and platform. The list is the actual filenames cibuildwheel produced for
    this release (see bin/create_release.py --wheels-dir), so this is a
    purely local check -- no PyPI query, no guessing at the CI build matrix."""
    wheels = _release_info().get("wheels", "")
    if not wheels:
        return False

    system, machine = platform.system(), platform.machine()
    if system == "Linux" and machine in ("x86_64", "AMD64"):
        platform_tags = ("manylinux", "x86_64")
    elif system == "Darwin" and machine in ("arm64", "aarch64"):
        platform_tags = ("macosx", "arm64")
    else:
        return False  # release wheels only ever target those two platforms

    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return any(
        name.startswith(f"{PACKAGE_NAME}-")
        and python_tag in name
        and all(tag in name for tag in platform_tags)
        for name in wheels.split(",")
    )


def load_settings() -> dict:
    for path in (SETTINGS_FILE, _LEGACY_SETTINGS_FILE):
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return {}


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


# Starting over
#
# A rebuild is normally incremental: the CMake tree in build/ is what makes it
# take seconds instead of recompiling madspace and its vendored OpenBLAS from
# scratch, so it is kept unless the user asks otherwise with --clean.


def clean_install_dirs() -> list[Path]:
    """Delete the CMake build tree and the install directory. Returns the
    directories that were actually removed."""
    removed = []
    for target in (BUILD_DIR, INSTALL_DIR):
        if target.is_dir():
            shutil.rmtree(target)
            print(f"Removed {target}")
            removed.append(target)
    return removed


# Main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Install mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--bin",
        action="store_true",
        default=False,
        help="Install pre-compiled package (default when interactive).",
    )
    mode.add_argument(
        "--source",
        action="store_true",
        default=False,
        help="Build and install from source.",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Non-interactive, no prompts: compile options given on the command "
        "line are used, the others reuse the previous source build's settings, "
        "else the platform defaults. Combine with --source/--bin to force the "
        "install mode.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel compilation jobs for the source build "
        f"(default: every available core, {default_jobs()} here). MG5's "
        "'nb_core' option is forwarded here by 'install madspace'.",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        default=False,
        help="Install system-wide instead of into the local install/ directory.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Delete the build/ and install/ directories first, for a rebuild "
        "from scratch (slow: madspace and its vendored OpenBLAS are recompiled). "
        "The saved settings are kept.",
    )

    # Compile option flags (each defaults to None = not specified via CLI)
    cuda_grp = parser.add_mutually_exclusive_group()
    cuda_grp.add_argument(
        "--cuda", dest="cuda", action="store_true", help="Enable CUDA backend."
    )
    cuda_grp.add_argument(
        "--no-cuda", dest="cuda", action="store_false", help="Disable CUDA backend."
    )

    hip_grp = parser.add_mutually_exclusive_group()
    hip_grp.add_argument(
        "--hip", dest="hip", action="store_true", help="Enable HIP/ROCm backend."
    )
    hip_grp.add_argument(
        "--no-hip", dest="hip", action="store_false", help="Disable HIP backend."
    )

    openblas_grp = parser.add_mutually_exclusive_group()
    openblas_grp.add_argument(
        "--openblas",
        dest="openblas",
        action="store_true",
        help="Build OpenBLAS from source (default on Linux/Windows).",
    )
    openblas_grp.add_argument(
        "--no-openblas",
        dest="openblas",
        action="store_false",
        help="Use system BLAS library (default on Apple).",
    )

    simd_grp = parser.add_mutually_exclusive_group()
    simd_grp.add_argument(
        "--simd",
        dest="simd",
        action="store_true",
        help="Enable SIMD backend (experimental).",
    )
    simd_grp.add_argument(
        "--no-simd", dest="simd", action="store_false", help="Disable SIMD backend."
    )

    docs_grp = parser.add_mutually_exclusive_group()
    docs_grp.add_argument(
        "--docs",
        dest="docs",
        action="store_true",
        help="Generate API docstrings (requires doxygen) and .pyi type stubs.",
    )
    docs_grp.add_argument(
        "--no-docs",
        dest="docs",
        action="store_false",
        help="Skip docstring and .pyi generation (default; no doxygen needed).",
    )

    debug_grp = parser.add_mutually_exclusive_group()
    debug_grp.add_argument(
        "--debug",
        dest="build_type",
        action="store_const",
        const="RelWithDebInfo",
        help="Build optimized with debug symbols (RelWithDebInfo).",
    )
    debug_grp.add_argument(
        "--full-debug",
        dest="build_type",
        action="store_const",
        const="Debug",
        help="Full debug build, no optimization (Debug).",
    )
    debug_grp.add_argument(
        "--no-debug",
        dest="build_type",
        action="store_const",
        const="Release",
        help="Optimized build without debug symbols (Release, default).",
    )

    # Architecture overrides
    parser.add_argument(
        "--cuda-arch",
        default=None,
        metavar="ARCHS",
        help="Semicolon-separated CUDA compute capabilities (default: the "
        f"previous source build's, else {DEFAULT_CUDA_ARCH}). Does not enable "
        'CUDA by itself (add --cuda). Example: "75;80;86".',
    )
    parser.add_argument(
        "--hip-arch",
        default=None,
        metavar="ARCHS",
        help="Semicolon-separated HIP GPU architectures (default: the previous "
        f"source build's, else {DEFAULT_HIP_ARCH}). Does not enable HIP by "
        'itself (add --hip). Example: "gfx900;gfx906;gfx1100".',
    )

    # None = not provided by user; overridden by set_defaults below
    parser.set_defaults(
        cuda=None, hip=None, openblas=None, simd=None, docs=None, build_type=None
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _set_noninteractive(args.yes)

    # The saved settings describe how the user wants madspace built, so they
    # are read whether or not install/ holds a previous installation: it never
    # does under --system, nor after a --clean whose rebuild failed. Read
    # before --clean on purpose, since the legacy location is inside build/.
    saved = load_settings()

    if args.clean:
        clean_install_dirs()

    # The PyPI wheel is only offered/defaulted-to in an actual release tarball
    # (input/.release, written by bin/create_release.py) that still matches
    # this checkout's madspace version -- otherwise pip would pull in an
    # unrelated madspace release -- and only when that release actually built
    # a wheel for this exact platform/Python (checked locally against the
    # wheel filenames recorded in the marker, no PyPI query).
    release_version = _release_version()
    is_release = release_version is not None and release_version == _madspace_version()
    bin_available = is_release and _release_wheel_available()

    # Determine install mode. An explicit --bin/--source always wins; --yes
    # alone reuses the saved mode, defaulting to bin only when available.
    if args.bin:
        if not bin_available:
            print(
                "WARNING: no madspace wheel was published for this platform/Python "
                "version; --bin will likely fail."
            )
        from_source = False
    elif args.source:
        from_source = True
    elif args.yes:
        from_source = (
            saved.get("mode", "bin" if bin_available else "source") == "source"
        )
    else:
        print("Welcome to the MadSpace interactive installer")
        print()
        if bin_available:
            default_is_bin = saved.get("mode", "bin") != "source"
            from_source = not ask_yes_no(
                "Install pre-compiled package? (recommended)", default=default_is_bin
            )
        else:
            from_source = True

    # PyPI installation, pinned to this checkout's madspace version so the
    # wheel matches the bundled source (see SOURCE_HASH check in mg7/launch.py).
    if not from_source:
        pip_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"{PACKAGE_NAME}=={_madspace_version()}",
        ]
        if not args.system:
            pip_cmd.append(f"--target={INSTALL_DIR}")
        run(pip_cmd)
        save_settings({"mode": "bin"})
        if args.system:
            print("\nInstalled system-wide.")
        else:
            print(f"\nInstalled to: {INSTALL_DIR}")
        return

    # Source build — compile options (see resolve_compile_options)
    options = resolve_compile_options(args, saved)
    if options is not None:
        enable_cuda = options["cuda"]
        enable_hip = options["hip"]
        enable_openblas = options["openblas"]
        enable_simd = options["simd"]
        build_type = options["build_type"]
    else:
        # Show saved source settings if available, else platform-appropriate defaults
        from_saved = saved.get("mode") == "source"
        menu_defaults = saved if from_saved else _PLATFORM_SOURCE_DEFAULTS

        # Basic prompt: just CUDA/HIP plus an opt-in to the expert options below
        basic = ask_compile_options(
            menu_defaults,
            from_saved=from_saved,
            keys=("cuda", "hip"),
            show_expert_option=True,
        )
        enable_cuda = basic.get("cuda", menu_defaults.get("cuda", False))
        enable_hip = basic.get("hip", menu_defaults.get("hip", False))

        if basic.get("_expert", False):
            expert = ask_compile_options(
                menu_defaults, from_saved=from_saved, keys=("openblas", "simd")
            )
            enable_openblas = expert["openblas"]
            enable_simd = expert["simd"]
            build_type = ask_build_type(menu_defaults)
        else:
            enable_openblas = menu_defaults.get(
                "openblas", _PLATFORM_SOURCE_DEFAULTS["openblas"]
            )
            enable_simd = menu_defaults.get("simd", False)
            build_type = _saved_build_type(menu_defaults)

    # Docs/.pyi generation: explicit flag wins, else reuse saved value under
    # --yes, else off. Not part of the interactive menu (niche / CI use).
    if args.docs is not None:
        enable_docs = args.docs
    elif args.yes:
        enable_docs = saved.get("docs", False)
    else:
        enable_docs = False

    # Compute capabilities (see resolve_gpu_arch); prompt only for an enabled
    # backend whose architectures were neither given nor fixed by --yes
    cuda_arch = resolve_gpu_arch(args, saved, "cuda")
    hip_arch = resolve_gpu_arch(args, saved, "hip")

    if enable_cuda and args.cuda_arch is None and not args.yes:
        print()
        cuda_arch = ask_string(
            "CUDA compute capabilities (semicolon-separated, e.g. 75;80;86)",
            default=cuda_arch,
        )

    if enable_hip and args.hip_arch is None and not args.yes:
        print()
        hip_arch = ask_string(
            "HIP GPU architectures (semicolon-separated, e.g. gfx900;gfx906;gfx1100)",
            default=hip_arch,
        )

    for backend, enabled in (("cuda", enable_cuda), ("hip", enable_hip)):
        if not enabled and getattr(args, f"{backend}_arch") is not None:
            print(
                f"WARNING: --{backend}-arch has no effect, the {backend.upper()} "
                f"backend is not enabled (add --{backend})."
            )

    # Assemble pip command
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--upgrade",
        "-Cbuild-dir=build",
        ".",
    ]
    if not args.system:
        cmd.append(f"--target={INSTALL_DIR}")

    if enable_cuda:
        cmd += [
            "-Ccmake.define.ENABLE_CUDA=ON",
            f"-Ccmake.define.CMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        ]
    if enable_hip:
        cmd += [
            "-Ccmake.define.ENABLE_HIP=ON",
            f"-Ccmake.define.CMAKE_HIP_ARCHITECTURES={hip_arch}",
        ]
    cmd.append(f"-Ccmake.define.ENABLE_OPENBLAS={'ON' if enable_openblas else 'OFF'}")
    if enable_simd:
        cmd.append("-Ccmake.define.ENABLE_SIMD=ON")
    if enable_docs:
        cmd.append("-Ccmake.define.ENABLE_DOCS=ON")
    cmd.append(f"-Ccmake.build-type={build_type}")

    env = install_build_deps(system=args.system)
    env = set_build_parallelism(env, args.jobs)
    run(cmd, env=env)
    save_settings(
        {
            "mode": "source",
            "cuda": enable_cuda,
            "cuda_arch": cuda_arch,
            "hip": enable_hip,
            "hip_arch": hip_arch,
            "openblas": enable_openblas,
            "simd": enable_simd,
            "docs": enable_docs,
            "build_type": build_type,
        }
    )
    if args.system:
        print("\nInstalled system-wide.")
    else:
        print(f"\nInstalled to: {INSTALL_DIR}")


if __name__ == "__main__":
    main()
