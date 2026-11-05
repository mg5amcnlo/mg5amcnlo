#!/usr/bin/env python3
"""Build and install MadtRex (and the bundled Rex fork it needs) from source.

MadtRex sits on top of madspace, which has to be installed first (by default
from ../madspace/install, i.e. `python madspace/install.py`); madspace itself
does not depend on MadtRex.

Usage:
  python install.py                 # interactive
  python install.py --yes           # no prompts
  python install.py --yes -j 8
  python install.py --clean         # rebuild from scratch
  python install.py --system        # into the current Python environment

Steps:
  1. check that madspace is importable (from ../madspace/install, or from the
     current environment with --system);
  2. pip install ./Rex (the `pyrex` package) into install/;
  3. pip install this directory (the `madtrex` package) into install/,
     built against that madspace.

Rebuilds reuse the CMake tree in build/ and are therefore incremental. --clean
deletes build/ and install/ first.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
INSTALL_DIR = SCRIPT_DIR / "install"
BUILD_DIR = SCRIPT_DIR / "build"
REX_DIR = SCRIPT_DIR / "Rex"
MADSPACE_SRC_DIR = SCRIPT_DIR.parent / "madspace"
MADSPACE_INSTALL_DIR = MADSPACE_SRC_DIR / "install"


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            answer = input(prompt + suffix).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def default_jobs() -> int:
    """Number of compilation jobs to use when none was requested."""
    try:
        return len(os.sched_getaffinity(0))  # Linux: respects cgroup/taskset
    except AttributeError:
        return os.cpu_count() or 1


def set_build_parallelism(env: dict, jobs: int | None) -> dict:
    """See madspace/install.py: scikit-build-core runs `cmake --build` without
    -j, so the job count has to come from CMAKE_BUILD_PARALLEL_LEVEL."""
    if jobs is not None and jobs <= 0:
        jobs = None
    if jobs is None:
        if env.get("CMAKE_BUILD_PARALLEL_LEVEL"):
            return env
        jobs = default_jobs()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    print(f"Compiling with {jobs} parallel job(s) (-j to change).")
    return env


def run(cmd: list, env: dict | None = None, cwd: Path = SCRIPT_DIR) -> None:
    display = " ".join(str(c) for c in cmd)
    print(f"\n$ {display}\n")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def with_pythonpath(env: dict, *dirs: Path) -> dict:
    parts = [str(d) for d in dirs]
    if existing := env.get("PYTHONPATH"):
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def find_madspace(env: dict) -> Path | None:
    """The madspace package directory the build would use, or None if
    madspace cannot be imported with this environment."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import madspace, os; "
            "print(os.path.dirname(os.path.abspath(madspace.__file__)))",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip().splitlines()[-1])


def madspace_problem(madspace_dir: Path, env: dict) -> str | None:
    """Why the madspace in madspace_dir is too old to build MadtRex against
    (it predates the API MadtRex uses: umami.h installed with the headers,
    MatrixElementApi.set_parameter/supported_inputs), or None."""
    if not (madspace_dir / "include").is_dir():
        return ("it has no headers (include/), as a precompiled copy of madspace "
                "(e.g. in a gridpack) does")
    if not (madspace_dir / "include" / "madspace" / "umami.h").is_file():
        return "it does not install include/madspace/umami.h"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import madspace; api = madspace.MatrixElementApi; "
            "missing = [n for n in ('supports_set_parameter', 'set_parameter', "
            "'supported_inputs') if not hasattr(api, n)]; print(' '.join(missing))",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "importing it failed:\n" + result.stderr.strip()
    missing = result.stdout.strip().splitlines()[-1:] if result.stdout.strip() else []
    if missing and missing[0]:
        return "its MatrixElementApi has no " + ", ".join(missing[0].split())
    return None


def install_build_deps(env: dict, system: bool) -> None:
    """Install the [build-system] requires of pyproject.toml, needed because
    madtrex is built with --no-build-isolation (so that CMake sees the same
    Python environment, and therefore the same madspace, as the install)."""
    with open(SCRIPT_DIR / "pyproject.toml", "rb") as f:
        requires = tomllib.load(f).get("build-system", {}).get("requires", [])
    if not requires:
        return
    print("\nInstalling build dependencies...")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *requires]
    if not system:
        cmd.append(f"--target={INSTALL_DIR}")
    run(cmd, env=env)


def clean_install_dirs() -> None:
    for target in (BUILD_DIR, INSTALL_DIR, REX_DIR / "build"):
        if target.is_dir():
            shutil.rmtree(target)
            print(f"Removed {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Non-interactive, no prompts.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel compilation jobs "
        f"(default: every available core, {default_jobs()} here).",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        default=False,
        help="Install into the current Python environment instead of the local "
        "install/ directory; madspace must then be importable from it too.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Delete the build/ and install/ directories first.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.clean:
        clean_install_dirs()

    env = os.environ.copy()
    if not args.system:
        env = with_pythonpath(env, INSTALL_DIR, MADSPACE_INSTALL_DIR)

    # 1. madspace
    madspace_dir = find_madspace(env)
    if madspace_dir is None:
        where = (
            "the current Python environment"
            if args.system
            else str(MADSPACE_INSTALL_DIR)
        )
        madspace_cmd = [sys.executable, str(MADSPACE_SRC_DIR / "install.py")]
        if args.system:
            madspace_cmd.append("--system")
        print(f"madspace is not installed in {where}; MadtRex needs it.")
        if args.yes or not ask_yes_no("Run madspace/install.py now?", default=True):
            print(
                "Install madspace first:\n    "
                + " ".join(madspace_cmd)
                + "\nthen re-run this script."
            )
            sys.exit(1)
        run(madspace_cmd, cwd=MADSPACE_SRC_DIR)
        madspace_dir = find_madspace(env)
        if madspace_dir is None:
            print("madspace is still not importable after installing it; giving up.")
            sys.exit(1)
    if problem := madspace_problem(madspace_dir, env):
        print(
            f"The madspace in {madspace_dir} is too old for MadtRex: {problem}.\n"
            "Reinstall madspace from this checkout's sources first:\n    "
            f"{sys.executable} {MADSPACE_SRC_DIR / 'install.py'} --source --yes\n"
            "(or 'install madspace --source' in MadGraph), then re-run this script."
        )
        sys.exit(1)
    print(f"Using madspace from {madspace_dir}")

    env = set_build_parallelism(env, args.jobs)

    # 2. the bundled Rex fork (pyrex)
    rex_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-deps",
        "-Cbuild-dir=build",
        ".",
    ]
    if not args.system:
        rex_cmd.append(f"--target={INSTALL_DIR}")
    run(rex_cmd, env=env, cwd=REX_DIR)

    # 3. madtrex itself. --no-deps: madspace and pyrex were just checked /
    # installed above; letting pip resolve them would pull unrelated
    # releases from PyPI instead of the local ones.
    install_build_deps(env, args.system)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "--upgrade",
        "-Cbuild-dir=build",
        f"-Ccmake.define.MADSPACE_DIR={madspace_dir}",
        ".",
    ]
    if not args.system:
        cmd.append(f"--target={INSTALL_DIR}")
    run(cmd, env=env)

    if args.system:
        print("\nInstalled system-wide.")
    else:
        print(f"\nInstalled to: {INSTALL_DIR}")
        print(
            "To use it outside MadGraph, put both install directories on "
            f"PYTHONPATH:\n    {INSTALL_DIR}{os.pathsep}{MADSPACE_INSTALL_DIR}"
        )


if __name__ == "__main__":
    main()
