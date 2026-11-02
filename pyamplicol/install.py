#!/usr/bin/env python3
"""Install pyamplicol either using pre-compiled binaries or built from source.

The pre-compiled PyPI wheel (--bin) is only offered/defaulted-to in an
official MadGraph release tarball, where it is guaranteed to match the
bundled source; a plain git checkout always builds from source.

Interactive usage (no arguments):  python install.py
Non-interactive examples:
  python install.py --bin
  python install.py --source -j 8
  python install.py --source --clean          # rebuild from scratch

By default everything goes into install/ next to this script, so the active
Python environment is left untouched. The source build needs a Rust
toolchain: the version pinned in rust-toolchain.toml is taken from rustup
when available, otherwise a project-local rustup is installed into build/.

Rebuilds reuse the Cargo build tree in build/ and are therefore incremental.
--clean deletes build/ and install/ first; the saved settings survive it.

The prepared-model packs shipped in the package are taken from the archive
attached to the MadGraph release (the matching one, else the latest). When
they are stale for this source tree, or cannot be downloaded, they are
regenerated locally (one extra build).
"""

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
INSTALL_DIR = SCRIPT_DIR / "install"
BUILD_DIR = SCRIPT_DIR / "build"
SETTINGS_FILE = SCRIPT_DIR / "install_settings.json"
RUST_DIR = BUILD_DIR / "rust"
CACHE_DIR = BUILD_DIR / "cache"
PREPARED_MODELS_DIR = BUILD_DIR / "prepared-models"
RELEASE_TOOLS = SCRIPT_DIR / "tools" / "release"

PACKAGE_NAME = "pyamplicol"
RUSTUP_INIT_URL = "https://sh.rustup.rs"


# Interactive helpers (same behaviour as madspace/install.py: prompts go
# through MadGraph's cmd.ask when importable, else input()).

_NONINTERACTIVE = (not sys.stdin.isatty()) or any(
    a in ("-y", "--yes") for a in sys.argv[1:]
)


def _set_noninteractive(flag: bool) -> None:
    global _NONINTERACTIVE
    if flag:
        _NONINTERACTIVE = True


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_s = "y" if default else "n"
    answer = None
    try:
        from madgraph.interface.extended_cmd import Cmd

        answer = Cmd().ask(
            prompt, default_s, choices=["y", "n"], timeout=0, force=_NONINTERACTIVE
        )
    except Exception:
        if not _NONINTERACTIVE:
            with contextlib.suppress(EOFError):
                answer = input(f"{prompt} [{default_s}]: ").strip()
    answer = (answer or default_s).strip().lower()
    return answer in ("y", "yes") if answer in ("y", "yes", "n", "no") else default


# Command execution


def run(cmd: list, env: dict | None = None, cwd: Path = SCRIPT_DIR) -> None:
    display = " ".join(str(c) for c in cmd)
    print(f"\n$ {display}\n", flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def default_jobs() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# Versions and release information


def _pyamplicol_version() -> str:
    """The MadGraph version, which pyamplicol is released in lockstep with."""
    for line in (SCRIPT_DIR.parent / "VERSION").read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "version":
            return value.strip()
    raise RuntimeError("no 'version' line in the MadGraph VERSION file")


def _release_info() -> dict[str, str]:
    """Parse input/.release, written into MadGraph release tarballs."""
    marker = SCRIPT_DIR.parent / "input" / ".release"
    if not marker.is_file():
        return {}
    info = {}
    for line in marker.read_text().splitlines():
        name, _, value = line.partition("=")
        info[name.strip()] = value.strip()
    return info


def _release_wheel_available() -> bool:
    """Whether input/.release lists a pyamplicol wheel for this platform.

    The wheels are abi3 (cp311 and newer), so only the platform matters.
    """
    wheels = [
        name
        for name in _release_info().get("wheels", "").split(",")
        if name.startswith(f"{PACKAGE_NAME}-")
    ]
    system, machine = platform.system(), platform.machine()
    if system == "Linux" and machine in ("x86_64", "AMD64"):
        platform_tags = ("manylinux", "x86_64")
    elif system == "Darwin" and machine in ("arm64", "aarch64"):
        platform_tags = ("macosx", "arm64")
    elif system == "Darwin" and machine == "x86_64":
        platform_tags = ("macosx", "x86_64")
    else:
        return False
    return any(all(tag in name for tag in platform_tags) for name in wheels)


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def clean_install_dirs() -> None:
    for target in (BUILD_DIR, INSTALL_DIR):
        if target.is_dir():
            shutil.rmtree(target)
            print(f"Removed {target}")


# Rust toolchain


def _pinned_toolchain() -> str:
    with open(SCRIPT_DIR / "rust-toolchain.toml", "rb") as f:
        return tomllib.load(f)["toolchain"]["channel"]


def setup_rust(env: dict) -> dict:
    """Make the pinned toolchain available through rustup and return env.

    A rustup on PATH is used as is; otherwise a project-local one is installed
    into build/rust, leaving ~/.rustup and ~/.cargo alone.
    """
    toolchain = _pinned_toolchain()
    local_bin = RUST_DIR / "cargo" / "bin"
    if (local_bin / "rustup").is_file():
        rustup = local_bin / "rustup"
    elif shutil.which("rustup", path=env.get("PATH")):
        rustup = Path(shutil.which("rustup", path=env.get("PATH")))
    else:
        rustup = None

    # The build sets its own CARGO_HOME; rustup itself needs the local one.
    rust_env = env
    if rustup is None or rustup.parent == local_bin:
        env["RUSTUP_HOME"] = str(RUST_DIR / "rustup")
        env["PATH"] = os.pathsep.join([str(local_bin), env.get("PATH", "")])
        rust_env = dict(
            env, CARGO_HOME=str(RUST_DIR / "cargo"), RUSTUP_INIT_SKIP_PATH_CHECK="yes"
        )
    if rustup is None:
        print(f"\nNo rustup found; installing Rust {toolchain} into {RUST_DIR}")
        RUST_DIR.mkdir(parents=True, exist_ok=True)
        script = RUST_DIR / "rustup-init.sh"
        with urllib.request.urlopen(RUSTUP_INIT_URL, timeout=120) as response:
            script.write_bytes(response.read())
        run(
            [
                "sh",
                str(script),
                "-y",
                "--no-modify-path",
                "--profile",
                "minimal",
                "--default-toolchain",
                toolchain,
            ],
            env=rust_env,
        )
    else:
        run(
            [str(rustup), "toolchain", "install", toolchain, "--profile", "minimal"],
            env=rust_env,
        )
    return env


# Build


def install_build_deps(env: dict, system: bool) -> dict:
    with open(SCRIPT_DIR / "pyproject.toml", "rb") as f:
        requires = tomllib.load(f)["build-system"]["requires"]
    print("\nInstalling build dependencies...")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *requires]
    if not system:
        cmd.append(f"--target={INSTALL_DIR}")
    run(cmd, env=env)
    if not system:
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(INSTALL_DIR), env.get("PYTHONPATH")) if p
        )
        # maturin's console script, needed by the build backend
        env["PATH"] = os.pathsep.join([str(INSTALL_DIR / "bin"), env.get("PATH", "")])
    return env


def build_environment(jobs: int | None) -> dict:
    env = os.environ.copy()
    env["PYAMPLICOL_LOCAL_BUILD"] = "1"
    env["PYAMPLICOL_BUILD_MODE"] = "release"
    env["PYAMPLICOL_BUILD_CACHE_ROOT"] = str(CACHE_DIR)
    jobs = default_jobs() if jobs is None or jobs <= 0 else jobs
    env["PYAMPLICOL_BUILD_JOBS"] = str(jobs)
    print(f"Compiling with {jobs} parallel job(s) (-j to change).")
    return env


def _prepared_models_stale(env: dict) -> str | None:
    """Validate the unpacked prepared models against this source tree.

    Returns None when they are usable, else the reason they are not.
    """
    check = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(SCRIPT_DIR / 'build_backend')!r}); "
        "from prepared_models import stage_packaged_prepared_models; "
        f"stage_packaged_prepared_models(Path({str(SCRIPT_DIR)!r}), 'release')"
    )
    result = subprocess.run(
        [sys.executable, "-c", check], env=env, capture_output=True, text=True
    )
    if result.returncode == 0:
        return None
    lines = (result.stderr or result.stdout).strip().splitlines()
    return lines[-1] if lines else "validation failed"


def fetch_prepared_models(env: dict) -> bool:
    """Download and unpack the release archive; True if the packs are usable."""
    archive_tool = RELEASE_TOOLS / "prepared_models_archive.py"
    result = subprocess.run(
        [
            sys.executable,
            str(archive_tool),
            "fetch",
            "--version",
            _pyamplicol_version(),
        ],
        env=env,
    )
    if result.returncode != 0:
        print("Could not download the prepared-model archive.")
        return False
    reason = _prepared_models_stale(env)
    if reason is not None:
        print(f"The released prepared models do not match this source tree: {reason}")
        return False
    return True


def regenerate_prepared_models(env: dict) -> None:
    """Build a bootstrap wheel and produce the prepared models with it."""
    print("\nRegenerating the prepared models locally (one extra build)...")
    PREPARED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(RELEASE_TOOLS / "prepared_models_archive.py"),
            "regenerate",
            "--work",
            str(PREPARED_MODELS_DIR),
        ],
        env=env,
    )


def check_license(target: Path | None) -> None:
    """Fail unless pyamplicol's Symbolica library key is accepted."""
    env = os.environ.copy()
    env.pop("SYMBOLICA_LICENSE", None)
    if target is not None:
        env["PYTHONPATH"] = str(target)
    code = (
        "import json, pyamplicol; "
        "from pyamplicol.licensing import detect_symbolica_license; "
        "print(json.dumps({'file': pyamplicol.__file__, "
        "'licensed': detect_symbolica_license(suggest=False).licensed}))"
    )
    # Run outside the MadGraph tree, where pyamplicol/ is a plain directory.
    with tempfile.TemporaryDirectory() as cwd:
        result = subprocess.run(
            [sys.executable, "-I" if target is None else "-s", "-c", code],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        print(result.stdout + result.stderr)
        sys.exit("ERROR: the installed pyamplicol cannot be imported.")
    if target is not None and not Path(info["file"]).resolve().is_relative_to(target):
        sys.exit(f"ERROR: imported pyamplicol from {info['file']}, not from {target}.")
    if not info["licensed"]:
        sys.exit(
            "ERROR: Symbolica does not accept pyamplicol's library license key. "
            "Model generation would run in Symbolica's restricted mode."
        )
    print("Symbolica license: pyamplicol library key accepted.")


# Main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--bin",
        action="store_true",
        help="Install the pre-compiled package (default in release tarballs).",
    )
    mode.add_argument(
        "--source", action="store_true", help="Build and install from source."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Non-interactive, no prompts: reuse the previous install mode, else "
        "the default. Combine with --source/--bin to force the install mode.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel compilation jobs for the source build "
        f"(default: every available core, {default_jobs()} here). MG5's "
        "'nb_core' option is forwarded here by 'install pyamplicol'.",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        help="Install into the active Python environment instead of install/.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete build/ and install/ first, for a rebuild from scratch. "
        "The saved settings are kept.",
    )
    parser.add_argument(
        "--build-models",
        action="store_true",
        help="Regenerate the prepared models locally instead of downloading them.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _set_noninteractive(args.yes)
    saved = load_settings()
    if args.clean:
        clean_install_dirs()

    release_version = _release_info().get("version")
    is_release = release_version == _pyamplicol_version()
    bin_available = is_release and _release_wheel_available()

    if args.bin:
        if not bin_available:
            print(
                "WARNING: no pyamplicol wheel was published for this platform; "
                "--bin will likely fail."
            )
        from_source = False
    elif args.source:
        from_source = True
    elif args.yes:
        from_source = (
            saved.get("mode", "bin" if bin_available else "source") == "source"
        )
    else:
        print("Welcome to the pyAmpliCol installer")
        from_source = not (
            bin_available
            and ask_yes_no(
                "Install pre-compiled package? (recommended)",
                default=saved.get("mode", "bin") != "source",
            )
        )

    target = None if args.system else INSTALL_DIR
    if not from_source:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"{PACKAGE_NAME}=={_pyamplicol_version()}",
        ]
        if target is not None:
            cmd.append(f"--target={target}")
        run(cmd)
        save_settings({"mode": "bin"})
    else:
        env = build_environment(args.jobs)
        env = setup_rust(env)
        env = install_build_deps(env, args.system)
        if args.build_models or not fetch_prepared_models(env):
            regenerate_prepared_models(env)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-build-isolation",
            # A wheel for this host only; manylinux tags need a manylinux container.
            "--config-settings=pyamplicol.host-wheel=true",
            ".",
        ]
        if target is not None:
            cmd.append(f"--target={target}")
        run(cmd, env=env)
        save_settings({"mode": "source"})

    check_license(target)
    print(f"\nInstalled to: {target}" if target else "\nInstalled system-wide.")


if __name__ == "__main__":
    main()
