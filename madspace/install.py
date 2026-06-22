#!/usr/bin/env python3
"""Install madspace either using pre-compiled binaries or built from source.

Interactive usage (no arguments):  python install.py
Non-interactive examples:
  python install.py --bin
  python install.py --source
  python install.py --source --cuda --cuda-arch "75;80;86"
  python install.py --source --cuda --hip --simd --debug
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
INSTALL_DIR = SCRIPT_DIR / "install"
SETTINGS_FILE = SCRIPT_DIR / "build" / "install_settings.json"

PACKAGE_NAME = "madspace"

DEFAULT_CUDA_ARCH = "75"
DEFAULT_HIP_ARCH = "gfx900"

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


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


def ask_string(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [default: {default}]: ").strip()
    return raw if raw else default


def ask_compile_options(
    saved: dict | None = None, from_saved: bool = False
) -> dict[str, bool]:
    """Multi-select menu for compile options; returns {output_key: bool}.

    Each entry is (menu_key, label, output_key, invert).  When invert=True,
    selecting the item sets output_key=False; not selecting it sets it True.
    This lets the BLAS item be opt-in on Apple ("build OpenBLAS") and opt-out
    on Linux ("use system BLAS") while the default behavior of pressing Enter
    always matches the platform default.

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

    while True:
        raw = input(
            f"Enter numbers separated by commas/spaces, or press {hint}: "
        ).strip()
        if not raw:
            # Convert checkbox state back to output values
            return {ok: (prev[mk] != inv) for mk, _, ok, inv in entries}
        try:
            chosen = {int(x) for x in raw.replace(",", " ").split()}
        except ValueError:
            print("  Invalid input — please enter numbers, e.g. 1,3 or 1 3")
            continue
        if not all(1 <= c <= len(entries) for c in chosen):
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
    hint = f"Enter to keep {current}" if current != "Release" else "Enter for Release"
    while True:
        raw = input(f"Choose (1-{len(options)}), or press {hint}: ").strip()
        if not raw:
            return current
        try:
            idx = int(raw)
        except ValueError:
            print(f"  Please enter a number between 1 and {len(options)}.")
            continue
        if not 1 <= idx <= len(options):
            print(f"  Please enter a number between 1 and {len(options)}.")
            continue
        return options[idx - 1][0]


# Command execution


def run(cmd: list, env: dict | None = None) -> None:
    display = " ".join(str(c) for c in cmd)
    print(f"\n$ {display}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def install_build_deps(system: bool = False) -> dict:
    """Install build-system dependencies and return an updated env.

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
    return env


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


# Main


def main() -> None:
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
        help="Re-install non-interactively using saved settings; falls back to built-in defaults.",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        default=False,
        help="Install system-wide instead of into the local install/ directory.",
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
        help=f"Semicolon-separated CUDA compute capabilities (default: {DEFAULT_CUDA_ARCH}). "
        'Example: "75;80;86".',
    )
    parser.add_argument(
        "--hip-arch",
        default=None,
        metavar="ARCHS",
        help=f"Semicolon-separated HIP GPU architectures (default: {DEFAULT_HIP_ARCH}). "
        'Example: "gfx900;gfx906;gfx1100".',
    )

    # None = not provided by user; overridden by set_defaults below
    parser.set_defaults(cuda=None, hip=None, openblas=None, simd=None, build_type=None)
    args = parser.parse_args()

    # Load saved settings when a previous installation is present
    saved = load_settings() if (INSTALL_DIR / "madspace").is_dir() else {}

    # Determine install mode
    if args.yes:
        from_source = saved.get("mode", "bin") == "source"
    elif args.bin:
        from_source = False
    elif args.source:
        from_source = True
    else:
        print("Welcome to the MadSpace interactive installer")
        print()
        default_is_bin = saved.get("mode", "bin") != "source"
        from_source = not ask_yes_no(
            "Install pre-compiled package? (recommended)", default=default_is_bin
        )

    # PyPI installation
    if not from_source:
        pip_cmd = [sys.executable, "-m", "pip", "install", PACKAGE_NAME]
        if not args.system:
            pip_cmd.append(f"--target={INSTALL_DIR}")
        run(pip_cmd)
        save_settings({"mode": "bin"})
        if args.system:
            print("\nInstalled system-wide.")
        else:
            print(f"\nInstalled to: {INSTALL_DIR}")
        return

    # Source build — compile options
    compile_flags_given = any(
        getattr(args, attr) is not None
        for attr in ("cuda", "hip", "openblas", "simd", "build_type")
    )

    if args.yes:
        enable_cuda = saved.get("cuda", _PLATFORM_SOURCE_DEFAULTS["cuda"])
        enable_hip = saved.get("hip", _PLATFORM_SOURCE_DEFAULTS["hip"])
        enable_openblas = saved.get("openblas", _PLATFORM_SOURCE_DEFAULTS["openblas"])
        enable_simd = saved.get("simd", _PLATFORM_SOURCE_DEFAULTS["simd"])
        build_type = _saved_build_type(saved)
    elif compile_flags_given:
        enable_cuda = bool(args.cuda)
        enable_hip = bool(args.hip)
        enable_openblas = (
            bool(args.openblas)
            if args.openblas is not None
            else _PLATFORM_SOURCE_DEFAULTS["openblas"]
        )
        enable_simd = bool(args.simd)
        build_type = args.build_type or "Release"
    else:
        # Show saved source settings if available, else platform-appropriate defaults
        from_saved = saved.get("mode") == "source"
        menu_defaults = saved if from_saved else _PLATFORM_SOURCE_DEFAULTS
        opts = ask_compile_options(menu_defaults, from_saved=from_saved)
        enable_cuda = opts.get("cuda", menu_defaults.get("cuda", False))
        enable_hip = opts.get("hip", menu_defaults.get("hip", False))
        enable_openblas = opts["openblas"]
        enable_simd = opts["simd"]
        build_type = ask_build_type(menu_defaults)

    # Compute capability prompts
    cuda_arch = saved.get("cuda_arch", DEFAULT_CUDA_ARCH)
    hip_arch = saved.get("hip_arch", DEFAULT_HIP_ARCH)

    if enable_cuda:
        if args.yes:
            cuda_arch = saved.get("cuda_arch", DEFAULT_CUDA_ARCH)
        elif args.cuda_arch is not None:
            cuda_arch = args.cuda_arch
        else:
            print()
            cuda_arch = ask_string(
                "CUDA compute capabilities (semicolon-separated, e.g. 75;80;86)",
                default=cuda_arch,
            )

    if enable_hip:
        if args.yes:
            hip_arch = saved.get("hip_arch", DEFAULT_HIP_ARCH)
        elif args.hip_arch is not None:
            hip_arch = args.hip_arch
        else:
            print()
            hip_arch = ask_string(
                "HIP GPU architectures (semicolon-separated, e.g. gfx900;gfx906;gfx1100)",
                default=hip_arch,
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
    cmd.append(f"-Ccmake.build-type={build_type}")

    env = install_build_deps(system=args.system)
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
            "build_type": build_type,
        }
    )
    if args.system:
        print("\nInstalled system-wide.")
    else:
        print(f"\nInstalled to: {INSTALL_DIR}")


if __name__ == "__main__":
    main()
