#!/usr/bin/env python3
"""Run the madspace unit tests once for every SIMD mode supported on this machine.

The mode is selected through the MADSPACE_SIMD_MODE environment variable, which the
CPU backend reads when it is first loaded, so every mode runs in its own pytest
process. Arguments that are not recognized here are passed on to pytest; without
any, the tests next to this script are run.

    python run_simd_modes.py                    # all supported modes
    python run_simd_modes.py --modes scalar     # only the given modes
    python run_simd_modes.py -x -k vegas        # extra pytest arguments
"""

import argparse
import os
import subprocess
import sys

import madspace as ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--modes",
        help="comma-separated list of SIMD modes (default: all supported ones)",
    )
    args, pytest_args = parser.parse_known_args()

    supported = ms.supported_simd_modes()
    modes = args.modes.split(",") if args.modes else supported
    if not pytest_args:
        pytest_args = [os.path.dirname(os.path.realpath(__file__))]
    print(f"supported SIMD modes: {', '.join(supported)}", flush=True)

    results = {}
    for mode in modes:
        print(f"\n===== SIMD mode: {mode} =====", flush=True)
        env = dict(os.environ, MADSPACE_SIMD_MODE=mode)
        results[mode] = subprocess.call(
            [sys.executable, "-m", "pytest", *pytest_args], env=env
        )

    print("\n===== summary =====")
    for mode, code in results.items():
        print(f"{mode:>10}: {'passed' if code == 0 else f'FAILED (exit code {code})'}")
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
