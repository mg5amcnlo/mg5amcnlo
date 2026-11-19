#!/usr/bin/env python3
"""Regenerate all 22 affected MC@NLO samples, then update their plots.

The completed fixed-order integrations and POWHEG sample are reused.
Runs are sequential; --cores controls integration and parallel Pythia showers.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    ("run_central.py", "central_runs_fixed"),
    ("run_g_variations.py", "g_runs_fixed"),
    ("run_shower_scales.py", "scale_runs_fixed"),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=64,
                        help="cores for integration and parallel showers (default: 64)")
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="parent of the three fresh campaign directories")
    parser.add_argument("--dry-run", action="store_true",
                        help="list all run definitions without writing or running")
    parser.add_argument("--prepare-only", action="store_true",
                        help="export and configure all runs without integrating")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="omit the final comparison plots and manuscript tables")
    args = parser.parse_args(argv)
    if args.cores < 1:
        parser.error("--cores must be positive")
    output = args.output_root.resolve()
    if any(c.isspace() for c in str(output)):
        parser.error("use an output path without whitespace for MG5 commands")

    for script, directory in RUNNERS:
        command = [sys.executable, str(ROOT / "paper" / script),
                   "--cores", str(args.cores), "--output-dir", str(output / directory)]
        if args.dry_run:
            command.append("--dry-run")
        elif args.prepare_only:
            command.append("--prepare-only")
        print("\n" + shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    if args.dry_run or args.prepare_only or args.skip_analysis:
        return

    central = output / "central_runs_fixed"
    compare = [sys.executable, str(ROOT / "Analyses/compare_pp_tj_recoil.py"),
               "--central-dir", str(central), "--patched", "--two-patches", "--powheg"]
    commands = [
        [sys.executable, str(ROOT / "paper/analyse_matched_runs.py"),
         "--central-dir", str(central), "--g-dir", str(output / "g_runs_fixed"),
         "--scale-dir", str(output / "scale_runs_fixed"), "--require-complete"],
        compare,
        compare + ["--shape"],
        [sys.executable, str(ROOT / "paper/figures/make_plots.py"),
         "--central-dir", str(central)],
    ]
    for command in commands:
        print("\n" + shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\nAll 22 MC@NLO runs and comparison plots are complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, KeyboardInterrupt) as error:
        raise SystemExit(error.returncode if isinstance(error, subprocess.CalledProcessError) else 130)
