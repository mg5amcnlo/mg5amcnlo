#!/usr/bin/env python3
"""Run the 12 inclusive fixed-order comparisons in proposed_runs.csv.

Each integration gets a fresh export under fo_runs/; existing production
directories are only read. Use --dry-run to list the runs without starting.
"""

import argparse
import csv
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def replace(path, pattern, value):
    text, count = re.subn(pattern, lambda m: m[1] + str(value) + m[2],
                          path.read_text(), flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one matching setting in {path}: {pattern}")
    path.write_text(text)


def prepare(row, directory, cores, log):
    source = ROOT / row["source_process"]
    commands = (source / "Cards/proc_card_mg5.dat").read_text()
    commands, count = re.subn(r"^output\b.*$", lambda m: f"output {directory}",
                              commands, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("Expected one output command in the process card")
    command_file = directory.with_suffix(".mg5")
    command_file.write_text(commands + "\nquit\n")
    subprocess.run([sys.executable, str(ROOT / "bin/mg5_aMC"), str(command_file)],
                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    for name in ("param_card.dat", "run_card.dat", "FKS_params.dat",
                 "FO_analyse_card.dat", "amcatnlo_configuration.txt"):
        shutil.copy2(source / "Cards" / name, directory / "Cards" / name)
    for name, value in (("iseed", row["iseed"]), ("req_acc_FO", row["req_acc_FO"]),
                        ("reweight_scale", "False"), ("reweight_PDF", "False"),
                        ("store_rwgt_info", "False")):
        replace(directory / "Cards/run_card.dat",
                rf"^([ \t]*)[^!\n=]+?([ \t]*=[ \t]*{name}\b.*)$", value)
    for name in ("xicut", "deltaI", "deltaO", "xiScut", "deltaS", "xiBSVcut"):
        replace(directory / "SubProcesses/fks_powers.inc",
                rf"^(\s*parameter\s*\(\s*{name}\s*=)[^)]*(\).*)$",
                row[name] + "d0")
    replace(directory / "Cards/FO_analyse_card.dat",
            r"^(FO_ANALYSE\s*=\s*)\S+(.*)$", "analysis_HwU_template.o")
    with (directory / "Cards/amcatnlo_configuration.txt").open("a") as stream:
        stream.write(f"\nrun_mode = 2\nnb_core = {cores}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=4, help="cores per run (default: 4)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fo_runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cores < 1:
        parser.error("--cores must be positive")
    output = args.output_dir.resolve()
    if any(c.isspace() for c in str(output)):
        parser.error("use an output path without whitespace for MG5 commands")
    with (ROOT / "paper/proposed_runs.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["group"] == "F"]
    if len(rows) != 12:
        parser.error("expected exactly 12 fixed-order rows in proposed_runs.csv")

    for row in rows:
        name = row["run_id"]
        directory = output / name
        summary = directory / "Events" / name / "summary.txt"
        print(f"{name}: seed={row['iseed']}, xiScut={row['xiScut']}, "
              f"deltaS={row['deltaS']}, accuracy={row['req_acc_FO']}", flush=True)
        if args.dry_run:
            continue
        if summary.is_file():
            print(f"  Already finished: {summary}", flush=True)
            continue
        if directory.exists():
            sys.exit(f"Incomplete export exists: {directory}. Inspect it before retrying.")
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / f"{name}.log"
        print(f"  Running; log: {log_path}", flush=True)
        with log_path.open("w") as log:
            prepare(row, directory, args.cores, log)
            subprocess.run([str(directory / "bin/calculate_xsect"), "NLO", "-f",
                            f"--name={name}"], cwd=directory,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        # MG5 can report a command error without returning a nonzero exit code.
        if not summary.is_file():
            sys.exit(f"No completed integration summary; inspect {log_path}")
        print(f"  Finished: {summary}", flush=True)


if __name__ == "__main__":
    main()
