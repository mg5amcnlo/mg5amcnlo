#!/usr/bin/env python3
"""Retain compact benchmark evidence without copying large LHE event files."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.run_dir.resolve()
    destination = args.output.resolve()

    def copy(path, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    for name in ('run_card.dat', 'param_card.dat', 'FKS_params.dat'):
        copy(source / 'Cards' / name, Path('ampli/Cards') / name)
    for pattern in ('P*/G*/log_MINT*.txt', 'P*/G*/res_*.dat',
                    'ampli_production.json', 'nevents_unweighted'):
        for path in sorted((source / 'SubProcesses').glob(pattern)):
            if path.is_file() and not path.is_symlink():
                copy(path, Path('ampli') / path.relative_to(source))
    for pattern in ('*/summary.txt', '*/res_*.txt', '*/ampli_production.json'):
        for path in sorted((source / 'Events').glob(pattern)):
            copy(path, Path('ampli/Events') / path.name)
    for parent in (source, source.parent):
        for pattern in ('*started.json', '*finished.json', '*.cmd',
                        'run_benchmark.py', 'ampli.log'):
            for path in sorted(parent.glob(pattern)):
                copy(path, Path('ampli') / path.name)
    print(destination)


if __name__ == '__main__':
    main()
