#!/usr/bin/env python3
"""Retain compact benchmark evidence without copying large LHE/pool files."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=('mint', 'ampli'), required=True)
    parser.add_argument('--run-name', default='benchmark_10k')
    parser.add_argument('--include-incomplete', action='store_true', help='Also retain active worker log.txt files for an interrupted/failed run.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.run_dir.resolve()
    destination = args.output.resolve() / args.backend

    def copy(path, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    for name in ('run_card.dat', 'param_card.dat', 'FKS_params.dat',
                 'proc_card_mg5.dat', 'proc_card.dat', 'amcatnlo_configuration.txt'):
        path = source / 'Cards' / name
        if path.is_file():
            copy(path, path.relative_to(source))
    for pattern in ('P*/G*/log_MINT*.txt', 'P*/G*/res_*.dat',
                    'P*/G*/ampli_job.dat', 'ampli_production.json',
                    'nevents_unweighted', 'randinit', 'subproc.mg', 'procdef_mg5.dat'):
        for path in sorted((source / 'SubProcesses').glob(pattern)):
            if path.is_file() and not path.is_symlink():
                copy(path, path.relative_to(source))
    if args.include_incomplete:
        for pattern in ('P*/G*/log.txt', 'P*/G*/input_app.txt',
                        'P*/fks_info.inc', 'P*/fks_j_from_i.inc', 'P*/symfact.dat'):
            for path in sorted((source / 'SubProcesses').glob(pattern)):
                if path.is_file() and not path.is_symlink():
                    copy(path, path.relative_to(source))
    selected_file = source / 'SubProcesses' / 'subproc.mg'
    if selected_file.is_file():
        for name in selected_file.read_text().split():
            selected = source / 'SubProcesses' / name
            for pattern in ('fks_info.inc', 'born_support.json', 'born_support.f',
                            'parton_lum_*.f', 'leshouche_info.dat'):
                for path in sorted(selected.glob(pattern)):
                    if path.is_file():
                        copy(path, path.relative_to(source))
    event_dir = source / 'Events' / args.run_name
    for pattern in ('summary.txt', 'res_*.txt', 'ampli_production.json', '*banner.txt'):
        for path in sorted(event_dir.glob(pattern)):
            copy(path, path.relative_to(source))
    for parent in (source, source.parent):
        for pattern in (args.backend + '*started.json', args.backend + '*finished.json',
                        args.backend + '*.cmd', args.backend + '*.log',
                        'run_benchmark.py', 'generate*.cmd', 'generate*.log',
                        'generation*.json', 'selection*.json', 'source*.json',
                        'selected*.json', 'subproc_complete.mg', 'export*.json',
                        'verification.json', 'verify_component.py', 'accuracy_conversion.json'):
            for path in sorted(parent.glob(pattern)):
                if path.is_file():
                    copy(path, Path('launch') / path.name)
    print(destination)


if __name__ == '__main__':
    main()
