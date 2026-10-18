#!/usr/bin/env python3
"""Controlled finite-term pilots in the existing isolated FxFx work area.

Both schemes use the corrected fixed-order driver and the same initial grids,
seeds and integration settings. Original production events are only read.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--sample', choices=('w0j', 'w1j'), default='w1j')
args = parser.parse_args()
work = args.work.resolve()
sub = work / args.sample / 'SubProcesses'
template = ROOT / 'Template/NLO/SubProcesses'
for name in ('cluster.f', 'FKSParams.f90', 'driver_mintFO.f', 'makefile_fks_dir'):
    shutil.copyfile(template / name, sub / name)
card = (ROOT / 'Template/NLO/Cards/FKS_params.dat').read_text()
assert card.count('#FxFxPythia8FiniteTerms\n.false.') == 1
records = []


def run_process(p):
    logpath = work / (args.sample + '_' + p.name + '_finite_build.log')
    with logpath.open('w') as log:
        result = subprocess.run(['make', 'madevent_mintFO'], cwd=p,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(logpath.read_text()[-5000:])
    executable = p / 'madevent_finite_comparison'
    (p / 'madevent_mintFO').rename(executable)
    output = []
    for variant, enabled in [('finite_legacy', False), ('finite_pythia', True)]:
        d = p / variant
        d.mkdir(exist_ok=False)
        original = ROOT / args.sample / 'SubProcesses' / p.name / 'all_G1'
        for item in original.iterdir():
            if item.is_symlink() and item.name != 'FKS_params.dat':
                (d / item.name).symlink_to(item.readlink())
        for name in ('mint_grids', 'grid.MC_integer'):
            shutil.copyfile(original / name, d / name)
        # Match the earlier restored-scale pilot, including its point count.
        shutil.copyfile(p / 'restored/input_app.txt', d / 'input_app.txt')
        localcard = card.replace('#FxFxPythia8FiniteTerms\n.false.',
                                '#FxFxPythia8FiniteTerms\n' + ('.true.' if enabled else '.false.'))
        (d / 'FKS_params.dat').write_text(localcard)
        with (d / 'log.txt').open('w') as log:
            result = subprocess.run([str(executable)], cwd=d,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not (d / 'res.dat').exists():
            raise RuntimeError((d / 'log.txt').read_text()[-5000:])
        print(p.name, variant, (d / 'res.dat').read_text().splitlines()[0], flush=True)
        output.append(dict(process=p.name, variant=variant,
            directory=str(d), input=(d/'input_app.txt').read_text(),
            result=(d/'res.dat').read_text()))
    return output


processes = sorted(p for p in sub.glob('P0_*') if p.is_dir())
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for output in pool.map(run_process, processes):
        records.extend(output)
target = ROOT / 'fxfx_diagnostics' / (args.sample + '_finite_fo_runs.json')
target.write_text(json.dumps(records, indent=2)+'\n')
print(target, flush=True)
