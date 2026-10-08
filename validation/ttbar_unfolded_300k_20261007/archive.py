#!/usr/bin/env python3
"""Archive numerical evidence without copying matrix-element objects or candidate LHE spools."""
from pathlib import Path
import argparse,json,shutil
parser=argparse.ArgumentParser()
parser.add_argument('--work',type=Path,required=True)
parser.add_argument('--destination',type=Path,default=Path(__file__).resolve().parent)
a=parser.parse_args();w=a.work.resolve();v=a.destination.resolve()
def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
for name in ['generate.cmd','generate.log','run_benchmark.py']:
    if (w/name).exists():copy(w/name,v/name)
for backend in ['mint','ampli','mint_tail']:
    out=w/backend;dst=v/backend
    for name in [backend+'.cmd',backend+'.log',backend+'_started.json',backend+'_finished.json']:
        if (w/name).exists():copy(w/name,v/name)
    for folder in ['Cards','Events/benchmark_300k']:
        for source in (out/folder).iterdir():
            if source.is_file():copy(source,dst/source.relative_to(out))
    sp=out/'SubProcesses'
    for pattern in ['P*/G*/log_MINT*.txt','P*/G*/res_*.dat','P*/G*/results.dat',
                    'P*/G*/ampli_job.dat','P*/G*/ampli_pool.dat',
                    'P*/G*/ampli_grids','P*/G*/grid.MC_integer','P*/G*/input_app.txt',
                    'P*/G*/mint_overweight_trials.dat',
                    'ampli_production.json','res_*.txt','nevents_unweighted','subproc.mg','randinit']:
        for source in sp.glob(pattern):
            if source.is_file() and not source.is_symlink():copy(source,dst/source.relative_to(out))
    if backend in ('mint','mint_tail'):
        for source in sp.glob('P*/G*/events.lhe'):
            if source.is_file():copy(source,dst/source.relative_to(out))
    for name in ['mint_module.f90','ampli_mint_adapter.f90','simple_integrator.f90',
                 'integrator_helpers.f90','driver_mintMC.f','genps_fks_radiation.f']:
        copy(sp/name,dst/'SubProcesses'/name)
    for name in ['ampli_pool.py','amcatnlo_run_interface.py']:
        copy(out/'bin/internal'/name,dst/'bin/internal'/name)
print('Evidence archived at',v)
