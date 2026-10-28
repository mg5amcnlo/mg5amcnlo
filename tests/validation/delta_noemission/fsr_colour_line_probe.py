#!/usr/bin/env python3
"""Inspect and trial individual FSR colour ends in the actual Simple shower."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ['work', 'pythia', 'output']:
        parser.add_argument(key, type=Path)
    parser.add_argument('--trials', type=int, default=200000)
    a = parser.parse_args()
    work, install, out = [p.resolve() for p in [a.work,a.pythia,a.output]]
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_suffix('.cc')
    wrapper = ROOT/'vendor/SudGen/pythia_interface_for_sudgen.cc'
    executable = work/'fsr_colour_line_probe'
    build = ['c++','-O2','-std=c++17',str(source),str(wrapper),
             '-I'+str(install/'include'),'-L'+str(install/'lib'),
             '-Wl,-rpath,'+str(install/'lib'),'-lpythia8','-o',str(executable)]
    subprocess.run(build,check=True)
    prefix = work/'fsr_colour_line_probe'
    command = [str(executable),str(prefix),str(a.trials)]
    env = dict(os.environ,PYTHIA8DATA=str(install/'share/Pythia8/xmldoc'))
    result = subprocess.run(command,env=env,capture_output=True,text=True,check=True)
    log = Path(str(prefix)+'.log')
    log.write_text(result.stdout+result.stderr)
    cases = {}
    for line in result.stdout.splitlines():
        values = line.split()
        if not values or values[0] not in ['COUNTS','END','RESULT']:
            continue
        key = (int(values[1]),int(values[2]))
        row = cases.setdefault(key,dict(type=key[0],pdg=key[1],ends=[],trials=[]))
        if values[0]=='COUNTS':
            row.update(end_count=int(values[3]),radiator_count=int(values[4]))
        elif values[0]=='END':
            row['ends'].append(dict(radiator=int(values[3]),recoiler=int(values[4]),
                colour_type=int(values[5]),isr_type=int(values[6]),pTmax=float(values[7])))
        else:
            probability=float(values[5]);trials=int(values[6])
            row['trials'].append(dict(subset=['one_line','one_radiator','full_system'][int(values[3])],
                ends=int(values[4]),probability=probability,trials=trials,
                standard_error=math.sqrt(probability*(1-probability)/trials)))
    assert len(cases)==4
    for (topology,pdg),row in cases.items():
        lines = 2 if pdg==21 else 1
        radiators = 2 if topology==2 else 1
        assert row['end_count']==lines*radiators
        assert row['radiator_count']==radiators
        assert len(row['ends'])==row['end_count']
        assert all(t['trials']==a.trials for t in row['trials'])
        line = row['trials'][0]
        row['power_checks'] = []
        for trial,power in [(row['trials'][1],lines),(row['trials'][2],lines*radiators)]:
            expected = line['probability']**power
            error = math.hypot(power*line['probability']**(power-1)*line['standard_error'],
                               trial['standard_error'])
            pull = (trial['probability']-expected)/error
            assert abs(pull)<5,(row,pull)
            row['power_checks'].append(dict(subset=trial['subset'],power=power,
                predicted_from_one_line=expected,pull=pull))
    data = dict(cases=list(cases.values()),Q=200.,q=20.,dipole_mass=500.,
        pythia_version='8.313',build=build,command=command,
        definition='Actual SimpleTimeShower::prepare() builds the API-style massless dummy states. '
            'Custom pTnext trials are then restricted to one prepared colour end, '
            'one radiator, or all prepared ends. Independent seeds for each subset.',
        conclusion='FI q/g raw probabilities cover1/2 ends; FF q/g cover2/4 ends. '
            'One emitter/one colour-line factors therefore require FI gluon sqrt, '
            'FF quark sqrt and FF gluon fourth root.',
        limitations=['Dummy massless radiators; does not validate massive-top showering',
            'FI dummy ends have isrType=0 and therefore do not exercise the physical '
            'initial-recoiler PDF-weight block gated by isrType!=0',
            'Header-only private-access shim is diagnostic; the linked Pythia library is unchanged'],
        input_sha256={str(p):sha(p) for p in [Path(__file__),source,wrapper,executable,
            install/'lib/libpythia8.so',install/'src/SimpleTimeShower.cc',log,
            Path(str(prefix)+'.settings.cmnd')]})
    (out/'fsr_colour_line_trials.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data['cases'],indent=2),flush=True)


if __name__=='__main__':
    main()
