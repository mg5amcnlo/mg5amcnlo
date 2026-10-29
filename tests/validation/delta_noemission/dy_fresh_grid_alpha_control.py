#!/usr/bin/env python3
"""Repeat the fresh quark API check with alphaS(MZ)=.1365 for ISR.

Usage: dy_fresh_grid_alpha_control.py WORK PYTHIA_INSTALL OUTPUT
Run dy_fresh_grid.py first. Only the ISR coupling literal is replaced in an
isolated C++ source; the original diagnostic and prior results are preserved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess

from dy_fresh_grid import ROOT, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('work',type=Path);p.add_argument('pythia',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    work,install,out=[p.resolve() for p in [args.work,args.pythia,args.output]]
    original=Path(__file__).with_name('dy_fresh_grid_probe.cc')
    text=original.read_text()
    needle='?"0.130":"0.1365"'
    assert text.count(needle)==1
    source=work/'dy_fresh_grid_probe_alpha1365.cc'
    source.write_text(text.replace(needle,'?"0.1365":"0.1365"'))
    exe=work/'dy_fresh_grid_probe_alpha1365'
    wrapper=ROOT/'vendor/SudGen/pythia_interface_for_sudgen.cc'
    build=['c++','-O2','-std=c++17',str(source),str(wrapper),
        '-I'+str(install/'include'),'-L'+str(install/'lib'),
        '-Wl,-rpath,'+str(install/'lib'),'-lpythia8','-o',str(exe)]
    subprocess.run(build,check=True)
    points=work/'points.txt';inp=points.read_text()
    reference=out/'fresh_grid.json'
    old=json.loads(reference.read_text())
    def run(mode):
        prefix=work/f'alpha1365_mode{mode}'
        command=[str(exe),str(mode),str(prefix)]
        env=dict(os.environ,PYTHIA8DATA=str(install/'share/Pythia8/xmldoc'))
        result=subprocess.run(command,input=inp,text=True,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,env=env,check=True)
        log=Path(str(prefix)+'.log');log.write_text(result.stdout)
        assert 'SIMPLE_TYPES_CONFIRMED 1 1 1' in result.stdout
        rows=[]
        for l in result.stdout.splitlines():
            if not l.startswith('RESULT '):continue
            a=l.split();mode_,q,Q,m,pdg,typ,seed,prob,n=a[1:]
            assert int(mode_)==mode
            r=next(r for r in old if r['mode']==mode and r['seed']==int(seed))
            value=float(prob);n=int(n)
            rows.append(dict(mode=mode,q=float(q),Q=float(Q),mass=float(m),pdg=int(pdg),
                topology=int(typ),seed=int(seed),fresh=value,trials=n,
                fresh_se=(value*(1-value)/n)**.5,stored=r['stored'],
                stored_over_fresh=r['stored']/value,
                difference_sigma=(r['stored']-value)/((value*(1-value)/n)**.5)))
        return rows,dict(command=command,log_sha256=sha(log),
            settings_sha256=sha(str(prefix)+'.settings.cmnd'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,[1,2]))
    rows=[r for rs,_ in results for r in rs]
    for mode in [1,2]:
        text=(work/f'alpha1365_mode{mode}.settings.cmnd').read_text()
        assert 'SpaceShower:alphaSvalue = 0.1365' in text
    with (out/'fresh_grid_alpha1365.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (out/'fresh_grid_alpha1365.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'fresh_grid_alpha1365_provenance.json').write_text(json.dumps(dict(
        build=build,definition='Only ISRalphaSvalue changes .130->.1365 relative to fresh_grid.json',
        reference_sha256=sha(reference),source_sha256={str(p):sha(p) for p in
            [Path(__file__),original,source,exe,wrapper,points,install/'lib/libpythia8.so']},
        modes={str(mode):provenance for mode,(_,provenance) in zip([1,2],results)}),indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
