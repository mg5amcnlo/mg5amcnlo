#!/usr/bin/env python3
"""Compare archived quark Sudakov ratios with fresh Simple-shower API trials.

Usage: dy_fresh_grid.py WORK PYTHIA_INSTALL OUTPUT
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess

from delta_evaluator import DeltaEvaluator, ROOT, _array, _ptr


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('work',type=Path);p.add_argument('pythia',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    work,install,out=[p.resolve() for p in [args.work,args.pythia,args.output]]
    work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('dy_fresh_grid_probe.cc')
    wrapper=ROOT/'vendor/SudGen/pythia_interface_for_sudgen.cc'
    exe=work/'dy_fresh_grid_probe'
    build=['c++','-O2','-std=c++17',str(source),str(wrapper),
        '-I'+str(install/'include'),'-L'+str(install/'lib'),
        '-Wl,-rpath,'+str(install/'lib'),'-lpythia8','-o',str(exe)]
    subprocess.run(build,check=True)
    points=[(q,91.1876,91.1876,2,1,10282800+i) for i,q in enumerate([10.,20.,50.])]
    points.append((20.,200.,500.,2,1,10282803))
    # Repeat one point independently to check statistical uncertainty.
    points.append((20.,91.1876,91.1876,2,1,10282804))
    inp=''.join(' '.join(map(str,p))+'\n' for p in points)
    points_path=work/'points.txt';points_path.write_text(inp)
    def run(mode):
        prefix=work/f'mode{mode}'
        command=[str(exe),str(mode),str(prefix)]
        # The Pythia constructor uses PYTHIA8DATA; set it explicitly.
        import os
        env=dict(os.environ,PYTHIA8DATA=str(install/'share/Pythia8/xmldoc'))
        result=subprocess.run(command,input=inp,text=True,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,env=env,check=True)
        Path(str(prefix)+'.log').write_text(result.stdout)
        assert 'SIMPLE_TYPES_CONFIRMED 1 1 1' in result.stdout
        rows=[]
        for line in result.stdout.splitlines():
            if not line.startswith('RESULT '):continue
            a=line.split(); mode_,q,Q,m,pdg,typ,seed,prob,n=a[1:]
            assert int(mode_)==mode
            probability=float(prob);n=int(n)
            rows.append(dict(mode=mode,q=float(q),Q=float(Q),mass=float(m),
                pdg=int(pdg),topology=int(typ),seed=int(seed),fresh=probability,
                trials=n,fresh_se=(probability*(1-probability)/n)**.5))
        assert len(rows)==len(points)
        return rows,dict(command=command,log_sha256=sha(str(prefix)+'.log'),
            settings_sha256=sha(str(prefix)+'.settings.cmnd'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,[1,2]))
    masses=_array([.33,.33,.5,1.5,4.8,0,0,0,0,0,
                   .000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0])
    rows=[];provenance={}
    for mode,(mode_rows,run_provenance) in zip([1,2],results):
        table=ROOT/f'vendor/SudGen/sudakov_r515_PDFmode{mode}.f.gz'
        with DeltaEvaluator(work=work/f'table_mode{mode}',table=table) as e:
            fn=e.lib.pysudakov_;fn.restype=C.c_double
            def value(q,mass,pdg,topology):
                a=[C.c_double(q),C.c_double(mass),C.c_int(pdg),C.c_int(topology)]
                return fn(*[C.byref(x) for x in a],_ptr(masses))
            for r in mode_rows:
                r['table_num']=value(r['q'],r['mass'],r['pdg'],r['topology'])
                r['table_den']=value(r['Q'],r['mass'],r['pdg'],r['topology'])
                r['stored']=r['table_num']/r['table_den']
                r['stored_over_fresh']=r['stored']/r['fresh']
                r['difference_sigma']=(r['stored']-r['fresh'])/r['fresh_se']
                rows.append(r)
            provenance[str(mode)]=dict(run=run_provenance,table=e.provenance)
    with (out/'fresh_grid.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (out/'fresh_grid.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'fresh_grid_provenance.json').write_text(json.dumps(dict(
        build=build,pythia_version='8.313',topology='II, one incoming up-quark end',
        definition='Fresh direct probability P(q,Q,m) versus archive ratio T(q,m)/T(Q,m). '
            'No external PDFs or gluon roots. Each direct API call uses 500000 Bernoulli trials.',
        settings='Recovered wrapper/defaults: ISRalphaS=.130, one-loop, CMWoff,pT0=0, '
            'masses1-4zero,b4.8,PDF13,nQuark5,unitrenorm/factor,ME/phi/rapidityoff. '
            'The current wrapper is linked unchanged; overrides are explicit in the driver.',
        source_sha256={str(p):sha(p) for p in [Path(__file__),source,wrapper,
            install/'src/SimpleSpaceShower.cc',install/'lib/libpythia8.so',points_path,exe]},
        modes=provenance),indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
