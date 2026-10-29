#!/usr/bin/env python3
"""Fresh whole-API probabilities for every SudGen topology and colour charge.

Usage: dipole_normalization_trials.py WORK PYTHIA_INSTALL OUTPUT
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes as C
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from delta_evaluator import DeltaEvaluator, ROOT, _array, _ptr
from singlet_run_showers import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['work','pythia','output']:p.add_argument(key,type=Path)
    a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('dipole_normalization_probe.cc')
    wrapper=ROOT/'vendor/SudGen/pythia_interface_for_sudgen.cc'
    exe=a.work/'dipole_normalization_probe'
    build=['c++','-O2','-std=c++17',str(source),str(wrapper),'-I'+str(a.pythia/'include'),
           '-L'+str(a.pythia/'lib'),'-Wl,-rpath,'+str(a.pythia/'lib'),'-lpythia8','-o',str(exe)]
    subprocess.run(build,check=True)
    jobs=[]
    for topology in [1,2,3,4]:
        points=[]
        for pdg in [2,21]:
            for cut in [10.,20.,50.]:
                for repeat in range(4 if topology in [2,4] else 1):
                    seed=10283000+1000*topology+100*(pdg==21)+10*int(cut)+repeat
                    points.append((cut,200.,500.,pdg,topology,seed))
        prefix=a.work/f'type{topology}'
        jobs.append(dict(topology=topology,prefix=str(prefix),points=points,
                         command=[str(exe),str(prefix)]))
    def run(job):
        inp=''.join(' '.join(map(str,p))+'\n' for p in job['points'])
        env=dict(os.environ,PYTHIA8DATA=str(a.pythia/'share/Pythia8/xmldoc'))
        result=subprocess.run(job['command'],input=inp,text=True,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,env=env,check=True)
        prefix=Path(job['prefix']);log=Path(str(prefix)+'.log');log.write_text(result.stdout)
        assert 'SIMPLE_TYPES_CONFIRMED 1 1 1' in result.stdout
        rows=[]
        for line in result.stdout.splitlines():
            if not line.startswith('RESULT '):continue
            q,Q,m,pdg,topology,seed,prob,n=line.split()[1:]
            rows.append(dict(q=float(q),Q=float(Q),mass=float(m),pdg=int(pdg),
                topology=int(topology),seed=int(seed),probability=float(prob),trials=int(n)))
        assert len(rows)==len(job['points'])
        messages={msg:int(n) for n,msg in re.findall(r'^\s*\|\s*(\d+)\s+((?:Warning|Error|Abort).*?)\s*\|',result.stdout,re.M)}
        assert not any(msg.startswith(('Error','Abort')) for msg in messages),messages
        job.update(messages=messages,log_sha256=sha(log),settings_sha256=sha(str(prefix)+'.settings.cmnd'))
        (a.output/f'type{job["topology"]}.settings.cmnd').write_bytes(Path(str(prefix)+'.settings.cmnd').read_bytes())
        return rows
    with ThreadPoolExecutor(max_workers=4) as pool:
        batches=list(pool.map(run,jobs))
    trials=[r for b in batches for r in b]
    table=ROOT/'vendor/SudGen/sudakov_r515_PDFmode1.f.gz'
    masses=_array([.33,.33,.5,1.5,4.8,0,0,0,0,0,.000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0])
    rows=[]
    with DeltaEvaluator(work=a.work/'raw_archive',table=table) as e:
        fun=e.lib.pysudakov_;fun.restype=C.c_double
        def value(q,m,pdg,topology):
            x=[C.c_double(q),C.c_double(m),C.c_int(pdg),C.c_int(topology)]
            return fun(*[C.byref(z) for z in x],_ptr(masses))
        for topology in [1,2,3,4]:
            for pdg in [2,21]:
                # The expected line count is independently audited from API
                # preparation, and separately tested by explicit-end trials.
                lines=(4 if pdg==21 else 2) if topology==2 else (2 if pdg==21 else 1)
                for cut in [10.,20.,50.]:
                    selected=[r for r in trials if r['topology']==topology and r['pdg']==pdg and r['q']==cut]
                    probs=np.array([r['probability'] for r in selected]);n=sum(r['trials'] for r in selected)
                    prob=float(probs.mean());binomial=(prob*(1-prob)/n)**.5
                    batch_se=float(probs.std(ddof=1)/len(probs)**.5) if len(probs)>1 else binomial
                    error=max(binomial,batch_se)
                    stored=value(cut,500.,pdg,topology)/value(200.,500.,pdg,topology)
                    line=prob**(1/lines)
                    rows.append(dict(topology=topology,name={1:'II',2:'FF',3:'IF',4:'FI'}[topology],
                        pdg=pdg,q=cut,Q=200.,mass=500.,batches=len(selected),trials=n,
                        fresh_raw=prob,binomial_se=binomial,batch_se=batch_se,raw_se=error,
                        stored_raw_ratio=stored,stored_over_fresh=stored/prob,
                        colour_lines_in_raw=lines,raw_to_one_line_power=1/lines,
                        fresh_one_line=line,one_line_se=error/(lines*prob**(1-1/lines)),
                        stored_one_line=stored**(1/lines)))
        table_provenance=e.provenance
    for name,data in [('whole_api.csv',rows),('whole_api_trials.csv',trials)]:
        with (a.output/name).open('w') as out:
            writer=csv.DictWriter(out,fieldnames=data[0]);writer.writeheader();writer.writerows(data)
    result=dict(rows=rows,trials=trials,jobs=jobs,build=build,
        definition='Whole public Simple-shower noEmissionProbability API versus raw archived PDFmode1 ratios. '
            'Four independent10k calls per FSRpoint;500k per ISRpoint. Candidate per-line powers follow '
            'the independently audited number of active colour ends, not a Casimir-scaling assumption.',
        settings='ISRalphaS=.130,FSR=.1365,one-loop,CMWoff,pT0=0,PDF13,SpacepdfMode1,TimepdfMode0, '
            'masses1-4zero,bottom4.8,ME/azimuth/QEDoff,globalRecoiloff,dampenBeamRecoilon,weightGluonToQuark4.',
        uncertainties='Reported raw_se=max(binomialSE,batchSEM). Batch fluctuations may include fixed '
            'dummy orientation or valence/sea preparation differences; archive interpolation/generation '
            'uncertainty is not included.',
        total_trials=sum(r['trials'] for r in trials),table_provenance=table_provenance,
        input_sha256={str(p):sha(p) for p in [Path(__file__),source,wrapper,exe,
            a.pythia/'src/SimpleTimeShower.cc',a.pythia/'src/SimpleSpaceShower.cc',
            a.pythia/'lib/libpythia8.so',table]})
    (a.output/'whole_api.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(r['name'],r['pdg'],r['q'],'raw',r['fresh_raw'],'stored',r['stored_raw_ratio'],
              'power',r['raw_to_one_line_power'],'line',r['fresh_one_line'],flush=True)


if __name__=='__main__':main()
