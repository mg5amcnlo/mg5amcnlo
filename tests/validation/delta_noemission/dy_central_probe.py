#!/usr/bin/env python3
"""Measure a central u-ubar DY Born point with the matched Simple shower.

Usage: dy_central_probe.py MATCHED_STUDY WORK PYTHIA_INSTALL OUTPUT
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from singlet_run_showers import sha


def main():
    p = argparse.ArgumentParser()
    for key in ['matched', 'work', 'pythia', 'output']:
        p.add_argument(key, type=Path)
    p.add_argument('--repeats', type=int, default=100000)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    exe = a.matched/'pdf13/singlet_matched_probe'
    source = a.matched/'pdf13/singlet_matched_probe.cc'
    mass = 91.1876
    half = mass/2
    lhe = a.work/'central.lhe'
    lhe.write_text(f'''<LesHouchesEvents version="3.0">
<header></header>
<init>
2212 2212 6500 6500 0 0 0 0 3 1
1 0 1 1
</init>
<event>
4 1 1 {mass:.17g} 0.0072973525693 0.13
2 -1 0 0 501 0 0 0 {half:.17g} {half:.17g} 0 0 9
-2 -1 0 0 0 501 0 0 {-half:.17g} {half:.17g} 0 0 9
11 1 1 2 0 0 0 0 {half:.17g} {half:.17g} 0 0 9
-11 1 1 2 0 0 0 0 {-half:.17g} {half:.17g} 0 0 9
<scales scalup_1_2="{mass:.17g}" scalup_2_1="{mass:.17g}"/>
</event>
</LesHouchesEvents>
''')
    jobs = []
    for i in range(a.workers):
        n = a.repeats*(i+1)//a.workers-a.repeats*i//a.workers
        prefix = a.work/f'shard{i:02d}'
        jobs.append(dict(prefix=str(prefix), repeats=n, seed=10282900+i,
            command=[str(exe), str(a.pythia/'share/Pythia8/xmldoc'),
                     'shower', str(lhe), str(prefix), str(n), str(10282900+i)]))

    def run(job):
        prefix = Path(job['prefix'])
        log = Path(str(prefix)+'.log')
        with log.open('w') as stream:
            subprocess.run(job['command'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        text = log.read_text()
        assert 'SIMPLE' in text.upper()
        assert f"Born=1 repeats={job['repeats']} failures=0" in text
        counts = re.findall(r'^\s*\|\s*(\d+)\s+((?:Warning|Error|Abort).*?)\s*\|', text, re.M)
        assert not any(msg.startswith(('Error', 'Abort')) for n,msg in counts)
        trials = [json.loads(line) for line in Path(str(prefix)+'.trials.jsonl').open()]
        assert len(trials)==job['repeats']
        assert all(t['ok'] and t['event']==0 and t['trial']==i
                   and 0<=t['pt']<=mass*(1+1.e-10)
                   and ((t['pt']>0 and t['type']==2) or (t['pt']==0 and t['type']==0))
                   for i,t in enumerate(trials))
        job['messages'] = {msg:int(n) for n,msg in counts}
        job['output_sha256'] = {str(prefix)+'.'+suffix:sha(str(prefix)+'.'+suffix)
            for suffix in ['log', 'settings.cmnd', 'particle_masses.json', 'born.jsonl', 'trials.jsonl']}
        return [t['pt'] for t in trials]

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        points = np.concatenate(list(pool.map(run,jobs)))
    assert len(points)==a.repeats
    rows = []
    for q in [5.,10.,20.,30.,50.,75.]:
        prob = float((points<=q).mean())
        rows.append(dict(q=q, no_emission_probability=prob,
            standard_error=(prob*(1-prob)/len(points))**.5,
            trials=len(points), passing=int((points<=q).sum())))
    result = dict(born=dict(ids=[2,-2,11,-11], M=mass, Q=mass,
                           x=[mass/13000]*2, weight=1),
        definition='One central massless incoming u-ubar Born state; matched historical settings, '
                   'PDF13, Simple Pythia8.313. Every repetition contributes to first-QCD-emission CDF.',
        rows=rows, jobs=jobs, input_sha256={str(f):sha(f) for f in
            [Path(__file__),exe,source,lhe,a.pythia/'lib/libpythia8.so']})
    (a.output/'central_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    main()
