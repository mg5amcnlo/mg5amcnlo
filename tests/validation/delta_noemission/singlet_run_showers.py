#!/usr/bin/env python3
"""Generate and replay DY and ggH Born ensembles for common-cut Sudakov tests.

Usage: singlet_run_showers.py WORK_DIRECTORY PYTHIA_INSTALL OUTPUT_DIRECTORY
The four samples share Born states within each process and differ only in their
starting-scale prescription. All repetitions can be used as CDF references.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path); parser.add_argument('pythia', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--born', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=300)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--reuse-shards', action='store_true')
    args = parser.parse_args()
    work, install, out = args.work.resolve(), args.pythia.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'tests/validation/delta_noemission/singlet_shower_probe.cc'
    exe = work/'singlet_probe'; xml = install/'share/Pythia8/xmldoc'
    build = ['c++', '-O2', '-std=c++17', str(source), '-I'+str(install/'include'),
        '-L'+str(install/'lib'), '-Wl,-rpath,'+str(install/'lib'), '-lpythia8', '-o', str(exe)]
    subprocess.run(build, check=True)
    generation, jobs = {}, []
    for process, seed in [('dy', 10281001), ('higgs', 10281002)]:
        command = [str(exe), str(xml), 'generate', process, str(work/process), str(args.born), str(seed)]
        with (work/(process+'.generation.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        generation[process] = dict(seed=seed, command=command,
            files={str(p):sha(p) for p in [work/(process+'.generation.log'),
                work/(process+'.generation.settings.cmnd'),work/(process+'_fixed.lhe'),work/(process+'_damped.lhe')]})
        shutil.copyfile(work/(process+'.generation.settings.cmnd'), out/(process+'.generation.settings.cmnd'))
        for start in ['fixed','damped']:
            name = process+'_'+start
            content = (work/(name+'.lhe')).read_text()
            events = list(re.finditer(r'<event>.*?</event>', content, re.S))
            assert len(events)==args.born
            header, footer = content[:events[0].start()], content[events[-1].end():]
            for shard in range(args.shards):
                first, last = args.born*shard//args.shards, args.born*(shard+1)//args.shards
                prefix = work/f'{name}_shard{shard:02d}'
                lhe = Path(str(prefix)+'.lhe')
                lhe.write_text(header+'\n'.join(m.group() for m in events[first:last])+footer)
                seed = 10282000 + (1000 if process=='higgs' else 0) + (100 if start=='damped' else 0) + shard
                command = [str(exe), str(xml), 'shower', str(lhe), str(prefix), str(args.repeats), str(seed)]
                jobs.append(dict(sample=name, shard=shard, first=first, last=last,
                    prefix=str(prefix), seed=seed, command=command, input_lhe_sha256=sha(lhe)))
    def run(job):
        log = Path(job['prefix']+'.log')
        if not(args.reuse_shards and log.exists() and 'failures=0' in log.read_text()):
            with log.open('w') as stream:
                subprocess.run(job['command'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        text = log.read_text()
        assert 'PROOF SimpleShowerModel=1 SimpleSpaceShower=1 SimpleTimeShower=1' in text
        m = re.search(r'AUDIT process_checks=(\d+) maximum_relative_Born_replay_difference=(\S+) emitted=(\d+) Born=(\d+) repeats=(\d+) failures=(\d+)', text)
        assert m, text[-3000:]
        checks, difference, emitted, born, repeats, failures = m.groups()
        assert int(born)==job['last']-job['first'] and int(repeats)==args.repeats and int(failures)==0
        assert int(checks)==int(born)*args.repeats and float(difference)<1e-10
        job['audit'] = dict(process_checks=int(checks), maximum_relative_Born_replay_difference=float(difference),
            emitted=int(emitted), born=int(born), repeats=int(repeats), failures=int(failures), log_sha256=sha(log))
        return job
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for i, future in enumerate(as_completed(futures),1):
            job = future.result()
            print(f'{i}/{len(jobs)} completed: {job["sample"]} shard{job["shard"]}, '
                f'{time.monotonic()-started:.1f}s', flush=True)
    samples = {}
    born_by_name = {}
    for name in ['dy_fixed','dy_damped','higgs_fixed','higgs_damped']:
        chosen = sorted((j for j in jobs if j['sample']==name), key=lambda j:j['shard'])
        prefix = work/name
        borns, ntrial, emitted = [], 0, 0
        with Path(str(prefix)+'.born.jsonl').open('w') as bo, Path(str(prefix)+'.trials.jsonl').open('w') as tr:
            for job in chosen:
                for line in Path(job['prefix']+'.born.jsonl').open():
                    b=json.loads(line);b['event']+=job['first'];assert b['event']==len(borns)
                    assert len(b['ids'])==(4 if name.startswith('dy') else 3)
                    assert all(c==[0,0] for c in b['colors'][2:])
                    if name.startswith('dy'):assert b['ids'][0]==-b['ids'][1] and b['ids'][2:]==[11,-11]
                    else:assert b['ids']==[21,21,25]
                    bo.write(json.dumps(b,separators=(',',':'))+'\n');borns.append(b)
                for line in Path(job['prefix']+'.trials.jsonl').open():
                    t=json.loads(line);t['event']+=job['first']
                    assert (t['event'],t['trial'])==divmod(ntrial,args.repeats) and t['ok']
                    assert 0<=t['pt']<=borns[t['event']]['scalup']*(1+1e-10)
                    assert (t['type']==2 and t['pt']>0) or (t['type']==0 and t['pt']==0)
                    emitted+=t['pt']>0;ntrial+=1
                    tr.write(json.dumps(t,separators=(',',':'))+'\n')
        assert len(borns)==args.born and ntrial==args.born*args.repeats
        born_by_name[name] = borns
        shutil.copyfile(chosen[0]['prefix']+'.settings.cmnd',Path(str(prefix)+'.settings.cmnd'))
        shutil.copyfile(Path(str(prefix)+'.settings.cmnd'),out/(name+'.settings.cmnd'))
        samples[name] = dict(born=len(borns),trials=ntrial,emitting=emitted,no_emission=ntrial-emitted,
            files={str(p):sha(p) for p in [Path(str(prefix)+'.'+suffix) for suffix in ['born.jsonl','trials.jsonl','settings.cmnd']]},
            maximum_relative_Born_replay_difference=max(j['audit']['maximum_relative_Born_replay_difference'] for j in chosen))
    for process in ['dy','higgs']:
        for fixed,damped in zip(born_by_name[process+'_fixed'],born_by_name[process+'_damped']):
            assert {k:v for k,v in fixed.items() if k not in ['scalup','starts']} == \
                   {k:v for k,v in damped.items() if k not in ['scalup','starts']}
            assert .1*fixed['scalup']<damped['scalup']<fixed['scalup']
    result = dict(pythia_version='8.313',samples=samples,generation=generation,jobs=jobs,build=build,
        source_sha256=sha(source),driver_sha256=sha(__file__),executable_sha256=sha(exe),
        library_sha256=sha(install/'lib/libpythia8.so'),
        pdf_grid_sha256=sha(install/'share/Pythia8/pdfdata/NNPDF23_nlo_as_0119_qed_0000.dat'),
        elapsed_seconds=time.monotonic()-started,
        setup='13TeV internal LO Born samples, pure qqbar->gamma*/Z->e-e+ with66<mll<116GeV; '
              'puregg->H withnominal125GeV andphysicalSMwidth, Hdecaysdisabled; no signed NLO event weighting.',
        starting_scales='fixed:Q=eventsingletmass; damped:Q=M*(.1+.9*u), u=sqrt(r)/(sqrt(r)+sqrt(1-r)),uniformr.',
        reference='All300independent successful shower repetitions perBorn are common-cutCDF references; no probe selection.',
        checks=['RuntimeSimpleShower/SpaceShower/TimeShower classes verified',
            'Ordinary backwardpdfMode0 in actual showers; PDF15,ISRalphaS=.130,FSR=.1365,one-loop,CMWoff',
            'No QED,MPI,hadronization,primordialKT; stableHiggs; firstQCDemissionISR',
            'Every replayed Born identity,colour andmomentum checked before shower',
            'AllfirstemissionpT<=SCALUP; nofailedshowers; event/trialorderingcomplete',
            'Fixed/damped shareexactBornrecords exceptstarts/scalup'])
    for path in [work/'shower_provenance.json',out/'shower_provenance.json']:
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(samples,indent=2),flush=True)


if __name__=='__main__':main()
