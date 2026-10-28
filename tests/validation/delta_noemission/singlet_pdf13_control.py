#!/usr/bin/env python3
"""Replay existing fixed-start DY/ggH Born samples with Pythia PDF13.

Usage: singlet_pdf13_control.py WORK_DIRECTORY PYTHIA_INSTALL OUTPUT_DIRECTORY
The only change in the isolated C++ source is PDF:pSet=15 ->13. Born samples,
couplings, seeds, starts, shower model and all other settings are preserved.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from singlet_run_showers import ROOT, sha


def settings(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('!'):
            key, value = line.split('=', 1)
            result[key.strip()] = value.strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path); parser.add_argument('pythia', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    work, install, out = args.work.resolve(), args.pythia.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    builddir = work/'pdf13_control'; builddir.mkdir(exist_ok=True)
    template = ROOT/'tests/validation/delta_noemission/singlet_shower_probe.cc'
    original = template.read_text()
    assert original.count('"PDF:pSet = 15"')==1
    original_copy = builddir/'singlet_shower_probe_original.cc'
    original_copy.write_text(original)
    source = builddir/'singlet_shower_probe_pdf13.cc'
    source.write_text(original.replace('"PDF:pSet = 15"', '"PDF:pSet = 13"'))
    exe = builddir/'singlet_pdf13_probe'
    build = ['c++', '-O2', '-std=c++17', str(source), '-I'+str(install/'include'),
        '-L'+str(install/'lib'), '-Wl,-rpath,'+str(install/'lib'), '-lpythia8', '-o', str(exe)]
    subprocess.run(build, check=True)
    prior_file = work/'shower_provenance.json'
    prior = json.loads(prior_file.read_text())
    assert sha(template)==prior['source_sha256']
    jobs = []
    for old in prior['jobs']:
        if old['sample'] not in ['dy_fixed', 'higgs_fixed']:
            continue
        name = old['sample']+'_pdf13'
        prefix = work/f'{name}_shard{old["shard"]:02d}'
        lhe = Path(old['command'][3])
        assert sha(lhe)==old['input_lhe_sha256']
        repeats = old['audit']['repeats']
        command = [str(exe), str(install/'share/Pythia8/xmldoc'), 'shower', str(lhe),
            str(prefix), str(repeats), str(old['seed'])]
        jobs.append(dict(sample=name, original_sample=old['sample'], shard=old['shard'],
            first=old['first'], last=old['last'], prefix=str(prefix),
            original_prefix=old['prefix'], seed=old['seed'], repeats=repeats,
            command=command, input_lhe_sha256=sha(lhe)))
    def run(job):
        log = Path(job['prefix']+'.log')
        with log.open('w') as stream:
            subprocess.run(job['command'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        text = log.read_text()
        assert 'PROOF SimpleShowerModel=1 SimpleSpaceShower=1 SimpleTimeShower=1' in text
        m = re.search(r'AUDIT process_checks=(\d+) maximum_relative_Born_replay_difference=(\S+) emitted=(\d+) Born=(\d+) repeats=(\d+) failures=(\d+)', text)
        assert m
        checks, difference, emitted, born, repeats, failures = m.groups()
        assert int(born)==job['last']-job['first'] and int(repeats)==job['repeats'] and int(failures)==0
        assert int(checks)==int(born)*int(repeats) and float(difference)<1e-10
        old, new = settings(job['original_prefix']+'.settings.cmnd'), settings(job['prefix']+'.settings.cmnd')
        diff = {k: [old.get(k),new.get(k)] for k in old.keys() | new.keys() if old.get(k)!=new.get(k)}
        assert diff=={'PDF:pSet':['15','13']}, diff
        messages = Counter()
        for n, msg in re.findall(r'^\s*\|\s*(\d+)\s+((?:Warning|Error|Abort).*?)\s*\|', text, re.M):
            messages[msg] += int(n)
        assert not any(msg.startswith(('Error','Abort')) for msg in messages)
        job['audit'] = dict(process_checks=int(checks), maximum_relative_Born_replay_difference=float(difference),
            emitted=int(emitted), born=int(born), repeats=int(repeats), failures=int(failures),
            messages=dict(messages), settings_differences=diff, log_sha256=sha(log))
        return job
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            j = future.result()
            print(f'{i}/{len(jobs)} completed: {j["sample"]} shard{j["shard"]}, '
                f'{time.monotonic()-started:.1f}s', flush=True)
    samples = {}
    for original_name in ['dy_fixed', 'higgs_fixed']:
        name = original_name+'_pdf13'; prefix = work/name
        original_born = [json.loads(l) for l in (work/(original_name+'.born.jsonl')).open()]
        selected = sorted((j for j in jobs if j['sample']==name), key=lambda j:j['shard'])
        borns, trial_count, emitted = [], 0, 0
        messages = Counter()
        with Path(str(prefix)+'.born.jsonl').open('w') as bo, Path(str(prefix)+'.trials.jsonl').open('w') as tr:
            for j in selected:
                messages.update(j['audit']['messages'])
                for line in Path(j['prefix']+'.born.jsonl').open():
                    b=json.loads(line);b['event']+=j['first'];assert b['event']==len(borns)
                    assert b==original_born[b['event']]
                    borns.append(b);bo.write(json.dumps(b,separators=(',',':'))+'\n')
                for line in Path(j['prefix']+'.trials.jsonl').open():
                    t=json.loads(line);t['event']+=j['first']
                    assert (t['event'],t['trial'])==divmod(trial_count,j['repeats']) and t['ok']
                    assert 0<=t['pt']<=borns[t['event']]['scalup']*(1+1e-10)
                    assert (t['type']==2 and t['pt']>0) or (t['type']==0 and t['pt']==0)
                    emitted+=t['pt']>0;trial_count+=1
                    tr.write(json.dumps(t,separators=(',',':'))+'\n')
        assert len(borns)==len(original_born) and trial_count==prior['samples'][original_name]['trials']
        shutil.copyfile(selected[0]['prefix']+'.settings.cmnd', Path(str(prefix)+'.settings.cmnd'))
        shutil.copyfile(Path(str(prefix)+'.settings.cmnd'), out/(name+'.settings.cmnd'))
        samples[name] = dict(born=len(borns),trials=trial_count,emitting=emitted,no_emission=trial_count-emitted,
            original_born_equal_bitwise=sha(str(prefix)+'.born.jsonl')==sha(work/(original_name+'.born.jsonl')),
            messages=dict(messages), files={str(p):sha(p) for p in
                [Path(str(prefix)+'.'+s) for s in ['born.jsonl','trials.jsonl','settings.cmnd']]},
            maximum_relative_Born_replay_difference=max(j['audit']['maximum_relative_Born_replay_difference'] for j in selected))
        assert samples[name]['original_born_equal_bitwise']
    result = dict(samples=samples, jobs=jobs, build=build, pythia_version='8.313',
        definition='Existing PDF15 hard Born ensemble and fixed starts replayed with PDF13; '
            'no new Born generation. Same per-shard RNG seeds as PDF15. All other settings unchanged.',
        replacement=['"PDF:pSet = 15"','"PDF:pSet = 13"'],
        source_sha256={str(p):sha(p) for p in [template, original_copy, source, Path(__file__)]},
        executable_sha256=sha(exe), library_sha256=sha(install/'lib/libpythia8.so'),
        pdf_grid_sha256=sha(install/'share/Pythia8/pdfdata/NNPDF23_lo_as_0130_qed_0000.dat'),
        prior_provenance_sha256=sha(prior_file), elapsed_seconds=time.monotonic()-started,
        checks=['All Born JSON files match original PDF15 Born files byte for byte',
            'Every shard settings comparison differs only in PDF:pSet 15->13',
            'Every trial successful; first accepted QCD emission ISR and below start',
            'All process-level Born replay audits pass', 'Simple shower classes checked at runtime',
            'All repetitions included in the reference CDF; no emission-conditioned selection'])
    for path in [work/'pdf13_control_provenance.json', out/'pdf13_control_provenance.json']:
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(samples,indent=2),flush=True)


if __name__=='__main__':main()
