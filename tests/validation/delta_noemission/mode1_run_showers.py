#!/usr/bin/env python3
"""Generate coupling-matched Simple-shower references for the mode-1 audit.

The archive itself is untouched. Actual ttbar showers use ordinary backward
PDF evolution; PDFmode=1 applies only to the previously generated table.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = """! Couplings from the historical SudGen wrapper and Pythia8.225 defaults.
SpaceShower:alphaSvalue = 0.130
TimeShower:alphaSvalue = 0.1365
SpaceShower:alphaSorder = 1
TimeShower:alphaSorder = 1
SpaceShower:alphaSuseCMW = off
TimeShower:alphaSuseCMW = off
PartonShowers:model = 1
SpaceShower:pdfMode = 0
TimeShower:pdfMode = 0
"""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('study', type=Path)
    p.add_argument('pythia', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--shards', type=int, default=8)
    p.add_argument('--repeats', type=int, default=300)
    p.add_argument('--reuse-shards', action='store_true')
    args = p.parse_args()
    study, install, out = args.study.resolve(), args.pythia.resolve(), args.output.resolve()
    work = study/'mode1_study'
    work.mkdir(exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'tests/validation/delta_noemission/pythia_probe.cc'
    executable = work/'pythia_probe'
    build = ['c++', '-O2', '-std=c++17', str(source), '-I'+str(install/'include'),
             '-L'+str(install/'lib'), '-Wl,-rpath,'+str(install/'lib'), '-lpythia8',
             '-o', str(executable)]
    subprocess.run(build, check=True)
    override = work/'historical_couplings.cmnd'
    override.write_text(SETTINGS)
    shutil.copyfile(override, out/override.name)
    lhe = study/'ttbar_S.lhe'
    content = lhe.read_text()
    matches = list(re.finditer(r'<event>.*?</event>', content, re.S))
    assert len(matches)==871
    header, footer = content[:matches[0].start()], content[matches[-1].end():]
    jobs = []
    for shard in range(args.shards):
        first = len(matches)*shard//args.shards
        last = len(matches)*(shard+1)//args.shards
        shard_lhe = work/f'input_{shard:02d}.lhe'
        shard_lhe.write_text(header+'\n'.join(m.group() for m in matches[first:last])+footer)
        for pdf in [13, 15]:
            prefix = work/f'matched_pdf{pdf}_shard{shard:02d}'
            seed = 10278000 + pdf*100 + shard
            command = [str(executable), str(install/'share/Pythia8/xmldoc'), 'shower',
                       str(shard_lhe), str(prefix), str(args.repeats), str(seed), str(pdf),
                       'both', str(override)]
            jobs.append(dict(pdf=pdf, shard=shard, first=first, last=last, seed=seed,
                             command=command, prefix=str(prefix)))
    def run(job):
        prefix = Path(job['prefix']); log = Path(str(prefix)+'.log')
        if args.reuse_shards and log.exists() and 'failures=0' in log.read_text():
            return job
        with log.open('w') as stream:
            subprocess.run(job['command'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        text = log.read_text()
        assert 'PROOF SimpleShowerModel=1 SimpleSpaceShower=1 SimpleTimeShower=1' in text
        assert f'Born={job["last"]-job["first"]} repeats={args.repeats} failures=0' in text
        return job
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            job = future.result()
            print(f'Completed {i}/{len(jobs)}: PDF{job["pdf"]} shard{job["shard"]} '
                  f'({time.monotonic()-started:.1f}s)', flush=True)
    baseline = [json.loads(line) for line in (study/'branch_exact.born.jsonl').open()]
    samples = {}
    for pdf in [13, 15]:
        prefix = work/f'matched_pdf{pdf}'
        selected = sorted((j for j in jobs if j['pdf']==pdf), key=lambda j:j['shard'])
        born_count = trial_count = emit_count = fail_count = 0
        with Path(str(prefix)+'.born.jsonl').open('w') as bo, Path(str(prefix)+'.trials.jsonl').open('w') as tr:
            for job in selected:
                with Path(job['prefix']+'.born.jsonl').open() as stream:
                    for line in stream:
                        b = json.loads(line); b['event'] += job['first']
                        assert b==baseline[b['event']], (pdf, b['event'])
                        assert b['event']==born_count
                        bo.write(json.dumps(b, separators=(',', ':'))+'\n'); born_count += 1
                with Path(job['prefix']+'.trials.jsonl').open() as stream:
                    for line in stream:
                        t = json.loads(line); t['event'] += job['first']
                        assert (t['event'], t['trial'])==divmod(trial_count, args.repeats)
                        fail_count += not t['ok']
                        if t['ok']:
                            emit_count += t['pt']>0
                            if t['pt']>0:
                                assert len(t['real']['ids'])==5 and t['real']['ids'][2:4]==[6,-6]
                        tr.write(json.dumps(t, separators=(',', ':'))+'\n'); trial_count += 1
        assert born_count==len(matches) and trial_count==len(matches)*args.repeats and fail_count==0
        shutil.copyfile(selected[0]['prefix']+'.settings.cmnd', Path(str(prefix)+'.settings.cmnd'))
        shutil.copyfile(Path(str(prefix)+'.settings.cmnd'), out/f'matched_pdf{pdf}.settings.cmnd')
        samples[f'matched_pdf{pdf}'] = dict(prefix=str(prefix), born=born_count,
            trials=trial_count, emitted=emit_count, failures=fail_count,
            input_born_exactly_matches_baseline=True,
            merged_sha256={suffix:sha(str(prefix)+suffix) for suffix in
                           ['.born.jsonl','.trials.jsonl','.settings.cmnd']})
        print(json.dumps(samples[f'matched_pdf{pdf}']), flush=True)
    provenance = dict(pythia_version='8.313', build=build, samples=samples, shards=jobs,
        source_sha256=sha(source), executable_sha256=sha(executable),
        library_sha256=sha(install/'lib/libpythia8.so'), input_lhe_sha256=sha(lhe),
        settings_override_sha256=sha(override),
        original_settings='Existing ttbar first-emission probe settings, overridden only as recorded',
        runtime_assertions=['SimpleShowerModel','SimpleSpaceShower','SimpleTimeShower',
                            'PartonShowers:model=1','SpaceShower:pdfMode=0','TimeShower:pdfMode=0'],
        historical_couplings=dict(commit='df4187461e8d10f05e78987dd8de5f02fe4e719a',
            isr_source='SudGen/pythia_interface_for_sudgen.cc:75, alphaSvalue=0.130',
            fsr_source='Pythia/Pythia/pythia-src/share/Pythia8/xmldoc/TimelikeShowers.xml:171, default0.1365',
            fsr_order_source='TimelikeShowers.xml:181, default1',
            fsr_cmw_source='TimelikeShowers.xml:197, defaultoff'),
        caveat='Couplings match the historical source; producing archive binary and all generation settings remain unrecorded.',
        elapsed_seconds=time.monotonic()-started)
    for name in ['baseline_driver_check', 'settings_differences']:
        audit = out/f'{name}.json'
        if audit.exists():
            provenance[name] = json.loads(audit.read_text())
    (out/'shower_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (work/'shower_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__=='__main__':
    main()
