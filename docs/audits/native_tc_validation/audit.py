#!/usr/bin/env python3
"""Read-only native accounting and actual continuation-card audit."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re

from run import DEFAULT_BASE, PROCESSES, ROOT, dump, sha

REFERENCE = ROOT/'docs/audits/macnlops_wbb4fs_nocuts_1M_20261101/validation/audit_warnings_accounting.py'
spec=importlib.util.spec_from_file_location('native_warning_reference',REFERENCE)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
THRESHOLD='Error in SimpleSpaceShower::pT2nearThreshold: xPDF = 0'


def audit(base,p,run,events):
    process=base/'processes'/p['directory'];result=base/'results'/p['key']/run
    eventdir=process/'Events'/run
    firstpath=eventdir/'macnlops_first_emission.json'
    source=json.loads(firstpath.read_text());counts=Counter();warnings=Counter();channel_rows=[]
    for channel in source['channels']:
        if channel['status']!='complete' or channel.get('error') or channel['rejected_events']!='kept':
            raise RuntimeError('Incomplete or filtered first-emission channel')
        own=channel['counts']
        for kind in ('S','H'):
            if own.get('accepted_'+kind,0)+own.get('rejected_'+kind,0)!=own.get('input_'+kind,0):
                raise RuntimeError('First matching count mismatch')
        counts.update(own);warnings.update(channel.get('native_trial_warnings',{}))
        channel_rows.append(dict(directory=channel['directory'],seed=channel['seed'],counts=own,
            native_trial_warnings=channel.get('native_trial_warnings',{})))
    if dict(counts)!=source['counts'] or counts['input_S']+counts['input_H']!=events:
        raise RuntimeError('First source aggregate mismatch')
    report=dict(process=p['key'],run=run,events=events,
        first=dict(summary_sha256=sha(firstpath),counts=dict(counts),channels=channel_rows,
                   native_warning_occurrences=dict(warnings)),
        policy='Read-only native tables. No warning-based removal, retry or reseeding.',
        pythia_source_modified=False)
    profile=process/'Cards/macnlops_profile.json'
    if profile.is_file():report['first_profile']=dict(path=str(profile),sha256=sha(profile),values=json.loads(profile.read_text()))
    fullpath=result/'full.json'
    if fullpath.exists():
        full=json.loads(fullpath.read_text());directory=Path(full['shower_directory'])
        match=re.search(r'RUN_PYTHIA8_(\d+)$',str(directory));index=match[1]
        jobs=[];allcounts=Counter();messages=Counter()
        from madgraph.various import macnlops
        masses=macnlops.hard_model_masses(process)
        required={'spaceshower:nquarkin':str(p['flavours']),
            'timeshower:ngluontoquark':str(p['flavours']),
            'standardmodel:alphasnfmax':'5','spaceshower:alphasorder':'1',
            'timeshower:alphasorder':'1','pdf:lepton':'off',
            'spaceshower:mecorrections':'off','timeshower:mecorrections':'off'}
        for item in full['bindings']:
            folder=Path(item['path']).parent
            job=int(folder.name.split('_')[-1]) if folder.name.startswith('run_') else 1
            log=folder/'mcatnlo_run.log'
            if not log.is_file():log=eventdir/f'mcatnlo_run_PYTHIA8_{index}__{job}.log'
            if not log.is_file():log=eventdir/f'mcatnlo_run_PYTHIA8_{index}.log'
            row=reference.full_log(log)
            n=item['events']
            unexpected={k:v for k,v in row['fatal_native_statistics'].items() if k!=THRESHOLD}
            if row['tried']!=n or row['selected']!=n or row['accepted']!=n or unexpected:
                raise RuntimeError(f'Native continuation accounting failed: {job}: {row}')
            card=folder/'Pythia8.cmd';settings={};effective={}
            for line in card.read_text().splitlines():
                text=line.split('!')[0].split('#')[0].strip()
                if '=' in text:
                    key,value=text.split('=',1);settings[key.strip()]=value.strip()
                    effective[key.strip().lower()]=value.strip().lower()
            for key,value in required.items():
                if effective.get(key)!=value:raise RuntimeError(f'Actual full profile differs: job{job} {key}')
            for key in ('spaceshower:alphasvalue','timeshower:alphasvalue'):
                if float(effective.get(key,'nan'))!=.118:raise RuntimeError('Actual full alpha_s differs')
            for pid,mass in masses.items():
                if float(effective.get(str(pid)+':m0','nan'))!=mass:
                    raise RuntimeError(f'Actual full mass differs: job{job}, pdg{pid}')
            row.update(job=job,events=n,card_sha256=sha(card),settings=settings,
                effective_settings=effective,physical_profile_passed=True)
            jobs.append(row);messages.update(row['messages'])
            allcounts.update({key:row[key] for key in ('tried','selected','accepted')})
        if any(allcounts[key]!=events for key in ('tried','selected','accepted')):
            raise RuntimeError('Native total count mismatch')
        report['full']=dict(counts=dict(allcounts),messages=dict(messages),jobs=jobs,
            limits='Warnings count occurrences, not distinct affected events; they do not bound proposal bias.')
        report['clean_native_error_statistics']=not any(
            v and k.startswith(('Error in ','Abort from ')) for k,v in messages.items())
        if messages[THRESHOLD]:
            report['native_threshold_diagnostic']=dict(message=THRESHOLD,occurrences=messages[THRESHOLD],
                interpretation='Native dipole evolution returns without a threshold candidate. Event statistics independently show every input event tried, selected and accepted; no external retry.',
                reference='docs/audits/macnlops_wbb4fs_nocuts_1M_20261101/validation/original_full_warning_accounting_audit.json',
                reference_source_sha256='79b9f2334af443005b7c4722dc629ca4facdce6663b73815c1fa0278aacfa789',
                reference_source_lines=[603,669,1262,1274,1563,1576],
                limitation='The effect of missing threshold dipole candidates on distributions has not been measured.')
    report['native_event_accounting_passed']=True
    report['passed']=True
    dump(result/'native_accounting.json',report)
    print('PASS',p['key'],run,'first',counts['input_S']+counts['input_H'],
          'full',report.get('full',{}).get('counts',{}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--process',nargs='+',default=['EE2j','DY'],choices=[p['key'] for p in PROCESSES])
    parser.add_argument('--run',default='run_1M_native_tc')
    parser.add_argument('--events',type=int,default=1000000)
    a=parser.parse_args()
    for p in PROCESSES:
        if p['key'] in a.process:audit(a.base,p,a.run,a.events)


if __name__=='__main__':main()
