#!/usr/bin/env python3
"""Audit native warning occurrences and complete first/full event accounting.

Reads existing output only. Printed warnings are not selected, dropped or retried.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,re,struct
BASE=Path(__file__).resolve().parents[1]
PROCESS=BASE/'processes/Wbb_4FS'
TRIAL='Warning in SimpleSpaceShower::pT2nextQCD: weight above unity'

def binding(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def full_log(path):
    text=path.read_text(errors='replace')
    sums=[];messages=collections.Counter();inside=False;tables=0
    for line in text.splitlines():
        if 'PYTHIA Error and Warning Messages Statistics' in line and 'End PYTHIA' not in line:
            assert not inside,(path,'nested native statistics table')
            inside=True;tables+=1
        elif 'End PYTHIA Error and Warning Messages Statistics' in line:
            assert inside,(path,'unmatched statistics end')
            inside=False
        if inside:
            m=re.match(r'^\|\s*(\d+)\s+(.+?)\s*\|$',line.strip())
            if m and m[2].startswith(('Warning in ','Error in ','Abort from ','Info from ')):
                messages[m[2]]+=int(m[1])
        m=re.match(r'^\|\s*sum\s*\|\s*(\d+)\s+(\d+)\s+(\d+)\s*\|',line.strip())
        if m:sums.append(tuple(map(int,m.groups())))
    assert not inside and tables==1,(path,'incomplete or multiple native warning tables',tables)
    assert len(sums)==1,(path,'expected exactly one final event-statistics sum',sums)
    fatal={k:v for k,v in messages.items() if v and k.startswith(('Error in ','Abort from '))}
    return dict(tried=sums[0][0],selected=sums[0][1],accepted=sums[0][2],messages=dict(messages),fatal_native_statistics=fatal,binding=binding(path))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='run_1M_wbb4fs_nocuts')
    p.add_argument('--shower-index',type=int,default=3);p.add_argument('--expected-events',type=int,default=1000000)
    p.add_argument('--expected-jobs',type=int,default=50);p.add_argument('--first-only',action='store_true')
    p.add_argument('--output',default='warning_accounting_audit.json');a=p.parse_args()
    events=PROCESS/'Events'/a.run
    firstpath=events/'macnlops_first_emission.json'
    source=json.loads(firstpath.read_text());channels=source['channels']
    counts=collections.Counter();warnings=collections.Counter();rows=[]
    for c in channels:
        assert c['status']=='complete' and not c.get('error'),c.get('directory')
        assert c['rejected_events']=='kept',c.get('directory')
        own=c['counts']
        for kind in ('S','H'):
            assert own.get('accepted_'+kind,0)+own.get('rejected_'+kind,0)==own.get('input_'+kind,0),(c['directory'],kind)
        counts.update(own);warnings.update(c.get('native_trial_warnings',{}))
        rows.append(dict(directory=c['directory'],seed=c['seed'],input_S=own.get('input_S',0),input_H=own.get('input_H',0),native_trial_warnings=c.get('native_trial_warnings',{})))
    assert dict(counts)==source['counts'],('first aggregate count mismatch',counts,source['counts'])
    assert counts['input_S']+counts['input_H']==a.expected_events
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=a.run,
      requested_events=a.expected_events,first=dict(channels=len(channels),counts=dict(counts),native_trial_warning_occurrences=dict(warnings),channels_with_warnings=sum(any(v>0 for v in c.get('native_trial_warnings',{}).values()) for c in channels),source_summary=binding(firstpath),channel_bindings=rows),
      method='First counts are native logger counter deltas around actual S-event next() calls, aggregated from complete matching-channel summaries. Full counts come only from complete native error-statistics tables, not from a count of printed warning lines.',
      warning_policy='Native warnings are retained. No warning-based event removal, retry or reseeding is performed.',
      limits='Warning occurrence counts are not distinct affected-event counts or a bound on proposal bias. Native ISR warning suppression below 1 GeV can hide trial-envelope violations. Conditional sampling errors elsewhere do not include this effect. All species/mass/coupling settings must be read from the separate profile audit.')
    if not a.first_only:
        directory=PROCESS/'MCatNLO'/f'RUN_PYTHIA8_{a.shower_index}'
        fullrows=[];total=collections.Counter();allmessages=collections.Counter()
        assert a.expected_events%a.expected_jobs==0
        per=a.expected_events//a.expected_jobs
        for job in range(1,a.expected_jobs+1):
            path=events/f'mcatnlo_run_PYTHIA8_{a.shower_index}__{job}.log'
            row=full_log(path)
            assert row['tried']==row['selected']==row['accepted']==per,(job,row['tried'],row['selected'],row['accepted'])
            assert not row['fatal_native_statistics'],(job,row['fatal_native_statistics'])
            binary=directory/f'run_{job}/wbb_observables.bin'
            with binary.open('rb') as stream: nobs,nweights=struct.unpack('=ii',stream.read(8))
            recordbytes=8*(nobs+nweights)
            assert nobs==19 and nweights>=4,(job,nobs,nweights)
            assert (binary.stat().st_size-8)%recordbytes==0,(job,'truncated observable binary')
            nrecords=(binary.stat().st_size-8)//recordbytes
            assert nrecords==per,(job,'observable row count',nrecords)
            row.update(job=job,observable_records=nrecords,nobs=nobs,nweights=nweights,observable_binary=binding(binary),continuation_card=binding(directory/f'run_{job}/Pythia8.cmd'))
            allmessages.update(row['messages']);total.update({k:row[k] for k in ('tried','selected','accepted')})
            fullrows.append(row)
        assert total['tried']==total['selected']==total['accepted']==a.expected_events
        result['full']=dict(shower_directory=str(directory),jobs=len(fullrows),events_per_job=per,
                            counts=dict(total),native_statistics_occurrences=dict(allmessages),
                            native_ISR_weight_above_unity_occurrences=allmessages[TRIAL],
                            observable_records=sum(r['observable_records'] for r in fullrows),job_records=fullrows)
    result['passed']=True
    target=BASE/'validation'/a.output
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS',a.run,'first channels',len(channels),'first native warnings',dict(warnings))
    if 'full' in result:print('Full counts',result['full']['counts'],'native ISR weight-above-unity',result['full']['native_ISR_weight_above_unity_occurrences'])
    print(target)
if __name__=='__main__':main()
