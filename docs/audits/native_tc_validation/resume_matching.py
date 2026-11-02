#!/usr/bin/env python3
"""Plan/replay the complete original matching cohort without hard regeneration.

Default writes a plan only. Execution requires the explicitly reviewed
MacNLOPSMatch.cc SHA and unchanged source/card/normalization bindings. Every
source channel is replayed with its original seed, including successful ones.
"""
import argparse
from collections import Counter
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from run import DEFAULT_BASE,PROCESSES,ROOT,PYTHIA,dump,sha
from madgraph.various import macnlops,collect_events
from madgraph.various.banner import RunCardNLO


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def bind(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def source_binding(path):
    digest=hashlib.sha256();count=0;tail=b''
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk);joined=tail+chunk;count+=joined.count(b'<event>');tail=joined[-6:]
        stream.seek(max(0,path.stat().st_size-128));footer=stream.read()
    if not footer.strip().endswith(b'</LesHouchesEvents>'):raise RuntimeError('Truncated hard source: '+str(path))
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest(),events=count)


def cohort(process,seed):
    rows=[]
    for line in (process/'SubProcesses/nevents_unweighted').read_text().splitlines():
        fields=line.split()
        if not fields or int(fields[1])==0:continue
        path=(process/'SubProcesses'/fields[0]).resolve();row=source_binding(path)
        if row['events']!=int(fields[1]):raise RuntimeError('Hard source count differs from nevents_unweighted: '+str(path))
        index=len(rows);actual_seed=1+(seed+index)%899999999
        previous=path.parent/'events_macnlops.json';record=None
        if previous.exists():
            try:record=json.loads(previous.read_text())
            except ValueError:record=dict(status='invalid_json')
        if record and 'seed' in record and record['seed']!=actual_seed:raise RuntimeError('Recorded native seed differs from original cohort order')
        row.update(index=index,seed=actual_seed,nevents_unweighted_fields=fields,
            previous_summary=record,previous_summary_binding=bind(previous) if previous.exists() else None,
            normalization_files={name:dict(bind(path.parent/name),text=(path.parent/name).read_text())
                for name in ('res_1','res.dat','res_2.dat') if (path.parent/name).is_file()})
        rows.append(row)
    return rows


def no_live_matching(rows):
    directories={str(Path(row['path']).parent) for row in rows}
    found=[]
    for pid in Path('/proc').iterdir():
        if not pid.name.isdigit():continue
        try:
            if str((pid/'cwd').resolve()) not in directories:continue
            command=(pid/'cmdline').read_bytes().replace(b'\0',b' ')
            if b'macnlops_onestep' in command:found.append(dict(pid=int(pid.name),command=command.decode(errors='replace')))
        except (OSError,RuntimeError):continue
    if found:raise RuntimeError('Original matching processes are still live: '+str(found))


def verify_inputs(process,plan):
    if macnlops.source_files(process)!=plan['generation_sources']:raise RuntimeError('Hard generation sources/cards changed since the recovery plan')
    for name,item in plan['controls'].items():
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Recovery control changed: '+name)
    for item in plan['python_apis'].values():
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Native Python collection/preparation API changed during recovery')
    for row in plan['source_bindings']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Original hard source changed: '+row['path'])
        for name,item in row['normalization_files'].items():
            if sha(item['path'])!=item['sha256']:raise RuntimeError('Original normalization changed: '+item['path'])


def archive_previous(process,plan,destination):
    compact=destination/'previous_compact';bulky=destination/'previous_bulky'
    if compact.exists() or bulky.exists():raise RuntimeError('Recovery archival already exists; use a new --label for another full-cohort replay')
    compact.mkdir();bulky.mkdir();records=[]
    for row in plan['source_bindings']:
        channel=Path(row['path']).parent
        for name in ('events_macnlops.json','macnlops_onestep.log','events_macnlops.lhe.settings.cmnd'):
            source=channel/name
            if not source.is_file():continue
            item=bind(source);relative=source.relative_to(process)
            if name.endswith('.cmnd'):
                target=compact/'deduplicated_settings'/(item['sha256']+'.cmnd')
            else:target=compact/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(source,target)
            records.append(dict(item,archived=str(target),operation='copied compact evidence'))
        source=channel/'events_macnlops.lhe'
        if source.is_file():
            item=bind(source);target=bulky/source.relative_to(process);target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target);records.append(dict(item,archived=str(target),operation='moved old matched output'))
    dump(destination/'archive_manifest.json',dict(created_utc=now(),records=records,
        hard_source_files_moved_or_changed=False,
        bulky_policy='Retain until the full replacement cohort and native collection verify; remove only after successful replacement.'))
    return bind(destination/'archive_manifest.json')


def collected_weights(path):
    counts=Counter();sumw=0.;compensation=0.;sumabs=0.
    with gzip.open(path,'rb') as stream:
        for line in stream:
            if not line.lstrip().startswith(b'<event>'):continue
            header=next(stream).split();w=float(header[2].replace(b'D',b'E').replace(b'd',b'e'))
            if not math.isfinite(w):raise RuntimeError('Nonfinite collected source weight')
            step=w-compensation;total=sumw+step;compensation=(total-sumw)-step;sumw=total
            sumabs+=abs(w);counts['events']+=1;counts['negative']+=w<0
    return dict(counts=dict(counts),source_sumw=sumw,source_sumabs=sumabs)


def physical_events(path):
    """Hash complete written endpoints, excluding intentional decision weights."""
    event=None;in_weights=False
    with Path(path).open('rb') as stream:
        for line in stream:
            stripped=line.lstrip()
            if stripped.startswith(b'<event>'):
                event=hashlib.sha256();in_weights=False
            if event is None:continue
            if stripped.startswith(b'<rwgt>'):in_weights=True
            if not in_weights and b'<macnlops_decision ' not in line:event.update(line)
            if stripped.startswith(b'</rwgt>'):in_weights=False
            if stripped.startswith(b'</event>'):
                yield event.hexdigest();event=None
    # An interrupted old matcher can leave a partial final event; complete
    # earlier endpoints remain usable as a prefix control.


def compare_previous_endpoints(process,plan,destination):
    rows=[]
    for source in plan['source_bindings']:
        channel=Path(source['path']).parent
        previous=destination/'previous_bulky'/channel.relative_to(process)/'events_macnlops.lhe'
        if not previous.is_file():continue
        replacement=channel/'events_macnlops.lhe';new=iter(physical_events(replacement));old_digest=hashlib.sha256();new_digest=hashlib.sha256();n=0
        for n,old in enumerate(physical_events(previous),1):
            try:current=next(new)
            except StopIteration:raise RuntimeError('Replacement endpoint sequence shorter than old prefix: '+str(channel))
            old_digest.update(old.encode());new_digest.update(current.encode())
            if current!=old:raise RuntimeError('Physical endpoint/caps changed in original-prefix control: '+str(channel)+' event '+str(n))
        if (source['previous_summary'] or {}).get('status')=='complete' and n!=source['events']:
            raise RuntimeError('Previously complete output has a truncated endpoint control: '+str(channel))
        rows.append(dict(channel=str(channel),events_compared=n,old_physical_sequence_sha256=old_digest.hexdigest(),
            new_physical_sequence_sha256=new_digest.hexdigest(),passed=True))
    report=dict(passed=True,channels=rows,events_compared=sum(row['events_compared'] for row in rows),
        definition='Byte-exact event records excluding the rwgt block and macnlops_decision tag. Includes source nominal header, physical particles, immutable hard packet, continuation scale and all scale caps. Interrupted old files contribute complete-event prefixes only.')
    dump(destination/'previous_endpoint_comparison.json',report);return bind(destination/'previous_endpoint_comparison.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--process',default='Wj',choices=[p['key'] for p in PROCESSES])
    parser.add_argument('--run',default='run_1M_native_tc');parser.add_argument('--events',type=int,default=1000000)
    parser.add_argument('--workers',type=int,default=32);parser.add_argument('--seed',type=int)
    parser.add_argument('--label',default='global_gluon_quadrature')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--matcher-sha')
    a=parser.parse_args();base=a.base.resolve();p=next(p for p in PROCESSES if p['key']==a.process)
    process=base/'processes'/p['directory'];destination=base/'recovery'/a.process/(a.run+'_'+a.label)
    destination.mkdir(parents=True,exist_ok=True);planpath=destination/'plan.json'
    randinit=process/'SubProcesses/randinit';seed=int(randinit.read_text().strip().split('=')[1])
    if a.seed is not None and seed!=a.seed:raise RuntimeError('Requested seed differs from original randinit')
    card=RunCardNLO(str(process/'Cards/run_card.dat'))
    if (card['nevents']!=a.events or not card['macnlops_keep_rejected'] or card['event_norm']!='sum'
        or any(card['reweight_scale']) or any(card['reweight_pdf']) or len(card['dynamical_scale_choice'])!=1
        or len(card['lhaid'])!=1 or card['store_rwgt_info']):
        raise RuntimeError('Recovery requires the original sum-normalized, kept-rejection, single-scale/PDF source')
    banner=process/'Events'/a.run/(a.run+'_tag_1_banner.txt');target=banner.parent/'events.lhe.gz'
    if not banner.is_file():raise RuntimeError('Original run banner missing')
    if target.exists() or target.with_suffix('').exists():raise RuntimeError('Refusing to overwrite a collected canonical event file')
    matcher=process/'MCatNLO/srcPythia8/MacNLOPSMatch.cc'
    if planpath.exists():
        plan=json.loads(planpath.read_text())
        if plan['status']!='planned':raise RuntimeError('This recovery epoch already executed; use another --label')
        if plan['base_seed']!=seed or plan['source_events']!=a.events:raise RuntimeError('Recovery seed/events differ from recorded plan')
        verify_inputs(process,plan)
    else:
        rows=cohort(process,seed)
        if sum(row['events'] for row in rows)!=a.events:raise RuntimeError('Original cohort is incomplete')
        plan=dict(schema='native-tc-full-cohort-recovery-v1',created_utc=now(),status='planned',
            process=p,run=a.run,source_events=a.events,source_jobs=len(rows),base_seed=seed,workers=a.workers,
            source_order='Every nonzero nevents_unweighted row in original file order, exactly as native reweight_and_collect_events.',
            seed_rule='1+(base_seed+zero_based_source_index)%899999999',source_bindings=rows,
            controls={name:bind(path) for name,path in dict(nevents_unweighted=process/'SubProcesses/nevents_unweighted',
                randinit=randinit,banner=banner).items()},generation_sources=macnlops.source_files(process),
            previous_matcher_source=bind(matcher),stock_pythia_library=bind(PYTHIA/'lib/libpythia8.so'),
            previous_compiled_matcher=bind(process/'MCatNLO/srcPythia8/MacNLOPSMatch.o') if (process/'MCatNLO/srcPythia8/MacNLOPSMatch.o').is_file() else None,
            previous_provider_executables={str(path):bind(path) for path in process.glob('SubProcesses/P*/macnlops_onestep')},
            python_apis={str(path):bind(path) for path in (Path(macnlops.__file__),Path(collect_events.__file__))},
            source_generation_repeated=False,subset_selection=False,warning_based_retry=False)
        dump(planpath,plan)
    print('Recovery plan:',a.process,len(plan['source_bindings']),'jobs,',a.events,'immutable hard events; seed',seed,flush=True)
    print('Previous statuses:',dict(Counter((r['previous_summary'] or {}).get('status','missing') for r in plan['source_bindings'])),flush=True)
    if not a.execute:
        print('Plan only; no source mutation, rebuild, matching, archival or collection.',planpath,flush=True);return
    if not a.matcher_sha or sha(matcher)!=a.matcher_sha or sha(ROOT/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc')!=a.matcher_sha:
        raise RuntimeError('--execute requires the reviewed frozen matcher SHA in both process and repository')
    if sha(PYTHIA/'lib/libpythia8.so')!=plan['stock_pythia_library']['sha256']:raise RuntimeError('Stock Pythia library changed')
    no_live_matching(plan['source_bindings']);verify_inputs(process,plan)
    plan['archive']=archive_previous(process,plan,destination);plan.update(status='building',started_utc=now(),matcher_source=bind(matcher));dump(planpath,plan)
    (process/'MCatNLO/srcPythia8/MacNLOPSMatch.o').unlink(missing_ok=True)
    for executable in process.glob('SubProcesses/P*/macnlops_onestep'):executable.unlink()
    inputs=[row['path'] for row in plan['source_bindings']]
    jobs=macnlops.prepare_first_emission(process,card,PYTHIA,inputs,seed,keep_rejected=True,workers=a.workers)
    if len(jobs)!=len(inputs):raise RuntimeError('Native preparation changed the complete source cohort')
    for job,row in zip(jobs,plan['source_bindings']):
        if Path(job['input'])!=Path(row['path']) or int(job['args'][job['args'].index('--seed')+1])!=row['seed']:
            raise RuntimeError('Native prepared order or seed differs from original cohort')
    if sha(matcher)!=a.matcher_sha:raise RuntimeError('Matcher changed during build')
    verify_inputs(process,plan)
    plan.update(status='matching',jobs=jobs,compiled_matcher=bind(process/'MCatNLO/srcPythia8/MacNLOPSMatch.o'),
        executables={str(path):bind(path) for path in process.glob('SubProcesses/P*/macnlops_onestep')});dump(planpath,plan)
    def run(job):
        command=[str(Path(job['cwd'])/job['exe']),*job['args']]
        result=subprocess.run(command,cwd=job['cwd'])
        return dict(input=job['input'],returncode=result.returncode)
    exits=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            exits.append(future.result());print('Replayed',len(exits),'/',len(jobs),'last exit',exits[-1]['returncode'],flush=True)
    plan['job_exit_codes']=exits;verify_inputs(process,plan)
    if any(item['returncode'] for item in exits):
        plan.update(status='matching_failed',finished_utc=now());dump(planpath,plan)
        macnlops.check_first_emission(jobs);raise RuntimeError('At least one replay failed')
    summaries,report=macnlops.check_first_emission(jobs)
    if report['counts'].get('input_S',0)+report['counts'].get('input_H',0)!=a.events:raise RuntimeError('Replay source population changed')
    plan['previous_endpoint_comparison']=compare_previous_endpoints(process,plan,destination)
    summary=banner.parent/'macnlops_first_emission.json';dump(summary,dict(report,channels=summaries))
    outputs,header=macnlops.first_emission_collection(jobs,summaries,str(banner),str(banner.parent))
    plan.update(status='collecting',native_matching_summary=bind(summary));dump(planpath,plan)
    if target.exists() or target.with_suffix('').exists():raise RuntimeError('Canonical event file appeared during replay; refusing replacement')
    collect_events.collect_events(output=str(target),header_template=header,
        template_header_keep_tags=['MGVersion','MG5ProcCard','MGRunCard','slha','MGShowerCard'],
        input_files=outputs,seed=seed,subset=None,workers=a.workers,mode='auto',prefer_pigz=True,gzip_level=6,verbose=False)
    macnlops.write_source_manifest(process,a.run,inputs);verify_inputs(process,plan)
    contract=macnlops.first_emission_contract(target);weights=collected_weights(target)
    if contract['events']!=a.events or weights['counts']['events']!=a.events:raise RuntimeError('Collected population changed')
    if not math.isclose(weights['source_sumw'],report['weight_sums']['input'],rel_tol=2.e-7,abs_tol=1.e-8):
        raise RuntimeError('Native collection changed the original full-cohort source normalization')
    for source in (process/'SubProcesses').glob('res*.txt'):
        if not (banner.parent/source.name).exists():shutil.copy2(source,banner.parent/source.name)
    plan.update(status='complete',finished_utc=now(),collected_output=bind(target),contract=contract,
        source_hashes_and_normalizations_unchanged=True,collected_weights=weights,
        matching_input_sumw=report['weight_sums']['input'],normalization_tolerance=dict(relative=2.e-7,absolute=1.e-8),
        previous_bulky_outputs_removed_after_verified_replacement=True)
    shutil.rmtree(destination/'previous_bulky');dump(planpath,plan)
    print('Complete full-cohort native matching/collection:',target,flush=True)


if __name__=='__main__':main()
