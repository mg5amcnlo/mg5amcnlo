#!/usr/bin/env python3
"""Recover native first-emission matching/collection without source regeneration.

Default is a read-only plan. --execute requires an explicitly frozen matcher
source hash and archives every old channel result before native preparation.
Original source order, seeds, native collection and source manifests are reused.
"""
from pathlib import Path
import argparse,collections,concurrent.futures,datetime,gzip,hashlib,json,os,shutil,subprocess,sys
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS')
BASE=Path(__file__).resolve().parent
PROCESS=BASE/'processes/Wbb_4FS'
PYTHIA=Path('/nfs/home/rikkert/Downloads/pythia8')
sys.path.insert(0,str(ROOT))
from madgraph.various import macnlops
from madgraph.various import collect_events
from madgraph.various.banner import RunCardNLO

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()
def bind(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def source_binding(path):
    h=hashlib.sha256();count=0;tail=b''
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):
            h.update(data);joined=tail+data;count+=joined.count(b'<event>');tail=joined[-6:]
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest(),events=count)
def source_rows(seed):
    rows=[]
    for line in (PROCESS/'SubProcesses/nevents_unweighted').read_text().splitlines():
        fields=line.split()
        if not fields or fields[1]=='0':continue
        path=(PROCESS/'SubProcesses'/fields[0]).resolve();expected=int(fields[1])
        record=source_binding(path);assert record['events']==expected,(path,record['events'],expected)
        index=len(rows);actual=1+(seed+index)%899999999
        old=path.parent/'events_macnlops.json';previous=None
        if old.is_file():
            try:previous=json.loads(old.read_text())
            except ValueError:pass
        if previous is not None:assert previous['seed']==actual,(path,previous['seed'],actual)
        record.update(index=index,seed=actual,nevents_unweighted_fields=fields,previous_status=previous.get('status') if previous else None)
        rows.append(record)
    return rows

def archive_previous(rows,destination):
    destination.mkdir(parents=True,exist_ok=False);moved={};copied={}
    for row in rows:
        channel=Path(row['path']).parent
        for name in ['events_macnlops.lhe','events_macnlops.json','events_macnlops.lhe.settings.cmnd','macnlops_onestep.log']:
            path=channel/name
            if not path.is_file():continue
            relative=path.relative_to(PROCESS);target=destination/'previous_matching'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            moved[str(relative)]=bind(path);path.rename(target)
    preserve=[PROCESS/'SubProcesses/nevents_unweighted',PROCESS/'SubProcesses/randinit',PROCESS/'SubProcesses/macnlops_onestep.sh',PROCESS/'Cards/run_card.dat',PROCESS/'Cards/param_card.dat',PROCESS/'Cards/macnlops_profile.json',PROCESS/'Cards/macnlops_shower.cmnd',BASE/'validation/source_epoch.json']
    preserve += [p for p in PROCESS.glob('SubProcesses/P*/macnlops_onestep') if p.is_file()]
    preserve += [p for pattern in ['SubProcesses/P*/macnlops_onestep.build.log','SubProcesses/P*/macnlops_routes.dat'] for p in PROCESS.glob(pattern) if p.is_file()]
    preserve += [PROCESS/'MCatNLO/srcPythia8'/name for name in ['MacNLOPSMatch.o','MacNLOPSMatch.cc','MacNLOPSMatch.build.log','MacNLOPS.h','Makefile']]
    for path in preserve:
        if not path.is_file():continue
        relative=path.relative_to(BASE);target=destination/'bindings'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);copied[str(relative)]=bind(path)
    dump(destination/'archive_manifest.json',dict(created_utc=stamp(),moved=moved,copied=copied,source_event_files_moved_or_changed=False))
    return dict(directory=str(destination),files_moved=len(moved),files_copied=len(copied),manifest=bind(destination/'archive_manifest.json'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='run_1M_wbb4fs_nocuts')
    parser.add_argument('--seed',type=int,default=4020101)
    parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--matcher-sha',help='Required frozen SHA256 of process MacNLOPSMatch.cc when executing')
    parser.add_argument('--archive-label',default='before_pdf_support_guard')
    args=parser.parse_args()
    randinit=PROCESS/'SubProcesses/randinit'
    actual=int(randinit.read_text().strip()[2:]);assert actual==args.seed,(actual,args.seed)
    card=RunCardNLO(str(PROCESS/'Cards/run_card.dat'))
    assert card['nevents']==1000000 and card['macnlops_keep_rejected']
    assert not any(card['reweight_scale']) and not any(card['reweight_pdf'])
    assert len(card['dynamical_scale_choice'])==len(card['lhaid'])==1 and not card['store_rwgt_info']
    banner=PROCESS/'Events'/args.run/f'{args.run}_tag_1_banner.txt'
    assert banner.is_file(),banner
    target=banner.parent/'events.lhe.gz';assert not target.exists(),('Refusing to replace collected events',target)
    rows=source_rows(args.seed)
    assert len(rows)==101 and sum(row['events'] for row in rows)==1000000
    failed=next(row for row in rows if '/GF1.0_26/' in row['path'] and '/P0_udx_' in row['path'])
    assert failed['index']==25 and failed['seed']==4020127
    source=PROCESS/'MCatNLO/srcPythia8/MacNLOPSMatch.cc'
    plan=dict(created_utc=stamp(),run=args.run,status='planned',source_events=1000000,source_jobs=len(rows),
              base_seed=args.seed,workers=args.workers,source_order='Exactly the nonzero rows of SubProcesses/nevents_unweighted, as in reweight_and_collect_events.',
              seed_rule='1 + (base_seed + zero_based_source_index) % 899999999',source_bindings=rows,
              banner=bind(banner),nevents_unweighted=bind(PROCESS/'SubProcesses/nevents_unweighted'),randinit=bind(randinit),
              matcher_source=bind(source),pythia_library=bind(PYTHIA/'lib/libpythia8.so'),
              generation_sources=macnlops.source_files(PROCESS),
              native_python={str(path):bind(path) for path in [Path(macnlops.__file__),Path(collect_events.__file__)]},
              source_generation_repeated=False,warning_based_selection_or_retry=False,
              native_APIs=['macnlops.prepare_first_emission','macnlops.check_first_emission','macnlops.first_emission_collection','collect_events.collect_events','macnlops.write_source_manifest'])
    planfile=BASE/'validation/recovery_plan.json';dump(planfile,plan)
    print('Recovery plan:',len(rows),'source jobs,',sum(row['events'] for row in rows),'events; seed',args.seed,flush=True)
    if not args.execute:
        print('No matching, collection, source mutation, build or output archival performed.',flush=True);return
    assert args.matcher_sha and len(args.matcher_sha)==64,'--execute requires --matcher-sha for the frozen process source'
    assert sha(source)==args.matcher_sha,('Matcher source does not match frozen hash',sha(source),args.matcher_sha)
    assert sha(ROOT/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc')==args.matcher_sha,'Process source differs from reviewed repository fix'
    archive=BASE/'recovery'/args.archive_label
    plan['archive']=archive_previous(rows,archive);plan['status']='building';dump(planfile,plan)
    inputs=[row['path'] for row in rows]
    # Native make performs the rebuild; removing archived compiled products
    # prevents an accidentally preserved timestamp from reusing the old matcher.
    (PROCESS/'MCatNLO/srcPythia8/MacNLOPSMatch.o').unlink(missing_ok=True)
    for path in PROCESS.glob('SubProcesses/P*/macnlops_onestep'):path.unlink()
    jobs=macnlops.prepare_first_emission(PROCESS,card,PYTHIA,inputs,args.seed,keep_rejected=True,workers=args.workers)
    assert sha(source)==args.matcher_sha,'Matcher source changed during native build'
    assert macnlops.source_files(PROCESS)==plan['generation_sources'],'Generation sources changed during native build'
    assert len(jobs)==len(rows)
    for job,row in zip(jobs,rows):
        assert Path(job['input'])==Path(row['path'])
        assert int(job['args'][job['args'].index('--seed')+1])==row['seed']
    plan['jobs']=jobs;plan['status']='matching';plan['matching_started_utc']=stamp()
    plan['provider_executables']={str(path):bind(path) for path in sorted(PROCESS.glob('SubProcesses/P*/macnlops_onestep'))}
    plan['compiled_matcher_object']=bind(PROCESS/'MCatNLO/srcPythia8/MacNLOPSMatch.o');dump(planfile,plan)
    def run(job):
        command=[str(Path(job['cwd'])/job['exe'])]+job['args']
        result=subprocess.run(command,cwd=job['cwd'])
        return dict(input=job['input'],returncode=result.returncode)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result());print('Matching completed',len(results),'of',len(jobs),'last exit',results[-1]['returncode'],flush=True)
    plan['job_exit_codes']=results;plan['matching_finished_utc']=stamp()
    for row in rows:assert sha(row['path'])==row['sha256'],('Source LHE changed',row['path'])
    plan['source_hashes_unchanged_after_matching']=True
    if any(row['returncode'] for row in results):
        plan['status']='matching_failed';dump(planfile,plan)
        macnlops.check_first_emission(jobs)
        raise RuntimeError('At least one first-emission job failed')
    summaries,report=macnlops.check_first_emission(jobs)
    assert report['counts'].get('input_S',0)+report['counts'].get('input_H',0)==1000000
    summary=banner.parent/'macnlops_first_emission.json'
    dump(summary,dict(report,channels=summaries));plan['matching_summary']=bind(summary)
    outputs,header=macnlops.first_emission_collection(jobs,summaries,str(banner),str(banner.parent))
    plan['status']='collecting';dump(planfile,plan)
    collect_events.collect_events(output=str(target),header_template=header,
        template_header_keep_tags=['MGVersion','MG5ProcCard','MGRunCard','slha','MGShowerCard'],
        input_files=outputs,seed=args.seed,subset=None,workers=args.workers,
        mode='auto',prefer_pigz=True,gzip_level=6,verbose=False)
    macnlops.write_source_manifest(PROCESS,args.run,inputs)
    assert macnlops.source_files(PROCESS)==plan['generation_sources'],'Generation sources changed during recovery'
    for row in rows:assert sha(row['path'])==row['sha256'],('Source LHE changed',row['path'])
    assert sha(randinit)==plan['randinit']['sha256'],'Global random seed file changed'
    for path in (PROCESS/'SubProcesses').glob('res*.txt'):
        destination=banner.parent/path.name
        if not destination.exists():shutil.copy2(path,destination)
    contract=macnlops.first_emission_contract(target);assert contract['events']==1000000
    plan.update(status='complete',completed_utc=stamp(),collected_output=bind(target),
                source_hashes_unchanged_after_collection=True,collected_contract=contract)
    dump(planfile,plan)
    print('Complete: same 1M immutable source events rematched and collected with native APIs:',target,flush=True)
if __name__=='__main__':main()
