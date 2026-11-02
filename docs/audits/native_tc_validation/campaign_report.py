#!/usr/bin/env python3
"""Consolidate available audited native-t_c results without reading events.

Outputs remain explicitly partial until every requested process has complete
first/full results and all required individual audits pass.
"""
import argparse
import csv
import datetime
import io
import json
from pathlib import Path

from run import DEFAULT_BASE, PROCESSES, ROOT, dump, sha

TRIAL='Warning in SimpleSpaceShower::pT2nextQCD: weight above unity'
THRESHOLD='Error in SimpleSpaceShower::pT2nearThreshold: xPDF = 0'
PROFILE_KEYS=('pdf:pset','pdf:lepton','spaceshower:alphasvalue',
    'timeshower:alphasvalue','spaceshower:alphasorder','timeshower:alphasorder',
    'spaceshower:nquarkin','timeshower:ngluontoquark','standardmodel:alphasnfmax',
    '4:m0','5:m0','6:m0','timeshower:globalrecoil',
    'beams:setproductionscalesfromlhef','beams:setdipoleshowerstartingscalesfromlhef')
CSV_FIELDS=('process','status','N_source','N_native_accepted','N_matched_nonzero',
    'negative_H_before','negative_H_after','negative_S_before','negative_S_after',
    'inclusive_matched_over_source','inclusive_relative_change',
    'first_H_over_S_5_20','first_H_over_S_error_5_20',
    'full_H_over_S_5_20','full_H_over_S_error_5_20',
    'paired_shower_H_over_S_drift_5_20','paired_shower_H_over_S_drift_error_5_20',
    'first_net_pb_5_20','first_net_error_pb_5_20','full_net_pb_5_20','full_net_error_pb_5_20',
    'paired_shower_net_drift_pb_5_20','paired_shower_net_drift_error_pb_5_20',
    'first_native_warning_total','full_native_warning_total',
    'first_ISR_weight_above_unity','full_ISR_weight_above_unity',
    'full_threshold_xPDF_zero','pending_or_failed_checks')


def bind(path):return dict(path=str(path),sha256=sha(path))


def matching_recovery(base,key,run):
    paths=list((base/'recovery'/key).glob(run+'_*/plan.json'))
    if not paths:return None
    path=max(paths,key=lambda x:x.stat().st_mtime);plan=json.loads(path.read_text())
    report=dict(status=plan['status'],plan=bind(path),base_seed=plan['base_seed'],
        original_jobs=plan['source_jobs'],original_hard_events=plan['source_events'],
        source_order=plan['source_order'],seed_rule=plan['seed_rule'],
        original_sources=[{k:row[k] for k in ('index','path','sha256','events','seed')} for row in plan['source_bindings']],
        original_statuses={status:sum((row['previous_summary'] or {}).get('status','missing')==status
            for row in plan['source_bindings']) for status in ('complete','failed','missing')},
        source_generation_repeated=plan['source_generation_repeated'],subset_selection=plan['subset_selection'],
        warning_based_retry=plan['warning_based_retry'],matcher_source=plan.get('matcher_source'),
        numerical_scope='Stable dead-cone algebra, exact native-floor breakpoints and cosine endpoint integration map. Analytic kernel and requested quadrature tolerance are unchanged.',
        qualification='Every original matching job is replayed; the replacement is not conditioned on previously successful shards. Other live provider epochs are preserved.')
    for name in ('execution.json','previous_endpoint_comparison.json','previous_successful_summary_comparison.json','archive_manifest.json'):
        candidate=path.parent/name
        if candidate.is_file():report[name[:-5]]=dict(binding=bind(candidate),record=json.loads(candidate.read_text()))
    for name in ('source_hashes_and_normalizations_unchanged','collected_weights','matching_input_sumw',
                 'normalization_tolerance','collected_output','contract','finished_utc'):
        if name in plan:report[name]=plan[name]
    return report


def actual_matching_epoch(base,p):
    process=base/'processes'/p['directory'];directory=process/'MCatNLO/srcPythia8'
    files={name:dict(bind(directory/name),mtime_ns=(directory/name).stat().st_mtime_ns)
        for name in ('MacNLOPSMatch.cc','MacNLOPSMatch.o','MacNLOPSMatch.build.log') if (directory/name).is_file()}
    return dict(actual_provider=files,
        provider_executables=[bind(path) for path in sorted(process.glob('SubProcesses/P*/macnlops_onestep')) if path.is_file()],
        repository_snapshot_is_runtime_identity=False,
        note='The repository can change during an already exported live run. Actual provider source/object/executable bindings, not later ROOT snapshot hashes, identify its matching runtime.')


def runtime_epochs(base,run):
    key='Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc';repository=sha(ROOT/key);records={}
    for p in PROCESSES:
        actual=actual_matching_epoch(base,p);source=actual['actual_provider'].get('MacNLOPSMatch.cc',{}).get('sha256')
        snapshots={}
        for stage in ('generation','shower'):
            path=base/'validation'/p['key']/(run+'_'+stage+'_epoch.json')
            if path.is_file():
                epoch=json.loads(path.read_text());snapshots[stage]=dict(binding=bind(path),repository_matcher_sha256=epoch.get('source_sha256',{}).get(key))
        first=base/'results'/p['key']/run/'first.json'
        if first.is_file():
            epoch=json.loads(first.read_text())['provenance'];snapshots['first_analysis']=dict(binding=bind(first),repository_matcher_sha256=epoch.get('source_sha256',{}).get(key))
        records[p['key']]=dict(actual=actual,repository_snapshots=snapshots,
            actual_source_matches_current_repository=source==repository,
            recovery=matching_recovery(base,p['key'],run))
    path=base/'validation/actual_matching_runtime_epochs.json'
    dump(path,dict(schema='native-tc-actual-matching-runtime-epochs-v1',run=run,
        current_repository_matcher_sha256=repository,processes=records,
        interpretation='Exported source/object/provider binaries identify actual matching runtime. Generation, first-analysis and shower epoch files record repository snapshots at their respective times; a later repository numerical repair does not mutate already exported providers.',
        physical_changes='The verified Wj numerical repair retains the analytic kernel/tolerance, original hard points, seeds, model/couplings/PDF and installed Pythia. Other providers keep their successful frozen numerical epochs.'))
    return path


def collect(base,p,run,expected_events):
    key=p['key'];result=base/'results'/key/run
    independent=base/'validation'/(key+('_1M_event_audit.json' if expected_events==1000000 else '_pilot_event_audit.json'))
    paths={name:result/(name+'.json') for name in
           ('first','full','native_accounting','inclusive_closure','shape_summary')}
    paths['independent_event_audit']=independent
    missing=[name for name,path in paths.items() if not path.is_file()]
    row=dict(process=key,status='pending',pending_or_failed_checks='; '.join('missing '+x for x in missing))
    recovery=matching_recovery(base,key,run)
    if missing:return dict(row=row,missing=missing,result_directory=str(result),matching_recovery=recovery)
    docs={name:json.loads(path.read_text()) for name,path in paths.items()}
    first,full=docs['first'],docs['full'];native=docs['native_accounting'];closure=docs['inclusive_closure']
    shapes=docs['shape_summary'];event=docs['independent_event_audit']
    payload=first.get('diagnostic_payload_sha256')
    if payload is None and (result/'encoding_audit.json').is_file():
        encoding=json.loads((result/'encoding_audit.json').read_text())
        if (encoding.get('schema')=='native-tc-encoding-audit-v2'
            and encoding.get('first_summary_sha256')==sha(paths['first'])
            and encoding.get('weight_sequence_audit',{}).get('passed') is True):
            payload=encoding.get('current_uncompressed_sha256')
    checks={
        'first_event_count':first.get('events')==expected_events,
        'full_event_count':full.get('events')==expected_events,
        'independent_event_audit':event.get('passed') is True and event.get('counts',{}).get('events')==expected_events,
        'native_accounting':native.get('passed') is True and native.get('native_event_accounting_passed') is True,
        'native_every_input_accepted':all(native.get('full',{}).get('counts',{}).get(k)==expected_events
                                          for k in ('tried','selected','accepted')),
        'native_physical_profiles':bool(native.get('full',{}).get('jobs')) and all(
            item.get('physical_profile_passed') is True for item in native['full']['jobs']),
        'per_job_inclusive_token_closure':closure.get('passed') is True,
        'first_inclusive_weight_closure':shapes.get('stages',{}).get('first',{}).get('inclusive_weight_closure',{}).get('passed') is True,
        'full_inclusive_weight_closure':shapes.get('stages',{}).get('full',{}).get('inclusive_weight_closure',{}).get('passed') is True,
        'paired_first_full_drift_present':'primary_5_20' in shapes.get('paired_first_full_drift',{}),
        'source_epoch':first.get('provenance',{}).get('source_construction')=='mcatnlo-delta-native-tc-v1',
        'independent_payload_binding':bool(payload) and event.get('sha256_uncompressed')==payload,
        'stock_pythia_unchanged':native.get('pythia_source_modified') is False,
    }
    if recovery is not None:checks['complete_original_cohort_recovery']=recovery['status']=='complete' and recovery.get('source_hashes_and_normalizations_unchanged') is True and recovery['subset_selection'] is False
    mapping={'negative_H':'H_negative_source','retained_negative_H':'H_negative_retained',
             'negative_S':'S_negative_source','retained_negative_S':'S_negative_retained',
             'S':'S','H':'H'}
    checks['independent_signed_counts']=all(first['counts'].get(left,0)==event['counts'].get(right,0)
                                            for left,right in mapping.items())
    failed=[name for name,passed in checks.items() if not passed]
    row['status']='complete' if not failed else 'audit_failed'
    row['pending_or_failed_checks']='; '.join(failed)
    if failed:return dict(row=row,checks=checks,bindings={k:bind(v) for k,v in paths.items()},result_directory=str(result))
    counts=first['counts'];fs=shapes['stages']['first'];ss=shapes['stages']['full']
    f,s=fs['primary_5_20'],ss['primary_5_20'];drift=shapes['paired_first_full_drift']['primary_5_20']
    fw=native['first']['native_warning_occurrences'];sw=native['full']['messages']
    ratio=fs['inclusive']['matched_over_source']
    row.update(N_source=counts['events'],N_native_accepted=native['full']['counts']['accepted'],
        N_matched_nonzero=counts.get('retained_H',0)+counts.get('retained_S',0),
        negative_H_before=counts.get('negative_H',0),negative_H_after=counts.get('retained_negative_H',0),
        negative_S_before=counts.get('negative_S',0),negative_S_after=counts.get('retained_negative_S',0),
        inclusive_matched_over_source=ratio,inclusive_relative_change=ratio-1.,
        first_H_over_S_5_20=f['H_over_S'],first_H_over_S_error_5_20=f['H_over_S_error'],
        full_H_over_S_5_20=s['H_over_S'],full_H_over_S_error_5_20=s['H_over_S_error'],
        paired_shower_H_over_S_drift_5_20=drift.get('H_over_S_drift'),
        paired_shower_H_over_S_drift_error_5_20=drift.get('H_over_S_drift_error'),
        first_net_pb_5_20=f['net_pb'],first_net_error_pb_5_20=f['net_error_pb'],
        full_net_pb_5_20=s['net_pb'],full_net_error_pb_5_20=s['net_error_pb'],
        paired_shower_net_drift_pb_5_20=drift['net_drift_pb'],
        paired_shower_net_drift_error_pb_5_20=drift['net_drift_error_pb'],
        first_native_warning_total=sum(v for k,v in fw.items() if k.startswith('Warning in ')),
        full_native_warning_total=sum(v for k,v in sw.items() if k.startswith('Warning in ')),
        first_ISR_weight_above_unity=fw.get(TRIAL,0),full_ISR_weight_above_unity=sw.get(TRIAL,0),
        full_threshold_xPDF_zero=sw.get(THRESHOLD,0))
    profiles=[{k:item['effective_settings'].get(k) for k in PROFILE_KEYS} for item in native['full']['jobs']]
    checks['same_selected_profile_all_jobs']=all(x==profiles[0] for x in profiles)
    if not checks['same_selected_profile_all_jobs']:
        row.update(status='audit_failed',pending_or_failed_checks='same_selected_profile_all_jobs')
    extras={}
    for name in ('epoch_comparison','encoding_audit','resume_identity'):
        path=result/(name+'.json')
        if path.is_file():extras[name]=bind(path)
    if 'resume_identity' in extras:
        resume=json.loads((result/'resume_identity.json').read_text())
        checks['recorded_resume_identity']=resume.get('verified') is True
        if not checks['recorded_resume_identity']:row.update(status='audit_failed',pending_or_failed_checks='recorded_resume_identity')
    fiducial=None
    if key=='Wj' and (result/'fiducial40_comparison.json').is_file():
        path=result/'fiducial40_comparison.json';fid=json.loads(path.read_text())
        fiducial=dict(binding=bind(path),stages={stage:doc['fiducial40'] for stage,doc in fid['stages'].items()},
            paired_shower_drift=fid['paired_full_minus_first'].get('fiducial40'),
            historical_qualification=fid['historical']['qualification'])
    return dict(row=row,checks=checks,bindings={k:bind(v) for k,v in paths.items()},extra_bindings=extras,
        matching_runtime_epoch=actual_matching_epoch(base,p),matching_recovery=recovery,
        fiducial40=fiducial,
        result_directory=str(result),native_warning_occurrences=dict(first=fw,full=sw),
        clean_native_error_statistics=native.get('clean_native_error_statistics'),
        first_physical_profile=native.get('first_profile'),full_selected_profile=profiles[0],
        inclusive=dict(first=fs['inclusive'],full=ss['inclusive']),
        primary_5_20=dict(first=f,full=s,paired_shower_drift=drift),
        source_provenance=first['provenance'])


def uncertainty(value,error,digits=5):
    if value is None:return 'undefined'
    if error is None:return f'{value:.{digits}f}'
    return f'{value:.{digits}f} ± {error:.{digits}f}'


def markdown(data):
    completed=data['completed_processes'];pending=data['pending_processes'];rows=data['processes']
    text=['# Fresh native-t_c campaign results','',
        f"Run `{data['run']}`: **{len(completed)}/{len(rows)} processes complete and audited**. "+
        ('All eight samples are complete.' if data['all_eight_complete'] else 'The campaign remains partial.'),'',
        f"Generated {data['generated_utc']}. Expected source events per process: {data['expected_events']:,}.",'',
        '| Process | Status | Source / native accepted | Matched nonzero | Negative H before → after | Negative S before → after | Inclusive M/source − 1 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key,item in rows.items():
        r=item['row']
        if r['status']!='complete':text.append(f"| {key} | {r['status']} | — | — | — | — | — |");continue
        text.append(f"| {key} | complete | {r['N_source']:,} / {r['N_native_accepted']:,} | {r['N_matched_nonzero']:,} | {r['negative_H_before']:,} → {r['negative_H_after']:,} | {r['negative_S_before']:,} → {r['negative_S_after']:,} | {100*r['inclusive_relative_change']:+.5f}% |")
    text+=['','The following ratios use the primary-observable interval 5–20 GeV. Δ is full minus first, with paired first/full covariance.','',
        '| Process | H/S first | H/S full | Paired shower Δ(H/S) | Net first [pb] | Net full [pb] |',
        '|---|---:|---:|---:|---:|---:|']
    for key,item in rows.items():
        r=item['row']
        if r['status']!='complete':continue
        text.append('| '+key+' | '+' | '.join((
            uncertainty(r['first_H_over_S_5_20'],r['first_H_over_S_error_5_20']),
            uncertainty(r['full_H_over_S_5_20'],r['full_H_over_S_error_5_20']),
            uncertainty(r['paired_shower_H_over_S_drift_5_20'],r['paired_shower_H_over_S_drift_error_5_20']),
            uncertainty(r['first_net_pb_5_20'],r['first_net_error_pb_5_20'],4),
            uncertainty(r['full_net_pb_5_20'],r['full_net_error_pb_5_20'],4)))+' |')
    significant=[]
    for key,item in rows.items():
        r=item['row'];error=r.get('paired_shower_H_over_S_drift_error_5_20');value=r.get('paired_shower_H_over_S_drift_5_20')
        if r['status']=='complete' and error and value is not None and abs(value)>=3*error:
            significant.append(f"{key} ({uncertainty(value,error)})")
    if significant:text+=['','Material remaining shower drift in H/S(5–20), exceeding three quoted conditional Monte Carlo errors: '+', '.join(significant)+'. These discrepancies are preserved as physics/model limitations.']
    wj=rows.get('Wj',{})
    if wj.get('row',{}).get('status')=='complete' and wj.get('fiducial40'):
        fid=wj['fiducial40'];text+=['','Wj fiducial40 uses the highest-pT qualifying anti-kT R=0.5 jet, pT>40 GeV and |y|<4.5, with original signed normalization.','',
            '| Stage | Fiducial M/source − 1 | H/S selected total | H/S recoil5–20 |',
            '|---|---:|---:|---:|']
        for stage,doc in fid['stages'].items():
            total=doc['selected_total'];selected=doc['recoil5_20']
            text.append(f"| {stage} | {100*(total['matched_over_source']-1):+.5f}% | {uncertainty(total['H_over_S'],total['H_over_S_error'])} | {uncertainty(selected['H_over_S'],selected['H_over_S_error'])} |")
        text+=['','This material fiducial shift remains after the native-t_c endpoint change. Its dependence on the cuts and matching-density model is part of the validation result. Differential curves and the qualified historical comparison are in fiducial40_comparison.json/pdf.']
    text+=['','| Process | ISR weight-above-unity first / full | Full threshold xPDF=0 | Native error table clean |',
        '|---|---:|---:|---|']
    for key,item in rows.items():
        r=item['row']
        if r['status']=='complete':text.append(f"| {key} | {r['first_ISR_weight_above_unity']:,} / {r['full_ISR_weight_above_unity']:,} | {r['full_threshold_xPDF_zero']:,} | {item['clean_native_error_statistics']} |")
    text+=['','Warning numbers are native occurrences, not distinct affected-event counts. Exact dictionaries, including routine SLHA warnings and other evolution warnings, are retained in JSON. Every completed sample has all requested native inputs tried, selected and accepted. Threshold diagnostics can coexist with complete event accounting; no warning-based filtering or retry is performed.','',
        'All uncertainties shown are paired conditional Monte Carlo errors from source chunks/native shower jobs. Hard-integration, channel-normalization and stock-shower modeling uncertainties are **not included**. Net means matched minus source; positive H removal and signed S veto are retained separately. Matched nonzero counts differ from native accepted counts because zero-weight rejected events are still showered.','',
        'The primary observable is sqrt(Durham dmerge_N) for EE Nj and hard-system transverse recoil for DY/H/ttbar/Wbb; Wj uses pT(W+leading jet). These process-dependent observables share units, not identical physical meaning.','',
        'Settings: EE sqrt(s)=91.188 GeV, fixed hard scales and bare leptons; EE3j/4j Durham generation ycut=0.01. Hadronic beams are 13 TeV, hard PDF331700 (Wbb PDF334700). DY has mll>60 GeV. Wj generation uses pTj>20 GeV and anti-kT R=0.5, with separate inclusive and pTj>40 GeV, |y|<4.5 analyses in individual reports. Wbb is massive 4FS, mb=4.7 GeV, without generation or analysis cuts.','',
        'First emission uses alpha_s(MZ)=0.1365, continuation0.118 at order one. Hadronic continuation uses the corresponding NLO LHAPDF set (NNPDF40_nlo_as_01180, nf4 for Wbb), as in the historical hadronic cards. Bare-lepton EE cards contain inactive PDF13 with PDF:lepton off. All fresh full showers explicitly use hard-model particle masses (5FS c=b=0; Wbb b=4.7), physical species4/5, stock alphaSnfmax5, no MPI/hadronization/primordial kT/QED, and stable W/top/H. The installed Pythia is unchanged.','',
        'Historical comparisons are complete-epoch comparisons. The new source, native Born-scale correction, gluon-FSR fixes and explicit full-shower mass profile differ from older runs. In particular, archived DY full showers retained Pythia charm/bottom defaults whereas fresh 5FS full showers set both to zero. Historical full-shape changes cannot be attributed to t_c alone. Each available epoch_comparison.json preserves its archived labels and errors.','']
    if pending:
        text+=['Pending or failed gates:','']
        text.extend(f"- {key}: {rows[key]['row']['pending_or_failed_checks']}" for key in pending)
        text.append('')
    text+=['Individual evidence:','']
    for key,item in rows.items():
        if item['row']['status']=='complete':text.append(f"- [{key} shape summary]({Path(item['result_directory'])/'shape_summary.json'}), [native accounting]({Path(item['result_directory'])/'native_accounting.json'}), [per-job closure]({Path(item['result_directory'])/'inclusive_closure.json'}).")
    recovered=[(key,item['matching_recovery']) for key,item in rows.items() if item.get('matching_recovery')]
    if recovered:
        text+=['','Matching numerical recovery:','']
        for key,recovery in recovered:
            matcher=(recovery.get('matcher_source') or {}).get('sha256','not installed yet')
            text.append(f"- {key}: {recovery['status']}; all {recovery['original_jobs']} original jobs / {recovery['original_hard_events']:,} hard events, original base seed {recovery['base_seed']}. Matching source `{matcher}`. All original sources and individual seeds are bound in [{Path(recovery['plan']['path']).name}]({recovery['plan']['path']}); no successful-shard conditioning or hard-point regeneration.")
        text+=['','The repair changes numerical evaluation/integration only: the analytic kernel and requested tolerance remain unchanged. Actual exported matcher source/object/executable hashes are recorded per process in JSON. Later repository snapshots can carry the repair while an already exported live EE4j/Wbb provider still uses the earlier numerical epoch. Successful earlier kernels are not rerun solely for this metadata distinction.','']
    return '\n'.join(text)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--run',default='run_1M_native_tc')
    parser.add_argument('--events',type=int,default=1000000)
    parser.add_argument('--output',type=Path,help='Output directory; default BASE/results')
    a=parser.parse_args();base=a.base.resolve();output=a.output or base/'results';output.mkdir(parents=True,exist_ok=True)
    records={}
    for p in PROCESSES:
        try:records[p['key']]=collect(base,p,a.run,a.events)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            records[p['key']]=dict(row=dict(process=p['key'],status='audit_failed',pending_or_failed_checks=str(exc)))
    completed=[k for k,v in records.items() if v['row']['status']=='complete']
    pending=[k for k in records if k not in completed]
    runtime_path=runtime_epochs(base,a.run)
    global_evidence={name:bind(base/'validation'/name) for name in
        ('final_source_epoch_contract.json','old_cleanup_current_state.json','durham_cut_validation.json',
         'native_tc_core_tests.json','independent_statistics_review.json','final_physical_profiles.json')
        if (base/'validation'/name).is_file()}
    global_evidence['actual_matching_runtime_epochs.json']=bind(runtime_path)
    data=dict(schema='native-tc-campaign-results-v1',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base=str(base),run=a.run,expected_events=a.events,completed_processes=completed,pending_processes=pending,
        all_eight_complete=len(completed)==8,processes=records,
        global_evidence=global_evidence,
        completion_policy='All eight individual first/full, native accounting/profile, token-precision closure and independent event audits must pass. Missing/failed samples remain explicit.',
        uncertainty_scope='Conditional paired Monte Carlo errors only; no hard-integration or model uncertainty included.',
        generator=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))))
    dump(output/'campaign_summary.json',data)
    csv_buffer=io.StringIO();writer=csv.DictWriter(csv_buffer,fieldnames=CSV_FIELDS,extrasaction='ignore')
    writer.writeheader();writer.writerows(v['row'] for v in records.values())
    for name,text in (('campaign_summary.csv',csv_buffer.getvalue()),('campaign_summary.md',markdown(data))):
        target=output/name;temporary=target.with_name(target.name+'.tmp');temporary.write_text(text);temporary.replace(target)
    print(f"Campaign report: {len(completed)}/8 complete; pending: {', '.join(pending) or 'none'}; {output/'campaign_summary.md'}",flush=True)


if __name__=='__main__':main()
