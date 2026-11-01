from pathlib import Path
import datetime,hashlib,json,sys
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(ROOT));from madgraph.various import macnlops
BASE=Path(__file__).resolve().parent
load=lambda p:json.loads((BASE/p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def bind(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
manifest=load('baseline_manifest.json');strict=load('raw_invariance.json');review=load('independent_review/raw_review.json');heft=load('direct/H_HEFT_diagnostic/summary.json');prior=load('Wbb_prior_factor_comparison.json');tt=load('direct_TT_instrumented_results.json')['processes'][0];ttnog=load('direct_TT_noG_results.json')['processes'][0];dy=load('direct/DY/summary.json')
assert strict['total_events']==19812 and strict['failed_processes']==['Wj']
for p in strict['processes']:
 bad=p['invariant_failures'];assert 'event_bytes_except_decision_and_match_weight' not in bad
 if p['process']=='Wj':assert set(bad)<={'r','d','k','total_k','marking'}
 else:assert not bad
 if p['process'].startswith('EE'):assert p['fixed_ee_G_and_decisions_exact']
 assert p['kernel_scale_equals_old_times_G_max_relative_difference']<1e-13
wj={k:v['relative'] for k,v in review['Wj_maximum_numeric_drifts'].items()};assert max(wj.values())<1e-12
assert heft['passed'] and heft['eligible_points']==321 and prior['passed'] and dy['all_processes_succeeded'] and tt['all_processes_succeeded'] and ttnog['all_processes_succeeded']
for dataset in [dy,tt]:
 for provider in dataset['providers']:
  for p in provider['records']:
   for key in ['r','d','k','alphaK','A']:assert p[key]==p['old_'+key]
for p,q in zip(tt['providers'],ttnog['providers']):
 assert p['provider']==q['provider'] and len(p['records'])==len(q['records'])
 for a,b in zip(p['records'],q['records']):
  assert a['index']==b['index']
  for key in ['r','d','k','alphaK','q2fact1','q2fact2','A']:assert a[key]==b[key]
for p in dy['providers']:
 for r in p['records']:assert r['coupling']==1.
root_sources={};names=['weight_lines.f','montecarlocounter.f','driver_macnlops.f','macnlops_capi.f','macnlops_evaluator.f']
for name in names:root_sources['SubProcesses/'+name]=sha(ROOT/'Template/NLO/SubProcesses'/name)
root_sources['MCatNLO/srcPythia8/MacNLOPSMatch.cc']=sha(ROOT/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc')
source_checks=[]
for p in manifest['processes']:
 clone=Path(p['clone']);original=Path(p['original']);assert macnlops.source_files(original)==p['generation_sources']
 for name in names:
  expected=root_sources['SubProcesses/'+name]
  for file in [clone/'SubProcesses'/name]+list(clone.glob('SubProcesses/P*/'+name)):assert sha(file)==expected,file
 cpp=clone/'MCatNLO/srcPythia8'/('MacNLOPSMatch.native.cc' if p['direct_matched_input'] else 'MacNLOPSMatch.cc');assert sha(cpp)==root_sources['MCatNLO/srcPythia8/MacNLOPSMatch.cc']
 for row in p['samples']:
  for key in ['source','subset','production_matched','production_prefix']:assert sha(row[key]['path'])==row[key]['sha256'],row[key]['path']
 if p['direct_matched_input']:
  for key in ['source','subset']:assert sha(p['direct_matched_input'][key]['path'])==p['direct_matched_input'][key]['sha256']
 source_checks.append(dict(process=p['key'],original_generation_sources_unchanged=True,selected_source_and_reference_files_unchanged=True,all_five_Fortran_sources_and_included_native_CPP_match_frozen_ROOT=True,final_provider_executables={str(file.relative_to(clone)):sha(file) for file in clone.glob('SubProcesses/P*/macnlops_onestep')}))
lib=Path(manifest['Pythia_library']['path']);assert sha(lib)==manifest['Pythia_library']['sha256']=='35203dfab4e20b044fe6e669b8170f5905f56c43c6dad6e762ceff86158a283c'
H=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101/diagnosis_transfer_20261101/scales/H_bound_all.json');assert H.is_file()
summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
 scope=dict(processes=['DY','H_HEFT','TTbar','EE2','EE3','EE4','Wj','Wbb'],raw_paired_events=19812,raw_jobs=41,direct_eligible_points=1700,direct_point_counts=dict(DY=441,H_HEFT=321,TTbar=938),fresh_source_generation=False,new_1M_or_full_shower_run=False),
 raw_invariance=dict(physical_event_payload_source_packets_starting_scales_caps_bit_exact=True,source_A_shower_marginal_CDF_and_emission_factors_bit_exact=True,fixed_ee_G_equals_one_and_all_decisions_bit_exact=True,Wj_density_roundoff_max_relative=wj,qualification='OnlyWj R/D/K,totalK,marking show native-cache floating rescaling noise<=6.43e-16. Strict string-check failure is retained inraw_invariance.json; qualified numerical invariance passes. SourceA,CDF andphysicalpayloads areexact.',per_process=strict['processes']),
 direct_invariance=dict(first_RDK_alphaK_sourceA_bit_exact_to_archived_all1700=True,TT_first_RDK_actual_muF_A_bit_exact_to_noG_control=True,HEFT_first_RDK_alphaK_A_bit_exact_to_noG_control=True,repeat_noise='Strict repeated-request bit equality is not promised: pre-existing nativeBorn cache rescaling causes<=3.23e-15 RDK and<=1.10e-15 coupling-ratio differences. No-G controls preserve the samefirst densities and reproduce repeated-request roundoff.'),
 independent_G=dict(DY_Born_coupling_exactly_one=True,HEFT_effective_Born_and_hard_PDF=heft['reference_factor'],Wbb_prior_independent_reference=prior,unit_fixture_reference='Root compiledFortran9casefixture separately validatesorders0/1/2,HEFT-like higherpowers,mixedcustomcpower,commonactualmuF,bareee,canonicalboost/fullrestore.'),
 protected_charm_zero_PDF_point=next(p['protected_pdf_point'] for p in strict['processes'] if p['process']=='Wbb'),
 all_negative_H_targeted_reference=bind(H),frozen_ROOT_six_source_sha256=root_sources,source_checks=source_checks,stock_Pythia_library=bind(lib),stock_Pythia_unchanged=True,
 limitations=['Controlled event prefixes and cached native density points establish implementation/generalization and preservation, not new1M statistical transfer closure or full-shower validation.','Local signedS/B density-shape approximation remains separate and is not changed bythisfix.','No claim thatthe nativefinite emissionkernel or ISR envelope diagnostics are corrected.','HistoricalDY/HEFT/tt rawGF sources werecleaned; directchecks usefrozen matchedendpoints/hardBorn packets. Currentdependency modules were refreshedonlyinisolatedcopies and exactarchived first densities were required.'],
 evidence={str(p):bind(p) for p in [BASE/'baseline_manifest.json',BASE/'baseline_raw.json',BASE/'corrected_raw.json',BASE/'raw_invariance.json',BASE/'independent_review/raw_review.json',BASE/'direct/DY/summary.json',BASE/'direct_TT_instrumented_results.json',BASE/'direct_TT_noG_results.json',BASE/'TT_repeat_classification.json',BASE/'direct/H_HEFT_diagnostic/summary.json',BASE/'direct/H_HEFT_diagnostic/evidence_hashes.json',BASE/'Wbb_prior_factor_comparison.json']},
 code_files={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file() and 'processes' not in p.relative_to(BASE).parts and p.suffix in ['.py','.cc','.f','.f90']})
(BASE/'validation_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print('PASS:8processes;19812pairedrawevents;1700directeligiblepoints; alloriginalsourcesandstockPythiaunchanged; frozenROOTsixsources verified.')
print(BASE/'validation_summary.json')
