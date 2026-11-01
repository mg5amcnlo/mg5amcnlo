#!/usr/bin/env python3
from pathlib import Path
import hashlib,itertools,json,sys
import numpy as np
R=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(R));from madgraph.various import macnlops as mc
B=Path('/export/slow1/rikkert/macnlops_born_scale_fix_20261101/generalisation');O=Path(__file__).parent
summary=json.loads((B/'direct/DY/summary.json').read_text());inputs=summary['direct_input'];source=Path(inputs['source']['path']);subset=Path(inputs['subset']['path'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
assert sha(source)==inputs['source']['sha256'];assert sha(subset)==inputs['subset']['sha256']
with mc.LHEStream(source) as original,mc.LHEStream(subset) as prefix:
 assert original.header==prefix.header
 blocks=list(prefix);first=list(itertools.islice(original,1500));assert blocks==first and len(blocks)==1500
records=[]
for p in summary['providers']:
 assert p['returncode']==0
 for r in p['records']:
  assert r['repeat_identical'] and r['ready']==1 and r['coupling']==1.
  assert all(r[field]==r['old_'+field] for field in ['r','d','k','alphaK','A'])
  assert r['pdf']>0 and np.isfinite(r['pdf'])
  records.append(r)
assert len(records)==441 and len(set(r['index'] for r in records))==441
mu_error=max(abs(r['q2fact1']/(r['muReal_HT2']**2)-1) for r in records)
assert mu_error<1e-10
clone=B/'processes/DY';native=clone/'MCatNLO/srcPythia8/MacNLOPSMatch.native.cc';probe=clone/'MCatNLO/srcPythia8/MacNLOPSMatch.cc'
assert sha(native)==summary['source_native_CPP_sha256'];assert sha(probe)==summary['probe_sha256']
logs=[]
for p in summary['providers']:
 log=B/'direct/DY'/(p['provider']+'.log');exe=clone/'SubProcesses'/p['provider']/'macnlops_onestep'
 assert sha(log)==p['log_sha256'];assert sha(exe)==p['exe_sha256'];logs.append(dict(provider=p['provider'],records=len(p['records']),log_sha256=sha(log),exe_sha256=sha(exe)))
result=dict(passed=True,source_prefix_events=1500,prefix_header_and_events_exact=True,source_file_sha256=sha(source),subset_sha256=sha(subset),evaluated_unique_points=len(records),all_recorded_RDK_A_alpha_exact=True,all_repeated_requests_exact=True,Born_coupling_factor_exactly_one=True,pdf_factor_range=[min(r['pdf'] for r in records),max(r['pdf'] for r in records)],max_real_HT2_scale_relative_error=mu_error,native_cpp_sha256=sha(native),probe_cpp_sha256=sha(probe),providers=logs,limits='Direct native reevaluation of pre-existing first endpoints, not a new DY first-emission/shower run. It confirms provider factors/state and unchanged old densities at441 supported selected contexts; no source regeneration or validation-distribution claim.')
(O/'dy_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
