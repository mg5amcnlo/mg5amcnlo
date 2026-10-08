"""Record immutable source/executable and worker-input hashes for all replays."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
paths=json.loads((HERE/'paths.json').read_text())
process=Path(paths['isolated'])/paths['process']
inputs={};builds={}
for summary in sorted(HERE.glob('*/summary.json')):
 variant=summary.parent.name;rows=json.loads(summary.read_text());inputs[variant]={}
 for label,row in rows.items():
  worker=Path(row['worker'])
  names=['input_app.txt','ampli_job.dat','ampli_grids','grid.MC_integer','FKS_params.dat','randinit','param_card.dat','res_1','moffset.dat']
  inputs[variant][label]={name:hashlib.sha256((worker/name).read_bytes()).hexdigest() for name in names}
  inputs[variant][label]['randinit_text']=(worker/'randinit').read_text().strip()
 a=summary.parent/'instrumented_ampli_mint_adapter.f90';b=process/('madevent_mintMC_'+variant)
 build={}
 if a.exists():build['instrumented_adapter_sha256']=hashlib.sha256(a.read_bytes()).hexdigest()
 if b.exists():build['executable_sha256']=hashlib.sha256(b.read_bytes()).hexdigest()
 elif variant=='baseline':build['executable_note']='Fresh GF1 baseline used initial madevent_mintMC before per-variant binary copies; reused GF3 results have exact source hashes and verified numerical instrumentation invariance.'
 builds[variant]=build
(HERE/'worker_input_provenance.json').write_text(json.dumps(inputs,indent=2)+'\n')
(HERE/'build_provenance.json').write_text(json.dumps(builds,indent=2)+'\n')
for v,rows in inputs.items():
 for label,row in rows.items():
  reference=inputs['baseline'][label]
  for name in reference:
   if name in ('randinit','randinit_text'):continue
   assert reference[name]==row[name],(v,label,name)
print('All worker inputs match current baseline except explicit seed overrides.')
