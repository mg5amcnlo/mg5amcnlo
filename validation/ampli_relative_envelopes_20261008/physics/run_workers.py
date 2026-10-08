"""Build snapshotted sources and replay fixed-input workers in an isolated export.

Usage: python run_workers.py VARIANT [gf3_small gf3_large gf1_large]
Set AMPLI_STREAM_DIAGNOSTICS=1 to instrument only the copied export adapter.
"""
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
paths=json.loads((HERE/'paths.json').read_text())
variant=sys.argv[1]
workloads=sys.argv[2:] or ['gf3_small','gf3_large','gf1_large']
sources=HERE/(variant+'_sources')
export=Path(paths['isolated'])
process=export/paths['process']
out=HERE/variant
resolved_export=export.resolve()
WORKLOADS={
 'gf3_small':('GF3.0','2500',Path(paths['original'])),
 'gf3_large':('GF3.0','30000',Path(paths['large'])),
 'gf1_large':('GF1.0','gf1_large',Path(paths['large'])),
}
for original_key in ('original','large'):
 original_export=Path(paths[original_key]).resolve()
 assert resolved_export != original_export and original_export not in resolved_export.parents
for name in workloads:
 channel,label,original=WORKLOADS[name]
 worker=process/(channel+'_'+variant+'_'+name)
 if worker.exists():
  raise RuntimeError('Replay worker already exists: '+str(worker))
 checkpoint=process/channel
 if not checkpoint.exists():
  shutil.copytree(original/paths['process']/channel,checkpoint,symlinks=True)
for source in (sources/'Template/NLO/SubProcesses').iterdir():
 destination=(export/'SubProcesses'/source.name).resolve()
 assert resolved_export in destination.parents
 shutil.copyfile(source,export/'SubProcesses'/source.name)
out.mkdir(exist_ok=True)
if os.environ.get('AMPLI_STREAM_DIAGNOSTICS')=='1':
 subprocess.run([sys.executable,str(HERE/'instrument_streams.py'),str(export/'SubProcesses/ampli_mint_adapter.f90')],check=True)
 shutil.copyfile(export/'SubProcesses/ampli_mint_adapter.f90',out/'instrumented_ampli_mint_adapter.f90')
with (out/'build.log').open('w') as log:
 result=subprocess.run(['make','-j1','madevent_mintMC'],cwd=process,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,out/'build.log'
variant_executable=process/('madevent_mintMC_'+variant)
shutil.copyfile(process/'madevent_mintMC',variant_executable)
variant_executable.chmod(0o755)
spec=importlib.util.spec_from_file_location('ampli_pool_snapshot',sources/'madgraph/various/ampli_pool.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
rows=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
for name in workloads:
 channel,label,original=WORKLOADS[name]
 worker=process/(channel+'_'+variant+'_'+name)
 shutil.copytree(original/paths['process']/(channel+'_1'),worker,symlinks=True)
 if os.environ.get('AMPLI_SEED'):
  seed=int(os.environ['AMPLI_SEED'])
  (worker/'randinit').unlink()
  (worker/'randinit').write_text('r='+str(seed)+'\n')
 target=out/label;target.mkdir(exist_ok=True)
 start=time.monotonic()
 with (worker/'input_app.txt').open('rb') as data,(target/'run.log').open('wb') as log:
  run=subprocess.run(['../'+variant_executable.name],cwd=worker,stdin=data,stdout=log,stderr=subprocess.STDOUT)
 wall=time.monotonic()-start
 if run.returncode:raise RuntimeError('Worker failed: '+str(target/'run.log'))
 pool=validator.read_pool(worker)
 retained,tails=validator._native_worker_status(pool)
 assert all(value<.01 for value in tails.values()),tails
 log=(target/'run.log').read_text()
 timing=dict((key,float(value)) for key,value in re.findall(r'Time spent in ([A-Za-z_0-9]+)\s*:\s*([\d.E+-]+)',log))
 epochs=pool['epochs'];active_trials=sum(epoch['trials'] for epoch in epochs if epoch['eligible'])
 result_values=list(map(float,(worker/'res.dat').read_text().split()))
 row=dict(worker=str(worker),exit_code=run.returncode,wall_seconds=wall,randinit=(worker/'randinit').read_text().strip(),
  timing=timing,trials=pool['trials'],candidates=pool['ncandidates'],quota=pool['final_quota'],reserve=pool['generated_target'],
  mean_signed=pool['mean_signed'],mean_abs=pool['mean_abs'],
  pooled_trial_error_signed=math.sqrt(pool['m2_signed']/(pool['trials']*(pool['trials']-1))),
  pooled_trial_error_abs=math.sqrt(pool['m2_abs']/(pool['trials']*(pool['trials']-1))),
  active_trials=active_trials,expired_trial_fraction=1-active_trials/pool['trials'],epochs=len(epochs),
  eligible_epochs=sum(epoch['eligible'] for epoch in epochs),adaptation=pool['adaptation'],tails=tails,validator_passed=True,
  published_rate=dict(absolute=result_values[0],error_abs=result_values[1],signed=result_values[2],error_signed=result_values[3]),
  res_dat=(worker/'res.dat').read_text().strip(),
  numerical_artifact_sha256={name:hashlib.sha256((worker/name).read_bytes()).hexdigest() for name in ['ampli_pool.dat','ampli_candidates.lhe']})
 for artifact in ['ampli_pool.dat','res.dat','input_app.txt','ampli_job.dat','moffset.dat','randinit']:
  shutil.copyfile(worker/artifact,target/artifact)
 if (worker/'ampli_stream_trials.dat').exists():
  shutil.copyfile(worker/'ampli_stream_trials.dat',target/'ampli_stream_trials.dat')
 (target/'candidate_lhe_path.txt').write_text(str(worker/'ampli_candidates.lhe')+'\n')
 rows[label]=row
 (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(label,json.dumps(row),flush=True)
