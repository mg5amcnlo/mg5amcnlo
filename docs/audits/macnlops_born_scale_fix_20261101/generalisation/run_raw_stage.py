#!/usr/bin/env python3
"""Run a controlled native stage only in external copies of source prefixes."""
from pathlib import Path
import argparse,concurrent.futures,datetime,hashlib,json,shutil,subprocess,sys
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(ROOT))
from madgraph.various import macnlops
from madgraph.various.banner import RunCardNLO
BASE=Path(__file__).resolve().parent;PYTHIA=Path('/nfs/home/rikkert/Downloads/pythia8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def run_process(p,stage):
 clone=Path(p['clone']);inputs=[row['subset']['path'] for row in p['samples']]
 if not inputs:return None
 template=ROOT/'Template/NLO' if stage=='corrected' else Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101/processes/Wbb_4FS')
 for name in ['weight_lines.f','montecarlocounter.f','driver_macnlops.f','macnlops_capi.f','macnlops_evaluator.f']:
  source=template/'SubProcesses'/name;shutil.copy2(source,clone/'SubProcesses'/name)
  for provider in clone.glob('SubProcesses/P*'):
   if provider.is_dir():
    shutil.copy2(source,provider/name);(provider/(Path(name).stem+'.o')).unlink(missing_ok=True)
 shutil.copy2(template/'MCatNLO/srcPythia8/MacNLOPSMatch.cc',clone/'MCatNLO/srcPythia8/MacNLOPSMatch.cc')
 if stage=='baseline':assert sha(clone/'MCatNLO/srcPythia8/MacNLOPSMatch.cc')=='0e2bbd496540ed2692cb0d2665544aba8380add65771776a9baaaec7312dc7e1'
 (clone/'MCatNLO/srcPythia8/MacNLOPSMatch.o').unlink(missing_ok=True)
 for path in clone.glob('SubProcesses/P*/macnlops_onestep'):path.unlink(missing_ok=True)
 jobs=macnlops.prepare_first_emission(clone,RunCardNLO(str(clone/'Cards/run_card.dat')),PYTHIA,inputs,1,keep_rejected=True,workers=2)
 for job,row in zip(jobs,p['samples']):job['args'][job['args'].index('--seed')+1]=str(row['seed'])
 def launch(job):
  result=subprocess.run([str(Path(job['cwd'])/job['exe'])]+job['args'],cwd=job['cwd']);return result.returncode
 with concurrent.futures.ThreadPoolExecutor(8) as pool:status=list(pool.map(launch,jobs))
 summaries,report=macnlops.check_first_emission(jobs);assert not any(status)
 expected=sum(row['events'] for row in p['samples']);assert report['counts'].get('input_S',0)+report['counts'].get('input_H',0)==expected
 destination=BASE/stage/p['key'];destination.mkdir(parents=True,exist_ok=False)
 for job in jobs:
  where=destination/Path(job['cwd']).relative_to(clone/'SubProcesses');where.mkdir(parents=True)
  for name in ['events_macnlops.lhe','events_macnlops.json','events_macnlops.lhe.settings.cmnd','macnlops_onestep.log']:shutil.copy2(Path(job['cwd'])/name,where/name)
 for row in p['samples']:assert sha(row['subset']['path'])==row['subset']['sha256']
 result=dict(process=p['key'],expected_events=expected,report=report,channels=summaries,
   matching_sources={str(path.relative_to(clone)):sha(path) for path in [clone/'SubProcesses'/name for name in ['weight_lines.f','montecarlocounter.f','driver_macnlops.f','macnlops_capi.f','macnlops_evaluator.f']]+[clone/'MCatNLO/srcPythia8/MacNLOPSMatch.cc']},
   provider_executables={str(path.relative_to(clone)):sha(path) for path in clone.glob('SubProcesses/P*/macnlops_onestep')},
   original_seeds_preserved=True,input_subsets_unchanged=True)
 (destination/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(stage,p['key'],'PASS',expected,'events',flush=True);return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['baseline','corrected']);a=ap.parse_args()
 manifest=json.loads((BASE/'baseline_manifest.json').read_text());assert not (BASE/f'{a.stage}_raw.json').exists()
 with concurrent.futures.ThreadPoolExecutor(3) as pool:results=list(pool.map(lambda p:run_process(p,a.stage),manifest['processes']))
 output=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=a.stage,processes=[r for r in results if r],script_sha256=sha(__file__),stock_Pythia_sha256=sha(PYTHIA/'lib/libpythia8.so'))
 (BASE/f'{a.stage}_raw.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
