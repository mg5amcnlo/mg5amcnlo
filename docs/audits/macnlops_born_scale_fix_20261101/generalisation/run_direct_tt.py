from pathlib import Path
import concurrent.futures,datetime,hashlib,json,shutil,subprocess,sys
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(ROOT));from madgraph.various import macnlops
from madgraph.various.banner import RunCardNLO
BASE=Path(__file__).resolve().parent;PYTHIA=Path('/nfs/home/rikkert/Downloads/pythia8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(p):
 clone=Path(p['clone']);data=p['direct_matched_input'];providers=[d for d in clone.glob('SubProcesses/P*') if d.is_dir()]
 for name in ['mcatnlo_delta_scales.f90','weight_lines.f','montecarlocounter.f','driver_macnlops.f','macnlops_capi.f','macnlops_evaluator.f']:
  source=ROOT/'Template/NLO/SubProcesses'/name;shutil.copy2(source,clone/'SubProcesses'/name)
  for provider in providers:shutil.copy2(source,provider/name);(provider/(Path(name).stem+'.o')).unlink(missing_ok=True)
 src=clone/'MCatNLO/srcPythia8';shutil.copy2(ROOT/'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc',src/'MacNLOPSMatch.native.cc');shutil.copy2(BASE/'direct_probe_tt.cc',src/'MacNLOPSMatch.cc');(src/'MacNLOPSMatch.o').unlink(missing_ok=True)
 inputs=[]
 for provider in providers:
  (provider/'macnlops_onestep').unlink(missing_ok=True);dummy=provider/'direct_point_input';dummy.mkdir(exist_ok=True);inputs.append(str(dummy/'events.lhe'))
 macnlops.prepare_first_emission(clone,RunCardNLO(str(clone/'Cards/run_card.dat')),PYTHIA,inputs,1,keep_rejected=True,workers=2)
 pdf,info=macnlops.shower_pdf(clone,RunCardNLO(str(clone/'Cards/run_card.dat')));env=macnlops.shower_environment(PYTHIA,info);output=BASE/'direct_TT_instrumented'/p['key'];output.mkdir(parents=True,exist_ok=False);results=[]
 def point(provider):
  log=output/(provider.name+'.log')
  with log.open('w') as stream:r=subprocess.run([str(provider/'macnlops_onestep'),data['subset']['path'],str(provider/'macnlops_routes.dat')],cwd=provider,env=env,stdout=stream,stderr=subprocess.STDOUT)
  records=[]
  for line in log.read_text(errors='replace').splitlines():
   if line.startswith('GPOINT '):records.append(json.loads(line[7:]))
  result=dict(provider=provider.name,returncode=r.returncode,records=records,log_sha256=sha(log),exe_sha256=sha(provider/'macnlops_onestep'))
  print(p['key'],provider.name,'status',r.returncode,'points',len(records),flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(4) as pool:results=list(pool.map(point,providers))
 result=dict(process=p['key'],direct_input=data,providers=results,records=sum(len(r['records']) for r in results),all_processes_succeeded=all(r['returncode']==0 for r in results),source_native_CPP_sha256=sha(src/'MacNLOPSMatch.native.cc'),probe_sha256=sha(src/'MacNLOPSMatch.cc'))
 (output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result
manifest=json.loads((BASE/'baseline_manifest.json').read_text())
with concurrent.futures.ThreadPoolExecutor(3) as pool:results=list(pool.map(run,[p for p in manifest['processes'] if p['key']=='TTbar']))
(BASE/'direct_TT_instrumented_results.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),processes=results),indent=2,sort_keys=True)+'\n')
