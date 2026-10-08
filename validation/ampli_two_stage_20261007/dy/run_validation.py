from pathlib import Path
import argparse,datetime,hashlib,json,os,resource,shutil,subprocess,sys,time
ROOT=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
WORK=Path(__file__).resolve().parent
OUT=WORK/'drell_yan'
sys.path.insert(0,str(ROOT))
from madgraph.various.banner import RunCardNLO
parser=argparse.ArgumentParser();parser.add_argument('--restart',action='store_true');args=parser.parse_args()
name='restart2000' if args.restart else 'fresh2000'
sources=['Template/NLO/SubProcesses/ampli_mint_adapter.f90','Template/NLO/SubProcesses/simple_integrator.f90','Template/NLO/SubProcesses/driver_mintMC.f','madgraph/interface/amcatnlo_run_interface.py','madgraph/various/ampli_pool.py']
settings=dict(nevents=2000,req_acc=-1,nevt_job=2500,iseed=19731 if args.restart else 19730,ebeam1=6500.,ebeam2=6500.,pdlabel='nn23nlo',parton_shower='PYTHIA8',folding=[2,2,2],born_spreading=False,event_norm='sum',reweight_scale=[False],reweight_pdf=[False],store_rwgt_info=False)
card=RunCardNLO(str(OUT/'Cards/run_card.dat'))
for k,v in settings.items():card[k]=v
card.write(str(OUT/'Cards/run_card.dat'))
fks=(OUT/'Cards/FKS_params.dat').read_text().replace('#NLOPSIntegrator\n0','#NLOPSIntegrator\n1').replace('#UsePolyVirtual\n.False.','#UsePolyVirtual\n.True.')
(OUT/'Cards/FKS_params.dat').write_text(fks)
cmd=WORK/(name+'.cmd')
cmd.write_text('set automatic_html_opening False --no_save\nset notification_center False --no_save\nset run_mode 2 --no_save\nset nb_core 3 --no_save\nlaunch aMC@NLO -f -p '+('--only_generation ' if args.restart else '')+'--name='+name+'\nquit\n')
def hashes(paths):return {str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
checkpoint_paths=[p for pattern in ('P*/GF*/ampli_grids','P*/GF*/grid.MC_integer') for p in (OUT/'SubProcesses').glob(pattern) if not p.is_symlink()]
meta=dict(name=name,settings=settings,cores=3,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},checkpoints_before=hashes(checkpoint_paths))
(WORK/(name+'_started.json')).write_text(json.dumps(meta,indent=2)+'\n')
for p in sources:
 target=WORK/'sources'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target)
t=time.monotonic()
with (WORK/(name+'.log')).open('w') as log:
 r=subprocess.run([sys.executable,'-O',str(OUT/'bin/aMCatNLO'),str(cmd)],cwd=OUT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
checkpoint_paths=[p for pattern in ('P*/GF*/ampli_grids','P*/GF*/grid.MC_integer') for p in (OUT/'SubProcesses').glob(pattern) if not p.is_symlink()]
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode,wall_seconds=time.monotonic()-t,events_exist=(OUT/'Events'/name/'events.lhe.gz').exists(),source_sha256_after={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},checkpoints_after=hashes(checkpoint_paths))
(WORK/(name+'_finished.json')).write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2),flush=True)
if not meta['events_exist']:sys.exit(1)
