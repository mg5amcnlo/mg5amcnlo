"""Launch only after the parent agent announces the scheduler source freeze."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
WORK=Path(__file__).resolve().parent;OUT=WORK/'drell_yan';NAME='bounded2000'
sys.path.insert(0,str(ROOT))
from madgraph.various.banner import RunCardNLO
mapping={'Template/NLO/SubProcesses/simple_integrator.f90':'SubProcesses/simple_integrator.f90','Template/NLO/SubProcesses/ampli_mint_adapter.f90':'SubProcesses/ampli_mint_adapter.f90','Template/NLO/SubProcesses/driver_mintMC.f':'SubProcesses/driver_mintMC.f','madgraph/interface/amcatnlo_run_interface.py':'bin/internal/amcatnlo_run_interface.py','madgraph/various/ampli_pool.py':'bin/internal/ampli_pool.py'}
for src,dst in mapping.items():
 target=OUT/dst
 if target.is_symlink():target.unlink()
 shutil.copy2(ROOT/src,target)
 target=WORK/'sources'/src;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,target)
settings=dict(nevents=2000,req_acc=-1,nevt_job=2500,iseed=19731,ebeam1=6500.,ebeam2=6500.,pdlabel='nn23nlo',parton_shower='PYTHIA8',folding=[2,2,2],born_spreading=False,event_norm='sum',reweight_scale=[False],reweight_pdf=[False],store_rwgt_info=False)
card=RunCardNLO(str(OUT/'Cards/run_card.dat'))
for k,v in settings.items():card[k]=v
card.write(str(OUT/'Cards/run_card.dat'))
cmd=WORK/'run.cmd';cmd.write_text('set automatic_html_opening False --no_save\nset notification_center False --no_save\nset run_mode 2 --no_save\nset nb_core 2 --no_save\nlaunch aMC@NLO -f -p --only_generation --name='+NAME+'\nquit\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def checkpoint_hashes():return {str(p.relative_to(OUT)):digest(p) for pattern in ('P*/GF*/ampli_grids','P*/GF*/grid.MC_integer') for p in (OUT/'SubProcesses').glob(pattern) if not p.is_symlink()}
meta=dict(name=NAME,original_survey='/tmp/mg5-ampli-two-stage-dy-ovcj8baq/drell_yan',settings=settings,cores=2,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={p:digest(ROOT/p) for p in mapping},exported_sha256={p:digest(OUT/dst) for p,dst in mapping.items()},checkpoints_before=checkpoint_hashes())
(WORK/'started.json').write_text(json.dumps(meta,indent=2)+'\n')
t=time.monotonic()
with (WORK/'run.log').open('w') as log:
 r=subprocess.run([sys.executable,'-O',str(OUT/'bin/aMCatNLO'),str(cmd)],cwd=OUT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode,wall_seconds=time.monotonic()-t,events_exist=(OUT/'Events'/NAME/'events.lhe.gz').exists(),source_sha256_after={p:digest(ROOT/p) for p in mapping},checkpoints_after=checkpoint_hashes())
(WORK/'finished.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if not meta['events_exist']:sys.exit(1)
