from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
ROOT=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
WORK=Path(__file__).parent; OUT=WORK/'drell_yan'
sys.path.insert(0,str(ROOT))
from madgraph.various.banner import RunCardNLO
shutil.copy2(ROOT/'madgraph/various/ampli_pool.py',OUT/'bin/internal/ampli_pool.py')
card=RunCardNLO(str(OUT/'Cards/run_card.dat'))
settings=dict(nevents=60,nevt_job=10,iseed=19722,event_norm='unity',reweight_scale=[True],store_rwgt_info=True)
for k,v in settings.items():card[k]=v
card.write(str(OUT/'Cards/run_card.dat'))
checkpoints={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'SubProcesses').glob('P*/G*/ampli_grids') if not p.is_symlink()}
cmd=WORK/'restart.cmd'
cmd.write_text(chr(10).join(['set automatic_html_opening False --no_save','set notification_center False --no_save','set run_mode 2 --no_save','set nb_core 5 --no_save','launch aMC@NLO -f -p --only_generation --name=restart60','quit','']))
meta=dict(settings=settings,survey_grid_sha256=checkpoints,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),collector_sha256=hashlib.sha256((OUT/'bin/internal/ampli_pool.py').read_bytes()).hexdigest())
t=time.monotonic()
with (WORK/'restart.log').open('w') as log:
 r=subprocess.run([sys.executable,'-O',str(OUT/'bin/aMCatNLO'),str(cmd)],cwd=OUT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
meta.update(returncode=r.returncode,wall_seconds=time.monotonic()-t,events_exist=(OUT/'Events/restart60/events.lhe.gz').exists(),survey_grids_unchanged=all(hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h for p,h in checkpoints.items()))
(WORK/'restart_finished.json').write_text(json.dumps(meta,indent=2)+chr(10))
print(json.dumps(meta,indent=2),flush=True)
