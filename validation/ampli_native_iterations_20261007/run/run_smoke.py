from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
WORK=Path(__file__).resolve().parent
OUT=WORK/'drell_yan'
sys.path.insert(0,str(ROOT))
from madgraph.various.banner import RunCardNLO
import shutil
# Synchronize completed sources after the matrix-element export was generated.
for name in ('ampli_mint_adapter.f90','simple_integrator.f90','driver_mintMC.f'):
    shutil.copy2(ROOT/'Template/NLO/SubProcesses'/name,OUT/'SubProcesses'/name)
for src,dst in [('madgraph/interface/amcatnlo_run_interface.py','bin/internal/amcatnlo_run_interface.py'),('madgraph/various/ampli_pool.py','bin/internal/ampli_pool.py')]:
    shutil.copy2(ROOT/src,OUT/dst)
card=RunCardNLO(str(OUT/'Cards/run_card.dat'))
settings=dict(nevents=6000,req_acc=.15,nevt_job=2000,iseed=19726,
              ebeam1=6500.,ebeam2=6500.,pdlabel='nn23nlo',parton_shower='PYTHIA8',
              folding=[2,2,2],born_spreading=False,event_norm='sum',
              reweight_scale=[False],reweight_pdf=[False],store_rwgt_info=False)
for k,v in settings.items():card[k]=v
card.write(str(OUT/'Cards/run_card.dat'))
fks=(OUT/'Cards/FKS_params.dat').read_text().replace('#NLOPSIntegrator\n0','#NLOPSIntegrator\n1').replace('#UsePolyVirtual\n.False.','#UsePolyVirtual\n.True.')
(OUT/'Cards/FKS_params.dat').write_text(fks)
cmd=WORK/'restart.cmd'
cmd.write_text('set automatic_html_opening False --no_save\nset notification_center False --no_save\nset run_mode 2 --no_save\nset nb_core 5 --no_save\nlaunch aMC@NLO -f -p --only_generation --name=native_restart6000\nquit\n')
meta=dict(settings=settings,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['Template/NLO/SubProcesses/ampli_mint_adapter.f90','Template/NLO/SubProcesses/simple_integrator.f90','Template/NLO/SubProcesses/driver_mintMC.f','madgraph/interface/amcatnlo_run_interface.py','madgraph/various/ampli_pool.py']})
(WORK/'restart_started.json').write_text(json.dumps(meta,indent=2)+'\n')
t=time.monotonic()
with (WORK/'restart.log').open('w') as log:
 r=subprocess.run([sys.executable,'-O',str(OUT/'bin/aMCatNLO'),str(cmd)],cwd=OUT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
meta.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode,wall_seconds=time.monotonic()-t,events_exist=(OUT/'Events/native_restart6000/events.lhe.gz').exists())
(WORK/'restart_finished.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2),flush=True)
