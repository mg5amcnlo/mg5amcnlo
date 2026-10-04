from pathlib import Path
import sys, shutil, subprocess, os
root=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
sys.path.insert(0,str(root))
from madgraph.various.banner import RunCardNLO
work=Path('/tmp/mg5-amplicol-backend-sq5iorrd')
out=work/'drell_yan'
for name in ['ampli_mint_adapter.f90','mint_module.f90','driver_mintMC.f','FKSParams.f90','simple_integrator.f90','integrator_helpers.f90']:
    shutil.copyfile(root/'Template/NLO/SubProcesses'/name,out/'SubProcesses'/name)
shutil.copyfile(root/'madgraph/interface/amcatnlo_run_interface.py',out/'bin/internal/amcatnlo_run_interface.py')
for proc in (out/'SubProcesses').glob('P*'):
    if proc.is_dir(): shutil.copyfile(root/'Template/NLO/SubProcesses/makefile_fks_dir',proc/'makefile')
card=RunCardNLO(str(out/'Cards/run_card.dat'))
settings=dict(nevents=100,nevt_job=25,req_acc=0.1,iseed=314159,parton_shower='PYTHIA8',folding=[2,2,2],born_spreading=False,reweight_scale=[True],reweight_pdf=[False],store_rwgt_info=True,event_norm='sum',fixed_ren_scale=True,fixed_fac_scale=True,mur_ref_fixed=91.188,muf_ref_fixed=91.188,dynamical_scale_choice=[-2],ptl=0.,etal=-1.,drll=0.,drll_sf=0.,mll=0.,mll_sf=60.)
for k,v in settings.items(): card[k]=v
card.write(str(out/'Cards/run_card.dat'))
fks=(root/'Template/NLO/Cards/FKS_params.dat').read_text().replace('#NLOPSIntegrator\n0','#NLOPSIntegrator\n1')
(out/'Cards/FKS_params.dat').write_text(fks)
cmd=work/'run_dy.cmd'
cmd.write_text('set automatic_html_opening False --no_save\nset notification_center False --no_save\nset run_mode 2 --no_save\nset nb_core 4 --no_save\nlaunch aMC@NLO -f -p --name=ampli_folded_v3\nquit\n')
with (work/'run_dy.log').open('w') as log:
    result=subprocess.run([sys.executable,'-O',str(out/'bin/aMCatNLO'),str(cmd)],cwd=out,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'})
print('Run exit status:',result.returncode)
print('Summary exists:',(out/'Events/ampli_folded_v3/summary.txt').exists())
