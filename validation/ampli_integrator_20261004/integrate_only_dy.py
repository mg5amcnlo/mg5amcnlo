from pathlib import Path
import sys,subprocess,os
root=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
sys.path.insert(0,str(root))
from madgraph.various.banner import RunCardNLO
work=Path('/tmp/mg5-amplicol-backend-sq5iorrd'); out=work/'drell_yan'
card=RunCardNLO(str(out/'Cards/run_card.dat'))
card['nevents']=0
card['req_acc']=0.15
card['iseed']=57721
card.write(str(out/'Cards/run_card.dat'))
cmd=work/'integrate_only_dy.cmd'
cmd.write_text('set automatic_html_opening False --no_save\nset notification_center False --no_save\nset run_mode 2 --no_save\nset nb_core 4 --no_save\nlaunch aMC@NLO -f -p --name=ampli_integrate_only\nquit\n')
with (work/'integrate_only_dy.log').open('w') as log:
    p=subprocess.run([sys.executable,'-O',str(out/'bin/aMCatNLO'),str(cmd)],cwd=out,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'})
print('status',p.returncode)
