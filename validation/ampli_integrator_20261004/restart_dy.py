from pathlib import Path
import sys,subprocess,os,json,gzip,re
root=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
sys.path.insert(0,str(root))
from madgraph.various.banner import RunCardNLO
work=Path('/tmp/mg5-amplicol-backend-sq5iorrd'); out=work/'drell_yan'
card=RunCardNLO(str(out/'Cards/run_card.dat'))
card['nevents']=60
card['nevt_job']=12
card['event_norm']='unity'
card['iseed']=271828
card.write(str(out/'Cards/run_card.dat'))
cmd=work/'restart_dy.cmd'
cmd.write_text('set automatic_html_opening False --no_save\nset notification_center False --no_save\nset run_mode 2 --no_save\nset nb_core 4 --no_save\nlaunch aMC@NLO -f -p --only_generation --name=ampli_restart_unity\nquit\n')
with (work/'restart_dy.log').open('w') as log:
    p=subprocess.run([sys.executable,'-O',str(out/'bin/aMCatNLO'),str(cmd)],cwd=out,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'})
print('status',p.returncode)
result=out/'Events/ampli_restart_unity/events.lhe.gz'
if result.exists():
    s=gzip.open(result,'rt').read()
    weights=[float(x.split()[2]) for x in re.findall(r'<event(?:\s[^>]*)?>\s*([^\n]+)',s)]
    print('count',len(weights),'abs weights',sorted(set(abs(x) for x in weights)))
else: print('No events')
